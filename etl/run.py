"""ETL command line: ``python -m etl.run --all`` (or ``--source imd|power|gee|crops|boundaries|weather|validate``).

Order for ``--all``: boundaries -> imd -> power -> weather -> gee (satellite) -> crops -> validate.
Every step is idempotent: existing outputs are kept unless ``--force`` is given. Real data always
wins; a synthetic generator is used only for a source that fails, and its rows carry
``is_synthetic=True``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from config import get_path, load_env, load_settings
from etl import (
    boundaries,
    crop_yield,
    gee_satellite,
    imd_common,
    imd_rainfall,
    imd_temperature,
    nasa_power,
    validate,
)
from etl.synthetic.crops import generate_crops
from etl.synthetic.satellite import generate_satellite
from etl.weather import assemble_weather

LOG = logging.getLogger("etl.run")

SOURCES = ("boundaries", "imd", "power", "weather", "gee", "crops", "validate")


def _processed(name: str) -> Path:
    return get_path("data_processed") / f"{name}.parquet"


def _interim(name: str) -> Path:
    return get_path("data_interim") / name


def _window(key: str) -> tuple[int, int]:
    years = load_settings()[key]
    return int(years["start"]), int(years["end"])


def step_boundaries(force: bool) -> None:
    """districts.parquet"""
    boundaries.build(force)


def step_imd(force: bool) -> None:
    """Download (resumable) and aggregate IMD rain/Tmax/Tmin to district daily means."""
    districts = boundaries.districts_geodataframe()
    start, end = _window("weather_years")
    years = range(start, end + 1)
    for var in imd_common.VARIABLES:
        ok, failed = imd_common.ensure_years(var, years)
        LOG.info(
            "IMD %s: %d/%d years available%s",
            var,
            len(ok),
            len(years),
            f" (failed: {failed})" if failed else "",
        )
    rain = imd_rainfall.district_rainfall(districts, years)
    temp = imd_temperature.district_temperature(districts, years)
    rain.to_parquet(_interim("imd_rain_daily.parquet"), index=False)
    temp.to_parquet(_interim("imd_temp_daily.parquet"), index=False)
    LOG.info("IMD aggregated: rain %d rows, temperature %d rows", len(rain), len(temp))


def step_power(force: bool) -> None:
    """NASA POWER daily humidity/wind/solar (and temperature/rain fallback) at district centroids."""
    districts = pd.read_parquet(_processed("districts"))
    start, end = _window("weather_years")
    out = _interim("power_daily.parquet")
    if out.exists() and not force:
        LOG.info("power_daily.parquet exists, skipping (use --force)")
        return
    frame = nasa_power.district_daily(districts, start, end, force=force)
    frame.to_parquet(out, index=False)
    LOG.info(
        "NASA POWER: %d rows for %d districts",
        len(frame),
        frame["district_id"].nunique() if len(frame) else 0,
    )


def _read_optional(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def step_weather(force: bool) -> None:
    """weather_daily.parquet from IMD + POWER (+ synthetic gap filling if allowed)."""
    out = _processed("weather_daily")
    if out.exists() and not force:
        LOG.info("weather_daily.parquet exists, skipping (use --force)")
        return
    settings = load_settings()
    districts = pd.read_parquet(_processed("districts"))
    start, end = _window("weather_years")
    weather, provenance = assemble_weather(
        districts,
        start,
        end,
        _read_optional(_interim("imd_rain_daily.parquet")),
        _read_optional(_interim("imd_temp_daily.parquet")),
        _read_optional(_interim("power_daily.parquet")),
        allow_synthetic=bool(settings["etl"]["allow_synthetic_fallback"]),
        seed=int(settings["project"]["random_seed"]),
    )
    weather.to_parquet(out, index=False)
    _interim("weather_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    LOG.info(
        "weather_daily: %d rows, synthetic share %.2f%%",
        len(weather),
        100 * weather["is_synthetic"].mean(),
    )


def step_gee(force: bool) -> None:
    """satellite_16d.parquet from Google Earth Engine, or flagged synthetic data if unavailable."""
    out = _processed("satellite_16d")
    if out.exists() and not force:
        LOG.info("satellite_16d.parquet exists, skipping (use --force)")
        return
    settings = load_settings()
    districts = pd.read_parquet(_processed("districts"))
    start, end = _window("satellite_years")
    note_path = _interim("satellite_report.json")
    try:
        table = gee_satellite.build_satellite(districts, range(start, end + 1), force)
        try:
            gee_satellite.build_chirps(districts, range(start, end + 1), force)
        except gee_satellite.GeeUnavailableError as exc:
            LOG.warning("CHIRPS check series skipped: %s", exc)
        note = f"Satellite features come from Google Earth Engine (MODIS MOD13Q1 NDVI, {settings['etl']['soil_moisture_source']} soil moisture)."
    except gee_satellite.GeeUnavailableError as exc:
        if not settings["etl"]["allow_synthetic_fallback"]:
            raise
        LOG.warning(
            "Earth Engine unavailable (%s) -> SYNTHETIC satellite data. Run `python -m etl.gee_auth_check`.",
            exc,
        )
        weather = pd.read_parquet(_processed("weather_daily"))
        table = generate_satellite(weather, start, end, int(settings["project"]["random_seed"]))
        note = (
            "**Satellite features (NDVI, soil moisture) are SYNTHETIC**: Google Earth Engine is not authenticated here "
            f"({str(exc)[:120]}). They are derived from rainfall by construction and carry no independent information; "
            "run `python -m etl.gee_auth_check` and `python -m etl.run --source gee --force` to replace them with real data."
        )
    table.to_parquet(out, index=False)
    note_path.write_text(
        json.dumps({"note": note, "synthetic": bool(table["is_synthetic"].any())}), encoding="utf-8"
    )
    LOG.info("satellite_16d: %d rows, synthetic=%s", len(table), bool(table["is_synthetic"].any()))


def step_crops(force: bool) -> None:
    """crops_annual.parquet from ICRISAT / APY CSV, or flagged synthetic data if none is available."""
    out = _processed("crops_annual")
    if out.exists() and not force:
        LOG.info("crops_annual.parquet exists, skipping (use --force)")
        return
    settings = load_settings()
    districts = pd.read_parquet(_processed("districts"))
    try:
        crops, report = crop_yield.build_crops(districts, force)
    except crop_yield.CropDataUnavailableError as exc:
        if not settings["etl"]["allow_synthetic_fallback"]:
            raise
        LOG.warning("%s -> SYNTHETIC crop yields", exc)
        years = settings["years"]
        crops = generate_crops(
            districts,
            pd.read_parquet(_processed("weather_daily")),
            _read_optional(_processed("satellite_16d")),
            int(years.get("history_start", years["start"])),
            int(years["end"]),
            int(settings["project"]["random_seed"]),
        )
        report = {
            "sources": [],
            "unmatched": [],
            "coverage_by_source": {},
            "join_coverage": 1.0,
            "rows": int(len(crops)),
        }
    crops.to_parquet(out, index=False)
    _interim("crops_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    LOG.info(
        "crops_annual: %d rows from %s",
        len(crops),
        ", ".join(report["sources"]) or "synthetic generator",
    )


def step_validate(force: bool) -> None:
    """Run data-quality checks and write docs/DATA_QUALITY.md."""
    validate.validate_all(strict=True)


STEPS = {
    "boundaries": step_boundaries,
    "imd": step_imd,
    "power": step_power,
    "weather": step_weather,
    "gee": step_gee,
    "crops": step_crops,
    "validate": step_validate,
}


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for the ETL CLI."""
    parser = argparse.ArgumentParser(prog="python -m etl.run", description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="run every step in order")
    group.add_argument("--source", choices=(*SOURCES, "all"), help="run one step")
    parser.add_argument(
        "--years",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="override the weather and satellite year windows",
    )
    parser.add_argument("--states", nargs="+", help="override the study states in settings.yaml")
    parser.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested ETL steps; returns a process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    load_env()
    settings = load_settings()
    if args.years:
        settings["weather_years"] = {"start": args.years[0], "end": args.years[1]}
        settings["satellite_years"] = {"start": max(2000, args.years[0]), "end": args.years[1]}
    if args.states:
        settings["states"] = args.states
    selected = list(SOURCES) if args.all or args.source == "all" else [args.source]
    started = time.time()
    for name in selected:
        LOG.info("=== step: %s ===", name)
        STEPS[name](args.force)
    print(f"ETL finished in {time.time() - started:.0f}s: steps {', '.join(selected)}")  # noqa: T201 - CLI summary
    for table in ("districts", "weather_daily", "satellite_16d", "crops_annual"):
        path = _processed(table)
        if path.exists():
            frame = pd.read_parquet(path)
            synthetic = (
                f", synthetic {frame['is_synthetic'].mean():.1%}" if "is_synthetic" in frame else ""
            )
            print(f"  {table:<14} {len(frame):>9,} rows{synthetic}")  # noqa: T201 - CLI summary
    return 0


if __name__ == "__main__":
    sys.exit(main())
