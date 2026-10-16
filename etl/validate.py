"""Data-quality checks for the four processed tables and the ``docs/DATA_QUALITY.md`` report.

Checks: duplicate primary keys, physical ranges, missing-value rate per column, orphan
``district_id`` values, daily-date completeness, crop-to-district join coverage and the share of
synthetic rows. Key and range violations raise :class:`DataQualityError` ("fail loudly"); everything
is written to the report first so the evidence is on disk.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import ROOT, get_path

LOG = logging.getLogger(__name__)

KEYS: dict[str, list[str]] = {
    "districts": ["district_id"],
    "weather_daily": ["district_id", "date"],
    "satellite_16d": ["district_id", "date"],
    "crops_annual": ["district_id", "year", "season", "crop"],
}
# (low, high) inclusive physical bounds.
RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "weather_daily": {
        "rain_mm": (0.0, 1000.0),
        "tmax_c": (0.0, 55.0),
        "tmin_c": (-10.0, 45.0),
        "rh_pct": (0.0, 100.0),
        "wind_ms": (0.0, 60.0),
        "solar_mj": (0.0, 40.0),
    },
    "satellite_16d": {"ndvi": (-0.2, 1.0), "soil_moisture": (0.0, 1.0)},
    "crops_annual": {
        "area_ha": (0.0, 1.0e8),
        "production_t": (0.0, 1.0e9),
        "yield_t_per_ha": (0.0, 150.0),
    },
    "districts": {
        "centroid_lat": (6.0, 38.0),
        "centroid_lon": (66.0, 100.0),
        "area_km2": (1.0, 1.0e6),
    },
}
MIN_JOIN_COVERAGE = 0.90


class DataQualityError(RuntimeError):
    """A key or physical-range rule was violated."""


@dataclass
class Check:
    """One validation result."""

    table: str
    name: str
    passed: bool
    detail: str
    blocking: bool = True


def check_keys(table: str, frame: pd.DataFrame) -> Check:
    """Primary key must be unique and non-null."""
    key = KEYS[table]
    nulls = int(frame[key].isna().any(axis=1).sum())
    dupes = int(frame.duplicated(subset=key).sum())
    return Check(
        table,
        "unique non-null key " + "+".join(key),
        nulls == 0 and dupes == 0,
        f"{dupes} duplicate, {nulls} null keys",
    )


def check_ranges(table: str, frame: pd.DataFrame) -> list[Check]:
    """Every value within physical bounds (NaN is a missing-value matter, not a range error)."""
    checks = []
    for column, (low, high) in RANGES.get(table, {}).items():
        if column not in frame:
            checks.append(Check(table, f"{column} present", False, "column missing"))
            continue
        values = frame[column].dropna()
        bad = int(((values < low) | (values > high)).sum())
        checks.append(
            Check(
                table,
                f"{column} in [{low:g}, {high:g}]",
                bad == 0,
                f"{bad} out-of-range of {len(values)}",
            )
        )
    return checks


def check_tmin_tmax(weather: pd.DataFrame) -> Check:
    """Tmin must not exceed Tmax."""
    bad = int((weather["tmin_c"] > weather["tmax_c"]).sum())
    return Check("weather_daily", "tmin_c <= tmax_c", bad == 0, f"{bad} violations")


def check_orphans(table: str, frame: pd.DataFrame, districts: pd.DataFrame) -> Check:
    """Every district_id must exist in districts."""
    orphans = sorted(set(frame["district_id"]) - set(districts["district_id"]))
    return Check(
        table,
        "district_id exists in districts",
        not orphans,
        f"{len(orphans)} orphan ids {orphans[:5]}",
    )


def check_daily_complete(weather: pd.DataFrame) -> Check:
    """Each district must have one row per day between its first and last date, equal across districts."""
    spans = weather.groupby("district_id")["date"].agg(["min", "max", "count"])
    expected = (spans["max"] - spans["min"]).dt.days + 1
    gaps = int((expected != spans["count"]).sum())
    return Check("weather_daily", "no missing days", gaps == 0, f"{gaps} districts with gaps")


def check_join_coverage(
    crops: pd.DataFrame, districts: pd.DataFrame, reported: float | None
) -> Check:
    """Share of crop rows that join to a district (and the source-level coverage reported by the loader)."""
    joined = (
        float(crops["district_id"].isin(districts["district_id"]).mean()) if len(crops) else 0.0
    )
    coverage = joined if reported is None else min(joined, reported)
    return Check(
        "crops_annual",
        f"join coverage >= {MIN_JOIN_COVERAGE:.0%}",
        coverage >= MIN_JOIN_COVERAGE,
        f"{coverage:.1%}",
    )


def missing_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Missing-value rate per column."""
    rates = (
        frame.isna().mean().rename("missing_rate").reset_index().rename(columns={"index": "column"})
    )
    return rates


def run_checks(tables: dict[str, pd.DataFrame], crop_report: dict | None = None) -> list[Check]:
    """Run every check against the loaded tables."""
    districts = tables["districts"]
    checks: list[Check] = []
    for name, frame in tables.items():
        checks.append(check_keys(name, frame))
        checks.extend(check_ranges(name, frame))
        if name != "districts":
            checks.append(check_orphans(name, frame, districts))
    checks.append(check_tmin_tmax(tables["weather_daily"]))
    checks.append(check_daily_complete(tables["weather_daily"]))
    reported = (crop_report or {}).get("join_coverage")
    checks.append(check_join_coverage(tables["crops_annual"], districts, reported))
    return checks


def _md_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def render_report(
    tables: dict[str, pd.DataFrame],
    checks: list[Check],
    crop_report: dict,
    weather_provenance: dict,
    satellite_note: str,
) -> str:
    """Render docs/DATA_QUALITY.md as markdown."""
    synthetic = [
        (name, int(frame["is_synthetic"].sum()), len(frame))
        for name, frame in tables.items()
        if "is_synthetic" in frame
    ]
    lines = [
        "# Data quality report",
        "",
        "Generated by `python -m etl.validate` (do not edit by hand).",
        "",
    ]
    lines += ["## Real versus synthetic data", ""]
    syn_df = pd.DataFrame(
        [(n, f"{s:,}", f"{t:,}", f"{(s / t if t else 0):.1%}") for n, s, t in synthetic],
        columns=["table", "synthetic rows", "total rows", "synthetic share"],
    )
    lines += [_md_table(syn_df), ""]
    if any(s for _, s, _ in synthetic):
        lines += [
            "**WARNING: parts of the data are synthetic (`is_synthetic=True`). Any analysis using them must say so.**",
            "",
        ]
    lines += [satellite_note, ""]
    if weather_provenance:
        rows = []
        for var, shares in weather_provenance.items():
            if var == "tmin_clipped_to_tmax_days":
                continue
            rows.append(
                (var, *(f"{shares.get(k, 0.0):.1%}" for k in ("imd", "nasa_power", "synthetic")))
            )
        lines += ["## Weather provenance (share of district-days by source)", ""]
        lines += [
            _md_table(
                pd.DataFrame(rows, columns=["variable", "IMD grid", "NASA POWER", "synthetic"])
            ),
            "",
        ]
        clipped = weather_provenance.get("tmin_clipped_to_tmax_days", {}).get("count", 0)
        lines += [
            f"Days where Tmin exceeded Tmax after mixing sources and was clipped: {int(clipped)}.",
            "",
        ]
    lines += ["## Checks", ""]
    check_df = pd.DataFrame(
        [(c.table, c.name, "PASS" if c.passed else "FAIL", c.detail) for c in checks],
        columns=["table", "check", "result", "detail"],
    )
    lines += [_md_table(check_df), ""]
    lines += ["## Table sizes", ""]
    size_df = pd.DataFrame(
        [
            (n, f"{len(f):,}", f"{f['district_id'].nunique():,}" if "district_id" in f else "")
            for n, f in tables.items()
        ],
        columns=["table", "rows", "districts"],
    )
    lines += [_md_table(size_df), ""]
    lines += ["## Missing-value rate per column", ""]
    for name, frame in tables.items():
        rates = missing_rates(frame.drop(columns=["geometry_wkt"], errors="ignore"))
        rates["missing_rate"] = rates["missing_rate"].map(lambda x: f"{x:.2%}")
        lines += [f"**{name}**", "", _md_table(rates), ""]
    if crop_report:
        lines += ["## Crop sources and district join coverage", ""]
        lines += [f"- Sources: {', '.join(crop_report.get('sources', [])) or 'none (synthetic)'}"]
        for source, cov in crop_report.get("coverage_by_source", {}).items():
            lines += [f"- Join coverage, {source}: {cov:.1%} (threshold {MIN_JOIN_COVERAGE:.0%})"]
        unmatched = crop_report.get("unmatched", [])
        lines += ["", f"Unmatched source districts: {len(unmatched)}"]
        if unmatched:
            lines += ["", _md_table(pd.DataFrame(unmatched).drop_duplicates())]
        lines += [""]
        crops = tables["crops_annual"]
        by_crop = (
            crops.groupby("crop")
            .agg(
                rows=("year", "size"),
                districts=("district_id", "nunique"),
                first_year=("year", "min"),
                last_year=("year", "max"),
                mean_yield_t_ha=("yield_t_per_ha", "mean"),
                flagged_yield_mismatch=("yield_flag", "sum"),
            )
            .reset_index()
            .round(3)
        )
        lines += ["", _md_table(by_crop), ""]
        dropped = crop_report.get("dropped_unreliable", {})
        if dropped.get("n"):
            lines += [
                f"**Rows dropped as unreliable: {dropped['n']}** "
                f"(by reason {dropped['by_reason']}; by crop {dropped['by_crop']}). "
                "Rule: the source's reported yield contradicts production/area by more than "
                "`max_yield_mismatch`, or the yield exceeds the crop's plausible maximum (settings.yaml).",
                "",
                _md_table(pd.DataFrame(dropped["rows"])),
                "",
            ]
    lines += [
        "## Known limitations",
        "",
        "- District units are reconstructions of ICRISAT's 1966 parent districts from 2021 polygons (docs/DECISIONS.md D-006); borders are approximate.",
        "- ICRISAT reports annual totals per crop; each crop is attributed to one season (D-007). Rice also grows in rabi in AP/TS/TN, so kharif-labelled rice yield is an annual average.",
        "- ICRISAT sugarcane values (~8 t/ha) are far below usual cane yields (~70-100 t/ha); they are probably jaggery-equivalent, not cane tonnes. Treat as an index, do not compare with cane yields from other sources.",
        "- IMD Tmax/Tmin are 1 degree grids: neighbouring districts share cells and differ little.",
        "- NASA POWER is a reanalysis/model product at ~0.5 degree sampled at the district centroid, not an area mean.",
        "",
    ]
    return "\n".join(lines)


def validate_all(strict: bool = True) -> list[Check]:
    """Load the processed parquet files, run checks, write docs/DATA_QUALITY.md and reports/data_quality.json."""
    processed = get_path("data_processed")
    tables = {name: pd.read_parquet(processed / f"{name}.parquet") for name in KEYS}
    interim = get_path("data_interim")
    crop_report = _load_json(interim / "crops_report.json")
    provenance = _load_json(interim / "weather_provenance.json")
    satellite_note = _load_json(interim / "satellite_report.json").get("note", "")
    checks = run_checks(tables, crop_report)
    report = render_report(tables, checks, crop_report, provenance, satellite_note)
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "DATA_QUALITY.md").write_text(report, encoding="utf-8")
    summary = {
        "passed": all(c.passed for c in checks if c.blocking),
        "failed_checks": [f"{c.table}: {c.name} ({c.detail})" for c in checks if not c.passed],
        "synthetic_share": {
            n: float(f["is_synthetic"].mean()) for n, f in tables.items() if "is_synthetic" in f
        },
        "rows": {n: int(len(f)) for n, f in tables.items()},
        "crop_join_coverage": crop_report.get("join_coverage"),
    }
    (get_path("reports") / "data_quality.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    failures = [c for c in checks if c.blocking and not c.passed]
    for check in failures:
        LOG.error("DATA QUALITY FAIL [%s] %s: %s", check.table, check.name, check.detail)
    if strict and failures:
        raise DataQualityError(
            f"{len(failures)} data-quality checks failed; see docs/DATA_QUALITY.md"
        )
    return checks


def main() -> None:
    """CLI: validate the processed tables."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    validate_all()


if __name__ == "__main__":
    main()
