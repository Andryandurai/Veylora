"""Export ``simulator/sim_data.json``: every number the 3D simulator shows, computed by the Phase 4-6 code.

For N representative districts (one per climate cluster) and each crop with yield history, the exporter evaluates a
grid of scenarios (rainfall multiplier x temperature shift x irrigation strategy) for the first 16 weeks of the
season. The browser interpolates bilinearly between grid points, so no backend is needed. Contract and field
definitions: ``simulator/README.md``; validated against ``simulator/sim_schema.json``.

Sources of each number:
* yield (mean, lo, hi): ``predict_yield`` at normal rainfall and the scenario temperature x the water-stress ratio of
  the daily water balance (docs/DECISIONS.md D-020);
* soil moisture, ETc/ETa, irrigation, stress, excess water: the daily FAO-56 water balance (Phase 6) scored per strategy;
  ``optimized`` irrigation is the Phase 6 weekly LP;
* ``drought_p``: the Phase 5 drought model on scenario rainfall; ``heat_p``: the share of IMD-qualifying heatwave
  days under the temperature shift (not a model probability);
* ``ndvi``: a **visual proxy** from the crop-coefficient curve and accumulated stress, NOT satellite NDVI;
* ``pond_mm`` / ``seepage_mm`` (paddy rice): standing water and seepage of the ponded-field water balance (D-029);
* ``stage_rel``: the four FAO-33 stage factors of the water balance (their product is the relative yield);
* ``reference_daily``: the observed daily weather of the reference year (unscaled; the page applies the sliders);
* ``reference_ndvi``: satellite NDVI of the reference year, flagged ``is_synthetic`` while the dataset is synthetic;
* ``elevation_m``: real relief of a 10 km window at the district centre (``simulator/terrain.py``), else the procedural
  ``elevation_proxy`` is all there is and ``elevation_source`` says so;
* ``development`` / ``senescence`` (per crop): a *visual* growth curve built from the stage lengths, not measured growth.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator

from analytics.predictive import features as F
from analytics.predictive.extreme_weather import risk_forecast
from analytics.predictive.predict import predict_yield
from analytics.predictive.risk_scenarios import get_scenario_risk, heat_threshold_note
from analytics.prescriptive import scenario as S
from analytics.prescriptive import water_balance as wb
from analytics.prescriptive.params import load_crop, load_globals, taw_mm
from config import ROOT, get_path, load_settings
from simulator import terrain

LOG = logging.getLogger(__name__)

SIM_DIR = ROOT / "simulator"
OUT_PATH = SIM_DIR / "sim_data.json"
DISTRICT_INDEX = "sim_districts.json"
SCHEMA_PATH = SIM_DIR / "sim_schema.json"
GRID_RAIN = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
GRID_TEMP = [-2, 0, 1, 2, 3, 4]
STRATEGIES = ["rainfed", "baseline", "optimized"]
SIM_ORDER = {"rainfed": "rainfed", "baseline": "farmer_baseline", "optimized": "optimized"}
MAX_BYTES = 15 * 1024 * 1024
MIN_CROP_ROWS = 10
YIELD_NOTES = {
    "sugarcane": "Yield unit in the source data looks like jaggery equivalent, not cane tonnes (D-007); compare scenarios, not absolute values."
}
DEFINITIONS = {
    "yield": "t/ha. predict_yield at normal rainfall and the scenario temperature x water-stress ratio of the daily water balance relative to the farmer baseline (D-020). lo/hi = 80% interval.",
    "rain_mm": "Weekly rainfall (mm) of the reference year, multiplied by the rainfall multiplier.",
    "tmax_c": "Weekly mean of daily Tmax (C) of the reference year plus the temperature shift.",
    "soil_moisture_0_1": "Root-zone water at the end of the week as a fraction of total available water (FAO-56 daily balance).",
    "ndvi": "VISUAL PROXY from the crop-coefficient curve and accumulated water stress (0.15 + 0.65 x canopy^0.8 x (1 - 0.45 x smoothed stress)); not satellite NDVI.",
    "etc_mm": "Weekly unstressed crop evapotranspiration (mm).",
    "eta_mm": "Weekly actual crop evapotranspiration (mm).",
    "irrigation_mm": "Net irrigation applied in the week (mm).",
    "excess_mm": "Water that did not enter the root zone or drained below it (runoff + drainage, mm).",
    "stress_0_1": "1 - ETa/ETc for the week.",
    "drought_p": "Phase 5 drought model: probability that the SPI-3 of the week's month is <= -1, given data up to the end of the previous month, with rainfall scaled by the multiplier (temperature does not change it).",
    "heat_p": "Share of the week's days that qualify as IMD heatwave days under the temperature shift (Tmax >= 40 C and >= 4.5 C above the district normal). A deterministic indicator, not a model probability.",
    "elevation_proxy": "Procedural relief 0-1 seeded by the district id; NOT real elevation. Used only when elevation_m is absent.",
    "elevation_m": "Real elevation (m) on a 32 x 32 lattice over a 10 km window at the district centre (row 0 = north), from open terrain tiles (SRTM and other public data). The scene draws it with vertical exaggeration; fields and plants are schematic.",
    "pond_mm": "Standing water in a ponded field (paddy rice) at the end of the week (mm); 0 for upland crops.",
    "pond_daily": "Daily standing water (mm) of the 112-day window for ponded crops only; the weekly pond_mm is its value on the last day of each week.",
    "seepage_mm": "Seepage and percolation loss of a ponded field in the week (mm); 0 for upland crops.",
    "stage_rel": "FAO-33 factor of each growth stage (1 = no loss), 1 - Ky_s (1 - ETa_s / ETc_s); the product is the relative yield of the water balance (before it is combined with the machine-learning yield).",
    "reference_daily": "Observed daily rain (mm), Tmax and Tmin (C) of the reference year for the 16-week window, before the scenario multiplier and shift. The page applies the sliders; model state remains weekly.",
    "reference_ndvi": "Weekly satellite NDVI of the reference year (mean of the 16-day composites in the week). is_synthetic is true while the satellite dataset is synthetic.",
    "development": "VISUAL growth curve 0-1 by week built from the stage lengths (slow establishment, fast vegetative growth, plateau); not measured growth.",
    "senescence": "VISUAL ripening 0-1 by week: progress through the last growth stage; drives the golden colour.",
    "tiles": "8x8 soil class per village tile (0 sandy loam, 1 loam, 2 clay loam, 3 clay); procedural, used for visual variation.",
}
DISCLAIMER = "Explanation tool: it visualises model outputs and what-if scenarios. It is not a physical crop growth simulation and not advice for farm decisions."


def git_commit() -> str:
    """Short hash of HEAD, or 'unknown'."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) & 0xFFFFFFFF


def smooth_field(seed: int, size: int, octaves: tuple[tuple[int, float], ...]) -> np.ndarray:
    """Seeded value noise: bilinearly upsampled random grids of several resolutions, summed and normalised to 0..1."""
    rng = np.random.default_rng(seed)
    out = np.zeros((size, size))
    for cells, weight in octaves:
        grid = rng.random((cells + 1, cells + 1))
        xs = np.linspace(0, cells, size)
        i0 = np.minimum(xs.astype(int), cells - 1)
        t = xs - i0
        rows = grid[i0][:, :] * (1 - t)[:, None] + grid[i0 + 1][:, :] * t[:, None]
        out += weight * (rows[:, i0] * (1 - t)[None, :] + rows[:, i0 + 1] * t[None, :])
    out -= out.min()
    return out / out.max() if out.max() > 0 else out


def elevation_proxy(district_id: str) -> list[list[float]]:
    """32x32 procedural relief in 0..1 (not real elevation)."""
    return np.round(
        smooth_field(_seed("elev:" + district_id), 32, ((3, 1.0), (6, 0.5), (12, 0.2))), 3
    ).tolist()


def soil_tiles(district_id: str) -> list[list[int]]:
    """8x8 soil class per tile (0..3), spatially smooth, biased to clay loam."""
    field = smooth_field(_seed("soil:" + district_id), 8, ((2, 1.0), (4, 0.4)))
    edges = np.quantile(field, [0.2, 0.45, 0.8])
    return np.digitize(field, edges).astype(int).tolist()


def development_curves(stage_days: list[int], n_weeks: int = 16) -> tuple[list[float], list[float]]:
    """Visual growth and ripening curves (0-1 per week) built from the four stage lengths.

    Growth is slow during establishment, rises smoothly through the vegetative stage, plateaus in the mid-season and
    stays at 1 afterwards; ripening is the progress through the last stage. Both are presentation curves, not growth data.
    """
    ini, dev, mid, late = stage_days
    growth, ripening = [], []
    for week in range(n_weeks):
        day = 7 * week + 3.0
        if day < ini:
            g = 0.02 + 0.08 * day / ini
        elif day < ini + dev:
            t = (day - ini) / dev
            g = 0.10 + 0.75 * t * t * (3.0 - 2.0 * t)
        elif day < ini + dev + mid:
            g = 0.85 + 0.15 * (day - ini - dev) / mid
        else:
            g = 1.0
        growth.append(g)
        ripening.append(float(np.clip((day - (ini + dev + mid)) / late, 0.0, 1.0)))
    return growth, ripening


def reference_daily(inputs: S.SeasonInputs) -> dict[str, list[float]]:
    """Observed daily weather of the 16-week window (reference year, unscaled), rounded for the file."""
    return {
        "rain_mm": np.round(inputs.rain / inputs.rain_multiplier, 1).tolist(),
        "tmax_c": np.round(inputs.tmax - inputs.temperature_delta, 1).tolist(),
        "tmin_c": np.round(inputs.tmin - inputs.temperature_delta, 1).tolist(),
    }


def reference_ndvi(
    satellite: pd.DataFrame, district_id: str, year: int, season: str, n_weeks: int = 16
) -> dict[str, Any] | None:
    """Weekly satellite NDVI of the reference year: the 16-day composite that covers each week's last day.

    ``is_synthetic`` is true if any composite used is synthetic, so the page labels the line honestly.
    """
    start, _ = F.season_bounds(year, season)
    group = (
        satellite[satellite["district_id"] == district_id]
        .sort_values("date")
        .assign(
            is_synthetic=lambda d: d["is_synthetic"].astype(float),
            ndvi=lambda d: d["ndvi"].astype(float),
            # one datetime resolution on both sides of the merge (parquet gives ns, Python timestamps give us)
            date=lambda d: d["date"].astype("datetime64[ns]"),
        )
    )
    ends = pd.DataFrame(
        {"end": pd.to_datetime([start + pd.Timedelta(days=7 * w + 6) for w in range(n_weeks)])}
    ).astype({"end": "datetime64[ns]"})
    merged = pd.merge_asof(
        ends, group[["date", "ndvi", "is_synthetic"]], left_on="end", right_on="date"
    )
    stale = (merged["end"] - merged["date"]).dt.days > 16
    merged.loc[stale, ["ndvi", "is_synthetic"]] = [np.nan, np.nan]
    if merged["ndvi"].isna().all():
        return None
    values = [None if np.isnan(v) else float(np.clip(v, 0.0, 1.0)) for v in merged["ndvi"]]
    return {
        "weekly": values,
        "is_synthetic": bool((merged["is_synthetic"].fillna(0.0) > 0).any()),
    }


def real_relief(lat: float, lon: float) -> dict[str, Any] | None:
    """Elevation lattice and relief of a 10 km window at a district centre (None when the tiles are unavailable)."""
    window = terrain.elevation_window(lat, lon)
    if window is None:
        return None
    return {"elevation_m": np.round(window, 1).tolist(), **terrain.relief_stats(window)}


def select_districts(n: int) -> list[str]:
    """One representative district per climate cluster (largest clusters first), preferring many crops incl. rice."""
    crops = pd.read_parquet(get_path("data_processed") / "crops_annual.parquet")
    counts = crops.pivot_table(
        index="district_id", columns="crop", values="year", aggfunc="count", fill_value=0
    )
    valid = (counts >= MIN_CROP_ROWS).sum(axis=1)
    rice_ok = counts.get("rice", pd.Series(0, index=counts.index)) >= MIN_CROP_ROWS
    clusters = F.load_clusters()
    feats = pd.read_parquet(get_path("data_processed") / "district_clusters.parquet")
    chosen: list[str] = []
    sizes = clusters["cluster_id"].value_counts()
    for cluster in sizes.index:
        members = feats[feats["cluster_id"] == cluster]
        members = members[members["district_id"].map(rice_ok).fillna(False)]
        if members.empty:
            continue
        centre = members[["mean_rain_mm", "mean_tmax_c"]].mean()
        score = members.assign(
            crops=members["district_id"].map(valid),
            dist=((members["mean_rain_mm"] - centre["mean_rain_mm"]) / 100).abs(),
        ).sort_values(["crops", "dist"], ascending=[False, True])
        chosen.append(str(score["district_id"].iloc[0]))
        if len(chosen) == n:
            break
    return chosen


def weekly_records(
    inputs: S.SeasonInputs,
    sim: wb.WaterBalance,
    row: int,
    drought_p: np.ndarray,
    heat_p: np.ndarray,
) -> list[dict[str, float]]:
    """Weekly dict list for one strategy (``row`` of the simulation arrays)."""
    crop = inputs.crop_params
    n_weeks = inputs.n_days // 7
    weekly = lambda a: wb.to_weekly(a, n_weeks)  # noqa: E731
    rain_w, etc_w, eta_w = weekly(inputs.rain), weekly(sim.etc[row]), weekly(sim.eta[row])
    irr_w = weekly(sim.irrigation[row])
    seep_w = weekly(sim.percolation[row]) if sim.percolation is not None else np.zeros(n_weeks)
    pond_w = np.array([sim.pond()[row, 7 * w + 6] for w in range(n_weeks)])
    excess_w = weekly(sim.drainage[row]) + weekly(inputs.rain - sim.peff[row])
    tmax_w = inputs.tmax[: n_weeks * 7].reshape(n_weeks, 7).mean(axis=1)
    moisture = np.array(
        [(sim.taw - sim.depletion[row, 7 * w + 6]) / sim.taw for w in range(n_weeks)]
    )
    stress = np.clip(
        1.0 - np.divide(eta_w, etc_w, out=np.ones_like(eta_w), where=etc_w > 0), 0.0, 1.0
    )
    kc_w = wb.kc_curve(crop, inputs.n_days)[: n_weeks * 7].reshape(n_weeks, 7).mean(axis=1)
    canopy = np.clip(kc_w / crop.kc_mid, 0.0, 1.0)
    smoothed, acc = np.zeros(n_weeks), 0.0
    for w in range(n_weeks):
        acc = 0.5 * acc + 0.5 * stress[w]
        smoothed[w] = acc
    ndvi = np.clip(0.15 + 0.65 * canopy**0.8 * (1.0 - 0.45 * smoothed), 0.0, 1.0)
    return [
        {
            "week": w + 1,
            "rain_mm": float(rain_w[w]),
            "tmax_c": float(tmax_w[w]),
            "soil_moisture_0_1": float(np.clip(moisture[w], 0.0, 1.0)),
            "ndvi": float(ndvi[w]),
            "etc_mm": float(etc_w[w]),
            "eta_mm": float(eta_w[w]),
            "irrigation_mm": float(irr_w[w]),
            "excess_mm": float(max(excess_w[w], 0.0)),
            "stress_0_1": float(stress[w]),
            "pond_mm": float(pond_w[w]),
            "seepage_mm": float(seep_w[w]),
            "drought_p": float(np.clip(drought_p[w], 0.0, 1.0)),
            "heat_p": float(np.clip(heat_p[w], 0.0, 1.0)),
        }
        for w in range(n_weeks)
    ]


def round_floats(obj: Any, digits: int = 3) -> Any:
    """Recursively round floats (keeps the file small and the diff stable)."""
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: round_floats(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, digits) for v in obj]
    return obj


def crop_scenarios(
    district_id: str, crop: str, season: str, year: int, risk: Any, out: dict[str, Any]
) -> None:
    """Fill ``out`` with all grid scenarios of one district-crop."""
    params = load_crop(crop)
    budget = params.default_budget_mm
    for temp in GRID_TEMP:
        ml = predict_yield(district_id, crop, season, {"temperature_delta": float(temp)}, year=year)
        evaluated: dict[float, tuple[S.SeasonInputs, pd.DataFrame, dict]] = {}
        for rain in GRID_RAIN:
            inputs = S.load_season_inputs(
                district_id, crop, season, year, rain_multiplier=rain, temperature_delta=float(temp)
            )
            table, details = S.evaluate_strategies(inputs, budget)
            evaluated[rain] = (inputs, table, details)
        ref_table = evaluated[1.0][1].set_index("strategy")
        reference = float(ref_table.loc["farmer_baseline", "rel_yield"])
        heat = risk.weekly_heat_share(district_id, year, season, float(temp))
        for rain in GRID_RAIN:
            inputs, table, details = evaluated[rain]
            table = table.set_index("strategy")
            drought = risk.weekly_drought_p(district_id, year, season, rain)
            names = list(table.index)
            for strategy in STRATEGIES:
                sim_name = SIM_ORDER[strategy]
                ratio = min(
                    float(table.loc[sim_name, "rel_yield"]) / max(reference, 1e-6), S.MAX_GAIN
                )
                row = names.index(sim_name)
                sim = details["simulation"]
                out[f"{district_id}|{crop}|{rain:g}|{temp:g}|{strategy}"] = {
                    "yield": {
                        "mean": ml["yield_t_ha"] * ratio,
                        "lo": ml["lo"] * ratio,
                        "hi": ml["hi"] * ratio,
                    },
                    "water_used_mm": float(table.loc[sim_name, "water_used_mm"]),
                    "stage_rel": [
                        float(v) for v in wb.stage_factors(sim.eta[row], sim.etc[row], params)[0]
                    ],
                    "weekly": weekly_records(inputs, sim, row, drought, heat),
                }
                if params.ponded:  # daily standing water: the irrigation event on the first day of a week is visible in it
                    out[f"{district_id}|{crop}|{rain:g}|{temp:g}|{strategy}"]["pond_daily"] = (
                        np.round(sim.pond()[row], 1).tolist()
                    )


def build(n_districts: int = 6, districts: list[str] | None = None) -> dict[str, Any]:
    """Assemble the whole data structure."""
    started = time.time()
    settings = load_settings()
    glob = load_globals()
    risk = get_scenario_risk()
    chosen = districts or select_districts(n_districts)
    meta_districts = pd.read_parquet(get_path("data_processed") / "districts.parquet").set_index(
        "district_id"
    )
    clusters = pd.read_parquet(get_path("data_processed") / "district_clusters.parquet").set_index(
        "district_id"
    )
    crops_df = pd.read_parquet(get_path("data_processed") / "crops_annual.parquet")
    satellite = pd.read_parquet(get_path("data_processed") / "satellite_16d.parquet")
    quality = json.loads((get_path("reports") / "data_quality.json").read_text(encoding="utf-8"))
    extreme_meta = json.loads(
        (get_path("reports") / "extreme_weather.json").read_text(encoding="utf-8")
    )
    scenarios: dict[str, Any] = {}
    district_records, crop_info = [], {}
    for district_id in chosen:
        row = meta_districts.loc[district_id]
        available = []
        reference_year, budget, taw = {}, {}, {}
        ref_daily, ref_ndvi = {}, {}
        for crop in settings["crops"]:
            season = settings["crop_season"][crop]
            rows = ((crops_df["district_id"] == district_id) & (crops_df["crop"] == crop)).sum()
            if rows < MIN_CROP_ROWS:
                continue
            year = S.typical_year(district_id, season)
            try:
                predict_yield(district_id, crop, season, None, year=year)
            except ValueError as exc:
                LOG.warning("skipping %s %s: %s", district_id, crop, exc)
                continue
            available.append(crop)
            reference_year[crop] = year
            budget[crop] = load_crop(crop).default_budget_mm
            taw[crop] = taw_mm(load_crop(crop))
            cp = load_crop(crop)
            growth, ripening = development_curves(list(cp.stage_days))
            crop_info[crop] = {
                "season": season,
                "note": YIELD_NOTES.get(crop, ""),
                "stage_days": list(cp.stage_days),
                "season_start": f"{F.season_bounds(2000, season)[0]:%m-%d}",
                "ponded": cp.ponded,
                "pond_max_mm": cp.pond_max_mm,
                "percolation_mm_day": cp.percolation_mm_day,
                "initial_pond_mm": cp.initial_pond_mm,
                "development": growth,
                "senescence": ripening,
            }
            ref_daily[crop] = reference_daily(S.load_season_inputs(district_id, crop, season, year))
            ref_ndvi[crop] = reference_ndvi(satellite, district_id, year, season)
            LOG.info("%s %s (year %d)", district_id, crop, year)
            crop_scenarios(district_id, crop, season, year, risk, scenarios)
        warnings_ew = {}
        for season in sorted({settings["crop_season"][c] for c in available}):
            year = next(
                reference_year[c] for c in available if settings["crop_season"][c] == season
            )
            start, _ = F.season_bounds(year, season)
            as_of = (start + pd.Timedelta(days=56)).strftime("%Y-%m-%d")
            warnings_ew[season] = risk_forecast(district_id, as_of)
        relief = real_relief(float(row["centroid_lat"]), float(row["centroid_lon"]))
        district_records.append(
            {
                "id": district_id,
                "name": str(row["district"]),
                "state": str(row["state"]),
                "cluster": int(clusters.loc[district_id, "cluster_id"]),
                "cluster_label": str(clusters.loc[district_id, "cluster_label"]),
                "lat": float(row["centroid_lat"]),
                "lon": float(row["centroid_lon"]),
                "area_km2": float(row["area_km2"]),
                "elevation_proxy": elevation_proxy(district_id),
                **(
                    {
                        "elevation_m": relief["elevation_m"],
                        "mean_elevation_m": relief["mean_elevation_m"],
                        "relief_m": relief["relief_m"],
                    }
                    if relief
                    else {}
                ),
                "elevation_source": terrain.ATTRIBUTION
                if relief
                else "procedural (real relief unavailable)",
                "tiles": soil_tiles(district_id),
                "reference_daily": ref_daily,
                "reference_ndvi": ref_ndvi,
                "crops": available,
                "reference_year": reference_year,
                "budget_mm": budget,
                "taw_mm": taw,
                "early_warning": warnings_ew,
            }
        )
    heat_served = extreme_meta["heatwave"].get("served", {}).get("source", "unknown")
    data = {
        "meta": {
            "schema_version": 2,
            "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "is_synthetic_inputs": {
                "weather": quality["synthetic_share"].get("weather_daily", 0) > 0,
                "crop_labels": quality["synthetic_share"].get("crops_annual", 0) > 0,
                "satellite_dataset": quality["synthetic_share"].get("satellite_16d", 0) > 0,
                "satellite_used_by_served_models": False,
            },
            "model_versions": {
                "yield": "gradient-boosted trees on weather only, temperature-monotone",
                "drought": f"gradient-boosted trees on {extreme_meta['drought']['served_feature_set']} features, calibrated",
                "heatwave_early_warning": f"probability from {heat_served}",
                "water_balance": "FAO-56 daily root-zone balance, Hargreaves-Samani ET0",
                "irrigation": "weekly linear programme, end-of-week stress basis",
            },
            "weeks": glob.n_weeks,
            "risk_levels": settings["risk_levels"],
            "grid": {"rain_mult": GRID_RAIN, "temp_delta": GRID_TEMP, "strategies": STRATEGIES},
            "definitions": {**DEFINITIONS, "heat_p_basis": heat_threshold_note()},
            "disclaimer": DISCLAIMER,
            "attribution": [terrain.ATTRIBUTION],
        },
        "crops": crop_info,
        "districts": district_records,
        "scenarios": scenarios,
    }
    LOG.info(
        "built %d scenarios for %d districts in %.0fs",
        len(scenarios),
        len(district_records),
        time.time() - started,
    )
    return data


def validate(data: dict[str, Any]) -> None:
    """Validate against the JSON Schema and check grid completeness; raises on any problem."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(data), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        raise ValueError(
            f"sim_data.json violates the schema at {'/'.join(map(str, first.path)) or '<root>'}: {first.message}"
        )
    grid = data["meta"]["grid"]
    for district in data["districts"]:
        for crop in district["crops"]:
            for rain in grid["rain_mult"]:
                for temp in grid["temp_delta"]:
                    for strategy in grid["strategies"]:
                        key = f"{district['id']}|{crop}|{rain:g}|{temp:g}|{strategy}"
                        if key not in data["scenarios"]:
                            raise ValueError(f"missing scenario {key}")


def write(data: dict[str, Any], path: Path = OUT_PATH) -> Path:
    """Round, validate and write the JSON (compact separators); enforces the 15 MB limit."""
    data = round_floats(data)
    validate(data)
    text = json.dumps(data, separators=(",", ":"))
    if len(text.encode()) > MAX_BYTES:
        raise ValueError(f"sim_data.json is {len(text) / 1e6:.1f} MB, above the 15 MB limit")
    path.write_text(text, encoding="utf-8")
    LOG.info("wrote %s (%.1f MB)", path, len(text) / 1e6)
    # A small index of the exported districts lets the dashboards map any district to a simulated one without
    # loading the full scenario grid.
    index = [
        {key: d[key] for key in ("id", "name", "state", "cluster", "lat", "lon", "crops")}
        for d in data["districts"]
    ]
    (path.parent / DISTRICT_INDEX).write_text(
        json.dumps(index, ensure_ascii=False), encoding="utf-8"
    )
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--districts",
        type=int,
        default=6,
        help="number of representative districts (one per cluster)",
    )
    parser.add_argument(
        "--ids", nargs="+", help="explicit district ids instead of the automatic selection"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    write(build(args.districts, args.ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
