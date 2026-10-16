"""Season scenarios: build weather inputs, score irrigation strategies with the daily simulator, combine with the ML yield.

``evaluate_strategies`` is the single place where schedules are compared: the LP's weekly plan, the even-split
and the farmer baseline are all scored by the *same* nonlinear daily water balance, so the comparison is fair and
no number is hard-coded.

Yield combination (docs/DECISIONS.md D-020): the ML model already responds to rainfall, so to avoid counting water
stress twice we take its prediction at **normal rainfall** (scenario temperature only) as the district's
typical-practice yield, and scale it by the water-stress ratio

    yield(strategy, scenario) = ML(normal rain, scenario T) x RYF(strategy, scenario) / RYF(farmer baseline, normal rain, scenario T)

so that the farmer baseline at normal rainfall reproduces the ML yield, droughts lower it and better irrigation
raises it. The ratio is capped at ``MAX_GAIN`` because no schedule can add more than that over typical practice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from analytics.prescriptive import irrigation_lp as lp
from analytics.prescriptive import water_balance as wb
from analytics.prescriptive.params import CropParams, Globals, load_crop, load_globals, taw_mm
from config import get_path, load_settings

LOG = logging.getLogger(__name__)

MAX_GAIN = 1.5
STRATEGIES = ("rainfed", "farmer_baseline", "even_split", "optimized")


@dataclass
class SeasonInputs:
    """Daily weather for the simulated window of one district-season-year, already scaled by the scenario."""

    district_id: str
    crop: str
    season: str
    year: int
    lat: float
    rain: np.ndarray  # mm/day, scenario applied
    tmax: np.ndarray
    tmin: np.ndarray
    et0: np.ndarray
    dates: pd.DatetimeIndex
    rain_multiplier: float
    temperature_delta: float
    soil_class: str
    crop_params: CropParams
    glob: Globals

    @property
    def taw(self) -> float:
        return taw_mm(self.crop_params, self.soil_class)

    @property
    def n_days(self) -> int:
        return len(self.rain)


@lru_cache(maxsize=1)
def _weather() -> pd.DataFrame:
    return pd.read_parquet(get_path("data_processed") / "weather_daily.parquet").sort_values(
        ["district_id", "date"]
    )


@lru_cache(maxsize=1)
def _districts() -> pd.DataFrame:
    return pd.read_parquet(get_path("data_processed") / "districts.parquet").set_index(
        "district_id"
    )


def typical_year(district_id: str, season: str) -> int:
    """Training year whose season rainfall is the median (same rule as ``analytics.predictive.predict.typical_year``)."""
    from analytics.predictive import features as F

    settings = load_settings()
    first, last = int(settings["years"]["start"]), int(settings["splits"]["yield"]["train_end"])
    weather = _weather()
    group = weather[weather["district_id"] == district_id]
    totals = {}
    for year in range(first, last + 1):
        start, end = F.season_bounds(year, season)
        window = group[(group["date"] >= start) & (group["date"] <= end)]
        if len(window) == (end - start).days + 1:
            totals[year] = float(window["rain_mm"].sum())
    median = float(np.median(list(totals.values())))
    return min(totals, key=lambda y: abs(totals[y] - median))


def load_season_inputs(
    district_id: str,
    crop: str,
    season: str,
    year: int | None = None,
    rain_multiplier: float = 1.0,
    temperature_delta: float = 0.0,
    soil_class: str | None = None,
) -> SeasonInputs:
    """Weather of the first 16 weeks of ``season`` in ``year`` (default: the typical year), with scenario changes."""
    from analytics.predictive import features as F

    glob = load_globals()
    districts = _districts()
    if district_id not in districts.index:
        raise ValueError(f"unknown district_id {district_id!r}")
    year = typical_year(district_id, season) if year is None else int(year)
    start, _ = F.season_bounds(year, season)
    weather = _weather()
    group = weather[(weather["district_id"] == district_id) & (weather["date"] >= start)].head(
        glob.season_days
    )
    if len(group) < glob.season_days:
        raise ValueError(f"not enough weather after {start:%Y-%m-%d} for {district_id}")
    rain = group["rain_mm"].to_numpy(float) * rain_multiplier
    tmax = group["tmax_c"].to_numpy(float) + temperature_delta
    tmin = group["tmin_c"].to_numpy(float) + temperature_delta
    lat = float(districts.loc[district_id, "centroid_lat"])
    ra = wb.extraterrestrial_radiation(lat, group["date"].dt.dayofyear.to_numpy())
    et0 = wb.hargreaves_et0(tmax, tmin, ra)
    return SeasonInputs(
        district_id, crop, season, year, lat, rain, tmax, tmin, et0, pd.DatetimeIndex(group["date"]),
        rain_multiplier, temperature_delta, soil_class or glob.default_soil_class, load_crop(crop), glob,
    )  # fmt: skip


def lp_inputs(inputs: SeasonInputs) -> lp.LPInputs:
    """Weekly planning inputs from the daily scenario weather (unstressed ETc, effective rain)."""
    kc = wb.kc_curve(inputs.crop_params, inputs.n_days)
    etc = kc * inputs.et0
    peff = wb.effective_rainfall(inputs.rain, inputs.glob.rain_efficiency)
    return lp.build_inputs(
        inputs.crop_params,
        etc,
        peff,
        inputs.taw,
        inputs.crop_params.initial_depletion_mm(inputs.taw, inputs.glob.initial_depletion_fraction),
    )


def simulate_schedules(inputs: SeasonInputs, daily_irrigation: np.ndarray) -> wb.WaterBalance:
    """Daily water balance for one or several irrigation schedules ``(n, T)`` under the scenario weather."""
    kc = wb.kc_curve(inputs.crop_params, inputs.n_days)
    n = np.atleast_2d(daily_irrigation).shape[0]
    return wb.simulate(
        np.tile(inputs.et0, (n, 1)),
        np.tile(inputs.rain, (n, 1)),
        daily_irrigation,
        kc,
        inputs.taw,
        inputs.crop_params.p,
        initial_depletion=inputs.crop_params.initial_depletion_mm(
            inputs.taw, inputs.glob.initial_depletion_fraction
        ),
        glob=inputs.glob,
        pond_max=inputs.crop_params.pond_max_mm,
        percolation=inputs.crop_params.percolation_mm_day,
    )


def evaluate_strategies(
    inputs: SeasonInputs,
    budget_mm: float,
    weekly_cap_mm: float | None = None,
    stress_basis: str = "end",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score rainfed, farmer baseline, even split and the LP schedule with the daily simulator.

    Returns:
        (table with one row per strategy: water_used_mm, eta_mm, etc_mm, rel_yield, drainage_mm, plus the weekly
        irrigation as a list; details with the LP result and inputs).
    """
    cap = inputs.glob.weekly_cap_mm if weekly_cap_mm is None else weekly_cap_mm
    n_days, n_weeks = inputs.n_days, inputs.n_days // 7
    planning = lp_inputs(inputs)
    result = lp.solve_irrigation(planning, budget_mm, cap, stress_basis)
    weekly = {
        "rainfed": np.zeros(n_weeks),
        "even_split": lp.even_split(budget_mm, cap, n_weeks),
        "optimized": result.schedule_mm,
    }
    daily = {name: wb.weekly_to_daily_irrigation(w, n_days) for name, w in weekly.items()}
    daily["farmer_baseline"] = lp.fixed_interval_daily(n_days)
    weekly["farmer_baseline"] = wb.to_weekly(daily["farmer_baseline"], n_weeks)
    order = list(STRATEGIES)
    sim = simulate_schedules(inputs, np.vstack([daily[s] for s in order]))
    ryf = wb.relative_yield(sim.eta, sim.etc, inputs.crop_params)
    table = pd.DataFrame(
        {
            "strategy": order,
            "water_used_mm": sim.irrigation.sum(axis=1),
            "eta_mm": sim.eta.sum(axis=1),
            "etc_mm": sim.etc.sum(axis=1),
            "drainage_mm": sim.drainage.sum(axis=1),
            "rel_yield": ryf,
            "weekly_irrigation_mm": [weekly[s].round(2).tolist() for s in order],
        }
    )
    table["eta_over_etc"] = table["eta_mm"] / table["etc_mm"]
    base_water = float(table.loc[table["strategy"] == "farmer_baseline", "water_used_mm"].iloc[0])
    table["water_saved_vs_baseline_pct"] = (
        100.0 * (base_water - table["water_used_mm"]) / base_water if base_water > 0 else np.nan
    )
    return table, {"lp": result, "planning": planning, "simulation": sim, "weekly": weekly}


def strategy_yields(
    inputs: SeasonInputs, budget_mm: float, ml: dict[str, float], weekly_cap_mm: float | None = None
) -> pd.DataFrame:
    """Strategy table plus yield mean / interval = ML potential x water-stress ratio (see module docstring).

    Args:
        inputs: scenario weather (rain multiplier and temperature delta already applied).
        budget_mm: seasonal irrigation budget.
        ml: ``predict_yield`` output at **normal rainfall** and the scenario temperature (keys yield_t_ha, lo, hi).
    """
    table, _ = evaluate_strategies(inputs, budget_mm, weekly_cap_mm)
    normal = load_season_inputs(
        inputs.district_id,
        inputs.crop,
        inputs.season,
        inputs.year,
        1.0,
        inputs.temperature_delta,
        inputs.soil_class,
    )
    ref_table, _ = evaluate_strategies(normal, budget_mm, weekly_cap_mm)
    reference = float(
        ref_table.loc[ref_table["strategy"] == "farmer_baseline", "rel_yield"].iloc[0]
    )
    ratio = np.minimum(table["rel_yield"].to_numpy() / max(reference, 1e-6), MAX_GAIN)
    table["yield_t_ha"] = ml["yield_t_ha"] * ratio
    table["yield_lo"] = ml["lo"] * ratio
    table["yield_hi"] = ml["hi"] * ratio
    table["reference_rel_yield"] = reference
    return table
