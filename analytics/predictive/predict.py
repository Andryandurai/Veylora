"""Serving API for yield forecasts: ``predict_yield`` and ``predict_batch``.

``predict_yield(district_id, crop, season, overrides=None)`` returns::

    {"yield_t_ha": float, "lo": float, "hi": float, "model": str, ...extra context...}

``lo``/``hi`` form an 80 percent conformal interval. ``overrides`` is what the what-if dashboard and the
3D simulator use to change the weather the model sees:

* ``rain_multiplier`` (0 .. 3): multiplies every daily rainfall value of the season;
* ``temperature_delta`` (-10 .. +10 C): added to daily Tmax and Tmin.

Overrides are applied to the **daily** weather of the reference year and the season features are then
recomputed with the same code as the SQL view, so a 1.2 rain multiplier changes rain totals, rainy days,
dry spells and the derived anomalies consistently. The serving model uses weather features only
(NDVI / soil moisture cannot be recomputed under a scenario, docs/DECISIONS.md D-013).

Reference year: by default the district-season's "typical year", the training year (<= train_end) whose season
rainfall is the median, so no future information enters; pass ``year=`` to use a specific year. The forecast is
issued ``lead_days`` (30, 60 or 90) after the season starts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd

from analytics.predictive import features as F
from config import get_path, load_settings

LOG = logging.getLogger(__name__)

DEFAULT_LEAD = 60
RAIN_MULTIPLIER_RANGE = (0.0, 3.0)
TEMPERATURE_DELTA_RANGE = (-10.0, 10.0)
ARTIFACT = "yield_xgb.joblib"


class ModelNotTrainedError(RuntimeError):
    """models/yield_xgb.joblib is missing; run ``make train-yield``."""


@dataclass
class _Context:
    """Everything needed to build feature rows quickly (loaded once per process)."""

    prepared: pd.DataFrame  # history + anomaly columns, all years
    seasons: pd.DataFrame  # season_table (weather only) for climatologies
    daily: dict[str, dict[str, np.ndarray]]  # per district: dates, rain, tmax, tmin
    clusters: dict[str, int]
    states: dict[str, str]
    levels: dict[str, list]
    source: str


@lru_cache(maxsize=1)
def _bundle() -> dict[str, Any]:
    path = get_path("models") / ARTIFACT
    if not path.exists():
        raise ModelNotTrainedError(
            f"{path} not found: run `make train-yield` (or `python -m analytics.predictive.train_yield`)"
        )
    return joblib.load(path)


@lru_cache(maxsize=1)
def _context() -> _Context:
    frame, source = F.load_training_frame()
    prepared = F.prepare_frame(frame)
    weather = F.load_weather_daily().sort_values(["district_id", "date"])
    daily = {
        d: {
            "dates": g["date"].to_numpy("datetime64[ns]"),
            "rain": g["rain_mm"].to_numpy(float),
            "tmax": g["tmax_c"].to_numpy(float),
            "tmin": g["tmin_c"].to_numpy(float),
        }
        for d, g in weather.groupby("district_id")
    }
    seasons = F.season_table(weather, None)
    clusters_df = F.load_clusters()
    clusters = (
        dict(zip(clusters_df["district_id"], clusters_df["cluster_id"], strict=True))
        if clusters_df is not None
        else {}
    )
    states = dict(zip(prepared["district_id"], prepared["state"], strict=True))
    return _Context(prepared, seasons, daily, clusters, states, _bundle()["levels"], source)


def reset_cache() -> None:
    """Forget loaded models and data (tests, or after retraining)."""
    _bundle.cache_clear()
    _context.cache_clear()


def _validate_overrides(overrides: dict[str, float] | None) -> tuple[float, float]:
    overrides = overrides or {}
    unknown = set(overrides) - {"rain_multiplier", "temperature_delta"}
    if unknown:
        raise ValueError(
            f"unknown override(s) {sorted(unknown)}; allowed: rain_multiplier, temperature_delta"
        )
    mult = float(overrides.get("rain_multiplier", 1.0))
    delta = float(overrides.get("temperature_delta", 0.0))
    if not RAIN_MULTIPLIER_RANGE[0] <= mult <= RAIN_MULTIPLIER_RANGE[1]:
        raise ValueError(f"rain_multiplier must be within {RAIN_MULTIPLIER_RANGE}, got {mult}")
    if not TEMPERATURE_DELTA_RANGE[0] <= delta <= TEMPERATURE_DELTA_RANGE[1]:
        raise ValueError(
            f"temperature_delta must be within {TEMPERATURE_DELTA_RANGE} C, got {delta}"
        )
    return mult, delta


def typical_year(ctx: _Context, district_id: str, season: str) -> int:
    """Training year whose season rainfall is the median for the district-season (no test-year information)."""
    train_end = int(load_settings()["splits"]["yield"]["train_end"])
    first = int(load_settings()["years"]["start"])
    rows = ctx.seasons[
        (ctx.seasons["district_id"] == district_id)
        & (ctx.seasons["season"] == season)
        & ctx.seasons["year"].between(first, train_end)
    ]
    if rows.empty:
        raise ValueError(f"no weather for {district_id} / {season}")
    median = rows["total_rain"].median()
    return int(rows.iloc[(rows["total_rain"] - median).abs().argsort().iloc[0]]["year"])


def _window(
    ctx: _Context, district_id: str, year: int, season: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    entry = ctx.daily.get(district_id)
    if entry is None:
        raise ValueError(f"unknown district_id {district_id!r}")
    start, end = F.season_bounds(year, season)
    lo = int(np.searchsorted(entry["dates"], np.datetime64(start, "ns"), side="left"))
    hi = int(np.searchsorted(entry["dates"], np.datetime64(end, "ns"), side="right"))
    if hi - lo != (end - start).days + 1:
        raise ValueError(f"no complete {season} weather for {district_id} in {year}")
    return entry["rain"][lo:hi], entry["tmax"][lo:hi], entry["tmin"][lo:hi]


def _history(
    ctx: _Context, district_id: str, crop: str, season: str, year: int
) -> dict[str, float]:
    hist = ctx.prepared[
        (ctx.prepared["district_id"] == district_id)
        & (ctx.prepared["crop"] == crop)
        & (ctx.prepared["season"] == season)
        & (ctx.prepared["year"] < year)
    ].sort_values("year")
    if len(hist) < F.MIN_PRIOR_YEARS:
        raise ValueError(
            f"not enough yield history for {district_id} / {crop} / {season} before {year} (need {F.MIN_PRIOR_YEARS}, have {len(hist)})"
        )
    years, values = hist["year"].to_numpy(), hist["yield_label"].to_numpy(float)
    last3 = hist[hist["year"] >= year - 3]["yield_label"]
    trend = np.nan
    if len(hist) >= F.MIN_TREND_YEARS:
        slope, intercept = np.polyfit(years.astype(float), values, 1)
        trend = intercept + slope * year
    return {
        "n_prior": float(len(hist)),
        "yield_mean_prior_all": float(values.mean()),
        "prev_year_yield": float(values[-1]) if years[-1] == year - 1 else np.nan,
        "yield_mean_prev3": float(last3.mean()) if len(last3) else np.nan,
        "yield_trend_pred": float(trend),
    }


def _climatology(ctx: _Context, district_id: str, season: str, year: int) -> dict[str, float]:
    prior = ctx.seasons[
        (ctx.seasons["district_id"] == district_id)
        & (ctx.seasons["season"] == season)
        & (ctx.seasons["year"] < year)
    ]
    if len(prior) < F.MIN_CLIMATOLOGY_YEARS:
        return {}
    return prior.mean(numeric_only=True).to_dict()


def feature_frame(ctx: _Context, request: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    """One-row ``prepared``-style frame for a request (applies overrides to daily weather)."""
    district_id, crop, season = request["district_id"], request["crop"], request["season"]
    mult, delta = _validate_overrides(request.get("overrides"))
    year = (
        int(request["year"])
        if request.get("year") is not None
        else typical_year(ctx, district_id, season)
    )
    rain, tmax, tmin = _window(ctx, district_id, year, season)
    row: dict[str, Any] = {
        "district_id": district_id,
        "state": ctx.states.get(district_id, ""),
        "crop": crop,
        "season": season,
        "year": year,
    }
    row.update(F.window_features(rain * mult, tmax + delta, tmin + delta))
    row.update(_history(ctx, district_id, crop, season, year))
    clim = _climatology(ctx, district_id, season, year)
    for lead in F.LEADS:
        row[f"rain_d{lead}_ratio"] = (
            row[f"rain_d{lead}"] / clim[f"rain_d{lead}"] if clim else np.nan
        )
        row[f"mean_tmax_d{lead}_anom"] = (
            row[f"mean_tmax_d{lead}"] - clim[f"mean_tmax_d{lead}"] if clim else np.nan
        )
    row["cluster_id"] = ctx.clusters.get(district_id, np.nan)
    return pd.DataFrame([row]), year


def predict_batch(
    requests: pd.DataFrame | list[dict[str, Any]], lead_days: int = DEFAULT_LEAD
) -> pd.DataFrame:
    """Forecast many scenarios at once.

    Args:
        requests: rows with ``district_id``, ``crop``, ``season`` and optionally ``rain_multiplier``,
            ``temperature_delta`` and ``year``.
        lead_days: forecast lead (30, 60 or 90).

    Returns:
        DataFrame with the request columns plus ``yield_t_ha``, ``lo``, ``hi``, ``reference_year``,
        ``scale_t_ha`` (prior mean yield) and ``model``.
    """
    if lead_days not in F.LEADS:
        raise ValueError(f"lead_days must be one of {F.LEADS}")
    bundle = _bundle()
    ctx = _context()
    art = bundle["leads"][lead_days]
    records = requests.to_dict("records") if isinstance(requests, pd.DataFrame) else list(requests)
    frames, years = [], []
    for rec in records:
        overrides = {
            k: rec[k]
            for k in ("rain_multiplier", "temperature_delta")
            if rec.get(k) is not None and not pd.isna(rec.get(k))
        }
        frame, year = feature_frame(ctx, {**rec, "overrides": rec.get("overrides") or overrides})
        frames.append(frame)
        years.append(year)
    prepared = pd.concat(frames, ignore_index=True)
    X, _, meta = F.build_features(
        prepared.assign(yield_label=np.nan), lead_days, include_satellite=False, levels=ctx.levels
    )
    if len(X) != len(prepared):
        raise ValueError("some requests lack enough yield history")
    X = X[art["columns"]]
    point = art["model"].predict(X)
    quant = art["conformal_model"].predict(X)
    lo = np.minimum(np.minimum(quant[:, 0], quant[:, 1]) - art["conformal_qhat"], point)
    hi = np.maximum(np.maximum(quant[:, 0], quant[:, 1]) + art["conformal_qhat"], point)
    scale = meta["scale"].to_numpy()
    out = pd.DataFrame(records)
    out["yield_t_ha"] = point * scale
    out["lo"] = np.maximum(lo, 0.0) * scale
    out["hi"] = hi * scale
    out["reference_year"] = years
    out["scale_t_ha"] = scale
    out["model"] = f"xgboost[{bundle['feature_set']},lead{lead_days}]"
    return out


def predict_yield(
    district_id: str,
    crop: str,
    season: str,
    overrides: dict[str, float] | None = None,
    *,
    year: int | None = None,
    lead_days: int = DEFAULT_LEAD,
) -> dict[str, Any]:
    """Point forecast and 80 percent interval for one district-crop-season (t/ha).

    Args:
        district_id: e.g. ``tamil_nadu__thanjavur``.
        crop: rice, wheat, cotton or sugarcane.
        season: kharif or rabi (must be the season the crop has yield history for).
        overrides: optional ``{"rain_multiplier": x, "temperature_delta": dC}`` what-if changes.
        year: reference year whose weather is used (default: the district-season's typical training year).
        lead_days: 30, 60 or 90.

    Returns:
        ``{"yield_t_ha", "lo", "hi", "model"}`` plus ``reference_year``, ``lead_days``, ``scale_t_ha``,
        ``overrides`` for context.
    """
    mult, delta = _validate_overrides(overrides)
    row = predict_batch(
        [
            {
                "district_id": district_id,
                "crop": crop,
                "season": season,
                "rain_multiplier": mult,
                "temperature_delta": delta,
                "year": year,
            }
        ],
        lead_days,
    ).iloc[0]
    return {
        "yield_t_ha": float(row["yield_t_ha"]),
        "lo": float(row["lo"]),
        "hi": float(row["hi"]),
        "model": str(row["model"]),
        "reference_year": int(row["reference_year"]),
        "lead_days": int(lead_days),
        "scale_t_ha": float(row["scale_t_ha"]),
        "overrides": {"rain_multiplier": mult, "temperature_delta": delta},
    }


SCENARIOS: dict[str, tuple[float, float]] = {
    "rain x0.6": (0.6, 0.0),
    "rain x0.8": (0.8, 0.0),
    "rain x1.2": (1.2, 0.0),
    "rain x1.4": (1.4, 0.0),
    "-2 C": (1.0, -2.0),
    "+1 C": (1.0, 1.0),
    "+2 C": (1.0, 2.0),
    "+4 C": (1.0, 4.0),
}


def scenario_response(
    years: tuple[int, int] | None = None, lead_days: int = DEFAULT_LEAD
) -> pd.DataFrame:
    """Average what-if response of the served model, measured on real district-years.

    For every district-crop-season-year in ``years`` (default: the validation years, never the test
    years) the baseline forecast is compared with each scenario of :data:`SCENARIOS`. Returns one row
    per scenario and crop (plus ``all``) with the mean change in t/ha and in percent, and the share of
    cases where the forecast rises. Used to check that scenarios behave sensibly and reported in
    ``reports/yield_model_comparison.md``.
    """
    ctx = _context()
    split = load_settings()["splits"]["yield"]
    lo, hi = years or tuple(split["valid"])
    rows = (
        ctx.prepared[
            ctx.prepared["year"].between(lo, hi) & (ctx.prepared["n_prior"] >= F.MIN_PRIOR_YEARS)
        ][["district_id", "crop", "season", "year"]]
        .drop_duplicates()
        .to_dict("records")
    )
    base = predict_batch(rows, lead_days)["yield_t_ha"].to_numpy()
    records = []
    for label, (mult, delta) in SCENARIOS.items():
        scen = predict_batch(
            [{**r, "rain_multiplier": mult, "temperature_delta": delta} for r in rows], lead_days
        )["yield_t_ha"].to_numpy()
        frame = pd.DataFrame(
            {
                "crop": [r["crop"] for r in rows],
                "change": scen - base,
                "pct": 100.0 * (scen - base) / base,
            }
        )
        for crop, group in [("all", frame), *frame.groupby("crop")]:
            records.append(
                {
                    "scenario": label,
                    "crop": crop,
                    "n": int(len(group)),
                    "mean_change_t_ha": float(group["change"].mean()),
                    "mean_change_pct": float(group["pct"].mean()),
                    "share_increase": float((group["change"] > 1e-9).mean()),
                }
            )
    return pd.DataFrame(records)
