"""Leakage-safe features for yield forecasting.

Everything here is computed from information available at the *forecast date* = season start + L days
(L = 30, 60 or 90): weather of season days 0..L-1, satellite composites that had *finished* by then
(a MODIS composite dated d covers d..d+15), yields of strictly earlier crop years and climatologies
built only from earlier years.

The module contains

* a Python re-implementation of the SQL season aggregates (``database/views.sql``). It is the engine
  behind what-if overrides (rain multiplier, temperature shift) and the fallback when PostgreSQL is
  down; ``tests/test_predictive.py`` checks it against the database view;
* history features (previous-year yield, expanding means, a trend extrapolation);
* ``build_features(df, lead_days)`` returning ``X, y, meta`` where ``y`` is yield divided by the
  district-crop's prior mean yield (a scale-free target, so crops with different yield levels pool).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from config import get_path, load_settings

LOG = logging.getLogger(__name__)

LEADS = (30, 60, 90)
RAIN_DAY_MM = 2.5
HEAT_TMAX_C = 40.0
GDD_BASE_C = 10.0
COMPOSITE_DAYS_AFTER_START = 15
MIN_PRIOR_YEARS = 3
MIN_TREND_YEARS = 6
MIN_CLIMATOLOGY_YEARS = 5
N_WEEKS = 16

WEATHER_STEMS = ("rain", "rainy_days", "dry_spell", "mean_tmax", "heat_days", "gdd")
SEASON_COLUMNS = (
    "total_rain",
    "rainy_days",
    "longest_dry_spell_days",
    "mean_tmax",
    "heat_days",
    "growing_degree_days",
)


# ----------------------------------------------------------------------------- season aggregates
def longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of True values (0 if none)."""
    if mask.size == 0 or not mask.any():
        return 0
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    delta = np.diff(padded)
    starts, ends = np.flatnonzero(delta == 1), np.flatnonzero(delta == -1)
    return int((ends - starts).max())


def window_features(rain: np.ndarray, tmax: np.ndarray, tmin: np.ndarray) -> dict[str, float]:
    """Season aggregates (and the first-L-days versions) for one season given daily arrays from day 0.

    Mirrors ``v_season_weather``: rainy day >= 2.5 mm, dry day < 2.5 mm, heat day Tmax >= 40 C,
    growing degree days = sum(max((Tmax+Tmin)/2 - 10, 0)).
    """
    dry = rain < RAIN_DAY_MM
    gdd_daily = np.maximum((tmax + tmin) / 2.0 - GDD_BASE_C, 0.0)
    out = {
        "total_rain": float(rain.sum()),
        "rainy_days": float((rain >= RAIN_DAY_MM).sum()),
        "longest_dry_spell_days": float(longest_run(dry)),
        "mean_tmax": float(tmax.mean()),
        "heat_days": float((tmax >= HEAT_TMAX_C).sum()),
        "growing_degree_days": float(gdd_daily.sum()),
    }
    for lead in LEADS:
        sl = slice(0, lead)
        out[f"rain_d{lead}"] = float(rain[sl].sum())
        out[f"rainy_days_d{lead}"] = float((rain[sl] >= RAIN_DAY_MM).sum())
        out[f"dry_spell_d{lead}"] = float(longest_run(dry[sl]))
        out[f"mean_tmax_d{lead}"] = float(tmax[sl].mean())
        out[f"heat_days_d{lead}"] = float((tmax[sl] >= HEAT_TMAX_C).sum())
        out[f"gdd_d{lead}"] = float(gdd_daily[sl].sum())
    return out


def season_bounds(
    year: int, season: str, seasons: dict | None = None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last day of ``season`` in crop year ``year`` (rabi ends in the following year)."""
    cfg = (seasons or load_settings()["seasons"])[season]
    start = pd.Timestamp(year=year, month=cfg["start_month"], day=1)
    end_year = year + 1 if cfg["end_month"] < cfg["start_month"] else year
    end = pd.Timestamp(year=end_year, month=cfg["end_month"], day=1) + pd.offsets.MonthEnd(0)
    return start, end


def _satellite_lookup(sat: pd.DataFrame | None) -> dict[str, dict[str, np.ndarray]]:
    """Per-district arrays for fast 'latest finished composite' queries."""
    if sat is None or sat.empty:
        return {}
    lookup = {}
    for district, group in sat.sort_values("date").groupby("district_id"):
        valid = group[group["ndvi"].notna()]
        lookup[district] = {
            "finished": (valid["date"] + pd.Timedelta(days=COMPOSITE_DAYS_AFTER_START)).to_numpy(
                "datetime64[ns]"
            ),
            "ndvi": valid["ndvi"].to_numpy(float),
            "soil": valid["soil_moisture"].to_numpy(float),
            "all_dates": group["date"].to_numpy("datetime64[ns]"),
            "all_ndvi": group["ndvi"].to_numpy(float),
            "all_soil": group["soil_moisture"].to_numpy(float),
            "all_synth": group["is_synthetic"].to_numpy(bool),
        }
    return lookup


def satellite_features(
    lookup: dict, district_id: str, start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, float]:
    """Satellite columns of ``v_season_weather`` for one season window."""
    names = [("ndvi_at_sowing", "soil_moisture_at_sowing", 0)] + [
        (f"ndvi_d{lead}", f"soil_moisture_d{lead}", lead) for lead in LEADS
    ]
    out: dict[str, float] = {}
    entry = lookup.get(district_id)
    for ndvi_name, soil_name, offset in names:
        out[ndvi_name] = out[soil_name] = np.nan
        if entry is None:
            continue
        cutoff = np.datetime64(start + pd.Timedelta(days=offset), "ns")
        idx = int(np.searchsorted(entry["finished"], cutoff, side="right")) - 1
        if idx >= 0:
            out[ndvi_name], out[soil_name] = float(entry["ndvi"][idx]), float(entry["soil"][idx])
    out["ndvi_peak"] = out["soil_moisture_mean"] = np.nan
    out["satellite_is_synthetic"] = False
    if entry is not None:
        inside = (entry["all_dates"] >= np.datetime64(start, "ns")) & (
            entry["all_dates"] <= np.datetime64(end, "ns")
        )
        if inside.any():
            ndvi = entry["all_ndvi"][inside]
            soil = entry["all_soil"][inside]
            out["ndvi_peak"] = float(np.nanmax(ndvi)) if np.isfinite(ndvi).any() else np.nan
            out["soil_moisture_mean"] = (
                float(np.nanmean(soil)) if np.isfinite(soil).any() else np.nan
            )
            out["satellite_is_synthetic"] = bool(entry["all_synth"][inside].any())
    return out


def season_table(
    weather: pd.DataFrame, satellite: pd.DataFrame | None, seasons: dict | None = None
) -> pd.DataFrame:
    """Python equivalent of ``v_season_weather`` (complete seasons only).

    Args:
        weather: columns district_id, date, rain_mm, tmax_c, tmin_c.
        satellite: columns district_id, date, ndvi, soil_moisture, is_synthetic (or None).
    """
    seasons = seasons or load_settings()["seasons"]
    lookup = _satellite_lookup(satellite)
    rows = []
    for district, group in weather.sort_values("date").groupby("district_id"):
        dates = group["date"].to_numpy("datetime64[ns]")
        rain = group["rain_mm"].to_numpy(float)
        tmax = group["tmax_c"].to_numpy(float)
        tmin = group["tmin_c"].to_numpy(float)
        years = range(int(group["date"].dt.year.min()), int(group["date"].dt.year.max()) + 1)
        for year in years:
            for season in seasons:
                start, end = season_bounds(year, season, seasons)
                lo = int(np.searchsorted(dates, np.datetime64(start, "ns"), side="left"))
                hi = int(np.searchsorted(dates, np.datetime64(end, "ns"), side="right"))
                if hi - lo != (end - start).days + 1:
                    continue  # incomplete season (weather ends or has gaps)
                row = {
                    "district_id": district,
                    "year": year,
                    "season": season,
                    "season_start": start,
                    "season_end": end,
                }
                row.update(window_features(rain[lo:hi], tmax[lo:hi], tmin[lo:hi]))
                row.update(satellite_features(lookup, district, start, end))
                rows.append(row)
    return pd.DataFrame(rows)


def training_table_py(
    weather: pd.DataFrame,
    satellite: pd.DataFrame | None,
    crops: pd.DataFrame,
    districts: pd.DataFrame,
) -> pd.DataFrame:
    """Python equivalent of ``v_training_table`` (used as the fallback when PostgreSQL is down)."""
    seasons_df = season_table(weather, satellite)
    out = crops.rename(
        columns={"yield_t_per_ha": "yield_label", "is_synthetic": "label_is_synthetic"}
    ).copy()
    out["year"] = out["year"].astype(int)
    out = out.sort_values(["district_id", "season", "crop", "year"]).reset_index(drop=True)
    grouped = out.groupby(["district_id", "season", "crop"], sort=False)
    prev_year = grouped["year"].shift(1)
    out["prev_year_yield"] = grouped["yield_label"].shift(1).where(prev_year == out["year"] - 1)
    stats = []
    for _, group in grouped:
        values = group["yield_label"].to_numpy(float)
        years = group["year"].to_numpy(int)
        mean3 = np.full(len(group), np.nan)
        count = np.zeros(len(group), dtype=int)
        for i in range(len(group)):
            sel = (years < years[i]) & (years >= years[i] - 3)
            count[i] = sel.sum()
            if count[i]:
                mean3[i] = values[sel].mean()
        stats.append(
            pd.DataFrame({"yield_mean_prev3": mean3, "n_prev_years": count}, index=group.index)
        )
    out = out.join(pd.concat(stats))
    out = out.merge(districts[["district_id", "state"]], on="district_id")
    out = out.merge(seasons_df, on=["district_id", "year", "season"], how="inner")
    return out.sort_values(["district_id", "crop", "year"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- data loading
def _processed(name: str) -> pd.DataFrame:
    path = get_path("data_processed") / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m etl.run --all`")
    return pd.read_parquet(path)


def load_weather_daily() -> pd.DataFrame:
    """Daily district weather (parquet)."""
    return _processed("weather_daily")


def load_training_frame(prefer_db: bool = True) -> tuple[pd.DataFrame, str]:
    """The model-ready table (view columns) and its source label ("postgres" or "parquet")."""
    if prefer_db:
        try:
            from analytics import common

            return common.training_table(), "postgres"
        except Exception as exc:  # DB down, view missing, ... -> documented fallback
            LOG.warning(
                "PostgreSQL unavailable (%s); computing the training table from parquet", exc
            )
    weather, crops = _processed("weather_daily"), _processed("crops_annual")
    satellite, districts = _processed("satellite_16d"), _processed("districts")
    return training_table_py(weather, satellite, crops, districts), "parquet"


def load_clusters() -> pd.DataFrame | None:
    """Phase 3 climate clusters (district_id, cluster_id) or None if not computed yet."""
    path = get_path("data_processed") / "district_clusters.parquet"
    return (
        pd.read_parquet(path)[["district_id", "cluster_id", "cluster_label"]]
        if path.exists()
        else None
    )


# ----------------------------------------------------------------------------- history features
def _prior_regression(years: np.ndarray, values: np.ndarray) -> np.ndarray:
    """For each row, the linear trend fitted on strictly earlier rows evaluated at the row's year."""
    n_rows = len(years)
    pred = np.full(n_rows, np.nan)
    for i in range(n_rows):
        sel = years < years[i]
        if sel.sum() >= MIN_TREND_YEARS:
            slope, intercept = np.polyfit(years[sel].astype(float), values[sel], 1)
            pred[i] = intercept + slope * years[i]
    return pred


def add_history(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``yield_mean_prior_all``, ``n_prior`` and ``yield_trend_pred`` using strictly earlier years only."""
    out = df.sort_values(["district_id", "crop", "season", "year"]).reset_index(drop=True)
    mean_all, n_prior, trend = (np.full(len(out), np.nan) for _ in range(3))
    for _, idx in out.groupby(["district_id", "crop", "season"]).indices.items():
        years = out["year"].to_numpy()[idx]
        values = out["yield_label"].to_numpy(float)[idx]
        for k, row in enumerate(idx):
            sel = years < years[k]
            n_prior[row] = sel.sum()
            mean_all[row] = values[sel].mean() if sel.any() else np.nan
        trend[idx] = _prior_regression(years, values)
    out["yield_mean_prior_all"], out["n_prior"], out["yield_trend_pred"] = mean_all, n_prior, trend
    return out


def add_weather_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Add rain ratios and temperature anomalies versus the district-season mean of *earlier* years."""
    columns = (
        [f"rain_d{lead}" for lead in LEADS]
        + [f"mean_tmax_d{lead}" for lead in LEADS]
        + ["total_rain", "mean_tmax"]
    )
    unique = df.drop_duplicates(["district_id", "season", "year"])[
        ["district_id", "season", "year", *columns]
    ]
    unique = unique.sort_values(["district_id", "season", "year"])
    grouped = unique.groupby(["district_id", "season"])
    for column in columns:
        prior_mean = grouped[column].transform(
            lambda s: s.shift(1).expanding(MIN_CLIMATOLOGY_YEARS).mean()
        )
        if column.startswith("rain") or column == "total_rain":
            unique[f"{column}_ratio"] = unique[column] / prior_mean
        else:
            unique[f"{column}_anom"] = unique[column] - prior_mean
    keep = ["district_id", "season", "year"] + [
        c for c in unique if c.endswith(("_ratio", "_anom"))
    ]
    return df.drop(
        columns=[c for c in keep if c in df and c not in ("district_id", "season", "year")]
    ).merge(unique[keep], on=["district_id", "season", "year"], how="left")


def prepare_frame(df: pd.DataFrame, clusters: pd.DataFrame | None = None) -> pd.DataFrame:
    """History + anomaly features + climate cluster, ready for ``build_features``."""
    out = add_weather_anomalies(add_history(df))
    clusters = load_clusters() if clusters is None else clusters
    if clusters is not None:
        out = out.drop(columns=[c for c in ("cluster_id", "cluster_label") if c in out]).merge(
            clusters, on="district_id", how="left"
        )
    return out


# ----------------------------------------------------------------------------- feature matrix
def weather_columns(lead_days: int) -> list[str]:
    """Raw season-to-date columns usable at ``lead_days``."""
    return [f"{stem}_d{lead}" for lead in LEADS if lead <= lead_days for stem in WEATHER_STEMS]


def anomaly_columns(lead_days: int) -> list[str]:
    """Anomaly columns usable at ``lead_days``."""
    return [f"rain_d{lead}_ratio" for lead in LEADS if lead <= lead_days] + [
        f"mean_tmax_d{lead}_anom" for lead in LEADS if lead <= lead_days
    ]


def satellite_columns(lead_days: int) -> list[str]:
    """Satellite columns usable at ``lead_days`` (finished composites only)."""
    cols = ["ndvi_at_sowing", "soil_moisture_at_sowing"]
    for lead in LEADS:
        if lead <= lead_days:
            cols += [f"ndvi_d{lead}", f"soil_moisture_d{lead}"]
    return cols


def category_levels(clusters: pd.DataFrame | None = None) -> dict[str, list]:
    """Fixed one-hot levels (so train/serve matrices always have identical columns)."""
    settings = load_settings()
    clusters = load_clusters() if clusters is None else clusters
    n_clusters = (
        int(clusters["cluster_id"].max()) + 1 if clusters is not None and len(clusters) else 0
    )
    return {
        "crop": list(settings["crops"]),
        "season": list(settings["seasons"]),
        "cluster": list(range(n_clusters)),
    }


def build_features(
    df: pd.DataFrame,
    lead_days: int,
    include_satellite: bool = True,
    levels: dict[str, list] | None = None,
    min_prior_years: int = MIN_PRIOR_YEARS,
) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame]:
    """Feature matrix, relative target and metadata at forecast lead ``lead_days``.

    Args:
        df: output of :func:`prepare_frame` (needs the season columns and history columns).
        lead_days: 30, 60 or 90 days after the season start.
        include_satellite: add NDVI / soil-moisture columns (synthetic until GEE is authenticated).
        levels: one-hot levels from :func:`category_levels` (so serving matches training).
        min_prior_years: rows with fewer earlier yields of the same district-crop are dropped, because the
            scale (prior mean yield) would be unreliable.

    Returns:
        ``X`` (float DataFrame, NaN allowed), ``y`` (yield / prior mean yield, None if the label column is
        absent) and ``meta`` (ids, scale, baselines in original units).
    """
    if lead_days not in LEADS:
        raise ValueError(f"lead_days must be one of {LEADS}")
    levels = levels or category_levels()
    data = df[df["n_prior"] >= min_prior_years].reset_index(drop=True)
    scale = data["yield_mean_prior_all"].to_numpy(float)
    cols: dict[str, np.ndarray | pd.Series] = {}
    for column in weather_columns(lead_days) + anomaly_columns(lead_days):
        cols[column] = data[column]
    if include_satellite:
        for column in satellite_columns(lead_days):
            cols[column] = data[column]
    cols["prev_year_rel"] = data["prev_year_yield"] / scale
    cols["mean_prev3_rel"] = data["yield_mean_prev3"] / scale
    cols["trend_pred_rel"] = data["yield_trend_pred"] / scale
    cols["log_scale"] = np.log(scale)
    for crop in levels["crop"]:
        cols[f"crop_{crop}"] = (data["crop"] == crop).astype(float)
    for season in levels["season"]:
        cols[f"season_{season}"] = (data["season"] == season).astype(float)
    for cluster in levels["cluster"]:
        cols[f"cluster_{cluster}"] = (data["cluster_id"] == cluster).astype(float)
    X = pd.DataFrame(cols).astype("float64")
    y = (data["yield_label"] / scale).rename("y_rel") if "yield_label" in data else None
    meta = data[["district_id", "state", "year", "crop", "season"]].copy()
    meta["scale"] = scale
    meta["yield"] = data.get("yield_label", np.nan)
    meta["prev_year_yield"] = data["prev_year_yield"]
    meta["trend_pred"] = data["yield_trend_pred"]
    meta["lead_days"] = lead_days
    return X, y, meta


def split_masks(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks for train / valid / test from settings.yaml (time-based, never random)."""
    split = load_settings()["splits"]["yield"]
    first = int(load_settings()["years"]["start"])
    year = meta["year"].to_numpy()
    return {
        "train": (year >= first) & (year <= int(split["train_end"])),
        "valid": (year >= split["valid"][0]) & (year <= split["valid"][1]),
        "test": (year >= split["test"][0]) & (year <= split["test"][1]),
    }


# ----------------------------------------------------------------------------- weekly sequences (LSTM)
SEQUENCE_CHANNELS = ("rain_mm", "tmax_c", "tmin_c", "ndvi", "soil_moisture")


def build_sequences(
    meta: pd.DataFrame,
    weather: pd.DataFrame,
    satellite: pd.DataFrame | None,
    lead_days: int,
    n_weeks: int = N_WEEKS,
) -> tuple[np.ndarray, np.ndarray]:
    """Weekly sequences for the first ``n_weeks`` weeks, with unobserved weeks zeroed and masked.

    A week is observed at forecast time if it ends before ``lead_days`` (so ``lead_days // 7`` weeks);
    its NDVI / soil moisture come from the latest composite that had finished by the end of that week.

    Returns:
        ``(array (N, n_weeks, 5), mask (N, n_weeks))`` with channels ``SEQUENCE_CHANNELS``.
    """
    n_obs = min(n_weeks, lead_days // 7)
    daily = {d: g.sort_values("date") for d, g in weather.groupby("district_id")}
    lookup = _satellite_lookup(satellite)
    seasons = load_settings()["seasons"]
    out = np.zeros((len(meta), n_weeks, len(SEQUENCE_CHANNELS)), dtype=np.float32)
    mask = np.zeros((len(meta), n_weeks), dtype=np.float32)
    for i, row in enumerate(meta.itertuples()):
        start, _ = season_bounds(int(row.year), row.season, seasons)
        group = daily[row.district_id]
        dates = group["date"].to_numpy("datetime64[ns]")
        lo = int(np.searchsorted(dates, np.datetime64(start, "ns")))
        block = group.iloc[lo : lo + 7 * n_obs]
        if len(block) < 7 * n_obs:
            continue
        weekly = block[["rain_mm", "tmax_c", "tmin_c"]].to_numpy(float).reshape(n_obs, 7, 3)
        out[i, :n_obs, 0] = weekly[:, :, 0].sum(axis=1)
        out[i, :n_obs, 1] = weekly[:, :, 1].mean(axis=1)
        out[i, :n_obs, 2] = weekly[:, :, 2].mean(axis=1)
        entry = lookup.get(row.district_id)
        for w in range(n_obs):
            if entry is None:
                continue
            cutoff = np.datetime64(start + pd.Timedelta(days=7 * (w + 1)), "ns")
            idx = int(np.searchsorted(entry["finished"], cutoff, side="right")) - 1
            if idx >= 0:
                out[i, w, 3], out[i, w, 4] = entry["ndvi"][idx], entry["soil"][idx]
        mask[i, :n_obs] = 1.0
    return out, mask


def describe_missing(X: pd.DataFrame, top: int = 8) -> pd.Series:
    """Columns with the highest share of NaN (for the report)."""
    return X.isna().mean().sort_values(ascending=False).head(top)


def check_no_future_columns(columns: Sequence[str], lead_days: int) -> list[str]:
    """Names of columns that refer to a lead later than ``lead_days`` (must be empty)."""
    bad = []
    for column in columns:
        for later in (other for other in LEADS if other > lead_days):
            if column.endswith((f"_d{later}", f"_d{later}_ratio", f"_d{later}_anom")):
                bad.append(column)
        if column in {
            "total_rain",
            "rainy_days",
            "longest_dry_spell_days",
            "mean_tmax",
            "heat_days",
            "growing_degree_days",
            "ndvi_peak",
            "soil_moisture_mean",
            "area_ha",
            "production_t",
        }:
            bad.append(column)
    return bad


def models_dir() -> Path:
    """models/ directory."""
    return get_path("models")
