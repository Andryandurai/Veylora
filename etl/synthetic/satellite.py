"""SYNTHETIC 16-day NDVI and soil moisture (fallback when Google Earth Engine is unavailable).

NDVI follows a smoothed, lagged response to district rainfall (a 30-day exponentially weighted
rainfall index passed through a saturating curve) plus a seasonal baseline and noise. Soil moisture
comes from a one-bucket water balance driven by the same rainfall. Because both are *derived from
rainfall by construction*, any analysis that treats them as independent evidence would be circular:
reports must say so whenever these rows (``is_synthetic=True``) are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import get_seed

COMPOSITE_DAYS = 16
NDVI_EWMA_DAYS = 30.0
BUCKET_MM = 120.0  # root-zone storage capacity of the toy bucket
FIELD_CAPACITY = 0.42  # volumetric soil moisture at full bucket
WILTING = 0.08


def composite_dates(start_year: int, end_year: int) -> pd.DatetimeIndex:
    """MODIS-like composite start dates: Jan 1 + 16-day steps, restarting every year (day 1, 17, ... 353)."""
    dates = []
    for year in range(start_year, end_year + 1):
        start = pd.Timestamp(year=year, month=1, day=1)
        dates.extend(start + pd.Timedelta(days=COMPOSITE_DAYS * k) for k in range(23))
    return pd.DatetimeIndex(dates)


def _bucket(rain: np.ndarray, et_mm: np.ndarray) -> np.ndarray:
    storage = np.empty_like(rain)
    s = 0.5 * BUCKET_MM
    for i, (p, et) in enumerate(zip(rain, et_mm, strict=True)):
        s = min(BUCKET_MM, max(0.0, s + p - et * (s / BUCKET_MM) ** 0.7))
        storage[i] = s
    return storage / BUCKET_MM


def generate_district(
    district_id: str, weather: pd.DataFrame, dates: pd.DatetimeIndex, rng: np.random.Generator
) -> pd.DataFrame:
    """Synthetic satellite rows for one district from its daily weather (needs rain_mm, tmax_c)."""
    w = weather.set_index("date").sort_index()
    rain = w["rain_mm"].to_numpy(dtype="float64")
    tmax = w["tmax_c"].to_numpy(dtype="float64")
    alpha = 1.0 - np.exp(-1.0 / NDVI_EWMA_DAYS)
    ewma = np.empty_like(rain)
    acc = 0.0
    for i, p in enumerate(rain):
        acc += alpha * (p - acc)
        ewma[i] = acc
    et = np.clip(0.18 * tmax - 1.0, 0.5, 7.0)
    storage_frac = _bucket(rain, et)
    index = w.index
    pos = index.get_indexer(dates, method="nearest")
    greenness = 1.0 - np.exp(-ewma[pos] / 2.5)
    ndvi = 0.18 + 0.62 * greenness + rng.normal(0, 0.03, len(dates))
    soil = (
        WILTING + (FIELD_CAPACITY - WILTING) * storage_frac[pos] + rng.normal(0, 0.01, len(dates))
    )
    return pd.DataFrame(
        {
            "district_id": district_id,
            "date": dates,
            "ndvi": np.clip(ndvi, -0.1, 0.95).astype("float32"),
            "soil_moisture": np.clip(soil, 0.02, 0.6).astype("float32"),
            "is_synthetic": True,
        }
    )


def generate_satellite(
    weather: pd.DataFrame, start_year: int, end_year: int, seed: int | None = None
) -> pd.DataFrame:
    """Synthetic satellite_16d table driven by ``weather`` (any is_synthetic mix)."""
    seed = get_seed() if seed is None else seed
    dates = composite_dates(start_year, end_year)
    frames = []
    for i, (district_id, group) in enumerate(weather.groupby("district_id", sort=True)):
        rng = np.random.default_rng([seed, 1000 + i])
        inside = group[
            (group["date"] >= dates[0])
            & (group["date"] <= dates[-1] + pd.Timedelta(days=COMPOSITE_DAYS))
        ]
        usable = dates[(dates >= inside["date"].min()) & (dates <= inside["date"].max())]
        frames.append(generate_district(district_id, inside, usable, rng))
    return pd.concat(frames, ignore_index=True)
