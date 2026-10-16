"""SYNTHETIC daily weather generator (fallback only; every row is ``is_synthetic=True``).

Used only when neither IMD nor NASA POWER can supply a district-day. It produces monsoon-shaped
rainfall (SW monsoon peak mid-July, NE monsoon bump around November for the south-east coast),
a seasonal temperature cycle peaking in May, and humidity / wind / solar consistent with them.
Parameters depend only on latitude so results are reproducible; this is *not* a climate model and
must never be reported as observed data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import get_seed

SW_PEAK_DOY = 200  # ~19 July
NE_PEAK_DOY = 310  # ~6 November
TMAX_PEAK_DOY = 125  # early May


def _gaussian(doy: np.ndarray, centre: float, width: float) -> np.ndarray:
    delta = np.minimum(np.abs(doy - centre), 365 - np.abs(doy - centre))
    return np.exp(-0.5 * (delta / width) ** 2)


def generate_district(
    district_id: str,
    lat: float,
    lon: float,
    dates: pd.DatetimeIndex,
    year_factors: pd.Series,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate daily weather for one district.

    Args:
        district_id: id copied into the output.
        lat, lon: centroid in degrees; controls climate (wetter/hotter towards the south coast).
        dates: daily index to generate.
        year_factors: multiplicative annual rainfall anomaly per calendar year (shared by all
            districts so that droughts and good years are spatially coherent).
        rng: seeded generator (callers derive one per district).
    """
    doy = dates.dayofyear.to_numpy()
    n = len(dates)
    east_coast = 1.0 if lon > 79 and lat < 16 else 0.3
    wet_prob = (
        0.03
        + 0.55 * _gaussian(doy, SW_PEAK_DOY, 35)
        + 0.30 * east_coast * _gaussian(doy, NE_PEAK_DOY, 25)
    )
    mean_wet = (
        4.0
        + 14.0 * _gaussian(doy, SW_PEAK_DOY, 35)
        + 9.0 * east_coast * _gaussian(doy, NE_PEAK_DOY, 25)
    )
    factor = year_factors.reindex(dates.year).to_numpy() * np.exp(rng.normal(0, 0.1))
    wet = rng.random(n) < np.clip(wet_prob * np.sqrt(factor), 0, 0.95)
    amount = rng.gamma(shape=0.7, scale=mean_wet / 0.7, size=n) * factor
    rain = np.where(wet, amount, 0.0)
    rain = np.minimum(rain, 400.0)

    base_tmax = 36.5 - 0.35 * (lat - 12.0)
    seasonal = 4.0 * np.cos(2 * np.pi * (doy - TMAX_PEAK_DOY) / 365.0)
    noise = np.zeros(n)
    eps = rng.normal(0, 1.1, n)
    for i in range(1, n):  # AR(1) temperature anomalies
        noise[i] = 0.7 * noise[i - 1] + eps[i]
    tmax = base_tmax - 1.0 + seasonal - 0.08 * np.minimum(rain, 40.0) + noise
    tmin = tmax - (11.0 - 4.0 * _gaussian(doy, SW_PEAK_DOY, 45)) + rng.normal(0, 0.8, n)
    tmax = np.clip(tmax, 12.0, 49.0)
    tmin = np.clip(np.minimum(tmin, tmax - 1.0), 5.0, 36.0)
    rh = np.clip(
        55
        + 30 * _gaussian(doy, SW_PEAK_DOY, 50)
        + 0.3 * np.minimum(rain, 30)
        + rng.normal(0, 5, n),
        15,
        100,
    )
    wind = np.clip(2.2 + 1.5 * _gaussian(doy, SW_PEAK_DOY, 40) + rng.normal(0, 0.5, n), 0.2, 12)
    solar = np.clip(
        22
        - 9 * _gaussian(doy, SW_PEAK_DOY, 45)
        - 0.15 * np.minimum(rain, 40)
        + rng.normal(0, 1.5, n),
        2,
        30,
    )
    return pd.DataFrame(
        {
            "district_id": district_id,
            "date": dates,
            "rain_mm": rain.astype("float32"),
            "tmax_c": tmax.astype("float32"),
            "tmin_c": tmin.astype("float32"),
            "rh_pct": rh.astype("float32"),
            "wind_ms": wind.astype("float32"),
            "solar_mj": solar.astype("float32"),
            "is_synthetic": True,
        }
    )


def generate_weather(
    districts: pd.DataFrame, start_year: int, end_year: int, seed: int | None = None
) -> pd.DataFrame:
    """Synthetic daily weather for every district between Jan 1 of start_year and Dec 31 of end_year."""
    seed = get_seed() if seed is None else seed
    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    year_rng = np.random.default_rng(seed)
    years = np.arange(start_year, end_year + 1)
    factors = pd.Series(np.exp(year_rng.normal(0, 0.18, len(years))), index=years)
    frames = []
    for i, row in enumerate(districts.itertuples()):
        rng = np.random.default_rng([seed, i])
        frames.append(
            generate_district(
                row.district_id, row.centroid_lat, row.centroid_lon, dates, factors, rng
            )
        )
    return pd.concat(frames, ignore_index=True)
