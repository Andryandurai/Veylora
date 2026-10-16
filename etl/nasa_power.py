"""NASA POWER daily point data at district centroids.

POWER (https://power.larc.nasa.gov, MERRA-2 / CERES derived, ~0.5 degree, free, no key) supplies the
variables IMD does not publish on a grid: relative humidity, wind speed and solar radiation. It also
carries temperature and corrected precipitation, which we keep with a ``p_`` prefix as a *real*
fallback when an IMD year is missing.

Output columns: district_id, date, rh_pct, wind_ms, solar_mj, p_tmax_c, p_tmin_c, p_rain_mm.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import get_path, load_settings

LOG = logging.getLogger(__name__)

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
FILL_VALUE = -999.0
RENAME = {
    "RH2M": "rh_pct",
    "WS2M": "wind_ms",
    "ALLSKY_SFC_SW_DWN": "solar_mj",
    "T2M_MAX": "p_tmax_c",
    "T2M_MIN": "p_tmin_c",
    "PRECTOTCORR": "p_rain_mm",
}
OUTPUT_COLUMNS = ["rh_pct", "wind_ms", "solar_mj", "p_tmax_c", "p_tmin_c", "p_rain_mm"]


def raw_dir() -> Path:
    """Per-district cache directory for POWER responses."""
    path = get_path("data_raw") / "nasa_power"
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_response(payload: dict) -> pd.DataFrame:
    """Convert a POWER JSON payload to a tidy frame indexed by date; -999 becomes NaN."""
    parameters = payload["properties"]["parameter"]
    series = {}
    for name, values in parameters.items():
        if name not in RENAME:
            continue
        s = pd.Series(values, dtype="float64")
        series[RENAME[name]] = s.where(s != FILL_VALUE)
    frame = pd.DataFrame(series)
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d")
    frame = frame.rename_axis("date").reset_index()
    for column in OUTPUT_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    frame[OUTPUT_COLUMNS] = frame[OUTPUT_COLUMNS].astype("float32")
    return frame[["date", *OUTPUT_COLUMNS]]


def fetch_point(
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
    parameters: list[str] | None = None,
    retries: int = 5,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download daily POWER data for one point, retrying on throttling/server errors.

    Raises:
        RuntimeError: if every attempt fails.
    """
    params = {
        "parameters": ",".join(parameters or load_settings()["etl"]["power_parameters"]),
        "community": "AG",
        "longitude": round(lon, 4),
        "latitude": round(lat, 4),
        "start": f"{start_year}0101",
        "end": f"{end_year}1231",
        "format": "JSON",
    }
    http = session or requests
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = http.get(POWER_URL, params=params, timeout=180)
            if response.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {response.status_code}")
            response.raise_for_status()
            return parse_response(response.json())
        except (requests.RequestException, KeyError, ValueError) as exc:
            last_error = exc
            wait = 5 * attempt
            LOG.warning(
                "POWER (%.2f, %.2f) attempt %d/%d failed: %s; retry in %ds",
                lat,
                lon,
                attempt,
                retries,
                exc,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"NASA POWER request failed for ({lat}, {lon}): {last_error}")


def _district_frame(row: pd.Series, start_year: int, end_year: int, force: bool) -> pd.DataFrame:
    cache = raw_dir() / f"{row['district_id']}_{start_year}_{end_year}.parquet"
    if cache.exists() and not force:
        frame = pd.read_parquet(cache)
    else:
        frame = fetch_point(row["centroid_lat"], row["centroid_lon"], start_year, end_year)
        frame.to_parquet(cache, index=False)
    frame.insert(0, "district_id", row["district_id"])
    return frame


def district_daily(
    districts: pd.DataFrame, start_year: int, end_year: int, force: bool = False, workers: int = 3
) -> pd.DataFrame:
    """POWER daily data for every district centroid (parallel, cached per district).

    Districts whose download fails are logged and omitted so that one bad request does not lose
    the rest; the caller sees which ids are missing through the returned frame.
    """
    rows = [row for _, row in districts.iterrows()]
    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    def work(row: pd.Series) -> pd.DataFrame | None:
        try:
            return _district_frame(row, start_year, end_year, force)
        except RuntimeError as exc:
            LOG.error("%s", exc)
            failed.append(str(row["district_id"]))
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for frame in pool.map(work, rows):
            if frame is not None:
                frames.append(frame)
    if failed:
        LOG.error("NASA POWER failed for %d districts: %s", len(failed), failed)
    if not frames:
        return pd.DataFrame(columns=["district_id", "date", *OUTPUT_COLUMNS])
    result = pd.concat(frames, ignore_index=True)
    result["date"] = result["date"].astype("datetime64[ns]")
    return result.sort_values(["district_id", "date"], ignore_index=True)
