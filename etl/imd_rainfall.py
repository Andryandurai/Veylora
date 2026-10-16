"""IMD gridded daily rainfall (0.25 degree) aggregated to district means.

Source: India Meteorological Department high-resolution gridded daily rainfall, fetched through
the ``imdlib`` package (see etl/imd_common.py). Each district value is the area-weighted mean of
the grid cells that overlap its polygon.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from config import get_path
from etl import grid, imd_common

LOG = logging.getLogger(__name__)


def fingerprint(districts: gpd.GeoDataFrame) -> str:
    """Short hash of the district ids and areas; changes whenever the polygons change."""
    text = "|".join(
        f"{i}:{a:.1f}" for i, a in zip(districts["district_id"], districts["area_km2"], strict=True)
    )
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def cache_dir(name: str) -> Path:
    """Interim cache directory for per-year aggregated frames."""
    path = get_path("data_interim") / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def aggregate_variable(
    districts: gpd.GeoDataFrame,
    var: str,
    years: Sequence[int],
    column: str,
    raw_dir: Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Aggregate one IMD variable to district daily means for the given years.

    Years whose IMD file is missing are skipped with a warning (the caller decides on fallbacks).

    Returns:
        DataFrame with columns district_id, date, ``column`` (float32).
    """
    tag = fingerprint(districts)
    out_dir = cache_dir("imd")
    frames: list[pd.DataFrame] = []
    weights = None
    for year in years:
        cached = out_dir / f"{var}_{year}_{tag}.parquet"
        if use_cache and cached.exists():
            frames.append(pd.read_parquet(cached))
            continue
        if not imd_common.is_complete(var, year, raw_dir):
            LOG.warning("IMD %s %d not available locally; skipped", var, year)
            continue
        data = imd_common.open_year(var, year, raw_dir)
        lat, lon = data["lat"].values, data["lon"].values
        field = data.transpose("time", "lat", "lon").values
        if weights is None:
            weights = grid.build_weights(districts.geometry, lat, lon, grid.valid_cells(field))
        means = grid.aggregate(field.reshape(field.shape[0], -1), weights)
        dates = pd.to_datetime(data["time"].values).normalize()
        frame = pd.DataFrame(means, index=dates, columns=districts["district_id"].tolist())
        frame = (
            frame.rename_axis("date")
            .reset_index()
            .melt(id_vars="date", var_name="district_id", value_name=column)
        )
        frame[column] = frame[column].astype("float32")
        frame.to_parquet(cached, index=False)
        frames.append(frame)
        LOG.info("IMD %s %d aggregated (%d districts)", var, year, len(districts))
    if not frames:
        return pd.DataFrame(
            {
                "district_id": pd.Series(dtype="str"),
                "date": pd.Series(dtype="datetime64[ns]"),
                column: pd.Series(dtype="float32"),
            }
        )
    result = pd.concat(frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"]).astype("datetime64[ns]")
    return result[["district_id", "date", column]].sort_values(
        ["district_id", "date"], ignore_index=True
    )


def district_rainfall(
    districts: gpd.GeoDataFrame, years: Sequence[int], raw_dir: Path | None = None
) -> pd.DataFrame:
    """District daily rainfall (mm/day) from IMD, columns district_id, date, rain_mm."""
    frame = aggregate_variable(districts, "rain", years, "rain_mm", raw_dir)
    # IMD stores tiny negatives as float noise; physical rainfall cannot be negative.
    frame["rain_mm"] = frame["rain_mm"].clip(lower=0).astype(np.float32)
    return frame
