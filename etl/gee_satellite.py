"""Satellite district statistics from Google Earth Engine (GEE).

Products (all reduced over the district polygons with ``reduceRegions``, one year per request so
that responses stay below GEE's 5000-feature limit):

* **NDVI**: ``MODIS/061/MOD13Q1`` (250 m, 16-day composites), band ``NDVI`` scaled by 0.0001.
* **Soil moisture**: ``ECMWF/ERA5_LAND/DAILY_AGGR`` band ``volumetric_soil_water_layer_1`` (m3/m3),
  averaged over each MODIS 16-day window so it aligns with NDVI. SMAP (the plan's first choice,
  ``NASA_USDA/HSL/SMAP10KM_soil_moisture``) only starts in April 2015 and cannot cover the 2000-2017
  yield window; see docs/DECISIONS.md D-008.
* **CHIRPS rainfall check series**: ``UCSB-CHG/CHIRPS/DAILY`` monthly sums, written to
  ``data/interim/chirps_monthly.parquet`` to cross-check IMD rainfall (not part of satellite_16d).

Per-year CSVs are cached in ``data/interim/gee`` so re-runs and partial failures are cheap.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import shapely

from config import get_path, load_env
from etl.boundaries import districts_geodataframe

LOG = logging.getLogger(__name__)

NDVI_COLLECTION = "MODIS/061/MOD13Q1"
NDVI_SCALE_FACTOR = 0.0001
ERA5_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
ERA5_BAND = "volumetric_soil_water_layer_1"
CHIRPS_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"
SIMPLIFY_DEG = 0.005  # ~500 m; keeps request payloads small without changing district means


class GeeUnavailableError(RuntimeError):
    """Earth Engine is not installed, not authenticated or the request failed."""


def cache_dir() -> Path:
    """Per-year CSV cache directory (data/interim/gee)."""
    path = get_path("data_interim") / "gee"
    path.mkdir(parents=True, exist_ok=True)
    return path


def initialize() -> Any:
    """Initialise and return the ``ee`` module, or raise :class:`GeeUnavailableError`."""
    from etl.gee_auth_check import check_auth

    load_env()
    ok, message = check_auth()
    if not ok:
        raise GeeUnavailableError(message)
    import ee

    return ee


def feature_collection(ee: Any, districts: pd.DataFrame) -> Any:
    """Build an ``ee.FeatureCollection`` with one simplified polygon per district."""
    gdf = districts_geodataframe(districts)
    features = []
    for district_id, geom in zip(gdf["district_id"], gdf.geometry, strict=True):
        simple = geom.simplify(SIMPLIFY_DEG, preserve_topology=True)
        geojson = json.loads(shapely.to_geojson(simple))
        features.append(ee.Feature(ee.Geometry(geojson, None, False), {"district_id": district_id}))
    return ee.FeatureCollection(features)


def parse_features(features: list[dict], value_name: str) -> pd.DataFrame:
    """Turn the ``features`` list from ``getInfo()`` into a tidy frame (district_id, date, value).

    Features whose ``mean`` is null (fully masked pixels, e.g. cloud-covered composites) are kept as
    NaN rather than dropped, so gaps stay visible.
    """
    rows = []
    for feature in features:
        props = feature.get("properties", {})
        rows.append(
            {
                "district_id": props.get("district_id"),
                "date": props.get("date"),
                value_name: props.get("mean"),
            }
        )
    frame = pd.DataFrame(rows, columns=["district_id", "date", value_name])
    frame["date"] = pd.to_datetime(frame["date"])
    frame[value_name] = pd.to_numeric(frame[value_name], errors="coerce")
    return frame


def _per_image_stats(ee: Any, fc: Any, image: Any, scale: float, date: Any) -> Any:
    stats = image.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=scale, tileScale=4)
    return stats.map(lambda f: f.set("date", date).setGeometry(None))


def ndvi_year(ee: Any, fc: Any, year: int) -> list[dict]:
    """District mean NDVI (raw MODIS integer units) for every 16-day composite in ``year``."""
    ic = (
        ee.ImageCollection(NDVI_COLLECTION)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .select("NDVI")
    )
    stats = ic.map(
        lambda img: _per_image_stats(ee, fc, img, 250, img.date().format("YYYY-MM-dd"))
    ).flatten()
    return stats.select(["district_id", "date", "mean"]).getInfo()["features"]


def soil_moisture_year(ee: Any, fc: Any, year: int) -> list[dict]:
    """District mean ERA5-Land layer-1 soil moisture averaged over each MODIS composite window."""
    modis = (
        ee.ImageCollection(NDVI_COLLECTION)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .select("NDVI")
    )
    era5 = ee.ImageCollection(ERA5_COLLECTION).select(ERA5_BAND)

    def per_composite(img: Any) -> Any:
        start = img.date()
        window = era5.filterDate(start, start.advance(16, "day")).mean().rename("sm")
        return _per_image_stats(ee, fc, window, 11132, start.format("YYYY-MM-dd"))

    return (
        modis.map(per_composite)
        .flatten()
        .select(["district_id", "date", "mean"])
        .getInfo()["features"]
    )


def chirps_year(ee: Any, fc: Any, year: int) -> list[dict]:
    """District mean CHIRPS monthly rainfall totals (mm) for ``year``."""
    chirps = ee.ImageCollection(CHIRPS_COLLECTION).select("precipitation")

    def per_month(month: Any) -> Any:
        start = ee.Date.fromYMD(year, 1, 1).advance(month, "month")
        total = chirps.filterDate(start, start.advance(1, "month")).sum().rename("chirps_mm")
        return _per_image_stats(ee, fc, total, 5566, start.format("YYYY-MM-dd"))

    months = ee.List.sequence(0, 11)
    flat = ee.FeatureCollection(months.map(per_month)).flatten()
    return flat.select(["district_id", "date", "mean"]).getInfo()["features"]


def _year_csv(kind: str, year: int, fetch: Any, value_name: str, force: bool) -> pd.DataFrame:
    """Return the cached or freshly fetched per-year frame for one product."""
    path = cache_dir() / f"{kind}_{year}.csv"
    if path.exists() and not force:
        return pd.read_csv(path, parse_dates=["date"])
    try:
        frame = parse_features(fetch(year), value_name)
    except Exception as exc:  # ee.EEException and transport errors share no useful base class
        raise GeeUnavailableError(f"GEE {kind} {year} failed: {exc}") from exc
    frame.to_csv(path, index=False)
    LOG.info("GEE %s %d: %d rows", kind, year, len(frame))
    return frame


def build_satellite(
    districts: pd.DataFrame, years: Sequence[int], force: bool = False
) -> pd.DataFrame:
    """Fetch NDVI + soil moisture for all districts and years; returns the satellite_16d table.

    Raises:
        GeeUnavailableError: when Earth Engine cannot be used (callers then fall back to synthetic).
    """
    ee = initialize()
    fc = feature_collection(ee, districts)
    frames = []
    for year in years:
        ndvi = _year_csv("ndvi", year, lambda y: ndvi_year(ee, fc, y), "ndvi_raw", force)
        soil = _year_csv(
            "soil", year, lambda y: soil_moisture_year(ee, fc, y), "soil_moisture", force
        )
        frames.append(ndvi.merge(soil, on=["district_id", "date"], how="outer"))
    table = pd.concat(frames, ignore_index=True)
    table["ndvi"] = (table["ndvi_raw"] * NDVI_SCALE_FACTOR).astype("float32")
    table["soil_moisture"] = table["soil_moisture"].astype("float32")
    table["is_synthetic"] = False
    table["date"] = table["date"].astype("datetime64[ns]")
    return table[["district_id", "date", "ndvi", "soil_moisture", "is_synthetic"]].sort_values(
        ["district_id", "date"], ignore_index=True
    )


def build_chirps(districts: pd.DataFrame, years: Sequence[int], force: bool = False) -> Path:
    """Fetch CHIRPS monthly district rainfall into data/interim/chirps_monthly.parquet."""
    ee = initialize()
    fc = feature_collection(ee, districts)
    frames = [
        _year_csv("chirps", y, lambda yr: chirps_year(ee, fc, yr), "chirps_mm", force)
        for y in years
    ]
    out = get_path("data_interim") / "chirps_monthly.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
    return out


def project_from_env() -> str:
    """The configured GEE Cloud project id (empty string when unset)."""
    load_env()
    return os.environ.get("GEE_PROJECT", "").strip()
