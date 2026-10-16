"""Cached data access for the dashboards.

Large tables come from the processed parquet files (fast, no server needed, identical to the warehouse contents);
the model-ready training frame uses PostgreSQL when it is reachable and falls back to parquet
(``analytics.predictive.features.load_training_frame``). The active source is reported on the Methodology page.
"""

from __future__ import annotations

import json
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st

from config import get_path, load_settings
from dashboards.components import ROOT


def _processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(get_path("data_processed") / f"{name}.parquet")


@st.cache_data(show_spinner=False)
def load_districts() -> pd.DataFrame:
    """The 51 yield-reporting district units (without geometry columns)."""
    return (
        _processed("districts")
        .drop(columns=["geometry_wkt"])
        .sort_values(["state", "district"], ignore_index=True)
    )


@st.cache_data(show_spinner=False)
def load_geojson() -> dict[str, Any]:
    """Simplified district polygons as GeoJSON with ``properties.district_id`` (for maps)."""
    frame = _processed("districts")
    geometry = gpd.GeoSeries.from_wkt(frame["geometry_wkt"], crs="EPSG:4326").simplify(
        0.01, preserve_topology=True
    )
    gdf = gpd.GeoDataFrame(
        frame[["district_id", "state", "district"]], geometry=geometry, crs="EPSG:4326"
    )
    return json.loads(gdf.to_json())


@st.cache_data(show_spinner=False)
def centroids_and_clusters() -> tuple[dict[str, tuple[float, float]], dict[str, int]]:
    """District centroids (lat, lon) and climate-cluster ids (empty clusters if the analytics step was not run)."""
    districts = load_districts()
    centroids = {
        row.district_id: (float(row.centroid_lat), float(row.centroid_lon))
        for row in districts.itertuples()
    }
    path = get_path("data_processed") / "district_clusters.parquet"
    clusters = {}
    if path.exists():
        frame = pd.read_parquet(path)
        clusters = dict(zip(frame["district_id"], frame["cluster_id"].astype(int), strict=True))
    return centroids, clusters


@st.cache_data(show_spinner=False)
def map_center() -> tuple[float, float]:
    """(lat, lon) centre of the study region."""
    d = load_districts()
    return float(d["centroid_lat"].mean()), float(d["centroid_lon"].mean())


@st.cache_data(show_spinner=False)
def load_weather() -> pd.DataFrame:
    """Daily district weather (IMD rain/Tmax/Tmin, NASA POWER humidity/wind/solar)."""
    return _processed("weather_daily")


@st.cache_data(show_spinner=False)
def load_crops() -> pd.DataFrame:
    """Yield table (crop-year level)."""
    return _processed("crops_annual")


def crops_list() -> list[str]:
    """Study crops from settings.yaml."""
    return list(load_settings()["crops"])


def season_of(crop: str) -> str:
    """The season a crop is assigned to (settings.yaml ``crop_season``)."""
    return load_settings()["crop_season"][crop]


@st.cache_data(show_spinner=False)
def recent_weather(district_id: str, as_of: str, days: int = 60) -> pd.DataFrame:
    """The last ``days`` days of weather up to and including ``as_of``."""
    w = load_weather()
    end = pd.Timestamp(as_of)
    sel = w[
        (w["district_id"] == district_id)
        & (w["date"] <= end)
        & (w["date"] > end - pd.Timedelta(days=days))
    ]
    return sel.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def five_year_mean(district_id: str, crop: str, reference_year: int) -> tuple[float | None, int]:
    """Mean yield of the five crop years before ``reference_year`` (t/ha) and how many years were available."""
    crops = load_crops()
    sel = crops[
        (crops["district_id"] == district_id)
        & (crops["crop"] == crop)
        & crops["year"].between(reference_year - 5, reference_year - 1)
    ]
    return (float(sel["yield_t_per_ha"].mean()) if len(sel) else None), int(sel["year"].nunique())


@st.cache_data(show_spinner=False)
def yield_history(district_id: str, crop: str) -> pd.DataFrame:
    """Observed yields of one district-crop."""
    crops = load_crops()
    return crops[(crops["district_id"] == district_id) & (crops["crop"] == crop)].sort_values(
        "year"
    )


@st.cache_data(show_spinner=False)
def vulnerability_table() -> pd.DataFrame:
    """Long-term climate vulnerability per district (defined for the dashboard, see decision D-025).

    Score = mean percentile rank of (kharif drought frequency, rice-yield variability, mean heat days per kharif
    season); higher = more vulnerable. Drought = season rain < 75 percent of the district mean. Decision support
    only: it summarises 1991-2024 weather and 2000-2017 yields, not a forecast.
    """
    from analytics.predictive import features as F

    weather = load_weather()
    seasons = F.season_table(weather, None)
    kharif = seasons[seasons["season"] == "kharif"].copy()
    mean_rain = kharif.groupby("district_id")["total_rain"].transform("mean")
    kharif["drought"] = kharif["total_rain"] < 0.75 * mean_rain
    base = kharif.groupby("district_id").agg(
        drought_frequency=("drought", "mean"), mean_heat_days=("heat_days", "mean")
    )
    rice = load_crops()
    rice = rice[(rice["crop"] == "rice") & (rice["year"] >= 2000)]
    variability = (
        rice.groupby("district_id")["yield_t_per_ha"]
        .agg(lambda s: s.std() / s.mean() if len(s) >= 10 else np.nan)
        .rename("rice_yield_cv")
    )
    table = base.join(variability).dropna().reset_index()
    ranks = table[["drought_frequency", "rice_yield_cv", "mean_heat_days"]].rank(pct=True)
    table["vulnerability_score"] = ranks.mean(axis=1)
    districts = load_districts()[["district_id", "state", "district"]]
    return table.merge(districts, on="district_id").sort_values(
        "vulnerability_score", ascending=False, ignore_index=True
    )


@st.cache_data(show_spinner=False)
def data_source() -> str:
    """'postgres' if the warehouse answers, else 'parquet' (reported on the Methodology page)."""
    try:
        from database import db_available

        return "postgres" if db_available(1) else "parquet"
    except Exception:
        return "parquet"


def season_start(year: int, season: str) -> pd.Timestamp:
    """First day of a season in a crop year."""
    from analytics.predictive import features as F

    return F.season_bounds(year, season)[0]


def static_dir() -> str:
    """Folder served by Streamlit's static file serving."""
    return str(ROOT / "dashboards" / "static")
