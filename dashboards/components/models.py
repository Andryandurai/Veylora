"""Cached model calls for the dashboards (thin wrappers: no modelling logic lives here).

``st.cache_data`` memoises results per argument set; the heavy artefacts (XGBoost, LSTM, weather context) are
loaded once per process by the underlying modules and warmed up by :func:`warm_up`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from config import load_settings
from dashboards.components import data


@st.cache_resource(show_spinner="Loading models (first load only)…")
def warm_up() -> bool:
    """Load the yield and early-warning artefacts once so that interactions afterwards are fast."""
    from analytics.predictive.extreme_weather import risk_forecast
    from analytics.predictive.predict import predict_yield

    ids = data.load_districts()["district_id"].tolist()
    crop = data.crops_list()[0]
    predict_yield(ids[0], crop, data.season_of(crop))
    risk_forecast(ids[0], "2015-06-15")
    return True


@st.cache_data(show_spinner=False)
def risk_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Dates for which an early-warning forecast can be issued (observed weather record)."""
    from analytics.predictive.extreme_weather import forecast_date_range

    return forecast_date_range()


@st.cache_data(show_spinner=False)
def risk(district_id: str, as_of: str) -> dict[str, Any]:
    """Early-warning forecast of one district (see ``risk_forecast``)."""
    from analytics.predictive.extreme_weather import risk_forecast

    return risk_forecast(district_id, as_of)


@st.cache_data(show_spinner="Computing risk for all districts…")
def risk_all(as_of: str) -> pd.DataFrame:
    """Early-warning forecast for every district on one date (districts without enough history are dropped)."""
    from analytics.predictive.extreme_weather import risk_forecast

    rows = []
    for district_id in data.load_districts()["district_id"]:
        try:
            rows.append({"district_id": district_id, **risk_forecast(district_id, as_of)})
        except ValueError:
            continue
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def yield_forecast(
    district_id: str,
    crop: str,
    rain_multiplier: float = 1.0,
    temperature_delta: float = 0.0,
    year: int | None = None,
    lead_days: int = 60,
) -> dict[str, Any]:
    """Yield forecast (t/ha) with an 80 percent interval for the crop's season."""
    from analytics.predictive.predict import predict_yield

    overrides = {"rain_multiplier": rain_multiplier, "temperature_delta": temperature_delta}
    return predict_yield(
        district_id, crop, data.season_of(crop), overrides, year=year, lead_days=lead_days
    )


@st.cache_data(show_spinner=False)
def yield_explanation(
    district_id: str,
    crop: str,
    rain_multiplier: float = 1.0,
    temperature_delta: float = 0.0,
    year: int | None = None,
    lead_days: int = 60,
) -> dict[str, Any]:
    """SHAP contributions (t/ha) of the forecast shown by :func:`yield_forecast`."""
    from analytics.predictive.predict import explain_yield

    overrides = {"rain_multiplier": rain_multiplier, "temperature_delta": temperature_delta}
    return explain_yield(
        district_id, crop, data.season_of(crop), overrides, year=year, lead_days=lead_days
    )


@st.cache_data(show_spinner=False)
def crop_ranking(
    district_id: str,
    season: str,
    rain_multiplier: float = 1.0,
    temperature_delta: float = 0.0,
    year: int | None = None,
) -> pd.DataFrame:
    """Crops of one season ranked by expected net return (Phase 6 economics; approximate prices)."""
    from analytics.prescriptive import crop_switch

    overrides = {"rain_multiplier": rain_multiplier, "temperature_delta": temperature_delta}
    return crop_switch.recommend(district_id, season, overrides, year)


@st.cache_data(show_spinner="Composing advisory…")
def advisory(district_id: str, crop: str, as_of: str, lang: str):
    """The composed advisory (text plus reasoning) for a district, crop, date and language."""
    from advisory.generate import compose

    return compose(district_id, crop, as_of, lang)


def split_of_year(year: int) -> str:
    """Which yield-model split a crop year belongs to (train / validation / test)."""
    split = load_settings()["splits"]["yield"]
    if year <= int(split["train_end"]):
        return "train"
    lo, hi = split["valid"]
    if lo <= year <= hi:
        return "validation"
    return "test"


def yield_years() -> list[int]:
    """Crop years offered as the weather reference of a forecast."""
    years = load_settings()["years"]
    return list(range(int(years["start"]), int(years["end"]) + 1))
