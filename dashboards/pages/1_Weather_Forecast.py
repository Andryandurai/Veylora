"""Weather and early warning: recent observed weather and the heatwave / drought risk for a district and date."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from config import load_settings
from dashboards.components import charts, data, guards, labels, layout, models

LEVEL_HELP = {
    "low": "No action needed beyond routine monitoring.",
    "moderate": "Watch the forecast and prepare: check irrigation sources and shade or mulch options.",
    "high": "Act now: schedule irrigation, protect sensitive stages and tell neighbouring farmers.",
    "severe": "Urgent: the model sees a high chance of a damaging event; contact the agriculture office.",
}


def main() -> None:
    district, _ = layout.start(
        "Early warning",
        needs=("processed data", "early-warning models"),
        crop=False,
        label="Early warning",
        intro="Replay of the early-warning models on observed history. No live forecast feed is connected.",
    )
    first, last = models.risk_range()
    default = min(max(pd.Timestamp("2024-06-15"), first), last)
    as_of = st.sidebar.date_input(
        "As-of date",
        value=default.date(),
        min_value=first.date(),
        max_value=last.date(),
        help="The forecast uses only weather observed up to and including this date.",
    )
    stamp = pd.Timestamp(as_of).strftime("%Y-%m-%d")

    forecast = models.risk(district, stamp)
    thresholds = load_settings()["risk_levels"]
    level = forecast["level"]

    head, detail = st.columns([1, 2])
    with head:
        st.metric("Overall risk level", charts.LEVEL_LABEL[level])
        st.write(LEVEL_HELP[level])
        hazard = {"heatwave": "heatwave within 14 days", "drought": "drought next month"}.get(
            forecast["hazard"], forecast["hazard"]
        )
        if level != "low":
            st.caption(f"Set by: {hazard}.")
    with detail:
        cols = st.columns(2)
        cols[0].metric(
            "Heatwave onset within 14 days",
            f"{forecast['heatwave_p']:.1%}",
            help="Probability that a heatwave starts in the next 14 days. Zero outside the March-June season the model covers.",
        )
        cols[1].metric(
            "Drought next month",
            f"{forecast['drought_p']:.1%}",
            help=f"Chance that rainfall over the coming months will be unusually low (a drought month), using weather up to {forecast['drought_month']}.",
        )
        if not forecast["heatwave_in_season"]:
            st.caption(
                "Heatwave model is out of season (it covers March-June); the probability is reported as 0."
            )
        else:
            st.caption(
                f"This heatwave chance is based on {labels.HEATWAVE_SOURCE.get(forecast['heatwave_source'], forecast['heatwave_source'])}. A more complex model was tested and did not do better (see Methodology)."
            )
        if not forecast["drought_relevant"]:
            st.caption(
                f"The coming month normally receives only {forecast['drought_normal_month_rain_mm']:.0f} mm of rain, so a drought signal is shown but does not set the overall level."
            )

    probabilities = {
        "Heatwave in 3 days": forecast["heatwave_p3"],
        "Heatwave in 7 days": forecast["heatwave_p7"],
        "Heatwave in 14 days": forecast["heatwave_p"],
        "Drought next month": forecast["drought_p"],
    }
    st.subheader("Hazard probabilities")
    st.altair_chart(charts.probability_chart(probabilities), width="stretch")
    st.caption(
        "Risk levels (probability): heatwave moderate/high/severe from "
        f"{thresholds['heatwave']['moderate']:.0%}/{thresholds['heatwave']['high']:.0%}/{thresholds['heatwave']['severe']:.0%}; "
        f"drought from {thresholds['drought']['moderate']:.0%}/{thresholds['drought']['high']:.0%}/{thresholds['drought']['severe']:.0%}."
    )

    st.subheader("Observed weather, last 60 days")
    weather = data.recent_weather(district, stamp, 60)
    if weather.empty:
        st.info("No weather observations for this window.")
    else:
        left, right = st.columns(2)
        left.altair_chart(charts.rain_chart(weather), width="stretch")
        right.altair_chart(charts.temperature_chart(weather), width="stretch")
        cols = st.columns(3)
        cols[0].metric("Rain, 60 days", f"{weather['rain_mm'].sum():.0f} mm")
        cols[1].metric("Hottest day", f"{weather['tmax_c'].max():.1f} °C")
        cols[2].metric(
            f"Days at or above {charts.HEAT_TMAX_C:.0f} °C",
            int((weather["tmax_c"] >= charts.HEAT_TMAX_C).sum()),
        )
        with st.expander("Data table"):
            st.dataframe(
                weather[["date", "rain_mm", "tmax_c", "tmin_c"]].rename(
                    columns={
                        "date": "Date",
                        "rain_mm": "Rain (mm)",
                        "tmax_c": "Highest temperature (°C)",
                        "tmin_c": "Lowest temperature (°C)",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
    st.caption(
        "Weather: IMD gridded rainfall and temperature, area-weighted over the district unit."
    )


guards.run_guarded(main)
