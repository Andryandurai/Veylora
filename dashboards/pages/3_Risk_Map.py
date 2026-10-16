"""Risk map: early-warning probabilities on a date, and the long-term climate vulnerability of each district."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dashboards.components import charts, data, guards, layout, maps, models


def early_warning_tab() -> None:
    first, last = models.risk_range()
    default = min(max(pd.Timestamp("2024-06-15"), first), last)
    as_of = st.date_input(
        "As-of date",
        value=default.date(),
        min_value=first.date(),
        max_value=last.date(),
        key="map_date",
        help="Replay on observed history; no live weather feed is connected.",
    )
    stamp = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    hazard = st.radio("Hazard", ["Drought next month", "Heatwave within 14 days"], horizontal=True)
    table = models.risk_all(stamp)
    if table.empty:
        st.info("No district has enough history for this date.")
        return
    column = "drought_p" if hazard.startswith("Drought") else "heatwave_p"
    table = table.assign(
        probability=table[column],
        level_label=table["level"].map(charts.LEVEL_ICON),
        counts_towards_level=table["drought_relevant"].map({True: "yes", False: "no (dry month)"})
        if column == "drought_p"
        else "yes",
    ).merge(data.load_districts()[["district_id", "district", "state"]], on="district_id")
    extra = ["counts_towards_level"]
    st.altair_chart(
        maps.choropleth(table, "probability", "Probability", (0.0, 1.0), extra),
        width="stretch",
    )
    levels = table["level"].value_counts().reindex(list(charts.LEVEL_ICON), fill_value=0)
    cols = st.columns(4)
    for col, (level, label) in zip(cols, charts.LEVEL_ICON.items(), strict=True):
        col.metric(
            f"Districts {label}",
            int(levels[level]),
            help="Overall level (heatwave and drought combined) for each district.",
        )
    if column == "heatwave_p" and not bool(table["heatwave_in_season"].iloc[0]):
        st.caption(
            "The heatwave model covers March-June only; outside the season every probability is reported as 0."
        )
    top = table.sort_values("probability", ascending=False).head(10)[
        ["district", "state", "probability", "level_label", "counts_towards_level"]
    ]
    with st.expander("Ten districts with the highest probability"):
        st.dataframe(
            top.rename(
                columns={"level_label": "overall level", "counts_towards_level": "sets the level"}
            ),
            hide_index=True,
            width="stretch",
            column_config={"probability": st.column_config.NumberColumn(format="percent")},
        )


def vulnerability_tab() -> None:
    table = data.vulnerability_table()
    st.markdown(
        "Long-term vulnerability = average percentile rank of three indicators per district: **kharif drought frequency** "
        "(season rain below 75% of the district mean), **rice-yield variability** (coefficient of variation, 2000-2017) and "
        "**mean heat days** (Tmax of 40 °C or more) per kharif season. Higher means more exposed. It summarises history; it is not a forecast."
    )
    st.altair_chart(
        maps.choropleth(table, "vulnerability_score", "Vulnerability score (0-1)", (0.0, 1.0)),
        width="stretch",
    )
    shown = table[
        [
            "district",
            "state",
            "vulnerability_score",
            "drought_frequency",
            "rice_yield_cv",
            "mean_heat_days",
        ]
    ]
    with st.expander("Most vulnerable districts"):
        st.dataframe(
            shown.head(15),
            hide_index=True,
            width="stretch",
            column_config={
                "vulnerability_score": st.column_config.NumberColumn("score", format="%.2f"),
                "drought_frequency": st.column_config.NumberColumn(
                    "drought years", format="percent"
                ),
                "rice_yield_cv": st.column_config.NumberColumn("rice yield CV", format="%.2f"),
                "mean_heat_days": st.column_config.NumberColumn(
                    "heat days / kharif", format="%.1f"
                ),
            },
        )


def main() -> None:
    layout.start(
        "Risk map",
        "🗺️",
        needs=("processed data", "early-warning models"),
        crop=False,
        intro="Where heatwave and drought risk is highest, and which districts are structurally most exposed.",
    )
    early, structural = st.tabs(["Early warning on a date", "Long-term vulnerability"])
    with early:
        early_warning_tab()
    with structural:
        vulnerability_tab()


guards.run_guarded(main)
