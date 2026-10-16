"""Home page of the dashboards (routed by ``app.py``)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from config import load_settings
from dashboards.components import data, guards, layout, models


def _headline_numbers() -> list[tuple[str, str, str]]:
    """(label, value, note) tiles built from the report files; empty if the pipeline has not been run."""
    tiles = []
    metrics = guards.ROOT / "reports" / "yield_metrics.csv"
    if metrics.exists():
        m = pd.read_csv(metrics)
        sel = m[
            (m["split"] == "test")
            & (m["crop"] == "all")
            & (m["lead_days"] == 60)
            & (m["feature_set"] == "weather")
        ].set_index("model")
        baselines = [
            b for b in ("naive_last_year", "district_trend", "prior_mean_3y") if b in sel.index
        ]
        if "xgboost" in sel.index and baselines:
            best = min(baselines, key=lambda name: sel.loc[name, "rmse_rel"])
            tiles.append(
                (
                    "Yield error vs mean yield (test 2015-17)",
                    f"{sel.loc['xgboost', 'rmse_rel']:.0%}",
                    f"best simple baseline ({best.replace('_', ' ')}): {sel.loc[best, 'rmse_rel']:.0%}; "
                    f"pooled RMSE {sel.loc['xgboost', 'rmse']:.2f} t/ha, dominated by sugarcane",
                )
            )
    extreme = guards.read_report("extreme_weather.json")
    if extreme:
        test = extreme["drought"]["feature_sets"][extreme["drought"]["served_feature_set"]]["test"]
        tiles.append(
            (
                "Drought-warning F1 (test 2020-24)",
                f"{test['f1']:.2f}",
                f"PR-AUC {test['pr_auc']:.2f} vs climatology {extreme['drought']['baselines_test']['climatology']['pr_auc']:.2f}",
            )
        )
    return tiles


def main() -> None:
    layout.start(
        "ClimateResilience India",
        "🌾",
        crop=False,
        intro="District-level climate risk, yield forecasts and irrigation decision support for Andhra Pradesh, Telangana, Karnataka and Tamil Nadu.",
    )
    settings = load_settings()
    years, weather_years = settings["years"], settings["weather_years"]
    districts = data.load_districts()
    cols = st.columns(4)
    cols[0].metric("Yield-reporting districts", len(districts))
    cols[1].metric("Crops", len(settings["crops"]))
    cols[2].metric("Weather record", f"{weather_years['start']}-{weather_years['end']}")
    cols[3].metric("Yield record", f"{years['start']}-{years['end']}")

    st.subheader("Where to start")
    st.page_link(
        "pages/1_Weather_Forecast.py",
        label="Weather and early warning: heatwave and drought risk for a district and date",
        icon="🌦️",
    )
    st.page_link(
        "pages/2_Yield_Predictor.py",
        label="Yield predictor: what-if rainfall and temperature, drivers, crop choice",
        icon="📈",
    )
    st.page_link(
        "pages/3_Risk_Map.py",
        label="Risk map: early-warning and long-term vulnerability by district",
        icon="🗺️",
    )
    st.page_link(
        "pages/4_Advisory.py",
        label="Farmer advisory: SMS text in English, Hindi, Tamil and Telugu",
        icon="📱",
    )
    st.page_link(
        "pages/5_3D_Simulator.py",
        label="3D simulator: the season week by week under climate scenarios",
        icon="🧊",
    )
    st.page_link(
        "pages/6_Methodology.py",
        label="Methodology: data, models, measured results and limits",
        icon="📚",
    )

    tiles = _headline_numbers()
    if tiles:
        st.subheader("Measured results (held-out years)")
        for col, (label, value, note) in zip(st.columns(len(tiles)), tiles, strict=True):
            col.metric(label, value, help=note)
            col.caption(note)
        st.caption(
            "The yield model is level with the best simple baselines, not clearly better; see Methodology for the confidence intervals and the other findings."
        )

    with st.expander("Read this before using any number", expanded=False):
        st.markdown(
            "- Units are the 51 district groups for which ICRISAT reports yields (1966 boundaries), not villages or today's 114 districts.\n"
            "- There is **no live weather feed**: the early-warning pages replay the models on observed history up to the end of the weather record.\n"
            "- The forecasts are decision support with uncertainty. They are not a substitute for local agronomic advice.\n"
            "- Crop-economics parameters are approximate and must be checked before real use; advisory translations are drafts that need native-speaker review."
        )
    models_ready = not guards.missing(["yield models", "early-warning models"])
    if models_ready:
        models.warm_up()


guards.run_guarded(main)
