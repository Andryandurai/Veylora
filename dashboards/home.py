"""Home page of the dashboards (routed by ``app.py``)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from config import load_settings
from dashboards.components import data, guards, labels, layout, models, ui


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
                    "Typical yield forecast error (2015-17)",
                    f"{sel.loc['xgboost', 'rmse_rel']:.0%}",
                    f"of the average yield. Using {labels.BASELINE_PHRASE[best]} instead gives "
                    f"{sel.loc[best, 'rmse_rel']:.0%}.",
                )
            )
    extreme = guards.read_report("extreme_weather.json")
    if extreme:
        test = extreme["drought"]["feature_sets"][extreme["drought"]["served_feature_set"]]["test"]
        tiles.append(
            (
                "Drought warning score (2020-24)",
                f"{test['f1']:.2f}",
                f"Out of 1.00 (higher is better). A guess from past averages scores "
                f"{extreme['drought']['baselines_test']['climatology']['f1']:.2f}.",
            )
        )
    return tiles


def main() -> None:
    layout.start(
        ui.SITE_NAME,
        crop=False,
        hero=True,
        label="Climate risk · Crop yield · Irrigation",
        intro="District-level climate risk, yield forecasts and irrigation decision support for Andhra Pradesh, Telangana, Karnataka and Tamil Nadu.",
    )
    settings = load_settings()
    years, weather_years = settings["years"], settings["weather_years"]
    districts = data.load_districts()
    cols = st.columns(4)
    cols[0].metric("Districts", len(districts))
    cols[1].metric("Crops", len(settings["crops"]))
    cols[2].metric("Weather record", f"{weather_years['start']}-{weather_years['end']}")
    cols[3].metric("Yield record", f"{years['start']}-{years['end']}")

    st.subheader("Explore")
    tiles = [
        (
            "pages/1_Weather_Forecast.py",
            "Early warning",
            "Heatwave and drought risk for a district and a date.",
        ),
        (
            "pages/2_Yield_Predictor.py",
            "Yield",
            "Forecast with an interval, what-if rainfall and temperature, crop choice.",
        ),
        (
            "pages/3_Risk_Map.py",
            "Risk map",
            "Early-warning probabilities and long-term vulnerability by district.",
        ),
        (
            "pages/4_Advisory.py",
            "Advisory",
            "Short farmer messages in English, Hindi, Tamil and Telugu.",
        ),
        (
            "pages/5_3D_Simulator.py",
            "Simulator",
            "The season week by week under climate and irrigation scenarios.",
        ),
        ("pages/6_Methodology.py", "Methodology", "Data, models, measured results and limits."),
    ]
    for row in (tiles[:3], tiles[3:]):
        for col, (page, title, text) in zip(st.columns(3), row, strict=True):
            with col:
                ui.tile(page, title, text)

    measured = _headline_numbers()
    if measured:
        st.subheader("Measured results")
        st.caption(
            "Checked on years the models never saw. The yield forecast is about as good as simple methods, not clearly better; Methodology has the details."
        )
        for col, (label, value, note) in zip(st.columns(len(measured)), measured, strict=True):
            col.metric(label, value, help=note)
            col.caption(note)

    with st.expander("Read this before using any number"):
        notes = [
            "Units are the 51 district groups for which ICRISAT reports yields (1966 boundaries), not villages or today's 114 districts.",
            "There is **no live weather feed**: the early-warning pages replay the models on observed history up to the end of the weather record.",
            "The forecasts are decision support with uncertainty. They are not a substitute for local agronomic advice.",
            "Crop-economics parameters are approximate and must be checked before real use; advisory translations are drafts that need native-speaker review.",
        ]
        st.markdown("\n".join(f"- {n}" for n in notes))
    models_ready = not guards.missing(["yield models", "early-warning models"])
    if models_ready:
        models.warm_up()


guards.run_guarded(main)
