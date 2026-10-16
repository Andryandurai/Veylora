"""Yield predictor: weather what-if forecast with an 80 percent interval, drivers and crop comparison."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dashboards.components import charts, data, guards, layout, models

LEADS = {
    30: "30 days into the season",
    60: "60 days into the season",
    90: "90 days into the season",
}


def _report_note() -> str | None:
    """One honest sentence on how the served model compares with the district-trend baseline (from reports)."""
    report = guards.read_report("yield_models.json")
    rows = [
        r
        for r in report.get("bootstrap_vs_baseline", [])
        if r["model"] == "xgboost"
        and r["baseline"] == "district_trend"
        and r["feature_set"] == "weather"
        and r["lead_days"] == 60
    ]
    if not rows:
        return None
    r = rows[0]
    verdict = (
        "no clear improvement over"
        if r["lo"] <= 1.0 <= r["hi"]
        else ("better than" if r["hi"] < 1.0 else "worse than")
    )
    return f"On held-out years 2015-17 the served model shows {verdict} a simple district trend (relative-RMSE ratio {r['ratio']:.2f}, 95% interval {r['lo']:.2f}-{r['hi']:.2f})."


def _sensitivity_note() -> str | None:
    """How strongly the served model reacts to weather what-ifs on average (validation years, from reports)."""
    rows = {
        r["scenario"]: r["mean_change_pct"]
        for r in guards.read_report("yield_models.json").get("scenario_response", [])
        if r["crop"] == "all"
    }
    if "rain x0.6" not in rows or "+4 C" not in rows:
        return None
    return (
        f"Sensitivity check: across validation years the model moves the forecast by only {rows['rain x0.6']:+.1f}% "
        f"for 40% less rain and {rows['+4 C']:+.1f}% for +4 °C. It learned yield mostly from each district's own history, "
        "so read the sliders as a mild weather adjustment, not a crop-physiology simulation (see the irrigation results and the 3D simulator for water stress)."
    )


def main() -> None:
    district, crop = layout.start(
        "Yield",
        needs=("processed data", "district clusters", "yield models"),
        label="Forecast",
        intro="Seasonal yield forecast from weather observed so far plus what-if changes to rainfall and temperature.",
    )
    season = data.season_of(crop)
    history = data.yield_history(district, crop)
    if history.empty:
        st.info(
            f"{crop.capitalize()} has no yield records for this district, so no forecast can be made. Choose another crop or district."
        )
        return

    with st.sidebar:
        st.header("What-if")
        rain = st.slider(
            "Rainfall vs the reference season",
            0.4,
            1.6,
            1.0,
            0.05,
            format="x%.2f",
            help="Multiplies every daily rainfall value of the reference season.",
        )
        delta = st.slider(
            "Temperature change (°C)",
            -2.0,
            4.0,
            0.0,
            0.5,
            help="Added to every daily Tmax and Tmin. The model's response to warming is constrained to be non-increasing.",
        )
        lead = st.selectbox("Forecast issued", list(LEADS), index=1, format_func=LEADS.get)
        years = ["typical"] + models.yield_years()
        pick = st.selectbox(
            "Reference season weather",
            years,
            format_func=lambda y: "Typical year (median rainfall)" if y == "typical" else str(y),
        )
        year = None if pick == "typical" else int(pick)

    forecast = models.yield_forecast(district, crop, rain, delta, year, lead)
    ref_year = int(forecast["reference_year"])
    mean5, n5 = data.five_year_mean(district, crop, ref_year)

    cols = st.columns(4)
    cols[0].metric(
        "Forecast yield",
        f"{forecast['yield_t_ha']:.2f} t/ha",
        help="Point forecast of the served XGBoost model.",
    )
    cols[1].metric(
        "80% interval",
        f"{forecast['lo']:.2f} - {forecast['hi']:.2f}",
        help="Conformalised quantile interval; calibrated on 2012-14 and checked on 2015-17.",
    )
    if mean5 is not None:
        change = forecast["yield_t_ha"] / mean5 - 1
        cols[2].metric(
            f"Mean of previous {n5} years",
            f"{mean5:.2f} t/ha",
            f"{change:+.0%} forecast vs mean",
            delta_color="off",
        )
    observed = history[history["year"] == ref_year]["yield_t_per_ha"]
    if len(observed) and pick != "typical":
        cols[3].metric(
            f"Observed {ref_year}",
            f"{observed.iloc[0]:.2f} t/ha",
            help=f"{ref_year} belongs to the model's {models.split_of_year(ref_year)} years.",
        )
    else:
        cols[3].metric("Reference weather year", ref_year)
    st.caption(
        f"{crop.capitalize()} {season} season, reference weather from {ref_year}, rainfall x{rain:.2f}, temperature {delta:+.1f} °C. "
        + (_report_note() or "")
    )

    if (sensitivity := _sensitivity_note()) is not None:
        st.caption(sensitivity)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Yield history and forecast")
        st.altair_chart(charts.history_chart(history, forecast, ref_year), width="stretch")
    with right:
        st.subheader("What drives this forecast")
        explanation = models.yield_explanation(district, crop, rain, delta, year, lead)
        st.altair_chart(charts.contributions_chart(explanation["contributions"]), width="stretch")
        st.caption(
            f"Effects relative to the average forecast of {explanation['base_t_ha']:.2f} t/ha (SHAP values). Association, not proof of cause."
        )

    st.subheader(f"Which {season} crop pays best here?")
    try:
        ranking = models.crop_ranking(district, season, rain, delta, year)
    except ValueError as exc:
        st.info(f"No crop comparison available: {exc}")
    else:
        st.altair_chart(charts.crop_ranking_chart(ranking), width="stretch")
        shown = ranking[
            [
                "crop",
                "yield_point_t_ha",
                "yield_lo_t_ha",
                "yield_hi_t_ha",
                "net_mid_inr_ha",
                "net_lower_inr_ha",
                "rank_mid",
                "rank_lower",
            ]
        ].rename(
            columns={
                "yield_point_t_ha": "yield t/ha",
                "yield_lo_t_ha": "low t/ha",
                "yield_hi_t_ha": "high t/ha",
                "net_mid_inr_ha": "net return INR/ha",
                "net_lower_inr_ha": "net at low end",
                "rank_mid": "rank (expected)",
                "rank_lower": "rank (pessimistic)",
            }
        )
        with st.expander("Table"):
            st.dataframe(shown, hide_index=True, width="stretch")
        if (ranking["rank_mid"] != ranking["rank_lower"]).any():
            st.warning(
                "The ranking changes when the pessimistic end of each interval is used: the best expected crop is not the safest."
            )
        st.caption("Prices and costs are approximate and must be verified before real decisions.")

    with st.expander("Forecast details"):
        details = pd.DataFrame(
            {
                "Item": [
                    "Forecast yield",
                    "80% interval",
                    "Reference weather year",
                    "Forecast issued",
                    "Method",
                ],
                "Value": [
                    f"{forecast['yield_t_ha']:.2f} t/ha",
                    f"{forecast['lo']:.2f} to {forecast['hi']:.2f} t/ha",
                    str(forecast["reference_year"]),
                    f"{forecast['lead_days']} days into the season",
                    "Gradient-boosted trees on weather and yield history, with a conformal interval",
                ],
            }
        )
        st.dataframe(details, hide_index=True, width="stretch")


guards.run_guarded(main)
