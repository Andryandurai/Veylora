"""Methodology: data provenance, model design, measured results and limitations, all read from reports/*."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from analytics.prescriptive.wording import paired_phrase
from config import load_settings
from dashboards.components import guards, labels, layout


def _satellite_is_synthetic(quality: dict) -> bool:
    """True when the data-quality report says any satellite rows are synthetic (or there is no report)."""
    return quality.get("synthetic_share", {}).get("satellite_16d", 1.0) > 0


def data_section() -> None:
    st.header("Data and provenance")
    quality = guards.read_report("data_quality.json")
    satellite_years = load_settings()["satellite_years"]
    synthetic_satellite = _satellite_is_synthetic(quality)
    rows = [
        (
            "Rainfall, Tmax, Tmin",
            "IMD gridded data (imdlib), area-weighted per district unit",
            "real",
            "1991-2024",
        ),
        ("Humidity, wind, solar", "NASA POWER daily point data", "real", "1991-2024"),
        ("Crop yield, area, production", "ICRISAT DLD district database", "real", "2000-2017"),
        (
            "NDVI, soil moisture (satellite)",
            "Placeholder values, not real satellite observations"
            if synthetic_satellite
            else "MODIS NDVI and ERA5-Land soil moisture (Earth Engine)",
            "SYNTHETIC" if synthetic_satellite else "real",
            f"{satellite_years['start']}-{satellite_years['end']}",
        ),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["Variable", "Source", "Status", "Years"]),
        hide_index=True,
        width="stretch",
    )
    if quality:
        counts = quality.get("rows", {})
        verdict = (
            "all checks passed"
            if quality.get("passed")
            else f"{len(quality.get('failed_checks') or [])} check(s) failed"
        )
        st.markdown(
            f"Data-quality report: {verdict}. Records: "
            + ", ".join(f"{labels.table_label(name)} {n:,}" for name, n in counts.items())
            + f". Crop-to-district join coverage {quality.get('crop_join_coverage', 0):.0%}."
        )
        for table, share in quality.get("synthetic_share", {}).items():
            if share > 0:
                st.warning(
                    f"The {labels.table_label(table)} are {share:.0%} synthetic (placeholder values, marked as such in the data). They are excluded from the served models."
                )
    st.markdown(
        "**District units.** ICRISAT reports yields for the 1966 parent districts, so the 114 current district polygons (geoBoundaries) were regrouped "
        "into 51 yield-reporting units by name anchors, nearest anchor and a small table of manual overrides."
    )


def yield_section() -> None:
    st.header("Yield forecasts")
    report = guards.read_report("yield_models.json")
    metrics_path = guards.ROOT / "reports" / "yield_metrics.csv"
    if not report or not metrics_path.exists():
        st.info("The yield results are not available right now.")
        return
    metrics = pd.read_csv(metrics_path)
    test = metrics[
        (metrics["split"] == "test")
        & (metrics["crop"] == "all")
        & (metrics["lead_days"] == 60)
        & (metrics["feature_set"] == "weather")
    ]
    st.markdown(
        "Target: yield relative to the district's prior mean. Features: weather windows up to the forecast date, anomalies against the district climatology and yield history. "
        "Splits are by time only (train to 2011, validation 2012-14, test 2015-17). The served model is XGBoost with monotone temperature constraints and a conformal 80% interval."
    )
    st.dataframe(
        test[["model", "rmse", "mae", "r2", "rmse_rel"]]
        .rename(
            columns={
                "rmse": "RMSE t/ha",
                "mae": "MAE t/ha",
                "r2": "R²",
                "rmse_rel": "RMSE / mean yield",
            }
        )
        .round(3),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Test years 2015-17, 60-day lead, weather features, all crops pooled. R² is high for every model because yields differ strongly between districts; compare RMSE with the baselines."
    )
    boot = pd.DataFrame(report["bootstrap_vs_baseline"])
    boot = boot[(boot["feature_set"] == "weather") & (boot["lead_days"] == 60)]
    st.markdown(
        "**Relative-RMSE ratio against each baseline** (below 1.0 means better; 95% cluster-bootstrap interval over districts; relative RMSE is the error as a share of each district's prior mean yield):"
    )
    st.dataframe(
        boot[["model", "baseline", "ratio", "lo", "hi", "p_not_better"]]
        .rename(columns={"lo": "ratio low", "hi": "ratio high", "p_not_better": "P(not better)"})
        .round(3),
        hide_index=True,
        width="stretch",
    )
    coverage = report["intervals_test"]["60"]
    st.markdown(
        f"**80% interval coverage on the test years:** {coverage['coverage']:.0%} (target {coverage['target_coverage']:.0%}); by crop "
        + ", ".join(
            f"{k.removeprefix('coverage_')} {v:.0%}"
            for k, v in coverage.items()
            if k.startswith("coverage_") and k != "coverage_valid_calibration"
        )
        + "."
    )
    st.info(
        "Plain reading: the models improve clearly on repeating last year's yield, but they do not clearly beat a simple per-district trend or the prior-three-year mean (the intervals above include or touch 1.0). "
        "Treat the forecast as a weather-adjusted baseline with an honest interval, not as a precise prediction."
    )
    figure = guards.ROOT / "reports" / "figures" / "yield_shap_summary_lead60.png"
    if figure.exists():
        st.image(
            str(figure), caption="SHAP summary of the served model (60-day lead).", width="stretch"
        )


def extreme_section() -> None:
    st.header("Early warning")
    report = guards.read_report("extreme_weather.json")
    if not report:
        st.info("The early-warning results are not available right now.")
        return
    heat, drought = report["heatwave"], report["drought"]
    st.markdown(
        f"**Heatwave onset (March-June).** Rolling-origin evaluation pooled over {heat['pooled_onsets_tested']} onsets; compared with a climatology forecast and persistence."
    )
    pooled = pd.DataFrame(heat["pooled_metrics"])
    pooled = pooled[pooled["horizon_days"] == 14]
    st.dataframe(
        pooled[
            ["forecast", "n_positive", "f1", "pr_auc", "roc_auc", "brier_skill_vs_climatology"]
        ].round(3),
        hide_index=True,
        width="stretch",
    )
    st.info(
        f"Served heatwave probability: **{heat['served']['source']}**. {heat['served']['reason']}."
    )
    served = drought["served_feature_set"]
    test = drought["feature_sets"][served]["test"]
    base = drought["baselines_test"]
    st.markdown(
        "**Drought next month (SPI-3 at or below -1).** Gradient-boosted trees on rainfall-deficit features with isotonic calibration, test years 2020-24."
    )
    table = pd.DataFrame(
        [
            {
                "forecast": "XGBoost (served)",
                "F1": test["f1"],
                "precision": test["precision"],
                "recall": test["recall"],
                "PR-AUC": test["pr_auc"],
                "Brier": test["brier"],
            },
            {
                "forecast": "climatology",
                "F1": base["climatology"]["f1"],
                "precision": base["climatology"]["precision"],
                "recall": base["climatology"]["recall"],
                "PR-AUC": base["climatology"]["pr_auc"],
                "Brier": base["climatology"]["brier"],
            },
            {
                "forecast": "persistence",
                "F1": base["persistence"]["f1"],
                "precision": base["persistence"]["precision"],
                "recall": base["persistence"]["recall"],
                "PR-AUC": base["persistence"]["pr_auc"],
                "Brier": base["persistence"]["brier"],
            },
        ]
    )
    st.dataframe(table.round(3), hide_index=True, width="stretch")
    st.caption(
        "A drought signal for a month that is normally almost dry is shown but does not set the overall risk level."
    )
    figure = guards.ROOT / "reports" / "figures" / "extreme_reliability.png"
    if figure.exists():
        st.image(
            str(figure),
            caption="Reliability of the calibrated probabilities on the test years.",
            width="stretch",
        )


def prescriptive_section() -> None:
    st.header("Irrigation optimization")
    report = guards.read_report("prescriptive_results.json")
    if not report:
        st.info("The irrigation results are not available right now.")
        return
    st.markdown(
        "Daily FAO-56 soil-water balance (Hargreaves ET0, crop coefficients, stress factor Ks; paddy rice as a ponded field with seepage, approximate parameters) with an FAO-33 yield response, and a weekly linear programme (PuLP/CBC) that allocates a seasonal water budget. "
        f"Evaluated on {report['summary'][0]['n_cases']} district-season cases per rainfall scenario; the baseline is a farmer applying 600 mm."
    )
    summary = pd.DataFrame(report["summary"])
    st.dataframe(
        summary[
            [
                "scenario",
                "strategy",
                "water_used_mm",
                "water_saved_pct",
                "rel_yield",
                "yield_vs_baseline",
            ]
        ]
        .rename(
            columns={
                "water_used_mm": "water mm",
                "water_saved_pct": "water saved %",
                "rel_yield": "relative yield",
                "yield_vs_baseline": "yield vs baseline",
            }
        )
        .round(3),
        hide_index=True,
        width="stretch",
    )
    normal = summary[
        (summary["scenario"] == "normal rainfall") & (summary["strategy"] == "optimized")
    ].iloc[0]
    paired = pd.DataFrame(report["paired_optimized_vs_even"]).iloc[0]
    verdict = paired_phrase(
        paired["delta_rel_yield"],
        paired["delta_rel_yield_lo"],
        paired["delta_rel_yield_hi"],
        "yield gain",
        "yield loss",
        "no clear yield difference",
    )
    st.info(
        f"Plain reading (normal rainfall): against the 600 mm baseline the optimized schedule saves {normal['water_saved_pct']:.0f}% of the water "
        f"but keeps only {normal['yield_vs_baseline']:.0%} of the baseline yield. Against an even split it uses {-paired['delta_water_mm']:.0f} mm less water "
        f"with {verdict} ({paired['delta_rel_yield']:+.3f}, interval {paired['delta_rel_yield_lo']:+.4f} to {paired['delta_rel_yield_hi']:+.4f}; "
        f"the optimized schedule is worse in {int(paired['cases_optimized_worse'])} of {int(paired['cases'])} cases). "
        "Whether the yield trade-off is acceptable is a decision for agronomists and farmers."
    )


def limits_section() -> None:
    st.header("Limitations")
    if _satellite_is_synthetic(guards.read_report("data_quality.json")):
        satellite_limit = "The satellite data (vegetation and soil moisture) is synthetic placeholder data, not real observations."
    else:
        satellite_limit = "Satellite vegetation and soil moisture are district averages from coarse grids (250 m vegetation index, about 9 km soil moisture), so clouds and mixed land use blur them."
    st.markdown(
        "- Yields are observed to 2017 only, so test years 2015-17 are the latest honest hold-out; nothing here is validated on recent seasons.\n"
        "- Forecast skill over simple baselines is small or unproven (see above). Use the intervals, not just the point forecast.\n"
        f"- {satellite_limit}\n"
        "- Crop and price parameters are approximate and not verified against current sources.\n"
        "- Advisory translations (Hindi, Tamil, Telugu) are drafts and need native-speaker and agronomist review; this site does not send messages.\n"
        "- The 3D simulator was tested with software rendering only; its frame rate on real hardware is not measured.\n"
        "- District units are 1966 parent districts; results are not village-level."
    )


def main() -> None:
    layout.start(
        "Methodology",
        needs=("processed data",),
        crop=False,
        label="Evidence",
        intro="How the numbers on the other pages were produced, and how far to trust them. Everything below comes from the project's own evaluation results.",
    )
    tabs = st.tabs(["Data", "Yield", "Early warning", "Irrigation", "Limitations"])
    for tab, section in zip(
        tabs,
        (data_section, yield_section, extreme_section, prescriptive_section, limits_section),
        strict=True,
    ):
        with tab:
            section()


guards.run_guarded(main)
