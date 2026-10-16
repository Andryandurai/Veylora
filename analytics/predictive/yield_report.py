"""Write reports/yield_model_comparison.md from the evaluation results (no number typed by hand)."""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from analytics import common
from config import ROOT, get_path

LOG = logging.getLogger(__name__)

ORDER = [
    "naive_last_year",
    "district_trend",
    "prior_mean_3y",
    "ridge",
    "random_forest",
    "xgboost",
    "xgboost_unconstrained",
    "lstm",
]
LABELS = {
    "naive_last_year": "naive (last year)",
    "district_trend": "district linear trend",
    "prior_mean_3y": "mean of previous 3 years",
    "ridge": "ridge",
    "random_forest": "random forest",
    "xgboost": "XGBoost (temperature-monotone)",
    "xgboost_unconstrained": "XGBoost (unconstrained)",
    "lstm": "LSTM",
}


def _table(frame: pd.DataFrame, digits: int = 3) -> str:
    return common.markdown_table(frame.reset_index(drop=True), digits)


def _verdict(row: dict[str, Any]) -> str:
    """Plain-language verdict from a bootstrap row (ratio of relative RMSE, model / baseline)."""
    ratio, lo, hi = row["ratio"], row["lo"], row["hi"]
    if hi < 1.0:
        return f"better ({(1 - ratio) * 100:.1f}% lower relative RMSE; CI {lo:.3f}-{hi:.3f} excludes 1)"
    if lo > 1.0:
        return f"WORSE ({(ratio - 1) * 100:.1f}% higher relative RMSE; CI {lo:.3f}-{hi:.3f} excludes 1)"
    return f"not distinguishable (ratio {ratio:.3f}, CI {lo:.3f}-{hi:.3f} includes 1)"


def write_report(results: dict[str, Any], metrics: pd.DataFrame) -> None:
    """Render the markdown report and a copy of the key numbers for the dashboards."""
    meta = results["meta"]
    split = meta["split"]
    lines: list[str] = []
    if meta["labels_synthetic"]:
        lines.append(
            "> **SYNTHETIC DATA: the yield labels themselves are synthetic. Nothing below describes real crops.**"
        )
    else:
        lines.append(
            "> Yield labels: real (ICRISAT District Level Database, 1991-2017), weather: IMD / NASA POWER (see docs/DATA_QUALITY.md)."
        )
    if meta["satellite_features_synthetic"]:
        lines.append(
            "> **Satellite features (NDVI, soil moisture) are SYNTHETIC** (Google Earth Engine not authenticated): they are derived from rainfall, "
            "so the `weather+satellite` rows below cannot show whether real satellite data would help."
        )
    lines += [
        "",
        "# Yield model comparison",
        "",
        f"Generated {meta['generated_at']} (commit `{meta['git_commit']}`, seed {meta['seed']}, data from {meta['data_source']}, {meta['n_optuna_trials']} Optuna trials, {meta['runtime_s']:.0f} s).",
        "",
        "## Setup",
        "",
        "- **Unit**: district x crop x season x crop year; target `y_rel` = yield / mean of the same district-crop's earlier yields (scale-free so crops pool); reported errors are converted back to t/ha.",
        f"- **Time split** (never random): train crop years 2000-{split['train_end']}, validation {split['valid'][0]}-{split['valid'][1]} (tuning, early stopping, conformal calibration), test {split['test'][0]}-{split['test'][1]} (touched once, here). "
        "The plan's 2018 / 2019-21 / 2022-24 split is not possible because real yields end in 2017 (D-006).",
        "- **Forecast leads**: 30, 60 and 90 days after the season starts. Features use only season days before the lead, satellite composites that had finished by then, and yields of earlier years (leakage test in `tests/test_predictive.py`).",
        "- **Baselines** (no weather, earlier years only): naive last-year yield (required by the plan), a district linear trend, and the mean of the previous three years. The 3-year-mean baseline was added after the first evaluation round, when the first test numbers had already been seen; adding a stronger baseline can only make the comparison more conservative.",
        "- **Feature sets**: `weather` (season-to-date weather and anomalies, history, crop/season/cluster one-hots) and `weather+satellite` (adds NDVI / soil moisture).",
        "",
        "Rows per dataset (train / valid / test): "
        + "; ".join(
            f"{k}: {v['train']} / {v['valid']} / {v['test']}"
            for k, v in list(results["n_rows"].items())[:3]
        ),
        "",
    ]

    # Headline
    boot = pd.DataFrame(results["bootstrap_vs_baseline"])
    lead_head = 60
    sel = boot[
        (boot["feature_set"] == "weather")
        & (boot["lead_days"] == lead_head)
        & (boot["model"] == "xgboost")
    ]
    lines += ["## Headline", ""]
    for _, row in sel.iterrows():
        lines.append(
            f"- XGBoost (weather, lead {lead_head} d) vs **{LABELS[row['baseline']]}** on the test years: {_verdict(row.to_dict())}."
        )
    best = results["best_by_validation"][f"weather|{lead_head}"]
    lines += [
        f"- Best learned model on the validation years (weather, lead {lead_head} d): **{LABELS[best['model']]}**.",
        "",
    ]
    valid_pooled = metrics[
        (metrics["split"] == "valid")
        & (metrics["crop"] == "all")
        & (metrics["feature_set"] == "weather")
        & (metrics["lead_days"] == lead_head)
    ].set_index("model")["rmse_rel"]
    test_pooled = metrics[
        (metrics["split"] == "test")
        & (metrics["crop"] == "all")
        & (metrics["feature_set"] == "weather")
        & (metrics["lead_days"] == lead_head)
    ].set_index("model")["rmse_rel"]
    lines += [
        "Relative RMSE at this lead, validation vs test years (a large gap means validation results do not carry over):",
        "",
        _table(
            pd.DataFrame(
                {
                    "model": [LABELS[m] for m in ORDER],
                    "valid_rmse_rel": [valid_pooled[m] for m in ORDER],
                    "test_rmse_rel": [test_pooled[m] for m in ORDER],
                }
            )
        ),
        "",
    ]

    # Pooled test metrics
    test = metrics[(metrics["split"] == "test") & (metrics["crop"] == "all")].copy()
    test["model"] = pd.Categorical(test["model"], ORDER, ordered=True)
    lines += [
        "## Test-set results, all crops pooled",
        "",
        "`rmse_rel` is comparable across crops; `rmse`/`mae` are in t/ha and dominated by sugarcane.",
        "",
    ]
    for feature_set in ("weather", "weather+satellite"):
        sub = test[test["feature_set"] == feature_set].sort_values(["lead_days", "model"])
        lines += [
            f"### Feature set: {feature_set}",
            "",
            _table(
                sub[["lead_days", "model", "n", "rmse_rel", "rmse", "mae", "r2", "mape"]].assign(
                    model=lambda d: d["model"].map(LABELS)
                )
            ),
            "",
        ]

    # By crop at lead 60
    by_crop = metrics[
        (metrics["split"] == "test")
        & (metrics["crop"] != "all")
        & (metrics["feature_set"] == "weather")
        & (metrics["lead_days"] == lead_head)
    ].copy()
    by_crop["model"] = pd.Categorical(by_crop["model"], ORDER, ordered=True)
    lines += [
        f"## Test-set results by crop (weather, lead {lead_head} d)",
        "",
        _table(
            by_crop.sort_values(["crop", "model"]).assign(model=lambda d: d["model"].map(LABELS))[
                ["crop", "model", "n", "rmse", "mae", "r2", "mape", "rmse_rel"]
            ]
        ),
        "",
    ]

    # Satellite effect
    pivot = test.pivot_table(
        index=["lead_days", "model"], columns="feature_set", values="rmse_rel", observed=True
    ).reset_index()
    if {"weather", "weather+satellite"} <= set(pivot.columns):
        pivot["change_with_satellite"] = pivot["weather+satellite"] - pivot["weather"]
        pivot["model"] = pivot["model"].map(LABELS)
        lines += [
            "## Does the satellite feature set help?",
            "",
            "Test `rmse_rel`; negative change = satellite helps."
            + (
                " **Satellite data is synthetic, so this is not evidence about real NDVI.**"
                if meta["satellite_features_synthetic"]
                else ""
            ),
            "",
            _table(pivot),
            "",
        ]

    # Bootstrap
    lines += [
        "## Paired cluster bootstrap against the baselines (test years)",
        "",
        "Ratio of relative RMSE (model / baseline), 2000 resamples of districts; < 1 means the model is better.",
        "",
    ]
    show = boot[boot["feature_set"] == "weather"].copy()
    show["verdict"] = [_verdict(r) for r in show.to_dict("records")]
    show["model"], show["baseline"] = show["model"].map(LABELS), show["baseline"].map(LABELS)
    lines += [
        _table(
            show[
                [
                    "lead_days",
                    "model",
                    "baseline",
                    "selected_on_validation",
                    "ratio",
                    "lo",
                    "hi",
                    "p_not_better",
                    "verdict",
                ]
            ]
        ),
        "",
    ]

    # LOSO
    loso = pd.DataFrame(results["leave_one_state_out"]["rows"])
    loso["xgb_vs_naive"] = loso["xgb_rmse_rel"] / loso["naive_rmse_rel"]
    lines += [
        f"## Spatial generalisation: leave-one-state-out (XGBoost, weather, lead {results['leave_one_state_out']['lead_days']} d)",
        "",
        "Trained on the other states' rows up to the end of validation, tested on the held-out state's test years.",
        "",
        _table(loso),
        "",
    ]

    # Intervals
    iv = pd.DataFrame(results["intervals_test"]).T.reset_index(names="lead_days")
    iv["n"] = iv["n"].astype(int)
    lines += [
        "## 80 percent prediction intervals (conformalized quantile regression, weather)",
        "",
        f"Calibrated on the validation years; empirical coverage on the test years (target {iv['target_coverage'].iloc[0]:.0%}).",
        "",
        _table(
            iv[
                [
                    "lead_days",
                    "n",
                    "coverage",
                    "mean_width_rel",
                    "coverage_valid_calibration",
                    "conformal_qhat_rel",
                ]
            ]
        ),
        "",
    ]

    # What-if behaviour
    scen = pd.DataFrame(results["scenario_response"])
    if len(scen):
        order = list(dict.fromkeys(scen["scenario"]))
        wide = (
            scen[scen["crop"].isin(["all", "rice", "cotton"])]
            .pivot_table(index="scenario", columns="crop", values="mean_change_pct", sort=False)
            .reindex(order)
            .reset_index()
        )
        share = (
            scen[scen["crop"] == "all"]
            .set_index("scenario")["share_increase"]
            .reindex(order)
            .reset_index(drop=True)
        )
        wide["share_of_cases_with_higher_forecast"] = share
        lines += [
            "## What-if behaviour of the served model (validation years, lead 60 d)",
            "",
            "Mean percentage change of the forecast when the season's daily weather is rescaled (`predict_yield` overrides), over every district-crop-year of the validation period. "
            "Temperature features are constrained to be non-increasing (D-014) because the unconstrained model predicted yield *gains* from large warming (extrapolation artefact of trees).",
            "",
            _table(wide, 2),
            "",
        ]
        phase3 = ROOT / "reports" / "analytics" / "regression.json"
        if phase3.exists():
            fe = (
                json.loads(phase3.read_text(encoding="utf-8"))
                .get("rice", {})
                .get("M3_district_FE_trend_weather", {})
                .get("tmax_c", {})
            )
            if fe:
                rice_1c = scen[(scen["scenario"] == "+1 C") & (scen["crop"] == "rice")]
                model_text = (
                    f" The served model's average response for rice is {rice_1c['mean_change_t_ha'].iloc[0]:+.3f} t/ha per +1 C, i.e. it understates the regression sensitivity (a conservative what-if)."
                    if len(rice_1c)
                    else ""
                )
                lines += [
                    f"For comparison, the Phase 3 district-FE regression gives {fe['coef']:+.3f} t/ha per +1 C for rice (p = {fe['p']:.3g}).{model_text}",
                    "",
                ]

    # SHAP
    shap_info = results.get("shap", {})
    lines += ["## Explanation (SHAP)", ""]
    if "top_features" in shap_info:
        lines += [
            "![SHAP summary](figures/yield_shap_summary_lead60.png)",
            "",
            "![SHAP dependence](figures/yield_shap_dependence_lead60.png)",
            "",
            "Top features by mean |SHAP| (relative-yield units):",
            "",
            _table(pd.DataFrame(shap_info["top_features"])),
            "",
        ]
    else:
        lines += [f"SHAP could not be computed: {shap_info.get('error', 'unknown')}", ""]

    # params
    lines += [
        "## Tuned hyperparameters",
        "",
        "```json",
        json.dumps(
            {
                "xgboost": results["xgboost_params"],
                "random_forest": results["rf_params"],
                "ridge_alpha": results["ridge_alpha"],
                "lstm_epochs": results["lstm_epochs"],
            },
            indent=1,
            default=str,
        ),
        "```",
        "",
    ]

    lines += [
        "## Limitations",
        "",
        "- **District aggregates, not fields.** Predictions are district-level averages; they do not describe individual farms (ecological fallacy). District units are reconstructed (D-006).",
        "- **Three test years.** The test period 2015-2017 is short; the bootstrap intervals above reflect resampling districts, not additional years, so they understate year-to-year uncertainty. Models that look better on the validation years than on the test years should be read with that in mind.",
        "- **Weather explains little beyond history.** Phase 3 found that weather adds only a small share of variance beyond district and year effects (see docs/FINDINGS.md), so models gain mostly from yield history. Skill claims must be read against the district-trend baseline.",
        "- **Satellite features** are synthetic until Earth Engine is authenticated; serving models therefore use weather only (D-013).",
        "- **Interventions are not modelled.** Irrigation, varieties and prices drive yields but are not features; what-if answers describe weather sensitivity under the fitted statistical relationship, not agronomic causation.",
        "- **Season attribution and sugarcane units** (D-007) limit comparisons across crops.",
        "- Shipped models are fitted on train years only (the ones evaluated here), not refit on validation or test years.",
        "",
    ]
    path = ROOT / "reports" / "yield_model_comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("wrote %s", path)
    get_path("reports")
