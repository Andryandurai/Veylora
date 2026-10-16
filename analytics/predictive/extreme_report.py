"""Write reports/extreme_weather_report.md and the reliability / precision-recall figures."""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from analytics import common
from analytics.predictive import extreme_weather as X
from config import ROOT

LOG = logging.getLogger(__name__)

MIN_ONSETS_FOR_CONCLUSIONS = 20


def _table(frame: pd.DataFrame, digits: int = 3) -> str:
    return common.markdown_table(frame.reset_index(drop=True), digits)


def draw_figures(figs: dict[str, np.ndarray]) -> None:
    """Reliability diagrams and precision-recall curves for the headline forecasts."""
    common.apply_mpl_theme()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, (title, y, p) in zip(
        axes,
        [
            (
                "LSTM: heatwave onset within 14 days (pooled out-of-sample)",
                figs["heat_y"],
                figs["heat_p"],
            ),
            ("Next-month drought (SPI-3 <= -1), 2020-2024", figs["y"], figs["p"]),
        ],
        strict=True,
    ):
        table = X.reliability_table(y, p, bins=10 if len(y) > 5000 else 8)
        top = max(float(table[["mean_pred", "observed"]].to_numpy().max()) * 1.15, 0.01)
        ax.plot([0, top], [0, top], color=common.AXIS, linewidth=1, linestyle=(0, (4, 3)))
        ax.plot(
            table["mean_pred"],
            table["observed"],
            marker="o",
            color=common.SERIES[0],
            markersize=6,
            markeredgecolor=common.SURFACE,
        )
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed frequency")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
    fig.suptitle(
        "Reliability (dashed = perfectly calibrated)", x=0.01, ha="left", fontweight="bold"
    )
    common.save_mpl(fig, "extreme_reliability")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    sets = [
        (
            "Heatwave onset within 14 days (pooled)",
            figs["heat_y"],
            {
                "LSTM": figs["heat_p"],
                "climatology": figs["heat_clim"],
                "persistence": figs["heat_pers"] + 1e-6 * figs["heat_clim"],
            },
        ),
        (
            "Next-month drought, 2020-2024",
            figs["y"],
            {"XGBoost": figs["p"], "climatology": figs["clim"]},
        ),
    ]
    colors = [common.SERIES[0], common.SERIES[1], common.INK_MUTED]
    for ax, (title, y, forecasts) in zip(axes, sets, strict=True):
        for color, (name, p) in zip(colors, forecasts.items(), strict=False):
            precision, recall, _ = precision_recall_curve(y, p)
            ax.plot(recall, precision, color=color, linewidth=2, label=name)
        ax.axhline(float(np.mean(y)), color=common.AXIS, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper right")
    fig.suptitle(
        "Precision-recall curves (dashed = base rate)", x=0.01, ha="left", fontweight="bold"
    )
    common.save_mpl(fig, "extreme_pr_curves")


def _verdict(entry: dict[str, Any], reference: str) -> str:
    delta, lo, hi = entry[f"delta_vs_{reference}"], entry["delta_lo"], entry["delta_hi"]
    if lo > 0:
        return f"skill over {reference}: PR-AUC {delta:+.3f} (CI {lo:+.3f} to {hi:+.3f})"
    if hi < 0:
        return f"WORSE than {reference}: PR-AUC {delta:+.3f} (CI {lo:+.3f} to {hi:+.3f})"
    return f"no demonstrated skill over {reference}: PR-AUC {delta:+.3f} (CI {lo:+.3f} to {hi:+.3f} includes 0)"


def write_report(results: dict[str, Any], figs: dict[str, np.ndarray]) -> None:
    """Render the markdown report from the results dict (every number comes from ``results``)."""
    meta, heat, drought = results["meta"], results["heatwave"], results["drought"]
    draw_figures(figs)
    lines: list[str] = []
    if meta["weather_synthetic"]:
        lines.append("> **SYNTHETIC DATA: parts of the weather record are synthetic.**")
    else:
        lines.append(
            "> Weather: real (IMD gridded rain, Tmax, Tmin; NASA POWER humidity, wind, solar)."
        )
    if meta["satellite_features_synthetic"]:
        lines.append(
            "> **Soil moisture and NDVI anomaly features are SYNTHETIC** (derived from rainfall; Google Earth Engine not authenticated). The shipped drought model therefore uses weather-derived features only."
        )
    folds = pd.DataFrame(heat["folds"])
    plan = folds[folds["plan_split"]].iloc[0]
    lines += [
        "",
        "# Extreme-weather early warning: heatwave and drought",
        "",
        f"Generated {meta['generated_at']} (seed {meta['seed']}, {meta['runtime_s']:.0f} s{', QUICK RUN: not representative' if meta['quick'] else ''}).",
        "",
        "## Definitions and setup",
        "",
        "- **Heatwave (IMD plains criterion)**: Tmax >= 40 C and >= 4.5 C above the district's daily normal for at least 2 consecutive days (severe: >= 6.5 C). "
        f"Normals = district mean Tmax per calendar day over 1991-{meta['reference_end_year']}, smoothed with a circular 30-day mean. The plan asks for 1991-2020; the training years are used so that no label depends on the validation/test years (D-018).",
        "- **Heatwave target**: an *onset* (first day of a heatwave spell) within the next 3, 7 or 14 days, forecast from the last 30 days of [Tmax, Tmin, rain, RH, wind, solar, sin/cos(doy)] with an LSTM; as-of days March-June only (all observed onsets are in May-June).",
        f"- **Drought target**: SPI-3 of the *next* month <= -1 (SPI from gamma fits per district and calendar month on 1991-{meta['reference_end_year']}, zeros handled with the mixed distribution), predicted by XGBoost from SPI-1/SPI-3 (and lags), rainfall deficit, Tmax anomaly and month.",
        "- **Calibration**: isotonic regression on the validation years (or a 2-parameter logistic fit when there are fewer than 50 validation positives). Thresholds for precision/recall/F1 maximise F1 on the validation years.",
        f"- **Plan split**: train <= {meta['split']['train_end']}, validation {meta['split']['valid'][0]}-{meta['split']['valid'][1]}, test {meta['split']['test'][0]}-{meta['split']['test'][1]}.",
        "",
        "### Heatwave onsets per fold (why a single test window is not enough)",
        "",
        _table(
            folds[
                [
                    "name",
                    "plan_split",
                    "train",
                    "valid",
                    "test",
                    "train_onsets",
                    "valid_onsets",
                    "test_onsets",
                ]
            ].assign(
                train=lambda d: d["train"].map(lambda x: f"{x[0]}-{x[1]}"),
                valid=lambda d: d["valid"].map(lambda x: f"{x[0]}-{x[1]}"),
                test=lambda d: d["test"].map(lambda x: f"{x[0]}-{x[1]}"),
            )
        ),
        "",
    ]
    if plan["test_onsets"] < MIN_ONSETS_FOR_CONCLUSIONS:
        lines += [
            f"**The plan's test window ({plan['test'][0]}-{plan['test'][1]}) contains only {int(plan['test_onsets'])} heatwave onsets, so its metrics are anecdotal and no conclusion should be drawn from them.** "
            f"Skill is therefore judged on the pooled out-of-sample predictions of all rolling-origin folds ({heat['pooled_onsets_tested']} onsets), each fold trained only on earlier years.",
            "",
        ]

    # heatwave results
    fm = pd.DataFrame(heat["fold_metrics"])
    fm = fm[fm["forecast"] != "lstm_calibrator"]
    cols = [
        "horizon_days",
        "forecast",
        "n",
        "n_positive",
        "base_rate",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "brier",
    ]
    lines += [
        "## Heatwave results",
        "",
        "### Plan split (final fold)",
        "",
        _table(fm[fm["plan_split"]][cols], 4),
        "",
    ]
    pooled = pd.DataFrame(heat["pooled_metrics"])
    lines += [
        "### Pooled out-of-sample, all rolling-origin folds",
        "",
        _table(
            pooled[
                [
                    "horizon_days",
                    "forecast",
                    "n",
                    "n_positive",
                    "base_rate",
                    "precision",
                    "recall",
                    "f1",
                    "pr_auc",
                    "roc_auc",
                    "brier",
                    "brier_skill_vs_climatology",
                ]
            ],
            4,
        ),
        "",
    ]
    lines += [
        "### Per-fold PR-AUC (LSTM / climatology / persistence)",
        "",
        _table(
            fm.pivot_table(
                index=["fold", "horizon_days"], columns="forecast", values="pr_auc"
            ).reset_index(),
            4,
        ),
        "",
    ]
    lines += [
        "### Is there skill, and at which lead time?",
        "",
        "Block bootstrap (district-years) of the difference in PR-AUC against climatology, pooled out-of-sample:",
        "",
    ]
    achieved = []
    for h in ("3", "7", "14"):
        entry = heat["bootstrap_ap"][h]
        lstm = entry["lstm"]
        verdict_clim = _verdict(lstm, "climatology")
        pers = entry["persistence"]
        lines.append(
            f"- **{h}-day horizon**: LSTM PR-AUC {lstm['ap']:.3f} (95% CI {lstm['ap_lo']:.3f}-{lstm['ap_hi']:.3f}), climatology {entry['climatology']['ap']:.3f}, persistence {pers['ap']:.3f}; LSTM: {verdict_clim}."
        )
        if lstm["delta_lo"] > 0:
            achieved.append(int(h))
    if achieved:
        lines += [
            "",
            f"Lead time supported by the evidence: skill over climatology is demonstrated for onset within **{max(achieved)} days** (largest horizon whose confidence interval excludes zero).",
        ]
    else:
        lines += [
            "",
            "**No horizon shows demonstrated skill over climatology**, so no usable heatwave lead time can be claimed; the 14-day output must be read as a climatology-like probability.",
        ]
    lines += [
        "",
        f"**Served heatwave probability (`risk_forecast`): {heat['served']['source']}.** {heat['served']['reason']}. The LSTM is still saved (`models/heatwave_lstm.pt`) for comparison and re-evaluation as more onsets accumulate.",
        "",
    ]

    # drought
    lines += ["## Drought results (plan split)", ""]
    base = drought["base_rate"]
    lines += [
        f"Base rate of next-month drought: train {base['train']:.3f}, validation {base['valid']:.3f}, **test {base['test']:.3f}**. "
        + (
            "The test years are much wetter than the 1991-2014 reference, so probabilities calibrated on the validation years over-predict drought on the test years (see mean predicted vs observed below). "
            if drought["feature_sets"]["weather"]["mean_predicted_test"] > 1.3 * base["test"]
            else ""
        )
        + f"Rows (train/valid/test): {drought['n_rows']['train']}/{drought['n_rows']['valid']}/{drought['n_rows']['test']}.",
        "",
    ]
    rows = []
    for name, entry in drought["feature_sets"].items():
        rows.append(
            {
                "forecast": f"XGBoost ({name})",
                **entry["test"],
                "brier_skill_vs_climatology": entry["brier_skill_vs_climatology_test"],
                "mean_predicted": entry["mean_predicted_test"],
            }
        )
    for name, entry in drought["baselines_test"].items():
        rows.append({"forecast": name, **entry})
    frame = pd.DataFrame(rows)
    lines += [
        _table(
            frame[
                [
                    "forecast",
                    "n",
                    "n_positive",
                    "base_rate",
                    "precision",
                    "recall",
                    "f1",
                    "pr_auc",
                    "roc_auc",
                    "brier",
                    "brier_skill_vs_climatology",
                    "mean_predicted",
                ]
            ],
            4,
        ),
        "",
    ]
    b = drought["bootstrap_ap"]
    lines += [
        "Block bootstrap (district-years) of PR-AUC on the test years:",
        "",
        f"- XGBoost (weather): {b['xgboost_weather']['ap']:.3f}; {_verdict(b['xgboost_weather'], 'climatology')}.",
        f"- persistence: {b['persistence']['ap']:.3f}; {_verdict(b['persistence'], 'climatology')}.",
        "",
    ]
    served = drought["served_feature_set"]
    lines += [
        f"Shipped drought model: **XGBoost, {served} features** ({'forced to weather-only because the satellite features are synthetic' if drought['satellite_features_synthetic'] else 'best validation PR-AUC'}).",
        "",
    ]
    lines += [
        "![Reliability](figures/extreme_reliability.png)",
        "",
        "![Precision-recall](figures/extreme_pr_curves.png)",
        "",
    ]

    # risk levels
    thresholds = meta["risk_levels"]
    heat_p, drought_p = figs["heat_p"], figs["p"]
    dist = []
    for hazard, probs in (
        ("heatwave (14-day, in season, pooled folds)", heat_p),
        ("drought (next month, test years)", drought_p),
    ):
        key = "heatwave" if hazard.startswith("heatwave") else "drought"
        levels = pd.Series([X.level_from_probability(key, float(p), thresholds) for p in probs])
        shares = levels.value_counts(normalize=True).reindex(X.LEVELS).fillna(0.0)
        dist.append({"hazard": hazard, **{f"share_{k}": float(v) for k, v in shares.items()}})
    lines += [
        "## Risk levels",
        "",
        "`risk_forecast(district_id, as_of_date)` returns `heatwave_p` (14-day onset probability, 0 outside March-June), `drought_p` (next-month drought probability), the higher of the two hazard levels, and the lead time of the hazard that sets it (14 days for heatwave, 30 days for drought). Thresholds (settings.yaml `risk_levels`) apply to each hazard's calibrated probability and differ because base rates differ by an order of magnitude:",
        "",
        _table(pd.DataFrame([{"hazard": h, **v} for h, v in thresholds.items()]), 2),
        "",
        "Share of forecasts falling in each level:",
        "",
        _table(pd.DataFrame(dist), 3),
        "",
        "## Limitations",
        "",
        "- **Rare events.** The IMD heatwave criterion applied to 1-degree district-mean data is met on ~0.06 percent of days (district means dilute station extremes); coastal criteria and station data are not used. Results describe district-mean heatwaves, not station heatwaves.",
        "- **Few onsets.** Even pooled over rolling folds the number of onsets is small, confidence intervals are wide, and per-district claims are not possible.",
        "- **Non-stationarity.** The reference period (1991-2014) is cooler and drier than 2020-2024; base rates change between splits, which hurts calibration. Operational use would need periodic recalibration.",
        "- **Drought predictability is partly structural.** Next month's SPI-3 window shares two of its three months with the current SPI-3, so part of the skill is arithmetic persistence of observed rainfall; it is a one-month outlook, not a seasonal forecast, and the skill measured against persistence (above) is the more informative number.",
        "- **Calibration on small samples.** With fewer than 50 validation positives a logistic calibrator replaces isotonic regression; probabilities are then smoother but less flexible.",
        "- **Lead times** above are what the evaluation supports, not what the plan hoped for; the 14-day output is only as good as the horizon verdicts above.",
        "- **Synthetic inputs** are flagged at the top; drought features from satellite are not used in the shipped model while they are synthetic.",
        "",
    ]
    path = ROOT / "reports" / "extreme_weather_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("wrote %s", path)
