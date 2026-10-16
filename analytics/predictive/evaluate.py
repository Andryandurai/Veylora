"""Evaluation of the yield models: metrics, paired bootstrap, leave-one-state-out CV, SHAP, report.

Metrics are computed in t/ha (RMSE, MAE, R2, MAPE) per crop and pooled. Because crops differ in scale
(sugarcane ~8 t/ha, cotton ~0.35), pooled errors are also given in *relative* units
(``rmse_rel`` = RMSE of yield / prior mean yield) which is comparable across crops. The test years are
only used here, after all tuning is finished.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analytics import common
from analytics.predictive.yield_models import Data, XGBoostModel, YieldModel

LOG = logging.getLogger(__name__)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, R2 and MAPE (percent) of predictions against truth."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - (err**2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
        "mape": float(100.0 * np.mean(np.abs(err) / np.abs(y_true))),
    }


def metric_rows(
    meta: pd.DataFrame, y_pred_rel: np.ndarray, base: dict[str, Any], split: str
) -> list[dict[str, Any]]:
    """Metrics for the pooled set and for each crop; ``base`` carries feature_set / lead / model."""
    scale = meta["scale"].to_numpy()
    truth = meta["yield"].to_numpy()
    pred = y_pred_rel * scale
    rows = []
    rel = regression_metrics(truth / scale, y_pred_rel)
    pooled = regression_metrics(truth, pred)
    pooled["rmse_rel"] = rel["rmse"]
    rows.append({**base, "split": split, "crop": "all", **pooled})
    for crop in sorted(meta["crop"].unique()):
        sel = (meta["crop"] == crop).to_numpy()
        entry = regression_metrics(truth[sel], pred[sel])
        entry["rmse_rel"] = regression_metrics(truth[sel] / scale[sel], y_pred_rel[sel])["rmse"]
        rows.append({**base, "split": split, "crop": crop, **entry})
    return rows


def paired_cluster_bootstrap(
    meta: pd.DataFrame,
    pred_model_rel: np.ndarray,
    pred_base_rel: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Ratio of relative RMSE (model / baseline) with a 95 % cluster-bootstrap CI over districts.

    A ratio below 1 means the model beats the baseline. ``p_not_better`` is the share of bootstrap
    resamples in which the model is *not* better (ratio >= 1).
    """
    truth = (meta["yield"] / meta["scale"]).to_numpy()
    sq_model = (pred_model_rel - truth) ** 2
    sq_base = (pred_base_rel - truth) ** 2
    districts = meta["district_id"].to_numpy()
    labels, inverse = np.unique(districts, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(labels))]
    rng = np.random.default_rng(seed)
    ratios = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        ratios[b] = np.sqrt(sq_model[idx].mean() / sq_base[idx].mean())
    return {
        "ratio": float(np.sqrt(sq_model.mean() / sq_base.mean())),
        "lo": float(np.percentile(ratios, 2.5)),
        "hi": float(np.percentile(ratios, 97.5)),
        "p_not_better": float(np.mean(ratios >= 1.0)),
        "n": int(len(truth)),
    }


def leave_one_state_out(
    data_all: Data,
    train_end_mask: np.ndarray,
    test_mask: np.ndarray,
    params: dict[str, Any],
    baseline_rel: np.ndarray,
) -> list[dict[str, Any]]:
    """Train on all other states' rows up to the end of validation, test on the held-out state's test years.

    ``train_end_mask`` selects rows up to the end of the validation period. The held-out state's own
    history only enters through its *features* (previous yields), never as training rows.
    """
    rows = []
    meta = data_all.meta
    for state in sorted(meta["state"].unique()):
        in_state = (meta["state"] == state).to_numpy()
        train = data_all.subset(train_end_mask & ~in_state)
        test = data_all.subset(test_mask & in_state)
        if len(test.y) == 0:
            continue
        model = XGBoostModel(params=params).fit(train, train)
        pred = model.predict_rel(test)
        base = baseline_rel[test_mask & in_state]
        entry = regression_metrics(
            test.meta["yield"].to_numpy(), pred * test.meta["scale"].to_numpy()
        )
        naive = regression_metrics(
            test.meta["yield"].to_numpy(), base * test.meta["scale"].to_numpy()
        )
        rows.append(
            {
                "held_out_state": state,
                "n_test": entry["n"],
                "xgb_rmse_rel": float(np.sqrt(np.mean((pred - test.y.to_numpy()) ** 2))),
                "naive_rmse_rel": float(np.sqrt(np.mean((base - test.y.to_numpy()) ** 2))),
                "xgb_mae": entry["mae"],
                "naive_mae": naive["mae"],
            }
        )
    return rows


def interval_metrics(
    meta: pd.DataFrame, lo_rel: np.ndarray, hi_rel: np.ndarray
) -> dict[str, float]:
    """Empirical coverage and mean width of prediction intervals (relative and t/ha)."""
    truth_rel = (meta["yield"] / meta["scale"]).to_numpy()
    inside = (truth_rel >= lo_rel) & (truth_rel <= hi_rel)
    out = {
        "coverage": float(inside.mean()),
        "mean_width_rel": float(np.mean(hi_rel - lo_rel)),
        "n": int(len(truth_rel)),
    }
    for crop in sorted(meta["crop"].unique()):
        sel = (meta["crop"] == crop).to_numpy()
        out[f"coverage_{crop}"] = float(inside[sel].mean())
    return out


def shap_figures(model: YieldModel, test: Data, tag: str) -> dict[str, Any]:
    """SHAP beeswarm and dependence plots for an XGBoost model on the test rows; returns top features."""
    import shap

    common.apply_mpl_theme()
    explainer = shap.TreeExplainer(model.model)
    values = explainer(test.X)
    importance = pd.Series(np.abs(values.values).mean(axis=0), index=test.X.columns).sort_values(
        ascending=False
    )
    plt.figure(figsize=(8, 6))
    shap.plots.beeswarm(values, max_display=15, show=False, color_bar=True)
    plt.title("SHAP summary (relative yield), XGBoost, test years")
    common.save_mpl(plt.gcf(), f"yield_shap_summary_{tag}")
    top = importance.head(3).index.tolist()
    fig, axes = plt.subplots(1, len(top), figsize=(4.2 * len(top), 3.8))
    for ax, feature in zip(np.atleast_1d(axes), top, strict=True):
        shap.plots.scatter(values[:, feature], ax=ax, show=False, color=common.SERIES[0])
        ax.set_title(feature)
    fig.suptitle(
        "SHAP dependence of the three most important features", x=0.01, ha="left", fontweight="bold"
    )
    common.save_mpl(fig, f"yield_shap_dependence_{tag}")
    return {
        "top_features": [
            {"feature": f, "mean_abs_shap": float(importance[f])} for f in importance.head(10).index
        ]
    }


def markdown_metrics(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    """Markdown table of selected metric columns."""
    return common.markdown_table(frame[columns].reset_index(drop=True), digits)
