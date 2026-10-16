"""Train and evaluate every yield model; save artifacts and the evaluation JSON.

Run: ``python -m analytics.predictive.train_yield`` (``make train-yield``). ``--quick`` uses 8 Optuna
trials for smoke tests. Outputs:

* ``models/yield_xgb.joblib``, ``models/yield_rf.joblib`` (weather-only serving models per lead, with conformal
  intervals for XGBoost), ``models/yield_lstm.pt`` (comparison model);
* ``reports/yield_metrics.csv`` and ``reports/yield_models.json`` (all numbers the report and dashboards read);
* ``reports/yield_model_comparison.md`` and SHAP figures.

Serving models use weather features only: NDVI / soil moisture cannot be recomputed under what-if
overrides and are synthetic until Google Earth Engine is authenticated (docs/DECISIONS.md D-013).
Models fitted on the train years only are the ones evaluated *and* shipped.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from analytics.predictive import features as F
from analytics.predictive.evaluate import (
    interval_metrics,
    leave_one_state_out,
    metric_rows,
    paired_cluster_bootstrap,
    shap_figures,
)
from analytics.predictive.yield_models import (
    ConformalXGB,
    Data,
    DistrictTrend,
    LSTMModel,
    NaiveLastYear,
    PriorMean3,
    RandomForestModel,
    RidgeModel,
    XGBoostModel,
    YieldModel,
)
from config import get_path, get_seed, load_settings, set_global_seed

LOG = logging.getLogger(__name__)

BASELINES = ("naive_last_year", "district_trend", "prior_mean_3y")
FEATURE_SETS = {"weather": False, "weather+satellite": True}
LEARNED = ("ridge", "random_forest", "xgboost", "lstm")
SERVING_FEATURE_SET = "weather"
ARTIFACT_VERSION = 1


def git_commit() -> str:
    """Short git hash of HEAD, or 'unknown'."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=get_path("reports").parent,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def make_data(
    prepared: pd.DataFrame,
    weather: pd.DataFrame,
    satellite: pd.DataFrame,
    lead: int,
    include_satellite: bool,
    levels: dict,
) -> Data:
    """Features, target, metadata and weekly sequences for one lead."""
    X, y, meta = F.build_features(prepared, lead, include_satellite, levels)
    leaked = F.check_no_future_columns(X.columns, lead)
    if leaked:
        raise AssertionError(f"features that use information after the forecast date: {leaked}")
    seq, mask = F.build_sequences(meta, weather, satellite, lead)
    return Data(X, y, meta, seq, mask)


def split_data(data: Data) -> dict[str, Data]:
    """Train / valid / test subsets by the time split in settings.yaml."""
    masks = F.split_masks(data.meta)
    return {name: data.subset(mask) for name, mask in masks.items()}


def fit_models(
    parts: dict[str, Data], lead: int, include_satellite: bool, n_trials: int
) -> dict[str, YieldModel]:
    """Fit baselines and learned models for one dataset."""
    train, valid = parts["train"], parts["valid"]
    models: dict[str, YieldModel] = {
        "naive_last_year": NaiveLastYear(),
        "district_trend": DistrictTrend(),
        "prior_mean_3y": PriorMean3(),
    }
    learned = (
        RidgeModel(),
        RandomForestModel(),
        XGBoostModel(n_trials=n_trials),
        XGBoostModel(n_trials=n_trials, monotone_temperature=False, name="xgboost_unconstrained"),
        LSTMModel(lead, use_satellite=include_satellite),
    )
    for model in learned:
        started = time.time()
        model.fit(train, valid)
        LOG.info("lead %d %s fitted in %.0fs", lead, model.name, time.time() - started)
        models[model.name] = model
    for name in BASELINES:
        models[name].fit(train, valid)
    return models


def run(n_trials: int = 40) -> dict[str, Any]:
    """Full training and evaluation; returns the results dict that is also written to JSON."""
    set_global_seed()
    started = time.time()
    settings = load_settings()
    frame, source = F.load_training_frame()
    prepared = F.prepare_frame(frame)
    prepared = prepared[
        (prepared["year"] >= int(settings["years"]["start"])) | prepared["yield_label"].notna()
    ]
    clusters = F.load_clusters()
    levels = F.category_levels(clusters)
    weather = F.load_weather_daily()
    satellite = pd.read_parquet(get_path("data_processed") / "satellite_16d.parquet")
    synthetic_satellite = bool(frame["satellite_is_synthetic"].any())
    synthetic_label = bool(frame["label_is_synthetic"].any())

    metrics: list[dict[str, Any]] = []
    tests: dict[tuple[str, int], dict[str, Any]] = {}
    served: dict[int, dict[str, Any]] = {}
    for feature_set, include_satellite in FEATURE_SETS.items():
        for lead in F.LEADS:
            data = make_data(prepared, weather, satellite, lead, include_satellite, levels)
            parts = split_data(data)
            LOG.info(
                "lead %d / %s: train %d valid %d test %d rows, %d features",
                lead,
                feature_set,
                *(len(parts[s].y) for s in ("train", "valid", "test")),
                data.X.shape[1],
            )
            models = fit_models(parts, lead, include_satellite, n_trials)
            predictions: dict[str, dict[str, np.ndarray]] = {}
            for name, model in models.items():
                predictions[name] = {
                    split: model.predict_rel(parts[split]) for split in ("valid", "test")
                }
                for split in ("valid", "test"):
                    metrics += metric_rows(
                        parts[split].meta,
                        predictions[name][split],
                        {"feature_set": feature_set, "lead_days": lead, "model": name},
                        split,
                    )
            tests[(feature_set, lead)] = {
                "parts": parts,
                "models": models,
                "predictions": predictions,
                "data": data,
            }
            if feature_set == SERVING_FEATURE_SET:
                served[lead] = _serve_artifacts(models, parts)

    metrics_df = pd.DataFrame(metrics)
    results = _summarise(metrics_df, tests, served, levels, synthetic_satellite)
    results["meta"] = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "seed": get_seed(),
        "data_source": source,
        "n_optuna_trials": n_trials,
        "satellite_features_synthetic": synthetic_satellite,
        "labels_synthetic": synthetic_label,
        "split": settings["splits"]["yield"],
        "levels": levels,
        "runtime_s": round(time.time() - started, 1),
    }
    _save_artifacts(served, tests, levels, results["meta"])
    from analytics.predictive import predict

    predict.reset_cache()
    results["scenario_response"] = predict.scenario_response().to_dict("records")
    reports = get_path("reports")
    metrics_df.to_csv(reports / "yield_metrics.csv", index=False)
    (reports / "yield_models.json").write_text(
        json.dumps(results, indent=2, default=_default), encoding="utf-8"
    )
    from analytics.predictive.yield_report import write_report

    write_report(results, metrics_df)
    return results


def _serve_artifacts(models: dict[str, YieldModel], parts: dict[str, Data]) -> dict[str, Any]:
    """Conformal intervals for the XGBoost model and the objects needed for serving."""
    xgb_model: XGBoostModel = models["xgboost"]  # type: ignore[assignment]
    conformal = ConformalXGB(params=xgb_model.params, constraints=xgb_model.constraints).fit(
        parts["train"], parts["valid"]
    )
    return {
        "xgboost": xgb_model,
        "conformal": conformal,
        "random_forest": models["random_forest"],
        "columns": list(parts["train"].X.columns),
    }


def _summarise(
    metrics_df: pd.DataFrame, tests: dict, served: dict, levels: dict, synthetic_satellite: bool
) -> dict[str, Any]:
    """Bootstrap tests, LOSO, intervals, SHAP and headline selections."""
    results: dict[str, Any] = {"metrics_file": "reports/yield_metrics.csv"}
    pooled = metrics_df[metrics_df["crop"] == "all"]
    best_by_lead: dict[str, Any] = {}
    bootstrap: list[dict[str, Any]] = []
    for (feature_set, lead), entry in tests.items():
        parts, preds = entry["parts"], entry["predictions"]
        valid_scores = {
            m: float(
                pooled[
                    (pooled["feature_set"] == feature_set)
                    & (pooled["lead_days"] == lead)
                    & (pooled["model"] == m)
                    & (pooled["split"] == "valid")
                ]["rmse_rel"].iloc[0]
            )
            for m in LEARNED
        }
        best = min(valid_scores, key=valid_scores.get)
        best_by_lead[f"{feature_set}|{lead}"] = {
            "model": best,
            "valid_rmse_rel": valid_scores[best],
        }
        test_meta = parts["test"].meta
        for challenger in (best, "xgboost"):
            for baseline in BASELINES:
                bootstrap.append(
                    {
                        "feature_set": feature_set,
                        "lead_days": lead,
                        "model": challenger,
                        "baseline": baseline,
                        "selected_on_validation": challenger == best,
                        **paired_cluster_bootstrap(
                            test_meta, preds[challenger]["test"], preds[baseline]["test"]
                        ),
                    }
                )
    results["best_by_validation"] = best_by_lead
    results["bootstrap_vs_baseline"] = [
        dict(t) for t in {tuple(sorted(b.items(), key=lambda kv: kv[0])) for b in bootstrap}
    ]
    results["bootstrap_vs_baseline"].sort(
        key=lambda b: (b["feature_set"], b["lead_days"], b["model"], b["baseline"])
    )

    # Intervals (serving feature set only).
    intervals = {}
    for lead, art in served.items():
        parts = tests[(SERVING_FEATURE_SET, lead)]["parts"]
        point = art["xgboost"].predict_rel(parts["test"])
        lo, hi = art["conformal"].interval_rel(parts["test"], point)
        intervals[str(lead)] = {
            **interval_metrics(parts["test"].meta, lo, hi),
            "conformal_qhat_rel": art["conformal"].qhat,
            "coverage_valid_calibration": art["conformal"].coverage_valid,
            "target_coverage": 1 - art["conformal"].alpha,
        }
    results["intervals_test"] = intervals

    # Leave-one-state-out CV (XGBoost, serving feature set, lead 60).
    lead_loso = 60
    entry = tests[(SERVING_FEATURE_SET, lead_loso)]
    data_all: Data = entry["data"]
    masks = F.split_masks(data_all.meta)
    valid_end = int(load_settings()["splits"]["yield"]["valid"][1])
    first = int(load_settings()["years"]["start"])
    year = data_all.meta["year"].to_numpy()
    upto_valid = (year >= first) & (year <= valid_end)
    baseline_rel = NaiveLastYear().predict_rel(data_all)
    results["leave_one_state_out"] = {
        "lead_days": lead_loso,
        "feature_set": SERVING_FEATURE_SET,
        "rows": leave_one_state_out(
            data_all, upto_valid, masks["test"], served[lead_loso]["xgboost"].params, baseline_rel
        ),
    }

    # SHAP + importance for the serving model at lead 60, and the best params.
    try:
        results["shap"] = shap_figures(
            served[lead_loso]["xgboost"], entry["parts"]["test"], f"lead{lead_loso}"
        )
    except Exception as exc:  # SHAP/xgboost version mismatch must not lose the whole run
        LOG.warning("SHAP failed: %s", exc)
        results["shap"] = {"error": str(exc)}
    results["xgboost_params"] = {str(lead): art["xgboost"].params for lead, art in served.items()}
    results["rf_params"] = {str(lead): art["random_forest"].params for lead, art in served.items()}
    results["ridge_alpha"] = {
        f"{fs}|{lead}": e["models"]["ridge"].alpha for (fs, lead), e in tests.items()
    }
    results["lstm_epochs"] = {
        f"{fs}|{lead}": len(e["models"]["lstm"].history) for (fs, lead), e in tests.items()
    }
    results["n_features"] = {
        f"{fs}|{lead}": int(e["data"].X.shape[1]) for (fs, lead), e in tests.items()
    }
    results["n_rows"] = {
        f"{fs}|{lead}": {s: int(len(p.y)) for s, p in e["parts"].items()}
        for (fs, lead), e in tests.items()
    }
    results["xgb_valid_rmse_rel_optuna"] = {
        str(lead): art["xgboost"].best_value for lead, art in served.items()
    }
    results["synthetic_satellite"] = synthetic_satellite
    return results


def _save_artifacts(served: dict, tests: dict, levels: dict, meta: dict) -> None:
    models = get_path("models")
    header = {
        "version": ARTIFACT_VERSION,
        "feature_set": SERVING_FEATURE_SET,
        "levels": levels,
        "meta": meta,
    }
    joblib.dump(
        {
            **header,
            "leads": {
                lead: {
                    "model": a["xgboost"].model,
                    "params": a["xgboost"].params,
                    "columns": a["columns"],
                    "conformal_model": a["conformal"].model,
                    "conformal_qhat": a["conformal"].qhat,
                    "alpha": a["conformal"].alpha,
                }
                for lead, a in served.items()
            },
        },
        models / "yield_xgb.joblib",
    )
    joblib.dump(
        {
            **header,
            "leads": {
                lead: {
                    "model": a["random_forest"].model,
                    "params": a["random_forest"].params,
                    "columns": a["columns"],
                }
                for lead, a in served.items()
            },
        },
        models / "yield_rf.joblib",
    )
    torch.save(
        {
            **header,
            "leads": {
                f"{fs}|{lead}": e["models"]["lstm"].state() for (fs, lead), e in tests.items()
            },
        },
        models / "yield_lstm.pt",
    )
    LOG.info("saved models to %s", models)


def _default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--quick", action="store_true", help="8 Optuna trials (smoke test)")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(n_trials=8 if args.quick else 40)


if __name__ == "__main__":
    main()
