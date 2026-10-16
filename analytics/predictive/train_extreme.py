"""Train and evaluate the heatwave LSTM and the drought XGBoost; save artifacts and results.

Run: ``python -m analytics.predictive.train_extreme`` (``make train-extreme``); ``--quick`` trains for few
epochs/trials. Outputs ``models/heatwave_lstm.pt``, ``models/drought_xgb.joblib``,
``reports/extreme_weather.json`` and ``reports/extreme_weather_report.md`` (+ figures).

Design notes (docs/DECISIONS.md D-018):
* Event definitions use a fixed reference period (1991-2014 = the training years) for heatwave normals
  and SPI distributions, so no fitted quantity depends on validation/test years.
* Heatwave onsets are so rare that the plan's 2020-2024 test window has 3 of them, so skill is also assessed
  with rolling-origin folds that always train on earlier years.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from analytics.predictive import extreme_data as E
from analytics.predictive import extreme_weather as X
from analytics.predictive import labels as L
from config import get_path, get_seed, load_settings, set_global_seed

LOG = logging.getLogger(__name__)

N_BOOT = 300


def folds() -> list[dict[str, Any]]:
    """Rolling-origin folds from settings plus the plan's fixed split as the final fold."""
    cfg = load_settings()["splits"]["extreme_weather"]
    out = [
        {
            "name": chr(ord("A") + i),
            "train": tuple(f["train"]),
            "valid": tuple(f["valid"]),
            "test": tuple(f["test"]),
            "plan_split": False,
        }
        for i, f in enumerate(cfg.get("rolling_origin", []))
    ]
    out.append(
        {
            "name": chr(ord("A") + len(out)),
            "train": (1991, int(cfg["train_end"])),
            "valid": tuple(cfg["valid"]),
            "test": tuple(cfg["test"]),
            "plan_split": True,
        }
    )
    return out


def persistence_heat(heat: E.HeatData, d_idx: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
    """1 if a qualifying (>= 40 C and >= 4.5 C above normal) day occurred in the last 7 days, else 0."""
    return heat.recent("qualifying", 7, d_idx, t_idx).astype(float)


def _block_ids(heat: E.HeatData, d_idx: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
    return d_idx * 10_000 + heat.dates.year.to_numpy()[t_idx]


def paired_ap_bootstrap(
    y: np.ndarray,
    probs: dict[str, np.ndarray],
    blocks: np.ndarray,
    reference: str,
    n_boot: int = N_BOOT,
    seed: int = 42,
) -> dict[str, Any]:
    """Block bootstrap of average precision for several forecasts; differences are versus ``reference``."""
    from sklearn.metrics import average_precision_score

    labels, inverse = np.unique(blocks, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(labels))]
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in probs}
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        if y[idx].sum() == 0:
            continue
        for name, p in probs.items():
            draws[name].append(average_precision_score(y[idx], p[idx]))
    out: dict[str, Any] = {}
    ref = np.array(draws[reference])
    for name, values in draws.items():
        arr = np.array(values)
        entry = {
            "ap": float(average_precision_score(y, probs[name])),
            "ap_lo": float(np.percentile(arr, 2.5)),
            "ap_hi": float(np.percentile(arr, 97.5)),
        }
        if name != reference:
            diff = arr - ref
            entry.update(
                {
                    "delta_vs_" + reference: float(
                        average_precision_score(y, probs[name])
                        - average_precision_score(y, probs[reference])
                    ),
                    "delta_lo": float(np.percentile(diff, 2.5)),
                    "delta_hi": float(np.percentile(diff, 97.5)),
                }
            )
        out[name] = entry
    out["n_boot_used"] = len(ref)
    return out


def run_heatwave(
    heat: E.HeatData, quick: bool
) -> tuple[dict[str, Any], X.HeatwaveModel, dict[str, np.ndarray]]:
    """Train per fold, evaluate on each fold's test years, pool, and return the plan-split model."""
    fold_rows, pooled = [], {"y": [], "model": [], "clim": [], "pers": [], "block": [], "thr": []}
    plan_model = None
    onset_counts = {}
    for fold in folds():
        started = time.time()
        model = X.HeatwaveModel(max_epochs=4 if quick else 25)
        model.fit(heat, fold["train"], fold["valid"])
        LOG.info(
            "heatwave fold %s trained in %.0fs (%d epochs)",
            fold["name"],
            time.time() - started,
            len(model.history),
        )
        d_te, t_te = heat.sample_index(*fold["test"], stride=1)
        y_te = heat.targets(d_te, t_te)
        prob = model.predict_proba(heat, d_te, t_te)
        clim = X.climatology_heat(heat, fold["train"], d_te, t_te)
        pers = persistence_heat(heat, d_te, t_te)
        sel = (heat.dates.year >= fold["test"][0]) & (heat.dates.year <= fold["test"][1])
        onset_counts[fold["name"]] = {
            "train_onsets": int(
                heat.labels["onset"][
                    :, (heat.dates.year >= fold["train"][0]) & (heat.dates.year <= fold["train"][1])
                ].sum()
            ),
            "valid_onsets": int(
                heat.labels["onset"][
                    :, (heat.dates.year >= fold["valid"][0]) & (heat.dates.year <= fold["valid"][1])
                ].sum()
            ),
            "test_onsets": int(heat.labels["onset"][:, sel].sum()),
        }
        for j, h in enumerate(L.HORIZONS):
            for name, p in (
                ("lstm", prob[:, j]),
                ("climatology", clim[:, j]),
                ("persistence", pers),
            ):
                thr = (
                    model.thresholds[j]
                    if name == "lstm"
                    else (float(np.quantile(clim[:, j], 0.99)) if name == "climatology" else 0.5)
                )
                fold_rows.append(
                    {
                        "fold": fold["name"],
                        "plan_split": fold["plan_split"],
                        "horizon_days": h,
                        "forecast": name,
                        "test_years": f"{fold['test'][0]}-{fold['test'][1]}",
                        **X.binary_metrics(y_te[:, j], p, thr),
                    }
                )
            fold_rows.append(
                {
                    "fold": fold["name"],
                    "plan_split": fold["plan_split"],
                    "horizon_days": h,
                    "forecast": "lstm_calibrator",
                    "test_years": fold["test"],
                    "n": model.calibrators[j].n_positive,
                    "n_positive": model.calibrators[j].n_positive,
                    "base_rate": np.nan,
                    "threshold": model.thresholds[j],
                    "calibration": model.calibrators[j].kind,
                }
            )
        pooled["y"].append(y_te)
        pooled["model"].append(prob)
        pooled["clim"].append(clim)
        pooled["pers"].append(np.repeat(pers[:, None], len(L.HORIZONS), axis=1))
        pooled["block"].append(_block_ids(heat, d_te, t_te))
        pooled["thr"].append(np.tile(model.thresholds, (len(d_te), 1)))
        if fold["plan_split"]:
            plan_model = model
    arrays = {k: np.concatenate(v) for k, v in pooled.items()}
    pooled_rows, bootstrap = [], {}
    for j, h in enumerate(L.HORIZONS):
        y = arrays["y"][:, j]
        for name, key in (("lstm", "model"), ("climatology", "clim"), ("persistence", "pers")):
            p = arrays[key][:, j]
            thr = (
                arrays["thr"][:, j]
                if name == "lstm"
                else (0.5 if name == "persistence" else np.quantile(p, 0.99))
            )
            decisions = p >= thr
            metrics = X.binary_metrics(y, p, 0.5)
            tp, fp, fn = (
                int((decisions & (y == 1)).sum()),
                int((decisions & (y == 0)).sum()),
                int((~decisions & (y == 1)).sum()),
            )
            precision = tp / (tp + fp) if tp + fp else float("nan")
            recall = tp / (tp + fn) if tp + fn else float("nan")
            metrics.update(
                {
                    "precision": precision,
                    "recall": recall,
                    "f1": 2 * precision * recall / (precision + recall) if tp else 0.0,
                }
            )
            metrics["brier_skill_vs_climatology"] = X.brier_skill(y, p, arrays["clim"][:, j])
            pooled_rows.append({"horizon_days": h, "forecast": name, **metrics})
        bootstrap[str(h)] = paired_ap_bootstrap(
            y,
            {
                "lstm": arrays["model"][:, j],
                "climatology": arrays["clim"][:, j],
                "persistence": arrays["pers"][:, j] + 1e-6 * arrays["clim"][:, j],
            },
            arrays["block"],
            reference="climatology",
            n_boot=30 if quick else N_BOOT,
        )
    results = {
        "folds": [
            {
                **f,
                "train": list(f["train"]),
                "valid": list(f["valid"]),
                "test": list(f["test"]),
                **onset_counts[f["name"]],
            }
            for f in folds()
        ],
        "fold_metrics": fold_rows,
        "pooled_metrics": pooled_rows,
        "bootstrap_ap": bootstrap,
        "pooled_onsets_tested": int(sum(c["test_onsets"] for c in onset_counts.values())),
    }
    return results, plan_model, arrays


def split_drought(table: pd.DataFrame, plan: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Train/valid/test rows by as-of month; rows whose label month falls in another split are dropped."""
    frame = table.dropna(subset=["target", "spi3", "spi3_lag2", "spi1_lag1"]).copy()
    year = frame["month"].dt.year
    label_year = (frame["month"] + pd.offsets.MonthBegin(1)).dt.year
    masks = {
        "train": year <= plan["train_end"],
        "valid": year.between(*plan["valid"]),
        "test": year.between(*plan["test"]),
    }
    label_masks = {
        "train": label_year <= plan["train_end"],
        "valid": label_year.between(*plan["valid"]),
        "test": label_year.between(*plan["test"]),
    }
    return {k: frame[masks[k] & label_masks[k]].copy() for k in masks}


def run_drought(
    table: pd.DataFrame, satellite_synthetic: bool, quick: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit both feature sets on the plan split; return results and the serving artifact."""
    plan = load_settings()["splits"]["extreme_weather"]
    parts = split_drought(table, plan)
    train, valid, test = parts["train"], parts["valid"], parts["test"]
    sets = {
        "weather": E.DROUGHT_FEATURES_WEATHER,
        "weather+satellite": (*E.DROUGHT_FEATURES_WEATHER, *E.DROUGHT_FEATURES_SATELLITE),
    }
    results: dict[str, Any] = {
        "n_rows": {k: int(len(v)) for k, v in parts.items()},
        "base_rate": {k: float(v["target"].mean()) for k, v in parts.items()},
        "feature_sets": {},
    }
    fitted: dict[str, X.DroughtModel] = {}
    clim_test = X.climatology_drought(train, test)
    clim_valid = X.climatology_drought(train, valid)
    y_test = test["target"].to_numpy(int)
    blocks = (
        test["district_id"].astype("category").cat.codes.astype("int64") * 10_000
        + test["month"].dt.year
    ).to_numpy()
    for name, features in sets.items():
        model = X.DroughtModel(features=tuple(features), n_trials=4 if quick else 20).fit(
            train, valid
        )
        fitted[name] = model
        p_test = model.predict_proba(test)
        p_valid = model.predict_proba(valid)
        results["feature_sets"][name] = {
            "params": model.params,
            "calibration": model.calibrator.kind,
            "threshold": model.threshold,
            "valid": X.binary_metrics(valid["target"].to_numpy(int), p_valid, model.threshold),
            "test": X.binary_metrics(y_test, p_test, model.threshold),
            "reliability_test": X.reliability_table(y_test, p_test).to_dict("records"),
            "brier_skill_vs_climatology_test": X.brier_skill(y_test, p_test, clim_test),
            "mean_predicted_test": float(p_test.mean()),
            "mean_predicted_valid": float(p_valid.mean()),
        }
    baselines = {
        "climatology": X.binary_metrics(y_test, clim_test, float(np.quantile(clim_valid, 0.8))),
        "persistence": X.binary_metrics(y_test, test["persistence"].to_numpy(), 0.5),
    }
    results["baselines_test"] = baselines
    probs = {
        "xgboost_weather": fitted["weather"].predict_proba(test),
        "climatology": clim_test,
        "persistence": test["persistence"].to_numpy() + 1e-6 * clim_test,
    }
    results["bootstrap_ap"] = paired_ap_bootstrap(
        y_test, probs, blocks, reference="climatology", n_boot=30 if quick else N_BOOT
    )
    scores = {k: results["feature_sets"][k]["valid"]["pr_auc"] for k in fitted}
    served = "weather" if satellite_synthetic else max(scores, key=scores.get)
    results["served_feature_set"] = served
    results["satellite_features_synthetic"] = satellite_synthetic
    artifact = {
        "model_wrapper": fitted[served],
        "features": list(fitted[served].features),
        "feature_set": served,
        "meta": {"reference_end_year": int(plan["train_end"])},
    }
    results["test_predictions"] = {
        "y": y_test.tolist(),
        "p": probs["xgboost_weather"].tolist(),
        "clim": clim_test.tolist(),
    }
    return results, artifact


def run(quick: bool = False) -> dict[str, Any]:
    """Full training and evaluation; writes artifacts, JSON, figures and the report."""
    set_global_seed()
    started = time.time()
    plan = load_settings()["splits"]["extreme_weather"]
    reference_end = int(plan["train_end"])
    processed = get_path("data_processed")
    weather = pd.read_parquet(processed / "weather_daily.parquet")
    satellite = pd.read_parquet(processed / "satellite_16d.parquet")
    satellite_synthetic = bool(satellite["is_synthetic"].any())
    heat = E.build_heat_data(weather, reference_end)
    heat_results, heat_model, heat_arrays = run_heatwave(heat, quick)
    drought_table = E.drought_table(weather, satellite, reference_end)
    drought_results, drought_artifact = run_drought(drought_table, satellite_synthetic, quick)
    meta = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "seed": get_seed(),
        "reference_end_year": reference_end,
        "quick": quick,
        "weather_synthetic": bool(weather["is_synthetic"].any()),
        "satellite_features_synthetic": satellite_synthetic,
        "split": {k: plan[k] for k in ("train_end", "valid", "test")},
        "season_months": list(E.SEASON_MONTHS),
        "risk_levels": load_settings()["risk_levels"],
        "runtime_s": round(time.time() - started, 1),
    }
    models = get_path("models")
    lstm_skilful = heat_results["bootstrap_ap"]["14"]["lstm"]["delta_lo"] > 0
    served_heat = "lstm" if lstm_skilful else "climatology"
    meta["heatwave_served"] = served_heat
    heat_results["served"] = {
        "source": served_heat,
        "reason": "LSTM beats climatology at 14 days (95% CI excludes 0)"
        if lstm_skilful
        else "LSTM shows no demonstrated PR-AUC skill over climatology at 14 days, so the climatological probability is served",
    }
    torch.save(
        heat_model.state(meta, X.climatology_table(heat, (1991, reference_end)), served_heat),
        models / X.HEAT_ARTIFACT,
    )
    joblib.dump(
        {**drought_artifact, "meta": {**drought_artifact["meta"], **meta}},
        models / X.DROUGHT_ARTIFACT,
    )
    X.reset_cache()
    figure_data = {
        "heat_y": heat_arrays["y"][:, 2].astype(int),
        "heat_p": heat_arrays["model"][:, 2],
        "heat_clim": heat_arrays["clim"][:, 2],
        "heat_pers": heat_arrays["pers"][:, 2],
        **{k: np.asarray(v) for k, v in drought_results.pop("test_predictions").items()},
    }
    results = {"meta": meta, "heatwave": heat_results, "drought": drought_results}
    (get_path("reports") / "extreme_weather.json").write_text(
        json.dumps(results, indent=1, default=_default), encoding="utf-8"
    )
    from analytics.predictive.extreme_report import write_report

    write_report(results, figure_data)
    return results


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
    parser.add_argument("--quick", action="store_true", help="few epochs/trials (smoke test)")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(quick=args.quick)


if __name__ == "__main__":
    main()
