"""Heatwave and drought early-warning models, calibration, evaluation helpers and ``risk_forecast``.

* **Heatwave**: PyTorch LSTM over the last 30 days of [Tmax, Tmin, rain, RH, wind, solar, sin(doy), cos(doy)]
  with three outputs, the probability of a heatwave *onset* within the next 3, 7 and 14 days (so the lead time
  that is actually achievable can be measured instead of assumed). Weighted BCE, early stopping, seed 42,
  trained on as-of days from March to June (all observed onsets fall in May-June).
* **Drought**: XGBoost classifier on monthly SPI / anomaly features for "SPI-3 of next month <= -1".
* Both are calibrated on the validation years (isotonic regression, or a 2-parameter logistic fit when the
  validation set has too few positives for isotonic to be stable) and compared with climatology and persistence.

Thresholds for the risk level live in ``config/settings.yaml`` under ``risk_levels``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from torch import nn

from analytics.predictive import extreme_data as E
from analytics.predictive import labels as L
from analytics.predictive import spi
from config import get_path, get_seed, load_settings

LOG = logging.getLogger(__name__)

MIN_POSITIVES_FOR_ISOTONIC = 50
HEAT_HORIZONS = L.HORIZONS  # (3, 7, 14)
LEVELS = ("low", "moderate", "high", "severe")
HEAT_ARTIFACT = "heatwave_lstm.pt"
DROUGHT_ARTIFACT = "drought_xgb.joblib"
DROUGHT_LEAD_DAYS = 30
HEAT_LEAD_DAYS = 14


# ----------------------------------------------------------------------------- calibration and metrics
@dataclass
class Calibrator:
    """Maps raw model probabilities to calibrated ones (isotonic, or logistic if positives are scarce)."""

    kind: str = "identity"
    isotonic: IsotonicRegression | None = None
    platt: tuple[float, float] | None = None  # intercept, slope on logit scale
    n_positive: int = 0

    def fit(self, raw: np.ndarray, y: np.ndarray) -> Calibrator:
        raw, y = np.asarray(raw, float), np.asarray(y, float)
        self.n_positive = int(y.sum())
        if self.n_positive == 0 or self.n_positive == len(y):
            self.kind = "identity"
        elif self.n_positive >= MIN_POSITIVES_FOR_ISOTONIC:
            self.kind = "isotonic"
            self.isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
                raw, y
            )
        else:
            self.kind = "platt"
            z = logit(np.clip(raw, 1e-6, 1 - 1e-6)).reshape(-1, 1)
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
            self.platt = (float(lr.intercept_[0]), float(lr.coef_[0, 0]))
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, float)
        if self.kind == "isotonic":
            return self.isotonic.predict(raw)
        if self.kind == "platt":
            a, b = self.platt
            return expit(a + b * logit(np.clip(raw, 1e-6, 1 - 1e-6)))
        return raw


def binary_metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    """Precision, recall, F1 at ``threshold`` plus PR-AUC (average precision), ROC-AUC and Brier score."""
    y = np.asarray(y, int)
    prob = np.asarray(prob, float)
    pred = prob >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if tp and precision == precision and recall == recall
        else 0.0
    )
    positive = int(y.sum())
    ok = 0 < positive < len(y)
    return {
        "n": int(len(y)),
        "n_positive": positive,
        "base_rate": float(y.mean()),
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": float(f1),
        "pr_auc": float(average_precision_score(y, prob)) if ok else float("nan"),
        "roc_auc": float(roc_auc_score(y, prob)) if ok else float("nan"),
        "brier": float(brier_score_loss(y, np.clip(prob, 0, 1))),
    }


def best_f1_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    """Probability threshold that maximises F1 on the given (validation) data; 0.5 if undefined."""
    y = np.asarray(y, int)
    if y.sum() == 0:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y, prob)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))]) if len(thresholds) else 0.5


def reliability_table(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Reliability diagram data: equal-count bins of predicted probability versus observed frequency."""
    frame = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(prob, float)})
    if frame["p"].nunique() < 2:
        return pd.DataFrame(
            {"mean_pred": [frame["p"].mean()], "observed": [frame["y"].mean()], "n": [len(frame)]}
        )
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), bins, labels=False, duplicates="drop")
    grouped = frame.groupby("bin").agg(
        mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size")
    )
    return grouped.reset_index(drop=True)


def brier_skill(y: np.ndarray, prob: np.ndarray, reference: np.ndarray) -> float:
    """Brier skill score versus a reference forecast (1 = perfect, 0 = as good as the reference)."""
    ref = brier_score_loss(np.asarray(y, int), np.clip(reference, 0, 1))
    return (
        float(1.0 - brier_score_loss(np.asarray(y, int), np.clip(prob, 0, 1)) / ref)
        if ref > 0
        else float("nan")
    )


# ----------------------------------------------------------------------------- heatwave LSTM
class _HeatNet(nn.Module):
    def __init__(
        self,
        n_features: int = len(E.FEATURE_NAMES),
        hidden: int = 32,
        n_out: int = len(HEAT_HORIZONS),
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, n_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


@dataclass
class HeatwaveModel:
    """LSTM with three onset-within-h-days outputs, plus standardiser and calibrators."""

    hidden: int = 32
    max_epochs: int = 25
    patience: int = 5
    batch_size: int = 512
    seed: int = field(default_factory=get_seed)
    net: _HeatNet | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    calibrators: list[Calibrator] = field(default_factory=list)
    thresholds: list[float] = field(default_factory=list)
    history: list[tuple[float, float]] = field(default_factory=list)

    def fit(
        self,
        heat: E.HeatData,
        train_years: tuple[int, int],
        valid_years: tuple[int, int],
        stride: int = 2,
    ) -> HeatwaveModel:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)  # noqa: NPY002 - reproducibility of any numpy-based tooling
        self.mean, self.std = heat.fit_standardiser(*train_years)
        d_tr, t_tr = heat.sample_index(*train_years, stride=stride)
        d_va, t_va = heat.sample_index(*valid_years, stride=3)
        y_tr, y_va = heat.targets(d_tr, t_tr), heat.targets(d_va, t_va)
        pos = np.maximum(y_tr.sum(axis=0), 1.0)
        pos_weight = torch.tensor(np.minimum((len(y_tr) - pos) / pos, 100.0), dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.net = _HeatNet(hidden=self.hidden)
        optimiser = torch.optim.AdamW(self.net.parameters(), lr=2e-3, weight_decay=1e-4)
        x_va = torch.tensor(heat.windows(d_va, t_va, self.mean, self.std))
        yv = torch.tensor(y_va)
        generator = torch.Generator().manual_seed(self.seed)
        best_state, best, stale = None, np.inf, 0
        for _ in range(self.max_epochs):
            self.net.train()
            order = torch.randperm(len(y_tr), generator=generator).numpy()
            total = 0.0
            for start in range(0, len(order), self.batch_size):
                idx = np.sort(order[start : start + self.batch_size])
                xb = torch.tensor(heat.windows(d_tr[idx], t_tr[idx], self.mean, self.std))
                optimiser.zero_grad()
                loss = loss_fn(self.net(xb), torch.tensor(y_tr[idx]))
                loss.backward()
                optimiser.step()
                total += float(loss) * len(idx)
            self.net.eval()
            with torch.no_grad():
                val = float(loss_fn(self.net(x_va), yv))
            self.history.append((total / len(order), val))
            if val < best - 1e-4:
                best, stale = val, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        # Calibrate on the validation years at their natural frequency (stride 1, no subsampling).
        d_cal, t_cal = heat.sample_index(*valid_years, stride=1)
        raw = self.predict_raw(heat, d_cal, t_cal)
        y_cal = heat.targets(d_cal, t_cal)
        self.calibrators = [Calibrator().fit(raw[:, j], y_cal[:, j]) for j in range(raw.shape[1])]
        calibrated = self.predict_proba(heat, d_cal, t_cal)
        self.thresholds = [
            best_f1_threshold(y_cal[:, j], calibrated[:, j]) for j in range(raw.shape[1])
        ]
        return self

    def predict_raw(
        self, heat: E.HeatData, d_idx: np.ndarray, t_idx: np.ndarray, batch: int = 4096
    ) -> np.ndarray:
        """Uncalibrated sigmoid outputs (N, 3)."""
        out = []
        with torch.no_grad():
            for start in range(0, len(d_idx), batch):
                sl = slice(start, start + batch)
                x = torch.tensor(heat.windows(d_idx[sl], t_idx[sl], self.mean, self.std))
                out.append(torch.sigmoid(self.net(x)).numpy())
        return np.concatenate(out) if out else np.zeros((0, len(HEAT_HORIZONS)))

    def predict_proba(self, heat: E.HeatData, d_idx: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
        """Calibrated probabilities (N, 3) for onset within 3, 7, 14 days."""
        raw = self.predict_raw(heat, d_idx, t_idx)
        return np.column_stack([cal.predict(raw[:, j]) for j, cal in enumerate(self.calibrators)])

    def state(
        self, meta: dict[str, Any], climatology: np.ndarray | None = None, served: str = "lstm"
    ) -> dict[str, Any]:
        """Serialisable state for ``models/heatwave_lstm.pt`` (``served`` = which probability risk_forecast returns)."""
        return {
            "served": served,
            "climatology_table": climatology,
            "net": self.net.state_dict(),
            "hidden": self.hidden,
            "mean": self.mean,
            "std": self.std,
            "calibrators": self.calibrators,
            "thresholds": self.thresholds,
            "horizons": HEAT_HORIZONS,
            "window_days": E.WINDOW_DAYS,
            "feature_names": E.FEATURE_NAMES,
            "season_months": E.SEASON_MONTHS,
            "epochs_trained": len(self.history),
            "meta": meta,
        }


def climatology_table(
    heat: E.HeatData, train_years: tuple[int, int], half_window: int = 15
) -> np.ndarray:
    """Climatological onset-within-h frequency, shape (n_districts, 366, 3).

    For every district and calendar-day slot: the share of training-year as-of days followed by an onset
    within 3 / 7 / 14 days, smoothed circularly over +-15 days.
    """
    slot = L._slot(pd.Series(heat.dates))
    in_train = (heat.dates.year >= train_years[0]) & (heat.dates.year <= train_years[1])
    table = np.zeros((len(heat.districts), 366, len(HEAT_HORIZONS)))
    kernel = np.ones(2 * half_window + 1) / (2 * half_window + 1)
    for j, h in enumerate(HEAT_HORIZONS):
        y = heat.labels[f"y{h}"].astype(float)
        daily = np.zeros((len(heat.districts), 366))
        for s_ in range(366):
            sel = in_train & (slot == s_)
            if sel.any():
                daily[:, s_] = y[:, sel].mean(axis=1)
        padded = np.concatenate([daily[:, -half_window:], daily, daily[:, :half_window]], axis=1)
        table[:, :, j] = np.array([np.convolve(row, kernel, mode="valid") for row in padded])
    return table


def climatology_heat(
    heat: E.HeatData,
    train_years: tuple[int, int],
    d_idx: np.ndarray,
    t_idx: np.ndarray,
    half_window: int = 15,
) -> np.ndarray:
    """Climatological probabilities (N, 3) for the given district / day samples (see :func:`climatology_table`)."""
    table = climatology_table(heat, train_years, half_window)
    slot = L._slot(pd.Series(heat.dates))
    return table[d_idx, slot[t_idx]]


# ----------------------------------------------------------------------------- drought XGBoost
@dataclass
class DroughtModel:
    """XGBoost classifier for next-month drought with calibration and a validation-chosen threshold."""

    features: tuple[str, ...]
    n_trials: int = 20
    model: Any = None
    calibrator: Calibrator = field(default_factory=Calibrator)
    threshold: float = 0.5
    params: dict[str, Any] = field(default_factory=dict)

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> DroughtModel:
        import optuna
        import xgboost as xgb

        X_tr, y_tr = train[list(self.features)], train["target"].astype(int)
        X_va, y_va = valid[list(self.features)], valid["target"].astype(int)
        weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

        def make(params: dict[str, Any], early: bool) -> Any:
            args: dict[str, Any] = {
                "tree_method": "hist",
                "random_state": get_seed(),
                "n_jobs": 4,
                "scale_pos_weight": weight,
                "eval_metric": "aucpr",
                "n_estimators": 600,
            }
            if early:
                args["early_stopping_rounds"] = 30
            args.update(params)  # tuned values (incl. n_estimators) override the defaults
            return xgb.XGBClassifier(**args)

        def objective(trial: Any) -> float:
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 5),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            }
            clf = make(params, True).fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            trial.set_user_attr("n_estimators", int(clf.best_iteration) + 1)
            return float(average_precision_score(y_va, clf.predict_proba(X_va)[:, 1]))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=get_seed())
        )
        study.optimize(objective, n_trials=self.n_trials)
        self.params = {
            **study.best_params,
            "n_estimators": study.best_trial.user_attrs["n_estimators"],
        }
        self.model = make(self.params, False).fit(X_tr, y_tr)
        raw_va = self.model.predict_proba(X_va)[:, 1]
        self.calibrator = Calibrator().fit(raw_va, y_va.to_numpy())
        self.threshold = best_f1_threshold(y_va.to_numpy(), self.calibrator.predict(raw_va))
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated probability that next month's SPI-3 is <= -1."""
        return self.calibrator.predict(self.model.predict_proba(frame[list(self.features)])[:, 1])


def climatology_drought(train: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """Frequency of next-month drought per district and calendar month over the training rows."""
    freq = train.groupby(["district_id", train["month"].dt.month])["target"].mean()
    keys = list(zip(rows["district_id"], rows["month"].dt.month, strict=True))
    return np.array([freq.get(k, train["target"].mean()) for k in keys])


# ----------------------------------------------------------------------------- risk levels
def level_from_probability(
    hazard: str, p: float, thresholds: dict[str, dict[str, float]] | None = None
) -> str:
    """'low', 'moderate', 'high' or 'severe' for one hazard from its calibrated probability."""
    cfg = (thresholds or load_settings()["risk_levels"])[hazard]
    if p >= cfg["severe"]:
        return "severe"
    if p >= cfg["high"]:
        return "high"
    if p >= cfg["moderate"]:
        return "moderate"
    return "low"


def combine_levels(
    heatwave_p: float, drought_p: float, thresholds: dict | None = None
) -> tuple[str, str, int]:
    """Overall level, the hazard that sets it and that hazard's lead time in days."""
    thresholds = thresholds or load_settings()["risk_levels"]
    levels = {
        "heatwave": level_from_probability("heatwave", heatwave_p, thresholds),
        "drought": level_from_probability("drought", drought_p, thresholds),
    }
    rank = {name: LEVELS.index(lv) for name, lv in levels.items()}
    if rank["heatwave"] != rank["drought"]:
        hazard = max(rank, key=rank.get)
    else:  # same level: the hazard closer to escalating (larger share of its 'moderate' threshold; drought wins ties)
        hazard = max(
            ("heatwave", "drought"),
            key=lambda h: (
                (heatwave_p if h == "heatwave" else drought_p) / thresholds[h]["moderate"],
                h == "drought",
            ),
        )
    return levels[hazard], hazard, HEAT_LEAD_DAYS if hazard == "heatwave" else DROUGHT_LEAD_DAYS


# ----------------------------------------------------------------------------- serving
class ModelsNotTrainedError(RuntimeError):
    """The early-warning artifacts are missing; run ``make train-extreme``."""


@lru_cache(maxsize=1)
def _serving() -> dict[str, Any]:
    heat_path, drought_path = (
        get_path("models") / HEAT_ARTIFACT,
        get_path("models") / DROUGHT_ARTIFACT,
    )
    if not heat_path.exists() or not drought_path.exists():
        raise ModelsNotTrainedError(
            "run `make train-extreme` to create models/heatwave_lstm.pt and models/drought_xgb.joblib"
        )
    heat_state = torch.load(heat_path, weights_only=False)
    drought = joblib.load(drought_path)
    processed = get_path("data_processed")
    weather = pd.read_parquet(processed / "weather_daily.parquet")
    satellite = pd.read_parquet(processed / "satellite_16d.parquet")
    reference_end = int(heat_state["meta"]["reference_end_year"])
    net = _HeatNet(hidden=heat_state["hidden"])
    net.load_state_dict(heat_state["net"])
    net.eval()
    heat = E.build_heat_data(weather, reference_end)
    table = E.drought_table(weather, satellite, reference_end)
    monthly = spi.monthly_rain(weather)
    reference = monthly[monthly["month"].dt.year <= reference_end]
    normals = reference.groupby(["district_id", reference["month"].dt.month])["rain_mm"].mean()
    return {
        "net": net,
        "heat_state": heat_state,
        "heat": heat,
        "drought": drought,
        "drought_table": table,
        "month_normals": normals.to_dict(),
    }


def reset_cache() -> None:
    """Forget loaded artifacts (after retraining or in tests)."""
    _serving.cache_clear()


def forecast_date_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last ``as_of_date`` that :func:`risk_forecast` accepts.

    The heatwave network needs ``WINDOW_DAYS`` of observed weather and the drought model needs complete SPI
    features for the last complete month, so the usable range is narrower than the weather record.
    """
    serving = _serving()
    dates = serving["heat"].dates
    table = serving["drought_table"]
    complete = table[list(serving["drought"]["features"])].notna().all(axis=1)
    months = table.loc[complete, "month"]
    first = max(dates[E.WINDOW_DAYS - 1], months.min().to_period("M").end_time.normalize())
    last = min(
        dates[-1], (months.max().to_period("M") + 1).end_time.normalize() - pd.Timedelta(days=1)
    )
    return first, last


def risk_forecast(district_id: str, as_of_date: str | pd.Timestamp) -> dict[str, Any]:
    """Early-warning probabilities and a risk level for a district on a date.

    Returns:
        ``{"heatwave_p", "drought_p", "level", "lead_days"}`` where ``heatwave_p`` is the calibrated probability of a
        heatwave onset within the next 14 days (0.0 outside the March-June season the model is trained for),
        ``drought_p`` the probability that next month's SPI-3 is <= -1, ``level`` is ``low|moderate|high|severe``
        (thresholds per hazard in settings.yaml ``risk_levels``) and ``lead_days`` the lead of the hazard that sets
        the level (14 heatwave, 30 drought). ``heatwave_source`` says whether the probability comes from the LSTM or
        from climatology (used unless the LSTM shows skill over it in the evaluation). Extra keys: ``heatwave_p3``,
        ``heatwave_p7``, ``heatwave_in_season``, ``hazard``, ``drought_month``, ``as_of``, ``drought_relevant`` (False when the
        target month is normally almost dry: the drought probability is then reported but does not set ``level``).

    Raises:
        ValueError: unknown district, or a date outside the observed weather record (a forecast needs data
            up to the as-of date).
    """
    s = _serving()
    heat: E.HeatData = s["heat"]
    state = s["heat_state"]
    as_of = pd.Timestamp(as_of_date).normalize()
    if district_id not in heat.districts:
        raise ValueError(f"unknown district_id {district_id!r}")
    if as_of < heat.dates[E.WINDOW_DAYS - 1] or as_of > heat.dates[-1]:
        raise ValueError(
            f"as_of_date must be between {heat.dates[E.WINDOW_DAYS - 1].date()} and {heat.dates[-1].date()} (observed record)"
        )
    d_idx = np.array([heat.districts.index(district_id)])
    t_idx = np.array([int(heat.dates.get_loc(as_of))])
    in_season = int(as_of.month) in state["season_months"]
    probs = np.zeros(3)
    source = state.get("served", "lstm")
    if in_season and source == "climatology":
        slot = int(L._slot(pd.Series([as_of]))[0])
        probs = np.asarray(state["climatology_table"][d_idx[0], slot], float)
    elif in_season:
        window = heat.windows(d_idx, t_idx, state["mean"], state["std"])
        with torch.no_grad():
            raw = torch.sigmoid(s["net"](torch.tensor(window))).numpy()[0]
        probs = np.array(
            [cal.predict(np.array([raw[j]]))[0] for j, cal in enumerate(state["calibrators"])]
        )
    # Drought uses the last *complete* month on or before the as-of date.
    period = as_of.to_period("M")
    if as_of != period.end_time.normalize():
        period = period - 1
    month_start = period.to_timestamp()
    table = s["drought_table"]
    row = table[(table["district_id"] == district_id) & (table["month"] == month_start)]
    art = s["drought"]
    if row.empty or row[list(art["features"])].isna().any(axis=1).iloc[0]:
        raise ValueError(
            f"not enough history for drought features of {district_id} in {month_start:%Y-%m}"
        )
    drought_p = float(art["model_wrapper"].predict_proba(row)[0])
    # SPI of a month that is normally almost dry is not agronomically meaningful: such a drought may be reported
    # (drought_p) but must not set the risk level (early Jan 2005 'severe dry spell' in a dry-season month).
    target_month = (month_start + pd.offsets.MonthBegin(1)).month
    normal_rain = float(s["month_normals"].get((district_id, target_month), np.nan))
    minimum = float(load_settings()["early_warning"]["drought_min_month_rain_mm"])
    relevant = bool(normal_rain >= minimum)
    level, hazard, lead = combine_levels(float(probs[2]), drought_p if relevant else 0.0)
    return {
        "heatwave_p": float(probs[2]),
        "drought_p": drought_p,
        "drought_relevant": relevant,
        "drought_normal_month_rain_mm": normal_rain,
        "level": level,
        "lead_days": lead,
        "heatwave_p3": float(probs[0]),
        "heatwave_p7": float(probs[1]),
        "heatwave_in_season": bool(in_season),
        "heatwave_source": source if in_season else "out_of_season",
        "hazard": hazard,
        "drought_month": f"{month_start:%Y-%m}",
        "as_of": f"{as_of:%Y-%m-%d}",
    }
