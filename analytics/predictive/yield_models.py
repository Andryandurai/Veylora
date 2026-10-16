"""Yield models on the relative target ``y_rel = yield / prior mean yield``.

* ``NaiveLastYear`` (required baseline), ``DistrictTrend`` (linear trend of the district-crop's earlier years)
  and ``PriorMean3`` (mean of the previous three years) use no weather at all.
* ``RidgeModel``, ``RandomForestModel``, ``XGBoostModel`` (Optuna, 40 trials, validation years only) and
  ``LSTMModel`` (weekly sequences + static features) add weather and, optionally, satellite features.
* ``ConformalXGB`` gives 80 percent intervals by conformalized quantile regression, calibrated on the
  validation years.

Every model has ``fit(train, valid)`` and ``predict_rel(data)``; ``predict`` multiplies by the scale to
return t/ha. Seeds are fixed (42) and nothing here looks at the test years.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from config import get_seed

LOG = logging.getLogger(__name__)

N_OPTUNA_TRIALS = 40
STATIC_PREFIXES = ("crop_", "season_", "cluster_")
STATIC_NUMERIC = ("prev_year_rel", "mean_prev3_rel", "trend_pred_rel", "log_scale")


@dataclass
class Data:
    """One split of a dataset: features, relative target, metadata and (for the LSTM) sequences."""

    X: pd.DataFrame
    y: pd.Series | None
    meta: pd.DataFrame
    seq: np.ndarray | None = None
    mask: np.ndarray | None = None

    def subset(self, rows: np.ndarray) -> Data:
        """Rows selected by a boolean mask."""
        return Data(
            self.X[rows].reset_index(drop=True),
            None if self.y is None else self.y[rows].reset_index(drop=True),
            self.meta[rows].reset_index(drop=True),
            None if self.seq is None else self.seq[rows],
            None if self.mask is None else self.mask[rows],
        )


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


TEMPERATURE_PREFIXES = ("mean_tmax_d", "heat_days_d", "gdd_d")


def temperature_constraints(columns: Any) -> dict[str, int]:
    """Monotone-decreasing constraints (-1) on temperature-driven features, 0 elsewhere.

    Why: tree models cannot extrapolate. For a what-if shift beyond the historical temperature range they
    return the highest-temperature leaf, which in our first run *raised* predicted yield by ~9 percent at
    +4 C although Phase 3's fixed-effects estimates show heat lowering rice and cotton yields. The
    constraint encodes that domain assumption (warmer-than-normal early-season weather does not increase
    yield); it is applied to the shipped models and its accuracy cost is reported (docs/DECISIONS.md D-014).
    """
    return {c: (-1 if str(c).startswith(TEMPERATURE_PREFIXES) else 0) for c in columns}


class YieldModel:
    """Interface: ``fit`` on train (+ valid for tuning), ``predict_rel`` returns relative yield."""

    name = "model"

    def fit(self, train: Data, valid: Data) -> YieldModel:
        raise NotImplementedError

    def predict_rel(self, data: Data) -> np.ndarray:
        raise NotImplementedError

    def predict(self, data: Data) -> np.ndarray:
        """Yield in t/ha."""
        return self.predict_rel(data) * data.meta["scale"].to_numpy()


class NaiveLastYear(YieldModel):
    """Last year's yield of the same district-crop-season (falls back to the prior mean if missing)."""

    name = "naive_last_year"

    def fit(self, train: Data, valid: Data) -> NaiveLastYear:
        return self

    def predict_rel(self, data: Data) -> np.ndarray:
        return data.X["prev_year_rel"].fillna(1.0).to_numpy()


class DistrictTrend(YieldModel):
    """Linear trend through the district-crop's earlier yields (falls back to last year, then the mean)."""

    name = "district_trend"

    def fit(self, train: Data, valid: Data) -> DistrictTrend:
        return self

    def predict_rel(self, data: Data) -> np.ndarray:
        return data.X["trend_pred_rel"].fillna(data.X["prev_year_rel"]).fillna(1.0).to_numpy()


class PriorMean3(YieldModel):
    """Mean of the previous three crop years (falls back to last year, then the prior mean)."""

    name = "prior_mean_3y"

    def fit(self, train: Data, valid: Data) -> PriorMean3:
        return self

    def predict_rel(self, data: Data) -> np.ndarray:
        return data.X["mean_prev3_rel"].fillna(data.X["prev_year_rel"]).fillna(1.0).to_numpy()


class RidgeModel(YieldModel):
    """Median-imputed, standardised ridge regression; alpha chosen on the validation years."""

    name = "ridge"

    def __init__(self) -> None:
        self.alpha = 1.0
        self.pipeline: Any = None

    def _make(self, alpha: float) -> Any:
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=alpha),
        )

    def fit(self, train: Data, valid: Data) -> RidgeModel:
        best = (np.inf, 1.0)
        for alpha in np.logspace(-2, 3, 11):
            score = rmse(valid.y, self._make(alpha).fit(train.X, train.y).predict(valid.X))
            best = min(best, (score, float(alpha)))
        self.alpha = best[1]
        self.pipeline = self._make(self.alpha).fit(train.X, train.y)
        return self

    def predict_rel(self, data: Data) -> np.ndarray:
        return self.pipeline.predict(data.X)


class RandomForestModel(YieldModel):
    """Random forest (NaN-aware); leaf size and max_features chosen on the validation years."""

    name = "random_forest"

    def __init__(self, n_estimators: int = 400) -> None:
        self.n_estimators = n_estimators
        self.params: dict[str, Any] = {}
        self.model: RandomForestRegressor | None = None

    def _make(self, **params: Any) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.n_estimators, n_jobs=-1, random_state=get_seed(), **params
        )

    def fit(self, train: Data, valid: Data) -> RandomForestModel:
        best: tuple[float, dict[str, Any]] = (np.inf, {})
        for leaf in (2, 5, 10):
            for max_features in (0.3, 0.6, 1.0):
                params = {"min_samples_leaf": leaf, "max_features": max_features}
                score = rmse(valid.y, self._make(**params).fit(train.X, train.y).predict(valid.X))
                if score < best[0]:
                    best = (score, params)
        self.params = best[1]
        self.model = self._make(**self.params).fit(train.X, train.y)
        return self

    def predict_rel(self, data: Data) -> np.ndarray:
        return self.model.predict(data.X)


class XGBoostModel(YieldModel):
    """XGBoost regressor tuned with Optuna (validation RMSE only), early stopping on the validation years."""

    name = "xgboost"

    def __init__(
        self,
        n_trials: int = N_OPTUNA_TRIALS,
        params: dict[str, Any] | None = None,
        monotone_temperature: bool = True,
        name: str | None = None,
    ) -> None:
        """
        Args:
            n_trials: Optuna trials (validation RMSE).
            params: fixed hyperparameters (skips tuning), e.g. for leave-one-state-out.
            monotone_temperature: forbid temperature, heat-day and degree-day features from *raising*
                predicted yield (see :func:`temperature_constraints`).
            name: model name used in reports (defaults to ``xgboost``).
        """
        self.n_trials = n_trials
        self.params = params
        self.monotone_temperature = monotone_temperature
        if name:
            self.name = name
        self.constraints: dict[str, int] | None = None
        self.model: Any = None
        self.best_value: float | None = None

    def _regressor(self, params: dict[str, Any], early_stopping: bool) -> Any:
        import xgboost as xgb

        args = {
            "tree_method": "hist",
            "random_state": get_seed(),
            "n_jobs": 4,
            "objective": "reg:squarederror",
            "n_estimators": 1000,
            **params,
        }
        if self.constraints:
            args["monotone_constraints"] = self.constraints
        if early_stopping:
            args["early_stopping_rounds"] = 40
        return xgb.XGBRegressor(**args)

    @staticmethod
    def _suggest(trial: Any) -> dict[str, Any]:
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 0.05),
        }

    def fit(self, train: Data, valid: Data) -> XGBoostModel:
        import optuna

        self.constraints = (
            temperature_constraints(train.X.columns) if self.monotone_temperature else None
        )
        if self.params is None:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial: Any) -> float:
                model = self._regressor(self._suggest(trial), early_stopping=True)
                model.fit(train.X, train.y, eval_set=[(valid.X, valid.y)], verbose=False)
                trial.set_user_attr("n_estimators", int(model.best_iteration) + 1)
                return rmse(valid.y, model.predict(valid.X))

            study = optuna.create_study(
                direction="minimize", sampler=optuna.samplers.TPESampler(seed=get_seed())
            )
            study.optimize(objective, n_trials=self.n_trials)
            self.params = {
                **study.best_params,
                "n_estimators": study.best_trial.user_attrs["n_estimators"],
            }
            self.best_value = float(study.best_value)
        self.model = self._regressor(self.params, early_stopping=False).fit(train.X, train.y)
        return self

    def predict_rel(self, data: Data) -> np.ndarray:
        return self.model.predict(data.X)


@dataclass
class ConformalXGB:
    """Conformalized quantile regression (CQR) around an XGBoost model: 80 % intervals by default."""

    params: dict[str, Any]
    constraints: dict[str, int] | None = None
    alpha: float = 0.2
    qhat: float = 0.0
    model: Any = None
    coverage_valid: float | None = field(default=None)

    def fit(self, train: Data, valid: Data) -> ConformalXGB:
        import xgboost as xgb

        args = {k: v for k, v in self.params.items() if k != "n_jobs"}
        self.model = xgb.XGBRegressor(
            tree_method="hist",
            objective="reg:quantileerror",
            quantile_alpha=np.array([self.alpha / 2, 1 - self.alpha / 2]),
            random_state=get_seed(),
            n_jobs=4,
            **({"monotone_constraints": self.constraints} if self.constraints else {}),
            **args,
        ).fit(train.X, train.y)
        pred = self.model.predict(valid.X)
        lo, hi = np.minimum(pred[:, 0], pred[:, 1]), np.maximum(pred[:, 0], pred[:, 1])
        y = valid.y.to_numpy()
        scores = np.maximum(lo - y, y - hi)
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.qhat = float(np.quantile(scores, level, method="higher"))
        self.coverage_valid = float(np.mean((y >= lo - self.qhat) & (y <= hi + self.qhat)))
        return self

    def interval_rel(
        self, data: Data, point_rel: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Relative lower/upper bounds (widened by the conformal quantile, always containing the point forecast)."""
        pred = self.model.predict(data.X)
        lo = np.minimum(pred[:, 0], pred[:, 1]) - self.qhat
        hi = np.maximum(pred[:, 0], pred[:, 1]) + self.qhat
        if point_rel is not None:
            lo, hi = np.minimum(lo, point_rel), np.maximum(hi, point_rel)
        return np.maximum(lo, 0.0), hi


# ----------------------------------------------------------------------------- LSTM
class _Net(nn.Module):
    def __init__(
        self, n_channels: int, n_static: int, hidden: int, n_obs: int, dropout: float
    ) -> None:
        super().__init__()
        self.n_obs = n_obs
        self.lstm = nn.LSTM(n_channels + 1, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, 48), nn.ReLU(), nn.Dropout(dropout), nn.Linear(48, 1)
        )

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(torch.cat([seq, mask.unsqueeze(-1)], dim=-1))
        last = out[:, self.n_obs - 1, :]  # hidden state after the last observed week
        return self.head(torch.cat([last, static], dim=-1)).squeeze(-1)


class LSTMModel(YieldModel):
    """LSTM over the first weeks of the season (rain, Tmax, Tmin, NDVI, soil moisture) plus static features."""

    name = "lstm"

    def __init__(
        self,
        lead_days: int,
        use_satellite: bool = True,
        hidden: int = 32,
        max_epochs: int = 300,
        patience: int = 25,
    ) -> None:
        self.lead_days = lead_days
        self.use_satellite = use_satellite
        self.hidden, self.max_epochs, self.patience = hidden, max_epochs, patience
        self.net: _Net | None = None
        self.seq_stats: tuple[np.ndarray, np.ndarray] | None = None
        self.static_stats: tuple[pd.Series, pd.Series] | None = None
        self.static_columns: list[str] = []
        self.history: list[tuple[float, float]] = []

    # -- preprocessing -------------------------------------------------------------------
    def _channels(self) -> list[int]:
        return [0, 1, 2, 3, 4] if self.use_satellite else [0, 1, 2]

    def _static(self, data: Data) -> np.ndarray:
        frame = data.X[self.static_columns].copy()
        mean, std = self.static_stats
        return ((frame.fillna(mean) - mean) / std).to_numpy(dtype=np.float32)

    def _seq(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.seq_stats
        seq = (data.seq[:, :, self._channels()] - mean) / std * data.mask[:, :, None]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(data.mask, dtype=torch.float32)

    def fit(self, train: Data, valid: Data) -> LSTMModel:
        torch.manual_seed(get_seed())
        np.random.seed(get_seed())  # noqa: NPY002 - torch DataLoader-free loop, kept for reproducibility
        self.static_columns = [
            c for c in train.X.columns if c in STATIC_NUMERIC or c.startswith(STATIC_PREFIXES)
        ]
        stats_frame = train.X[self.static_columns]
        self.static_stats = (
            stats_frame.mean().fillna(0.0),
            stats_frame.std().replace(0, 1).fillna(1.0),
        )
        observed = train.mask.astype(bool)
        channels = self._channels()
        flat = train.seq[:, :, channels][observed]
        self.seq_stats = (flat.mean(axis=0), np.where(flat.std(axis=0) == 0, 1.0, flat.std(axis=0)))
        n_obs = int(train.mask[0].sum())
        self.net = _Net(len(channels), len(self.static_columns), self.hidden, n_obs, dropout=0.2)
        seq_tr, mask_tr = self._seq(train)
        seq_va, mask_va = self._seq(valid)
        static_tr, static_va = torch.tensor(self._static(train)), torch.tensor(self._static(valid))
        y_tr = torch.tensor(train.y.to_numpy(), dtype=torch.float32)
        y_va = torch.tensor(valid.y.to_numpy(), dtype=torch.float32)
        optimiser = torch.optim.AdamW(self.net.parameters(), lr=3e-3, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss(beta=0.2)
        generator = torch.Generator().manual_seed(get_seed())
        best_state, best_val, stale = None, np.inf, 0
        for _ in range(self.max_epochs):
            self.net.train()
            order = torch.randperm(len(y_tr), generator=generator)
            for start in range(0, len(order), 128):
                idx = order[start : start + 128]
                optimiser.zero_grad()
                loss = loss_fn(self.net(seq_tr[idx], mask_tr[idx], static_tr[idx]), y_tr[idx])
                loss.backward()
                optimiser.step()
            self.net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.net(seq_va, mask_va, static_va), y_va))
                train_loss = float(loss_fn(self.net(seq_tr, mask_tr, static_tr), y_tr))
            self.history.append((train_loss, val_loss))
            if val_loss < best_val - 1e-5:
                best_val, stale = val_loss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict_rel(self, data: Data) -> np.ndarray:
        seq, mask = self._seq(data)
        with torch.no_grad():
            return self.net(seq, mask, torch.tensor(self._static(data))).numpy().astype(float)

    def state(self) -> dict[str, Any]:
        """Serialisable state for ``models/yield_lstm.pt``."""
        return {
            "net": self.net.state_dict(),
            "lead_days": self.lead_days,
            "use_satellite": self.use_satellite,
            "hidden": self.hidden,
            "static_columns": self.static_columns,
            "seq_stats": self.seq_stats,
            "static_stats": (self.static_stats[0].to_dict(), self.static_stats[1].to_dict()),
            "n_obs": self.net.n_obs,
            "epochs_trained": len(self.history),
        }
