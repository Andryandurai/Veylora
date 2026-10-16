"""Scenario-aware risk indicators for the 3D simulator.

``risk_forecast`` (Phase 5) answers "what is the risk on this date given the observed weather". The simulator also
needs risk *under what-if weather* (rainfall x0.6, +3 C), so this module re-evaluates the Phase 5 drought model on
scenario weather and derives a heat indicator from the same event definition used for the heatwave labels.

* ``drought_p``: the shipped drought model (``models/drought_xgb.joblib``) applied to monthly SPI-1/SPI-3,
  deficit, rainfall and Tmax-anomaly features recomputed from the scenario daily weather. The SPI distributions
  stay fixed on the unmodified reference years (1991-2014), so a drier scenario produces a more negative SPI.
  It is the probability that the SPI-3 of a month is <= -1 given data up to the end of the previous month.
* ``heat_share``: share of days in a week that qualify as IMD heatwave days (Tmax >= 40 C and >= 4.5 C above the
  district's daily normal). It is a deterministic function of the scenario weather, **not a model probability**;
  the heatwave LSTM has no demonstrated skill over climatology (D-019) and climatology cannot react to a scenario.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from analytics.predictive import extreme_data as E
from analytics.predictive import features as F
from analytics.predictive import labels as L
from analytics.predictive import spi
from config import get_path, load_settings

LOG = logging.getLogger(__name__)

WEATHER_ONLY_FEATURES = set(E.DROUGHT_FEATURES_WEATHER)


class ScenarioRisk:
    """Loads the reference statistics once and evaluates risks for many scenarios."""

    def __init__(self) -> None:
        processed = get_path("data_processed")
        weather = pd.read_parquet(processed / "weather_daily.parquet").sort_values(
            ["district_id", "date"]
        )
        artifact = joblib.load(get_path("models") / "drought_xgb.joblib")
        self.reference_end = int(artifact["meta"]["reference_end_year"])
        self.model = artifact["model_wrapper"]
        extra = set(self.model.features) - WEATHER_ONLY_FEATURES
        if extra:
            raise NotImplementedError(
                f"drought model uses satellite features {sorted(extra)} that cannot be recomputed for scenarios"
            )
        monthly = spi.monthly_rain(weather)
        self.references = spi.fit_reference(monthly, self.reference_end, (1, 3))
        tmax_month = (
            weather.assign(month=weather["date"].dt.to_period("M").dt.to_timestamp())
            .groupby(["district_id", "month"], as_index=False)["tmax_c"]
            .mean()
        )
        ref = tmax_month[tmax_month["month"].dt.year <= self.reference_end]
        self.tmax_clim = (
            ref.groupby(["district_id", ref["month"].dt.month])["tmax_c"].mean().to_dict()
        )
        normals = L.daily_normals(weather, self.reference_end)
        self.normals = {
            d: g.sort_values("slot")["tmax_normal"].to_numpy()
            for d, g in normals.groupby("district_id")
        }
        self.daily = {d: g.reset_index(drop=True) for d, g in weather.groupby("district_id")}

    def scenario_daily(
        self, district_id: str, year: int, season: str, rain_mult: float, temp_delta: float
    ) -> pd.DataFrame:
        """The district's daily weather with the scenario applied to every day of the season in ``year``."""
        frame = self.daily[district_id].copy()
        start, end = F.season_bounds(year, season)
        inside = (frame["date"] >= start) & (frame["date"] <= end)
        frame.loc[inside, "rain_mm"] = frame.loc[inside, "rain_mm"] * rain_mult
        frame.loc[inside, "tmax_c"] = frame.loc[inside, "tmax_c"] + temp_delta
        frame.loc[inside, "tmin_c"] = frame.loc[inside, "tmin_c"] + temp_delta
        return frame

    def drought_by_month(
        self, district_id: str, year: int, season: str, rain_mult: float
    ) -> pd.Series:
        """Probability that SPI-3 of each month <= -1, indexed by the *target* month start, for the season's months.

        The probability for month m uses features at the end of month m-1. Only rainfall is scenario-scaled: SPI
        drought is defined by rainfall, and the model's Tmax-anomaly input would be pushed outside its training
        range by a temperature shift (a tree-model extrapolation artefact that lowered drought risk under warming
        in our first test), so temperature stays at its observed value here.
        """
        daily = self.scenario_daily(district_id, year, season, rain_mult, 0.0)
        start, end = F.season_bounds(year, season)
        lo, hi = start - pd.DateOffset(months=5), end + pd.DateOffset(months=1)
        window = daily[(daily["date"] >= lo) & (daily["date"] <= hi)].copy()
        window["month"] = window["date"].dt.to_period("M").dt.to_timestamp()
        monthly = window.groupby("month").agg(
            rain_mm=("rain_mm", "sum"), tmax_month=("tmax_c", "mean")
        )
        table = spi.apply_reference(
            monthly[["rain_mm"]], district_id, self.references, (1, 3)
        ).join(monthly["tmax_month"])
        cal = table.index.month
        table["tmax_anom"] = table["tmax_month"] - pd.Series(
            [self.tmax_clim.get((district_id, m), np.nan) for m in cal], index=table.index
        )
        table["spi1_lag1"] = table["spi1"].shift(1)
        table["spi3_lag1"] = table["spi3"].shift(1)
        table["spi3_lag2"] = table["spi3"].shift(2)
        table["month_sin"] = np.sin(2 * np.pi * cal / 12)
        table["month_cos"] = np.cos(2 * np.pi * cal / 12)
        usable = table.dropna(subset=list(self.model.features))
        probabilities = pd.Series(
            self.model.predict_proba(usable), index=usable.index + pd.offsets.MonthBegin(1)
        )
        target_months = pd.date_range(
            start.to_period("M").to_timestamp(), end.to_period("M").to_timestamp(), freq="MS"
        )
        return probabilities.reindex(target_months)

    def weekly_drought_p(
        self, district_id: str, year: int, season: str, rain_mult: float, n_weeks: int = 16
    ) -> np.ndarray:
        """Drought probability for each of the first ``n_weeks`` weeks (the month containing the week's middle day)."""
        by_month = self.drought_by_month(district_id, year, season, rain_mult)
        start, _ = F.season_bounds(year, season)
        values = []
        for w in range(n_weeks):
            middle = start + pd.Timedelta(days=7 * w + 3)
            values.append(by_month.get(middle.to_period("M").to_timestamp(), np.nan))
        out = np.asarray(values, float)
        return np.where(np.isnan(out), np.nanmean(out) if np.isfinite(out).any() else 0.0, out)

    def weekly_heat_share(
        self, district_id: str, year: int, season: str, temp_delta: float, n_weeks: int = 16
    ) -> np.ndarray:
        """Share of days per week that qualify as IMD heatwave days under the temperature shift."""
        start, _ = F.season_bounds(year, season)
        daily = self.daily[district_id]
        window = daily[daily["date"] >= start].head(7 * n_weeks)
        normal = self.normals[district_id][L._slot(window["date"])]
        qualifying = L.qualifying_days(window["tmax_c"].to_numpy(float) + temp_delta, normal)
        return qualifying.reshape(n_weeks, 7).mean(axis=1)


@lru_cache(maxsize=1)
def get_scenario_risk() -> ScenarioRisk:
    """Shared instance (loading the weather and fitting the references takes a few seconds)."""
    return ScenarioRisk()


def heat_threshold_note() -> str:
    """Human-readable definition used in the simulator's metadata."""
    return f"share of days with Tmax >= {L.TMAX_MIN_C:g} C and >= {L.DEPARTURE_C:g} C above the district daily normal (1991-{load_settings()['splits']['extreme_weather']['train_end']})"
