"""Data builders for the extreme-weather models.

Heatwave: a dense array of daily features per district so that 30-day windows can be gathered quickly,
plus the onset labels of :mod:`analytics.predictive.labels`. Drought: a monthly table of SPI features and the
next-month SPI-3 drought label. Everything that is *fitted* (normals, SPI distributions, anomaly climatologies)
uses a reference period passed in by the caller (the training years), never the evaluation years.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from analytics.predictive import labels as L
from analytics.predictive import spi

LOG = logging.getLogger(__name__)

WINDOW_DAYS = 30
SEASON_MONTHS = (
    3,
    4,
    5,
    6,
)  # as-of months for heatwave forecasts; all observed onsets fall in May-June
RAW_FEATURES = ("tmax_c", "tmin_c", "rain_mm", "rh_pct", "wind_ms", "solar_mj")
FEATURE_NAMES = (*RAW_FEATURES, "doy_sin", "doy_cos")


@dataclass
class HeatData:
    """Daily features and labels for all districts on a common date axis."""

    districts: list[str]
    dates: pd.DatetimeIndex
    features: np.ndarray  # (n_districts, T, 8) float32, unstandardised
    labels: dict[
        str, np.ndarray
    ]  # "y3","y7","y14","onset","heatwave","qualifying" -> (n_districts, T) bool

    def sample_index(
        self,
        first_year: int,
        last_year: int,
        stride: int = 1,
        months: tuple[int, ...] = SEASON_MONTHS,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(district index, time index) of every as-of day in the season window and year range."""
        t_ok = (
            (self.dates.year >= first_year)
            & (self.dates.year <= last_year)
            & np.isin(self.dates.month, months)
        )
        t_idx = np.flatnonzero(t_ok)
        t_idx = t_idx[t_idx >= WINDOW_DAYS - 1]
        if stride > 1:
            t_idx = t_idx[::stride]
        d_idx = np.repeat(np.arange(len(self.districts)), len(t_idx))
        return d_idx, np.tile(t_idx, len(self.districts))

    def windows(
        self, d_idx: np.ndarray, t_idx: np.ndarray, mean: np.ndarray, std: np.ndarray
    ) -> np.ndarray:
        """Standardised 30-day windows ending on the as-of day, shape (N, 30, 8)."""
        steps = t_idx[:, None] - np.arange(WINDOW_DAYS - 1, -1, -1)[None, :]
        raw = self.features[d_idx[:, None], steps]
        return ((raw - mean) / std).astype(np.float32)

    def fit_standardiser(self, first_year: int, last_year: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-feature mean and std over the given years (training years only)."""
        sel = (self.dates.year >= first_year) & (self.dates.year <= last_year)
        values = self.features[:, sel].reshape(-1, self.features.shape[-1])
        std = values.std(axis=0)
        return values.mean(axis=0), np.where(std == 0, 1.0, std)

    def targets(self, d_idx: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
        """(N, 3) float array of onset-within-3/7/14-days labels."""
        return np.column_stack([self.labels[f"y{h}"][d_idx, t_idx] for h in L.HORIZONS]).astype(
            np.float32
        )

    def recent(self, name: str, days: int, d_idx: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
        """True if ``name`` was True on any of the last ``days`` days up to and including the as-of day."""
        csum = np.concatenate(
            [np.zeros((len(self.districts), 1)), np.cumsum(self.labels[name], axis=1)], axis=1
        )
        lo = np.maximum(t_idx - days + 1, 0)
        return (csum[d_idx, t_idx + 1] - csum[d_idx, lo]) > 0


def build_heat_data(weather: pd.DataFrame, reference_end_year: int) -> HeatData:
    """Assemble :class:`HeatData`; heatwave normals use years <= ``reference_end_year``."""
    weather = weather.sort_values(["district_id", "date"])
    districts = sorted(weather["district_id"].unique())
    dates = pd.DatetimeIndex(np.sort(weather["date"].unique()))
    counts = weather.groupby("district_id")["date"].count()
    if (counts != len(dates)).any():
        raise ValueError("weather_daily must have the same complete daily range for every district")
    normals = L.daily_normals(weather, reference_end_year)
    flags = L.heatwave_frame(weather, normals).sort_values(["district_id", "date"])
    n_t = len(dates)
    doy = dates.dayofyear.to_numpy() / 365.25 * 2 * np.pi
    seasonal = np.column_stack([np.sin(doy), np.cos(doy)]).astype(np.float32)
    features = np.zeros((len(districts), n_t, len(FEATURE_NAMES)), dtype=np.float32)
    for i, district in enumerate(districts):
        group = weather[weather["district_id"] == district]
        features[i, :, : len(RAW_FEATURES)] = group[list(RAW_FEATURES)].to_numpy(np.float32)
        features[i, :, len(RAW_FEATURES) :] = seasonal
    labels = {}
    for name in ("y3", "y7", "y14", "onset", "heatwave", "qualifying"):
        labels[name] = (
            flags.pivot_table(index="district_id", columns="date", values=name, aggfunc="first")
            .loc[districts]
            .to_numpy(bool)
        )
    return HeatData(districts, dates, features, labels)


# ----------------------------------------------------------------------------- drought
DROUGHT_FEATURES_WEATHER = (
    "spi1",
    "spi1_lag1",
    "spi3",
    "spi3_lag1",
    "spi3_lag2",
    "deficit3_pct",
    "rain_mm",
    "tmax_anom",
    "month_sin",
    "month_cos",
)
DROUGHT_FEATURES_SATELLITE = ("soil_moisture", "ndvi_anom")


def monthly_means(weather: pd.DataFrame, satellite: pd.DataFrame | None) -> pd.DataFrame:
    """Monthly mean Tmax and (if available) mean NDVI / soil moisture per district.

    Satellite composites count toward a month only if they had finished by its last day (date + 15 days).
    """
    w = weather[["district_id", "date", "tmax_c"]].copy()
    w["month"] = w["date"].dt.to_period("M").dt.to_timestamp()
    out = (
        w.groupby(["district_id", "month"], as_index=False)["tmax_c"]
        .mean()
        .rename(columns={"tmax_c": "tmax_month"})
    )
    if satellite is not None and len(satellite):
        s = satellite[["district_id", "date", "ndvi", "soil_moisture", "is_synthetic"]].copy()
        s["month"] = (
            (s["date"] + pd.Timedelta(days=15)).dt.to_period("M").dt.to_timestamp()
        )  # month in which it finished
        agg = s.groupby(["district_id", "month"], as_index=False).agg(
            ndvi=("ndvi", "mean"),
            soil_moisture=("soil_moisture", "mean"),
            satellite_synthetic=("is_synthetic", "max"),
        )
        out = out.merge(agg, on=["district_id", "month"], how="left")
    return out


def drought_table(
    weather: pd.DataFrame, satellite: pd.DataFrame | None, reference_end_year: int
) -> pd.DataFrame:
    """Monthly feature table with the next-month drought label.

    Row = (district, as-of month m): features use data through the end of m; ``target`` = SPI-3 of month m+1
    <= -1. ``split_month`` is the as-of month start. Anomalies and SPI use the reference period only.
    """
    monthly = spi.monthly_rain(weather)
    table = spi.spi_table(monthly, reference_end_year, scales=(1, 3))
    table = table.merge(monthly_means(weather, satellite), on=["district_id", "month"], how="left")
    table = table.sort_values(["district_id", "month"]).reset_index(drop=True)
    cal = table["month"].dt.month
    ref = table["month"].dt.year <= reference_end_year
    for column, name in (("tmax_month", "tmax_anom"), ("ndvi", "ndvi_anom")):
        if column in table:
            clim = table[ref].groupby(["district_id", cal[ref]])[column].mean()
            keyed = pd.Series(
                list(zip(table["district_id"], cal, strict=True)), index=table.index
            ).map(clim.to_dict())
            table[name] = table[column] - keyed
    grouped = table.groupby("district_id", sort=False)
    table["spi1_lag1"] = grouped["spi1"].shift(1)
    table["spi3_lag1"] = grouped["spi3"].shift(1)
    table["spi3_lag2"] = grouped["spi3"].shift(2)
    table["month_sin"] = np.sin(2 * np.pi * cal / 12)
    table["month_cos"] = np.cos(2 * np.pi * cal / 12)
    next_spi3 = grouped["spi3"].shift(-1)
    consecutive = grouped["month"].shift(-1) == table["month"] + pd.offsets.MonthBegin(1)
    table["target"] = np.where(
        next_spi3.notna() & consecutive, L.drought_label(next_spi3.fillna(0)), np.nan
    )
    table["persistence"] = L.drought_label(table["spi3"].fillna(0)).astype(float)
    table.loc[table["spi3"].isna(), "persistence"] = np.nan
    return table
