"""Standardised Precipitation Index (SPI) with a gamma fit per district and calendar month.

SPI-k = Phi^-1( H(x) ) where x is the k-month rainfall total ending in a given month, H is the mixed
distribution ``q + (1 - q) * Gamma(x)`` (q = probability of a zero total, gamma fitted to the non-zero
totals with location fixed at 0). Zero totals are given the plotting-position probability
``(zeros + 1) / (2 (n + 1))`` of the zero mass so SPI stays finite.

Parameters are fitted on a **reference period only** (the training years) and applied to every year, so
SPI of a validation or test month never uses a later distribution (docs/DECISIONS.md D-018).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

LOG = logging.getLogger(__name__)

MIN_NONZERO = 8  # fewer non-zero totals than this: fall back to an empirical (rank) transform


@dataclass(frozen=True)
class GammaFit:
    """Distribution of one (district, calendar month, scale) combination."""

    shape: float
    scale: float
    zero_share: float
    n: int
    zeros: int
    empirical: tuple[float, ...] | None = None  # sorted totals if too few non-zero values


def fit_gamma(totals: np.ndarray) -> GammaFit:
    """Fit the mixed gamma distribution to reference-period totals (mm)."""
    totals = np.asarray(totals, float)
    totals = totals[np.isfinite(totals)]
    n = len(totals)
    nonzero = totals[totals > 0]
    zeros = n - len(nonzero)
    if len(nonzero) < MIN_NONZERO:
        return GammaFit(np.nan, np.nan, zeros / max(n, 1), n, zeros, tuple(np.sort(totals)))
    shape, _, scale = stats.gamma.fit(nonzero, floc=0)
    return GammaFit(float(shape), float(scale), zeros / n, n, zeros)


def transform(fit: GammaFit, totals: np.ndarray) -> np.ndarray:
    """SPI values for ``totals`` under ``fit``."""
    x = np.asarray(totals, float)
    if fit.empirical is not None:
        ref = np.asarray(fit.empirical)
        prob = (np.searchsorted(ref, x, side="right") + 0.5) / (len(ref) + 1)
    else:
        prob = np.where(
            x <= 0,
            (fit.zeros + 1) / (2.0 * (fit.n + 1)),
            fit.zero_share
            + (1 - fit.zero_share)
            * stats.gamma.cdf(np.maximum(x, 1e-9), fit.shape, scale=fit.scale),
        )
    eps = 1.0 / (2.0 * (fit.n + 1))
    return stats.norm.ppf(np.clip(prob, eps, 1 - eps))


def monthly_rain(weather: pd.DataFrame) -> pd.DataFrame:
    """Daily weather to monthly rainfall totals: columns district_id, month (Timestamp, month start), rain_mm."""
    frame = weather[["district_id", "date", "rain_mm"]].copy()
    frame["month"] = frame["date"].dt.to_period("M").dt.to_timestamp()
    return frame.groupby(["district_id", "month"], as_index=False)["rain_mm"].sum()


@dataclass(frozen=True)
class MonthReference:
    """Reference-period distribution and mean of the k-month totals of one (district, calendar month, scale)."""

    fit: GammaFit
    mean: float


def _totals(frame: pd.DataFrame, k: int) -> pd.Series:
    return frame["rain_mm"].rolling(k, min_periods=k).sum()


def fit_reference(
    monthly: pd.DataFrame, reference_end_year: int, scales: tuple[int, ...] = (1, 3)
) -> dict[tuple[str, int, int], MonthReference]:
    """Fit the reference distributions once: ``{(district_id, calendar_month, k): MonthReference}``.

    Years <= ``reference_end_year`` only; combinations with fewer than 5 reference totals are omitted.
    """
    references: dict[tuple[str, int, int], MonthReference] = {}
    for district, group in monthly.sort_values("month").groupby("district_id"):
        frame = group.set_index("month")[["rain_mm"]]
        for k in scales:
            total = _totals(frame, k)
            for month in range(1, 13):
                sel = (frame.index.month == month) & total.notna().to_numpy()
                ref = sel & (frame.index.year <= reference_end_year)
                if ref.sum() < 5:
                    continue
                values = total[ref].to_numpy()
                references[(district, month, k)] = MonthReference(
                    fit_gamma(values), float(values.mean())
                )
    return references


def apply_reference(
    frame: pd.DataFrame,
    district: str,
    references: dict[tuple[str, int, int], MonthReference],
    scales: tuple[int, ...] = (1, 3),
) -> pd.DataFrame:
    """SPI-k and rainfall deficit for one district's monthly series (index = month start, column ``rain_mm``).

    The distributions come from ``references`` (fitted on unmodified reference years), so the series being
    transformed may be a *scenario* (e.g. rainfall scaled by 0.7) without changing the reference.
    """
    out = frame.copy()
    for k in scales:
        total = _totals(out, k)
        spi = pd.Series(np.nan, index=out.index)
        deficit = pd.Series(np.nan, index=out.index)
        for month in range(1, 13):
            ref = references.get((district, month, k))
            sel = (out.index.month == month) & total.notna().to_numpy()
            if ref is None or not sel.any():
                continue
            spi[sel] = transform(ref.fit, total[sel].to_numpy())
            deficit[sel] = 100.0 * (total[sel] / ref.mean - 1.0) if ref.mean > 0 else np.nan
        out[f"spi{k}"] = spi
        out[f"deficit{k}_pct"] = deficit
    return out


def spi_table(
    monthly: pd.DataFrame, reference_end_year: int, scales: tuple[int, ...] = (1, 3)
) -> pd.DataFrame:
    """SPI-k and rainfall deficit for every district-month.

    Args:
        monthly: output of :func:`monthly_rain`.
        reference_end_year: last year of the reference period used to fit the gamma distributions.
        scales: SPI accumulation periods in months.

    Returns:
        DataFrame with district_id, month, rain_mm and, per scale k, ``spi{k}`` and ``deficit{k}_pct``
        (percent deviation of the k-month total from its reference mean; negative = deficit).
    """
    references = fit_reference(monthly, reference_end_year, scales)
    out = []
    for district, group in monthly.sort_values("month").groupby("district_id"):
        frame = group.set_index("month")[["rain_mm"]]
        result = apply_reference(frame, district, references, scales)
        result.insert(0, "district_id", district)
        out.append(result.reset_index())
    return pd.concat(out, ignore_index=True)
