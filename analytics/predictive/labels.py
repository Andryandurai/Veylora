"""Event labels for the early-warning models: heatwaves (IMD plains criterion) and meteorological drought.

**Heatwave** (IMD, plains): a day qualifies if Tmax >= 40 C and the departure from the district's daily
climatological normal is >= 4.5 C (severe: >= 6.5 C); a heatwave needs at least 2 consecutive qualifying
days. Normals are the district's mean Tmax per calendar day over a **reference period**, smoothed with a
circular 30-day moving average. The plan asks for 1991-2020; we use the training years (1991-2014) so
that labels and features never depend on the validation/test years (docs/DECISIONS.md D-018).

**Drought**: SPI-3 <= -1.0 (see :mod:`analytics.predictive.spi`).

The forecast target for heatwaves is *onset* within the next h days, h in (3, 7, 14): an onset is the
first day of a heatwave spell.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

TMAX_MIN_C = 40.0
DEPARTURE_C = 4.5
SEVERE_DEPARTURE_C = 6.5
MIN_SPELL_DAYS = 2
SMOOTH_DAYS = 30
HORIZONS = (3, 7, 14)
DROUGHT_SPI = -1.0


def smooth_circular(values: np.ndarray, window: int = SMOOTH_DAYS) -> np.ndarray:
    """Circular moving average (the year wraps around), used to smooth a 366-day climatology."""
    n = len(values)
    padded = np.concatenate([values[-window:], values, values[:window]])
    kernel = np.ones(window) / window
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[window : window + n]


def daily_normals(
    weather: pd.DataFrame, reference_end_year: int, first_year: int = 1991
) -> pd.DataFrame:
    """Smoothed daily normal Tmax per district and calendar day (month-day slot, 366 values).

    Returns:
        DataFrame with columns district_id, slot (0..365) and tmax_normal.
    """
    frame = weather[
        (weather["date"].dt.year >= first_year) & (weather["date"].dt.year <= reference_end_year)
    ].copy()
    frame["slot"] = _slot(frame["date"])
    out = []
    for district, group in frame.groupby("district_id"):
        daily = group.groupby("slot")["tmax_c"].mean().reindex(range(366))
        daily = daily.interpolate(limit_direction="both").to_numpy()
        out.append(
            pd.DataFrame(
                {
                    "district_id": district,
                    "slot": np.arange(366),
                    "tmax_normal": smooth_circular(daily),
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def _slot(dates: pd.Series) -> np.ndarray:
    """Month-day slot 0..365 that treats Feb 29 as its own slot, so leap years do not shift the calendar."""
    month, day = dates.dt.month.to_numpy(), dates.dt.day.to_numpy()
    cumulative = np.array(
        [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    )  # leap-year month starts
    return cumulative[month - 1] + day - 1


def qualifying_days(
    tmax: np.ndarray, normal: np.ndarray, departure: float = DEPARTURE_C
) -> np.ndarray:
    """Boolean array: Tmax >= 40 C and departure from normal >= ``departure``."""
    return (tmax >= TMAX_MIN_C) & ((tmax - normal) >= departure)


def spell_mask(qualifying: np.ndarray, min_days: int = MIN_SPELL_DAYS) -> np.ndarray:
    """Days that belong to a run of at least ``min_days`` consecutive qualifying days."""
    mask = np.zeros(len(qualifying), dtype=bool)
    n = len(qualifying)
    i = 0
    while i < n:
        if qualifying[i]:
            j = i
            while j < n and qualifying[j]:
                j += 1
            if j - i >= min_days:
                mask[i:j] = True
            i = j
        else:
            i += 1
    return mask


def onset_days(heatwave: np.ndarray) -> np.ndarray:
    """First day of every heatwave spell."""
    previous = np.concatenate([[False], heatwave[:-1]])
    return heatwave & ~previous


def onset_within(onset: np.ndarray, horizon: int) -> np.ndarray:
    """y[t] = 1 if an onset occurs on any of the days t+1 .. t+horizon.

    Near the end of the record the window is cut off and only its observed part is searched, so a day can be
    labelled positive but never wrongly negative *because of* an onset that is known to lie inside the record;
    a truncated window without an observed onset is simply False (unknown). Forecast samples are restricted
    to March-June, far from the end of a calendar-year record, so this never affects training or evaluation.
    """
    n = len(onset)
    csum = np.concatenate([[0], np.cumsum(onset.astype(np.int64))])
    idx = np.arange(n)
    hi = np.minimum(idx + horizon + 1, n)
    lo = np.minimum(idx + 1, n)
    return (csum[hi] - csum[lo]) > 0


def heatwave_frame(weather: pd.DataFrame, normals: pd.DataFrame) -> pd.DataFrame:
    """Per district-day heatwave flags: qualifying, heatwave, severe, onset and onset-within-h labels.

    Columns added to ``weather[["district_id","date","tmax_c"]]``: ``tmax_normal``, ``departure``,
    ``qualifying``, ``heatwave``, ``severe``, ``onset`` and ``y3``, ``y7``, ``y14``.
    """
    frame = (
        weather[["district_id", "date", "tmax_c"]]
        .sort_values(["district_id", "date"])
        .reset_index(drop=True)
    )
    frame["slot"] = _slot(frame["date"])
    frame = frame.merge(normals, on=["district_id", "slot"], how="left")
    frame["departure"] = frame["tmax_c"] - frame["tmax_normal"]
    parts = []
    for _, group in frame.groupby("district_id", sort=False):
        tmax, normal = group["tmax_c"].to_numpy(float), group["tmax_normal"].to_numpy(float)
        qualifying = qualifying_days(tmax, normal)
        heatwave = spell_mask(qualifying)
        severe = spell_mask(qualifying_days(tmax, normal, SEVERE_DEPARTURE_C))
        onset = onset_days(heatwave)
        data = {"qualifying": qualifying, "heatwave": heatwave, "severe": severe, "onset": onset}
        for h in HORIZONS:
            data[f"y{h}"] = onset_within(onset, h)
        parts.append(pd.DataFrame(data, index=group.index))
    return frame.join(pd.concat(parts)).drop(columns="slot")


def drought_label(spi3: pd.Series | np.ndarray) -> np.ndarray:
    """True where SPI-3 <= -1.0."""
    return np.asarray(spi3, float) <= DROUGHT_SPI
