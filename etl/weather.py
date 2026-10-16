"""Assemble ``weather_daily`` from IMD grids, NASA POWER and (only if unavoidable) synthetic data.

Precedence per variable and district-day:

* rain_mm, tmax_c, tmin_c : IMD gridded district mean  >  NASA POWER  >  synthetic
* rh_pct, wind_ms, solar_mj : NASA POWER  >  synthetic

``is_synthetic`` is True for a row if *any* of its values came from the synthetic generator. The
share contributed by each source is returned as a provenance dict and stored next to the data so
that ``docs/DATA_QUALITY.md`` can state exactly what is real.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from etl.synthetic.weather import generate_weather

LOG = logging.getLogger(__name__)

WEATHER_COLUMNS = [
    "district_id",
    "date",
    "rain_mm",
    "tmax_c",
    "tmin_c",
    "rh_pct",
    "wind_ms",
    "solar_mj",
    "is_synthetic",
]
VALUE_COLUMNS = ["rain_mm", "tmax_c", "tmin_c", "rh_pct", "wind_ms", "solar_mj"]
POWER_FALLBACK = {"rain_mm": "p_rain_mm", "tmax_c": "p_tmax_c", "tmin_c": "p_tmin_c"}
POWER_ONLY = ("rh_pct", "wind_ms", "solar_mj")


def assemble_weather(
    districts: pd.DataFrame,
    start_year: int,
    end_year: int,
    imd_rain: pd.DataFrame | None,
    imd_temp: pd.DataFrame | None,
    power: pd.DataFrame | None,
    allow_synthetic: bool = True,
    seed: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Merge the sources into the ``weather_daily`` contract.

    Args:
        districts: districts table (``district_id``, ``centroid_lat``, ``centroid_lon``).
        start_year, end_year: inclusive calendar-year window (a complete daily grid is produced).
        imd_rain: columns district_id, date, rain_mm (or None / empty).
        imd_temp: columns district_id, date, tmax_c, tmin_c (or None / empty).
        power: NASA POWER frame from :func:`etl.nasa_power.district_daily` (or None / empty).
        allow_synthetic: fill what is still missing with flagged synthetic values.
        seed: seed for the synthetic generator.

    Returns:
        (weather_daily frame, provenance dict of source shares per variable)
    """
    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    grid = pd.MultiIndex.from_product(
        [districts["district_id"].tolist(), dates], names=["district_id", "date"]
    )
    frame = pd.DataFrame(index=grid).reset_index()
    for source in (imd_rain, imd_temp, power):
        if source is not None and len(source):
            frame = frame.merge(source, on=["district_id", "date"], how="left")
    for column in (*VALUE_COLUMNS, *POWER_FALLBACK.values()):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = frame[column].astype("float32")

    provenance: dict[str, dict[str, float]] = {}
    for target, fallback in POWER_FALLBACK.items():
        from_imd = frame[target].notna()
        use_power = ~from_imd & frame[fallback].notna()
        frame.loc[use_power, target] = frame.loc[use_power, fallback]
        provenance[target] = {"imd": float(from_imd.mean()), "nasa_power": float(use_power.mean())}
    for target in POWER_ONLY:
        provenance[target] = {"nasa_power": float(frame[target].notna().mean())}

    synthetic_row = np.zeros(len(frame), dtype=bool)
    missing_any = frame[VALUE_COLUMNS].isna().any(axis=1)
    if missing_any.any():
        if allow_synthetic:
            LOG.warning(
                "%d district-days lack real data; filling with SYNTHETIC values",
                int(missing_any.sum()),
            )
            synthetic = generate_weather(districts, start_year, end_year, seed)
            synthetic = synthetic.set_index(["district_id", "date"]).reindex(grid)
            for column in VALUE_COLUMNS:
                gap = frame[column].isna().to_numpy()
                if gap.any():
                    frame.loc[gap, column] = synthetic[column].to_numpy()[gap]
                    provenance.setdefault(column, {})["synthetic"] = float(gap.mean())
            synthetic_row = missing_any.to_numpy()
        else:
            LOG.warning(
                "%d district-days lack real data and synthetic fallback is disabled",
                int(missing_any.sum()),
            )
    for column in VALUE_COLUMNS:
        provenance[column].setdefault("synthetic", 0.0)

    # Mixing IMD Tmax with POWER Tmin (or the reverse) can leave Tmin above Tmax on a few days.
    swapped = frame["tmin_c"] > frame["tmax_c"]
    provenance["tmin_clipped_to_tmax_days"] = {"count": float(swapped.sum())}
    frame.loc[swapped, "tmin_c"] = frame.loc[swapped, "tmax_c"]

    frame["is_synthetic"] = synthetic_row
    result = frame[WEATHER_COLUMNS].sort_values(["district_id", "date"], ignore_index=True)
    return result, provenance
