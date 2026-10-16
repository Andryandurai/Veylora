"""IMD gridded daily Tmax / Tmin (1 degree) aggregated to district means.

Source: IMD gridded daily maximum and minimum temperature through ``imdlib``. The product is
1 x 1 degree, so neighbouring districts share cells and differ only through their overlap weights;
that coarse resolution is a documented limitation (docs/DATA_QUALITY.md).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import pandas as pd

from etl.imd_rainfall import aggregate_variable

LOG = logging.getLogger(__name__)


def district_temperature(
    districts: gpd.GeoDataFrame, years: Sequence[int], raw_dir: Path | None = None
) -> pd.DataFrame:
    """District daily Tmax/Tmin (deg C) from IMD, columns district_id, date, tmax_c, tmin_c.

    Days where only one of the two variables exists keep the other as NaN.
    """
    tmax = aggregate_variable(districts, "tmax", years, "tmax_c", raw_dir)
    tmin = aggregate_variable(districts, "tmin", years, "tmin_c", raw_dir)
    merged = tmax.merge(tmin, on=["district_id", "date"], how="outer")
    return merged.sort_values(["district_id", "date"], ignore_index=True)
