"""SYNTHETIC crop yields (fallback when neither ICRISAT nor an APY CSV is available).

Yield = crop baseline x (1 + trend) x rainfall response x vegetation response x noise, with
season rainfall and mean NDVI taken from the supplied weather/satellite tables. This lets the
downstream models run end-to-end, but the relationships are *assumed by us*, so every row is
``is_synthetic=True`` and results must be labelled as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import get_seed, load_settings

BASE_YIELD_T_HA = {"rice": 2.9, "wheat": 2.4, "cotton": 0.35, "sugarcane": 8.0}
TREND_PER_YEAR = 0.012
SEASON_RAIN_REF_MM = {"kharif": 600.0, "rabi": 120.0}


def _season_window(year: int, season: str, seasons: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    cfg = seasons[season]
    start = pd.Timestamp(year=year, month=cfg["start_month"], day=1)
    end_year = year + 1 if cfg["end_month"] < cfg["start_month"] else year
    end = pd.Timestamp(year=end_year, month=cfg["end_month"], day=1) + pd.offsets.MonthEnd(0)
    return start, end


def generate_crops(
    districts: pd.DataFrame,
    weather: pd.DataFrame,
    satellite: pd.DataFrame | None,
    start_year: int,
    end_year: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """Synthetic crops_annual for every district, crop and year in the window."""
    settings = load_settings()
    seed = get_seed() if seed is None else seed
    rng = np.random.default_rng(seed + 7)
    crop_season = settings["crop_season"]
    seasons = settings["seasons"]
    daily = weather.set_index(["district_id", "date"]).sort_index()
    sat = (
        satellite.set_index(["district_id", "date"]).sort_index() if satellite is not None else None
    )
    rows = []
    for district in districts.itertuples():
        scale = float(np.exp(rng.normal(0, 0.15)))
        for crop in settings["crops"]:
            season = crop_season[crop]
            grown_here = (
                rng.random() < {"rice": 0.95, "wheat": 0.4, "cotton": 0.7, "sugarcane": 0.7}[crop]
            )
            if not grown_here:
                continue
            area_base = float(np.exp(rng.normal(10.8, 0.9)))
            for year in range(start_year, end_year + 1):
                start, end = _season_window(year, season, seasons)
                try:
                    window = daily.loc[district.district_id].loc[start:end, "rain_mm"]
                except KeyError:
                    continue
                if window.empty:
                    continue
                rain_ratio = float(window.sum()) / SEASON_RAIN_REF_MM[season]
                rain_effect = 1.0 + 0.25 * np.tanh(rain_ratio - 1.0)
                veg_effect = 1.0
                if sat is not None:
                    ndvi = sat.loc[district.district_id].loc[start:end, "ndvi"]
                    if not ndvi.empty:
                        veg_effect = 1.0 + 0.4 * (float(ndvi.mean()) - 0.5)
                trend = 1.0 + TREND_PER_YEAR * (year - start_year)
                yld = (
                    BASE_YIELD_T_HA[crop]
                    * scale
                    * trend
                    * rain_effect
                    * veg_effect
                    * np.exp(rng.normal(0, 0.06))
                )
                area = area_base * np.exp(rng.normal(0, 0.05))
                rows.append((district.district_id, year, season, crop, area, area * yld))
    frame = pd.DataFrame(
        rows, columns=["district_id", "year", "season", "crop", "area_ha", "production_t"]
    )
    frame["yield_t_per_ha"] = frame["production_t"] / frame["area_ha"]
    frame["year"] = frame["year"].astype("int16")
    frame["yield_flag"] = False
    frame["is_synthetic"] = True
    return frame.sort_values(["district_id", "crop", "season", "year"], ignore_index=True)
