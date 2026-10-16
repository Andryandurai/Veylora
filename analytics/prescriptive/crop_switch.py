"""Crop-switch decision support: rank candidate crops by expected net return.

**This is a decision-support estimate, not financial advice.** Prices and costs in ``config/economics.yaml`` are
approximate and unverified, yields are model forecasts with wide intervals, and the model ignores risk attitudes,
inputs, labour, rotations, water rights and market access.

Expected net return per hectare = price x marketed-unit yield x interval midpoint - cost, where the *interval
midpoint* is the middle of the 80 percent prediction interval of ``predict_yield``. A pessimistic ("lower-bound")
return uses the lower end of the interval. Yields are converted to the unit the price refers to (milled rice to
paddy, lint to seed cotton) using the factors in the economics file; sugarcane is excluded by default because its
yield unit in the source data is unclear.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from analytics.prescriptive.params import load_economics
from config import load_settings

LOG = logging.getLogger(__name__)

QUINTALS_PER_TONNE = 10.0
DISCLAIMER = "Decision-support estimate based on model forecasts and illustrative prices; not financial advice."


def candidate_crops(season: str, include_sugarcane: bool | None = None) -> list[str]:
    """Crops grown in ``season`` according to settings.yaml ``crop_season`` (sugarcane only if enabled)."""
    econ = load_economics()
    include = (
        econ.get("include_sugarcane", False) if include_sugarcane is None else include_sugarcane
    )
    crops = [c for c, s in load_settings()["crop_season"].items() if s == season]
    return [c for c in crops if include or c != "sugarcane"]


def revenue_per_ha(yield_t_ha: float, crop: str, econ: dict[str, Any] | None = None) -> float:
    """Revenue in INR/ha for a yield in the data's unit (t/ha), after unit conversion to the priced commodity."""
    entry = (econ or load_economics())["crops"][crop]
    marketed_t = yield_t_ha * float(entry["marketed_unit_per_yield_unit"])
    return marketed_t * QUINTALS_PER_TONNE * float(entry["price_per_quintal"])


def net_return_per_ha(yield_t_ha: float, crop: str, econ: dict[str, Any] | None = None) -> float:
    """Revenue minus cost of cultivation, INR/ha."""
    econ = econ or load_economics()
    return revenue_per_ha(yield_t_ha, crop, econ) - float(econ["crops"][crop]["cost_per_ha"])


def rank_crops(
    predictions: dict[str, dict[str, float]], econ: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Rank crops by expected net return (interval midpoint) and by the lower-bound scenario.

    Args:
        predictions: ``{crop: {"yield_t_ha", "lo", "hi"}}`` (the output of ``predict_yield`` per crop).

    Returns:
        DataFrame sorted by ``net_mid`` (best first) with revenue, cost, both net returns and both ranks;
        ``rank_mid != rank_lower`` shows crops whose ranking depends on the pessimistic scenario.
    """
    econ = econ or load_economics()
    rows = []
    for crop, pred in predictions.items():
        mid = 0.5 * (pred["lo"] + pred["hi"])
        rows.append(
            {
                "crop": crop,
                "yield_point_t_ha": pred["yield_t_ha"],
                "yield_lo_t_ha": pred["lo"],
                "yield_mid_t_ha": mid,
                "yield_hi_t_ha": pred["hi"],
                "revenue_mid_inr_ha": revenue_per_ha(mid, crop, econ),
                "cost_inr_ha": float(econ["crops"][crop]["cost_per_ha"]),
                "net_mid_inr_ha": net_return_per_ha(mid, crop, econ),
                "net_lower_inr_ha": net_return_per_ha(pred["lo"], crop, econ),
            }
        )
    frame = pd.DataFrame(rows).sort_values("net_mid_inr_ha", ascending=False, ignore_index=True)
    frame["rank_mid"] = frame["net_mid_inr_ha"].rank(ascending=False, method="min").astype(int)
    frame["rank_lower"] = frame["net_lower_inr_ha"].rank(ascending=False, method="min").astype(int)
    frame.attrs["disclaimer"] = DISCLAIMER
    return frame


def recommend(
    district_id: str,
    season: str,
    overrides: dict[str, float] | None = None,
    year: int | None = None,
    include_sugarcane: bool | None = None,
) -> pd.DataFrame:
    """Forecast every candidate crop with ``predict_yield`` and rank them (crops without yield history are skipped).

    Raises:
        ValueError: if no candidate crop has a forecast for this district and season.
    """
    from analytics.predictive.predict import predict_yield

    predictions, skipped = {}, []
    for crop in candidate_crops(season, include_sugarcane):
        try:
            predictions[crop] = predict_yield(district_id, crop, season, overrides, year=year)
        except ValueError as exc:
            skipped.append(f"{crop}: {exc}")
    if not predictions:
        raise ValueError(f"no forecastable crop for {district_id} in {season}: {skipped}")
    frame = rank_crops(predictions)
    frame.attrs["skipped"] = skipped
    return frame
