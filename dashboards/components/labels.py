"""Plain-language names for internal tables and data groups, used wherever a visitor can read them."""

from __future__ import annotations

TABLE_LABELS = {
    "satellite_16d": "satellite vegetation and soil-moisture data",
    "weather_daily": "weather data",
    "crops_annual": "crop yield data",
    "districts": "district boundaries",
}


def table_label(name: str) -> str:
    """The reader-facing name of an internal table (the internal name itself is never shown)."""
    return TABLE_LABELS.get(name, name.replace("_", " "))
