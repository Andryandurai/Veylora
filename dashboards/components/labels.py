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


MODEL_LABELS = {
    "xgboost": "Gradient-boosted trees (served)",
    "xgboost_unconstrained": "Gradient-boosted trees, no temperature constraint",
    "random_forest": "Random forest",
    "ridge": "Ridge regression",
    "lstm": "Neural network (LSTM)",
    "naive_last_year": "Last year's yield",
    "district_trend": "District trend",
    "prior_mean_3y": "Average of the last 3 years",
    "climatology": "Past averages (climatology)",
    "persistence": "Same as last month (persistence)",
    "rainfed": "No irrigation",
    "farmer_baseline": "Farmer baseline (600 mm)",
    "even_split": "Even split of the budget",
    "optimized": "Optimized schedule",
}

BASELINE_PHRASE = {
    "district_trend": "the district trend",
    "naive_last_year": "last year's yield",
    "prior_mean_3y": "the average of the last 3 years",
}

HEATWAVE_SOURCE = {
    "climatology": "how often heatwaves started at this time of year in past years",
    "lstm": "a neural-network forecast",
}

_FEATURE_FIXED = {
    "prev_year_rel": "Last year's yield",
    "mean_prev3_rel": "Last 3 years' average",
    "trend_pred_rel": "The district's long-term yield trend",
    "log_scale": "The district's typical yield level",
    "other features": "All other factors",
}
_FEATURE_WINDOWED = {
    "rain": "Rainfall",
    "rainy_days": "Number of rainy days",
    "dry_spell": "Longest dry spell",
    "mean_tmax": "Average daily maximum temperature",
    "heat_days": "Days of extreme heat",
    "gdd": "Accumulated warmth (growing degree days)",
    "ndvi": "Vegetation greenness",
    "soil_moisture": "Soil moisture",
}


def model_label(name: str) -> str:
    """The reader-facing name of a model, baseline or irrigation strategy (never the internal id)."""
    return MODEL_LABELS.get(name, name.replace("_", " ").capitalize())


def feature_label(name: str) -> str:
    """A plain description of a model input such as ``rain_d60`` (the internal name is never shown)."""
    import re

    if name in _FEATURE_FIXED:
        return _FEATURE_FIXED[name]
    match = re.fullmatch(r"(?P<base>[a-z_]+?)_d(?P<days>\d+)(?P<suffix>_ratio|_anom)?", name)
    if match and match["base"] in _FEATURE_WINDOWED:
        label = f"{_FEATURE_WINDOWED[match['base']]}, first {match['days']} days"
        if match["suffix"] == "_ratio":
            return f"{label}, compared with normal"
        if match["suffix"] == "_anom":
            return f"{label}, difference from normal"
        return label
    for prefix, template in (
        ("crop_", "Crop: {}"),
        ("season_", "Season: {}"),
        ("cluster_", "Climate group {}"),
    ):
        if name.startswith(prefix):
            return template.format(name.removeprefix(prefix))
    return name.replace("_", " ").capitalize()
