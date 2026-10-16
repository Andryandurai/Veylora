"""Drought and flood-risk frequency per district.

Definitions (plan, Phase 3):

* **drought**: season rainfall below 75 percent of the district's long-term mean for that season;
* **flood risk**: season rainfall above 125 percent of the mean **and** at least 3 days above 100 mm.

Rainfall here is a *district-area mean*, so single days above 100 mm are rare; the strict flood-risk
count is therefore small, and we also report the looser "wet season" count (> 125 percent) so the
map is informative. Outputs: ``drought_flood_frequency.png`` and three interactive Folium maps.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd

from analytics import common, maps

LOG = logging.getLogger(__name__)

DROUGHT_RATIO = 0.75
WET_RATIO = 1.25
HEAVY_DAY_MM = 100.0
FLOOD_HEAVY_DAYS = 3

HEAVY_DAYS_SQL = """
SELECT sw.district_id, sw.year, sw.season,
       count(*) FILTER (WHERE w.rain_mm > :heavy) AS heavy_days
FROM v_season_weather sw
JOIN weather_daily w ON w.district_id = sw.district_id AND w.date BETWEEN sw.season_start AND sw.season_end
GROUP BY sw.district_id, sw.year, sw.season
"""


def classify_events(
    frame: pd.DataFrame,
    drought_ratio: float = DROUGHT_RATIO,
    wet_ratio: float = WET_RATIO,
    flood_heavy_days: int = FLOOD_HEAVY_DAYS,
) -> pd.DataFrame:
    """Add ``rain_ratio``, ``drought``, ``wet_season`` and ``flood_risk`` columns.

    Args:
        frame: rows with ``district_id``, ``season``, ``total_rain`` and ``heavy_days``
            (days above 100 mm within the season).
        drought_ratio: drought if ``total_rain < drought_ratio * long-term mean``.
        wet_ratio: wet season if ``total_rain > wet_ratio * long-term mean``.
        flood_heavy_days: flood risk additionally needs at least this many heavy-rain days.

    The long-term mean is the district's mean over all rows of the same season in ``frame``.
    """
    out = frame.copy()
    mean = out.groupby(["district_id", "season"])["total_rain"].transform("mean")
    out["rain_ratio"] = out["total_rain"] / mean
    out["drought"] = out["rain_ratio"] < drought_ratio
    out["wet_season"] = out["rain_ratio"] > wet_ratio
    out["flood_risk"] = out["wet_season"] & (out["heavy_days"] >= flood_heavy_days)
    return out


def event_counts(events: pd.DataFrame) -> pd.DataFrame:
    """Per district and season: number of seasons and counts of each event type."""
    grouped = events.groupby(["district_id", "season"])
    return grouped.agg(
        seasons=("year", "count"),
        drought_events=("drought", "sum"),
        wet_seasons=("wet_season", "sum"),
        flood_risk_events=("flood_risk", "sum"),
    ).reset_index()


def run() -> dict:
    """Compute events, draw the maps and save results."""
    sw = common.season_weather()
    heavy = common.read_sql(HEAVY_DAYS_SQL, {"heavy": HEAVY_DAY_MM})
    events = classify_events(
        sw.merge(heavy, on=["district_id", "year", "season"], how="left").fillna({"heavy_days": 0})
    )
    counts = event_counts(events)
    gdf = common.districts()
    meta = gdf[["district_id", "state", "district"]]
    common.save_table(counts.merge(meta, on="district_id"), "drought_flood_counts_by_district")

    common.apply_mpl_theme()
    panels = [
        ("kharif", "drought_events", "Kharif drought seasons (< 75 % of mean)"),
        ("rabi", "drought_events", "Rabi drought seasons (< 75 % of mean)"),
        ("kharif", "wet_seasons", "Kharif wet seasons (> 125 % of mean)"),
        (
            "kharif",
            "flood_risk_events",
            "Kharif flood-risk seasons (> 125 % and >= 3 days > 100 mm)",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    vmax = float(max(counts[c].max() for _, c, _ in panels))
    for ax, (season, column, title) in zip(axes.ravel(), panels, strict=True):
        data = gdf.merge(
            counts[counts["season"] == season][["district_id", column]], on="district_id"
        )
        maps.mpl_choropleth(
            ax,
            data,
            column,
            vmin=0,
            vmax=max(vmax, 1.0),
            title=title,
            legend_label="number of seasons",
        )
    fig.suptitle(
        "Drought and flood-risk frequency, 1991-2024", x=0.01, ha="left", fontweight="bold"
    )
    common.save_mpl(fig, "drought_flood_frequency")
    for season, column, title in panels[:2] + panels[3:]:
        name = f"{column.split('_')[0]}_{season}_frequency"
        data = gdf.merge(
            counts[counts["season"] == season][["district_id", column]], on="district_id"
        )
        maps.folium_choropleth(
            data,
            column,
            name,
            title,
            vmin=0,
            vmax=max(float(data[column].max()), 1.0),
            legend_caption="number of seasons",
            tooltip_columns=["district", "state", column],
            value_format="{:.0f}",
        )

    names = meta.set_index("district_id")["district"]
    kh = counts[counts["season"] == "kharif"].set_index("district_id")
    rb = counts[counts["season"] == "rabi"].set_index("district_id")
    by_year = (
        events[events["season"] == "kharif"]
        .groupby("year")["drought"]
        .mean()
        .sort_values(ascending=False)
    )
    n_seasons = int(kh["seasons"].max())
    results = {
        "definitions": {
            "drought": f"season rain < {DROUGHT_RATIO:.0%} of the district's long-term mean",
            "flood_risk": f"season rain > {WET_RATIO:.0%} of mean and >= {FLOOD_HEAVY_DAYS} days > {HEAVY_DAY_MM:.0f} mm",
        },
        "n_seasons_per_district": n_seasons,
        "kharif_drought": {
            "mean_events_per_district": float(kh["drought_events"].mean()),
            "mean_frequency": float(kh["drought_events"].sum() / kh["seasons"].sum()),
            "most_drought_prone": [
                {
                    "district": names[i],
                    "events": int(r["drought_events"]),
                    "frequency": float(r["drought_events"] / r["seasons"]),
                }
                for i, r in kh.sort_values("drought_events", ascending=False).head(3).iterrows()
            ],
            "worst_regional_years": [
                {"year": int(y), "share_of_districts_in_drought": float(v)}
                for y, v in by_year.head(3).items()
            ],
        },
        "rabi_drought": {
            "mean_frequency": float(rb["drought_events"].sum() / rb["seasons"].sum()),
            "most_drought_prone": [
                {
                    "district": names[i],
                    "events": int(r["drought_events"]),
                    "frequency": float(r["drought_events"] / r["seasons"]),
                }
                for i, r in rb.sort_values("drought_events", ascending=False).head(3).iterrows()
            ],
        },
        "kharif_wet_seasons_mean_per_district": float(kh["wet_seasons"].mean()),
        "flood_risk_events_total": int(counts["flood_risk_events"].sum()),
        "flood_risk_districts_affected": int(
            (counts.groupby("district_id")["flood_risk_events"].sum() > 0).sum()
        ),
        "max_heavy_days_in_a_season": int(events["heavy_days"].max()),
    }
    common.save_results("drought_flood_frequency", results)
    return results


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
