"""Rainfall variability: coefficient of variation of season rainfall per district.

Outputs (reports/figures): ``rainfall_anomaly_heatmap.{png,html}`` (districts x years, % of the
district's own mean, diverging around normal) and ``rainfall_cv_kharif.{png,html}`` (choropleth).
Numbers go to ``reports/analytics/rainfall_variability.json``.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from analytics import common, maps

LOG = logging.getLogger(__name__)

ANOMALY_CLIP = 60.0  # colour scale limit, % of normal


def season_stats(sw: pd.DataFrame) -> pd.DataFrame:
    """Per district and season: mean, standard deviation and coefficient of variation of total rain."""
    grouped = sw.groupby(["district_id", "season"])["total_rain"]
    out = grouped.agg(n_years="count", mean_rain_mm="mean", sd_rain_mm="std").reset_index()
    out["cv"] = out["sd_rain_mm"] / out["mean_rain_mm"]
    return out


def anomaly_matrix(sw: pd.DataFrame, season: str) -> pd.DataFrame:
    """Districts x years matrix of season rainfall as % deviation from each district's own mean."""
    frame = sw[sw["season"] == season]
    mean = frame.groupby("district_id")["total_rain"].transform("mean")
    frame = frame.assign(anomaly_pct=100.0 * (frame["total_rain"] / mean - 1.0))
    return frame.pivot_table(
        index="district_id", columns="year", values="anomaly_pct", aggfunc="first"
    )


def _row_order(gdf_meta: pd.DataFrame, stats_k: pd.DataFrame) -> list[str]:
    """Districts grouped by state (settings order) and sorted by mean kharif rain within a state."""
    order = {s: i for i, s in enumerate(common.load_settings()["states"])}
    meta = gdf_meta.merge(stats_k[["district_id", "mean_rain_mm"]], on="district_id")
    meta["state_rank"] = meta["state"].map(order)
    return meta.sort_values(["state_rank", "mean_rain_mm"])["district_id"].tolist()


def heatmap_png(matrix: pd.DataFrame, meta: pd.DataFrame, season: str, name: str) -> None:
    """Static heatmap with state separators."""
    common.apply_mpl_theme()
    labels = meta.set_index("district_id").loc[matrix.index, "district"].tolist()
    states = meta.set_index("district_id").loc[matrix.index, "state"].tolist()
    fig, ax = plt.subplots(figsize=(10, max(6, 0.2 * len(matrix))))
    image = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        cmap=common.CMAP_DIV,
        vmin=-ANOMALY_CLIP,
        vmax=ANOMALY_CLIP,
    )
    ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
    years = matrix.columns.tolist()
    ax.set_xticks(range(0, len(years), 3), years[::3])
    ax.grid(False)
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            ax.axhline(i - 0.5, color=common.INK, linewidth=1.0)
    first = {s: states.index(s) for s in dict.fromkeys(states)}
    last = {s: len(states) - 1 - states[::-1].index(s) for s in first}
    for state in first:
        ax.text(
            len(years) - 0.3,
            (first[state] + last[state]) / 2,
            state,
            va="center",
            ha="left",
            fontsize=8,
            color=common.INK_SECONDARY,
            clip_on=False,
        )
    ax.set_title(f"{season.capitalize()} rainfall, % deviation from each district's mean")
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.12)
    cbar.set_label("% of district mean (red = deficit, blue = excess)", color=common.INK_SECONDARY)
    cbar.outline.set_visible(False)
    common.save_mpl(fig, name)


def heatmap_html(matrix: pd.DataFrame, meta: pd.DataFrame, season: str, name: str) -> None:
    """Interactive heatmap with per-cell hover."""
    labels = meta.set_index("district_id").loc[matrix.index, "district"].tolist()
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns.tolist(),
            y=labels,
            zmin=-ANOMALY_CLIP,
            zmax=ANOMALY_CLIP,
            zmid=0,
            colorscale=[
                [0, common.DIVERGING_LOW],
                [0.5, common.DIVERGING_MID],
                [1, common.DIVERGING_HIGH],
            ],
            colorbar={"title": "% of mean"},
            hovertemplate="%{y}, %{x}<br>%{z:.0f}% of the district mean<extra></extra>",
        )
    )
    common.plotly_layout(
        fig, f"{season.capitalize()} rainfall, % deviation from each district's mean", height=900
    )
    fig.update_yaxes(autorange="reversed", tickfont={"size": 9})
    common.save_plotly(fig, name)


def run() -> dict:
    """Compute, draw and save everything; returns the headline numbers."""
    sw = common.season_weather()
    gdf = common.districts()
    meta = gdf[["district_id", "state", "district"]]
    table = season_stats(sw).merge(meta, on="district_id")
    common.save_table(
        table[["district_id", "state", "season", "n_years", "mean_rain_mm", "sd_rain_mm", "cv"]],
        "rainfall_variability_by_district",
    )
    kharif = table[table["season"] == "kharif"].set_index("district_id")
    rabi = table[table["season"] == "rabi"].set_index("district_id")

    matrix = anomaly_matrix(sw, "kharif")
    ordered = [d for d in _row_order(meta, kharif.reset_index()) if d in matrix.index]
    matrix = matrix.loc[ordered]
    heatmap_png(matrix, meta, "kharif", "rainfall_anomaly_heatmap")
    heatmap_html(matrix, meta, "kharif", "rainfall_anomaly_heatmap")

    plot = gdf.merge(kharif[["mean_rain_mm", "cv"]].reset_index(), on="district_id")
    common.apply_mpl_theme()
    fig, ax = plt.subplots(figsize=(6.5, 7))
    maps.mpl_choropleth(
        ax,
        plot,
        "cv",
        title="Kharif rainfall variability (CV)",
        legend_label="coefficient of variation",
    )
    common.save_mpl(fig, "rainfall_cv_kharif")
    maps.folium_choropleth(
        plot,
        "cv",
        "rainfall_cv_kharif",
        "Kharif rainfall variability (CV of season total)",
        legend_caption="CV of kharif rainfall",
        tooltip_columns=["district", "state", "mean_rain_mm", "cv"],
        value_format="{:.3f}",
    )

    rho, p_value = stats.spearmanr(kharif["mean_rain_mm"], kharif["cv"])
    region_year = matrix.mean(axis=0)
    share_below = (matrix < 0).mean(axis=0)
    by_state = table[table["season"] == "kharif"].groupby("state")["cv"].mean()
    top = kharif.sort_values("cv", ascending=False).head(5)
    names = meta.set_index("district_id")["district"]
    results = {
        "years": [int(matrix.columns.min()), int(matrix.columns.max())],
        "n_districts": int(len(kharif)),
        "kharif_cv_mean": float(kharif["cv"].mean()),
        "kharif_cv_median": float(kharif["cv"].median()),
        "rabi_cv_mean": float(rabi["cv"].mean()),
        "kharif_cv_by_state": {k: float(v) for k, v in by_state.items()},
        "top_cv_districts": [
            {
                "district_id": i,
                "district": names[i],
                "cv": float(r["cv"]),
                "mean_rain_mm": float(r["mean_rain_mm"]),
            }
            for i, r in top.iterrows()
        ],
        "lowest_cv_district": {
            "district_id": kharif["cv"].idxmin(),
            "district": names[kharif["cv"].idxmin()],
            "cv": float(kharif["cv"].min()),
        },
        "spearman_meanrain_vs_cv": {"rho": float(rho), "p": float(p_value)},
        "driest_regional_years": [
            {
                "year": int(y),
                "mean_anomaly_pct": float(v),
                "share_districts_below_normal": float(share_below[y]),
            }
            for y, v in region_year.sort_values().head(3).items()
        ],
        "wettest_regional_years": [
            {"year": int(y), "mean_anomaly_pct": float(v)}
            for y, v in region_year.sort_values().tail(3)[::-1].items()
        ],
    }
    common.save_results("rainfall_variability", results)
    return results


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
