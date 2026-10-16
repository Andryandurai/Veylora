"""Choropleth helpers: a static matplotlib PNG and an interactive Folium HTML from the same data."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import branca.colormap as bcm
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, Normalize, to_hex

from analytics import common

LOG = logging.getLogger(__name__)


def _norm(values: Sequence[float], vmin: float | None, vmax: float | None) -> Normalize:
    arr = np.asarray(values, dtype=float)
    lo = np.nanmin(arr) if vmin is None else vmin
    hi = np.nanmax(arr) if vmax is None else vmax
    if hi <= lo:
        hi = lo + 1.0
    return Normalize(vmin=lo, vmax=hi)


def mpl_choropleth(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    column: str,
    cmap: Colormap = common.CMAP_SEQ,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    legend_label: str | None = None,
    missing_color: str = "#e1e0d9",
) -> None:
    """Draw one choropleth on ``ax`` with a slim colour bar; missing values are drawn grey."""
    norm = _norm(gdf[column].dropna(), vmin, vmax)
    gdf.plot(
        column=column,
        ax=ax,
        cmap=cmap,
        norm=norm,
        edgecolor=common.SURFACE,
        linewidth=0.6,
        missing_kwds={"color": missing_color, "edgecolor": common.SURFACE},
    )
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = ax.figure.colorbar(mappable, ax=ax, fraction=0.04, pad=0.02, shrink=0.8)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=common.INK_MUTED, labelsize=8)
    if legend_label:
        cbar.set_label(legend_label, color=common.INK_SECONDARY, fontsize=9)


def mpl_ordinal_map(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    column: str,
    ordered_labels: Sequence[str],
    title: str | None = None,
) -> None:
    """Map of an ordered class column using the one-hue ordinal ramp, with a labelled legend."""
    colors = dict(zip(ordered_labels, common.ordinal_ramp(len(ordered_labels)), strict=True))
    gdf.assign(_color=gdf[column].map(colors)).plot(
        color=gdf[column].map(colors), ax=ax, edgecolor=common.SURFACE, linewidth=0.6
    )
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in ordered_labels]
    ax.legend(handles, ordered_labels, loc="lower left", fontsize=8, title=None, handlelength=1.2)


def folium_choropleth(
    gdf: gpd.GeoDataFrame,
    column: str,
    name: str,
    title: str,
    cmap: Colormap = common.CMAP_SEQ,
    vmin: float | None = None,
    vmax: float | None = None,
    legend_caption: str | None = None,
    tooltip_columns: Sequence[str] | None = None,
    tiles: str | None = "OpenStreetMap",
    value_format: str = "{:.2f}",
) -> Path:
    """Save an interactive Folium choropleth to reports/figures/<name>.html and return the path."""
    norm = _norm(gdf[column].dropna(), vmin, vmax)
    colormap = bcm.LinearColormap(
        [to_hex(cmap(x)) for x in np.linspace(0, 1, 7)], vmin=norm.vmin, vmax=norm.vmax
    )
    colormap.caption = legend_caption or column
    minx, miny, maxx, maxy = gdf.total_bounds
    centre = [float((miny + maxy) / 2), float((minx + maxx) / 2)]
    fmap = folium.Map(location=centre, zoom_start=6, tiles=tiles, control_scale=True)
    fmap.get_root().html.add_child(
        folium.Element(
            f'<div style="position:fixed;top:10px;left:55px;z-index:9999;background:{common.SURFACE};'
            f"padding:6px 12px;border-radius:6px;font:600 14px system-ui,sans-serif;color:{common.INK};"
            f'border:1px solid {common.AXIS}">{title}</div>'
        )
    )
    cols = list(tooltip_columns or ["district_id", column])
    data = gdf.copy()
    data[column + "__fmt"] = [
        "n/a" if np.isnan(v) else value_format.format(v) for v in data[column].astype(float)
    ]

    def style(feature: dict) -> dict:
        value = feature["properties"][column]
        missing = value is None or (isinstance(value, float) and np.isnan(value))
        return {
            "fillColor": "#e1e0d9" if missing else colormap(value),
            "color": common.SURFACE,
            "weight": 1.0,
            "fillOpacity": 0.85,
        }

    keep = [*dict.fromkeys([*cols, column, column + "__fmt"])]
    folium.GeoJson(
        data[[*keep, "geometry"]].to_json(),
        style_function=style,
        highlight_function=lambda _f: {"weight": 3, "color": common.INK},
        tooltip=folium.GeoJsonTooltip(
            fields=[c for c in keep if c != column],
            aliases=[c.replace("__fmt", "").replace("_", " ") for c in keep if c != column],
            sticky=True,
        ),
    ).add_to(fmap)
    colormap.add_to(fmap)
    path = common.figures_dir() / f"{name}.html"
    fmap.save(str(path))
    LOG.info("wrote %s", path.name)
    return path
