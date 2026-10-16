"""Shared helpers for the analytics packages: database access, plotting theme, saving outputs.

All Phase 3 statistics are computed from PostgreSQL (``v_training_table`` / ``v_season_weather``
from Phase 2). Every module writes its numbers to ``reports/analytics/<name>.json`` so that
``analytics/make_findings.py`` can build ``docs/FINDINGS.md`` without any hard-coded values.

Colour follows the validated reference palette (docs: dataviz skill): categorical slots in fixed
order (max three on maps), one-hue blue ramp for magnitude, blue<->red with a grey midpoint for
anomalies, ink/grid tokens for chrome.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")  # headless: figures are written to files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from config import get_path, load_settings  # noqa: E402
from database import db_available, get_engine  # noqa: E402

LOG = logging.getLogger(__name__)

# --------------------------------------------------------------------------- palette tokens (light)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = {
    100: "#cde2fb",
    150: "#b7d3f6",
    200: "#9ec5f4",
    250: "#86b6ef",
    300: "#6da7ec",
    350: "#5598e7",
    400: "#3987e5",
    450: "#2a78d6",
    500: "#256abf",
    550: "#1c5cab",
    600: "#184f95",
    650: "#104281",
    700: "#0d366b",
}
DIVERGING_LOW = "#e34948"  # red pole: deficit / below normal
DIVERGING_MID = "#f0efec"
DIVERGING_HIGH = "#2a78d6"  # blue pole: excess / above normal
FONT_STACK = ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]

CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", [SEQ_BLUE[k] for k in sorted(SEQ_BLUE)])
CMAP_DIV = LinearSegmentedColormap.from_list(
    "div_red_blue", [DIVERGING_LOW, DIVERGING_MID, DIVERGING_HIGH]
)


def state_colors(states: list[str]) -> dict[str, str]:
    """Fixed categorical colour per state (order follows settings.yaml, never re-ranked by data)."""
    order = load_settings()["states"]
    return {s: SERIES[order.index(s) % len(SERIES)] for s in states}


def crop_colors() -> dict[str, str]:
    """Fixed categorical colour per crop (order follows settings.yaml)."""
    return {c: SERIES[i % len(SERIES)] for i, c in enumerate(load_settings()["crops"])}


def ordinal_ramp(n: int) -> list[str]:
    """``n`` ordered colours from the blue ramp, never lighter than step 250 (ordinal rule)."""
    steps = [250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    picks = np.linspace(0, len(steps) - 1, n).round().astype(int)
    return [SEQ_BLUE[steps[i]] for i in picks]


def apply_mpl_theme() -> None:
    """Apply the project matplotlib theme (thin marks, recessive grid, ink tokens)."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "text.color": INK,
            "font.family": FONT_STACK,
            "font.size": 10,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "figure.dpi": 110,
            "savefig.dpi": 150,
        }
    )


# --------------------------------------------------------------------------- paths and outputs
def figures_dir() -> Path:
    """reports/figures (created on demand)."""
    return get_path("figures")


def analytics_dir() -> Path:
    """reports/analytics: JSON results and CSV/markdown tables."""
    path = get_path("reports") / "analytics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_mpl(fig: plt.Figure, name: str) -> Path:
    """Save a matplotlib figure as reports/figures/<name>.png and close it."""
    path = figures_dir() / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", path.name)
    return path


def plotly_layout(fig: Any, title: str, height: int | None = None) -> Any:
    """Apply the project Plotly look (surface, ink, recessive grid) to a figure."""
    fig.update_layout(
        title={"text": title, "x": 0.0, "xanchor": "left", "font": {"size": 16, "color": INK}},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": ", ".join(FONT_STACK), "color": INK_SECONDARY, "size": 12},
        margin={"l": 60, "r": 30, "t": 70, "b": 50},
        legend={"orientation": "h", "y": -0.18},
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    if height:
        fig.update_layout(height=height)
    return fig


def save_plotly(fig: Any, name: str) -> Path:
    """Save a Plotly figure as self-contained-ish HTML (Plotly JS from CDN) in reports/figures."""
    path = figures_dir() / f"{name}.html"
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    LOG.info("wrote %s", path.name)
    return path


def save_table(frame: pd.DataFrame, name: str, floatfmt: int = 3) -> Path:
    """Write a table as CSV and as a markdown table next to each other in reports/analytics."""
    csv_path = analytics_dir() / f"{name}.csv"
    frame.to_csv(csv_path, index=False)
    md = markdown_table(frame, floatfmt)
    (analytics_dir() / f"{name}.md").write_text(md + "\n", encoding="utf-8")
    return csv_path


def markdown_table(frame: pd.DataFrame, floatfmt: int = 3) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""

    def fmt(value: Any) -> str:
        if isinstance(value, float | np.floating):
            return "" if np.isnan(value) else f"{value:.{floatfmt}f}"
        return str(value)

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def save_results(name: str, payload: dict[str, Any]) -> Path:
    """Write a module's headline numbers to reports/analytics/<name>.json."""
    path = analytics_dir() / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def load_results(name: str) -> dict[str, Any]:
    """Read back reports/analytics/<name>.json (raises FileNotFoundError with a hint)."""
    path = analytics_dir() / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} missing: run `make analytics-eda`")
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"not JSON serialisable: {type(value)}")


# --------------------------------------------------------------------------- database access
class DatabaseUnavailableError(RuntimeError):
    """The warehouse cannot be reached; analytics read from PostgreSQL by design."""


def read_sql(sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Run a query against the warehouse with SQLAlchemy and return a DataFrame."""
    if not db_available():
        raise DatabaseUnavailableError(
            "PostgreSQL warehouse unreachable. Run `make db-up` then `make db-all` (see database/README.md)."
        )
    from sqlalchemy import text

    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def season_weather() -> pd.DataFrame:
    """v_season_weather: one row per district, year, season (complete seasons only)."""
    frame = read_sql("SELECT * FROM v_season_weather ORDER BY district_id, year, season")
    for col in ("season_start", "season_end"):
        frame[col] = pd.to_datetime(frame[col])
    return frame


def training_table() -> pd.DataFrame:
    """v_training_table: yield label with season weather features."""
    return read_sql("SELECT * FROM v_training_table ORDER BY district_id, crop, year")


def districts() -> gpd.GeoDataFrame:
    """District polygons from PostGIS as a GeoDataFrame (EPSG:4326)."""
    frame = read_sql(
        "SELECT district_id, state, district, area_km2, ST_AsText(geom) AS wkt FROM districts ORDER BY district_id"
    )
    geometry = gpd.GeoSeries.from_wkt(frame["wkt"], crs="EPSG:4326")
    return gpd.GeoDataFrame(frame.drop(columns=["wkt"]), geometry=geometry, crs="EPSG:4326")


def train_end_year() -> int:
    """Last year of the yield training period (settings.yaml); anything fitted for later reuse stops here."""
    return int(load_settings()["splits"]["yield"]["train_end"])


def synthetic_satellite_used(frame: pd.DataFrame | None = None) -> bool:
    """True when any satellite-derived column in the warehouse is synthetic."""
    if frame is not None and "satellite_is_synthetic" in frame:
        return bool(frame["satellite_is_synthetic"].any())
    return bool(
        read_sql("SELECT coalesce(bool_or(is_synthetic), false) AS s FROM satellite_16d")["s"].iloc[
            0
        ]
    )
