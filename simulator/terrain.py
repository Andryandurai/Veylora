"""Real relief for the simulator scene: open terrain tiles (SRTM and other public elevation data via AWS Terrain Tiles).

For a district centre this module fetches the Terrarium PNG tiles that cover a square window (default 10 km), stitches
them and resamples the window to a 32 x 32 grid of elevations in metres. The scene draws that relief (vertically
exaggerated, and flattened into terraced plots for paddy); only the *shape and size of the relief* are real. Fields,
soil classes and plants remain schematic.

Tiles are cached under ``data/raw/elevation`` so that an export without network access still works after the first run.
When a tile cannot be fetched and is not cached, :func:`elevation_window` returns ``None`` and the exporter falls back to
the procedural relief, labelling it as such.

Terrarium encoding: ``elevation_m = R * 256 + G + B / 256 - 32768``.
Attribution required by the tile provider (SRTM, GMTED2010 and 3DEP data): see :data:`ATTRIBUTION`.
"""

from __future__ import annotations

import io
import logging
import math
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from config import get_path

LOG = logging.getLogger(__name__)

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
ZOOM = 12
TILE_PX = 256
EARTH_CIRCUMFERENCE_M = 40_075_016.686
WINDOW_KM = 10.0
GRID = 32
ATTRIBUTION = "United States 3DEP (formerly NED) and global GMTED2010 and SRTM terrain data courtesy of the U.S. Geological Survey, via AWS Terrain Tiles."
Fetcher = Callable[[int, int, int], np.ndarray]


def tile_coordinates(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """Fractional slippy-map tile coordinates ``(x, y)`` of a point (x grows east, y grows south)."""
    n = 2.0**zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def decode_terrarium(png: bytes) -> np.ndarray:
    """Elevations (m) of one Terrarium tile as a float array ``(256, 256)``."""
    rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=float)
    return rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0


def cache_dir() -> Path:
    """Folder of cached tiles (``data/raw/elevation``)."""
    path = get_path("data_raw") / "elevation"
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_tile(z: int, x: int, y: int) -> np.ndarray:
    """One tile, from the cache or the network (cached after download); raises ``OSError`` on failure."""
    path = cache_dir() / f"terrarium_{z}_{x}_{y}.png"
    if not path.exists():
        try:
            with urllib.request.urlopen(TILE_URL.format(z=z, x=x, y=y), timeout=30) as response:
                path.write_bytes(response.read())
        except (urllib.error.URLError, TimeoutError) as exc:
            raise OSError(f"terrain tile {z}/{x}/{y} unavailable: {exc}") from exc
    return decode_terrarium(path.read_bytes())


def elevation_window(
    lat: float,
    lon: float,
    window_km: float = WINDOW_KM,
    grid: int = GRID,
    zoom: int = ZOOM,
    fetcher: Fetcher | None = None,
) -> np.ndarray | None:
    """Elevations (m) on a ``grid x grid`` lattice over a square window centred on a point.

    Row 0 is the northern edge and column 0 the western edge. Returns ``None`` if any needed tile is unavailable.
    """
    fetch = fetcher or fetch_tile
    cx, cy = tile_coordinates(lat, lon, zoom)
    metres_per_tile = EARTH_CIRCUMFERENCE_M * math.cos(math.radians(lat)) / 2.0**zoom
    half = window_km * 1000.0 / metres_per_tile / 2.0
    x0, x1 = int(math.floor(cx - half)), int(math.floor(cx + half))
    y0, y1 = int(math.floor(cy - half)), int(math.floor(cy + half))
    try:
        rows = [
            np.hstack([fetch(zoom, x, y) for x in range(x0, x1 + 1)]) for y in range(y0, y1 + 1)
        ]
    except OSError as exc:
        LOG.warning("no real relief for (%.3f, %.3f): %s", lat, lon, exc)
        return None
    mosaic = np.vstack(rows)
    # pixel coordinates (row, col) of the sampling lattice inside the mosaic; pixel centres sit at +0.5
    cols = (cx - half - x0) * TILE_PX + np.linspace(0.0, 2.0 * half * TILE_PX, grid) - 0.5
    rows_px = (cy - half - y0) * TILE_PX + np.linspace(0.0, 2.0 * half * TILE_PX, grid) - 0.5
    yy, xx = np.meshgrid(rows_px, cols, indexing="ij")
    return np.asarray(map_coordinates(mosaic, [yy, xx], order=1, mode="nearest"))


def relief_stats(elevation: np.ndarray) -> dict[str, float]:
    """Mean elevation and 5th-95th percentile relief (m) of a window; the percentile range ignores single-pixel spikes."""
    p5, p95 = np.percentile(elevation, [5, 95])
    return {"mean_elevation_m": float(elevation.mean()), "relief_m": float(p95 - p5)}
