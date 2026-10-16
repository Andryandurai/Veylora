"""Area-weighted aggregation of regular lat/lon grids onto district polygons.

For a grid of cells and a set of polygons we build a sparse matrix ``W`` (n_polygons x n_cells)
whose entry is the area of the intersection of polygon *i* with cell *j* (computed in an equal-area
CRS). The district mean of a field on day *t* is then ``sum_j W_ij * x_tj / sum_j W_ij`` over the
cells that have data, so ocean/missing cells are ignored and weights are renormalised per day.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import shapely
from scipy import sparse

LOG = logging.getLogger(__name__)

AREA_CRS = "EPSG:6933"


def grid_step(coords: np.ndarray) -> float:
    """Spacing of a regular, monotonic coordinate axis (degrees)."""
    if len(coords) < 2:
        raise ValueError("a grid axis needs at least two coordinates")
    return float(np.median(np.abs(np.diff(coords))))


def cell_boxes(lat: np.ndarray, lon: np.ndarray) -> gpd.GeoSeries:
    """Cell footprints (EPSG:4326) in row-major order: index = i_lat * n_lon + i_lon."""
    step_lat, step_lon = grid_step(lat), grid_step(lon)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    boxes = shapely.box(
        lon_grid.ravel() - step_lon / 2,
        lat_grid.ravel() - step_lat / 2,
        lon_grid.ravel() + step_lon / 2,
        lat_grid.ravel() + step_lat / 2,
    )
    return gpd.GeoSeries(boxes, crs="EPSG:4326")


def build_weights(
    polygons: gpd.GeoSeries,
    lat: np.ndarray,
    lon: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> sparse.csr_matrix:
    """Intersection-area weight matrix (n_polygons x n_cells), in km2.

    Args:
        polygons: polygon per district in EPSG:4326, in the order rows should appear.
        lat, lon: 1-D cell-centre coordinates of the grid.
        valid_mask: optional flat boolean array (n_cells) marking cells that carry data. Cells
            outside the mask get zero weight; a polygon with no valid overlap at all is attached
            to the valid cell nearest its centroid so that it never ends up empty.
    """
    cells = cell_boxes(lat, lon)
    cells_area = cells.to_crs(AREA_CRS).values
    polys_area = polygons.to_crs(AREA_CRS).values
    tree = shapely.STRtree(cells_area)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i, geom in enumerate(polys_area):
        idx = tree.query(geom, predicate="intersects")
        if idx.size == 0:
            continue
        if valid_mask is not None:
            idx = idx[valid_mask[idx]]
            if idx.size == 0:
                continue
        areas = shapely.area(shapely.intersection(cells_area[idx], geom)) / 1e6
        keep = areas > 0
        rows.extend([i] * int(keep.sum()))
        cols.extend(idx[keep].tolist())
        vals.extend(areas[keep].tolist())
    weights = sparse.csr_matrix(
        (vals, (rows, cols)), shape=(len(polygons), len(cells)), dtype=np.float64
    )

    empty = np.flatnonzero(np.asarray(weights.sum(axis=1)).ravel() == 0)
    if empty.size:
        centres = np.column_stack([np.repeat(lat, len(lon)), np.tile(lon, len(lat))])
        candidates = np.arange(len(cells)) if valid_mask is None else np.flatnonzero(valid_mask)
        weights = weights.tolil()
        for i in empty:
            centroid = polygons.iloc[i].centroid
            distance = np.hypot(
                centres[candidates, 0] - centroid.y, centres[candidates, 1] - centroid.x
            )
            weights[i, candidates[int(distance.argmin())]] = 1.0
            LOG.warning("polygon %d has no valid grid overlap; using the nearest valid cell", i)
        weights = weights.tocsr()
    return weights


def aggregate(values: np.ndarray, weights: sparse.csr_matrix) -> np.ndarray:
    """Weighted mean per polygon for every time step, ignoring NaN cells.

    Args:
        values: array (T, n_cells) with NaN for missing cells.
        weights: matrix from :func:`build_weights`.

    Returns:
        float32 array (T, n_polygons); NaN where a polygon had no valid cell that day.
    """
    valid = ~np.isnan(values)
    filled = np.where(valid, values, 0.0).astype(np.float64)
    numerator = weights @ filled.T
    denominator = weights @ valid.astype(np.float64).T
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(denominator > 0, numerator / denominator, np.nan)
    return mean.T.astype(np.float32)


def valid_cells(field: np.ndarray) -> np.ndarray:
    """Flat mask of cells with at least one finite value across time; ``field`` is (T, n_lat, n_lon)."""
    return np.isfinite(field).any(axis=0).ravel()
