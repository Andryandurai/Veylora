"""District boundaries: geoBoundaries ADM2 polygons grouped into yield-reporting units.

Why units and not raw ADM2 polygons: ICRISAT's *apportioned* yield series is stated on 1966 parent
districts (e.g. one "Thanjavur" that today is Thanjavur + Thiruvarur + Nagapattinam + ...), so a
modern polygon cannot be joined to a yield by name. We therefore:

1. load modern (2021) ADM2 polygons and attribute each to a state via ADM1,
2. anchor every ICRISAT district to the modern polygons whose names match it (alias map in
   ``config/district_name_map.csv``),
3. attach every remaining modern polygon to the nearest anchor of the same state,
4. dissolve each group into one unit polygon. That unit is a row of ``districts.parquet``.

This reconstruction is an approximation of the 1966 boundaries (documented as D-006). If ICRISAT is
unreachable and uncached, units fall back to the modern polygons themselves.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import MultiPolygon, Polygon

from config import ROOT, get_path, load_settings
from etl import icrisat
from etl.names import canonical_key, make_district_id, split_aliases, state_key

LOG = logging.getLogger(__name__)

OVERRIDES_PATH = ROOT / "config" / "unit_overrides.csv"
GEO_API = "https://www.geoboundaries.org/api/current/gbOpen/{iso3}/{level}/"
AREA_CRS = "EPSG:6933"  # WGS 84 / NSIDC EASE-Grid 2.0 Global (cylindrical equal-area)
DIST_CRS = "EPSG:7755"  # WGS 84 / India NSF LCC (distances)

DISTRICT_COLUMNS = [
    "district_id",
    "state",
    "district",
    "centroid_lat",
    "centroid_lon",
    "geometry_wkt",
    "area_km2",
    "member_districts",
]


def boundaries_raw_dir() -> Path:
    """Cache directory for downloaded boundary files."""
    path = get_path("data_raw") / "boundaries"
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_geoboundaries(level: str, iso3: str = "IND", force: bool = False) -> Path:
    """Download the simplified geoBoundaries GeoJSON for ``level`` (ADM1/ADM2) unless cached.

    geoBoundaries gbOpen is licensed ODbL 1.0; the metadata JSON is stored beside the data.
    """
    target = boundaries_raw_dir() / f"geoBoundaries-{iso3}-{level}_simplified.geojson"
    if target.exists() and not force:
        return target
    meta = requests.get(GEO_API.format(iso3=iso3, level=level), timeout=60)
    meta.raise_for_status()
    info = meta.json()
    data = requests.get(info["simplifiedGeometryGeoJSON"], timeout=300)
    data.raise_for_status()
    target.write_bytes(data.content)
    keep = (
        "boundaryName",
        "boundaryYearRepresented",
        "boundaryLicense",
        "boundarySource",
        "gjDownloadURL",
    )
    meta_path = target.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({k: info.get(k) for k in keep}, indent=1), encoding="utf-8")
    LOG.info("downloaded %s (%d KB)", target.name, target.stat().st_size // 1024)
    return target


def load_modern(states: list[str], exclude: list[str] | None = None) -> gpd.GeoDataFrame:
    """Modern ADM2 polygons of the study states with canonical ``state`` and ``district`` columns."""
    cfg = load_settings()["etl"]["boundaries"]
    adm1 = gpd.read_file(fetch_geoboundaries("ADM1", cfg["iso3"]))
    adm2 = gpd.read_file(fetch_geoboundaries(cfg["level"], cfg["iso3"]))
    points = adm2.copy()
    points["geometry"] = adm2.representative_point()
    joined = gpd.sjoin(
        points, adm1[["shapeName", "geometry"]], how="left", predicate="within", rsuffix="adm1"
    )
    joined = joined[~joined.index.duplicated(keep="first")]
    adm2["state_key"] = joined["shapeName_adm1"].map(
        lambda s: state_key(s) if isinstance(s, str) else ""
    )
    wanted = {state_key(s): s for s in states}
    modern = adm2[adm2["state_key"].isin(wanted)].copy()
    modern["state"] = modern["state_key"].map(wanted)
    modern["district"] = modern["shapeName"].str.strip()
    excluded = {canonical_key("", d) for d in (exclude or [])}
    modern = modern[~modern["district"].map(lambda d: canonical_key("", d) in excluded)]
    modern["geometry"] = modern.geometry.make_valid()
    return modern[["state", "state_key", "district", "geometry"]].reset_index(drop=True)


def _match_anchors(
    modern_state: gpd.GeoDataFrame, unit_names: list[str], state: str
) -> dict[str, list[int]]:
    """Map each unit name to the indices of modern polygons whose names match it.

    Exact (alias-normalised) matches win; prefix matches ("warangal" -> "warangal r", "warangal u")
    are only used for polygons no unit matched exactly.
    """
    poly_keys = {idx: canonical_key(state, name) for idx, name in modern_state["district"].items()}
    anchors: dict[str, list[int]] = {name: [] for name in unit_names}
    claimed: set[int] = set()
    for name in unit_names:
        for part in split_aliases(name):
            key = canonical_key(state, part)
            hits = [i for i, k in poly_keys.items() if k == key and i not in claimed]
            anchors[name].extend(hits)
            claimed.update(hits)
    for name in unit_names:
        for part in split_aliases(name):
            key = canonical_key(state, part)
            hits = [i for i, k in poly_keys.items() if k.startswith(key + " ") and i not in claimed]
            anchors[name].extend(hits)
            claimed.update(hits)
    return anchors


def load_overrides(path: Path | None = None) -> dict[tuple[str, str], str]:
    """Read config/unit_overrides.csv into ``{(state_key, canonical district key): unit name}``."""
    target = path or OVERRIDES_PATH
    if not target.exists():
        return {}
    table = pd.read_csv(target, comment="#", dtype=str).dropna(subset=["modern_district"])
    return {
        (state_key(row.state), canonical_key(row.state, row.modern_district)): row.unit
        for row in table.itertuples()
    }


def assign_units(
    modern: gpd.GeoDataFrame,
    units: pd.DataFrame,
    overrides: dict[tuple[str, str], str] | None = None,
) -> gpd.GeoDataFrame:
    """Add a ``unit`` column (ICRISAT district name) to every modern polygon.

    Args:
        modern: output of :func:`load_modern`.
        units: DataFrame with ``state``, ``name``, ``lat``, ``lon`` of ICRISAT districts.
    """
    overrides = load_overrides() if overrides is None else overrides
    projected = modern.to_crs(DIST_CRS)
    centroids = projected.geometry.centroid
    unit_of: dict[int, str] = {}
    for state in modern["state"].unique():
        state_idx = modern.index[modern["state"] == state]
        sub = modern.loc[state_idx]
        skey = state_key(state)
        unit_names = units.loc[units["state"].map(state_key) == skey, "name"].tolist()
        if not unit_names:
            for idx in state_idx:
                unit_of[idx] = modern.at[idx, "district"]
            continue
        anchors = _match_anchors(sub, unit_names, state)
        for name, idxs in anchors.items():
            for idx in idxs:
                unit_of[idx] = name
        missing = [n for n, idxs in anchors.items() if not idxs]
        for name in missing:  # last resort: polygon nearest to ICRISAT's own (approximate) point
            row = units[(units["state"].map(state_key) == skey) & (units["name"] == name)].iloc[0]
            pt = gpd.GeoSeries(
                gpd.points_from_xy([row["lon"]], [row["lat"]]), crs="EPSG:4326"
            ).to_crs(DIST_CRS)
            nearest = projected.loc[state_idx].distance(pt.iloc[0]).idxmin()
            if nearest not in unit_of:
                unit_of[nearest] = name
                anchors[name] = [nearest]
                LOG.warning(
                    "unit %r (%s) anchored by ICRISAT coordinates -> %s",
                    name,
                    state,
                    modern.at[nearest, "district"],
                )
            else:
                LOG.warning("unit %r (%s) has no anchor polygon and is dropped", name, state)
        for idx in state_idx:
            target = overrides.get((skey, canonical_key(state, modern.at[idx, "district"])))
            if target in unit_names:
                unit_of[idx] = target
            elif target:
                LOG.warning(
                    "override for %s -> %r ignored: unknown unit",
                    modern.at[idx, "district"],
                    target,
                )
        anchored = [i for i in state_idx if i in unit_of]
        for idx in state_idx:
            if idx in unit_of:
                continue
            dists = {i: centroids[idx].distance(centroids[i]) for i in anchored}
            unit_of[idx] = unit_of[min(dists, key=dists.get)]
    result = modern.copy()
    result["unit"] = pd.Series(unit_of)
    return result[result["unit"].notna()]


def build_districts(assigned: gpd.GeoDataFrame) -> pd.DataFrame:
    """Dissolve polygons by (state, unit) into the ``districts`` table (data contract in CLAUDE.md)."""
    groups = []
    for (state, unit), part in assigned.groupby(["state", "unit"], sort=True):
        geom = part.geometry.union_all().buffer(0)
        if isinstance(geom, Polygon):
            geom = MultiPolygon([geom])
        groups.append(
            {
                "state": state,
                "district": split_aliases(unit)[-1],
                "icrisat_name": unit,
                "members": "|".join(sorted(part["district"])),
                "geometry": geom,
            }
        )
    frame = gpd.GeoDataFrame(groups, geometry="geometry", crs="EPSG:4326")
    projected = frame.to_crs(AREA_CRS)
    centroid = projected.geometry.centroid.to_crs("EPSG:4326")
    out = pd.DataFrame(
        {
            "district_id": [
                make_district_id(s, d)
                for s, d in zip(frame["state"], frame["district"], strict=True)
            ],
            "state": frame["state"].astype(str),
            "district": frame["district"].astype(str),
            "centroid_lat": centroid.y.astype("float64"),
            "centroid_lon": centroid.x.astype("float64"),
            "geometry_wkt": frame.geometry.to_wkt(rounding_precision=5),
            "area_km2": (projected.area / 1e6).astype("float64"),
            "member_districts": frame["members"],
            "icrisat_name": frame["icrisat_name"],
        }
    )
    if out["district_id"].duplicated().any():
        dupes = out.loc[out["district_id"].duplicated(keep=False), "district_id"].tolist()
        raise ValueError(f"duplicate district_id values: {dupes}")
    return out


def load_units_spec(states: list[str]) -> pd.DataFrame:
    """ICRISAT districts (with yield data) of the study states; empty if ICRISAT is unreachable and uncached."""
    wanted = {state_key(s) for s in states}
    try:
        units = icrisat.fetch_units()
    except icrisat.IcrisatUnavailableError:
        LOG.warning(
            "ICRISAT district list unavailable: falling back to modern ADM2 polygons as units"
        )
        return pd.DataFrame(columns=["state", "name", "lat", "lon", "dist_code"])
    return units[units["state"].map(state_key).isin(wanted)].reset_index(drop=True)


def build(force: bool = False) -> Path:
    """Build data/processed/districts.parquet (idempotent unless ``force``)."""
    out_path = get_path("data_processed") / "districts.parquet"
    if out_path.exists() and not force:
        LOG.info("districts.parquet exists, skipping (use --force)")
        return out_path
    settings = load_settings()
    states = settings["states"]
    modern = load_modern(states, settings["etl"]["boundaries"].get("exclude_districts"))
    units = load_units_spec(states)
    assigned = assign_units(modern, units)
    districts = build_districts(assigned)
    districts.to_parquet(out_path, index=False)
    members = assigned.assign(unit_district=assigned["unit"].map(lambda u: split_aliases(u)[-1]))
    members[["state", "district", "unit", "unit_district"]].to_parquet(
        get_path("data_interim") / "unit_members.parquet", index=False
    )
    LOG.info("districts.parquet: %d units from %d modern polygons", len(districts), len(modern))
    return out_path


def load_districts() -> pd.DataFrame:
    """Read the processed districts table."""
    path = get_path("data_processed") / "districts.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "districts.parquet missing: run `python -m etl.run --source boundaries`"
        )
    return pd.read_parquet(path)


def districts_geodataframe(districts: pd.DataFrame | None = None) -> gpd.GeoDataFrame:
    """Districts as a GeoDataFrame (geometry parsed from WKT, EPSG:4326)."""
    districts = load_districts() if districts is None else districts
    geometry = gpd.GeoSeries.from_wkt(districts["geometry_wkt"], crs="EPSG:4326")
    return gpd.GeoDataFrame(
        districts.drop(columns=["geometry_wkt"]), geometry=geometry, crs="EPSG:4326"
    )
