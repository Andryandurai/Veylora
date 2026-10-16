"""Crop area / production / yield per district-season-crop.

Sources, in order of preference:

1. **ICRISAT District Level Database** (apportioned, 1966-2017) through its public JSON API,
   cached in ``data/raw/crops``. Units: area 1000 ha, production 1000 tons, yield kg/ha.
2. **data.gov.in / Ministry of Agriculture APY CSVs** that the user places in ``data/raw/crops/``
   (columns ``State_Name, District_Name, Crop_Year, Season, Crop, Area, Production``; hectares and
   tonnes, cotton in 170 kg bales). Used to extend years beyond ICRISAT's last year or to replace it.

Both are mapped onto the yield *units* of ``districts.parquet``. If neither is available the
caller falls back to ``etl.synthetic.crops`` (rows flagged ``is_synthetic=True``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_settings
from etl import icrisat
from etl.names import canonical_key, state_key

LOG = logging.getLogger(__name__)

CROP_COLUMNS = [
    "district_id",
    "year",
    "season",
    "crop",
    "area_ha",
    "production_t",
    "yield_t_per_ha",
    "yield_flag",
    "is_synthetic",
]
APY_CROP_NAMES = {
    "rice": "rice",
    "wheat": "wheat",
    "cotton(lint)": "cotton",
    "sugarcane": "sugarcane",
}
COTTON_BALE_T = 0.17  # one bale of lint = 170 kg
YIELD_FLAG_TOLERANCE = 0.05
APY_SEASON_MAP = {
    "kharif": "kharif",
    "rabi": "rabi",
    "autumn": "kharif",
    "winter": "kharif",
    "summer": "rabi",
}


class CropDataUnavailableError(RuntimeError):
    """No real crop source could be loaded (ICRISAT unreachable/uncached and no CSV present)."""


def _finalise(frame: pd.DataFrame, flag: pd.Series | None = None) -> pd.DataFrame:
    """Recompute yield, apply dtypes and the 5 percent reported-vs-recomputed flag."""
    out = frame.copy()
    out["yield_t_per_ha"] = out["production_t"] / out["area_ha"]
    out["yield_flag"] = (
        flag.reindex(out.index).fillna(False).astype(bool) if flag is not None else False
    )
    out["year"] = out["year"].astype("int16")
    out["is_synthetic"] = False
    for col in ("area_ha", "production_t", "yield_t_per_ha"):
        out[col] = out[col].astype("float64")
    return out[CROP_COLUMNS].sort_values(
        ["district_id", "crop", "season", "year"], ignore_index=True
    )


def icrisat_to_tidy(
    apy: pd.DataFrame, districts: pd.DataFrame, settings: dict | None = None
) -> pd.DataFrame:
    """Reshape the wide ICRISAT table into the ``crops_annual`` contract.

    Args:
        apy: output of :func:`etl.icrisat.fetch_apy`.
        districts: processed districts table (needs ``district_id``, ``state`` and ``icrisat_name``).
        settings: settings dict (defaults to config/settings.yaml).
    """
    settings = settings or load_settings()
    crop_season: dict[str, str] = settings["crop_season"]
    years = settings["years"]
    unit_lookup = {
        (state_key(r.state), r.icrisat_name): r.district_id for r in districts.itertuples()
    }
    frame = apy.copy()
    frame["district_id"] = [
        unit_lookup.get((state_key(s), n))
        for s, n in zip(frame["State Name"], frame["Dist Name"], strict=True)
    ]
    frame = frame[frame["district_id"].notna()]
    first = int(years.get("history_start", years["start"]))
    frame = frame[(frame["Year"] >= first) & (frame["Year"] <= int(years["end"]))]
    parts = []
    for crop in settings["crops"]:
        name = crop.upper()
        needed = [f"{name} AREA", f"{name} PRODUCTION", f"{name} YIELD"]
        if not all(c in frame.columns for c in needed):
            LOG.warning("ICRISAT table has no columns for crop %s; skipped", crop)
            continue
        part = pd.DataFrame(
            {
                "district_id": frame["district_id"],
                "year": frame["Year"],
                "season": crop_season[crop],
                "crop": crop,
                "area_ha": frame[needed[0]] * 1000.0,
                "production_t": frame[needed[1]] * 1000.0,
                "reported_t_per_ha": frame[needed[2]] / 1000.0,
            }
        )
        parts.append(part)
    tidy = pd.concat(parts, ignore_index=True)
    # ICRISAT marks "no data" with -1 and "crop not grown" with 0.
    valid = (tidy["area_ha"] > 0) & (tidy["production_t"] > 0)
    tidy = tidy[valid].reset_index(drop=True)
    recomputed = tidy["production_t"] / tidy["area_ha"]
    mismatch = (tidy["reported_t_per_ha"] - recomputed).abs() / recomputed > YIELD_FLAG_TOLERANCE
    return _finalise(tidy.drop(columns="reported_t_per_ha"), mismatch)


def _member_lookup(districts: pd.DataFrame) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """Lookups from modern district names to unit ids (state-aware and state-agnostic)."""
    by_state: dict[tuple[str, str], str] = {}
    by_name: dict[str, str] = {}
    for row in districts.itertuples():
        for member in str(row.member_districts).split("|"):
            by_state[(state_key(row.state), canonical_key(row.state, member))] = row.district_id
            by_name.setdefault(canonical_key(row.state, member), row.district_id)
    return by_state, by_name


def apy_csv_to_tidy(
    csv_path: Path, districts: pd.DataFrame, settings: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Load a data.gov.in style APY CSV and aggregate modern districts into yield units.

    Returns:
        (tidy crops frame, unmatched State_Name/District_Name pairs for the quality report,
        share of the file's rows for study states that joined to a unit)
    """
    settings = settings or load_settings()
    crop_season: dict[str, str] = settings["crop_season"]
    raw = pd.read_csv(csv_path)
    raw.columns = [c.strip() for c in raw.columns]
    required = {"State_Name", "District_Name", "Crop_Year", "Season", "Crop", "Area", "Production"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns {sorted(missing)}")
    raw["crop_key"] = raw["Crop"].astype(str).str.strip().str.lower()
    raw = raw[raw["crop_key"].isin(APY_CROP_NAMES)].copy()
    raw["crop"] = raw["crop_key"].map(APY_CROP_NAMES)
    raw["season_key"] = raw["Season"].astype(str).str.strip().str.lower()
    raw["season"] = [
        crop_season[c] if s.startswith("whole") else APY_SEASON_MAP.get(s, crop_season[c])
        for c, s in zip(raw["crop"], raw["season_key"], strict=True)
    ]
    raw["Production"] = np.where(
        raw["crop_key"] == "cotton(lint)", raw["Production"] * COTTON_BALE_T, raw["Production"]
    )
    by_state, by_name = _member_lookup(districts)
    ids = []
    for state, district in zip(raw["State_Name"], raw["District_Name"], strict=True):
        key = canonical_key(state, str(district).strip())
        ids.append(by_state.get((state_key(state), key)) or by_name.get(key))
    raw["district_id"] = ids
    study = {state_key(st) for st in settings["states"]}
    in_study = raw["State_Name"].map(state_key).isin(study)
    coverage = float(raw.loc[in_study, "district_id"].notna().mean()) if in_study.any() else 0.0
    unmatched = raw[in_study & raw["district_id"].isna()][
        ["State_Name", "District_Name"]
    ].drop_duplicates()
    matched = raw[raw["district_id"].notna()].copy()
    matched["year"] = pd.to_numeric(matched["Crop_Year"].astype(str).str[:4], errors="coerce")
    matched = matched.dropna(subset=["year", "Area", "Production"])
    grouped = (
        matched.groupby(["district_id", "year", "season", "crop"], as_index=False)[
            ["Area", "Production"]
        ]
        .sum()
        .rename(columns={"Area": "area_ha", "Production": "production_t"})
    )
    grouped = grouped[(grouped["area_ha"] > 0) & (grouped["production_t"] > 0)]
    return _finalise(grouped), unmatched.reset_index(drop=True), coverage


def build_crops(districts: pd.DataFrame, force: bool = False) -> tuple[pd.DataFrame, dict]:
    """Assemble the real ``crops_annual`` table from ICRISAT and any APY CSVs.

    Returns:
        (crops frame, report dict with source, coverage and unmatched names for DATA_QUALITY.md)

    Raises:
        CropDataUnavailableError: if no real source is available.
    """
    report: dict = {"sources": [], "unmatched": [], "coverage_by_source": {}, "join_coverage": None}
    settings = load_settings()
    frames: list[pd.DataFrame] = []
    try:
        apy = icrisat.fetch_apy(force)
        frames.append(icrisat_to_tidy(apy, districts))
        study = {state_key(st) for st in settings["states"]}
        rows = apy[apy["State Name"].map(state_key).isin(study)][["State Name", "Dist Name"]]
        known = {(state_key(r.state), r.icrisat_name) for r in districts.itertuples()}
        hit = [
            (state_key(s_), n) in known
            for s_, n in zip(rows["State Name"], rows["Dist Name"], strict=True)
        ]
        report["sources"].append("ICRISAT DLD apportioned (1966-2017)")
        report["coverage_by_source"]["ICRISAT"] = float(np.mean(hit)) if hit else 0.0
        report["unmatched"].extend(
            rows[[not h for h in hit]]
            .drop_duplicates()
            .rename(columns={"State Name": "State_Name", "Dist Name": "District_Name"})
            .to_dict("records")
        )
    except icrisat.IcrisatUnavailableError as exc:
        LOG.warning("ICRISAT unavailable: %s", exc)
    for csv_path in sorted(icrisat.raw_dir().glob("*.csv")):
        try:
            frame, unmatched, coverage = apy_csv_to_tidy(csv_path, districts, settings)
        except ValueError as exc:
            LOG.warning("skipping %s: %s", csv_path.name, exc)
            continue
        frames.append(frame)
        report["sources"].append(f"APY CSV {csv_path.name}")
        report["coverage_by_source"][csv_path.name] = coverage
        report["unmatched"].extend(unmatched.to_dict("records"))
    if not frames:
        raise CropDataUnavailableError(
            "No real crop data found. Download the ICRISAT District Level Database "
            "(http://data.icrisat.org/dld/) or the data.gov.in Area-Production-Yield CSV "
            "(https://data.gov.in, search 'District-wise, season-wise crop production statistics') "
            f"and place the CSV in {icrisat.raw_dir()}."
        )
    combined = pd.concat(frames, ignore_index=True)
    # ICRISAT (consistent apportioned boundaries) wins where both sources cover a record.
    combined = combined.drop_duplicates(
        subset=["district_id", "year", "season", "crop"], keep="first"
    )
    combined = combined.sort_values(["district_id", "crop", "season", "year"], ignore_index=True)
    coverages = list(report["coverage_by_source"].values())
    report["join_coverage"] = float(min(coverages)) if coverages else None
    report["rows"] = int(len(combined))
    return combined, report
