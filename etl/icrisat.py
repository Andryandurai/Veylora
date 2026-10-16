"""Client for the ICRISAT District Level Database (DLD) JSON API.

The public portal (http://data.icrisat.org/dld/) is a thin front end over
``http://data.icrisat.org/dldAPI/``. We use the *apportioned* dataset, which re-states every district
on a consistent set of parent boundaries (1966 districts, ~313 districts in 20 states), so yields are
comparable across the 1966-2017 record even though many districts were later split.

Responses are cached under ``data/raw/crops`` so the pipeline works offline after the first run.
Cite: ICRISAT District Level Database, http://data.icrisat.org/dld/ (check terms of use before
redistributing derived data).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config import get_path

LOG = logging.getLogger(__name__)

API_BASE = "http://data.icrisat.org/dldAPI/"
APY_FILE = "icrisat_apportioned_apy.json"


class IcrisatUnavailableError(RuntimeError):
    """Raised when the API cannot be reached and no cached copy exists."""


def raw_dir() -> Path:
    """Directory holding cached ICRISAT responses and any manually downloaded crop CSVs."""
    path = get_path("data_raw") / "crops"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_json(endpoint: str, timeout: int = 180, retries: int = 3) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(API_BASE + endpoint, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            LOG.warning("ICRISAT %s attempt %d/%d failed: %s", endpoint, attempt, retries, exc)
    raise IcrisatUnavailableError(f"ICRISAT API unreachable ({endpoint}): {last_error}")


def _cached(filename: str, endpoint: str, force: bool) -> Any:
    path = raw_dir() / filename
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        payload = _get_json(endpoint)
    except IcrisatUnavailableError:
        if path.exists():
            LOG.warning("using stale cached %s", path.name)
            return json.loads(path.read_text(encoding="utf-8"))
        raise
    path.write_text(json.dumps(payload), encoding="utf-8")
    LOG.info("cached %s (%d KB)", path.name, path.stat().st_size // 1024)
    return payload


def fetch_units(force: bool = False) -> pd.DataFrame:
    """Districts that actually have yield data: columns state, name, lat, lon, dist_code.

    Derived from the Area-Production-Yield table (not the separate district endpoint, which lists
    districts without yields and spells composite names without the " / " separator). ICRISAT's own
    coordinates are approximate and wrong for a few districts, so they are only a last-resort
    anchor; see :mod:`etl.boundaries`.
    """
    apy = fetch_apy(force)
    units = apy[["State Name", "Dist Name", "LATITUDE", "LONGITUDE", "Dist Code"]].drop_duplicates(
        subset=["State Name", "Dist Name"]
    )
    units = units.rename(
        columns={
            "State Name": "state",
            "Dist Name": "name",
            "LATITUDE": "lat",
            "LONGITUDE": "lon",
            "Dist Code": "dist_code",
        }
    )
    return units.reset_index(drop=True)


def fetch_apy(force: bool = False) -> pd.DataFrame:
    """Return the apportioned Area-Production-Yield table as a wide DataFrame (one row per district-year).

    Columns are the ICRISAT headers (e.g. ``RICE AREA`` in 1000 ha, ``RICE PRODUCTION`` in 1000 tons,
    ``RICE YIELD`` in kg/ha) plus ``State Name``, ``Dist Name`` and ``Year``.
    """
    payload = _cached(APY_FILE, "apportioned/area-production-yield", force)
    columns = [header["header"] for header in payload["headers"]]
    frame = pd.DataFrame(payload["data"], columns=columns)
    numeric = [c for c in columns if c not in {"State Name", "Dist Name", "Region Name"}]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    return frame
