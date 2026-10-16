"""Static assets for the dashboards: the self-contained 3D simulator is copied into ``dashboards/static``."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlencode

from dashboards.components import ROOT

SOURCE = ROOT / "simulator" / "index_embedded.html"
SIM_DISTRICTS = ROOT / "simulator" / "sim_districts.json"
TARGET = ROOT / "dashboards" / "static" / "simulator.html"
URL = "/app/static/simulator.html"


@dataclass(frozen=True)
class SimulatorChoice:
    """The simulator district and crop to show, and a note when they differ from the sidebar selection."""

    district_id: str
    crop: str
    note: str | None = None


def ensure_simulator() -> Path | None:
    """Copy the embedded simulator to the static folder when missing or older than its source (None if no source)."""
    if not SOURCE.exists():
        return TARGET if TARGET.exists() else None
    if not TARGET.exists() or TARGET.stat().st_mtime < SOURCE.stat().st_mtime:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE, TARGET)
    return TARGET


@lru_cache(maxsize=1)
def simulator_districts() -> tuple[dict, ...]:
    """The representative districts exported to the simulator (id, name, cluster, lat, lon, crops)."""
    if not SIM_DISTRICTS.exists():
        return ()
    return tuple(json.loads(SIM_DISTRICTS.read_text(encoding="utf-8")))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Approximate great-circle distance in degrees (good enough to rank neighbours)."""
    return math.hypot(a[0] - b[0], (a[1] - b[1]) * math.cos(math.radians((a[0] + b[0]) / 2)))


def simulator_choice(
    district_id: str,
    crop: str,
    centroids: dict[str, tuple[float, float]],
    clusters: dict[str, int],
) -> SimulatorChoice:
    """Pick what the simulator shows for the sidebar selection.

    The simulator holds one representative district per climate cluster, not all 51 units. A selection outside it
    is mapped to the representative of the same cluster (nearest one if several) or else the nearest district.
    """
    sims = simulator_districts()
    if not sims:
        return SimulatorChoice(district_id, crop)
    chosen = next((s for s in sims if s["id"] == district_id), None)
    note = None
    if chosen is None:
        here = centroids.get(district_id)
        same = [
            s
            for s in sims
            if clusters.get(district_id) is not None and s["cluster"] == clusters[district_id]
        ]
        pool = same or list(sims)
        chosen = min(pool, key=lambda s: _distance(here, (s["lat"], s["lon"])) if here else 0.0)
        reason = (
            "the representative of the same climate cluster"
            if same
            else "the nearest district it contains"
        )
        note = f"The simulator holds {len(sims)} representative districts. For the selected district it shows {chosen['name']}, {reason}."
    crops = chosen["crops"]
    use_crop = crop if crop in crops else crops[0]
    if use_crop != crop:
        extra = f"{crop.capitalize()} is not simulated for {chosen['name']}; showing {use_crop}."
        note = f"{note} {extra}" if note else extra
    return SimulatorChoice(chosen["id"], use_crop, note)


def simulator_url(district_id: str, crop: str) -> str:
    """URL of the simulator with the district and crop preselected (the page reads these query parameters)."""
    return f"{URL}?{urlencode({'district': district_id, 'crop': crop})}"
