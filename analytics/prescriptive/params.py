"""Loaders for config/crop_params.yaml and config/economics.yaml (see the provenance warnings in those files)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import yaml

from config import ROOT

CROP_PARAMS_PATH = ROOT / "config" / "crop_params.yaml"
ECONOMICS_PATH = ROOT / "config" / "economics.yaml"
N_STAGES = 4


@dataclass(frozen=True)
class CropParams:
    """Crop-specific parameters of the water balance and yield response."""

    name: str
    kc_ini: float
    kc_mid: float
    kc_end: float
    stage_days: tuple[int, int, int, int]
    root_depth_m: float
    p: float
    ky: tuple[float, float, float, float]
    default_budget_mm: float


@dataclass(frozen=True)
class Globals:
    """Settings shared by all crops."""

    season_days: int
    weekly_cap_mm: float
    rain_efficiency: float
    initial_depletion_fraction: float
    irrigation_efficiency: float
    soil_classes: dict[str, float]
    default_soil_class: str

    @property
    def n_weeks(self) -> int:
        return self.season_days // 7


@lru_cache(maxsize=1)
def _raw() -> dict[str, Any]:
    with CROP_PARAMS_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_globals() -> Globals:
    """Shared parameters (season length, weekly cap, efficiencies, soil classes)."""
    raw = _raw()
    return Globals(
        season_days=int(raw["season_days"]),
        weekly_cap_mm=float(raw["weekly_cap_mm"]),
        rain_efficiency=float(raw["rain_efficiency"]),
        initial_depletion_fraction=float(raw["initial_depletion_fraction"]),
        irrigation_efficiency=float(raw["irrigation_efficiency"]),
        soil_classes={k: float(v) for k, v in raw["soil_classes"].items()},
        default_soil_class=str(raw["default_soil_class"]),
    )


def load_crop(crop: str) -> CropParams:
    """Parameters of one crop; validates that stage lengths add up to the season length."""
    raw = _raw()
    if crop not in raw["crops"]:
        raise KeyError(f"no parameters for crop {crop!r}; known: {sorted(raw['crops'])}")
    c = raw["crops"][crop]
    stages = tuple(int(d) for d in c["stage_days"])
    if len(stages) != N_STAGES or sum(stages) != int(raw["season_days"]):
        raise ValueError(
            f"{crop}: stage_days must have {N_STAGES} entries summing to {raw['season_days']}, got {stages}"
        )
    ky = tuple(float(k) for k in c["ky"])
    if len(ky) != N_STAGES:
        raise ValueError(f"{crop}: ky needs {N_STAGES} values")
    return CropParams(
        crop,
        float(c["kc"]["ini"]),
        float(c["kc"]["mid"]),
        float(c["kc"]["end"]),
        stages,
        float(c["root_depth_m"]),
        float(c["p"]),
        ky,
        float(c["default_budget_mm"]),
    )  # type: ignore[arg-type]


def taw_mm(crop: CropParams, soil_class: str | None = None) -> float:
    """Total available water of the root zone in mm (soil class mm/m x root depth)."""
    g = load_globals()
    soil = soil_class or g.default_soil_class
    if soil not in g.soil_classes:
        raise KeyError(f"unknown soil class {soil!r}; known: {sorted(g.soil_classes)}")
    return g.soil_classes[soil] * crop.root_depth_m


def load_economics() -> dict[str, Any]:
    """Prices, costs and unit conversions for the crop-switch estimate."""
    with ECONOMICS_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
