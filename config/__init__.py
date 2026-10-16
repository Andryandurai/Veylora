"""Project configuration: one place to read settings.yaml, resolve paths and seed randomness."""

from __future__ import annotations

import logging
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = ROOT / "config" / "settings.yaml"

LOG = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_settings(path: Path | None = None) -> dict[str, Any]:
    """Load settings.yaml once and return it as a plain dict.

    Args:
        path: Optional alternative YAML file (used by tests). Defaults to config/settings.yaml.
    """
    target = Path(path) if path else SETTINGS_PATH
    with target.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if not isinstance(settings, dict):
        raise ValueError(f"{target} must contain a YAML mapping at the top level")
    return settings


def load_env(path: Path | None = None) -> dict[str, str]:
    """Load KEY=VALUE lines from .env into ``os.environ`` without overriding existing variables.

    Returns the variables that were read from the file (empty if there is no .env).
    """
    target = Path(path) if path else ROOT / ".env"
    loaded: dict[str, str] = {}
    if not target.is_file():
        return loaded
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def get_path(key: str) -> Path:
    """Return an absolute path for a key under `paths:` in settings.yaml, creating it if needed."""
    relative = load_settings()["paths"][key]
    path = ROOT / relative
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_seed() -> int:
    """Return the global random seed (42 unless overridden in settings.yaml)."""
    return int(load_settings()["project"]["random_seed"])


def set_global_seed(seed: int | None = None) -> int:
    """Seed Python, NumPy and (if installed) PyTorch so runs are reproducible."""
    seed = get_seed() if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global seed kept for third-party libraries
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # torch is optional for ETL-only environments
        LOG.debug("torch not installed; skipping torch seed")
    return seed


def year_range() -> range:
    """Inclusive study period from settings as a range of years."""
    years = load_settings()["years"]
    return range(int(years["start"]), int(years["end"]) + 1)
