"""Canonical state/district names and stable ``district_id`` slugs.

Every source spells districts differently (diacritics, "Tiruchirapalli / Trichy", "Kadapa(YSR)").
All joins in the pipeline go through :func:`name_key` (a lossy, comparable form) and the alias file
``config/district_name_map.csv`` so that spelling variants are fixed in exactly one place.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from config import ROOT

NAME_MAP_PATH = ROOT / "config" / "district_name_map.csv"

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_STATE_ALIASES = {"orissa": "odisha", "uttaranchal": "uttarakhand", "pondicherry": "puducherry"}


def strip_diacritics(text: str) -> str:
    """Remove accents/macrons: 'Tamil Nādu' -> 'Tamil Nadu'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def name_key(text: str) -> str:
    """Lossy comparison key: ascii, lowercase, alphanumerics separated by single spaces.

    >>> name_key("Kadapa(YSR)")
    'kadapa ysr'
    >>> name_key("Tamil Nādu")
    'tamil nadu'
    """
    ascii_text = strip_diacritics(str(text)).lower().replace("&", " and ")
    return _NON_ALNUM.sub(" ", ascii_text).strip()


def state_key(state: str) -> str:
    """Comparison key for a state name, with historical renames applied (Orissa -> Odisha)."""
    key = name_key(state)
    return _STATE_ALIASES.get(key, key)


def slugify(text: str) -> str:
    """Underscore slug of :func:`name_key`: 'Tamil Nadu' -> 'tamil_nadu'."""
    return name_key(text).replace(" ", "_")


def make_district_id(state: str, district: str) -> str:
    """Stable id ``<state>__<district>``, e.g. ``tamil_nadu__tiruchirapalli``."""
    return f"{slugify(state_key(state))}__{slugify(district)}"


def split_aliases(name: str) -> list[str]:
    """Split ICRISAT-style composite names: 'North Arcot / Vellore' -> both parts (and the whole)."""
    parts = [p.strip() for p in re.split(r"\s*/\s*", str(name)) if p.strip()]
    return parts if len(parts) > 1 else [str(name).strip()]


@lru_cache(maxsize=4)
def load_name_map(path: Path | None = None) -> dict[tuple[str, str], str]:
    """Load the alias file into ``{(state_key, alias_key): canonical_key}``.

    The CSV has columns ``state,alias,canonical``. A blank state means "any state".
    """
    target = Path(path) if path else NAME_MAP_PATH
    mapping: dict[tuple[str, str], str] = {}
    if not target.exists():
        return mapping
    with target.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row.get("state") or "").lstrip().startswith("#"):
                continue  # comment row
            state = state_key(row["state"]) if row.get("state") else ""
            mapping[(state, name_key(row["alias"]))] = name_key(row["canonical"])
    return mapping


def canonical_key(state: str, district: str, path: Path | None = None) -> str:
    """Return the canonical comparison key for a district name after applying the alias map."""
    mapping = load_name_map(path)
    key = name_key(district)
    skey = state_key(state) if state else ""
    return mapping.get((skey, key)) or mapping.get(("", key)) or key
