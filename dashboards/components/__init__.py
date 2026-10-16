"""Shared helpers for the Streamlit dashboards (data access, models, charts, maps, guards)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if (
    str(ROOT) not in sys.path
):  # lets pages import analytics/, advisory/, config/ without an editable install
    sys.path.insert(0, str(ROOT))
