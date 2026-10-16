"""Friendly handling of missing data, models and expected failures: pages never show stack traces."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import streamlit as st

from config import bundle
from dashboards.components import ROOT, labels, ui

LOG = logging.getLogger(__name__)

REQUIRED = {
    "processed data": (
        [
            "data/processed/districts.parquet",
            "data/processed/weather_daily.parquet",
            "data/processed/crops_annual.parquet",
            "data/processed/satellite_16d.parquet",
        ],
        "run `make etl` (python -m etl.run --all)",
    ),
    "district clusters": (["data/processed/district_clusters.parquet"], "run `make analytics-eda`"),
    "yield models": (["models/yield_xgb.joblib"], "run `make train-yield`"),
    "early-warning models": (
        ["models/heatwave_lstm.pt", "models/drought_xgb.joblib"],
        "run `make train-extreme`",
    ),
    "simulator data": (["simulator/index_embedded.html"], "run `make export-sim embed-sim`"),
}


def missing(groups: list[str]) -> list[tuple[str, str]]:
    """(group, instruction) for every requested group with a missing file (after trying the demo bundle)."""
    if any(not (ROOT / f).exists() for group in groups for f in REQUIRED[group][0]):
        bundle.restore_if_missing()  # a fresh clone or a deployed app starts from data/sample
    problems = []
    for group in groups:
        files, hint = REQUIRED[group]
        if any(not (ROOT / f).exists() for f in files):
            problems.append((group, hint))
    return problems


def require(*groups: str) -> None:
    """Stop the page with a plain message if required artefacts are absent (the fix for the operator goes to the log)."""
    problems = missing(list(groups))
    if problems:
        for group, hint in problems:
            LOG.warning("missing %s: %s", group, hint)
        st.error(
            "This page needs data or models that have not been built yet. "
            "It is not available right now; please try again later."
        )
        st.stop()


def run_guarded(body: Callable[[], Any]) -> None:
    """Run a page body and turn expected failures into readable messages (unexpected ones are still raised)."""
    try:
        body()
    except st.errors.StreamlitAPIException:
        raise
    except FileNotFoundError as exc:
        LOG.error("required file missing: %s", exc)
        st.error("Something this page needs is not available right now. Please try again later.")
    except ValueError as exc:
        LOG.warning("combination cannot be computed: %s", exc)
        st.warning("This combination of choices cannot be computed. Please try different settings.")
    except Exception as exc:  # models/artefacts built by an older version, DB or library problems
        if exc.__class__.__name__ in {
            "ModelNotTrainedError",
            "ModelsNotTrainedError",
            "DatabaseUnavailableError",
            "NotImplementedError",
        }:
            LOG.error("%s: %s", exc.__class__.__name__, exc)
            st.error("This page is not available right now. Please try again later.")
        else:
            raise


def synthetic_banner() -> None:
    """One quiet line on every page while any pipeline input is synthetic (reads reports/data_quality.json)."""
    path = ROOT / "reports" / "data_quality.json"
    if not path.exists():
        return
    share = json.loads(path.read_text(encoding="utf-8")).get("synthetic_share", {})
    synthetic = {t: s for t, s in share.items() if s > 0}
    if synthetic:
        parts = ", ".join(f"{labels.table_label(t)} ({s:.0%})" for t, s in synthetic.items())
        ui.note(
            f"<b>Synthetic data in use: {parts}.</b> The forecasts shown use observed weather and "
            "ICRISAT yields only; the synthetic satellite data appears only in an experimental comparison (see Methodology)."
        )


def read_report(name: str) -> dict[str, Any]:
    """Read ``reports/<name>.json`` (empty dict if absent)."""
    path = Path(ROOT) / "reports" / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
