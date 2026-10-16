"""Friendly handling of missing data, models and expected failures: pages never show stack traces."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import streamlit as st

from config import bundle
from dashboards.components import ROOT, ui

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
    """Stop the page with a clear instruction if required artefacts are absent."""
    problems = missing(list(groups))
    if problems:
        st.error("This page needs data or models that have not been built yet.")
        for group, hint in problems:
            st.markdown(f"- **{group}**: {hint}")
        st.info("See the README quickstart, or run `make pipeline` to build everything.")
        st.stop()


def run_guarded(body: Callable[[], Any]) -> None:
    """Run a page body and turn expected failures into readable messages (unexpected ones are still raised)."""
    try:
        body()
    except st.errors.StreamlitAPIException:
        raise
    except FileNotFoundError as exc:
        st.error(f"A required file is missing: {exc}. Run the pipeline steps listed in the README.")
    except ValueError as exc:
        st.warning(f"This combination cannot be computed: {exc}")
    except Exception as exc:  # models/artefacts built by an older version, DB or library problems
        if exc.__class__.__name__ in {
            "ModelNotTrainedError",
            "ModelsNotTrainedError",
            "DatabaseUnavailableError",
            "NotImplementedError",
        }:
            st.error(f"{exc}")
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
        parts = ", ".join(f"{t} ({s:.0%})" for t, s in synthetic.items())
        ui.note(
            f"<b>Synthetic data in the pipeline: {parts}.</b> The forecasts shown use observed weather and "
            "ICRISAT yields only; the synthetic satellite table appears only in an experimental comparison (see Methodology)."
        )


def read_report(name: str) -> dict[str, Any]:
    """Read ``reports/<name>.json`` (empty dict if absent)."""
    path = Path(ROOT) / "reports" / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
