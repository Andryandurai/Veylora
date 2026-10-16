"""Phase 0 smoke test: the scientific stack imports and settings.yaml loads."""

from __future__ import annotations

import importlib

import pytest

from config import ROOT, get_seed, load_settings, year_range

REQUIRED_MODULES = [
    "pandas",
    "numpy",
    "geopandas",
    "pyarrow",
    "sklearn",
    "xgboost",
    "torch",
    "statsmodels",
    "scipy",
    "pulp",
    "plotly",
    "folium",
    "streamlit",
    "sqlalchemy",
    "psycopg",
    "ee",
    "imdlib",
    "requests",
    "yaml",
]

PACKAGES = [
    "etl",
    "etl.synthetic",
    "database",
    "analytics",
    "analytics.descriptive",
    "analytics.diagnostic",
    "analytics.predictive",
    "analytics.prescriptive",
    "simulator",
    "dashboards",
    "advisory",
]


@pytest.mark.parametrize("module", REQUIRED_MODULES)
def test_dependency_imports(module: str) -> None:
    assert importlib.import_module(module) is not None


@pytest.mark.parametrize("package", PACKAGES)
def test_project_packages_import(package: str) -> None:
    assert importlib.import_module(package) is not None


def test_settings_load() -> None:
    settings = load_settings()
    assert get_seed() == 42
    assert set(settings["crops"]) == {"rice", "wheat", "cotton", "sugarcane"}
    assert "Tamil Nadu" in settings["states"]
    years = year_range()
    # Crop-year window follows real data availability (ICRISAT ends 2017, docs/DECISIONS.md D-006).
    assert years[0] == 2000 and years[-1] == 2017
    assert settings["weather_years"] == {"start": 1991, "end": 2024}
    assert settings["splits"]["yield"]["test"][1] <= years[-1]


def test_expected_layout_exists() -> None:
    for relative in ("CLAUDE.md", "docs/DECISIONS.md", "docker-compose.yml", ".env.example"):
        assert (ROOT / relative).is_file(), relative
    for relative in ("data/raw", "data/interim", "data/processed", "models", "reports"):
        assert (ROOT / relative).is_dir(), relative
