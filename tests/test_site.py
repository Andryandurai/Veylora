"""The static Veylora site served by Vercel: numbers come from the result files, links and notes follow the options."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts import make_site

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def site(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("site")
    make_site.build(out, dashboards_url=None, copy_simulator=False)
    return out


def _page(out: Path) -> str:
    return (out / "index.html").read_text(encoding="utf-8")


def test_site_is_named_veylora_and_has_no_placeholders(site):
    page = _page(site)
    assert "<title>Veylora" in page and "<h1>Veylora</h1>" in page
    assert "ClimateResilience" not in page
    assert not re.search(r"TODO|lorem|MISSING", page, flags=re.I)
    # nothing aimed at developers: no commands, repository links, file paths or internal table names
    assert not re.search(
        r"github|make [a-z]|\.yaml|\.json|reports/|satellite_16d|<code>", page, flags=re.I
    )
    assert (site / "styles.css").is_file()


def test_headline_numbers_are_computed_from_the_result_files(site):
    page = _page(site)
    extreme = json.loads((ROOT / "reports" / "extreme_weather.json").read_text(encoding="utf-8"))
    drought = extreme["drought"]
    f1 = drought["feature_sets"][drought["served_feature_set"]]["test"]["f1"]
    assert f'<span class="v">{f1:.2f}</span>' in page
    results = json.loads(
        (ROOT / "reports" / "prescriptive_results.json").read_text(encoding="utf-8")
    )
    optimized = next(
        r
        for r in results["summary"]
        if r["scenario"] == "normal rainfall" and r["strategy"] == "optimized"
    )
    assert f"{optimized['water_saved_pct']:.0f}% less water" in page
    assert f"{optimized['yield_vs_baseline']:.0%} of its yield" in page


def test_dashboards_link_only_when_a_url_is_given(site, tmp_path):
    assert "Open the dashboards" not in _page(site)
    assert "not available from this page yet" in _page(site)
    out = tmp_path / "with-link"
    make_site.build(out, dashboards_url="https://example.invalid/app", copy_simulator=False)
    page = _page(out)
    assert 'href="https://example.invalid/app"' in page and "Open the dashboards" in page
    assert "not available from this page yet" not in page


def test_synthetic_data_note_follows_the_data_quality_report(site):
    quality = json.loads((ROOT / "reports" / "data_quality.json").read_text(encoding="utf-8"))
    shares = [n for n, s in quality["synthetic_share"].items() if s > 0]
    page = _page(site)
    assert ("Synthetic data in use" in page) == bool(shares)
    assert "satellite_16d" not in page


def test_missing_results_show_a_visible_marker(monkeypatch):
    monkeypatch.setattr(make_site, "REPORTS", ROOT / "reports" / "does-not-exist")
    for result in (
        make_site.yield_result(),
        make_site.drought_result(),
        make_site.irrigation_result(),
    ):
        assert "MISSING" in result.value


def test_page_carries_the_dashboard_colour_tokens(site):
    page = _page(site)
    for name, value in make_site.theme.TOKENS.items():
        assert f"--{name}: {value};" in page


def test_vercel_config_serves_the_web_folder_and_copies_the_simulator():
    config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    assert config["outputDirectory"] == "web"
    assert (
        config["buildCommand"].startswith("cp ") and "web/simulator.html" in config["buildCommand"]
    )
    assert (ROOT / "data" / "sample" / "simulator" / "index_embedded.html").is_file()
    assert "web/simulator.html" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_satellite_credit_appears_only_when_the_satellite_data_is_real(site):
    quality = json.loads((ROOT / "reports" / "data_quality.json").read_text(encoding="utf-8"))
    real = quality["synthetic_share"]["satellite_16d"] == 0
    page = _page(site)
    assert ("MODIS MOD13Q1" in page and "Copernicus Climate Change Service" in page) == real
