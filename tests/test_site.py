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
    make_site.build(out, dashboards_url=None, design="field", copy_simulator=False)
    return out


def _page(out: Path) -> str:
    return (out / "index.html").read_text(encoding="utf-8")


def test_site_is_named_veylora_and_has_no_placeholders(site):
    page = _page(site)
    assert "<title>Veylora" in page and "<h1>Veylora</h1>" in page
    assert "ClimateResilience" not in page
    assert not re.search(r"TODO|lorem|MISSING", page, flags=re.I)
    assert (site / "styles.css").is_file()


def test_headline_numbers_are_computed_from_the_result_files(site):
    page = _page(site)
    extreme = json.loads((ROOT / "reports" / "extreme_weather.json").read_text(encoding="utf-8"))
    drought = extreme["drought"]
    f1 = drought["feature_sets"][drought["served_feature_set"]]["test"]["f1"]
    assert f"F1 {f1:.2f}" in page
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
    assert "not hosted on this page" in _page(site)
    out = tmp_path / "with-link"
    make_site.build(out, dashboards_url="https://example.invalid/app", copy_simulator=False)
    page = _page(out)
    assert 'href="https://example.invalid/app"' in page and "Open the dashboards" in page
    assert "not hosted on this page" not in page


def test_synthetic_data_note_follows_the_data_quality_report(site):
    quality = json.loads((ROOT / "reports" / "data_quality.json").read_text(encoding="utf-8"))
    shares = [n for n, s in quality["synthetic_share"].items() if s > 0]
    page = _page(site)
    assert ("Synthetic data in the pipeline" in page) == bool(shares)


def test_missing_results_show_a_visible_marker(monkeypatch):
    monkeypatch.setattr(make_site, "REPORTS", ROOT / "reports" / "does-not-exist")
    for result in (
        make_site.yield_result(),
        make_site.drought_result(),
        make_site.irrigation_result(),
    ):
        assert "MISSING" in result.value


def test_both_designs_build_and_unknown_design_is_rejected(tmp_path):
    for design in ("field", "almanac"):
        out = tmp_path / design
        make_site.build(out, design=design, copy_simulator=False)
        assert "--paper:" in _page(out)
    with pytest.raises(ValueError):
        make_site.build(tmp_path / "x", design="neon", copy_simulator=False)


def test_vercel_config_serves_the_web_folder_and_copies_the_simulator():
    config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    assert config["outputDirectory"] == "web"
    assert (
        config["buildCommand"].startswith("cp ") and "web/simulator.html" in config["buildCommand"]
    )
    assert (ROOT / "data" / "sample" / "simulator" / "index_embedded.html").is_file()
    assert "web/simulator.html" in (ROOT / ".gitignore").read_text(encoding="utf-8")
