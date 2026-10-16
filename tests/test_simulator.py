"""Phase 7 tests: data contract, procedural helpers, embedding, JavaScript syntax and the browser test."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from config import ROOT
from simulator import embed
from simulator import export_sim_data as X

SIM = ROOT / "simulator"
DATA_PATH = SIM / "sim_data.json"
data_ready = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="run `python -m simulator.export_sim_data` first"
)


def _tiny_data() -> dict:
    """A minimal valid document (one district, one crop, full grid) for schema tests."""
    weekly = [
        {
            "week": w + 1,
            "rain_mm": 10.0,
            "tmax_c": 33.0,
            "soil_moisture_0_1": 0.5,
            "ndvi": 0.6,
            "etc_mm": 30.0,
            "eta_mm": 28.0,
            "irrigation_mm": 0.0,
            "excess_mm": 0.0,
            "stress_0_1": 0.1,
            "drought_p": 0.2,
            "heat_p": 0.0,
        }
        for w in range(16)
    ]
    grid = {
        "rain_mult": [1.0, 1.2],
        "temp_delta": [0, 1],
        "strategies": ["rainfed", "baseline", "optimized"],
    }
    scenarios = {
        f"a__b|rice|{r:g}|{t:g}|{s}": {
            "yield": {"mean": 2.0, "lo": 1.5, "hi": 2.5},
            "water_used_mm": 0.0,
            "weekly": weekly,
        }
        for r in grid["rain_mult"]
        for t in grid["temp_delta"]
        for s in grid["strategies"]
    }
    district = {
        "id": "a__b", "name": "B", "state": "A", "cluster": 0, "cluster_label": "x", "lat": 12.0, "lon": 78.0, "area_km2": 1000.0,
        "elevation_proxy": X.elevation_proxy("a__b"), "tiles": X.soil_tiles("a__b"), "crops": ["rice"],
        "reference_year": {"rice": 2005}, "budget_mm": {"rice": 350.0}, "taw_mm": {"rice": 80.0}, "early_warning": {},
    }  # fmt: skip
    meta = {
        "schema_version": 1, "generated_at": "now", "git_commit": "abc", "weeks": 16, "risk_levels": {}, "grid": grid, "definitions": {}, "disclaimer": "x",
        "is_synthetic_inputs": {"weather": False, "crop_labels": False, "satellite_dataset": True, "satellite_used_by_served_models": False},
        "model_versions": {"yield": "v"},
    }  # fmt: skip
    return {
        "meta": meta,
        "crops": {"rice": {"season": "kharif", "note": "", "stage_days": [20, 30, 40, 22]}},
        "districts": [district],
        "scenarios": scenarios,
    }


def test_procedural_terrain_and_soil_are_deterministic_and_in_range() -> None:
    a, b = X.elevation_proxy("tamil_nadu__madurai"), X.elevation_proxy("tamil_nadu__madurai")
    assert a == b and X.elevation_proxy("karnataka__kolar") != a
    arr = np.array(a)
    assert arr.shape == (32, 32) and arr.min() == 0.0 and arr.max() == 1.0
    assert (np.abs(np.diff(arr, axis=0)) < 0.35).all()  # smooth relief, no spikes
    tiles = np.array(X.soil_tiles("tamil_nadu__madurai"))
    assert (
        tiles.shape == (8, 8)
        and set(np.unique(tiles)) <= {0, 1, 2, 3}
        and len(np.unique(tiles)) >= 2
    )


def test_round_floats_and_schema_accepts_a_valid_document() -> None:
    assert X.round_floats({"a": [1.23456, {"b": 2.0004}], "c": 3}) == {
        "a": [1.235, {"b": 2.0}],
        "c": 3,
    }
    X.validate(_tiny_data())


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["scenarios"]["a__b|rice|1|0|rainfed"]["weekly"].pop(), "weekly"),
        (
            lambda d: d["scenarios"]["a__b|rice|1|0|rainfed"]["weekly"][3].update(
                soil_moisture_0_1=1.4
            ),
            "soil_moisture",
        ),
        (lambda d: d["scenarios"]["a__b|rice|1|0|rainfed"]["yield"].update(mean=-1), "mean"),
        (lambda d: d["districts"][0].update(tiles=[[0] * 8] * 7), "tiles"),
        (lambda d: d["meta"].pop("disclaimer"), "disclaimer"),
        (lambda d: d["scenarios"].pop("a__b|rice|1.2|1|optimized"), "missing scenario"),
        (lambda d: d["crops"]["rice"].update(stage_days=[1, 2, 3]), "stage_days"),
    ],
)
def test_validation_rejects_broken_documents(mutate, message: str) -> None:
    data = copy.deepcopy(_tiny_data())
    mutate(data)
    with pytest.raises(ValueError, match=message):
        X.validate(data)


@data_ready
def test_exported_file_is_valid_complete_and_within_size_limit() -> None:
    assert DATA_PATH.stat().st_size < X.MAX_BYTES
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    X.validate(data)
    grid = data["meta"]["grid"]
    expected = (
        sum(len(d["crops"]) for d in data["districts"])
        * len(grid["rain_mult"])
        * len(grid["temp_delta"])
        * len(grid["strategies"])
    )
    assert len(data["scenarios"]) == expected
    assert data["meta"]["is_synthetic_inputs"]["satellite_used_by_served_models"] is False
    assert "not satellite ndvi" in data["meta"]["definitions"]["ndvi"].lower()


@data_ready
def test_scenarios_are_internally_consistent_with_the_models() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    sc = data["scenarios"]
    for d in data["districts"][:3]:
        crop = d["crops"][0]
        base = sc[f"{d['id']}|{crop}|1|0|baseline"]
        rainfed, opt = sc[f"{d['id']}|{crop}|1|0|rainfed"], sc[f"{d['id']}|{crop}|1|0|optimized"]
        assert (
            base["water_used_mm"] == pytest.approx(600.0, abs=0.01)
            and rainfed["water_used_mm"] == 0
        )
        assert 0 <= opt["water_used_mm"] <= d["budget_mm"][crop] + 1e-6
        assert sum(w["irrigation_mm"] for w in base["weekly"]) == pytest.approx(
            base["water_used_mm"], abs=0.5
        )
        assert all(w["irrigation_mm"] == 0 for w in rainfed["weekly"])
        for s in (base, rainfed, opt):
            assert s["yield"]["lo"] <= s["yield"]["mean"] <= s["yield"]["hi"]
        # weekly rain scales with the multiplier (same reference weather)
        rain1 = sum(w["rain_mm"] for w in base["weekly"])
        rain14 = sum(w["rain_mm"] for w in sc[f"{d['id']}|{crop}|1.4|0|baseline"]["weekly"])
        assert rain14 == pytest.approx(1.4 * rain1, rel=0.01)
        # drought risk does not fall when rainfall falls; the heat share does not fall when it gets warmer
        dry = np.mean([w["drought_p"] for w in sc[f"{d['id']}|{crop}|0.4|0|baseline"]["weekly"]])
        wet = np.mean([w["drought_p"] for w in sc[f"{d['id']}|{crop}|1.6|0|baseline"]["weekly"]])
        assert dry >= wet
        hot = np.mean([w["heat_p"] for w in sc[f"{d['id']}|{crop}|1|4|baseline"]["weekly"]])
        cool = np.mean([w["heat_p"] for w in sc[f"{d['id']}|{crop}|1|-2|baseline"]["weekly"]])
        assert hot >= cool
        tmax4 = np.mean([w["tmax_c"] for w in sc[f"{d['id']}|{crop}|1|4|baseline"]["weekly"]])
        tmax0 = np.mean([w["tmax_c"] for w in base["weekly"]])
        assert tmax4 == pytest.approx(tmax0 + 4.0, abs=0.01)


def test_embedding_inlines_valid_json_and_is_safe_for_script_tags(tmp_path) -> None:
    data_file = tmp_path / "d.json"
    payload = _tiny_data()
    payload["meta"]["disclaimer"] = "contains </script> and </div> text"
    data_file.write_text(json.dumps(payload), encoding="utf-8")
    out = embed.build_embedded(vendor=False, data_path=data_file, out_path=tmp_path / "e.html")
    html = out.read_text(encoding="utf-8")
    start = html.index('<script id="sim-data" type="application/json">') + len(
        '<script id="sim-data" type="application/json">'
    )
    block = html[start : html.index("</script>", start)]
    assert (
        json.loads(block)["meta"]["disclaimer"] == "contains </script> and </div> text"
    )  # JSON escape round-trips
    assert html.count('<script type="module">') == 1 and "three-src" not in html.replace(
        "getElementById('three-src')", ""
    )


@pytest.mark.skipif(
    not embed.THREE_FILE.exists(), reason="Three.js not vendored yet (python -m simulator.embed)"
)
def test_vendored_three_is_inlined_and_cannot_close_the_script_tag(tmp_path) -> None:
    data_file = tmp_path / "d.json"
    data_file.write_text(json.dumps(_tiny_data()), encoding="utf-8")
    html = embed.build_embedded(
        vendor=True, data_path=data_file, out_path=tmp_path / "e.html"
    ).read_text(encoding="utf-8")
    assert '<script id="three-src" type="text/plain">' in html
    assert "</script" not in embed.THREE_FILE.read_text(encoding="utf-8").lower()
    assert "sha256" in (embed.VENDOR_DIR / "THREE_NOTICE.txt").read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_page_javascript_has_valid_module_syntax(tmp_path) -> None:
    html = (SIM / "index.html").read_text(encoding="utf-8")
    start = html.index('<script type="module">') + len('<script type="module">')
    script = html[start : html.rindex("</script>")]
    module = tmp_path / "page.mjs"
    module.write_text(script, encoding="utf-8")
    result = subprocess.run(["node", "--check", str(module)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_page_has_no_hard_coded_result_numbers_and_states_its_limits() -> None:
    html = (SIM / "index.html").read_text(encoding="utf-8")
    for needle in (
        "not a physical crop-growth simulation",
        "proxy",
        'aria-label="Rainfall multiplier"',
        "prefers-reduced-motion",
        'role="meter"',
    ):
        assert needle in html
    assert "fetch('sim_data.json')" in html and 'id="dropzone"' in html


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_interpolation_logic_matches_python_reference(tmp_path) -> None:
    """Run the page's bracket()/bilinear code in node against a numpy reference."""
    html = (SIM / "index.html").read_text(encoding="utf-8")
    start = html.index("function bracket(")
    end = html.index("function weekState(")
    snippet = html[start:end]
    script = (
        "const FIELDS=['v'];let DATA,GRID;const clamp=(v,a=0,b=1)=>Math.min(b,Math.max(a,v));const lerp=(a,b,t)=>a+(b-a)*t;"
        + snippet
        + "GRID={rain_mult:[0.4,1,1.6],temp_delta:[-2,0,4]};DATA={meta:{weeks:2},scenarios:{}};"
        "for(const r of GRID.rain_mult)for(const t of GRID.temp_delta)for(const s of ['rainfed','baseline','optimized'])"
        "DATA.scenarios[key('d','c',r,t,s)]={yield:{mean:r*10+t,lo:0,hi:0},water_used_mm:r+t,weekly:[{v:0,rain_mm:r*10+t,tmax_c:0,soil_moisture_0_1:0,ndvi:0,etc_mm:0,eta_mm:0,irrigation_mm:0,excess_mm:0,stress_0_1:0,drought_p:0,heat_p:0},{v:0,rain_mm:0,tmax_c:0,soil_moisture_0_1:0,ndvi:0,etc_mm:0,eta_mm:0,irrigation_mm:0,excess_mm:0,stress_0_1:0,drought_p:0,heat_p:0}]};"
        "const out=[[0.7,-1],[1.3,2],[0.4,-2],[1.6,4],[2.0,9]].map(([r,t])=>scenarioAt('d','c',r,t,'baseline').yield.mean);console.log(JSON.stringify(out));"
    )
    script = script.replace(
        "FIELDS=['v']",
        "FIELDS=['rain_mm','tmax_c','soil_moisture_0_1','ndvi','etc_mm','eta_mm','irrigation_mm','excess_mm','stress_0_1','drought_p','heat_p']",
    )
    file = tmp_path / "interp.mjs"
    file.write_text(script, encoding="utf-8")
    result = subprocess.run(["node", str(file)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    got = json.loads(result.stdout)
    # the test function is affine in (rain, temp), so bilinear interpolation must reproduce it (inputs outside the grid are clamped)
    expected = [0.7 * 10 - 1, 1.3 * 10 + 2, 0.4 * 10 - 2, 1.6 * 10 + 4, 1.6 * 10 + 4]
    np.testing.assert_allclose(got, expected, atol=1e-9)


@pytest.mark.skipif(
    not (SIM / "index_embedded.html").exists(), reason="run `python -m simulator.embed` first"
)
def test_browser_test_passes() -> None:
    pytest.importorskip("playwright")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "test_sim.py")],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if "Executable doesn't exist" in result.stdout + result.stderr:
        pytest.skip("Chromium not installed (python -m playwright install chromium)")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
