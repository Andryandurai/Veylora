"""Phase 7 tests: data contract, procedural helpers, embedding, JavaScript syntax and the browser test."""

from __future__ import annotations

import copy
import io
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from config import ROOT
from simulator import embed, terrain
from simulator import export_sim_data as X

SIM = ROOT / "simulator"
DATA_PATH = SIM / "sim_data.json"
data_ready = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="run `python -m simulator.export_sim_data` first"
)


def _tiny_data(ponded: bool = False) -> dict:
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
            "pond_mm": 0.0,
            "seepage_mm": 0.0,
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
            "stage_rel": [1.0, 0.95, 0.9, 1.0],
            "weekly": weekly,
            **({"pond_daily": [0.0] * 112} if ponded else {}),
        }
        for r in grid["rain_mult"]
        for t in grid["temp_delta"]
        for s in grid["strategies"]
    }
    growth, ripening = X.development_curves([20, 30, 40, 22])
    daily = {"rain_mm": [1.0] * 112, "tmax_c": [33.0] * 112, "tmin_c": [24.0] * 112}
    district = {
        "id": "a__b", "name": "B", "state": "A", "cluster": 0, "cluster_label": "x", "lat": 12.0, "lon": 78.0, "area_km2": 1000.0,
        "elevation_proxy": X.elevation_proxy("a__b"), "elevation_source": "procedural (real relief unavailable)", "tiles": X.soil_tiles("a__b"), "crops": ["rice"],
        "reference_year": {"rice": 2005}, "reference_daily": {"rice": daily}, "reference_ndvi": {"rice": {"weekly": [0.5] * 16, "is_synthetic": True}},
        "budget_mm": {"rice": 350.0}, "taw_mm": {"rice": 80.0}, "early_warning": {},
    }  # fmt: skip
    meta = {
        "schema_version": 2, "generated_at": "now", "git_commit": "abc", "weeks": 16, "risk_levels": {}, "grid": grid, "definitions": {}, "disclaimer": "x", "attribution": ["terrain"],
        "is_synthetic_inputs": {"weather": False, "crop_labels": False, "satellite_dataset": True, "satellite_used_by_served_models": False},
        "model_versions": {"yield": "v"},
    }  # fmt: skip
    crop = {
        "season": "kharif", "note": "", "stage_days": [20, 30, 40, 22], "season_start": "06-01", "ponded": ponded,
        "pond_max_mm": 75.0 if ponded else 0.0, "percolation_mm_day": 2.0 if ponded else 0.0, "initial_pond_mm": 25.0 if ponded else 0.0,
        "development": growth, "senescence": ripening,
    }  # fmt: skip
    return {"meta": meta, "crops": {"rice": crop}, "districts": [district], "scenarios": scenarios}


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
        (
            lambda d: d["scenarios"]["a__b|rice|1|0|rainfed"].update(stage_rel=[1, 1, 1]),
            "stage_rel",
        ),
        (
            lambda d: d["scenarios"]["a__b|rice|1|0|rainfed"]["stage_rel"].__setitem__(2, 1.3),
            "stage_rel",
        ),
        (lambda d: d["scenarios"]["a__b|rice|1|0|rainfed"]["weekly"][2].pop("pond_mm"), "pond_mm"),
        (lambda d: d["districts"][0]["reference_daily"]["rice"]["rain_mm"].pop(), "rain_mm"),
        (
            lambda d: d["districts"][0]["reference_daily"]["rice"].update(rain_mm=[-1.0] * 112),
            "rain_mm",
        ),
        (lambda d: d["districts"][0].pop("elevation_source"), "elevation_source"),
        (lambda d: d["meta"].update(schema_version=1), "schema_version"),
        (lambda d: d["meta"].pop("attribution"), "attribution"),
        (lambda d: d["crops"]["rice"].update(development=[0.1] * 15), "development"),
        (lambda d: d["districts"][0].update(elevation_m=[[0.0] * 31] * 32), "elevation_m"),
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
        'id="card-stage"',
        'id="card-pond"',
        'id="btn-uncertainty"',
        'id="rng-hour"',
        "reference_daily",
        "pond_daily",
        "stage_rel",
        "schematic",
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
        "DATA.scenarios[key('d','c',r,t,s)]={yield:{mean:r*10+t,lo:0,hi:0},water_used_mm:r+t,stage_rel:[r,t,1,0],pond_daily:[r*10+t,0,5],weekly:[{v:0,rain_mm:r*10+t,tmax_c:0,soil_moisture_0_1:0,ndvi:0,etc_mm:0,eta_mm:0,irrigation_mm:0,excess_mm:0,stress_0_1:0,pond_mm:0,seepage_mm:0,drought_p:0,heat_p:0},{v:0,rain_mm:0,tmax_c:0,soil_moisture_0_1:0,ndvi:0,etc_mm:0,eta_mm:0,irrigation_mm:0,excess_mm:0,stress_0_1:0,pond_mm:0,seepage_mm:0,drought_p:0,heat_p:0}]};"
        "const pts=[[0.7,-1],[1.3,2],[0.4,-2],[1.6,4],[2.0,9]];"
        "const out=pts.map(([r,t])=>scenarioAt('d','c',r,t,'baseline').yield.mean);"
        "const rel=pts.map(([r,t])=>scenarioAt('d','c',r,t,'baseline').stage_rel);"
        "const pond=pts.map(([r,t])=>scenarioAt('d','c',r,t,'baseline').pond_daily);"
        "console.log(JSON.stringify({out,rel,pond}));"
    )
    script = script.replace(
        "FIELDS=['v']",
        "FIELDS=['rain_mm','tmax_c','soil_moisture_0_1','ndvi','etc_mm','eta_mm','irrigation_mm','excess_mm','stress_0_1','pond_mm','seepage_mm','drought_p','heat_p']",
    )
    file = tmp_path / "interp.mjs"
    file.write_text(script, encoding="utf-8")
    result = subprocess.run(["node", str(file)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    got = json.loads(result.stdout)
    # the test function is affine in (rain, temp), so bilinear interpolation must reproduce it (inputs outside the grid are clamped)
    points = [(0.7, -1), (1.3, 2), (0.4, -2), (1.6, 4), (1.6, 4)]
    np.testing.assert_allclose(got["out"], [10 * r + t for r, t in points], atol=1e-9)
    np.testing.assert_allclose(
        got["rel"], [[r, t, 1, 0] for r, t in points], atol=1e-9
    )  # the stage factors interpolate too
    np.testing.assert_allclose(
        got["pond"], [[10 * r + t, 0, 5] for r, t in points], atol=1e-9
    )  # and so does the daily pond series


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


# ---------------------------------------------------------------- schema v2: ponded paddy, daily weather, growth curves
def test_ponded_documents_validate_and_the_daily_pond_series_is_checked() -> None:
    data = _tiny_data(ponded=True)
    X.validate(data)
    data["scenarios"]["a__b|rice|1|0|rainfed"]["pond_daily"].pop()
    with pytest.raises(ValueError, match="pond_daily"):
        X.validate(data)


@pytest.mark.parametrize(
    "stage_days", [[20, 30, 40, 22], [18, 27, 40, 27], [25, 30, 35, 22], [20, 25, 42, 25]]
)
def test_development_curves_are_bounded_monotone_and_follow_the_stage_lengths(stage_days) -> None:
    growth, ripening = X.development_curves(stage_days)
    ini, dev, mid, late = stage_days
    assert len(growth) == len(ripening) == 16
    assert all(0 <= v <= 1 for v in growth + ripening)
    assert growth == sorted(growth) and ripening == sorted(ripening)
    assert growth[0] < 0.12  # a seedling in week 1
    assert growth[-1] == pytest.approx(1.0)  # full size by the end of the window
    full_week = next(w for w in range(16) if 7 * w + 3 >= ini + dev + mid)
    assert growth[full_week] == pytest.approx(1.0)  # once the mid-season has passed
    assert (
        all(v == 0 for v in ripening[:full_week]) and ripening[-1] > 0
    )  # ripening only in the last stage
    assert ripening[-1] == pytest.approx(min(1.0, (7 * 15 + 3 - (ini + dev + mid)) / late))


def test_reference_ndvi_uses_the_covering_composite_and_keeps_the_synthetic_flag() -> None:
    dates = pd.date_range("2000-05-20", periods=4, freq="16D")
    frame = pd.DataFrame(
        {
            "district_id": "a__b",
            "date": dates,
            "ndvi": np.linspace(0.2, 0.8, 4),
            "is_synthetic": True,
        }
    )
    other = frame.assign(district_id="c__d", ndvi=0.99)
    sat = pd.concat([frame, other], ignore_index=True)
    result = X.reference_ndvi(sat, "a__b", 2000, "kharif")  # kharif starts on 1 June
    assert result["is_synthetic"] is True and len(result["weekly"]) == 16
    start = pd.Timestamp("2000-06-01")
    for week, value in enumerate(result["weekly"]):
        end = start + pd.Timedelta(days=7 * week + 6)
        covering = frame[frame["date"] <= end].iloc[-1]
        if (end - covering["date"]).days > 16:
            assert value is None
        else:
            assert value == pytest.approx(float(np.clip(covering["ndvi"], 0, 1)))
    assert (
        any(v is None for v in result["weekly"]) and result["weekly"][0] is not None
    )  # the composites stop after 4 x 16 days
    real = X.reference_ndvi(sat.assign(is_synthetic=False), "a__b", 2000, "kharif")
    assert real["is_synthetic"] is False
    assert X.reference_ndvi(sat, "nowhere__x", 2000, "kharif") is None
    one_synthetic = sat.assign(is_synthetic=[i == 3 for i in range(len(sat))])
    assert X.reference_ndvi(one_synthetic, "a__b", 2000, "kharif")["is_synthetic"] is True


@pytest.mark.skipif(
    not (ROOT / "data" / "processed" / "weather_daily.parquet").exists(),
    reason="run `make etl` first",
)
def test_reference_daily_undoes_the_scenario_multiplier_and_shift() -> None:
    from analytics.prescriptive import scenario as S

    plain = S.load_season_inputs("tamil_nadu__thanjavur", "rice", "kharif", 2005)
    scenario = S.load_season_inputs(
        "tamil_nadu__thanjavur", "rice", "kharif", 2005, rain_multiplier=1.4, temperature_delta=2.0
    )
    a, b = X.reference_daily(plain), X.reference_daily(scenario)
    assert len(a["rain_mm"]) == len(a["tmax_c"]) == len(a["tmin_c"]) == 112
    for key in ("rain_mm", "tmax_c", "tmin_c"):
        np.testing.assert_allclose(
            a[key], b[key], atol=0.11
        )  # equal up to the one-decimal rounding
    assert min(a["rain_mm"]) >= 0 and all(
        t1 >= t0 for t0, t1 in zip(a["tmin_c"], a["tmax_c"], strict=True)
    )


# ---------------------------------------------------------------- real relief (open terrain tiles)
def test_tile_coordinates_and_terrarium_decoding() -> None:
    assert terrain.tile_coordinates(0.0, 0.0, 2) == pytest.approx((2.0, 2.0))
    x, y = terrain.tile_coordinates(10.78, 79.14, 12)
    assert 2800 < x < 3000 and 1800 < y < 1950  # southern India at zoom 12
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = (
        128,
        100,
        128,
    )  # 128 * 256 + 100 + 128 / 256 - 32768 = 100.5 m
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    decoded = terrain.decode_terrarium(buffer.getvalue())
    assert decoded.shape == (256, 256) and decoded[0, 0] == pytest.approx(100.5)


def _plane_fetcher(z: int, x: int, y: int) -> np.ndarray:
    """Elevation rises 0.5 m per pixel eastwards and falls 2 m per pixel southwards (global pixel coordinates)."""
    cols = x * 256 + np.arange(256)
    rows = y * 256 + np.arange(256)
    return 0.5 * cols[None, :] - 2.0 * rows[:, None] + 5000.0


def test_elevation_window_has_the_right_orientation_and_scale() -> None:
    lat, lon = 12.0, 78.0
    window = terrain.elevation_window(lat, lon, window_km=10.0, grid=32, fetcher=_plane_fetcher)
    assert window.shape == (32, 32)
    assert (np.diff(window, axis=1) > 0).all()  # east is to the right
    assert (
        np.diff(window, axis=0) < 0
    ).all()  # row 0 is the north edge; this plane falls towards the south
    metres_per_pixel = (
        terrain.EARTH_CIRCUMFERENCE_M * np.cos(np.radians(lat)) / 2**terrain.ZOOM / terrain.TILE_PX
    )
    pixels = 10_000.0 / metres_per_pixel
    assert window[0, -1] - window[0, 0] == pytest.approx(
        0.5 * pixels, rel=0.02
    )  # the window spans 10 km
    assert window[0, 0] - window[-1, 0] == pytest.approx(
        2.0 * pixels, rel=0.02
    )  # 10 km north-south too
    centre = window[15:17, 15:17].mean()
    cx, cy = terrain.tile_coordinates(lat, lon, terrain.ZOOM)
    assert centre == pytest.approx(
        0.5 * cx * 256 - 2.0 * cy * 256 + 5000.0, abs=3.0
    )  # centred on the district


def test_elevation_window_returns_none_when_a_tile_is_unavailable_and_stats_ignore_spikes() -> None:
    def broken(z: int, x: int, y: int) -> np.ndarray:
        raise OSError("offline")

    assert terrain.elevation_window(12.0, 78.0, fetcher=broken) is None
    flat = np.full((32, 32), 100.0)
    flat[3, 3] = 1100.0  # a single-pixel spike
    stats = terrain.relief_stats(flat)
    assert stats["relief_m"] == pytest.approx(0.0) and stats["mean_elevation_m"] > 100.0
    ramp = np.tile(np.linspace(0.0, 100.0, 32), (32, 1))
    assert terrain.relief_stats(ramp)["relief_m"] == pytest.approx(90.0, abs=5.0)


@data_ready
def test_exported_v2_content_is_consistent_with_the_water_balance_and_the_reference_weather() -> (
    None
):
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    sc = data["scenarios"]
    assert data["meta"]["schema_version"] == 2 and "U.S. Geological Survey" in " ".join(
        data["meta"]["attribution"]
    )
    assert {name for name, c in data["crops"].items() if c["ponded"]} == {"rice"}
    for crop in data["crops"].values():
        growth, ripening = X.development_curves(crop["stage_days"])
        np.testing.assert_allclose(
            crop["development"], growth, atol=1e-3
        )  # the file rounds to three decimals
        np.testing.assert_allclose(crop["senescence"], ripening, atol=1e-3)
        assert crop["season_start"] in {"06-01", "11-01"}
    sat_synthetic = data["meta"]["is_synthetic_inputs"]["satellite_dataset"]
    for d in data["districts"]:
        assert d["elevation_source"] and (
            ("elevation_m" in d) == ("U.S. Geological Survey" in d["elevation_source"])
        )
        for crop in d["crops"]:
            ref = d["reference_daily"][crop]
            ndvi = d["reference_ndvi"][crop]
            assert ndvi is None or ndvi["is_synthetic"] == sat_synthetic
            base = sc[f"{d['id']}|{crop}|1|0|baseline"]
            for rain, temp in ((1, 0), (1.4, 2), (0.6, -2)):
                scenario = sc[f"{d['id']}|{crop}|{rain:g}|{temp:g}|optimized"]
                for week, record in enumerate(scenario["weekly"]):
                    days = slice(7 * week, 7 * week + 7)
                    assert record["rain_mm"] == pytest.approx(
                        rain * sum(ref["rain_mm"][days]), abs=0.6
                    )  # one-decimal rounding of 7 days
                    assert record["tmax_c"] == pytest.approx(
                        np.mean(ref["tmax_c"][days]) + temp, abs=0.1
                    )
            factors = np.array(base["stage_rel"])
            assert ((factors >= 0) & (factors <= 1)).all()
            ponded = data["crops"][crop]["ponded"]
            for strategy in ("rainfed", "baseline", "optimized"):
                scenario = sc[f"{d['id']}|{crop}|1|0|{strategy}"]
                assert ("pond_daily" in scenario) == ponded
                if ponded:
                    daily = scenario["pond_daily"]
                    assert max(daily) <= data["crops"][crop]["pond_max_mm"] + 0.05
                    for week, record in enumerate(scenario["weekly"]):
                        assert record["pond_mm"] == pytest.approx(daily[7 * week + 6], abs=0.06)
                else:
                    assert all(
                        r["pond_mm"] == 0 and r["seepage_mm"] == 0 for r in scenario["weekly"]
                    )
            if ponded:  # the 50 mm irrigation event on day 10 of the farmer baseline is visible in the standing water
                daily = base["pond_daily"]
                assert daily[10] > daily[9] or daily[9] >= data["crops"][crop]["pond_max_mm"] - 1


@data_ready
def test_ponded_rice_starts_from_the_initial_puddle_and_loses_water_to_seepage() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    rice_districts = [d for d in data["districts"] if "rice" in d["crops"]]
    assert rice_districts
    for d in rice_districts:
        rainfed = data["scenarios"][f"{d['id']}|rice|1|0|rainfed"]
        assert rainfed["pond_daily"][0] <= data["crops"]["rice"]["initial_pond_mm"] + 0.05
        wet = data["scenarios"][f"{d['id']}|rice|1|0|baseline"]
        assert (
            sum(w["seepage_mm"] for w in wet["weekly"]) > 50
        )  # seepage is a real loss while water stands


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_curve_and_pond_helpers_in_the_page(tmp_path) -> None:
    html = (SIM / "index.html").read_text(encoding="utf-8")
    start = html.index("/** Standing water (mm) on the day")
    end = html.index(
        "// ===================================================================================== three.js"
    )
    script = (
        "const clamp=(v,a=0,b=1)=>Math.min(b,Math.max(a,v));const lerp=(a,b,t)=>a+(b-a)*t;"
        + html[start:end]
        + "console.log(JSON.stringify({a:curveAt([0,1,2],1),b:curveAt([0,1,2],2.5),c:curveAt([0,1,2],99),d:curveAt([null,1,2],1),e:curveAt([null,null],1),"
        "f:pondToday({pond_daily:[5,6,7]},1,{pond_mm:9}),g:pondToday({pond_daily:[5,6,7]},99,{pond_mm:9}),h:pondToday({},1,{pond_mm:9})}));"
    )
    file = tmp_path / "helpers.mjs"
    file.write_text(script, encoding="utf-8")
    result = subprocess.run(["node", str(file)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "a": 0,
        "b": 1.5,
        "c": 2,
        "d": 1,
        "e": None,
        "f": 6,
        "g": 7,
        "h": 9,
    }
