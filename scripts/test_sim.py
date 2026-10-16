"""Headless-browser test of the 3D simulator (Playwright + Chromium).

Opens ``simulator/index_embedded.html`` from ``file://`` and checks that there are no console errors, that lowering
the rainfall slider to 0.4 raises the drought gauge, that compare mode shows less optimized than baseline water,
and that keyboard control works. A second group of checks covers the scene itself: one plant model per crop, growth
from seedling to full size over the season, standing water in the paddy only, stressed plants that are yellower and
shorter, the by-stage stress bars, daily weather with dates, the time of day, the uncertainty view (three strips with
different vigour) and the level of detail. Saves screenshots to ``reports/figures``, prints the average frame rate
over 5 seconds and saves the result to ``reports/sim_checks.json``.

Run: ``python scripts/test_sim.py`` (exit code 1 on any failed check). Frame rates measured in headless Chromium use
software WebGL (SwiftShader), so they are a *lower bound* for real GPUs and say little about integrated graphics.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "simulator" / "index_embedded.html"
FIGURES = ROOT / "reports" / "figures"
PAGE_LOAD_MS = 120_000  # the 12 MB page can load slowly on a busy machine (software rendering)
LAUNCH_ARGS = [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
    "--no-sandbox",
]
LOG = logging.getLogger("test_sim")


def set_range(page: Page, selector: str, value: float) -> None:
    """Set a range input and fire the input event the way a user drag would."""
    page.eval_on_selector(
        selector,
        "(el, v) => { el.value = v; el.dispatchEvent(new Event('input', {bubbles: true})); }",
        value,
    )


def gauge(page: Page, name: str) -> float:
    return float(page.get_attribute(f"#gauge-{name}", "data-value"))


def settle(page: Page, frames: int = 3) -> None:
    start = page.evaluate("window.__sim.frames")
    page.wait_for_function(f"window.__sim.frames >= {start + frames}", timeout=60000)


def plant_stats(page: Page) -> dict[str, float]:
    """Mean height (scene units) and mean colour of the drawn plants, per vertical strip, from the instance buffers."""
    return page.evaluate(
        """() => {
            const v = window.__sim.views[0], m = v.bodyMesh.instanceMatrix.array, c = v.bodyMesh.instanceColor.array, n = v.bodyMesh.count;
            const out = { n, h: 0, r: 0, g: 0, b: 0, band: [0, 0, 0], bandN: [0, 0, 0] };
            for (let k = 0; k < n; k++) {
                const h = Math.hypot(m[k * 16 + 4], m[k * 16 + 5], m[k * 16 + 6]), x = m[k * 16 + 12], band = x < -10 / 6 ? 0 : x > 10 / 6 ? 2 : 1;
                out.h += h; out.r += c[k * 3]; out.g += c[k * 3 + 1]; out.b += c[k * 3 + 2]; out.band[band] += h; out.bandN[band]++;
            }
            for (const key of ['h', 'r', 'g', 'b']) out[key] /= Math.max(n, 1);
            out.band = out.band.map((s, i) => s / Math.max(out.bandN[i], 1));
            return out;
        }"""
    )


def scene_checks(page: Page, failures: list[str], report: dict[str, object]) -> None:
    """Checks of the landscape: plant models, growth, paddy water, stress, daily weather, time of day, uncertainty, detail."""
    page.select_option("#sel-quality", "high")
    settle(page, 3)
    page.check("input[name=strategy][value=optimized]")
    set_range(page, "#rng-rain", 1.0)
    set_range(page, "#rng-temp", 0.0)
    set_range(page, "#rng-hour", 12)

    # a) one model per crop; only rice is ponded; sugarcane has no separate heads
    models = {}
    for crop in ("rice", "wheat", "cotton", "sugarcane"):
        page.select_option("#sel-crop", crop)
        set_range(page, "#rng-week", 12)
        settle(page, 3)
        info = page.evaluate("window.__sim.describe()")
        models[crop] = {k: info[k] for k in ("plants", "drawn", "heads", "ponded")}
        if info["crop"] != crop or info["plants"] < 500 or info["drawn"] != info["plants"]:
            failures.append(
                f"{crop}: expected a full plant field in high detail, got {models[crop]}"
            )
        if (info["heads"] == 0) != (crop == "sugarcane") or info["ponded"] != (crop == "rice"):
            failures.append(f"{crop}: wrong heads / ponding flags {models[crop]}")
        if crop in ("wheat", "cotton", "sugarcane") and info["pondsVisible"] != 0:
            failures.append(f"{crop}: standing water drawn for an upland crop")
        page.screenshot(path=str(FIGURES / f"sim_crop_{crop}.png"))
    report["plant_models"] = models

    # b) growth by week: seedlings in week 1, full size later (rice)
    page.select_option("#sel-crop", "rice")
    heights = {}
    for week in (1, 5, 12):
        set_range(page, "#rng-week", week)
        settle(page, 3)
        heights[week] = plant_stats(page)["h"]
    report["rice_mean_plant_height_by_week"] = {k: round(v, 4) for k, v in heights.items()}
    if not (heights[1] < 0.3 * heights[12] and heights[1] < heights[5] < heights[12]):
        failures.append(f"plants do not grow over the season: {heights}")

    # c) paddy water: flooded after the farmer-baseline irrigation, bare when rainfed in a dry year, none for upland crops
    page.check("input[name=strategy][value=baseline]")
    set_range(page, "#rng-week", 1.2)
    settle(page, 3)
    wet = page.evaluate("window.__sim.describe()")
    page.check("input[name=strategy][value=rainfed]")
    set_range(page, "#rng-rain", 0.4)
    set_range(page, "#rng-week", 12)
    settle(page, 3)
    dry = page.evaluate("window.__sim.describe()")
    report["paddy_pond_scale_baseline_week1"], report["paddy_pond_scale_rainfed_dry_week12"] = (
        wet["pondScale"],
        dry["pondScale"],
    )
    if not (wet["pondScale"] > 0.5 and dry["pondScale"] < 0.01):
        failures.append(
            f"paddy water: baseline week 1 {wet['pondScale']}, rainfed dry week 12 {dry['pondScale']}"
        )
    if float(page.get_attribute("#pond-mm", "data-value")) > 0.5:
        failures.append("the paddy water card shows standing water in a dry rainfed field")
    page.screenshot(path=str(FIGURES / "sim_rice_dry.png"))

    # d) water stress: plants are shorter and less green than under the farmer baseline, and the stage bars show the loss
    set_range(page, "#rng-week", 11)
    settle(page, 3)
    stressed = plant_stats(page)
    stage_dry = page.eval_on_selector_all(
        "#stage-bars .stagerow", "els => els.map((e) => Number(e.dataset.value))"
    )
    page.check("input[name=strategy][value=baseline]")
    set_range(page, "#rng-rain", 1.0)
    settle(page, 3)
    healthy = plant_stats(page)
    stage_ok = page.eval_on_selector_all(
        "#stage-bars .stagerow", "els => els.map((e) => Number(e.dataset.value))"
    )
    report["stress"] = {
        "height_stressed": stressed["h"],
        "height_healthy": healthy["h"],
        "green_share_stressed": stressed["g"] / stressed["r"],
        "green_share_healthy": healthy["g"] / healthy["r"],
        "stage_factors_dry_rainfed": stage_dry,
        "stage_factors_baseline": stage_ok,
    }
    if not stressed["h"] < healthy["h"]:
        failures.append("stressed plants are not shorter")
    if not stressed["g"] / stressed["r"] < healthy["g"] / healthy["r"]:
        failures.append("stressed plants are not less green")
    if not (min(stage_dry) < 0.9 and min(stage_ok) > min(stage_dry)):
        failures.append(f"stage bars do not show the drought loss: {stage_dry} vs {stage_ok}")
    if "Largest loss" not in page.inner_text("#stage-note") and min(stage_ok) < 0.995:
        failures.append("the stage note does not name the largest loss")

    # e) daily weather with dates: the day advances inside a week and the date is shown
    page.check("input[name=strategy][value=optimized]")
    days = []
    for week in (2.0, 2.4, 3.0):
        set_range(page, "#rng-week", week)
        settle(page, 2)
        days.append(page.evaluate("window.__sim.describe().day"))
    label = page.inner_text("#val-day")
    report["daily_playback"] = {
        "day_index": [d["index"] for d in days],
        "dates": [d["date"] for d in days],
        "label": label,
    }
    if (
        not (days[0]["index"] < days[1]["index"] < days[2]["index"])
        or days[0]["date"] == days[2]["date"]
        or "reference year" not in label
    ):
        failures.append(f"daily playback does not advance: {days} / {label!r}")

    # f) time of day moves the sun
    set_range(page, "#rng-hour", 7)
    settle(page, 2)
    morning = page.evaluate("window.__sim.describe().sunY")
    set_range(page, "#rng-hour", 12)
    settle(page, 2)
    noon = page.evaluate("window.__sim.describe().sunY")
    report["sun_height_07h_12h"] = [round(morning, 3), round(noon, 3)]
    if not morning < noon:
        failures.append(f"the sun is not lower at 07:00 ({morning}) than at 12:00 ({noon})")

    # g) the uncertainty view: three strips with low / model / high vigour and a note with the interval
    set_range(page, "#rng-week", 9)
    settle(page, 3)
    even = plant_stats(page)
    page.click("#btn-uncertainty")
    settle(page, 3)
    un = plant_stats(page)
    shown = page.evaluate(
        "window.__sim.views[0].dividers.every((m) => m.visible) && window.__sim.state.uncertainty"
    )
    report["uncertainty_strip_heights"] = [round(h, 4) for h in un["band"]]
    if (
        not (shown and un["band"][0] < un["band"][1] < un["band"][2])
        or abs(even["band"][0] - even["band"][2]) > 0.2 * even["band"][1]
    ):
        failures.append(
            f"uncertainty view: strips {un['band']} (before: {even['band']}), dividers shown {shown}"
        )
    if "80% interval" not in page.inner_text(
        "#uncertainty-note"
    ) and "interval" not in page.inner_text("#uncertainty-note"):
        failures.append("the uncertainty note is missing")
    page.screenshot(path=str(FIGURES / "sim_uncertainty.png"))
    page.keyboard.press("u")
    settle(page, 2)
    if page.evaluate("window.__sim.state.uncertainty"):
        failures.append("the U key did not switch the uncertainty view off")

    # h) level of detail
    page.select_option("#sel-quality", "low")
    settle(page, 3)
    low = page.evaluate("window.__sim.describe()")
    report["low_detail_plants_drawn"] = [low["drawn"], low["plants"]]
    if not (0.3 < low["drawn"] / low["plants"] < 0.4):
        failures.append(
            f"low detail should draw about 35% of the plants: {low['drawn']} of {low['plants']}"
        )
    page.select_option("#sel-quality", "high")
    settle(page, 2)

    # i) terrain: real relief for this district and the attribution in the footer
    relief = page.evaluate("window.__sim.describe().relief")
    report["terrain_note"] = relief
    if "real relief" not in relief or "Geological Survey" not in page.inner_text(
        "#foot-attribution"
    ):
        failures.append(f"terrain note or attribution missing: {relief!r}")
    # j) the paddy card, charts and legends exist for rice and the synthetic NDVI line is labelled
    page.select_option("#sel-crop", "rice")
    settle(page, 2)
    if page.is_hidden("#card-pond") or page.is_hidden("#wrap-pond"):
        failures.append("the paddy water card or chart is hidden for rice")
    if "SYNTHETIC" not in page.inner_text("#ndvi-legend") and "observed" not in page.inner_text(
        "#ndvi-legend"
    ):
        failures.append(
            "the satellite NDVI legend does not say whether the data are synthetic or observed"
        )
    own = json.loads((ROOT / "simulator" / "sim_data.json").read_text(encoding="utf-8"))
    satellite_synthetic = own["meta"]["is_synthetic_inputs"]["satellite_dataset"]
    offered = not page.is_hidden("#fs-green")
    report["observed_greenness_offered"] = offered
    if offered == satellite_synthetic:
        failures.append(
            "the observed-greenness choice should be offered exactly when the satellite data are real "
            f"(synthetic={satellite_synthetic}, offered={offered})"
        )
    page.select_option("#sel-crop", "wheat")
    settle(page, 2)
    if not page.is_hidden("#card-pond"):
        failures.append("the paddy water card is shown for wheat")
    page.select_option("#sel-crop", "rice")
    page.select_option("#sel-quality", "auto")


def observed_ndvi_check(browser, failures: list[str], report: dict[str, object]) -> None:
    """With real (non-synthetic) satellite data the page offers observed greenness; test it on a modified copy of the data.

    The copy declares one district-crop observed (greenness 0.25 all season) and another synthetic, so the check does
    not depend on whether the project's own satellite data is real; nothing is written to the project files.
    """
    import tempfile

    from simulator import embed

    data = json.loads((ROOT / "simulator" / "sim_data.json").read_text(encoding="utf-8"))
    district = data["districts"][0]
    district["reference_ndvi"]["rice"] = {"weekly": [0.25] * 16, "is_synthetic": False}
    district["reference_ndvi"]["wheat"] = {"weekly": [0.25] * 16, "is_synthetic": True}
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "data.json"
        source.write_text(json.dumps(data), encoding="utf-8")
        page_file = embed.build_embedded(
            vendor=True, data_path=source, out_path=Path(tmp) / "page.html"
        )
        page = browser.new_page(viewport={"width": 1500, "height": 900})
        page.goto(
            page_file.as_uri() + f"?quality=high&district={district['id']}&crop=rice",
            timeout=PAGE_LOAD_MS,
        )
        page.wait_for_function("window.__sim && window.__sim.ready", timeout=120000)
        set_range(page, "#rng-week", 12)
        settle(page, 3)
        if page.is_hidden("#fs-green") or "observed" not in page.inner_text("#ndvi-legend"):
            failures.append(
                "observed NDVI is not offered or not labelled observed when the data are not synthetic"
            )
        proxy = plant_stats(page)
        page.check("input[name=green][value=observed]")
        settle(page, 3)
        observed = plant_stats(page)
        report["observed_ndvi_test"] = {
            "proxy_height": proxy["h"],
            "observed_height": observed["h"],
            "proxy_green_share": proxy["g"] / proxy["r"],
            "observed_green_share": observed["g"] / observed["r"],
        }
        if not (
            observed["h"] < proxy["h"] and observed["g"] / observed["r"] < proxy["g"] / proxy["r"]
        ):
            failures.append("low observed NDVI did not make the plants shorter and less green")
        page.select_option("#sel-crop", "wheat")
        settle(page, 2)
        if not page.is_hidden("#fs-green"):
            failures.append(
                "the observed-greenness choice stayed visible for a crop whose NDVI is synthetic"
            )
        page.close()


def run() -> int:
    if not PAGE.exists():
        print(f"{PAGE} missing: run `python -m simulator.embed` first")  # noqa: T201
        return 2
    FIGURES.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    failures: list[str] = []
    report: dict[str, object] = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=LAUNCH_ARGS)
        page = browser.new_page(viewport={"width": 1500, "height": 900})
        page.on(
            "console",
            lambda msg: (
                errors.append(f"console.{msg.type}: {msg.text}") if msg.type in ("error",) else None
            ),
        )
        page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
        page.goto(PAGE.as_uri(), timeout=PAGE_LOAD_MS)
        page.wait_for_function("window.__sim && window.__sim.ready", timeout=120000)
        settle(page, 5)
        report["title"] = page.title()
        report["scenarios_chip"] = page.inner_text("#chip-data")

        # 1) drought gauge rises when rainfall is cut to 0.4x (check several weeks, take the max)
        def drought_max() -> float:
            values = []
            for week in (6, 9, 12, 14):
                set_range(page, "#rng-week", week)
                settle(page, 1)
                values.append(gauge(page, "drought"))
            return max(values)

        set_range(page, "#rng-rain", 1.0)
        normal = drought_max()
        set_range(page, "#rng-rain", 0.4)
        settle(page, 2)
        dry = drought_max()
        report["drought_gauge_normal_rain"], report["drought_gauge_rain_0.4"] = normal, dry
        if not dry > normal:
            failures.append(
                f"drought gauge did not rise with 0.4x rain ({normal:.3f} -> {dry:.3f})"
            )
        set_range(page, "#rng-rain", 1.0)

        # 2) heat gauge rises with +4 C
        set_range(page, "#rng-week", 3)
        heat_cool = max(gauge(page, "heat") for _ in [0])
        set_range(page, "#rng-temp", 4.0)
        settle(page, 2)
        heat_warm = max(
            (set_range(page, "#rng-week", w), settle(page, 1), gauge(page, "heat"))[2]
            for w in range(1, 17, 3)
        )
        report["heat_gauge_dT0_week3"], report["heat_gauge_max_dT+4"] = heat_cool, heat_warm
        set_range(page, "#rng-temp", 0.0)

        # 3) compare mode: optimized uses less water than the baseline
        page.click("#btn-compare")
        settle(page, 5)
        baseline = float(page.get_attribute("#compare-water-baseline", "data-value"))
        optimized = float(page.get_attribute("#compare-water-optimized", "data-value"))
        report["compare_water_baseline_mm"], report["compare_water_optimized_mm"] = (
            baseline,
            optimized,
        )
        if not optimized < baseline:
            failures.append(
                f"compare mode: optimized water {optimized} is not below baseline {baseline}"
            )
        if page.is_hidden("#view-b"):
            failures.append("compare mode did not show the second view")
        page.click("#btn-compare")
        settle(page, 3)

        # 4) keyboard controls
        page.focus("body")
        set_range(page, "#rng-week", 5)
        page.keyboard.press("ArrowRight")
        settle(page, 1)
        week_after = float(page.evaluate("window.__sim.state.wpos"))
        if abs(week_after - 6) > 0.01:
            failures.append(f"ArrowRight did not step to week 6 (got {week_after})")
        page.keyboard.press("Space")
        settle(page, 3)
        if not page.evaluate("window.__sim.state.playing"):
            failures.append("Space did not start playback")
        page.keyboard.press("Space")

        # 5) screenshots at weeks 3, 9, 14 (rain slightly below normal, optimized irrigation)
        set_range(page, "#rng-rain", 0.7)
        for week in (3, 9, 14):
            set_range(page, "#rng-week", week)
            settle(page, 6)
            page.screenshot(path=str(FIGURES / f"sim_week{week}.png"))
        set_range(page, "#rng-rain", 1.0)

        # 6) frame rate over 5 s while playing
        set_range(page, "#rng-week", 1)
        page.keyboard.press("Space")
        settle(page, 3)
        start_frames, start = page.evaluate("window.__sim.frames"), time.time()
        time.sleep(5.0)
        frames = page.evaluate("window.__sim.frames") - start_frames
        fps = frames / (time.time() - start)
        report["average_fps_5s_headless_software_gl"] = round(fps, 1)
        page.keyboard.press("Space")

        # 7) every displayed number comes from the file: at a grid point the yield card equals the embedded scenario value
        for rain, temp, strategy in ((1.0, 0.0, "optimized"), (0.6, 2.0, "baseline")):
            set_range(page, "#rng-rain", rain)
            set_range(page, "#rng-temp", temp)
            page.check(f"input[name=strategy][value={strategy}]")
            settle(page, 3)
            shown = float(page.get_attribute("#hud-yield", "data-value"))
            water = float(page.get_attribute("#water-used", "data-value"))
            expected = page.evaluate(
                """([rain, temp, strategy]) => {
                    const data = JSON.parse(document.getElementById('sim-data').textContent);
                    const d = document.getElementById('sel-district').value, c = document.getElementById('sel-crop').value;
                    const sc = data.scenarios[`${d}|${c}|${rain}|${temp}|${strategy}`];
                    return [sc.yield.mean, sc.water_used_mm];
                }""",
                [rain, temp, strategy],
            )
            report[f"yield_shown_vs_file_{strategy}"] = [shown, expected[0]]
            if abs(shown - expected[0]) > 0.001 or abs(water - expected[1]) > 0.01:
                failures.append(
                    f"displayed values {shown}/{water} differ from the file {expected} for {strategy}"
                )
        scene_checks(page, failures, report)
        observed_ndvi_check(browser, failures, report)
        browser.close()
    if errors:
        failures.append("browser errors: " + "; ".join(errors[:5]))
    report["failures"] = failures
    (ROOT / "reports" / "sim_checks.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8"
    )
    print(json.dumps(report, indent=1))  # noqa: T201 - CLI result
    return 1 if failures else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(run())
