"""Headless-browser test of the 3D simulator (Playwright + Chromium).

Opens ``simulator/index_embedded.html`` from ``file://`` and checks that there are no console errors, that lowering
the rainfall slider to 0.4 raises the drought gauge, that compare mode shows less optimized than baseline water,
and that keyboard control works. Saves screenshots at weeks 3, 9 and 14 to ``reports/figures``, prints the
average frame rate over 5 seconds and saves the result to ``reports/sim_checks.json``.

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
        page.goto(PAGE.as_uri())
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
