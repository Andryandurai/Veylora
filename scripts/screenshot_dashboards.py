"""Browser check of the Streamlit dashboards (Playwright + headless Chromium).

Starts ``streamlit run dashboards/app.py`` on a free port, opens every page twice (the second visit measures the
median time to content over three warm-cache visits), fails on console errors or Streamlit exception boxes, checks that the 3D
simulator iframe loads with the sidebar's district preselected, and saves screenshots to
``reports/figures/dash_*.png`` and the timings to ``reports/dashboard_checks.json``.

Run: ``python scripts/screenshot_dashboards.py`` (exit code 1 when a check fails).
"""

from __future__ import annotations

import json
import logging
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"
LOG = logging.getLogger("screenshot_dashboards")
LAUNCH_ARGS = [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
    "--no-sandbox",
]
WARM_BUDGET_S = 3.0
WARM_VISITS = (
    3  # the budget applies to the median, one slow visit on a busy machine is not a regression
)
IGNORED_PROBES = re.compile(r"/_stcore/(health|host-config)$")

# name, URL path, text that proves the page rendered its content
PAGES = [
    ("home", "/", "Measured results"),
    ("weather", "/weather", "Hazard probabilities"),
    ("yield", "/yield", "Which kharif crop pays best here?"),
    ("risk_map", "/risk-map", "Districts at low risk"),
    ("advisory", "/advisory", "Why this message"),
    ("simulator", "/simulator", "Open the simulator in its own tab"),
    ("methodology", "/methodology", "Data and provenance"),
]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_server(port: int, timeout_s: float = 90.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/_stcore/health", timeout=2
            ) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("streamlit did not start")


def load(page: Page, base: str, path: str, marker: str) -> float:
    """Open a page and return the seconds until ``marker`` is visible and the script has stopped running."""
    start = time.perf_counter()
    page.goto(base + path, wait_until="domcontentloaded")
    page.get_by_text(marker).first.wait_for(state="visible", timeout=180000)
    page.wait_for_function(
        "() => document.querySelector('[data-testid=\"stApp\"]')?.getAttribute('data-test-script-state') === 'notRunning'",
        timeout=180000,
    )
    return time.perf_counter() - start


def run() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "dashboards/app.py",
            "--server.port",
            str(port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    failures: list[str] = []
    report: dict[str, object] = {"pages": {}, "warm_budget_s": WARM_BUDGET_S}
    try:
        wait_for_server(port)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=LAUNCH_ARGS)
            page = browser.new_page(viewport={"width": 1500, "height": 1900})
            errors: list[str] = []
            # Streamlit probes /<page>/_stcore/health and /host-config on deep links and logs a harmless 404 for
            # each; failed loads are therefore judged by URL (below) rather than by the URL-less console message.
            page.on(
                "console",
                lambda msg: (
                    errors.append(f"console.{msg.type}: {msg.text}")
                    if msg.type == "error" and not msg.text.startswith("Failed to load resource")
                    else None
                ),
            )
            page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
            page.on(
                "response",
                lambda resp: (
                    errors.append(f"HTTP {resp.status} {resp.url}")
                    if resp.status >= 400 and not IGNORED_PROBES.search(resp.url)
                    else None
                ),
            )
            for name, path, marker in PAGES:
                errors.clear()
                cold = load(page, base, path, marker)
                warm = statistics.median(load(page, base, path, marker) for _ in range(WARM_VISITS))
                exceptions = page.locator('[data-testid="stException"]').count()
                page.screenshot(path=str(FIGURES / f"dash_{name}.png"), full_page=True)
                info = {
                    "cold_s": round(cold, 2),
                    "warm_s": round(warm, 2),
                    "exception_boxes": exceptions,
                    "console_errors": list(errors),
                }
                report["pages"][name] = info
                LOG.info("%-12s cold %.1fs warm %.1fs", name, cold, warm)
                if exceptions:
                    failures.append(f"{name}: {exceptions} Streamlit exception box(es)")
                if errors:
                    failures.append(f"{name}: console errors {errors[:3]}")
                if warm > WARM_BUDGET_S:
                    failures.append(f"{name}: warm load {warm:.1f}s exceeds {WARM_BUDGET_S:.0f}s")
            report["simulator_embed"] = check_simulator(page, base, failures)
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
    report["failures"] = failures
    (ROOT / "reports" / "dashboard_checks.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    for failure in failures:
        LOG.error("FAIL %s", failure)
    LOG.info(
        "%s", "all dashboard checks passed" if not failures else f"{len(failures)} check(s) failed"
    )
    return 1 if failures else 0


def check_simulator(page: Page, base: str, failures: list[str]) -> dict[str, object]:
    """The simulator iframe must load its scene and honour the district chosen in the sidebar."""
    load(page, base, "/simulator", "Open the simulator in its own tab")
    frame = None
    for _ in range(120):
        frame = next((f for f in page.frames if "simulator.html" in f.url), None)
        if frame is not None:
            break
        time.sleep(0.5)
    if frame is None:
        failures.append("simulator: iframe with simulator.html not found")
        return {"loaded": False}
    try:
        frame.wait_for_function("window.__sim && window.__sim.ready", timeout=120000)
    except Exception as exc:  # report, do not crash the run
        failures.append(f"simulator: scene did not become ready ({exc.__class__.__name__})")
        return {"loaded": False, "url": frame.url}
    selected = frame.eval_on_selector("#sel-district", "el => el.value")
    expected = frame.url.split("district=")[-1].split("&")[0]
    if selected != expected:
        failures.append(
            f"simulator: district {selected!r} does not match URL parameter {expected!r}"
        )
    page.screenshot(path=str(FIGURES / "dash_simulator_loaded.png"), full_page=False)
    return {"loaded": True, "url": frame.url, "district": selected}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(run())
