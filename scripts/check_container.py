"""Clean-container verification: build the image from scratch, start the app, exercise the models, run the tests.

Steps (each recorded in ``reports/container_checks.json`` with its duration):

1. ``docker build`` of the serving image (``requirements.txt``);
2. start it, wait for Streamlit's health endpoint, confirm the process is not root, restore the demo bundle, call
   ``predict_yield``, ``risk_forecast`` and ``compose`` inside the container and record the memory in use;
3. build the full image (``requirements.lock``) and run the whole test suite inside it, on the compose network so that the
   warehouse tests can reach the ``db`` service when it is up (they skip otherwise).

Run: ``python scripts/check_container.py [--skip-full]`` (needs Docker; the first build downloads several hundred MB).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("check_container")
SERVING, FULL, NAME, PORT = "climateresilience:latest", "climateresilience:full", "cr-check", 8598
NETWORK = "climate-resilience_default"
SMOKE = """
import json, warnings
warnings.filterwarnings('ignore')
from analytics.predictive.predict import predict_yield
from analytics.predictive.extreme_weather import risk_forecast
from advisory.generate import compose
y = predict_yield('tamil_nadu__thanjavur', 'rice', 'kharif')
r = risk_forecast('telangana__warangal', '2003-06-15')
a = compose('telangana__warangal', 'cotton', '2003-06-15', 'hi')
print(json.dumps({'yield_t_ha': round(y['yield_t_ha'], 3), 'level': r['level'], 'advisory_chars': len(a.text), 'messages': a.messages}))
"""


def run(cmd: list[str], timeout: int = 3600) -> subprocess.CompletedProcess[str]:
    """Run a command and capture its output (never raises on a non-zero exit)."""
    return subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False
    )


def step(
    report: dict[str, object], name: str, ok: bool, started: float, detail: object = None
) -> None:
    """Record one step."""
    report[name] = {"ok": ok, "seconds": round(time.time() - started, 1), "detail": detail}
    LOG.info("%-26s %s (%.0fs)", name, "ok" if ok else "FAIL", time.time() - started)


def wait_health(timeout_s: int = 120) -> bool:
    """True once Streamlit's health endpoint answers."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/_stcore/health", timeout=3
            ) as response:
                if response.status == 200:
                    return True
        except OSError:
            time.sleep(2)
    return False


def check_serving(report: dict[str, object]) -> None:
    """Steps 1 and 2."""
    started = time.time()
    build = run(["docker", "build", "-t", SERVING, "."])
    size = run(["docker", "image", "inspect", SERVING, "--format", "{{.Size}}"]).stdout.strip()
    step(
        report,
        "build_serving_image",
        build.returncode == 0,
        started,
        {
            "image_mb": round(int(size) / 1e6) if size.isdigit() else None,
            "tail": build.stdout[-300:] + build.stderr[-300:],
        }
        if build.returncode
        else {"image_mb": round(int(size) / 1e6)},
    )
    if build.returncode:
        return
    run(["docker", "rm", "-f", NAME])
    started = time.time()
    up = run(["docker", "run", "-d", "--name", NAME, "-p", f"{PORT}:8501", SERVING])
    healthy = up.returncode == 0 and wait_health()
    step(
        report, "app_health_endpoint", healthy, started, up.stderr[-200:] if up.returncode else None
    )
    if not healthy:
        run(["docker", "rm", "-f", NAME])
        return
    started = time.time()
    user = run(["docker", "exec", NAME, "id", "-u"]).stdout.strip()
    step(report, "runs_as_non_root", user.isdigit() and int(user) != 0, started, {"uid": user})
    started = time.time()
    restore = run(["docker", "exec", NAME, "python", "-m", "config.bundle", "restore"])
    step(
        report,
        "restore_demo_bundle",
        restore.returncode == 0,
        started,
        (restore.stdout + restore.stderr)[-200:],
    )
    started = time.time()
    smoke = run(["docker", "exec", NAME, "python", "-W", "ignore", "-c", SMOKE])
    lines = [line for line in smoke.stdout.splitlines() if line.startswith("{")]
    step(
        report,
        "models_work_in_container",
        smoke.returncode == 0 and bool(lines),
        started,
        json.loads(lines[-1]) if lines else smoke.stderr[-300:],
    )
    memory = run(
        ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", NAME]
    ).stdout.strip()
    report["memory_in_use_after_smoke"] = memory
    run(["docker", "rm", "-f", NAME])


def check_full(report: dict[str, object]) -> None:
    """Step 3."""
    started = time.time()
    build = run(
        ["docker", "build", "--build-arg", "REQUIREMENTS=requirements.lock", "-t", FULL, "."]
    )
    step(
        report,
        "build_full_image",
        build.returncode == 0,
        started,
        None if build.returncode == 0 else build.stdout[-300:] + build.stderr[-300:],
    )
    if build.returncode:
        return
    started = time.time()
    network = (
        [
            "--network",
            NETWORK,
            "-e",
            "DB_URL=postgresql+psycopg://postgres:postgres@db:5432/postgres",
        ]
        if NETWORK in run(["docker", "network", "ls", "--format", "{{.Name}}"]).stdout.split()
        else []
    )
    tests = run(
        [
            "docker",
            "run",
            "--rm",
            *network,
            FULL,
            "sh",
            "-c",
            "python -m config.bundle restore && python -m pytest -q -p no:cacheprovider -W ignore -o addopts=",
        ]
    )
    summary = re.findall(r"^(\d+ (?:passed|failed|error).*)$", tests.stdout, re.M)
    step(
        report,
        "full_test_suite_in_container",
        tests.returncode == 0,
        started,
        {
            "summary": summary[-1] if summary else tests.stdout[-300:],
            "database_reachable": bool(network),
        },
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-full", action="store_true", help="only the serving image")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report: dict[str, object] = {}
    check_serving(report)
    if not args.skip_full:
        check_full(report)
    (ROOT / "reports" / "container_checks.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    failed = [k for k, v in report.items() if isinstance(v, dict) and not v.get("ok", True)]
    LOG.info("%s", "all container checks passed" if not failed else f"failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
