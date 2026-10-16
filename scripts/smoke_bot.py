"""End-to-end smoke test of the advisory webhook: starts uvicorn, calls it over HTTP, writes ``reports/bot_smoke.json``.

Checks ``/health``, a signed message ("Thanjavur rice"), a message in Hindi, a forged signature (must be 403) and the
unsigned-refusal default. Uses a throwaway auth token and a temporary state folder; nothing is sent to Twilio.

Run: ``python scripts/smoke_bot.py`` (exit code 1 on any failed check).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from twilio.request_validator import RequestValidator

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("smoke_bot")
TOKEN = "smoke-test-token"
SENDER = "whatsapp:+919876543210"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start(port: int, state: str, token: str) -> subprocess.Popen:
    env = {
        **os.environ,
        "ADVISORY_STATE_DIR": state,
        "TWILIO_AUTH_TOKEN": token,
        "DRY_RUN": "true",
        "PYTHONUTF8": "1",
    }
    env.pop("ALLOW_UNSIGNED_WEBHOOK", None)
    env.pop("PUBLIC_BASE_URL", None)
    return subprocess.Popen(
        [
            sys.executable,
            "-W",
            "ignore",
            "-m",
            "uvicorn",
            "advisory.bot:app",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait(port: int) -> None:
    for _ in range(90):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(1)
    raise RuntimeError("bot did not start")


def post(port: int, body: str, token: str | None) -> httpx.Response:
    url = f"http://127.0.0.1:{port}/whatsapp"
    form = {"From": SENDER, "Body": body}
    headers = (
        {"X-Twilio-Signature": RequestValidator(token).compute_signature(url, form)}
        if token
        else {}
    )
    return httpx.post(url, data=form, headers=headers, timeout=120)


def run() -> int:
    results: dict[str, object] = {}
    failures: list[str] = []

    def check(name: str, ok: bool, detail: object) -> None:
        results[name] = {"ok": ok, "detail": detail}
        LOG.info("%-28s %s", name, "ok" if ok else "FAIL")
        if not ok:
            failures.append(name)

    with tempfile.TemporaryDirectory() as state:
        port = free_port()
        server = start(port, state, TOKEN)
        try:
            wait(port)
            health = httpx.get(f"http://127.0.0.1:{port}/health", timeout=5).json()
            check("health", health.get("status") == "ok" and health.get("dry_run") is True, health)
            started = time.perf_counter()
            english = post(port, "Thanjavur rice", TOKEN)
            first = time.perf_counter() - started
            check(
                "signed_message_english",
                english.status_code == 200
                and "Thanjavur" in english.text
                and "<Message>" in english.text,
                english.text[:200],
            )
            started = time.perf_counter()
            hindi = post(port, "warangal cotton hindi", TOKEN)
            second = time.perf_counter() - started
            check(
                "signed_message_hindi",
                hindi.status_code == 200 and "Warangal" in hindi.text,
                hindi.text[:200],
            )
            check(
                "forged_signature_refused",
                post(port, "Thanjavur rice", "wrong-token").status_code == 403,
                None,
            )
            check(
                "missing_signature_refused",
                post(port, "Thanjavur rice", None).status_code == 403,
                None,
            )
            typo = post(port, "thanjvur rice", TOKEN)
            check("misspelt_district_is_understood", "Thanjavur" in typo.text, typo.text[:200])
            results["seconds"] = {
                "first_advisory_incl_model_load": round(first, 2),
                "second_advisory": round(second, 2),
            }
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
        # a bot without an auth token must refuse everything by default
        port = free_port()
        server = start(port, state, "")
        try:
            wait(port)
            refused = httpx.post(
                f"http://127.0.0.1:{port}/whatsapp",
                data={"From": SENDER, "Body": "help"},
                timeout=10,
            )
            check("unconfigured_bot_refuses", refused.status_code == 403, refused.text[:120])
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
    results["failures"] = failures
    (ROOT / "reports" / "bot_smoke.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(run())
