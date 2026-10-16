"""Fail if a file contains something that looks like a credential (used by the pre-commit hook).

Patterns: Twilio account and API-key SIDs and a 32-hex auth token assigned to ``TWILIO_AUTH_TOKEN``, private-key
headers, AWS access keys, Google API keys and GitHub tokens. Matches are reported as ``file:line: rule`` without the
secret itself. ``.env.example`` is skipped (it holds placeholders only) and binary files are ignored.

Run: ``python scripts/check_secrets.py FILE...`` (exit code 1 when something suspicious is found).
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from pathlib import Path

RULES: dict[str, re.Pattern[str]] = {
    "Twilio account SID": re.compile(r"\bAC[0-9a-f]{32}\b"),
    "Twilio API key SID": re.compile(r"\bSK[0-9a-f]{32}\b"),
    "Twilio auth token": re.compile(r"TWILIO_AUTH_TOKEN\s*[=:]\s*['\"]?[0-9a-f]{32}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
}
SKIP_NAMES = {".env.example"}


def scan_text(text: str) -> list[tuple[int, str]]:
    """``(line number, rule name)`` for every suspicious line of ``text``."""
    hits = []
    for number, line in enumerate(text.splitlines(), start=1):
        hits.extend((number, name) for name, pattern in RULES.items() if pattern.search(line))
    return hits


def scan_files(paths: Iterable[Path]) -> list[str]:
    """Findings (``file:line: rule``) over many files; unreadable and binary files are skipped."""
    findings = []
    for path in paths:
        if path.name in SKIP_NAMES or not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            continue
        findings.extend(
            f"{path}:{n}: {rule}" for n, rule in scan_text(data.decode("utf-8", "replace"))
        )
    return findings


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    findings = scan_files(Path(p) for p in (argv if argv is not None else sys.argv[1:]))
    for finding in findings:
        print(finding)  # noqa: T201 - CLI output
    if findings:
        print("possible credentials found: remove them and use .env (git-ignored)")  # noqa: T201
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
