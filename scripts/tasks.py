"""Small cross-platform helpers called from the Makefile (no shell-specific syntax needed)."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("tasks")

CACHE_DIRS = (".pytest_cache", ".ruff_cache", "build", "climateresilience_india.egg-info")


def ensure_env() -> None:
    """Create .env from .env.example if it does not exist yet (never overwrites)."""
    env, example = ROOT / ".env", ROOT / ".env.example"
    if env.exists():
        LOG.info(".env already exists, leaving it untouched")
        return
    shutil.copyfile(example, env)
    LOG.info("created .env from .env.example - edit it to add real credentials")


def clean() -> None:
    """Remove caches, bytecode and build artefacts. Data and models are left alone."""
    for name in CACHE_DIRS:
        shutil.rmtree(ROOT / name, ignore_errors=True)
    for pycache in ROOT.rglob("__pycache__"):
        if ".venv" in pycache.parts:
            continue
        shutil.rmtree(pycache, ignore_errors=True)
    LOG.info("cleaned caches (data/ and models/ untouched)")


COMMANDS = {"ensure-env": ensure_env, "clean": clean}


def main(argv: list[str]) -> int:
    """Dispatch a single sub-command."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(argv) != 2 or argv[1] not in COMMANDS:
        LOG.error("usage: tasks.py {%s}", "|".join(COMMANDS))
        return 2
    COMMANDS[argv[1]]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
