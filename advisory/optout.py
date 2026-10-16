"""Opt-out list: recipients who replied STOP are never contacted again.

Only SHA-256 hashes of the phone numbers are stored (``advisory_optout.json`` in the state folder, ``data/interim`` by default), so the file does not
leak numbers; the hash is salted with ``OPTOUT_SALT`` from the environment when set.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path

from config import get_path

LOG = logging.getLogger(__name__)
_LOCK = threading.Lock()


def state_dir() -> Path:
    """Folder for the opt-out list and the send audit log (``ADVISORY_STATE_DIR`` or ``data/interim``)."""
    override = os.environ.get("ADVISORY_STATE_DIR")
    if override:
        path = Path(override)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return get_path("data_interim")


def _path() -> Path:
    return state_dir() / "advisory_optout.json"


def digest(number: str) -> str:
    """Stable, non-reversible key of a phone number (``whatsapp:`` prefix and spaces ignored)."""
    cleaned = number.strip().removeprefix("whatsapp:").replace(" ", "")
    salt = os.environ.get("OPTOUT_SALT", "")
    return hashlib.sha256(f"{salt}|{cleaned}".encode()).hexdigest()


def _load() -> set[str]:
    path = _path()
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")))


def is_opted_out(number: str) -> bool:
    """True if this number asked to stop receiving messages."""
    with _LOCK:
        return digest(number) in _load()


def opt_out(number: str) -> None:
    """Record that ``number`` must not be contacted again."""
    with _LOCK:
        keys = _load()
        keys.add(digest(number))
        _path().write_text(json.dumps(sorted(keys)), encoding="utf-8")
    LOG.info("recorded an opt-out")


def opt_in(number: str) -> None:
    """Remove ``number`` from the opt-out list (the person asked to restart)."""
    with _LOCK:
        keys = _load()
        keys.discard(digest(number))
        _path().write_text(json.dumps(sorted(keys)), encoding="utf-8")
    LOG.info("removed an opt-out")
