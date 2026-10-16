"""Build ``simulator/index_embedded.html``: ``index.html`` with the data (and Three.js) inlined.

The embedded file opens by double-click (``file://``) and works offline:

* ``sim_data.json`` is placed in ``<script id="sim-data" type="application/json">``;
* Three.js r160.1 (MIT licence, downloaded once from jsDelivr into ``simulator/vendor/``) is placed in
  ``<script id="three-src" type="text/plain">`` and imported through a Blob URL at start-up.

Usage: ``python -m simulator.embed`` (add ``--no-vendor`` to keep loading Three.js from the CDN).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import requests

from config import ROOT

LOG = logging.getLogger(__name__)

SIM_DIR = ROOT / "simulator"
VENDOR_DIR = SIM_DIR / "vendor"
THREE_VERSION = "0.160.1"
THREE_URL = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/build/three.module.min.js"
THREE_FILE = VENDOR_DIR / "three.module.min.js"
MARKER = '<script type="module">'


def vendor_three(refresh: bool = False) -> Path:
    """Download Three.js once and cache it under simulator/vendor (with its SHA-256 and licence note)."""
    VENDOR_DIR.mkdir(exist_ok=True)
    if THREE_FILE.exists() and not refresh:
        return THREE_FILE
    response = requests.get(THREE_URL, timeout=120)
    response.raise_for_status()
    THREE_FILE.write_bytes(response.content)
    digest = hashlib.sha256(response.content).hexdigest()
    (VENDOR_DIR / "THREE_NOTICE.txt").write_text(
        f"three.module.min.js, three@{THREE_VERSION}, downloaded from {THREE_URL}\nsha256 {digest}\n"
        "Three.js is MIT licensed, Copyright (c) 2010-2024 three.js authors. https://github.com/mrdoob/three.js\n",
        encoding="utf-8",
    )
    LOG.info("vendored %s (%d KB)", THREE_FILE.name, len(response.content) // 1024)
    return THREE_FILE


def build_embedded(
    vendor: bool = True, data_path: Path | None = None, out_path: Path | None = None
) -> Path:
    """Write the single-file page and return its path."""
    html = (SIM_DIR / "index.html").read_text(encoding="utf-8")
    data = (data_path or SIM_DIR / "sim_data.json").read_text(encoding="utf-8")
    if MARKER not in html:
        raise ValueError("index.html has no module script marker")
    safe_data = data.replace(
        "</", "<\\/"
    )  # "<\/" is a valid JSON escape for "/" and cannot close the script tag
    blocks = [f'<script id="sim-data" type="application/json">{safe_data}</script>']
    if vendor:
        source = vendor_three().read_text(encoding="utf-8")
        if "</script" in source.lower():
            raise ValueError("vendored Three.js contains '</script' and cannot be inlined")
        blocks.append(f'<script id="three-src" type="text/plain">{source}</script>')
    page = html.replace(MARKER, "\n".join(blocks) + "\n" + MARKER, 1)
    target = out_path or SIM_DIR / "index_embedded.html"
    target.write_text(page, encoding="utf-8")
    LOG.info(
        "wrote %s (%.1f MB, three.js %s)",
        target,
        len(page.encode()) / 1e6,
        "inlined" if vendor else "from CDN",
    )
    return target


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--no-vendor",
        action="store_true",
        help="do not inline Three.js (the page then needs internet)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_embedded(vendor=not args.no_vendor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
