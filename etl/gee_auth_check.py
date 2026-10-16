"""Check Google Earth Engine authentication and print clear next steps if it is missing.

Run: ``python -m etl.gee_auth_check``. Exit code 0 means Earth Engine is ready; 1 means it is not.
"""

from __future__ import annotations

import logging
import os
import sys

from config import load_env

LOG = logging.getLogger(__name__)

INSTRUCTIONS = """\
Google Earth Engine is not ready. One-time setup:
  1. Register a (free, non-commercial) Cloud project for Earth Engine:
       https://console.cloud.google.com/earth-engine
  2. Authenticate this machine (opens a browser):
       python -m ee authenticate        (or:  earthengine authenticate)
  3. Put the project id in .env:
       GEE_PROJECT=your-cloud-project-id
  4. Re-run:  python -m etl.gee_auth_check
Until then the pipeline uses clearly flagged synthetic satellite data (is_synthetic=True).
Details: {detail}"""


def check_auth() -> tuple[bool, str]:
    """Try to initialise Earth Engine and run a trivial server-side computation.

    Returns:
        (ok, message) where message is a human-readable status or the failure reason.
    """
    load_env()
    project = os.environ.get("GEE_PROJECT", "").strip()
    try:
        import ee
    except ImportError:
        return False, "the earthengine-api package is not installed"
    if not project:
        return False, "GEE_PROJECT is empty in .env"
    try:
        ee.Initialize(project=project)
        value = ee.Number(1).add(1).getInfo()
    except Exception as exc:  # ee raises several unrelated exception types for auth problems
        return False, f"{type(exc).__name__}: {str(exc)[:200]}"
    if value != 2:
        return False, "Earth Engine answered a test computation incorrectly"
    return True, f"Earth Engine initialised for project {project!r}"


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ok, message = check_auth()
    if ok:
        print(message)  # noqa: T201 - CLI summary
        return 0
    print(INSTRUCTIONS.format(detail=message))  # noqa: T201 - CLI summary
    return 1


if __name__ == "__main__":
    sys.exit(main())
