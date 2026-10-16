"""Start the dashboards in a chosen design: ``python -m dashboards.serve --theme almanac --port 8502``.

The design is fixed when the server starts (Streamlit reads its own theme options then), so two designs can run
side by side on different ports. ``--theme field`` is the default design of ``make dashboards``.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from dashboards.components import theme

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def command(port: int) -> list[str]:
    """The ``streamlit run`` command line for the dashboards on ``port``."""
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "dashboards" / "app.py"),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]


def main(argv: list[str] | None = None) -> int:
    """Run the server in the foreground; returns its exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", choices=sorted(theme.THEMES), default="field")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args(argv)
    env = {**os.environ, "PYTHONUTF8": "1", **theme.streamlit_environment(args.theme)}
    LOG.info("serving the %s design on http://localhost:%d", args.theme, args.port)
    return subprocess.call(command(args.port), env=env, cwd=ROOT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
