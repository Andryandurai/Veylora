"""Run every Phase 3 analysis and regenerate docs/FINDINGS.md (``make analytics-eda``)."""

from __future__ import annotations

import logging
import time

from analytics import make_findings
from analytics.descriptive import drought_flood_frequency, rainfall_variability, yield_trends
from analytics.diagnostic import clustering, correlations, regression

LOG = logging.getLogger("analytics.run_eda")

STEPS = [
    ("rainfall variability", rainfall_variability.run),
    ("yield trends", yield_trends.run),
    ("drought / flood frequency", drought_flood_frequency.run),
    ("correlations", correlations.run),
    ("regression", regression.run),
    ("clustering", clustering.run),
]


def main() -> None:
    """Run all steps in order, then write the findings document."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    started = time.time()
    for name, step in STEPS:
        LOG.info("=== %s ===", name)
        step()
    make_findings.main()
    print(f"analytics-eda finished in {time.time() - started:.0f}s")  # noqa: T201 - CLI summary


if __name__ == "__main__":
    main()
