"""Build and execute notebooks/01_eda.ipynb from the Phase 3 result files.

The notebook only *displays* what ``make analytics-eda`` produced (figures, tables, FINDINGS.md), so it
never recomputes or retypes a statistic. Run: ``python scripts/make_eda_notebook.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("make_eda_notebook")

FIGURES = [
    ("rainfall_anomaly_heatmap", "Kharif rainfall anomalies, districts x years"),
    ("rainfall_cv_kharif", "Kharif rainfall variability (CV)"),
    ("yield_trends", "State yield trends with Theil-Sen fits"),
    ("drought_flood_frequency", "Drought and flood-risk frequency"),
    ("correlations_within", "Within-district weather-yield correlations"),
    ("regression_residuals", "Fixed-effects regression residuals"),
    ("clusters_map", "Climate-vulnerability clusters"),
]
TABLES = [
    ("yield_trends_table", "Yield trends (Theil-Sen, Mann-Kendall)"),
    ("regression_coefficients", "Fixed-effects regression coefficients"),
    ("cluster_profiles", "Cluster profiles (centroids in original units)"),
]


def build() -> nbformat.NotebookNode:
    """Assemble the notebook cells."""
    cells = [
        new_markdown_cell(
            "# 01 - Exploratory analysis\n\nDescriptive and diagnostic analytics for the study districts. "
            "All numbers are produced by `make analytics-eda` from the PostgreSQL warehouse; this notebook only displays them. "
            "Check the data-status banner first."
        ),
        new_code_cell(
            "from pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image, Markdown, display\n"
            "pd.set_option('display.width', 200)\nROOT = Path.cwd()\n"
            "findings = (ROOT / 'docs' / 'FINDINGS.md').read_text(encoding='utf-8')\n"
            "display(Markdown(findings))"
        ),
    ]
    for name, title in FIGURES:
        cells.append(new_markdown_cell(f"## {title}"))
        cells.append(new_code_cell(f"display(Image(filename=str(ROOT / 'reports' / 'figures' / '{name}.png')))"))
    for name, title in TABLES:
        cells.append(new_markdown_cell(f"## {title}"))
        cells.append(new_code_cell(f"pd.read_csv(ROOT / 'reports' / 'analytics' / '{name}.csv').round(4)"))
    notebook = new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return notebook


def main() -> None:
    """Build, execute (so outputs are embedded) and save the notebook."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    notebook = build()
    NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    target = ROOT / "notebooks" / "01_eda.ipynb"
    target.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, target)
    LOG.info("wrote %s", target)


if __name__ == "__main__":
    main()
