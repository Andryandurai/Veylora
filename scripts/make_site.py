"""Build the static Veylora site that Vercel serves (``web/``).

Vercel cannot run the Streamlit dashboards (no websockets, small function size limit), so it hosts a static site:
a landing page, the self-contained 3D simulator and the headline results. The dashboards run on a Streamlit host and
are linked with ``--dashboards-url``. Every number on the page is read from ``reports/*.json`` or ``reports/*.csv``;
a missing result prints a visible ``MISSING`` marker instead of a value (CLAUDE.md hard rule 2).

Usage::

    python -m scripts.make_site                                   # build web/ (no dashboards link)
    python -m scripts.make_site --dashboards-url https://veylora.streamlit.app
    python -m scripts.make_site --design almanac                  # dark editorial palette

``web/simulator.html`` is a copy of the bundled simulator page; Vercel makes the same copy at build time
(``vercel.json``), so it is not committed.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from config import load_settings
from dashboards.components import assets, theme

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
REPORTS = ROOT / "reports"
SIMULATOR_SOURCE = ROOT / "data" / "sample" / "simulator" / "index_embedded.html"
PREVIEW_SOURCE = REPORTS / "figures" / "sim_week9.png"
REPO_URL = "https://github.com/Andryandurai/Veylora"
SITE_NAME = "Veylora"
MISSING = '<span class="missing">MISSING</span>'
BASELINES = ("naive_last_year", "district_trend", "prior_mean_3y")


@dataclass(frozen=True)
class Result:
    """One headline result: label, value (HTML), and the sentence that explains it (HTML)."""

    label: str
    value: str
    text: str


def _read_json(name: str) -> dict | None:
    path = REPORTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _missing(label: str, what: str) -> Result:
    return Result(label, MISSING, f"The result file for {html.escape(what)} was not found.")


def yield_result() -> Result:
    """Held-out yield error (served XGBoost, 60-day lead) against the best simple baseline, worded from the numbers."""
    label = "Yield forecast error, test 2015 to 2017"
    path = REPORTS / "yield_metrics.csv"
    if not path.exists():
        return _missing(label, "the yield models")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["split"] == "test"
            and row["crop"] == "all"
            and row["lead_days"] == "60"
            and row["feature_set"] == "weather"
        ]
    by_model = {row["model"]: float(row["rmse_rel"]) for row in rows}
    candidates = {name: by_model[name] for name in BASELINES if name in by_model}
    if "xgboost" not in by_model or not candidates:
        return _missing(label, "the yield models")
    best = min(candidates, key=candidates.get)
    ours, theirs = by_model["xgboost"], candidates[best]
    gap = ours - theirs
    verdict = "better than" if gap < -0.02 else "worse than" if gap > 0.02 else "about level with"
    return Result(
        label,
        f"{ours:.0%}",
        f"of mean yield (RMSE relative to each district's prior mean). That is {verdict} the best simple "
        f"baseline ({best.replace('_', ' ')}, {theirs:.0%}).",
    )


def drought_result() -> Result:
    """Drought-warning F1 and PR-AUC on the held-out years, against climatology."""
    label = "Drought warning, test 2020 to 2024"
    report = _read_json("extreme_weather.json")
    if not report:
        return _missing(label, "the early-warning models")
    drought = report["drought"]
    test = drought["feature_sets"][drought["served_feature_set"]]["test"]
    clim = drought["baselines_test"]["climatology"]
    return Result(
        label,
        f"F1 {test['f1']:.2f}",
        f"PR-AUC {test['pr_auc']:.2f} against {clim['pr_auc']:.2f} for a climatology forecast.",
    )


def irrigation_result() -> Result:
    """Water saved and yield kept by the optimized schedule against the farmer baseline, normal rainfall."""
    label = "Irrigation plan, normal rainfall"
    report = _read_json("prescriptive_results.json")
    if not report:
        return _missing(label, "the irrigation optimization")
    rows = {(row["scenario"], row["strategy"]): row for row in report["summary"]}
    optimized = rows.get(("normal rainfall", "optimized"))
    baseline = rows.get(("normal rainfall", "farmer_baseline"))
    if not optimized or not baseline:
        return _missing(label, "the irrigation optimization")
    return Result(
        label,
        f"{optimized['water_saved_pct']:.0f}% less water",
        f"than a {baseline['water_used_mm']:.0f} mm baseline, keeping {optimized['yield_vs_baseline']:.0%} of its "
        f"yield ({optimized['n_cases']} district-season cases). Whether that trade-off is acceptable is for "
        "agronomists and farmers to decide.",
    )


def synthetic_note() -> str:
    """The data caveat (empty when no table contains synthetic rows)."""
    quality = _read_json("data_quality.json") or {}
    tables = [
        f"{name} ({share:.0%})"
        for name, share in quality.get("synthetic_share", {}).items()
        if share > 0
    ]
    if not tables:
        return ""
    return (
        '<p class="note"><span class="tag">Note</span><b>Synthetic data in the pipeline: '
        f"{html.escape(', '.join(tables))}.</b> The forecasts use observed weather and ICRISAT yields only; "
        "the synthetic satellite table appears only in an experimental comparison.</p>"
    )


def _on_primary(tokens: dict[str, str]) -> str:
    """Text colour that reads on the primary colour (white or the page ink, whichever contrasts more)."""

    def lum(colour: str) -> float:
        channels = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    def ratio(a: str, b: str) -> float:
        hi, lo = sorted((lum(a), lum(b)), reverse=True)
        return (hi + 0.05) / (lo + 0.05)

    white, ink = "#ffffff", tokens["paper"]
    return white if ratio(white, tokens["primary"]) >= ratio(ink, tokens["primary"]) else ink


def _facts() -> list[tuple[str, str]]:
    settings = load_settings()
    quality = _read_json("data_quality.json") or {}
    districts = quality.get("rows", {}).get("districts")
    years, weather = settings["years"], settings["weather_years"]
    return [
        ("Yield-reporting districts", str(districts) if districts else MISSING),
        ("Crops", str(len(settings["crops"]))),
        ("Weather record", f"{weather['start']}–{weather['end']}"),
        ("Yield record", f"{years['start']}–{years['end']}"),
    ]


def render(
    tokens: dict[str, str],
    dashboards_url: str | None,
    repo_url: str,
    has_logo: bool,
    has_preview: bool,
) -> str:
    """The landing page as one HTML document."""
    root_vars = " ".join(
        f"--{name}: {value};"
        for name, value in {**tokens, "on-primary": _on_primary(tokens)}.items()
    )
    results = [yield_result(), drought_result(), irrigation_result()]
    facts = "".join(
        f'<li><span class="v">{v}</span><span class="l">{html.escape(label)}</span></li>'
        for label, v in _facts()
    )
    result_html = "".join(
        f'<div class="result"><span class="l">{html.escape(r.label)}</span><span class="v">{r.value}</span>'
        f"<p>{r.text}</p></div>"
        for r in results
    )
    brand_img = '<img src="logo.jpg" alt="" width="42" height="42">' if has_logo else ""
    icon = (
        '<link rel="icon" href="favicon.png" type="image/png">'
        if has_logo
        else "<link rel=\"icon\" href=\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Crect width='16' height='16' fill='%231f6f5a'/%3E%3C/svg%3E\">"
    )
    dashboards_button = (
        f'<a class="btn" href="{html.escape(dashboards_url)}" rel="noopener">Open the dashboards</a>'
        if dashboards_url
        else ""
    )
    dashboards_nav = (
        f'<a href="{html.escape(dashboards_url)}" rel="noopener">Dashboards</a>'
        if dashboards_url
        else ""
    )
    aside = (
        ""
        if dashboards_url
        else '<p class="aside">The forecast, risk-map and advisory dashboards are a Streamlit app and are not hosted on '
        "this page; run them with <code>make dashboards</code> (see the repository).</p>"
    )
    preview = (
        '<figure class="preview"><img src="preview.jpg" width="1400" height="840" loading="lazy" '
        'alt="Screenshot of the Veylora 3D simulator: a paddy-field scene with sliders on the left and yield, water and '
        'stress panels on the right.">'
        "<figcaption>The simulator: a rice field whose plants, ponded water and stress panels follow the rainfall, "
        "temperature and irrigation sliders. Software-rendered screenshot.</figcaption></figure>"
        if has_preview
        else ""
    )
    description = (
        "Veylora: district-level early warning for heatwaves and drought, crop-yield forecasts with honest "
        "uncertainty, and irrigation plans for Andhra Pradesh, Telangana, Karnataka and Tamil Nadu."
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{SITE_NAME}: climate risk, yield forecasts and irrigation planning for southern India</title>
<meta name="description" content="{html.escape(description)}">
<meta property="og:title" content="{SITE_NAME}">
<meta property="og:description" content="{html.escape(description)}">
<meta property="og:type" content="website">
<meta name="theme-color" content="{tokens["paper"]}">
{icon}
<style>:root {{ {root_vars} }}</style>
<link rel="stylesheet" href="styles.css">
</head>
<body>
<header class="site"><div class="wrap">
  <a class="brand" href="/">{brand_img}<span class="brand-name">{SITE_NAME}</span></a>
  <nav class="site" aria-label="Main">
    <a href="simulator">Simulator</a>
    {dashboards_nav}
    <a href="#results">Results</a>
    <a href="#limits">Limits</a>
    <a href="{html.escape(repo_url)}" rel="noopener">GitHub</a>
  </nav>
</div></header>
<main>
<div class="wrap">
  <div class="hero">
    <p class="eyebrow">Climate resilience, India</p>
    <h1>{SITE_NAME}</h1>
    <p class="lead">District-level early warning for heatwaves and drought, crop-yield forecasts with honest uncertainty,
    and irrigation plans for Andhra Pradesh, Telangana, Karnataka and Tamil Nadu.</p>
    <div class="actions">
      <a class="btn primary" href="simulator">Open the 3D simulator</a>
      {dashboards_button}
      <a class="btn" href="{html.escape(repo_url)}" rel="noopener">Source on GitHub</a>
    </div>
    {aside}
    {synthetic_note()}
  </div>
  <ul class="facts" aria-label="At a glance">{facts}</ul>

  <section id="simulator">
    <h2>See a season, week by week</h2>
    <p class="sub">The simulator runs in your browser, with no sign-in and nothing to install. Change rainfall and
    temperature, compare irrigation strategies and watch yield, water use and stress respond.</p>
    {preview}
  </section>

  <section id="what">
    <h2>What Veylora does</h2>
    <p class="sub">Four tools for one decision: what the coming weeks mean for a crop in a district.</p>
    <div class="grid">
      <div class="item"><span class="n">01</span><h3>Early warning</h3><p>Heatwave and drought probabilities for a district and date, replayed on observed history. No live forecast feed is connected.</p></div>
      <div class="item"><span class="n">02</span><h3>Yield forecasts</h3><p>A forecast with an 80% interval, what-if rainfall and temperature, and a comparison with simple baselines.</p></div>
      <div class="item"><span class="n">03</span><h3>Irrigation plans</h3><p>A weekly water plan from a soil-water balance and a linear programme; paddy rice is modelled as a ponded field.</p></div>
      <div class="item"><span class="n">04</span><h3>Farmer advisories</h3><p>Short, template-driven messages in English, Hindi, Tamil and Telugu. The translations are drafts awaiting native-speaker review.</p></div>
    </div>
  </section>

  <section id="results">
    <h2>What the evaluation shows</h2>
    <p class="sub">Measured on held-out years only, and reported as found. Intervals, baselines and the negative results are on the Methodology page of the dashboards.</p>
    <div class="results">{result_html}</div>
  </section>

  <section id="limits">
    <h2>Read this before using a number</h2>
    <ul class="limits">
      <li>Units are the 51 district groups for which ICRISAT reports yields (1966 boundaries), not villages or today's districts.</li>
      <li>Yields are observed to 2017 only, so the latest hold-out years are 2015 to 2017.</li>
      <li>Forecast skill over simple baselines is small or unproven; use the intervals, not just the point forecast.</li>
      <li>Crop and price parameters are approximate and have not been verified against current sources.</li>
      <li>The simulator's frame rate was tested with software rendering only.</li>
    </ul>
  </section>
</div>
</main>
<footer class="site"><div class="wrap">
  <p>Data: IMD gridded rainfall and temperature, NASA POWER, ICRISAT district-level database, geoBoundaries. Relief: AWS Terrain Tiles (USGS 3DEP, GMTED2010, SRTM). 3D: three.js.</p>
  <p>Veylora is decision support with uncertainty, not a substitute for local agronomic advice.
  <a href="{html.escape(repo_url)}" rel="noopener">Source code and licence notes</a>.</p>
</div></footer>
</body>
</html>
"""


def _write_logo_assets(out_dir: Path) -> bool:
    """Write the web logo and tab icon when the logo file exists; True if written."""
    if not assets.LOGO_SOURCE.is_file():
        for stale in ("logo.jpg", "favicon.png"):
            (out_dir / stale).unlink(missing_ok=True)
        return False
    with Image.open(assets.LOGO_SOURCE) as source:
        small = source.convert("RGB")
        small.thumbnail((256, 256), Image.Resampling.LANCZOS)
        small.save(out_dir / "logo.jpg", "JPEG", quality=90, optimize=True)
    icon = assets.favicon()
    if icon is not None:
        icon.save(out_dir / "favicon.png", "PNG", optimize=True)
    return True


def _write_preview(out_dir: Path) -> bool:
    """Write the simulator preview image (resized JPEG) when the capture exists; True if written."""
    if not PREVIEW_SOURCE.is_file():
        (out_dir / "preview.jpg").unlink(missing_ok=True)
        return False
    with Image.open(PREVIEW_SOURCE) as source:
        image = source.convert("RGB")
        image.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
        image.save(out_dir / "preview.jpg", "JPEG", quality=85, optimize=True)
    return True


def build(
    out_dir: Path = WEB,
    dashboards_url: str | None = None,
    design: str = "field",
    repo_url: str = REPO_URL,
    copy_simulator: bool = True,
) -> Path:
    """Write the site into ``out_dir`` and return the path of ``index.html``.

    Args:
        out_dir: output directory (``web/`` by default; ``styles.css`` is copied from there when different).
        dashboards_url: address of the hosted Streamlit dashboards, or None to say they are not hosted here.
        design: dashboard design whose colours the page uses (``field`` or ``almanac``).
        repo_url: link to the source repository.
        copy_simulator: also copy the bundled simulator page to ``simulator.html`` (Vercel does this at build time).
    """
    if design not in theme.THEMES:
        raise ValueError(f"unknown design {design!r}; choose one of {sorted(theme.THEMES)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.resolve() != WEB.resolve():
        shutil.copyfile(WEB / "styles.css", out_dir / "styles.css")
    has_logo = _write_logo_assets(out_dir)
    has_preview = _write_preview(out_dir)
    page = render(theme.THEMES[design]["tokens"], dashboards_url, repo_url, has_logo, has_preview)
    target = out_dir / "index.html"
    target.write_text(page, encoding="utf-8")
    if copy_simulator and SIMULATOR_SOURCE.is_file():
        shutil.copyfile(SIMULATOR_SOURCE, out_dir / "simulator.html")
    LOG.info(
        "wrote %s (logo=%s, preview=%s, dashboards=%s)",
        target,
        has_logo,
        has_preview,
        dashboards_url,
    )
    return target


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dashboards-url", default=None, help="address of the hosted Streamlit dashboards"
    )
    parser.add_argument("--design", choices=sorted(theme.THEMES), default="field")
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--out", type=Path, default=WEB)
    parser.add_argument(
        "--no-simulator", action="store_true", help="do not copy the simulator page"
    )
    args = parser.parse_args(argv)
    build(args.out, args.dashboards_url, args.design, args.repo_url, not args.no_simulator)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
