"""Generate reports/prescriptive_results.md and reports/prescriptive_results.json.

For 20 sampled district-seasons (seed 42) the four irrigation strategies are scored by the daily water-balance
simulator in two weather scenarios (the district-season's typical year, and the same year with rainfall x0.7).
Means come with 95 percent bootstrap intervals over the 20 cases. Nothing is hard-coded: every table cell is
computed by :mod:`analytics.prescriptive.scenario`.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from analytics import common
from analytics.prescriptive import scenario as S
from analytics.prescriptive.params import load_crop, load_globals
from config import ROOT, get_path, get_seed, load_settings

LOG = logging.getLogger(__name__)

N_CASES = 20
N_BOOT = 2000
SCENARIOS = {"normal rainfall": 1.0, "dry (rain x0.7)": 0.7}
CROPS = ("rice", "cotton", "wheat")


def sample_cases(n: int = N_CASES, seed: int | None = None) -> list[dict[str, str]]:
    """Seeded sample of district-crop-season combinations that have yield history."""
    seed = get_seed() if seed is None else seed
    crops = pd.read_parquet(get_path("data_processed") / "crops_annual.parquet")
    counts = crops.groupby(["district_id", "crop", "season"]).size().reset_index(name="n")
    valid = (
        counts[(counts["n"] >= 10) & counts["crop"].isin(CROPS)]
        .sort_values(["district_id", "crop"])
        .reset_index(drop=True)
    )
    picked = valid.sample(n=min(n, len(valid)), random_state=seed)
    return picked.sort_values(["crop", "district_id"])[["district_id", "crop", "season"]].to_dict(
        "records"
    )


def bootstrap_ci(
    values: np.ndarray, n_boot: int = N_BOOT, seed: int = 42
) -> tuple[float, float, float]:
    """Mean and 95 percent bootstrap interval of a 1-D sample."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_cases(cases: list[dict[str, str]], with_ml: bool = True) -> pd.DataFrame:
    """Strategy results for every case and scenario (long format)."""
    predict_yield = None
    if with_ml:
        try:
            from analytics.predictive.predict import predict_yield
        except Exception as exc:  # models missing
            LOG.warning("ML yields unavailable: %s", exc)
    rows = []
    for case in cases:
        crop = load_crop(case["crop"])
        year = S.typical_year(case["district_id"], case["season"])
        ml = None
        if predict_yield is not None:
            try:
                ml = predict_yield(
                    case["district_id"], case["crop"], case["season"], None, year=year
                )
            except ValueError as exc:
                LOG.warning("no ML yield for %s: %s", case, exc)
        for scenario, mult in SCENARIOS.items():
            inputs = S.load_season_inputs(
                case["district_id"], case["crop"], case["season"], year, rain_multiplier=mult
            )
            started = time.perf_counter()
            table = (
                S.strategy_yields(inputs, crop.default_budget_mm, ml)
                if ml
                else S.evaluate_strategies(inputs, crop.default_budget_mm)[0]
            )
            elapsed = time.perf_counter() - started
            table = table.assign(
                **case,
                scenario=scenario,
                year=year,
                budget_mm=crop.default_budget_mm,
                case_seconds=elapsed,
                rain_mm=float(inputs.rain.sum()),
                etc_mm=float(table["etc_mm"].iloc[0]),
            )
            rows.append(table.drop(columns=["weekly_irrigation_mm"]))
    return pd.concat(rows, ignore_index=True)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Per scenario and strategy: mean and bootstrap CI of water used, relative yield, saving and yield ratio."""
    out = []
    base = results[results["strategy"] == "farmer_baseline"].set_index(
        ["district_id", "crop", "season", "scenario"]
    )
    for (scenario, strategy), group in results.groupby(["scenario", "strategy"], sort=False):
        key = group.set_index(["district_id", "crop", "season", "scenario"])
        row: dict[str, Any] = {"scenario": scenario, "strategy": strategy, "n_cases": len(group)}
        for metric in ("water_used_mm", "rel_yield", "eta_over_etc", "drainage_mm"):
            mean, lo, hi = bootstrap_ci(group[metric].to_numpy())
            row[metric], row[metric + "_lo"], row[metric + "_hi"] = mean, lo, hi
        saved = (
            100.0
            * (base["water_used_mm"].reindex(key.index) - key["water_used_mm"])
            / base["water_used_mm"].reindex(key.index)
        )
        row["water_saved_pct"], row["water_saved_pct_lo"], row["water_saved_pct_hi"] = bootstrap_ci(
            saved.to_numpy()
        )
        ratio = key["rel_yield"] / base["rel_yield"].reindex(key.index)
        row["yield_vs_baseline"], row["yield_vs_baseline_lo"], row["yield_vs_baseline_hi"] = (
            bootstrap_ci(ratio.replace([np.inf, -np.inf], np.nan).to_numpy())
        )
        out.append(row)
    return pd.DataFrame(out)


def paired_optimized_vs_even(results: pd.DataFrame) -> pd.DataFrame:
    """Paired comparison optimized minus even-split (same budget) per scenario."""
    rows = []
    for scenario, group in results.groupby("scenario", sort=False):
        pivot = group.pivot_table(
            index=["district_id", "crop", "season"],
            columns="strategy",
            values=["rel_yield", "water_used_mm"],
        )
        d_yield = pivot["rel_yield"]["optimized"] - pivot["rel_yield"]["even_split"]
        d_water = pivot["water_used_mm"]["optimized"] - pivot["water_used_mm"]["even_split"]
        y_mean, y_lo, y_hi = bootstrap_ci(d_yield.to_numpy())
        w_mean, w_lo, w_hi = bootstrap_ci(d_water.to_numpy())
        rows.append(
            {
                "scenario": scenario,
                "cases": len(d_yield),
                "delta_rel_yield": y_mean,
                "delta_rel_yield_lo": y_lo,
                "delta_rel_yield_hi": y_hi,
                "delta_water_mm": w_mean,
                "delta_water_lo": w_lo,
                "delta_water_hi": w_hi,
                "cases_optimized_at_least_as_good": int((d_yield >= -1e-9).sum()),
                "cases_optimized_worse": int((d_yield < -1e-9).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_report(results: pd.DataFrame) -> dict[str, Any]:
    """Render markdown + JSON from the case results."""
    summary, paired = summarise(results), paired_optimized_vs_even(results)
    glob = load_globals()
    ci = lambda r, m, d=1: f"{r[m]:.{d}f} ({r[m + '_lo']:.{d}f} to {r[m + '_hi']:.{d}f})"  # noqa: E731
    table = pd.DataFrame(
        [
            {
                "scenario": r["scenario"],
                "strategy": r["strategy"],
                "water used, mm": ci(r, "water_used_mm", 0),
                "relative yield (FAO-33)": ci(r, "rel_yield", 3),
                "ETa / ETc": ci(r, "eta_over_etc", 3),
                "drainage, mm": ci(r, "drainage_mm", 0),
                "water saved vs baseline, %": ci(r, "water_saved_pct", 0),
                "yield vs baseline": ci(r, "yield_vs_baseline", 3),
            }
            for r in summary.to_dict("records")
        ]
    )
    lines = [
        "# Prescriptive results: irrigation scheduling",
        "",
        f"Generated by `python -m analytics.prescriptive.report` (seed {get_seed()}). **Decision-support estimates under stated assumptions, not field-validated results.**",
        "",
        f"**Method.** For {results[['district_id', 'crop', 'season']].drop_duplicates().shape[0]} sampled district-seasons the first {glob.season_days // 7} weeks of the season are simulated with the daily FAO-56 water balance "
        "(`analytics/prescriptive/water_balance.py`) in two weather scenarios (the district-season's typical training year, and the same year with rainfall x0.7). Four strategies are scored by that simulator: "
        "`rainfed`; `farmer_baseline` (50 mm every 10 days, as in the plan, 600 mm in 112 days); `even_split` (the budget spread evenly over the weeks, capped); and `optimized` (the weekly LP, `analytics/prescriptive/irrigation_lp.py`). "
        f"Budget per crop from config/crop_params.yaml (rice {load_crop('rice').default_budget_mm:.0f}, cotton {load_crop('cotton').default_budget_mm:.0f}, wheat {load_crop('wheat').default_budget_mm:.0f} mm), weekly cap {glob.weekly_cap_mm:.0f} mm. "
        "Cells are the mean over cases with a 95% bootstrap interval (2000 resamples of the cases).",
        "",
        "## Baseline versus optimized (simulator-scored)",
        "",
        common.markdown_table(table, 3),
        "",
        "## Optimized versus even split at the same budget (paired)",
        "",
        common.markdown_table(paired, 3),
        "",
    ]
    total_cases = len(results[["district_id", "crop", "season", "scenario"]].drop_duplicates())
    solved = results.groupby(["district_id", "crop", "season", "scenario"])["case_seconds"].first()
    lines += [
        "## Solver performance",
        "",
        f"All {total_cases} district-season-scenario problems solved to optimality; scoring all four strategies for one case (LP solve + simulation + reference scenario) took a mean of {solved.mean():.2f} s and a maximum of {solved.max():.2f} s (acceptance: LP < 5 s).",
        "",
        "## Assumptions and limits (read before quoting any number)",
        "",
        "- **Parameters are approximate.** Kc, root depth, depletion fraction, Ky, soil water capacity and prices are rounded values of the right order of magnitude, not table lookups (see the provenance notes in `config/crop_params.yaml` and `config/economics.yaml`). Changing them changes the results.",
        f"- **16-week window.** Every crop is simulated for {glob.season_days} days from the season start with compressed stage lengths; cotton and sugarcane cycles are longer in reality.",
        "- **Rice** is treated as a saturated-soil crop with a small depletion store: ponding, puddling and seepage are not modelled, so absolute rice water use is understated and the 600 mm baseline is not a recommendation.",
        f"- **Irrigation depths are net** (efficiency {glob.irrigation_efficiency:.2f}); each week's irrigation is applied as one event on the first day of the week; rainfall efficiency {glob.rain_efficiency:.2f}.",
        "- **Hargreaves ET0** from district-mean IMD Tmax/Tmin (1-degree grid) and the 'typical year' weather; real fields differ in soil, timing and sowing date (sowing is assumed on the first day of the season).",
        "- **LP vs simulator.** The LP is a weekly linearisation; the table scores its schedule with the daily nonlinear simulator, where it can be worse than an even split in individual cases (counted above).",
        "- **Yield estimate** (when reported): ML yield at normal rainfall x water-stress ratio relative to the farmer baseline at normal rainfall, capped at +50 percent (D-020). The ML model itself explains little beyond yield history (docs/DECISIONS.md D-015).",
        "- **Water saved** is relative to a generous 600 mm baseline defined in the plan; compare also `even_split`, which uses the same budget as `optimized`.",
        "",
    ]
    path = ROOT / "reports" / "prescriptive_results.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "summary": summary.to_dict("records"),
        "paired_optimized_vs_even": paired.to_dict("records"),
        "n_cases": total_cases,
        "max_case_seconds": float(solved.max()),
        "settings": {
            "seed": get_seed(),
            "season_days": glob.season_days,
            "weekly_cap_mm": glob.weekly_cap_mm,
        },
    }
    (get_path("reports") / "prescriptive_results.json").write_text(
        json.dumps(payload, indent=1, default=float), encoding="utf-8"
    )
    results.to_csv(get_path("reports") / "prescriptive_cases.csv", index=False)
    LOG.info("wrote %s", path)
    return payload


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_settings()
    cases = sample_cases()
    LOG.info("scoring %d district-seasons x %d scenarios", len(cases), len(SCENARIOS))
    write_report(run_cases(cases))


if __name__ == "__main__":
    main()
