"""Yield trends by state and crop: Theil-Sen slope and Mann-Kendall significance.

State-level yield per year is production-weighted (sum production / sum area) over a **balanced
panel** of districts, i.e. only districts that report in every year of the window, so that a district
entering or leaving the data cannot create a trend. Series with fewer than three such districts are
skipped and listed in the results. Outputs: ``yield_trends.{png,html}`` and
``reports/analytics/yield_trends.json`` plus ``yield_trends_table.{csv,md}``.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pymannkendall as mk
from plotly.subplots import make_subplots
from scipy import stats

from analytics import common

LOG = logging.getLogger(__name__)

MIN_BALANCED_DISTRICTS = 3
MIN_YEARS = 10


def mann_kendall(values: np.ndarray | pd.Series) -> dict[str, float | str]:
    """Mann-Kendall trend test (pymannkendall original test) returning tau, p and the trend label."""
    result = mk.original_test(np.asarray(values, dtype=float))
    return {
        "trend": str(result.trend),
        "p": float(result.p),
        "tau": float(result.Tau),
        "z": float(result.z),
    }


def theil_sen(years: np.ndarray | pd.Series, values: np.ndarray | pd.Series) -> dict[str, float]:
    """Theil-Sen slope per year with its 95 % confidence interval (scipy)."""
    slope, intercept, lo, hi = stats.theilslopes(
        np.asarray(values, float), np.asarray(years, float), 0.95
    )
    return {"slope": float(slope), "intercept": float(intercept), "lo": float(lo), "hi": float(hi)}


def balanced_state_series(
    train: pd.DataFrame, state: str, crop: str
) -> tuple[pd.Series, int] | None:
    """Production-weighted yield by year over districts present in every year; None if too thin."""
    subset = train[(train["state"] == state) & (train["crop"] == crop)]
    if subset.empty:
        return None
    years = sorted(subset["year"].unique())
    counts = subset.groupby("district_id")["year"].nunique()
    complete = counts[counts == len(years)].index
    if len(years) < MIN_YEARS or len(complete) < MIN_BALANCED_DISTRICTS:
        return None
    panel = subset[subset["district_id"].isin(complete)]
    grouped = panel.groupby("year")[["production_t", "area_ha"]].sum()
    return (grouped["production_t"] / grouped["area_ha"]).rename("yield_t_ha"), int(len(complete))


def trend_table(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.Series], list[dict]]:
    """Theil-Sen + Mann-Kendall for every state-crop series; returns table, series, skipped list."""
    rows, series, skipped = [], {}, []
    for state in train["state"].unique():
        for crop in common.load_settings()["crops"]:
            built = balanced_state_series(train, state, crop)
            if built is None:
                skipped.append({"state": state, "crop": crop})
                continue
            values, n_districts = built
            ts = theil_sen(values.index, values.to_numpy())
            kendall = mann_kendall(values.to_numpy())
            rows.append(
                {
                    "state": state,
                    "crop": crop,
                    "n_years": int(len(values)),
                    "n_districts": n_districts,
                    "mean_yield_t_ha": float(values.mean()),
                    "slope_t_ha_per_year": ts["slope"],
                    "slope_ci_lo": ts["lo"],
                    "slope_ci_hi": ts["hi"],
                    "slope_pct_of_mean_per_year": 100.0 * ts["slope"] / float(values.mean()),
                    "mk_tau": kendall["tau"],
                    "mk_p": kendall["p"],
                    "mk_trend": kendall["trend"],
                }
            )
            series[(state, crop)] = values
    return pd.DataFrame(rows), series, skipped


def _spread(values: list[float], min_gap: float) -> list[float]:
    """Nudge label positions apart so neighbouring labels are at least ``min_gap`` away (order kept)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = [0.0] * len(values)
    previous = -np.inf
    for i in order:
        previous = max(values[i], previous + min_gap)
        placed[i] = previous
    return placed


def draw_png(table: pd.DataFrame, series: dict[tuple[str, str], pd.Series]) -> None:
    """Small multiples: one panel per crop, one line per state, Theil-Sen trend dashed."""
    common.apply_mpl_theme()
    crops = [c for c in common.load_settings()["crops"] if c in set(table["crop"])]
    colors = common.state_colors(sorted(set(table["state"])))
    n_cols = 2
    n_rows = int(np.ceil(len(crops) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 4.2 * n_rows), sharex=True, squeeze=False)
    flat = axes.ravel()
    for ax, crop in zip(flat, crops, strict=False):
        labels: list[tuple[float, str, str]] = []
        for state in colors:
            if (state, crop) not in series:
                continue
            values = series[(state, crop)]
            row = table[(table["state"] == state) & (table["crop"] == crop)].iloc[0]
            x = values.index.to_numpy(dtype=float)
            ax.plot(x, values.to_numpy(), color=colors[state], linewidth=2)
            fit = row["slope_t_ha_per_year"] * (x - x.mean()) + np.median(values.to_numpy())
            ax.plot(x, fit, color=colors[state], linestyle=(0, (4, 3)), linewidth=1.2)
            labels.append((float(values.iloc[-1]), state, colors[state]))
        low, high = ax.get_ylim()
        for y, (_, state, color) in zip(
            _spread([v for v, _, _ in labels], 0.07 * (high - low)), labels, strict=True
        ):
            ax.annotate(
                state,
                xy=(2017.4, y),
                color=common.INK_SECONDARY,
                fontsize=8,
                va="center",
                annotation_clip=False,
            )
            ax.plot([2017.2], [y], marker="o", markersize=4, color=color, clip_on=False)
        suffix = "  (unit unclear: probably jaggery-equivalent)" if crop == "sugarcane" else ""
        ax.set_title(crop.capitalize() + suffix)
        ax.set_xlim(1990.5, 2017)
        ax.margins(x=0)
    for ax in flat[len(crops) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("crop year")
    for ax in axes[:, 0]:
        ax.set_ylabel("yield, t/ha")
    fig.suptitle(
        "State yield trends: production-weighted yield (solid) and Theil-Sen fit (dashed)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.subplots_adjust(right=0.86, hspace=0.35, wspace=0.55)
    common.save_mpl(fig, "yield_trends")


def draw_html(table: pd.DataFrame, series: dict[tuple[str, str], pd.Series]) -> None:
    """Interactive small multiples with hover and slope in the hover text."""
    crops = [c for c in common.load_settings()["crops"] if c in set(table["crop"])]
    colors = common.state_colors(sorted(set(table["state"])))
    fig = make_subplots(
        rows=1, cols=len(crops), subplot_titles=[c.capitalize() for c in crops], shared_xaxes=True
    )
    for j, crop in enumerate(crops, start=1):
        for state, color in colors.items():
            if (state, crop) not in series:
                continue
            values = series[(state, crop)]
            row = table[(table["state"] == state) & (table["crop"] == crop)].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=values.index,
                    y=values.to_numpy(),
                    mode="lines",
                    name=state,
                    legendgroup=state,
                    showlegend=j == 1,
                    line={"color": color, "width": 2},
                    hovertemplate=(
                        f"{state}: %{{y:.2f}} t/ha in %{{x}}<br>Theil-Sen {row['slope_t_ha_per_year']:+.3f} t/ha/yr "
                        f"(MK p={row['mk_p']:.3f})<extra></extra>"
                    ),
                ),
                row=1,
                col=j,
            )
    common.plotly_layout(
        fig, "State yield trends (production-weighted, balanced panel)", height=430
    )
    common.save_plotly(fig, "yield_trends")


def run() -> dict:
    """Compute, plot and save; returns headline numbers."""
    train = common.training_table()
    table, series, skipped = trend_table(train)
    draw_png(table, series)
    draw_html(table, series)
    common.save_table(table, "yield_trends_table", floatfmt=4)
    significant = table[table["mk_p"] < 0.05]
    results = {
        "years": [int(train["year"].min()), int(train["year"].max())],
        "n_series": int(len(table)),
        "skipped_series": skipped,
        "n_significant_increasing": int(
            ((table["mk_p"] < 0.05) & (table["mk_trend"] == "increasing")).sum()
        ),
        "n_significant_decreasing": int(
            ((table["mk_p"] < 0.05) & (table["mk_trend"] == "decreasing")).sum()
        ),
        "series": table.to_dict("records"),
        "strongest_increasing": (
            significant.sort_values("slope_pct_of_mean_per_year", ascending=False)
            .head(3)
            .to_dict("records")
        ),
        "rice_slope_pct_range": [
            float(table[table["crop"] == "rice"]["slope_pct_of_mean_per_year"].min()),
            float(table[table["crop"] == "rice"]["slope_pct_of_mean_per_year"].max()),
        ]
        if (table["crop"] == "rice").any()
        else None,
    }
    common.save_results("yield_trends", results)
    return results


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
