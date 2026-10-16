"""Yield versus weather/vegetation correlations with district-cluster bootstrap confidence intervals.

Two views of the same relationship, because they answer different questions:

* ``raw``: pooled district-years. Mixes *where* (wetter districts yield differently) and *when*, and
  both series trend upwards over time, so it is the easy-to-misread number.
* ``within``: each variable is first detrended inside its district (district intercept + linear
  trend removed), so only year-to-year co-movement remains. This is the one to interpret.

Confidence intervals come from a **cluster bootstrap over districts** (1000 resamples, seed 42):
resampling whole districts keeps each district's years together and partly respects the spatial
dependence between neighbours. Spearman uses full-sample ranks inside the bootstrap (rank-then-
resample), a standard approximation. Because districts are still spatially autocorrelated (Moran's I
is reported), results are also given **per state** and effective sample sizes are smaller than n.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analytics import common

LOG = logging.getLogger(__name__)

PREDICTORS = {
    "total_rain": "Season rainfall",
    "mean_tmax": "Mean Tmax",
    "ndvi_peak": "Peak NDVI",
    "soil_moisture_mean": "Mean soil moisture",
}
SATELLITE_PREDICTORS = {"ndvi_peak", "soil_moisture_mean"}
N_BOOT = 1000


def detrend_within_district(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Residuals of each column after a per-district OLS on a constant and a linear year trend."""
    out = frame.copy()
    for column in columns:
        residuals = pd.Series(np.nan, index=out.index)
        for _, group in out.groupby("district_id"):
            valid = group[column].notna()
            if valid.sum() < 4:
                continue
            x = group.loc[valid, "year"].to_numpy(float)
            y = group.loc[valid, column].to_numpy(float)
            slope, intercept = np.polyfit(x, y, 1)
            residuals.loc[group.index[valid]] = y - (intercept + slope * x)
        out[column] = residuals
    return out


def cluster_bootstrap_corr(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_boot: int = N_BOOT, seed: int = 42
) -> dict[str, float]:
    """Pearson and Spearman correlation with 95 % cluster-bootstrap CIs (resampling groups)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y, groups = x[mask], y[mask], groups[mask]
    if len(x) < 8 or np.std(x) == 0 or np.std(y) == 0:
        return {
            k: float("nan")
            for k in (
                "pearson",
                "pearson_lo",
                "pearson_hi",
                "spearman",
                "spearman_lo",
                "spearman_hi",
            )
        } | {"n": int(len(x))}
    rank_x, rank_y = stats.rankdata(x), stats.rankdata(y)
    labels, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(labels))]
    rng = np.random.default_rng(seed)
    pearson, spearman = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        pearson[b] = _corr(x[idx], y[idx])
        spearman[b] = _corr(rank_x[idx], rank_y[idx])
    return {
        "pearson": float(stats.pearsonr(x, y)[0]),
        "pearson_lo": float(np.nanpercentile(pearson, 2.5)),
        "pearson_hi": float(np.nanpercentile(pearson, 97.5)),
        "spearman": float(stats.spearmanr(x, y)[0]),
        "spearman_lo": float(np.nanpercentile(spearman, 2.5)),
        "spearman_hi": float(np.nanpercentile(spearman, 97.5)),
        "n": int(len(x)),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    da, db = a - a.mean(), b - b.mean()
    denom = np.sqrt((da * da).sum() * (db * db).sum())
    return float((da * db).sum() / denom) if denom > 0 else float("nan")


def morans_i(
    values: np.ndarray, coords: np.ndarray, k: int = 5, n_perm: int = 999, seed: int = 42
) -> dict[str, float]:
    """Moran's I with row-standardised k-nearest-neighbour weights and a permutation p-value.

    Args:
        values: one value per district.
        coords: (n, 2) projected coordinates (any planar units).
    """
    n = len(values)
    z = values - values.mean()
    dist = np.hypot(
        coords[:, None, 0] - coords[None, :, 0], coords[:, None, 1] - coords[None, :, 1]
    )
    np.fill_diagonal(dist, np.inf)
    weights = np.zeros((n, n))
    neighbours = np.argsort(dist, axis=1)[:, :k]
    for i, row in enumerate(neighbours):
        weights[i, row] = 1.0 / k

    def statistic(v: np.ndarray) -> float:
        return float(n / weights.sum() * (v @ weights @ v) / (v @ v))

    observed = statistic(z)
    rng = np.random.default_rng(seed)
    permuted = np.array([statistic(rng.permutation(z)) for _ in range(n_perm)])
    p = float((1 + (permuted >= observed).sum()) / (n_perm + 1))
    return {"I": observed, "expected": -1.0 / (n - 1), "p_one_sided": p, "k": k, "n": n}


def correlation_table(data: pd.DataFrame, predictors: list[str], crops: list[str]) -> pd.DataFrame:
    """All crop x predictor x scope x variant correlations."""
    rows = []
    scopes: list[tuple[str, pd.DataFrame]]
    for crop in crops:
        subset = data[data["crop"] == crop]
        within = detrend_within_district(
            subset.assign(yield_label=subset["yield_label"]), ["yield_label", *predictors]
        )
        scopes = [("all states", subset)] + [
            (s, subset[subset["state"] == s]) for s in sorted(subset["state"].unique())
        ]
        within_scopes = [("all states", within)] + [
            (s, within[within["state"] == s]) for s in sorted(within["state"].unique())
        ]
        for variant, scope_list in (("raw", scopes), ("within", within_scopes)):
            for scope, frame in scope_list:
                for predictor in predictors:
                    est = cluster_bootstrap_corr(
                        frame[predictor].to_numpy(float),
                        frame["yield_label"].to_numpy(float),
                        frame["district_id"].to_numpy(),
                    )
                    rows.append(
                        {
                            "crop": crop,
                            "predictor": predictor,
                            "scope": scope,
                            "variant": variant,
                            **est,
                        }
                    )
    return pd.DataFrame(rows)


def draw_forest(table: pd.DataFrame, satellite_synthetic: bool) -> None:
    """Dot-and-whisker plot of the within-district Pearson correlations for all states."""
    common.apply_mpl_theme()
    data = table[(table["variant"] == "within") & (table["scope"] == "all states")]
    crops = [c for c in common.crop_colors() if c in set(data["crop"])]
    colors = common.crop_colors()
    fig, ax = plt.subplots(figsize=(8, 5))
    predictors = list(PREDICTORS)
    for j, crop in enumerate(crops):
        for i, predictor in enumerate(predictors):
            row = data[(data["crop"] == crop) & (data["predictor"] == predictor)]
            if row.empty or np.isnan(row["pearson"].iloc[0]):
                continue
            r = row.iloc[0]
            y = i + (j - (len(crops) - 1) / 2) * 0.17
            ax.plot([r["pearson_lo"], r["pearson_hi"]], [y, y], color=colors[crop], linewidth=1.6)
            ax.plot(
                r["pearson"],
                y,
                "o",
                color=colors[crop],
                markersize=6,
                markeredgecolor=common.SURFACE,
                label=crop.capitalize() if i == 0 else None,
            )
    ax.axvline(0, color=common.AXIS, linewidth=1)
    labels = [
        PREDICTORS[p]
        + (" (synthetic)" if satellite_synthetic and p in SATELLITE_PREDICTORS else "")
        for p in predictors
    ]
    ax.set_yticks(range(len(predictors)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("within-district Pearson r with yield (95% cluster-bootstrap CI)")
    ax.set_title("Year-to-year weather-yield co-movement (trend and district level removed)")
    ax.legend(loc="lower right", ncol=2)
    ax.grid(axis="y", visible=False)
    common.save_mpl(fig, "correlations_within")


def run() -> dict:
    """Compute all correlations and Moran's I, plot and save."""
    train = common.training_table()
    settings = common.load_settings()
    data = train[
        (train["year"] >= int(settings["years"]["start"]))
        & (train["year"] <= int(settings["years"]["end"]))
    ]
    synthetic = common.synthetic_satellite_used(data)
    crops = [c for c in settings["crops"] if (data["crop"] == c).any()]
    table = correlation_table(data, list(PREDICTORS), crops)
    common.save_table(table, "correlations", floatfmt=3)
    draw_forest(table, synthetic)

    gdf = common.districts().to_crs("EPSG:7755")
    centroids = gdf.geometry.centroid
    coords = np.column_stack([centroids.x, centroids.y]) / 1000.0
    moran = {}
    for crop in crops:
        means = data[data["crop"] == crop].groupby("district_id")["yield_label"].mean()
        index = gdf["district_id"].isin(means.index).to_numpy()
        moran[crop] = morans_i(
            means.reindex(gdf.loc[index, "district_id"]).to_numpy(), coords[index]
        )

    def pick(crop: str, predictor: str, variant: str, scope: str = "all states") -> dict:
        row = table[
            (table["crop"] == crop)
            & (table["predictor"] == predictor)
            & (table["variant"] == variant)
            & (table["scope"] == scope)
        ]
        return row.iloc[0].to_dict() if len(row) else {}

    results = {
        "period": [int(data["year"].min()), int(data["year"].max())],
        "n_bootstrap": N_BOOT,
        "satellite_features_synthetic": synthetic,
        "morans_i_district_mean_yield": moran,
        "rice_rain_raw": pick("rice", "total_rain", "raw"),
        "rice_rain_within": pick("rice", "total_rain", "within"),
        "rice_tmax_within": pick("rice", "mean_tmax", "within"),
        "within_by_crop_rain": {c: pick(c, "total_rain", "within") for c in crops},
        "within_by_crop_tmax": {c: pick(c, "mean_tmax", "within") for c in crops},
        "rice_rain_within_by_state": {
            s: pick("rice", "total_rain", "within", s) for s in sorted(data["state"].unique())
        },
    }
    common.save_results("correlations", results)
    return results


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
