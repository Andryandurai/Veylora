"""Cluster districts into climate-vulnerability groups (KMeans, k chosen by silhouette).

Features per district (monsoon/kharif climate and rice-yield stability): mean rain, rain CV, mean Tmax,
mean heat days, mean longest dry spell, detrended rice-yield CV. They are standardised, k is chosen in
3..8 by the silhouette score, and each cluster is labelled from its centroid profile (e.g. "dry-hot").

**Leakage guard:** every feature uses only years <= ``splits.yield.train_end`` (2011). The cluster id is
a model feature in Phase 4, so it must not be computed from validation/test-year yields or weather.
Clusters are numbered by increasing mean rainfall so that the id is ordinal and stable.

Output: ``data/processed/district_clusters.parquet`` (district_id, cluster_id, cluster_label, features),
``clusters_map.{png,html}``, ``cluster_profiles.{csv,md}`` and ``reports/analytics/clustering.json``.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from analytics import common, maps
from config import get_path

LOG = logging.getLogger(__name__)

FEATURES = {
    "mean_rain_mm": "Mean kharif rain (mm)",
    "rain_cv": "Kharif rain CV",
    "mean_tmax_c": "Mean kharif Tmax (C)",
    "heat_days": "Heat days / kharif season",
    "dry_spell_days": "Longest dry spell (days)",
    "yield_cv": "Rice yield CV (detrended)",
}
K_RANGE = range(3, 9)
Z_CUT = 0.5


def build_features(sw: pd.DataFrame, rice: pd.DataFrame, last_year: int) -> pd.DataFrame:
    """District feature table from years <= ``last_year`` only.

    Args:
        sw: v_season_weather rows.
        rice: v_training_table rows for rice (district_id, year, yield_label).
        last_year: last year allowed (the yield training end).
    """
    kharif = sw[(sw["season"] == "kharif") & (sw["year"] <= last_year)]
    grouped = kharif.groupby("district_id")
    features = pd.DataFrame(
        {
            "mean_rain_mm": grouped["total_rain"].mean(),
            "rain_cv": grouped["total_rain"].std() / grouped["total_rain"].mean(),
            "mean_tmax_c": grouped["mean_tmax"].mean(),
            "heat_days": grouped["heat_days"].mean(),
            "dry_spell_days": grouped["longest_dry_spell_days"].mean(),
        }
    )
    features["yield_cv"] = detrended_cv(rice[rice["year"] <= last_year])
    return features


def detrended_cv(rice: pd.DataFrame) -> pd.Series:
    """Per-district CV of the residuals from a linear yield trend (stability net of technology gains)."""
    values = {}
    for district, group in rice.groupby("district_id"):
        if len(group) < 8:
            continue
        slope, intercept = np.polyfit(
            group["year"].to_numpy(float), group["yield_label"].to_numpy(float), 1
        )
        resid = group["yield_label"].to_numpy(float) - (
            intercept + slope * group["year"].to_numpy(float)
        )
        values[district] = float(resid.std(ddof=1) / group["yield_label"].mean())
    return pd.Series(values, name="yield_cv")


def choose_k(scaled: np.ndarray, seed: int = 42) -> tuple[int, dict[int, float], dict[int, KMeans]]:
    """Fit KMeans for each k in 3..8 and pick the best silhouette (smaller k wins ties)."""
    scores, models = {}, {}
    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=50, random_state=seed).fit(scaled)
        scores[k] = float(silhouette_score(scaled, model.labels_))
        models[k] = model
    best = max(scores, key=lambda k: (round(scores[k], 6), -k))
    return best, scores, models


def label_clusters(centroid_z: pd.DataFrame) -> list[str]:
    """Human-readable names from standardised centroids: '<rain>-<temperature>[-variable|-stable]'.

    Base name (z = centroid in standardised units): rain z >= +0.5 'wet', <= -0.5 'dry', else
    'moderate'; Tmax z >= +0.5 'hot', <= -0.5 'cool', else 'warm'. Clusters that still share a name
    get distinguishing traits appended, tried in this order until the names are unique: 'variable-rain'
    / 'stable-rain' (rain CV), 'heat-prone' (heat days), 'long-dry-spells', 'volatile-yield' /
    'stable-yield' (rice yield CV). Anything still tied gets its cluster number appended.
    """
    base = []
    for _, row in centroid_z.iterrows():
        rain = (
            "wet"
            if row["mean_rain_mm"] >= Z_CUT
            else "dry"
            if row["mean_rain_mm"] <= -Z_CUT
            else "moderate"
        )
        temp = (
            "hot"
            if row["mean_tmax_c"] >= Z_CUT
            else "cool"
            if row["mean_tmax_c"] <= -Z_CUT
            else "warm"
        )
        base.append(f"{rain}-{temp}")

    def traits(row: pd.Series) -> list[str | None]:
        return [
            "variable-rain"
            if row["rain_cv"] >= Z_CUT
            else "stable-rain"
            if row["rain_cv"] <= -Z_CUT
            else None,
            "heat-prone" if row["heat_days"] >= Z_CUT else None,
            "long-dry-spells" if row["dry_spell_days"] >= Z_CUT else None,
            "volatile-yield"
            if row["yield_cv"] >= Z_CUT
            else "stable-yield"
            if row["yield_cv"] <= -Z_CUT
            else None,
        ]

    labels = list(base)
    rows = [traits(row) for _, row in centroid_z.iterrows()]
    for slot in range(4):
        for name in set(labels):
            members = [i for i, label in enumerate(labels) if label == name]
            if len(members) > 1:
                for i in members:
                    if rows[i][slot]:
                        labels[i] = f"{labels[i]}-{rows[i][slot]}"
    return [
        label if labels.count(label) == 1 else f"{label} #{i}" for i, label in enumerate(labels)
    ]


def run() -> dict:
    """Cluster, label, map and save."""
    settings = common.load_settings()
    last_year = common.train_end_year()
    sw = common.season_weather()
    train = common.training_table()
    features = build_features(sw, train[train["crop"] == "rice"], last_year).dropna()
    scaler = StandardScaler().fit(features[list(FEATURES)])
    scaled = scaler.transform(features[list(FEATURES)])
    best_k, scores, models = choose_k(scaled, int(settings["project"]["random_seed"]))
    model = models[best_k]

    centroid_z = pd.DataFrame(model.cluster_centers_, columns=list(FEATURES))
    order = centroid_z["mean_rain_mm"].argsort().to_numpy()  # id 0 = driest
    remap = {int(old): new for new, old in enumerate(order)}
    centroid_z = centroid_z.iloc[order].reset_index(drop=True)
    names = label_clusters(centroid_z)
    assignment = pd.DataFrame(
        {
            "district_id": features.index,
            "cluster_id": [remap[int(c)] for c in model.labels_],
        }
    )
    assignment["cluster_label"] = assignment["cluster_id"].map(dict(enumerate(names)))
    clusters = assignment.merge(features.reset_index(names="district_id"), on="district_id")
    out_path = get_path("data_processed") / "district_clusters.parquet"
    clusters.to_parquet(out_path, index=False)

    centroids_raw = pd.DataFrame(
        scaler.inverse_transform(centroid_z.to_numpy()), columns=list(FEATURES)
    )
    profile = pd.concat(
        [
            pd.DataFrame(
                {
                    "cluster_id": range(best_k),
                    "cluster_label": names,
                    "n_districts": clusters["cluster_id"].value_counts().sort_index().to_numpy(),
                }
            ),
            centroids_raw,
        ],
        axis=1,
    )
    common.save_table(profile, "cluster_profiles", floatfmt=2)

    gdf = common.districts().merge(
        clusters[["district_id", "cluster_id", "cluster_label"]], on="district_id"
    )
    common.apply_mpl_theme()
    fig, (ax_map, ax_sil) = plt.subplots(
        1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    legend_labels = [
        f"{i}: {n} ({int((clusters['cluster_id'] == i).sum())})" for i, n in enumerate(names)
    ]
    gdf["cluster_legend"] = gdf["cluster_id"].map(dict(enumerate(legend_labels)))
    maps.mpl_ordinal_map(
        ax_map,
        gdf,
        "cluster_legend",
        legend_labels,
        title="Climate-vulnerability clusters (ordered dry to wet)",
    )
    ks = list(scores)
    ax_sil.plot(ks, [scores[k] for k in ks], marker="o", color=common.SERIES[0], markersize=6)
    ax_sil.plot(
        [best_k],
        [scores[best_k]],
        marker="o",
        markersize=10,
        color=common.SERIES[0],
        markeredgecolor=common.INK,
        markeredgewidth=1.5,
    )
    ax_sil.set_xlabel("number of clusters k")
    ax_sil.set_ylabel("silhouette score")
    ax_sil.set_title("Choosing k")
    common.save_mpl(fig, "clusters_map")
    ramp = LinearSegmentedColormap.from_list("ordinal", common.ordinal_ramp(best_k), N=best_k)
    maps.folium_choropleth(
        gdf,
        "cluster_id",
        "clusters_map",
        "Climate-vulnerability clusters",
        cmap=ramp,
        vmin=0,
        vmax=max(best_k - 1, 1),
        legend_caption="cluster id (0 = driest)",
        tooltip_columns=["district", "state", "cluster_label", "cluster_id"],
        value_format="{:.0f}",
    )

    results = {
        "train_years_up_to": last_year,
        "n_districts": int(len(clusters)),
        "silhouette_by_k": {str(k): v for k, v in scores.items()},
        "chosen_k": int(best_k),
        "chosen_silhouette": scores[best_k],
        "clusters": [
            {
                "cluster_id": int(i),
                "label": names[i],
                "n_districts": int((clusters["cluster_id"] == i).sum()),
                "example_districts": gdf[gdf["cluster_id"] == i]
                .sort_values("district")["district"]
                .head(4)
                .tolist(),
                **{f: float(centroids_raw.loc[i, f]) for f in FEATURES},
            }
            for i in range(best_k)
        ],
    }
    common.save_results("clustering", results)
    return results


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
