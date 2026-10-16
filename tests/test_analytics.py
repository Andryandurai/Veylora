"""Phase 3 tests on synthetic series with known answers (no database needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.common import markdown_table, ordinal_ramp
from analytics.descriptive.drought_flood_frequency import classify_events, event_counts
from analytics.descriptive.rainfall_variability import anomaly_matrix, season_stats
from analytics.descriptive.yield_trends import balanced_state_series, mann_kendall, theil_sen
from analytics.diagnostic.clustering import detrended_cv, label_clusters
from analytics.diagnostic.correlations import (
    cluster_bootstrap_corr,
    detrend_within_district,
    morans_i,
)
from analytics.diagnostic.regression import within_vif
from analytics.make_findings import effect_words


# ---------------------------------------------------------------- drought / flood definitions
def _season_frame() -> pd.DataFrame:
    rain = [400, 500, 600, 700, 800, 100, 1400]  # mean = 642.86
    return pd.DataFrame(
        {
            "district_id": "d1",
            "season": "kharif",
            "year": range(2000, 2007),
            "total_rain": rain,
            "heavy_days": [0, 0, 0, 0, 0, 0, 3],
        }
    )


def test_drought_is_below_75_percent_of_long_term_mean() -> None:
    events = classify_events(_season_frame())
    mean = 4500 / 7
    expected = [r < 0.75 * mean for r in [400, 500, 600, 700, 800, 100, 1400]]
    assert events["drought"].tolist() == expected
    assert events["drought"].sum() == 2  # only 400 and 100 are below 0.75 x 642.9 = 482.1
    assert events.loc[events["year"] == 2005, "drought"].item()


def test_flood_risk_needs_wet_season_and_three_heavy_days() -> None:
    frame = _season_frame()
    frame.loc[frame["year"] == 2004, "heavy_days"] = 5  # heavy days but season not wet enough
    events = classify_events(frame)
    assert events.loc[
        events["year"] == 2006, "flood_risk"
    ].item()  # 1400 mm > 125% of mean and 3 heavy days
    assert not events.loc[events["year"] == 2004, "flood_risk"].item()
    wet_but_dry_days = _season_frame().assign(heavy_days=2)
    assert not classify_events(wet_but_dry_days)["flood_risk"].any()


def test_thresholds_are_strict_inequalities_and_per_season() -> None:
    frame = pd.DataFrame(
        {
            "district_id": ["a"] * 4 + ["a"] * 4,
            "season": ["kharif"] * 4 + ["rabi"] * 4,
            "year": list(range(4)) * 2,
            "total_rain": [100.0, 100.0, 100.0, 100.0, 10.0, 10.0, 10.0, 10.0],
            "heavy_days": 0,
        }
    )
    events = classify_events(frame)
    assert not events["drought"].any()  # ratio exactly 1.0 in each season: normals are per season
    counts = event_counts(events)
    assert set(counts["season"]) == {"kharif", "rabi"} and (counts["drought_events"] == 0).all()


# ---------------------------------------------------------------- Mann-Kendall / Theil-Sen wrapper
def test_mann_kendall_detects_known_trends() -> None:
    rng = np.random.default_rng(42)
    t = np.arange(30)
    up = mann_kendall(0.05 * t + rng.normal(0, 0.1, 30))
    down = mann_kendall(-0.05 * t + rng.normal(0, 0.1, 30))
    flat = mann_kendall(rng.normal(0, 1, 30))
    assert up["trend"] == "increasing" and up["p"] < 0.001 and up["tau"] > 0.8
    assert down["trend"] == "decreasing" and down["tau"] < -0.8
    assert flat["trend"] == "no trend" and flat["p"] > 0.05


def test_theil_sen_recovers_slope_and_resists_outliers() -> None:
    years = np.arange(2000, 2020)
    values = 2.0 + 0.1 * (years - 2000)
    values[5] = 40.0  # one wild outlier
    fit = theil_sen(years, values)
    assert fit["slope"] == pytest.approx(0.1, abs=0.01)
    assert fit["lo"] <= fit["slope"] <= fit["hi"]


def test_balanced_panel_ignores_districts_with_gaps_and_thin_series() -> None:
    rows = []
    for district in ("a", "b", "c", "gappy"):
        for year in range(2000, 2012):
            if district == "gappy" and year == 2005:
                continue
            rows.append(("S", district, "rice", year, 100.0, 300.0))
    frame = pd.DataFrame(
        rows, columns=["state", "district_id", "crop", "year", "area_ha", "production_t"]
    )
    series, n = balanced_state_series(frame, "S", "rice")
    assert n == 3 and series.round(6).eq(3.0).all()
    assert balanced_state_series(frame[frame["district_id"].isin(["a", "b"])], "S", "rice") is None


# ---------------------------------------------------------------- rainfall variability
def test_cv_and_anomaly_matrix() -> None:
    sw = pd.DataFrame(
        {
            "district_id": ["a"] * 4 + ["b"] * 4,
            "season": "kharif",
            "year": list(range(4)) * 2,
            "total_rain": [100.0, 100.0, 100.0, 100.0, 50.0, 150.0, 50.0, 150.0],
        }
    )
    stats = season_stats(sw).set_index("district_id")
    assert stats.loc["a", "cv"] == 0.0
    assert stats.loc["b", "cv"] == pytest.approx(np.std([50, 150, 50, 150], ddof=1) / 100)
    matrix = anomaly_matrix(sw, "kharif")
    assert matrix.loc["b", 0] == pytest.approx(-50.0) and matrix.loc["b", 1] == pytest.approx(50.0)
    assert (matrix.loc["a"] == 0).all()


# ---------------------------------------------------------------- correlations
def test_cluster_bootstrap_ci_contains_true_correlation_and_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(40), 10)
    x = rng.normal(size=400)
    y = 0.6 * x + 0.8 * rng.normal(size=400)  # true r = 0.6
    a = cluster_bootstrap_corr(x, y, groups, n_boot=300, seed=42)
    b = cluster_bootstrap_corr(x, y, groups, n_boot=300, seed=42)
    assert a == b
    assert a["pearson_lo"] < 0.6 < a["pearson_hi"]
    assert a["pearson_lo"] < a["pearson"] < a["pearson_hi"]
    assert a["spearman_lo"] < a["spearman"] < a["spearman_hi"]


def test_detrending_removes_district_level_and_trend() -> None:
    frame = pd.DataFrame(
        {
            "district_id": ["a"] * 10 + ["b"] * 10,
            "year": list(range(10)) * 2,
            "y": [5 + 0.3 * t for t in range(10)] + [50 - 1.0 * t for t in range(10)],
        }
    )
    resid = detrend_within_district(frame, ["y"])
    assert resid["y"].abs().max() < 1e-9  # pure level + trend leaves nothing


def test_pooled_correlation_can_have_opposite_sign_to_within_correlation() -> None:
    """Simpson's paradox: the situation the raw/within split exists to expose."""
    rng = np.random.default_rng(0)
    rows = []
    for d in range(12):
        level_x, level_y = 10 * d, 100 - 8 * d  # wetter districts yield less
        for year in range(20):
            x = level_x + rng.normal()
            rows.append(
                (f"d{d}", 2000 + year, x, level_y + 0.5 * (x - level_x) + 0.2 * rng.normal())
            )
    frame = pd.DataFrame(rows, columns=["district_id", "year", "x", "y"])
    pooled = cluster_bootstrap_corr(
        frame["x"].to_numpy(), frame["y"].to_numpy(), frame["district_id"].to_numpy(), 100
    )
    within = detrend_within_district(frame, ["x", "y"])
    inner = cluster_bootstrap_corr(
        within["x"].to_numpy(), within["y"].to_numpy(), within["district_id"].to_numpy(), 100
    )
    assert pooled["pearson"] < -0.5 and inner["pearson"] > 0.5


def test_morans_i_positive_for_smooth_surface_and_near_zero_for_noise() -> None:
    gx, gy = np.meshgrid(np.arange(8), np.arange(8))
    coords = np.column_stack([gx.ravel(), gy.ravel()]).astype(float)
    smooth = morans_i(coords[:, 0] + coords[:, 1], coords, k=4, n_perm=199)
    noise = morans_i(np.random.default_rng(3).normal(size=64), coords, k=4, n_perm=199)
    assert smooth["I"] > 0.6 and smooth["p_one_sided"] < 0.01
    assert abs(noise["I"]) < 0.25 and noise["p_one_sided"] > 0.05


# ---------------------------------------------------------------- regression VIF
def test_within_vif_detects_collinearity_after_partialling_out_fixed_effects() -> None:
    rng = np.random.default_rng(2)
    n_d, n_t = 10, 12
    frame = pd.DataFrame(
        {"district_id": np.repeat(np.arange(n_d), n_t), "year": np.tile(np.arange(n_t), n_d)}
    )
    frame["a"] = rng.normal(size=len(frame))
    frame["b"] = frame["a"] + 0.05 * rng.normal(size=len(frame))  # nearly collinear with a
    frame["c"] = rng.normal(size=len(frame))
    vif = within_vif(frame, ["a", "b", "c"], "C(district_id) + C(year)")
    assert vif["a"] > 50 and vif["b"] > 50
    assert vif["c"] < 2


# ---------------------------------------------------------------- clustering helpers
def test_cluster_labels_are_unique_and_follow_the_rules() -> None:
    z = pd.DataFrame(
        {
            "mean_rain_mm": [-1.5, 0.0, 0.1, 1.8],
            "rain_cv": [-1.0, 0.7, -0.8, -0.9],
            "mean_tmax_c": [1.2, 0.0, 0.2, -1.1],
            "heat_days": [0.0, 0.0, 0.0, -0.5],
            "dry_spell_days": [1.0, 0.0, 0.0, -1.0],
            "yield_cv": [0.0, 0.0, 0.0, 0.0],
        }
    )
    labels = label_clusters(z)
    assert labels[0] == "dry-hot" and labels[3] == "wet-cool"
    assert len(set(labels)) == 4
    assert labels[1].startswith("moderate-warm") and labels[2].startswith("moderate-warm")
    assert labels[1] != labels[2]  # distinguished by rain variability


def test_detrended_yield_cv_ignores_trend() -> None:
    years = np.arange(2000, 2012)
    steady = pd.DataFrame(
        {"district_id": "a", "year": years, "yield_label": 2 + 0.1 * (years - 2000)}
    )
    noisy = pd.DataFrame(
        {
            "district_id": "b",
            "year": years,
            "yield_label": 2 + 0.1 * (years - 2000) + np.tile([0.5, -0.5], 6),
        }
    )
    cv = detrended_cv(pd.concat([steady, noisy]))
    assert cv["a"] < 1e-9 and cv["b"] > 0.1


# ---------------------------------------------------------------- helpers
def test_findings_wording_follows_the_numbers() -> None:
    assert effect_words(0.5, 0.2) == "has no statistically detectable association with yield"
    assert effect_words(0.5, 0.01) == "is associated with higher yield"
    assert effect_words(-0.5, 0.01) == "is associated with lower yield"


def test_markdown_table_and_ordinal_ramp() -> None:
    text = markdown_table(pd.DataFrame({"a": [1.23456], "b": ["x"]}), 2)
    assert text.splitlines()[0] == "| a | b |" and "1.23" in text
    ramp = ordinal_ramp(6)
    assert len(ramp) == 6 and len(set(ramp)) == 6
    assert ramp[0] == "#86b6ef"  # never lighter than step 250
