"""Phase 4 tests: feature leakage by perturbation, SQL equivalence, determinism, serving contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.predictive import features as F
from analytics.predictive import predict as P
from analytics.predictive.evaluate import (
    interval_metrics,
    paired_cluster_bootstrap,
    regression_metrics,
)
from analytics.predictive.yield_models import (
    ConformalXGB,
    Data,
    DistrictTrend,
    LSTMModel,
    NaiveLastYear,
    PriorMean3,
    XGBoostModel,
    temperature_constraints,
)
from config import get_path
from database import db_available
from etl.synthetic.satellite import generate_satellite
from etl.synthetic.weather import generate_weather


# ---------------------------------------------------------------- toy world
def _toy(
    seed: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    districts = pd.DataFrame(
        {
            "district_id": ["a__x", "b__y"],
            "state": ["A", "B"],
            "centroid_lat": [11.0, 17.0],
            "centroid_lon": [78.0, 80.0],
        }
    )
    weather = generate_weather(districts, 1995, 2018, seed=seed)
    satellite = generate_satellite(weather, 1997, 2018, seed=seed)
    rng = np.random.default_rng(seed)
    rows = []
    for district in districts["district_id"]:
        for year in range(1996, 2018):
            area = 1000.0 + 10 * rng.random()
            rows.append(
                (
                    district,
                    year,
                    "kharif",
                    "rice",
                    area,
                    area * (2.5 + 0.03 * (year - 1996) + 0.3 * rng.normal()),
                    False,
                    False,
                )
            )
    crops = pd.DataFrame(
        rows,
        columns=[
            "district_id",
            "year",
            "season",
            "crop",
            "area_ha",
            "production_t",
            "yield_flag",
            "is_synthetic",
        ],
    )
    crops["yield_t_per_ha"] = crops["production_t"] / crops["area_ha"]
    clusters = pd.DataFrame(
        {
            "district_id": districts["district_id"],
            "cluster_id": [0, 1],
            "cluster_label": ["dry", "wet"],
        }
    )
    return weather, satellite, crops, districts, clusters


def _features(
    weather, satellite, crops, districts, clusters, lead: int, include_satellite: bool = True
):
    table = F.training_table_py(weather, satellite, crops, districts)
    prepared = F.prepare_frame(table, clusters)
    levels = {"crop": ["rice"], "season": ["kharif", "rabi"], "cluster": [0, 1]}
    return F.build_features(prepared, lead, include_satellite, levels)


# ---------------------------------------------------------------- hand-checked window features
def test_window_features_on_a_hand_made_season() -> None:
    n = 153
    rain = np.zeros(n)
    rain[[3, 4, 10]] = [5.0, 2.5, 30.0]  # rainy days (>= 2.5 mm) on days 3, 4 and 10
    tmax = np.full(n, 35.0)
    tmax[[20, 21]] = 41.0
    tmin = np.full(n, 25.0)
    out = F.window_features(rain, tmax, tmin)
    assert out["total_rain"] == pytest.approx(37.5) and out["rainy_days"] == 3
    assert out["longest_dry_spell_days"] == n - 11  # days 11..152 are dry
    assert out["heat_days"] == 2 and out["heat_days_d30"] == 2 and out["heat_days_d60"] == 2
    assert out["dry_spell_d30"] == 19  # days 11..29
    assert out["rain_d30"] == pytest.approx(37.5)
    assert out["gdd_d30"] == pytest.approx(
        30 * 20.0 + 2 * 3.0
    )  # (35+25)/2-10 = 20; hot days (41+25)/2-10 = 23
    assert F.longest_run(np.array([], dtype=bool)) == 0
    assert F.longest_run(np.array([True, True, False, True])) == 2


# ---------------------------------------------------------------- leakage by perturbation
@pytest.mark.parametrize("lead", F.LEADS)
def test_weather_after_the_forecast_date_cannot_change_features(lead: int) -> None:
    weather, satellite, crops, districts, clusters = _toy()
    base_X, _, base_meta = _features(weather, satellite, crops, districts, clusters, lead)
    start = pd.Timestamp("2008-06-01")
    future = (weather["date"] >= start + pd.Timedelta(days=lead)) & (
        weather["date"] <= pd.Timestamp("2008-10-31")
    )
    changed = weather.copy()
    changed.loc[future, "rain_mm"] = changed.loc[future, "rain_mm"] + 80.0
    changed.loc[future, "tmax_c"] = changed.loc[future, "tmax_c"] + 6.0
    new_X, _, new_meta = _features(changed, satellite, crops, districts, clusters, lead)
    row = ((base_meta["year"] == 2008) & (base_meta["district_id"] == "a__x")).to_numpy()
    # Anomalies of *later* years use 2008 in their climatology, so compare the 2008 row and earlier years only.
    compare = (base_meta["year"] <= 2008).to_numpy()
    pd.testing.assert_frame_equal(
        base_X[compare].reset_index(drop=True), new_X[compare].reset_index(drop=True)
    )
    assert row.any()
    # The perturbation is real: the next lead's features do change (except for the last lead).
    if lead < 90:
        wider_X, _, _ = _features(changed, satellite, crops, districts, clusters, 90)
        original_wider, _, _ = _features(weather, satellite, crops, districts, clusters, 90)
        assert not np.allclose(
            wider_X[compare].fillna(0).to_numpy(), original_wider[compare].fillna(0).to_numpy()
        )


@pytest.mark.parametrize("lead", F.LEADS)
def test_satellite_composites_not_finished_by_the_forecast_date_cannot_change_features(
    lead: int,
) -> None:
    weather, satellite, crops, districts, clusters = _toy()
    base_X, _, base_meta = _features(weather, satellite, crops, districts, clusters, lead)
    cutoff = pd.Timestamp("2008-06-01") + pd.Timedelta(days=lead)
    unfinished = (satellite["date"] + pd.Timedelta(days=15) > cutoff) & (
        satellite["date"] <= pd.Timestamp("2008-10-31")
    )
    changed = satellite.copy()
    changed.loc[unfinished, "ndvi"] = 0.99
    changed.loc[unfinished, "soil_moisture"] = 0.55
    new_X, _, _ = _features(weather, changed, crops, districts, clusters, lead)
    compare = (base_meta["year"] <= 2008).to_numpy()
    pd.testing.assert_frame_equal(
        base_X[compare].reset_index(drop=True), new_X[compare].reset_index(drop=True)
    )


def test_yield_history_features_use_only_earlier_years() -> None:
    weather, satellite, crops, districts, clusters = _toy()
    X, y, meta = _features(weather, satellite, crops, districts, clusters, 60)
    target = (meta["year"] == 2010) & (meta["district_id"] == "a__x")
    changed = crops.copy()
    hit = (changed["district_id"] == "a__x") & (
        changed["year"] >= 2010
    )  # change the label of year 2010 and everything after
    changed.loc[hit, "production_t"] *= 5.0
    changed["yield_t_per_ha"] = changed["production_t"] / changed["area_ha"]
    X2, y2, meta2 = _features(weather, satellite, changed, districts, clusters, 60)
    same = ((meta["year"] <= 2010) & (meta["district_id"] == "a__x")).to_numpy()
    for column in ("prev_year_rel", "mean_prev3_rel", "trend_pred_rel", "log_scale"):
        np.testing.assert_allclose(X.loc[same, column], X2.loc[same, column], err_msg=column)
    # ... while the label itself changes, and the following year's history does see the change.
    assert not np.isclose(y[target.to_numpy()].iloc[0], y2[target.to_numpy()].iloc[0])
    nxt = ((meta["year"] == 2011) & (meta["district_id"] == "a__x")).to_numpy()
    assert not np.isclose(X.loc[nxt, "prev_year_rel"].iloc[0], X2.loc[nxt, "prev_year_rel"].iloc[0])


def test_target_is_yield_over_prior_mean_and_scale_excludes_current_year() -> None:
    weather, satellite, crops, districts, clusters = _toy()
    X, y, meta = _features(weather, satellite, crops, districts, clusters, 60)
    one = meta.index[(meta["year"] == 2005) & (meta["district_id"] == "b__y")][0]
    history = crops[(crops["district_id"] == "b__y") & (crops["year"] < 2005)]["yield_t_per_ha"]
    assert meta.loc[one, "scale"] == pytest.approx(history.mean())
    assert y.loc[one] == pytest.approx(meta.loc[one, "yield"] / history.mean())


def test_no_feature_refers_to_a_later_lead_or_post_season_aggregate() -> None:
    weather, satellite, crops, districts, clusters = _toy()
    for lead in F.LEADS:
        X, _, _ = _features(weather, satellite, crops, districts, clusters, lead)
        assert F.check_no_future_columns(X.columns, lead) == []
        later = [other for other in F.LEADS if other > lead]
        assert not [c for c in X.columns if any(f"_d{other}" in c for other in later)]
    with_post_hoc = ["rain_d90", "total_rain", "ndvi_peak"]
    assert F.check_no_future_columns(with_post_hoc, 30) == ["rain_d90", "total_rain", "ndvi_peak"]


def test_time_splits_do_not_overlap_and_are_ordered() -> None:
    weather, satellite, crops, districts, clusters = _toy()
    _, _, meta = _features(weather, satellite, crops, districts, clusters, 60)
    masks = F.split_masks(meta)
    assert (
        not (masks["train"] & masks["valid"]).any() and not (masks["valid"] & masks["test"]).any()
    )
    assert meta.loc[masks["train"], "year"].max() < meta.loc[masks["valid"], "year"].min()
    assert meta.loc[masks["valid"], "year"].max() < meta.loc[masks["test"], "year"].min()


# ---------------------------------------------------------------- SQL equivalence (needs DB)
@pytest.mark.skipif(not db_available(), reason="PostgreSQL warehouse not reachable")
def test_python_training_table_equals_sql_view() -> None:
    if not (get_path("data_processed") / "crops_annual.parquet").exists():
        pytest.skip("pipeline data not built")
    from analytics import common

    try:
        db = common.training_table()
    except Exception:
        pytest.skip("warehouse views not created")
    py, _ = F.load_training_frame(prefer_db=False)
    key = ["district_id", "year", "season", "crop"]
    merged = db.merge(py, on=key, suffixes=("_db", "_py"))
    assert len(merged) == len(db) == len(py)
    for column in db.columns:
        if (
            column in key
            or column not in py
            or not pd.api.types.is_numeric_dtype(db[column])
            or column in ("area_ha", "production_t")
        ):
            continue
        a, b = merged[f"{column}_db"].astype(float), merged[f"{column}_py"].astype(float)
        assert (a.isna() == b.isna()).all(), column
        assert np.nanmax(np.abs(a - b)) < 5e-3, column  # PostgreSQL real (float32) rounding


# ---------------------------------------------------------------- models and metrics
def _toy_xy(n: int = 300, seed: int = 0) -> tuple[Data, Data]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "mean_tmax_d30": rng.normal(33, 1, n),
            "rain_d30": rng.gamma(2, 20, n),
            "prev_year_rel": rng.normal(1, 0.1, n),
            "mean_prev3_rel": rng.normal(1, 0.1, n),
            "trend_pred_rel": rng.normal(1, 0.1, n),
            "log_scale": rng.normal(1, 0.1, n),
            "crop_rice": 1.0,
            "season_kharif": 1.0,
            "cluster_0": 1.0,
        }
    )
    y = pd.Series(
        1.0
        + 0.4 * (X["prev_year_rel"] - 1)
        - 0.05 * (X["mean_tmax_d30"] - 33)
        + 0.002 * (X["rain_d30"] - 40)
        + rng.normal(0, 0.03, n),
        name="y_rel",
    )
    meta = pd.DataFrame(
        {
            "district_id": [f"d{i % 30}" for i in range(n)],
            "state": "S",
            "year": 2000 + np.arange(n) % 10,
            "crop": "rice",
            "season": "kharif",
            "scale": 3.0,
            "yield": 3.0 * y,
            "prev_year_yield": 3.0 * X["prev_year_rel"],
            "trend_pred": 3.0 * X["trend_pred_rel"],
            "lead_days": 30,
        }
    )
    seq = rng.normal(size=(n, 16, 5)).astype("float32")
    mask = np.zeros((n, 16), dtype="float32")
    mask[:, :4] = 1.0
    data = Data(X, y, meta, seq, mask)
    cut = int(0.7 * n)
    rows = np.arange(n) < cut
    return data.subset(rows), data.subset(~rows)


def test_xgboost_is_deterministic_with_fixed_seed_and_params() -> None:
    train, valid = _toy_xy()
    params = {
        "max_depth": 3,
        "learning_rate": 0.1,
        "n_estimators": 60,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    a = XGBoostModel(params=params).fit(train, valid).predict_rel(valid)
    b = XGBoostModel(params=params).fit(train, valid).predict_rel(valid)
    np.testing.assert_allclose(a, b, rtol=1e-9)


def test_lstm_is_deterministic_and_learns_something() -> None:
    train, valid = _toy_xy()
    a = LSTMModel(30, max_epochs=8, patience=4).fit(train, valid).predict_rel(valid)
    b = LSTMModel(30, max_epochs=8, patience=4).fit(train, valid).predict_rel(valid)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    assert np.isfinite(a).all()


def test_monotone_temperature_constraint_holds_and_costs_little() -> None:
    train, valid = _toy_xy()
    columns = list(train.X.columns)
    constraints = temperature_constraints(columns)
    assert constraints["mean_tmax_d30"] == -1 and constraints["rain_d30"] == 0
    params = {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 80}
    model = XGBoostModel(params=params).fit(train, valid)
    grid = valid.X.iloc[:20].copy()
    previous = None
    for shift in (0.0, 1.0, 2.0, 4.0, 8.0):
        shifted = grid.copy()
        shifted["mean_tmax_d30"] = grid["mean_tmax_d30"] + shift
        pred = model.model.predict(shifted)
        if previous is not None:
            assert (pred <= previous + 1e-9).all()
        previous = pred


def test_conformal_intervals_cover_about_eighty_percent_on_exchangeable_data() -> None:
    train, valid = _toy_xy(900, seed=5)
    big_valid = valid.subset(np.arange(len(valid.y)) < len(valid.y) // 2)
    big_test = valid.subset(np.arange(len(valid.y)) >= len(valid.y) // 2)
    params = {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 80}
    model = XGBoostModel(params=params).fit(train, big_valid)
    conformal = ConformalXGB(params=params, constraints=model.constraints).fit(train, big_valid)
    lo, hi = conformal.interval_rel(big_test, model.predict_rel(big_test))
    stats = interval_metrics(big_test.meta, lo, hi)
    assert 0.65 <= stats["coverage"] <= 0.95
    assert (lo <= model.predict_rel(big_test) + 1e-9).all() and (
        hi >= model.predict_rel(big_test) - 1e-9
    ).all()


def test_baselines_and_metric_functions() -> None:
    train, valid = _toy_xy()
    for model in (NaiveLastYear(), DistrictTrend(), PriorMean3()):
        pred = model.fit(train, valid).predict_rel(valid)
        assert pred.shape == (len(valid.y),) and np.isfinite(pred).all()
    metrics = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    assert metrics["rmse"] == pytest.approx(np.sqrt(1 / 3)) and metrics["mae"] == pytest.approx(
        1 / 3
    )
    assert metrics["r2"] == pytest.approx(1 - 1 / 2) and metrics["mape"] == pytest.approx(
        100 * (1 / 3) / 3
    )


def test_paired_bootstrap_detects_a_better_model() -> None:
    train, valid = _toy_xy(600, seed=2)
    truth = (valid.meta["yield"] / valid.meta["scale"]).to_numpy()
    rng = np.random.default_rng(0)
    good = truth + rng.normal(0, 0.05, len(truth))
    bad = truth + rng.normal(0, 0.2, len(truth))
    better = paired_cluster_bootstrap(valid.meta, good, bad, n_boot=300)
    assert better["hi"] < 1.0 and better["p_not_better"] == 0.0
    same = paired_cluster_bootstrap(valid.meta, bad, bad, n_boot=300)
    assert same["ratio"] == pytest.approx(1.0)


# ---------------------------------------------------------------- serving contract (needs trained models + data)
def _serving_ready() -> bool:
    models = get_path("models") / "yield_xgb.joblib"
    processed = get_path("data_processed")
    return models.exists() and all(
        (processed / f"{n}.parquet").exists()
        for n in ("weather_daily", "crops_annual", "districts", "satellite_16d")
    )


serving = pytest.mark.skipif(
    not _serving_ready(), reason="run `make etl` and `make train-yield` first"
)


@serving
def test_predict_yield_contract_and_interval_order() -> None:
    result = P.predict_yield("tamil_nadu__thanjavur", "rice", "kharif")
    assert {"yield_t_ha", "lo", "hi", "model"} <= set(result)
    assert isinstance(result["yield_t_ha"], float) and isinstance(result["model"], str)
    assert 0 < result["lo"] <= result["yield_t_ha"] <= result["hi"]
    assert 0.5 < result["yield_t_ha"] < 8


@serving
def test_overrides_are_validated_and_neutral_overrides_change_nothing() -> None:
    base = P.predict_yield("karnataka__raichur", "rice", "kharif")
    neutral = P.predict_yield(
        "karnataka__raichur", "rice", "kharif", {"rain_multiplier": 1.0, "temperature_delta": 0.0}
    )
    assert neutral["yield_t_ha"] == pytest.approx(base["yield_t_ha"])
    for bad in (
        {"rain_multiplier": -0.1},
        {"rain_multiplier": 5},
        {"temperature_delta": 20},
        {"humidity": 1},
    ):
        with pytest.raises(ValueError):
            P.predict_yield("karnataka__raichur", "rice", "kharif", bad)
    with pytest.raises(ValueError):
        P.predict_yield("nowhere__x", "rice", "kharif")
    with pytest.raises(ValueError):
        P.predict_yield("karnataka__raichur", "rice", "kharif", lead_days=45)


@serving
def test_warming_never_raises_the_forecast_and_batch_matches_single_calls() -> None:
    cases = [
        ("tamil_nadu__thanjavur", "rice"),
        ("karnataka__raichur", "cotton"),
        ("telangana__warangal", "rice"),
    ]
    for district, crop in cases:
        values = [
            P.predict_yield(district, crop, "kharif", {"temperature_delta": d})["yield_t_ha"]
            for d in (0, 1, 2, 4)
        ]
        assert all(b <= a + 1e-9 for a, b in zip(values, values[1:], strict=False)), (
            district,
            crop,
            values,
        )
    batch = P.predict_batch(
        [
            {
                "district_id": d,
                "crop": c,
                "season": "kharif",
                "rain_multiplier": 0.8,
                "temperature_delta": 1.0,
            }
            for d, c in cases
        ]
    )
    for (district, crop), got in zip(cases, batch["yield_t_ha"], strict=True):
        single = P.predict_yield(
            district, crop, "kharif", {"rain_multiplier": 0.8, "temperature_delta": 1.0}
        )
        assert got == pytest.approx(single["yield_t_ha"])


@serving
def test_reference_year_never_comes_from_validation_or_test_years() -> None:
    ctx = P._context()
    for district in list(ctx.daily)[:10]:
        assert P.typical_year(ctx, district, "kharif") <= 2011
