"""Phase 5 tests: labels on hand-made series, SPI sanity, calibration, leakage, serving contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analytics.predictive import extreme_data as E
from analytics.predictive import extreme_weather as X
from analytics.predictive import labels as L
from analytics.predictive import spi
from config import get_path


# ---------------------------------------------------------------- heatwave labels
def test_smooth_circular_wraps_around_the_year() -> None:
    values = np.zeros(366)
    values[:5] = 30.0
    smooth = L.smooth_circular(values, 30)
    assert smooth.shape == (366,)
    assert smooth[-1] > 0 and smooth[0] > 0  # the bump leaks across the year boundary both ways
    assert smooth.sum() == pytest.approx(values.sum(), rel=1e-6)


def test_qualifying_days_need_both_the_absolute_and_the_departure_condition() -> None:
    normal = np.full(6, 36.0)
    tmax = np.array([40.0, 41.0, 39.9, 45.0, 40.5, 40.4])
    qualifying = L.qualifying_days(tmax, normal)
    # departure >= 4.5 needs Tmax >= 40.5 here, and Tmax >= 40 as well
    assert qualifying.tolist() == [False, True, False, True, True, False]
    hot_normal = np.full(3, 38.0)
    assert not L.qualifying_days(
        np.array([41.0, 42.0, 42.4]), hot_normal
    ).any()  # >= 40 but < 4.5 above normal


def test_heatwave_needs_two_consecutive_days_and_onset_is_the_first_day() -> None:
    qualifying = np.array([0, 1, 0, 1, 1, 1, 0, 0, 1, 1], dtype=bool)
    spell = L.spell_mask(qualifying)
    assert spell.tolist() == [False, False, False, True, True, True, False, False, True, True]
    onset = L.onset_days(spell)
    assert np.flatnonzero(onset).tolist() == [3, 8]
    assert L.spell_mask(np.array([True]), 2).tolist() == [False]


def test_onset_within_horizon_looks_forward_only_and_masks_the_unobserved_end() -> None:
    onset = np.zeros(30, dtype=bool)
    onset[10] = True
    y3, y7 = L.onset_within(onset, 3), L.onset_within(onset, 7)
    assert np.flatnonzero(y3).tolist() == [
        7,
        8,
        9,
    ]  # onset on day 10 is within the next 3 days of days 7..9
    assert np.flatnonzero(y7).tolist() == [3, 4, 5, 6, 7, 8, 9]
    assert not y3[10]  # the onset day itself is not "in the next days"
    onset_late = np.zeros(30, dtype=bool)
    onset_late[28] = True
    late = L.onset_within(onset_late, 3)
    assert np.flatnonzero(late).tolist() == [
        25,
        26,
        27,
    ]  # day 27's window is cut off but contains the onset
    assert not late[28:].any()  # nothing after the onset day can see it
    assert not L.onset_within(np.zeros(30, dtype=bool), 3).any()


def test_heatwave_frame_on_a_synthetic_hot_spell() -> None:
    dates = pd.date_range("2000-01-01", "2002-12-31")
    tmax = np.full(len(dates), 33.0)
    spike = (dates >= "2002-05-10") & (dates <= "2002-05-13")
    tmax[spike] = 43.0
    weather = pd.DataFrame({"district_id": "d", "date": dates, "tmax_c": tmax})
    normals = L.daily_normals(weather[weather["date"].dt.year <= 2001], 2001, first_year=2000)
    frame = L.heatwave_frame(weather, normals)
    assert frame.loc[frame["onset"], "date"].tolist() == [pd.Timestamp("2002-05-10")]
    assert (
        int(frame["heatwave"].sum()) == 4 and int(frame["severe"].sum()) == 4
    )  # departure 10 C >= 6.5
    assert frame.loc[frame["date"] == "2002-05-05", "y7"].item()
    assert not frame.loc[frame["date"] == "2002-05-12", "y7"].item()  # onset already happened


def test_slot_does_not_shift_after_february_in_leap_years() -> None:
    dates = pd.Series(pd.to_datetime(["2000-03-01", "2001-03-01", "2000-12-31", "2001-12-31"]))
    slots = L._slot(dates).tolist()
    assert slots[0] == slots[1] == 60 and slots[2] == slots[3] == 365


# ---------------------------------------------------------------- SPI
def test_spi_has_mean_zero_and_sd_one_on_a_long_gamma_series() -> None:
    rng = np.random.default_rng(1)
    totals = rng.gamma(2.0, 60.0, 600)
    fit = spi.fit_gamma(totals)
    values = spi.transform(fit, totals)
    assert abs(values.mean()) < 0.05 and abs(values.std() - 1.0) < 0.05
    assert (
        spi.transform(fit, np.array([10.0]))[0] < spi.transform(fit, np.array([200.0]))[0]
    )  # monotone in rainfall


def test_spi_handles_zero_totals_without_infinities_and_is_lower_for_zero() -> None:
    rng = np.random.default_rng(2)
    totals = np.concatenate([np.zeros(40), rng.gamma(2.0, 30.0, 160)])
    fit = spi.fit_gamma(totals)
    values = spi.transform(fit, np.array([0.0, 20.0, 100.0]))
    assert np.isfinite(values).all() and values[0] < values[1] < values[2]
    assert fit.zero_share == pytest.approx(0.2)


def test_spi_table_reference_period_and_drought_label() -> None:
    dates = pd.date_range("1991-01-01", "2024-12-31")
    rng = np.random.default_rng(3)
    weather = pd.DataFrame(
        {"district_id": "d", "date": dates, "rain_mm": rng.gamma(0.4, 8.0, len(dates))}
    )
    table = spi.spi_table(spi.monthly_rain(weather), 2014, scales=(1, 3))
    reference = table[table["month"].dt.year <= 2014]["spi3"].dropna()
    assert abs(reference.mean()) < 0.1 and 0.85 < reference.std() < 1.1
    assert table["spi3"].iloc[:2].isna().all()  # needs three months
    assert L.drought_label(np.array([-1.0, -0.99, -2.0, np.nan])).tolist() == [
        True,
        False,
        True,
        False,
    ]
    # moving the evaluation years' rainfall must not change the reference-period SPI (no leakage)
    shifted = weather.copy()
    shifted.loc[shifted["date"].dt.year >= 2015, "rain_mm"] *= 3.0
    table2 = spi.spi_table(spi.monthly_rain(shifted), 2014, scales=(1, 3))
    np.testing.assert_allclose(
        table.loc[table["month"].dt.year <= 2014, "spi3"].to_numpy(),
        table2.loc[table2["month"].dt.year <= 2014, "spi3"].to_numpy(),
        equal_nan=True,
    )
    assert (
        table2.loc[table2["month"].dt.year >= 2016, "spi3"].mean()
        > table.loc[table["month"].dt.year >= 2016, "spi3"].mean() + 0.5
    )


# ---------------------------------------------------------------- calibration and metrics
def test_isotonic_calibration_fixes_a_biased_forecast() -> None:
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.0, 0.5, 4000)
    y = (rng.random(4000) < true_p).astype(int)
    raw = np.clip(true_p * 1.8 + 0.1, 0, 1)  # systematically over-confident
    calibrator = X.Calibrator().fit(raw, y)
    assert calibrator.kind == "isotonic"
    before = abs(raw.mean() - y.mean())
    after = abs(calibrator.predict(raw).mean() - y.mean())
    assert after < before / 5
    out = calibrator.predict(np.array([0.0, 0.5, 1.0]))
    assert (np.diff(out) >= -1e-12).all() and (
        (out >= 0) & (out <= 1)
    ).all()  # monotone and a valid probability


def test_calibrator_falls_back_to_logistic_when_positives_are_scarce_and_to_identity_without_any() -> (
    None
):
    rng = np.random.default_rng(1)
    raw = rng.uniform(0, 0.2, 3000)
    y = (rng.random(3000) < raw / 20).astype(int)
    assert 0 < y.sum() < X.MIN_POSITIVES_FOR_ISOTONIC
    assert X.Calibrator().fit(raw, y).kind == "platt"
    assert X.Calibrator().fit(raw, np.zeros(3000)).kind == "identity"
    np.testing.assert_allclose(X.Calibrator().fit(raw, np.zeros(3000)).predict(raw), raw)


def test_binary_metrics_and_threshold_choice() -> None:
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
    m = X.binary_metrics(y, p, 0.5)
    assert (
        m["precision"] == 1.0
        and m["recall"] == pytest.approx(2 / 3)
        and m["f1"] == pytest.approx(0.8)
    )
    assert m["roc_auc"] == pytest.approx(stats.mannwhitneyu(p[y == 1], p[y == 0]).statistic / 9)
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))
    assert 0.3 <= X.best_f1_threshold(y, p) <= 0.9
    assert X.brier_skill(y, p, np.full(6, 0.5)) > 0
    assert np.isnan(
        X.binary_metrics(np.zeros(5, int), np.zeros(5), 0.5)["pr_auc"]
    )  # undefined without positives


def test_reliability_table_bins_are_ordered() -> None:
    rng = np.random.default_rng(4)
    p = rng.uniform(0, 1, 5000)
    y = (rng.random(5000) < p).astype(int)
    table = X.reliability_table(y, p, 10)
    assert len(table) == 10 and (np.diff(table["mean_pred"]) > 0).all()
    assert np.abs(table["mean_pred"] - table["observed"]).max() < 0.1


# ---------------------------------------------------------------- risk levels
def test_levels_and_combination_use_per_hazard_thresholds() -> None:
    thresholds = {
        "heatwave": {"moderate": 0.05, "high": 0.15, "severe": 0.35},
        "drought": {"moderate": 0.30, "high": 0.50, "severe": 0.70},
    }
    assert X.level_from_probability("heatwave", 0.04, thresholds) == "low"
    assert X.level_from_probability("heatwave", 0.05, thresholds) == "moderate"
    assert X.level_from_probability("heatwave", 0.40, thresholds) == "severe"
    assert X.level_from_probability("drought", 0.40, thresholds) == "moderate"
    level, hazard, lead = X.combine_levels(0.20, 0.35, thresholds)
    assert (level, hazard, lead) == (
        "high",
        "heatwave",
        14,
    )  # heatwave 'high' outranks drought 'moderate'
    level, hazard, lead = X.combine_levels(0.01, 0.75, thresholds)
    assert (level, hazard, lead) == ("severe", "drought", 30)
    assert X.combine_levels(0.0, 0.0, thresholds)[0] == "low"


# ---------------------------------------------------------------- leakage of windows
def test_window_features_end_on_the_as_of_day_and_never_include_later_days() -> None:
    dates = pd.date_range("1991-01-01", "1993-12-31")
    n = len(dates)
    rng = np.random.default_rng(0)
    weather = pd.DataFrame(
        {
            "district_id": "d",
            "date": dates,
            "tmax_c": rng.normal(33, 2, n),
            "tmin_c": 22.0,
            "rain_mm": 0.0,
            "rh_pct": 60.0,
            "wind_ms": 2.0,
            "solar_mj": 18.0,
            "is_synthetic": False,
        }
    )
    heat = E.build_heat_data(weather, 1992)
    mean, std = heat.fit_standardiser(1991, 1992)
    t = int(np.flatnonzero(dates == "1993-05-20")[0])
    base = heat.windows(np.array([0]), np.array([t]), mean, std)
    changed = weather.copy()
    changed.loc[changed["date"] > "1993-05-20", "tmax_c"] += 20.0
    heat2 = E.build_heat_data(changed, 1992)
    again = heat2.windows(np.array([0]), np.array([t]), mean, std)
    np.testing.assert_allclose(base, again)  # days after the as-of day cannot enter the window
    assert base.shape == (1, E.WINDOW_DAYS, len(E.FEATURE_NAMES))
    d, ti = heat.sample_index(1993, 1993)
    assert set(dates[ti].month) <= set(E.SEASON_MONTHS) and (ti >= E.WINDOW_DAYS - 1).all()


# ---------------------------------------------------------------- serving contract (needs trained artifacts)
def _ready() -> bool:
    models, processed = get_path("models"), get_path("data_processed")
    return (
        (models / X.HEAT_ARTIFACT).exists()
        and (models / X.DROUGHT_ARTIFACT).exists()
        and (processed / "weather_daily.parquet").exists()
    )


serving = pytest.mark.skipif(not _ready(), reason="run `make etl` and `make train-extreme` first")


@serving
def test_risk_forecast_contract_and_ranges() -> None:
    result = X.risk_forecast("andhra_pradesh__east_godavari", "2023-05-20")
    assert {"heatwave_p", "drought_p", "level", "lead_days"} <= set(result)
    assert 0.0 <= result["heatwave_p"] <= 1.0 and 0.0 <= result["drought_p"] <= 1.0
    assert result["level"] in X.LEVELS and result["lead_days"] in (
        X.HEAT_LEAD_DAYS,
        X.DROUGHT_LEAD_DAYS,
    )
    assert result["heatwave_in_season"] is True
    assert result["heatwave_source"] in ("lstm", "climatology")


@serving
def test_risk_forecast_out_of_season_heatwave_is_zero_and_errors_are_clear() -> None:
    winter = X.risk_forecast("tamil_nadu__thanjavur", "2022-01-15")
    assert winter["heatwave_p"] == 0.0 and winter["heatwave_in_season"] is False
    with pytest.raises(ValueError, match="unknown district"):
        X.risk_forecast("nowhere__x", "2022-05-01")
    with pytest.raises(ValueError, match="observed record"):
        X.risk_forecast("tamil_nadu__thanjavur", "2031-05-01")


@serving
def test_risk_forecast_uses_only_the_last_complete_month_for_drought() -> None:
    mid = X.risk_forecast("karnataka__raichur", "2021-08-17")
    end_of_prev = X.risk_forecast("karnataka__raichur", "2021-07-31")
    assert mid["drought_month"] == "2021-07" and end_of_prev["drought_month"] == "2021-07"
    assert mid["drought_p"] == pytest.approx(end_of_prev["drought_p"])
    assert X.risk_forecast("karnataka__raichur", "2021-08-31")["drought_month"] == "2021-08"


@serving
def test_extreme_models_are_deterministic_across_loads() -> None:
    a = X.risk_forecast("telangana__khammam", "2019-05-28")
    X.reset_cache()
    b = X.risk_forecast("telangana__khammam", "2019-05-28")
    assert a == b


@serving
def test_served_heatwave_probability_matches_the_declared_source() -> None:
    import torch

    state = torch.load(get_path("models") / X.HEAT_ARTIFACT, weights_only=False)
    result = X.risk_forecast("andhra_pradesh__krishna", "2018-05-25")
    assert result["heatwave_source"] == state["served"]
    if state["served"] == "climatology":
        districts = X._serving()["heat"].districts
        table = state["climatology_table"]
        slot = int(L._slot(pd.Series([pd.Timestamp("2018-05-25")]))[0])
        assert result["heatwave_p"] == pytest.approx(
            table[districts.index("andhra_pradesh__krishna"), slot, 2]
        )


@serving
def test_risk_forecast_reports_whether_the_drought_signal_is_agronomically_relevant() -> None:
    wet = X.risk_forecast("andhra_pradesh__east_godavari", "2023-05-20")  # June is a monsoon month
    assert wet["drought_relevant"] is True and wet["drought_normal_month_rain_mm"] >= 40
    dry = X.risk_forecast("andhra_pradesh__krishna", "2005-01-10")  # February is almost rainless
    assert dry["drought_relevant"] is False and dry["drought_normal_month_rain_mm"] < 40
    assert (
        dry["level"] == X.combine_levels(dry["heatwave_p"], 0.0)[0]
    )  # an irrelevant drought never sets the level
    assert 0.0 <= dry["drought_p"] <= 1.0  # still reported


@serving
def test_forecast_date_range_is_the_set_of_dates_risk_forecast_accepts() -> None:
    first, last = X.forecast_date_range()
    assert first < last and last.year >= 2024
    X.risk_forecast("tamil_nadu__thanjavur", str(first.date()))
    X.risk_forecast("tamil_nadu__thanjavur", str(last.date()))
    with pytest.raises(ValueError):
        X.risk_forecast("tamil_nadu__thanjavur", str((last + pd.Timedelta(days=1)).date()))
    with pytest.raises(ValueError):
        X.risk_forecast("tamil_nadu__thanjavur", str((first - pd.Timedelta(days=1)).date()))
