"""Phase 6 tests: water balance conservation, FAO equations, LP feasibility and dominance, crop switch ordering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.prescriptive import crop_switch
from analytics.prescriptive import irrigation_lp as lp
from analytics.prescriptive import params as P
from analytics.prescriptive import scenario as S
from analytics.prescriptive import water_balance as wb
from config import get_path

GLOB = P.load_globals()
T = GLOB.season_days


# ---------------------------------------------------------------- parameters
def test_every_crop_has_consistent_stage_lengths_and_positive_parameters() -> None:
    for crop in ("rice", "wheat", "cotton", "sugarcane"):
        c = P.load_crop(crop)
        assert sum(c.stage_days) == T and len(c.ky) == 4 and 0 < c.p < 1 and c.root_depth_m > 0
        assert P.taw_mm(c) == pytest.approx(
            GLOB.soil_classes[GLOB.default_soil_class] * c.root_depth_m
        )
        assert (
            max(c.ky) == c.ky[2] or crop == "sugarcane"
        )  # flowering / mid stage is the most sensitive
    with pytest.raises(KeyError):
        P.load_crop("banana")
    with pytest.raises(KeyError):
        P.taw_mm(P.load_crop("rice"), "granite")


# ---------------------------------------------------------------- radiation and ET0
def test_extraterrestrial_radiation_matches_the_fao56_worked_example() -> None:
    ra_mj = wb.extraterrestrial_radiation(-20.0, np.array([246.0]))[0] / wb.LAMBDA_INV
    assert ra_mj == pytest.approx(
        32.2, abs=0.2
    )  # FAO-56 Example 8: 20 S, 3 September -> 32.2 MJ m-2 d-1
    equator = wb.extraterrestrial_radiation(0.0, np.arange(1, 366)) / wb.LAMBDA_INV
    assert (
        equator.min() > 33 and equator.max() < 38.5
    )  # nearly constant through the year at the equator


def test_hargreaves_et0_is_sensible_and_monotone() -> None:
    ra = wb.extraterrestrial_radiation(11.0, np.array([180.0, 180.0, 180.0]))
    et0 = wb.hargreaves_et0(np.array([33.0, 36.0, 33.0]), np.array([22.0, 22.0, 22.0]), ra)
    assert 4 < et0[0] < 7 and et0[1] > et0[0] and et0[2] == pytest.approx(et0[0])
    assert (
        wb.hargreaves_et0(np.array([20.0]), np.array([25.0]), ra[:1])[0] == 0.0
    )  # Tmax < Tmin floors the range at 0


def test_kc_curve_shape_and_stage_index() -> None:
    crop = P.load_crop("wheat")
    kc = wb.kc_curve(crop)
    ini, dev, mid, late = crop.stage_days
    assert (
        len(kc) == T
        and (kc[:ini] == crop.kc_ini).all()
        and (kc[ini + dev : ini + dev + mid] == crop.kc_mid).all()
    )
    assert kc[ini + dev - 2] < crop.kc_mid and kc[ini + dev - 1] == pytest.approx(
        crop.kc_mid
    )  # development ends at Kc_mid
    assert kc[-1] == pytest.approx(crop.kc_end)
    assert (np.diff(kc[ini : ini + dev]) > 0).all() and (np.diff(kc[-late:]) < 0).all()
    stage = wb.stage_index(crop)
    assert stage[0] == 0 and stage[-1] == 3 and np.bincount(stage).tolist() == list(crop.stage_days)


# ---------------------------------------------------------------- water balance
def _random_scenarios(n: int = 25, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    et0 = rng.uniform(3.0, 7.0, (n, T))
    rain = rng.gamma(0.3, 12.0, (n, T)) * rng.uniform(0, 2, (n, 1))
    irrigation = rng.choice([0.0, 0.0, 0.0, 40.0], size=(n, T))
    return et0, rain, irrigation


@pytest.mark.parametrize("crop_name", ["rice", "wheat", "cotton", "sugarcane"])
def test_water_balance_conserves_mass_and_respects_bounds(crop_name: str) -> None:
    crop = P.load_crop(crop_name)
    taw = P.taw_mm(crop)
    et0, rain, irrigation = _random_scenarios()
    sim = wb.simulate(et0, rain, irrigation, wb.kc_curve(crop), taw, crop.p)
    assert np.abs(sim.mass_balance_residual()).max() < 1e-9
    assert (sim.depletion >= -1e-12).all() and (sim.depletion <= taw + 1e-12).all()
    assert (sim.eta <= sim.etc + 1e-12).all() and (sim.eta >= 0).all() and (sim.drainage >= 0).all()
    assert ((sim.ks >= 0) & (sim.ks <= 1)).all()


def test_no_stress_when_water_is_plentiful_and_severe_stress_when_it_is_absent() -> None:
    crop = P.load_crop("cotton")
    taw = P.taw_mm(crop)
    et0 = np.full((2, T), 5.0)
    rain = np.zeros((2, T))
    irrigation = np.zeros((2, T))
    irrigation[0, ::3] = 30.0  # irrigate far more than the crop needs
    sim = wb.simulate(et0, rain, irrigation, wb.kc_curve(crop), taw, crop.p, initial_depletion=0.0)
    np.testing.assert_allclose(sim.eta[0], sim.etc[0])  # Ks = 1 throughout
    assert sim.drainage[0].sum() > 0  # the excess drains instead of piling up
    assert sim.eta[1].sum() < 0.5 * sim.etc[1].sum()  # a dry soil with no inputs starves
    ryf = wb.relative_yield(sim.eta, sim.etc, crop)
    assert ryf[0] == pytest.approx(1.0) and ryf[1] < 0.5


def test_effective_rainfall_and_weekly_helpers() -> None:
    np.testing.assert_allclose(wb.effective_rainfall(np.array([0.0, 10.0]), 0.85), [0.0, 8.5])
    daily = np.arange(T, dtype=float)
    weekly = wb.to_weekly(daily)
    assert weekly.shape == (T // 7,) and weekly.sum() == pytest.approx(daily.sum())
    irrigation = wb.weekly_to_daily_irrigation(np.array([10.0, 0.0, 20.0]), 21)
    assert irrigation[0] == 10 and irrigation[14] == 20 and irrigation.sum() == 30
    np.testing.assert_allclose(wb.to_weekly(irrigation), [10.0, 0.0, 20.0])


def test_relative_yield_is_monotone_in_eta_and_bounded() -> None:
    crop = P.load_crop("rice")
    etc = np.tile(wb.kc_curve(crop) * 5.0, (3, 1))
    eta = etc * np.array([[1.0], [0.8], [0.5]])
    ryf = wb.relative_yield(eta, etc, crop)
    assert ryf[0] == pytest.approx(1.0) and ryf[0] > ryf[1] > ryf[2] >= 0.0


# ---------------------------------------------------------------- LP
def _lp_inputs(rain_scale: float = 0.6, seed: int = 1, crop_name: str = "rice") -> lp.LPInputs:
    crop = P.load_crop(crop_name)
    rng = np.random.default_rng(seed)
    etc = wb.kc_curve(crop) * rng.uniform(4.0, 6.0, T)
    rain = rng.gamma(0.3, 12.0, T) * rain_scale
    taw = P.taw_mm(crop)
    return lp.build_inputs(
        crop,
        etc,
        wb.effective_rainfall(rain, GLOB.rain_efficiency),
        taw,
        GLOB.initial_depletion_fraction * taw,
    )


@pytest.mark.parametrize("budget", [0.0, 100.0, 350.0, 5000.0])
def test_lp_respects_budget_and_weekly_cap_and_solves_fast(budget: float) -> None:
    inputs = _lp_inputs()
    result = lp.solve_irrigation(inputs, budget, GLOB.weekly_cap_mm)
    assert result.status == "Optimal" and result.solve_seconds < 5.0
    assert result.schedule_mm.sum() <= budget + 1e-6
    assert (result.schedule_mm <= GLOB.weekly_cap_mm + 1e-6).all() and (
        result.schedule_mm >= -1e-9
    ).all()
    assert len(result.schedule_mm) == T // 7
    assert (result.storage >= -1e-6).all() and (result.storage <= inputs.taw + 1e-6).all()


def test_zero_budget_gives_zero_irrigation_and_more_budget_never_hurts() -> None:
    inputs = _lp_inputs()
    objectives = []
    for budget in (0.0, 50.0, 150.0, 350.0, 1000.0):
        result = lp.solve_irrigation(inputs, budget, GLOB.weekly_cap_mm)
        objectives.append(result.objective)
        if budget == 0.0:
            assert result.schedule_mm.sum() == 0.0
    assert all(b >= a - 1e-9 for a, b in zip(objectives, objectives[1:], strict=False))
    assert objectives[-1] > objectives[0]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("crop_name", ["rice", "cotton", "wheat"])
def test_lp_is_never_worse_than_even_split_under_the_same_model(seed: int, crop_name: str) -> None:
    inputs = _lp_inputs(rain_scale=0.4 + 0.2 * seed, seed=seed, crop_name=crop_name)
    budget = P.load_crop(crop_name).default_budget_mm
    result = lp.solve_irrigation(inputs, budget, GLOB.weekly_cap_mm)
    even = lp.penalized_value(inputs, lp.even_split(budget, GLOB.weekly_cap_mm, inputs.n_weeks))
    rainfed = lp.penalized_value(inputs, np.zeros(inputs.n_weeks))
    assert (
        result.penalized_objective >= even - 1e-7
    )  # the LP optimum dominates every feasible schedule
    assert result.penalized_objective >= rainfed - 1e-7
    assert result.objective >= rainfed - 1e-7 or result.schedule_mm.sum() == 0


def test_lp_puts_scarce_water_into_the_most_sensitive_stage() -> None:
    n = T // 7
    stage_w = np.repeat(np.arange(4), n // 4)
    inputs = lp.LPInputs(
        etc_w=np.full(n, 30.0),
        peff_w=np.zeros(n),
        stage_w=stage_w,
        ky=(0.1, 0.1, 2.0, 0.1),
        taw=60.0,
        p=0.5,
        s0=60.0,
    )
    result = lp.solve_irrigation(inputs, 120.0, 80.0)
    in_sensitive = result.schedule_mm[stage_w == 2].sum()
    assert (
        in_sensitive >= 0.9 * result.schedule_mm.sum() and result.schedule_mm.sum() > 100
    )  # stage-2 weeks get (nearly) all of it


def test_lp_solution_is_deterministic_and_bad_inputs_are_rejected() -> None:
    inputs = _lp_inputs()
    a = lp.solve_irrigation(inputs, 300.0, 80.0).schedule_mm
    b = lp.solve_irrigation(inputs, 300.0, 80.0).schedule_mm
    np.testing.assert_allclose(a, b)
    with pytest.raises(ValueError):
        lp.solve_irrigation(inputs, -1.0, 80.0)
    with pytest.raises(ValueError):
        lp.solve_irrigation(inputs, 100.0, 80.0, stress_basis="nonsense")
    assert (
        lp.fixed_interval_daily(112).sum() == 600.0 and lp.even_split(350.0, 20.0, 16).max() == 20.0
    )


def test_max_weekly_eta_respects_each_limit() -> None:
    assert lp.max_weekly_eta(40.0, 5.0, 100.0, 60.0, "start") == pytest.approx(
        40.0 / 60.0 * 5.0
    )  # Ks-limited
    assert lp.max_weekly_eta(40.0, 1.0, 100.0, 1e-3, "start") == pytest.approx(
        1.0
    )  # cannot use water that is not there
    assert lp.max_weekly_eta(40.0, 500.0, 100.0, 60.0, "end") == pytest.approx(
        40.0
    )  # plentiful water: full demand
    assert lp.max_weekly_eta(40.0, 30.0, 100.0, 60.0, "end") < lp.max_weekly_eta(
        40.0, 30.0, 100.0, 60.0, "start"
    )


# ---------------------------------------------------------------- crop switch
ECON = {
    "crops": {
        "a": {"price_per_quintal": 2000, "marketed_unit_per_yield_unit": 1.0, "cost_per_ha": 20000},
        "b": {"price_per_quintal": 6000, "marketed_unit_per_yield_unit": 2.0, "cost_per_ha": 40000},
        "c": {"price_per_quintal": 1000, "marketed_unit_per_yield_unit": 1.0, "cost_per_ha": 5000},
    }
}


def test_revenue_uses_unit_conversion_and_net_subtracts_cost() -> None:
    assert crop_switch.revenue_per_ha(3.0, "a", ECON) == pytest.approx(3.0 * 10 * 2000)
    assert crop_switch.revenue_per_ha(0.5, "b", ECON) == pytest.approx(0.5 * 2.0 * 10 * 6000)
    assert crop_switch.net_return_per_ha(3.0, "a", ECON) == pytest.approx(60000 - 20000)


def test_crop_ranking_orders_by_net_return_and_lower_bound_can_reorder() -> None:
    predictions = {
        "a": {"yield_t_ha": 3.0, "lo": 2.8, "hi": 3.2},  # safe: net mid 40 000
        "b": {
            "yield_t_ha": 0.5,
            "lo": 0.1,
            "hi": 0.9,
        },  # risky: net mid 0.5*2*10*6000 - 40 000 = 20 000, lower bound negative
        "c": {"yield_t_ha": 1.0, "lo": 0.9, "hi": 1.1},  # poor: net mid 5 000
    }
    ranking = crop_switch.rank_crops(predictions, ECON)
    assert ranking["crop"].tolist() == ["a", "b", "c"]
    assert ranking.loc[ranking["crop"] == "a", "rank_mid"].item() == 1
    b = ranking[ranking["crop"] == "b"].iloc[0]
    assert b["net_mid_inr_ha"] > ranking[ranking["crop"] == "c"].iloc[0]["net_mid_inr_ha"]
    assert (
        b["net_lower_inr_ha"] < 0 and b["rank_lower"] == 3
    )  # pessimistic scenario demotes the risky crop
    assert "not financial advice" in ranking.attrs["disclaimer"]
    boom = crop_switch.rank_crops(
        {**predictions, "b": {"yield_t_ha": 1.5, "lo": 1.3, "hi": 1.7}}, ECON
    )
    assert boom["crop"].iloc[0] == "b"  # a high-yield scenario flips the order


def test_candidates_follow_the_season_and_exclude_sugarcane_by_default() -> None:
    assert set(crop_switch.candidate_crops("kharif")) == {"rice", "cotton"}
    assert crop_switch.candidate_crops("rabi") == ["wheat"]
    assert "sugarcane" in crop_switch.candidate_crops("kharif", include_sugarcane=True)


def test_real_economics_file_is_complete() -> None:
    econ = P.load_economics()
    for crop in ("rice", "wheat", "cotton", "sugarcane"):
        entry = econ["crops"][crop]
        assert (
            entry["price_per_quintal"] > 0
            and entry["cost_per_ha"] > 0
            and entry["marketed_unit_per_yield_unit"] >= 1.0
        )
    assert "unverified" in econ["price_year"]  # provenance warning must stay in the file


# ---------------------------------------------------------------- integration with the real data
def _data_ready() -> bool:
    processed = get_path("data_processed")
    return all((processed / f"{n}.parquet").exists() for n in ("weather_daily", "districts"))


needs_data = pytest.mark.skipif(not _data_ready(), reason="run `make etl` first")


@needs_data
def test_strategy_comparison_on_a_real_district_is_consistent() -> None:
    inputs = S.load_season_inputs("karnataka__raichur", "cotton", "kharif")
    budget = P.load_crop("cotton").default_budget_mm
    table, details = S.evaluate_strategies(inputs, budget)
    row = table.set_index("strategy")
    assert row.loc["rainfed", "water_used_mm"] == 0
    assert row.loc["farmer_baseline", "water_used_mm"] == pytest.approx(600.0)
    assert (
        row.loc["optimized", "water_used_mm"] <= budget + 1e-6
        and row.loc["even_split", "water_used_mm"] <= budget + 1e-6
    )
    assert ((table["rel_yield"] >= 0) & (table["rel_yield"] <= 1)).all()
    assert row.loc["optimized", "rel_yield"] >= row.loc["rainfed", "rel_yield"] - 1e-9
    assert np.abs(details["simulation"].mass_balance_residual()).max() < 1e-9
    assert details["lp"].solve_seconds < 5.0


@needs_data
def test_baseline_at_normal_rain_reproduces_the_ml_yield_and_droughts_lower_it() -> None:
    ml = {"yield_t_ha": 3.0, "lo": 2.0, "hi": 4.0}
    normal = S.load_season_inputs("tamil_nadu__madurai", "rice", "kharif")
    dry = S.load_season_inputs(
        "tamil_nadu__madurai", "rice", "kharif", year=normal.year, rain_multiplier=0.5
    )
    budget = 350.0
    base = S.strategy_yields(normal, budget, ml).set_index("strategy")
    assert base.loc["farmer_baseline", "yield_t_ha"] == pytest.approx(3.0)
    assert base.loc["farmer_baseline", "yield_lo"] == pytest.approx(2.0)
    assert (base["yield_t_ha"] <= 3.0 * S.MAX_GAIN + 1e-9).all()
    drought = S.strategy_yields(dry, budget, ml).set_index("strategy")
    assert drought.loc["rainfed", "yield_t_ha"] <= base.loc["rainfed", "yield_t_ha"] + 1e-9
    assert drought.loc["optimized", "yield_t_ha"] >= drought.loc["rainfed", "yield_t_ha"] - 1e-9


@needs_data
def test_scenario_inputs_apply_rain_and_temperature_changes_and_use_a_training_year() -> None:
    base = S.load_season_inputs("telangana__warangal", "rice", "kharif")
    changed = S.load_season_inputs(
        "telangana__warangal",
        "rice",
        "kharif",
        year=base.year,
        rain_multiplier=0.5,
        temperature_delta=2.0,
    )
    np.testing.assert_allclose(changed.rain, 0.5 * base.rain)
    np.testing.assert_allclose(changed.tmax, base.tmax + 2.0)
    assert changed.et0.sum() > base.et0.sum()  # warmer -> more evaporative demand
    assert base.year <= 2011 and base.n_days == T
    with pytest.raises(ValueError):
        S.load_season_inputs("nowhere__x", "rice", "kharif")


@needs_data
def test_cli_runs_without_the_ml_models_and_reports_errors_cleanly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from analytics.prescriptive import run as cli

    assert (
        cli.main(
            [
                "--district",
                "andhra_pradesh__guntur",
                "--crop",
                "rice",
                "--season",
                "kharif",
                "--budget",
                "300",
                "--no-ml",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "optimized" in out and "farmer_baseline" in out
    assert (
        cli.main(["--district", "nowhere__x", "--crop", "rice", "--season", "kharif", "--no-ml"])
        == 2
    )


def test_report_helpers_bootstrap_and_summaries() -> None:
    from analytics.prescriptive import report

    mean, lo, hi = report.bootstrap_ci(np.arange(100, dtype=float))
    assert lo < mean < hi and mean == pytest.approx(49.5)
    rows = []
    for case in range(6):
        for strategy, water, ryf in (
            ("farmer_baseline", 600.0, 0.9),
            ("even_split", 300.0, 0.7 + 0.01 * case),
            ("optimized", 200.0, 0.75 + 0.01 * case),
            ("rainfed", 0.0, 0.3),
        ):
            rows.append(
                {
                    "district_id": f"d{case}",
                    "crop": "rice",
                    "season": "kharif",
                    "scenario": "normal rainfall",
                    "strategy": strategy,
                    "water_used_mm": water,
                    "rel_yield": ryf,
                    "eta_over_etc": 0.9,
                    "drainage_mm": 10.0,
                }
            )
    results = pd.DataFrame(rows)
    summary = report.summarise(results).set_index("strategy")
    assert summary.loc["optimized", "water_saved_pct"] == pytest.approx(100 * (600 - 200) / 600)
    paired = report.paired_optimized_vs_even(results)
    assert (
        paired["delta_rel_yield"].iloc[0] == pytest.approx(0.05)
        and paired["cases_optimized_worse"].iloc[0] == 0
    )
