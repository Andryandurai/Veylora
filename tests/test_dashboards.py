"""Smoke and behaviour tests for the Streamlit dashboards (Streamlit's AppTest runs each page headlessly)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGES = {
    "home": "dashboards/app.py",
    "weather": "dashboards/pages/1_Weather_Forecast.py",
    "yield": "dashboards/pages/2_Yield_Predictor.py",
    "map": "dashboards/pages/3_Risk_Map.py",
    "advisory": "dashboards/pages/4_Advisory.py",
    "simulator": "dashboards/pages/5_3D_Simulator.py",
    "methodology": "dashboards/pages/6_Methodology.py",
}
NEEDED = [
    "data/processed/weather_daily.parquet",
    "data/processed/districts.parquet",
    "data/processed/crops_annual.parquet",
    "models/yield_xgb.joblib",
    "models/drought_xgb.joblib",
    "simulator/index_embedded.html",
]
pytestmark = pytest.mark.skipif(
    any(not (ROOT / f).exists() for f in NEEDED),
    reason="dashboards need the pipeline outputs (make etl train-yield train-extreme embed-sim)",
)


def run(page: str, timeout: int = 120) -> AppTest:
    at = AppTest.from_file(str(ROOT / PAGES[page]), default_timeout=timeout)
    at.run()
    return at


def text_of(at: AppTest) -> str:
    """All visible text of a run, for content assertions."""
    parts = (
        [m.value for m in at.markdown]
        + [c.value for c in at.caption]
        + [w.value for w in at.warning]
        + [i.value for i in at.info]
    )
    parts += [f"{m.label} {m.value}" for m in at.metric] + [c.value for c in at.code]
    parts += [e.value for e in at.error]
    return "\n".join(map(str, parts))


@pytest.mark.parametrize("page", list(PAGES))
def test_page_runs_without_exception(page):
    at = run(page)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_every_page_warns_about_synthetic_satellite_data():
    for page in ("home", "weather", "yield", "map", "advisory", "methodology"):
        assert "Synthetic data in the pipeline" in text_of(run(page)), page


def test_home_lists_scope_and_reads_measured_results():
    text = text_of(run("home"))
    assert "51" in text and "Yield error vs mean yield" in text and "Drought-warning F1" in text


def test_weather_page_reports_levels_and_probabilities():
    at = run("weather")
    labels = [m.label for m in at.metric]
    assert "Overall risk level" in labels and "Drought next month" in labels
    assert any("Heatwave onset" in label for label in labels)
    for metric in at.metric:
        if metric.label in {"Drought next month"}:
            assert 0.0 <= float(metric.value.rstrip("%")) <= 100.0


def test_weather_page_reports_a_bad_date_range_without_a_stack_trace(monkeypatch):
    from dashboards.components import models

    def boom(*args, **kwargs):
        raise ValueError("as_of_date must be within the observed record")

    monkeypatch.setattr(models, "risk", boom)
    at = run("weather")
    assert not at.exception
    assert any("cannot be computed" in w.value for w in at.warning)


def test_yield_page_forecast_responds_to_the_what_if_sliders():
    at = run("yield")
    base = next(m for m in at.metric if m.label == "Forecast yield").value
    at.sidebar.slider[0].set_value(1.6).run()  # rainfall multiplier
    assert not at.exception
    dry = next(m for m in at.metric if m.label == "Forecast yield").value
    assert dry != base
    assert float(dry.split()[0]) > 0


def test_yield_page_shows_interval_contributions_and_crop_ranking():
    at = run("yield")
    labels = [m.label for m in at.metric]
    assert "80% interval" in labels
    assert "What drives this forecast" in [s.value for s in at.subheader]
    assert any("crop pays best" in s.value for s in at.subheader)


def test_yield_page_handles_a_crop_without_records():
    at = run("yield")
    at.session_state["district"] = "tamil_nadu__thanjavur"
    at.sidebar.selectbox(key="crop").select("wheat").run()
    assert not at.exception
    assert any("no yield records" in i.value for i in at.info)


def test_risk_map_has_both_views_and_district_counts():
    at = run("map")
    assert len(at.tabs) == 2
    assert sum(1 for m in at.metric if m.label.startswith("Districts")) == 4


def test_advisory_languages_stay_within_the_sms_limit_and_warn_about_translation():
    at = run("advisory")
    assert not any("draft" in w.value for w in at.warning)  # English needs no translation warning
    for code in ("hi", "ta", "te"):
        at.sidebar.selectbox(key="advisory_lang").select(code).run()
        assert not at.exception, code
        chars = next(m for m in at.metric if m.label == "Characters").value
        assert int(chars.split("/")[0]) <= int(chars.split("/")[1]), code
        assert any("draft" in w.value for w in at.warning), code


def test_advisory_text_differs_by_language():
    at = run("advisory")
    english = at.code[0].value
    at.sidebar.selectbox(key="advisory_lang").select("hi").run()
    assert at.code[0].value != english


def test_methodology_reads_the_reports_and_states_the_limits():
    at = run("methodology")
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "Data",
        "Yield",
        "Early warning",
        "Irrigation",
        "Limitations",
    ]
    text = text_of(at)
    assert "SYNTHETIC" in "".join(df.value.to_csv() for df in at.dataframe)
    assert "do not clearly beat" in text and "Served heatwave probability" in text


def test_missing_artifacts_give_instructions_not_a_stack_trace(monkeypatch, tmp_path):
    from dashboards.components import guards

    monkeypatch.setattr(guards, "ROOT", tmp_path)
    for page in ("yield", "map", "advisory", "simulator"):
        at = run(page)
        assert not at.exception, page
        assert any("not been built yet" in e.value for e in at.error), page
        assert any("make" in m.value for m in at.markdown), page


def test_simulator_page_embeds_the_static_file_with_preselected_district(tmp_path):
    from dashboards.components import assets

    at = run("simulator")
    assert not at.exception
    assert assets.TARGET.exists()
    url = assets.simulator_url("tamil_nadu__thanjavur", "rice")
    assert url.startswith("/app/static/simulator.html?") and "district=tamil_nadu__thanjavur" in url


def test_pages_are_fast_once_caches_are_warm():
    for page in ("weather", "yield", "advisory"):
        run(page)  # warm-up (loads models, caches data)
        start = time.perf_counter()
        at = run(page)
        elapsed = time.perf_counter() - start
        assert not at.exception
        assert elapsed < 3.0, f"{page} took {elapsed:.1f}s after warm-up"


def test_home_compares_the_model_with_the_best_baseline_not_the_weakest():
    import pandas as pd

    m = pd.read_csv(ROOT / "reports" / "yield_metrics.csv")
    sel = m[
        (m["split"] == "test")
        & (m["crop"] == "all")
        & (m["lead_days"] == 60)
        & (m["feature_set"] == "weather")
    ].set_index("model")
    best = min(
        ("naive_last_year", "district_trend", "prior_mean_3y"), key=lambda b: sel.loc[b, "rmse_rel"]
    )
    text = text_of(run("home"))
    assert (
        f"best simple baseline ({best.replace('_', ' ')}): {sel.loc[best, 'rmse_rel']:.0%}" in text
    )


def test_simulator_choice_maps_unavailable_districts_to_a_representative():
    from dashboards.components import assets, data

    centroids, clusters = data.centroids_and_clusters()
    exact = assets.simulator_choice("telangana__nizamabad", "rice", centroids, clusters)
    assert exact.district_id == "telangana__nizamabad" and exact.note is None
    mapped = assets.simulator_choice("tamil_nadu__thanjavur", "rice", centroids, clusters)
    assert mapped.district_id in {s["id"] for s in assets.simulator_districts()}
    assert mapped.note and "representative districts" in mapped.note
    no_wheat = assets.simulator_choice("tamil_nadu__trichy", "wheat", centroids, clusters)
    assert no_wheat.crop == "rice" and "not simulated" in (no_wheat.note or "")


def test_vulnerability_table_is_a_ranked_unit_interval_score():
    from dashboards.components import data

    table = data.vulnerability_table()
    assert 40 <= len(table) <= 51
    assert table["vulnerability_score"].between(0, 1).all()
    assert table["vulnerability_score"].is_monotonic_decreasing
    assert table[["drought_frequency", "mean_heat_days", "rice_yield_cv"]].notna().all().all()
