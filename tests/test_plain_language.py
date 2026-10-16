"""Visitors never see developer text: no commands, file names, internal ids or model class names where plain words belong."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from dashboards.components import labels

ROOT = Path(__file__).resolve().parents[1]

RAW_FEATURES = [
    "rain_d30",
    "rainy_days_d60",
    "dry_spell_d90",
    "mean_tmax_d30",
    "heat_days_d60",
    "gdd_d90",
    "rain_d30_ratio",
    "mean_tmax_d60_anom",
    "ndvi_d60",
    "soil_moisture_d90",
    "prev_year_rel",
    "mean_prev3_rel",
    "trend_pred_rel",
    "log_scale",
    "crop_rice",
    "crop_sugarcane",
    "season_kharif",
    "season_rabi",
    "cluster_0",
    "cluster_6",
    "other features",
]
RAW_MODELS = [
    "xgboost",
    "xgboost_unconstrained",
    "random_forest",
    "ridge",
    "lstm",
    "naive_last_year",
    "district_trend",
    "prior_mean_3y",
    "climatology",
    "persistence",
    "rainfed",
    "farmer_baseline",
    "even_split",
    "optimized",
]


def test_feature_labels_are_plain_and_distinct():
    shown = [labels.feature_label(name) for name in RAW_FEATURES]
    assert len(set(shown)) == len(shown)
    for name, label in zip(RAW_FEATURES, shown, strict=True):
        assert "_" not in label and not re.search(r"\b[dD]\d{2}\b", label), (name, label)
        assert label[0].isupper(), (name, label)
    assert labels.feature_label("rain_d60") == "Rainfall, first 60 days"
    assert labels.feature_label("crop_rice") == "Crop: rice"
    assert (
        labels.feature_label("something_new_x") == "Something new x"
    )  # unknown ids still lose underscores


def test_model_labels_cover_every_internal_name():
    for name in RAW_MODELS:
        label = labels.model_label(name)
        assert label != name and "_" not in label, name
    assert labels.model_label("never_heard_of_it") == "Never heard of it"


def test_explanation_chart_uses_plain_labels():
    from dashboards.components import charts

    contributions = [
        {"feature": "prev_year_rel", "value": 1.0, "contribution_t_ha": 0.3},
        {"feature": "rain_d60", "value": 100.0, "contribution_t_ha": -0.1},
        {"feature": "other features", "value": None, "contribution_t_ha": 0.02},
    ]
    spec = json.dumps(charts.contributions_chart(contributions).to_dict())
    for raw in ("prev_year_rel", "rain_d60", "other features"):
        assert raw not in spec
    assert "Last year's yield" in spec and "Rainfall, first 60 days" in spec


# ---------------------------------------------------------------- whole pages (Streamlit's AppTest)
pytest.importorskip("streamlit")
from google.protobuf.json_format import MessageToDict  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

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
needs_pipeline = pytest.mark.skipif(
    any(not (ROOT / f).exists() for f in NEEDED), reason="dashboards need the pipeline outputs"
)

# things that must not appear anywhere a visitor can read them (styles and chart specs are not read)
ALWAYS = {
    "a command": r"\bmake [a-z][a-z-]+|python -m|pip install|\bgit\b|docker",
    "a file name": r"\b[\w./-]+\.(yaml|yml|json|csv|parquet|joblib|py|toml)\b",
    "a repository path": r"\b(reports|config|data|models|scripts|dashboards)/[\w./-]+",
    "an internal flag or table": r"is_synthetic|satellite_16d|weather_daily|crops_annual|district_id",
    "an internal id": r"\b[a-z]{2,}(?:_[a-z0-9]+)+\b",
    "a hash or timestamp": r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b|\d{4}-\d\d-\d\dT\d\d:\d\d",
    "debug text": r"Traceback|Exception|\bNone\b|\bnan\b",
}
# method names are allowed on the Methodology page only (it documents how the numbers were made)
OUTSIDE_METHODOLOGY = {
    "a model or library name": r"XGBoost|SHAP|LSTM|conformal|GSM-7|FAO-\d+|Hargreaves|PuLP|bootstrap|served model|held-out|lead_days",
}


# fields of Streamlit's protobuf messages that are internal (ids, enum names, layout), not text a visitor reads
INTERNAL_KEYS = {"id", "formId", "elementId", "key", "type", "color", "icon", "width", "execution"}


def _strings(value, key: str = "") -> list[str]:
    if key in INTERNAL_KEYS:
        return []
    if isinstance(value, str):
        internal = (
            value.startswith(("$$ID", "/mock/media"))
            or re.fullmatch(r"[A-Z][A-Z_]+", value)
            or re.fullmatch(r"[0-9a-f]{32}", value)
        )
        return [] if internal else [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in _strings(v, k)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v, key)]
    return []


def visible_text(at: AppTest) -> list[str]:
    """Every string a visitor can read on the page: text, labels, tooltips, table headers and cells."""
    found: list[str] = []
    for block in (at.main, at.sidebar):
        stack = [block]
        while stack:
            node = stack.pop()
            children = getattr(node, "children", None)
            if children:
                stack.extend(children.values())
            if getattr(node, "type", None) in {"dataframe"}:
                frame = node.value
                shown = json.loads(
                    node.proto.columns or "{}"
                )  # display names set with column_config
                found += [shown.get(str(c), {}).get("label", str(c)) for c in frame.columns]
                found += [str(v) for v in frame.to_numpy().ravel() if isinstance(v, str)]
            elif getattr(node, "proto", None) is not None and getattr(node, "type", "") not in {
                "arrow_vega_lite_chart",
                "vega_lite_chart",
                "iframe",
            }:
                found += _strings(MessageToDict(node.proto))
    return [s for s in found if s and "<style>" not in s and "st-key" not in s]


@needs_pipeline
@pytest.mark.parametrize("page", list(PAGES))
def test_no_developer_text_is_visible(page):
    at = AppTest.from_file(str(ROOT / PAGES[page]), default_timeout=120)
    at.run()
    assert not at.exception
    rules = dict(ALWAYS)
    if page != "methodology":
        rules |= OUTSIDE_METHODOLOGY
    problems = []
    for text in visible_text(at):
        for name, pattern in rules.items():
            for match in re.finditer(
                pattern, text, flags=re.IGNORECASE if name != "an internal id" else 0
            ):
                problems.append(f"{name}: {match.group(0)!r} in {text[:80]!r}")
    assert not problems, "\n".join(problems[:15])
