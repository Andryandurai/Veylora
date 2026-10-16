"""Phase 10: the report generator reads numbers from result files, marks gaps and refuses contradicted prose."""

from __future__ import annotations

import copy
import sys

import pytest

from config import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import make_report as R  # noqa: E402

NEEDED = [
    "yield_models.json",
    "yield_metrics.csv",
    "extreme_weather.json",
    "prescriptive_results.json",
    "data_quality.json",
    "dashboard_checks.json",
    "bot_smoke.json",
    "sim_checks.json",
]
reports_exist = pytest.mark.skipif(
    any(not (ROOT / "reports" / n).exists() for n in NEEDED), reason="run the pipeline first"
)
MANUAL = {"real_gpu_fps", "deployed_url", "translation_review", "tests"}


def test_a_missing_figure_is_marked_and_recorded_never_invented() -> None:
    facts = R.Facts()
    assert facts.show("answer", lambda: 0.123456, "{:.2f}") == "0.12"
    assert facts.show("absent key", lambda: {}["x"]) == "MISSING (absent key)"
    assert facts.show("nan", lambda: float("nan")) == "MISSING (nan)"
    assert facts.show("none", lambda: None) == "MISSING (none)"
    assert facts.missing == ["absent key", "nan", "none"]


@reports_exist
def test_every_figure_comes_from_the_reports_except_the_declared_manual_ones() -> None:
    facts = R.Facts()
    values = R.collect(facts, count_tests=False)
    unexpected = {k: v for k, v in values.items() if "MISSING" in v and k not in MANUAL}
    assert not unexpected, unexpected
    for key in ("real_gpu_fps", "deployed_url", "translation_review"):
        assert values[key].startswith(
            "MISSING"
        )  # these cannot be known to the generator and must stay visible
    assert R.verify_claims(facts) == []


@reports_exist
def test_documents_contain_the_measured_numbers_and_the_cautions() -> None:
    facts = R.Facts()
    values = R.collect(facts, count_tests=False)
    report, demo, cv = R.final_report(values), R.demo_script(values), R.cv_bullets(values)
    for text in (report, demo, cv):
        assert (
            "{" not in text.replace("{{", "").replace("}}", "") or "```" in text
        )  # no unfilled placeholder
    assert values["d_f1"] in report and values["y_rel"] in report and values["w_saved"] in report
    assert "not better than simple baselines" in report and "no skill over climatology" in report
    assert "keeps less yield" in report and "synthetic" in report.lower()
    assert "Not claimable" in cv and "MISSING (not done)" in cv
    assert "Say the cautions out loud" in demo and values["h_ap_clim"] in demo
    assert "plan's figures were hypotheses" in report


@reports_exist
def test_claims_check_catches_prose_that_the_numbers_no_longer_support(monkeypatch) -> None:
    facts = R.Facts()
    originals = {
        n: copy.deepcopy(facts.json(n))
        for n in (
            "yield_models.json",
            "extreme_weather.json",
            "prescriptive_results.json",
            "data_quality.json",
        )
    }

    def with_changes(mutate):
        data = copy.deepcopy(originals)
        mutate(data)
        patched = R.Facts()
        patched._json.update(data)
        return R.verify_claims(patched)

    def beats_trend(d):
        for row in d["yield_models.json"]["bootstrap_vs_baseline"]:
            if row["model"] == "xgboost" and row["baseline"] == "district_trend":
                row["hi"] = 0.9

    def lstm_wins(d):
        d["extreme_weather.json"]["heatwave"]["served"]["source"] = "lstm"

    def optimizer_keeps_yield(d):
        for row in d["prescriptive_results.json"]["summary"]:
            row["yield_vs_baseline"] = 1.02

    def real_satellite(d):
        d["data_quality.json"]["synthetic_share"]["satellite_16d"] = 0.0

    assert with_changes(lambda d: None) == []
    assert any("district_trend" in c for c in with_changes(beats_trend))
    assert any("heatwave" in c for c in with_changes(lstm_wins))
    assert any("optimizer" in c for c in with_changes(optimizer_keeps_yield))
    # the satellite sentences are generated from the data-quality report, so real data is not a stale claim
    assert with_changes(real_satellite) == []
    empty = R.Facts()
    empty._json.update({name: None for name in originals})
    assert R.verify_claims(empty) == ["result files are missing or have an unexpected structure"]


def test_readme_results_block_is_replaced_in_place(tmp_path, monkeypatch) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{R.START}\nold\n{R.END}\noutro\n", encoding="utf-8")
    monkeypatch.setattr(R, "ROOT", tmp_path)
    assert R.write_readme_block(f"{R.START}\nnew table\n{R.END}")
    text = readme.read_text(encoding="utf-8")
    assert (
        "new table" in text
        and "old" not in text
        and text.startswith("intro")
        and text.rstrip().endswith("outro")
    )
    readme.write_text("no markers here", encoding="utf-8")
    assert not R.write_readme_block("x")


def test_the_repository_readme_has_the_markers_and_no_unfilled_missing_figures() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert R.START in text and R.END in text
    block = text.split(R.START)[1].split(R.END)[0]
    if "Generated by" in block:  # filled by make_report: only the declared manual gaps may remain
        assert (
            "MISSING (yield" not in block
            and "MISSING (drought" not in block
            and "MISSING (water" not in block
        )


@reports_exist
def test_satellite_sentences_follow_the_data_quality_report() -> None:
    def values(share: float) -> dict[str, str]:
        facts = R.Facts()
        facts.json("data_quality.json")["synthetic_share"]["satellite_16d"] = share
        return R.collect(facts, count_tests=False)

    synthetic, real = values(1.0), values(0.0)
    assert (
        "synthetic" in synthetic["sat_headline"]
        and synthetic["sat_next"] == "real satellite data, "
    )
    assert "yellow banner" in synthetic["sat_banner"]
    assert "real" in real["sat_headline"] and "synthetic" not in real["sat_headline"]
    assert real["sat_next"] == "" and real["sat_banner"] == ""
    assert "MISSING" not in real["sat_use"]
