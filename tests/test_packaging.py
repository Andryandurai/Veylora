"""Phase 10: packaging, container, demo bundle and secret-scanning checks (no network, no Docker needed)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from config import ROOT, bundle

sys.path.insert(0, str(ROOT / "scripts"))
import check_secrets  # noqa: E402


def pins(path: Path) -> dict[str, str]:
    """``{normalised package name: version}`` of the ``name==version`` lines of a requirements file."""
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==([^\s;#]+)", line.strip())
        if match:
            found[re.sub(r"[-_.]+", "-", match[1]).lower()] = match[2]
    return found


def test_serving_requirements_are_exact_and_agree_with_the_lock_file() -> None:
    serving, lock = pins(ROOT / "requirements.txt"), pins(ROOT / "requirements.lock")
    assert len(serving) >= 20
    for name, version in serving.items():
        assert lock.get(name) == version, (
            f"{name}: requirements.txt {version} vs lock {lock.get(name)}"
        )
    assert (
        "torch" in serving and "scikit-learn" in serving
    )  # pickled models need matching library versions


def test_every_declared_dependency_is_in_the_lock_file() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"dependencies = \[(.*?)\n\]", text, re.S)[1]
    declared = {
        re.sub(r"[-_.]+", "-", re.match(r"\s*\"([A-Za-z0-9_.\-]+)", line)[1]).lower()
        for line in block.splitlines()
        if line.strip().startswith('"')
    }
    lock = pins(ROOT / "requirements.lock")
    assert declared and not sorted(declared - set(lock)), sorted(declared - set(lock))


def test_lock_file_has_no_local_paths_or_editable_installs() -> None:
    text = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    assert "-e " not in text and "file://" not in text and "c:\\" not in text.lower()


def test_dockerfile_runs_as_a_non_root_user_with_a_health_check() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.startswith("# ClimateResilience") and "FROM python:3.11-slim" in dockerfile
    assert re.search(r"^USER app$", dockerfile, re.M) and "HEALTHCHECK" in dockerfile
    assert (
        "download.pytorch.org/whl/cpu" in dockerfile
    )  # CPU wheels, not gigabytes of CUDA libraries
    assert dockerfile.index("COPY requirements") < dockerfile.index(
        "COPY --chown=app:app . ."
    )  # layer caching


def test_docker_context_excludes_secrets_and_regenerable_data() -> None:
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    for entry in (".env", ".git", "data/raw", "data/processed", "models"):
        assert entry in ignored
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert {"db", "app"} <= set(compose["services"])
    assert compose["services"]["app"]["depends_on"]["db"]["condition"] == "service_healthy"


def test_pre_commit_runs_ruff_with_the_locked_version_and_the_secret_scan() -> None:
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    ruff = next(r for r in config["repos"] if "ruff-pre-commit" in r["repo"])
    assert ruff["rev"] == "v" + pins(ROOT / "requirements.lock")["ruff"]
    ids = {hook["id"] for repo in config["repos"] for hook in repo["hooks"]}
    assert {"ruff", "ruff-format", "fast-tests", "no-secrets", "check-added-large-files"} <= ids


def test_env_file_is_ignored_by_git_and_never_tracked() -> None:
    assert re.search(r"^\.env$", (ROOT / ".gitignore").read_text(encoding="utf-8"), re.M)
    if (ROOT / ".git").exists():
        tracked = subprocess.run(
            ["git", "ls-files", ".env"], cwd=ROOT, capture_output=True, text=True, check=False
        )
        assert tracked.stdout.strip() == ""


# --- secret scanner -----------------------------------------------------------------------------------------------


def test_secret_scanner_flags_credentials_without_echoing_them(tmp_path, capsys) -> None:
    sid = "AC" + "0123456789abcdef" * 2  # assembled here so that this file itself stays clean
    token = "TWILIO_AUTH_TOKEN=" + "fedcba9876543210" * 2
    key = "-----BEGIN " + "PRIVATE KEY-----"
    bad = tmp_path / "settings.py"
    bad.write_text(f'SID = "{sid}"\n{token}\nharmless = 1\n{key}\n', encoding="utf-8")
    assert [rule for _, rule in check_secrets.scan_text(bad.read_text(encoding="utf-8"))] == [
        "Twilio account SID",
        "Twilio auth token",
        "private key",
    ]
    assert check_secrets.main([str(bad)]) == 1
    out = capsys.readouterr().out
    assert "settings.py:1: Twilio account SID" in out and sid not in out and "fedcba98" not in out


def test_secret_scanner_ignores_placeholders_binaries_and_the_env_example(tmp_path) -> None:
    clean = tmp_path / "ok.md"
    clean.write_text("TWILIO_AUTH_TOKEN= (leave blank)\nACCOUNT_SID=ACxxxxxxxx\n", encoding="utf-8")
    binary = tmp_path / "blob.bin"
    binary.write_bytes(b"\0" + ("AC" + "0" * 32).encode())
    example = tmp_path / ".env.example"
    example.write_text("TWILIO_AUTH_TOKEN=" + "a" * 32, encoding="utf-8")
    assert (
        check_secrets.main([str(clean), str(binary), str(example), str(tmp_path / "missing.txt")])
        == 0
    )


def test_the_repository_itself_has_no_credentials() -> None:
    try:
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        pytest.skip("git is not installed (container)")
    if tracked.returncode != 0:
        pytest.skip("not a git checkout")
    paths = [
        ROOT / p
        for p in tracked.stdout.splitlines()
        if not p.endswith((".png", ".parquet", ".joblib", ".pt"))
    ]
    assert check_secrets.scan_files(paths) == []


# --- demo bundle --------------------------------------------------------------------------------------------------


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A throw-away repository root with fake pipeline outputs, so the real data is never touched."""
    monkeypatch.setattr(bundle, "ROOT", tmp_path)
    monkeypatch.setattr(bundle, "BUNDLE", tmp_path / "data" / "sample")
    monkeypatch.setattr(bundle, "MANIFEST", tmp_path / "data" / "sample" / "MANIFEST.json")
    for _, _, destination in bundle.entries():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(destination.name.encode() * 10)
    return tmp_path


def test_bundle_make_verify_restore_round_trip(sandbox) -> None:
    files = bundle.make()
    assert (
        len(files) == sum(len(names) for _, names in bundle.LAYOUT.values())
        and bundle.verify() == []
    )
    for _, _, destination in bundle.entries():
        destination.unlink()
    assert bundle.restore_if_missing() and all(dest.exists() for _, _, dest in bundle.entries())
    assert bundle.restore() == []  # nothing missing, nothing overwritten


def test_bundle_restore_never_overwrites_without_force_and_detects_tampering(sandbox) -> None:
    bundle.make()
    target = sandbox / "models" / "yield_xgb.joblib"
    target.write_bytes(b"my newer model")
    assert bundle.restore() == [] and target.read_bytes() == b"my newer model"
    written = bundle.restore(force=True)
    assert (
        any("yield_xgb.joblib" in w for w in written) and target.read_bytes() != b"my newer model"
    )
    (bundle.BUNDLE / "models" / "yield_xgb.joblib").write_bytes(b"corrupt")
    assert bundle.verify() == ["models/yield_xgb.joblib"]
    with pytest.raises(ValueError, match="incomplete or modified"):
        bundle.restore(force=True)
    assert (
        bundle.restore_if_missing() == []
    )  # a damaged bundle is ignored quietly, the app then explains what is missing


def test_bundle_make_requires_the_pipeline_outputs(sandbox) -> None:
    shutil.rmtree(sandbox / "models")
    with pytest.raises(FileNotFoundError, match="make pipeline"):
        bundle.make()


def test_committed_demo_bundle_is_intact_and_complete() -> None:
    if not bundle.MANIFEST.exists():
        pytest.skip("no demo bundle yet: run `python -m config.bundle make`")
    assert bundle.verify() == []
    manifest = json.loads(bundle.MANIFEST.read_text(encoding="utf-8"))
    assert "synthetic" in manifest["note"]
    assert (
        sum(f["bytes"] for f in manifest["files"].values()) < 60e6
    )  # small enough for a plain git repository
