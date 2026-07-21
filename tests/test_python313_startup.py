from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import app
from services.common import ServiceError
from services import nikkei_artifact

ROOT = Path(__file__).resolve().parents[1]


def test_windows_startup_targets_python_313_series() -> None:
    runner = (ROOT / "RUN_APP_INNER.cmd").read_text(encoding="utf-8")
    start_here = (ROOT / "START_HERE.cmd").read_text(encoding="utf-8")
    assert "3.12.13" not in runner
    assert "py -3.12" not in runner
    assert "Python 3.13.x" in runner
    assert ":try_path_python" in runner
    assert "where python" in runner
    assert r"%LocalAppData%\Programs\Python\Python313\python.exe" in runner
    assert "py -3.13" in runner
    assert 'sys.version_info[:2] == (3, 13)' in runner
    assert r'.venv\Scripts\python.exe' in runner
    assert '-m venv "%VENV_DIR%"' in runner
    assert '-m pip install -r "%~dp0requirements.txt"' in runner
    assert 'set "STOCK_APP_OPEN_BROWSER=1"' in runner
    assert "AppendAllText" in runner
    assert "pause\nexit /b 0" not in runner.replace("\r\n", "\n")
    assert '/k call "%~dp0RUN_APP_INNER.cmd"' in start_here
    assert "PROJECT_ROOT=%~dp0" in runner
    assert "Invoke-RestMethod" in runner
    assert ":port_conflict" in runner
    assert "No process was stopped automatically" in runner


def test_health_identifies_the_current_extracted_project() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert Path(response.json()["project_root"]).resolve() == ROOT.resolve()


def test_manifest_declares_tested_python_313_compatibility() -> None:
    manifest = json.loads(
        (ROOT / "model_settings" / "nikkei_model_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["python_version"] == "3.12.13"
    assert "3.13" in manifest["compatible_python_series"]


def test_runtime_validation_accepts_a_declared_compatible_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = nikkei_artifact.load_manifest()
    expected_packages = dict(manifest["library_versions"])
    monkeypatch.setattr(
        nikkei_artifact,
        "runtime_versions",
        lambda: {"python": "3.13.7", "packages": expected_packages},
    )
    nikkei_artifact._validate_runtime(manifest)


def test_runtime_validation_rejects_an_untested_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = nikkei_artifact.load_manifest()
    expected_packages = dict(manifest["library_versions"])
    monkeypatch.setattr(
        nikkei_artifact,
        "runtime_versions",
        lambda: {"python": "3.14.0", "packages": expected_packages},
    )
    with pytest.raises(ServiceError):
        nikkei_artifact._validate_runtime(manifest)


def test_saved_model_environment_versions_remain_pinned() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    model_sensitive = ("pandas", "numpy", "scikit-learn", "catboost", "joblib")
    for package in model_sensitive:
        assert any(line.startswith(f"{package}==") for line in requirements)
