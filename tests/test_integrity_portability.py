from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from services.common import ServiceError
from services.nikkei_artifact import (
    ARTIFACT_SCHEMA_VERSION,
    EVALUATION_PATHS,
    EVALUATION_SCHEMA_VERSION,
    MANIFEST_PATH,
    load_context_evaluation,
    load_manifest,
    load_portable_artifact,
    runtime_versions,
    sha256_file,
)
from services.nikkei_dual_market import DualMarketData, FEATURE_DEFINITION_VERSION, MODEL_ARTIFACT_PATH, MODEL_VERSION
from services import nikkei_artifact, nikkei_dual_model

ROOT = Path(__file__).resolve().parents[1]


def _context_data() -> DualMarketData:
    index = pd.bdate_range("2020-01-01", periods=500)
    alternating = np.arange(len(index)) % 2
    base = pd.DataFrame(
        {
            "target_direction": alternating,
            "target_change": np.where(alternating == 1, 1.0, -1.0),
            "return_5d": np.where(alternating == 1, 0.01, -0.01),
            "japan_feature": 1.0,
            "overseas_feature": 1.0,
        },
        index=index,
    )
    after_close = base.copy()
    after_close["japan_feature"] = 2.0
    return DualMarketData(
        base_frame=base,
        japanese_frame=base,
        intraday_frame=base,
        after_close_frame=after_close,
        inference_rows={"intraday": base.iloc[[-1]], "after_close": after_close.iloc[[-1]]},
        japan_features=["japan_feature"],
        overseas_features=["overseas_feature"],
        warnings=[],
        fetched_at="2026-07-21T00:00:00+09:00",
        metadata={
            "evaluation_period": {"start": index[-360].date().isoformat(), "end": index[-1].date().isoformat()},
        },
    )


def test_formal_evaluation_uses_the_requested_context_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _context_data()
    seen: list[float] = []

    def selection(train: pd.DataFrame, *_args: object) -> dict:
        seen.append(float(train["japan_feature"].mean()))
        side = {"selected_model": "logistic", "selected_weight": "uniform"}
        return {"japan": side, "overseas": side, "threshold": {"selected_threshold": 0.5}}

    monkeypatch.setattr(nikkei_dual_model, "_outer_splits", lambda index: [index[-20:]])
    monkeypatch.setattr(nikkei_dual_model, "select_operational_configuration", selection)
    monkeypatch.setattr(nikkei_dual_model, "fit_candidate_package", lambda *_args: {})
    monkeypatch.setattr(
        nikkei_dual_model,
        "predict_candidate_package",
        lambda _package, validation: validation["target_direction"].to_numpy(dtype=float),
    )
    monkeypatch.setattr(
        nikkei_dual_model,
        "_six_class_fold",
        lambda _train, validation, _features: {
            "y_true": validation["target_direction"].to_numpy(dtype=int),
            "prediction": validation["target_direction"].to_numpy(dtype=int),
            "accuracy": 1.0,
            "macro_f1": 1.0,
        },
    )
    monkeypatch.setattr(nikkei_dual_model, "_selection_record", lambda _selection: {})
    monkeypatch.setattr(
        nikkei_dual_model,
        "moving_block_bootstrap_intervals",
        lambda *_args: {
            "direction_accuracy_95ci": [1.0, 1.0],
            "best_baseline_gap_95ci": [0.0, 1.0],
            "method": "fixed-test",
            "block_length": 1,
            "resamples": 1,
            "seed": 1,
        },
    )

    intraday = nikkei_dual_model.evaluate_formal_two_years(data, "intraday")
    after_close = nikkei_dual_model.evaluate_formal_two_years(data, "after_close")
    assert seen == [1.0, 2.0]
    assert intraday["prediction_context"] == "intraday"
    assert after_close["prediction_context"] == "after_close"
    assert intraday["purge_trading_days"] == after_close["purge_trading_days"] == 5
    assert intraday["evaluation_schema_version"] == EVALUATION_SCHEMA_VERSION


def test_context_evaluations_are_separate_current_files() -> None:
    intraday = load_context_evaluation("intraday")
    after_close = load_context_evaluation("after_close")
    assert intraday is not None and after_close is not None
    assert intraday["prediction_context"] == "intraday"
    assert after_close["prediction_context"] == "after_close"
    assert EVALUATION_PATHS["intraday"] != EVALUATION_PATHS["after_close"]
    assert intraday["outer_fold_count"] == after_close["outer_fold_count"] == 8
    assert intraday["inner_fold_count"] == after_close["inner_fold_count"] == 3


def test_manifest_hash_and_real_bundled_models_load() -> None:
    manifest = load_manifest()
    assert manifest["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert manifest["model_sha256"] == sha256_file(MODEL_ARTIFACT_PATH)
    assert manifest["prediction_contexts"] == ["after_close", "intraday"]
    packages, loaded_manifest = load_portable_artifact()
    assert loaded_manifest == manifest
    assert set(packages["contexts"]) == {"intraday", "after_close"}
    for context in ("intraday", "after_close"):
        package = packages["contexts"][context]
        direction_features = sorted(set(package["japan"]["selected_features"] + package["overseas"]["selected_features"]))
        row = pd.DataFrame([{feature: np.nan for feature in direction_features}])
        japan = nikkei_dual_model.predict_candidate_package(package["japan"], row)
        overseas = nikkei_dual_model.predict_candidate_package(package["overseas"], row)
        six_features = package["six_class"]["features"]
        six_row = pd.DataFrame([{feature: np.nan for feature in six_features}])
        six = nikkei_dual_model._predict_six(package["six_class"], six_row)
        assert np.isfinite(japan).all() and np.isfinite(overseas).all()
        assert sum(six["probabilities"]) == pytest.approx(1.0)


def test_artifact_metadata_contains_no_pandas_objects() -> None:
    packages = joblib.load(MODEL_ARTIFACT_PATH)

    def walk(value: object, key: str = "") -> None:
        if key in {"model", "scaler"}:
            return
        assert not isinstance(value, (pd.Series, pd.Index, pd.DataFrame, pd.Timestamp, pd.api.extensions.ExtensionDtype))
        if isinstance(value, np.ndarray):
            assert value.dtype != object
        elif isinstance(value, dict):
            for child_key, child in value.items():
                walk(child, str(child_key))
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    walk(packages)
    for context in packages["contexts"].values():
        for side in ("japan", "overseas"):
            for component in context[side]["components"]:
                assert isinstance(component["features"], list)
                assert all(isinstance(item, str) for item in component["features"])
                assert isinstance(component["medians"], dict)
        assert isinstance(context["six_class"]["medians"], dict)


def test_corrupt_or_schema_mismatched_manifest_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact = tmp_path / "model.joblib"
    joblib.dump({"model_version": MODEL_VERSION, "contexts": {}}, artifact)
    manifest_path = tmp_path / "manifest.json"
    payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_schema_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_DEFINITION_VERSION,
        "prediction_contexts": ["after_close", "intraday"],
        "python_version": runtime_versions()["python"],
        "library_versions": runtime_versions()["packages"],
        "model_sha256": "0" * 64,
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(nikkei_artifact, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(nikkei_artifact, "MODEL_ARTIFACT_PATH", artifact)
    with pytest.raises(ServiceError, match="互換性"):
        nikkei_artifact.load_portable_artifact()
    payload["artifact_schema_version"] = "old-schema"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ServiceError, match="互換性"):
        nikkei_artifact.load_manifest()


def test_public_renderer_separates_direction_and_six_class_evaluations() -> None:
    renderer = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
    renderer = renderer.split("const renderNikkeiPublicPrediction", 1)[1].split("const renderSimulation", 1)[0]
    for required in (
        "prediction_context_label",
        "予測精度",
        "6段階区分の過去評価",
        "方向一致率の95％区間",
        "単純方法との差の95％区間",
        "現在使用した予測条件に対応する正式評価は、まだ作成されていません。",
        "方向一致率とは異なる指標",
    ):
        assert required in renderer
    assert "54.7" not in renderer and "16.8" not in renderer
    assert "将来の利益確率や売買の推奨度を示すものではありません" in (ROOT / "services" / "nikkei_service.py").read_text(encoding="utf-8")

def test_release_zip_builder_uses_only_portable_safe_paths(tmp_path: Path) -> None:
    import zipfile
    from tools.build_release_zip import build_release_zip, inspect_release_zip

    destination = tmp_path / "release.zip"
    build_release_zip(ROOT, destination)
    report = inspect_release_zip(destination, extract_smoke=True)
    assert report["top_level"] == "japanese-stock-strategy-app"
    assert report["backslash_member_count"] == 0
    assert report["testzip"] is None
    with zipfile.ZipFile(destination) as archive:
        names = set(archive.namelist())
    assert not any("individual_evaluations" in name for name in names)
    assert "japanese-stock-strategy-app/tools/check_individual_evaluation.py" not in names
    assert "japanese-stock-strategy-app/tools/update_individual_logistic_evaluation.py" not in names
    assert not any("individual_cache_" in name for name in names)

def test_missing_context_evaluation_is_not_replaced_by_another_context() -> None:
    from services.nikkei_service import _public_evaluation

    unavailable = _public_evaluation(None, "after_close")
    assert unavailable["available"] is False
    assert unavailable["prediction_context"] == "after_close"
    assert "まだ作成されていません" in unavailable["message"]
    source = (ROOT / "services" / "nikkei_dual_model.py").read_text(encoding="utf-8")
    build_body = source.split("def build_latest_result", 1)[1].split("def predict_with_saved_dual_market_model", 1)[0]
    assert "load_context_evaluation(context)" in build_body
    assert 'settings["formal_evaluation"]' not in build_body
