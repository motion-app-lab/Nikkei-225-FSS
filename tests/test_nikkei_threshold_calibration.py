from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as app_module
import services.nikkei_threshold_calibration as threshold
from services.common import PredictionData


ROOT = Path(__file__).resolve().parents[1]


def _frame(rows: int = 900) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-06", periods=rows)
    phase = np.arange(rows, dtype=float)
    score = np.clip(0.5 + np.sin(phase / 11.0) * 0.16 + np.cos(phase / 23.0) * 0.05, 0.05, 0.95)
    target = (np.sin((phase + 3) / 11.0) > -0.08).astype(int)
    values = {feature: np.sin(phase / (index + 5.0)) for index, feature in enumerate(threshold.FROZEN_FEATURES)}
    values["return_2d"] = score
    values["return_5d"] = np.where(np.cos(phase / 8.0) >= 0, 1.0, -1.0)
    values["target_direction"] = target
    values["target_change"] = np.where(target == 1, 1.0, -1.0)
    return pd.DataFrame(values, index=dates)


def _fake_fit(_train: pd.DataFrame, validation: pd.DataFrame):
    probability = validation["return_2d"].to_numpy(dtype=float)
    importance = {feature: 100.0 / len(threshold.FROZEN_FEATURES) for feature in threshold.FROZEN_FEATURES}
    return probability, importance, {"test": True}


def _valid_setting(decision_threshold: float = 0.5) -> dict:
    return {
        "schema_version": 1,
        "model_version": threshold.MODEL_VERSION,
        "model_settings_version": threshold.MODEL_SETTINGS_VERSION,
        "threshold_logic_version": threshold.THRESHOLD_LOGIC_VERSION,
        "decision_threshold": decision_threshold,
        "determined_at": "2026-07-19T12:00:00+09:00",
        "used_data_period": {"start": "2018-07-12", "end": "2024-12-03", "samples": 1561},
        "normal_prediction_recalculates_threshold": False,
        "model_reevaluation_can_update": True,
        "frozen_model": "Extra Trees",
        "frozen_parameters": dict(threshold.FROZEN_EXTRA_TREES_PARAMETERS),
        "frozen_class_balance": "None",
        "frozen_features": list(threshold.FROZEN_FEATURES),
        "threshold_logic": threshold.threshold_logic_metadata(),
        "outer_evaluation": {},
        "final_inner_selection": {},
        "fallback_applied": False,
        "fallback_reason": "test",
        "fixed_confirmation_used_for_selection": False,
        "latest_prediction_used_for_selection": False,
    }


def _candidate(
    value: float,
    balanced: float,
    worst: float,
    std: float,
    macro: float = 0.5,
    accuracy: float = 0.5,
    valid: bool = True,
) -> dict:
    return {
        "threshold": value,
        "mean_direction_balanced_accuracy": balanced,
        "worst_fold_direction_balanced_accuracy": worst,
        "direction_balanced_accuracy_std": std,
        "mean_direction_macro_f1": macro,
        "mean_direction_accuracy": accuracy,
        "valid_prediction_mix": valid,
    }


def test_frozen_extra_trees_features_period_and_parameters_are_unchanged() -> None:
    assert len(threshold.FROZEN_FEATURES) == 28
    assert threshold.FROZEN_EXTRA_TREES_PARAMETERS == {
        "n_estimators": 240,
        "max_depth": 7,
        "min_samples_leaf": 8,
        "max_features": "sqrt",
        "random_state": 42,
        "class_weight": None,
        "n_jobs": -1,
    }
    assert threshold.PURGE_TRADING_DAYS == 5
    assert threshold.PREDICTION_HORIZON == 5


def test_threshold_candidates_are_fixed_symmetric_constants() -> None:
    assert threshold.THRESHOLD_CANDIDATES == (0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60)
    assert threshold.THRESHOLD_CANDIDATE_STEP == 0.02
    assert threshold.THRESHOLD_SELECTION_METRIC == "mean_direction_balanced_accuracy"
    assert threshold.MIN_PREDICTION_SHARE == 0.10


def test_inner_and_outer_walk_forward_use_purge_and_past_only() -> None:
    frame = _frame(900)
    outer = threshold.expanding_purged_walk_forward_splits(frame)
    assert len(outer) == threshold.OUTER_WALK_FORWARD_FOLDS
    for outer_train, outer_validation in outer:
        assert outer_train.index.max() < outer_validation.index.min()
        boundary_position = frame.index.get_loc(outer_validation.index.min())
        assert frame.index.get_loc(outer_train.index.max()) <= boundary_position - threshold.PURGE_TRADING_DAYS - 1
        inner = threshold.expanding_purged_walk_forward_splits(outer_train)
        assert len(inner) == threshold.INNER_WALK_FORWARD_FOLDS
        for inner_train, inner_validation in inner:
            assert inner_train.index.max() < inner_validation.index.min()
            position = outer_train.index.get_loc(inner_validation.index.min())
            assert outer_train.index.get_loc(inner_train.index.max()) <= position - threshold.PURGE_TRADING_DAYS - 1


def test_tie_breaker_prefers_threshold_closest_to_fifty_percent() -> None:
    candidates = [
        _candidate(0.46, 0.55, 0.52, 0.02),
        _candidate(0.50, 0.5495, 0.5195, 0.0205),
        _candidate(0.54, 0.55, 0.52, 0.02),
    ]
    assert threshold.choose_threshold_candidate(candidates)["threshold"] == 0.50


def test_extreme_one_direction_candidate_is_excluded() -> None:
    candidates = [
        _candidate(0.40, 0.70, 0.68, 0.01, valid=False),
        _candidate(0.50, 0.53, 0.51, 0.02, valid=True),
    ]
    assert threshold.choose_threshold_candidate(candidates)["threshold"] == 0.50


def test_tiny_improvement_falls_back_to_fifty_percent() -> None:
    standard_folds = [
        {"direction_balanced_accuracy": value} for value in (0.50, 0.51, 0.52)
    ]
    chosen_folds = [
        {"direction_balanced_accuracy": value} for value in (0.503, 0.513, 0.523)
    ]
    standard = _candidate(0.50, 0.51, 0.50, 0.01)
    standard["folds"] = standard_folds
    chosen = _candidate(0.48, 0.513, 0.503, 0.01)
    chosen["folds"] = chosen_folds
    safe, reasons, details = threshold._threshold_selection_safety(chosen, standard)
    assert safe is False
    assert details["balanced_accuracy_improvement"] < threshold.MIN_BALANCED_IMPROVEMENT
    assert any("未満" in reason for reason in reasons)


def test_inner_selection_uses_only_supplied_outer_training(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame(700)
    observed: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fake_fit(train: pd.DataFrame, validation: pd.DataFrame):
        observed.append((train.index.max(), validation.index.min()))
        return _fake_fit(train, validation)

    monkeypatch.setattr(threshold, "_fit_frozen_extra_trees", fake_fit)
    result = threshold.select_threshold_with_inner_walk_forward(frame)
    assert len(observed) == 3
    assert all(train_end < validation_start for train_end, validation_start in observed)
    assert result["outer_validation_used_for_selection"] is False
    assert result["fixed_confirmation_used_for_selection"] is False
    assert result["latest_prediction_used_for_selection"] is False


def test_outer_validation_is_not_passed_to_inner_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame(900)
    observed_training_ends: list[pd.Timestamp] = []

    def fake_inner(outer_train: pd.DataFrame) -> dict:
        observed_training_ends.append(outer_train.index.max())
        return {
            "selected_threshold": 0.5,
            "fallback_applied": False,
            "fallback_reason": None,
            "candidate_results": [],
        }

    monkeypatch.setattr(threshold, "select_threshold_with_inner_walk_forward", fake_inner)
    monkeypatch.setattr(threshold, "_fit_frozen_extra_trees", _fake_fit)
    monkeypatch.setattr(
        threshold,
        "moving_block_bootstrap_intervals",
        lambda *_: {
            "direction_accuracy_95ci": [0.4, 0.6],
            "best_baseline_gap_95ci": [-0.1, 0.1],
            "block_length": 10,
            "resamples": 10,
            "seed": 42,
            "method": "test bootstrap",
        },
    )
    result = threshold.nested_walk_forward_threshold_evaluation(frame)
    for observed_end, fold in zip(observed_training_ends, result["outer_folds"]):
        assert observed_end < pd.Timestamp(fold["outer_validation_period"]["start"])
        assert fold["outer_validation_used_for_inner_selection"] is False
    assert result["fixed_confirmation_used_for_selection"] is False


def test_outer_evaluation_records_fixed_and_selected_threshold_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(threshold, "_fit_frozen_extra_trees", _fake_fit)
    monkeypatch.setattr(
        threshold,
        "moving_block_bootstrap_intervals",
        lambda *_: {
            "direction_accuracy_95ci": [0.4, 0.6],
            "best_baseline_gap_95ci": [-0.1, 0.1],
            "block_length": 10,
            "resamples": 10,
            "seed": 42,
            "method": "test bootstrap",
        },
    )
    result = threshold.nested_walk_forward_threshold_evaluation(_frame(900))
    assert result["evaluation_type"] == "formal_nested_walk_forward"
    assert result["validation_samples"] == sum(fold["validation_samples"] for fold in result["outer_folds"])
    assert all("fixed_50_metrics" in fold and "selected_threshold_metrics" in fold for fold in result["outer_folds"])


def test_latest_prediction_uses_saved_threshold_and_never_trains_on_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    training = _frame(500)
    inference = training.iloc[[-1]][threshold.FROZEN_FEATURES].copy()
    inference.index = pd.DatetimeIndex([training.index[-1] + pd.offsets.BDay(1)])
    data = PredictionData(training, list(threshold.FROZEN_FEATURES), training, inference, [], "now")
    monkeypatch.setattr(threshold, "_fit_frozen_extra_trees", lambda *_: (np.array([0.52]), {}, {"test": True}))
    monkeypatch.setattr(
        threshold,
        "_six_latest_prediction",
        lambda *_: {"probabilities": [0.2, 0.2, 0.2, 0.1, 0.2, 0.1], "prediction_label": "急騰", "top_probability": 0.2},
    )
    setting = _valid_setting(0.54)
    result = threshold.fit_latest_prediction(data, setting)
    assert result["direction"] == "下落傾向"
    assert result["decision_threshold"] == 0.54
    assert result["threshold_margin"] < 0
    assert result["up_probability"] + result["down_probability"] == pytest.approx(1.0)
    assert result["inference_row_in_training"] is False
    setting["decision_threshold"] = 0.50
    upward = threshold.fit_latest_prediction(data, setting)
    assert upward["direction"] == "上昇傾向"
    assert upward["threshold_margin"] > 0


def test_persistent_threshold_survives_result_cache_deletion_and_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "settings.json"
    history_path = tmp_path / "history.json"
    monkeypatch.setattr(threshold, "THRESHOLD_SETTINGS_PATH", settings_path)
    monkeypatch.setattr(threshold, "THRESHOLD_HISTORY_PATH", history_path)
    setting = _valid_setting(0.48)
    threshold.save_persistent_threshold(setting, "test")
    result_cache = tmp_path / "nikkei_threshold_result_test.json"
    result_cache.write_text("{}", encoding="utf-8")
    result_cache.unlink()
    first = threshold.load_persistent_threshold()
    second = threshold.load_persistent_threshold()
    assert first["decision_threshold"] == second["decision_threshold"] == 0.48


def test_prediction_cache_roundtrip_keeps_result_format(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(threshold, "OUTPUT_DIR", tmp_path)
    result = {"official_evaluation": {"direction_accuracy": 0.53}, "decision_threshold": 0.5}
    created_at = threshold._save_cache("same-key", result, {"version": "test"})
    loaded = threshold._load_cache("same-key")
    assert loaded is not None
    assert loaded["created_at"] == created_at
    assert loaded["result"] == result


def test_threshold_change_history_contains_required_fields(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(threshold, "THRESHOLD_SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(threshold, "THRESHOLD_HISTORY_PATH", tmp_path / "history.json")
    threshold.save_persistent_threshold(_valid_setting(0.50), "initial")
    changed = _valid_setting(0.48)
    changed["determined_at"] = "2026-07-20T12:00:00+09:00"
    threshold.save_persistent_threshold(changed, "explicit reevaluation")
    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    entry = history["changes"][0]
    assert entry == {
        "previous_threshold": 0.5,
        "new_threshold": 0.48,
        "reevaluated_at": "2026-07-20T12:00:00+09:00",
        "data_period": changed["used_data_period"],
        "model_settings_version": threshold.MODEL_SETTINGS_VERSION,
        "change_reason": "explicit reevaluation",
    }


def test_only_explicit_model_reevaluation_calls_threshold_recalibration(monkeypatch: pytest.MonkeyPatch) -> None:
    data_frame = _frame(900)
    inference = data_frame.iloc[[-1]][threshold.FROZEN_FEATURES]
    data = PredictionData(data_frame, list(threshold.FROZEN_FEATURES), data_frame, inference, [], "now")
    selection = data_frame.iloc[:700]
    confirmation = data_frame.iloc[700:]
    calls = {"reevaluate": 0}
    setting = _valid_setting(0.5)
    setting["outer_evaluation"] = {"test": True}

    def fake_reevaluate(_selection: pd.DataFrame) -> dict:
        calls["reevaluate"] += 1
        return setting

    monkeypatch.setattr(threshold, "fixed_period_split", lambda *_: (selection, confirmation))
    monkeypatch.setattr(threshold, "load_persistent_threshold", lambda: setting)
    monkeypatch.setattr(threshold, "reevaluate_and_persist_threshold", fake_reevaluate)
    monkeypatch.setattr(threshold, "_cache_descriptor", lambda *_: {"same": True})
    monkeypatch.setattr(threshold, "_load_cache", lambda *_: {"created_at": "now", "result": {"ok": True}})
    normal = threshold.run_nikkei_threshold_analysis(data, {}, model_reevaluation=False)
    assert normal["ok"] is True
    assert calls["reevaluate"] == 0
    monkeypatch.setattr(threshold, "evaluate_fixed_confirmation", lambda *_: {"feature_importance": []})
    monkeypatch.setattr(threshold, "fit_latest_prediction", lambda *_: {})
    monkeypatch.setattr(threshold, "frozen_configuration", lambda *_: {})
    monkeypatch.setattr(threshold, "_save_cache", lambda *_: "now")
    threshold.run_nikkei_threshold_analysis(data, {}, model_reevaluation=True)
    assert calls["reevaluate"] == 1


def test_api_exposes_explicit_model_reevaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[bool] = []
    monkeypatch.setattr(app_module, "predict_nikkei", lambda value: observed.append(value) or {"kind": "prediction"})
    monkeypatch.setattr(app_module, "save_last_result", lambda *_: None)
    client = TestClient(app_module.app)
    assert client.post("/api/predict/nikkei", json={"model_reevaluation": False}).status_code == 200
    assert client.post("/api/predict/nikkei", json={"model_reevaluation": True}).status_code == 200
    assert observed == [False, True]


def test_general_user_labels_and_model_details_remain_in_ui() -> None:
    script = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
    template = (ROOT / "templates" / "nikkei.html").read_text(encoding="utf-8")
    public_renderer = script.split("const renderNikkeiPublicPrediction", 1)[1].split(
        "const renderSimulation", 1
    )[0]
    for text in (
        "5営業日先の6段階予測",
        "予測精度",
        "上昇予測の一致率",
        "下落予測の一致率",
        "単純な予測方法との差",
        "今回のモデルが選択したファクター",
        "方向予測に使われた特徴量重要度",
        "採用構成と全特徴量の詳細を見る",
    ):
        assert text in public_renderer
    assert "data-model-reevaluation" not in template
    assert "decision_threshold" not in public_renderer
    assert "top_probability" not in public_renderer
    assert "<strong>${escapeHtml(result.direction)}</strong>" not in public_renderer
    assert "training_period" not in public_renderer

def test_six_class_model_implementation_is_not_modified() -> None:
    source = (ROOT / "services" / "nikkei_direction_comparison.py").read_text(encoding="utf-8")
    assert 'model = _new_catboost("enhanced_selected", "six", "Balanced")' in source
    assert "_six_latest_prediction(training, data.inference_row, FROZEN_FEATURES)" in (
        ROOT / "services" / "nikkei_threshold_calibration.py"
    ).read_text(encoding="utf-8")
