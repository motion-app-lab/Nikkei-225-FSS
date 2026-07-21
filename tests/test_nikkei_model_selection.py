from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as app_module
import services.nikkei_model_selection as selection_module
from services.common import PredictionData, ServiceError, _add_pce, _six_probabilities
from services.nikkei_model_selection import (
    _fit_latest_prediction,
    choose_reduced_candidate_or_fallback,
    expanding_walk_forward_splits,
    majority_baseline_predictions,
    momentum_baseline_predictions,
    run_nikkei_model_selection,
    split_selection_and_holdout,
)


def _synthetic_prediction_data(rows: int = 520) -> PredictionData:
    dates = pd.bdate_range("2022-01-03", periods=rows)
    cycle = np.array([-3.0, -1.0, -0.1, 0.8, 3.0, 6.0])
    target = np.resize(cycle, rows)
    frame = pd.DataFrame(
        {
            "return_5d": np.sin(np.arange(rows) / 7.0) * 3,
            "rsi14": 50 + np.cos(np.arange(rows) / 9.0) * 20,
            "extra": np.arange(rows, dtype=float),
            "target_change": target,
        },
        index=dates,
    )
    inference_index = pd.DatetimeIndex([dates[-1] + pd.offsets.BDay(1)])
    inference = pd.DataFrame(
        {"return_5d": [1.2], "rsi14": [55.0], "extra": [rows + 1.0]},
        index=inference_index,
    )
    return PredictionData(
        frame=frame,
        feature_columns=["return_5d", "rsi14", "extra"],
        training_frame=frame.copy(),
        inference_row=inference,
        warnings=[],
        fetched_at="2026-07-19T12:00:00+09:00",
    )


def _summary(direction: float, balanced: float, macro_f1: float = 0.2, std: float = 0.01) -> dict:
    return {
        "direction_accuracy_mean": direction,
        "direction_balanced_accuracy_mean": balanced,
        "six_class_macro_f1_mean": macro_f1,
        "direction_accuracy_std": std,
    }


def _candidate(name: str, direction: float, balanced: float, fold_values: list[float]) -> dict:
    return {
        "name": name,
        "evaluation": {
            "features": ["return_5d"],
            "summary": _summary(direction, balanced),
            "folds": [{"direction_accuracy": value} for value in fold_values],
        },
    }


def test_final_holdout_is_not_passed_to_model_selection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = _synthetic_prediction_data()
    expected_selection, expected_holdout = split_selection_and_holdout(data.training_frame)
    observed: dict[str, pd.DatetimeIndex] = {}
    configuration = {
        "selected_features": ["return_5d"],
        "catboost_setting": {"name": "test", "iterations": 5, "depth": 2, "learning_rate": 0.1, "l2_leaf_reg": 3.0},
        "class_balance": "None",
    }

    def fake_select(selection: pd.DataFrame, _features: list[str]) -> dict:
        observed["selection"] = selection.index
        return configuration

    def fake_final(selection: pd.DataFrame, holdout: pd.DataFrame, _configuration: dict) -> dict:
        observed["final_train"] = selection.index
        observed["holdout"] = holdout.index
        return {"period": {"start": str(holdout.index[0].date()), "end": str(holdout.index[-1].date())}}

    monkeypatch.setattr(selection_module, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(selection_module, "select_model_configuration", fake_select)
    monkeypatch.setattr(selection_module, "evaluate_final_holdout", fake_final)
    monkeypatch.setattr(selection_module, "_fit_latest_prediction", lambda *_: {"prediction_label": "上昇"})
    result = run_nikkei_model_selection(data)

    assert observed["selection"].equals(expected_selection.index)
    assert observed["final_train"].equals(expected_selection.index)
    assert observed["holdout"].equals(expected_holdout.index)
    assert observed["selection"].max() < observed["holdout"].min()
    assert result["cache"]["used"] is False


def test_latest_inference_row_cannot_be_in_training_data() -> None:
    data = _synthetic_prediction_data()
    duplicated_date = data.training_frame.index[-1]
    data.inference_row = data.training_frame.loc[[duplicated_date], data.feature_columns]
    configuration = {
        "selected_features": ["return_5d"],
        "catboost_setting": {"name": "test", "iterations": 5, "depth": 2, "learning_rate": 0.1, "l2_leaf_reg": 3.0},
        "class_balance": "None",
    }
    with pytest.raises(ServiceError, match="混入"):
        _fit_latest_prediction(data, configuration)


def test_walk_forward_training_is_always_before_validation() -> None:
    frame = _synthetic_prediction_data().training_frame
    folds = expanding_walk_forward_splits(frame)
    assert len(folds) == 3
    previous_train_size = 0
    for train, validation in folds:
        assert train.index.max() < validation.index.min()
        assert len(train) > previous_train_size
        previous_train_size = len(train)


def test_majority_baseline_uses_training_majority() -> None:
    predictions = majority_baseline_predictions(np.array([1, 1, 1, 0]), 5)
    assert predictions.tolist() == [1, 1, 1, 1, 1]


def test_momentum_baseline_does_not_use_future_target() -> None:
    validation = pd.DataFrame(
        {"return_5d": [-1.0, 0.0, 2.0], "target_change": [99.0, -99.0, 99.0]}
    )
    first = momentum_baseline_predictions(validation)
    validation["target_change"] = [-500.0, 500.0, -500.0]
    second = momentum_baseline_predictions(validation)
    assert first.tolist() == [0, 1, 1]
    assert np.array_equal(first, second)


def test_selected_feature_subset_matches_latest_inference_columns() -> None:
    data = _synthetic_prediction_data(260)
    configuration = {
        "selected_features": ["return_5d", "rsi14"],
        "catboost_setting": {"name": "test", "iterations": 8, "depth": 3, "learning_rate": 0.1, "l2_leaf_reg": 3.0},
        "class_balance": "None",
    }
    result = _fit_latest_prediction(data, configuration)
    assert len(result["probabilities"]) == 6
    assert sum(result["probabilities"]) == pytest.approx(1.0)


def test_factor_reduction_falls_back_when_it_does_not_improve() -> None:
    all_factors = _candidate("all_factors", 0.54, 0.53, [0.53, 0.55, 0.54])
    reduced = _candidate("reduced", 0.53, 0.52, [0.52, 0.54, 0.53])
    selected = choose_reduced_candidate_or_fallback([all_factors, reduced], "all_factors")
    assert selected["name"] == "all_factors"


def test_missing_catboost_classes_map_to_six_probabilities() -> None:
    class FakeModel:
        classes_ = np.array([0, 2, 5])

    probabilities = _six_probabilities(FakeModel(), np.array([0.2, 0.3, 0.5]))
    assert probabilities == [0.2, 0.0, 0.3, 0.0, 0.0, 0.5]


def test_fred_unset_continues_without_pce(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    macro = pd.DataFrame(index=pd.bdate_range("2025-01-01", periods=5))
    warnings: list[str] = []
    _add_pce(macro.index, macro, warnings)
    assert "pce" not in macro.columns
    assert any("PCEを除外" in warning for warning in warnings)


def test_cache_and_fresh_result_have_same_format(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = _synthetic_prediction_data()
    configuration = {
        "selected_features": ["return_5d"],
        "catboost_setting": {"name": "test", "iterations": 5, "depth": 2, "learning_rate": 0.1, "l2_leaf_reg": 3.0},
        "class_balance": "None",
    }
    calls = {"selection": 0}

    def fake_select(*_args) -> dict:
        calls["selection"] += 1
        return configuration

    monkeypatch.setattr(selection_module, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(selection_module, "select_model_configuration", fake_select)
    monkeypatch.setattr(selection_module, "evaluate_final_holdout", lambda *_: {"direction_accuracy": 0.5})
    monkeypatch.setattr(selection_module, "_fit_latest_prediction", lambda *_: {"prediction_label": "上昇", "probabilities": [0, 1, 0, 0, 0, 0]})

    fresh = run_nikkei_model_selection(data)
    cached = run_nikkei_model_selection(data)
    assert fresh.keys() == cached.keys()
    assert fresh["configuration"] == cached["configuration"]
    assert fresh["final_evaluation"] == cached["final_evaluation"]
    assert fresh["latest_prediction"].keys() == cached["latest_prediction"].keys()
    assert fresh["cache"]["used"] is False
    assert cached["cache"]["used"] is True
    assert calls["selection"] == 1


def test_force_refresh_does_not_trigger_model_reevaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, bool] = {}

    def fake_predict(force_refresh: bool) -> dict:
        observed["force_refresh"] = force_refresh
        return {"kind": "prediction", "ticker": "^N225"}

    monkeypatch.setattr(app_module, "predict_nikkei", fake_predict)
    monkeypatch.setattr(app_module, "save_last_result", lambda *_: None)
    client = TestClient(app_module.app)
    response = client.post("/api/predict/nikkei", json={"force_refresh": True})
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert observed["force_refresh"] is False
