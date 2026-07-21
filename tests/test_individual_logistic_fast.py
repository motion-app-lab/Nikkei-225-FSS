from __future__ import annotations

import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import services.individual_logistic_fast as fast
import services.individual_service as service
from services.common import CLASS_ORDER
from services.individual_market import IndividualMarketData


def _synthetic_data(rows: int = 2200, feature_count: int = 24) -> IndividualMarketData:
    index = pd.bdate_range("2017-01-04", periods=rows)
    rng = np.random.default_rng(20260721)
    frame = pd.DataFrame(index=index)
    frame["price"] = 2000 + np.cumsum(rng.normal(0.4, 12.0, rows))
    frame["open"] = frame["price"] * (1 + rng.normal(0, 0.002, rows))
    frame["high"] = np.maximum(frame["open"], frame["price"]) * 1.01
    frame["low"] = np.minimum(frame["open"], frame["price"]) * 0.99
    frame["volume"] = 1_000_000 + rng.integers(0, 100_000, rows)
    features = []
    prefixes = ["stock", "nikkei", "spx", "nasdaq", "vix", "usdjpy"]
    for number in range(feature_count):
        name = f"{prefixes[number % len(prefixes)]}_feature_{number}"
        frame[name] = rng.normal(size=rows) + np.sin(np.arange(rows) / (9 + number))
        features.append(name)
    classes = np.asarray([6.0, 3.0, 1.0, -0.1, -1.0, -3.0])
    changes = classes[np.arange(rows) % len(classes)]
    frame["target_change"] = changes
    frame["target_direction"] = (changes > 0).astype(float)
    frame["target_equal_close"] = False
    frame["stock_ret5"] = rng.normal(size=rows)
    frame.loc[index[-5:], ["target_change", "target_direction"]] = np.nan
    return IndividualMarketData(
        frame=frame,
        feature_groups={fast.FIXED_FEATURE_GROUP: features},
        group_valid_starts={fast.FIXED_FEATURE_GROUP: index[300]},
        basis_date=index[-1],
        prediction_context="after_close",
        prediction_mode="大引け後",
        prediction_cutoff_jst=pd.Timestamp(index[-1]).tz_localize("Asia/Tokyo"),
        first_training_date=index[300],
        warmup_start=index[0],
        warnings=[],
        fetched_at="2026-07-21T16:00:00+09:00",
        metadata={"feature_definition_version": "schema-test"},
    )


def test_fixed_logistic_settings_are_common_and_supported() -> None:
    assert fast.LOGISTIC_SETTINGS == {
        "solver": "lbfgs",
        "C": 1.0,
        "max_iter": 300,
        "tol": 1e-4,
        "class_weight": "balanced",
        "random_state": 42,
    }
    assert isinstance(fast._new_model(), LogisticRegression)


def test_normal_prediction_fits_exactly_two_logistic_models_and_no_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _synthetic_data()
    original = LogisticRegression.fit
    calls = {"count": 0}

    def counted(self, x, y, *args, **kwargs):
        calls["count"] += 1
        return original(self, x, y, *args, **kwargs)

    monkeypatch.setattr(LogisticRegression, "fit", counted)
    started = time.perf_counter()
    result = fast.latest_prediction(data)
    elapsed = time.perf_counter() - started
    assert calls["count"] == 2
    assert result["fit_counts"] == {"direction": 1, "six_class": 1, "total": 2}
    assert result["selection"]["cross_validation_count"] == 0
    assert result["selection"]["candidate_comparison_count"] == 0
    assert result["direction_model"]["algorithm"] == "logistic_regression"
    assert result["six_class_model"]["algorithm"] == "logistic_regression"
    assert elapsed < 5.0


def test_training_matrix_and_preprocessor_are_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _synthetic_data()
    original = fast.fit_preprocessor
    calls = {"count": 0}

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fast, "fit_preprocessor", counted)
    result = fast.latest_prediction(data)
    assert calls["count"] == 1
    assert result["direction_features"] == result["six_class_features"]


def test_preprocessing_fits_training_statistics_only_and_removes_constant_inf() -> None:
    training = pd.DataFrame({
        "signal": [1.0, 2.0, np.nan, 4.0],
        "constant": [5.0, 5.0, 5.0, 5.0],
        "infinite": [1.0, np.inf, 3.0, 4.0],
        "text": ["a", "b", "c", "d"],
    })
    fitted, x = fast.fit_preprocessor(training, ["signal", "constant", "infinite", "text"])
    assert fitted.medians["signal"] == pytest.approx(2.0)
    assert "constant" not in fitted.columns
    assert "text" not in fitted.columns
    assert np.isfinite(x).all()
    validation = pd.DataFrame({"signal": [1_000_000.0], "infinite": [np.nan]})
    fast.transform_features(validation, fitted)
    assert fitted.medians["signal"] == pytest.approx(2.0)


def test_six_class_probabilities_do_not_depend_on_model_class_order() -> None:
    fake = SimpleNamespace(
        classes_=np.asarray([5, 0, 2]),
        predict_proba=lambda _x: np.asarray([[0.2, 0.3, 0.5]]),
    )
    probabilities = fast._probabilities_for_classes(fake, np.zeros((1, 2)), 6)[0]
    assert probabilities.tolist() == pytest.approx([0.3, 0.0, 0.5, 0.0, 0.0, 0.2])
    assert probabilities.sum() == pytest.approx(1.0)


def test_public_six_stage_output_has_all_classes_and_totals_one() -> None:
    result = fast.latest_prediction(_synthetic_data())
    assert [item["raw_label"] for item in result["probabilities"]] == CLASS_ORDER
    assert sum(item["probability"] for item in result["probabilities"]) == pytest.approx(1.0)
    assert result["fixed_direction_threshold"] == 0.50


def test_normal_service_source_does_not_run_formal_evaluation_or_candidates() -> None:
    source = inspect.getsource(service.predict_individual)
    assert "update_and_save_formal_evaluation(" not in source
    assert "formal_outer" not in source
    assert "outer_fold" not in source
    assert "inner_fold" not in source
    assert "select_direction_configuration" not in source
    assert "select_six_configuration" not in source
    assert "latest_prediction(data)" in source
    combined = inspect.getsource(fast.latest_prediction) + inspect.getsource(service.predict_individual)
    assert "ExtraTrees" not in combined
    assert "CatBoost" not in combined


def test_fixed_module_has_no_topix_or_alternative_topix_ticker() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in ("services/individual_logistic_fast.py", "services/individual_service.py", "services/individual_market.py"):
        source = (root / relative).read_text(encoding="utf-8").lower()
        assert "topix" not in source
        assert "^topx" not in source
        assert "998405.t" not in source



def test_public_service_does_not_load_or_return_individual_accuracy() -> None:
    source = inspect.getsource(service.predict_individual)
    assert "load_formal_evaluation" not in source
    assert "saved_formal_evaluation" not in source
    assert '"validation"' not in source
    assert '"formal_evaluation_available"' not in source


def test_release_tree_has_no_individual_accuracy_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "outputs" / "individual_evaluations").exists()
    assert not (root / "model_settings" / "individual_evaluations").exists()
    assert not (root / "tools" / "check_individual_evaluation.py").exists()
    assert not (root / "tools" / "update_individual_logistic_evaluation.py").exists()
