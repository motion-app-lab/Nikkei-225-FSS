from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import services.nikkei_direction_comparison as direction
from services.common import PredictionData, ServiceError


def _fixed_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2018-07-12", "2026-07-17")
    x = np.arange(len(dates), dtype=float)
    price = 20_000 + x * 8 + np.sin(x / 11) * 300
    future = pd.Series(price, index=dates).shift(-5)
    frame = pd.DataFrame(
        {
            "open": price - 10,
            "high": price + 80,
            "low": price - 90,
            "price": price,
            "volume": 1_000_000 + x * 10,
            "return_5d": pd.Series(price, index=dates).pct_change(5) * 100,
            "f1": np.sin(x / 9),
            "f2": np.cos(x / 17),
            "target_change": (future / price - 1.0) * 100,
            "target_direction": np.where(future.notna(), (future > price).astype(float), np.nan),
        },
        index=dates,
    )
    return frame.dropna(subset=["target_change", "target_direction"])


def _prediction_data() -> PredictionData:
    training = _fixed_frame()
    features = ["open", "high", "low", "price", "volume", "return_5d", "f1", "f2"]
    inference_date = training.index[-1] + pd.offsets.BDay(6)
    inference = training.iloc[[-1]][features].copy()
    inference.index = pd.DatetimeIndex([inference_date])
    return PredictionData(
        frame=training.copy(),
        feature_columns=features,
        training_frame=training.copy(),
        inference_row=inference,
        warnings=[],
        fetched_at="2026-07-19T12:00:00+09:00",
    )


def _mock_market_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2018-04-23", "2026-07-17")
    x = np.arange(len(dates), dtype=float)
    price = 20_000 + x * 9 + np.sin(x / 13) * 200
    base = pd.DataFrame(
        {
            "open": price - 20,
            "high": price + 100,
            "low": price - 120,
            "price": price,
            "volume": 1_000_000 + x * 100,
        },
        index=dates,
    )
    macro = pd.DataFrame({"spx": 2_000 + x * 1.5}, index=dates)
    return base, macro


def test_fixed_period_and_direction_threshold_are_constants() -> None:
    assert direction.FIXED_COMPARISON_START == pd.Timestamp("2018-07-12")
    assert direction.FIXED_EVALUATION_START == pd.Timestamp("2024-12-04")
    assert direction.FIXED_EVALUATION_END == pd.Timestamp("2026-07-10")
    assert direction.DIRECTION_THRESHOLD == 0.5
    assert direction.direction_from_probability(np.array([0.4999, 0.5])).tolist() == [0, 1]


def test_direction_target_is_direct_five_day_binary_and_ties_are_missing() -> None:
    prices = pd.Series([100, 100, 100, 100, 100, 101, 99, 100, 100, 100, 100], dtype=float)
    target, future = direction.make_direction_target(prices, horizon=5)
    assert future.iloc[0] == 101
    assert target.iloc[0] == 1
    assert target.iloc[1] == 0
    assert pd.isna(target.iloc[2])  # 100 -> 100 is an exact tie
    assert target.tail(5).isna().all()


def test_fixed_split_keeps_evaluation_period_out_of_selection() -> None:
    selection, holdout = direction.fixed_period_split(_fixed_frame())
    assert selection.index.min() == direction.FIXED_COMPARISON_START
    assert selection.index.max() == direction.FIXED_SELECTION_END
    assert holdout.index.min() == direction.FIXED_EVALUATION_START
    assert holdout.index.max() == direction.FIXED_EVALUATION_END
    assert selection.index.max() < holdout.index.min()


def test_walk_forward_has_five_day_purge_and_no_target_overlap() -> None:
    selection, _ = direction.fixed_period_split(_fixed_frame())
    folds = direction.expanding_purged_walk_forward_splits(selection)
    previous_training_size = 0
    for train, validation in folds:
        train_last_position = selection.index.get_loc(train.index[-1])
        validation_first_position = selection.index.get_loc(validation.index[0])
        assert validation_first_position - train_last_position - 1 == direction.PURGE_TRADING_DAYS
        assert train_last_position + direction.PREDICTION_HORIZON < validation_first_position
        assert train.index.max() < validation.index.min()
        assert len(train) > previous_training_size
        previous_training_size = len(train)


def test_all_candidates_use_identical_validation_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    selection, _ = direction.fixed_period_split(_fixed_frame())
    features = ["return_5d", "f1", "f2"]

    def fake_selection(train: pd.DataFrame, all_features: list[str], _factors: dict) -> dict:
        return {
            "selected_features": all_features[:2],
            "selected_factors": ["nikkei_price"],
            "importance": {feature: float(index + 1) for index, feature in enumerate(all_features)},
            "factor_scores": [],
            "fit_period": {"start": str(train.index[0].date()), "end": str(train.index[-1].date())},
            "used_validation_data": False,
        }

    def fake_fit(_definition: dict, _train: pd.DataFrame, validation: pd.DataFrame, used: list[str]):
        probability = np.clip(0.5 + validation["f1"].to_numpy() * 0.1, 0, 1)
        return probability, {feature: 1.0 for feature in used}, {"fit_on_train": True}

    monkeypatch.setattr(direction, "_training_only_feature_selection", fake_selection)
    monkeypatch.setattr(direction, "_fit_component", fake_fit)
    monkeypatch.setattr(
        direction,
        "_six_class_evaluation",
        lambda _train, validation, _features: {
            "six_class_accuracy": 0.2,
            "six_class_macro_f1": 0.1,
            "six_class_distribution": {label: 0 for label in direction.CLASS_ORDER},
        },
    )
    result = direction.compare_direction_models(selection, features, {"nikkei_price": features})
    assert len(result["candidate_results"]) == 6
    assert any(candidate["id"] == "current_catboost" for candidate in result["candidate_results"])
    assert len(result["class_balance_screening"]) == 3
    assert all(item["used_fixed_evaluation"] is False for item in result["class_balance_screening"])
    reference_dates = [tuple(dates) for dates in result["validation_dates_by_fold"]]
    reference_sizes = [len(dates) for dates in reference_dates]
    for candidate in result["candidate_results"]:
        candidate_dates = [
            tuple(pd.bdate_range(fold["validation_start"], fold["validation_end"]).strftime("%Y-%m-%d"))
            for fold in candidate["folds"]
        ]
        assert candidate_dates == reference_dates
        assert [fold["validation_samples"] for fold in candidate["folds"]] == reference_sizes


def test_training_median_imputation_does_not_use_validation_values() -> None:
    train = pd.DataFrame({"f": [1.0, np.nan, 3.0]})
    validation = pd.DataFrame({"f": [np.nan, 10_000.0]})
    x_train, x_validation, medians = direction._imputed_arrays(train, validation, ["f"])
    assert medians["f"] == 2.0
    assert x_train[1, 0] == 2.0
    assert x_validation[0, 0] == 2.0


def test_feature_selection_receives_training_only(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, int] = {}

    class FakeModel:
        def fit(self, x: np.ndarray, y: np.ndarray) -> None:
            observed["rows"] = len(x)
            observed["labels"] = len(y)

        def get_feature_importance(self) -> np.ndarray:
            return np.array([2.0, 1.0])

    monkeypatch.setattr(direction, "_new_catboost", lambda *_: FakeModel())
    train = _fixed_frame().iloc[:300]
    result = direction._training_only_feature_selection(
        train,
        ["f1", "f2"],
        {"trend_regime": ["f1"], "return_range": ["f2"]},
    )
    assert observed == {"rows": len(train), "labels": len(train)}
    assert result["used_validation_data"] is False
    assert result["fit_period"]["end"] == str(train.index[-1].date())


def test_prepare_data_shifts_external_market_one_tokyo_session_and_excludes_pce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, macro = _mock_market_data()
    monkeypatch.setattr(direction, "_download_base_frame", lambda *_: base.copy())
    monkeypatch.setattr(direction, "_download_macro_frame", lambda *_: (macro.copy(), []))
    data, metadata = direction.prepare_nikkei_direction_data(now=pd.Timestamp("2026-07-19").to_pydatetime())
    assert pd.isna(data.frame["spx"].iloc[0])
    assert data.frame["spx"].iloc[10] == macro["spx"].iloc[9]
    assert metadata["external_alignment"]["spx"]["lag_trading_sessions"] == 1
    assert metadata["pce_used"] is False
    assert "公表日" in metadata["pce_exclusion_reason"]
    assert metadata["fixed_data_start"] == "2018-07-12"
    assert metadata["fixed_evaluation_period"]["start"] == "2024-12-04"


def test_rolling_features_use_current_and_past_rows_only(monkeypatch: pytest.MonkeyPatch) -> None:
    base, macro = _mock_market_data()
    monkeypatch.setattr(direction, "_download_base_frame", lambda *_: base.copy())
    monkeypatch.setattr(direction, "_download_macro_frame", lambda *_: (macro.copy(), []))
    data, _ = direction.prepare_nikkei_direction_data(now=pd.Timestamp("2026-07-19").to_pydatetime())
    position = 400
    date = base.index[position]
    expected_ma20 = base["price"].iloc[position - 19 : position + 1].mean()
    expected_gap = (base.loc[date, "price"] / expected_ma20 - 1.0) * 100
    expected_high = base["price"].iloc[position - 251 : position + 1].max()
    assert data.frame.loc[date, "ma20_gap"] == pytest.approx(expected_gap)
    assert data.frame.loc[date, "distance_high252"] == pytest.approx(
        (base.loc[date, "price"] / expected_high - 1.0) * 100
    )


def test_simple_average_ensemble_and_probability_sum() -> None:
    result = direction.simple_average_probabilities(
        [np.array([0.2, 0.8]), np.array([0.4, 0.6]), np.array([0.6, 0.4])]
    )
    assert result.tolist() == pytest.approx([0.4, 0.6])
    assert (result + (1.0 - result)).tolist() == pytest.approx([1.0, 1.0])


def _candidate(candidate_id: str, accuracy: float, balance: float) -> dict:
    return {
        "id": candidate_id,
        "summary": {
            "direction_accuracy_mean": accuracy,
            "direction_balanced_accuracy_mean": balance,
            "best_baseline_gap_mean": accuracy - 0.52,
            "direction_accuracy_std": 0.02,
            "direction_accuracy_worst_fold": accuracy - 0.03,
            "direction_macro_f1_mean": balance - 0.02,
        },
    }


def test_current_model_is_kept_when_new_candidates_do_not_improve() -> None:
    candidates = [_candidate("current_catboost", 0.54, 0.53)] + [
        _candidate(definition["id"], 0.53, 0.54)
        for definition in direction.CANDIDATE_DEFINITIONS
        if definition["id"] != "current_catboost"
    ]
    selected, reason = direction.choose_direction_candidate(candidates)
    assert selected["id"] == "current_catboost"
    assert "上回らなかった" in reason


def test_latest_inference_row_is_never_training_and_probabilities_sum_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _prediction_data()
    configuration = {
        "selected_features": ["return_5d", "f1"],
        "selected_model": {"id": "extra_trees_selected", "label": "Extra Trees", "kind": "extra_trees"},
    }
    monkeypatch.setattr(
        direction,
        "_predict_selected_model",
        lambda *_: (np.array([0.61]), {"f1": 1.0}, {"fit_on_training": True}),
    )
    monkeypatch.setattr(
        direction,
        "_six_latest_prediction",
        lambda *_: {
            "probabilities": [0.1, 0.2, 0.2, 0.2, 0.2, 0.1],
            "prediction_label": "上昇",
            "top_probability": 0.2,
        },
    )
    result = direction.fit_latest_prediction(data, configuration)
    assert result["inference_row_in_training"] is False
    assert result["up_probability"] + result["down_probability"] == pytest.approx(1.0)
    assert result["direction"] == "上昇傾向"

    duplicated = copy.deepcopy(data)
    duplicated.inference_row = duplicated.training_frame.iloc[[-1]][duplicated.feature_columns]
    with pytest.raises(ServiceError, match="混入"):
        direction.fit_latest_prediction(duplicated, configuration)


def test_moving_block_bootstrap_is_reproducible() -> None:
    model = np.resize(np.array([1, 0, 1, 1, 0], dtype=bool), 100)
    baseline = np.resize(np.array([1, 1, 0, 1, 0], dtype=bool), 100)
    first = direction.moving_block_bootstrap_intervals(model, baseline, block_length=5, resamples=200, seed=7)
    second = direction.moving_block_bootstrap_intervals(model, baseline, block_length=5, resamples=200, seed=7)
    assert first == second
    assert first["method"] == "circular moving-block bootstrap"


def test_cache_fresh_and_hit_have_same_result_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = _prediction_data()
    selection, holdout = direction.fixed_period_split(data.training_frame)
    metadata = {"factor_definitions": {"nikkei_price": ["return_5d"]}}
    configuration = {
        "selected_model": {"id": "current_catboost", "label": "current", "kind": "catboost"},
        "selected_features": ["return_5d"],
        "excluded_features": [],
        "selected_factors": ["日経平均"],
        "excluded_factors": [],
        "adoption_reason": "test",
    }
    calls = {"selection": 0}

    def fake_compare(*_args):
        calls["selection"] += 1
        return copy.deepcopy(configuration)

    monkeypatch.setattr(direction, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(direction, "_cache_descriptor", lambda *_: {"version": "test", "data": "same"})
    monkeypatch.setattr(direction, "fixed_period_split", lambda *_: (selection, holdout))
    monkeypatch.setattr(direction, "compare_direction_models", fake_compare)
    monkeypatch.setattr(
        direction,
        "evaluate_fixed_holdout",
        lambda *_: {"direction_accuracy": 0.5, "feature_importance": []},
    )
    monkeypatch.setattr(direction, "fit_latest_prediction", lambda *_: {"up_probability": 0.5})
    fresh = direction.run_nikkei_direction_analysis(data, metadata)
    cached = direction.run_nikkei_direction_analysis(data, metadata)
    assert fresh.keys() == cached.keys()
    assert fresh["configuration"] == cached["configuration"]
    assert fresh["final_evaluation"] == cached["final_evaluation"]
    assert fresh["latest_prediction"] == cached["latest_prediction"]
    assert fresh["cache"]["used"] is False
    assert cached["cache"]["used"] is True
    assert calls["selection"] == 1


def test_cache_signature_ignores_insignificant_vendor_float_noise() -> None:
    data = _prediction_data()
    data.training_frame["spx"] = 4_000.0
    data.inference_row["spx"] = 4_100.0
    data.feature_columns.append("spx")
    metadata = {"available_external_factors": ["spx"]}
    first = direction._data_signature(data, metadata)
    data.training_frame.loc[data.training_frame.index[10], "spx"] += 0.00004
    second = direction._data_signature(data, metadata)
    assert first == second
