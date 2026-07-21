from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from services.nikkei_artifact import load_context_evaluation
from services.nikkei_dual_market import (
    EVALUATION_YEARS,
    EXTERNAL_SOURCES,
    HISTORICAL_INTRADAY_CUTOFF_JST,
    INNER_FOLDS,
    JAPAN_FEATURES,
    JAPAN_SCORE_WEIGHT,
    LEARNING_YEARS,
    MARKET_CLOSE_CONFIRMATION_JST,
    OUTER_FOLDS,
    OVERSEAS_SCORE_WEIGHT,
    PURGE_TRADING_DAYS,
    THRESHOLD_CANDIDATES,
    WARMUP_TRADING_DAYS,
    asof_align_external,
    build_external_feature_frame,
    external_available_timestamp,
    historical_prediction_cutoffs,
    resolve_prediction_context,
)
from services.nikkei_dual_model import (
    _decay_is_safe,
    load_model_settings,
    make_inner_splits,
    select_combined_threshold,
    training_sample_weights,
)


ROOT = Path(__file__).resolve().parents[1]
JST = ZoneInfo("Asia/Tokyo")


def _base_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(["2026-07-15", "2026-07-16", "2026-07-17"])
    return pd.DataFrame(
        {
            "open": [50000.0, 50100.0, 50200.0],
            "high": [50200.0, 50300.0, 50400.0],
            "low": [49900.0, 50000.0, 50100.0],
            "price": [50100.0, 50200.0, 50300.0],
            "volume": [1.0, 1.0, 1.0],
        },
        index=index,
    )


def test_frozen_rolling_period_and_evaluation_constants() -> None:
    assert LEARNING_YEARS == 8
    assert EVALUATION_YEARS == 2
    assert WARMUP_TRADING_DAYS == 300
    assert OUTER_FOLDS == 8
    assert INNER_FOLDS == 3
    assert PURGE_TRADING_DAYS == 5
    assert JAPAN_SCORE_WEIGHT == pytest.approx(0.5)
    assert OVERSEAS_SCORE_WEIGHT == pytest.approx(0.5)


def test_inner_folds_are_expanding_purged_and_rolling_eight_years() -> None:
    index = pd.bdate_range("2012-01-02", periods=3000)
    splits = make_inner_splits(index)
    assert len(splits) == 3
    previous_train_end: pd.Timestamp | None = None
    for split in splits:
        train = split["train_index"]
        validation = split["validation_index"]
        assert train.max() < validation.min()
        assert len(split["purged_dates"]) == PURGE_TRADING_DAYS
        assert train.min() >= train.max() - pd.DateOffset(years=LEARNING_YEARS)
        if previous_train_end is not None:
            assert train.max() > previous_train_end
        previous_train_end = train.max()


def test_time_decay_weights_match_fixed_formula() -> None:
    training_end = pd.Timestamp("2026-07-10")
    index = pd.DatetimeIndex(
        [training_end - pd.DateOffset(years=8), training_end - pd.DateOffset(years=4), training_end]
    )
    weights = training_sample_weights(index, training_end, "half_life_4y")
    assert weights[0] == pytest.approx(0.25, abs=0.002)
    assert weights[1] == pytest.approx(0.50, abs=0.002)
    assert weights[2] == pytest.approx(1.00)
    assert np.all(training_sample_weights(index, training_end, "uniform") == 1.0)


def test_decay_falls_back_when_safety_gain_is_too_small() -> None:
    def record(values: list[float]) -> dict:
        folds = [
            {
                "direction_balanced_accuracy": value,
                "predicted_up": 50,
                "predicted_down": 50,
                "validation_samples": 100,
            }
            for value in values
        ]
        return {"folds": folds}

    safe, reason = _decay_is_safe(record([0.52, 0.53, 0.51]), record([0.521, 0.531, 0.511]))
    assert safe is False
    assert "均等重み" in reason


def test_us_close_availability_is_dst_aware() -> None:
    winter = external_available_timestamp("2026-03-06", "us_close")
    summer = external_available_timestamp("2026-03-09", "us_close")
    assert winter.tzinfo is not None and summer.tzinfo is not None
    assert winter.hour == 6
    assert summer.hour == 5
    before_end = external_available_timestamp("2026-10-30", "us_close")
    after_end = external_available_timestamp("2026-11-02", "us_close")
    assert before_end.hour == 5
    assert after_end.hour == 6


def test_asof_uses_latest_confirmed_us_session_across_japan_holiday() -> None:
    source_index = pd.DatetimeIndex(["2026-07-16", "2026-07-17", "2026-07-20"])
    values = pd.Series([100.0, 101.0, 102.0], index=source_index)
    source = build_external_feature_frame("spx", values)
    cutoff = pd.Series(
        [pd.Timestamp("2026-07-21 09:00", tz=JST)],
        index=pd.DatetimeIndex(["2026-07-17"]),
        dtype="object",
    )
    aligned, diagnostics = asof_align_external(cutoff, {"spx": source})
    assert aligned.loc[pd.Timestamp("2026-07-17"), "spx"] == pytest.approx(102.0)
    assert pd.Timestamp(diagnostics["spx"].iloc[0]["source_date"]).date().isoformat() == "2026-07-20"


def test_asof_never_uses_future_us_value_when_us_is_closed() -> None:
    source_index = pd.DatetimeIndex(["2026-07-02", "2026-07-06"])
    values = pd.Series([100.0, 110.0], index=source_index)
    source = build_external_feature_frame("spx", values)
    cutoff = pd.Series(
        [pd.Timestamp("2026-07-06 09:00", tz=JST)],
        index=pd.DatetimeIndex(["2026-07-03"]),
        dtype="object",
    )
    aligned, diagnostics = asof_align_external(cutoff, {"spx": source})
    assert aligned.iloc[0]["spx"] == pytest.approx(100.0)
    assert pd.Timestamp(diagnostics["spx"].iloc[0]["source_date"]).date().isoformat() == "2026-07-02"


def test_prediction_context_excludes_unconfirmed_intraday_bar() -> None:
    frame = _base_frame()
    intraday = resolve_prediction_context(frame, datetime(2026, 7, 17, 10, 0, tzinfo=JST))
    assert intraday["prediction_context"] == "intraday"
    assert intraday["nikkei_base_date"] == "2026-07-16"
    assert intraday["nikkei_base_close"] == pytest.approx(50200.0)
    assert "09:00:00+09:00" in intraday["external_cutoff_timestamp_jst"]


def test_prediction_context_uses_confirmed_after_close_bar() -> None:
    frame = _base_frame()
    after_close = resolve_prediction_context(frame, datetime(2026, 7, 17, 16, 0, tzinfo=JST))
    assert after_close["prediction_context"] == "after_close"
    assert after_close["nikkei_base_date"] == "2026-07-17"
    assert after_close["nikkei_base_close"] == pytest.approx(50300.0)
    assert "15:40:00+09:00" in after_close["external_cutoff_timestamp_jst"]


def test_feature_panels_are_strictly_separated() -> None:
    forbidden_japan = {factor for factor in EXTERNAL_SOURCES}
    assert forbidden_japan.isdisjoint(JAPAN_FEATURES)
    assert "price" in JAPAN_FEATURES and "return_5d" in JAPAN_FEATURES
    settings = load_model_settings()
    overseas = set(settings["adopted_features"]["overseas"])
    assert not {"open", "high", "low", "price", "volume"}.intersection(overseas)


def test_combined_threshold_uses_exact_fixed_half_scores() -> None:
    y_folds = [np.asarray([0, 1, 0, 1])] * 3
    japan = [np.asarray([0.1, 0.9, 0.2, 0.8])] * 3
    overseas = [np.asarray([0.3, 0.7, 0.4, 0.6])] * 3
    result = select_combined_threshold(y_folds, japan, overseas)
    assert result["selected_threshold"] in THRESHOLD_CANDIDATES
    combined = 0.5 * japan[0] + 0.5 * overseas[0]
    assert combined.tolist() == pytest.approx([0.2, 0.8, 0.3, 0.7])


def test_persisted_settings_are_new_version_and_normal_prediction_is_frozen() -> None:
    settings = load_model_settings()
    assert settings["model_version"] == "nikkei_dual_market_rolling_v1"
    assert settings["learning_years"] == 8
    assert settings["warmup_trading_days"] == 300
    assert settings["evaluation_years"] == 2
    assert settings["prediction_contexts"] == ["intraday", "after_close"]
    assert settings["combination"] == {"japan": 0.5, "overseas": 0.5, "method": "fixed_50_50"}
    assert settings["pce_used"] is False
    assert settings["normal_prediction_reselects_model"] is False
    assert settings["normal_prediction_reselects_weight"] is False
    assert settings["normal_prediction_reselects_threshold"] is False
    assert settings["normal_prediction_repeats_formal_evaluation"] is False
    intraday = load_context_evaluation("intraday")
    after_close = load_context_evaluation("after_close")
    assert intraday is not None and after_close is not None
    assert intraday["evaluation_not_used_for_selection"] is True
    assert after_close["evaluation_not_used_for_selection"] is True
    assert len(intraday["outer_folds"]) == len(after_close["outer_folds"]) == 8
    warmup_end = pd.Timestamp(settings["warmup_period"]["end"])
    first_training_start = pd.Timestamp(intraday["outer_folds"][0]["training_period"]["start"])
    assert warmup_end < first_training_start
    assert settings["warmup_period"]["included_in_training_or_evaluation"] is False


def test_all_candidates_use_the_same_inner_validation_dates() -> None:
    settings = load_model_settings()
    formal = load_context_evaluation("intraday")
    assert formal is not None
    first_outer = formal["outer_folds"][0]
    selection = first_outer["inner_selection"]
    expected = [
        (fold["validation_start"], fold["validation_end"])
        for fold in selection["japan"]["candidates"][0]["folds"]
    ]
    for side in ("japan", "overseas"):
        for candidate in selection[side]["candidates"]:
            observed = [(fold["validation_start"], fold["validation_end"]) for fold in candidate["folds"]]
            assert observed == expected


def test_normal_prediction_path_does_not_run_model_selection() -> None:
    source = (ROOT / "services" / "nikkei_dual_model.py").read_text(encoding="utf-8")
    normal_body = source.split("def predict_with_saved_dual_market_model", 1)[1]
    assert "select_operational_configuration(" not in normal_body
    assert "evaluate_formal_two_years(" not in normal_body


def test_ui_uses_six_class_first_and_keeps_formal_model_details_available() -> None:
    template = (ROOT / "templates" / "nikkei.html").read_text(encoding="utf-8")
    script = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
    warning = "本予測は参考情報であり、売買を推奨するものではありません。最終的な投資判断はご自身の責任でお願いします。"
    assert warning in template
    assert template.index("投資判断に関する注意") < template.index('id="result-panel"')
    assert "data-model-reevaluation" not in template

    public_renderer = script.split("const renderNikkeiPublicPrediction", 1)[1].split(
        "const renderSimulation", 1
    )[0]
    for text in (
        "5営業日先の6段階予測",
        "6段階トレンド予測レポート",
        "直近60営業日の株価推移",
        "直近2年間の株価推移",
        "予測精度",
        "今回のモデルが選択したファクター",
        "方向予測に使われた特徴量重要度",
        "採用構成と全特徴量の詳細を見る",
    ):
        assert text in public_renderer
    for text in (
        "日本側の上昇スコア",
        "米国・海外側の上昇スコア",
        "最終上昇スコア",
        "現在の判定ライン",
        "<strong>${escapeHtml(result.direction)}</strong>",
        "trend.symbol",
    ):
        assert text not in public_renderer

def test_no_intraday_interval_is_requested_in_new_direction_code() -> None:
    source = (ROOT / "services" / "nikkei_dual_market.py").read_text(encoding="utf-8")
    for interval in ('interval="1m"', 'interval="5m"', 'interval="15m"'):
        assert interval not in source
    assert 'interval="1d"' in source


def test_protected_original_sha256_matches_legacy_copy() -> None:
    for filename in ("nikkei_no2.py", "nikkei_kobetu_no1.py", "sim_research.py"):
        root_hash = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
        legacy_hash = hashlib.sha256((ROOT / "legacy_original" / filename).read_bytes()).hexdigest()
        assert root_hash == legacy_hash
