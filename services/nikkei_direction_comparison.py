from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .common import (
    CLASS_ORDER,
    OUTPUT_DIR,
    PREDICTION_HORIZON,
    PredictionData,
    ServiceError,
    _download_base_frame,
    _download_macro_frame,
    _six_probabilities,
    calc_rsi,
    channel_features,
    classify_change,
    overheat_score,
)


MODEL_VERSION = "nikkei_fixed_direction_v5"
EXTERNAL_ALIGNMENT_VERSION = "tokyo_close_safe_v2"
FEATURE_DEFINITION_VERSION = "market_regime_v2"
DIRECTION_TARGET_DEFINITION = "close_t_plus_5_gt_close_t"
DIRECTION_THRESHOLD = 0.50
PURGE_TRADING_DAYS = 5
WALK_FORWARD_FOLDS = 3
MODEL_TIE_TOLERANCE = 0.003
BOOTSTRAP_BLOCK_LENGTH = 10
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 42
RAW_LOOKBACK_YEARS = 8
RAW_WARMUP_DAYS = 90

# 変更前の保存済み結果で使われていた比較条件を固定する。
FIXED_COMPARISON_START = pd.Timestamp("2018-07-12")
FIXED_SELECTION_END = pd.Timestamp("2024-12-03")
FIXED_EVALUATION_START = pd.Timestamp("2024-12-04")
FIXED_EVALUATION_END = pd.Timestamp("2026-07-10")

STRONG_CRASH_MIN_PROBABILITY = 0.45
STRONG_CRASH_MIN_MARGIN = 0.15


BASE_FEATURES = [
    "open",
    "high",
    "low",
    "price",
    "volume",
    "return_1d",
    "return_5d",
    "volatility_5d",
    "volatility_20d",
    "volume_change",
    "rsi14",
    "ch_trend",
    "ch_upper",
    "ch_lower",
    "ch_pos",
    "overheat_score",
]

CURRENT_BASELINE_FEATURES = [
    "open",
    "high",
    "low",
    "price",
    "return_1d",
    "return_5d",
    "volume",
    "volume_change",
    "volatility_5d",
    "volatility_20d",
    "overheat_score",
    "rsi14",
    "ch_trend",
    "ch_upper",
    "ch_lower",
    "ch_pos",
    "dow",
    "dow_ret",
    "usdjpy",
    "usdjpy_ret",
    "oil",
    "oil_ret",
]

MARKET_REGIME_FEATURES = [
    "ma20_gap",
    "ma50_gap",
    "ma100_gap",
    "ma200_gap",
    "ma20_slope5",
    "ma50_slope5",
    "ma100_slope5",
    "ma20_above_ma100",
    "return_2d",
    "return_3d",
    "return_10d",
    "return_20d",
    "intraday_range",
    "opening_gap",
    "atr14_ratio",
    "volatility_10d",
    "volatility_60d",
    "volatility_5_60_ratio",
    "volatility_20d_change5",
    "distance_high20",
    "distance_high60",
    "distance_high252",
    "distance_low20",
    "distance_low60",
]

EXTERNAL_LABELS = OrderedDict(
    [
        ("spx", "S&P 500"),
        ("dow", "NYダウ"),
        ("nasdaq", "NASDAQ"),
        ("vix", "VIX"),
        ("sox", "SOX"),
        ("nvda", "NVIDIA"),
        ("usdjpy", "ドル円"),
        ("oil", "原油"),
        ("gold", "金"),
        ("btc", "Bitcoin"),
    ]
)

FACTOR_LABELS = OrderedDict(
    [
        ("nikkei_price", "日経平均の価格・リターン"),
        ("volume", "出来高"),
        ("rsi_channel", "RSI・価格チャネル"),
        ("trend_regime", "トレンド環境"),
        ("return_range", "リターン・値幅"),
        ("volatility_regime", "ボラティリティ環境"),
        ("price_position", "価格位置"),
        ("external_relationship", "外部市場との相関"),
        *EXTERNAL_LABELS.items(),
    ]
)

CATBOOST_SETTINGS = {
    "current": {
        "name": "current",
        "label": "現行設定",
        "iterations": 130,
        "depth": 5,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
    },
    "enhanced_all": {
        "name": "enhanced_all",
        "label": "相場環境・標準設定",
        "iterations": 160,
        "depth": 4,
        "learning_rate": 0.04,
        "l2_leaf_reg": 5.0,
    },
    "enhanced_selected": {
        "name": "enhanced_selected",
        "label": "相場環境・正則化設定",
        "iterations": 190,
        "depth": 4,
        "learning_rate": 0.03,
        "l2_leaf_reg": 8.0,
    },
    "importance": {
        "name": "importance",
        "label": "学習内重要度算出",
        "iterations": 90,
        "depth": 4,
        "learning_rate": 0.05,
        "l2_leaf_reg": 6.0,
    },
}

CANDIDATE_DEFINITIONS = [
    {
        "id": "current_catboost",
        "label": "現行CatBoost方向モデル",
        "kind": "catboost",
        "features": "current",
        "setting": "current",
        "class_balance": "Balanced",
        "simplicity": 3,
    },
    {
        "id": "enhanced_catboost_all",
        "label": "相場環境CatBoost（全特徴量）",
        "kind": "catboost",
        "features": "all",
        "setting": "enhanced_all",
        "class_balance": "None",
        "simplicity": 4,
    },
    {
        "id": "enhanced_catboost_selected",
        "label": "相場環境CatBoost（学習内選別）",
        "kind": "catboost",
        "features": "selected",
        "setting": "enhanced_selected",
        "class_balance": "None",
        "simplicity": 4,
    },
    {
        "id": "logistic_selected",
        "label": "ロジスティック回帰（学習内選別）",
        "kind": "logistic",
        "features": "selected",
        "class_balance": "None",
        "simplicity": 1,
    },
    {
        "id": "extra_trees_selected",
        "label": "Extra Trees（学習内選別）",
        "kind": "extra_trees",
        "features": "selected",
        "class_balance": "None",
        "simplicity": 2,
    },
    {
        "id": "simple_average_ensemble",
        "label": "3モデル単純平均",
        "kind": "ensemble",
        "features": "selected",
        "components": ["enhanced_catboost_selected", "logistic_selected", "extra_trees_selected"],
        "class_balance": "component settings",
        "simplicity": 5,
    },
]


def _now_jst() -> str:
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")


def _date_text(value: pd.Timestamp | datetime) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def make_direction_target(price: pd.Series, horizon: int = PREDICTION_HORIZON) -> tuple[pd.Series, pd.Series]:
    """5取引日後が高い=1、低い=0、同値または未来価格なし=欠損。"""
    values = price.astype(float)
    future = values.shift(-horizon)
    target = pd.Series(np.nan, index=values.index, dtype=float)
    target.loc[future.gt(values)] = 1.0
    target.loc[future.lt(values)] = 0.0
    return target, future


def _factor_definitions(available_external: Iterable[str]) -> OrderedDict[str, list[str]]:
    definitions: OrderedDict[str, list[str]] = OrderedDict(
        [
            ("nikkei_price", ["open", "high", "low", "price", "return_1d", "return_5d"]),
            ("volume", ["volume", "volume_change"]),
            ("rsi_channel", ["rsi14", "ch_trend", "ch_upper", "ch_lower", "ch_pos", "overheat_score"]),
            (
                "trend_regime",
                [
                    "ma20_gap",
                    "ma50_gap",
                    "ma100_gap",
                    "ma200_gap",
                    "ma20_slope5",
                    "ma50_slope5",
                    "ma100_slope5",
                    "ma20_above_ma100",
                ],
            ),
            (
                "return_range",
                [
                    "return_2d",
                    "return_3d",
                    "return_10d",
                    "return_20d",
                    "intraday_range",
                    "opening_gap",
                    "atr14_ratio",
                ],
            ),
            (
                "volatility_regime",
                [
                    "volatility_5d",
                    "volatility_10d",
                    "volatility_20d",
                    "volatility_60d",
                    "volatility_5_60_ratio",
                    "volatility_20d_change5",
                ],
            ),
            (
                "price_position",
                [
                    "distance_high20",
                    "distance_high60",
                    "distance_high252",
                    "distance_low20",
                    "distance_low60",
                ],
            ),
        ]
    )
    relationship_columns: list[str] = []
    for external in ("spx", "nasdaq", "usdjpy"):
        if external in available_external:
            relationship_columns.append(f"corr20_nikkei_{external}")
    if relationship_columns:
        definitions["external_relationship"] = relationship_columns
    for external in available_external:
        definitions[external] = [external, f"{external}_ret", f"{external}_ret5", f"{external}_ret20"]
    return definitions


def prepare_nikkei_direction_data(now: datetime | None = None) -> tuple[PredictionData, dict[str, Any]]:
    """日経平均専用に、東京終値時点で利用可能な特徴量を作る。"""
    started = time.perf_counter()
    end = now or datetime.now()
    requested_start = end - timedelta(days=365 * RAW_LOOKBACK_YEARS + RAW_WARMUP_DAYS)
    fetch_started = time.perf_counter()
    base = _download_base_frame("^N225", requested_start, end)
    macro, warnings = _download_macro_frame(base.index, requested_start, end)
    fetch_seconds = time.perf_counter() - fetch_started

    # 同じ暦日ラベルの米国・商品・為替・暗号資産の日次終値は、東京市場の
    # 当日終値時点では未確定になり得る。全系列を日経平均の1取引日遅らせる。
    available_external = [column for column in EXTERNAL_LABELS if column in macro.columns]
    macro = macro.loc[:, available_external].shift(1)
    alignment: dict[str, Any] = {}
    for column in EXTERNAL_LABELS:
        if column in available_external:
            alignment[column] = {
                "label": EXTERNAL_LABELS[column],
                "method": "source daily close shifted by one Nikkei trading session, then past-only forward fill",
                "lag_trading_sessions": 1,
                "available_at_prediction_time": True,
                "missing_handling": "training-fold median for models requiring imputation; no backward fill",
            }
        else:
            alignment[column] = {
                "label": EXTERNAL_LABELS[column],
                "method": "excluded because source data was unavailable",
                "lag_trading_sessions": None,
                "available_at_prediction_time": False,
                "missing_handling": "factor excluded",
            }
    alignment["pce"] = {
        "label": "PCE",
        "method": "excluded",
        "lag_trading_sessions": None,
        "available_at_prediction_time": False,
        "missing_handling": "factor excluded",
        "reason": "実際の公表日と当時利用可能だったビンテージを安全に再現できないため",
    }
    warnings.append("PCEは実際の公表日・当時ビンテージの安全な対応ができないため、日経平均方向モデルから除外しました。")

    frame = base.join(macro, how="left")
    if available_external:
        # shift後の欠損だけを過去値で補完する。bfillは使わない。
        frame[available_external] = frame[available_external].ffill()

    feature_started = time.perf_counter()
    price = frame["price"].astype(float)
    frame["return_1d"] = price.pct_change(fill_method=None) * 100
    frame["return_5d"] = price.pct_change(5, fill_method=None) * 100
    frame["volume_change"] = frame["volume"].pct_change(fill_method=None) * 100
    frame["rsi14"] = calc_rsi(price, 14)
    frame["volatility_5d"] = frame["return_1d"].rolling(5, min_periods=5).std()
    frame["volatility_20d"] = frame["return_1d"].rolling(20, min_periods=20).std()
    frame = frame.join(channel_features(price))
    frame["overheat_score"] = overheat_score(frame)

    moving_averages: dict[int, pd.Series] = {}
    for window in (20, 50, 100, 200):
        moving_averages[window] = price.rolling(window, min_periods=window).mean()
        frame[f"ma{window}_gap"] = (price / moving_averages[window] - 1.0) * 100
    for window in (20, 50, 100):
        frame[f"ma{window}_slope5"] = moving_averages[window].pct_change(5, fill_method=None) * 100
    ma_relation = (moving_averages[20] >= moving_averages[100]).astype(float)
    frame["ma20_above_ma100"] = ma_relation.where(moving_averages[20].notna() & moving_averages[100].notna())

    for window in (2, 3, 10, 20):
        frame[f"return_{window}d"] = price.pct_change(window, fill_method=None) * 100
    previous_close = price.shift(1)
    frame["intraday_range"] = (frame["high"] - frame["low"]) / price.replace(0, np.nan) * 100
    frame["opening_gap"] = (frame["open"] / previous_close - 1.0) * 100
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr14_ratio"] = true_range.rolling(14, min_periods=14).mean() / price * 100
    frame["volatility_10d"] = frame["return_1d"].rolling(10, min_periods=10).std()
    frame["volatility_60d"] = frame["return_1d"].rolling(60, min_periods=60).std()
    frame["volatility_5_60_ratio"] = frame["volatility_5d"] / frame["volatility_60d"].replace(0, np.nan)
    frame["volatility_20d_change5"] = frame["volatility_20d"].pct_change(5, fill_method=None) * 100
    for window in (20, 60, 252):
        rolling_high = price.rolling(window, min_periods=window).max()
        frame[f"distance_high{window}"] = (price / rolling_high - 1.0) * 100
    for window in (20, 60):
        rolling_low = price.rolling(window, min_periods=window).min()
        frame[f"distance_low{window}"] = (price / rolling_low - 1.0) * 100

    external_features: list[str] = []
    for column in available_external:
        for window, suffix in ((1, "ret"), (5, "ret5"), (20, "ret20")):
            name = f"{column}_{suffix}"
            frame[name] = frame[column].pct_change(window, fill_method=None) * 100
            external_features.append(name)
        external_features.insert(len(external_features) - 3, column)
    correlation_features: list[str] = []
    for column in ("spx", "nasdaq", "usdjpy"):
        if f"{column}_ret" in frame:
            name = f"corr20_nikkei_{column}"
            frame[name] = frame["return_1d"].rolling(20, min_periods=20).corr(frame[f"{column}_ret"])
            correlation_features.append(name)

    direction, future_price = make_direction_target(price)
    frame["target_change"] = np.where(
        future_price.notna(),
        (future_price / price - 1.0) * 100,
        np.nan,
    )
    frame["target_direction"] = direction
    tie_mask = future_price.notna() & future_price.eq(price)

    all_features = list(dict.fromkeys(BASE_FEATURES + MARKET_REGIME_FEATURES + external_features + correlation_features))
    frame = frame.replace([np.inf, -np.inf], np.nan)
    comparison = frame.loc[frame.index >= FIXED_COMPARISON_START].copy()
    training = comparison.loc[:, all_features + ["target_change", "target_direction"]].dropna(
        subset=["target_change", "target_direction"]
    )
    inference_date = frame.index[-1]
    if frame.loc[inference_date, ["open", "high", "low", "price"]].isna().any():
        raise ServiceError("最新の日経平均特徴量を生成できませんでした。", "市場データを再取得してください。")
    inference = frame.loc[[inference_date], all_features]
    training = training.drop(index=inference_date, errors="ignore")

    selection = training.loc[
        (training.index >= FIXED_COMPARISON_START) & (training.index <= FIXED_SELECTION_END)
    ]
    holdout = training.loc[
        (training.index >= FIXED_EVALUATION_START) & (training.index <= FIXED_EVALUATION_END)
    ]
    if len(selection) < 1_000 or len(holdout) < 250:
        raise ServiceError(
            "固定した時系列検証期間のデータが不足しています。",
            "市場データの取得状態を確認して再分析してください。",
        )

    complete_features = frame.loc[:, all_features].dropna()
    feature_valid_start = complete_features.index[0] if not complete_features.empty else None
    factor_definitions = _factor_definitions(available_external)
    feature_seconds = time.perf_counter() - feature_started
    fetched_at = _now_jst()
    data = PredictionData(frame, all_features, training, inference, warnings, fetched_at)
    metadata = {
        "fixed_data_start": _date_text(FIXED_COMPARISON_START),
        "fixed_data_end": _date_text(inference_date),
        "raw_requested_start": _date_text(requested_start),
        "raw_actual_start": _date_text(base.index[0]),
        "raw_actual_end": _date_text(base.index[-1]),
        "feature_valid_start": _date_text(feature_valid_start) if feature_valid_start is not None else None,
        "comparison_rows": int(len(training.loc[training.index <= FIXED_EVALUATION_END])),
        "selection_period": {
            "start": _date_text(selection.index[0]),
            "end": _date_text(selection.index[-1]),
            "samples": int(len(selection)),
        },
        "fixed_evaluation_period": {
            "start": _date_text(holdout.index[0]),
            "end": _date_text(holdout.index[-1]),
            "samples": int(len(holdout)),
        },
        "prediction_horizon": PREDICTION_HORIZON,
        "direction_threshold": DIRECTION_THRESHOLD,
        "direction_target_definition": DIRECTION_TARGET_DEFINITION,
        "purge_trading_days": PURGE_TRADING_DAYS,
        "equal_close_rows_excluded": int(tie_mask.loc[tie_mask.index >= FIXED_COMPARISON_START].sum()),
        "base_feature_count": int(len(BASE_FEATURES) + len(available_external) * 2),
        "enhanced_feature_count": int(len(all_features)),
        "market_regime_features": [feature for feature in MARKET_REGIME_FEATURES if feature in all_features]
        + correlation_features
        + [feature for feature in external_features if feature.endswith(("_ret5", "_ret20"))],
        "available_external_factors": available_external,
        "external_alignment": alignment,
        "external_alignment_version": EXTERNAL_ALIGNMENT_VERSION,
        "feature_definition_version": FEATURE_DEFINITION_VERSION,
        "pce_used": False,
        "pce_exclusion_reason": alignment["pce"]["reason"],
        "missing_handling": "no backward fill; external values use past-only forward fill; model imputation is fit on each training fold only",
        "factor_definitions": {key: [column for column in value if column in all_features] for key, value in factor_definitions.items()},
        "data_fetch_seconds": float(fetch_seconds),
        "feature_creation_seconds": float(feature_seconds),
        "preparation_seconds": float(time.perf_counter() - started),
    }
    return data, metadata


def fixed_period_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection = frame.loc[(frame.index >= FIXED_COMPARISON_START) & (frame.index <= FIXED_SELECTION_END)].copy()
    holdout = frame.loc[(frame.index >= FIXED_EVALUATION_START) & (frame.index <= FIXED_EVALUATION_END)].copy()
    if selection.empty or holdout.empty or selection.index.max() >= holdout.index.min():
        raise ServiceError("モデル選択期間と固定評価期間を分離できませんでした。")
    return selection, holdout


def expanding_purged_walk_forward_splits(
    selection: pd.DataFrame,
    n_splits: int = WALK_FORWARD_FOLDS,
    purge_days: int = PURGE_TRADING_DAYS,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    if n_splits != 3:
        raise ValueError("this comparison uses exactly three fixed folds")
    if purge_days < PREDICTION_HORIZON:
        raise ValueError("purge must be at least the prediction horizon")
    minimum_train = max(240, int(len(selection) * 0.52))
    validation_size = (len(selection) - minimum_train) // n_splits
    if validation_size < 45:
        raise ServiceError("ウォークフォワード検証に必要な期間が不足しています。")
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for fold_index in range(n_splits):
        boundary = minimum_train + validation_size * fold_index
        validation_end = len(selection) if fold_index == n_splits - 1 else boundary + validation_size
        unpurged_train = selection.iloc[:boundary]
        if len(unpurged_train) <= purge_days:
            raise ServiceError("パージ後の学習期間が不足しています。")
        train = unpurged_train.iloc[:-purge_days].copy()
        validation = selection.iloc[boundary:validation_end].copy()
        if train.index.max() >= validation.index.min():
            raise ServiceError("ウォークフォワードの時系列順序が不正です。")
        folds.append((train, validation))
    return folds


def majority_baseline_predictions(train_direction: np.ndarray, validation_size: int) -> np.ndarray:
    values = np.asarray(train_direction, dtype=int)
    majority = int(np.argmax(np.bincount(values, minlength=2)))
    return np.full(validation_size, majority, dtype=int)


def momentum_baseline_predictions(validation: pd.DataFrame) -> np.ndarray:
    """正解ラベルではなく、予測時点までの5日リターンだけを使う。"""
    return (validation["return_5d"].to_numpy(dtype=float) >= 0.0).astype(int)


def direction_from_probability(up_probability: np.ndarray | float) -> np.ndarray:
    return (np.asarray(up_probability, dtype=float) >= DIRECTION_THRESHOLD).astype(int)


def simple_average_probabilities(probabilities: Iterable[np.ndarray]) -> np.ndarray:
    rows = [np.asarray(row, dtype=float) for row in probabilities]
    if not rows:
        raise ValueError("at least one probability array is required")
    lengths = {len(row) for row in rows}
    if len(lengths) != 1:
        raise ValueError("probability arrays must have equal length")
    return np.mean(np.vstack(rows), axis=0)


def _training_medians(frame: pd.DataFrame, features: list[str]) -> pd.Series:
    medians = frame[features].median(axis=0, skipna=True).fillna(0.0)
    return medians.astype(float)


def _imputed_arrays(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    medians = _training_medians(train, features)
    train_values = train[features].fillna(medians).to_numpy(dtype=float)
    validation_values = validation[features].fillna(medians).to_numpy(dtype=float)
    return train_values, validation_values, medians


def _new_catboost(setting_name: str, task: str, class_balance: str):
    from catboost import CatBoostClassifier

    setting = CATBOOST_SETTINGS[setting_name]
    parameters: dict[str, Any] = {
        "iterations": int(setting["iterations"]),
        "depth": int(setting["depth"]),
        "learning_rate": float(setting["learning_rate"]),
        "l2_leaf_reg": float(setting["l2_leaf_reg"]),
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
        "loss_function": "Logloss" if task == "direction" else "MultiClass",
    }
    if class_balance == "Balanced":
        parameters["auto_class_weights"] = "Balanced"
    return CatBoostClassifier(**parameters)


def _class_one_probability(model: Any, probability_matrix: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probability_matrix, dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    if 1 not in classes:
        return np.zeros(len(probabilities), dtype=float)
    return probabilities[:, int(np.where(classes == 1)[0][0])]


def _normalize_importance(features: list[str], values: np.ndarray) -> dict[str, float]:
    raw = np.asarray(values, dtype=float)
    total = float(np.abs(raw).sum())
    if total <= 0:
        return {feature: 0.0 for feature in features}
    normalized = np.abs(raw) / total * 100.0
    return {feature: float(value) for feature, value in zip(features, normalized)}


def _fit_component(
    definition: dict[str, Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    y_train = train["target_direction"].to_numpy(dtype=int)
    if len(np.unique(y_train)) < 2:
        constant = float(y_train[0])
        return np.full(len(validation), constant), {feature: 0.0 for feature in features}, {"fallback": "single training class"}

    x_train, x_validation, medians = _imputed_arrays(train, validation, features)
    kind = definition["kind"]
    metadata: dict[str, Any] = {
        "imputation_fit_start": _date_text(train.index[0]),
        "imputation_fit_end": _date_text(train.index[-1]),
        "imputation": "training median; all-missing training columns use 0",
    }
    if kind == "catboost":
        if definition["id"] == "current_catboost":
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_validation = scaler.transform(x_validation)
            metadata["standardization"] = "fit on training fold only"
        model = _new_catboost(definition["setting"], "direction", definition["class_balance"])
        model.fit(x_train, y_train)
        probability = _class_one_probability(model, model.predict_proba(x_validation))
        importance = _normalize_importance(features, np.asarray(model.get_feature_importance(), dtype=float))
    elif kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_validation = scaler.transform(x_validation)
        model = LogisticRegression(
            C=0.5,
            class_weight="balanced" if definition["class_balance"] == "balanced" else None,
            max_iter=1_000,
            random_state=42,
            solver="liblinear",
        )
        model.fit(x_train, y_train)
        probability = _class_one_probability(model, model.predict_proba(x_validation))
        importance = _normalize_importance(features, model.coef_[0])
        metadata["standardization"] = "fit on training fold only"
    elif kind == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        model = ExtraTreesClassifier(
            n_estimators=240,
            max_depth=7,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced" if definition["class_balance"] == "balanced" else None,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        probability = _class_one_probability(model, model.predict_proba(x_validation))
        importance = _normalize_importance(features, model.feature_importances_)
    else:
        raise ValueError(f"unsupported component kind: {kind}")
    metadata["median_count"] = int(len(medians))
    return np.asarray(probability, dtype=float), importance, metadata


def _direction_metrics(
    y_true: np.ndarray,
    up_probability: np.ndarray,
    majority_prediction: np.ndarray,
    momentum_prediction: np.ndarray,
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score

    prediction = direction_from_probability(up_probability)
    majority_accuracy = float(accuracy_score(y_true, majority_prediction))
    momentum_accuracy = float(accuracy_score(y_true, momentum_prediction))
    best_baseline_accuracy = max(majority_accuracy, momentum_accuracy)
    best_baseline_name = "多数派基準モデル" if majority_accuracy >= momentum_accuracy else "直近方向継続モデル"
    accuracy = float(accuracy_score(y_true, prediction))
    return {
        "direction_accuracy": accuracy,
        "direction_balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "direction_macro_f1": float(f1_score(y_true, prediction, labels=[0, 1], average="macro", zero_division=0)),
        "up_recall": float(recall_score(y_true, prediction, labels=[1], average="macro", zero_division=0)),
        "down_recall": float(recall_score(y_true, prediction, labels=[0], average="macro", zero_division=0)),
        "predicted_up": int(np.sum(prediction == 1)),
        "predicted_down": int(np.sum(prediction == 0)),
        "actual_up": int(np.sum(y_true == 1)),
        "actual_down": int(np.sum(y_true == 0)),
        "correct_predictions": int(np.sum(prediction == y_true)),
        "validation_samples": int(len(y_true)),
        "majority_baseline_accuracy": majority_accuracy,
        "momentum_baseline_accuracy": momentum_accuracy,
        "best_baseline_accuracy": best_baseline_accuracy,
        "best_baseline_name": best_baseline_name,
        "baseline_gap": float(accuracy - best_baseline_accuracy),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).astype(int).tolist(),
    }


def _training_only_feature_selection(
    train: pd.DataFrame,
    all_features: list[str],
    factor_definitions: dict[str, list[str]],
) -> dict[str, Any]:
    y_train = train["target_direction"].to_numpy(dtype=int)
    x_train = train[all_features].fillna(_training_medians(train, all_features)).to_numpy(dtype=float)
    if len(np.unique(y_train)) < 2:
        importance = {feature: 0.0 for feature in all_features}
    else:
        model = _new_catboost("importance", "direction", "Balanced")
        model.fit(x_train, y_train)
        importance = _normalize_importance(all_features, np.asarray(model.get_feature_importance(), dtype=float))

    group_scores: list[tuple[str, float]] = []
    for factor_id, columns in factor_definitions.items():
        present = [column for column in columns if column in importance]
        if present:
            group_scores.append((factor_id, float(np.mean([importance[column] for column in present]))))
    group_scores.sort(key=lambda item: (-item[1], item[0]))
    mandatory = [factor for factor in ("nikkei_price", "trend_regime", "return_range") if factor in factor_definitions]
    selected_factors = list(dict.fromkeys(mandatory + [factor for factor, _ in group_scores[:8]]))
    allowed = {
        feature
        for factor in selected_factors
        for feature in factor_definitions.get(factor, [])
        if feature in all_features
    }
    ranked = sorted(allowed, key=lambda feature: (-importance[feature], feature))
    selected_features = ranked[:30]
    if len(selected_features) < min(15, len(all_features)):
        all_ranked = sorted(all_features, key=lambda feature: (-importance[feature], feature))
        selected_features = list(dict.fromkeys(selected_features + all_ranked))[: min(30, len(all_features))]
    return {
        "selected_features": selected_features,
        "selected_factors": selected_factors,
        "importance": importance,
        "factor_scores": [{"factor_id": factor, "score": score} for factor, score in group_scores],
        "fit_period": {"start": _date_text(train.index[0]), "end": _date_text(train.index[-1])},
        "used_validation_data": False,
    }


def _six_class_evaluation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score

    y_train = np.array([CLASS_ORDER.index(classify_change(float(value))) for value in train["target_change"]], dtype=int)
    y_validation = np.array(
        [CLASS_ORDER.index(classify_change(float(value))) for value in validation["target_change"]], dtype=int
    )
    x_train, x_validation, _ = _imputed_arrays(train, validation, features)
    if len(np.unique(y_train)) < 2:
        prediction = np.full(len(validation), int(y_train[0]), dtype=int)
    else:
        model = _new_catboost("enhanced_selected", "six", "Balanced")
        model.fit(x_train, y_train)
        prediction = model.predict(x_validation).astype(int).reshape(-1)
    return {
        "six_class_accuracy": float(accuracy_score(y_validation, prediction)),
        "six_class_macro_f1": float(
            f1_score(y_validation, prediction, labels=list(range(len(CLASS_ORDER))), average="macro", zero_division=0)
        ),
        "six_class_distribution": {
            label: int(np.sum(y_validation == class_id)) for class_id, label in enumerate(CLASS_ORDER)
        },
    }


def _candidate_summary(folds: list[dict[str, Any]]) -> dict[str, Any]:
    accuracy = np.array([fold["direction_accuracy"] for fold in folds], dtype=float)
    balance = np.array([fold["direction_balanced_accuracy"] for fold in folds], dtype=float)
    macro_f1 = np.array([fold["direction_macro_f1"] for fold in folds], dtype=float)
    gaps = np.array([fold["baseline_gap"] for fold in folds], dtype=float)
    return {
        "direction_accuracy_mean": float(accuracy.mean()),
        "direction_accuracy_std": float(accuracy.std()),
        "direction_accuracy_worst_fold": float(accuracy.min()),
        "direction_balanced_accuracy_mean": float(balance.mean()),
        "direction_macro_f1_mean": float(macro_f1.mean()),
        "best_baseline_gap_mean": float(gaps.mean()),
        "validation_samples_each_fold": [int(fold["validation_samples"]) for fold in folds],
    }


def _candidate_rank(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    summary = candidate["summary"]
    return (
        float(summary["direction_accuracy_mean"]),
        float(summary["direction_balanced_accuracy_mean"]),
        float(summary["best_baseline_gap_mean"]),
        -float(summary["direction_accuracy_std"]),
        float(summary["direction_accuracy_worst_fold"]),
        float(summary["direction_macro_f1_mean"]),
    )


def choose_direction_candidate(candidate_results: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    current = next(candidate for candidate in candidate_results if candidate["id"] == "current_catboost")
    best_new = max((candidate for candidate in candidate_results if candidate["id"] != "current_catboost"), key=_candidate_rank)
    current_accuracy = float(current["summary"]["direction_accuracy_mean"])
    new_accuracy = float(best_new["summary"]["direction_accuracy_mean"])
    gain = new_accuracy - current_accuracy
    if gain < 0:
        return current, "新候補の平均方向予測精度が現行モデルを上回らなかったため、現行モデルを採用しました。"
    if gain > MODEL_TIE_TOLERANCE:
        return best_new, f"平均方向予測精度が現行モデルを{gain * 100:.2f}ポイント上回ったため採用しました。"

    # 同程度では、指定された副指標の順序で比較する。精度が下回る新候補は採らない。
    secondary_new = _candidate_rank(best_new)[1:]
    secondary_current = _candidate_rank(current)[1:]
    if secondary_new > secondary_current:
        return best_new, (
            f"平均方向予測精度差が同程度（{gain * 100:.2f}ポイント）で、"
            "バランス精度・基準差・安定性の優先順で上回ったため採用しました。"
        )
    return current, "平均方向予測精度差が同程度で副指標の改善も確認できなかったため、現行モデルを採用しました。"


def compare_direction_models(
    selection: pd.DataFrame,
    all_features: list[str],
    factor_definitions: dict[str, list[str]],
) -> dict[str, Any]:
    """固定評価期間を受け取らず、モデル選択期間だけで6候補を比較する。"""
    started = time.perf_counter()
    folds = expanding_purged_walk_forward_splits(selection)
    definitions = {definition["id"]: definition for definition in CANDIDATE_DEFINITIONS}
    fold_results: dict[str, list[dict[str, Any]]] = {candidate_id: [] for candidate_id in definitions}
    screened_ids = ("enhanced_catboost_selected", "logistic_selected", "extra_trees_selected")
    balance_modes = {
        "enhanced_catboost_selected": ("None", "Balanced"),
        "logistic_selected": ("None", "balanced"),
        "extra_trees_selected": ("None", "balanced"),
    }
    balance_fold_results: dict[str, dict[str, list[dict[str, Any]]]] = {
        candidate_id: {mode: [] for mode in balance_modes[candidate_id]} for candidate_id in screened_ids
    }
    balance_probabilities: dict[str, dict[str, list[np.ndarray]]] = {
        candidate_id: {mode: [] for mode in balance_modes[candidate_id]} for candidate_id in screened_ids
    }
    feature_selections: list[dict[str, Any]] = []
    auxiliary_six_folds: list[dict[str, Any]] = []
    validation_dates: list[list[str]] = []
    fold_contexts: list[dict[str, Any]] = []

    current_features = [feature for feature in CURRENT_BASELINE_FEATURES if feature in all_features]
    for fold_number, (train, validation) in enumerate(folds, start=1):
        selection_result = _training_only_feature_selection(train, all_features, factor_definitions)
        fold_selected = selection_result["selected_features"]
        feature_selections.append(selection_result)
        validation_dates.append([_date_text(index) for index in validation.index])
        y_train = train["target_direction"].to_numpy(dtype=int)
        y_validation = validation["target_direction"].to_numpy(dtype=int)
        majority = majority_baseline_predictions(y_train, len(validation))
        momentum = momentum_baseline_predictions(validation)
        for candidate_id in (
            "current_catboost",
            "enhanced_catboost_all",
            "enhanced_catboost_selected",
            "logistic_selected",
            "extra_trees_selected",
        ):
            definition = definitions[candidate_id]
            feature_mode = definition["features"]
            features = current_features if feature_mode == "current" else all_features if feature_mode == "all" else fold_selected
            modes = balance_modes[candidate_id] if candidate_id in screened_ids else (definition["class_balance"],)
            for mode in modes:
                effective_definition = {**definition, "class_balance": mode}
                probability, importance, preprocessing = _fit_component(
                    effective_definition, train, validation, features
                )
                metrics = _direction_metrics(y_validation, probability, majority, momentum)
                record = {
                    "fold": fold_number,
                    "train_start": _date_text(train.index[0]),
                    "train_end": _date_text(train.index[-1]),
                    "validation_start": _date_text(validation.index[0]),
                    "validation_end": _date_text(validation.index[-1]),
                    "training_samples": int(len(train)),
                    "purge_trading_days": PURGE_TRADING_DAYS,
                    "feature_count": int(len(features)),
                    "features": features,
                    "feature_importance": importance,
                    "preprocessing": preprocessing,
                    "class_balance": mode,
                    "direction_distribution": {
                        "up": metrics["actual_up"],
                        "non_up": metrics["actual_down"],
                    },
                    **metrics,
                }
                if candidate_id in screened_ids:
                    balance_fold_results[candidate_id][mode].append(record)
                    balance_probabilities[candidate_id][mode].append(probability)
                else:
                    fold_results[candidate_id].append(record)

        fold_contexts.append(
            {
                "fold": fold_number,
                "train": train,
                "validation": validation,
                "selected_features": fold_selected,
                "y_validation": y_validation,
                "majority": majority,
                "momentum": momentum,
            }
        )
        six_metrics = _six_class_evaluation(train, validation, fold_selected)
        auxiliary_six_folds.append(
            {
                "fold": fold_number,
                "train_start": _date_text(train.index[0]),
                "train_end": _date_text(train.index[-1]),
                "validation_start": _date_text(validation.index[0]),
                "validation_end": _date_text(validation.index[-1]),
                "training_samples": int(len(train)),
                "validation_samples": int(len(validation)),
                **six_metrics,
            }
        )

    selected_balances: dict[str, str] = {}
    class_balance_screening: list[dict[str, Any]] = []
    for candidate_id in screened_ids:
        none_mode, balanced_mode = balance_modes[candidate_id]
        none_summary = _candidate_summary(balance_fold_results[candidate_id][none_mode])
        balanced_summary = _candidate_summary(balance_fold_results[candidate_id][balanced_mode])
        direction_gain = (
            balanced_summary["direction_accuracy_mean"] - none_summary["direction_accuracy_mean"]
        )
        balance_not_degraded = (
            balanced_summary["direction_balanced_accuracy_mean"]
            >= none_summary["direction_balanced_accuracy_mean"] - 0.005
        )
        selected_mode = (
            balanced_mode
            if direction_gain >= MODEL_TIE_TOLERANCE and balance_not_degraded
            else none_mode
        )
        selected_balances[candidate_id] = selected_mode
        fold_results[candidate_id] = balance_fold_results[candidate_id][selected_mode]
        class_balance_screening.append(
            {
                "candidate_id": candidate_id,
                "options": [
                    {"class_balance": none_mode, "summary": none_summary},
                    {"class_balance": balanced_mode, "summary": balanced_summary},
                ],
                "selected": selected_mode,
                "selection_rule": "balanced is selected only when mean direction accuracy improves by at least the tie tolerance without material balanced-accuracy degradation",
                "used_fixed_evaluation": False,
            }
        )

    ensemble_definition = definitions["simple_average_ensemble"]
    for fold_index, context in enumerate(fold_contexts):
        ensemble_probability = simple_average_probabilities(
            [
                balance_probabilities[component][selected_balances[component]][fold_index]
                for component in ensemble_definition["components"]
            ]
        )
        ensemble_metrics = _direction_metrics(
            context["y_validation"],
            ensemble_probability,
            context["majority"],
            context["momentum"],
        )
        train = context["train"]
        validation = context["validation"]
        fold_selected = context["selected_features"]
        fold_results["simple_average_ensemble"].append(
            {
                "fold": context["fold"],
                "train_start": _date_text(train.index[0]),
                "train_end": _date_text(train.index[-1]),
                "validation_start": _date_text(validation.index[0]),
                "validation_end": _date_text(validation.index[-1]),
                "training_samples": int(len(train)),
                "purge_trading_days": PURGE_TRADING_DAYS,
                "feature_count": int(len(fold_selected)),
                "features": fold_selected,
                "ensemble_components": ensemble_definition["components"],
                "component_class_balances": {
                    component: selected_balances[component]
                    for component in ensemble_definition["components"]
                },
                "ensemble_method": "unweighted arithmetic mean of up probabilities",
                "direction_distribution": {
                    "up": ensemble_metrics["actual_up"],
                    "non_up": ensemble_metrics["actual_down"],
                },
                **ensemble_metrics,
            }
        )

    feature_counts = Counter(feature for result in feature_selections for feature in result["selected_features"])
    mean_importance: dict[str, float] = {}
    for feature in all_features:
        mean_importance[feature] = float(np.mean([result["importance"].get(feature, 0.0) for result in feature_selections]))
    stable_ranked = sorted(
        (feature for feature in all_features if feature_counts[feature] >= 2),
        key=lambda feature: (-feature_counts[feature], -mean_importance[feature], feature),
    )
    stable_features = stable_ranked[:30]
    if len(stable_features) < min(15, len(all_features)):
        fallback_ranked = sorted(all_features, key=lambda feature: (-mean_importance[feature], feature))
        stable_features = list(dict.fromkeys(stable_features + fallback_ranked))[: min(30, len(all_features))]

    candidate_results: list[dict[str, Any]] = []
    for definition in CANDIDATE_DEFINITIONS:
        folds_for_candidate = fold_results[definition["id"]]
        effective_balance = selected_balances.get(definition["id"], definition["class_balance"])
        candidate_results.append(
            {
                **definition,
                "class_balance": effective_balance,
                "folds": folds_for_candidate,
                "summary": _candidate_summary(folds_for_candidate),
            }
        )
    selected_candidate, adoption_reason = choose_direction_candidate(candidate_results)
    if selected_candidate["features"] == "current":
        selected_features = current_features
    elif selected_candidate["features"] == "all":
        selected_features = all_features
    else:
        selected_features = stable_features

    selected_factor_ids = [
        factor_id
        for factor_id, columns in factor_definitions.items()
        if any(feature in selected_features for feature in columns)
    ]
    all_factor_ids = [factor_id for factor_id, columns in factor_definitions.items() if columns]
    selected_fold_results = selected_candidate["folds"]
    for direction_fold, six_fold in zip(selected_fold_results, auxiliary_six_folds):
        direction_fold.update(
            {
                "six_class_accuracy": six_fold["six_class_accuracy"],
                "six_class_macro_f1": six_fold["six_class_macro_f1"],
                "six_class_distribution": six_fold["six_class_distribution"],
            }
        )
    return {
        "selected_candidate": selected_candidate,
        "selected_model": {
            "id": selected_candidate["id"],
            "label": selected_candidate["label"],
            "kind": selected_candidate["kind"],
            "class_balance": selected_candidate["class_balance"],
        },
        "adoption_reason": adoption_reason,
        "selected_features": selected_features,
        "excluded_features": [feature for feature in all_features if feature not in selected_features],
        "stable_selected_features": stable_features,
        "selected_factor_ids": selected_factor_ids,
        "excluded_factor_ids": [factor for factor in all_factor_ids if factor not in selected_factor_ids],
        "selected_factors": [FACTOR_LABELS.get(factor, factor) for factor in selected_factor_ids],
        "excluded_factors": [
            FACTOR_LABELS.get(factor, factor) for factor in all_factor_ids if factor not in selected_factor_ids
        ],
        "candidate_results": candidate_results,
        "class_balance_screening": class_balance_screening,
        "component_class_balances": selected_balances,
        "walk_forward_folds": selected_fold_results,
        "walk_forward_summary": selected_candidate["summary"],
        "feature_selection_by_fold": feature_selections,
        "feature_selection_importance": mean_importance,
        "auxiliary_six_class_folds": auxiliary_six_folds,
        "validation_dates_by_fold": validation_dates,
        "selection_period": {
            "start": _date_text(selection.index[0]),
            "end": _date_text(selection.index[-1]),
            "samples": int(len(selection)),
        },
        "direction_threshold": DIRECTION_THRESHOLD,
        "purge_trading_days": PURGE_TRADING_DAYS,
        "fixed_evaluation_used_for_selection": False,
        "model_tie_tolerance": MODEL_TIE_TOLERANCE,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _definition(candidate_id: str) -> dict[str, Any]:
    return next(definition for definition in CANDIDATE_DEFINITIONS if definition["id"] == candidate_id)


def _predict_selected_model(
    candidate_id: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    component_class_balances: dict[str, str] | None = None,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    component_class_balances = component_class_balances or {}
    definition = {
        **_definition(candidate_id),
        "class_balance": component_class_balances.get(
            candidate_id, _definition(candidate_id)["class_balance"]
        ),
    }
    if definition["kind"] != "ensemble":
        return _fit_component(definition, train, validation, features)
    component_probabilities: list[np.ndarray] = []
    component_importances: list[dict[str, float]] = []
    component_metadata: dict[str, Any] = {}
    for component_id in definition["components"]:
        component_definition = {
            **_definition(component_id),
            "class_balance": component_class_balances.get(
                component_id, _definition(component_id)["class_balance"]
            ),
        }
        probability, importance, metadata = _fit_component(component_definition, train, validation, features)
        component_probabilities.append(probability)
        component_importances.append(importance)
        component_metadata[component_id] = metadata
    mean_importance = {
        feature: float(np.mean([importance.get(feature, 0.0) for importance in component_importances]))
        for feature in features
    }
    return (
        simple_average_probabilities(component_probabilities),
        mean_importance,
        {
            "ensemble_method": "unweighted arithmetic mean of up probabilities",
            "components": definition["components"],
            "component_class_balances": component_class_balances,
            "component_preprocessing": component_metadata,
        },
    )


def moving_block_bootstrap_intervals(
    model_correct: np.ndarray,
    baseline_correct: np.ndarray,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    model_values = np.asarray(model_correct, dtype=float)
    baseline_values = np.asarray(baseline_correct, dtype=float)
    if len(model_values) != len(baseline_values) or len(model_values) < block_length:
        raise ValueError("bootstrap inputs are invalid")
    rng = np.random.default_rng(seed)
    count = len(model_values)
    blocks_needed = int(np.ceil(count / block_length))
    accuracy_samples = np.empty(resamples, dtype=float)
    gap_samples = np.empty(resamples, dtype=float)
    base_indices = np.arange(count)
    for sample_index in range(resamples):
        starts = rng.integers(0, count, size=blocks_needed)
        sampled_indices = np.concatenate(
            [base_indices[(start + np.arange(block_length)) % count] for start in starts]
        )[:count]
        accuracy_samples[sample_index] = model_values[sampled_indices].mean()
        gap_samples[sample_index] = (model_values[sampled_indices] - baseline_values[sampled_indices]).mean()
    return {
        "direction_accuracy_95ci": [float(value) for value in np.quantile(accuracy_samples, [0.025, 0.975])],
        "best_baseline_gap_95ci": [float(value) for value in np.quantile(gap_samples, [0.025, 0.975])],
        "block_length": int(block_length),
        "resamples": int(resamples),
        "seed": int(seed),
        "method": "circular moving-block bootstrap",
    }


def evaluate_fixed_holdout(
    selection: pd.DataFrame,
    holdout: pd.DataFrame,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    train = selection.iloc[:-PURGE_TRADING_DAYS].copy()
    features = configuration["selected_features"]
    candidate_id = configuration["selected_model"]["id"]
    probability, importance, preprocessing = _predict_selected_model(
        candidate_id,
        train,
        holdout,
        features,
        configuration.get("component_class_balances"),
    )
    y_train = train["target_direction"].to_numpy(dtype=int)
    y_holdout = holdout["target_direction"].to_numpy(dtype=int)
    majority = majority_baseline_predictions(y_train, len(holdout))
    momentum = momentum_baseline_predictions(holdout)
    metrics = _direction_metrics(y_holdout, probability, majority, momentum)
    prediction = direction_from_probability(probability)
    best_baseline_prediction = majority if metrics["best_baseline_name"] == "多数派基準モデル" else momentum
    intervals = moving_block_bootstrap_intervals(prediction == y_holdout, best_baseline_prediction == y_holdout)
    six = _six_class_evaluation(train, holdout, features)
    gap = metrics["baseline_gap"]
    if gap >= 0.02:
        advantage_message = f"基準モデルより＋{gap * 100:.1f}ポイント"
        advantage_level = "clear"
    elif gap > 0:
        advantage_message = "基準モデルとの差は小さく、優位性は限定的です"
        advantage_level = "limited"
    else:
        advantage_message = "基準モデルに対する優位性は確認できませんでした"
        advantage_level = "none"
    return {
        "period": {"start": _date_text(holdout.index[0]), "end": _date_text(holdout.index[-1])},
        "training_period": {"start": _date_text(train.index[0]), "end": _date_text(train.index[-1])},
        "training_samples": int(len(train)),
        "selection_samples_before_purge": int(len(selection)),
        "validation_samples": int(len(holdout)),
        "direction_threshold": DIRECTION_THRESHOLD,
        "purge_trading_days": PURGE_TRADING_DAYS,
        "holdout_used_for_selection": False,
        "fixed_evaluation_note": "固定評価期間はモデルの学習および自動選択には使用していません。",
        "direction_distribution": {"up": metrics["actual_up"], "non_up": metrics["actual_down"]},
        "feature_importance": [
            {"feature": feature, "mean_importance": float(value), "std_importance": 0.0}
            for feature, value in sorted(importance.items(), key=lambda item: (-item[1], item[0]))
        ],
        "preprocessing": preprocessing,
        "advantage_message": advantage_message,
        "advantage_level": advantage_level,
        **metrics,
        **six,
        **intervals,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _six_latest_prediction(train: pd.DataFrame, inference: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    x_train, x_latest, _ = _imputed_arrays(train, inference, features)
    y_six = np.array([CLASS_ORDER.index(classify_change(float(value))) for value in train["target_change"]], dtype=int)
    if len(np.unique(y_six)) < 2:
        probabilities = [0.0] * len(CLASS_ORDER)
        probabilities[int(y_six[0])] = 1.0
    else:
        model = _new_catboost("enhanced_selected", "six", "Balanced")
        model.fit(x_train, y_six)
        probabilities = _six_probabilities(model, model.predict_proba(x_latest)[0])
    top_index = int(np.argmax(probabilities))
    return {
        "probabilities": probabilities,
        "prediction_label": CLASS_ORDER[top_index],
        "top_probability": float(probabilities[top_index]),
    }


def fit_latest_prediction(data: PredictionData, configuration: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    training = data.training_frame.copy()
    if data.inference_row.index[0] in training.index:
        raise ServiceError("最新推論行が学習データへ混入しています。")
    features = configuration["selected_features"]
    candidate_id = configuration["selected_model"]["id"]
    probability, _, preprocessing = _predict_selected_model(
        candidate_id,
        training,
        data.inference_row,
        features,
        configuration.get("component_class_balances"),
    )
    up_probability = float(np.clip(probability[0], 0.0, 1.0))
    down_probability = float(1.0 - up_probability)
    direction_id = int(up_probability >= DIRECTION_THRESHOLD)
    six = _six_latest_prediction(training, data.inference_row, features)
    ranked = sorted(six["probabilities"], reverse=True)
    crash_probability = float(six["probabilities"][CLASS_ORDER.index("急落")])
    crash_margin = crash_probability - float(ranked[1] if len(ranked) > 1 else 0.0)
    strong_crash = bool(
        six["prediction_label"] == "急落"
        and crash_probability >= STRONG_CRASH_MIN_PROBABILITY
        and crash_margin >= STRONG_CRASH_MIN_MARGIN
    )
    return {
        **six,
        "direction": "上昇傾向" if direction_id == 1 else "下落傾向",
        "direction_key": "up" if direction_id == 1 else "strong_down" if strong_crash else "down",
        "direction_probability": up_probability if direction_id == 1 else down_probability,
        "up_probability": up_probability,
        "down_probability": down_probability,
        "direction_threshold": DIRECTION_THRESHOLD,
        "strong_crash": strong_crash,
        "training_samples": int(len(training)),
        "training_end": _date_text(training.index[-1]),
        "inference_date": _date_text(data.inference_row.index[0]),
        "inference_row_in_training": False,
        "selected_model": configuration["selected_model"],
        "selected_feature_count": int(len(features)),
        "preprocessing": preprocessing,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _data_signature(data: PredictionData, metadata: dict[str, Any]) -> str:
    core_columns = [
        column
        for column in ("open", "high", "low", "price", "volume", "target_change", "target_direction")
        if column in data.training_frame.columns
    ]
    external_columns = [
        column
        for column in metadata["available_external_factors"]
        if column in data.training_frame.columns
    ]
    # yfinanceが同じ履歴へ浮動小数点の最下位桁だけ異なる値を返すことがある。
    # モデル入力として十分な小数3桁へ丸め、実質同一データのキャッシュを安定させる。
    core_values = data.training_frame.loc[:, core_columns].round(3)
    external_values = data.training_frame.loc[:, external_columns].round(1)
    hashed_core = pd.util.hash_pandas_object(core_values, index=True).to_numpy(dtype=np.uint64)
    hashed_external = pd.util.hash_pandas_object(external_values, index=True).to_numpy(dtype=np.uint64)
    inference_columns = [column for column in ("open", "high", "low", "price", *external_columns) if column in data.inference_row]
    inference = pd.util.hash_pandas_object(data.inference_row.loc[:, inference_columns].round(1), index=True).to_numpy(dtype=np.uint64)
    hasher = hashlib.sha256()
    hasher.update(hashed_core.tobytes())
    hasher.update(hashed_external.tobytes())
    hasher.update(inference.tobytes())
    hasher.update(json.dumps(metadata["available_external_factors"], sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()


def _cache_descriptor(data: PredictionData, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_version": MODEL_VERSION,
        "fixed_data_start": metadata["fixed_data_start"],
        "fixed_data_end": metadata["fixed_data_end"],
        "model_selection_period": metadata["selection_period"],
        "fixed_evaluation_period": metadata["fixed_evaluation_period"],
        "external_alignment_version": EXTERNAL_ALIGNMENT_VERSION,
        "feature_definition_version": FEATURE_DEFINITION_VERSION,
        "direction_target_definition": DIRECTION_TARGET_DEFINITION,
        "direction_threshold": DIRECTION_THRESHOLD,
        "purge_trading_days": PURGE_TRADING_DAYS,
        "candidate_models": CANDIDATE_DEFINITIONS,
        "catboost_settings": CATBOOST_SETTINGS,
        "ensemble_definition": _definition("simple_average_ensemble")["components"],
        "fred_used": False,
        "available_external_factors": metadata["available_external_factors"],
        "data_signature": _data_signature(data, metadata),
    }


def _hash_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def _manifest_path(base_key: str) -> Path:
    return OUTPUT_DIR / f"nikkei_direction_manifest_{base_key}.json"


def _payload_path(final_key: str) -> Path:
    return OUTPUT_DIR / f"nikkei_direction_cache_{final_key}.json"


def _load_cached(base_key: str) -> tuple[dict[str, Any], str] | None:
    manifest_path = _manifest_path(base_key)
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        final_key = str(manifest["final_key"])
        payload = json.loads(_payload_path(final_key).read_text(encoding="utf-8"))
        if payload.get("version") != MODEL_VERSION or payload.get("base_key") != base_key:
            return None
        return payload, final_key
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _save_cached(base_key: str, final_key: str, payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _payload_path(final_key).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"version": MODEL_VERSION, "base_key": base_key, "final_key": final_key}
    _manifest_path(base_key).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_nikkei_direction_analysis(
    data: PredictionData,
    preparation_metadata: dict[str, Any],
    force_refresh: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    descriptor = _cache_descriptor(data, preparation_metadata)
    base_key = _hash_payload(descriptor)
    cache_lookup_started = time.perf_counter()
    cached = None if force_refresh else _load_cached(base_key)
    lookup_seconds = time.perf_counter() - cache_lookup_started
    if cached is not None:
        payload, final_key = cached
        result = dict(payload["result"])
        result["preparation_metadata"] = preparation_metadata
        result["cache"] = {
            "used": True,
            "key": final_key,
            "base_key": base_key,
            "created_at": payload["created_at"],
            "force_refresh": False,
            "lookup_seconds": float(lookup_seconds),
        }
        result["total_seconds"] = float(time.perf_counter() - started)
        return result

    selection, holdout = fixed_period_split(data.training_frame)
    model_selection_started = time.perf_counter()
    factor_definitions = preparation_metadata["factor_definitions"]
    configuration = compare_direction_models(selection, data.feature_columns, factor_definitions)
    model_selection_seconds = time.perf_counter() - model_selection_started
    fixed_evaluation_started = time.perf_counter()
    evaluation = evaluate_fixed_holdout(selection, holdout, configuration)
    fixed_evaluation_seconds = time.perf_counter() - fixed_evaluation_started
    latest_started = time.perf_counter()
    latest = fit_latest_prediction(data, configuration)
    latest_seconds = time.perf_counter() - latest_started

    selected_definition = _definition(configuration["selected_model"]["id"])
    model_setting = (
        CATBOOST_SETTINGS[selected_definition["setting"]]
        if "setting" in selected_definition
        else {
            "name": selected_definition["kind"],
            "label": selected_definition["label"],
            "parameters": "predefined single configuration",
        }
    )
    configuration["model_setting"] = model_setting
    configuration["class_balance"] = configuration["selected_model"].get(
        "class_balance", selected_definition.get("class_balance", "component settings")
    )
    configuration["class_balance_label"] = {
        "Balanced": "学習Fold内のCatBoost自動クラス補正",
        "balanced": "学習Fold内のbalancedクラス重み",
        "None": "クラス補正なし",
        "component settings": "各構成の学習Fold内設定",
    }.get(configuration["class_balance"], str(configuration["class_balance"]))
    configuration["feature_importance"] = evaluation["feature_importance"]
    configuration["selected_features_count"] = len(configuration["selected_features"])
    configuration["excluded_features_count"] = len(configuration["excluded_features"])
    configuration["ensemble"] = {
        "adopted": selected_definition["kind"] == "ensemble",
        "components": selected_definition.get("components", []),
        "method": "unweighted arithmetic mean of up probabilities" if selected_definition["kind"] == "ensemble" else None,
    }

    result = {
        "version": MODEL_VERSION,
        "configuration": configuration,
        "final_evaluation": evaluation,
        "latest_prediction": latest,
        "preparation_metadata": preparation_metadata,
        "model_selection_seconds": float(model_selection_seconds),
        "final_evaluation_seconds": float(fixed_evaluation_seconds),
        "latest_inference_seconds": float(latest_seconds),
        "total_seconds": float(time.perf_counter() - started),
    }
    adopted_cache_fields = {
        "base_key": base_key,
        "adopted_model": configuration["selected_model"],
        "adopted_features": configuration["selected_features"],
        "class_balance": configuration["class_balance"],
        "model_setting": configuration["model_setting"],
        "ensemble": configuration["ensemble"],
    }
    final_key = _hash_payload(adopted_cache_fields)
    created_at = _now_jst()
    payload = {
        "version": MODEL_VERSION,
        "base_key": base_key,
        "final_key": final_key,
        "created_at": created_at,
        "descriptor": descriptor,
        "adopted_cache_fields": adopted_cache_fields,
        "result": result,
    }
    _save_cached(base_key, final_key, payload)
    result["cache"] = {
        "used": False,
        "key": final_key,
        "base_key": base_key,
        "created_at": created_at,
        "force_refresh": bool(force_refresh),
        "lookup_seconds": float(lookup_seconds),
    }
    return result
