from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .common import CLASS_ORDER, ServiceError, classify_change
from .individual_market import TRAINING_YEARS, IndividualMarketData
from .nikkei_dual_market import next_japan_market_days


MODEL_SCHEMA_VERSION = "individual_logistic_fast_v1"
FIXED_FEATURE_GROUP = "F_logistic_fast"
FIXED_DIRECTION_THRESHOLD = 0.50
PURGE_TRADING_DAYS = 5
RANDOM_SEED = 42
MIN_TRAINING_ROWS = 420
MIN_SIX_CLASS_COUNT = 3

LOGISTIC_SETTINGS: dict[str, Any] = {
    "solver": "lbfgs",
    "C": 1.0,
    "max_iter": 300,
    "tol": 1e-4,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}
MODEL_SETTINGS_HASH = hashlib.sha256(
    json.dumps(LOGISTIC_SETTINGS, sort_keys=True, ensure_ascii=True).encode("utf-8")
).hexdigest()[:16]


@dataclass(frozen=True)
class FittedPreprocessor:
    columns: list[str]
    medians: pd.Series
    means: pd.Series
    scales: pd.Series


def _six_class_ids(changes: pd.Series) -> np.ndarray:
    return np.asarray([CLASS_ORDER.index(classify_change(float(value))) for value in changes], dtype=int)


def _numeric_feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    if not available:
        raise ServiceError("個別銘柄予測に使用できる数値特徴量がありません。")
    numeric = frame.loc[:, available].apply(pd.to_numeric, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


def fit_preprocessor(training: pd.DataFrame, columns: list[str]) -> tuple[FittedPreprocessor, np.ndarray]:
    """中央値・標準化を学習行だけでfitし、定数列と全欠損列を除外する。"""
    numeric = _numeric_feature_frame(training, columns)
    medians = numeric.median(axis=0, skipna=True).dropna()
    numeric = numeric.loc[:, medians.index].fillna(medians)
    if numeric.empty:
        raise ServiceError("個別銘柄予測の前処理後に使用できる特徴量がありません。")
    means = numeric.mean(axis=0)
    scales = numeric.std(axis=0, ddof=0)
    scale_values = scales.to_numpy(dtype=float)
    usable = scales.index[np.isfinite(scale_values) & (scale_values > 1e-12)]
    if len(usable) == 0:
        raise ServiceError("個別銘柄予測の特徴量がすべて定数列になりました。")
    medians = medians.loc[usable].astype(float)
    means = means.loc[usable].astype(float)
    scales = scales.loc[usable].astype(float)
    transformed = ((numeric.loc[:, usable] - means) / scales).to_numpy(dtype=float, copy=False)
    if not np.isfinite(transformed).all():
        raise ServiceError("個別銘柄予測の前処理後に非数値が残りました。")
    fitted = FittedPreprocessor(list(usable), medians, means, scales)
    return fitted, transformed


def transform_features(frame: pd.DataFrame, fitted: FittedPreprocessor) -> np.ndarray:
    numeric = _numeric_feature_frame(frame, fitted.columns).reindex(columns=fitted.columns)
    transformed = ((numeric.fillna(fitted.medians) - fitted.means) / fitted.scales).to_numpy(dtype=float, copy=False)
    if not np.isfinite(transformed).all():
        raise ServiceError("個別銘柄予測の推論入力に非数値が残りました。")
    return transformed


def _new_model() -> LogisticRegression:
    return LogisticRegression(**LOGISTIC_SETTINGS)


def _fit_model(model: LogisticRegression, x: np.ndarray, y: np.ndarray) -> tuple[float, list[str]]:
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    convergence = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    return float(time.perf_counter() - started), convergence


def _probabilities_for_classes(model: LogisticRegression, x: np.ndarray, class_count: int) -> np.ndarray:
    source = model.predict_proba(x)
    result = np.zeros((len(x), class_count), dtype=float)
    for source_position, class_id in enumerate(model.classes_):
        index = int(class_id)
        if 0 <= index < class_count:
            result[:, index] = source[:, source_position]
    totals = result.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ServiceError("個別銘柄予測のモデル出力を6区分へ整列できませんでした。")
    return result / totals




def _importance_rows(coefficients: np.ndarray, columns: list[str]) -> list[dict[str, Any]]:
    values = np.asarray(coefficients, dtype=float)
    values = np.mean(np.abs(values), axis=0) if values.ndim == 2 else np.abs(values)
    total = float(values.sum())
    normalized = values / total if total > 0 else values
    order = np.argsort(-normalized)[:10]
    return [
        {
            "internal_name": columns[int(index)],
            "display_name": columns[int(index)],
            "importance": float(normalized[int(index)]),
            "coefficient_absolute": float(values[int(index)]),
        }
        for index in order
    ]


def latest_prediction(data: IndividualMarketData) -> dict[str, Any]:
    """通常API専用。固定前処理1回と固定ロジスティック回帰2回だけを実行する。"""
    timing: dict[str, float] = {}
    basis = data.basis_date
    training = data.frame.loc[
        (data.frame.index >= basis - pd.DateOffset(years=TRAINING_YEARS))
        & (data.frame.index >= data.first_training_date)
    ].dropna(subset=["target_direction", "target_change"]).sort_index()
    if len(training) < MIN_TRAINING_ROWS or basis in training.index:
        raise ServiceError(
            "最新予測に必要な株価履歴が不足しています。",
            "最大8年間の確定済み履歴を安全に分離できる銘柄で再度お試しください。",
        )
    features = list(data.feature_groups.get(FIXED_FEATURE_GROUP, []))
    if not features:
        raise ServiceError("固定ロジスティック回帰用の特徴量グループを作成できませんでした。")

    preprocess_started = time.perf_counter()
    preprocessor, x_train = fit_preprocessor(training, features)
    x_latest = transform_features(data.frame.loc[[basis]], preprocessor)
    timing["preprocess_seconds"] = float(time.perf_counter() - preprocess_started)

    y_direction = pd.to_numeric(training["target_direction"], errors="coerce").to_numpy(dtype=int)
    y_six = _six_class_ids(training["target_change"])
    if len(np.unique(y_direction)) != 2:
        raise ServiceError("方向予測の学習データに上昇・下落の両方がありません。")
    if len(np.unique(y_six)) < MIN_SIX_CLASS_COUNT:
        raise ServiceError("6段階予測を学習できる値動き区分が不足しています。")

    direction_model = _new_model()
    six_model = _new_model()
    timing["direction_fit_seconds"], direction_warnings = _fit_model(direction_model, x_train, y_direction)
    timing["six_class_fit_seconds"], six_warnings = _fit_model(six_model, x_train, y_six)

    inference_started = time.perf_counter()
    direction_probabilities = _probabilities_for_classes(direction_model, x_latest, 2)[0]
    six_probabilities = _probabilities_for_classes(six_model, x_latest, len(CLASS_ORDER))[0]
    timing["inference_seconds"] = float(time.perf_counter() - inference_started)
    up_score = float(direction_probabilities[1])
    predicted_direction = 1 if up_score >= FIXED_DIRECTION_THRESHOLD else 0
    predicted_six_id = int(np.argmax(six_probabilities))

    training_start = pd.Timestamp(training.index[0])
    common_feature_start = data.group_valid_starts.get(FIXED_FEATURE_GROUP, training_start)
    data_collection_start = max(training_start, pd.Timestamp(common_feature_start))
    target_date = pd.Timestamp(next_japan_market_days(basis, 5)[-1]).strftime("%Y-%m-%d")
    model_record = {
        "algorithm": "logistic_regression",
        "model_name": "LogisticRegression",
        "feature_group": FIXED_FEATURE_GROUP,
        "settings": dict(LOGISTIC_SETTINGS),
        "settings_hash": MODEL_SETTINGS_HASH,
        "fixed_for_all_tickers": True,
    }
    return {
        "direction": "上昇傾向" if predicted_direction == 1 else "下落傾向",
        "direction_key": "up" if predicted_direction == 1 else "down",
        "raw_up_score": up_score,
        "calibrated_up_score": up_score,
        "calibrated_down_score": 1.0 - up_score,
        "fixed_direction_threshold": FIXED_DIRECTION_THRESHOLD,
        "probabilities": [
            {"raw_label": label, "probability": float(six_probabilities[index]), "percentage": float(six_probabilities[index] * 100.0)}
            for index, label in enumerate(CLASS_ORDER)
        ],
        "prediction_class_raw": CLASS_ORDER[predicted_six_id],
        "forecast_target_date": target_date,
        "training_samples": int(len(training)),
        "training_start": training_start.strftime("%Y-%m-%d"),
        "training_end": pd.Timestamp(training.index[-1]).strftime("%Y-%m-%d"),
        "inference_row_in_training": False,
        "direction_model": {**model_record, "task": "binary_direction"},
        "six_class_model": {**model_record, "task": "multiclass_six_stage"},
        "direction_features": list(preprocessor.columns),
        "six_class_features": list(preprocessor.columns),
        "feature_importance_top10": _importance_rows(direction_model.coef_, preprocessor.columns),
        "six_class_feature_importance_top10": _importance_rows(six_model.coef_, preprocessor.columns),
        "calibration": {
            "calibration_status": "not_applied",
            "calibration_reason": "compatible calibrator unavailable",
            "public_output_source": "raw_predict_proba",
        },
        "selection": {
            "performed": False,
            "reason": "model, features, class_weight, time weighting and C are fixed before prediction",
            "cross_validation_count": 0,
            "candidate_comparison_count": 0,
        },
        "fit_counts": {"direction": 1, "six_class": 1, "total": 2},
        "timing": timing,
        "convergence_warnings": [*direction_warnings, *six_warnings],
        "data_collection_start": data_collection_start.strftime("%Y-%m-%d"),
        "data_collection_end": basis.strftime("%Y-%m-%d"),
        "data_collection_definition": (
            "市場ごとの利用可能時刻調整、backward as-of結合、300取引日の準備、特徴量計算を終えた後、"
            "固定特徴量を最大8年の学習入力として共通利用できた期間"
        ),
    }
