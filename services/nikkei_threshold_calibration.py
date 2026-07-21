from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import BASE_DIR, CLASS_ORDER, OUTPUT_DIR, PREDICTION_HORIZON, PredictionData, ServiceError
from .nikkei_direction_comparison import (
    BOOTSTRAP_BLOCK_LENGTH,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EXTERNAL_ALIGNMENT_VERSION,
    FEATURE_DEFINITION_VERSION,
    PURGE_TRADING_DAYS,
    STRONG_CRASH_MIN_MARGIN,
    STRONG_CRASH_MIN_PROBABILITY,
    _data_signature,
    _six_class_evaluation,
    _six_latest_prediction,
    expanding_purged_walk_forward_splits,
    fixed_period_split,
    majority_baseline_predictions,
    momentum_baseline_predictions,
    moving_block_bootstrap_intervals,
)


# These conditions were frozen before the fixed confirmation period was evaluated.
MODEL_VERSION = "nikkei_extra_trees_28f_threshold_v1"
MODEL_SETTINGS_VERSION = "extra_trees_28f_2026_07"
THRESHOLD_LOGIC_VERSION = "nested_walk_forward_threshold_v1"
STANDARD_THRESHOLD = 0.50
THRESHOLD_CANDIDATES = tuple(round(float(value), 2) for value in np.arange(0.40, 0.601, 0.02))
THRESHOLD_CANDIDATE_MIN = min(THRESHOLD_CANDIDATES)
THRESHOLD_CANDIDATE_MAX = max(THRESHOLD_CANDIDATES)
THRESHOLD_CANDIDATE_STEP = 0.02
THRESHOLD_SELECTION_METRIC = "mean_direction_balanced_accuracy"
THRESHOLD_TIE_TOLERANCE = 0.001
MIN_PREDICTION_SHARE = 0.10
MIN_BALANCED_IMPROVEMENT = 0.005
MIN_FOLD_IMPROVEMENT = 0.001
MIN_IMPROVED_FOLDS = 2
MAX_THRESHOLD_RANGE = 0.06
MAX_THRESHOLD_STD = 0.03
INNER_WALK_FORWARD_FOLDS = 3
OUTER_WALK_FORWARD_FOLDS = 3

FROZEN_EXTRA_TREES_PARAMETERS: dict[str, Any] = {
    "n_estimators": 240,
    "max_depth": 7,
    "min_samples_leaf": 8,
    "max_features": "sqrt",
    "random_state": 42,
    "class_weight": None,
    "n_jobs": -1,
}

FROZEN_FEATURES = [
    "ch_trend",
    "distance_high60",
    "gold",
    "ch_pos",
    "low",
    "ch_lower",
    "return_5d",
    "usdjpy",
    "usdjpy_ret5",
    "open",
    "usdjpy_ret20",
    "ma20_gap",
    "overheat_score",
    "intraday_range",
    "btc_ret20",
    "gold_ret20",
    "corr20_nikkei_nasdaq",
    "return_10d",
    "high",
    "ch_upper",
    "ma20_slope5",
    "return_20d",
    "ma200_gap",
    "gold_ret5",
    "price",
    "distance_low20",
    "ma100_slope5",
    "return_2d",
]

FROZEN_SELECTED_FACTORS = [
    "日経平均の価格・リターン",
    "RSI・価格チャネル",
    "トレンド環境",
    "リターン・値幅",
    "価格位置",
    "外部市場との相関",
    "ドル円",
    "金",
    "Bitcoin",
]

FROZEN_EXCLUDED_FACTORS = [
    "出来高",
    "ボラティリティ環境",
    "S&P 500",
    "NYダウ",
    "NASDAQ",
    "VIX",
    "SOX",
    "NVIDIA",
    "原油",
]

MODEL_SETTINGS_DIR = BASE_DIR / "model_settings"
THRESHOLD_SETTINGS_PATH = MODEL_SETTINGS_DIR / "nikkei_direction_threshold.json"
THRESHOLD_HISTORY_PATH = MODEL_SETTINGS_DIR / "nikkei_direction_threshold_history.json"


def _now_jst() -> str:
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")


def _date_text(value: pd.Timestamp | datetime) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def direction_from_probability(up_probability: np.ndarray | float, threshold: float) -> np.ndarray:
    return (np.asarray(up_probability, dtype=float) >= float(threshold)).astype(int)


def _training_medians(frame: pd.DataFrame) -> pd.Series:
    return frame[FROZEN_FEATURES].median(axis=0, skipna=True).fillna(0.0).astype(float)


def _fit_frozen_extra_trees(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    from sklearn.ensemble import ExtraTreesClassifier

    missing = [feature for feature in FROZEN_FEATURES if feature not in train or feature not in validation]
    if missing:
        raise ServiceError(
            "固定した28特徴量を揃えられませんでした。",
            f"不足している特徴量: {', '.join(missing)}",
        )
    y_train = train["target_direction"].to_numpy(dtype=int)
    if len(np.unique(y_train)) < 2:
        constant = float(y_train[0])
        return (
            np.full(len(validation), constant, dtype=float),
            {feature: 0.0 for feature in FROZEN_FEATURES},
            {"fallback": "single training class"},
        )
    medians = _training_medians(train)
    x_train = train[FROZEN_FEATURES].fillna(medians).to_numpy(dtype=float)
    x_validation = validation[FROZEN_FEATURES].fillna(medians).to_numpy(dtype=float)
    model = ExtraTreesClassifier(**FROZEN_EXTRA_TREES_PARAMETERS)
    model.fit(x_train, y_train)
    matrix = np.asarray(model.predict_proba(x_validation), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    if 1 in classes:
        probability = matrix[:, int(np.where(classes == 1)[0][0])]
    else:
        probability = np.zeros(len(validation), dtype=float)
    raw_importance = np.asarray(model.feature_importances_, dtype=float)
    total = float(np.abs(raw_importance).sum())
    normalized = np.zeros_like(raw_importance) if total <= 0 else np.abs(raw_importance) / total * 100.0
    importance = {feature: float(value) for feature, value in zip(FROZEN_FEATURES, normalized)}
    return (
        np.clip(probability, 0.0, 1.0),
        importance,
        {
            "imputation": "training-fold median; all-missing training columns use 0",
            "imputation_fit_start": _date_text(train.index[0]),
            "imputation_fit_end": _date_text(train.index[-1]),
            "class_balance": "None",
            "parameters": dict(FROZEN_EXTRA_TREES_PARAMETERS),
        },
    )


def _classification_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score

    truth = np.asarray(y_true, dtype=int)
    predicted = np.asarray(prediction, dtype=int)
    if len(truth) != len(predicted) or len(truth) == 0:
        raise ValueError("direction metric inputs are invalid")
    return {
        "direction_accuracy": float(accuracy_score(truth, predicted)),
        "direction_balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "direction_macro_f1": float(f1_score(truth, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "up_recall": float(recall_score(truth, predicted, labels=[1], average="macro", zero_division=0)),
        "down_recall": float(recall_score(truth, predicted, labels=[0], average="macro", zero_division=0)),
        "predicted_up": int(np.sum(predicted == 1)),
        "predicted_down": int(np.sum(predicted == 0)),
        "actual_up": int(np.sum(truth == 1)),
        "actual_down": int(np.sum(truth == 0)),
        "correct_predictions": int(np.sum(predicted == truth)),
        "validation_samples": int(len(truth)),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).astype(int).tolist(),
    }


def metrics_at_threshold(y_true: np.ndarray, up_probability: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = direction_from_probability(up_probability, threshold)
    metrics = _classification_metrics(y_true, prediction)
    metrics["decision_threshold"] = float(threshold)
    metrics["minimum_prediction_share"] = float(
        min(metrics["predicted_up"], metrics["predicted_down"]) / metrics["validation_samples"]
    )
    return metrics


def _candidate_summary(threshold: float, folds: list[dict[str, Any]]) -> dict[str, Any]:
    balanced = np.asarray([fold["direction_balanced_accuracy"] for fold in folds], dtype=float)
    accuracy = np.asarray([fold["direction_accuracy"] for fold in folds], dtype=float)
    macro_f1 = np.asarray([fold["direction_macro_f1"] for fold in folds], dtype=float)
    valid_predictions = all(
        fold["predicted_up"] > 0
        and fold["predicted_down"] > 0
        and fold["minimum_prediction_share"] >= MIN_PREDICTION_SHARE
        for fold in folds
    )
    return {
        "threshold": float(threshold),
        "mean_direction_balanced_accuracy": float(balanced.mean()),
        "mean_direction_accuracy": float(accuracy.mean()),
        "mean_direction_macro_f1": float(macro_f1.mean()),
        "direction_balanced_accuracy_std": float(balanced.std(ddof=0)),
        "worst_fold_direction_balanced_accuracy": float(balanced.min()),
        "predicted_up": int(sum(fold["predicted_up"] for fold in folds)),
        "predicted_down": int(sum(fold["predicted_down"] for fold in folds)),
        "up_recall_mean": float(np.mean([fold["up_recall"] for fold in folds])),
        "down_recall_mean": float(np.mean([fold["down_recall"] for fold in folds])),
        "valid_prediction_mix": bool(valid_predictions),
        "folds": folds,
    }


def _keep_near_best(
    candidates: list[dict[str, Any]],
    field: str,
    maximize: bool,
) -> list[dict[str, Any]]:
    values = [float(candidate[field]) for candidate in candidates]
    best = max(values) if maximize else min(values)
    if maximize:
        return [candidate for candidate in candidates if float(candidate[field]) >= best - THRESHOLD_TIE_TOLERANCE]
    return [candidate for candidate in candidates if float(candidate[field]) <= best + THRESHOLD_TIE_TOLERANCE]


def choose_threshold_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [candidate for candidate in candidates if candidate["valid_prediction_mix"]]
    if not valid:
        return next(candidate for candidate in candidates if candidate["threshold"] == STANDARD_THRESHOLD)
    remaining = _keep_near_best(valid, "mean_direction_balanced_accuracy", True)
    remaining = _keep_near_best(remaining, "worst_fold_direction_balanced_accuracy", True)
    remaining = _keep_near_best(remaining, "direction_balanced_accuracy_std", False)
    remaining = _keep_near_best(remaining, "mean_direction_macro_f1", True)
    remaining = _keep_near_best(remaining, "mean_direction_accuracy", True)
    return min(remaining, key=lambda candidate: (abs(candidate["threshold"] - STANDARD_THRESHOLD), candidate["threshold"]))


def _threshold_selection_safety(
    chosen: dict[str, Any],
    standard: dict[str, Any],
) -> tuple[bool, list[str], dict[str, Any]]:
    if chosen["threshold"] == STANDARD_THRESHOLD:
        return True, ["内側検証で標準の50％が選ばれました。"], {
            "balanced_accuracy_improvement": 0.0,
            "improved_folds": 0,
        }
    improvement = float(
        chosen["mean_direction_balanced_accuracy"] - standard["mean_direction_balanced_accuracy"]
    )
    fold_improvements = [
        float(chosen_fold["direction_balanced_accuracy"] - standard_fold["direction_balanced_accuracy"])
        for chosen_fold, standard_fold in zip(chosen["folds"], standard["folds"])
    ]
    improved_folds = int(sum(value >= MIN_FOLD_IMPROVEMENT for value in fold_improvements))
    reasons: list[str] = []
    if improvement < MIN_BALANCED_IMPROVEMENT:
        reasons.append(
            f"公平採点の平均改善が安全条件の{MIN_BALANCED_IMPROVEMENT * 100:.1f}ポイント未満でした。"
        )
    if improved_folds < MIN_IMPROVED_FOLDS:
        reasons.append(f"改善したFoldが{MIN_IMPROVED_FOLDS}期間未満でした。")
    if not chosen["valid_prediction_mix"]:
        reasons.append("上昇・下落のどちらかの予測割合が安全条件を下回りました。")
    return not reasons, reasons, {
        "balanced_accuracy_improvement": improvement,
        "fold_improvements": fold_improvements,
        "improved_folds": improved_folds,
    }


def select_threshold_with_inner_walk_forward(outer_training: pd.DataFrame) -> dict[str, Any]:
    folds = expanding_purged_walk_forward_splits(
        outer_training,
        n_splits=INNER_WALK_FORWARD_FOLDS,
        purge_days=PURGE_TRADING_DAYS,
    )
    probability_folds: list[dict[str, Any]] = []
    for fold_number, (train, validation) in enumerate(folds, start=1):
        probability, _, preprocessing = _fit_frozen_extra_trees(train, validation)
        probability_folds.append(
            {
                "fold": fold_number,
                "train": train,
                "validation": validation,
                "probability": probability,
                "preprocessing": preprocessing,
            }
        )
    candidates: list[dict[str, Any]] = []
    for threshold in THRESHOLD_CANDIDATES:
        fold_results: list[dict[str, Any]] = []
        for fold in probability_folds:
            validation = fold["validation"]
            y_true = validation["target_direction"].to_numpy(dtype=int)
            metrics = metrics_at_threshold(y_true, fold["probability"], threshold)
            fold_results.append(
                {
                    "fold": fold["fold"],
                    "train_period": {
                        "start": _date_text(fold["train"].index[0]),
                        "end": _date_text(fold["train"].index[-1]),
                    },
                    "validation_period": {
                        "start": _date_text(validation.index[0]),
                        "end": _date_text(validation.index[-1]),
                    },
                    "purge_trading_days": PURGE_TRADING_DAYS,
                    "training_samples": int(len(fold["train"])),
                    **metrics,
                }
            )
        candidates.append(_candidate_summary(threshold, fold_results))
    provisional = choose_threshold_candidate(candidates)
    standard = next(candidate for candidate in candidates if candidate["threshold"] == STANDARD_THRESHOLD)
    safe, safety_reasons, safety = _threshold_selection_safety(provisional, standard)
    selected_threshold = float(provisional["threshold"] if safe else STANDARD_THRESHOLD)
    fallback_applied = bool(not safe and provisional["threshold"] != STANDARD_THRESHOLD)
    fallback_reason = " ".join(safety_reasons) if fallback_applied else None
    selected = next(candidate for candidate in candidates if candidate["threshold"] == selected_threshold)
    return {
        "period": {
            "start": _date_text(outer_training.index[0]),
            "end": _date_text(outer_training.index[-1]),
            "samples": int(len(outer_training)),
        },
        "selected_threshold": selected_threshold,
        "provisional_threshold": float(provisional["threshold"]),
        "selected_candidate": selected,
        "standard_candidate": standard,
        "candidate_results": candidates,
        "fallback_applied": fallback_applied,
        "fallback_reason": fallback_reason,
        "safety": safety,
        "selection_rule": threshold_logic_metadata(),
        "outer_validation_used_for_selection": False,
        "fixed_confirmation_used_for_selection": False,
        "latest_prediction_used_for_selection": False,
    }


def _baseline_metrics(
    y_true: np.ndarray,
    majority_prediction: np.ndarray,
    momentum_prediction: np.ndarray,
) -> dict[str, Any]:
    majority = _classification_metrics(y_true, majority_prediction)
    momentum = _classification_metrics(y_true, momentum_prediction)
    return {"majority": majority, "momentum": momentum}


def _public_fold_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "minimum_prediction_share"}


def nested_walk_forward_threshold_evaluation(selection: pd.DataFrame) -> dict[str, Any]:
    started = time.perf_counter()
    outer_folds = expanding_purged_walk_forward_splits(
        selection,
        n_splits=OUTER_WALK_FORWARD_FOLDS,
        purge_days=PURGE_TRADING_DAYS,
    )
    fold_results: list[dict[str, Any]] = []
    pooled_truth: list[np.ndarray] = []
    pooled_adjusted: list[np.ndarray] = []
    pooled_standard: list[np.ndarray] = []
    pooled_majority: list[np.ndarray] = []
    pooled_momentum: list[np.ndarray] = []
    for fold_number, (outer_train, outer_validation) in enumerate(outer_folds, start=1):
        inner = select_threshold_with_inner_walk_forward(outer_train)
        selected_threshold = float(inner["selected_threshold"])
        probability, _, preprocessing = _fit_frozen_extra_trees(outer_train, outer_validation)
        y_train = outer_train["target_direction"].to_numpy(dtype=int)
        y_true = outer_validation["target_direction"].to_numpy(dtype=int)
        adjusted_prediction = direction_from_probability(probability, selected_threshold)
        standard_prediction = direction_from_probability(probability, STANDARD_THRESHOLD)
        adjusted = _classification_metrics(y_true, adjusted_prediction)
        standard = _classification_metrics(y_true, standard_prediction)
        majority_prediction = majority_baseline_predictions(y_train, len(outer_validation))
        momentum_prediction = momentum_baseline_predictions(outer_validation)
        baselines = _baseline_metrics(y_true, majority_prediction, momentum_prediction)
        fold_results.append(
            {
                "fold": fold_number,
                "outer_train_period": {
                    "start": _date_text(outer_train.index[0]),
                    "end": _date_text(outer_train.index[-1]),
                },
                "outer_validation_period": {
                    "start": _date_text(outer_validation.index[0]),
                    "end": _date_text(outer_validation.index[-1]),
                },
                "training_samples": int(len(outer_train)),
                "validation_samples": int(len(outer_validation)),
                "purge_trading_days": PURGE_TRADING_DAYS,
                "selected_threshold": selected_threshold,
                "threshold_fallback_applied": inner["fallback_applied"],
                "threshold_fallback_reason": inner["fallback_reason"],
                "fixed_50_metrics": _public_fold_metrics(standard),
                "selected_threshold_metrics": _public_fold_metrics(adjusted),
                "majority_baseline": baselines["majority"],
                "majority_direction": "上昇" if int(majority_prediction[0]) == 1 else "下落",
                "momentum_baseline": baselines["momentum"],
                "balanced_accuracy_change_from_50": float(
                    adjusted["direction_balanced_accuracy"] - standard["direction_balanced_accuracy"]
                ),
                "accuracy_change_from_50": float(adjusted["direction_accuracy"] - standard["direction_accuracy"]),
                "inner_selection": inner,
                "preprocessing": preprocessing,
                "outer_validation_used_for_inner_selection": False,
            }
        )
        pooled_truth.append(y_true)
        pooled_adjusted.append(adjusted_prediction)
        pooled_standard.append(standard_prediction)
        pooled_majority.append(majority_prediction)
        pooled_momentum.append(momentum_prediction)

    truth = np.concatenate(pooled_truth)
    adjusted_prediction = np.concatenate(pooled_adjusted)
    standard_prediction = np.concatenate(pooled_standard)
    majority_prediction = np.concatenate(pooled_majority)
    momentum_prediction = np.concatenate(pooled_momentum)
    adjusted = _classification_metrics(truth, adjusted_prediction)
    standard = _classification_metrics(truth, standard_prediction)
    majority = _classification_metrics(truth, majority_prediction)
    momentum = _classification_metrics(truth, momentum_prediction)
    best_baseline_name = "多数派基準" if majority["direction_accuracy"] >= momentum["direction_accuracy"] else "直近方向継続"
    best_baseline = majority_prediction if best_baseline_name == "多数派基準" else momentum_prediction
    intervals = moving_block_bootstrap_intervals(adjusted_prediction == truth, best_baseline == truth)
    selected_thresholds = np.asarray([fold["selected_threshold"] for fold in fold_results], dtype=float)
    adjusted_balanced = np.asarray(
        [fold["selected_threshold_metrics"]["direction_balanced_accuracy"] for fold in fold_results], dtype=float
    )
    standard_balanced = np.asarray(
        [fold["fixed_50_metrics"]["direction_balanced_accuracy"] for fold in fold_results], dtype=float
    )
    improved_folds = int(np.sum(adjusted_balanced - standard_balanced >= MIN_FOLD_IMPROVEMENT))
    threshold_range = float(selected_thresholds.max() - selected_thresholds.min())
    threshold_std = float(selected_thresholds.std(ddof=0))
    stable = bool(threshold_range <= MAX_THRESHOLD_RANGE and threshold_std <= MAX_THRESHOLD_STD)
    balanced_gain = float(adjusted["direction_balanced_accuracy"] - standard["direction_balanced_accuracy"])
    method_safe = bool(
        balanced_gain >= MIN_BALANCED_IMPROVEMENT
        and improved_folds >= MIN_IMPROVED_FOLDS
        and stable
    )
    method_reasons: list[str] = []
    if balanced_gain < MIN_BALANCED_IMPROVEMENT:
        method_reasons.append("外側検証で公平採点の改善が安全条件に届きませんでした。")
    if improved_folds < MIN_IMPROVED_FOLDS:
        method_reasons.append("外側検証で改善した期間が複数確認できませんでした。")
    if not stable:
        method_reasons.append("外側Foldで選ばれた判定ラインの変動が大きすぎました。")
    return {
        "evaluation_type": "formal_nested_walk_forward",
        "period": {
            "start": fold_results[0]["outer_validation_period"]["start"],
            "end": fold_results[-1]["outer_validation_period"]["end"],
        },
        "validation_samples": int(len(truth)),
        "correct_predictions": adjusted["correct_predictions"],
        "selected_threshold_method": adjusted,
        "fixed_50_method": standard,
        "direction_accuracy": adjusted["direction_accuracy"],
        "direction_balanced_accuracy": adjusted["direction_balanced_accuracy"],
        "direction_macro_f1": adjusted["direction_macro_f1"],
        "up_recall": adjusted["up_recall"],
        "down_recall": adjusted["down_recall"],
        "predicted_up": adjusted["predicted_up"],
        "predicted_down": adjusted["predicted_down"],
        "actual_up": adjusted["actual_up"],
        "actual_down": adjusted["actual_down"],
        "confusion_matrix": adjusted["confusion_matrix"],
        "majority_baseline": majority,
        "momentum_baseline": momentum,
        "majority_baseline_accuracy": majority["direction_accuracy"],
        "majority_baseline_balanced_accuracy": majority["direction_balanced_accuracy"],
        "momentum_baseline_accuracy": momentum["direction_accuracy"],
        "momentum_baseline_balanced_accuracy": momentum["direction_balanced_accuracy"],
        "best_baseline_name": best_baseline_name,
        "best_baseline_accuracy": max(majority["direction_accuracy"], momentum["direction_accuracy"]),
        "baseline_gap": float(
            adjusted["direction_accuracy"] - max(majority["direction_accuracy"], momentum["direction_accuracy"])
        ),
        "accuracy_change_from_fixed_50": float(adjusted["direction_accuracy"] - standard["direction_accuracy"]),
        "balanced_accuracy_change_from_fixed_50": balanced_gain,
        "macro_f1_change_from_fixed_50": float(adjusted["direction_macro_f1"] - standard["direction_macro_f1"]),
        "fold_mean_direction_accuracy": float(
            np.mean([fold["selected_threshold_metrics"]["direction_accuracy"] for fold in fold_results])
        ),
        "fold_mean_direction_balanced_accuracy": float(adjusted_balanced.mean()),
        "fold_balanced_accuracy_std": float(adjusted_balanced.std(ddof=0)),
        "worst_fold_direction_balanced_accuracy": float(adjusted_balanced.min()),
        "outer_folds": fold_results,
        "selected_thresholds": [float(value) for value in selected_thresholds],
        "threshold_stability": {
            "range": threshold_range,
            "std": threshold_std,
            "stable": stable,
            "assessment": "安定" if stable else "変動が大きい",
            "maximum_allowed_range": MAX_THRESHOLD_RANGE,
            "maximum_allowed_std": MAX_THRESHOLD_STD,
        },
        "improved_outer_folds": improved_folds,
        "threshold_method_safe_for_operation": method_safe,
        "threshold_method_safety_reasons": method_reasons,
        "fixed_confirmation_used_for_selection": False,
        "latest_prediction_used_for_selection": False,
        **intervals,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def threshold_logic_metadata() -> dict[str, Any]:
    return {
        "version": THRESHOLD_LOGIC_VERSION,
        "candidate_range": [THRESHOLD_CANDIDATE_MIN, THRESHOLD_CANDIDATE_MAX],
        "candidate_step": THRESHOLD_CANDIDATE_STEP,
        "candidates": list(THRESHOLD_CANDIDATES),
        "selection_metric": THRESHOLD_SELECTION_METRIC,
        "tie_tolerance": THRESHOLD_TIE_TOLERANCE,
        "minimum_prediction_share": MIN_PREDICTION_SHARE,
        "minimum_balanced_improvement": MIN_BALANCED_IMPROVEMENT,
        "minimum_fold_improvement": MIN_FOLD_IMPROVEMENT,
        "minimum_improved_folds": MIN_IMPROVED_FOLDS,
        "inner_folds": INNER_WALK_FORWARD_FOLDS,
        "outer_folds": OUTER_WALK_FORWARD_FOLDS,
        "purge_trading_days": PURGE_TRADING_DAYS,
        "priority": [
            "mean balanced accuracy (higher)",
            "worst-fold balanced accuracy (higher)",
            "balanced accuracy standard deviation (lower)",
            "mean Macro-F1 (higher)",
            "mean accuracy (higher)",
            "distance from 0.50 (lower)",
        ],
        "fallback_conditions": {
            "tiny_improvement": MIN_BALANCED_IMPROVEMENT,
            "insufficient_improved_folds": MIN_IMPROVED_FOLDS,
            "extreme_prediction_mix": MIN_PREDICTION_SHARE,
            "unstable_outer_threshold_range": MAX_THRESHOLD_RANGE,
            "unstable_outer_threshold_std": MAX_THRESHOLD_STD,
        },
    }


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_persistent_threshold() -> dict[str, Any]:
    if not THRESHOLD_SETTINGS_PATH.exists():
        raise ServiceError(
            "保存済みの日経平均判定ラインがありません。",
            "「モデル再評価」を一度実行して、提出版の判定ラインを保存してください。",
        )
    try:
        setting = json.loads(THRESHOLD_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise ServiceError(
            "保存済みの日経平均判定ラインを読み込めませんでした。",
            "モデル設定ファイルを確認し、「モデル再評価」を実行してください。",
        ) from error
    expected = {
        "model_version": MODEL_VERSION,
        "model_settings_version": MODEL_SETTINGS_VERSION,
        "threshold_logic_version": THRESHOLD_LOGIC_VERSION,
    }
    if any(setting.get(key) != value for key, value in expected.items()):
        raise ServiceError(
            "保存済み判定ラインのモデル設定バージョンが一致しません。",
            "明示的な「モデル再評価」を実行してください。",
        )
    threshold = float(setting.get("decision_threshold", -1.0))
    if not 0.0 < threshold < 1.0:
        raise ServiceError("保存済み判定ラインの値が不正です。", "モデル再評価を実行してください。")
    if setting.get("frozen_features") != FROZEN_FEATURES:
        raise ServiceError("保存済み設定の28特徴量が一致しません。", "モデル再評価を実行してください。")
    return setting


def _append_threshold_history(previous: dict[str, Any], current: dict[str, Any], reason: str) -> None:
    if float(previous["decision_threshold"]) == float(current["decision_threshold"]):
        return
    try:
        history = json.loads(THRESHOLD_HISTORY_PATH.read_text(encoding="utf-8")) if THRESHOLD_HISTORY_PATH.exists() else {}
    except (OSError, ValueError, TypeError):
        history = {}
    changes = history.get("changes") if isinstance(history.get("changes"), list) else []
    changes.append(
        {
            "previous_threshold": float(previous["decision_threshold"]),
            "new_threshold": float(current["decision_threshold"]),
            "reevaluated_at": current["determined_at"],
            "data_period": current["used_data_period"],
            "model_settings_version": MODEL_SETTINGS_VERSION,
            "change_reason": reason,
        }
    )
    _atomic_json_write(
        THRESHOLD_HISTORY_PATH,
        {"schema_version": 1, "threshold_logic_version": THRESHOLD_LOGIC_VERSION, "changes": changes},
    )


def save_persistent_threshold(setting: dict[str, Any], reason: str) -> None:
    previous: dict[str, Any] | None = None
    if THRESHOLD_SETTINGS_PATH.exists():
        try:
            previous = json.loads(THRESHOLD_SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            previous = None
    setting["previous_threshold"] = float(previous["decision_threshold"]) if previous and "decision_threshold" in previous else None
    setting["change_reason"] = reason
    if previous and "decision_threshold" in previous:
        _append_threshold_history(previous, setting, reason)
    _atomic_json_write(THRESHOLD_SETTINGS_PATH, setting)
    if not THRESHOLD_HISTORY_PATH.exists():
        _atomic_json_write(
            THRESHOLD_HISTORY_PATH,
            {"schema_version": 1, "threshold_logic_version": THRESHOLD_LOGIC_VERSION, "changes": []},
        )


def reevaluate_and_persist_threshold(selection: pd.DataFrame) -> dict[str, Any]:
    official = nested_walk_forward_threshold_evaluation(selection)
    final_selection = select_threshold_with_inner_walk_forward(selection)
    proposed = float(final_selection["selected_threshold"])
    final_threshold = proposed
    fallback_applied = bool(final_selection["fallback_applied"])
    fallback_reasons: list[str] = []
    if final_selection["fallback_reason"]:
        fallback_reasons.append(str(final_selection["fallback_reason"]))
    if proposed != STANDARD_THRESHOLD and not official["threshold_method_safe_for_operation"]:
        final_threshold = STANDARD_THRESHOLD
        fallback_applied = True
        fallback_reasons.extend(official["threshold_method_safety_reasons"])
    if final_threshold == STANDARD_THRESHOLD and not fallback_reasons:
        fallback_reasons.append("検証の結果、標準の50％を維持しました。")
    determined_at = _now_jst()
    setting = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "model_settings_version": MODEL_SETTINGS_VERSION,
        "threshold_logic_version": THRESHOLD_LOGIC_VERSION,
        "decision_threshold": float(final_threshold),
        "determined_at": determined_at,
        "used_data_period": {
            "start": _date_text(selection.index[0]),
            "end": _date_text(selection.index[-1]),
            "samples": int(len(selection)),
        },
        "normal_prediction_recalculates_threshold": False,
        "model_reevaluation_can_update": True,
        "frozen_model": "Extra Trees",
        "frozen_parameters": dict(FROZEN_EXTRA_TREES_PARAMETERS),
        "frozen_class_balance": "None",
        "frozen_features": list(FROZEN_FEATURES),
        "threshold_logic": threshold_logic_metadata(),
        "outer_evaluation": official,
        "final_inner_selection": final_selection,
        "fallback_applied": bool(fallback_applied),
        "fallback_reason": " ".join(dict.fromkeys(fallback_reasons)),
        "fixed_confirmation_used_for_selection": False,
        "latest_prediction_used_for_selection": False,
    }
    reason = (
        setting["fallback_reason"]
        if fallback_applied or final_threshold == STANDARD_THRESHOLD
        else "モデル選択期間内の二重時系列検証で、安全条件を満たした判定ラインを採用しました。"
    )
    save_persistent_threshold(setting, reason)
    return setting


def _advantage_text(gap: float) -> tuple[str, str]:
    if gap >= 0.02:
        return f"単純な比較方法より＋{gap * 100:.1f}ポイント", "clear"
    if gap > 0:
        return "単純な比較方法との差は小さく、優位性は限定的です", "limited"
    return "単純な比較方法を上回る結果は確認できませんでした", "none"


def evaluate_fixed_confirmation(
    selection: pd.DataFrame,
    confirmation: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    train = selection.iloc[:-PURGE_TRADING_DAYS].copy()
    probability, importance, preprocessing = _fit_frozen_extra_trees(train, confirmation)
    y_train = train["target_direction"].to_numpy(dtype=int)
    y_true = confirmation["target_direction"].to_numpy(dtype=int)
    adjusted_prediction = direction_from_probability(probability, threshold)
    standard_prediction = direction_from_probability(probability, STANDARD_THRESHOLD)
    adjusted = _classification_metrics(y_true, adjusted_prediction)
    standard = _classification_metrics(y_true, standard_prediction)
    majority_prediction = majority_baseline_predictions(y_train, len(confirmation))
    momentum_prediction = momentum_baseline_predictions(confirmation)
    majority = _classification_metrics(y_true, majority_prediction)
    momentum = _classification_metrics(y_true, momentum_prediction)
    best_baseline_name = "多数派基準" if majority["direction_accuracy"] >= momentum["direction_accuracy"] else "直近方向継続"
    best_prediction = majority_prediction if best_baseline_name == "多数派基準" else momentum_prediction
    intervals = moving_block_bootstrap_intervals(adjusted_prediction == y_true, best_prediction == y_true)
    six = _six_class_evaluation(train, confirmation, FROZEN_FEATURES)
    best_accuracy = max(majority["direction_accuracy"], momentum["direction_accuracy"])
    gap = float(adjusted["direction_accuracy"] - best_accuracy)
    advantage_message, advantage_level = _advantage_text(gap)
    majority_direction = "上昇" if int(majority_prediction[0]) == 1 else "下落"
    return {
        "evaluation_type": "fixed_confirmation_not_used_for_threshold_selection",
        "period": {"start": _date_text(confirmation.index[0]), "end": _date_text(confirmation.index[-1])},
        "training_period": {"start": _date_text(train.index[0]), "end": _date_text(train.index[-1])},
        "training_samples": int(len(train)),
        "selection_samples_before_purge": int(len(selection)),
        "validation_samples": int(len(confirmation)),
        "decision_threshold": float(threshold),
        "purge_trading_days": PURGE_TRADING_DAYS,
        "selected_threshold_metrics": adjusted,
        "fixed_50_metrics": standard,
        "direction_accuracy": adjusted["direction_accuracy"],
        "direction_balanced_accuracy": adjusted["direction_balanced_accuracy"],
        "direction_macro_f1": adjusted["direction_macro_f1"],
        "correct_predictions": adjusted["correct_predictions"],
        "up_recall": adjusted["up_recall"],
        "down_recall": adjusted["down_recall"],
        "predicted_up": adjusted["predicted_up"],
        "predicted_down": adjusted["predicted_down"],
        "actual_up": adjusted["actual_up"],
        "actual_down": adjusted["actual_down"],
        "confusion_matrix": adjusted["confusion_matrix"],
        "majority_baseline": majority,
        "majority_baseline_accuracy": majority["direction_accuracy"],
        "majority_baseline_balanced_accuracy": majority["direction_balanced_accuracy"],
        "majority_direction": majority_direction,
        "momentum_baseline": momentum,
        "momentum_baseline_accuracy": momentum["direction_accuracy"],
        "momentum_baseline_balanced_accuracy": momentum["direction_balanced_accuracy"],
        "best_baseline_name": best_baseline_name,
        "best_baseline_accuracy": best_accuracy,
        "baseline_gap": gap,
        "advantage_message": advantage_message,
        "advantage_level": advantage_level,
        "accuracy_change_from_fixed_50": float(adjusted["direction_accuracy"] - standard["direction_accuracy"]),
        "balanced_accuracy_change_from_fixed_50": float(
            adjusted["direction_balanced_accuracy"] - standard["direction_balanced_accuracy"]
        ),
        "macro_f1_change_from_fixed_50": float(adjusted["direction_macro_f1"] - standard["direction_macro_f1"]),
        "feature_importance": [
            {"feature": feature, "mean_importance": float(value), "std_importance": 0.0}
            for feature, value in sorted(importance.items(), key=lambda item: (-item[1], item[0]))
        ],
        "preprocessing": preprocessing,
        "fixed_confirmation_used_for_threshold_selection": False,
        "note": "判定ラインの決定には使用していない固定確認期間です。",
        **six,
        **intervals,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def fit_latest_prediction(data: PredictionData, threshold_setting: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    training = data.training_frame.copy()
    inference_date = data.inference_row.index[0]
    if inference_date in training.index:
        raise ServiceError("最新推論行が学習データへ混入しています。")
    probability, importance, preprocessing = _fit_frozen_extra_trees(training, data.inference_row)
    up_probability = float(np.clip(probability[0], 0.0, 1.0))
    down_probability = float(1.0 - up_probability)
    threshold = float(threshold_setting["decision_threshold"])
    direction_id = int(up_probability >= threshold)
    six = _six_latest_prediction(training, data.inference_row, FROZEN_FEATURES)
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
        "decision_threshold": threshold,
        "threshold_margin": float(up_probability - threshold),
        "threshold_margin_points": float((up_probability - threshold) * 100.0),
        "strong_crash": strong_crash,
        "training_samples": int(len(training)),
        "training_end": _date_text(training.index[-1]),
        "inference_date": _date_text(inference_date),
        "inference_row_in_training": False,
        "selected_model": {
            "id": "extra_trees_selected",
            "label": "Extra Trees",
            "kind": "extra_trees",
            "class_balance": "None",
        },
        "selected_feature_count": len(FROZEN_FEATURES),
        "selected_features": list(FROZEN_FEATURES),
        "model_settings_version": MODEL_SETTINGS_VERSION,
        "threshold_determined_at": threshold_setting["determined_at"],
        "feature_importance": [
            {"feature": feature, "mean_importance": float(value), "std_importance": 0.0}
            for feature, value in sorted(importance.items(), key=lambda item: (-item[1], item[0]))
        ],
        "preprocessing": preprocessing,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def frozen_configuration(data: PredictionData, evaluation: dict[str, Any], setting: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_model": {
            "id": "extra_trees_selected",
            "label": "Extra Trees",
            "kind": "extra_trees",
            "class_balance": "None",
        },
        "selected_features": list(FROZEN_FEATURES),
        "excluded_features": [feature for feature in data.feature_columns if feature not in FROZEN_FEATURES],
        "selected_factors": list(FROZEN_SELECTED_FACTORS),
        "excluded_factors": list(FROZEN_EXCLUDED_FACTORS),
        "feature_importance": evaluation["feature_importance"],
        "class_balance": "None",
        "class_balance_label": "クラス補正なし",
        "model_setting": {
            "name": "extra_trees",
            "label": "Extra Trees（固定済み28特徴量）",
            "parameters": dict(FROZEN_EXTRA_TREES_PARAMETERS),
        },
        "adoption_reason": "前工程の時系列検証で採用済みのExtra Treesと28特徴量を固定しています。",
        "ensemble": {"adopted": False, "components": [], "method": None},
        "selection_period": setting["used_data_period"],
        "walk_forward_folds": setting["outer_evaluation"]["outer_folds"],
        "walk_forward_summary": {
            "direction_accuracy": setting["outer_evaluation"]["direction_accuracy"],
            "direction_balanced_accuracy": setting["outer_evaluation"]["direction_balanced_accuracy"],
            "direction_macro_f1": setting["outer_evaluation"]["direction_macro_f1"],
        },
        "candidate_results": [],
        "class_balance_screening": [],
        "feature_selection_by_fold": [],
        "fixed_evaluation_used_for_selection": False,
        "model_tie_tolerance": None,
        "threshold_setting": setting,
    }


def _cache_descriptor(
    data: PredictionData,
    metadata: dict[str, Any],
    setting: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_version": MODEL_VERSION,
        "model_settings_version": MODEL_SETTINGS_VERSION,
        "fixed_data_start": metadata["fixed_data_start"],
        "fixed_data_end": metadata["fixed_data_end"],
        "model_selection_period": metadata["selection_period"],
        "fixed_confirmation_period": metadata["fixed_evaluation_period"],
        "external_alignment_version": EXTERNAL_ALIGNMENT_VERSION,
        "feature_definition_version": FEATURE_DEFINITION_VERSION,
        "direction_target_definition": "close_t_plus_5_gt_close_t",
        "threshold_logic": threshold_logic_metadata(),
        "saved_threshold": setting["decision_threshold"],
        "threshold_determined_at": setting["determined_at"],
        "purge_trading_days": PURGE_TRADING_DAYS,
        "frozen_model": "Extra Trees",
        "frozen_parameters": FROZEN_EXTRA_TREES_PARAMETERS,
        "frozen_features": FROZEN_FEATURES,
        "class_balance": "None",
        "fred_used": False,
        "available_external_factors": metadata["available_external_factors"],
        "data_signature": _data_signature(data, metadata),
    }


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _cache_path(key: str) -> Path:
    return OUTPUT_DIR / f"nikkei_threshold_result_{key}.json"


def _load_cache(key: str) -> dict[str, Any] | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if payload.get("model_version") != MODEL_VERSION or payload.get("cache_key") != key:
        return None
    return payload


def _save_cache(key: str, result: dict[str, Any], descriptor: dict[str, Any]) -> str:
    created_at = _now_jst()
    _atomic_json_write(
        _cache_path(key),
        {
            "model_version": MODEL_VERSION,
            "cache_key": key,
            "created_at": created_at,
            "descriptor": descriptor,
            "result": result,
        },
    )
    return created_at


def run_nikkei_threshold_analysis(
    data: PredictionData,
    preparation_metadata: dict[str, Any],
    model_reevaluation: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    selection, confirmation = fixed_period_split(data.training_frame)
    calibration_started = time.perf_counter()
    if model_reevaluation:
        setting = reevaluate_and_persist_threshold(selection)
        calibration_seconds = time.perf_counter() - calibration_started
    else:
        setting = load_persistent_threshold()
        calibration_seconds = 0.0
    descriptor = _cache_descriptor(data, preparation_metadata, setting)
    cache_key = _hash_payload(descriptor)
    lookup_started = time.perf_counter()
    cached = None if model_reevaluation else _load_cache(cache_key)
    lookup_seconds = time.perf_counter() - lookup_started
    if cached is not None:
        result = dict(cached["result"])
        result["preparation_metadata"] = preparation_metadata
        result["total_seconds"] = float(time.perf_counter() - started)
        result["cache"] = {
            "used": True,
            "key": cache_key,
            "created_at": cached["created_at"],
            "model_reevaluation": False,
            "lookup_seconds": float(lookup_seconds),
        }
        return result

    confirmation_started = time.perf_counter()
    fixed_confirmation = evaluate_fixed_confirmation(
        selection,
        confirmation,
        float(setting["decision_threshold"]),
    )
    confirmation_seconds = time.perf_counter() - confirmation_started
    latest_started = time.perf_counter()
    latest = fit_latest_prediction(data, setting)
    latest_seconds = time.perf_counter() - latest_started
    configuration = frozen_configuration(data, fixed_confirmation, setting)
    result = {
        "version": MODEL_VERSION,
        "configuration": configuration,
        "official_evaluation": setting["outer_evaluation"],
        "fixed_confirmation": fixed_confirmation,
        "latest_prediction": latest,
        "threshold_setting": setting,
        "preparation_metadata": preparation_metadata,
        "threshold_calibration_seconds": float(calibration_seconds),
        "fixed_confirmation_seconds": float(confirmation_seconds),
        "latest_inference_seconds": float(latest_seconds),
        "total_seconds": float(time.perf_counter() - started),
    }
    created_at = _save_cache(cache_key, result, descriptor)
    result["cache"] = {
        "used": False,
        "key": cache_key,
        "created_at": created_at,
        "model_reevaluation": bool(model_reevaluation),
        "lookup_seconds": float(lookup_seconds),
    }
    return result
