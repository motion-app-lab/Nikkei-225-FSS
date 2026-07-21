from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .common import (
    CLASS_ORDER,
    OUTPUT_DIR,
    UP_CLASS_IDS,
    PredictionData,
    ServiceError,
    _six_probabilities,
    classify_change,
)


MODEL_SELECTION_VERSION = "nikkei_nested_walk_forward_v3"
FINAL_HOLDOUT_RATIO = 0.20
WALK_FORWARD_FOLDS = 3
STRONG_CRASH_MIN_PROBABILITY = 0.45
STRONG_CRASH_MIN_MARGIN = 0.15

FACTOR_DEFINITIONS: OrderedDict[str, dict[str, Any]] = OrderedDict(
    [
        ("nikkei_price", {"label": "日経平均の価格・リターン", "columns": ["open", "high", "low", "price", "return_1d", "return_5d"]}),
        ("volume", {"label": "出来高", "columns": ["volume", "volume_change"]}),
        ("volatility", {"label": "ボラティリティ", "columns": ["volatility_5d", "volatility_20d", "overheat_score"]}),
        ("rsi", {"label": "RSI", "columns": ["rsi14"]}),
        ("channel", {"label": "価格チャネル", "columns": ["ch_trend", "ch_upper", "ch_lower", "ch_pos"]}),
        ("spx", {"label": "S&P 500", "columns": ["spx", "spx_ret"]}),
        ("dow", {"label": "NYダウ", "columns": ["dow", "dow_ret"]}),
        ("nasdaq", {"label": "NASDAQ", "columns": ["nasdaq", "nasdaq_ret"]}),
        ("vix", {"label": "VIX", "columns": ["vix", "vix_ret"]}),
        ("sox", {"label": "SOX", "columns": ["sox", "sox_ret"]}),
        ("nvda", {"label": "NVIDIA", "columns": ["nvda", "nvda_ret"]}),
        ("usdjpy", {"label": "ドル円", "columns": ["usdjpy", "usdjpy_ret"]}),
        ("oil", {"label": "原油", "columns": ["oil", "oil_ret"]}),
        ("gold", {"label": "金", "columns": ["gold", "gold_ret"]}),
        ("btc", {"label": "ビットコイン", "columns": ["btc", "btc_ret"]}),
        ("pce", {"label": "PCE", "columns": ["pce", "pce_ret"]}),
    ]
)

CORE_FACTOR_IDS = ["nikkei_price", "volume", "volatility", "rsi", "channel"]

SCREENING_SETTING = {
    "name": "screening",
    "label": "ファクター選別用軽量設定",
    "iterations": 90,
    "depth": 4,
    "learning_rate": 0.06,
    "l2_leaf_reg": 4.0,
}

PARAMETER_CANDIDATES = [
    {
        "name": "standard",
        "label": "標準設定",
        "iterations": 130,
        "depth": 5,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
    },
    {
        "name": "shallow",
        "label": "浅い木",
        "iterations": 150,
        "depth": 4,
        "learning_rate": 0.05,
        "l2_leaf_reg": 4.0,
    },
    {
        "name": "regularized",
        "label": "正則化を強化",
        "iterations": 150,
        "depth": 5,
        "learning_rate": 0.04,
        "l2_leaf_reg": 8.0,
    },
    {
        "name": "low_learning_rate",
        "label": "低学習率",
        "iterations": 190,
        "depth": 5,
        "learning_rate": 0.03,
        "l2_leaf_reg": 5.0,
    },
]


def _now_jst() -> str:
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")


def _label_ids(changes: pd.Series) -> np.ndarray:
    return np.array([CLASS_ORDER.index(classify_change(float(value))) for value in changes], dtype=int)


def _direction_ids(changes: pd.Series) -> np.ndarray:
    return np.isin(_label_ids(changes), list(UP_CLASS_IDS)).astype(int)


def split_selection_and_holdout(
    frame: pd.DataFrame,
    holdout_ratio: float = FINAL_HOLDOUT_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """最新側を未使用の最終評価期間として切り離す。"""
    if len(frame) < 400:
        raise ServiceError("二重の時系列検証に必要なデータが不足しています。")
    split_index = int(len(frame) * (1.0 - holdout_ratio))
    split_index = max(280, min(split_index, len(frame) - 80))
    selection = frame.iloc[:split_index].copy()
    holdout = frame.iloc[split_index:].copy()
    if selection.index.max() >= holdout.index.min():
        raise ServiceError("モデル選択期間と最終評価期間を正しく分離できませんでした。")
    return selection, holdout


def expanding_walk_forward_splits(
    frame: pd.DataFrame,
    n_splits: int = WALK_FORWARD_FOLDS,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """古い学習期間を拡大し、その直後だけを検証する。"""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    minimum_train = max(240, int(len(frame) * 0.52))
    validation_size = (len(frame) - minimum_train) // n_splits
    if validation_size < 45:
        raise ServiceError("ウォークフォワード検証に必要な期間が不足しています。")
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for fold_index in range(n_splits):
        train_end = minimum_train + validation_size * fold_index
        validation_end = len(frame) if fold_index == n_splits - 1 else train_end + validation_size
        train = frame.iloc[:train_end].copy()
        validation = frame.iloc[train_end:validation_end].copy()
        if train.index.max() >= validation.index.min():
            raise ServiceError("ウォークフォワードの時系列順序が不正です。")
        folds.append((train, validation))
    return folds


def majority_baseline_predictions(train_direction: np.ndarray, validation_size: int) -> np.ndarray:
    values = np.asarray(train_direction, dtype=int)
    if values.size == 0:
        raise ValueError("train_direction is empty")
    majority = int(np.argmax(np.bincount(values, minlength=2)))
    return np.full(validation_size, majority, dtype=int)


def momentum_baseline_predictions(validation: pd.DataFrame) -> np.ndarray:
    """予測時点までに判明している直近5取引日リターンだけを使う。"""
    if "return_5d" not in validation.columns:
        raise ValueError("return_5d is required")
    return (validation["return_5d"].to_numpy(dtype=float) >= 0.0).astype(int)


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls: list[float] = []
    for class_id in (0, 1):
        mask = y_true == class_id
        if mask.any():
            recalls.append(float(np.mean(y_pred[mask] == class_id)))
    return float(np.mean(recalls)) if recalls else 0.0


def _new_model(setting: dict[str, Any], task: str, class_balance: str):
    from catboost import CatBoostClassifier

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


def _fit_train_validate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    setting: dict[str, Any],
    class_balance: str,
    include_six_class: bool,
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[features])
    x_validation = scaler.transform(validation[features])
    y_train_direction = _direction_ids(train["target_change"])
    y_validation_direction = _direction_ids(validation["target_change"])

    if len(np.unique(y_train_direction)) < 2:
        direction_prediction = majority_baseline_predictions(y_train_direction, len(validation))
        importance = np.zeros(len(features), dtype=float)
    else:
        direction_model = _new_model(setting, "direction", class_balance)
        direction_model.fit(x_train, y_train_direction)
        direction_prediction = direction_model.predict(x_validation).astype(int).reshape(-1)
        importance = np.asarray(direction_model.get_feature_importance(), dtype=float)

    y_validation_six = _label_ids(validation["target_change"])
    six_prediction: np.ndarray | None = None
    if include_six_class:
        y_train_six = _label_ids(train["target_change"])
        if len(np.unique(y_train_six)) < 2:
            six_prediction = np.full(len(validation), int(y_train_six[0]), dtype=int)
        else:
            six_model = _new_model(setting, "six", class_balance)
            six_model.fit(x_train, y_train_six)
            six_prediction = six_model.predict(x_validation).astype(int).reshape(-1)

    majority_prediction = majority_baseline_predictions(y_train_direction, len(validation))
    momentum_prediction = momentum_baseline_predictions(validation)
    six_distribution = {
        label: int(np.sum(y_validation_six == class_id))
        for class_id, label in enumerate(CLASS_ORDER)
    }
    result: dict[str, Any] = {
        "direction_accuracy": float(accuracy_score(y_validation_direction, direction_prediction)),
        "direction_balanced_accuracy": _balanced_accuracy(y_validation_direction, direction_prediction),
        "six_class_accuracy": None,
        "six_class_macro_f1": None,
        "validation_samples": int(len(validation)),
        "direction_distribution": {
            "up": int(np.sum(y_validation_direction == 1)),
            "non_up": int(np.sum(y_validation_direction == 0)),
        },
        "six_class_distribution": six_distribution,
        "majority_baseline_accuracy": float(accuracy_score(y_validation_direction, majority_prediction)),
        "momentum_baseline_accuracy": float(accuracy_score(y_validation_direction, momentum_prediction)),
        "feature_importance": {feature: float(value) for feature, value in zip(features, importance)},
    }
    if six_prediction is not None:
        result["six_class_accuracy"] = float(accuracy_score(y_validation_six, six_prediction))
        result["six_class_macro_f1"] = float(
            f1_score(
                y_validation_six,
                six_prediction,
                labels=list(range(len(CLASS_ORDER))),
                average="macro",
                zero_division=0,
            )
        )
    return result


def _mean(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def evaluate_walk_forward(
    selection_frame: pd.DataFrame,
    features: list[str],
    setting: dict[str, Any],
    class_balance: str = "None",
    include_six_class: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    folds: list[dict[str, Any]] = []
    importance_rows: list[dict[str, float]] = []
    for fold_number, (train, validation) in enumerate(expanding_walk_forward_splits(selection_frame), start=1):
        metrics = _fit_train_validate(
            train,
            validation,
            features,
            setting,
            class_balance,
            include_six_class,
        )
        importance_rows.append(metrics.pop("feature_importance"))
        folds.append(
            {
                "fold": fold_number,
                "train_start": train.index[0].strftime("%Y-%m-%d"),
                "train_end": train.index[-1].strftime("%Y-%m-%d"),
                "validation_start": validation.index[0].strftime("%Y-%m-%d"),
                "validation_end": validation.index[-1].strftime("%Y-%m-%d"),
                "training_samples": int(len(train)),
                **metrics,
            }
        )

    direction_values = [fold["direction_accuracy"] for fold in folds]
    importance = []
    for feature in features:
        values = [row.get(feature, 0.0) for row in importance_rows]
        importance.append(
            {
                "feature": feature,
                "mean_importance": float(np.mean(values)),
                "std_importance": float(np.std(values)),
            }
        )
    importance.sort(key=lambda item: item["mean_importance"], reverse=True)
    return {
        "features": list(features),
        "setting_name": setting["name"],
        "class_balance": class_balance,
        "folds": folds,
        "summary": {
            "direction_accuracy_mean": float(np.mean(direction_values)),
            "direction_accuracy_std": float(np.std(direction_values)),
            "direction_balanced_accuracy_mean": float(np.mean([fold["direction_balanced_accuracy"] for fold in folds])),
            "six_class_accuracy_mean": _mean(fold["six_class_accuracy"] for fold in folds),
            "six_class_macro_f1_mean": _mean(fold["six_class_macro_f1"] for fold in folds),
            "majority_baseline_accuracy_mean": float(np.mean([fold["majority_baseline_accuracy"] for fold in folds])),
            "momentum_baseline_accuracy_mean": float(np.mean([fold["momentum_baseline_accuracy"] for fold in folds])),
        },
        "feature_importance": importance,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _candidate_rank(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    summary = candidate["evaluation"]["summary"]
    return (
        float(summary["direction_accuracy_mean"]),
        float(summary["direction_balanced_accuracy_mean"]),
        float(summary.get("six_class_macro_f1_mean") or 0.0),
        -float(summary["direction_accuracy_std"]),
    )


def _stable_improvement(candidate: dict[str, Any], fallback: dict[str, Any]) -> bool:
    candidate_summary = candidate["evaluation"]["summary"]
    fallback_summary = fallback["evaluation"]["summary"]
    direction_gain = candidate_summary["direction_accuracy_mean"] - fallback_summary["direction_accuracy_mean"]
    balance_gain = candidate_summary["direction_balanced_accuracy_mean"] - fallback_summary["direction_balanced_accuracy_mean"]
    fold_wins = sum(
        candidate_fold["direction_accuracy"] > fallback_fold["direction_accuracy"]
        for candidate_fold, fallback_fold in zip(candidate["evaluation"]["folds"], fallback["evaluation"]["folds"])
    )
    return bool(
        fold_wins >= 2
        and balance_gain >= -0.005
        and (direction_gain >= 0.003 or (direction_gain >= 0.0 and balance_gain >= 0.01))
    )


def choose_reduced_candidate_or_fallback(
    candidates: list[dict[str, Any]],
    fallback_name: str,
) -> dict[str, Any]:
    fallback = next(candidate for candidate in candidates if candidate["name"] == fallback_name)
    best = max(candidates, key=_candidate_rank)
    if best["name"] == fallback_name:
        return fallback
    return best if _stable_improvement(best, fallback) else fallback


def _available_factor_ids(feature_columns: list[str]) -> list[str]:
    available = set(feature_columns)
    return [
        factor_id
        for factor_id, definition in FACTOR_DEFINITIONS.items()
        if any(column in available for column in definition["columns"])
    ]


def _features_for_factors(factor_ids: Iterable[str], feature_columns: list[str]) -> list[str]:
    allowed = set(feature_columns)
    selected: list[str] = []
    for factor_id in factor_ids:
        for column in FACTOR_DEFINITIONS[factor_id]["columns"]:
            if column in allowed and column not in selected:
                selected.append(column)
    return selected


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": candidate["name"],
        "label": candidate.get("label", candidate["name"]),
        "factor_ids": candidate.get("factor_ids"),
        "feature_count": len(candidate["evaluation"]["features"]),
        "summary": candidate["evaluation"]["summary"],
    }


def select_model_configuration(
    selection_frame: pd.DataFrame,
    all_feature_columns: list[str],
) -> dict[str, Any]:
    """最終評価期間を受け取らず、モデル選択用期間だけで構成を決める。"""
    available_factors = _available_factor_ids(all_feature_columns)
    core_factors = [factor for factor in CORE_FACTOR_IDS if factor in available_factors]
    external_factors = [factor for factor in available_factors if factor not in core_factors]
    evaluation_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def evaluate(
        features: list[str],
        setting: dict[str, Any],
        class_balance: str = "None",
        include_six: bool = True,
    ) -> dict[str, Any]:
        key = (tuple(features), setting["name"], class_balance, include_six)
        if key not in evaluation_cache:
            evaluation_cache[key] = evaluate_walk_forward(
                selection_frame,
                features,
                setting,
                class_balance,
                include_six,
            )
        return evaluation_cache[key]

    core_features = _features_for_factors(core_factors, all_feature_columns)
    factor_screening: list[dict[str, Any]] = []
    for factor_id in external_factors:
        screening_features = _features_for_factors(core_factors + [factor_id], all_feature_columns)
        screening = evaluate(screening_features, SCREENING_SETTING, include_six=False)
        factor_screening.append(
            {
                "factor_id": factor_id,
                "label": FACTOR_DEFINITIONS[factor_id]["label"],
                "summary": screening["summary"],
            }
        )
    factor_screening.sort(
        key=lambda item: (
            item["summary"]["direction_accuracy_mean"],
            item["summary"]["direction_balanced_accuracy_mean"],
            -item["summary"]["direction_accuracy_std"],
        ),
        reverse=True,
    )
    ranked_external = [item["factor_id"] for item in factor_screening]

    factor_sets: list[tuple[str, str, list[str]]] = [
        ("all_factors", "全ファクター", available_factors),
        ("core_factors", "日経平均の内部ファクター", core_factors),
    ]
    if ranked_external:
        factor_sets.append(("core_plus_top3", "内部＋上位3外部ファクター", core_factors + ranked_external[:3]))
        factor_sets.append(("core_plus_top5", "内部＋上位5外部ファクター", core_factors + ranked_external[:5]))

    factor_candidates: list[dict[str, Any]] = []
    seen_features: set[tuple[str, ...]] = set()
    standard_setting = PARAMETER_CANDIDATES[0]
    for name, label, factor_ids in factor_sets:
        features = _features_for_factors(factor_ids, all_feature_columns)
        signature = tuple(features)
        if not features or signature in seen_features:
            continue
        seen_features.add(signature)
        factor_candidates.append(
            {
                "name": name,
                "label": label,
                "factor_ids": factor_ids,
                "evaluation": evaluate(features, standard_setting),
            }
        )
    selected_factor_candidate = choose_reduced_candidate_or_fallback(factor_candidates, "all_factors")

    factor_features = selected_factor_candidate["evaluation"]["features"]
    importance_ranking = [item["feature"] for item in selected_factor_candidate["evaluation"]["feature_importance"]]
    feature_sizes = []
    for size in (len(factor_features), 30, 20, 15):
        normalized_size = min(size, len(factor_features))
        if normalized_size >= 5 and normalized_size not in feature_sizes:
            feature_sizes.append(normalized_size)
    feature_candidates: list[dict[str, Any]] = []
    for size in feature_sizes:
        features = factor_features if size == len(factor_features) else importance_ranking[:size]
        name = "all_selected_features" if size == len(factor_features) else f"top_{size}_features"
        feature_candidates.append(
            {
                "name": name,
                "label": "選択ファクターの全特徴量" if size == len(factor_features) else f"重要度上位{size}特徴量",
                "evaluation": evaluate(features, standard_setting),
            }
        )
    selected_feature_candidate = choose_reduced_candidate_or_fallback(feature_candidates, "all_selected_features")
    selected_features = selected_feature_candidate["evaluation"]["features"]

    unbalanced_candidate = {
        "name": "none",
        "label": "クラス補正なし",
        "evaluation": evaluate(selected_features, standard_setting, "None"),
    }
    balanced_candidate = {
        "name": "balanced",
        "label": "CatBoost自動クラス補正",
        "evaluation": evaluate(selected_features, standard_setting, "Balanced"),
    }
    balance_candidates = [unbalanced_candidate, balanced_candidate]
    selected_balance_candidate = (
        balanced_candidate
        if _stable_improvement(balanced_candidate, unbalanced_candidate)
        else unbalanced_candidate
    )
    selected_balance = selected_balance_candidate["evaluation"]["class_balance"]

    parameter_candidates: list[dict[str, Any]] = []
    for setting in PARAMETER_CANDIDATES:
        parameter_candidates.append(
            {
                "name": setting["name"],
                "label": setting["label"],
                "setting": setting,
                "evaluation": evaluate(selected_features, setting, selected_balance),
            }
        )
    standard_parameter = parameter_candidates[0]
    best_parameter = max(parameter_candidates, key=_candidate_rank)
    standard_summary = standard_parameter["evaluation"]["summary"]
    best_summary = best_parameter["evaluation"]["summary"]
    if (
        best_parameter["name"] != "standard"
        and best_summary["direction_accuracy_mean"] - standard_summary["direction_accuracy_mean"] < 0.002
        and best_summary["direction_balanced_accuracy_mean"] - standard_summary["direction_balanced_accuracy_mean"] < 0.005
    ):
        best_parameter = standard_parameter

    final_evaluation = best_parameter["evaluation"]
    final_factor_ids = [
        factor_id
        for factor_id in available_factors
        if any(feature in selected_features for feature in FACTOR_DEFINITIONS[factor_id]["columns"])
    ]
    excluded_factor_ids = [factor for factor in available_factors if factor not in final_factor_ids]
    return {
        "selection_period": {
            "start": selection_frame.index[0].strftime("%Y-%m-%d"),
            "end": selection_frame.index[-1].strftime("%Y-%m-%d"),
            "samples": int(len(selection_frame)),
        },
        "selected_factor_ids": final_factor_ids,
        "selected_factors": [FACTOR_DEFINITIONS[factor]["label"] for factor in final_factor_ids],
        "excluded_factor_ids": excluded_factor_ids,
        "excluded_factors": [FACTOR_DEFINITIONS[factor]["label"] for factor in excluded_factor_ids],
        "selected_features": selected_features,
        "excluded_features": [feature for feature in all_feature_columns if feature not in selected_features],
        "feature_importance": final_evaluation["feature_importance"],
        "class_balance": selected_balance,
        "class_balance_label": "CatBoost自動クラス補正" if selected_balance == "Balanced" else "補正なし",
        "catboost_setting": best_parameter["setting"],
        "walk_forward_folds": final_evaluation["folds"],
        "walk_forward_summary": final_evaluation["summary"],
        "candidate_results": {
            "factor_screening": factor_screening,
            "factor_candidates": [_candidate_summary(candidate) for candidate in factor_candidates],
            "feature_candidates": [_candidate_summary(candidate) for candidate in feature_candidates],
            "class_balance_candidates": [_candidate_summary(candidate) for candidate in balance_candidates],
            "parameter_candidates": [_candidate_summary(candidate) for candidate in parameter_candidates],
        },
    }


def evaluate_final_holdout(
    selection_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """構成決定後にだけ、最新側の未使用期間を一度評価する。"""
    metrics = _fit_train_validate(
        selection_frame,
        holdout_frame,
        configuration["selected_features"],
        configuration["catboost_setting"],
        configuration["class_balance"],
        include_six_class=True,
    )
    metrics.pop("feature_importance", None)
    best_baseline_accuracy = max(
        metrics["majority_baseline_accuracy"],
        metrics["momentum_baseline_accuracy"],
    )
    best_baseline_name = (
        "多数派基準モデル"
        if metrics["majority_baseline_accuracy"] >= metrics["momentum_baseline_accuracy"]
        else "直近方向継続モデル"
    )
    gap = metrics["direction_accuracy"] - best_baseline_accuracy
    if gap >= 0.02:
        advantage = f"基準モデルより＋{gap * 100:.1f}ポイント"
        advantage_level = "clear"
    elif gap > 0:
        advantage = "基準モデルとの差は小さく、優位性は限定的です"
        advantage_level = "limited"
    else:
        advantage = "基準モデルに対する優位性は確認できませんでした"
        advantage_level = "none"
    return {
        "period": {
            "start": holdout_frame.index[0].strftime("%Y-%m-%d"),
            "end": holdout_frame.index[-1].strftime("%Y-%m-%d"),
        },
        "training_samples": int(len(selection_frame)),
        **metrics,
        "best_baseline_accuracy": float(best_baseline_accuracy),
        "best_baseline_name": best_baseline_name,
        "baseline_gap": float(gap),
        "advantage_message": advantage,
        "advantage_level": advantage_level,
        "holdout_used_for_selection": False,
    }


def _fit_latest_prediction(data: PredictionData, configuration: dict[str, Any]) -> dict[str, Any]:
    from sklearn.preprocessing import StandardScaler

    features = configuration["selected_features"]
    setting = configuration["catboost_setting"]
    class_balance = configuration["class_balance"]
    training = data.training_frame
    if data.inference_row.index[0] in training.index:
        raise ServiceError("最新推論行が学習データへ混入しています。")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(training[features])
    x_latest = scaler.transform(data.inference_row[features])
    y_direction = _direction_ids(training["target_change"])
    if len(np.unique(y_direction)) < 2:
        raise ServiceError("方向モデルに必要な2クラスが揃っていません。")
    direction_model = _new_model(setting, "direction", class_balance)
    direction_model.fit(x_train, y_direction)
    direction_raw = direction_model.predict_proba(x_latest)[0]
    up_probability = 0.0
    for column, class_id in enumerate(np.asarray(direction_model.classes_).reshape(-1)):
        if int(class_id) == 1:
            up_probability = float(direction_raw[column])
    direction_id = int(up_probability >= 0.5)

    y_six = _label_ids(training["target_change"])
    if len(np.unique(y_six)) < 2:
        raise ServiceError("6段階モデルに必要なクラスが不足しています。")
    six_model = _new_model(setting, "six", class_balance)
    six_model.fit(x_train, y_six)
    raw_six = six_model.predict_proba(x_latest)[0]
    probabilities = _six_probabilities(six_model, raw_six)
    predicted_id = int(np.argmax(probabilities))
    crash_probability = float(probabilities[CLASS_ORDER.index("急落")])
    second_probability = float(max(probability for index, probability in enumerate(probabilities) if index != CLASS_ORDER.index("急落")))
    strong_crash = bool(
        predicted_id == CLASS_ORDER.index("急落")
        and crash_probability >= STRONG_CRASH_MIN_PROBABILITY
        and crash_probability - second_probability >= STRONG_CRASH_MIN_MARGIN
    )
    if direction_id == 1:
        direction = "上昇傾向"
        direction_key = "up"
    else:
        direction = "下落傾向"
        direction_key = "strong_down" if strong_crash else "down"
    return {
        "prediction_label": CLASS_ORDER[predicted_id],
        "probabilities": probabilities,
        "top_probability": float(probabilities[predicted_id]),
        "direction": direction,
        "direction_key": direction_key,
        "direction_probability": float(up_probability if direction_id == 1 else 1.0 - up_probability),
        "up_probability": float(up_probability),
        "strong_crash": strong_crash,
        "crash_probability": crash_probability,
        "crash_margin": float(crash_probability - second_probability),
    }


def model_selection_cache_key(data: PredictionData) -> str:
    hasher = hashlib.sha256()
    hasher.update(MODEL_SELECTION_VERSION.encode("utf-8"))
    hasher.update(json.dumps(PARAMETER_CANDIDATES, sort_keys=True).encode("utf-8"))
    # 外部系列の調整済み価格は同一取得内容でも複数行で1 ULP程度揺れる。
    # 列の有無は署名へ含め、数値本体は安定している日経平均系列と正解だけを使う。
    # 基準日・行数・日経平均値が変われば必ず別署名になる。
    hasher.update(json.dumps(data.feature_columns, ensure_ascii=False).encode("utf-8"))
    stable_columns = [
        column
        for column in ("open", "high", "low", "price", "volume", "return_1d", "return_5d", "target_change")
        if column in data.training_frame.columns
    ]
    inference_columns = [column for column in stable_columns if column != "target_change"]
    normalized_training = data.training_frame[stable_columns].round(4)
    normalized_inference = data.inference_row[inference_columns].round(4)
    hashed = pd.util.hash_pandas_object(normalized_training, index=True).to_numpy()
    hasher.update(hashed.tobytes())
    hasher.update(pd.util.hash_pandas_object(normalized_inference, index=True).to_numpy().tobytes())
    return hasher.hexdigest()[:20]


def _cache_path(cache_key: str) -> Path:
    return OUTPUT_DIR / f"nikkei_model_selection_{cache_key}.json"


def _load_cache(cache_key: str) -> dict[str, Any] | None:
    path = _cache_path(cache_key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if payload.get("version") != MODEL_SELECTION_VERSION:
        return None
    return payload


def _save_cache(cache_key: str, payload: dict[str, Any]) -> None:
    path = _cache_path(cache_key)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def run_nikkei_model_selection(
    data: PredictionData,
    force_refresh: bool = False,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    selection_frame, holdout_frame = split_selection_and_holdout(data.training_frame)
    cache_key = model_selection_cache_key(data)
    cache_path = _cache_path(cache_key)
    if force_refresh and cache_path.exists():
        cache_path.unlink()

    cache_lookup_started = time.perf_counter()
    cached = _load_cache(cache_key)
    cache_lookup_seconds = time.perf_counter() - cache_lookup_started
    if cached is None:
        selection_started = time.perf_counter()
        configuration = select_model_configuration(selection_frame, data.feature_columns)
        selection_seconds = time.perf_counter() - selection_started
        final_started = time.perf_counter()
        final_evaluation = evaluate_final_holdout(selection_frame, holdout_frame, configuration)
        final_evaluation_seconds = time.perf_counter() - final_started
        cached = {
            "version": MODEL_SELECTION_VERSION,
            "created_at": _now_jst(),
            "configuration": configuration,
            "final_evaluation": final_evaluation,
            "model_selection_seconds": float(selection_seconds),
            "final_evaluation_seconds": float(final_evaluation_seconds),
        }
        _save_cache(cache_key, cached)
        cache_used = False
    else:
        cache_used = True

    inference_started = time.perf_counter()
    latest_prediction = _fit_latest_prediction(data, cached["configuration"])
    inference_seconds = time.perf_counter() - inference_started
    return {
        **cached,
        "latest_prediction": latest_prediction,
        "cache": {
            "used": cache_used,
            "key": cache_key,
            "created_at": cached["created_at"],
            "force_refresh": force_refresh,
            "lookup_seconds": float(cache_lookup_seconds),
        },
        "latest_inference_seconds": float(inference_seconds),
        "total_seconds": float(time.perf_counter() - total_started),
    }
