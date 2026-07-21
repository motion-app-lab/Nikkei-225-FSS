from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .common import (
    CLASS_ORDER,
    OUTPUT_DIR,
    PREDICTION_HORIZON,
    ServiceError,
    _six_probabilities,
    classify_change,
    generate_analysis_comment,
    plot_channel_chart,
)
from .individual_chart_report import build_six_stage_trend_report
from .nikkei_artifact import (
    EVALUATION_PATHS,
    EVALUATION_SCHEMA_VERSION,
    atomic_write_json,
    load_context_evaluation,
    load_portable_artifact,
    runtime_versions,
    save_portable_artifacts,
)
from .nikkei_direction_comparison import moving_block_bootstrap_intervals
from .nikkei_public_report import (
    NIKKEI_PUBLIC_SCHEMA_VERSION,
    build_nikkei_chart_payload,
)
from .nikkei_dual_market import (
    BOOTSTRAP_BLOCK_LENGTH,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CACHE_VERSION,
    EVALUATION_YEARS,
    EXTERNAL_ALIGNMENT_VERSION,
    FEATURE_DEFINITION_VERSION,
    HALF_LIFE_YEARS,
    HISTORY_PATH,
    INNER_FOLDS,
    JAPAN_SCORE_WEIGHT,
    LEARNING_YEARS,
    LEGACY_SIX_FEATURES,
    MIN_PREDICTION_SHARE,
    MODEL_ARTIFACT_PATH,
    MODEL_CANDIDATES,
    MODEL_PARAMETERS,
    MODEL_SELECTION_VERSION,
    MODEL_SETTINGS_VERSION,
    MODEL_TIE_TOLERANCE,
    MODEL_VERSION,
    OUTER_FOLDS,
    OVERSEAS_SCORE_WEIGHT,
    PURGE_TRADING_DAYS,
    RANDOM_SEED,
    SETTINGS_DIR,
    SETTINGS_PATH,
    STANDARD_THRESHOLD,
    THRESHOLD_CANDIDATES,
    THRESHOLD_LOGIC_VERSION,
    THRESHOLD_MIN_BALANCED_IMPROVEMENT,
    THRESHOLD_MIN_FOLD_IMPROVEMENT,
    THRESHOLD_MIN_IMPROVED_FOLDS,
    THRESHOLD_TIE_TOLERANCE,
    WARMUP_TRADING_DAYS,
    WEIGHT_MAX_WORST_FOLD_DETERIORATION,
    WEIGHT_MIN_BALANCED_IMPROVEMENT,
    DualMarketData,
    prepare_dual_market_data,
)


WEIGHT_METHODS = ("uniform", "half_life_4y")
ENSEMBLE_COMPONENTS = ("catboost_all", "logistic", "extra_trees")
MODEL_COMPLEXITY = {item["id"]: int(item["simplicity"]) for item in MODEL_CANDIDATES}
MODEL_LABELS = {item["id"]: item["label"] for item in MODEL_CANDIDATES}


def _date_text(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _timestamp_text(value: Any) -> str:
    return pd.Timestamp(value).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Index):
        return [_date_text(item) for item in value]
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    serializable = json.loads(json.dumps(payload, ensure_ascii=False, default=_json_default))
    atomic_write_json(path, serializable)


def _median_mapping(frame: pd.DataFrame, features: list[str]) -> dict[str, float]:
    values = frame[features].median(axis=0, skipna=True).fillna(0.0)
    return {str(feature): float(values[feature]) for feature in features}


def load_model_settings() -> dict[str, Any]:
    if not SETTINGS_PATH.exists() or not MODEL_ARTIFACT_PATH.exists():
        raise ServiceError(
            "保存済みの日経平均モデルがありません。モデル再評価を実行してください。",
            "管理者向けのモデル再評価を一度実行すると、通常予測で利用する設定が保存されます。",
            409,
        )
    try:
        settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ServiceError("保存済みモデル設定を読み込めませんでした。", "モデル再評価を実行してください。", 500) from error
    if settings.get("model_version") != MODEL_VERSION or settings.get("cache_version") != CACHE_VERSION:
        raise ServiceError(
            "保存済みモデルは現在の方式と互換性がありません。",
            "新しい日米別方式でモデル再評価を実行してください。",
            409,
        )
    return settings


def training_sample_weights(index: pd.DatetimeIndex, training_end: Any, method: str) -> np.ndarray:
    if method == "uniform":
        return np.ones(len(index), dtype=float)
    if method != "half_life_4y":
        raise ValueError(f"unknown weight method: {method}")
    end = pd.Timestamp(training_end)
    age_years = (end - pd.DatetimeIndex(index)).days.to_numpy(dtype=float) / 365.2425
    return np.power(0.5, age_years / HALF_LIFE_YEARS)


def _class_one_probability(model: Any, probabilities: np.ndarray) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    if 1 not in classes:
        return np.zeros(len(matrix), dtype=float)
    return matrix[:, int(np.where(classes == 1)[0][0])]


def _normalize_importance(features: list[str], values: Iterable[float]) -> dict[str, float]:
    raw = np.abs(np.asarray(list(values), dtype=float))
    total = float(raw.sum())
    if total <= 0:
        return {feature: 0.0 for feature in features}
    return {feature: float(value / total * 100.0) for feature, value in zip(features, raw)}


def _select_features(train: pd.DataFrame, features: list[str]) -> tuple[list[str], dict[str, float]]:
    from sklearn.ensemble import ExtraTreesClassifier

    medians = train[features].median(axis=0, skipna=True).fillna(0.0)
    x_train = train[features].fillna(medians).to_numpy(dtype=float)
    y_train = train["target_direction"].to_numpy(dtype=int)
    if len(np.unique(y_train)) < 2:
        return list(features), {feature: 0.0 for feature in features}
    selector = ExtraTreesClassifier(**MODEL_PARAMETERS["feature_selector"])
    selector.fit(x_train, y_train)
    importance = _normalize_importance(features, selector.feature_importances_)
    keep = min(20, len(features))
    selected = sorted(features, key=lambda column: (-importance[column], column))[:keep]
    return selected, importance


def _fit_component(
    component_id: str,
    train: pd.DataFrame,
    features: list[str],
    weight_method: str,
) -> dict[str, Any]:
    y_train = train["target_direction"].to_numpy(dtype=int)
    selected_features = list(features)
    selection_importance: dict[str, float] = {}
    if component_id == "catboost_selected":
        selected_features, selection_importance = _select_features(train, features)
    medians = _median_mapping(train, selected_features)
    x_train = train[selected_features].fillna(medians).to_numpy(dtype=float)
    sample_weight = training_sample_weights(train.index, train.index[-1], weight_method)
    if len(np.unique(y_train)) < 2:
        return {
            "component_id": component_id,
            "features": selected_features,
            "medians": medians,
            "constant": float(y_train[0]),
            "importance": {feature: 0.0 for feature in selected_features},
            "selection_importance": selection_importance,
        }
    scaler = None
    if component_id in {"catboost_all", "catboost_selected"}:
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            **MODEL_PARAMETERS["catboost"],
            loss_function="Logloss",
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)
        importance = _normalize_importance(selected_features, model.get_feature_importance())
    elif component_id == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        model = LogisticRegression(**MODEL_PARAMETERS["logistic"])
        model.fit(x_train, y_train, sample_weight=sample_weight)
        importance = _normalize_importance(selected_features, model.coef_[0])
    elif component_id == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        model = ExtraTreesClassifier(**MODEL_PARAMETERS["extra_trees"])
        model.fit(x_train, y_train, sample_weight=sample_weight)
        importance = _normalize_importance(selected_features, model.feature_importances_)
    else:
        raise ValueError(f"unknown component: {component_id}")
    return {
        "component_id": component_id,
        "features": selected_features,
        "medians": medians,
        "scaler": scaler,
        "model": model,
        "importance": importance,
        "selection_importance": selection_importance,
    }


def _predict_component(package: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    if "constant" in package:
        return np.full(len(frame), float(package["constant"]), dtype=float)
    features = package["features"]
    x_values = frame[features].fillna(package["medians"]).to_numpy(dtype=float)
    if package.get("scaler") is not None:
        x_values = package["scaler"].transform(x_values)
    return _class_one_probability(package["model"], package["model"].predict_proba(x_values))


def fit_candidate_package(
    candidate_id: str,
    train: pd.DataFrame,
    features: list[str],
    weight_method: str,
) -> dict[str, Any]:
    component_ids = ENSEMBLE_COMPONENTS if candidate_id == "simple_average" else (candidate_id,)
    components = [_fit_component(component_id, train, features, weight_method) for component_id in component_ids]
    combined_importance: dict[str, list[float]] = {}
    for component in components:
        for feature, importance in component["importance"].items():
            combined_importance.setdefault(feature, []).append(float(importance))
    importance = {feature: float(np.mean(values)) for feature, values in combined_importance.items()}
    selected_features = sorted({feature for component in components for feature in component["features"]})
    return {
        "candidate_id": candidate_id,
        "candidate_label": MODEL_LABELS[candidate_id],
        "weight_method": weight_method,
        "components": components,
        "selected_features": selected_features,
        "feature_importance": importance,
        "training_start": _date_text(train.index[0]),
        "training_end": _date_text(train.index[-1]),
        "training_samples": int(len(train)),
    }


def predict_candidate_package(package: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    probabilities = [_predict_component(component, frame) for component in package["components"]]
    return np.mean(np.vstack(probabilities), axis=0)


def make_inner_splits(index: pd.DatetimeIndex) -> list[dict[str, Any]]:
    count = len(index)
    validation_size = max(40, count // 8)
    first_validation = count - INNER_FOLDS * validation_size
    if first_validation < 180:
        validation_size = max(25, (count - 180) // INNER_FOLDS)
        first_validation = count - INNER_FOLDS * validation_size
    if validation_size < 20 or first_validation <= PURGE_TRADING_DAYS:
        raise ServiceError("内側時系列検証に必要な学習データが不足しています。")
    splits: list[dict[str, Any]] = []
    for fold_number in range(INNER_FOLDS):
        validation_start = first_validation + fold_number * validation_size
        validation_end = validation_start + validation_size
        train_end = validation_start - PURGE_TRADING_DAYS
        train_index = pd.DatetimeIndex(index[:train_end])
        fold_training_end = train_index[-1]
        minimum_date = fold_training_end - pd.DateOffset(years=LEARNING_YEARS)
        train_index = train_index[train_index >= minimum_date]
        validation_index = pd.DatetimeIndex(index[validation_start:validation_end])
        splits.append(
            {
                "fold": fold_number + 1,
                "train_index": train_index,
                "validation_index": validation_index,
                "train_start": _date_text(train_index[0]),
                "train_end": _date_text(train_index[-1]),
                "validation_start": _date_text(validation_index[0]),
                "validation_end": _date_text(validation_index[-1]),
                "purge_trading_days": PURGE_TRADING_DAYS,
                "purged_dates": [_date_text(value) for value in index[train_end:validation_start]],
            }
        )
    return splits


def _binary_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score

    y_values = np.asarray(y_true, dtype=int)
    predicted = np.asarray(prediction, dtype=int)
    return {
        "direction_accuracy": float(accuracy_score(y_values, predicted)),
        "direction_balanced_accuracy": float(balanced_accuracy_score(y_values, predicted)),
        "direction_macro_f1": float(f1_score(y_values, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "up_recall": float(recall_score(y_values, predicted, labels=[1], average="macro", zero_division=0)),
        "down_recall": float(recall_score(y_values, predicted, labels=[0], average="macro", zero_division=0)),
        "predicted_up": int(np.sum(predicted == 1)),
        "predicted_down": int(np.sum(predicted == 0)),
        "actual_up": int(np.sum(y_values == 1)),
        "actual_down": int(np.sum(y_values == 0)),
        "correct_predictions": int(np.sum(y_values == predicted)),
        "validation_samples": int(len(y_values)),
        "confusion_matrix": confusion_matrix(y_values, predicted, labels=[0, 1]).astype(int).tolist(),
    }


def _summary(folds: list[dict[str, Any]]) -> dict[str, Any]:
    accuracy = np.asarray([fold["direction_accuracy"] for fold in folds], dtype=float)
    balanced = np.asarray([fold["direction_balanced_accuracy"] for fold in folds], dtype=float)
    macro = np.asarray([fold["direction_macro_f1"] for fold in folds], dtype=float)
    return {
        "mean_direction_accuracy": float(accuracy.mean()),
        "mean_balanced_accuracy": float(balanced.mean()),
        "mean_macro_f1": float(macro.mean()),
        "balanced_accuracy_std": float(balanced.std()),
        "worst_fold_balanced_accuracy": float(balanced.min()),
        "fold_count": int(len(folds)),
        "validation_samples": int(sum(fold["validation_samples"] for fold in folds)),
    }


def _decay_is_safe(uniform: dict[str, Any], decay: dict[str, Any]) -> tuple[bool, str]:
    uniform_balanced = np.asarray([fold["direction_balanced_accuracy"] for fold in uniform["folds"]])
    decay_balanced = np.asarray([fold["direction_balanced_accuracy"] for fold in decay["folds"]])
    mean_gain = float(decay_balanced.mean() - uniform_balanced.mean())
    improved_folds = int(np.sum(decay_balanced > uniform_balanced))
    worst_deterioration = float(uniform_balanced.min() - decay_balanced.min())
    shares_ok = all(
        min(fold["predicted_up"], fold["predicted_down"]) / fold["validation_samples"] >= MIN_PREDICTION_SHARE
        for fold in decay["folds"]
    )
    safe = (
        mean_gain >= WEIGHT_MIN_BALANCED_IMPROVEMENT
        and improved_folds >= math.ceil(len(decay_balanced) / 2)
        and worst_deterioration <= WEIGHT_MAX_WORST_FOLD_DETERIORATION
        and shares_ok
    )
    if safe:
        return True, "複数Foldでバランス成績が改善し、安全条件を満たしました。"
    reasons: list[str] = []
    if mean_gain < WEIGHT_MIN_BALANCED_IMPROVEMENT:
        reasons.append("平均改善が0.5ポイント未満")
    if improved_folds < math.ceil(len(decay_balanced) / 2):
        reasons.append("改善Foldが半数未満")
    if worst_deterioration > WEIGHT_MAX_WORST_FOLD_DETERIORATION:
        reasons.append("最悪Foldが1.0ポイント超悪化")
    if not shares_ok:
        reasons.append("少ない側の予測割合が10%未満")
    return False, "、".join(reasons) + "のため均等重みを維持しました。"


def _candidate_is_better(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    left = candidate["summary"]
    right = incumbent["summary"]
    comparisons = (
        (left["mean_balanced_accuracy"], right["mean_balanced_accuracy"], True),
        (left["worst_fold_balanced_accuracy"], right["worst_fold_balanced_accuracy"], True),
        (left["balanced_accuracy_std"], right["balanced_accuracy_std"], False),
        (left["mean_macro_f1"], right["mean_macro_f1"], True),
        (left["mean_direction_accuracy"], right["mean_direction_accuracy"], True),
    )
    for candidate_value, incumbent_value, higher_is_better in comparisons:
        difference = candidate_value - incumbent_value
        if abs(difference) <= MODEL_TIE_TOLERANCE:
            continue
        return difference > 0 if higher_is_better else difference < 0
    return MODEL_COMPLEXITY[candidate["candidate_id"]] < MODEL_COMPLEXITY[incumbent["candidate_id"]]


def select_side_configuration(
    frame: pd.DataFrame,
    features: list[str],
    splits: list[dict[str, Any]],
    side: str,
) -> dict[str, Any]:
    candidate_records: dict[str, dict[str, Any]] = {}
    validation_outputs: dict[str, dict[str, list[np.ndarray]]] = {}
    for candidate in MODEL_CANDIDATES:
        candidate_id = candidate["id"]
        method_records: dict[str, Any] = {}
        validation_outputs[candidate_id] = {}
        for weight_method in WEIGHT_METHODS:
            folds: list[dict[str, Any]] = []
            fold_probabilities: list[np.ndarray] = []
            for split in splits:
                train = frame.loc[split["train_index"]].dropna(subset=["target_direction"])
                validation = frame.loc[split["validation_index"]].dropna(subset=["target_direction"])
                package = fit_candidate_package(candidate_id, train, features, weight_method)
                probability = predict_candidate_package(package, validation)
                metrics = _binary_metrics(validation["target_direction"].to_numpy(dtype=int), probability >= STANDARD_THRESHOLD)
                public_split = {
                    key: value for key, value in split.items() if key not in {"train_index", "validation_index"}
                }
                folds.append({**public_split, **metrics})
                fold_probabilities.append(probability)
            method_records[weight_method] = {"folds": folds, "summary": _summary(folds)}
            validation_outputs[candidate_id][weight_method] = fold_probabilities
        decay_safe, weight_reason = _decay_is_safe(
            method_records["uniform"], method_records["half_life_4y"]
        )
        selected_weight = "half_life_4y" if decay_safe else "uniform"
        candidate_records[candidate_id] = {
            "candidate_id": candidate_id,
            "candidate_label": candidate["label"],
            "side": side,
            "selected_weight": selected_weight,
            "weight_selection_reason": weight_reason,
            "weight_candidates": method_records,
            "folds": method_records[selected_weight]["folds"],
            "summary": method_records[selected_weight]["summary"],
        }
    selected = candidate_records[MODEL_CANDIDATES[0]["id"]]
    for candidate in MODEL_CANDIDATES[1:]:
        challenger = candidate_records[candidate["id"]]
        if _candidate_is_better(challenger, selected):
            selected = challenger
    selected_probabilities = validation_outputs[selected["candidate_id"]][selected["selected_weight"]]
    return {
        "side": side,
        "selected_model": selected["candidate_id"],
        "selected_model_label": selected["candidate_label"],
        "selected_weight": selected["selected_weight"],
        "selection_reason": "内側時系列検証のバランス成績を優先し、同程度なら最悪Fold、ばらつき、Macro-F1、正答率、単純さの順で選択しました。",
        "candidates": list(candidate_records.values()),
        "selected_fold_probabilities": selected_probabilities,
        "selected_fold_metrics": selected["folds"],
        "summary": selected["summary"],
    }


def _threshold_candidate_record(
    threshold: float,
    y_folds: list[np.ndarray],
    score_folds: list[np.ndarray],
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for y_true, score in zip(y_folds, score_folds):
        folds.append(_binary_metrics(y_true, score >= threshold))
    pooled_y = np.concatenate(y_folds)
    pooled_prediction = np.concatenate([score >= threshold for score in score_folds]).astype(int)
    pooled = _binary_metrics(pooled_y, pooled_prediction)
    return {"threshold": float(threshold), "folds": folds, "summary": _summary(folds), "pooled": pooled}


def _threshold_is_better(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    left = candidate["summary"]
    right = incumbent["summary"]
    comparisons = (
        (left["mean_balanced_accuracy"], right["mean_balanced_accuracy"], True),
        (left["worst_fold_balanced_accuracy"], right["worst_fold_balanced_accuracy"], True),
        (left["balanced_accuracy_std"], right["balanced_accuracy_std"], False),
        (left["mean_macro_f1"], right["mean_macro_f1"], True),
        (left["mean_direction_accuracy"], right["mean_direction_accuracy"], True),
    )
    for candidate_value, incumbent_value, higher_is_better in comparisons:
        difference = candidate_value - incumbent_value
        if abs(difference) <= THRESHOLD_TIE_TOLERANCE:
            continue
        return difference > 0 if higher_is_better else difference < 0
    return abs(candidate["threshold"] - STANDARD_THRESHOLD) < abs(incumbent["threshold"] - STANDARD_THRESHOLD)


def select_combined_threshold(
    y_folds: list[np.ndarray],
    japan_probabilities: list[np.ndarray],
    overseas_probabilities: list[np.ndarray],
) -> dict[str, Any]:
    combined = [
        JAPAN_SCORE_WEIGHT * japan + OVERSEAS_SCORE_WEIGHT * overseas
        for japan, overseas in zip(japan_probabilities, overseas_probabilities)
    ]
    candidates = [_threshold_candidate_record(value, y_folds, combined) for value in THRESHOLD_CANDIDATES]
    standard = next(item for item in candidates if item["threshold"] == STANDARD_THRESHOLD)
    best = candidates[0]
    for candidate in candidates[1:]:
        if _threshold_is_better(candidate, best):
            best = candidate
    fold_improvements = np.asarray(
        [fold["direction_balanced_accuracy"] for fold in best["folds"]]
    ) - np.asarray([fold["direction_balanced_accuracy"] for fold in standard["folds"]])
    mean_improvement = best["summary"]["mean_balanced_accuracy"] - standard["summary"]["mean_balanced_accuracy"]
    shares_ok = all(
        min(fold["predicted_up"], fold["predicted_down"]) / fold["validation_samples"] >= MIN_PREDICTION_SHARE
        for fold in best["folds"]
    )
    safe = (
        best["threshold"] != STANDARD_THRESHOLD
        and mean_improvement >= THRESHOLD_MIN_BALANCED_IMPROVEMENT
        and int(np.sum(fold_improvements >= THRESHOLD_MIN_FOLD_IMPROVEMENT)) >= THRESHOLD_MIN_IMPROVED_FOLDS
        and shares_ok
    )
    if safe:
        selected = best
        fallback = False
        reason = "複数の内側Foldでバランス成績が改善し、両方向の最低予測割合も満たしました。"
    else:
        selected = standard
        fallback = best["threshold"] != STANDARD_THRESHOLD
        reasons: list[str] = []
        if mean_improvement < THRESHOLD_MIN_BALANCED_IMPROVEMENT:
            reasons.append("平均改善が0.5ポイント未満")
        if int(np.sum(fold_improvements >= THRESHOLD_MIN_FOLD_IMPROVEMENT)) < THRESHOLD_MIN_IMPROVED_FOLDS:
            reasons.append("複数Foldで改善を確認できない")
        if not shares_ok:
            reasons.append("片方向の予測割合が10%未満")
        reason = ("、".join(reasons) or "50%が同等以上") + "のため標準の50%を維持しました。"
    return {
        "selected_threshold": float(selected["threshold"]),
        "fallback_to_50": bool(fallback),
        "fallback_reason": reason,
        "candidate_range": [float(THRESHOLD_CANDIDATES[0]), float(THRESHOLD_CANDIDATES[-1])],
        "candidate_step": float(THRESHOLD_CANDIDATES[1] - THRESHOLD_CANDIDATES[0]),
        "selection_metric": "balanced_accuracy",
        "tie_tolerance": THRESHOLD_TIE_TOLERANCE,
        "minimum_prediction_share": MIN_PREDICTION_SHARE,
        "candidates": candidates,
        "selected_summary": selected["summary"],
        "standard_summary": standard["summary"],
    }


def select_operational_configuration(
    japan_frame: pd.DataFrame,
    overseas_frame: pd.DataFrame,
    japan_features: list[str],
    overseas_features: list[str],
) -> dict[str, Any]:
    common_index = japan_frame.index.intersection(overseas_frame.index)
    splits = make_inner_splits(common_index)
    japan = select_side_configuration(japan_frame, japan_features, splits, "japan")
    overseas = select_side_configuration(overseas_frame, overseas_features, splits, "overseas")
    y_folds = [
        japan_frame.loc[split["validation_index"], "target_direction"].to_numpy(dtype=int) for split in splits
    ]
    threshold = select_combined_threshold(
        y_folds,
        japan["selected_fold_probabilities"],
        overseas["selected_fold_probabilities"],
    )
    public_splits = [
        {key: value for key, value in split.items() if key not in {"train_index", "validation_index"}}
        for split in splits
    ]
    return {"inner_folds": public_splits, "japan": japan, "overseas": overseas, "threshold": threshold}


def _selection_record(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _selection_record(item)
            for key, item in value.items()
            if key != "selected_fold_probabilities"
        }
    if isinstance(value, list):
        return [_selection_record(item) for item in value]
    return value


def _outer_splits(evaluation_index: pd.DatetimeIndex) -> list[pd.DatetimeIndex]:
    return [pd.DatetimeIndex(part) for part in np.array_split(evaluation_index, OUTER_FOLDS) if len(part)]


def _six_class_fold(train: pd.DataFrame, validation: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    from catboost import CatBoostClassifier
    from sklearn.metrics import accuracy_score, f1_score

    present = [feature for feature in features if feature in train and feature in validation]
    medians = _median_mapping(train, present)
    x_train = train[present].fillna(medians).to_numpy(dtype=float)
    x_validation = validation[present].fillna(medians).to_numpy(dtype=float)
    y_train = np.asarray([CLASS_ORDER.index(classify_change(value)) for value in train["target_change"]], dtype=int)
    y_validation = np.asarray(
        [CLASS_ORDER.index(classify_change(value)) for value in validation["target_change"]], dtype=int
    )
    if len(np.unique(y_train)) < 2:
        prediction = np.full(len(validation), int(y_train[0]), dtype=int)
    else:
        model = CatBoostClassifier(
            iterations=190,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=8.0,
            random_seed=RANDOM_SEED,
            verbose=False,
            allow_writing_files=False,
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            thread_count=-1,
        )
        model.fit(x_train, y_train)
        prediction = model.predict(x_validation).astype(int).reshape(-1)
    return {
        "y_true": y_validation,
        "prediction": prediction,
        "accuracy": float(accuracy_score(y_validation, prediction)),
        "macro_f1": float(
            f1_score(y_validation, prediction, labels=list(range(len(CLASS_ORDER))), average="macro", zero_division=0)
        ),
    }


def evaluate_formal_two_years(data: DualMarketData, prediction_context: str) -> dict[str, Any]:
    """予測時点と同じ情報条件のフレームだけを使って正式評価する。"""
    started = time.perf_counter()
    if prediction_context not in {"intraday", "after_close"}:
        raise ValueError(f"unknown prediction context: {prediction_context}")
    context_frame = data.intraday_frame if prediction_context == "intraday" else data.after_close_frame
    frame = context_frame.dropna(subset=["target_direction", "target_change"])
    evaluation_meta = data.metadata["evaluation_period"]
    evaluation = frame.loc[evaluation_meta["start"] : evaluation_meta["end"]]
    if len(evaluation) < 360:
        raise ServiceError("直近2年間の正式評価に必要な日経平均データが不足しています。")
    outer_results: list[dict[str, Any]] = []
    pooled_y: list[np.ndarray] = []
    pooled_prediction: list[np.ndarray] = []
    pooled_majority: list[np.ndarray] = []
    pooled_momentum: list[np.ndarray] = []
    pooled_six_y: list[np.ndarray] = []
    pooled_six_prediction: list[np.ndarray] = []
    for fold_number, validation_index in enumerate(_outer_splits(evaluation.index), start=1):
        validation_start_position = frame.index.get_loc(validation_index[0])
        train_end_position = validation_start_position - PURGE_TRADING_DAYS
        train = frame.iloc[:train_end_position].copy()
        training_end = train.index[-1]
        train = train.loc[train.index >= training_end - pd.DateOffset(years=LEARNING_YEARS)]
        validation = frame.loc[validation_index]
        selection = select_operational_configuration(
            train,
            train,
            data.japan_features,
            data.overseas_features,
        )
        japan_package = fit_candidate_package(
            selection["japan"]["selected_model"], train, data.japan_features, selection["japan"]["selected_weight"]
        )
        overseas_package = fit_candidate_package(
            selection["overseas"]["selected_model"],
            train,
            data.overseas_features,
            selection["overseas"]["selected_weight"],
        )
        japan_score = predict_candidate_package(japan_package, validation)
        overseas_score = predict_candidate_package(overseas_package, validation)
        final_score = JAPAN_SCORE_WEIGHT * japan_score + OVERSEAS_SCORE_WEIGHT * overseas_score
        threshold = float(selection["threshold"]["selected_threshold"])
        prediction = (final_score >= threshold).astype(int)
        fixed_prediction = (final_score >= STANDARD_THRESHOLD).astype(int)
        y_true = validation["target_direction"].to_numpy(dtype=int)
        majority_direction = int(train["target_direction"].value_counts().idxmax())
        majority_prediction = np.full(len(validation), majority_direction, dtype=int)
        momentum_prediction = (validation["return_5d"].to_numpy(dtype=float) >= 0).astype(int)
        metrics = _binary_metrics(y_true, prediction)
        fixed_metrics = _binary_metrics(y_true, fixed_prediction)
        majority_metrics = _binary_metrics(y_true, majority_prediction)
        momentum_metrics = _binary_metrics(y_true, momentum_prediction)
        six = _six_class_fold(train, validation, LEGACY_SIX_FEATURES)
        pooled_y.append(y_true)
        pooled_prediction.append(prediction)
        pooled_majority.append(majority_prediction)
        pooled_momentum.append(momentum_prediction)
        pooled_six_y.append(six["y_true"])
        pooled_six_prediction.append(six["prediction"])
        outer_results.append(
            {
                "fold": fold_number,
                "training_period": {"start": _date_text(train.index[0]), "end": _date_text(train.index[-1])},
                "training_samples": int(len(train)),
                "validation_period": {
                    "start": _date_text(validation.index[0]),
                    "end": _date_text(validation.index[-1]),
                },
                "purge_trading_days": PURGE_TRADING_DAYS,
                "validation_samples": int(len(validation)),
                "selected_japan_model": selection["japan"]["selected_model"],
                "selected_japan_weight": selection["japan"]["selected_weight"],
                "selected_overseas_model": selection["overseas"]["selected_model"],
                "selected_overseas_weight": selection["overseas"]["selected_weight"],
                "selected_threshold": threshold,
                "direction_metrics": metrics,
                "fixed_50_metrics": fixed_metrics,
                "majority_baseline": {"direction": "上昇" if majority_direction else "下落", **majority_metrics},
                "five_day_continuation_baseline": momentum_metrics,
                "six_class": {"accuracy": six["accuracy"], "macro_f1": six["macro_f1"]},
                "inner_selection": _selection_record(selection),
            }
        )
    from sklearn.metrics import accuracy_score, f1_score

    y_all = np.concatenate(pooled_y)
    prediction_all = np.concatenate(pooled_prediction)
    majority_all = np.concatenate(pooled_majority)
    momentum_all = np.concatenate(pooled_momentum)
    metrics = _binary_metrics(y_all, prediction_all)
    majority_metrics = _binary_metrics(y_all, majority_all)
    momentum_metrics = _binary_metrics(y_all, momentum_all)
    best_baseline_prediction = (
        majority_all
        if majority_metrics["direction_accuracy"] >= momentum_metrics["direction_accuracy"]
        else momentum_all
    )
    best_baseline_name = (
        "学習期間で多かった方向を毎回答える基準"
        if majority_metrics["direction_accuracy"] >= momentum_metrics["direction_accuracy"]
        else "直近5日方向の継続基準"
    )
    intervals = moving_block_bootstrap_intervals(
        prediction_all == y_all,
        best_baseline_prediction == y_all,
        BOOTSTRAP_BLOCK_LENGTH,
        BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED,
    )
    six_y = np.concatenate(pooled_six_y)
    six_prediction = np.concatenate(pooled_six_prediction)
    threshold_values = [fold["selected_threshold"] for fold in outer_results]
    return {
        "evaluation_role": "直近2年間の外側時系列評価（モデル選択には不使用）",
        "prediction_context": prediction_context,
        "model_schema_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_DEFINITION_VERSION,
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluated_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
        "data_final_date": _date_text(frame.index[-1]),
        "prediction_horizon_trading_days": PREDICTION_HORIZON,
        "learning_years": LEARNING_YEARS,
        "purge_trading_days": PURGE_TRADING_DAYS,
        "model_configuration": {
            "japan_score_weight": JAPAN_SCORE_WEIGHT,
            "overseas_score_weight": OVERSEAS_SCORE_WEIGHT,
            "candidates": list(MODEL_LABELS),
            "inner_fold_count": INNER_FOLDS,
        },
        "feature_schema": {
            "japan": list(data.japan_features),
            "overseas": list(data.overseas_features),
            "six_class": list(LEGACY_SIX_FEATURES),
        },
        "creation_environment": runtime_versions(),
        "period": {"start": _date_text(evaluation.index[0]), "end": _date_text(evaluation.index[-1])},
        "evaluation_samples": int(len(y_all)),
        "outer_fold_count": OUTER_FOLDS,
        "inner_fold_count": INNER_FOLDS,
        "outer_folds": outer_results,
        "direction_metrics": metrics,
        "majority_baseline": majority_metrics,
        "five_day_continuation_baseline": momentum_metrics,
        "best_baseline_name": best_baseline_name,
        "best_baseline_accuracy": float(max(majority_metrics["direction_accuracy"], momentum_metrics["direction_accuracy"])),
        "best_baseline_gap": float(
            metrics["direction_accuracy"]
            - max(majority_metrics["direction_accuracy"], momentum_metrics["direction_accuracy"])
        ),
        "six_class_accuracy": float(accuracy_score(six_y, six_prediction)),
        "six_class_macro_f1": float(
            f1_score(six_y, six_prediction, labels=list(range(len(CLASS_ORDER))), average="macro", zero_division=0)
        ),
        "selected_thresholds": threshold_values,
        "threshold_stability_std": float(np.std(threshold_values)),
        "evaluation_not_used_for_selection": True,
        **intervals,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _fit_six_package(train: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    from catboost import CatBoostClassifier

    present = [feature for feature in features if feature in train]
    medians = _median_mapping(train, present)
    x_train = train[present].fillna(medians).to_numpy(dtype=float)
    y_train = np.asarray([CLASS_ORDER.index(classify_change(value)) for value in train["target_change"]], dtype=int)
    if len(np.unique(y_train)) < 2:
        return {"features": present, "medians": medians, "constant": int(y_train[0])}
    model = CatBoostClassifier(
        iterations=190,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=8.0,
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        thread_count=-1,
    )
    model.fit(x_train, y_train)
    return {"features": present, "medians": medians, "model": model}


def _predict_six(package: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    if "constant" in package:
        probabilities = [0.0] * len(CLASS_ORDER)
        probabilities[int(package["constant"])] = 1.0
    else:
        values = frame[package["features"]].fillna(package["medians"]).to_numpy(dtype=float)
        probabilities = _six_probabilities(package["model"], package["model"].predict_proba(values)[0])
    top = int(np.argmax(probabilities))
    return {"probabilities": probabilities, "prediction_label": CLASS_ORDER[top], "top_probability": probabilities[top]}


def _build_current_packages(data: DualMarketData, selection: dict[str, Any]) -> dict[str, Any]:
    base_date = pd.Timestamp(data.metadata["prediction_context"]["nikkei_base_date"])
    latest_labeled = pd.Timestamp(data.metadata["latest_label_date"])
    training_start = latest_labeled - pd.DateOffset(years=LEARNING_YEARS)
    intraday_train = data.intraday_frame.loc[training_start:latest_labeled].dropna(
        subset=["target_direction", "target_change"]
    )
    after_close_train = data.after_close_frame.loc[training_start:latest_labeled].dropna(
        subset=["target_direction", "target_change"]
    )
    packages: dict[str, Any] = {"model_version": MODEL_VERSION, "contexts": {}}
    for context_name, train in (("intraday", intraday_train), ("after_close", after_close_train)):
        packages["contexts"][context_name] = {
            "japan": fit_candidate_package(
                selection["japan"]["selected_model"],
                train,
                data.japan_features,
                selection["japan"]["selected_weight"],
            ),
            "overseas": fit_candidate_package(
                selection["overseas"]["selected_model"],
                train,
                data.overseas_features,
                selection["overseas"]["selected_weight"],
            ),
            "six_class": _fit_six_package(train, LEGACY_SIX_FEATURES),
        }
    packages["training_period"] = {
        "start": _date_text(intraday_train.index[0]),
        "end": _date_text(intraday_train.index[-1]),
        "samples": int(len(intraday_train)),
        "base_date": _date_text(base_date),
    }
    return packages


def _history_entry(previous: dict[str, Any] | None, settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "old_model_version": previous.get("model_version") if previous else None,
        "new_model_version": settings["model_version"],
        "reevaluated_at": settings["created_at"],
        "raw_data_period": settings["raw_data_period"],
        "training_period": settings["operational_training_period"],
        "evaluation_periods": {context: item["period"] for context, item in settings["formal_evaluations"].items()},
        "old_threshold": previous.get("final_threshold") if previous else None,
        "new_threshold": settings["final_threshold"],
        "old_models": previous.get("adopted_models") if previous else None,
        "new_models": settings["adopted_models"],
        "old_training_weights": previous.get("training_weights") if previous else None,
        "new_training_weights": settings["training_weights"],
        "reason": "明示的なモデル再評価：日米別モデル、ローリング8年、直近2年の外側評価、学習重み比較へ移行",
    }


def reevaluate_dual_market_model() -> dict[str, Any]:
    started = time.perf_counter()
    data = prepare_dual_market_data(model_reevaluation=True)
    formal_evaluations = {
        context: evaluate_formal_two_years(data, context)
        for context in ("intraday", "after_close")
    }
    labeled = data.intraday_frame.dropna(subset=["target_direction", "target_change"])
    latest_labeled = labeled.index[-1]
    latest_train = labeled.loc[labeled.index >= latest_labeled - pd.DateOffset(years=LEARNING_YEARS)]
    operational_selection = select_operational_configuration(
        latest_train,
        latest_train,
        data.japan_features,
        data.overseas_features,
    )
    packages = _build_current_packages(data, operational_selection)
    created_at = pd.Timestamp.now(tz="Asia/Tokyo").isoformat()
    previous = None
    if SETTINGS_PATH.exists():
        try:
            previous = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
    else:
        legacy_path = SETTINGS_DIR / "nikkei_direction_threshold.json"
        if legacy_path.exists():
            try:
                legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
                previous = {
                    "model_version": legacy.get("model_version"),
                    "final_threshold": legacy.get("decision_threshold"),
                    "adopted_models": {"combined": legacy.get("frozen_model", "extra_trees")},
                    "training_weights": {"combined": "uniform"},
                }
            except (OSError, json.JSONDecodeError):
                previous = None
    settings = {
        "model_version": MODEL_VERSION,
        "model_settings_version": MODEL_SETTINGS_VERSION,
        "cache_version": CACHE_VERSION,
        "feature_definition_version": FEATURE_DEFINITION_VERSION,
        "external_alignment_version": EXTERNAL_ALIGNMENT_VERSION,
        "model_selection_version": MODEL_SELECTION_VERSION,
        "threshold_logic_version": THRESHOLD_LOGIC_VERSION,
        "created_at": created_at,
        "prediction_target": "日経平均終値が5日本取引日後に現在より上か下か",
        "prediction_horizon_trading_days": PREDICTION_HORIZON,
        "learning_years": LEARNING_YEARS,
        "warmup_trading_days": WARMUP_TRADING_DAYS,
        "evaluation_years": EVALUATION_YEARS,
        "prediction_contexts": ["intraday", "after_close"],
        "raw_data_period": data.metadata["raw_data_period"],
        "feature_valid_start": data.metadata["feature_valid_start"],
        "warmup_period": data.metadata["warmup_period"],
        "operational_training_period": packages["training_period"],
        "formal_evaluations": {
            context: {
                "prediction_context": context,
                "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
                "period": evaluation["period"],
                "path": str(EVALUATION_PATHS[context].relative_to(SETTINGS_DIR.parent)).replace("\\", "/"),
            }
            for context, evaluation in formal_evaluations.items()
        },
        "adopted_models": {
            "japan": operational_selection["japan"]["selected_model"],
            "japan_label": operational_selection["japan"]["selected_model_label"],
            "overseas": operational_selection["overseas"]["selected_model"],
            "overseas_label": operational_selection["overseas"]["selected_model_label"],
        },
        "adopted_features": {
            "japan": packages["contexts"]["intraday"]["japan"]["selected_features"],
            "overseas": packages["contexts"]["intraday"]["overseas"]["selected_features"],
        },
        "feature_importance": {
            "japan": packages["contexts"]["intraday"]["japan"]["feature_importance"],
            "overseas": packages["contexts"]["intraday"]["overseas"]["feature_importance"],
        },
        "context_model_details": {
            context: {
                "adopted_features": {
                    "japan": payload["japan"]["selected_features"],
                    "overseas": payload["overseas"]["selected_features"],
                },
                "feature_importance": {
                    "japan": payload["japan"]["feature_importance"],
                    "overseas": payload["overseas"]["feature_importance"],
                },
            }
            for context, payload in packages["contexts"].items()
        },
        "training_weights": {
            "japan": operational_selection["japan"]["selected_weight"],
            "overseas": operational_selection["overseas"]["selected_weight"],
            "half_life_years": HALF_LIFE_YEARS,
            "decay_formula": "0.5 ** (age_years / 4.0)",
        },
        "combination": {"japan": JAPAN_SCORE_WEIGHT, "overseas": OVERSEAS_SCORE_WEIGHT, "method": "fixed_50_50"},
        "final_threshold": operational_selection["threshold"]["selected_threshold"],
        "threshold_selection": operational_selection["threshold"],
        "operational_inner_selection": _selection_record(operational_selection),
        "external_alignment": data.metadata["external_sources"],
        "available_external_factors": data.metadata["available_external_factors"],
        "pce_used": False,
        "pce_exclusion_reason": "実際の公表日と当時利用可能だったデータを安全に再現できないため除外",
        "random_seed": RANDOM_SEED,
        "normal_prediction_reselects_model": False,
        "normal_prediction_reselects_weight": False,
        "normal_prediction_reselects_threshold": False,
        "normal_prediction_repeats_formal_evaluation": False,
        "artifact_path": str(MODEL_ARTIFACT_PATH.relative_to(SETTINGS_DIR.parent)).replace("\\", "/"),
    }
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = save_portable_artifacts(packages, formal_evaluations)
    _write_json(SETTINGS_PATH, settings)
    history: list[dict[str, Any]] = []
    if HISTORY_PATH.exists():
        try:
            existing = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                history = existing
        except (OSError, json.JSONDecodeError):
            history = []
    history.append(_history_entry(previous, settings))
    _write_json(HISTORY_PATH, history)
    result = build_latest_result(data, settings, packages, cache_used=False)
    result["reevaluation"] = {
        "completed": True,
        "elapsed_seconds": float(time.perf_counter() - started),
        "settings_path": str(SETTINGS_PATH),
        "artifact_path": str(MODEL_ARTIFACT_PATH),
        "history_path": str(HISTORY_PATH),
        "manifest_path": str(SETTINGS_DIR / "nikkei_model_manifest.json"),
        "model_sha256": manifest["model_sha256"],
    }
    return result


def _cache_path(context: str, base_date: str) -> Path:
    safe_date = base_date.replace("-", "")
    return OUTPUT_DIR / f"nikkei_dual_{CACHE_VERSION}_{context}_{safe_date}.json"


def _settings_fingerprint(settings: dict[str, Any]) -> str:
    keys = {
        "model_version": settings["model_version"],
        "cache_version": settings["cache_version"],
        "feature_definition_version": settings["feature_definition_version"],
        "external_alignment_version": settings["external_alignment_version"],
        "model_selection_version": settings["model_selection_version"],
        "threshold_logic_version": settings["threshold_logic_version"],
        "created_at": settings["created_at"],
        "threshold": settings["final_threshold"],
        "models": settings["adopted_models"],
        "weights": settings["training_weights"],
        "features": settings["adopted_features"],
        "combination": settings["combination"],
        "available_external_factors": settings["available_external_factors"],
        "pce_used": settings["pce_used"],
    }
    return hashlib.sha256(json.dumps(keys, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _data_fingerprint(data: DualMarketData) -> str:
    context = data.metadata["prediction_context"]
    frame = data.inference_rows[context["prediction_context"]]
    values = {
        "prediction_context": context["prediction_context"],
        "base_date": context["nikkei_base_date"],
        "base_close": context["nikkei_base_close"],
        "external_cutoff": context["external_cutoff_timestamp_jst"],
        "feature_values": {
            column: None if pd.isna(value) else round(float(value), 12)
            for column, value in frame.iloc[0].items()
        },
        "external_source_dates": {
            factor: item.get("selected_source_date")
            for factor, item in data.metadata["latest_external_usage"].items()
        },
    }
    return hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def build_latest_result(
    data: DualMarketData,
    settings: dict[str, Any],
    packages: dict[str, Any],
    cache_used: bool,
) -> dict[str, Any]:
    context_meta = data.metadata["prediction_context"]
    context = context_meta["prediction_context"]
    if context not in packages["contexts"]:
        raise ServiceError("現在の予測モードに対応する保存済みモデルがありません。", "モデル再評価を実行してください。")
    inference = data.inference_rows[context]
    full_context_frame = data.intraday_frame if context == "intraday" else data.after_close_frame
    six_inference = full_context_frame.loc[[pd.Timestamp(context_meta["nikkei_base_date"])]]
    package = packages["contexts"][context]
    if inference[data.overseas_features].isna().all(axis=None):
        raise ServiceError(
            "米国・海外側データを取得できないため、統合予測を作成できませんでした。",
            "通信状態を確認して、時間をおいて再度実行してください。",
            503,
        )
    japan_score = float(predict_candidate_package(package["japan"], inference)[0])
    overseas_score = float(predict_candidate_package(package["overseas"], inference)[0])
    final_score = JAPAN_SCORE_WEIGHT * japan_score + OVERSEAS_SCORE_WEIGHT * overseas_score
    threshold = float(settings["final_threshold"])
    direction = "上昇傾向" if final_score >= threshold else "下落傾向"
    six = _predict_six(package["six_class"], six_inference)
    raw_probability_rows = [
        {
            "class": label,
            "label": label,
            "raw_label": label,
            "probability": float(probability),
            "percentage": float(probability) * 100.0,
        }
        for label, probability in zip(CLASS_ORDER, six["probabilities"])
    ]
    six_stage = build_six_stage_trend_report(raw_probability_rows)
    public_analysis = build_nikkei_chart_payload(
        data.japanese_frame,
        context_meta["nikkei_base_date"],
        six_stage,
    )
    latest_row = data.japanese_frame.loc[pd.Timestamp(context_meta["nikkei_base_date"])]
    chart_url = plot_channel_chart(
        data.japanese_frame.loc[: pd.Timestamp(context_meta["nikkei_base_date"])],
        "^N225",
        "Nikkei 225",
        60,
    )
    latest_usage = data.metadata["latest_external_usage"]
    latest_us_dates = [
        item.get("selected_source_date")
        for factor, item in latest_usage.items()
        if factor in {"spx", "dow", "nasdaq", "vix", "sox", "nvda"} and item.get("selected_source_date")
    ]
    formal_evaluation = load_context_evaluation(context)
    context_details = settings.get("context_model_details", {}).get(context, {})
    adopted_features = context_details.get("adopted_features", settings["adopted_features"])
    feature_importance = context_details.get("feature_importance", settings["feature_importance"])
    context_label = "場中データによる予測" if context == "intraday" else "大引け後の確定データによる予測"
    return {
        "kind": "prediction",
        "nikkei_public_schema_version": NIKKEI_PUBLIC_SCHEMA_VERSION,
        "company_name": "日経平均株価",
        "ticker": "^N225",
        "model_version": settings["model_version"],
        "model_created_at": settings["created_at"],
        "cache_version": settings["cache_version"],
        "cache_used": bool(cache_used),
        "prediction_context": context,
        "prediction_context_label": context_label,
        "prediction_mode": context_meta["prediction_mode"],
        "prediction_timestamp_jst": context_meta["prediction_timestamp_jst"],
        "basis_date": context_meta["nikkei_base_date"],
        "latest_price": context_meta["nikkei_base_close"],
        "target_date": context_meta["target_date"],
        "external_cutoff_timestamp_jst": context_meta["external_cutoff_timestamp_jst"],
        "latest_us_market_date": max(latest_us_dates) if latest_us_dates else None,
        "japan_up_score": japan_score,
        "overseas_up_score": overseas_score,
        "final_up_score": float(final_score),
        "decision_threshold": threshold,
        "score_margin": float(final_score - threshold),
        "direction": direction,
        "direction_key": "up" if final_score >= threshold else "down",
        "prediction_class": six["prediction_label"],
        "top_probability": six["top_probability"],
        "probabilities": raw_probability_rows,
        "six_stage_trend": six_stage,
        **public_analysis,
        "rsi": float(latest_row["rsi14"]),
        "channel_position": float(latest_row["ch_pos"]),
        "analysis_comment": generate_analysis_comment(latest_row, six["prediction_label"]),
        "chart_url": chart_url,
        "formal_evaluation": formal_evaluation,
        "formal_evaluation_available": formal_evaluation is not None,
        "models": settings["adopted_models"],
        "training_weights": settings["training_weights"],
        "adopted_features": adopted_features,
        "feature_importance": feature_importance,
        "data_periods": {
            "raw": settings["raw_data_period"],
            "training": settings["operational_training_period"],
            "evaluation": formal_evaluation["period"] if formal_evaluation else None,
            "warmup": settings["warmup_period"],
        },
        "learning_years": LEARNING_YEARS,
        "warmup_trading_days": WARMUP_TRADING_DAYS,
        "evaluation_years": EVALUATION_YEARS,
        "combination": settings["combination"],
        "threshold_selection": settings["threshold_selection"],
        "external_alignment": settings["external_alignment"],
        "latest_external_usage": latest_usage,
        "pce_used": False,
        "pce_exclusion_reason": settings["pce_exclusion_reason"],
        "warnings": list(dict.fromkeys(data.warnings + ([context_meta["warning"]] if context_meta.get("warning") else []))),
        "fetched_at": data.fetched_at,
        "normal_prediction_reselects_configuration": False,
        "disclaimer": "本システムの予測およびシミュレーション結果は、情報提供および研究目的の参考情報です。特定の金融商品の売買を推奨するものではなく、利益を保証するものでもありません。投資判断は利用者自身の責任で行ってください。",
    }


def predict_with_saved_dual_market_model(force_refresh: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    settings = load_model_settings()
    packages, _manifest = load_portable_artifact()
    data = prepare_dual_market_data(model_reevaluation=False)
    context = data.metadata["prediction_context"]
    cache_path = _cache_path(context["prediction_context"], context["nikkei_base_date"])
    fingerprint = _settings_fingerprint(settings)
    data_fingerprint = _data_fingerprint(data)
    if not force_refresh and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("settings_fingerprint") == fingerprint
                and cached.get("data_fingerprint") == data_fingerprint
                and cached.get("nikkei_public_schema_version") == NIKKEI_PUBLIC_SCHEMA_VERSION
            ):
                cached["cache_used"] = True
                cached["total_seconds"] = float(time.perf_counter() - started)
                return cached
        except (OSError, json.JSONDecodeError):
            pass
    result = build_latest_result(data, settings, packages, cache_used=False)
    result["settings_fingerprint"] = fingerprint
    result["data_fingerprint"] = data_fingerprint
    result["total_seconds"] = float(time.perf_counter() - started)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(cache_path, result)
    _write_json(OUTPUT_DIR / "nikkei_latest.json", result)
    return result
