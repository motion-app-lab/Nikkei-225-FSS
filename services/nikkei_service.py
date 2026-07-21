from __future__ import annotations

from typing import Any

from .nikkei_dual_model import predict_with_saved_dual_market_model, reevaluate_dual_market_model
from .nikkei_public_report import NIKKEI_PUBLIC_SCHEMA_VERSION


SIX_CLASS_DISPLAY_LABELS = {
    "急騰": "上昇 Lv.3",
    "上昇": "上昇 Lv.2",
    "やや上昇": "上昇 Lv.1",
    "やや下落": "下落 Lv.1",
    "下落": "下落 Lv.2",
    "急落": "下落 Lv.3",
}


FEATURE_DISPLAY_NAMES = {
    "return_1d": "1日騰落率",
    "return_5d": "5日騰落率",
    "return_10d": "10日騰落率",
    "return_20d": "20日騰落率",
    "rsi14": "RSI（14日）",
    "ch_pos": "価格チャネル内の位置",
    "ch_trend": "価格チャネルの傾き",
    "ma20_gap": "20日移動平均乖離率",
    "ma50_gap": "50日移動平均乖離率",
    "ma100_gap": "100日移動平均乖離率",
    "ma200_gap": "200日移動平均乖離率",
    "ma20_slope5": "20日移動平均の傾き",
    "ma50_slope5": "50日移動平均の傾き",
    "ma100_slope5": "100日移動平均の傾き",
    "volatility_5d": "5日ボラティリティ",
    "volatility_20d": "20日ボラティリティ",
    "volatility_60d": "60日ボラティリティ",
    "distance_high60": "60日高値からの距離",
    "distance_high252": "252日高値からの距離",
    "distance_low60": "60日安値からの距離",
    "atr14_ratio": "14日平均値幅率",
    "opening_gap": "始値ギャップ率",
    "volume_change": "出来高変化率",
    "spx": "S&P 500",
    "dow": "NYダウ",
    "nasdaq": "NASDAQ",
    "vix": "VIX",
    "sox": "SOX",
    "nvda": "NVIDIA",
    "usdjpy": "ドル円",
    "oil": "原油",
    "gold": "金",
    "btc": "Bitcoin",
}

EXTERNAL_FACTOR_LABELS = {
    "spx": "S&P 500",
    "dow": "NYダウ",
    "nasdaq": "NASDAQ",
    "vix": "VIX",
    "sox": "SOX",
    "nvda": "NVIDIA",
    "usdjpy": "ドル円",
    "oil": "原油",
    "gold": "金",
    "btc": "Bitcoin",
}


def _feature_display_name(feature: str) -> str:
    if feature in FEATURE_DISPLAY_NAMES:
        return FEATURE_DISPLAY_NAMES[feature]
    for prefix, label in EXTERNAL_FACTOR_LABELS.items():
        if feature.startswith(f"{prefix}_"):
            suffix = feature[len(prefix) + 1 :]
            suffix_label = {
                "ret": "1日騰落率",
                "ret5": "5日騰落率",
                "ret20": "20日騰落率",
                "vol20": "20日ボラティリティ",
                "ma20_gap": "20日移動平均乖離率",
            }.get(suffix, suffix)
            return f"{label} {suffix_label}"
    return feature


def _factor_summary(result: dict[str, Any]) -> tuple[list[str], list[str]]:
    japan = set(result["adopted_features"]["japan"])
    overseas = set(result["adopted_features"]["overseas"])
    selected: list[str] = []
    groups = (
        ("日経平均自身の価格・リターン", {"open", "high", "low", "price", "return_1d", "return_5d", "return_10d", "return_20d"}),
        ("RSI等の価格オシレーター", {"rsi14", "overheat_score"}),
        ("トレンド環境・移動平均", {item for item in japan if item.startswith("ma") or item.startswith("ch_")}),
        ("価格位置", {item for item in japan if item.startswith("distance_")}),
        ("ボラティリティ・値幅", {item for item in japan if item.startswith("volatility_") or item in {"atr14_ratio", "intraday_range", "opening_gap"}}),
        ("出来高", {"volume", "volume_change"}),
    )
    for label, members in groups:
        if japan.intersection(members):
            selected.append(label)
    excluded: list[str] = []
    available = list(result.get("latest_external_usage", {}).keys())
    for factor in available:
        label = EXTERNAL_FACTOR_LABELS.get(factor, factor)
        if any(feature == factor or feature.startswith(f"{factor}_") for feature in overseas):
            selected.append(label)
        else:
            excluded.append(label)
    excluded.append("PCE（公表時点を安全に再現できないため対象外）")
    return list(dict.fromkeys(selected)), list(dict.fromkeys(excluded))


def _combined_feature_importance(result: dict[str, Any]) -> list[dict[str, Any]]:
    weights = result.get("combination", {})
    rows: list[dict[str, Any]] = []
    for side, side_label in (("japan", "日本側"), ("overseas", "米国・海外側")):
        side_weight = float(weights.get(side, 0.5))
        for feature, value in result.get("feature_importance", {}).get(side, {}).items():
            importance = max(0.0, float(value)) * side_weight
            rows.append({
                "internal_name": feature,
                "display_name": _feature_display_name(feature),
                "importance": importance,
                "side": side_label,
            })
    rows.sort(key=lambda item: (-item["importance"], item["internal_name"]))
    return rows[:10]

def _public_evaluation(formal: dict[str, Any] | None, prediction_context: str) -> dict[str, Any]:
    if not formal or formal.get("prediction_context") != prediction_context:
        return {
            "available": False,
            "prediction_context": prediction_context,
            "message": "現在使用した予測条件に対応する正式評価は、まだ作成されていません。",
        }
    metrics = formal["direction_metrics"]
    confusion = metrics["confusion_matrix"]
    true_down, false_up = int(confusion[0][0]), int(confusion[0][1])
    false_down, true_up = int(confusion[1][0]), int(confusion[1][1])
    predicted_up_count = true_up + false_up
    predicted_down_count = true_down + false_down
    up_precision = (true_up / predicted_up_count) if predicted_up_count else None
    down_precision = (true_down / predicted_down_count) if predicted_down_count else None
    majority = formal["majority_baseline"]
    continuation = formal["five_day_continuation_baseline"]
    best_accuracy = float(formal["best_baseline_accuracy"])
    accuracy = float(metrics["direction_accuracy"])
    balanced = float(metrics["direction_balanced_accuracy"])
    majority_balanced = float(majority["direction_balanced_accuracy"])
    if accuracy > best_accuracy and balanced > majority_balanced:
        comment = (
            "直近2年間の時系列評価では、単純な比較方法より方向を見分ける成績が良くなりました。"
            "ただし、今後の値動きを保証するものではありません。"
        )
    elif balanced > majority_balanced:
        comment = (
            "上昇と下落を分けて見ると、値動きを見分ける力が少し確認できました。"
            "一方、全体の正答率では最良の単純比較を上回っていません。"
        )
    else:
        comment = (
            "今回の時系列評価では、単純な比較方法を安定して上回る結果は確認できませんでした。"
            "予測結果を記録しながら、今後も検証を続けます。"
        )
    if abs(float(metrics["up_recall"]) - float(metrics["down_recall"])) >= 0.15:
        easier = "上昇" if metrics["up_recall"] > metrics["down_recall"] else "下落"
        harder = "下落" if easier == "上昇" else "上昇"
        comment += f" 現在のモデルは{easier}方向を見分けやすく、{harder}方向の判定には課題があります。"
    outer_folds = [
        {
            key: fold[key]
            for key in (
                "fold",
                "training_period",
                "training_samples",
                "validation_period",
                "purge_trading_days",
                "validation_samples",
                "selected_japan_model",
                "selected_japan_weight",
                "selected_overseas_model",
                "selected_overseas_weight",
                "selected_threshold",
                "direction_metrics",
                "fixed_50_metrics",
                "majority_baseline",
                "five_day_continuation_baseline",
                "six_class",
            )
        }
        for fold in formal["outer_folds"]
    ]
    return {
        "available": True,
        "period": formal["period"],
        "prediction_context": formal["prediction_context"],
        "validation_samples": formal["evaluation_samples"],
        "correct_predictions": metrics["correct_predictions"],
        "direction_accuracy": metrics["direction_accuracy"],
        "direction_balanced_accuracy": metrics["direction_balanced_accuracy"],
        "direction_macro_f1": metrics["direction_macro_f1"],
        "up_recall": metrics["up_recall"],
        "down_recall": metrics["down_recall"],
        "predicted_up": metrics["predicted_up"],
        "predicted_down": metrics["predicted_down"],
        "predicted_up_count": predicted_up_count,
        "correct_up_predictions": true_up,
        "up_prediction_precision": up_precision,
        "predicted_down_count": predicted_down_count,
        "correct_down_predictions": true_down,
        "down_prediction_precision": down_precision,
        "actual_up": metrics["actual_up"],
        "actual_down": metrics["actual_down"],
        "confusion_matrix": metrics["confusion_matrix"],
        "majority_baseline_accuracy": majority["direction_accuracy"],
        "majority_baseline_balanced_accuracy": majority["direction_balanced_accuracy"],
        "five_day_continuation_accuracy": continuation["direction_accuracy"],
        "five_day_continuation_balanced_accuracy": continuation["direction_balanced_accuracy"],
        "best_baseline_name": formal["best_baseline_name"],
        "best_baseline_accuracy": best_accuracy,
        "best_baseline_gap": formal["best_baseline_gap"],
        "direction_accuracy_95ci": formal["direction_accuracy_95ci"],
        "best_baseline_gap_95ci": formal["best_baseline_gap_95ci"],
        "bootstrap": {
            "method": formal["method"],
            "block_length": formal["block_length"],
            "resamples": formal["resamples"],
            "seed": formal["seed"],
        },
        "outer_folds": outer_folds,
        "six_class_accuracy": formal["six_class_accuracy"],
        "six_class_macro_f1": formal["six_class_macro_f1"],
        "evaluation_comment": comment,
        "evaluation_not_used_for_selection": formal["evaluation_not_used_for_selection"],
    }


def _finish_payload(result: dict[str, Any], model_reevaluation: bool) -> dict[str, Any]:
    technical_class = result["prediction_class"]
    result["prediction_class_technical"] = technical_class
    result["prediction_class"] = SIX_CLASS_DISPLAY_LABELS.get(technical_class, technical_class)
    for item in result["probabilities"]:
        technical_label = item.get("raw_label", item["label"])
        item["technical_label"] = technical_label
        item["label"] = SIX_CLASS_DISPLAY_LABELS.get(technical_label, technical_label)

    six_stage = result["six_stage_trend"]
    items = list(six_stage["items"])
    up_total = round(sum(float(item["percentage"]) for item in items[:3]), 1)
    down_total = round(sum(float(item["percentage"]) for item in items[3:]), 1)
    balance_gap = abs(up_total - down_total)
    if balance_gap < 10.0:
        direction_sentence = "上昇3区分と下落3区分の合計差は小さく、方向性が一方へ大きく偏った出力ではありません。"
    elif up_total > down_total:
        direction_sentence = "6区分全体では、上昇側3区分の合計が下落側3区分の合計を上回りました。"
    else:
        direction_sentence = "6区分全体では、下落側3区分の合計が上昇側3区分の合計を上回りました。"
    six_class_report = {
        "title": "6段階トレンド予測レポート",
        "body": " ".join(
            (
                six_stage["intro"],
                f"次に大きい区分は{six_stage['second_label']} {six_stage['second_percentage']:.1f}％です。"
                f"上昇3区分の合計は{up_total:.1f}％、下落3区分の合計は{down_total:.1f}％です。",
                direction_sentence,
                six_stage["distribution_description"],
            )
        ),
        "up_total": up_total,
        "down_total": down_total,
        "footer": six_stage["footer"],
    }
    result["up_total"] = up_total
    result["down_total"] = down_total
    result["six_class_probabilities"] = items
    result["six_class_report"] = six_class_report

    validation = _public_evaluation(result.get("formal_evaluation"), result["prediction_context"])
    result["validation"] = validation
    result.pop("formal_evaluation", None)
    selected_factors, excluded_factors = _factor_summary(result)
    importance_top10 = _combined_feature_importance(result)
    factor_definition = "最終再学習した保存済み日米統合モデルで、実際に使用された特徴量をファクター単位に集約しています。"
    result["selected_factors"] = selected_factors
    result["excluded_factors"] = excluded_factors
    result["factor_selection_definition"] = factor_definition
    result["feature_importance_top10"] = importance_top10
    result["model_name"] = f"日本側: {result['models']['japan_label']} / 米国・海外側: {result['models']['overseas_label']}"
    result["feature_group"] = "日本側特徴量＋利用可能時刻を調整した米国・海外側特徴量"
    result["model_selection"] = {
        "japan_model": result["models"]["japan"],
        "japan_model_label": result["models"]["japan_label"],
        "overseas_model": result["models"]["overseas"],
        "overseas_model_label": result["models"]["overseas_label"],
        "japan_features": result["adopted_features"]["japan"],
        "overseas_features": result["adopted_features"]["overseas"],
        "japan_weight": result["training_weights"]["japan"],
        "overseas_weight": result["training_weights"]["overseas"],
        "feature_importance": result["feature_importance"],
        "feature_importance_top10": importance_top10,
        "selected_factors": selected_factors,
        "excluded_factors": excluded_factors,
        "factor_selection_definition": factor_definition,
        "fixed_combination": result["combination"],
    }

    raw_period = result.get("data_periods", {}).get("raw", {})
    evaluation_period = validation.get("period", {})
    result["forecast_horizon_label"] = "5営業日先"
    result["forecast_base_date"] = result["basis_date"]
    result["forecast_target_date"] = result["target_date"]
    result["data_collection_start"] = raw_period.get("start")
    result["data_collection_end"] = result["basis_date"]
    result["data_collection_definition"] = "予測基準日以前に取得した日経平均の確定日足と、各市場で利用可能時刻を迎えた外部データの期間です。"
    result["evaluation_start"] = evaluation_period.get("start")
    result["evaluation_end"] = evaluation_period.get("end")
    result["accuracy_summary"] = {
        "available": bool(validation.get("available")),
        "evaluation_start": evaluation_period.get("start"),
        "evaluation_end": evaluation_period.get("end"),
        "validation_samples": validation.get("validation_samples"),
        "correct_predictions": validation.get("correct_predictions"),
        "direction_accuracy": validation.get("direction_accuracy"),
        "up_prediction_precision": validation.get("up_prediction_precision"),
        "down_prediction_precision": validation.get("down_prediction_precision"),
        "best_baseline_gap": validation.get("best_baseline_gap"),
        "six_class_accuracy": validation.get("six_class_accuracy"),
        "six_class_macro_f1": validation.get("six_class_macro_f1"),
        "message": validation.get("message"),
    }
    result["direction_evaluation"] = {
        key: validation.get(key)
        for key in (
            "available", "prediction_context", "period", "validation_samples", "correct_predictions",
            "direction_accuracy", "up_prediction_precision", "down_prediction_precision",
            "best_baseline_name", "best_baseline_accuracy", "best_baseline_gap",
            "direction_accuracy_95ci", "best_baseline_gap_95ci", "message",
        )
    }
    result["six_class_evaluation"] = {
        "available": bool(validation.get("available")),
        "prediction_context": validation.get("prediction_context"),
        "period": validation.get("period"),
        "six_class_accuracy": validation.get("six_class_accuracy"),
        "six_class_macro_f1": validation.get("six_class_macro_f1"),
        "message": validation.get("message"),
    }
    result["model_roles_note"] = (
        "方向予測と6段階予測は、それぞれ異なる目的のモデルで計算しています。"
        "方向予測は上昇・下落の一致を評価し、6段階予測は値動き幅の区分を評価します。"
    )
    result["cache"] = {
        "used": bool(result.get("cache_used")),
        "model_reevaluation": bool(model_reevaluation),
        "created_at": result.get("fetched_at"),
    }
    result["score_display_note"] = (
        "6段階予測の棒グラフは今回のモデル出力割合です。方向一致率や6段階完全一致率とは別であり、"
        "将来の利益確率や売買の推奨度を示すものではありません。"
    )
    return result

def predict_nikkei(model_reevaluation: bool = False) -> dict[str, Any]:
    """明示的な再評価時だけ設定を選び直し、通常予測では保存済みモデルを使う。"""
    if model_reevaluation:
        result = reevaluate_dual_market_model()
    else:
        result = predict_with_saved_dual_market_model()
    return _finish_payload(result, model_reevaluation)

def is_current_nikkei_result(result: dict[str, Any] | None) -> bool:
    if not isinstance(result, dict):
        return False
    required = (
        "six_class_probabilities",
        "six_class_report",
        "chart_60d",
        "short_term_report",
        "chart_2y",
        "medium_long_term_report",
        "selected_factors",
        "excluded_factors",
        "feature_importance_top10",
        "accuracy_summary",
        "prediction_context_label",
        "direction_evaluation",
        "six_class_evaluation",
        "model_roles_note",
    )
    return (
        result.get("nikkei_public_schema_version") == NIKKEI_PUBLIC_SCHEMA_VERSION
        and all(key in result for key in required)
    )
