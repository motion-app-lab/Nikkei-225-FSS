from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .common import OUTPUT_DIR, ServiceError, _download_base_frame, get_ticker_name
from .individual_chart_report import (
    CHART_ANALYSIS_VERSION,
    LONG_CHART_ANALYSIS_VERSION,
    SIX_STAGE_REPORT_VERSION,
    SIX_STAGE_DISPLAY_ORDER,
    build_chart_analysis,
    build_long_term_chart_analysis,
    build_six_stage_trend_report,
    chart_data,
    long_chart_data,
    plot_individual_chart,
    plot_individual_long_chart,
)
from .individual_logistic_fast import (
    FIXED_DIRECTION_THRESHOLD,
    FIXED_FEATURE_GROUP,
    LOGISTIC_SETTINGS,
    MODEL_SCHEMA_VERSION,
    MODEL_SETTINGS_HASH,
    latest_prediction,
)
from .individual_market import (
    EXTERNAL_SOURCES,
    INDIVIDUAL_FEATURE_DEFINITION_VERSION,
    INDIVIDUAL_MARKET_ALIGNMENT_VERSION,
    TRAINING_YEARS,
    WARMUP_TRADING_DAYS,
    IndividualMarketData,
    prepare_individual_market_data,
)


INDIVIDUAL_UI_SCHEMA_VERSION = "individual_public_ui_v7_no_accuracy"
INDIVIDUAL_CACHE_VERSION = "individual_prediction_cache_v8_no_accuracy"
INDIVIDUAL_MODEL_VERSION = MODEL_SCHEMA_VERSION
INDIVIDUAL_SELECTION_VERSION = "individual_fixed_configuration_v1"
INDIVIDUAL_CALIBRATION_VERSION = "individual_logistic_raw_output_v1"
INDIVIDUAL_PUBLIC_LABEL_VERSION = "individual_public_labels_level_v2"
INDIVIDUAL_DATA_COLLECTION_VERSION = "individual_data_collection_period_v1"
INDIVIDUAL_MAX_PROCESSING_SECONDS = 180.0

logger = logging.getLogger(__name__)
_prediction_lock = threading.Lock()
JST = timezone(timedelta(hours=9))
MIN_LIQUID_DAYS_RATIO = 0.80
DISPLAY_CLASS_LABELS = {raw: display for raw, display in SIX_STAGE_DISPLAY_ORDER}


class IndividualExecutionControl:
    """1件の予測に有限期限と少量の段階時間ログを持たせる。"""

    def __init__(self, max_seconds: float = INDIVIDUAL_MAX_PROCESSING_SECONDS):
        self.max_seconds = float(max_seconds)
        self.started = time.perf_counter()
        self.cancel_event = threading.Event()
        self.stage_seconds: dict[str, float] = {}
        self.cancel_reason = ""
        self._timing_lock = threading.Lock()

    def elapsed(self) -> float:
        return float(time.perf_counter() - self.started)

    def cancel(self, reason: str) -> None:
        self.cancel_reason = reason
        self.cancel_event.set()

    def checkpoint(self, stage: str) -> None:
        if self.cancel_event.is_set() or self.elapsed() >= self.max_seconds:
            self.cancel_event.set()
            logger.error(
                "individual prediction stopped stage=%s elapsed=%.3fs reason=%s",
                stage,
                self.elapsed(),
                self.cancel_reason or "deadline_exceeded",
            )
            raise ServiceError(
                "個別銘柄予測の計算が制限時間を超えたため終了しました。",
                "データ取得状況を確認して、時間をおいてもう一度実行してください。",
                504,
            )

    @contextmanager
    def stage(self, name: str):
        self.checkpoint(f"{name}:before")
        stage_started = time.perf_counter()
        try:
            yield
        finally:
            duration = float(time.perf_counter() - stage_started)
            with self._timing_lock:
                self.stage_seconds[name] = self.stage_seconds.get(name, 0.0) + duration
        self.checkpoint(f"{name}:after")




def normalize_japanese_security_code(value: str) -> tuple[str, str]:
    code = (value or "").strip().upper()
    if "." in code:
        raise ServiceError(
            "「.T」は付けず、証券コードだけを入力してください。",
            "「.T」を付けず、日本株の証券コードを1銘柄だけ入力してください。",
        )
    if any(separator in code for separator in (",", "\u3001", " ", "\t", "\n", "/")):
        raise ServiceError(
            "一度に入力できる証券コードは1銘柄だけです。",
            "「.T」を付けず、日本株の証券コードを1銘柄だけ入力してください。",
        )
    if not re.fullmatch(r"(?:\d{4}|\d{3}[A-Z])", code):
        raise ServiceError(
            "日本株の証券コード（ティッカー）を確認できませんでした。",
            "英数字4文字のコードを入力してください。海外ティッカーは対象外です。",
        )
    return code, f"{code}.T"


def _confirmed_basis_date(frame: pd.DataFrame, now: datetime | None = None) -> tuple[pd.Timestamp, str]:
    if frame.empty:
        raise ServiceError("確定した株価データがありません。")
    current = now.astimezone(JST) if now is not None else datetime.now(JST)
    today = pd.Timestamp(current.date())
    dates = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    available = dates[dates <= today]
    if len(available) == 0:
        raise ServiceError("確定した株価データがありません。")
    latest = available[-1]
    after_confirmed_close = current.weekday() < 5 and (current.hour, current.minute) >= (15, 40)
    if latest == today and not after_confirmed_close:
        previous = available[available < today]
        if len(previous) == 0:
            raise ServiceError("前営業日の確定終値を確認できませんでした。")
        return previous[-1], "場中"
    return latest, "大引け後" if latest == today else "寄り前"


def _validate_market_quality(data: IndividualMarketData) -> None:
    recent = data.frame.loc[: data.basis_date].tail(60)
    if recent.empty or float(recent["price"].isna().mean()) > 0.05:
        raise ServiceError("直近の終値に欠損が多いため予測できません。", "別の銘柄を指定するか、後ほど再実行してください。")
    volume = pd.to_numeric(recent.get("volume"), errors="coerce")
    if volume.notna().sum() >= 20:
        liquid_ratio = float((volume.fillna(0) > 0).mean())
        if liquid_ratio < MIN_LIQUID_DAYS_RATIO or float(volume.median()) <= 0:
            raise ServiceError(
                "直近の出来高が不足しているため予測対象にできません。",
                "流動性を確認できる別の日本株を指定してください。",
            )


def _download_nikkei_benchmark(index: pd.DatetimeIndex, warnings_list: list[str]) -> tuple[pd.Series | None, str | None]:
    """旧互換ヘルパー。通常予測は市場準備時の同一系列を再利用し、この関数を呼ばない。"""
    start = index[0].to_pydatetime() - timedelta(days=10)
    end = index[-1].to_pydatetime() + timedelta(days=1)
    try:
        frame = _download_base_frame("^N225", start, end)
        return frame["price"], "日経平均"
    except Exception:
        warnings_list.append("日経平均を取得できなかったため、チャートレポートの市場比較を省略しました。")
    return None, None


def _cache_fingerprint(data: IndividualMarketData, code: str) -> str:
    basis = data.basis_date
    recent = data.frame.loc[:basis, ["price", "volume"]].tail(10).round(6).to_json(date_format="iso")
    payload = "|".join([
        INDIVIDUAL_CACHE_VERSION,
        INDIVIDUAL_MODEL_VERSION,
        INDIVIDUAL_UI_SCHEMA_VERSION,
        INDIVIDUAL_MARKET_ALIGNMENT_VERSION,
        INDIVIDUAL_FEATURE_DEFINITION_VERSION,
        INDIVIDUAL_SELECTION_VERSION,
        INDIVIDUAL_CALIBRATION_VERSION,
        INDIVIDUAL_PUBLIC_LABEL_VERSION,
        INDIVIDUAL_DATA_COLLECTION_VERSION,
        MODEL_SETTINGS_HASH,
        FIXED_FEATURE_GROUP,
        CHART_ANALYSIS_VERSION,
        LONG_CHART_ANALYSIS_VERSION,
        SIX_STAGE_REPORT_VERSION,
        code,
        basis.strftime("%Y-%m-%d"),
        recent,
        json.dumps(data.metadata.get("used_japan_factors", []), ensure_ascii=False),
        json.dumps(data.metadata.get("used_external_factors", []), ensure_ascii=False),
        f"threshold={FIXED_DIRECTION_THRESHOLD:.2f}",
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _cache_path(code: str, fingerprint: str) -> Path:
    return OUTPUT_DIR / f"individual_cache_{code}_{fingerprint}.json"


def _cache_is_current(result: dict[str, Any]) -> bool:
    versions = result.get("individual_schema_versions", {})
    required_versions = {
        "cache": INDIVIDUAL_CACHE_VERSION,
        "market_alignment": INDIVIDUAL_MARKET_ALIGNMENT_VERSION,
        "feature_definition": INDIVIDUAL_FEATURE_DEFINITION_VERSION,
        "selection": INDIVIDUAL_SELECTION_VERSION,
        "calibration": INDIVIDUAL_CALIBRATION_VERSION,
        "public_labels": INDIVIDUAL_PUBLIC_LABEL_VERSION,
        "data_collection": INDIVIDUAL_DATA_COLLECTION_VERSION,
    }
    if result.get("individual_ui_schema_version") != INDIVIDUAL_UI_SCHEMA_VERSION:
        return False
    if result.get("model_version") not in (None, MODEL_SCHEMA_VERSION):
        return False
    if result.get("model_settings_hash") not in (None, MODEL_SETTINGS_HASH):
        return False
    if any(versions.get(key) != value for key, value in required_versions.items()):
        return False
    if float(result.get("fixed_direction_threshold", -1.0)) != FIXED_DIRECTION_THRESHOLD:
        return False
    if not result.get("data_collection_start") or result.get("data_collection_end") != result.get("basis_date"):
        return False
    labels = [item.get("label") for item in result.get("six_stage_trend", {}).get("items", [])]
    return labels == [display for _, display in SIX_STAGE_DISPLAY_ORDER]


def is_current_individual_result(result: dict[str, Any]) -> bool:
    return _cache_is_current(result)


def _load_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        chart_path = OUTPUT_DIR / Path(result.get("chart_url", "")).name
        long_chart_path = OUTPUT_DIR / Path(result.get("long_chart_url", "")).name
        if (
            not _cache_is_current(result)
            or result.get("chart_analysis", {}).get("schema_version") != CHART_ANALYSIS_VERSION
            or result.get("long_chart_analysis", {}).get("schema_version") != LONG_CHART_ANALYSIS_VERSION
            or result.get("six_stage_trend", {}).get("schema_version") != SIX_STAGE_REPORT_VERSION
            or not chart_path.exists()
            or not long_chart_path.exists()
        ):
            return None
        result["cache"] = {"used": True, "version": INDIVIDUAL_CACHE_VERSION, "key": path.stem}
        result.setdefault("processing_timing", {})["result_cache_hit"] = True
        return result
    except (OSError, ValueError, TypeError):
        return None


def _expected_current_basis_date(now: datetime | None = None) -> pd.Timestamp:
    current = now.astimezone(JST) if now is not None else datetime.now(JST)
    today = pd.Timestamp(current.date())
    use_today = current.weekday() < 5 and (current.hour, current.minute) >= (15, 40)
    try:
        import holidays
        holiday_dates = set(holidays.Japan(years=range(today.year - 1, today.year + 2)).keys())
    except Exception:
        holiday_dates = set()
    if use_today and today.date() not in holiday_dates:
        return today
    candidate = today - pd.Timedelta(days=1)
    while candidate.weekday() >= 5 or candidate.date() in holiday_dates:
        candidate -= pd.Timedelta(days=1)
    return candidate


def _load_current_code_cache(code: str) -> dict[str, Any] | None:
    expected = _expected_current_basis_date().strftime("%Y-%m-%d")
    candidates = sorted(OUTPUT_DIR.glob(f"individual_cache_{code}_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        cached = _load_cache(path)
        if cached is not None and cached.get("basis_date") == expected:
            cached["cache"]["fast_reuse"] = True
            return cached
    return None


def _save_cache(path: Path, result: dict[str, Any]) -> None:
    payload = dict(result)
    payload["cache"] = {"used": False, "version": INDIVIDUAL_CACHE_VERSION, "key": path.stem}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)




def _display_feature_name(name: str) -> str:
    replacements = {
        "stock_": "対象銘柄 ",
        "nikkei_": "日経平均 ",
        "spx_": "S&P 500 ",
        "nasdaq_": "NASDAQ ",
        "vix_": "VIX ",
        "usdjpy_": "ドル円 ",
    }
    result = name
    for prefix, label in replacements.items():
        if result.startswith(prefix):
            result = label + result[len(prefix):]
            break
    return result.replace("_", " ")


def _fixed_factor_lists(data: IndividualMarketData) -> tuple[list[str], list[str]]:
    selected = [str(value) for value in data.metadata.get("fixed_logistic_factors", [])]
    fixed_labels = {"S&P 500", "NASDAQ", "VIX", "ドル円"}
    excluded = [
        str(source["label"])
        for source in EXTERNAL_SOURCES.values()
        if str(source["label"]) not in fixed_labels
    ]
    missing = {
        str(EXTERNAL_SOURCES[factor]["label"])
        for factor in data.metadata.get("missing_external_factors", [])
        if factor in EXTERNAL_SOURCES and str(EXTERNAL_SOURCES[factor]["label"]) in fixed_labels
    }
    excluded.extend(sorted(missing))
    return list(dict.fromkeys(selected)), list(dict.fromkeys(excluded))


def _prepare_market_data(ticker: str, timing: dict[str, float | bool]) -> IndividualMarketData:
    try:
        return prepare_individual_market_data(ticker, timing=timing)
    except TypeError as error:
        if "timing" not in str(error):
            raise
        return prepare_individual_market_data(ticker)




def predict_individual(
    ticker: str,
    execution_control: IndividualExecutionControl | None = None,
) -> dict[str, Any]:
    """固定ロジスティック回帰2モデルだけで最新予測を作る高速な通常経路。"""
    started = time.perf_counter()
    control = execution_control or IndividualExecutionControl()
    if not _prediction_lock.acquire(blocking=False):
        raise ServiceError(
            "別の個別銘柄予測が実行中です。処理が終了してから、もう一度お試しください。",
            "実行中の計算が終了すると、再度実行できます。",
            409,
        )
    try:
        with control.stage("input_validation"):
            code, normalized_ticker = normalize_japanese_security_code(ticker)
        with control.stage("cache_lookup"):
            current_cache = _load_current_code_cache(code)
        if current_cache is not None:
            logger.info(
                "individual timing ticker=%s result_cache_hit=true total_elapsed_seconds=%.3f",
                code,
                time.perf_counter() - started,
            )
            return current_cache

        market_timing: dict[str, float | bool] = {}
        try:
            with control.stage("market_data_preparation"):
                data = _prepare_market_data(normalized_ticker, market_timing)
        except ServiceError as error:
            if "履歴" in error.message:
                raise ServiceError(
                    "この銘柄は、予測に必要な株価履歴が不足しています。",
                    "証券コードが正しい場合は、取引履歴が蓄積してから再度お試しください。",
                    error.status_code,
                ) from error
            if "株価データを取得できません" in error.message:
                raise ServiceError(
                    "対象銘柄の株価データを取得できなかったため、今回は予測できません。",
                    "通信状態とデータ提供元の状況を確認して、時間をおいてもう一度実行してください。",
                    503,
                ) from error
            raise
        _validate_market_quality(data)

        fingerprint = _cache_fingerprint(data, code)
        cache_path = _cache_path(code, fingerprint)
        with control.stage("fingerprint_cache_lookup"):
            cached = _load_cache(cache_path)
        if cached is not None:
            logger.info(
                "individual timing ticker=%s result_cache_hit=true total_elapsed_seconds=%.3f",
                code,
                time.perf_counter() - started,
            )
            return cached

        with control.stage("fixed_logistic_prediction"):
            latest = latest_prediction(data)
        with control.stage("reports_and_charts"):
            benchmark = data.benchmark_series
            chart_report = build_chart_analysis(data.frame, data.basis_date, benchmark, "日経平均")
            long_report = build_long_term_chart_analysis(data.frame, data.basis_date, benchmark, "日経平均")
            report_started = time.perf_counter()
            six_stage_trend = build_six_stage_trend_report(latest["probabilities"])
            report_seconds = float(time.perf_counter() - report_started)
            chart_started = time.perf_counter()
            chart_url = plot_individual_chart(data.frame, normalized_ticker, data.basis_date, chart_report)
            long_chart_url = plot_individual_long_chart(data.frame, normalized_ticker, data.basis_date)
            chart_seconds = float(time.perf_counter() - chart_started)
            display_name = get_ticker_name(normalized_ticker)
            latest_row = data.frame.loc[data.basis_date]
            prediction_class = DISPLAY_CLASS_LABELS[latest["prediction_class_raw"]]

        selected_factors, excluded_factors = _fixed_factor_lists(data)
        importance = [
            {**item, "display_name": _display_feature_name(str(item["internal_name"]))}
            for item in latest["feature_importance_top10"]
        ]
        schema_versions = {
            "cache": INDIVIDUAL_CACHE_VERSION,
            "market_alignment": INDIVIDUAL_MARKET_ALIGNMENT_VERSION,
            "feature_definition": INDIVIDUAL_FEATURE_DEFINITION_VERSION,
            "selection": INDIVIDUAL_SELECTION_VERSION,
            "calibration": INDIVIDUAL_CALIBRATION_VERSION,
                "public_labels": INDIVIDUAL_PUBLIC_LABEL_VERSION,
            "data_collection": INDIVIDUAL_DATA_COLLECTION_VERSION,
        }
        elapsed = float(time.perf_counter() - started)
        external_seconds = float(market_timing.get("stock_external_fetch_seconds", 0.0)) + float(
            market_timing.get("common_market_external_fetch_seconds", 0.0)
        )
        internal_seconds = max(0.0, elapsed - external_seconds)
        prediction_timing = latest["timing"]
        processing_timing = {
            "input_validation_seconds": float(control.stage_seconds.get("input_validation", 0.0)),
            "target_stock_external_fetch_seconds": float(market_timing.get("stock_external_fetch_seconds", 0.0)),
            "common_market_external_fetch_seconds": float(market_timing.get("common_market_external_fetch_seconds", 0.0)),
            "local_cache_read_seconds": float(market_timing.get("local_cache_read_seconds", 0.0)),
            "data_join_seconds": float(market_timing.get("data_join_seconds", 0.0)),
            "feature_build_seconds": float(market_timing.get("feature_build_seconds", 0.0)),
            "preprocess_seconds": float(prediction_timing.get("preprocess_seconds", 0.0)),
            "direction_fit_seconds": float(prediction_timing.get("direction_fit_seconds", 0.0)),
            "six_class_fit_seconds": float(prediction_timing.get("six_class_fit_seconds", 0.0)),
            "inference_seconds": float(prediction_timing.get("inference_seconds", 0.0)),
            "report_seconds": report_seconds,
            "chart_data_generation_seconds": chart_seconds,
            "internal_total_seconds": internal_seconds,
            "external_fetch_seconds": external_seconds,
            "total_elapsed_seconds": elapsed,
            "result_cache_hit": False,
            "common_market_cache_hit": bool(market_timing.get("common_market_cache_hit", False)),
        }
        result: dict[str, Any] = {
            "kind": "individual_prediction",
            "individual_ui_schema_version": INDIVIDUAL_UI_SCHEMA_VERSION,
            "individual_schema_versions": schema_versions,
            "model_version": INDIVIDUAL_MODEL_VERSION,
            "model_settings_hash": MODEL_SETTINGS_HASH,
            "chart_analysis_version": CHART_ANALYSIS_VERSION,
            "long_chart_analysis_version": LONG_CHART_ANALYSIS_VERSION,
            "six_stage_report_version": SIX_STAGE_REPORT_VERSION,
            "company_name": display_name,
            "security_code": code,
            "ticker": normalized_ticker,
            "basis_date": data.basis_date.strftime("%Y-%m-%d"),
            "forecast_target_date": latest["forecast_target_date"],
            "forecast_horizon_label": "5営業日先",
            "fetched_at": data.fetched_at,
            "latest_price": float(latest_row["price"]),
            "prediction_context": data.prediction_context,
            "prediction_mode": data.prediction_mode,
            "direction": latest["direction"],
            "direction_key": latest["direction_key"],
            "prediction_class": prediction_class,
            "prediction_class_raw": latest["prediction_class_raw"],
            "fixed_direction_threshold": FIXED_DIRECTION_THRESHOLD,
            "direction_scores": {
                "raw_up": latest["raw_up_score"],
                "calibrated_up": latest["calibrated_up_score"],
                "calibrated_down": latest["calibrated_down_score"],
                "calibration_status": latest["calibration"]["calibration_status"],
            },
            "probabilities": latest["probabilities"],
            "chart_url": chart_url,
            "chart_data": chart_data(data.frame, data.basis_date),
            "chart_analysis": chart_report,
            "long_chart_url": long_chart_url,
            "long_chart_data": long_chart_data(data.frame, data.basis_date),
            "long_chart_analysis": long_report,
            "six_stage_trend": six_stage_trend,
            "data_collection_start": latest["data_collection_start"],
            "data_collection_end": latest["data_collection_end"],
            "data_collection_period": {"start": latest["data_collection_start"], "end": latest["data_collection_end"]},
            "data_collection_definition": latest["data_collection_definition"],
            "model_name": "ロジスティック回帰",
            "direction_model_name": "二値ロジスティック回帰",
            "six_class_model_name": "多クラスロジスティック回帰",
            "model_configuration_label": "全銘柄共通固定構成",
            "feature_group": FIXED_FEATURE_GROUP,
            "selected_factors": selected_factors,
            "excluded_factors": excluded_factors,
            "factor_selection_definition": "候補比較は行わず、全銘柄共通の固定ファクター構成を使用しています。",
            "feature_importance_top10": importance,
            "model_training": {
                "target_specific": True,
                "other_tickers_used_as_training_rows": False,
                "integrated_market_model": True,
                "direction_target": "5営業日先の終値が基準日の終値より上か下か",
                "direction_model_is_independent_binary": True,
                "six_class_model_is_separate": True,
                "training_window_years": TRAINING_YEARS,
                "warmup_trading_days": WARMUP_TRADING_DAYS,
                "latest_training_start": latest["training_start"],
                "latest_training_end": latest["training_end"],
                "latest_training_samples": latest["training_samples"],
                "inference_row_in_training": latest["inference_row_in_training"],
                "direction_model": latest["direction_model"],
                "six_class_model": latest["six_class_model"],
                "direction_features": latest["direction_features"],
                "six_class_features": latest["six_class_features"],
                "feature_matrix_reused": True,
                "fit_counts": latest["fit_counts"],
                "cross_validation_count": 0,
                "candidate_comparison_count": 0,
                "global_external_shift_sessions": 0,
                "fixed_direction_threshold": FIXED_DIRECTION_THRESHOLD,
            },
            "internal_audit": {
                "market_data": data.metadata,
                "fixed_configuration": latest["selection"],
                "calibration": latest["calibration"],
                "data_collection_start": latest["data_collection_start"],
                "data_collection_end": latest["data_collection_end"],
                "data_collection_definition": latest["data_collection_definition"],
                "cache_schema_version": INDIVIDUAL_CACHE_VERSION,
                "model_schema_version": MODEL_SCHEMA_VERSION,
                "model_settings_hash": MODEL_SETTINGS_HASH,
                "processing_timing": processing_timing,
                "processing_timeout_seconds": control.max_seconds,
                "model_fit_count": latest["fit_counts"]["total"],
                "cross_validation_count": 0,
                "candidate_comparison_count": 0,
                "convergence_warnings": latest["convergence_warnings"],
            },
            "processing_timing": processing_timing,
            "warnings": list(dict.fromkeys([*data.warnings, *latest["convergence_warnings"]])),
            "cache": {"used": False, "version": INDIVIDUAL_CACHE_VERSION, "key": cache_path.stem},
            "processing_seconds": elapsed,
            "disclaimer": (
                "本予測は情報提供および研究目的の参考情報です。特定の金融商品の売買を推奨するものではなく、"
                "利益を保証するものでもありません。投資判断は利用者自身の責任で行ってください。"
            ),
        }
        _save_cache(cache_path, result)
        logger.info(
            "individual timing ticker=%s external_fetch_seconds=%.3f feature_build_seconds=%.3f "
            "preprocess_seconds=%.3f direction_fit_seconds=%.3f six_class_fit_seconds=%.3f "
            "inference_seconds=%.3f report_seconds=%.3f internal_total_seconds=%.3f "
            "total_elapsed_seconds=%.3f result_cache_hit=false",
            code,
            processing_timing["external_fetch_seconds"],
            processing_timing["feature_build_seconds"],
            processing_timing["preprocess_seconds"],
            processing_timing["direction_fit_seconds"],
            processing_timing["six_class_fit_seconds"],
            processing_timing["inference_seconds"],
            processing_timing["report_seconds"],
            processing_timing["internal_total_seconds"],
            processing_timing["total_elapsed_seconds"],
        )
        return result
    finally:
        _prediction_lock.release()
