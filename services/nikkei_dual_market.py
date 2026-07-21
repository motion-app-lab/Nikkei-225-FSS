from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, time as clock_time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from .common import (
    BASE_DIR,
    CLASS_ORDER,
    OUTPUT_DIR,
    PREDICTION_HORIZON,
    PredictionData,
    ServiceError,
    _download_base_frame,
    _six_probabilities,
    calc_rsi,
    channel_features,
    classify_change,
    extract_yfinance_column,
    normalize_datetime_index,
    overheat_score,
)
from .nikkei_direction_comparison import moving_block_bootstrap_intervals


MODEL_VERSION = "nikkei_dual_market_rolling_v1"
MODEL_SETTINGS_VERSION = "dual_market_8y_2y_v1"
CACHE_VERSION = "nikkei_dual_cache_v1"
FEATURE_DEFINITION_VERSION = "dual_calendar_asof_features_v1"
EXTERNAL_ALIGNMENT_VERSION = "source_calendar_available_at_asof_v1"
MODEL_SELECTION_VERSION = "dual_side_nested_selection_v1"
THRESHOLD_LOGIC_VERSION = "combined_score_threshold_v2"

LEARNING_YEARS = 8
EVALUATION_YEARS = 2
WARMUP_TRADING_DAYS = 300
OUTER_FOLDS = 8
INNER_FOLDS = 3
PURGE_TRADING_DAYS = 5
HALF_LIFE_YEARS = 4.0
JAPAN_SCORE_WEIGHT = 0.50
OVERSEAS_SCORE_WEIGHT = 0.50
RANDOM_SEED = 42
MODEL_TIE_TOLERANCE = 0.001
WEIGHT_MIN_BALANCED_IMPROVEMENT = 0.005
WEIGHT_MAX_WORST_FOLD_DETERIORATION = 0.010
MIN_PREDICTION_SHARE = 0.10
THRESHOLD_MIN_BALANCED_IMPROVEMENT = 0.005
THRESHOLD_MIN_IMPROVED_FOLDS = 2
THRESHOLD_MIN_FOLD_IMPROVEMENT = 0.001
THRESHOLD_TIE_TOLERANCE = 0.001
STANDARD_THRESHOLD = 0.50
THRESHOLD_CANDIDATES = tuple(round(float(value), 2) for value in np.arange(0.40, 0.601, 0.02))
BOOTSTRAP_BLOCK_LENGTH = 10
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 42
MARKET_CLOSE_CONFIRMATION_JST = clock_time(15, 40)
HISTORICAL_INTRADAY_CUTOFF_JST = clock_time(9, 0)

JST = ZoneInfo("Asia/Tokyo")
NEW_YORK = ZoneInfo("America/New_York")
UTC = timezone.utc

EXTERNAL_SOURCES: dict[str, dict[str, Any]] = {
    "spx": {"label": "S&P 500", "ticker": "^GSPC", "kind": "us_close", "safety_lag": 0},
    "dow": {"label": "NYダウ", "ticker": "^DJI", "kind": "us_close", "safety_lag": 0},
    "nasdaq": {"label": "NASDAQ", "ticker": "^IXIC", "kind": "us_close", "safety_lag": 0},
    "vix": {"label": "VIX", "ticker": "^VIX", "kind": "us_close", "safety_lag": 0},
    "sox": {"label": "SOX", "ticker": "^SOX", "kind": "us_close", "safety_lag": 0},
    "nvda": {"label": "NVIDIA", "ticker": "NVDA", "kind": "us_close", "safety_lag": 0},
    "usdjpy": {"label": "ドル円", "ticker": "JPY=X", "kind": "uncertain_daily", "safety_lag": 1},
    "oil": {"label": "原油", "ticker": "CL=F", "kind": "uncertain_daily", "safety_lag": 1},
    "gold": {"label": "金", "ticker": "GC=F", "kind": "uncertain_daily", "safety_lag": 1},
    "btc": {"label": "Bitcoin", "ticker": "BTC-USD", "kind": "utc_daily", "safety_lag": 1},
}

JAPAN_FEATURES = [
    "open",
    "high",
    "low",
    "price",
    "volume",
    "return_1d",
    "return_2d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "volume_change",
    "rsi14",
    "ch_trend",
    "ch_upper",
    "ch_lower",
    "ch_pos",
    "overheat_score",
    "ma20_gap",
    "ma50_gap",
    "ma100_gap",
    "ma200_gap",
    "ma20_slope5",
    "ma50_slope5",
    "ma100_slope5",
    "ma20_above_ma100",
    "intraday_range",
    "opening_gap",
    "atr14_ratio",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "volatility_60d",
    "volatility_5_60_ratio",
    "volatility_20d_change5",
    "distance_high20",
    "distance_high60",
    "distance_high252",
    "distance_low20",
    "distance_low60",
]

LEGACY_SIX_FEATURES = [
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

MODEL_CANDIDATES = [
    {"id": "catboost_all", "label": "CatBoost", "simplicity": 3},
    {"id": "catboost_selected", "label": "学習内特徴量選択CatBoost", "simplicity": 3},
    {"id": "logistic", "label": "Logistic Regression", "simplicity": 1},
    {"id": "extra_trees", "label": "Extra Trees", "simplicity": 2},
    {"id": "simple_average", "label": "主要3モデル単純平均", "simplicity": 4},
]

MODEL_PARAMETERS = {
    "catboost": {
        "iterations": 100,
        "depth": 4,
        "learning_rate": 0.04,
        "l2_leaf_reg": 6.0,
        "random_seed": RANDOM_SEED,
    },
    "logistic": {"C": 0.5, "max_iter": 1_000, "solver": "liblinear", "random_state": RANDOM_SEED},
    "extra_trees": {
        "n_estimators": 180,
        "max_depth": 7,
        "min_samples_leaf": 8,
        "max_features": "sqrt",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    },
    "feature_selector": {
        "n_estimators": 120,
        "max_depth": 6,
        "min_samples_leaf": 8,
        "max_features": "sqrt",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    },
}

SETTINGS_DIR = BASE_DIR / "model_settings"
SETTINGS_PATH = SETTINGS_DIR / "nikkei_dual_market.json"
MODEL_ARTIFACT_PATH = SETTINGS_DIR / "nikkei_dual_market_models.joblib"
HISTORY_PATH = SETTINGS_DIR / "nikkei_dual_market_history.json"


@dataclass
class DualMarketData:
    base_frame: pd.DataFrame
    japanese_frame: pd.DataFrame
    intraday_frame: pd.DataFrame
    after_close_frame: pd.DataFrame
    inference_rows: dict[str, pd.DataFrame]
    japan_features: list[str]
    overseas_features: list[str]
    warnings: list[str]
    fetched_at: str
    metadata: dict[str, Any]


def _now_jst() -> datetime:
    return datetime.now(JST)


def _date_text(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _timestamp_text(value: Any) -> str:
    return pd.Timestamp(value).isoformat()


def _safe_float(value: Any) -> float:
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return _timestamp_text(value)
    return value


def is_japan_market_day(value: Any) -> bool:
    date_value = pd.Timestamp(value).date()
    if date_value.weekday() >= 5 or (date_value.month == 12 and date_value.day == 31) or (
        date_value.month == 1 and date_value.day in {1, 2, 3}
    ):
        return False
    try:
        import holidays

        return date_value not in holidays.country_holidays("JP", years=[date_value.year])
    except ImportError:
        return True


def next_japan_market_days(base_date: Any, count: int = PREDICTION_HORIZON) -> list[pd.Timestamp]:
    days: list[pd.Timestamp] = []
    current = pd.Timestamp(base_date).normalize()
    while len(days) < count:
        current += pd.Timedelta(days=1)
        if is_japan_market_day(current):
            days.append(current)
    return days


def resolve_prediction_context(base: pd.DataFrame, now: datetime | None = None) -> dict[str, Any]:
    current = now.astimezone(JST) if now and now.tzinfo else (now.replace(tzinfo=JST) if now else _now_jst())
    today = pd.Timestamp(current.date())
    complete_columns = [column for column in ("open", "high", "low", "price") if column in base]
    today_is_market_day = is_japan_market_day(today)
    after_close = current.timetz().replace(tzinfo=None) >= MARKET_CLOSE_CONFIRMATION_JST
    warning = None
    if today_is_market_day and not after_close:
        eligible = base.loc[base.index < today]
        if eligible.empty:
            raise ServiceError("場中予測に使用できる前営業日の確定終値がありません。")
        base_date = eligible.index[-1]
        context = "intraday"
        mode_label = "場中"
        cutoff = pd.Timestamp.combine(today.date(), HISTORICAL_INTRADAY_CUTOFF_JST).tz_localize(JST)
    elif today_is_market_day and today in base.index and not base.loc[today, complete_columns].isna().any():
        base_date = today
        context = "after_close"
        mode_label = "大引け後"
        cutoff = pd.Timestamp.combine(today.date(), MARKET_CLOSE_CONFIRMATION_JST).tz_localize(JST)
    else:
        eligible = base.loc[base.index <= today]
        if eligible.empty:
            raise ServiceError("確定済みの日経平均終値を確認できませんでした。")
        base_date = eligible.index[-1]
        context = "after_close" if not today_is_market_day or after_close else "intraday"
        mode_label = "大引け後" if context == "after_close" else "場中"
        cutoff_time = MARKET_CLOSE_CONFIRMATION_JST if context == "after_close" else HISTORICAL_INTRADAY_CUTOFF_JST
        cutoff_date = base_date if context == "after_close" else today
        cutoff = pd.Timestamp.combine(pd.Timestamp(cutoff_date).date(), cutoff_time).tz_localize(JST)
        if today_is_market_day and after_close and today not in base.index:
            warning = "当日の確定日足を確認できなかったため、前営業日の確定終値を使用しました。"
    target_days = next_japan_market_days(base_date, PREDICTION_HORIZON)
    return {
        "prediction_context": context,
        "prediction_mode": mode_label,
        "prediction_timestamp_jst": _timestamp_text(current),
        "nikkei_base_date": _date_text(base_date),
        "nikkei_base_close": float(base.loc[base_date, "price"]),
        "external_cutoff_timestamp_jst": _timestamp_text(cutoff),
        "target_date": _date_text(target_days[-1]),
        "warning": warning,
    }


def _download_external_raw(start: datetime, end: datetime) -> tuple[dict[str, pd.Series], list[str]]:
    warnings: list[str] = []
    try:
        import yfinance as yf

        tickers = [source["ticker"] for source in EXTERNAL_SOURCES.values()]
        raw = yf.download(
            tickers,
            start=start,
            end=end + timedelta(days=1),
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column",
            timeout=30,
        )
    except Exception as error:
        raise ServiceError(
            "米国・海外側データを取得できないため、統合予測を作成できませんでした。",
            "インターネット接続を確認し、時間をおいて再度実行してください。",
            503,
        ) from error
    results: dict[str, pd.Series] = {}
    for factor, source in EXTERNAL_SOURCES.items():
        series = extract_yfinance_column(raw, "Close", source["ticker"])
        if series.empty:
            warnings.append(f"{source['label']}を取得できなかったため、モデル再評価候補から除外します。")
            continue
        series.index = normalize_datetime_index(series.index)
        series = series.groupby(level=0).last().sort_index().replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) < 260:
            warnings.append(f"{source['label']}は履歴不足のため、モデル再評価候補から除外します。")
            continue
        results[factor] = series.astype(float)
    if not results:
        raise ServiceError(
            "米国・海外側データを取得できないため、統合予測を作成できませんでした。",
            "Yahoo Financeの通信状態を確認してください。",
        )
    return results, warnings


def external_available_timestamp(source_date: Any, kind: str) -> pd.Timestamp:
    date_value = pd.Timestamp(source_date).date()
    if kind in {"us_close", "uncertain_daily"}:
        local_time = clock_time(16, 15) if kind == "us_close" else clock_time(17, 15)
        localized = datetime.combine(date_value, local_time, tzinfo=NEW_YORK)
        return pd.Timestamp(localized.astimezone(JST))
    utc_close = datetime.combine(date_value + timedelta(days=1), clock_time(0, 15), tzinfo=UTC)
    return pd.Timestamp(utc_close.astimezone(JST))


def build_external_feature_frame(factor: str, series: pd.Series) -> pd.DataFrame:
    source = EXTERNAL_SOURCES[factor]
    values = series.astype(float).sort_index()
    frame = pd.DataFrame(index=values.index)
    frame[factor] = values
    frame[f"{factor}_ret"] = values.pct_change(fill_method=None) * 100
    frame[f"{factor}_ret5"] = values.pct_change(5, fill_method=None) * 100
    frame[f"{factor}_ret20"] = values.pct_change(20, fill_method=None) * 100
    frame[f"{factor}_vol20"] = frame[f"{factor}_ret"].rolling(20, min_periods=20).std()
    moving_average = values.rolling(20, min_periods=20).mean()
    frame[f"{factor}_ma20_gap"] = (values / moving_average - 1.0) * 100
    frame["source_date"] = pd.to_datetime(frame.index)
    frame["available_at_jst"] = [external_available_timestamp(value, source["kind"]) for value in frame.index]
    lag = int(source["safety_lag"])
    feature_columns = [column for column in frame.columns if column not in {"source_date", "available_at_jst"}]
    if lag:
        frame[feature_columns] = frame[feature_columns].shift(lag)
        frame["source_date"] = frame["source_date"].shift(lag)
    frame = frame.dropna(subset=["source_date"]).sort_values("available_at_jst")
    return frame


def asof_align_external(
    prediction_cutoffs: pd.Series,
    external_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    cutoffs = pd.DataFrame(
        {"base_date": prediction_cutoffs.index, "prediction_cutoff_jst": list(prediction_cutoffs.values)}
    ).sort_values("prediction_cutoff_jst")
    aligned = pd.DataFrame(index=pd.DatetimeIndex(prediction_cutoffs.index))
    diagnostics: dict[str, pd.DataFrame] = {}
    for factor, source_frame in external_frames.items():
        feature_columns = [column for column in source_frame.columns if column not in {"source_date", "available_at_jst"}]
        right = source_frame[["available_at_jst", "source_date", *feature_columns]].sort_values("available_at_jst")
        merged = pd.merge_asof(
            cutoffs,
            right,
            left_on="prediction_cutoff_jst",
            right_on="available_at_jst",
            direction="backward",
            allow_exact_matches=True,
        ).set_index("base_date")
        aligned.loc[merged.index, feature_columns] = merged[feature_columns]
        diagnostics[factor] = merged[["prediction_cutoff_jst", "source_date", "available_at_jst"]].copy()
    return aligned.sort_index(), diagnostics


def build_japanese_features(base: pd.DataFrame) -> pd.DataFrame:
    frame = base.copy()
    price = frame["price"].astype(float)
    for window in (1, 2, 3, 5, 10, 20):
        frame[f"return_{window}d"] = price.pct_change(window, fill_method=None) * 100
    frame["volume_change"] = frame["volume"].pct_change(fill_method=None) * 100
    frame["rsi14"] = calc_rsi(price, 14)
    frame = frame.join(channel_features(price))
    frame["overheat_score"] = overheat_score(frame)
    moving_averages: dict[int, pd.Series] = {}
    for window in (20, 50, 100, 200):
        moving_averages[window] = price.rolling(window, min_periods=window).mean()
        frame[f"ma{window}_gap"] = (price / moving_averages[window] - 1.0) * 100
    for window in (20, 50, 100):
        frame[f"ma{window}_slope5"] = moving_averages[window].pct_change(5, fill_method=None) * 100
    frame["ma20_above_ma100"] = (moving_averages[20] >= moving_averages[100]).where(
        moving_averages[20].notna() & moving_averages[100].notna()
    ).astype(float)
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
    for window in (5, 10, 20, 60):
        frame[f"volatility_{window}d"] = frame["return_1d"].rolling(window, min_periods=window).std()
    frame["volatility_5_60_ratio"] = frame["volatility_5d"] / frame["volatility_60d"].replace(0, np.nan)
    frame["volatility_20d_change5"] = frame["volatility_20d"].pct_change(5, fill_method=None) * 100
    for window in (20, 60, 252):
        high = price.rolling(window, min_periods=window).max()
        frame[f"distance_high{window}"] = (price / high - 1.0) * 100
    for window in (20, 60):
        low = price.rolling(window, min_periods=window).min()
        frame[f"distance_low{window}"] = (price / low - 1.0) * 100
    future = price.shift(-PREDICTION_HORIZON)
    frame["target_change"] = np.where(future.notna(), (future / price - 1.0) * 100, np.nan)
    direction = pd.Series(np.nan, index=frame.index, dtype=float)
    direction.loc[future.gt(price)] = 1.0
    direction.loc[future.lt(price)] = 0.0
    frame["target_direction"] = direction
    return frame.replace([np.inf, -np.inf], np.nan)


def historical_prediction_cutoffs(index: pd.DatetimeIndex, context: str) -> pd.Series:
    values: list[pd.Timestamp] = []
    for position, date_value in enumerate(index):
        if context == "intraday":
            if position + 1 < len(index):
                prediction_date = index[position + 1]
            else:
                prediction_date = next_japan_market_days(date_value, 1)[0]
            cutoff_time = HISTORICAL_INTRADAY_CUTOFF_JST
        else:
            prediction_date = date_value
            cutoff_time = MARKET_CLOSE_CONFIRMATION_JST
        values.append(pd.Timestamp.combine(prediction_date.date(), cutoff_time).tz_localize(JST))
    return pd.Series(values, index=index, dtype="object")


def _add_legacy_six_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "nasdaq_ret" in result:
        result["corr20_nikkei_nasdaq"] = result["return_1d"].rolling(20, min_periods=20).corr(
            result["nasdaq_ret"]
        )
    return result


def _diagnostic_for_cutoff(
    cutoff: pd.Timestamp,
    external_frames: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for factor, frame in external_frames.items():
        eligible = frame.loc[frame["available_at_jst"] <= cutoff]
        source = EXTERNAL_SOURCES[factor]
        if eligible.empty:
            result[factor] = {
                "data_name": source["label"],
                "ticker": source["ticker"],
                "source_date": None,
                "available_at_jst": None,
                "prediction_timestamp_jst": _timestamp_text(cutoff),
                "selected_source_date": None,
                "missing_handling": "no past value available",
                "safety_lag_source_sessions": source["safety_lag"],
            }
            continue
        row = eligible.iloc[-1]
        result[factor] = {
            "data_name": source["label"],
            "ticker": source["ticker"],
            "source_date": _date_text(row["source_date"]),
            "available_at_jst": _timestamp_text(row["available_at_jst"]),
            "prediction_timestamp_jst": _timestamp_text(cutoff),
            "selected_source_date": _date_text(row["source_date"]),
            "missing_handling": "past-only asof; no bfill or nearest",
            "safety_lag_source_sessions": source["safety_lag"],
        }
    return result


def prepare_dual_market_data(
    model_reevaluation: bool,
    now: datetime | None = None,
) -> DualMarketData:
    started = time.perf_counter()
    current = now or _now_jst()
    current_naive = current.replace(tzinfo=None) if current.tzinfo else current
    approximate_years = LEARNING_YEARS + (EVALUATION_YEARS if model_reevaluation else 0) + 2
    requested_start = pd.Timestamp(current_naive) - pd.DateOffset(years=approximate_years)
    base = _download_base_frame("^N225", requested_start.to_pydatetime(), current_naive)
    context = resolve_prediction_context(base, current if current.tzinfo else current.replace(tzinfo=JST))
    base_date = pd.Timestamp(context["nikkei_base_date"])
    base = base.loc[base.index <= base_date].copy()
    latest_labeled_position = len(base) - PREDICTION_HORIZON - 1
    if latest_labeled_position < 500:
        raise ServiceError("日経平均の正解確定済みデータが不足しています。")
    latest_labeled_date = base.index[latest_labeled_position]
    evaluation_start_target = latest_labeled_date - pd.DateOffset(years=EVALUATION_YEARS)
    if model_reevaluation:
        evaluation_candidates = base.index[(base.index >= evaluation_start_target) & (base.index <= latest_labeled_date)]
        if len(evaluation_candidates) < 400:
            raise ServiceError("直近2年間の精度評価に必要な日経平均データが不足しています。")
        first_evaluation_position = int(base.index.get_loc(evaluation_candidates[0]))
        earliest_training_end_position = first_evaluation_position - PURGE_TRADING_DAYS - 1
        earliest_training_target = base.index[earliest_training_end_position] - pd.DateOffset(
            years=LEARNING_YEARS
        )
    else:
        evaluation_candidates = pd.DatetimeIndex([])
        earliest_training_target = base_date - pd.DateOffset(years=LEARNING_YEARS)
    first_training_position = int(base.index.searchsorted(earliest_training_target, side="left"))
    extension_attempts = 0
    while first_training_position < WARMUP_TRADING_DAYS and extension_attempts < 4:
        extension_attempts += 1
        requested_start -= pd.DateOffset(years=1)
        base = _download_base_frame("^N225", requested_start.to_pydatetime(), current_naive)
        context = resolve_prediction_context(base, current if current.tzinfo else current.replace(tzinfo=JST))
        base_date = pd.Timestamp(context["nikkei_base_date"])
        base = base.loc[base.index <= base_date].copy()
        latest_labeled_position = len(base) - PREDICTION_HORIZON - 1
        latest_labeled_date = base.index[latest_labeled_position]
        evaluation_start_target = latest_labeled_date - pd.DateOffset(years=EVALUATION_YEARS)
        evaluation_candidates = base.index[(base.index >= evaluation_start_target) & (base.index <= latest_labeled_date)]
        if model_reevaluation:
            first_evaluation_position = int(base.index.get_loc(evaluation_candidates[0]))
            earliest_training_end_position = first_evaluation_position - PURGE_TRADING_DAYS - 1
            earliest_training_target = base.index[earliest_training_end_position] - pd.DateOffset(
                years=LEARNING_YEARS
            )
        else:
            earliest_training_target = base_date - pd.DateOffset(years=LEARNING_YEARS)
        first_training_position = int(base.index.searchsorted(earliest_training_target, side="left"))
    if first_training_position < WARMUP_TRADING_DAYS:
        raise ServiceError(
            "300取引日のウォームアップ期間を確保できませんでした。",
            "市場データの取得状態を確認してモデル再評価を実行してください。",
        )
    warmup_start_position = first_training_position - WARMUP_TRADING_DAYS
    base = base.iloc[warmup_start_position:].copy()
    warmup_start = base.index[0]
    first_training_date = base.index[WARMUP_TRADING_DAYS]
    external_raw, warnings = _download_external_raw(
        warmup_start.to_pydatetime(),
        current_naive,
    )
    external_frames = {
        factor: build_external_feature_frame(factor, series)
        for factor, series in external_raw.items()
    }
    if len(external_frames) < 3:
        raise ServiceError(
            "米国・海外側データを取得できないため、統合予測を作成できませんでした。",
            "複数の外部系列を取得できる状態で再度実行してください。",
        )
    japanese = build_japanese_features(base)
    intraday_cutoffs = historical_prediction_cutoffs(base.index, "intraday")
    after_close_cutoffs = historical_prediction_cutoffs(base.index, "after_close")
    aligned_intraday, _ = asof_align_external(intraday_cutoffs, external_frames)
    aligned_after, _ = asof_align_external(after_close_cutoffs, external_frames)
    intraday = _add_legacy_six_columns(japanese.join(aligned_intraday, how="left"))
    after_close = _add_legacy_six_columns(japanese.join(aligned_after, how="left"))
    overseas_features = list(
        dict.fromkeys(
            column
            for factor in external_frames
            for column in (
                factor,
                f"{factor}_ret",
                f"{factor}_ret5",
                f"{factor}_ret20",
                f"{factor}_vol20",
                f"{factor}_ma20_gap",
            )
            if column in intraday.columns
        )
    )
    inference_context = context["prediction_context"]
    inference_frame = intraday if inference_context == "intraday" else after_close
    inference_rows = {
        "intraday": intraday.loc[[base_date], JAPAN_FEATURES + overseas_features],
        "after_close": after_close.loc[[base_date], JAPAN_FEATURES + overseas_features],
    }
    cutoff = pd.Timestamp(context["external_cutoff_timestamp_jst"])
    diagnostics = _diagnostic_for_cutoff(cutoff, external_frames)
    if context.get("warning"):
        warnings.append(str(context["warning"]))
    warnings.append("PCEは実際の公表日と当時利用可能だったデータを安全に再現できないため除外しました。")
    feature_valid = japanese[JAPAN_FEATURES].dropna()
    metadata = {
        "model_version": MODEL_VERSION,
        "raw_requested_start": _date_text(requested_start),
        "raw_data_start": _date_text(base.index[0]),
        "raw_data_end": _date_text(base.index[-1]),
        "raw_data_period": {
            "start": _date_text(base.index[0]),
            "end": _date_text(base.index[-1]),
            "rows": int(len(base)),
        },
        "feature_valid_start": _date_text(feature_valid.index[0]),
        "warmup_period": {
            "start": _date_text(warmup_start),
            "end": _date_text(base.index[WARMUP_TRADING_DAYS - 1]),
            "trading_days": WARMUP_TRADING_DAYS,
            "first_training_date": _date_text(first_training_date),
            "included_in_training_or_evaluation": False,
        },
        "latest_label_date": _date_text(latest_labeled_date),
        "evaluation_target_start": _date_text(evaluation_start_target),
        "evaluation_period": {
            "start": _date_text(evaluation_candidates[0]) if len(evaluation_candidates) else None,
            "end": _date_text(evaluation_candidates[-1]) if len(evaluation_candidates) else None,
            "samples": int(len(evaluation_candidates)),
            "years": EVALUATION_YEARS,
        },
        "learning_years": LEARNING_YEARS,
        "warmup_trading_days": WARMUP_TRADING_DAYS,
        "prediction_horizon": PREDICTION_HORIZON,
        "prediction_context": context,
        "external_alignment_version": EXTERNAL_ALIGNMENT_VERSION,
        "feature_definition_version": FEATURE_DEFINITION_VERSION,
        "external_sources": {
            factor: {
                **source,
                "availability_rule": (
                    "16:15 America/New_York converted to Asia/Tokyo"
                    if source["kind"] == "us_close"
                    else "conservative one-source-session lag; source close timestamp converted to Asia/Tokyo"
                ),
                "calculation_calendar": "source observation calendar before asof alignment",
                "join_rule": "latest available_at <= Japanese prediction cutoff; backward asof only",
            }
            for factor, source in EXTERNAL_SOURCES.items()
            if factor in external_frames
        },
        "latest_external_usage": diagnostics,
        "available_external_factors": list(external_frames),
        "pce_used": False,
        "pce_exclusion_reason": "実際の公表日と当時利用可能だったデータを安全に再現できないため除外",
        "intraday_data_used": False,
        "download_interval": "1d",
        "preparation_seconds": float(time.perf_counter() - started),
    }
    return DualMarketData(
        base_frame=base,
        japanese_frame=japanese,
        intraday_frame=intraday,
        after_close_frame=after_close,
        inference_rows=inference_rows,
        japan_features=list(JAPAN_FEATURES),
        overseas_features=overseas_features,
        warnings=warnings,
        fetched_at=_timestamp_text(_now_jst()),
        metadata=metadata,
    )
