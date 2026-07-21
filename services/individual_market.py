from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as clock_time, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .common import (
    OUTPUT_DIR,
    PREDICTION_HORIZON,
    ServiceError,
    _download_base_frame,
    calc_rsi,
    extract_yfinance_column,
    normalize_datetime_index,
)
from .nikkei_dual_market import next_japan_market_days, resolve_prediction_context


INDIVIDUAL_MARKET_ALIGNMENT_VERSION = "individual_market_available_at_asof_v3_nikkei_only"
INDIVIDUAL_FEATURE_DEFINITION_VERSION = "individual_normalized_market_features_v4_nikkei_only"
TRAINING_YEARS = 8
EVALUATION_YEARS = 2
WARMUP_TRADING_DAYS = 300
MIN_REQUIRED_OVERSEAS_FACTORS = 3
MIN_FACTOR_COVERAGE = 0.75
EXTERNAL_DOWNLOAD_TIMEOUT_SECONDS = 20
EXTERNAL_DOWNLOAD_MAX_ATTEMPTS = 2
EXTERNAL_DOWNLOAD_THREADS = False
FIXED_LOGISTIC_FEATURE_GROUP = "F_logistic_fast"
MARKET_DATA_CACHE_VERSION = "individual_market_data_cache_v1"

JST = ZoneInfo("Asia/Tokyo")
NEW_YORK = ZoneInfo("America/New_York")
UTC = timezone.utc

JAPAN_SOURCES: dict[str, dict[str, str]] = {

    "nikkei": {"label": "日経平均株価", "ticker": "^N225"},
}

# NVIDIAなどの特定企業は全銘柄共通の候補へ入れない。
EXTERNAL_SOURCES: dict[str, dict[str, Any]] = {
    "spx": {"label": "S&P 500", "ticker": "^GSPC", "group": "us_equity", "kind": "us_close", "safety_lag": 0},
    "nasdaq": {"label": "NASDAQ", "ticker": "^IXIC", "group": "us_equity", "kind": "us_close", "safety_lag": 0},
    "dow": {"label": "NYダウ", "ticker": "^DJI", "group": "us_equity", "kind": "us_close", "safety_lag": 0},
    "sox": {"label": "SOX", "ticker": "^SOX", "group": "us_equity", "kind": "us_close", "safety_lag": 0},
    "vix": {"label": "VIX", "ticker": "^VIX", "group": "risk_fx", "kind": "us_close", "safety_lag": 0},
    "usdjpy": {"label": "ドル円", "ticker": "JPY=X", "group": "risk_fx", "kind": "uncertain_daily", "safety_lag": 1},
    "oil": {"label": "原油", "ticker": "CL=F", "group": "optional", "kind": "uncertain_daily", "safety_lag": 1},
    "gold": {"label": "金", "ticker": "GC=F", "group": "optional", "kind": "uncertain_daily", "safety_lag": 1},
    "btc": {"label": "Bitcoin", "ticker": "BTC-USD", "group": "optional", "kind": "utc_daily", "safety_lag": 1},
}

REQUIRED_OVERSEAS_CANDIDATES = ("spx", "nasdaq", "vix", "usdjpy")


@dataclass
class IndividualMarketData:
    frame: pd.DataFrame
    feature_groups: dict[str, list[str]]
    group_valid_starts: dict[str, pd.Timestamp]
    basis_date: pd.Timestamp
    prediction_context: str
    prediction_mode: str
    prediction_cutoff_jst: pd.Timestamp
    first_training_date: pd.Timestamp
    warmup_start: pd.Timestamp
    warnings: list[str]
    fetched_at: str
    metadata: dict[str, Any]
    benchmark_series: pd.Series | None = None


def _date_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _timestamp_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def external_available_timestamp(source_date: Any, kind: str) -> pd.Timestamp:
    """ソース市場の確定時刻をJSTへ変換する。米国夏時間はzoneinfoへ委ねる。"""
    source_day = pd.Timestamp(source_date).date()
    if kind == "us_close":
        local = datetime.combine(source_day, clock_time(16, 15), tzinfo=NEW_YORK)
        return pd.Timestamp(local.astimezone(JST))
    if kind == "uncertain_daily":
        local = datetime.combine(source_day, clock_time(17, 15), tzinfo=NEW_YORK)
        return pd.Timestamp(local.astimezone(JST))
    utc_close = datetime.combine(source_day + timedelta(days=1), clock_time(0, 15), tzinfo=UTC)
    return pd.Timestamp(utc_close.astimezone(JST))


def historical_prediction_cutoffs(index: pd.DatetimeIndex, context: str) -> pd.Series:
    cutoffs: list[pd.Timestamp] = []
    for position, source_date in enumerate(index):
        if context == "intraday":
            prediction_date = index[position + 1] if position + 1 < len(index) else next_japan_market_days(source_date, 1)[0]
            cutoff_time = clock_time(9, 0)
        else:
            prediction_date = source_date
            cutoff_time = clock_time(15, 40)
        cutoffs.append(pd.Timestamp.combine(pd.Timestamp(prediction_date).date(), cutoff_time).tz_localize(JST))
    return pd.Series(cutoffs, index=index, dtype="object")


def build_external_feature_frame(factor: str, series: pd.Series) -> pd.DataFrame:
    """外部系列は元カレンダー上で正規化特徴量を完成させてから結合する。"""
    source = EXTERNAL_SOURCES[factor]
    values = pd.to_numeric(series, errors="coerce").dropna().sort_index().astype(float)
    frame = pd.DataFrame(index=values.index)
    daily_return = values.pct_change(fill_method=None) * 100.0
    for window in (1, 5, 20, 60):
        frame[f"{factor}_ret{window}"] = values.pct_change(window, fill_method=None) * 100.0
    for window in (5, 20, 60):
        frame[f"{factor}_vol{window}"] = daily_return.rolling(window, min_periods=window).std()
    for window in (20, 60):
        average = values.rolling(window, min_periods=window).mean()
        frame[f"{factor}_ma{window}_gap"] = (values / average - 1.0) * 100.0
    if factor == "vix":
        # VIXは価格指数ではなく市場不安の水準指標なので、水準自体も候補にする。
        frame["vix_level"] = values
    frame["source_date"] = pd.to_datetime(frame.index)
    frame["available_at_jst"] = [external_available_timestamp(value, str(source["kind"])) for value in frame.index]
    safety_lag = int(source["safety_lag"])
    feature_columns = [column for column in frame if column not in {"source_date", "available_at_jst"}]
    if safety_lag:
        # 時刻を断定しにくい系列だけを、各ソース自身の観測日で保守的に遅らせる。
        frame[feature_columns] = frame[feature_columns].shift(safety_lag)
        frame["source_date"] = frame["source_date"].shift(safety_lag)
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["source_date"]).sort_values("available_at_jst")


def asof_align_external(
    prediction_cutoffs: pd.Series,
    external_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    cutoffs = pd.DataFrame(
        {
            "base_date": prediction_cutoffs.index,
            "prediction_cutoff_utc": pd.to_datetime(
                list(prediction_cutoffs.values), utc=True, errors="coerce"
            ),
        }
    ).dropna(subset=["prediction_cutoff_utc"])
    cutoffs = cutoffs.sort_values("prediction_cutoff_utc").drop_duplicates("prediction_cutoff_utc", keep="last")
    aligned = pd.DataFrame(index=pd.DatetimeIndex(prediction_cutoffs.index))
    diagnostics: dict[str, pd.DataFrame] = {}
    for factor, source_frame in external_frames.items():
        feature_columns = [column for column in source_frame if column not in {"source_date", "available_at_jst"}]
        right = source_frame[["available_at_jst", "source_date", *feature_columns]].copy()
        right["available_at_utc"] = pd.to_datetime(right["available_at_jst"], utc=True, errors="coerce")
        right = (
            right.drop(columns=["available_at_jst"])
            .dropna(subset=["available_at_utc"])
            .sort_values("available_at_utc")
            .drop_duplicates("available_at_utc", keep="last")
        )
        merged = pd.merge_asof(
            cutoffs,
            right,
            left_on="prediction_cutoff_utc",
            right_on="available_at_utc",
            direction="backward",
            allow_exact_matches=True,
        ).set_index("base_date")
        merged["prediction_cutoff_jst"] = merged["prediction_cutoff_utc"].dt.tz_convert(JST)
        merged["available_at_jst"] = merged["available_at_utc"].dt.tz_convert(JST)
        aligned.loc[merged.index, feature_columns] = merged[feature_columns]
        diagnostics[factor] = merged[["prediction_cutoff_jst", "source_date", "available_at_jst"]].copy()
    return aligned.sort_index(), diagnostics


def build_stock_features(base: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = base.copy()
    price = pd.to_numeric(frame["price"], errors="coerce").astype(float)
    volume = pd.to_numeric(frame.get("volume", 0.0), errors="coerce").astype(float)
    daily_return = price.pct_change(fill_method=None) * 100.0
    features: list[str] = []
    for window in (1, 5, 10, 20, 60):
        column = f"stock_ret{window}"
        frame[column] = price.pct_change(window, fill_method=None) * 100.0
        features.append(column)
    for window in (5, 20, 60):
        vol_column = f"stock_vol{window}"
        absolute_column = f"stock_absret{window}"
        frame[vol_column] = daily_return.rolling(window, min_periods=window).std()
        frame[absolute_column] = daily_return.abs().rolling(window, min_periods=window).mean()
        features.extend([vol_column, absolute_column])
    averages: dict[int, pd.Series] = {}
    for window in (5, 20, 60):
        averages[window] = price.rolling(window, min_periods=window).mean()
        column = f"stock_ma{window}_gap"
        frame[column] = (price / averages[window] - 1.0) * 100.0
        features.append(column)
    for window in (20, 60):
        slope_column = f"stock_ma{window}_slope5"
        frame[slope_column] = averages[window].pct_change(5, fill_method=None) * 100.0
        high = price.rolling(window, min_periods=window).max()
        low = price.rolling(window, min_periods=window).min()
        width = (high - low).replace(0, np.nan)
        values = {
            f"stock_range_position{window}": (price - low) / width,
            f"stock_distance_high{window}": (price / high - 1.0) * 100.0,
            f"stock_distance_low{window}": (price / low - 1.0) * 100.0,
        }
        for column, series in values.items():
            frame[column] = series
            features.append(column)
    median5 = volume.rolling(5, min_periods=5).median()
    median20 = volume.rolling(20, min_periods=20).median()
    frame["stock_volume_median_ratio5"] = volume / median20.replace(0, np.nan)
    frame["stock_volume_median_ratio20"] = median5 / median20.replace(0, np.nan)
    frame["stock_volume_change"] = volume.pct_change(fill_method=None) * 100.0
    frame["stock_rsi14"] = calc_rsi(price, 14)
    previous_close = price.shift(1)
    true_range = pd.concat(
        [
            pd.to_numeric(frame["high"], errors="coerce") - pd.to_numeric(frame["low"], errors="coerce"),
            (pd.to_numeric(frame["high"], errors="coerce") - previous_close).abs(),
            (pd.to_numeric(frame["low"], errors="coerce") - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["stock_atr14_ratio"] = true_range.rolling(14, min_periods=14).mean() / price * 100.0
    frame["stock_opening_gap"] = (pd.to_numeric(frame["open"], errors="coerce") / previous_close - 1.0) * 100.0
    features.extend(
        [
            "stock_volume_median_ratio5",
            "stock_volume_median_ratio20",
            "stock_volume_change",
            "stock_rsi14",
            "stock_atr14_ratio",
            "stock_opening_gap",
        ]
    )
    return frame.replace([np.inf, -np.inf], np.nan), features


def build_japan_market_features(factor: str, series: pd.Series) -> pd.DataFrame:
    values = pd.to_numeric(series, errors="coerce").dropna().sort_index().astype(float)
    frame = pd.DataFrame(index=values.index)
    daily_return = values.pct_change(fill_method=None) * 100.0
    for window in (1, 5, 20, 60):
        frame[f"{factor}_ret{window}"] = values.pct_change(window, fill_method=None) * 100.0
    for window in (5, 20, 60):
        frame[f"{factor}_vol{window}"] = daily_return.rolling(window, min_periods=window).std()
    for window in (20, 60):
        average = values.rolling(window, min_periods=window).mean()
        frame[f"{factor}_ma{window}_gap"] = (values / average - 1.0) * 100.0
    return frame.replace([np.inf, -np.inf], np.nan)


def add_relative_market_features(frame: pd.DataFrame, available_japan: list[str]) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    columns: list[str] = []
    for factor in available_japan:
        for window in (5, 20, 60):
            column = f"stock_vs_{factor}_ret{window}"
            result[column] = result[f"stock_ret{window}"] - result[f"{factor}_ret{window}"]
            columns.append(column)
        correlation = f"stock_{factor}_corr60"
        beta = f"stock_{factor}_beta60"
        covariance = result["stock_ret1"].rolling(60, min_periods=60).cov(result[f"{factor}_ret1"])
        variance = result[f"{factor}_ret1"].rolling(60, min_periods=60).var().replace(0, np.nan)
        result[correlation] = result["stock_ret1"].rolling(60, min_periods=60).corr(result[f"{factor}_ret1"])
        result[beta] = covariance / variance
        columns.extend([correlation, beta])
    return result.replace([np.inf, -np.inf], np.nan), columns


def add_prediction_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """完全同値を上下どちらにも割り当てず、両モデルの対象外にする。"""
    result = frame.copy()
    future_price = result["price"].shift(-PREDICTION_HORIZON)
    direction = pd.Series(np.nan, index=result.index, dtype=float)
    direction.loc[future_price.gt(result["price"])] = 1.0
    direction.loc[future_price.lt(result["price"])] = 0.0
    equal_mask = future_price.notna() & future_price.eq(result["price"])
    change = (future_price / result["price"] - 1.0) * 100.0
    result["target_direction"] = direction
    result["target_change"] = change.where(future_price.notna()).mask(equal_mask)
    result["target_equal_close"] = equal_mask
    return result


def _download_series_map(
    sources: dict[str, dict[str, Any]],
    start: datetime,
    end: datetime,
) -> tuple[dict[str, pd.Series], list[str]]:
    warnings: list[str] = []
    raw = pd.DataFrame()
    try:
        import yfinance as yf

        for _attempt in range(EXTERNAL_DOWNLOAD_MAX_ATTEMPTS):
            try:
                raw = yf.download(
                    [str(source["ticker"]) for source in sources.values()],
                    start=start,
                    end=end + timedelta(days=1),
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    # yfinance内部の無制限な銘柄別スレッド待ちを避ける。
                    threads=EXTERNAL_DOWNLOAD_THREADS,
                    group_by="column",
                    timeout=EXTERNAL_DOWNLOAD_TIMEOUT_SECONDS,
                )
            except Exception:
                raw = pd.DataFrame()
            if raw is not None and not raw.empty:
                break
    except Exception:
        raw = pd.DataFrame()
    result: dict[str, pd.Series] = {}
    for factor, source in sources.items():
        series = extract_yfinance_column(raw, "Close", str(source["ticker"]))
        if series.empty:
            warnings.append(f"{source['label']}を取得できませんでした。")
            continue
        series.index = normalize_datetime_index(series.index)
        series = series.groupby(level=0).last().sort_index().replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) < 180:
            warnings.append(f"{source['label']}は履歴が不足したため除外しました。")
            continue
        result[factor] = series.astype(float)
    return result, warnings


def _series_cache_path(scope: str, sources: dict[str, dict[str, Any]], end: datetime) -> Path:
    source_signature = "|".join(f"{name}:{source['ticker']}" for name, source in sorted(sources.items()))
    digest = hashlib.sha256(source_signature.encode("utf-8")).hexdigest()[:12]
    return OUTPUT_DIR / f"individual_market_{scope}_{pd.Timestamp(end).strftime('%Y%m%d')}_{digest}.json"


def _read_series_cache(path: Path, sources: dict[str, dict[str, Any]], requested_start: datetime) -> tuple[dict[str, pd.Series], list[str]] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {name: source["ticker"] for name, source in sources.items()}
        if payload.get("schema_version") != MARKET_DATA_CACHE_VERSION or payload.get("sources") != expected or pd.Timestamp(payload.get("requested_start")) > pd.Timestamp(requested_start):
            return None
        result: dict[str, pd.Series] = {}
        for factor, rows in payload.get("series", {}).items():
            values = pd.Series([float(row[1]) for row in rows], index=pd.to_datetime([row[0] for row in rows]), dtype=float)
            values.index = normalize_datetime_index(values.index)
            result[factor] = values.groupby(level=0).last().sort_index()
        return result, [str(value) for value in payload.get("warnings", [])]
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _write_series_cache(path: Path, sources: dict[str, dict[str, Any]], requested_start: datetime, end: datetime, series_map: dict[str, pd.Series], source_warnings: list[str]) -> None:
    payload = {
        "schema_version": MARKET_DATA_CACHE_VERSION,
        "sources": {name: source["ticker"] for name, source in sources.items()},
        "requested_start": pd.Timestamp(requested_start).isoformat(),
        "requested_end": pd.Timestamp(end).isoformat(),
        "series": {factor: [[pd.Timestamp(index).isoformat(), float(value)] for index, value in series.items()] for factor, series in series_map.items()},
        "warnings": list(source_warnings),
    }
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _load_or_download_series_map(scope: str, sources: dict[str, dict[str, Any]], start: datetime, end: datetime) -> tuple[dict[str, pd.Series], list[str], bool, float, float]:
    if getattr(_download_series_map, "__module__", __name__) != __name__:
        started = time.perf_counter()
        values, source_warnings = _download_series_map(sources, start, end)
        return values, source_warnings, False, 0.0, float(time.perf_counter() - started)
    path = _series_cache_path(scope, sources, end)
    cache_started = time.perf_counter()
    cached = _read_series_cache(path, sources, start) if path.exists() else None
    cache_seconds = float(time.perf_counter() - cache_started)
    if cached is not None:
        return cached[0], cached[1], True, cache_seconds, 0.0
    external_started = time.perf_counter()
    series_map, source_warnings = _download_series_map(sources, start, end)
    external_seconds = float(time.perf_counter() - external_started)
    _write_series_cache(path, sources, start, end, series_map, source_warnings)
    return series_map, source_warnings, False, cache_seconds, external_seconds


def _factor_has_coverage(frame: pd.DataFrame, columns: list[str], index: pd.DatetimeIndex) -> bool:
    if not columns or len(index) == 0:
        return False
    view = frame.reindex(index=index, columns=columns)
    return bool(float(view.notna().mean().min()) >= MIN_FACTOR_COVERAGE)


def _latest_external_diagnostics(
    diagnostics: dict[str, pd.DataFrame],
    basis_date: pd.Timestamp,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for factor, diagnostic in diagnostics.items():
        source = EXTERNAL_SOURCES[factor]
        row = diagnostic.loc[basis_date] if basis_date in diagnostic.index else None
        result[factor] = {
            "data_name": source["label"],
            "ticker": source["ticker"],
            "source_date": _date_text(row["source_date"]) if row is not None else None,
            "available_at_jst": _timestamp_text(row["available_at_jst"]) if row is not None else None,
            "prediction_cutoff_jst": _timestamp_text(row["prediction_cutoff_jst"]) if row is not None else None,
            "join_rule": "latest available_at <= prediction cutoff; backward asof only",
            "safety_lag_source_sessions": int(source["safety_lag"]),
        }
    return result


def prepare_individual_market_data(
    ticker: str,
    now: datetime | None = None,
    timing: dict[str, float | bool] | None = None,
) -> IndividualMarketData:
    """対象銘柄・日本市場・確定済み海外市場を一つの時系列特徴量表へ統合する。"""
    timing_sink = timing if timing is not None else {}
    current = now or datetime.now(JST)
    current_naive = current.replace(tzinfo=None) if current.tzinfo else current
    requested_start = pd.Timestamp(current_naive) - pd.DateOffset(years=12) - pd.Timedelta(days=180)
    stock_fetch_started = time.perf_counter()
    base = _download_base_frame(ticker, requested_start.to_pydatetime(), current_naive)
    timing_sink["stock_external_fetch_seconds"] = float(time.perf_counter() - stock_fetch_started)
    context = resolve_prediction_context(base, current if current.tzinfo else current.replace(tzinfo=JST))
    basis_date = pd.Timestamp(context["nikkei_base_date"])
    base = base.loc[:basis_date].copy()
    if len(base) < WARMUP_TRADING_DAYS + 500:
        raise ServiceError(
            "この銘柄は、正式評価と予測に必要な株価履歴が不足しています。",
            "最大8年の学習、直近2年の評価、300取引日の準備期間を確保できる銘柄を指定してください。",
        )

    latest_labeled_position = len(base) - PREDICTION_HORIZON - 1
    if latest_labeled_position < 0:
        raise ServiceError("この銘柄は、正式評価と予測に必要な株価履歴が不足しています。")
    latest_labeled_date = base.index[latest_labeled_position]
    evaluation_target = latest_labeled_date - pd.DateOffset(years=EVALUATION_YEARS)
    evaluation_start_position = int(base.index.searchsorted(evaluation_target, side="left"))
    if evaluation_start_position >= len(base):
        raise ServiceError("直近2年間の正式評価期間を確保できませんでした。")
    evaluation_start = base.index[evaluation_start_position]
    earliest_training_target = evaluation_start - pd.DateOffset(years=TRAINING_YEARS)
    first_training_position = int(base.index.searchsorted(earliest_training_target, side="left"))
    if first_training_position < WARMUP_TRADING_DAYS:
        raise ServiceError(
            "この銘柄は、正式評価と予測に必要な株価履歴が不足しています。",
            "最初の評価期間より前の最大8年と、特徴量計算用の300取引日を確保できませんでした。",
        )
    warmup_start_position = first_training_position - WARMUP_TRADING_DAYS
    base = base.iloc[warmup_start_position:].copy()
    warmup_start = base.index[0]
    first_training_date = base.index[WARMUP_TRADING_DAYS]

    download_start = warmup_start.to_pydatetime() - timedelta(days=120)
    japan_raw, japan_warnings, japan_hit, japan_cache_seconds, japan_external_seconds = _load_or_download_series_map(
        "japan", JAPAN_SOURCES, download_start, current_naive
    )
    external_raw, external_warnings, external_hit, external_cache_seconds, external_fetch_seconds = _load_or_download_series_map(
        "external", EXTERNAL_SOURCES, download_start, current_naive
    )
    timing_sink["local_cache_read_seconds"] = float(japan_cache_seconds + external_cache_seconds)
    timing_sink["common_market_external_fetch_seconds"] = float(japan_external_seconds + external_fetch_seconds)
    timing_sink["common_market_cache_hit"] = bool(japan_hit and external_hit)
    warnings = [*japan_warnings, *external_warnings]

    feature_started = time.perf_counter()
    stock_frame, stock_features = build_stock_features(base)
    comparison_index = stock_frame.loc[first_training_date:basis_date].index
    japan_features_by_factor: dict[str, list[str]] = {}
    available_japan: list[str] = []
    for factor, series in japan_raw.items():
        features = build_japan_market_features(factor, series)
        aligned = features.reindex(stock_frame.index, method="ffill")
        columns = list(features.columns)
        if _factor_has_coverage(aligned, columns, comparison_index):
            stock_frame = stock_frame.join(aligned, how="left")
            japan_features_by_factor[factor] = columns
            available_japan.append(factor)
    if "nikkei" not in available_japan:
        raise ServiceError(
            "予測に必要な市場データを十分に取得できなかったため、今回は予測できません。",
            "日経平均株価を取得できる状態で再度実行してください。",
        )
    stock_frame, relative_features = add_relative_market_features(stock_frame, available_japan)

    external_frames = {factor: build_external_feature_frame(factor, series) for factor, series in external_raw.items()}
    join_started = time.perf_counter()
    cutoffs = historical_prediction_cutoffs(stock_frame.index, str(context["prediction_context"]))
    aligned_external, diagnostics = asof_align_external(cutoffs, external_frames)
    timing_sink["data_join_seconds"] = float(time.perf_counter() - join_started)
    external_features_by_factor: dict[str, list[str]] = {}
    available_external: list[str] = []
    for factor, source_frame in external_frames.items():
        columns = [column for column in source_frame if column not in {"source_date", "available_at_jst"}]
        if _factor_has_coverage(aligned_external, columns, comparison_index):
            external_features_by_factor[factor] = columns
            available_external.append(factor)
    if available_external:
        selected_external_columns = list(
            dict.fromkeys(column for factor in available_external for column in external_features_by_factor[factor])
        )
        stock_frame = stock_frame.join(aligned_external[selected_external_columns], how="left")
    available_required = [factor for factor in REQUIRED_OVERSEAS_CANDIDATES if factor in available_external]
    if len(available_required) < MIN_REQUIRED_OVERSEAS_FACTORS:
        raise ServiceError(
            "予測に必要な市場データを十分に取得できなかったため、今回は予測できません。",
            "S&P 500、NASDAQ、VIX、ドル円のうち3系列以上を取得できる状態で再度実行してください。",
        )

    japan_columns = list(
        dict.fromkeys(column for factor in available_japan for column in japan_features_by_factor[factor])
    ) + relative_features
    us_columns = list(
        dict.fromkeys(
            column
            for factor in available_external
            if EXTERNAL_SOURCES[factor]["group"] == "us_equity"
            for column in external_features_by_factor[factor]
        )
    )
    risk_columns = list(
        dict.fromkeys(
            column
            for factor in available_external
            if EXTERNAL_SOURCES[factor]["group"] == "risk_fx"
            for column in external_features_by_factor[factor]
        )
    )
    optional_columns = list(
        dict.fromkeys(
            column
            for factor in available_external
            if EXTERNAL_SOURCES[factor]["group"] == "optional"
            for column in external_features_by_factor[factor]
        )
    )
    feature_groups: dict[str, list[str]] = {"A_stock": list(stock_features)}
    feature_groups["B_stock_japan"] = list(dict.fromkeys([*stock_features, *japan_columns]))
    fixed_external_columns = list(
        dict.fromkeys(
            column
            for factor in REQUIRED_OVERSEAS_CANDIDATES
            if factor in available_external
            for column in external_features_by_factor[factor]
        )
    )
    feature_groups[FIXED_LOGISTIC_FEATURE_GROUP] = list(
        dict.fromkeys([*feature_groups["B_stock_japan"], *fixed_external_columns])
    )
    if us_columns:
        feature_groups["C_plus_us_equity"] = list(dict.fromkeys([*feature_groups["B_stock_japan"], *us_columns]))
    if us_columns and risk_columns:
        feature_groups["D_plus_risk_fx"] = list(
            dict.fromkeys([*feature_groups["C_plus_us_equity"], *risk_columns])
        )
    if "D_plus_risk_fx" in feature_groups and optional_columns:
        feature_groups["E_plus_optional"] = list(
            dict.fromkeys([*feature_groups["D_plus_risk_fx"], *optional_columns])
        )

    stock_frame = add_prediction_targets(stock_frame)
    equal_mask = stock_frame["target_equal_close"]
    stock_frame = stock_frame.replace([np.inf, -np.inf], np.nan)

    group_valid_starts: dict[str, pd.Timestamp] = {}
    for group_name, columns in feature_groups.items():
        starts = [stock_frame[column].first_valid_index() for column in columns]
        valid_starts = [pd.Timestamp(value) for value in starts if value is not None]
        group_valid_starts[group_name] = max(valid_starts) if valid_starts else first_training_date

    timing_sink["feature_build_seconds"] = float(time.perf_counter() - feature_started)
    cutoff = pd.Timestamp(cutoffs.loc[basis_date])
    used_japan_labels = [JAPAN_SOURCES[factor]["label"] for factor in available_japan]
    used_external_labels = [EXTERNAL_SOURCES[factor]["label"] for factor in available_external]
    missing_external = [factor for factor in EXTERNAL_SOURCES if factor not in available_external]
    metadata = {
        "market_alignment_version": INDIVIDUAL_MARKET_ALIGNMENT_VERSION,
        "feature_definition_version": INDIVIDUAL_FEATURE_DEFINITION_VERSION,
        "raw_requested_start": _date_text(requested_start),
        "raw_data_start": _date_text(base.index[0]),
        "raw_data_end": _date_text(base.index[-1]),
        "first_training_date": _date_text(first_training_date),
        "warmup_period": {
            "start": _date_text(warmup_start),
            "end": _date_text(base.index[WARMUP_TRADING_DAYS - 1]),
            "trading_days": WARMUP_TRADING_DAYS,
            "included_in_training_or_evaluation": False,
        },
        "evaluation_target_start": _date_text(evaluation_start),
        "latest_label_date": _date_text(latest_labeled_date),
        "prediction_context": context["prediction_context"],
        "prediction_mode": context["prediction_mode"],
        "prediction_cutoff_jst": _timestamp_text(cutoff),
        "basis_date": _date_text(basis_date),
        "used_japan_factors": used_japan_labels,
        "used_external_factors": used_external_labels,
        "missing_japan_factors": [factor for factor in JAPAN_SOURCES if factor not in available_japan],
        "missing_external_factors": missing_external,
        "required_overseas_candidates": list(REQUIRED_OVERSEAS_CANDIDATES),
        "required_overseas_minimum": MIN_REQUIRED_OVERSEAS_FACTORS,
        "latest_external_usage": _latest_external_diagnostics(diagnostics, basis_date),
        "external_sources": {
            factor: {
                **EXTERNAL_SOURCES[factor],
                "availability_rule": (
                    "16:15 America/New_York converted to Asia/Tokyo"
                    if EXTERNAL_SOURCES[factor]["kind"] == "us_close"
                    else "conservative one-source-session lag on source calendar"
                ),
                "calculation_calendar": "source observation calendar before asof alignment",
                "join_rule": "backward asof only; no bfill, nearest, or future value",
            }
            for factor in available_external
        },
        "global_external_shift_sessions": 0,
        "pce_used": False,
        "pce_exclusion_reason": "今回の個別銘柄モデルでは使用しない",
        "equal_close_rows": int(equal_mask.sum()),
        "available_feature_groups": {name: len(columns) for name, columns in feature_groups.items()},
        "fixed_logistic_feature_group": FIXED_LOGISTIC_FEATURE_GROUP,
        "fixed_logistic_factors": ["対象銘柄", *used_japan_labels, *[EXTERNAL_SOURCES[factor]["label"] for factor in REQUIRED_OVERSEAS_CANDIDATES if factor in available_external]],
        "timing": dict(timing_sink),
    }
    if context.get("warning"):
        warnings.append(str(context["warning"]))
    return IndividualMarketData(
        frame=stock_frame,
        feature_groups=feature_groups,
        group_valid_starts=group_valid_starts,
        basis_date=basis_date,
        prediction_context=str(context["prediction_context"]),
        prediction_mode=str(context["prediction_mode"]),
        prediction_cutoff_jst=cutoff,
        first_training_date=first_training_date,
        warmup_start=warmup_start,
        warnings=warnings,
        fetched_at=datetime.now(JST).isoformat(timespec="seconds"),
        metadata=metadata,
        benchmark_series=japan_raw.get("nikkei"),
    )
