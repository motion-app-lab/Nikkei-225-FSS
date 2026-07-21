from __future__ import annotations

"""日経平均の公開画面向けに、確定済み日足を定型文と図表データへ変換する。"""

from typing import Any

import numpy as np
import pandas as pd

from .common import ServiceError


NIKKEI_PUBLIC_SCHEMA_VERSION = "nikkei_public_ui_v3_context_integrity"
NIKKEI_CHART_REPORT_VERSION = "nikkei_chart_reports_v1"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _pct(current: float, previous: float) -> float | None:
    if not np.isfinite(current) or not np.isfinite(previous) or previous == 0:
        return None
    return (current / previous - 1.0) * 100.0


def _return_at(series: pd.Series, sessions: int) -> float | None:
    if len(series) <= sessions:
        return None
    return _pct(float(series.iloc[-1]), float(series.iloc[-sessions - 1]))


def _format_move(value: float | None, period: str) -> str:
    if value is None:
        return f"{period}の騰落率はデータ不足で算出できません。"
    if abs(value) < 0.05:
        return f"{period}は開始時点とほぼ同じ水準です。"
    return f"{period}は{abs(value):.1f}％{'上昇' if value > 0 else '下落'}しました。"


def _canonical(frame: pd.DataFrame, basis_date: str | pd.Timestamp) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ServiceError("日経平均のチャートデータがありません。")
    result = frame.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index, errors="coerce")).tz_localize(None).normalize()
    result = result[~result.index.isna()].sort_index()
    result = result[~result.index.duplicated(keep="last")]
    result = result.loc[: pd.Timestamp(basis_date).tz_localize(None).normalize()]
    close_column = "price" if "price" in result else "Close" if "Close" in result else None
    if close_column is None:
        raise ServiceError("日経平均の終値データがありません。")
    result["close"] = pd.to_numeric(result[close_column], errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
    if len(result) < 252:
        raise ServiceError("日経平均の中長期チャートに必要な履歴が不足しています。")
    for window in (5, 20, 60, 200):
        result[f"ma{window}"] = result["close"].rolling(window, min_periods=window).mean()
    return result


def _chart_records(view: pd.DataFrame, moving_averages: tuple[int, ...]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for date, row in view.iterrows():
        item: dict[str, Any] = {"date": date.strftime("%Y-%m-%d"), "close": _finite(row["close"])}
        for window in moving_averages:
            item[f"ma{window}"] = _finite(row[f"ma{window}"])
        records.append(item)
    return records


def _top_six_label(six_stage: dict[str, Any]) -> str:
    labels = list(six_stage.get("top_labels") or [])
    return "・".join(str(label) for label in labels) if labels else "最上位区分"


def build_nikkei_chart_payload(
    frame: pd.DataFrame,
    basis_date: str | pd.Timestamp,
    six_stage: dict[str, Any],
) -> dict[str, Any]:
    data = _canonical(frame, basis_date)
    short = data.tail(60).copy()
    long = data.tail(504).copy()
    close_short = short["close"]
    close_long = long["close"]
    current = float(close_long.iloc[-1])

    return_5 = _return_at(close_short, 5)
    return_20 = _return_at(close_short, 20)
    high_60 = float(close_short.max())
    low_60 = float(close_short.min())
    position_60 = 50.0 if high_60 == low_60 else (current - low_60) / (high_60 - low_60) * 100.0
    vol_20_raw = close_short.pct_change().tail(20).std() * np.sqrt(252) * 100.0
    vol_20 = float(vol_20_raw) if np.isfinite(vol_20_raw) else 0.0
    ma5 = _finite(short["ma5"].iloc[-1])
    ma20 = _finite(short["ma20"].iloc[-1])
    top_label = _top_six_label(six_stage)

    short_sentences = [
        f"{_format_move(return_5, '直近5営業日').rstrip('。')}、{_format_move(return_20, '直近20営業日')}"
    ]
    if ma5 is not None and ma20 is not None:
        short_sentences.append(
            f"現在値は5日移動平均を{'上回り' if current >= ma5 else '下回り'}、"
            f"20日移動平均を{'上回っています' if current >= ma20 else '下回っています'}。"
        )
    short_sentences.append(
        f"直近60営業日の値幅内では{position_60:.1f}％の位置にあり、20日年率換算ボラティリティは{vol_20:.1f}％です。"
    )
    short_sentences.append(
        f"5営業日先の6段階出力では{top_label}が最上位ですが、これは直近チャートの説明とは別のモデル出力です。"
    )

    return_2y = _pct(current, float(close_long.iloc[0]))
    return_1y = _return_at(close_long, 252)
    high_2y = float(close_long.max())
    low_2y = float(close_long.min())
    distance_high = _pct(current, high_2y) or 0.0
    distance_low = _pct(current, low_2y) or 0.0
    position_2y = 50.0 if high_2y == low_2y else (current - low_2y) / (high_2y - low_2y) * 100.0
    ma60 = _finite(long["ma60"].iloc[-1])
    ma200 = _finite(long["ma200"].iloc[-1])
    long_vol_raw = close_long.pct_change().tail(252).std() * np.sqrt(252) * 100.0
    long_vol = float(long_vol_raw) if np.isfinite(long_vol_raw) else 0.0
    long_sentences = [
        f"{_format_move(return_2y, '直近2年間').rstrip('。')}、{_format_move(return_1y, '直近1年間')}"
    ]
    long_sentences.append(
        f"現在値は2年高値から{abs(distance_high):.1f}％下、2年安値から{abs(distance_low):.1f}％上で、期間値幅の{position_2y:.1f}％の位置です。"
    )
    if ma60 is not None and ma200 is not None:
        long_sentences.append(
            f"現在値は60日移動平均を{'上回り' if current >= ma60 else '下回り'}、"
            f"200日移動平均を{'上回り' if current >= ma200 else '下回り'}、"
            f"直近1年の年率換算ボラティリティは{long_vol:.1f}％です。"
        )
    long_sentences.append(
        f"中長期の位置関係と、5営業日先のモデル出力で最上位の{top_label}は対象期間が異なるため、別々の情報として表示しています。"
    )

    return {
        "schema_version": NIKKEI_CHART_REPORT_VERSION,
        "chart_60d": _chart_records(short, (5, 20, 60)),
        "chart_2y": _chart_records(long, (60, 200)),
        "short_term_report": {
            "title": "短期動向レポート",
            "body": " ".join(short_sentences),
            "window_start": short.index[0].strftime("%Y-%m-%d"),
            "window_end": short.index[-1].strftime("%Y-%m-%d"),
        },
        "medium_long_term_report": {
            "title": "中長期トレンド分析レポート",
            "body": " ".join(long_sentences),
            "window_start": long.index[0].strftime("%Y-%m-%d"),
            "window_end": long.index[-1].strftime("%Y-%m-%d"),
        },
    }