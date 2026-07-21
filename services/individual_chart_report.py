from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import OUTPUT_DIR, ServiceError


CHART_ANALYSIS_VERSION = "individual_chart_analysis_v3_nikkei_benchmark"
SIX_STAGE_REPORT_VERSION = "individual_six_stage_report_v2"
LONG_CHART_ANALYSIS_VERSION = "individual_long_chart_analysis_v2_nikkei_benchmark"
CHART_ANALYSIS_DISCLAIMER = (
    "このレポートは、過去の株価・出来高などを機械的に整理し、チャートの形状を説明する参考情報です。"
    "将来の株価、売買の時期、保有・売却の判断、利益または損失を予測・推奨するものではありません。"
    "表示された価格水準や移動平均線を、売買判断の基準として示すものでもありません。"
    "また、利用者の資産状況、投資目的、経験、許容できる損失などは考慮していません。"
    "予測モデルの判定理由そのものを示すものでもありません。"
)
CHART_ANALYSIS_FOOTER = "※過去の短期チャートを自動整理した説明です。将来の値動きや売買判断を示すものではありません。"
LONG_CHART_ANALYSIS_FOOTER = "※過去の中長期チャートを自動整理した説明です。将来の値動きや売買判断を示すものではありません。"
SIX_STAGE_REPORT_FOOTER = (
    "表示割合は、本システムが5営業日先の値動きを6段階に分類した際のモデル出力割合です。"
    "実際にその値動きが発生する確率や、過去の予測精度そのものを示す数値ではありません。"
    "また、将来の株価を保証したり、購入・売却などの判断を示したりするものではありません。"
)

SIX_STAGE_DISPLAY_ORDER = (
    ("急騰", "上昇 Lv.3"),
    ("上昇", "上昇 Lv.2"),
    ("やや上昇", "上昇 Lv.1"),
    ("やや下落", "下落 Lv.1"),
    ("下落", "下落 Lv.2"),
    ("急落", "下落 Lv.3"),
)
SIX_STAGE_LEVEL_NOTE = "Lv.はモデル内の値動き幅の区分を示すもので、予測の確実性や売買の推奨度を示すものではありません。"

# 冒頭・末尾の注意文には適用せず、自動生成した分析本文だけを監査する。
FORBIDDEN_REPORT_TERMS = (
    "買い", "売り", "買い場", "売り場", "押し目", "戻り売り", "狙い目", "チャンス",
    "エントリー", "仕込み", "利確", "損切り", "保有継続", "売却", "購入", "反発が期待",
    "上昇が期待", "下落に注意", "今後上昇", "今後下落", "上がりそう", "下がりそう",
    "ここを超えると", "ここを割ると", "支持線", "抵抗線", "サポートライン",
    "レジスタンスライン", "上値目標", "下値目安", "注目価格", "目標株価", "強気",
    "弱気", "割安", "割高", "推奨", "有望", "危険", "警戒", "判断のポイント",
    "今後の注目点", "今後の確認項目", "観察対象",
)


def build_six_stage_trend_report(probabilities: list[dict[str, Any]]) -> dict[str, Any]:
    """6クラスモデル出力を固定順・小数1桁・合計100.0％の公開表示へ整える。"""
    by_raw_label = {str(item.get("raw_label", item.get("label", ""))): item for item in probabilities}
    raw_values = np.asarray(
        [max(0.0, float(by_raw_label.get(raw_label, {}).get("probability", 0.0))) for raw_label, _ in SIX_STAGE_DISPLAY_ORDER],
        dtype=float,
    )
    if not np.isfinite(raw_values).all() or float(raw_values.sum()) <= 0:
        raw_values = np.ones(len(SIX_STAGE_DISPLAY_ORDER), dtype=float)
    exact_tenths = raw_values / float(raw_values.sum()) * 1000.0
    rounded_tenths = np.floor(exact_tenths).astype(int)
    remainder = int(1000 - rounded_tenths.sum())
    fractions = exact_tenths - rounded_tenths
    for index in sorted(range(len(fractions)), key=lambda value: (-fractions[value], value))[:remainder]:
        rounded_tenths[index] += 1

    items = [
        {
            "raw_label": raw_label,
            "label": display_label,
            "percentage": float(rounded_tenths[index] / 10.0),
        }
        for index, (raw_label, display_label) in enumerate(SIX_STAGE_DISPLAY_ORDER)
    ]
    ranked = sorted(range(len(items)), key=lambda value: (-items[value]["percentage"], value))
    top_value = items[ranked[0]]["percentage"]
    top_indices = [index for index in ranked if items[index]["percentage"] == top_value]
    if len(top_indices) > 1:
        top_text = "と".join(f"{items[index]['label']} {top_value:.1f}％" for index in top_indices)
        intro = (
            f"本システムが導き出した5営業日先のトレンド予測結果では、{top_text}が同率最上位として"
            "6区分の中で最も大きい値を示しました。"
        )
        second_index = ranked[len(top_indices)] if len(top_indices) < len(ranked) else top_indices[-1]
    else:
        top_index = top_indices[0]
        intro = (
            f"本システムが導き出した5営業日先のトレンド予測結果では、"
            f"{items[top_index]['label']} {top_value:.1f}％が6区分の中で最も大きい値を示しました。"
        )
        second_index = ranked[1]
    second_value = items[second_index]["percentage"]
    gap = float(top_value - second_value)
    comparison = f"次に大きかった{items[second_index]['label']} {second_value:.1f}％との差は{gap:.1f}ポイントでした。"
    if gap < 3.0:
        gap_description = "上位2区分の値は非常に近い状態です。"
    elif gap < 10.0:
        gap_description = "上位2区分の値は比較的近い状態です。"
    elif gap < 20.0:
        gap_description = "6区分の中では、最上位区分がやや大きい状態です。"
    else:
        gap_description = "6区分の中では、最上位区分が相対的に大きい状態です。"
    if top_value < 25.0:
        distribution = "モデル出力は6区分へ広く分散しています。"
    elif top_value < 40.0:
        distribution = "モデル出力は、一つの区分へ大きく集中していません。"
    elif top_value < 55.0:
        distribution = "最上位区分の値は比較的大きいものの、他の区分にも分散しています。"
    else:
        distribution = "最上位区分の値は大きくなっていますが、将来の値動きを確定するものではありません。"
    return {
        "schema_version": SIX_STAGE_REPORT_VERSION,
        "items": items,
        "display_total_percentage": round(float(sum(item["percentage"] for item in items)), 1),
        "top_labels": [items[index]["label"] for index in top_indices],
        "top_percentage": top_value,
        "tie_for_top": len(top_indices) > 1,
        "second_label": items[second_index]["label"],
        "second_percentage": second_value,
        "gap_points": gap,
        "intro": intro,
        "comparison": comparison,
        "gap_description": gap_description,
        "distribution_description": distribution,
        "level_note": SIX_STAGE_LEVEL_NOTE,
        "footer": SIX_STAGE_REPORT_FOOTER,
    }


@dataclass(frozen=True)
class Drawdown:
    start: pd.Timestamp
    bottom: pd.Timestamp
    percent: float


def _finite(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _pct(current: float, previous: float) -> float | None:
    if not np.isfinite(current) or not np.isfinite(previous) or previous == 0:
        return None
    return (current / previous - 1.0) * 100.0


def _fmt_date(value: pd.Timestamp) -> str:
    date = pd.Timestamp(value)
    return f"{date.year}年{date.month}月{date.day}日"


def _fmt_pct(value: float, absolute: bool = False) -> str:
    numeric = abs(value) if absolute else value
    return f"{numeric:.1f}％"


def _canonical_frame(frame: pd.DataFrame, basis_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ServiceError("チャート分析に必要な株価データがありません。", "別の証券コードを確認してください。")
    result = frame.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index, errors="coerce")).tz_localize(None).normalize()
    result = result[~result.index.isna()].sort_index()
    result = result[~result.index.duplicated(keep="last")]
    if basis_date is not None:
        result = result.loc[:pd.Timestamp(basis_date).tz_localize(None).normalize()]
    close_name = "price" if "price" in result else "Close" if "Close" in result else None
    if close_name is None:
        raise ServiceError("チャート分析に必要な終値がありません。", "株価データを再取得してください。")
    result["close"] = pd.to_numeric(result[close_name], errors="coerce")
    volume_name = "volume" if "volume" in result else "Volume" if "Volume" in result else None
    result["volume_value"] = pd.to_numeric(result[volume_name], errors="coerce") if volume_name else np.nan
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
    if len(result) < 20:
        raise ServiceError("チャート分析に必要な履歴が不足しています。", "20営業日以上の履歴がある銘柄を指定してください。")
    result["ma5"] = result["close"].rolling(5, min_periods=5).mean()
    result["ma20"] = result["close"].rolling(20, min_periods=20).mean()
    result["ma60"] = result["close"].rolling(60, min_periods=min(60, len(result))).mean()
    result["ma200"] = result["close"].rolling(200, min_periods=min(200, len(result))).mean()
    return result


def maximum_drawdown(close: pd.Series) -> Drawdown:
    values = pd.to_numeric(close, errors="coerce").dropna()
    if values.empty:
        raise ValueError("close series is empty")
    running_high = values.cummax()
    drawdowns = values / running_high - 1.0
    bottom = pd.Timestamp(drawdowns.idxmin())
    prefix = values.loc[:bottom]
    start = pd.Timestamp(prefix.idxmax())
    return Drawdown(start=start, bottom=bottom, percent=float(drawdowns.loc[bottom] * 100.0))


def maximum_runup(close: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    values = pd.to_numeric(close, errors="coerce").dropna()
    running_low = values.cummin()
    runups = values / running_low - 1.0
    top = pd.Timestamp(runups.idxmax())
    start = pd.Timestamp(values.loc[:top].idxmin())
    return start, top, float(runups.loc[top] * 100.0)


def _return_at(close: pd.Series, sessions: int) -> float | None:
    if len(close) <= sessions:
        return None
    return _pct(float(close.iloc[-1]), float(close.iloc[-sessions - 1]))


def _position_text(position: float) -> str:
    if position <= 20:
        return "期間内では低い側"
    if position <= 40:
        return "中央より低い側"
    if position < 60:
        return "中央付近"
    if position < 80:
        return "中央より高い側"
    return "期間内では高い側"


def _long_position_level_text(position: float) -> str:
    if position <= 20:
        return "低い水準"
    if position <= 40:
        return "比較的低い水準"
    if position < 60:
        return "中間付近の水準"
    if position < 80:
        return "比較的高い水準"
    return "高い水準"


def _direction_text(value: float | None) -> str:
    if value is None or abs(value) < 0.1:
        return "開始時点とほぼ同じ水準"
    return "開始時点を上回っています" if value > 0 else "開始時点を下回っています"


def _safe_section(key: str, title: str, body: str) -> dict[str, str] | None:
    if not body or any(term in body for term in FORBIDDEN_REPORT_TERMS):
        return None
    if "nan" in body.lower() or "inf" in body.lower() or "none" in body.lower():
        return None
    return {"key": key, "title": title, "body": body}


def report_contains_forbidden_term(report: dict[str, Any]) -> list[str]:
    body = " ".join(str(section.get("body", "")) for section in report.get("sections", []))
    return [term for term in FORBIDDEN_REPORT_TERMS if term in body]


def _benchmark_returns(
    benchmark: pd.Series | pd.DataFrame | None,
    target_index: pd.DatetimeIndex,
) -> tuple[float | None, float | None]:
    if benchmark is None:
        return None, None
    if isinstance(benchmark, pd.DataFrame):
        column = "price" if "price" in benchmark else "Close" if "Close" in benchmark else None
        if column is None:
            return None, None
        series = benchmark[column]
    else:
        series = benchmark
    series = pd.to_numeric(series, errors="coerce")
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index, errors="coerce")).tz_localize(None).normalize()
    series = series[~series.index.isna()].sort_index().groupby(level=0).last()
    # 後方補完だけを使い、対象日より未来の指数値は使用しない。
    aligned = series.reindex(target_index, method="ffill")
    return _return_at(aligned, 20), _return_at(aligned, 60)


def build_chart_analysis(
    frame: pd.DataFrame,
    basis_date: str | pd.Timestamp | None = None,
    benchmark: pd.Series | pd.DataFrame | None = None,
    benchmark_name: str = "日経平均",
) -> dict[str, Any]:
    data = _canonical_frame(frame, basis_date)
    view = data.tail(60).copy()
    close = view["close"]
    current = float(close.iloc[-1])
    high = float(close.max())
    low = float(close.min())
    high_distance = _pct(current, high) or 0.0
    low_distance = _pct(current, low) or 0.0
    range_position = 50.0 if high == low else (current - low) / (high - low) * 100.0
    return_5 = _return_at(close, 5)
    return_20 = _return_at(close, 20)
    return_60 = _return_at(close, 59)
    drawdown = maximum_drawdown(close)
    drawdown_start_price = float(close.loc[drawdown.start])
    drawdown_bottom_price = float(close.loc[drawdown.bottom])
    recovery = _pct(current, drawdown_bottom_price) or 0.0
    from_drawdown_start = _pct(current, drawdown_start_price) or 0.0
    returns = close.pct_change()
    abs_20 = float(returns.tail(20).abs().mean() * 100.0) if returns.tail(20).notna().any() else None
    abs_60 = float(returns.tail(60).abs().mean() * 100.0) if returns.tail(60).notna().any() else None
    vol_20 = float(returns.tail(20).std() * np.sqrt(252) * 100.0) if returns.tail(20).count() >= 10 else None
    vol_60 = float(returns.tail(60).std() * np.sqrt(252) * 100.0) if returns.tail(60).count() >= 20 else None
    volatility_ratio = (abs_20 / abs_60) if abs_20 is not None and abs_60 not in (None, 0) else None
    market_20, market_60 = _benchmark_returns(benchmark, view.index)
    relative_20 = (return_20 - market_20) if return_20 is not None and market_20 is not None else None

    sections: list[dict[str, str]] = []
    if drawdown.percent <= -5.0:
        body = (
            f"{_fmt_date(drawdown.start)}から{_fmt_date(drawdown.bottom)}にかけて株価は{_fmt_pct(drawdown.percent, True)}下落しました。"
            f"その後は{_fmt_date(view.index[-1])}までに安値から{_fmt_pct(recovery, True)}戻しています。"
            f"現在値は下落開始時点と比べて{_fmt_pct(from_drawdown_start, True)}"
            f"{'上回っています' if from_drawdown_start >= 0 else '下回っています'}。"
        )
    else:
        rise_start, rise_top, rise_pct = maximum_runup(close)
        if rise_pct >= 5.0:
            after_top = _pct(current, float(close.loc[rise_top])) or 0.0
            body = (
                f"{_fmt_date(rise_start)}から{_fmt_date(rise_top)}にかけて株価は{_fmt_pct(rise_pct)}上昇しました。"
                f"その後の現在値は、その期間の高値を{_fmt_pct(after_top, True)}"
                f"{'上回っています' if after_top >= 0 else '下回っています'}。"
            )
        else:
            body = "直近60営業日では、5％を超える連続的な上昇または下落は見られず、比較的限られた範囲で推移しました。"
    section = _safe_section("story", "最近のチャートのあらすじ", body)
    if section:
        sections.append(section)

    if return_5 is not None and return_60 is not None:
        if return_5 * return_60 < 0:
            body = (
                f"直近5営業日は{_direction_text(return_5)}が、60営業日で見ると{_direction_text(return_60)}。"
                "見る期間によって、チャートの方向が異なって見える状態です。"
            )
        else:
            common = "上回る" if return_5 >= 0 and return_60 >= 0 else "下回る"
            body = (
                f"直近5営業日と60営業日のどちらでも、開始時点を{common}動きが見られます。"
                "対象とする期間によって、値動きの大きさには違いがあります。"
            )
        section = _safe_section("horizon_gap", "短期と中期の温度差", body)
        if section:
            sections.append(section)

    body = (
        f"現在値は直近60営業日の高値から{_fmt_pct(high_distance, True)}下、"
        f"安値から{_fmt_pct(low_distance, True)}上にあります。"
        f"期間内の値幅では、{_position_text(range_position)}にあります。"
    )
    section = _safe_section("position", "現在の位置", body)
    if section:
        sections.append(section)

    if volatility_ratio is not None:
        if volatility_ratio >= 1.25:
            body = "直近20営業日の1日当たりの値動きは、過去60営業日の平均より大きくなっています。最近は、上下の振れが比較的広い状態です。"
        elif volatility_ratio <= 0.8:
            body = "直近20営業日の1日当たりの値動きは、過去60営業日の平均より小さくなっています。最近は、上下の振れが比較的狭い状態です。"
        else:
            body = "直近20営業日の1日当たりの値動きは、過去60営業日の平均とおおむね同程度です。"
        section = _safe_section("volatility", "値動きの大きさ", body)
        if section:
            sections.append(section)

    if return_20 is not None and market_20 is not None:
        same_direction = (return_20 >= 0) == (market_20 >= 0)
        if same_direction and abs(relative_20 or 0.0) < 2.0:
            body = (
                f"この銘柄と{benchmark_name}は、直近20営業日でおおむね同じ方向に動いています。"
                f"騰落率はこの銘柄が{_fmt_pct(return_20)}、{benchmark_name}が{_fmt_pct(market_20)}でした。"
            )
        else:
            body = (
                f"この銘柄は直近20営業日で{_fmt_pct(return_20, True)}{'上昇' if return_20 >= 0 else '下落'}し、"
                f"同期間の{benchmark_name}は{_fmt_pct(market_20, True)}{'上昇' if market_20 >= 0 else '下落'}しました。"
                f"騰落率の差は{_fmt_pct(relative_20 or 0.0, True)}でした。"
            )
        section = _safe_section("market", "市場全体との違い", body)
        if section:
            sections.append(section)

    volume = view["volume_value"].dropna()
    volume_5 = float(volume.tail(5).median()) if len(volume) >= 5 else None
    volume_20 = float(volume.tail(20).median()) if len(volume) >= 20 else None
    volume_prior = float(volume.iloc[-60:-20].median()) if len(volume) >= 50 else None
    volume_ratio = volume_5 / volume_20 if volume_5 is not None and volume_20 not in (None, 0) else None
    if len(sections) < 6 and volume_ratio is not None:
        if volume_ratio >= 1.25:
            body = "直近5営業日の出来高中央値は、直近20営業日の中央値を上回っています。"
        elif volume_ratio <= 0.8:
            body = "直近5営業日の出来高中央値は、直近20営業日の中央値を下回っています。"
        else:
            body = "直近5営業日の出来高中央値は、直近20営業日の中央値とおおむね同程度です。"
        section = _safe_section("volume", "出来高の変化", body)
        if section:
            sections.append(section)

    if len(sections) < 6 and return_5 is not None and return_60 is not None:
        if return_5 * return_60 < 0:
            body = "短期と中期の方向がそろっていないため、選ぶ期間によってチャートの印象が変わりやすい状態です。"
        else:
            body = "短期と中期では、おおむね同じ方向の動きが見られます。ただし、対象とする期間によって値動きの大きさは異なります。"
        section = _safe_section("mixed_view", "見方が分かれやすいところ", body)
        if section:
            sections.append(section)

    sections = sections[:6]
    if len(sections) < 3:
        sections.append({"key": "data_scope", "title": "分析した期間", "body": "取得できた確定日足のうち、直近の営業日を中心にチャートを整理しています。"})

    report = {
        "schema_version": CHART_ANALYSIS_VERSION,
        "basis_date": view.index[-1].strftime("%Y-%m-%d"),
        "window_start": view.index[0].strftime("%Y-%m-%d"),
        "window_end": view.index[-1].strftime("%Y-%m-%d"),
        "disclaimer": CHART_ANALYSIS_DISCLAIMER,
        "sections": sections,
        "footer": CHART_ANALYSIS_FOOTER,
        "metrics": {
            "max_drawdown_start": drawdown.start.strftime("%Y-%m-%d"),
            "max_drawdown_bottom": drawdown.bottom.strftime("%Y-%m-%d"),
            "max_drawdown_pct": drawdown.percent,
            "recovery_from_bottom_pct": recovery,
            "difference_from_drawdown_start_pct": from_drawdown_start,
            "five_day_return_pct": return_5,
            "twenty_day_return_pct": return_20,
            "sixty_day_return_pct": return_60,
            "distance_from_high_pct": high_distance,
            "distance_from_low_pct": low_distance,
            "range_position_pct": range_position,
            "average_absolute_return_20_pct": abs_20,
            "average_absolute_return_60_pct": abs_60,
            "volatility_20_pct": vol_20,
            "volatility_60_pct": vol_60,
            "volatility_ratio": volatility_ratio,
            "benchmark_name": benchmark_name if market_20 is not None else None,
            "benchmark_twenty_day_return_pct": market_20,
            "benchmark_sixty_day_return_pct": market_60,
            "relative_return_vs_benchmark_pct": relative_20,
            "volume_recent_5_median": volume_5,
            "volume_recent_20_median": volume_20,
            "volume_prior_60_baseline_median": volume_prior,
            "volume_ratio_5_to_20": volume_ratio,
        },
    }
    forbidden = report_contains_forbidden_term(report)
    if forbidden:
        raise ServiceError("安全なチャート分析文を生成できませんでした。", "時間をおいて再度お試しください。")
    return report


def build_long_term_chart_analysis(
    frame: pd.DataFrame,
    basis_date: str | pd.Timestamp | None = None,
    benchmark: pd.Series | pd.DataFrame | None = None,
    benchmark_name: str = "日経平均",
) -> dict[str, Any]:
    """基準日以前の直近約2年を使い、中長期チャートの事実だけを整理する。"""
    data = _canonical_frame(frame, basis_date)
    view = data.tail(504).copy()
    if len(view) < 252:
        raise ServiceError("中長期チャートに必要な履歴が不足しています。", "十分な取引履歴がある銘柄を指定してください。")
    close = view["close"]
    current = float(close.iloc[-1])
    return_6m = _return_at(close, 126)
    return_1y = _return_at(close, 252)
    return_2y = _pct(current, float(close.iloc[0]))
    return_60 = _return_at(close, 60)
    high = float(close.max())
    low = float(close.min())
    distance_high = _pct(current, high) or 0.0
    distance_low = _pct(current, low) or 0.0
    range_position = 50.0 if high == low else (current - low) / (high - low) * 100.0
    drawdown = maximum_drawdown(close)
    rise_start, rise_top, rise_percent = maximum_runup(close)

    benchmark_1y: float | None = None
    benchmark_2y: float | None = None
    if benchmark is not None:
        if isinstance(benchmark, pd.DataFrame):
            column = "price" if "price" in benchmark else "Close" if "Close" in benchmark else None
            series = benchmark[column] if column else pd.Series(dtype=float)
        else:
            series = benchmark
        aligned = pd.to_numeric(series, errors="coerce")
        aligned.index = pd.DatetimeIndex(pd.to_datetime(aligned.index, errors="coerce")).tz_localize(None).normalize()
        aligned = aligned[~aligned.index.isna()].sort_index().groupby(level=0).last().reindex(view.index, method="ffill")
        benchmark_1y = _return_at(aligned, 252)
        if aligned.notna().sum() >= 252:
            valid = aligned.dropna()
            benchmark_2y = _pct(float(valid.iloc[-1]), float(valid.iloc[0])) if len(valid) > 1 else None

    sections: list[dict[str, str]] = []
    overall_direction = "上昇" if (return_2y or 0.0) >= 0 else "下落"
    body = (
        f"{_fmt_date(view.index[0])}から{_fmt_date(view.index[-1])}までの株価は、"
        f"開始時点と比べて{_fmt_pct(return_2y or 0.0, True)}{overall_direction}しました。"
        "この文章は期間内に確認できた終値の変化だけを表しています。"
    )
    section = _safe_section("two_year_story", "2年間の値動きのあらすじ", body)
    if section:
        sections.append(section)

    available_returns = []
    for label, value in (("6か月", return_6m), ("1年", return_1y), ("2年", return_2y)):
        if value is not None:
            available_returns.append(f"{label}では{_fmt_pct(value, True)}{'上昇' if value >= 0 else '下落'}")
    if available_returns:
        section = _safe_section(
            "horizon_returns",
            "6か月・1年・2年の騰落率",
            "、".join(available_returns) + "しました。期間の長さによって、騰落率と方向の見え方が異なる場合があります。",
        )
        if section:
            sections.append(section)

    section = _safe_section(
        "long_position",
        "2年高値・安値から見た現在位置",
        f"現在値は、期間内の高値より{_fmt_pct(distance_high, True)}低く、安値より{_fmt_pct(distance_low, True)}高い位置にあります。"
        f"2年間の値幅全体で見ると、現在は{_long_position_level_text(range_position)}にあります。",
    )
    if section:
        sections.append(section)

    section = _safe_section(
        "major_phases",
        "主な上昇局面と下落局面",
        f"期間内では、{_fmt_date(rise_start)}から{_fmt_date(rise_top)}にかけて{_fmt_pct(rise_percent, True)}上昇しました。"
        f"また、{_fmt_date(drawdown.start)}から{_fmt_date(drawdown.bottom)}にかけて{_fmt_pct(drawdown.percent, True)}下落しました。",
    )
    if section:
        sections.append(section)

    ma60 = _finite(view["ma60"].iloc[-1])
    ma200 = _finite(view["ma200"].iloc[-1])
    if ma60 is not None and ma200 is not None:
        body = (
            f"現在値は60日移動平均を{_fmt_pct(_pct(current, ma60) or 0.0, True)}"
            f"{'上回り' if current >= ma60 else '下回り'}、200日移動平均を{_fmt_pct(_pct(current, ma200) or 0.0, True)}"
            f"{'上回っています' if current >= ma200 else '下回っています'}。"
            f"60日移動平均は200日移動平均を{'上回っています' if ma60 >= ma200 else '下回っています'}。"
        )
        section = _safe_section("moving_averages", "60日・200日移動平均の関係", body)
        if section:
            sections.append(section)

    if benchmark_1y is not None and return_1y is not None:
        difference = return_1y - benchmark_1y
        body = (
            f"直近1年では、この銘柄は{_fmt_pct(return_1y, True)}{'上昇' if return_1y >= 0 else '下落'}し、"
            f"{benchmark_name}は{_fmt_pct(benchmark_1y, True)}{'上昇' if benchmark_1y >= 0 else '下落'}しました。"
            f"騰落率の差は{_fmt_pct(difference, True)}でした。"
        )
        section = _safe_section("long_market", "市場全体との違い", body)
    elif return_60 is not None and return_2y is not None:
        section = _safe_section(
            "short_long_gap",
            "短期チャートとの見え方の違い",
            f"直近60営業日は{_fmt_pct(return_60, True)}{'上昇' if return_60 >= 0 else '下落'}し、"
            f"2年間では{_fmt_pct(return_2y, True)}{'上昇' if return_2y >= 0 else '下落'}しました。"
            "短期と中長期では、選ぶ期間によってチャートの印象が異なる場合があります。",
        )
    else:
        section = None
    if section:
        sections.append(section)

    sections = sections[:6]
    report = {
        "schema_version": LONG_CHART_ANALYSIS_VERSION,
        "basis_date": view.index[-1].strftime("%Y-%m-%d"),
        "window_start": view.index[0].strftime("%Y-%m-%d"),
        "window_end": view.index[-1].strftime("%Y-%m-%d"),
        "sections": sections,
        "footer": LONG_CHART_ANALYSIS_FOOTER,
        "metrics": {
            "six_month_return_pct": return_6m,
            "one_year_return_pct": return_1y,
            "two_year_return_pct": return_2y,
            "sixty_day_return_pct": return_60,
            "distance_from_two_year_high_pct": distance_high,
            "distance_from_two_year_low_pct": distance_low,
            "two_year_range_position_pct": range_position,
            "max_drawdown_start": drawdown.start.strftime("%Y-%m-%d"),
            "max_drawdown_bottom": drawdown.bottom.strftime("%Y-%m-%d"),
            "max_drawdown_pct": drawdown.percent,
            "max_runup_start": rise_start.strftime("%Y-%m-%d"),
            "max_runup_top": rise_top.strftime("%Y-%m-%d"),
            "max_runup_pct": rise_percent,
            "ma60": ma60,
            "ma200": ma200,
            "benchmark_name": benchmark_name if benchmark_1y is not None else None,
            "benchmark_one_year_return_pct": benchmark_1y,
            "benchmark_two_year_return_pct": benchmark_2y,
        },
    }
    forbidden = report_contains_forbidden_term(report)
    if forbidden:
        raise ServiceError("安全な中長期チャート分析文を生成できませんでした。", "時間をおいて再度お試しください。")
    return report


def chart_data(frame: pd.DataFrame, basis_date: str | pd.Timestamp | None = None) -> list[dict[str, Any]]:
    view = _canonical_frame(frame, basis_date).tail(60)
    records: list[dict[str, Any]] = []
    for date, row in view.iterrows():
        records.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "close": _finite(row["close"]),
                "volume": _finite(row["volume_value"]),
                "ma5": _finite(row["ma5"]),
                "ma20": _finite(row["ma20"]),
                "ma60": _finite(row["ma60"]),
            }
        )
    return records


def long_chart_data(frame: pd.DataFrame, basis_date: str | pd.Timestamp | None = None) -> list[dict[str, Any]]:
    view = _canonical_frame(frame, basis_date).tail(504)
    return [
        {
            "date": date.strftime("%Y-%m-%d"),
            "close": _finite(row["close"]),
            "ma60": _finite(row["ma60"]),
            "ma200": _finite(row["ma200"]),
        }
        for date, row in view.iterrows()
    ]


def plot_individual_chart(
    frame: pd.DataFrame,
    ticker: str,
    basis_date: str | pd.Timestamp,
    report: dict[str, Any],
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "Noto Sans CJK JP", "DejaVu Sans"]

    view = _canonical_frame(frame, basis_date).tail(60)
    fig, axis = plt.subplots(figsize=(10, 4.8), facecolor="#111827")
    axis.set_facecolor("#111827")
    axis.plot(view.index, view["close"], color="#7dd3fc", linewidth=2.0, label="終値")
    for column, color, label, width in (
        ("ma5", "#f8fafc", "5日移動平均", 1.15),
        ("ma20", "#94a3b8", "20日移動平均", 1.25),
        ("ma60", "#818cf8", "60日移動平均", 1.35),
    ):
        if view[column].notna().any():
            axis.plot(view.index, view[column], color=color, linewidth=width, alpha=0.9, label=label)
    metrics = report.get("metrics", {})
    for key, marker, color, label in (
        ("max_drawdown_start", "o", "#f59e0b", "主な下落開始"),
        ("max_drawdown_bottom", "v", "#fb7185", "主な下落底"),
    ):
        date_text = metrics.get(key)
        if date_text:
            date = pd.Timestamp(date_text)
            if date in view.index:
                axis.scatter(date, view.loc[date, "close"], marker=marker, color=color, edgecolor="white", s=55, zorder=5, label=label)
    axis.scatter(view.index[-1], view["close"].iloc[-1], color="#22d3ee", edgecolor="white", s=65, zorder=6, label="最新値")
    axis.set_title(f"{ticker} 直近60営業日", color="#f8fafc")
    axis.set_ylabel("株価", color="#cbd5e1")
    axis.grid(color="#334155", linestyle="--", alpha=0.5)
    axis.tick_params(colors="#cbd5e1")
    for spine in axis.spines.values():
        spine.set_color("#475569")
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate(rotation=30)
    legend = axis.legend(loc="best", ncol=3, frameon=True, fontsize=8)
    legend.get_frame().set_facecolor("#1e293b")
    legend.get_frame().set_edgecolor("#475569")
    for text in legend.get_texts():
        text.set_color("#f8fafc")
    fig.tight_layout()
    safe_ticker = "".join(character if character.isalnum() else "_" for character in ticker)
    filename = f"{safe_ticker}_individual_chart_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    path: Path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=145, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return f"/outputs/{filename}"


def plot_individual_long_chart(
    frame: pd.DataFrame,
    ticker: str,
    basis_date: str | pd.Timestamp,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "Noto Sans CJK JP", "DejaVu Sans"]
    view = _canonical_frame(frame, basis_date).tail(504)
    if len(view) < 252:
        raise ServiceError("中長期チャートに必要な履歴が不足しています。")
    fig, axis = plt.subplots(figsize=(10, 4.8), facecolor="#111827")
    axis.set_facecolor("#111827")
    axis.plot(view.index, view["close"], color="#7dd3fc", linewidth=1.7, label="終値")
    axis.plot(view.index, view["ma60"], color="#94a3b8", linewidth=1.3, label="60日移動平均")
    axis.plot(view.index, view["ma200"], color="#818cf8", linewidth=1.45, label="200日移動平均")
    axis.scatter(view.index[-1], view["close"].iloc[-1], color="#22d3ee", edgecolor="white", s=55, zorder=5, label="最新値")
    axis.set_title(f"{ticker} 直近2年間", color="#f8fafc")
    axis.set_ylabel("株価", color="#cbd5e1")
    axis.grid(color="#334155", linestyle="--", alpha=0.5)
    axis.tick_params(colors="#cbd5e1")
    for spine in axis.spines.values():
        spine.set_color("#475569")
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30)
    legend = axis.legend(loc="best", frameon=True, fontsize=8)
    legend.get_frame().set_facecolor("#1e293b")
    legend.get_frame().set_edgecolor("#475569")
    for text in legend.get_texts():
        text.set_color("#f8fafc")
    fig.tight_layout()
    safe_ticker = "".join(character if character.isalnum() else "_" for character in ticker)
    filename = f"{safe_ticker}_individual_long_chart_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    path: Path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=145, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return f"/outputs/{filename}"
