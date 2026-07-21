from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services.common import ServiceError
from services.individual_chart_report import (
    CHART_ANALYSIS_DISCLAIMER,
    CHART_ANALYSIS_FOOTER,
    FORBIDDEN_REPORT_TERMS,
    LONG_CHART_ANALYSIS_FOOTER,
    SIX_STAGE_DISPLAY_ORDER,
    SIX_STAGE_REPORT_FOOTER,
    build_chart_analysis,
    build_long_term_chart_analysis,
    build_six_stage_trend_report,
    chart_data,
    long_chart_data,
    maximum_drawdown,
    report_contains_forbidden_term,
)


def _frame(prices: np.ndarray | list[float], volume: np.ndarray | list[float] | None = None) -> pd.DataFrame:
    index = pd.bdate_range("2025-01-06", periods=len(prices))
    if volume is None:
        volume = np.linspace(1_000, 2_000, len(prices))
    return pd.DataFrame({"price": prices, "volume": volume}, index=index)


def _mixed_frame() -> pd.DataFrame:
    prices = np.r_[np.linspace(90, 120, 45), np.linspace(120, 96, 10), np.linspace(96, 108, 25)]
    return _frame(prices)


def test_maximum_drawdown_detects_start_bottom_and_percent() -> None:
    frame = _frame([100, 120, 114, 90, 99, 105])
    result = maximum_drawdown(frame["price"])
    assert result.start == frame.index[1]
    assert result.bottom == frame.index[3]
    assert result.percent == pytest.approx(-25.0)


def test_recovery_and_difference_from_start_are_correct() -> None:
    frame = _frame([100] * 60 + [120, 90, 99])
    report = build_chart_analysis(frame)
    metrics = report["metrics"]
    assert metrics["recovery_from_bottom_pct"] == pytest.approx(10.0)
    assert metrics["difference_from_drawdown_start_pct"] == pytest.approx(-17.5)


def test_five_twenty_and_sixty_session_returns_are_correct() -> None:
    prices = np.arange(1, 82, dtype=float)
    report = build_chart_analysis(_frame(prices))
    metrics = report["metrics"]
    assert metrics["five_day_return_pct"] == pytest.approx((81 / 76 - 1) * 100)
    assert metrics["twenty_day_return_pct"] == pytest.approx((81 / 61 - 1) * 100)
    assert metrics["sixty_day_return_pct"] == pytest.approx((81 / 22 - 1) * 100)


def test_high_low_distances_and_range_position_are_correct() -> None:
    prices = np.r_[np.linspace(100, 200, 59), 150]
    report = build_chart_analysis(_frame(prices))
    metrics = report["metrics"]
    assert metrics["distance_from_high_pct"] == pytest.approx(-25.0)
    assert metrics["distance_from_low_pct"] == pytest.approx(50.0)
    assert metrics["range_position_pct"] == pytest.approx(50.0)


def test_constant_prices_do_not_divide_by_zero() -> None:
    report = build_chart_analysis(_frame(np.full(80, 100.0)))
    assert report["metrics"]["range_position_pct"] == 50.0
    assert report["metrics"]["volatility_ratio"] is None


def test_recent_volatility_expansion_is_detected() -> None:
    quiet = np.linspace(100, 101, 60)
    noisy = 101 + np.tile([4, -4], 10)
    report = build_chart_analysis(_frame(np.r_[quiet, noisy]))
    assert report["metrics"]["volatility_ratio"] > 1.25
    section = next(item for item in report["sections"] if item["key"] == "volatility")
    assert "大きく" in section["body"]


def test_nikkei_difference_is_computed_with_past_only_alignment() -> None:
    frame = _frame(np.linspace(100, 130, 80))
    benchmark = pd.Series(np.linspace(100, 108, 80), index=frame.index)
    report = build_chart_analysis(frame, benchmark=benchmark)
    expected = report["metrics"]["twenty_day_return_pct"] - report["metrics"]["benchmark_twenty_day_return_pct"]
    assert report["metrics"]["relative_return_vs_benchmark_pct"] == pytest.approx(expected)
    assert any(section["key"] == "market" for section in report["sections"])


def test_future_benchmark_value_is_not_used() -> None:
    frame = _frame(np.linspace(100, 110, 80))
    future_date = frame.index[-1] + pd.offsets.BDay(1)
    benchmark = pd.Series(np.r_[np.linspace(100, 105, 80), 9999], index=frame.index.append(pd.DatetimeIndex([future_date])))
    first = build_chart_analysis(frame, benchmark=benchmark)
    benchmark.loc[future_date] = 1
    second = build_chart_analysis(frame, benchmark=benchmark)
    assert first["metrics"]["benchmark_twenty_day_return_pct"] == second["metrics"]["benchmark_twenty_day_return_pct"]


def test_volume_comparison_is_correct() -> None:
    volume = np.r_[np.full(75, 1000.0), np.full(5, 2000.0)]
    report = build_chart_analysis(_frame(np.linspace(100, 110, 80), volume))
    assert report["metrics"]["volume_recent_5_median"] == 2000.0
    assert report["metrics"]["volume_recent_20_median"] == 1000.0
    assert report["metrics"]["volume_ratio_5_to_20"] == 2.0


def test_missing_volume_safely_omits_volume_section() -> None:
    frame = _frame(np.linspace(100, 110, 80), np.full(80, np.nan))
    report = build_chart_analysis(frame)
    assert not any(section["key"] == "volume" for section in report["sections"])


def test_missing_nikkei_safely_omits_market_section() -> None:
    report = build_chart_analysis(_mixed_frame(), benchmark=None)
    assert not any(section["key"] == "market" for section in report["sections"])


def test_same_input_always_generates_same_report() -> None:
    frame = _mixed_frame()
    assert build_chart_analysis(frame) == build_chart_analysis(frame.copy())


def test_basis_date_excludes_future_rows_and_dates() -> None:
    frame = _mixed_frame()
    basis = frame.index[-6]
    report = build_chart_analysis(frame, basis)
    assert report["basis_date"] == basis.strftime("%Y-%m-%d")
    text = " ".join(section["body"] for section in report["sections"])
    for future in frame.index[-5:]:
        assert f"{future.year}年{future.month}月{future.day}日" not in text


def test_disclaimer_and_footer_are_exact() -> None:
    report = build_chart_analysis(_mixed_frame())
    assert report["disclaimer"] == CHART_ANALYSIS_DISCLAIMER
    assert report["footer"] == CHART_ANALYSIS_FOOTER


def test_no_nan_infinity_or_none_in_report_body() -> None:
    frame = _mixed_frame()
    frame.iloc[5:10, frame.columns.get_loc("volume")] = np.nan
    report = build_chart_analysis(frame)
    body = " ".join(section["body"] for section in report["sections"])
    assert "nan" not in body.lower()
    assert "inf" not in body.lower()
    assert "none" not in body.lower()


def test_all_forbidden_terms_are_absent_from_analysis_body() -> None:
    report = build_chart_analysis(_mixed_frame())
    assert report_contains_forbidden_term(report) == []
    body = " ".join(section["body"] for section in report["sections"])
    assert all(term not in body for term in FORBIDDEN_REPORT_TERMS)


@pytest.mark.parametrize("term", ["今後の注目点", "今後の確認項目", "判断のポイント", "押し目", "買い場", "反発が期待"])
def test_specific_prohibited_phrases_are_absent(term: str) -> None:
    body = " ".join(section["body"] for section in build_chart_analysis(_mixed_frame())["sections"])
    assert term not in body


@pytest.mark.parametrize(
    "prices",
    [
        np.r_[np.linspace(100, 130, 55), np.linspace(130, 105, 10), np.linspace(105, 115, 15)],
        np.r_[np.linspace(100, 80, 40), np.linspace(80, 125, 25), np.linspace(125, 112, 15)],
        np.r_[np.linspace(100, 130, 70), np.linspace(130, 122, 10)],
        np.r_[np.linspace(130, 100, 70), np.linspace(100, 108, 10)],
        np.linspace(100, 150, 80),
        np.linspace(150, 100, 80),
        100 + np.sin(np.arange(80) / 3) * 0.2,
        np.r_[np.linspace(100, 101, 60), 101 + np.tile([4, -4], 10)],
    ],
)
def test_synthetic_chart_patterns_generate_safe_sections(prices: np.ndarray) -> None:
    report = build_chart_analysis(_frame(prices))
    assert 3 <= len(report["sections"]) <= 6
    assert report_contains_forbidden_term(report) == []


def test_chart_data_has_dates_close_volume_and_three_moving_averages() -> None:
    records = chart_data(_mixed_frame())
    assert len(records) == 60
    assert set(records[-1]) == {"date", "close", "volume", "ma5", "ma20", "ma60"}
    assert records[-1]["ma60"] is not None


def test_chart_data_contains_no_nonfinite_numbers() -> None:
    records = chart_data(_mixed_frame())
    for record in records:
        for key in ("close", "volume", "ma5", "ma20", "ma60"):
            assert record[key] is None or math.isfinite(record[key])


def test_six_stage_report_uses_fixed_order_and_display_total_is_exactly_100() -> None:
    report = build_six_stage_trend_report([
        {"raw_label": raw_label, "probability": probability}
        for (raw_label, _), probability in zip(
            SIX_STAGE_DISPLAY_ORDER,
            (0.1111, 0.2222, 0.1234, 0.1567, 0.3001, 0.0865),
            strict=True,
        )
    ])
    assert [item["label"] for item in report["items"]] == [label for _, label in SIX_STAGE_DISPLAY_ORDER]
    assert sum(item["percentage"] for item in report["items"]) == pytest.approx(100.0)
    leader = max(report["items"], key=lambda item: item["percentage"])
    assert report["intro"] == (
        f"本システムが導き出した5営業日先のトレンド予測結果では、"
        f"{leader['label']} {leader['percentage']:.1f}％が6区分の中で最も大きい値を示しました。"
    )
    assert report["footer"] == SIX_STAGE_REPORT_FOOTER


def test_six_stage_report_handles_rounded_top_tie() -> None:
    report = build_six_stage_trend_report([
        {"raw_label": raw_label, "probability": probability}
        for (raw_label, _), probability in zip(
            SIX_STAGE_DISPLAY_ORDER,
            (0.3, 0.3, 0.1, 0.1, 0.1, 0.1),
            strict=True,
        )
    ])
    assert report["tie_for_top"] is True
    assert "同率最上位" in report["intro"]
    assert "上昇 Lv.3 30.0％" in report["intro"]
    assert "上昇 Lv.2 30.0％" in report["intro"]


def test_long_report_and_chart_use_two_year_history_without_future_rows() -> None:
    prices = 100 + np.linspace(0, 35, 560) + np.sin(np.arange(560) / 11) * 7
    frame = _frame(prices)
    basis = frame.index[-6]
    report = build_long_term_chart_analysis(frame, basis)
    assert 3 <= len(report["sections"]) <= 6
    assert report["basis_date"] == basis.strftime("%Y-%m-%d")
    assert report["footer"] == LONG_CHART_ANALYSIS_FOOTER
    assert report_contains_forbidden_term(report) == []
    records = long_chart_data(frame, basis)
    assert records[-1]["date"] == basis.strftime("%Y-%m-%d")
    assert set(records[-1]) == {"date", "close", "ma60", "ma200"}
    assert records[-1]["ma200"] is not None


def test_long_report_is_deterministic_and_contains_no_nonfinite_text() -> None:
    prices = 80 + np.linspace(0, 25, 540) + np.sin(np.arange(540) / 9) * 4
    frame = _frame(prices)
    first = build_long_term_chart_analysis(frame)
    second = build_long_term_chart_analysis(frame.copy())
    assert first == second
    body = " ".join(section["body"] for section in first["sections"])
    assert "nan" not in body.lower()
    assert "inf" not in body.lower()
    assert "none" not in body.lower()


def test_long_position_sentence_uses_natural_dynamic_high_low_wording() -> None:
    prices = np.r_[np.linspace(100, 200, 400), np.linspace(200, 130, 104)]
    report = build_long_term_chart_analysis(_frame(prices))
    section = next(item for item in report["sections"] if item["key"] == "long_position")
    assert section["body"] == (
        "現在値は、期間内の高値より35.0％低く、安値より30.0％高い位置にあります。"
        "2年間の値幅全体で見ると、現在は比較的低い水準にあります。"
    )
    assert "高値から" not in section["body"]
    assert "安値から" not in section["body"]


def test_too_little_chart_history_is_rejected() -> None:
    with pytest.raises(ServiceError):
        build_chart_analysis(_frame(np.arange(10, dtype=float)))


def test_chart_report_service_does_not_reference_external_ai_clients() -> None:
    source = Path("services/individual_chart_report.py").read_text(encoding="utf-8").lower()
    assert "openai" not in source
    assert "gemini" not in source
    assert "anthropic" not in source
    assert "requests.post" not in source
