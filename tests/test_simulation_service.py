from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services.common import ServiceError
from services.simulation_service import (
    LOT_MODE,
    LOT_SIZE,
    PRICE_ADJUSTMENT_METHOD,
    SIMULATION_SCHEMA_VERSION,
    calculate_lot_purchase,
    download_adjusted_ohlc,
    is_current_simulation_result,
    max_drawdown,
    normalize_security_code,
    simulate,
)


ROOT = Path(__file__).resolve().parents[1]


def frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=pd.bdate_range("2025-01-06", periods=len(rows)))


@pytest.mark.parametrize(
    ("raw", "code", "ticker"),
    [("7203", "7203", "7203.T"), ("130A", "130A", "130A.T"), ("130a", "130A", "130A.T")],
)
def test_japanese_security_code_is_normalized_only_for_internal_download(raw: str, code: str, ticker: str) -> None:
    assert normalize_security_code(raw) == (code, ticker)


@pytest.mark.parametrize("raw", ["", "8035.T", "AAPL", "NVDA", "720", "72030", "7203,130A", "72-3"])
def test_invalid_public_security_codes_are_rejected(raw: str) -> None:
    with pytest.raises(ServiceError):
        normalize_security_code(raw)


def test_at_least_one_exit_condition_is_required() -> None:
    with pytest.raises(ServiceError, match="どちらか"):
        simulate(frame([(100, 101, 99, 100), (100, 101, 99, 100)]), None, None)


@pytest.mark.parametrize("value", [0, -1, np.inf, -np.inf, np.nan])
def test_rates_must_be_positive_finite_numbers(value: float) -> None:
    with pytest.raises(ServiceError):
        simulate(frame([(100, 101, 99, 100), (100, 101, 99, 100)]), value, 4)


def test_first_entry_uses_first_open_and_period_end_uses_last_close() -> None:
    result = simulate(frame([(100, 101, 99, 100), (102, 104, 101, 103), (104, 106, 103, 105)]), 100, 100)
    trade = result["trades"][0]
    assert trade["entry_price"] == 100
    assert trade["exit_price"] == 105
    assert trade["reason"] == "検証期間終了"
    assert trade["held_trading_days"] == 3


def test_exit_day_is_not_reused_and_next_trading_day_is_entry() -> None:
    result = simulate(frame([(100, 111, 99, 110), (105, 106, 104, 105), (105, 106, 104, 105)]), 10, 50)
    assert result["settlement_points"][0]["date"] == result["price_curve"][0]["date"]
    assert result["purchase_points"][1]["date"] == result["price_curve"][1]["date"]
    assert result["purchase_points"][1]["date"] != result["settlement_points"][0]["date"]


def test_final_day_does_not_create_zero_duration_trade() -> None:
    result = simulate(frame([(100, 111, 99, 110), (105, 116, 104, 115), (115, 116, 114, 115)]), 10, 50)
    assert all(trade["entry_date"] != result["simulation_end"] for trade in result["trades"])


@pytest.mark.parametrize(
    ("second_row", "rate", "field", "expected_price", "reason"),
    [
        ((90, 92, 88, 91), 4, "stop", 90, "損切り条件"),
        ((112, 113, 111, 112), 10, "take", 112, "利益確定条件"),
    ],
)
def test_open_gap_executes_at_open(second_row, rate: float, field: str, expected_price: float, reason: str) -> None:
    take, stop = (rate, 50) if field == "take" else (100, rate)
    result = simulate(frame([(100, 101, 99, 100), second_row, (100, 101, 99, 100)]), take, stop)
    assert result["trades"][0]["exit_price"] == expected_price
    assert result["trades"][0]["reason"] == reason


def test_intraday_hit_executes_at_condition_price() -> None:
    result = simulate(frame([(100, 109, 99, 105), (105, 111, 104, 110), (110, 111, 109, 110)]), 10, 50)
    assert result["trades"][0]["exit_price"] == pytest.approx(110)
    assert result["trades"][0]["reason"] == "利益確定条件"


def test_same_day_both_triggers_prioritizes_stop_loss() -> None:
    result = simulate(frame([(100, 112, 94, 100), (100, 101, 99, 100)]), 10, 4)
    assert result["trades"][0]["exit_price"] == pytest.approx(96)
    assert result["trades"][0]["reason"] == "損切り条件"


def test_trade_outcomes_add_up_to_completed_round_trips() -> None:
    result = simulate(frame([(100, 111, 99, 110), (100, 101, 94, 96), (100, 101, 99, 100)]), 10, 4)
    assert result["profitable_trades"] + result["losing_trades"] + result["break_even_trades"] == result["trade_count"]
    assert result["trade_count"] == len(result["trades"]) == len(result["settlement_points"])


def test_maximum_drawdown_uses_daily_mark_to_market_values() -> None:
    values = pd.Series([100.0, 120.0, 90.0, 108.0], index=pd.bdate_range("2025-01-01", periods=4))
    assert max_drawdown(values) == pytest.approx(-25.0)


def test_buy_and_hold_uses_same_start_open_and_end_close() -> None:
    result = simulate(frame([(100, 101, 99, 100), (102, 104, 101, 103), (104, 106, 103, 105)]), 100, 100, 100_000)
    assert result["benchmark_shares"] == 1_000
    assert result["benchmark_cash"] == 0
    assert result["buy_hold_final_assets"] == pytest.approx(105_000)
    assert result["buy_hold_profit_rate"] == pytest.approx(5.0)
    assert result["buy_hold_difference_points"] == pytest.approx(result["profit_rate"] - 5.0)


def test_download_uses_target_unadjusted_daily_ohlc_and_split_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    index = pd.bdate_range("2025-01-06", periods=3)
    raw = pd.DataFrame({"Open": [100, 50, 51], "High": [102, 52, 53], "Low": [99, 49, 50], "Close": [100, 51, 52], "Stock Splits": [0, 2, 0]}, index=index)

    def fake_download(ticker, **kwargs):
        calls.append({"ticker": ticker, **kwargs})
        return raw

    monkeypatch.setattr("yfinance.download", fake_download)
    downloaded, metadata = download_adjusted_ohlc("7203.T", datetime(2025, 1, 1), datetime(2025, 2, 1))
    assert len(calls) == 1 and calls[0]["ticker"] == "7203.T"
    assert calls[0]["auto_adjust"] is False and calls[0]["actions"] is True and calls[0]["interval"] == "1d"
    assert list(downloaded.columns) == ["open", "high", "low", "close", "stock_splits"]
    assert downloaded.iloc[1]["stock_splits"] == 2
    assert metadata["price_adjustment_method"] == PRICE_ADJUSTMENT_METHOD


def test_simulation_source_contains_no_prediction_model_or_external_market_fetch() -> None:
    source = (ROOT / "services" / "simulation_service.py").read_text(encoding="utf-8")
    for forbidden in ("RandomForest", "make_predictions", "prediction_signal", "calc_rsi", "MACRO_TICKERS", "^VIX", "JPY=X"):
        assert forbidden not in source


def test_current_schema_requires_new_chart_and_comparison_fields() -> None:
    current = {
        "schema_version": SIMULATION_SCHEMA_VERSION,
        "simulation_start": "2025-01-01", "simulation_end": "2025-02-01",
        "data_collection_start": "2025-01-01", "data_collection_end": "2025-02-01",
        "equity_curve": [], "price_curve": [], "purchase_points": [], "settlement_points": [],
        "buy_hold_profit_rate": 0.0, "lot_size": LOT_SIZE, "lot_mode": LOT_MODE,
        "initial_purchase_shares": 100, "benchmark_shares": 100,
    }
    assert is_current_simulation_result(current)
    assert not is_current_simulation_result({**current, "schema_version": "old"})
