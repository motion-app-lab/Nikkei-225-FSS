from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from services.common import ServiceError
from services.simulation_service import (
    INITIAL_LOT_ERROR,
    LOT_MODE,
    LOT_SIZE,
    SIMULATION_SCHEMA_VERSION,
    SPLIT_INFORMATION_ERROR,
    UNEXPLAINED_PRICE_DISCONTINUITY_ERROR,
    calculate_lot_purchase,
    is_current_simulation_result,
    simulate,
    simulate_strategy,
)


ROOT = Path(__file__).resolve().parents[1]


def price_frame(
    rows: list[tuple[float, float, float, float] | tuple[float, float, float, float, float]],
) -> pd.DataFrame:
    columns = ["open", "high", "low", "close"] + (["stock_splits"] if len(rows[0]) == 5 else [])
    return pd.DataFrame(rows, columns=columns, index=pd.bdate_range("2025-01-06", periods=len(rows)))


def test_maximum_affordable_100_share_lot_calculation_matches_specification() -> None:
    purchase = calculate_lot_purchase(1_000_000, 3_200)
    assert purchase == {
        "lot_cost": 320_000.0,
        "lot_count": 3,
        "entry_shares": 300,
        "entry_value": 960_000.0,
        "cash_after_entry": 40_000.0,
    }


def test_decimal_lot_floor_does_not_drop_or_add_an_affordable_lot() -> None:
    assert calculate_lot_purchase(320_000, 3_200)["entry_shares"] == 100
    assert calculate_lot_purchase(319_999.999, 3_200)["entry_shares"] == 0


def test_every_new_entry_is_100_share_multiple_and_cash_is_carried_forward() -> None:
    data = price_frame(
        [
            (3_200, 3_520, 3_190, 3_500),
            (4_000, 4_400, 3_990, 4_350),
            (4_300, 4_310, 4_290, 4_300),
        ]
    )
    result = simulate(data, 10, 50, 1_000_000)
    assert [point["entry_shares"] for point in result["purchase_points"]] == [300, 200]
    assert [point["cash_after_entry"] for point in result["purchase_points"]] == pytest.approx([40_000, 296_000])
    assert all(point["entry_shares"] % 100 == 0 for point in result["purchase_points"])
    assert result["trades"][0]["cash_after_exit"] == pytest.approx(1_096_000)
    assert result["trades"][1]["cash_before_entry"] == pytest.approx(1_096_000)


def test_take_profit_and_stop_prices_are_recomputed_from_each_entry_price() -> None:
    data = price_frame(
        [
            (100, 110, 99, 109, 0),
            (190, 209, 189, 205, 0),
            (210, 211, 209, 210, 0),
        ]
    )
    result = simulate(data, 10, 50, 100_000)
    assert [trade["entry_price"] for trade in result["trades"][:2]] == [100, 190]
    assert [trade["exit_price"] for trade in result["trades"][:2]] == pytest.approx([110, 209])


def test_initial_insufficient_cash_returns_the_required_japanese_error() -> None:
    with pytest.raises(ServiceError, match="100株を購入できません") as exc:
        simulate(price_frame([(100, 101, 99, 100), (100, 101, 99, 100)]), 10, 4, 9_999)
    assert str(exc.value) == INITIAL_LOT_ERROR


def test_reentry_insufficient_cash_stops_new_entries_and_keeps_cash() -> None:
    data = price_frame(
        [
            (100, 101, 50, 50),
            (60, 61, 59, 60),
            (70, 71, 69, 70),
        ]
    )
    result = simulate(data, 100, 50, 10_000)
    assert result["trade_count"] == 1
    assert result["reentry_stopped_due_to_insufficient_cash"] is True
    assert result["reentry_stop_date"] == "2025-01-07"
    assert result["reentry_stop_available_cash"] == pytest.approx(5_000)
    assert result["reentry_stop_required_cash"] == pytest.approx(6_000)
    assert result["final_assets"] == pytest.approx(5_000)
    assert result["equity_curve"][-1]["strategy_assets"] == pytest.approx(5_000)


def test_daily_equity_and_trade_return_use_invested_amount_not_total_cash() -> None:
    result = simulate(
        price_frame([(100, 101, 99, 100), (110, 111, 109, 110)]),
        100,
        100,
        10_500,
    )
    trade = result["trades"][0]
    assert result["equity_curve"][0]["strategy_assets"] == pytest.approx(10_500)
    assert result["final_assets"] == pytest.approx(11_500)
    assert trade["entry_value"] == pytest.approx(10_000)
    assert trade["cash_after_entry"] == pytest.approx(500)
    assert trade["trade_pnl"] == pytest.approx(1_000)
    assert trade["trade_pnl_rate"] == pytest.approx(10.0)


def test_benchmark_uses_same_100_share_lots_and_keeps_residual_cash() -> None:
    result = simulate(
        price_frame([(100, 101, 99, 100), (110, 111, 109, 110)]),
        100,
        100,
        10_500,
    )
    assert result["benchmark_shares"] == 100
    assert result["benchmark_cash"] == pytest.approx(500)
    assert result["benchmark_final_assets"] == pytest.approx(11_500)
    assert result["buy_hold_profit_rate"] == pytest.approx(1000 / 10500 * 100)


def test_two_for_one_split_preserves_cost_basis_and_asset_continuity() -> None:
    data = price_frame(
        [
            (1_000, 1_010, 990, 1_000, 0),
            (500, 505, 495, 500, 2),
            (510, 515, 505, 510, 0),
        ]
    )
    result = simulate(data, 100, 100, 1_000_000)
    trade = result["trades"][0]
    assert trade["entry_shares"] == 1_000
    assert trade["exit_shares"] == 2_000
    assert trade["entry_cost_basis"] == pytest.approx(1_000_000)
    assert trade["split_events"][0]["basis_price_after"] == pytest.approx(500)
    assert result["equity_curve"][0]["strategy_assets"] == pytest.approx(1_000_000)
    assert result["equity_curve"][1]["strategy_assets"] == pytest.approx(1_000_000)
    assert result["benchmark_final_shares"] == 2_000


def test_split_adjusts_trigger_before_that_days_ohlc_decision() -> None:
    data = price_frame(
        [
            (100, 101, 99, 100, 0),
            (50, 55, 49, 54, 2),
            (54, 55, 53, 54, 0),
        ]
    )
    result = simulate(data, 10, 50, 100_000)
    assert result["trades"][0]["exit_price"] == pytest.approx(55)
    assert result["trades"][0]["exit_shares"] == 2_000
    assert result["trades"][0]["trade_pnl_rate"] == pytest.approx(10.0)


def test_reverse_split_preserves_total_asset_continuity() -> None:
    data = price_frame(
        [
            (500, 505, 495, 500, 0),
            (1_000, 1_010, 990, 1_000, 0.5),
            (1_010, 1_020, 1_000, 1_010, 0),
        ]
    )
    result = simulate(data, 100, 100, 1_000_000)
    assert result["equity_curve"][0]["strategy_assets"] == pytest.approx(1_000_000)
    assert result["equity_curve"][1]["strategy_assets"] == pytest.approx(1_000_000)
    assert result["trades"][0]["exit_shares"] == 1_000
    assert result["benchmark_final_shares"] == 1_000


def test_unexplained_large_unadjusted_price_jump_is_not_silently_used() -> None:
    data = price_frame([(1_000, 1_010, 990, 1_000), (500, 505, 495, 500)])
    with pytest.raises(ServiceError, match="株式分割・株式併合") as exc:
        simulate(data, 10, 4, 1_000_000)
    assert str(exc.value) == UNEXPLAINED_PRICE_DISCONTINUITY_ERROR


def test_schema_rejects_fractional_legacy_results() -> None:
    current = {
        "schema_version": SIMULATION_SCHEMA_VERSION,
        "simulation_start": "2025-01-01",
        "simulation_end": "2025-02-01",
        "data_collection_start": "2025-01-01",
        "data_collection_end": "2025-02-01",
        "equity_curve": [],
        "price_curve": [],
        "purchase_points": [],
        "settlement_points": [],
        "buy_hold_profit_rate": 0.0,
        "lot_size": LOT_SIZE,
        "lot_mode": LOT_MODE,
        "initial_purchase_shares": 100,
        "benchmark_shares": 100,
    }
    assert is_current_simulation_result(current)
    assert not is_current_simulation_result({**current, "schema_version": "take_profit_stop_loss_simulation_v2"})
    assert not is_current_simulation_result({**current, "lot_size": 1})


def test_strategy_api_result_records_lot_and_unadjusted_split_method(monkeypatch: pytest.MonkeyPatch) -> None:
    data = price_frame([(100, 101, 99, 100, 0), (101, 102, 100, 101, 0)])
    metadata = {
        "auto_adjust": False,
        "actions": True,
        "stock_split_information_available": True,
        "price_adjustment_method": "unadjusted with splits",
    }
    monkeypatch.setattr("services.simulation_service.download_unadjusted_ohlc", lambda *_: (data, metadata))
    monkeypatch.setattr("services.simulation_service.get_ticker_name", lambda _: "テスト銘柄")
    result = simulate_strategy("7203", 100_000, 10, 4)
    assert result["schema_version"] == SIMULATION_SCHEMA_VERSION
    assert result["lot_size"] == 100 and result["lot_mode"] == LOT_MODE
    assert result["conditions"]["lot_size"] == 100
    assert result["rules"]["lot_size"] == 100
    assert result["costs"]["board_lot_and_integer_shares"] is True
    assert result["data_source"]["auto_adjust"] is False


def test_simulation_source_remains_independent_from_prediction_systems() -> None:
    source = (ROOT / "services" / "simulation_service.py").read_text(encoding="utf-8")
    for forbidden in ("RandomForest", "prediction_signal", "calc_rsi", "MACRO_TICKERS", "^VIX", "JPY=X"):
        assert forbidden not in source
    assert "LOT_SIZE = 100" in source
    assert "auto_adjust=False" in source


def test_simulation_download_timeout_remains_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    raw = pd.DataFrame(
        {"Open": [100, 101], "High": [101, 102], "Low": [99, 100], "Close": [100, 101], "Stock Splits": [0, 0]},
        index=pd.bdate_range("2025-01-06", periods=2),
    )

    def fake_download(ticker, **kwargs):
        captured.update(kwargs)
        return raw

    monkeypatch.setattr("yfinance.download", fake_download)
    from services.simulation_service import download_unadjusted_ohlc

    download_unadjusted_ohlc("7203.T", datetime(2025, 1, 1), datetime(2025, 2, 1))
    assert captured["timeout"] == 20
    assert captured["threads"] is False

@pytest.mark.parametrize("with_split_column", [False, True])
def test_normal_price_series_succeeds_with_missing_or_all_zero_split_column(with_split_column: bool) -> None:
    rows = [(100, 102, 99, 101), (101, 103, 100, 102), (102, 104, 101, 103)]
    if with_split_column:
        rows = [(*row, 0) for row in rows]
    result = simulate(price_frame(rows), 20, 20, 100_000)
    assert result["trade_count"] == 1


def test_all_zero_split_column_does_not_hide_large_discontinuity() -> None:
    data = price_frame([(1_000, 1_010, 990, 1_000, 0), (500, 505, 495, 500, 0)])
    with pytest.raises(ServiceError, match="シミュレーションを中止"):
        simulate(data, 10, 4, 1_000_000)


def test_three_for_one_split_is_consistent() -> None:
    data = price_frame([(900, 910, 890, 900, 0), (300, 305, 295, 300, 3), (306, 310, 302, 306, 0)])
    result = simulate(data, 100, 100, 900_000)
    assert result["trades"][0]["entry_shares"] == 1_000
    assert result["trades"][0]["exit_shares"] == 3_000
    assert result["trades"][0]["entry_cost_basis"] == pytest.approx(900_000)


def test_split_ratio_and_price_must_be_consistent() -> None:
    data = price_frame([(1_000, 1_010, 990, 1_000, 0), (900, 910, 890, 900, 2)])
    with pytest.raises(ServiceError, match="正確に計算"):
        simulate(data, 10, 4, 1_000_000)


@pytest.mark.parametrize("bad_ratio", [float("nan"), float("inf"), -2.0, 101.0])
def test_invalid_split_ratio_is_rejected(bad_ratio: float) -> None:
    data = price_frame([(1_000, 1_010, 990, 1_000, 0), (500, 505, 495, 500, bad_ratio)])
    with pytest.raises(ServiceError, match="正確に計算"):
        simulate(data, 10, 4, 1_000_000)


def test_split_created_odd_lot_is_not_rounded_but_new_entries_stay_board_lots() -> None:
    data = price_frame(
        [
            (100, 101, 99, 100, 0),
            (80, 88, 79, 87, 1.25),
            (90, 91, 89, 90, 0),
            (90, 91, 89, 90, 0),
        ]
    )
    result = simulate(data, 10, 90, 100_000)
    assert result["trades"][0]["exit_shares"] == 1_250
    assert result["trades"][0]["entry_cost_basis"] == pytest.approx(100_000)
    assert all(point["entry_shares"] % 100 == 0 for point in result["purchase_points"])


def test_missing_split_information_never_returns_a_fake_fifty_percent_loss() -> None:
    data = price_frame([(1_000, 1_010, 990, 1_000, 0), (500, 505, 495, 500, 0)])
    with pytest.raises(ServiceError) as exc:
        simulate(data, 100, 100, 1_000_000)
    assert "大きな価格変化" in str(exc.value)