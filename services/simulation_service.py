from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from typing import Any

import numpy as np
import pandas as pd

from .common import ServiceError, extract_yfinance_column, get_ticker_name, normalize_datetime_index


SIMULATION_SCHEMA_VERSION = "take_profit_stop_loss_simulation_v3_100share_lots"
SIMULATION_DATA_YEARS = 3
LOT_SIZE = 100
LOT_MODE = "max_affordable_100_share_lots"
PRICE_ADJUSTMENT_METHOD = (
    "yfinance auto_adjust=Falseによる未調整OHLCとStock Splitsを使用し、保有中の株数・基準価格を分割比率で調整"
)
DATA_COLLECTION_DEFINITION = (
    "対象銘柄の日足について、同一の未調整基準で取得できたOpen・High・Low・Closeと株式分割情報を用いた期間"
)
SAME_DAY_TRIGGER_RULE = "利益確定条件と損切り条件へ同日に到達した場合は、損切り条件を先に成立したものとして扱う"
REPURCHASE_RULE = "決済日の同日には再購入せず、次の取引可能日の始値で100株以上を購入できる場合に再購入する"
INITIAL_LOT_ERROR = (
    "指定した初期投資額では、検証開始日の株価で100株を購入できません。"
    "初期投資額を増やして、もう一度実行してください。"
)
SPLIT_INFORMATION_ERROR = (
    "対象期間内の株式分割・併合情報を確認できないため、"
    "100株単位のシミュレーションを正確に計算できませんでした。"
)
UNEXPLAINED_PRICE_DISCONTINUITY_ERROR = (
    "対象期間内に、株式分割・株式併合の可能性がある大きな価格変化を検出しましたが、"
    "対応する企業行動情報を確認できませんでした。正確な100株単位計算ができないため、"
    "シミュレーションを中止しました。"
)
SPLIT_PRICE_RATIO_LOW = 0.55
SPLIT_PRICE_RATIO_HIGH = 1.8
# 分割日の通常の市場変動を許容しつつ、分割比率と明らかに逆行する価格系列を拒否する。
SPLIT_CONSISTENCY_LOW = 0.65
SPLIT_CONSISTENCY_HIGH = 1.35
MAX_REALISTIC_SPLIT_RATIO = 100.0


def normalize_security_code(value: str) -> tuple[str, str]:
    """公開入力を日本株コードへ限定し、内部取得用だけに .T を付ける。"""
    code = (value or "").strip().upper()
    if not re.fullmatch(r"(?:\d{4}|\d{3}[A-Z])", code):
        raise ServiceError(
            "日本株の証券コードを確認できませんでした。",
            ".Tを付けず、7203または130Aのように1銘柄だけ入力してください。",
        )
    return code, f"{code}.T"


def _validate_positive_number(value: float, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ServiceError(f"{label}は0より大きい数値を入力してください。") from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise ServiceError(f"{label}は0より大きい有限の数値を入力してください。")
    return numeric


def _validate_rate(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    return _validate_positive_number(value, label)


def _decimal(value: Any, label: str = "数値") -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ServiceError(f"{label}を正しく計算できませんでした。") from exc
    if not result.is_finite():
        raise ServiceError(f"{label}を正しく計算できませんでした。")
    return result


def _float(value: Decimal | float | int) -> float:
    numeric = float(value)
    return 0.0 if abs(numeric) < 1e-12 else numeric


def _display_share_value(value: Decimal | float | int) -> int | float:
    numeric = _float(value)
    rounded = round(numeric)
    return int(rounded) if abs(numeric - rounded) < 1e-9 else numeric


def calculate_lot_purchase(
    available_cash: float | Decimal,
    entry_price: float | Decimal,
    lot_size: int = LOT_SIZE,
) -> dict[str, Any]:
    """利用可能資金から最大の100株単位をDecimalで算出する。"""
    cash = _decimal(available_cash, "利用可能資金")
    price = _decimal(entry_price, "購入価格")
    if cash < 0 or price <= 0 or lot_size <= 0:
        raise ServiceError("購入可能株数を正しく計算できませんでした。")
    lot_cost = price * Decimal(lot_size)
    lot_count = int((cash / lot_cost).to_integral_value(rounding=ROUND_FLOOR))
    shares = lot_count * lot_size
    entry_value = price * Decimal(shares)
    cash_after_entry = cash - entry_value
    if cash_after_entry < 0:
        raise ServiceError("利用可能資金を超える仮想購入を検出したため、計算を中止しました。")
    return {
        "lot_cost": _float(lot_cost),
        "lot_count": lot_count,
        "entry_shares": shares,
        "entry_value": _float(entry_value),
        "cash_after_entry": _float(cash_after_entry),
    }


def _validate_split_price_continuity(frame: pd.DataFrame) -> None:
    """企業行動列の有無にかかわらず、大きな価格断絶と分割比率の整合性を検査する。"""
    if len(frame) < 2:
        return
    previous_close = pd.to_numeric(frame["close"], errors="coerce").shift(1)
    current_open = pd.to_numeric(frame["open"], errors="coerce")
    price_ratio = current_open / previous_close.replace(0, np.nan)
    splits = pd.to_numeric(frame["stock_splits"], errors="coerce")
    for position in range(1, len(frame)):
        observed = float(price_ratio.iloc[position])
        split = float(splits.iloc[position])
        has_action = split not in (0.0, 1.0)
        suspicious = observed >= SPLIT_PRICE_RATIO_HIGH or observed <= SPLIT_PRICE_RATIO_LOW
        if suspicious and not has_action:
            raise ServiceError(
                UNEXPLAINED_PRICE_DISCONTINUITY_ERROR,
                "株式分割情報を取得できる状態で、もう一度実行してください。",
            )
        if has_action:
            expected = 1.0 / split
            consistency = observed / expected
            if not SPLIT_CONSISTENCY_LOW <= consistency <= SPLIT_CONSISTENCY_HIGH:
                raise ServiceError(
                    SPLIT_INFORMATION_ERROR,
                    "株式分割比率と分割日の価格変化が整合しないため、企業行動情報を確認してください。",
                )


def _validate_ohlc(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    required = ["open", "high", "low", "close"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ServiceError(
            "シミュレーションに必要な日足データが不足しています。",
            "始値・高値・安値・終値を同じ調整基準で取得できる銘柄を指定してください。",
        )
    source_rows = len(frame)
    split_information_available = "stock_splits" in frame.columns
    columns = required + (["stock_splits"] if split_information_available else [])
    usable = frame.loc[:, columns].copy()
    usable[required] = usable[required].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    usable = usable.dropna(subset=required)
    usable = usable[(usable[required] > 0).all(axis=1)]
    consistent = (
        (usable["high"] >= usable[["open", "close", "low"]].max(axis=1))
        & (usable["low"] <= usable[["open", "close", "high"]].min(axis=1))
    )
    usable = usable.loc[consistent]
    usable.index = normalize_datetime_index(usable.index)
    usable = usable[~usable.index.duplicated(keep="last")].sort_index()
    if split_information_available:
        splits = pd.to_numeric(usable["stock_splits"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if splits.isna().any() or (splits < 0).any() or (splits > MAX_REALISTIC_SPLIT_RATIO).any():
            raise ServiceError(SPLIT_INFORMATION_ERROR, "株式分割情報を取得できる状態で、もう一度実行してください。")
        usable["stock_splits"] = splits.fillna(0.0)
    else:
        usable["stock_splits"] = 0.0
    _validate_split_price_continuity(usable)
    if len(usable) < 2:
        raise ServiceError(
            "シミュレーションに必要な株価履歴が不足しています。",
            "別の日本株証券コードを指定してください。",
        )
    usable.attrs["split_information_available"] = split_information_available
    return usable, source_rows - len(usable)


def download_unadjusted_ohlc(ticker: str, start: datetime, end: datetime) -> tuple[pd.DataFrame, dict[str, Any]]:
    """対象銘柄の未調整日足OHLCと株式分割情報だけを取得する。"""
    try:
        import yfinance as yf

        raw = yf.download(
            ticker,
            start=start,
            end=end + timedelta(days=1),
            interval="1d",
            auto_adjust=False,
            actions=True,
            progress=False,
            threads=False,
            timeout=20,
        )
    except Exception as exc:
        raise ServiceError(
            "対象銘柄の株価データを取得できませんでした。",
            "インターネット接続を確認し、時間をおいて再度実行してください。",
            503,
        ) from exc
    if raw is None or raw.empty:
        raise ServiceError("対象銘柄の株価データを取得できませんでした。", "証券コードと上場状況を確認してください。")

    columns: dict[str, pd.Series] = {}
    for source, destination in (("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close")):
        series = extract_yfinance_column(raw, source, ticker)
        if series.empty:
            raise ServiceError(
                "シミュレーションに必要な日足データが不足しています。",
                "始値・高値・安値・終値を同じ調整基準で取得できる銘柄を指定してください。",
            )
        series.index = normalize_datetime_index(series.index)
        columns[destination] = series.groupby(level=0).last()

    split_series = extract_yfinance_column(raw, "Stock Splits", ticker)
    split_information_available = not split_series.empty
    if split_information_available:
        split_series.index = normalize_datetime_index(split_series.index)
        columns["stock_splits"] = split_series.groupby(level=0).last()
    frame = pd.concat(columns, axis=1)
    if split_information_available:
        frame["stock_splits"] = frame["stock_splits"].fillna(0.0)
    usable, dropped_rows = _validate_ohlc(frame)
    return usable, {
        "requested_ticker": ticker,
        "source": "Yahoo Finance",
        "interval": "1d",
        "auto_adjust": False,
        "actions": True,
        "stock_split_information_available": split_information_available,
        "price_adjustment_method": PRICE_ADJUSTMENT_METHOD,
        "missing_price_handling": "欠損OHLC行を除外し、価格補間は行わない",
        "source_rows": int(len(frame)),
        "usable_rows": int(len(usable)),
        "dropped_rows": int(dropped_rows),
    }


def download_adjusted_ohlc(ticker: str, start: datetime, end: datetime) -> tuple[pd.DataFrame, dict[str, Any]]:
    """旧呼出し名との互換用。返すデータは未調整OHLCと株式分割情報。"""
    return download_unadjusted_ohlc(ticker, start, end)


def max_drawdown(equity: pd.Series) -> float:
    numeric = pd.to_numeric(equity, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return 0.0
    drawdown = numeric / numeric.cummax().replace(0, np.nan) - 1.0
    return float(drawdown.min() * 100) if drawdown.notna().any() else 0.0


def _exit_decision(
    row: pd.Series,
    entry_price: float,
    take_profit: float | None,
    stop_loss: float | None,
    *,
    entry_day: bool,
) -> tuple[float, str] | None:
    entry = _decimal(entry_price, "購入価格")
    hundred = Decimal("100")
    take_price = (
        entry * (Decimal("1") + _decimal(take_profit) / hundred)
        if take_profit is not None
        else None
    )
    stop_price = (
        entry * (Decimal("1") - _decimal(stop_loss) / hundred)
        if stop_loss is not None
        else None
    )
    open_price = _decimal(row["open"], "始値")
    if not entry_day:
        if stop_price is not None and open_price <= stop_price:
            return _float(open_price), "損切り条件"
        if take_price is not None and open_price >= take_price:
            return _float(open_price), "利益確定条件"
    stop_hit = stop_price is not None and _decimal(row["low"], "安値") <= stop_price
    take_hit = take_price is not None and _decimal(row["high"], "高値") >= take_price
    if stop_hit:
        return _float(stop_price), "損切り条件"
    if take_hit:
        return _float(take_price), "利益確定条件"
    return None


def _split_ratio(row: pd.Series) -> Decimal | None:
    value = _decimal(row.get("stock_splits", 0.0), "株式分割比率")
    if value == 0:
        return None
    if value <= 0:
        raise ServiceError(SPLIT_INFORMATION_ERROR, "株式分割情報を確認して、もう一度実行してください。")
    return value


def simulate(
    frame: pd.DataFrame,
    take_profit: float | None,
    stop_loss: float | None,
    initial_investment: float = 1_000_000,
) -> dict[str, Any]:
    """指定率を、最大購入可能な100株単位と余剰現金で反復適用する。"""
    initial = _validate_positive_number(initial_investment, "初期投資額")
    take_profit = _validate_rate(take_profit, "利益確定条件")
    stop_loss = _validate_rate(stop_loss, "損切り条件")
    if take_profit is None and stop_loss is None:
        raise ServiceError(
            "利益確定条件または損切り条件のどちらかを入力してください。",
            "両方を使用しない場合は、同期間に保有し続けた場合と同じ条件になります。",
        )

    usable, dropped_rows = _validate_ohlc(frame)
    dates = usable.index
    last_position = len(usable) - 1
    initial_purchase = calculate_lot_purchase(initial, float(usable.iloc[0]["open"]))
    if initial_purchase["entry_shares"] == 0:
        raise ServiceError(INITIAL_LOT_ERROR, "初期投資額を増やして、もう一度実行してください。")

    cash = _decimal(initial, "初期投資額")
    shares = Decimal("0")
    entry_original_price = Decimal("0")
    threshold_basis_price = Decimal("0")
    entry_value = Decimal("0")
    entry_cost_basis = Decimal("0")
    cash_before_entry = Decimal("0")
    entry_index = -1
    next_entry_index = 0
    entry_date: pd.Timestamp | None = None
    entry_split_events: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    purchases: list[dict[str, Any]] = []
    settlements: list[dict[str, Any]] = []
    equity_values: list[float] = []
    reentry_stopped = False
    reentry_stop_date: str | None = None
    reentry_stop_open: float | None = None
    reentry_stop_cash: float | None = None
    reentry_required_cash: float | None = None
    initial_purchase_shares = 0
    initial_cash_after_entry = 0.0

    benchmark_purchase = calculate_lot_purchase(initial, float(usable.iloc[0]["open"]))
    benchmark_initial_shares = Decimal(benchmark_purchase["entry_shares"])
    benchmark_shares = benchmark_initial_shares
    benchmark_cash = _decimal(benchmark_purchase["cash_after_entry"], "同期間保有の余剰現金")
    benchmark_equity_values: list[float] = []

    for position, (date, row) in enumerate(usable.iterrows()):
        split_ratio = _split_ratio(row)
        if position > 0 and split_ratio is not None:
            benchmark_shares *= split_ratio
        benchmark_equity_values.append(
            _float(benchmark_cash + benchmark_shares * _decimal(row["close"], "終値"))
        )

        if shares > 0 and split_ratio is not None:
            shares_before = shares
            basis_before = threshold_basis_price
            shares *= split_ratio
            threshold_basis_price /= split_ratio
            entry_split_events.append(
                {
                    "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                    "ratio": _float(split_ratio),
                    "shares_before": _display_share_value(shares_before),
                    "shares_after": _display_share_value(shares),
                    "basis_price_before": _float(basis_before),
                    "basis_price_after": _float(threshold_basis_price),
                }
            )

        if shares == 0 and not reentry_stopped and position >= next_entry_index and position < last_position:
            purchase = calculate_lot_purchase(cash, float(row["open"]))
            if purchase["entry_shares"] == 0:
                if not purchases:
                    raise ServiceError(INITIAL_LOT_ERROR, "初期投資額を増やして、もう一度実行してください。")
                reentry_stopped = True
                reentry_stop_date = pd.Timestamp(date).strftime("%Y-%m-%d")
                reentry_stop_open = float(row["open"])
                reentry_stop_cash = _float(cash)
                reentry_required_cash = _float(_decimal(row["open"]) * Decimal(LOT_SIZE))
            else:
                cash_before_entry = cash
                entry_original_price = _decimal(row["open"], "購入価格")
                threshold_basis_price = entry_original_price
                shares = Decimal(purchase["entry_shares"])
                entry_value = entry_original_price * shares
                entry_cost_basis = entry_value
                cash = cash_before_entry - entry_value
                entry_index = position
                entry_date = pd.Timestamp(date)
                entry_split_events = []
                if not purchases:
                    initial_purchase_shares = int(purchase["entry_shares"])
                    initial_cash_after_entry = _float(cash)
                purchases.append(
                    {
                        "trade_number": len(trades) + 1,
                        "date": entry_date.strftime("%Y-%m-%d"),
                        "price": _float(entry_original_price),
                        "entry_shares": int(purchase["entry_shares"]),
                        "entry_value": _float(entry_value),
                        "cash_after_entry": _float(cash),
                        "label": "仮想購入",
                    }
                )

        if shares > 0:
            decision = _exit_decision(
                row,
                _float(threshold_basis_price),
                take_profit,
                stop_loss,
                entry_day=position == entry_index,
            )
            if decision is None and position == last_position:
                decision = (float(row["close"]), "検証期間終了")
            if decision is not None:
                exit_price, reason = decision
                exit_value = _decimal(exit_price, "決済価格") * shares
                cash_after_exit = cash + exit_value
                trade_pnl = exit_value - entry_cost_basis
                trade_pnl_rate = trade_pnl / entry_cost_basis * Decimal("100")
                trade_number = len(trades) + 1
                trade = {
                    "trade_number": trade_number,
                    "entry_date": entry_date.strftime("%Y-%m-%d") if entry_date is not None else "",
                    "entry_price": _float(entry_original_price),
                    "entry_shares": int(_display_share_value(Decimal(purchases[-1]["entry_shares"]))),
                    "exit_date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                    "exit_price": float(exit_price),
                    "exit_shares": _display_share_value(shares),
                    "held_trading_days": int(position - entry_index + 1),
                    "reason": reason,
                    "entry_value": _float(entry_value),
                    "cash_before_entry": _float(cash_before_entry),
                    "cash_after_entry": _float(cash),
                    "exit_value": _float(exit_value),
                    "cash_after_exit": _float(cash_after_exit),
                    "entry_cost_basis": _float(entry_cost_basis),
                    "trade_pnl": _float(trade_pnl),
                    "trade_pnl_rate": _float(trade_pnl_rate),
                    "profit_amount": _float(trade_pnl),
                    "profit_percent": _float(trade_pnl_rate),
                    "capital_before": _float(cash_before_entry),
                    "capital_after": _float(cash_after_exit),
                    "split_events": list(entry_split_events),
                    "had_stock_split_or_consolidation": bool(entry_split_events),
                }
                trades.append(trade)
                settlements.append(
                    {
                        "trade_number": trade_number,
                        "date": trade["exit_date"],
                        "price": float(exit_price),
                        "exit_shares": _display_share_value(shares),
                        "reason": reason,
                        "profit_percent": _float(trade_pnl_rate),
                        "label": "仮想決済",
                    }
                )
                cash = cash_after_exit
                shares = Decimal("0")
                entry_original_price = Decimal("0")
                threshold_basis_price = Decimal("0")
                entry_value = Decimal("0")
                entry_cost_basis = Decimal("0")
                cash_before_entry = Decimal("0")
                entry_index = -1
                entry_date = None
                entry_split_events = []
                next_entry_index = position + 1

        daily_equity = cash if shares == 0 else cash + shares * _decimal(row["close"], "終値")
        equity_values.append(_float(daily_equity))

    equity = pd.Series(equity_values, index=dates, dtype=float)
    buy_hold_equity = pd.Series(benchmark_equity_values, index=dates, dtype=float)
    final_assets = _float(equity.iloc[-1])
    buy_hold_final_assets = _float(buy_hold_equity.iloc[-1])
    profit_amount = _float(_decimal(final_assets) - _decimal(initial))
    profit_rate = _float(_decimal(profit_amount) / _decimal(initial) * Decimal("100"))
    buy_hold_profit_rate = _float(
        (_decimal(buy_hold_final_assets) - _decimal(initial)) / _decimal(initial) * Decimal("100")
    )
    comparison_points = _float(_decimal(profit_rate) - _decimal(buy_hold_profit_rate))

    tolerance = max(initial * 1e-12, 1e-9)
    profitable_trades = sum(1 for trade in trades if trade["trade_pnl"] > tolerance)
    losing_trades = sum(1 for trade in trades if trade["trade_pnl"] < -tolerance)
    break_even_trades = len(trades) - profitable_trades - losing_trades
    profitable_trade_rate = profitable_trades / len(trades) * 100.0 if trades else None
    equity_curve = [
        {
            "date": date.strftime("%Y-%m-%d"),
            "strategy_assets": _float(equity.loc[date]),
            "buy_hold_assets": _float(buy_hold_equity.loc[date]),
            "initial_assets": initial,
        }
        for date in dates
    ]
    price_curve = [
        {"date": date.strftime("%Y-%m-%d"), "close": float(usable.loc[date, "close"])}
        for date in dates
    ]
    return {
        "lot_size": LOT_SIZE,
        "lot_mode": LOT_MODE,
        "initial_purchase_shares": initial_purchase_shares,
        "initial_cash_after_entry": initial_cash_after_entry,
        "reentry_stopped_due_to_insufficient_cash": reentry_stopped,
        "reentry_stop_date": reentry_stop_date,
        "reentry_stop_open": reentry_stop_open,
        "reentry_stop_available_cash": reentry_stop_cash,
        "reentry_stop_required_cash": reentry_required_cash,
        "initial_assets": initial,
        "final_assets": final_assets,
        "profit_amount": profit_amount,
        "profit_rate": profit_rate,
        "max_drawdown": max_drawdown(equity),
        "trade_count": int(len(trades)),
        "profitable_trades": int(profitable_trades),
        "losing_trades": int(losing_trades),
        "break_even_trades": int(break_even_trades),
        "profitable_trade_rate": profitable_trade_rate,
        "benchmark_shares": int(_display_share_value(benchmark_initial_shares)),
        "benchmark_final_shares": _display_share_value(benchmark_shares),
        "benchmark_cash": _float(benchmark_cash),
        "benchmark_final_assets": buy_hold_final_assets,
        "buy_hold_final_assets": buy_hold_final_assets,
        "buy_hold_profit_rate": buy_hold_profit_rate,
        "buy_hold_difference_points": comparison_points,
        "trades": trades,
        "equity_curve": equity_curve,
        "price_curve": price_curve,
        "purchase_points": purchases,
        "settlement_points": settlements,
        "simulation_start": dates[0].strftime("%Y-%m-%d"),
        "simulation_end": dates[-1].strftime("%Y-%m-%d"),
        "data_collection_start": dates[0].strftime("%Y-%m-%d"),
        "data_collection_end": dates[-1].strftime("%Y-%m-%d"),
        "data_collection_definition": DATA_COLLECTION_DEFINITION,
        "split_information_available": bool(usable.attrs.get("split_information_available", False)),
        "dropped_price_rows": int(dropped_rows),
    }


def _condition_text(take_profit: float | None, stop_loss: float | None) -> str:
    parts: list[str] = []
    if take_profit is not None:
        parts.append(f"利益確定{take_profit:.1f}％")
    if stop_loss is not None:
        parts.append(f"損切り{stop_loss:.1f}％")
    return "、".join(parts)


def _result_summary(result: dict[str, Any], take_profit: float | None, stop_loss: float | None) -> str:
    condition_text = _condition_text(take_profit, stop_loss)
    return (
        "本システムが、各購入時点で購入可能な最大の100株単位を仮想購入し、"
        f"{condition_text}の条件を過去の株価データへ繰り返し適用した結果、"
        f"{result['trade_count']}回の取引が決済まで完了しました。"
        f"そのうち{result['profitable_trades']}回で1回の取引損益がプラスになりました。"
        f"損失は{result['losing_trades']}回、損益ゼロは{result['break_even_trades']}回でした。"
        "同じ期間に対象銘柄を100株単位で保有し続けた場合との差は"
        f"{result['buy_hold_difference_points']:+.1f}ポイントでした。"
    )


def is_current_simulation_result(result: dict[str, Any] | None) -> bool:
    if not isinstance(result, dict):
        return False
    required = {
        "schema_version",
        "simulation_start",
        "simulation_end",
        "data_collection_start",
        "data_collection_end",
        "equity_curve",
        "price_curve",
        "purchase_points",
        "settlement_points",
        "buy_hold_profit_rate",
        "lot_size",
        "lot_mode",
        "initial_purchase_shares",
        "benchmark_shares",
    }
    return (
        result.get("schema_version") == SIMULATION_SCHEMA_VERSION
        and result.get("lot_size") == LOT_SIZE
        and result.get("lot_mode") == LOT_MODE
        and required.issubset(result)
    )


def simulate_strategy(
    ticker: str,
    initial_investment: float = 1_000_000,
    take_profit: float | None = 10,
    stop_loss: float | None = 4,
) -> dict[str, Any]:
    security_code, normalized_ticker = normalize_security_code(ticker)
    initial = _validate_positive_number(initial_investment, "初期投資額")
    take_profit = _validate_rate(take_profit, "利益確定条件")
    stop_loss = _validate_rate(stop_loss, "損切り条件")
    if take_profit is None and stop_loss is None:
        raise ServiceError(
            "利益確定条件または損切り条件のどちらかを入力してください。",
            "両方を使用しない場合は、同期間に保有し続けた場合と同じ条件になります。",
        )

    end = datetime.now()
    start = (pd.Timestamp(end) - pd.DateOffset(years=SIMULATION_DATA_YEARS)).to_pydatetime()
    frame, download_metadata = download_unadjusted_ohlc(normalized_ticker, start, end)
    result = simulate(frame, take_profit=take_profit, stop_loss=stop_loss, initial_investment=initial)
    result.update(
        {
            "kind": "simulation",
            "schema_version": SIMULATION_SCHEMA_VERSION,
            "security_code": security_code,
            "ticker": normalized_ticker,
            "company_name": get_ticker_name(normalized_ticker),
            "fetched_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"),
            "conditions": {
                "initial_investment": initial,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "lot_size": LOT_SIZE,
                "lot_mode": LOT_MODE,
            },
            "data_source": download_metadata,
            "rules": {
                "position": "現物買いのみ",
                "capital": "各購入時点で購入可能な最大の100株単位を購入し、余剰現金を保持する複利計算",
                "entry": "検証期間の最初の取引可能日の始値。以後は決済日の次の取引可能日の始値",
                "lot_size": LOT_SIZE,
                "lot_mode": LOT_MODE,
                "repurchase": REPURCHASE_RULE,
                "gap_execution": "条件価格を始値で飛び越えた場合は、その始値で決済",
                "intraday_execution": "日中到達は条件価格で決済",
                "same_day_both_triggers": SAME_DAY_TRIGGER_RULE,
                "final_liquidation": "検証期間終了時の保有分は最終取引可能日の終値で決済",
                "stock_split_handling": "効力日のOHLC判定前に保有株数と1株当たり基準価格を分割比率で調整し、総取得原価は維持",
            },
            "costs": {
                "trading_fees": False,
                "taxes": False,
                "dividends": False,
                "slippage": False,
                "market_liquidity": False,
                "board_lot_and_integer_shares": True,
            },
            "result_summary": _result_summary(result, take_profit, stop_loss),
            "assumptions": [
                "対象銘柄自身の未調整日足OHLCと株式分割情報だけを使用し、予測モデルや外部市場データは使用しない",
                "現物買いのみ・空売りなし。新規購入は最大購入可能な100株単位とし、余剰現金を保持する",
                REPURCHASE_RULE,
                SAME_DAY_TRIGGER_RULE,
                "株式分割・併合日は、価格判定前に保有株数と基準価格を同じ比率で調整する",
                "欠損価格を補間せず、同一の未調整基準のOHLCが揃う取引日だけを使用",
            ],
            "disclaimer": (
                "このシミュレーションは、各購入時点で購入可能な最大の100株単位を用い、指定した利益確定率と損切り率を"
                "過去の株価データへ機械的に適用した仮想計算です。実際の取引結果や将来の利益を保証するものではなく、"
                "購入・売却の判断を示すものでもありません。"
            ),
            "cost_disclaimer": "売買手数料、税金、配当、スリッページ、市場流動性は計算に含めていません。",
            "cost_frequency_note": "取引回数が多い条件ほど、手数料等を含む実際の取引との差が大きくなる場合があります。",
        }
    )
    return result
