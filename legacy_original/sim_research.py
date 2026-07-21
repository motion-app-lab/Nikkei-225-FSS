# -*- coding: utf-8 -*-
"""
sim_research_realistic.py
=========================
5営業日予測に基づく実売型AIシミュレーション（現物取引のみ）
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import timedelta
import warnings

warnings.filterwarnings("ignore")


# -----------------------------
# RSI と 最大ドローダウン
# -----------------------------
def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.clip(0, 100)

def max_drawdown(equity):
    roll_max = equity.cummax()
    dd = equity / roll_max - 1
    return float(dd.min() * 100) if len(dd) else 0.0


# -----------------------------
# モデル：5営業日先の上昇確率を予測
# -----------------------------
def make_predictions(df):
    df["Ret1"] = df["Close"].pct_change()
    df["RSI"] = rsi(df["Close"])
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["VolChg"] = df["Volume"].pct_change()

    # 5営業日後の上昇をターゲットに
    df["Target"] = (df["Close"].shift(-5) > df["Close"]).astype(int)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    features = ["Ret1", "RSI", "MA5", "MA20", "VolChg"]
    X = df[features].astype(np.float32)
    y = df["Target"]

    if len(X) < 100:
        raise ValueError("データが少なすぎます。期間を延ばすか他の銘柄を試してください。")

    split = int(len(df) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    p_up = model.predict_proba(X_test)[:, 1]
    df["PredSignal"] = np.nan
    df.loc[X_test.index, "PredSignal"] = np.where(p_up >= 0.5, 1, 0)
    df["PredSignal"] = df["PredSignal"].fillna(0)
    return df


# -----------------------------
# 売買シミュレーション（現物買いのみ）
# -----------------------------
def simulate(df, tp, sl, buy_on_up=True):
    cash = 1_000_000
    pos = 0
    entry_px = 0.0
    entry_day = None
    equity_curve = []
    trades = []

    tp_active = isinstance(tp, (int, float)) and tp > 0
    sl_active = isinstance(sl, (int, float)) and sl > 0

    dates = df.index
    prices = df["Close"].values
    signals = df["PredSignal"].values

    for i in range(1, len(dates)):  # 翌日以降にシグナルを反映
        px = float(prices[i])
        sig = int(signals[i - 1])  # 1日遅れで反映（翌日売買）

        # ポジション保有中
        if pos == 1:
            pnl_pct = float((px / entry_px - 1.0) * 100.0)
            held_days = (dates[i] - entry_day).days

            # 利確・損切・5営業日経過で決済
            if (tp_active and pnl_pct >= tp) or (sl_active and pnl_pct <= -sl) or held_days >= 7:
                cash *= (1.0 + pnl_pct / 100.0)
                trades.append(pnl_pct)
                pos = 0
                entry_px = 0.0
                entry_day = None

        # ポジションなし：買いシグナルが出たら購入
        elif pos == 0 and buy_on_up and sig == 1:
            pos = 1
            entry_px = px
            entry_day = dates[i]

        # 評価資産
        eq = cash if pos == 0 else cash * (1.0 + (px / entry_px - 1.0))
        equity_curve.append(eq)

    # 未決済ポジションの最終評価
    if pos == 1:
        pnl_pct = float((prices[-1] / entry_px - 1.0) * 100.0)
        cash *= (1.0 + pnl_pct / 100.0)
        trades.append(pnl_pct)

    # 結果
    equity = pd.Series(equity_curve, index=dates[-len(equity_curve):])
    total_profit = cash - 1_000_000
    win_rate = np.mean([t > 0 for t in trades]) * 100.0 if trades else 0.0
    dd = max_drawdown(equity)

    print("\n========== 結果 ==========")
    print(f"勝率: {win_rate:.2f}%　最大DD: {dd:.2f}%")
    print(f"資金1,000,000　総資産{cash:,.0f}")
    print(f"利益{total_profit:,.0f}")


# -----------------------------
# メイン処理
# -----------------------------
def main():
    print("===================================")
    print("📈 実売型AIシミュレーション（5営業日予測・現物取引）")
    print("===================================\n")

    ticker = input("ティッカーコード（例: 8035.T）: ").strip() or "8035.T"

    tp_in = input("利確(％)：1~50、off → ").strip().lower()
    sl_in = input("損切(％)：1~50、off → ").strip().lower()
    buy_in = input("予測システム上昇で買い：on,off → ").strip().lower()

    tp = float(tp_in) if tp_in not in ("off", "") else 0
    sl = float(sl_in) if sl_in not in ("off", "") else 0
    buy_on_up = buy_in != "off"

    print("\n[INFO] 株価データ取得中...")
    df = yf.download(ticker, start="2015-01-01", progress=False)
    if df.empty:
        print("❌ データ取得に失敗しました。ティッカーを確認してください。")
        return

    df = df.rename(columns=str.capitalize)
    try:
        df = make_predictions(df)
    except ValueError as e:
        print(f"❌ モデル学習エラー: {e}")
        return

    print("\n[INFO] 売買シミュレーション実行中...")
    simulate(df, tp, sl, buy_on_up)


if __name__ == "__main__":
    main()
