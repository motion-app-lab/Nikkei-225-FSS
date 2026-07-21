# ==============================================================
# nikkei_kobetu_no1_named_20d_realtime_v2.py
# 複数銘柄対応：5営業日予測＋直近20日チャネル＋外部ファクタ
# ・当日価格（分足 or 気配）反映
# ・matplotlibバックエンドを非GUI化（複数銘柄でも安定動作）
# ==============================================================

import os, warnings, sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import yfinance as yf
from prophet import Prophet
from fredapi import Fred
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# GUI非依存バックエンドを使用（複数銘柄描画時のTk競合回避）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.random.seed(42)

# ========= 設定 =========
PRED_HORIZON = 5
CHAN_WIN = 55
CHAN_K = 2.0
ZOOM_BARS = 20
CAT_ITER = 320
CAT_DEPTH = 5
CAT_LR = 0.05
OUT_DIR = r"C:\week_yosoku"

CLASS_ORDER = ["急騰", "上昇", "やや上昇", "やや下落", "下落", "急落"]

# ========= ユーティリティ =========
def classify_change(ch):
    if ch >= 5.0: return "急騰"
    if ch >= 2.0: return "上昇"
    if ch >= 0.5: return "やや上昇"
    if ch >= -0.5: return "やや下落"
    if ch >= -2.0: return "下落"
    return "急落"

def calc_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def channel_features(close, win=CHAN_WIN, k=CHAN_K):
    trend, upper, lower, pos = [], [], [], []
    for i in range(len(close)):
        if i < win:
            trend.extend([np.nan]); upper.extend([np.nan]); lower.extend([np.nan]); pos.extend([np.nan]); continue
        y = close.iloc[i-win:i]
        x = np.arange(win)
        b1 = np.cov(x, y)[0,1] / np.var(x)
        b0 = y.mean() - b1 * x.mean()
        resid = y - (b0 + b1*x)
        sigma = resid.std()
        tr = b0 + b1*(win-1)
        up = tr + k*sigma
        lo = tr - k*sigma
        width = max(up-lo, 1e-9)
        trend.append(tr); upper.append(up); lower.append(lo); pos.append((close.iloc[i]-lo)/width)
    return pd.DataFrame({"ch_trend":trend,"ch_upper":upper,"ch_lower":lower,"ch_pos":pos}, index=close.index)

def overheat_score(df):
    if "RSI14" not in df.columns:
        df["RSI14"] = calc_rsi(df["price"], 14)
    rsi = df["RSI14"]
    ch = df["ch_pos"].clip(0,1)
    vola_ratio = df["price"].rolling(5).std() / df["price"].rolling(20).std()
    return ch * (rsi/100) * vola_ratio

def generate_comment(row, pred_label):
    ch, rsi, over = row["ch_pos"], row.get("RSI14", np.nan), row.get("overheat_score", np.nan)
    if ch > 0.95 and rsi > 68: return "上限タッチ直後で反落注意。ただし短期反発余地も。"
    if ch > 0.9 and rsi < 65:  return "上限圏だが過熱感は限定。トレンド内の一服。"
    if ch < 0.1 and rsi < 35:  return "下限付近の売られ過ぎ。反発の芽。"
    if ch < 0.15 and rsi > 45: return "下限近くでRSI回復。反発初動の可能性。"
    if 0.3 <= ch <= 0.7:       return "チャネル中央帯で方向感乏しい。様子見。"
    if over > 1.2:             return "過熱スコア高く乱高下に注意。利益確定売り警戒。"
    if pred_label in ["急騰","上昇"] and rsi < 70: return "過熱感の薄い健全な上昇局面。継続余地。"
    if pred_label in ["下落","急落"] and ch > 0.9: return "上限圏からの下落サイン。調整初期。"
    return "明確な過熱・冷却サインなし。穏やかな推移。"

def get_display_name(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).get_info()
        for k in ("shortName","longName","name"):
            v = info.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    except Exception:
        pass
    return ticker

def normalize_to_tokyo_index(idx_like):
    idx = pd.to_datetime(idx_like)
    try:
        idx = idx.tz_localize("UTC").tz_convert("Asia/Tokyo").tz_localize(None)
    except Exception:
        idx = pd.to_datetime(idx).tz_localize(None)
    return idx

# ========= 当日価格反映 =========
def fetch_runtime_price(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        m = tk.history(period="2d", interval="1m", auto_adjust=True)
        if m is not None and not m.empty:
            m.index = normalize_to_tokyo_index(m.index)
            last_ts = m.index[-1]
            last_px = float(m["Close"].iloc[-1])
            if np.isfinite(last_px):
                return last_ts, last_px
        fp = getattr(tk, "fast_info", None)
        if fp:
            lp = getattr(fp, "last_price", None)
            if lp is not None and np.isfinite(lp):
                return normalize_to_tokyo_index([datetime.utcnow()])[0], float(lp)
        inf = tk.get_info()
        rmp = inf.get("regularMarketPrice", None) if isinstance(inf, dict) else None
        if rmp is not None and np.isfinite(rmp):
            return normalize_to_tokyo_index([datetime.utcnow()])[0], float(rmp)
    except Exception:
        pass
    return None, None

def inject_runtime_price(base: pd.DataFrame, ticker: str) -> pd.DataFrame:
    ts, px = fetch_runtime_price(ticker)
    if ts is None or px is None:
        return base
    day = pd.Timestamp(ts.date())
    if day in base.index:
        base.loc[day, "price"] = float(px)
    elif day > base.index[-1]:
        new = pd.DataFrame(index=pd.DatetimeIndex([day]))
        new["price"] = float(px)
        base = pd.concat([base, new]).sort_index()
    return base

def safe_fetch_close(ticker, start, end, index_like):
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True)
        if isinstance(df, pd.DataFrame) and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = normalize_to_tokyo_index(df.index)
            ser = df["Close"]
            return ser.reindex(index_like).ffill()
    except Exception:
        pass
    return pd.Series(index=index_like, dtype="float64")

# ========= チャート描画 =========
def plot_channel_chart(df, display_name, ticker, save_prefix, proba_row=None, label_next=None):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df.index, df["price"], color="skyblue", linewidth=1.8, label=f"{display_name} ({ticker})")
    ax.plot(df.index, df["ch_upper"], "m--", label="Channel Upper")
    ax.plot(df.index, df["ch_lower"], "m--", label="Channel Lower")
    ax.scatter(df.index[-1], df["price"].iloc[-1], color="red", s=80, label="Current")

    right_pad_days = 5
    if len(df) > ZOOM_BARS:
        ax.set_xlim(df.index[-ZOOM_BARS], df.index[-1] + timedelta(days=right_pad_days))
    else:
        ax.set_xlim(df.index.min(), df.index.max() + timedelta(days=right_pad_days))

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.gcf().autofmt_xdate(rotation=45)

    ymin = float(df["price"].min())
    ymax = float(df["price"].max())
    ax.set_ylim(ymin - (ymax-ymin)*0.02, ymax * 1.03)

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    ax.set_title(f"{display_name} ({ticker}) – Channel (Last {ZOOM_BARS} days)", fontsize=12)

    last = df.iloc[-1]
    comment = generate_comment(last, label_next if label_next else "")
    lines = [f"ch_pos={last['ch_pos']:.2f}", comment]
    if proba_row is not None:
        lines.append(" | ".join([f"{lab}:{p*100:.1f}%" for lab,p in zip(CLASS_ORDER, proba_row)]))
    ax.text(df.index[-1] + timedelta(days=2), float(df["price"].iloc[-1]),
            "\n".join(lines), fontsize=9, color="black", ha="left", va="bottom")

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.today().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_DIR, f"{save_prefix}_channel_{ts}.png")
    plt.savefig(path, dpi=150)
    plt.close("all")
    print(f"🖼️ チャネルグラフ保存: {path}")

# ========= データ構築・予測 =========
def build_feature_frame(ticker, fred, start_date, end_date):
    print(f"\n[INFO] {ticker} のデータ取得…")
    base = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True)
    if base.empty: raise RuntimeError(f"データ取得失敗: {ticker}")
    if isinstance(base.columns, pd.MultiIndex):
        base.columns = base.columns.get_level_values(0)
    base.index = normalize_to_tokyo_index(base.index)
    base["price"] = base["Close"]

    base = inject_runtime_price(base, ticker)
    base["RSI14"] = calc_rsi(base["price"], 14)

    macro_ticks = {
        "spx":"^GSPC","dow":"^DJI","nasdaq":"^IXIC","vix":"^VIX",
        "btc":"BTC-USD","oil":"CL=F","gold":"GC=F","usdjpy":"JPY=X",
        "sox":"^SOX","nvda":"NVDA"
    }
    macro = pd.DataFrame(index=base.index)
    for name, tk in macro_ticks.items():
        macro[name] = safe_fetch_close(tk, start_date, end_date, base.index)

    try:
        pce = fred.get_series("PCEPILFE")
        pce.index = pd.to_datetime(pce.index)
        macro["pce"] = pce.reindex(base.index, method="ffill")
    except Exception:
        macro["pce"] = pd.Series(index=base.index, dtype="float64")

    df = pd.concat([base, macro], axis=1).ffill()
    factor_cols = [c for c in df.columns if c not in ["price","RSI14"]]
    for c in factor_cols:
        df[f"{c}_ret"] = df[c].pct_change()*100

    ch = channel_features(df["price"])
    df = df.join(ch)
    df["overheat_score"] = overheat_score(df)
    df["Target_Change"] = df["price"].pct_change(PRED_HORIZON).shift(-PRED_HORIZON)*100
    df = df.dropna()
    return df

def fit_predict_one(df, ticker, display_name=None):
    display_name = display_name or get_display_name(ticker)
    print("[INFO] Prophet分析…")
    p_df = pd.DataFrame({"ds": df.index, "y": df["price"].values})
    p_model = Prophet()
    p_model.fit(p_df)
    fc = p_model.predict(p_df)
    df["prophet_change"] = (fc["yhat"].values - df["price"].values)/df["price"].values*100

    feat_cols = [c for c in df.columns if c not in ["Target_Change","prophet_change"]]
    X = df[feat_cols].values
    y = np.array([CLASS_ORDER.index(classify_change(v)) for v in df["Target_Change"].values])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = CatBoostClassifier(iterations=CAT_ITER, depth=CAT_DEPTH, learning_rate=CAT_LR, verbose=False, loss_function="MultiClass")
    model.fit(Xs, y)
    proba = model.predict_proba(Xs)
    pred_idx = np.argmax(proba, axis=1)

    acc6 = accuracy_score(y, pred_idx)
    updown_true = np.array([1 if CLASS_ORDER[i] in ["急騰","上昇","やや上昇"] else 0 for i in y])
    updown_pred = np.array([1 if CLASS_ORDER[i] in ["急騰","上昇","やや上昇"] else 0 for i in pred_idx])
    acc2 = accuracy_score(updown_true, updown_pred)

    print(f"\n[RESULT] 精度評価（6段階）: {acc6*100:.2f}%")
    print(f"[RESULT] 方向精度（上昇系/下落系）: {acc2*100:.2f}%\n")
    print("【5営業日後の傾向予測】")
    for cls, p in zip(CLASS_ORDER, proba[-1]):
        print(f"{cls}: {p*100:.2f}%")

    label_next = CLASS_ORDER[np.argmax(proba[-1])]
    action = "→ 今後5営業日は【買い傾向】" if label_next in ["急騰","上昇"] \
             else ("→ 今後5営業日は【売り傾向】" if label_next in ["下落","急落"] else "→ 今後5営業日は【様子見】")
    print(f"\n予測クラス: {label_next}　{action}")
    print(f"💬 コメント: {generate_comment(df.iloc[-1], label_next)}")

    plot_channel_chart(df, display_name, ticker, ticker.replace("^","").replace(".","_"),
                       proba_row=proba[-1], label_next=label_next)

# ========= メイン =========
if __name__ == "__main__":
    print("----------------------------------------------")
    print("📈 個別銘柄：5営業日予測＋直近20日チャネル＋外部ファクタ（当日価格反映）")
    print("----------------------------------------------")
    fred_key = input("🔑 AIPパスキーを入力してください（FRED APIキー）: ").strip()
    fred = Fred(api_key=fred_key)

    raw = input("🎯 予測したいティッカーをカンマ区切りで入力（例: 7203.T, 6758.T, AAPL, NVDA）: ").strip()
    tickers = [t.strip() for t in raw.split(",") if t.strip()]
    if not tickers:
        print("ティッカーが入力されていません。終了します。")
        sys.exit(0)

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)

    for tk in tickers:
        try:
            df = build_feature_frame(tk, fred, start_date, end_date)
            fit_predict_one(df, tk)
            print("----------------------------------------------")
        except Exception as e:
            print(f"[ERROR] {tk}: {e}")
            print("----------------------------------------------")
