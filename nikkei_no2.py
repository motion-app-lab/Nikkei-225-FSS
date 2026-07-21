# ==============================================================
# nikkei_no22_zoom_daily_view.py
# 日経平均（日足）チャネル分析（直近60営業日拡大＋上余白3000）
# ==============================================================

import os, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from prophet import Prophet
from fredapi import Fred
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.random.seed(42)
tf.random.set_seed(42)

# --------------------------------------------------------------
# 補助関数群
# --------------------------------------------------------------

def classify_change(ch):
    if ch >= 5.0: return "急騰"
    if ch >= 2.0: return "上昇"
    if ch >= 0.5: return "やや上昇"
    if ch >= -0.5: return "やや下落"
    if ch >= -2.0: return "下落"
    return "急落"

def calc_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def channel_features(close, win=55, k=2.0):
    trend, upper, lower, pos = [], [], [], []
    for i in range(len(close)):
        if i < win:
            trend.append(np.nan); upper.append(np.nan)
            lower.append(np.nan); pos.append(np.nan)
            continue
        y = close.iloc[i - win:i]
        x = np.arange(win)
        b1 = np.cov(x, y)[0,1] / np.var(x)
        b0 = y.mean() - b1 * x.mean()
        resid = y - (b0 + b1 * x)
        sigma = resid.std()
        tr = b0 + b1 * (win - 1)
        up = tr + k * sigma
        lo = tr - k * sigma
        trend.append(tr); upper.append(up); lower.append(lo)
        pos.append((close.iloc[i] - lo) / (up - lo))
    return pd.DataFrame({"ch_trend":trend,"ch_upper":upper,"ch_lower":lower,"ch_pos":pos}, index=close.index)

def overheat_score(df):
    if "RSI14" not in df.columns:
        df["RSI14"] = calc_rsi(df["price"], 14)
    rsi = df["RSI14"]
    ch = df["ch_pos"]
    vola_ratio = df["price"].rolling(5).std() / df["price"].rolling(20).std()
    return (ch.clip(0,1)) * (rsi/100) * vola_ratio

def generate_comment(row, pred_label):
    ch, rsi, over = row["ch_pos"], row["RSI14"], row["overheat_score"]
    if ch > 0.95 and rsi > 68:
        return "上限タッチ直後のため反落注意。ただし短期反発余地あり。"
    elif ch > 0.9 and rsi < 65:
        return "上限圏ながら過熱感は限定的。上昇トレンド内の一服局面。"
    elif ch < 0.1 and rsi < 35:
        return "下限付近で売られすぎ圏。反発の兆し。"
    elif ch < 0.15 and rsi > 45:
        return "下限近くでRSI回復傾向。反発初動の可能性。"
    elif 0.3 <= ch <= 0.7:
        return "チャネル中央帯で方向感に乏しい。様子見が妥当。"
    elif over > 1.2:
        return "過熱スコア高く乱高下に注意。利益確定売り警戒。"
    elif pred_label in ["急騰","上昇"] and rsi < 70:
        return "過熱感の薄い健全な上昇局面。継続余地あり。"
    elif pred_label in ["下落","急落"] and ch > 0.9:
        return "上限圏からの下落サイン。調整入り初期。"
    else:
        return "明確な過熱・冷却サインなし。穏やかな推移。"

# --------------------------------------------------------------
# チャネル可視化（直近60営業日＋上余白3000）
# --------------------------------------------------------------

def plot_channel_chart(df, pred_label_next):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df.index, df["price"], color="skyblue", linewidth=1.8, label="Nikkei 225")
    ax.plot(df.index, df["ch_upper"], "m--", label="Channel Upper")
    ax.plot(df.index, df["ch_lower"], "m--", label="Channel Lower")
    ax.scatter(df.index[-1], df["price"].iloc[-1], color="red", s=80, label="Current Position")

    from matplotlib.dates import date2num
    last = df.iloc[-1]
    text = f"ch_pos={last['ch_pos']:.2f}\n{generate_comment(last, pred_label_next)}"
    ax.text(date2num(df.index[-1] + timedelta(days=3)), float(df["price"].iloc[-1]),
            text, fontsize=9, color="black", ha="left", va="bottom", transform=ax.transData)

    ax.legend(loc="upper left")
    ax.set_title("Nikkei 225 Daily Channel Trend (Zoomed - Last 60 Days)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)

    # ✅ 直近60営業日を拡大表示
    if len(df) > 60:
        ax.set_xlim(df.index[-60], df.index[-1] + timedelta(days=5))
    else:
        ax.set_xlim(df.index.min(), df.index.max() + timedelta(days=5))

    # ✅ 日付フォーマット（3日おき）
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.gcf().autofmt_xdate(rotation=45)

    # ✅ Y軸：最高値＋3000円
    ymin = int(df["price"].min() // 1000 * 1000)
    ymax = int(df["price"].max() // 1000 * 1000) + 3000
    y_ticks = np.arange(ymin, ymax + 2000, 2000)
    ax.set_yticks(y_ticks)
    ax.set_ylim(ymin - 500, ymax)

    plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.15)

    # 保存
    ts = datetime.today().strftime("%Y%m%d_%H%M%S")
    outdir = r"C:\week_yosoku"
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"n225_channel_zoom_{ts}.png")
    plt.savefig(outpath, dpi=150)
    plt.close("all")
    print(f"\n🖼️ チャネルグラフ（拡大版）を保存しました → {outpath}")
    try:
        os.startfile(outpath)
    except Exception:
        pass

# --------------------------------------------------------------
# メイン処理
# --------------------------------------------------------------

print("----------------------------------------------")
print("📈 日経平均（日足）5営業日予測＋直近60日拡大＋上余白3000")
print("----------------------------------------------")

fred_key = input("🔑 AIPパスキーを入力してください（FRED APIキー）: ").strip()
fred = Fred(api_key=fred_key)

end_date = datetime.today()
start_date = end_date - timedelta(days=365)

# --- 日経平均 ---
print("[INFO] 日経平均データ（日足）取得中...")
df_nk = yf.download("^N225", start=start_date, end=end_date, interval="1d", auto_adjust=True)
if isinstance(df_nk.columns, pd.MultiIndex):
    df_nk.columns = df_nk.columns.get_level_values(0)
df_nk["price"] = df_nk["Close"]

latest_close = yf.Ticker("^N225").history(period="1d")["Close"].iloc[-1]
df_nk.loc[df_nk.index[-1], "price"] = latest_close
print(f"[INFO] 最新終値を反映: {latest_close:.2f}")

df_nk["RSI14"] = calc_rsi(df_nk["price"], 14)
df_base = df_nk[["price","RSI14"]].copy()

# --- 外部ファクタ ---
macro_tickers = {
    "spx":"^GSPC","dow":"^DJI","nasdaq":"^IXIC","vix":"^VIX",
    "btc":"BTC-USD","oil":"CL=F","gold":"GC=F","usdjpy":"JPY=X",
    "sox":"^SOX","nvda":"NVDA"
}
print("[INFO] 外部ファクタ取得中...")
macro_df = pd.DataFrame(index=df_base.index)
for name,tkr in macro_tickers.items():
    tmp = yf.download(tkr, start=start_date, end=end_date, interval="1d", auto_adjust=True)
    if isinstance(tmp.columns, pd.MultiIndex):
        tmp.columns = tmp.columns.get_level_values(0)
    tmp = tmp["Close"].reindex(df_base.index).fillna(method="ffill")
    macro_df[name] = tmp
macro_df["pce"] = fred.get_series('PCEPILFE').reindex(df_base.index, method="ffill")

df = pd.concat([df_base, macro_df], axis=1).fillna(method="ffill")

# 特徴量
for col in list(macro_tickers.keys()) + ["pce"]:
    df[f"{col}_ret"] = df[col].pct_change() * 100
ch = channel_features(df["price"], 55, 2.0)
df = df.join(ch)
df["overheat_score"] = overheat_score(df)
df["Target_Change"] = df["price"].pct_change(5).shift(-5) * 100
df = df.dropna()

# Prophet
print("[INFO] Prophet分析中...")
prophet_df = pd.DataFrame({"ds": df.index, "y": df["price"].values})
model_prophet = Prophet()
model_prophet.fit(prophet_df)
forecast = model_prophet.predict(prophet_df)
df["prophet_change"] = (forecast["yhat"].values - df["price"].values) / df["price"].values * 100

# CatBoost
feat_cols = [c for c in df.columns if c not in ["Target_Change","prophet_change"]]
X = df[feat_cols].values
y_labels = np.array([classify_change(v) for v in df["Target_Change"]])
labels = ["急騰","上昇","やや上昇","やや下落","下落","急落"]
y_idx = np.array([labels.index(v) for v in y_labels])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cat = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05, verbose=False)
cat.fit(X_scaled, y_idx)
proba = cat.predict_proba(X_scaled)
pred_idx = np.argmax(proba, axis=1)
acc = accuracy_score(y_idx, pred_idx)

# 結果
print(f"\n[RESULT] 精度評価: {acc*100:.2f}%")
print("----------------------------------------------")
print("【5営業日後の傾向予測】")
print("----------------------------------------------")
for cls, p in zip(labels, proba[-1]):
    print(f"{cls}: {p*100:.2f}%")

pred_label_next = labels[np.argmax(proba[-1])]
if pred_label_next in ["急騰","上昇"]:
    action = "→ 今後5営業日は【買い傾向】を予測"
elif pred_label_next in ["下落","急落"]:
    action = "→ 今後5営業日は【売り傾向】を予測"
else:
    action = "→ 今後5営業日は【様子見】を推奨"

print(f"\n予測クラス: {pred_label_next}　{action}")
print(f"\n💬 【AIコメント】{generate_comment(df.iloc[-1], pred_label_next)}")

plot_channel_chart(df, pred_label_next)
print("----------------------------------------------")
