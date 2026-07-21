from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(BASE_DIR / ".env")

PREDICTION_HORIZON = 5
CHANNEL_WINDOW = 55
CHANNEL_K = 2.0
CLASS_ORDER = ["急騰", "上昇", "やや上昇", "やや下落", "下落", "急落"]
UP_CLASS_IDS = {0, 1, 2}

MACRO_TICKERS = {
    "spx": "^GSPC",
    "dow": "^DJI",
    "nasdaq": "^IXIC",
    "vix": "^VIX",
    "btc": "BTC-USD",
    "oil": "CL=F",
    "gold": "GC=F",
    "usdjpy": "JPY=X",
    "sox": "^SOX",
    "nvda": "NVDA",
}


class ServiceError(RuntimeError):
    """画面へ安全に返せる業務エラー。"""

    def __init__(self, message: str, action: str = "時間をおいて再度お試しください。", status_code: int = 422):
        super().__init__(message)
        self.message = message
        self.action = action
        self.status_code = status_code


@dataclass
class PredictionData:
    frame: pd.DataFrame
    feature_columns: list[str]
    training_frame: pd.DataFrame
    inference_row: pd.DataFrame
    warnings: list[str]
    fetched_at: str


def classify_change(change_percent: float) -> str:
    """既存CMD版の6段階境界を維持する。"""
    if change_percent >= 5.0:
        return "急騰"
    if change_percent >= 2.0:
        return "上昇"
    if change_percent >= 0.5:
        return "やや上昇"
    if change_percent >= -0.5:
        return "やや下落"
    if change_percent >= -2.0:
        return "下落"
    return "急落"


def trend_for_class(label: str) -> tuple[str, str]:
    if label in {"急騰", "上昇", "やや上昇"}:
        return "上昇傾向", "up"
    if label == "やや下落":
        return "中立傾向（弱い下落を含む）", "neutral"
    return "下落傾向", "down"


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.astype(float).diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    average_up = up.rolling(period).mean()
    average_down = down.rolling(period).mean()
    relative_strength = average_up / average_down.replace(0, np.nan)
    result = 100 - (100 / (1 + relative_strength))
    return result.where(average_down.ne(0), 100.0).clip(0, 100)


def channel_features(close: pd.Series, window: int = CHANNEL_WINDOW, k: float = CHANNEL_K) -> pd.DataFrame:
    trend: list[float] = []
    upper: list[float] = []
    lower: list[float] = []
    position: list[float] = []
    values = close.astype(float)

    for index in range(len(values)):
        if index < window:
            trend.append(np.nan)
            upper.append(np.nan)
            lower.append(np.nan)
            position.append(np.nan)
            continue
        history = values.iloc[index - window:index]
        x = np.arange(window, dtype=float)
        slope = float(np.cov(x, history)[0, 1] / np.var(x))
        intercept = float(history.mean() - slope * x.mean())
        residual = history - (intercept + slope * x)
        sigma = float(residual.std())
        current_trend = intercept + slope * (window - 1)
        current_upper = current_trend + k * sigma
        current_lower = current_trend - k * sigma
        width = max(current_upper - current_lower, 1e-9)
        trend.append(current_trend)
        upper.append(current_upper)
        lower.append(current_lower)
        position.append((float(values.iloc[index]) - current_lower) / width)

    return pd.DataFrame(
        {"ch_trend": trend, "ch_upper": upper, "ch_lower": lower, "ch_pos": position},
        index=values.index,
    )


def overheat_score(frame: pd.DataFrame) -> pd.Series:
    volatility_ratio = frame["price"].rolling(5).std() / frame["price"].rolling(20).std().replace(0, np.nan)
    return frame["ch_pos"].clip(0, 1) * (frame["rsi14"] / 100) * volatility_ratio


def generate_analysis_comment(row: pd.Series, prediction_label: str) -> str:
    channel = float(row["ch_pos"])
    rsi = float(row["rsi14"])
    overheat = float(row["overheat_score"])
    if channel > 0.95 and rsi > 68:
        return "チャネル上限付近かつRSIが高く、短期的な過熱と反動に注意が必要です。"
    if channel > 0.9 and rsi < 65:
        return "チャネル上限付近ですが、RSIの過熱感は限定的です。トレンド継続性を確認してください。"
    if channel < 0.1 and rsi < 35:
        return "チャネル下限付近でRSIも低く、売られ過ぎとその反動の両方に注意が必要です。"
    if channel < 0.15 and rsi > 45:
        return "チャネル下限付近でRSIは持ち直しています。反発初動の可能性を観察する局面です。"
    if 0.3 <= channel <= 0.7:
        return "チャネル中央帯にあり、方向感は限定的です。予測確率の偏りと外部環境を併せて確認してください。"
    if overheat > 1.2:
        return "過熱スコアが高く、値動きの拡大に注意が必要です。"
    if prediction_label in {"急騰", "上昇", "やや上昇"} and rsi < 70:
        return "モデルは上向きのクラスを最上位としていますが、確率分布を含めて不確実性をご確認ください。"
    if prediction_label in {"下落", "急落"} and channel > 0.9:
        return "チャネル上限圏で下向きクラスが最上位です。短期的な調整リスクを示す組み合わせです。"
    return "明確な過熱・冷却シグナルは限定的です。単独の予測ではなく複数の情報と併用してください。"


def normalize_ticker(ticker: str, append_tokyo_suffix: bool = True) -> str:
    normalized = (ticker or "").strip().upper()
    if append_tokyo_suffix and re.fullmatch(r"\d{4}", normalized):
        normalized = f"{normalized}.T"
    if not normalized or len(normalized) > 20 or not re.fullmatch(r"[A-Z0-9.^=\-]+", normalized):
        raise ServiceError(
            "ティッカーの形式を確認できませんでした。",
            "7203.T、6758.T、AAPL、NVDAのように入力してください。",
        )
    return normalized


def normalize_datetime_index(index: Iterable[Any]) -> pd.DatetimeIndex:
    result = pd.to_datetime(index, errors="coerce")
    if getattr(result, "tz", None) is not None:
        result = result.tz_convert("Asia/Tokyo").tz_localize(None)
    return pd.DatetimeIndex(result).normalize()


def extract_yfinance_column(data: pd.DataFrame, field: str, ticker: str | None = None) -> pd.Series:
    """yfinanceの通常列・MultiIndex列の双方から1列を取り出す。"""
    if data is None or data.empty:
        return pd.Series(dtype="float64")

    if isinstance(data.columns, pd.MultiIndex):
        for level in range(data.columns.nlevels):
            values = data.columns.get_level_values(level)
            matches = [value for value in values.unique() if str(value).lower() == field.lower()]
            if not matches:
                continue
            selected = data.xs(matches[0], axis=1, level=level, drop_level=True)
            if isinstance(selected, pd.Series):
                return selected.astype(float)
            if ticker:
                for column in selected.columns:
                    if str(column).upper() == ticker.upper():
                        return selected[column].astype(float)
            if selected.shape[1] == 1:
                return selected.iloc[:, 0].astype(float)
            return pd.Series(dtype="float64")

    for column in data.columns:
        if str(column).lower() == field.lower():
            return data[column].astype(float)
    return pd.Series(dtype="float64")


def _download_base_frame(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    try:
        import yfinance as yf

        raw = yf.download(
            ticker,
            start=start,
            end=end + timedelta(days=1),
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=20,
        )
    except Exception as exc:
        raise ServiceError(
            "市場データの取得中に通信エラーが発生しました。",
            "インターネット接続を確認し、時間をおいて再度実行してください。",
            503,
        ) from exc

    if raw is None or raw.empty:
        raise ServiceError(
            f"{ticker} の株価データを取得できませんでした。",
            "ティッカーが正しいか確認してください。日本株は7203.Tのように入力します。",
        )

    close = extract_yfinance_column(raw, "Close", ticker)
    if close.empty:
        raise ServiceError("終値データを確認できませんでした。", "ティッカーを確認して再度実行してください。")

    frame = pd.DataFrame(index=normalize_datetime_index(close.index))
    for source, destination in (("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "price"), ("Volume", "volume")):
        series = extract_yfinance_column(raw, source, ticker)
        if not series.empty:
            series.index = normalize_datetime_index(series.index)
            frame[destination] = series.groupby(level=0).last().reindex(frame.index)

    if "price" not in frame:
        raise ServiceError("終値データを確認できませんでした。")
    for column in ("open", "high", "low"):
        if column not in frame:
            frame[column] = frame["price"]
    if "volume" not in frame:
        frame["volume"] = 0.0

    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["price"])
    if len(frame) < 180:
        raise ServiceError(
            "モデル学習に必要な株価履歴が不足しています。",
            "十分な取引履歴があるティッカーを指定してください。",
        )
    return frame


def _download_macro_frame(index: pd.DatetimeIndex, start: datetime, end: datetime) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    macro = pd.DataFrame(index=index)
    try:
        import yfinance as yf

        raw = yf.download(
            list(MACRO_TICKERS.values()),
            start=start,
            end=end + timedelta(days=1),
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column",
            timeout=20,
        )
    except Exception:
        raw = pd.DataFrame()
        warnings.append("外部市場指標を取得できなかったため、株価固有の特徴量のみで分析しました。")

    for name, ticker in MACRO_TICKERS.items():
        series = extract_yfinance_column(raw, "Close", ticker)
        if series.empty:
            warnings.append(f"外部指標 {name} を取得できなかったため除外しました。")
            continue
        series.index = normalize_datetime_index(series.index)
        series = series.groupby(level=0).last().sort_index()
        aligned = series.reindex(index).ffill()
        if aligned.notna().sum() >= 100:
            macro[name] = aligned
        else:
            warnings.append(f"外部指標 {name} の履歴が不足したため除外しました。")

    return macro, warnings


def _add_pce(index: pd.DatetimeIndex, macro: pd.DataFrame, warnings: list[str]) -> None:
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        warnings.append("FRED APIキーが未設定のためPCEを除外しました。")
        return
    try:
        from fredapi import Fred

        pce = Fred(api_key=api_key).get_series("PCEPILFE")
        pce.index = normalize_datetime_index(pce.index)
        # PCEは観測月の後に公表されるため、観測月の日付へそのまま結合すると
        # 当時未公表だった値を使うことになる。保守的な45日ラグを置く。
        pce.index = pce.index + pd.Timedelta(days=45)
        pce = pce.groupby(level=0).last().sort_index()
        aligned = pce.reindex(index, method="ffill")
        if aligned.notna().sum() < 100:
            raise ValueError("PCE history is too short")
        macro["pce"] = aligned
    except Exception:
        warnings.append("FRED APIエラーのためPCEを除外して処理を継続しました。APIキーと通信状態をご確認ください。")


def prepare_prediction_data(
    ticker: str,
    years: int = 5,
    lag_external_one_session: bool = False,
) -> PredictionData:
    """正解がある過去行と、最新推論行を分離して特徴量を生成する。"""
    end = datetime.now()
    start = end - timedelta(days=365 * years + 90)
    base = _download_base_frame(ticker, start, end)
    macro, warnings = _download_macro_frame(base.index, start, end)
    _add_pce(base.index, macro, warnings)

    if lag_external_one_session and not macro.empty:
        # 東京市場の当日終値より後に確定する米国市場・商品等の日次終値を
        # 同じ日の日経平均予測へ使わないよう、全外部系列を1取引日遅らせる。
        macro = macro.shift(1)

    frame = base.join(macro, how="left")
    macro_columns = list(macro.columns)
    if macro_columns:
        frame[macro_columns] = frame[macro_columns].ffill()

    frame["return_1d"] = frame["price"].pct_change() * 100
    frame["return_5d"] = frame["price"].pct_change(5) * 100
    frame["volatility_5d"] = frame["return_1d"].rolling(5).std()
    frame["volatility_20d"] = frame["return_1d"].rolling(20).std()
    frame["volume_change"] = frame["volume"].pct_change() * 100
    frame["rsi14"] = calc_rsi(frame["price"], 14)
    frame = frame.join(channel_features(frame["price"]))
    frame["overheat_score"] = overheat_score(frame)

    for column in macro_columns:
        frame[f"{column}_ret"] = frame[column].pct_change() * 100

    # future_priceが存在しない最新5取引日は正解ラベルを持たせない。
    future_price = frame["price"].shift(-PREDICTION_HORIZON)
    frame["target_change"] = np.where(
        future_price.notna(),
        (future_price / frame["price"] - 1.0) * 100,
        np.nan,
    )

    feature_columns = [
        "open",
        "high",
        "low",
        "price",
        "volume",
        "return_1d",
        "return_5d",
        "volatility_5d",
        "volatility_20d",
        "volume_change",
        "rsi14",
        "ch_trend",
        "ch_upper",
        "ch_lower",
        "ch_pos",
        "overheat_score",
    ]
    for column in macro_columns:
        feature_columns.extend([column, f"{column}_ret"])

    frame = frame.replace([np.inf, -np.inf], np.nan)
    latest_market_date = frame.index[-1]
    latest_features = frame.loc[latest_market_date, feature_columns]
    missing = latest_features[latest_features.isna()].index.tolist()
    if missing:
        complete = frame[feature_columns].dropna()
        if complete.empty:
            raise ServiceError(
                "最新特徴量を生成できませんでした。",
                f"欠損している特徴量: {', '.join(missing)}。十分な履歴があるティッカーで再度お試しください。",
            )
        inference_date = complete.index[-1]
        warnings.append(
            f"最新取引日の特徴量（{', '.join(missing)}）が欠損したため、利用可能な最新行 {inference_date:%Y-%m-%d} を基準日に使用しました。"
        )
    else:
        inference_date = latest_market_date

    training_frame = frame.loc[:, feature_columns + ["target_change"]].dropna()
    inference_row = frame.loc[[inference_date], feature_columns]
    # 欠損により推論基準日が過去へ戻った場合でも、その行自身を学習へ含めない。
    training_frame = training_frame.drop(index=inference_date, errors="ignore")
    if len(training_frame) < 160:
        raise ServiceError(
            "時系列検証に必要なデータが不足しています。",
            "より長い履歴があるティッカーを選ぶか、外部データ取得後に再度お試しください。",
        )

    fetched_at = datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")
    return PredictionData(frame, feature_columns, training_frame, inference_row, warnings, fetched_at)


def chronological_split(frame: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        raise ServiceError("検証対象データがありません。")
    split_index = int(len(frame) * train_ratio)
    split_index = max(1, min(split_index, len(frame) - 1))
    return frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy()


def _label_ids(changes: pd.Series) -> np.ndarray:
    return np.array([CLASS_ORDER.index(classify_change(float(value))) for value in changes], dtype=int)


def _new_catboost_model():
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ServiceError(
            "CatBoostを読み込めませんでした。",
            "setup_windows.batを実行して依存関係をインストールしてください。",
            500,
        ) from exc
    return CatBoostClassifier(
        iterations=220,
        depth=5,
        learning_rate=0.05,
        loss_function="MultiClass",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )


def _six_probabilities(model: Any, probability_row: np.ndarray) -> list[float]:
    probabilities = [0.0] * len(CLASS_ORDER)
    classes = np.asarray(model.classes_).reshape(-1)
    for model_column, class_id in enumerate(classes):
        normalized_id = int(class_id)
        if 0 <= normalized_id < len(probabilities):
            probabilities[normalized_id] = float(probability_row[model_column])
    return probabilities


def fit_validate_and_predict(data: PredictionData) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    train_part, validation_part = chronological_split(data.training_frame)
    y_train = _label_ids(train_part["target_change"])
    y_validation = _label_ids(validation_part["target_change"])
    if len(np.unique(y_train)) < 2:
        raise ServiceError(
            "学習期間に必要な予測クラスが不足しています。",
            "別のティッカーを選ぶか、データが蓄積してから再度お試しください。",
        )

    validation_scaler = StandardScaler()
    x_train = validation_scaler.fit_transform(train_part[data.feature_columns])
    x_validation = validation_scaler.transform(validation_part[data.feature_columns])
    validation_model = _new_catboost_model()
    validation_model.fit(x_train, y_train)
    validation_prediction = validation_model.predict(x_validation).astype(int).reshape(-1)
    six_class_accuracy = float(accuracy_score(y_validation, validation_prediction))
    direction_true = np.isin(y_validation, list(UP_CLASS_IDS)).astype(int)
    direction_prediction = np.isin(validation_prediction, list(UP_CLASS_IDS)).astype(int)
    direction_accuracy = float(accuracy_score(direction_true, direction_prediction))

    # 現在予測用モデルは、正解が判明した全履歴だけで再学習する。
    full_y = _label_ids(data.training_frame["target_change"])
    if len(np.unique(full_y)) < 2:
        raise ServiceError("モデル学習に必要な予測クラスが不足しています。")
    inference_scaler = StandardScaler()
    full_x = inference_scaler.fit_transform(data.training_frame[data.feature_columns])
    latest_x = inference_scaler.transform(data.inference_row[data.feature_columns])
    inference_model = _new_catboost_model()
    inference_model.fit(full_x, full_y)
    raw_probabilities = inference_model.predict_proba(latest_x)[0]
    probabilities = _six_probabilities(inference_model, raw_probabilities)
    predicted_id = int(np.argmax(probabilities))

    return {
        "prediction_label": CLASS_ORDER[predicted_id],
        "probabilities": probabilities,
        "top_probability": float(probabilities[predicted_id]),
        "six_class_accuracy": six_class_accuracy,
        "direction_accuracy": direction_accuracy,
        "training_samples": int(len(train_part)),
        "validation_samples": int(len(validation_part)),
    }


def get_ticker_name(ticker: str) -> str:
    if ticker == "^N225":
        return "日経平均株価"
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).get_info()
        if isinstance(info, dict):
            for key in ("shortName", "longName", "displayName"):
                value = info.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    except Exception:
        pass
    return ticker


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return cleaned or "chart"


def plot_channel_chart(frame: pd.DataFrame, ticker: str, display_name: str, bars: int) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    view = frame.dropna(subset=["price", "ch_upper", "ch_lower"]).tail(bars)
    if view.empty:
        raise ServiceError("チャネルグラフを生成できませんでした。")
    fig, axis = plt.subplots(figsize=(10, 4.8), facecolor="#111827")
    axis.set_facecolor("#111827")
    axis.plot(view.index, view["price"], color="#7dd3fc", linewidth=2.0, label="Price")
    axis.plot(view.index, view["ch_upper"], color="#a78bfa", linestyle="--", linewidth=1.3, label="Channel upper")
    axis.plot(view.index, view["ch_lower"], color="#a78bfa", linestyle="--", linewidth=1.3, label="Channel lower")
    axis.scatter(view.index[-1], view["price"].iloc[-1], color="#22d3ee", edgecolor="white", s=70, zorder=5, label="Latest")
    # 日本語フォントがないWindows環境でも文字化けしないよう、画像内はティッカー表記に限定する。
    axis.set_title(f"{ticker} - Last {len(view)} trading days", color="#f8fafc")
    axis.set_ylabel("Price", color="#cbd5e1")
    axis.grid(color="#334155", linestyle="--", alpha=0.55)
    axis.tick_params(colors="#cbd5e1")
    for spine in axis.spines.values():
        spine.set_color("#475569")
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate(rotation=30)
    legend = axis.legend(loc="best", frameon=True)
    legend.get_frame().set_facecolor("#1e293b")
    legend.get_frame().set_edgecolor("#475569")
    for text in legend.get_texts():
        text.set_color("#f8fafc")
    fig.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{_safe_filename(ticker)}_channel_{timestamp}.png"
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=145, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return f"/outputs/{filename}"


def serialize_prediction_result(
    data: PredictionData,
    model_result: dict[str, Any],
    ticker: str,
    display_name: str,
    chart_bars: int,
) -> dict[str, Any]:
    inference_date = data.inference_row.index[-1]
    latest = data.frame.loc[inference_date]
    label = model_result["prediction_label"]
    direction, direction_key = trend_for_class(label)
    chart_url = plot_channel_chart(data.frame.loc[:inference_date], ticker, display_name, chart_bars)
    probability_items = [
        {"label": class_label, "probability": probability, "percentage": probability * 100}
        for class_label, probability in zip(CLASS_ORDER, model_result["probabilities"])
    ]
    result = {
        "kind": "prediction",
        "company_name": display_name,
        "ticker": ticker,
        "basis_date": inference_date.strftime("%Y-%m-%d"),
        "fetched_at": data.fetched_at,
        "latest_price": float(latest["price"]),
        "prediction_class": label,
        "direction": direction,
        "direction_key": direction_key,
        "top_probability": model_result["top_probability"],
        "probabilities": probability_items,
        "analysis_comment": generate_analysis_comment(latest, label),
        "rsi": float(latest["rsi14"]),
        "channel_position": float(latest["ch_pos"]),
        "chart_url": chart_url,
        "validation": {
            "six_class_accuracy": model_result["six_class_accuracy"],
            "direction_accuracy": model_result["direction_accuracy"],
            "training_samples": model_result["training_samples"],
            "validation_samples": model_result["validation_samples"],
            "note": "過去データを時系列分割した検証結果であり、将来の精度を保証するものではありません。",
        },
        "warnings": data.warnings,
        "disclaimer": "本結果は情報提供および研究目的の参考情報であり、特定の金融商品の売買を推奨するものではありません。",
    }
    return result


def save_last_result(name: str, result: dict[str, Any]) -> None:
    payload = dict(result)
    payload["saved_at"] = datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")
    path = OUTPUT_DIR / f"last_{_safe_filename(name)}.json"
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def load_last_result(name: str) -> dict[str, Any] | None:
    path = OUTPUT_DIR / f"last_{_safe_filename(name)}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        chart_url = payload.get("chart_url")
        if chart_url and not (OUTPUT_DIR / Path(chart_url).name).exists():
            payload["chart_url"] = None
        return payload
    except (OSError, ValueError, TypeError):
        return None
