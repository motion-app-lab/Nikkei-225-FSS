from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from services.common import CLASS_ORDER, chronological_split, classify_change
from services.simulation_service import simulate


ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


@pytest.mark.parametrize("path", ["/", "/nikkei", "/individual", "/simulation", "/about"])
def test_pages_return_http_200(path: str) -> None:
    response = client.get(path)
    assert response.status_code == 200
    assert "日経平均株価予測・戦略支援システム" in response.text
    assert "日本株式予測・戦略支援システム" not in response.text


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (5.0, "急騰"),
        (4.999, "上昇"),
        (2.0, "上昇"),
        (1.999, "やや上昇"),
        (0.5, "やや上昇"),
        (0.499, "やや下落"),
        (-0.5, "やや下落"),
        (-0.501, "下落"),
        (-2.0, "下落"),
        (-2.001, "急落"),
    ],
)
def test_six_class_boundaries(change: float, expected: str) -> None:
    assert classify_change(change) == expected
    assert expected in CLASS_ORDER


def test_chronological_split_keeps_order() -> None:
    frame = pd.DataFrame({"value": range(10)}, index=pd.date_range("2025-01-01", periods=10))
    train, validation = chronological_split(frame)
    assert train["value"].tolist() == list(range(7))
    assert validation["value"].tolist() == list(range(7, 10))
    assert train.index.max() < validation.index.min()


def _simulation_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=18)
    closes = np.array([100, 101, 102, 103, 104, 105, 106, 104, 103, 102, 101, 100, 102, 103, 104, 105, 106, 107], dtype=float)
    opens = np.r_[100.0, closes[:-1]]
    return pd.DataFrame(
        {"open": opens, "high": np.maximum(opens, closes) + 0.4, "low": np.minimum(opens, closes) - 0.4, "close": closes},
        index=dates,
    )


def test_simulation_returns_dictionary_and_consistent_assets() -> None:
    result = simulate(
        _simulation_frame(),
        take_profit=5,
        stop_loss=3,
        initial_investment=1_000_000,
    )
    assert isinstance(result, dict)
    assert result["initial_assets"] == 1_000_000
    assert result["final_assets"] == pytest.approx(result["initial_assets"] + result["profit_amount"])
    expected_rate = result["profit_amount"] / result["initial_assets"] * 100
    assert result["profit_rate"] == pytest.approx(expected_rate)
    assert result["trade_count"] == len(result["trades"])


def test_simulation_never_opens_a_new_trade_on_the_final_day() -> None:
    result = simulate(_simulation_frame(), take_profit=5, stop_loss=3)
    assert result["trades"]
    assert all(trade["entry_date"] != result["simulation_end"] for trade in result["trades"])


def test_invalid_ticker_does_not_stop_application() -> None:
    response = client.post("/api/predict/individual", json={"ticker": "???"})
    assert response.status_code == 422
    body = response.json()
    assert body["ok"] is False
    assert "ティッカー" in body["error"]["message"]
    assert client.get("/health").status_code == 200


def test_original_files_are_preserved_byte_for_byte() -> None:
    for filename in ("nikkei_no2.py", "nikkei_kobetu_no1.py", "sim_research.py"):
        original = ROOT / filename
        preserved = ROOT / "legacy_original" / filename
        assert preserved.exists()
        assert hashlib.sha256(original.read_bytes()).digest() == hashlib.sha256(preserved.read_bytes()).digest()


def test_new_implementation_has_no_fixed_output_path_or_gui_open() -> None:
    source_files = [ROOT / "app.py", *sorted((ROOT / "services").glob("*.py"))]
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_files)
    forbidden_path = "C:" + "\\" + "week_yosoku"
    assert forbidden_path not in source
    assert "os.startfile" not in source
    assert "tensorflow" not in source.lower()
    assert "from prophet" not in source.lower()


def test_api_key_is_not_hard_coded_and_target_is_not_a_feature() -> None:
    common_source = (ROOT / "services" / "common.py").read_text(encoding="utf-8")
    assert 'os.getenv("FRED_API_KEY"' in common_source
    assert "feature_columns + [\"target_change\"]" in common_source
    assert 'inference_row = frame.loc[[inference_date], feature_columns]' in common_source
    assert "validation_model.predict(x_train)" not in common_source
