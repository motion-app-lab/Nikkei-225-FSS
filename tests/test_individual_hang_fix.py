from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as app_module
import services.individual_market as market
import services.individual_service as individual_service
from services.common import ServiceError


def test_7203_api_request_completes_in_finite_time_when_prediction_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app_module,
        "predict_individual",
        lambda ticker, **_kwargs: {"kind": "individual_prediction", "ticker": f"{ticker}.T"},
    )
    monkeypatch.setattr(app_module, "save_last_result", lambda *_args, **_kwargs: None)
    started = time.perf_counter()
    response = TestClient(app_module.app).post("/api/predict/individual", json={"ticker": "7203"})
    assert time.perf_counter() - started < 1.0
    assert response.status_code == 200
    assert response.json()["result"]["ticker"] == "7203.T"


def test_backend_timeout_cancels_control_and_returns_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped = {"value": False}

    def waits_for_cancel(_ticker: str, *, execution_control):
        while not execution_control.cancel_event.wait(0.002):
            pass
        stopped["value"] = True
        execution_control.checkpoint("test_after_cancel")

    monkeypatch.setattr(app_module, "INDIVIDUAL_MAX_PROCESSING_SECONDS", 0.03)
    monkeypatch.setattr(app_module, "predict_individual", waits_for_cancel)
    response = TestClient(app_module.app).post("/api/predict/individual", json={"ticker": "7203"})
    assert response.status_code == 504
    assert "制限時間" in response.json()["error"]["message"]
    deadline = time.perf_counter() + 1.0
    while not stopped["value"] and time.perf_counter() < deadline:
        time.sleep(0.005)
    assert stopped["value"] is True


def test_duplicate_individual_calculation_is_rejected_without_waiting() -> None:
    assert individual_service._prediction_lock.acquire(blocking=False)
    try:
        with pytest.raises(ServiceError) as raised:
            individual_service.predict_individual("7203")
        assert raised.value.status_code == 409
        assert "実行中" in raised.value.message
    finally:
        individual_service._prediction_lock.release()


def test_target_download_failure_is_not_misreported_as_short_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(individual_service, "_load_current_code_cache", lambda _code: None)
    monkeypatch.setattr(
        individual_service,
        "prepare_individual_market_data",
        lambda _ticker: (_ for _ in ()).throw(ServiceError("7203.T の株価データを取得できませんでした。")),
    )
    with pytest.raises(ServiceError) as raised:
        individual_service.predict_individual("7203")
    assert raised.value.status_code == 503
    assert "取得できなかった" in raised.value.message
    assert "履歴が不足" not in raised.value.message
    assert individual_service._prediction_lock.acquire(blocking=False)
    individual_service._prediction_lock.release()


def test_external_download_has_finite_attempts_timeout_and_no_internal_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def failed_download(*_args, **kwargs):
        calls.append(kwargs)
        raise TimeoutError("synthetic timeout")

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=failed_download))
    values, warnings = market._download_series_map(
        {"spx": market.EXTERNAL_SOURCES["spx"]},
        datetime(2024, 1, 1),
        datetime(2026, 1, 1),
    )
    assert values == {}
    assert len(warnings) == 1
    assert len(calls) == market.EXTERNAL_DOWNLOAD_MAX_ATTEMPTS
    assert all(call["timeout"] == market.EXTERNAL_DOWNLOAD_TIMEOUT_SECONDS for call in calls)
    assert all(call["threads"] is False for call in calls)


def test_merge_asof_keys_are_explicitly_same_utc_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    original = pd.merge_asof
    captured: dict[str, str] = {}

    def checked_merge(left, right, **kwargs):
        captured["left"] = str(left[kwargs["left_on"]].dtype)
        captured["right"] = str(right[kwargs["right_on"]].dtype)
        return original(left, right, **kwargs)

    monkeypatch.setattr(market.pd, "merge_asof", checked_merge)
    friday = market.external_available_timestamp("2026-03-06", "us_close")
    source = pd.DataFrame(
        {
            "available_at_jst": [friday],
            "source_date": pd.to_datetime(["2026-03-06"]),
            "spx_ret1": [1.25],
        }
    )
    cutoffs = pd.Series(
        [pd.Timestamp("2026-03-09 09:00", tz="Asia/Tokyo")],
        index=pd.DatetimeIndex(["2026-03-06"]),
        dtype="object",
    )
    aligned, diagnostics = market.asof_align_external(cutoffs, {"spx": source})
    assert captured["left"] == captured["right"]
    assert captured["left"].startswith("datetime64[") and captured["left"].endswith(", UTC]")
    assert aligned.iloc[0]["spx_ret1"] == pytest.approx(1.25)
    assert diagnostics["spx"].iloc[0]["available_at_jst"] <= cutoffs.iloc[0]


def test_frontend_always_clears_individual_busy_state_and_ignores_old_responses() -> None:
    source = (Path(__file__).resolve().parents[1] / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert "INDIVIDUAL_REQUEST_TIMEOUT_MS" in source
    assert 'form.setAttribute("aria-busy", "true")' in source
    assert 'panel.setAttribute("aria-busy", "true")' in source
    assert 'form.removeAttribute("aria-busy")' in source
    assert 'panel.removeAttribute("aria-busy")' in source
    assert "overlay.hidden = true" in source
    assert "button.disabled = false" in source
    assert "currentGeneration !== requestGeneration" in source
    assert "activeController?.abort()" in source
    assert "clearStaleResult();" in source
    assert "個別銘柄予測の計算が制限時間を超えたため終了しました。" in source
