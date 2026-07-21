from __future__ import annotations

import inspect
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as app_module
import services.individual_chart_report as chart_report
import services.individual_market as market
import services.individual_service as individual_service
from services.common import ServiceError


def _long_base_frame() -> pd.DataFrame:
    index = pd.bdate_range("2012-01-04", periods=3600)
    prices = 1000.0 + np.arange(len(index), dtype=float) * 0.25
    return pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "price": prices,
            "volume": 1_000_000.0 + np.arange(len(index), dtype=float),
        },
        index=index,
    )


def _prepare_without_network(monkeypatch: pytest.MonkeyPatch):
    base = _long_base_frame()
    requested_tickers: list[str] = []

    monkeypatch.setattr(market, "_download_base_frame", lambda *_args, **_kwargs: base.copy())
    monkeypatch.setattr(
        market,
        "resolve_prediction_context",
        lambda _frame, _now: {
            "nikkei_base_date": base.index[-1],
            "prediction_context": "after_close",
            "prediction_mode": "大引け後",
        },
    )

    def fake_series_map(sources, _start, _end):
        requested_tickers.extend(str(source["ticker"]) for source in sources.values())
        if sources is market.JAPAN_SOURCES:
            return {"nikkei": pd.Series(np.linspace(25000.0, 40000.0, len(base)), index=base.index)}, []
        return {
            factor: pd.Series(np.linspace(100.0 + offset, 180.0 + offset, len(base)), index=base.index)
            for offset, factor in enumerate(("spx", "nasdaq", "vix", "usdjpy"))
        }, []

    monkeypatch.setattr(market, "_download_series_map", fake_series_map)
    now = datetime(2026, 1, 5, 16, 0, tzinfo=market.JST)
    return market.prepare_individual_market_data("7203.T", now=now), requested_tickers


def test_individual_japan_source_is_nikkei_only() -> None:
    assert market.JAPAN_SOURCES == {
        "nikkei": {"label": "日経平均株価", "ticker": "^N225"}
    }


def test_individual_market_download_never_requests_topix_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, requested = _prepare_without_network(monkeypatch)
    assert "^N225" in requested
    assert "^TOPX" not in requested
    assert "998405.T" not in requested
    assert data.metadata["used_japan_factors"] == ["日経平均株価"]


def test_nikkei_market_group_succeeds_without_topix_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, _requested = _prepare_without_network(monkeypatch)
    japan_group = data.feature_groups["B_stock_japan"]
    assert any(column.startswith("nikkei_") for column in japan_group)
    assert any(column.startswith("stock_vs_nikkei_") for column in japan_group)
    assert all("topix" not in column.lower() for column in data.frame.columns)
    assert all("topix" not in column.lower() for column in japan_group)


def test_individual_production_modules_have_no_topix_identifier() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "services/individual_market.py",
        "services/individual_service.py",
        "services/individual_chart_report.py",
        "templates/individual.html",
    ):
        source = (root / relative).read_text(encoding="utf-8").lower()
        assert "topix" not in source
        assert "^topx" not in source
        assert "998405.t" not in source


def test_chart_benchmark_download_requests_nikkei_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []
    index = pd.bdate_range("2024-01-04", periods=300)
    frame = pd.DataFrame({"price": np.linspace(30000.0, 40000.0, len(index))}, index=index)

    def fake_download(ticker, _start, _end):
        requested.append(ticker)
        return frame

    monkeypatch.setattr(individual_service, "_download_base_frame", fake_download)
    values, name = individual_service._download_nikkei_benchmark(index, [])
    assert requested == ["^N225"]
    assert name == "日経平均"
    assert values.equals(frame["price"])


def test_old_market_schema_cache_is_rejected() -> None:
    old = {
        "individual_ui_schema_version": individual_service.INDIVIDUAL_UI_SCHEMA_VERSION,
        "basis_date": "2026-07-17",
        "data_collection_start": "2018-07-17",
        "data_collection_end": "2026-07-17",
        "fixed_direction_threshold": individual_service.FIXED_DIRECTION_THRESHOLD,
        "six_stage_trend": {
            "items": [{"label": label} for _, label in individual_service.SIX_STAGE_DISPLAY_ORDER]
        },
        "individual_schema_versions": {
            "cache": "individual_prediction_cache_v5_hang_fix",
            "market_alignment": "individual_market_available_at_asof_v2",
            "feature_definition": "individual_normalized_market_features_v3",
            "selection": individual_service.INDIVIDUAL_SELECTION_VERSION,
            "calibration": individual_service.INDIVIDUAL_CALIBRATION_VERSION,
            "public_labels": individual_service.INDIVIDUAL_PUBLIC_LABEL_VERSION,
            "data_collection": individual_service.INDIVIDUAL_DATA_COLLECTION_VERSION,
        },
    }
    assert individual_service.is_current_individual_result(old) is False


def test_prediction_lock_is_released_after_cached_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        individual_service,
        "_load_current_code_cache",
        lambda _code: {"kind": "individual_prediction", "ticker": "7203.T"},
    )
    assert individual_service.predict_individual("7203")["ticker"] == "7203.T"
    assert individual_service._prediction_lock.acquire(blocking=False)
    individual_service._prediction_lock.release()


def test_409_response_uses_public_message_and_no_previous_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = "別の個別銘柄予測が実行中です。処理が終了してから、もう一度お試しください。"
    monkeypatch.setattr(
        app_module,
        "predict_individual",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ServiceError(message, "実行中の計算が終了すると、再度実行できます。", 409)
        ),
    )
    response = TestClient(app_module.app).post("/api/predict/individual", json={"ticker": "7203"})
    assert response.status_code == 409
    assert response.json()["error"]["message"] == message
    assert response.json()["last_result"] is None


def test_frontend_blocks_duplicate_posts_and_recovers_from_409() -> None:
    source = (Path(__file__).resolve().parents[1] / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert source.count('form.addEventListener("submit"') == 1
    assert 'activeController !== null || form.getAttribute("aria-busy") === "true"' in source
    assert "response.status === 409" in source
    assert "別の個別銘柄予測が実行中です。処理が終了してから、もう一度お試しください。" in source
    assert "finally {" in source
    assert "overlay.hidden = true" in source
    assert "button.disabled = false" in source
    assert 'form.removeAttribute("aria-busy")' in source


def test_report_defaults_and_metric_are_nikkei_generic() -> None:
    assert inspect.signature(chart_report.build_chart_analysis).parameters["benchmark_name"].default == "日経平均"
    values = np.linspace(100.0, 125.0, 80)
    index = pd.bdate_range("2025-01-06", periods=80)
    frame = pd.DataFrame(
        {
            "price": values,
            "open": values,
            "high": values,
            "low": values,
            "volume": np.full(80, 1000.0),
        },
        index=index,
    )
    benchmark = pd.Series(np.linspace(100.0, 110.0, 80), index=index)
    report = chart_report.build_chart_analysis(frame, benchmark=benchmark)
    assert report["metrics"]["benchmark_name"] == "日経平均"
    assert "relative_return_vs_benchmark_pct" in report["metrics"]