from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as app_module
import services.individual_service as individual_service
from services.common import ServiceError
from services.individual_service import (
    _confirmed_basis_date,
    normalize_japanese_security_code,
)


@pytest.mark.parametrize("value,code,ticker", [("7203", "7203", "7203.T"), ("130a", "130A", "130A.T")])
def test_japanese_code_is_normalized_without_user_suffix(value: str, code: str, ticker: str) -> None:
    assert normalize_japanese_security_code(value) == (code, ticker)


@pytest.mark.parametrize("value", ["AAPL", "NVDA", "MSFT", "7203.T", "7203,6758", "7203 6758", "", "123"])
def test_overseas_suffix_multiple_and_invalid_codes_are_rejected(value: str) -> None:
    with pytest.raises(ServiceError):
        normalize_japanese_security_code(value)


def test_intraday_uses_previous_confirmed_session() -> None:
    index = pd.to_datetime(["2026-07-16", "2026-07-17"])
    frame = pd.DataFrame({"price": [100, 101]}, index=index)
    date, context = _confirmed_basis_date(frame, datetime(2026, 7, 17, 10, tzinfo=timezone(timedelta(hours=9))))
    assert date == pd.Timestamp("2026-07-16")
    assert context == "場中"


def test_after_close_uses_confirmed_same_day() -> None:
    index = pd.to_datetime(["2026-07-16", "2026-07-17"])
    frame = pd.DataFrame({"price": [100, 101]}, index=index)
    date, context = _confirmed_basis_date(frame, datetime(2026, 7, 17, 16, tzinfo=timezone(timedelta(hours=9))))
    assert date == pd.Timestamp("2026-07-17")
    assert context == "大引け後"




def test_individual_api_error_never_returns_previous_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        app_module,
        "predict_individual",
        lambda _ticker, **_kwargs: (_ for _ in ()).throw(ServiceError("失敗", "確認")),
    )
    monkeypatch.setattr(app_module, "load_last_result", lambda _: {"ticker": "7203.T"})
    response = TestClient(app_module.app).post("/api/predict/individual", json={"ticker": "6758"})
    assert response.status_code == 422
    assert response.json()["last_result"] is None


def test_individual_last_result_rejects_old_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "load_last_result", lambda _: {"kind": "prediction", "ticker": "7203.T"})
    response = TestClient(app_module.app).get("/api/last/individual")
    assert response.status_code == 404


def test_individual_last_result_accepts_current_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    current = {
        "kind": "individual_prediction",
        "ticker": "7203.T",
        "individual_ui_schema_version": individual_service.INDIVIDUAL_UI_SCHEMA_VERSION,
        "chart_analysis": {"schema_version": individual_service.CHART_ANALYSIS_VERSION},
        "long_chart_analysis": {"schema_version": individual_service.LONG_CHART_ANALYSIS_VERSION},
        "six_stage_trend": {
            "schema_version": individual_service.SIX_STAGE_REPORT_VERSION,
            "items": [
                {"label": label}
                for _, label in individual_service.SIX_STAGE_DISPLAY_ORDER
            ],
        },
        "long_chart_url": "/outputs/long_chart.png",
        "basis_date": "2026-07-17",
        "data_collection_start": "2018-07-17",
        "data_collection_end": "2026-07-17",
        "fixed_direction_threshold": individual_service.FIXED_DIRECTION_THRESHOLD,
        "individual_schema_versions": {
            "cache": individual_service.INDIVIDUAL_CACHE_VERSION,
            "market_alignment": individual_service.INDIVIDUAL_MARKET_ALIGNMENT_VERSION,
            "feature_definition": individual_service.INDIVIDUAL_FEATURE_DEFINITION_VERSION,
            "selection": individual_service.INDIVIDUAL_SELECTION_VERSION,
            "calibration": individual_service.INDIVIDUAL_CALIBRATION_VERSION,
            "public_labels": individual_service.INDIVIDUAL_PUBLIC_LABEL_VERSION,
            "data_collection": individual_service.INDIVIDUAL_DATA_COLLECTION_VERSION,
        },
    }
    monkeypatch.setattr(app_module, "load_last_result", lambda _: current)
    response = TestClient(app_module.app).get("/api/last/individual")
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["individual_ui_schema_version"] == individual_service.INDIVIDUAL_UI_SCHEMA_VERSION
    assert "formal_evaluation_available" not in result
    assert "validation" not in result


def _write_individual_cache(tmp_path: Path, *, basis_date: str, chart_version: str) -> Path:
    chart = tmp_path / "chart.png"
    chart.write_bytes(b"png")
    long_chart = tmp_path / "long_chart.png"
    long_chart.write_bytes(b"png")
    path = tmp_path / "individual_cache_7203_test.json"
    path.write_text(
        json.dumps(
            {
                "kind": "individual_prediction",
                "basis_date": basis_date,
                "chart_url": "/outputs/chart.png",
                "long_chart_url": "/outputs/long_chart.png",
                "individual_ui_schema_version": individual_service.INDIVIDUAL_UI_SCHEMA_VERSION,
                "chart_analysis": {"schema_version": chart_version},
                "long_chart_analysis": {"schema_version": individual_service.LONG_CHART_ANALYSIS_VERSION},
                "six_stage_trend": {
                    "schema_version": individual_service.SIX_STAGE_REPORT_VERSION,
                    "items": [
                        {"label": label}
                        for _, label in individual_service.SIX_STAGE_DISPLAY_ORDER
                    ],
                },
                "data_collection_start": "2018-07-17",
                "data_collection_end": basis_date,
                "fixed_direction_threshold": individual_service.FIXED_DIRECTION_THRESHOLD,
                "individual_schema_versions": {
                    "cache": individual_service.INDIVIDUAL_CACHE_VERSION,
                    "market_alignment": individual_service.INDIVIDUAL_MARKET_ALIGNMENT_VERSION,
                    "feature_definition": individual_service.INDIVIDUAL_FEATURE_DEFINITION_VERSION,
                    "selection": individual_service.INDIVIDUAL_SELECTION_VERSION,
                    "calibration": individual_service.INDIVIDUAL_CALIBRATION_VERSION,
                            "public_labels": individual_service.INDIVIDUAL_PUBLIC_LABEL_VERSION,
                    "data_collection": individual_service.INDIVIDUAL_DATA_COLLECTION_VERSION,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_current_basis_cache_is_reused_before_market_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_individual_cache(
        tmp_path,
        basis_date="2026-07-17",
        chart_version=individual_service.CHART_ANALYSIS_VERSION,
    )
    monkeypatch.setattr(individual_service, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(individual_service, "_expected_current_basis_date", lambda: pd.Timestamp("2026-07-17"))
    cached = individual_service._load_current_code_cache("7203")
    assert cached is not None
    assert cached["cache"]["used"] is True
    assert cached["cache"]["fast_reuse"] is True


def test_cache_without_current_chart_analysis_schema_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _write_individual_cache(tmp_path, basis_date="2026-07-17", chart_version="old")
    monkeypatch.setattr(individual_service, "OUTPUT_DIR", tmp_path)
    assert individual_service._load_cache(path) is None
