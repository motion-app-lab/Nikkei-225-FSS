from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services.individual_chart_report import build_six_stage_trend_report
from services.nikkei_public_report import (
    NIKKEI_PUBLIC_SCHEMA_VERSION,
    build_nikkei_chart_payload,
)
from services.nikkei_artifact import load_context_evaluation
from services.nikkei_service import (
    SIX_CLASS_DISPLAY_LABELS,
    _public_evaluation,
    is_current_nikkei_result,
)


ROOT = Path(__file__).resolve().parents[1]


def _frame() -> pd.DataFrame:
    index = pd.bdate_range("2024-06-03", periods=530)
    trend = np.linspace(35_000.0, 42_000.0, len(index))
    cycle = np.sin(np.arange(len(index)) / 12.0) * 650.0
    close = trend + cycle
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.006,
            "low": close * 0.994,
            "price": close,
            "volume": np.linspace(90_000_000, 120_000_000, len(index)),
        },
        index=index,
    )


def _six_stage() -> dict:
    raw_labels = ("急騰", "上昇", "やや上昇", "やや下落", "下落", "急落")
    values = (0.12, 0.18, 0.24, 0.21, 0.15, 0.10)
    return build_six_stage_trend_report(
        [{"raw_label": label, "label": label, "probability": value} for label, value in zip(raw_labels, values)]
    )


def test_nikkei_six_class_public_labels_are_exact_and_ordered() -> None:
    assert [SIX_CLASS_DISPLAY_LABELS[key] for key in ("急騰", "上昇", "やや上昇", "やや下落", "下落", "急落")] == [
        "上昇 Lv.3", "上昇 Lv.2", "上昇 Lv.1", "下落 Lv.1", "下落 Lv.2", "下落 Lv.3"
    ]


def test_chart_payload_uses_sorted_confirmed_rows_and_common_close_values() -> None:
    frame = _frame()
    basis_date = frame.index[-2]
    payload = build_nikkei_chart_payload(frame, basis_date, _six_stage())
    short = payload["chart_60d"]
    long = payload["chart_2y"]
    assert len(short) == 60
    assert len(long) == 504
    assert [row["date"] for row in short] == sorted(row["date"] for row in short)
    assert len({row["date"] for row in long}) == len(long)
    assert short[-1]["date"] == basis_date.strftime("%Y-%m-%d")
    long_by_date = {row["date"]: row["close"] for row in long}
    assert all(row["close"] == pytest.approx(long_by_date[row["date"]]) for row in short)
    assert frame.index[-1].strftime("%Y-%m-%d") not in {row["date"] for row in long}


def test_reports_are_deterministic_short_and_do_not_claim_two_year_forecast() -> None:
    frame = _frame()
    first = build_nikkei_chart_payload(frame, frame.index[-1], _six_stage())
    second = build_nikkei_chart_payload(frame, frame.index[-1], _six_stage())
    assert first == second
    short_body = first["short_term_report"]["body"]
    long_body = first["medium_long_term_report"]["body"]
    assert "5営業日先" in short_body and "6段階出力" in short_body
    assert "将来2年間" not in long_body and "2年後" not in long_body
    for forbidden in ("買い時", "売り時", "購入推奨", "売却推奨", "利益を期待", "必ず上昇"):
        assert forbidden not in short_body
        assert forbidden not in long_body


def test_reports_use_no_external_generation_api() -> None:
    source = (ROOT / "services" / "nikkei_public_report.py").read_text(encoding="utf-8").lower()
    for forbidden in ("openai", "gemini", "anthropic", "claude", "requests.post", "httpx"):
        assert forbidden not in source


def test_public_evaluation_prediction_precision_matches_confusion_matrix() -> None:
    formal = load_context_evaluation("intraday")
    assert formal is not None
    public = _public_evaluation(formal, "intraday")
    confusion = public["confusion_matrix"]
    true_down, false_up = confusion[0]
    false_down, true_up = confusion[1]
    assert public["predicted_up_count"] == true_up + false_up
    assert public["predicted_down_count"] == true_down + false_down
    assert public["up_prediction_precision"] == pytest.approx(true_up / (true_up + false_up))
    assert public["down_prediction_precision"] == pytest.approx(true_down / (true_down + false_down))


def test_old_public_cache_schema_is_rejected() -> None:
    assert is_current_nikkei_result({"nikkei_public_schema_version": "old"}) is False
    current = {
        "nikkei_public_schema_version": NIKKEI_PUBLIC_SCHEMA_VERSION,
        "six_class_probabilities": [],
        "six_class_report": {},
        "chart_60d": [],
        "short_term_report": {},
        "chart_2y": [],
        "medium_long_term_report": {},
        "selected_factors": [],
        "excluded_factors": [],
        "feature_importance_top10": [],
        "accuracy_summary": {},
        "prediction_context_label": "場中データによる予測",
        "direction_evaluation": {},
        "six_class_evaluation": {},
        "model_roles_note": "方向予測と6段階予測は別モデルです。",
    }
    assert is_current_nikkei_result(current) is True


def test_nikkei_production_paths_do_not_add_topix() -> None:
    paths = (
        ROOT / "services" / "nikkei_service.py",
        ROOT / "services" / "nikkei_dual_market.py",
        ROOT / "services" / "nikkei_dual_model.py",
        ROOT / "services" / "nikkei_public_report.py",
        ROOT / "templates" / "nikkei.html",
    )
    production = "\n".join(path.read_text(encoding="utf-8") for path in paths).lower()
    assert "^topx" not in production
    assert "998405.t" not in production
    assert "topix" not in production

def test_chart_reports_stay_within_two_to_four_sentences() -> None:
    payload = build_nikkei_chart_payload(_frame(), _frame().index[-1], _six_stage())
    for key in ("short_term_report", "medium_long_term_report"):
        sentence_count = payload[key]["body"].count("。")
        assert 2 <= sentence_count <= 4
