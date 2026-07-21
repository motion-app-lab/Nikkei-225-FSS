from pathlib import Path

from fastapi.testclient import TestClient

import app as app_module


ROOT = Path(__file__).resolve().parents[1]
WARNING = (
    "本予測は参考情報であり、売買を推奨するものではありません。"
    "最終的な投資判断はご自身の責任でお願いします。"
)
LABELS = ("上昇 Lv.3", "上昇 Lv.2", "上昇 Lv.1", "下落 Lv.1", "下落 Lv.2", "下落 Lv.3")


def _public_renderer() -> str:
    script = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
    return script.split("const renderNikkeiPublicPrediction", 1)[1].split("const renderSimulation", 1)[0]


def test_nikkei_warning_is_exact_and_precedes_result() -> None:
    template = (ROOT / "templates" / "nikkei.html").read_text(encoding="utf-8")
    assert template.count(WARNING) == 1
    assert template.index("投資判断に関する注意") < template.index('id="result-panel"')
    assert 'class="investment-warning"' in template


def test_six_class_labels_and_fixed_display_order() -> None:
    renderer = _public_renderer()
    service = (ROOT / "services" / "nikkei_service.py").read_text(encoding="utf-8")
    previous = -1
    for label in LABELS:
        position = service.index(f'"{label}"')
        assert position > previous
        previous = position
    assert "six_class_probabilities" in renderer
    assert "percentage.toFixed(1)" in renderer
    assert "display_total_percentage" in renderer
    assert "Lv.はモデル内の値動き幅の区分" in (ROOT / "services" / "individual_chart_report.py").read_text(encoding="utf-8")


def test_large_direction_conclusion_and_arrows_are_absent_from_nikkei_renderer() -> None:
    renderer = _public_renderer()
    for forbidden in ("<strong>${escapeHtml(result.direction)}</strong>", "trend.symbol", "direction-result", "separated-signal", "巨大", "↗", "↘", "↓"):
        assert forbidden not in renderer
    assert "5営業日先の6段階予測" in renderer


def test_new_public_sections_appear_in_required_order() -> None:
    renderer = _public_renderer()
    headings = (
        "5営業日先の6段階予測",
        "6段階トレンド予測レポート",
        "直近60営業日の株価推移",
        "短期動向レポート",
        "直近2年間の株価推移",
        "中長期トレンド分析レポート",
        "予測精度",
        "今回のモデルが選択したファクター",
        "方向予測に使われた特徴量重要度",
        "採用構成と全特徴量の詳細を見る",
        "データ・検証条件・注意事項",
    )
    for heading in headings:
        assert heading in renderer
    markers = (
        'data-nikkei-section="six-class"', 'data-nikkei-section="six-report"',
        'data-nikkei-section="chart-60d"', 'data-nikkei-section="short-report"',
        'data-nikkei-section="chart-2y"', 'data-nikkei-section="long-report"',
        'data-nikkei-section="accuracy"', 'data-nikkei-section="factors"',
        'data-nikkei-section="importance"', 'data-nikkei-section="details"',
        'data-nikkei-section="conditions"',
    )
    positions = [renderer.index(marker) for marker in markers]
    assert positions == sorted(positions)


def test_header_metadata_is_dynamic_and_uses_five_days_ahead_wording() -> None:
    renderer = _public_renderer()
    for field in (
        "forecast_horizon_label",
        "forecast_base_date",
        "forecast_target_date",
        "data_collection_start",
        "data_collection_end",
        "evaluation_start",
        "evaluation_end",
        "fetched_at",
    ):
        assert field in renderer
    assert "5営業日後" not in renderer


def test_accuracy_uses_existing_evaluation_and_prediction_precision() -> None:
    renderer = _public_renderer()
    for field in (
        "validation_samples",
        "correct_predictions",
        "direction_accuracy",
        "up_prediction_precision",
        "down_prediction_precision",
        "best_baseline_gap",
    ):
        assert field in renderer
    assert "上昇予測の一致率" in renderer
    assert "下落予測の一致率" in renderer
    assert "単純な予測方法との差" in renderer
    assert "勝率" not in renderer
    assert "利益確率" not in renderer


def test_factor_importance_and_details_are_public_but_not_fake() -> None:
    renderer = _public_renderer()
    for field in ("selected_factors", "excluded_factors", "factor_selection_definition", "feature_importance_top10"):
        assert field in renderer
    assert "因果関係を意味しません" in renderer
    assert '<summary>採用構成と全特徴量の詳細を見る</summary>' in renderer
    assert "Math.random" not in renderer


def test_chart_renderer_has_tooltips_and_shared_series_names() -> None:
    renderer = _public_renderer()
    for field in ("chart_60d", "chart_2y", "close", "ma5", "ma20", "ma60", "ma200"):
        assert field in renderer
    assert 'data-chart-tooltip' in renderer
    assert "pointerenter" in renderer
    assert "click" in renderer


def test_nikkei_css_is_scoped_and_mobile_safe() -> None:
    style = (ROOT / "static" / "css" / "style.css").read_text(encoding="utf-8")
    for selector in (".nikkei-section", ".nikkei-svg-chart", ".nikkei-factor-columns", ".nikkei-importance-row"):
        assert selector in style
    assert "@media (max-width: 400px)" in style
    assert ".nikkei-public-result { padding: 0.65rem; }" in style
    assert "overflow: hidden" in style


def test_nikkei_page_returns_http_200() -> None:
    response = TestClient(app_module.app).get("/nikkei")
    assert response.status_code == 200
    assert WARNING in response.text


def test_other_public_pages_remain_http_200() -> None:
    client = TestClient(app_module.app)
    assert client.get("/individual").status_code == 200
    assert client.get("/simulation").status_code == 200

def test_chart_renderer_does_not_coerce_missing_moving_averages_to_zero() -> None:
    script = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
    chart_renderer = script.split("const nikkeiChartMarkup", 1)[1].split("const renderNikkeiPublicPrediction", 1)[0]
    assert 'value === null || value === undefined || value === ""' in chart_renderer
    assert "series.map((item) => chartNumber(row[item.key]))" in chart_renderer
    assert "series.map((item) => Number(row[item.key]))" not in chart_renderer
