from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app import app


HTML = Path("templates/individual.html").read_text(encoding="utf-8")
JS = Path("static/js/app.js").read_text(encoding="utf-8")
CSS = Path("static/css/style.css").read_text(encoding="utf-8")
INDIVIDUAL_RENDERER = JS[
    JS.index("const renderIndividualPrediction") : JS.index("const nikkeiChartMarkup")
]


def test_individual_page_is_http_200_and_uses_separate_japanese_code_examples() -> None:
    response = TestClient(app).get("/individual")
    assert response.status_code == 200
    assert "入力例（いずれか1銘柄）" in response.text
    assert '<code>7203</code>' in response.text
    assert '<code>130A</code>' in response.text
    assert response.text.index('<code>7203</code>') < response.text.index('<code>130A</code>')
    assert ".Tは不要です。日本株の証券コードを1銘柄だけ入力してください。" in response.text
    joined_example = "7203" + "、" + "130A"
    assert joined_example not in response.text


def test_result_order_is_prediction_short_long_model_warning_notice() -> None:
    main_tokens = [
        'data-individual-section="prediction"',
        "${shortChart}",
        'data-individual-section="chart-analysis"',
        "${longChart}",
        'data-individual-section="long-chart-analysis"',
    ]
    positions = [INDIVIDUAL_RENDERER.index(token) for token in main_tokens]
    assert positions == sorted(positions)
    output_model = INDIVIDUAL_RENDERER.rindex("${modelAuditMarkup}")
    output_warning = INDIVIDUAL_RENDERER.rindex("${warningsMarkup(result.warnings)}")
    output_notice = INDIVIDUAL_RENDERER.rindex('data-individual-section="notice"')
    assert output_model < output_warning < output_notice

def test_public_prediction_uses_six_stage_distribution_without_direction_hero() -> None:
    assert "5営業日先の6段階トレンド予測" in INDIVIDUAL_RENDERER
    assert "モデル出力割合" in INDIVIDUAL_RENDERER
    assert "six-stage-list" in INDIVIDUAL_RENDERER
    assert "6段階トレンド予測レポート" in INDIVIDUAL_RENDERER
    assert "sixStage.level_note" in INDIVIDUAL_RENDERER
    for forbidden in (
        "${escapeHtml(result.direction)}",
        "trend.symbol",
        "individual-direction",
        "individual-move-size",
        "最上位確率",
        "判定ライン",
    ):
        assert forbidden not in INDIVIDUAL_RENDERER


def test_data_collection_period_replaces_future_target_period() -> None:
    assert "データ収集期間" in INDIVIDUAL_RENDERER
    assert "result.data_collection_period" in INDIVIDUAL_RENDERER
    assert "判定対象期間" not in INDIVIDUAL_RENDERER
    assert "result.target_period" not in INDIVIDUAL_RENDERER


def test_public_six_stage_labels_use_level_names() -> None:
    chart_source = Path("services/individual_chart_report.py").read_text(encoding="utf-8")
    for label in ("上昇 Lv.3", "上昇 Lv.2", "上昇 Lv.1", "下落 Lv.1", "下落 Lv.2", "下落 Lv.3"):
        assert label in chart_source
    for old_label in ("上昇level3", "上昇level2", "上昇level1", "下落level1", "下落level2", "下落level3"):
        assert old_label not in chart_source
    assert "Lv.はモデル内の値動き幅の区分を示すもので、予測の確実性や売買の推奨度を示すものではありません。" in chart_source
    for old_label in ("大幅上昇", "大幅下落"):
        assert old_label not in INDIVIDUAL_RENDERER


def test_two_chart_headings_and_reports_are_rendered() -> None:
    for required in (
        "直近60営業日のチャート",
        "短期チャート分析レポート",
        "直近2年間のチャート",
        "中長期チャート分析レポート",
    ):
        assert required in INDIVIDUAL_RENDERER
    assert INDIVIDUAL_RENDERER.index("直近60営業日のチャート") < INDIVIDUAL_RENDERER.index("短期チャート分析レポート")
    assert INDIVIDUAL_RENDERER.index("直近2年間のチャート") < INDIVIDUAL_RENDERER.index("中長期チャート分析レポート")


def test_individual_prediction_does_not_show_accuracy_evaluation() -> None:
    for forbidden in (
        "予測精度",
        "上昇予測の一致率",
        "下落予測の一致率",
        "6段階完全一致率",
        "単純な予測方法との差",
        "この銘柄の保存済み予測精度データはありません",
        "accuracyMarkup",
        "individual-accuracy-summary",
        "individual-evaluation-details",
    ):
        assert forbidden not in INDIVIDUAL_RENDERER
    assert "予測精度" not in HTML


def test_new_prediction_and_error_clear_old_individual_result() -> None:
    assert 'const isIndividual = form.id === "individual-form"' in JS
    assert "sessionStorage.removeItem(storageKey)" in JS
    assert "const localPrevious = isolatesErrors\n            ? null" in JS
    assert 'form.addEventListener("invalid", clearIndividualResult, true)' in JS
    handler_start = JS.index('form.addEventListener("submit"')
    handler = JS[handler_start:JS.index("submitButtons.forEach", handler_start)]
    assert handler.index('const isIndividual') < handler.index('form.reportValidity()')


def test_mobile_css_prevents_overflow_and_stacks_six_stage_rows() -> None:
    assert "@media (max-width: 700px)" in CSS
    assert ".six-stage-row" in CSS
    assert "grid-template-columns: 1fr" in CSS
    assert "overflow-wrap: anywhere" in CSS
    assert ".individual-public-precision-grid" not in CSS


def test_public_model_names_and_individual_loading_guidance_are_plain_japanese() -> None:
    base = Path("templates/base.html").read_text(encoding="utf-8")
    guidance = "市場データの取得とモデル分析を実行しています。処理には数十秒から1分程度かかる場合があります。画面を閉じずにお待ちください。"
    assert 'result.direction_model_name === "二値ロジスティック回帰"' in INDIVIDUAL_RENDERER
    assert '"ロジスティック回帰（上昇・下落）"' in INDIVIDUAL_RENDERER
    assert 'result.six_class_model_name === "多クラスロジスティック回帰"' in INDIVIDUAL_RENDERER
    assert '"ロジスティック回帰（6段階）"' in INDIVIDUAL_RENDERER
    assert guidance in base
    assert "individual-loading-guidance" in CSS
    assert "overlay.hidden = false" in JS
    assert "overlay.hidden = true" in JS


def test_six_stage_bars_use_the_same_color_as_primary_button() -> None:
    assert ".primary-button" in CSS
    assert "background: var(--cyan);" in CSS
    assert ".six-stage-fill { display: block; height: 100%; border-radius: inherit; background: var(--cyan); }" in CSS


def test_no_old_five_session_wording_remains_in_public_assets() -> None:
    old_wording = "5営業日" + "後"
    assert old_wording not in HTML
    assert old_wording not in JS
