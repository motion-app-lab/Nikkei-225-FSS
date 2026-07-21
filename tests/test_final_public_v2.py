from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app import app


ROOT = Path(__file__).resolve().parents[1]
CLIENT = TestClient(app)
FORMAL_NAME = "日経平均株価予測・戦略支援システム"
OLD_NAME = "日本株式予測・戦略支援システム"
WAITING_GUIDANCE = "市場データの取得とモデル分析を実行しています。処理には数十秒から1分程度かかる場合があります。画面を閉じずにお待ちください。"


def test_formal_system_name_is_public_on_every_page() -> None:
    for path in ("/", "/nikkei", "/individual", "/simulation", "/about"):
        body = CLIENT.get(path).text
        assert FORMAL_NAME in body
        assert OLD_NAME not in body
        assert "Nikkei 225 Forecast &amp; Strategy Support System" in body


def test_nikkei_result_renderer_uses_formal_name_and_accuracy_heading() -> None:
    source = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
    renderer = source.split("const renderNikkeiPublicPrediction", 1)[1].split("const renderSimulation", 1)[0]
    assert FORMAL_NAME in renderer
    assert "予測精度" in renderer
    assert "方向予測の過去評価" not in renderer


def test_individual_waiting_guidance_is_only_rendered_on_individual_page() -> None:
    individual = CLIENT.get("/individual").text
    assert "分析実施中" in individual
    assert WAITING_GUIDANCE in individual
    for path in ("/nikkei", "/simulation"):
        assert WAITING_GUIDANCE not in CLIENT.get(path).text


def test_current_walk_forward_and_python_documentation_are_public() -> None:
    assert "8-Fold Walk-Forward" in CLIENT.get("/").text
    about = CLIENT.get("/about").text
    assert "直近2年間を対象とした8分割ウォークフォワード検証" in about
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Python 3.13.x" in readme
    assert "Python 3.11" not in readme
    assert "py -3.11" not in readme
    assert "START_HERE.cmd" in readme


def test_long_formal_name_and_waiting_text_have_mobile_overflow_guards() -> None:
    css = (ROOT / "static" / "css" / "style.css").read_text(encoding="utf-8")
    assert "@media (max-width: 700px)" in css
    assert ".brand > span:last-child" in css
    assert ".individual-loading-guidance" in css
    assert "overflow-wrap: anywhere" in css



def test_individual_accuracy_artifacts_are_not_bundled() -> None:
    assert not (ROOT / "outputs" / "individual_evaluations").exists()
    assert not (ROOT / "model_settings" / "individual_evaluations").exists()
    assert not (ROOT / "tools" / "check_individual_evaluation.py").exists()
    assert not (ROOT / "tools" / "update_individual_logistic_evaluation.py").exists()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "個別銘柄ごとの予測精度や方向一致率は公開・表示しません" in readme
