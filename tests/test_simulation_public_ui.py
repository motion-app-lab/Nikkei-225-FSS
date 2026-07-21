from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as app_module
from services.simulation_service import SIMULATION_SCHEMA_VERSION, _result_summary


ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app_module.app)


def test_simulation_page_has_four_inputs_and_separate_code_examples() -> None:
    html = client.get("/simulation").text
    assert "利確・損切りシミュレーション" in html
    assert "日本株の証券コード" in html
    assert "入力例（いずれか1銘柄）" in html
    assert "<code>7203</code>" in html and "<code>130A</code>" in html
    assert "7203、130A" not in html
    assert ".Tは不要です。日本株の証券コードを1銘柄だけ入力してください。" in html
    assert 'name="buy_on_up"' not in html
    assert "data-disable-target" not in html
    assert "20260721-simulation-100share-v1" in html


def test_simulation_defaults_and_optional_condition_explanation() -> None:
    html = client.get("/simulation").text
    assert 'name="take_profit" type="number" value="10"' in html
    assert 'name="stop_loss" type="number" value="4"' in html
    assert 'class="rate-row simulation-rate-stack"' in html
    help_text = "数値を入力した場合、この条件を使用します。使用しない場合は空欄にしてください。"
    assert html.count(help_text) == 1
    assert html.index(help_text) > html.index('id="stop-loss"')
    assert "最大の100株単位で仮想購入します" in html
    assert "購入に使用しなかった資金は、現金として次回の取引へ繰り越します" in html


def test_public_simulation_renderer_uses_neutral_metrics_and_no_old_signal_ui() -> None:
    source = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
    block = source.split("const renderSimulation =", 1)[1].split("const wireSimulationChartTooltips", 1)[0]
    for required in (
        "仮想最終資産", "仮想損益額", "総損益率", "決済済み取引回数",
        "利益になった取引の割合", "最大下落率", "同期間保有の総損益率", "同期間保有との差",
        "データ収集期間", "株価推移と仮想売買ポイント",
        "売買単位：100株", "購入方法：各購入時点で購入可能な最大株数", "仮想購入株数",
    ):
        assert required in source
    for forbidden in ("勝率", "最大ドローダウン", "上昇予測時の買い", "profitClass", "profitSymbol", "trend-up", "trend-down", "simulation-count-summary"):
        assert forbidden not in block
    assert "全${formatInteger(result.trade_count)}回の決済済み取引" not in block


def test_virtual_purchase_and_settlement_chart_markers_have_touch_and_keyboard_details() -> None:
    source = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert "simulation-purchase-marker" in source
    assert "simulation-settlement-marker" in source
    assert 'tabindex="0"' in source
    assert "data-chart-tooltip" in source
    assert "仮想購入日" in source and "仮想決済日" in source
    assert "point.entry_shares" in source and "point.cash_after_entry" in source and "point.exit_shares" in source
    assert "利益確定条件" in source and "損切り条件" in source and "検証期間終了" not in source.split("const simulationPriceChart", 1)[1].split("const trendMeta", 1)[0]


def test_simulation_specific_css_stacks_conditions_and_distinguishes_charts() -> None:
    css = (ROOT / "static" / "css" / "style.css").read_text(encoding="utf-8")
    assert ".simulation-result strong" in css and "color: var(--text)" in css
    assert ".simulation-rate-stack" in css and "grid-template-columns: minmax(0, 1fr)" in css
    assert ".simulation-line-strategy { stroke: #5db7c9; stroke-width: 2.1" in css
    assert ".simulation-line-hold { stroke: #9e98ca; stroke-width: 1.55; stroke-dasharray: 6 4" in css
    assert ".simulation-purchase-marker { fill: #3aa8b8; stroke: #d8f3f7" in css
    assert ".simulation-settlement-marker { fill: #c79255; stroke: #f7e6cc" in css


def test_period_cards_are_data_collection_then_simulation_in_dom_order() -> None:
    source = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
    block = source.split('<div class="simulation-period-grid">', 1)[1].split('<div class="metrics-grid simulation-metrics">', 1)[0]
    assert block.index("データ収集期間") < block.index("検証期間")


def test_trade_profit_cells_use_scoped_sign_classes_and_rounded_zero_is_neutral() -> None:
    source = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
    css = (ROOT / "static" / "css" / "style.css").read_text(encoding="utf-8")
    assert "simulationPnlClass(trade.profit_amount, 0)" in source
    assert "simulationPnlClass(trade.profit_percent, 1)" in source
    assert 'return "simulation-pnl-neutral"' in source
    assert ".simulation-trade-table .simulation-pnl-positive" in css
    assert ".simulation-trade-table .simulation-pnl-negative" in css
    assert ".simulation-trade-table .simulation-pnl-neutral" in css


def test_purchase_and_settlement_markers_use_different_svg_shapes() -> None:
    source = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
    block = source.split("const simulationPriceChart", 1)[1].split("const trendMeta", 1)[0]
    assert block.count('<polygon points="${points}"') == 2
    assert "y - 5.5" in block and "x + 5" in block
    assert "y - 7" not in block and "x + 6.5" not in block


def test_result_summary_keeps_counts_but_omits_asset_amounts() -> None:
    result = {
        "trade_count": 71,
        "profitable_trades": 24,
        "losing_trades": 47,
        "break_even_trades": 0,
        "initial_assets": 1_000_000,
        "final_assets": 1_316_765,
        "buy_hold_difference_points": 3.2,
    }
    summary = _result_summary(result, 10.0, 4.0)
    assert "71回" in summary and "24回" in summary and "損失は47回" in summary and "損益ゼロは0回" in summary
    assert "1,000,000" not in summary and "1,316,765" not in summary
    assert "初期資金" not in summary and "仮想的に" not in summary
    assert "最大の100株単位" in summary and "100株単位で保有し続けた場合" in summary


def test_simulation_error_never_returns_previous_result() -> None:
    response = client.post(
        "/api/simulate",
        json={"ticker": "8035.T", "initial_investment": 1_000_000, "take_profit": 10, "stop_loss": 4},
    )
    assert response.status_code == 422
    assert response.json()["last_result"] is None


def test_empty_both_conditions_is_rejected_without_previous_result() -> None:
    response = client.post(
        "/api/simulate",
        json={"ticker": "7203", "initial_investment": 1_000_000, "take_profit": None, "stop_loss": None},
    )
    assert response.status_code == 422
    body = response.json()
    assert "どちらか" in body["error"]["message"]
    assert body["last_result"] is None


@pytest.mark.parametrize("code", ["7203", "130a"])
def test_api_accepts_supported_japanese_code_forms(monkeypatch: pytest.MonkeyPatch, code: str) -> None:
    def fake_strategy(**kwargs):
        return {"kind": "simulation", "schema_version": SIMULATION_SCHEMA_VERSION, "security_code": kwargs["ticker"].upper()}

    monkeypatch.setattr(app_module, "simulate_strategy", fake_strategy)
    monkeypatch.setattr(app_module, "save_last_result", lambda *_: None)
    response = client.post(
        "/api/simulate",
        json={"ticker": code, "initial_investment": 1_000_000, "take_profit": 10, "stop_loss": 4},
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_old_simulation_schema_is_not_returned_as_saved_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "load_last_result", lambda _: {"schema_version": "old", "kind": "simulation"})
    assert client.get("/api/last/simulation").status_code == 404


def test_new_request_clears_old_result_and_stale_response_is_ignored() -> None:
    source = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert 'form.id === "simulation-form"' in source
    assert "clearStaleResult();" in source
    assert "AbortController" in source
    assert "currentGeneration !== requestGeneration" in source
    assert 'form.id !== "simulation-form"' in source


def test_simulation_disclaimers_are_present_in_service() -> None:
    source = (ROOT / "services" / "simulation_service.py").read_text(encoding="utf-8")
    assert "このシミュレーションは、各購入時点で購入可能な最大の100株単位を用い" in source
    assert "売買手数料、税金、配当、スリッページ、市場流動性は計算に含めていません。" in source
    assert "実際の株数や単元株は考慮していません" not in source
