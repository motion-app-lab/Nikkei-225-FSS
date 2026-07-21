from __future__ import annotations

import asyncio
import logging
import os
import threading
import webbrowser
from functools import partial
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from services.common import BASE_DIR, OUTPUT_DIR, ServiceError, load_last_result, save_last_result
from services.individual_service import (
    INDIVIDUAL_MAX_PROCESSING_SECONDS,
    INDIVIDUAL_UI_SCHEMA_VERSION,
    IndividualExecutionControl,
    is_current_individual_result,
    predict_individual,
)
from services.nikkei_service import is_current_nikkei_result, predict_nikkei
from services.simulation_service import is_current_simulation_result, simulate_strategy


APP_NAME = "日経平均株価予測・戦略支援システム"
logger = logging.getLogger(__name__)
DISCLAIMER = (
    "本システムの予測およびシミュレーション結果は、情報提供および研究目的の参考情報です。"
    "特定の金融商品の売買を推奨するものではなく、利益を保証するものでもありません。"
    "投資判断は利用者自身の責任で行ってください。"
)

app = FastAPI(title=APP_NAME, version="1.0.0", docs_url="/api/docs", redoc_url=None)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


class IndividualRequest(BaseModel):
    ticker: str = Field(default="7203", min_length=1, max_length=20)


class NikkeiRequest(BaseModel):
    model_reevaluation: bool = False
    # 旧クライアントとの互換性。Trueの場合は明示的なモデル再評価として扱う。
    force_refresh: bool = False


class SimulationRequest(BaseModel):
    ticker: str = Field(default="7203", min_length=1, max_length=20)
    initial_investment: float = Field(default=1_000_000, gt=0, allow_inf_nan=False)
    take_profit: float | None = Field(default=10, gt=0, allow_inf_nan=False)
    stop_loss: float | None = Field(default=4, gt=0, allow_inf_nan=False)


def page_context(request: Request, page: str) -> dict:
    return {
        "request": request,
        "page": page,
        "app_name": APP_NAME,
        "disclaimer": DISCLAIMER,
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    fields = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", []) if item != "body")
        fields.append(location or "入力値")
    return JSONResponse(
        status_code=422,
        content={
            "ok": False,
            "error": {
                "message": "入力内容を確認できませんでした。",
                "action": f"{', '.join(dict.fromkeys(fields))} の値と範囲を確認してください。",
            },
        },
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context=page_context(request, "home"))


@app.get("/nikkei", response_class=HTMLResponse)
async def nikkei_page(request: Request):
    return templates.TemplateResponse(request=request, name="nikkei.html", context=page_context(request, "nikkei"))


@app.get("/individual", response_class=HTMLResponse)
async def individual_page(request: Request):
    return templates.TemplateResponse(request=request, name="individual.html", context=page_context(request, "individual"))


@app.get("/simulation", response_class=HTMLResponse)
async def simulation_page(request: Request):
    return templates.TemplateResponse(request=request, name="simulation.html", context=page_context(request, "simulation"))


@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse(request=request, name="about.html", context=page_context(request, "about"))


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": APP_NAME,
        "project_root": str(BASE_DIR.resolve()),
    }


def _error_response(error: Exception, cache_key: str, include_previous: bool = True) -> JSONResponse:
    previous = load_last_result(cache_key) if include_previous else None
    if isinstance(error, ServiceError):
        message = error.message
        action = error.action
        status_code = error.status_code
    else:
        message = "分析処理を完了できませんでした。"
        action = "入力値と通信状態を確認し、時間をおいて再度お試しください。"
        status_code = 500
    return JSONResponse(
        status_code=status_code,
        content={
            "ok": False,
            "error": {"message": message, "action": action},
            "last_result": previous,
        },
    )


@app.post("/api/predict/nikkei")
async def api_predict_nikkei(payload: NikkeiRequest):
    try:
        # 結果キャッシュの更新とモデル設定の再選択は別操作。明示指定だけを再評価として扱う。
        model_reevaluation = bool(payload.model_reevaluation)
        result = await run_in_threadpool(predict_nikkei, model_reevaluation)
        save_last_result("nikkei", result)
        return {"ok": True, "result": result}
    except Exception as error:
        return _error_response(error, "nikkei", include_previous=False)


@app.post("/api/predict/individual")
async def api_predict_individual(payload: IndividualRequest):
    control = IndividualExecutionControl(INDIVIDUAL_MAX_PROCESSING_SECONDS)
    try:
        task = partial(predict_individual, payload.ticker, execution_control=control)
        result = await asyncio.wait_for(
            run_in_threadpool(task),
            timeout=INDIVIDUAL_MAX_PROCESSING_SECONDS,
        )
        save_last_result("individual", result)
        return {"ok": True, "result": result}
    except TimeoutError:
        control.cancel("api_timeout")
        logger.error(
            "individual prediction API timeout ticker=%s elapsed=%.3fs",
            payload.ticker,
            control.elapsed(),
        )
        return _error_response(
            ServiceError(
                "個別銘柄予測の計算が制限時間を超えたため終了しました。",
                "データ取得状況を確認して、時間をおいてもう一度実行してください。",
                504,
            ),
            "individual",
            include_previous=False,
        )
    except asyncio.CancelledError:
        control.cancel("client_disconnected_or_request_cancelled")
        raise
    except ServiceError as error:
        logger.warning(
            "individual prediction stopped ticker=%s status=%s elapsed=%.3fs message=%s",
            payload.ticker,
            error.status_code,
            control.elapsed(),
            error.message,
        )
        return _error_response(error, "individual", include_previous=False)
    except Exception as error:
        logger.exception(
            "individual prediction API failed ticker=%s elapsed=%.3fs",
            payload.ticker,
            control.elapsed(),
        )
        return _error_response(error, "individual", include_previous=False)


@app.post("/api/simulate")
async def api_simulate(payload: SimulationRequest):
    try:
        task = partial(
            simulate_strategy,
            ticker=payload.ticker,
            initial_investment=payload.initial_investment,
            take_profit=payload.take_profit,
            stop_loss=payload.stop_loss,
        )
        result = await run_in_threadpool(task)
        save_last_result("simulation", result)
        return {"ok": True, "result": result}
    except Exception as error:
        # 入力・通信・計算エラー時に、以前の正常結果は返さない。
        return _error_response(error, "simulation", include_previous=False)


@app.get("/api/last/{kind}")
async def api_last_result(kind: Literal["nikkei", "individual", "simulation"]):
    result = load_last_result(kind)
    if kind == "nikkei" and result is not None and not is_current_nikkei_result(result):
        result = None
    if kind == "individual" and result is not None:
        required = ("chart_analysis", "long_chart_analysis", "six_stage_trend", "long_chart_url")
        if not is_current_individual_result(result) or not all(
            key in result for key in required
        ):
            result = None
    if kind == "simulation" and result is not None and not is_current_simulation_result(result):
        result = None
    if result is None:
        return JSONResponse(
            status_code=404,
            content={"ok": False, "error": {"message": "保存済みの正常結果はありません。", "action": "一度分析を実行してください。"}},
        )
    return {"ok": True, "result": result, "cached": True}


def _open_browser_once() -> None:
    if os.getenv("STOCK_APP_OPEN_BROWSER", "1") != "0":
        webbrowser.open("http://127.0.0.1:8000")


if __name__ == "__main__":
    import uvicorn

    threading.Timer(1.2, _open_browser_once).start()
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
