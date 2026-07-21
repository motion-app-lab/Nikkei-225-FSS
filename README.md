# Nikkei 225 Forecast & Strategy Support System

**OpenAI Build Week 2026 submission by Komukai**

A local Windows web application for:

1. forecasting the direction of the Nikkei 225 five Japanese trading sessions ahead,
2. forecasting the direction of one Japanese stock five trading sessions ahead, and
3. testing user-defined take-profit and stop-loss rules on historical stock-price data.

The application combines Python, FastAPI, Jinja2, HTML/CSS, Vanilla JavaScript, scikit-learn, CatBoost, pandas, Matplotlib, and market data obtained through `yfinance`. It runs locally on `127.0.0.1` and does not expose the server to the public network.

> **Interface language:** The current application interface is Japanese. This README provides the English setup, testing, technical, and Build Week documentation.

## Important interpretation

This project is a research and analysis support tool. It does **not** provide investment advice, recommend a purchase or sale, or guarantee future returns.

The forecast features estimate **direction**, not a future price target or profit amount. The six displayed levels are internal movement-range classes and are not confidence levels or recommendation strength.

The take-profit / stop-loss simulator is separate from the forecast models. It mechanically applies the user's exit conditions to historical OHLC data and compares the result with holding the same stock over the same period.

## Main features

### 1. Nikkei 225 five-session direction forecast

The Nikkei page forecasts whether the confirmed Nikkei 225 closing price will be higher or lower five Japanese trading sessions later.

Key characteristics:

- Uses confirmed daily bars only.
- During Japanese market hours, uses the previous confirmed Japanese close.
- After the close, uses the current day's bar only when it is confirmed and available.
- Does not use intraday bars or unfinished daily bars.
- Produces separate upward scores from:
  - a Japan-only model, and
  - a US / overseas model.
- Combines the two scores with a fixed 50 / 50 weight.
- Shows an independent six-class model output in this fixed order:
  - Upward Lv.3
  - Upward Lv.2
  - Upward Lv.1
  - Downward Lv.1
  - Downward Lv.2
  - Downward Lv.3
- Displays:
  - the five-session direction result,
  - a 60-session chart and short-term report,
  - a two-year chart and medium- to long-term report,
  - selected and excluded factors,
  - feature importance,
  - historical directional evaluation,
  - model and time-alignment details.

The upward score is the model's strength for the current input. It is not historical accuracy and should not be interpreted as a literal probability that the market will rise.

### 2. Individual Japanese stock forecast

The individual-stock page accepts one Japanese security code at a time, such as `7203` or `130A`. The `.T` suffix is added internally.

Key characteristics:

- Accepts one Japanese stock code without `.T`.
- Rejects overseas tickers, multiple codes, and `.T`-suffixed input.
- Uses the selected stock's historical data together with broader market data available by the prediction timestamp.
- Uses fixed logistic-regression configurations for the normal prediction path.
- Fits one binary direction model and one six-class movement model.
- Shows the six classes in the same fixed order used by the Nikkei page.
- Displays:
  - the current model output,
  - a 60-session short-term chart,
  - a two-year medium- to long-term chart,
  - deterministic reports for both periods,
  - the market factors used for the current forecast,
  - feature importance.

**Individual-stock accuracy metrics are intentionally not displayed or bundled.** The page focuses on the current model output, recent chart behavior, longer-term chart behavior, and the market factors behind the forecast.

The chart reports are generated from deterministic, safety-checked templates. They do not call an external large language model and do not provide trade timing advice.

### 3. Take-profit / stop-loss simulator

The simulator does not use the Nikkei or individual-stock forecast models.

Inputs:

- one Japanese security code without `.T`,
- initial investment amount,
- optional take-profit percentage,
- optional stop-loss percentage.

Default example:

- stock code: `7203`
- initial investment: JPY 1,000,000
- take profit: 10%
- stop loss: 4%

Simulation rules include:

- long-only trading,
- purchases in the largest possible 100-share lot,
- uninvested cash carried forward,
- entry at the first available opening price,
- re-entry at the next available opening price after an exit,
- gap exits executed at the opening price,
- intraday exits executed at the threshold price,
- stop loss taking priority when both thresholds are reached on the same day,
- stock splits and reverse splits reflected in the position,
- final open position closed at the last available close,
- comparison with holding the same stock over the same period using the same lot and cash assumptions,
- maximum drawdown calculated from the daily marked-to-market portfolio value.

The simulator does not include commissions, taxes, dividends, slippage, or market-liquidity effects.

## Quick start for judges — Windows 10 / 11

### Requirements

- Windows 10 or Windows 11
- Python 3.13.x
- Internet access for installing dependencies and retrieving market data

### Start the application

1. Download the ZIP file.
2. Right-click the ZIP and select **Extract All**.
3. Open the extracted `japanese-stock-strategy-app` folder.
4. Double-click `START_HERE.cmd`.
5. Keep the command window open while using the application.
6. On the first run, wait while the local virtual environment and dependencies are installed.
7. The browser should open automatically at:

```text
http://127.0.0.1:8000
```

To stop the application, press `Ctrl + C` in the command window or close that window.

> Do not run `START_HERE.cmd` from inside the ZIP. Extract the project first.

### Suggested test flow

1. Open **Nikkei 225 Trend Forecast** and run the normal forecast.
2. Open **Individual Stock Forecast**, enter `7203`, and run the analysis.
3. Open **Take-Profit / Stop-Loss Simulation**, use `7203`, JPY 1,000,000, 10% take profit, and 4% stop loss.

The normal Nikkei forecast uses the bundled saved model. Do not run the explicit model reevaluation during a short judging session unless a full re-evaluation is specifically required.

The individual-stock request has a finite overall processing limit of 180 seconds. External market-data availability can affect completion time.

## Manual setup

From PowerShell or Command Prompt in the project folder:

```powershell
python --version
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

The application is then available at:

```text
http://127.0.0.1:8000
```

## Optional FRED API key

Copy `.env.example` to `.env` and add a key only when required by an optional data path:

```dotenv
FRED_API_KEY=your_fred_api_key_here
```

The current Nikkei direction model does not retrieve PCE, regardless of whether a FRED key is present, because the project does not attempt to reconstruct historical macroeconomic release availability without a reliable vintage-data process.

The `.env` file is excluded from version control.

## Technology stack

- Python 3.13.x
- FastAPI
- Uvicorn
- Jinja2
- HTML / CSS / Vanilla JavaScript
- pandas
- NumPy
- scikit-learn
- CatBoost
- joblib
- yfinance
- fredapi
- Matplotlib using the non-interactive Agg backend
- pytest

Prophet and TensorFlow are not used in the final prediction path because they did not provide a meaningful contribution to the final system.

## Project structure

```text
japanese-stock-strategy-app/
├─ app.py
├─ START_HERE.cmd
├─ RUN_APP_INNER.cmd
├─ requirements.txt
├─ services/
│  ├─ common.py
│  ├─ nikkei_dual_market.py
│  ├─ nikkei_dual_model.py
│  ├─ nikkei_service.py
│  ├─ individual_chart_report.py
│  ├─ individual_market.py
│  ├─ individual_logistic_fast.py
│  ├─ individual_service.py
│  └─ simulation_service.py
├─ model_settings/
│  ├─ nikkei_dual_market.json
│  ├─ nikkei_dual_market_models.joblib
│  ├─ nikkei_dual_market_history.json
│  ├─ nikkei_after_close_evaluation.json
│  ├─ nikkei_intraday_evaluation.json
│  └─ nikkei_model_manifest.json
├─ templates/
├─ static/
├─ tests/
├─ tools/
├─ legacy_original/
├─ BUILD_WEEK_NOTES.md
├─ DEMO_SCRIPT.md
└─ SUBMISSION_NOTES.md
```

Runtime output files are created locally when the application runs and are not bundled in the submission ZIP.

## Application routes and API endpoints

Pages:

```text
GET  /
GET  /nikkei
GET  /individual
GET  /simulation
GET  /about
```

API:

```text
GET  /health
POST /api/predict/nikkei
POST /api/predict/individual
POST /api/simulate
GET  /api/last/{kind}
```

Interactive FastAPI documentation is available locally at:

```text
http://127.0.0.1:8000/api/docs
```

## Data sources

Main Yahoo Finance symbols used by the project include:

- Japan: `^N225`
- US equities: `^GSPC`, `^DJI`, `^IXIC`, `^VIX`, `^SOX`, `NVDA`
- Other markets: `JPY=X`, `CL=F`, `GC=F`, `BTC-USD`

The simulator uses the selected Japanese stock's unadjusted daily OHLC data and stock-split information.

Yahoo Finance data is not an exchange-certified feed, and availability or revisions may affect runtime results.

## Market-time alignment and leakage prevention

External series are not first forced onto the Nikkei calendar and then shifted uniformly. Instead, the project:

1. keeps each series on its original trading or observation calendar,
2. calculates returns, moving-average distance, and volatility on that original calendar,
3. assigns a timezone-aware availability timestamp,
4. joins only the latest row available before the Japanese prediction timestamp with a backward as-of join,
5. prohibits future-direction filling, backward filling from future data, and nearest-time matching that could select a future value.

US equity data uses 16:15 `America/New_York` as its availability time, including daylight-saving-time conversion.

Because exact daily close availability is less certain for FX, crude oil, gold, and Bitcoin, those series receive a conservative one-source-session delay.

## Nikkei model design

The binary target is whether the Nikkei 225 close five Japanese trading sessions later is above the current confirmed close.

The Japan-side and US / overseas-side models produce separate upward scores:

```text
final upward score = Japan-side score × 0.50
                   + US / overseas score × 0.50
```

The 50 / 50 weight is fixed and was not optimized after reviewing the formal evaluation result.

Candidate model families include:

- CatBoost
- CatBoost with feature selection performed inside the training fold
- Logistic Regression
- Extra Trees
- a simple average of CatBoost, Logistic Regression, and Extra Trees

The primary model-selection metric is Balanced Accuracy. Macro-F1, ordinary accuracy, worst-fold performance, fold-to-fold stability, and model simplicity are also considered.

## Rolling training and formal Nikkei evaluation

For each prediction point:

- training uses the preceding rolling eight calendar years,
- an additional 300 Japanese market sessions are used only to warm up long-horizon features,
- warm-up rows are not counted as training or evaluation samples.

Formal evaluation uses the latest two years for which the five-session outcomes are already known.

The two-year period is divided into eight outer folds. Each outer training period contains three inner folds for model, feature, weighting, and decision-threshold selection. A five-Japanese-session purge is applied at the inner and outer boundaries.

The outer evaluation results are not used to modify the settings being evaluated.

Two predeclared training-weight choices are compared:

- equal weighting,
- four-year half-life time-decay weighting.

Time decay is adopted only when it passes predefined improvement and safety conditions; otherwise the system returns to equal weighting.

The combined score compares thresholds from 40% to 60% in two-point increments inside the training folds. If predefined improvement and prediction-balance conditions are not met, the operating threshold returns to 50%.

## Bundled Nikkei evaluation result

The explicit reevaluation completed on 2026-07-19 produced the following bundled result:

- source data: 2015-04-10 to 2026-07-17, 2,753 rows
- feature warm-up: 2015-04-10 to 2016-07-01, 300 Japanese sessions
- final model training: 2018-07-10 to 2026-07-10, 1,952 samples
- formal evaluation: 2024-07-10 to 2026-07-10, 488 samples under the intraday condition
- Japan-side model: Extra Trees, 40 features, equal weighting
- US / overseas model: training-fold feature-selection CatBoost, 20 features, equal weighting
- operating threshold: 50%
- directional accuracy: 54.71% — 267 correct out of 488
- Balanced Accuracy: 53.36%
- Macro-F1: 53.36%
- upward recall: 62.28%
- downward recall: 44.44%
- majority-direction baseline: 57.58%
- recent-five-session continuation baseline: 46.11%
- difference from the strongest baseline: -2.87 percentage points
- directional-accuracy 95% confidence interval: 48.77% to 60.86%
- baseline-difference 95% confidence interval: -11.07 to +4.72 percentage points

The model did not outperform the majority baseline in this formal evaluation. The application and documentation report this result rather than hiding or relabeling it.

## Individual-stock model path

The normal individual-stock forecast:

- uses up to eight years of history before the prediction date,
- uses up to 300 earlier sessions for feature preparation,
- combines stock-specific normalized features, Nikkei features, and timestamp-safe overseas market features,
- uses one fixed binary Logistic Regression model configuration,
- uses one fixed six-class Logistic Regression model configuration,
- fits preprocessing only on the training data,
- reuses the same transformed feature matrix for both models,
- uses a common 50% binary direction boundary,
- does not perform cross-validation, candidate-model comparison, stock-specific threshold optimization, or stock-specific accuracy evaluation during a normal request.

The fixed configuration uses:

```text
solver=lbfgs
C=1.0
max_iter=300
tol=1e-4
class_weight=balanced
```

If a compatible saved calibrator is unavailable, the system uses raw `predict_proba` output and records internally that calibration was not applied.

## Deterministic chart reports

The 60-session report may include, when sufficient data exists:

- 5-, 20-, and 60-session changes,
- maximum decline from an earlier high to a later low,
- recovery from that low to the current close,
- distance from the 60-session high and low,
- current position inside the 60-session range,
- recent average absolute daily movement,
- difference from the Nikkei 225,
- volume statistics.

The two-year report may include:

- six-month, one-year, and two-year changes,
- position relative to the period high and low,
- major rising and falling phases,
- the relationship between the 60- and 200-session moving averages,
- difference from the market index.

These reports describe historical chart facts. They do not explain causality, forecast the future, or recommend a trade.

## Testing

Run the local automated test suite:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Run Python compilation checks:

```powershell
.venv\Scripts\python.exe -m compileall app.py services tests
```

The submission package was checked with 291 automated tests, with zero failures, before the English README was added. The application code and model artifacts were not changed while adding this documentation.

Coverage includes:

- page and API responses,
- forecast classification boundaries,
- rolling training windows,
- warm-up separation,
- inner and outer purging,
- Japanese and US holiday differences,
- US daylight-saving-time transitions,
- backward as-of joining,
- normal prediction versus explicit model reevaluation,
- individual-stock timeout and stale-result handling,
- deterministic short- and long-term reports,
- exclusion of individual-stock accuracy from the public UI,
- simulator lot sizing, gap handling, same-day threshold handling, re-entry, final liquidation, splits, drawdown, and hold comparison,
- rejection of outdated result schemas,
- preservation checks for the original command-line files.

Network-dependent live-data checks are kept separate from ordinary unit tests.

## Codex and GPT-5.6 collaboration

Codex and GPT-5.6 were used as development collaborators. They are not called by the application at runtime.

### How Codex accelerated the project

Codex was used to:

- inspect and preserve the three original command-line programs,
- integrate the three functions into one FastAPI application,
- create and refine API routes, templates, JavaScript interactions, and Windows launchers,
- identify potential future-data leakage and timestamp-alignment problems,
- implement rolling evaluation, inner / outer time-series folds, purging, and persisted model settings,
- separate ordinary inference from explicit model reevaluation,
- redesign the simulator so it tests only user-defined exit rules rather than forecast signals,
- implement deterministic individual-stock chart reports,
- diagnose and fix the individual-stock request that could remain indefinitely in an analyzing state,
- add finite data-download attempts and an overall processing timeout,
- build regression tests for the final behavior,
- prepare the release package and submission documentation.

### How GPT-5.6 contributed

GPT-5.6 was used to:

- refine the product concept and user-facing scope,
- translate the creator's requirements into precise technical specifications,
- compare design alternatives and identify misleading financial-language risks,
- review calculation rules and edge cases,
- distinguish forecast output, historical evaluation, chart analysis, and simulation results,
- refine the six-level public labels and explanatory text,
- review the final interface and demo narrative,
- help prepare the English submission materials.

### Human product and engineering decisions

The creator personally decided:

- the three-function product concept,
- the Japanese local-app experience,
- the exact input and output design,
- the five-trading-session horizon,
- the separation of the forecast features from the simulator,
- the simulator's 100-share-lot and cash-handling rules,
- the use of confirmed daily data only,
- the decision not to display individual-stock accuracy metrics,
- the neutral presentation of results without red / green persuasion or profit claims,
- the requirement to display weak evaluation results rather than optimize the story after seeing them,
- the final acceptance or rejection of every implementation change.

The development workflow was iterative: the creator specified behavior, tested each build, identified mismatches, and made the final product, design, and risk-communication decisions; Codex generated and revised implementation work under those constraints.

## Build Week improvements

The following work was completed during Build Week:

- integrated three independent command-line programs into one local FastAPI web application,
- preserved the original programs under `legacy_original/`,
- separated the latest inference row from rows with known training labels,
- separated the binary direction model from the six-class movement model,
- redesigned Japanese and overseas market-time alignment,
- added rolling eight-year training and 300-session feature warm-up,
- added two-year, eight-outer-fold Nikkei evaluation with three inner folds and purging,
- added safe comparison between equal and time-decay training weights,
- persisted models, selected settings, model versions, and reevaluation history,
- added deterministic 60-session and two-year reports,
- removed individual-stock accuracy from the final public output,
- removed forecast-model dependence from the take-profit / stop-loss simulator,
- added Windows one-click setup and startup handling,
- added error messages, stale-result rejection, request timeouts, and regression tests,
- added English setup, testing, and collaboration documentation for judging.

## Original command-line programs

The original pre-Build-Week programs are retained unchanged in `legacy_original/`:

- `nikkei_no2.py`
- `nikkei_kobetu_no1.py`
- `sim_research.py`

The Build Week work integrated and extended these functions without replacing the preserved originals.

## Security and privacy

- The server binds only to `127.0.0.1`.
- API keys are not hard-coded into the application.
- `.env` is excluded from version control.
- The application does not send source code or analysis text to an external generative-AI API at runtime.
- Model and result files are stored locally.
- CatBoost file output is disabled where it is used by the current application path.

## Known limitations

- Market data is obtained from third-party services and may be delayed, revised, missing, or temporarily unavailable.
- The Japanese daily bar may not be available immediately after the market close.
- Conservative source-session lags are applied to some non-equity series.
- Adjacent five-session targets overlap, so evaluation observations are not fully independent.
- Historical evaluation does not guarantee future performance.
- The Nikkei model did not beat the strongest simple baseline in the bundled formal evaluation.
- The individual-stock page intentionally does not provide stock-specific accuracy metrics.
- The simulator omits real-world costs and execution effects.
- The application interface is currently Japanese.

## Reproducible model environment

The bundled Nikkei model artifacts were created with Python 3.12.13. `requirements.txt` pins the direct dependencies used by the measured model environment.

The Windows launcher accepts Python 3.13.x patch releases, recreates an incompatible local `.venv`, installs the pinned dependencies, and always starts the application with `.venv\Scripts\python.exe`.

Before loading the bundled artifact, the model manifest validates the supported Python series and model-sensitive library versions.

## Disclaimer

This system is intended for research, analysis, and informational use only. Its forecasts and simulations do not constitute financial advice, do not recommend buying or selling any financial instrument, and do not guarantee future performance. All investment decisions remain the user's responsibility.
