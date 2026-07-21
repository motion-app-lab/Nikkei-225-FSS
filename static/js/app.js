(() => {
  "use strict";

  const INDIVIDUAL_REQUEST_TIMEOUT_MS = 190_000;

  const escapeHtml = (value) => String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

  const numberFormatter = new Intl.NumberFormat("ja-JP", { maximumFractionDigits: 2 });
  const integerFormatter = new Intl.NumberFormat("ja-JP", { maximumFractionDigits: 0 });
  const formatNumber = (value) => Number.isFinite(Number(value)) ? numberFormatter.format(Number(value)) : "—";
  const formatInteger = (value) => Number.isFinite(Number(value)) ? integerFormatter.format(Number(value)) : "—";
  const formatPercent = (value, digits = 1) => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(digits)}%` : "—";
  const formatRawPercent = (value, digits = 1) => Number.isFinite(Number(value)) ? `${Number(value).toFixed(digits)}%` : "—";
  const formatSignedPoints = (value, digits = 1) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return "—";
    return `${numeric >= 0 ? "+" : "−"}${Math.abs(numeric * 100).toFixed(digits)}ポイント`;
  };
  const formatSignedCurrency = (value) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return "—";
    if (Math.round(numeric) === 0) return "¥0";
    return `${numeric >= 0 ? "+" : "−"}¥${formatInteger(Math.abs(numeric))}`;
  };
  const formatCurrency = (value) => Number.isFinite(Number(value)) ? `¥${formatInteger(value)}` : "—";
  const formatSignedRawPercent = (value, digits = 1) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return "—";
    if (Number(numeric.toFixed(digits)) === 0) return `${(0).toFixed(digits)}%`;
    return `${numeric >= 0 ? "+" : "−"}${Math.abs(numeric).toFixed(digits)}%`;
  };
  const simulationPnlClass = (value, digits) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return "simulation-pnl-neutral";
    const displayedValue = Number(numeric.toFixed(digits));
    if (displayedValue > 0) return "simulation-pnl-positive";
    if (displayedValue < 0) return "simulation-pnl-negative";
    return "simulation-pnl-neutral";
  };

  const simulationChartGeometry = (rows, keys, width = 900, height = 320) => {
    const padding = { left: 72, right: 24, top: 20, bottom: 48 };
    const values = rows.flatMap((row) => keys.map((key) => Number(row[key]))).filter(Number.isFinite);
    if (rows.length < 2 || values.length === 0) return null;
    let minimum = Math.min(...values);
    let maximum = Math.max(...values);
    const span = Math.max(maximum - minimum, Math.abs(maximum) * 0.04, 1);
    minimum -= span * 0.08;
    maximum += span * 0.08;
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const x = (index) => padding.left + (index / (rows.length - 1)) * plotWidth;
    const y = (value) => padding.top + ((maximum - Number(value)) / (maximum - minimum)) * plotHeight;
    const path = (key) => rows.map((row, index) => `${index ? "L" : "M"}${x(index).toFixed(2)},${y(row[key]).toFixed(2)}`).join(" ");
    return { width, height, padding, minimum, maximum, plotWidth, plotHeight, x, y, path };
  };

  const simulationAxisMarkup = (rows, geometry) => {
    const ticks = [0, 1, 2, 3, 4];
    const grid = ticks.map((tick) => {
      const ratio = tick / 4;
      const y = geometry.padding.top + ratio * geometry.plotHeight;
      const value = geometry.maximum - ratio * (geometry.maximum - geometry.minimum);
      return `<line x1="${geometry.padding.left}" y1="${y.toFixed(2)}" x2="${geometry.width - geometry.padding.right}" y2="${y.toFixed(2)}" class="simulation-chart-grid"/><text x="${geometry.padding.left - 10}" y="${(y + 4).toFixed(2)}" text-anchor="end" class="simulation-chart-axis-label">${escapeHtml(formatInteger(value))}</text>`;
    }).join("");
    const positions = [0, Math.floor((rows.length - 1) / 2), rows.length - 1];
    const labels = positions.map((index) => `<text x="${geometry.x(index).toFixed(2)}" y="${geometry.height - 16}" text-anchor="${index === 0 ? "start" : index === rows.length - 1 ? "end" : "middle"}" class="simulation-chart-axis-label">${escapeHtml(rows[index].date)}</text>`).join("");
    return `${grid}${labels}`;
  };

  const simulationAssetChart = (result) => {
    const rows = Array.isArray(result.equity_curve) ? result.equity_curve : [];
    const geometry = simulationChartGeometry(rows, ["strategy_assets", "buy_hold_assets", "initial_assets"]);
    if (!geometry) return "";
    const finalIndex = rows.length - 1;
    return `<section class="result-section simulation-chart-section" data-simulation-chart="assets">
      <div class="section-title"><h3>仮想資産の推移</h3><small>今回の条件と同期間保有を同じ基準で比較</small></div>
      <figure class="simulation-svg-chart">
        <svg viewBox="0 0 ${geometry.width} ${geometry.height}" role="img" aria-label="今回の条件、同期間保有、初期資金の推移">
          ${simulationAxisMarkup(rows, geometry)}
          <path d="${geometry.path("initial_assets")}" class="simulation-line-initial"/>
          <path d="${geometry.path("buy_hold_assets")}" class="simulation-line-hold"/>
          <path d="${geometry.path("strategy_assets")}" class="simulation-line-strategy"/>
          <circle cx="${geometry.x(finalIndex).toFixed(2)}" cy="${geometry.y(rows[finalIndex].buy_hold_assets).toFixed(2)}" r="4" class="simulation-final-marker simulation-final-marker-hold"><title>同期間保有の最終時点 ${rows[finalIndex].date} ${formatCurrency(rows[finalIndex].buy_hold_assets)}</title></circle>
          <circle cx="${geometry.x(finalIndex).toFixed(2)}" cy="${geometry.y(rows[finalIndex].strategy_assets).toFixed(2)}" r="4.8" class="simulation-final-marker simulation-final-marker-strategy"><title>今回の条件の最終時点 ${rows[finalIndex].date} ${formatCurrency(rows[finalIndex].strategy_assets)}</title></circle>
        </svg>
        <figcaption class="simulation-chart-legend"><span class="legend-strategy">今回の条件による仮想資産推移</span><span class="legend-hold">同期間保有の推移</span><span class="legend-initial">初期資金</span></figcaption>
      </figure>
    </section>`;
  };

  const simulationPriceChart = (result) => {
    const rows = Array.isArray(result.price_curve) ? result.price_curve : [];
    const geometry = simulationChartGeometry(rows, ["close"]);
    if (!geometry) return "";
    const dateIndex = new Map(rows.map((row, index) => [row.date, index]));
    const purchases = (result.purchase_points || []).map((point) => {
      const index = dateIndex.get(point.date);
      if (index == null || !Number.isFinite(Number(point.price))) return "";
      const tooltip = `仮想購入日 ${point.date} / 価格 ${formatNumber(point.price)}円 / 仮想購入株数 ${formatInteger(point.entry_shares)}株 / 購入後現金残高 ${formatCurrency(point.cash_after_entry)} / 取引${formatInteger(point.trade_number)}`;
      const x = geometry.x(index);
      const y = geometry.y(point.price);
      const points = `${x.toFixed(2)},${(y - 5.5).toFixed(2)} ${(x - 5).toFixed(2)},${(y + 4.2).toFixed(2)} ${(x + 5).toFixed(2)},${(y + 4.2).toFixed(2)}`;
      return `<polygon points="${points}" class="simulation-purchase-marker" tabindex="0" role="img" aria-label="${escapeHtml(tooltip)}" data-chart-tooltip="${escapeHtml(tooltip)}"><title>${escapeHtml(tooltip)}</title></polygon>`;
    }).join("");
    const settlements = (result.settlement_points || []).map((point) => {
      const index = dateIndex.get(point.date);
      if (index == null || !Number.isFinite(Number(point.price))) return "";
      const tooltip = `仮想決済日 ${point.date} / 価格 ${formatNumber(point.price)}円 / 仮想決済株数 ${formatNumber(point.exit_shares)}株 / ${point.reason} / 取引損益率 ${formatSignedRawPercent(point.profit_percent)}`;
      const x = geometry.x(index);
      const y = geometry.y(point.price);
      const points = `${x.toFixed(2)},${(y - 5).toFixed(2)} ${(x + 5).toFixed(2)},${y.toFixed(2)} ${x.toFixed(2)},${(y + 5).toFixed(2)} ${(x - 5).toFixed(2)},${y.toFixed(2)}`;
      return `<polygon points="${points}" class="simulation-settlement-marker" tabindex="0" role="img" aria-label="${escapeHtml(tooltip)}" data-chart-tooltip="${escapeHtml(tooltip)}"><title>${escapeHtml(tooltip)}</title></polygon>`;
    }).join("");
    return `<section class="result-section simulation-chart-section" data-simulation-chart="prices">
      <div class="section-title"><h3>株価推移と仮想売買ポイント</h3><small>対象銘柄の終値と仮想購入・仮想決済</small></div>
      <figure class="simulation-svg-chart">
        <svg viewBox="0 0 ${geometry.width} ${geometry.height}" role="img" aria-label="対象銘柄の終値と仮想売買ポイント">
          ${simulationAxisMarkup(rows, geometry)}
          <path d="${geometry.path("close")}" class="simulation-line-price"/>
          ${purchases}${settlements}
        </svg>
        <figcaption class="simulation-chart-legend"><span class="legend-purchase">仮想購入</span><span class="legend-settlement">仮想決済</span></figcaption>
        <div class="simulation-chart-tooltip" role="status" aria-live="polite" hidden></div>
      </figure>
    </section>`;
  };

  const trendMeta = (key) => {
    if (key === "up") return { className: "trend-up", symbol: "↑" };
    if (key === "down") return { className: "trend-down", symbol: "↓" };
    return { className: "trend-neutral", symbol: "→" };
  };

  const cachedBanner = (result, cached) => {
    if (!cached) return "";
    const savedAt = result.saved_at || result.fetched_at || "時刻不明";
    return `<div class="cached-banner">保存結果を表示しています。ライブ分析結果ではありません。保存日時: ${escapeHtml(savedAt)}</div>`;
  };

  const chartMarkup = (result, caption) => {
    if (!result.chart_url) return "";
    return `
      <section class="result-section">
        <div class="section-title"><h3>チャート</h3><small>${escapeHtml(caption)}</small></div>
        <figure class="chart-frame">
          <img src="${escapeHtml(result.chart_url)}" alt="${escapeHtml(caption)}" loading="eager">
          <figcaption>グラフ内は環境依存の文字化けを避けるため英数字表記です。</figcaption>
        </figure>
      </section>`;
  };

  const warningsMarkup = (warnings) => {
    if (!Array.isArray(warnings) || warnings.length === 0) return "";
    return `<ul class="warning-list">${warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("")}</ul>`;
  };

  const renderPrediction = (result, cached = false) => {
    const trend = trendMeta(result.direction_key);
    const probabilities = Array.isArray(result.probabilities) ? result.probabilities : [];
    const probabilityRows = probabilities.map((item) => {
      const percentage = Math.max(0, Math.min(100, Number(item.percentage) || 0));
      return `
        <div class="probability-row">
          <span>${escapeHtml(item.label)}</span>
          <span class="probability-track"><span class="probability-fill" style="width:${percentage.toFixed(2)}%"></span></span>
          <b>${percentage.toFixed(1)}%</b>
        </div>`;
    }).join("");

    return `
      ${cachedBanner(result, cached)}
      <header class="result-header">
        <div>
          <p class="result-kicker">ANALYSIS RESULT</p>
          <h2>${escapeHtml(result.company_name)} <code>${escapeHtml(result.ticker)}</code></h2>
        </div>
        <div class="result-date"><span>基準日 ${escapeHtml(result.basis_date)}</span><span>取得 ${escapeHtml(result.fetched_at)}</span></div>
      </header>
      <div class="signal-card">
        <div>
          <small>5営業日先の方向</small>
          <strong class="${trend.className}">${trend.symbol} ${escapeHtml(result.direction)}</strong>
        </div>
        <div class="prediction-class">
          <small>予測クラス / 最上位確率</small>
          <strong>${escapeHtml(result.prediction_class)} · ${formatPercent(result.top_probability)}</strong>
        </div>
      </div>
      <div class="metrics-grid">
        <div class="metric-card"><span>最新値</span><strong>${formatNumber(result.latest_price)}</strong></div>
        <div class="metric-card"><span>RSI (14)</span><strong>${formatNumber(result.rsi)}</strong></div>
        <div class="metric-card"><span>チャネル位置</span><strong>${formatRawPercent(Number(result.channel_position) * 100)}</strong></div>
        <div class="metric-card"><span>検証データ数</span><strong>${formatInteger(result.validation?.validation_samples)}</strong></div>
      </div>
      <section class="result-section">
        <div class="section-title"><h3>分析コメント</h3><small>ルールベース解説</small></div>
        <p class="analysis-comment">${escapeHtml(result.analysis_comment)}</p>
      </section>
      <section class="result-section">
        <div class="section-title"><h3>6段階のモデル出力確率</h3><small>合計は丸めにより100%と一致しない場合があります</small></div>
        <div class="probability-list">${probabilityRows}</div>
      </section>
      ${chartMarkup(result, `${result.ticker} の直近チャネル`)}
      <section class="result-section">
        <div class="section-title"><h3>時系列検証</h3><small>前半70%学習 / 後半30%検証</small></div>
        <div class="metrics-grid">
          <div class="metric-card"><span>6段階分類精度</span><strong>${formatPercent(result.validation?.six_class_accuracy)}</strong></div>
          <div class="metric-card"><span>方向予測精度</span><strong>${formatPercent(result.validation?.direction_accuracy)}</strong></div>
          <div class="metric-card"><span>学習データ数</span><strong>${formatInteger(result.validation?.training_samples)}</strong></div>
          <div class="metric-card"><span>検証データ数</span><strong>${formatInteger(result.validation?.validation_samples)}</strong></div>
        </div>
        <p class="disclaimer-box">${escapeHtml(result.validation?.note)}</p>
      </section>
      ${warningsMarkup(result.warnings)}
      <p class="disclaimer-box">${escapeHtml(result.disclaimer)}</p>`;
  };

  const renderIndividualPrediction = (result, cached = false) => {
    const shortAnalysis = result.chart_analysis || {};
    const longAnalysis = result.long_chart_analysis || {};
    const sixStage = result.six_stage_trend || {};
    const collectionPeriod = result.data_collection_period || {
      start: result.data_collection_start,
      end: result.data_collection_end,
    };
    const shortSections = (shortAnalysis.sections || []).map((section) => `
      <article class="chart-report-section" data-report-key="${escapeHtml(section.key)}">
        <h4>${escapeHtml(section.title)}</h4>
        <p>${escapeHtml(section.body)}</p>
      </article>`).join("");
    const longSections = (longAnalysis.sections || []).map((section) => `
      <article class="chart-report-section" data-report-key="${escapeHtml(section.key)}">
        <h4>${escapeHtml(section.title)}</h4>
        <p>${escapeHtml(section.body)}</p>
      </article>`).join("");
    const sixRows = (sixStage.items || []).map((item) => {
      const percentage = Math.max(0, Math.min(100, Number(item.percentage) || 0));
      return `<div class="six-stage-row"><span>${escapeHtml(item.label)}</span><span class="six-stage-track"><span class="six-stage-fill" style="width:${percentage.toFixed(1)}%"></span></span><b>${percentage.toFixed(1)}％</b></div>`;
    }).join("");
    const shortChart = result.chart_url ? `
      <section class="individual-section individual-chart-section" data-individual-section="chart">
        <div class="section-title"><h3>直近60営業日のチャート</h3><small>確定日足による短期表示</small></div>
        <figure class="chart-frame individual-chart-frame">
          <img src="${escapeHtml(result.chart_url)}" alt="${escapeHtml(result.company_name)}の直近60営業日チャート" loading="eager">
          <figcaption>終値と5日・20日・60日移動平均を表示しています。</figcaption>
        </figure>
      </section>` : "";
    const longChart = result.long_chart_url ? `
      <section class="individual-section individual-chart-section" data-individual-section="long-chart">
        <div class="section-title"><h3>直近2年間のチャート</h3><small>確定日足による中長期表示</small></div>
        <figure class="chart-frame individual-chart-frame">
          <img src="${escapeHtml(result.long_chart_url)}" alt="${escapeHtml(result.company_name)}の直近2年間チャート" loading="lazy">
          <figcaption>終値と60日・200日移動平均を表示しています。</figcaption>
        </figure>
      </section>` : "";
    const directionModelPublicName = result.direction_model_name === "二値ロジスティック回帰"
      ? "ロジスティック回帰（上昇・下落）"
      : result.direction_model_name;
    const sixClassModelPublicName = result.six_class_model_name === "多クラスロジスティック回帰"
      ? "ロジスティック回帰（6段階）"
      : result.six_class_model_name;
    const factorChips = (result.selected_factors || []).map((factor) => `<span class="factor-chip selected">${escapeHtml(factor)}</span>`).join("") || '<span class="factor-chip excluded">なし</span>';
    const excludedChips = (result.excluded_factors || []).map((factor) => `<span class="factor-chip excluded">${escapeHtml(factor)}</span>`).join("") || '<span class="factor-chip excluded">なし</span>';
    const importance = Array.isArray(result.feature_importance_top10) ? result.feature_importance_top10 : [];
    const maximumImportance = Math.max(...importance.map((item) => Number(item.importance) || 0), 1e-9);
    const importanceRows = importance.map((item) => {
      const width = Math.max(1, (Number(item.importance) || 0) / maximumImportance * 100);
      return `<div class="importance-row individual-importance-row" title="内部特徴量名: ${escapeHtml(item.internal_name)}"><span>${escapeHtml(item.display_name)}</span><span class="importance-track"><span style="width:${width.toFixed(1)}%"></span></span><b>${(Number(item.importance) * 100).toFixed(2)}</b></div>`;
    }).join("");
    const modelAuditMarkup = `
      <section class="individual-section individual-model-summary" data-individual-section="model-summary">
        <div class="section-title"><h3>予測モデル</h3><small>全銘柄共通固定構成</small></div>
        <div class="model-detail-grid individual-logistic-model-grid">
          <div><span>方向予測</span><p>${escapeHtml(directionModelPublicName)}</p></div>
          <div><span>6段階予測</span><p>${escapeHtml(sixClassModelPublicName)}</p></div>
          <div><span>予測期間</span><p>${escapeHtml(result.forecast_horizon_label)}</p></div>
          <div><span>モデル設定</span><p>${escapeHtml(result.model_configuration_label)}</p></div>
        </div>
      </section>
      <section class="individual-section individual-factor-section" data-individual-section="factors">
        <div class="section-title"><h3>今回の予測に使用したファクター</h3><small>候補比較を行わない固定構成</small></div>
        <p class="details-note">${escapeHtml(result.factor_selection_definition)}</p>
        <div class="factor-columns individual-factor-columns"><div><h4>使用ファクター</h4><div class="factor-list">${factorChips}</div></div><div><h4>固定構成で未使用</h4><div class="factor-list">${excludedChips}</div></div></div>
      </section>
      <section class="individual-section individual-importance-section" data-individual-section="importance">
        <div class="section-title"><h3>方向予測に使われた特徴量重要度</h3><small>標準化後係数の絶対値 上位10件</small></div>
        <div class="importance-list">${importanceRows}</div>
        <p class="details-note">特徴量重要度は、標準化後のロジスティック回帰係数の絶対値を基にした相対的な指標であり、因果関係を意味しません。</p>
      </section>
      <details class="model-details individual-logistic-details" data-individual-section="model-details">
        <summary>モデル詳細を見る</summary>
        <div class="individual-detail-content"><p>通常予測では交差検証、候補比較、正式評価の再計算を行わず、方向予測用・6段階予測用のロジスティック回帰を各1回だけ学習します。</p><p>確率補正：${escapeHtml(result.direction_scores?.calibration_status === "not_applied" ? "未適用（互換性のある保存済み補正器なし）" : result.direction_scores?.calibration_status)}</p></div>
      </details>`;
    return `
      ${cachedBanner(result, cached || result.cache?.used)}
      <section class="individual-section individual-current-prediction" data-individual-section="prediction">
        <header class="result-header individual-result-header">
          <div>
            <p class="result-kicker">今回の予測</p>
            <h2>${escapeHtml(result.company_name)}</h2>
            <p class="security-code">証券コード ${escapeHtml(result.security_code)}</p>
          </div>
          <div class="result-date"><span>データ取得 ${escapeHtml(result.fetched_at)}</span></div>
        </header>
        <dl class="individual-facts">
          <div><dt>基準日</dt><dd>${escapeHtml(result.basis_date)}</dd></div>
          <div><dt>基準終値</dt><dd>${formatNumber(result.latest_price)}</dd></div>
          <div><dt>データ収集期間</dt><dd>${escapeHtml(collectionPeriod.start)} ～ ${escapeHtml(collectionPeriod.end)}</dd></div>
        </dl>
        <div class="six-stage-panel">
          <div class="section-title"><h3>5営業日先の6段階トレンド予測</h3><small>モデル出力割合</small></div>
          <div class="six-stage-list">${sixRows}</div>
          <p class="six-stage-level-note">${escapeHtml(sixStage.level_note)}</p>
        </div>
        <section class="six-stage-report" data-individual-section="six-stage-report">
          <h3>6段階トレンド予測レポート</h3>
          <p>${escapeHtml(sixStage.intro)}</p>
          <p>${escapeHtml(sixStage.comparison)} ${escapeHtml(sixStage.gap_description)}</p>
          <p>${escapeHtml(sixStage.distribution_description)}</p>
          <p class="six-stage-report-footer">${escapeHtml(sixStage.footer)}</p>
        </section>
      </section>
      ${shortChart}
      <section class="individual-section chart-analysis-card" data-individual-section="chart-analysis">
        <div class="section-title"><h3>短期チャート分析レポート</h3><small>${escapeHtml(shortAnalysis.window_start)} ～ ${escapeHtml(shortAnalysis.window_end)}</small></div>
        <aside class="chart-report-disclaimer" aria-label="このレポートについて">
          <h4>このレポートについて</h4>
          <p>${escapeHtml(shortAnalysis.disclaimer)}</p>
        </aside>
        <div class="chart-report-sections">${shortSections}</div>
        <p class="chart-report-footer">${escapeHtml(shortAnalysis.footer)}</p>
      </section>
      ${longChart}
      <section class="individual-section chart-analysis-card" data-individual-section="long-chart-analysis">
        <div class="section-title"><h3>中長期チャート分析レポート</h3><small>${escapeHtml(longAnalysis.window_start)} ～ ${escapeHtml(longAnalysis.window_end)}</small></div>
        <div class="chart-report-sections">${longSections}</div>
        <p class="chart-report-footer">${escapeHtml(longAnalysis.footer)}</p>
      </section>
      ${modelAuditMarkup}
      ${warningsMarkup(result.warnings)}
      <section class="individual-section individual-investment-notice" data-individual-section="notice">
        <h3>投資判断に関する注意</h3>
        <p>${escapeHtml(result.disclaimer)}</p>
      </section>`;
  };

  const nikkeiChartMarkup = (rows, title, series) => {
    const chartNumber = (value) => value === null || value === undefined || value === "" ? Number.NaN : Number(value);
    const values = Array.isArray(rows) ? rows : [];
    const width = 820;
    const height = 340;
    const padding = { left: 70, right: 22, top: 24, bottom: 52 };
    const numericValues = values.flatMap((row) => series.map((item) => chartNumber(row[item.key]))).filter(Number.isFinite);
    if (values.length < 2 || numericValues.length === 0) return `<p class="nikkei-empty-note">表示できるチャートデータがありません。</p>`;
    let minimum = Math.min(...numericValues);
    let maximum = Math.max(...numericValues);
    const span = Math.max(maximum - minimum, Math.abs(maximum) * 0.03, 1);
    minimum -= span * 0.08;
    maximum += span * 0.08;
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const x = (index) => padding.left + (index / (values.length - 1)) * plotWidth;
    const y = (value) => padding.top + ((maximum - value) / (maximum - minimum)) * plotHeight;
    const pathFor = (key) => {
      let drawing = false;
      return values.map((row, index) => {
        const value = chartNumber(row[key]);
        if (!Number.isFinite(value)) { drawing = false; return ""; }
        const command = drawing ? "L" : "M";
        drawing = true;
        return `${command}${x(index).toFixed(2)},${y(value).toFixed(2)}`;
      }).join(" ");
    };
    const grid = Array.from({ length: 5 }, (_, index) => {
      const value = maximum - ((maximum - minimum) * index / 4);
      const lineY = y(value);
      return `<line x1="${padding.left}" y1="${lineY.toFixed(2)}" x2="${width - padding.right}" y2="${lineY.toFixed(2)}" class="nikkei-chart-grid"/><text x="${padding.left - 10}" y="${(lineY + 4).toFixed(2)}" text-anchor="end" class="nikkei-chart-axis-label">${escapeHtml(formatInteger(value))}</text>`;
    }).join("");
    const labelIndexes = [...new Set([0, Math.round((values.length - 1) / 3), Math.round((values.length - 1) * 2 / 3), values.length - 1])];
    const dateLabels = labelIndexes.map((index) => `<text x="${x(index).toFixed(2)}" y="${height - 18}" text-anchor="${index === 0 ? "start" : index === values.length - 1 ? "end" : "middle"}" class="nikkei-chart-axis-label">${escapeHtml(values[index].date)}</text>`).join("");
    const lines = series.map((item) => `<path d="${pathFor(item.key)}" class="nikkei-chart-line ${item.className}"/>`).join("");
    const hitTargets = values.map((row, index) => {
      const close = chartNumber(row.close);
      if (!Number.isFinite(close)) return "";
      const detail = series.map((item) => Number.isFinite(chartNumber(row[item.key])) ? `${item.label} ${formatNumber(row[item.key])}` : "").filter(Boolean).join(" / ");
      return `<circle cx="${x(index).toFixed(2)}" cy="${y(close).toFixed(2)}" r="8" class="nikkei-chart-hit" tabindex="0" role="img" aria-label="${escapeHtml(`${row.date} / ${detail}`)}" data-chart-tooltip="${escapeHtml(`${row.date} / ${detail}`)}"><title>${escapeHtml(`${row.date} / ${detail}`)}</title></circle>`;
    }).join("");
    const legend = series.map((item) => `<span class="${item.className}">${escapeHtml(item.label)}</span>`).join("");
    return `<figure class="nikkei-svg-chart"><h4>${escapeHtml(title)}</h4><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(title)}">${grid}${dateLabels}${lines}${hitTargets}</svg><figcaption class="nikkei-chart-legend">${legend}</figcaption><div class="nikkei-chart-tooltip" role="status" aria-live="polite" hidden></div></figure>`;
  };

  const renderNikkeiPublicPrediction = (result, cached = false) => {
    const validation = result.validation || {};
    const sixStage = result.six_stage_trend || {};
    const sixReport = result.six_class_report || {};
    const items = Array.isArray(result.six_class_probabilities) ? result.six_class_probabilities : [];
    const topValue = Math.max(...items.map((item) => Number(item.percentage) || 0), 0);
    const sixRows = items.map((item) => {
      const percentage = Math.max(0, Math.min(100, Number(item.percentage) || 0));
      const isTop = percentage === topValue;
      return `<div class="six-stage-row${isTop ? " nikkei-six-top" : ""}"><span>${escapeHtml(item.label)}</span><span class="six-stage-track"><span class="six-stage-fill" style="width:${percentage.toFixed(1)}%"></span></span><b>${percentage.toFixed(1)}％</b></div>`;
    }).join("");
    const precisionText = (value) => value == null || !Number.isFinite(Number(value)) ? "算出できません" : formatPercent(value);
    const baselineGap = Number(validation.best_baseline_gap);
    const baselineSentence = Number.isFinite(baselineGap)
      ? baselineGap > 0
        ? `本システムの方向一致率は、同じ期間で最も成績が良かった単純な予測方法を${Math.abs(baselineGap * 100).toFixed(1)}ポイント上回りました。`
        : baselineGap < 0
          ? `本システムの方向一致率は、同じ期間で最も成績が良かった単純な予測方法を${Math.abs(baselineGap * 100).toFixed(1)}ポイント下回りました。`
          : "本システムの方向一致率は、同じ期間で最も成績が良かった単純な予測方法と同じでした。"
      : "単純な予測方法との差は算出できませんでした。";
    const factorSelection = result.model_selection || {};
    const selectedFactors = (result.selected_factors || []).map((factor) => `<span class="factor-chip selected">${escapeHtml(factor)}</span>`).join("") || '<span class="factor-chip excluded">なし</span>';
    const excludedFactors = (result.excluded_factors || []).map((factor) => `<span class="factor-chip excluded">${escapeHtml(factor)}</span>`).join("") || '<span class="factor-chip excluded">なし</span>';
    const importance = Array.isArray(result.feature_importance_top10) ? result.feature_importance_top10 : [];
    const maxImportance = Math.max(...importance.map((item) => Number(item.importance) || 0), 1);
    const importanceRows = importance.map((item) => {
      const width = Math.max(1, (Number(item.importance) || 0) / maxImportance * 100);
      return `<div class="importance-row nikkei-importance-row" title="内部特徴量名: ${escapeHtml(item.internal_name)}"><span>${escapeHtml(item.display_name)}</span><span class="importance-track"><span style="width:${width.toFixed(1)}%"></span></span><b>${formatNumber(item.importance)}</b></div>`;
    }).join("");
    const cacheNotice = result.cache?.used ? `<div class="cache-status cached">同じ基準日・データ・保存済みモデル設定の新画面対応結果を再利用しました。</div>` : "";
    const shortChart = nikkeiChartMarkup(result.chart_60d, "直近60営業日の株価推移", [
      { key: "close", label: "終値", className: "nikkei-line-close" },
      { key: "ma5", label: "5日移動平均", className: "nikkei-line-ma5" },
      { key: "ma20", label: "20日移動平均", className: "nikkei-line-ma20" },
      { key: "ma60", label: "60日移動平均", className: "nikkei-line-ma60" },
    ]);
    const longChart = nikkeiChartMarkup(result.chart_2y, "直近2年間の株価推移", [
      { key: "close", label: "終値", className: "nikkei-line-close" },
      { key: "ma60", label: "60日移動平均", className: "nikkei-line-ma60" },
      { key: "ma200", label: "200日移動平均", className: "nikkei-line-ma200" },
    ]);
    const allFeatures = [...(factorSelection.japan_features || []), ...(factorSelection.overseas_features || [])];
    const directionEvaluation = result.direction_evaluation || validation;
    const sixClassEvaluation = result.six_class_evaluation || validation;
    const evaluationAvailable = directionEvaluation.available !== false && sixClassEvaluation.available !== false;
    const intervalText = (values, formatter) => Array.isArray(values) && values.length === 2
      ? `${formatter(values[0])} ～ ${formatter(values[1])}`
      : "算出できません";
    const accuracyMarkup = evaluationAvailable ? `
      <div class="nikkei-evaluation-split">
        <article class="nikkei-evaluation-card" data-nikkei-evaluation="direction">
          <h4>予測精度</h4>
          <strong class="individual-accuracy-value">${formatPercent(directionEvaluation.direction_accuracy)}</strong>
          <p>方向一致率は、5営業日先の日経平均株価が基準日より上か下かについて、過去データで実際の方向と一致した割合です。</p>
          <dl class="nikkei-evaluation-list"><div><dt>評価期間</dt><dd>${escapeHtml(directionEvaluation.period?.start)} ～ ${escapeHtml(directionEvaluation.period?.end)}</dd></div><div><dt>予測条件</dt><dd>${escapeHtml(result.prediction_context_label)}</dd></div><div><dt>検証回数</dt><dd>${formatInteger(directionEvaluation.validation_samples)}回</dd></div><div><dt>一致回数</dt><dd>${formatInteger(directionEvaluation.correct_predictions)}回</dd></div><div><dt>上昇予測の一致率</dt><dd>${precisionText(directionEvaluation.up_prediction_precision)}</dd></div><div><dt>下落予測の一致率</dt><dd>${precisionText(directionEvaluation.down_prediction_precision)}</dd></div><div><dt>最良の単純方法との差</dt><dd>${formatSignedPoints(directionEvaluation.best_baseline_gap)}</dd></div><div><dt>方向一致率の95％区間</dt><dd>${intervalText(directionEvaluation.direction_accuracy_95ci, formatPercent)}</dd></div><div><dt>単純方法との差の95％区間</dt><dd>${intervalText(directionEvaluation.best_baseline_gap_95ci, formatSignedPoints)}</dd></div></dl>
          <p>${escapeHtml(baselineSentence)}</p>
        </article>
        <article class="nikkei-evaluation-card" data-nikkei-evaluation="six-class">
          <h4>6段階区分の過去評価</h4>
          <strong class="individual-accuracy-value">${formatPercent(sixClassEvaluation.six_class_accuracy)}</strong>
          <p>6段階完全一致率は、実際の値動き幅の区分と、モデルが最も高く出力した区分が一致した割合です。方向一致率とは異なる指標です。</p>
          <dl class="nikkei-evaluation-list"><div><dt>6段階完全一致率</dt><dd>${formatPercent(sixClassEvaluation.six_class_accuracy)}</dd></div><div><dt>6段階Macro-F1</dt><dd>${formatPercent(sixClassEvaluation.six_class_macro_f1)}</dd></div><div><dt>評価期間</dt><dd>${escapeHtml(sixClassEvaluation.period?.start)} ～ ${escapeHtml(sixClassEvaluation.period?.end)}</dd></div><div><dt>予測条件</dt><dd>${escapeHtml(result.prediction_context_label)}</dd></div></dl>
        </article>
      </div>` : `<div class="nikkei-evaluation-unavailable" role="status">${escapeHtml(directionEvaluation.message || sixClassEvaluation.message || "現在使用した予測条件に対応する正式評価は、まだ作成されていません。")}</div>`;

    return `
      ${cachedBanner(result, cached)}${cacheNotice}
      <header class="result-header nikkei-result-header">
        <div><p class="result-kicker">NIKKEI 225 · 5-DAY OUTLOOK</p><h2>日経平均株価予測・戦略支援システム</h2><p>予測対象：日経平均株価</p></div>
        <div class="result-date"><span>予測期間 ${escapeHtml(result.forecast_horizon_label)}</span><span>データ取得日時 ${escapeHtml(result.fetched_at)}</span></div>
      </header>
      <dl class="nikkei-meta-grid">
        <div><dt>予測基準日</dt><dd>${escapeHtml(result.forecast_base_date)}</dd></div>
        <div><dt>予測対象日</dt><dd>${escapeHtml(result.forecast_target_date)}</dd></div>
        <div><dt>予測条件</dt><dd>${escapeHtml(result.prediction_context_label)}</dd></div>
        <div><dt>データ収集期間</dt><dd>${escapeHtml(result.data_collection_start)} ～ ${escapeHtml(result.data_collection_end)}</dd></div>
        <div><dt>評価期間</dt><dd>${escapeHtml(result.evaluation_start)} ～ ${escapeHtml(result.evaluation_end)}</dd></div>
      </dl>

      <section class="nikkei-section nikkei-six-section" data-nikkei-section="six-class">
        <div class="nikkei-section-heading"><span>01</span><div><h3>5営業日先の6段階予測</h3><p>実際の6クラスモデルが出力した割合です。</p></div></div>
        <div class="six-stage-panel"><div class="section-title"><h3>モデル出力割合</h3><small>合計 ${formatRawPercent(sixStage.display_total_percentage)}</small></div><div class="six-stage-list">${sixRows}</div><p class="six-stage-level-note">${escapeHtml(sixStage.level_note)}</p><p class="nikkei-model-output-note">${escapeHtml(result.score_display_note)}</p><p class="nikkei-model-role-note">${escapeHtml(result.model_roles_note)}</p></div>
      </section>

      <section class="nikkei-section six-stage-report" data-nikkei-section="six-report">
        <div class="nikkei-section-heading"><span>02</span><div><h3>6段階トレンド予測レポート</h3><p>モデル出力の分布を固定テンプレートで整理します。</p></div></div>
        <p>${escapeHtml(sixReport.body)}</p><p class="six-stage-report-footer">${escapeHtml(sixReport.footer)}</p>
      </section>

      <section class="nikkei-section nikkei-chart-card" data-nikkei-section="chart-60d">
        <div class="nikkei-section-heading"><span>03</span><div><h3>直近60営業日の株価推移</h3><p>${escapeHtml(result.short_term_report?.window_start)} ～ ${escapeHtml(result.short_term_report?.window_end)}</p></div></div>${shortChart}
      </section>
      <section class="nikkei-section nikkei-report-card" data-nikkei-section="short-report"><div class="nikkei-section-heading"><span>04</span><div><h3>短期動向レポート</h3><p>確定済み日足と今回の6段階出力を区別して整理します。</p></div></div><p>${escapeHtml(result.short_term_report?.body)}</p></section>

      <section class="nikkei-section nikkei-chart-card" data-nikkei-section="chart-2y">
        <div class="nikkei-section-heading"><span>05</span><div><h3>直近2年間の株価推移</h3><p>${escapeHtml(result.medium_long_term_report?.window_start)} ～ ${escapeHtml(result.medium_long_term_report?.window_end)}</p></div></div>${longChart}
      </section>
      <section class="nikkei-section nikkei-report-card" data-nikkei-section="long-report"><div class="nikkei-section-heading"><span>06</span><div><h3>中長期トレンド分析レポート</h3><p>将来2年間の予測ではなく、過去2年間の位置関係です。</p></div></div><p>${escapeHtml(result.medium_long_term_report?.body)}</p></section>

      <section class="nikkei-section nikkei-accuracy-card" data-nikkei-section="accuracy">
        <div class="nikkei-section-heading"><span>07</span><div><h3>予測精度</h3><p>今回使用した予測条件と一致する正式な時系列評価だけを表示します。</p></div></div>
        <p class="nikkei-prediction-context">予測条件：${escapeHtml(result.prediction_context_label)}</p>
        ${accuracyMarkup}
      </section>

      <section class="nikkei-section nikkei-factor-section" data-nikkei-section="factors">
        <div class="nikkei-section-heading"><span>08</span><div><h3>今回のモデルが選択したファクター</h3><p>${escapeHtml(result.factor_selection_definition)}</p></div></div>
        <div class="factor-columns nikkei-factor-columns"><div><h4>採用ファクター</h4><div class="factor-list">${selectedFactors}</div></div><div><h4>除外ファクター</h4><div class="factor-list">${excludedFactors}</div></div></div>
        <p class="nikkei-factor-note">評価期間の時系列検証を基に、最終再学習モデルで採用されたファクターを表示しています。採用は因果関係や将来の有効性を保証するものではありません。</p>
      </section>

      <section class="nikkei-section nikkei-importance-section" data-nikkei-section="importance">
        <div class="nikkei-section-heading"><span>09</span><div><h3>方向予測に使われた特徴量重要度</h3><p>日米50対50の固定合算比率を反映した上位10項目です。</p></div></div><div class="importance-list">${importanceRows}</div><p class="nikkei-factor-note">特徴量重要度は、モデル内部で予測に使用された相対的な寄与度であり、因果関係を意味しません。</p>
      </section>

      <details class="model-details nikkei-config-details" data-nikkei-section="details">
        <summary>採用構成と全特徴量の詳細を見る</summary><div class="nikkei-detail-content"><div class="model-detail-grid"><div><span>採用モデル</span><p>${escapeHtml(result.model_name)}</p></div><div><span>採用ファクターグループ</span><p>${escapeHtml(result.feature_group)}</p></div><div><span>モデル選択方式</span><p>内側3Foldで選択し、外側8Foldは正式評価にのみ使用します。</p></div><div><span>予測対象・期間</span><p>日経平均終値の方向と値動き幅 · 5営業日先</p></div><div><span>データ利用可能時刻</span><p>日本の予測時点以前に確定していた外部データだけを後方as-of結合します。</p></div><div><span>評価期間</span><p>${escapeHtml(result.evaluation_start)} ～ ${escapeHtml(result.evaluation_end)}</p></div></div><h4>採用特徴量（${formatInteger(allFeatures.length)}項目）</h4><p class="nikkei-feature-list">${allFeatures.map(escapeHtml).join("、")}</p><h4>除外ファクター</h4><p class="nikkei-feature-list">${(result.excluded_factors || []).map(escapeHtml).join("、") || "なし"}</p></div>
      </details>

      <section class="nikkei-section nikkei-condition-section" data-nikkei-section="conditions"><div class="nikkei-section-heading"><span>10</span><div><h3>データ・検証条件・注意事項</h3><p>表示値の前提をまとめています。</p></div></div><div class="nikkei-condition-grid"><div><span>学習期間</span><strong>各評価時点より前の最大8年間</strong></div><div><span>ウォームアップ</span><strong>300取引日</strong></div><div><span>正式評価</span><strong>直近2年 · 外側8Fold</strong></div><div><span>内側選択</span><strong>時系列順3Fold · 5取引日パージ</strong></div></div><p>${escapeHtml(result.data_collection_definition)}</p>${warningsMarkup(result.warnings)}<p class="disclaimer-box">${escapeHtml(result.disclaimer)}</p></section>`;
  };

  const wireNikkeiChartTooltips = (panel) => {
    panel.querySelectorAll(".nikkei-svg-chart [data-chart-tooltip]").forEach((point) => {
      const tooltip = point.closest("figure")?.querySelector(".nikkei-chart-tooltip");
      if (!tooltip) return;
      const show = () => { tooltip.textContent = point.dataset.chartTooltip || ""; tooltip.hidden = false; };
      const hide = () => { tooltip.hidden = true; };
      point.addEventListener("pointerenter", show);
      point.addEventListener("pointerleave", hide);
      point.addEventListener("focus", show);
      point.addEventListener("blur", hide);
      point.addEventListener("click", show);
    });
  };
  const renderSimulation = (result, cached = false) => {
    const trades = Array.isArray(result.trades) ? result.trades : [];
    const tradeTable = trades.length ? `
      <div class="trade-table-wrap">
        <table class="trade-table simulation-trade-table">
          <thead><tr><th>取引番号</th><th>仮想購入日</th><th>仮想購入価格</th><th>仮想購入株数</th><th>仮想決済日</th><th>仮想決済価格</th><th>保有取引日数</th><th>決済理由</th><th>仮想損益額</th><th>仮想損益率</th></tr></thead>
          <tbody>${trades.map((trade) => `
            <tr>
              <td>${formatInteger(trade.trade_number)}</td><td>${escapeHtml(trade.entry_date)}</td><td>${formatNumber(trade.entry_price)}円</td><td title="仮想購入金額 ${formatCurrency(trade.entry_value)} / 購入後現金残高 ${formatCurrency(trade.cash_after_entry)} / 仮想決済金額 ${formatCurrency(trade.exit_value)}">${formatInteger(trade.entry_shares)}株</td>
              <td>${escapeHtml(trade.exit_date)}</td><td>${formatNumber(trade.exit_price)}円</td><td>${formatInteger(trade.held_trading_days)}日</td>
              <td>${escapeHtml(trade.reason)}</td><td><span class="simulation-pnl-value ${simulationPnlClass(trade.profit_amount, 0)}">${formatSignedCurrency(trade.profit_amount)}</span></td><td><span class="simulation-pnl-value ${simulationPnlClass(trade.profit_percent, 1)}">${formatSignedRawPercent(trade.profit_percent)}</span></td>
            </tr>`).join("")}</tbody>
        </table>
      </div>` : `<p class="simulation-neutral-note">決済まで完了した仮想取引はありません。</p>`;
    const conditions = result.conditions || {};
    const conditionList = [
      `初期投資額 ${formatCurrency(conditions.initial_investment)}`,
      `利益確定条件 ${conditions.take_profit == null ? "使用しない" : `${formatNumber(conditions.take_profit)}％`}`,
      `損切り条件 ${conditions.stop_loss == null ? "使用しない" : `${formatNumber(conditions.stop_loss)}％`}`,
      "売買単位：100株",
      "購入方法：各購入時点で購入可能な最大株数",
      "現物買い・余剰現金を保持する仮想計算",
    ];
    const profitableRate = result.profitable_trade_rate == null ? "算出できません" : formatRawPercent(result.profitable_trade_rate);

    return `
      ${cachedBanner(result, cached)}
      <div class="simulation-result">
      <header class="result-header simulation-result-header">
        <div><p class="result-kicker">SIMULATION RESULT</p><h2>${escapeHtml(result.company_name)}</h2><p class="security-code">証券コード ${escapeHtml(result.security_code)}</p></div>
        <div class="result-date"><span>データ取得日時 ${escapeHtml(result.fetched_at)}</span></div>
      </header>
      <div class="simulation-period-grid">
        <div><span>データ収集期間</span><strong>${escapeHtml(result.data_collection_start)} ～ ${escapeHtml(result.data_collection_end)}</strong></div>
        <div><span>検証期間</span><strong>${escapeHtml(result.simulation_start)} ～ ${escapeHtml(result.simulation_end)}</strong></div>
      </div>
      <div class="metrics-grid simulation-metrics">
        <div class="metric-card"><span>仮想最終資産</span><strong>${formatCurrency(result.final_assets)}</strong></div>
        <div class="metric-card"><span>仮想損益額</span><strong>${formatSignedCurrency(result.profit_amount)}</strong></div>
        <div class="metric-card"><span>総損益率</span><strong>${formatSignedRawPercent(result.profit_rate)}</strong></div>
        <div class="metric-card"><span>決済済み取引回数</span><strong>${formatInteger(result.trade_count)}回</strong></div>
        <div class="metric-card"><span>利益になった取引の割合</span><strong>${profitableRate}</strong></div>
        <div class="metric-card"><span>最大下落率</span><strong>${formatRawPercent(result.max_drawdown)}</strong></div>
        <div class="metric-card"><span>同期間保有の総損益率</span><strong>${formatSignedRawPercent(result.buy_hold_profit_rate)}</strong></div>
        <div class="metric-card"><span>同期間保有との差</span><strong>${formatSignedRawPercent(result.buy_hold_difference_points).replace("%", "ポイント")}</strong></div>
      </div>
      <section class="result-section simulation-summary-section">
        <div class="section-title"><h3>結果説明</h3><small>実際の計算結果を定型文で整理</small></div>
        <p>${escapeHtml(result.result_summary)}</p>
      </section>
      ${result.reentry_stopped_due_to_insufficient_cash ? `<p class="simulation-neutral-note">途中の再購入時点で100株を購入できる資金を下回ったため、その時点で新規購入を終了し、残りの資金を現金として保持しました。</p>` : ""}
      ${simulationAssetChart(result)}
      ${simulationPriceChart(result)}
      <section class="result-section">
        <div class="section-title"><h3>シミュレーション条件</h3><small>指定率を機械的に反復適用</small></div>
        <div class="condition-list">${conditionList.map((condition) => `<span>${escapeHtml(condition)}</span>`).join("")}</div>
        <p class="simulation-rule-note">始値で条件価格を飛び越えた場合は始値で決済します。日中に両条件へ到達した日は、日足だけでは先後を判断できないため損切り条件を優先します。決済日の同日には再購入しません。</p>
      </section>
      <section class="result-section">
        <div class="section-title"><h3>取引履歴</h3><small>仮想購入から仮想決済までを1回として集計</small></div>
        ${tradeTable}
      </section>
      <section class="result-section">
        <div class="section-title"><h3>計算条件</h3><small>結果を読む前にご確認ください</small></div>
        <ul class="assumption-list">${(result.assumptions || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      </section>
      <div class="simulation-disclaimer"><p>${escapeHtml(result.disclaimer)}</p><p>${escapeHtml(result.cost_disclaimer)}</p><p>${escapeHtml(result.cost_frequency_note)}</p></div>
      </div>`;
  };

  const wireSimulationChartTooltips = (panel) => {
    panel.querySelectorAll("[data-chart-tooltip]").forEach((marker) => {
      const tooltip = marker.closest("figure")?.querySelector(".simulation-chart-tooltip");
      if (!tooltip) return;
      const show = () => { tooltip.textContent = marker.dataset.chartTooltip || ""; tooltip.hidden = false; };
      const hide = () => { tooltip.hidden = true; };
      marker.addEventListener("pointerenter", show);
      marker.addEventListener("pointerleave", hide);
      marker.addEventListener("focus", show);
      marker.addEventListener("blur", hide);
      marker.addEventListener("click", show);
    });
  };

  const renderResult = (panel, result, cached = false) => {
    panel.classList.remove("empty-state");
    panel.innerHTML = result.kind === "simulation"
      ? renderSimulation(result, cached)
      : result.ticker === "^N225" && result.model_selection
        ? renderNikkeiPublicPrediction(result, cached)
        : result.kind === "individual_prediction"
          ? renderIndividualPrediction(result, cached)
          : renderPrediction(result, cached);
    if (result.kind === "simulation") wireSimulationChartTooltips(panel);
    if (result.ticker === "^N225" && result.model_selection) wireNikkeiChartTooltips(panel);
    panel.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const renderError = (panel, error, previousResult) => {
    panel.classList.remove("empty-state");
    panel.innerHTML = `
      <div class="error-card">
        <h2>処理を完了できませんでした</h2>
        <p>${escapeHtml(error?.message || "通信または分析処理でエラーが発生しました。")}</p>
        <p>${escapeHtml(error?.action || "時間をおいて再度お試しください。")}</p>
        ${previousResult ? '<button type="button" class="secondary-button" data-show-previous>前回の正常結果を表示</button>' : ""}
      </div>`;
    if (previousResult) {
      panel.querySelector("[data-show-previous]").addEventListener("click", () => renderResult(panel, previousResult, true));
    }
  };

  const buildPayload = (form, submitter = null) => {
    if (form.id === "nikkei-form") return { model_reevaluation: submitter?.dataset.modelReevaluation === "true" };
    if (form.id === "individual-form") return { ticker: form.elements.ticker.value.trim() };
    const optionalNumber = (input) => input.value.trim() === "" ? null : Number(input.value);
    return {
      ticker: form.elements.ticker.value.trim(),
      initial_investment: Number(form.elements.initial_investment.value),
      take_profit: optionalNumber(form.elements.take_profit),
      stop_loss: optionalNumber(form.elements.stop_loss),
    };
  };

  const wireAnalysisForm = (form) => {
    const panel = document.getElementById("result-panel");
    const overlay = document.getElementById("loading-overlay");
    const submitButtons = [...form.querySelectorAll("button[type='submit']")];
    const storageKey = `stock-app:${form.id}:last-result`;
    const isolatesErrors = form.id === "individual-form" || form.id === "simulation-form" || form.id === "nikkei-form";
    let requestGeneration = 0;
    let activeController = null;
    const clearIndividualResult = () => {
      if (!isolatesErrors) return;
      sessionStorage.removeItem(storageKey);
      panel.classList.remove("empty-state");
      panel.innerHTML = "";
    };
    const clearStaleResult = clearIndividualResult;

    form.addEventListener("invalid", clearIndividualResult, true);

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const isIndividual = form.id === "individual-form";
      const isSimulation = form.id === "simulation-form";
      if (isIndividual && (activeController !== null || form.getAttribute("aria-busy") === "true")) return;
      clearStaleResult();
      if (!form.reportValidity()) return;
      activeController?.abort();
      activeController = new AbortController();
      const currentGeneration = ++requestGeneration;
      submitButtons.forEach((button) => { button.disabled = true; });
      overlay.hidden = false;
      form.setAttribute("aria-busy", "true");
      panel.setAttribute("aria-busy", "true");
      let requestTimedOut = false;
      const requestTimeout = isIndividual
        ? window.setTimeout(() => {
            requestTimedOut = true;
            activeController?.abort();
          }, INDIVIDUAL_REQUEST_TIMEOUT_MS)
        : null;
      try {
        const response = await fetch(form.dataset.api, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(buildPayload(form, event.submitter)),
          signal: activeController.signal,
        });
        const payload = await response.json();
        if (currentGeneration !== requestGeneration) return;
        if (!response.ok || !payload.ok) {
          const localPrevious = isolatesErrors
            ? null
            : payload.last_result || JSON.parse(sessionStorage.getItem(storageKey) || "null");
          const responseError = isIndividual && response.status === 409
            ? {
                message: "別の個別銘柄予測が実行中です。処理が終了してから、もう一度お試しください。",
                action: "実行中の計算が終了すると、再度実行できます。",
              }
            : payload.error;
          renderError(panel, responseError, localPrevious);
          return;
        }
        if (form.id !== "simulation-form") sessionStorage.setItem(storageKey, JSON.stringify(payload.result));
        renderResult(panel, payload.result, false);
      } catch (error) {
        if (currentGeneration !== requestGeneration) return;
        if (error?.name === "AbortError" && !requestTimedOut) return;
        let localPrevious = null;
        if (!isolatesErrors) {
          try { localPrevious = JSON.parse(sessionStorage.getItem(storageKey) || "null"); } catch (_) { localPrevious = null; }
        }
        renderError(panel, {
          message: requestTimedOut && isIndividual
            ? "個別銘柄予測の計算が制限時間を超えたため終了しました。"
            : "サーバーと通信できませんでした。",
          action: requestTimedOut && isIndividual
            ? "データ取得状況を確認して、時間をおいてもう一度実行してください。"
            : "アプリが起動しているか、ネットワーク接続が有効か確認してください。",
        }, localPrevious);
      } finally {
        if (requestTimeout !== null) window.clearTimeout(requestTimeout);
        if (currentGeneration === requestGeneration) {
          overlay.hidden = true;
          submitButtons.forEach((button) => { button.disabled = false; });
          form.removeAttribute("aria-busy");
          panel.removeAttribute("aria-busy");
          activeController = null;
        }
      }
    });
  };

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("form[data-api]").forEach(wireAnalysisForm);
  });
})();
