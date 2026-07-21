# 2～3分デモ動画用台本

## 撮影前の準備

- `START_HERE.cmd` で起動し、Python 3.13.xの仮想環境準備が完了するまで待つ
- 日経平均の明示的なモデル再評価は事前に完了させる
- `model_settings/nikkei_dual_market.json` と学習済みモデルが存在することを確認
- 動画では通常の「予測を実行」を使い、モデル比較を繰り返さない
- 推奨順序: 日経平均 → 個別銘柄`7203` → シミュレーション`7203`、100万円、利益確定10%、損切り4%

## 日本語台本（約3分）

### 0:00～0:20　課題と概要

**画面:** トップ画面。

「日本株の短期的な方向予測と、売買条件の検証を一つの画面で行うローカルWebアプリです。日経平均、個別銘柄、利確・損切りシミュレーションの3機能を統合しています。」

### 0:20～1:10　日経平均5営業日先予測

**操作:** 日経平均を開き、「予測を実行」。6段階横棒、予測レポート、60日・2年チャートを順に示す。

「日経平均は、現在の確定終値を基準に5営業日先を予測します。場中のリアルタイム株価や未確定日足は使いません。6段階の割合は、方向モデルとは別の6クラスモデルの実出力で、上昇 Lv.3から下落 Lv.3まで固定順・合計100.0パーセントで表示します。」

「Lv.は値動き幅の区分で、確実性や売買の推奨度ではありません。60日・2年レポートは確定済みチャートの事実を固定テンプレートで整理し、5営業日先のモデル出力とは区別します。」

### 1:10～1:40　精度・ファクター・特徴量重要度

**画面で指す場所:** 採用／除外ファクター、特徴量重要度、初期状態が閉じた詳細欄。

「採用・除外ファクターは最終再学習モデルで実際に使われた構成を示し、特徴量重要度は相対的な寄与度であって因果関係を意味しません。詳細欄では採用モデル、特徴量、データ利用可能時刻の扱いを確認できます。」
### 1:40～2:10　個別銘柄予測とチャート分析

**操作:** 個別銘柄へ移動し `7203` を実行。6段階トレンド、60日／2年チャート、各分析レポート、予測モデル情報を順にスクロールする。

「個別銘柄は、日本株の証券コードを一度に1件だけ分析し、5営業日先の株価方向を予測します。対象銘柄の株価データと市場全体の動きを機械学習モデルで分析し、結果を上昇 Lv.3から下落 Lv.3まで6段階で表示します。」

「さらに、60営業日チャートによる短期分析と、2年チャートによる中長期分析を、それぞれのレポートとともに表示します。今回の予測に大きく影響した市場データも確認できます。」

「個別銘柄ごとの予測精度は表示しません。今回のモデル出力、直近の動き、中長期の流れ、予測の背景にある市場要因を確認する機能として整理しています。」

### 2:10～2:35　利確・損切りシミュレーション

**操作:** `.T`を付けずに`7203`、1,000,000円、利益確定10%、損切り4%で実行。

「この機能は予測モデルを使いません。各購入日の始値と資金から購入可能な最大の100株単位を毎回計算し、余剰現金を繰り越しながら、指定率を過去OHLCへ機械的に適用します。同期間保有も同じ100株単位で比較します。手数料、税金、配当、スリッページなどは含みません。」

### 2:35～2:55　Codexによる拡張

「Build Weekでは、CodexとGPT-5.6を使い、3つのCMD版をWebアプリへ統合しました。最新日推論、時系列評価、モデル設定の保存に加え、個別銘柄の安全な定型チャート分析と回帰テストを追加しました。」

「明示的なモデル再評価が終わった後は、通常予測や画面再読み込みのたびにモデルや判定ラインを選び直しません。」

### 2:55～3:00　終了

「結果は研究・情報提供目的で、売買推奨や利益保証ではありません。投資判断は利用者自身の責任で行ってください。」

## Short English narration

### Opening

“This local web app combines a five-trading-day Nikkei direction forecast, a single-stock forecast, and a take-profit and stop-loss backtest.”

### Nikkei 225

“The target is whether the confirmed Nikkei close will be higher or lower five Japanese trading sessions later. During market hours, the app uses the previous confirmed close. It never uses intraday bars or an unfinished daily bar.”

“A Japan-only model and a US-and-overseas model produce separate upward scores. The final score is their fixed fifty-fifty average. This score is model strength for the current input, not historical accuracy or a literal market probability.”

“Each training point uses only the preceding rolling eight years, with three hundred earlier Japanese sessions for feature warm-up. Formal performance is measured over the latest two years in eight outer folds, with three inner folds and a five-session purge.”

“The current two-year result is 54.7 percent, or 267 correct directions out of 488. Balanced accuracy is 53.4 percent. The majority baseline is higher at 57.6 percent, so the interface reports that no baseline advantage was confirmed.”

### Other features

“The individual-stock page accepts one Japanese security code at a time. It shows the current five-session direction, a moving-average chart, and a deterministic report of past chart facts. The report uses no external generative AI and gives no trading timing advice.”

“The page also includes a sixty-session short-term chart, a two-year medium- to long-term chart, analysis reports for both periods, and the market factors behind the current prediction. Individual-stock accuracy is not displayed. The simulation page tests simple long-only exit rules with one million yen.”

### Build Week

“Codex and GPT-5.6 helped preserve the original command-line programs, redesign time alignment and validation, integrate the APIs and interface, persist model settings, and add regression tests.”

### Closing

“Results are for research and information only. They are not trading advice and do not guarantee future returns.”

## デモで避けること

- 上昇スコアを「実際に上がる確率」「予測精度」と呼ばない
- 6段階区分を方向モデルの主要精度と混同しない
- 日経平均の多数派基準を下回った結果を省略しない
- 動画中に重い日経平均の「モデル再評価」を実行しない
- 「必ず当たる」「高精度」「利益を保証」「最適な売買を指示」と表現しない
