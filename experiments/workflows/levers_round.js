export const meta = {
  name: 'abgt-levers',
  description: 'Quantify the four capital levers for ABGT (diversification, costs, capital curve, leverage) with independent skeptic verification',
  phases: [
    { title: 'Analyze', detail: 'one analyst per lever writes a script + results json' },
    { title: 'Verify', detail: 'one skeptic per lever recomputes key numbers independently' },
    { title: 'Synthesize', detail: 'levers.md + structured recommendations' },
  ],
}

const ROOT = 'D:/feihong/tradingView/ABGT'
const PY = `${ROOT}/.venv/Scripts/python.exe`

const COMMON = `
你在幫一個加密貨幣趨勢跟隨策略專案做量化分析。專案根目錄 ${ROOT}（Windows，Bash 裡用正斜線）。
先讀 ${ROOT}/CLAUDE.md（尤其第 1、5 節）、${ROOT}/experiments/rounds/round_4.md 的結論、${ROOT}/abgt/strategies/regime.py（RiskParams：max_exposure、borrow_rate_annual、vol_target、commission_pct、slippage_pct、funding…）、${ROOT}/abgt/portfolio.py（simulate_portfolio、buy_and_hold_portfolio）、${ROOT}/evaluate_portfolio.py（--fee --slip --max-exposure --borrow-rate --vol-target --symbols --weights --stress）、${ROOT}/evaluate_variant.py。
核心事實：
- 冠軍 ema50_plain：close > EMA50 滿倉否則空手，只做多，收盤決策次日開盤成交。四輪 36 個變體沒有任何疊加規則在 OOS 對它有可分辨改善，優化已停；現在的目標是把「本金 × 邊際」的三個槓桿量化，以及回答槓桿能不能合理使用。
- 資料快取在 ${ROOT}/data/：永續 BTC/USDT:USDT、ETH/USDT:USDT、SOL/USDT:USDT（2020-09 起）、BNB/USDT:USDT（2020-02 起）；現貨 BTC/USDT、ETH/USDT、SOL/USDT、BNB/USDT（2019-09 起，SOL 2020-08）；資金費率 *_funding.csv。用 abgt.data.load_cached("data", symbol, "1d", "2019-09-01")。
- 區間：IS 2020-01→2022-12、OOS 2023-01→今、FULL、IS2 2020-07→2023-12、OOS2 2024-01→今。閘門 = Sharpe、Calmar 高於同期持有且最大回撤更小。
- 已知：BTC/ETH 50/50 ema50_plain 組合十格對兩種持有基準全過，Sharpe IS 1.24 / OOS 1.13 / FULL 1.17 / IS2 1.39 / OOS2 0.77，回撤 −59% vs 持有 −78%；1.5× 固定槓桿 + 借貸 8% 的煙霧：FULL Sharpe 1.13（比 1× 的 1.17 低）、回撤 −74%。
- 上線口徑：現貨、費 0.10%/邊（BNB 抵扣 0.075%，maker 0.02%）、滑 0.05%。真錢還沒投入。
- 每個分析寫成一個可重跑的腳本 ${ROOT}/experiments/levers/<key>.py，結果寫 ${ROOT}/results/levers/<key>.json 與 <key>.md（表格），不要改任何既有檔案。Python 用 PYTHONIOENCODING=utf-8 ${PY}。
- 所有數字都要能被別人獨立重算；不用未來資料；假設要明寫。結論要誠實：不好的結果照寫。
`

const RESULT = {
  type: 'object',
  properties: {
    key: { type: 'string' }, ok: { type: 'boolean' }, error: { type: 'string' },
    script: { type: 'string' }, results_json: { type: 'string' }, results_md: { type: 'string' },
    headline: { type: 'string', description: '一句話結論' },
    key_numbers: { type: 'object', description: '別人要重算的關鍵數字，鍵值對' },
    recommendation: { type: 'string' },
    assumptions: { type: 'string' },
    caveats: { type: 'string' },
  },
  required: ['key', 'ok', 'headline', 'key_numbers', 'recommendation'],
}
const VERDICT = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['pass', 'fail'] }, severity: { type: 'string', enum: ['none', 'low', 'medium', 'high'] },
    reasons: { type: 'string' }, recomputed: { type: 'object', description: '你獨立重算出的關鍵數字' }, corrections: { type: 'string' },
  },
  required: ['verdict', 'severity', 'reasons'],
}
const SYNTH = {
  type: 'object',
  properties: {
    report_path: { type: 'string' },
    diversification: { type: 'string' }, costs: { type: 'string' }, capital: { type: 'string' }, leverage: { type: 'string' },
    leverage_policy: { type: 'object', description: '{max_exposure, instrument, borrow_rate_assumed, conditions, why}' },
    executor_requirements: { type: 'array', items: { type: 'string' } },
    risks: { type: 'array', items: { type: 'string' } },
    one_paragraph: { type: 'string', description: '給使用者的一段話：現在比純持有強在哪、弱在哪、三個槓桿各能加多少、槓桿該不該用' },
  },
  required: ['report_path', 'diversification', 'costs', 'capital', 'leverage', 'leverage_policy', 'executor_requirements', 'risks', 'one_paragraph'],
}

const LEVERS = [
  { key: 'diversification', prompt: `標的分散。用 simulate_portfolio / evaluate_portfolio 跑等權組合：BTC 單獨、BTC+ETH、BTC+ETH+SOL、BTC+ETH+SOL+BNB，各用 ema50_plain（另跑 vol_atr_channel 對照），現貨資料為主（永續為對照），費 0.10%/邊 滑 0.05%（上線口徑）。每個組合報十格（IS/OOS/FULL/IS2/OOS2 × 對「持有不再平衡」與「持有月再平衡」）Sharpe / Calmar / 回撤 / 相對落後 / 閘門、名目換手、各標的平均曝險。注意 SOL 2020-08 起、BNB 2020-02 起，IS 視窗不同要標明；也要報「同視窗下 BTC+ETH 的數字」讓比較公平。回答：多加標的的 Sharpe 增量是多少、回撤有沒有變深、哪個組合在 OOS 最穩、等權以外要不要試 BTC 加權（例如 60/40、50/30/20）— 加權只能事前用「市值或流動性」這種非績效理由，不得用回測挑。` },
  { key: 'costs', prompt: `成本。對 BTC 單獨、ETH 單獨、BTC+ETH 50/50（ema50_plain，現貨資料）跑手續費 × 滑點網格：費 {0.02%(maker), 0.075%(BNB 抵扣), 0.10%(taker)} × 滑 {0.02%, 0.05%, 0.10%, 0.20%}，報十格閘門通過數與 OOS/OOS2 Sharpe；找出每個標的的「損益兩平總成本」（OOS 閘門翻掛的每邊成本）；用名目換手估年化成本拖累（%/年）。再評估執行層的「先掛限價當 maker、N 分鐘沒成交轉市價」：用日線 open/high/low 估「開盤價掛限價在當天成交」的機率上下界（例如買單掛在開盤價，當日 low ≤ 開盤價即視為成交），算 maker 成交率 p 下的期望成本 = p×0.02% + (1−p)×0.10%，以及未成交那部分延遲到收盤的機會成本（用同日 open→close 的報酬分布）。結論：執行層該用哪種下單方式、預期把成本從多少壓到多少、對 ETH 閘門有沒有決定性影響。` },
  { key: 'capital', prompt: `本金曲線。用 BTC+ETH 50/50 ema50_plain 組合（現貨、費 0.10% 滑 0.05%）的日權益曲線做定期投入模擬：起始本金 {1 萬, 5 萬, 30 萬} × 每月投入 {0, 1000, 3000} 美元，起點分別 2020-01、2021-11（頂部）、2023-01，投入策略 vs 投入同權重買入持有；報期末 NAV、投入總額、金錢加權報酬（IRR）、時間加權報酬、最深美元回撤與最深百分比回撤、水下最長天數。再做「本金 × 年化」表：用 OOS 2023→ 的組合 CAGR 當基準、其一半當保守、IS 當樂觀，算本金 1/5/10/30/100 萬在 1/3/5/10 年的期望 NAV 與可能的最壞回撤金額；算「從 1 萬 + 每月 X 到 30 萬」在三種 CAGR 下要幾年。回答：本金曲線對結果的影響比策略參數大多少（用同一條策略換本金和投入節奏的差距量化）、在回撤中加碼是否有意義（用歷史回撤段做「回撤 ≥ 20% 時加倍投入」的對照）。` },
  { key: 'leverage', prompt: `槓桿。(1) 從 BTC 單獨與 BTC+ETH 50/50 組合（ema50_plain，現貨，費 0.10% 滑 0.05%）的日對數報酬算年化 μ、σ、Sharpe，Kelly 最適槓桿 f* = μ/σ²（含扣借貸成本的版本），半 Kelly；分 IS / OOS / FULL 各算一次，看 f* 穩不穩。(2) 固定槓桿網格：max_exposure ∈ {1.0, 1.25, 1.5, 2.0, 2.5} × 借貸年化 {5%, 10%, 15%}，用 evaluate_portfolio --max-exposure --borrow-rate 跑十格，報 Sharpe / 回撤 / 最低權益 / 借貸成本 / 閘門；找出「回撤不超過持有」與「Sharpe 不低於 1×」的最大槓桿。(3) 波動目標型槓桿：--vol-target {0.4, 0.6} × --max-exposure {1.5, 2.0}，同上。(4) 爆倉與跳空風險：以 Binance 現貨全倉保證金（借 USDT 買幣、維持保證金率 1.1 強平）估各槓桿下的強平價格距離，對照歷史最大單日跌幅（BTC 2020-03-12、ETH 同日）與策略在場時的最大單日跌幅分布；報「策略出場（EMA50）先於強平的機率」— 用歷史每次趨勢內的最深回撤與單日跌幅算。(5) 借貸成本用 Binance 現貨保證金 USDT 借款利率的量級（年化 5–15%，寫明假設）。結論要給一個可執行的政策：合理槓桿是多少、用什麼工具（現貨保證金 / 永續）、什麼條件下才啟用（例如本金 ≥ X、紙上前向 ≥ N 月、回撤 < Y）、什麼條件下降回 1×；或者結論是「不用」也要說清楚為什麼。` },
]

phase('Analyze')
const results = await pipeline(
  LEVERS,
  l => agent(`${COMMON}
你負責「${l.key}」這個槓桿的分析。任務：${l.prompt}
交付：腳本 ${ROOT}/experiments/levers/${l.key}.py（用 Bash mkdir -p 建目錄；可重跑、印表格、寫 ${ROOT}/results/levers/${l.key}.json 與 .md）。跑完後把關鍵數字填進 key_numbers（別人要能獨立重算），headline 一句話，recommendation 要具體到數字，assumptions 與 caveats 誠實列。只回傳結構化結果。`,
    { label: `analyze:${l.key}`, phase: 'Analyze', schema: RESULT }).then(r => r ? { ...r, lever: l } : null),
  r => r ? agent(`${COMMON}
你是懷疑者，負責獨立重算「${r.lever.key}」分析的關鍵數字並找錯。分析者的交付：腳本 ${r.script || ROOT + '/experiments/levers/' + r.lever.key + '.py'}、結果 ${r.results_json || ''}、headline「${r.headline}」、key_numbers ${JSON.stringify(r.key_numbers)}、recommendation「${r.recommendation}」、assumptions「${r.assumptions || ''}」。
原始任務：${r.lever.prompt}
做法：不要只讀他的腳本，自己另寫一段 Python（放 ${ROOT}/results/levers/${r.lever.key}_verify.py）用專案引擎或純 numpy 重算至少 5 個關鍵數字，比對差異；檢查是否用到未來資料、成本有沒有算進去（手續費、滑點、借貸）、視窗是否公平（不同標的起點）、假設是否樂觀（maker 成交率、借貸利率、無強平）、結論是否被數字支持。輸出 recomputed（你的數字）、reasons、corrections（哪些結論該改）。找不到問題才 pass。`,
    { label: `verify:${r.lever.key}`, phase: 'Verify', schema: VERDICT }).then(v => ({ ...r, verdict: v })) : null,
)

const done = results.filter(Boolean)
for (const r of done) log(`${r.lever.key}: ${r.ok ? 'ok' : 'FAIL'} | ${r.headline} | 驗證 ${r.verdict ? r.verdict.verdict + '(' + r.verdict.severity + ')' : '無'}`)

phase('Synthesize')
const synth = await agent(`${COMMON}
你是綜合者。四個槓桿分析與各自的獨立驗證如下（JSON）：
${JSON.stringify(done.map(r => ({ key: r.lever.key, ok: r.ok, error: r.error, script: r.script, results_md: r.results_md, headline: r.headline, key_numbers: r.key_numbers, recommendation: r.recommendation, assumptions: r.assumptions, caveats: r.caveats, verdict: r.verdict })))}
任務：
1. 讀每個 results/levers/<key>.md 與驗證者的 corrections，以驗證者的數字為準處理衝突（兩者差異要寫出來）。
2. 寫 ${ROOT}/experiments/rounds/levers.md（繁體中文）：每個槓桿一節（表格 + 結論 + 驗證結果 + 該改的地方）、槓桿政策一節（可執行：倍數、工具、啟用/退場條件、借貸假設）、執行層需求一節（下單方式、成本目標、對帳、風險控制）、風險清單、最後一段給使用者的話。
3. 回傳結構化摘要；leverage_policy 要對得上引擎欄位（max_exposure、borrow_rate_annual、vol_target）。`,
  { label: 'synthesize', phase: 'Synthesize', schema: SYNTH })

return { levers: done.map(r => ({ key: r.lever.key, ok: r.ok, headline: r.headline, key_numbers: r.key_numbers, recommendation: r.recommendation, verdict: r.verdict })), synthesis: synth }
