export const meta = {
  name: 'abgt-optimize-round',
  description: 'Design, implement, evaluate and adversarially verify exposure-based trend strategy variants for ABGT (round N)',
  phases: [
    { title: 'Design', detail: 'directions × variants, params chosen a priori' },
    { title: 'Implement', detail: 'one agent per variant: write file, run evaluate_variant.py with alt-split / sensitivity / baseline / stress' },
    { title: 'Verify', detail: 'candidates vs 4 lenses: lookahead, overfit (excess Sharpe), executability, holdout SOL/BNB' },
    { title: 'Synthesize', detail: 'ranking, champion, engine proposals, round report' },
  ],
}

const ROOT = 'D:/feihong/tradingView/ABGT'
const PY = `${ROOT}/.venv/Scripts/python.exe`
const round = args.round
const baseline = args.baseline
const ledger = args.ledger || []
const directions = args.directions || []
const CELLS6 = ['BTC_IS', 'BTC_OOS', 'BTC_FULL', 'ETH_IS', 'ETH_OOS', 'ETH_FULL']
const CELLS10 = CELLS6.concat(['BTC_IS2', 'BTC_OOS2', 'ETH_IS2', 'ETH_OOS2'])
const OOS_CELLS = ['BTC_OOS', 'ETH_OOS', 'BTC_OOS2', 'ETH_OOS2']

const COMMON = `
你在幫一個加密貨幣趨勢跟隨策略專案做研究（第 ${round} 輪）。專案根目錄 ${ROOT}（Windows，Bash 裡用正斜線）。
先讀 ${ROOT}/CLAUDE.md 與 ${ROOT}/experiments/rounds/round_${round - 1}.md（上一輪報告：哪些方向有效、哪些已證明無效），再讀 ${ROOT}/abgt/strategies/regime.py（引擎與 RiskParams 全部欄位）、${ROOT}/abgt/indicators.py、${ROOT}/experiments/variants/vol_atr_channel.py（現任冠軍的變體檔）。
核心事實：
- 引擎只做多。策略提供趨勢判定：布林陣列 in_regime，或 dict(enter=..., exit=...) 讓引擎維護狀態機（推薦，少寫一段迴圈）。
- RiskParams 欄位：max_exposure、vol_target / vol_target_mode(absolute|relative) / vol_side(total|downside) / vol_lookback / vol_ref_lookback、atr_len、atr_stop_mult、stop_trigger(low|close)、reentry(reset|immediate|close_above_stop)、reentry_min_bars、dd_pause / dd_pause_resume(regime_reset|new_high|bars) / dd_resume_high_n / dd_resume_bars、funding_rate_8h、exec_delay_bars、rebalance_threshold。
- 評估：BTC 與 ETH 日線；區間 IS 2020-01→2022-12、OOS 2023-01→今、FULL、替換切分 IS2 2020-07→2023-12、OOS2 2024-01→今；閘門 = Sharpe、Calmar 高於同期買入持有且最大回撤更小；敏感度擾動、對基準逐格 ΔSharpe 與配對 t 值、四組壓力情境（高費、高滑點、資金費率 0.01%/8h、延遲一根）。
- 現任冠軍/基準 ${baseline.id}：${baseline.note || ''}。平均 Sharpe（六格）${baseline.meanSharpe}。各格：${JSON.stringify(baseline.cells)}
- 取代冠軍的門檻：十格全過（IS/OOS/FULL/IS2/OOS2 × 2 標的）、六格平均 Sharpe 高於冠軍、四個 OOS 格對冠軍的 ΔSharpe 都 ≥ −0.05、OOS 敏感度通過率 ≥ 0.67、前視/過擬合/holdout 驗證全過。只是「閘門過」不夠。
- 已試過的變體（不要重複）：${ledger.map(l => `${l.id}：${l.note}`).join('；')}
- 不可用未來資料：任何在第 i 根用到的值只能來自 ≤ i 根的收盤資料。日線調倉每次成本約 0.07%。
`

const DESIGN_SCHEMA = {
  type: 'object',
  properties: {
    variants: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string', description: '小寫英數與底線，將成為檔名，需唯一且具描述性' },
          name: { type: 'string' },
          description: { type: 'string' },
          rules: { type: 'string', description: '精確到可直接實作的規則與偽碼，含 enter/exit 或 in_regime 定義與 RiskParams 設定' },
          params: { type: 'object', description: '數值參數與預設值；事前決定，不得看 OOS 調' },
          rationale: { type: 'string' },
          failure_modes: { type: 'string' },
        },
        required: ['id', 'name', 'description', 'rules', 'params', 'rationale'],
      },
    },
  },
  required: ['variants'],
}

const RESULT_SCHEMA = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    file: { type: 'string' },
    ok: { type: 'boolean' },
    error: { type: 'string' },
    cells: { type: 'object', description: '鍵 BTC_IS/BTC_OOS/BTC_FULL/BTC_IS2/BTC_OOS2/ETH_… → {sharpe, calmar, max_dd, net_return, round_trips, bh_sharpe, gate, d_sharpe_vs_baseline, t_daily}' },
    sens: { type: 'object', description: '鍵 BTC_IS/…/ETH_OOS2 → {pass_ratio, sharpe_min, sharpe_max}；另加 n_effective' },
    stress: { type: 'object', description: '四組壓力情境 → 各區間 pass 布林（兩標的）' },
    extra: { type: 'object', description: 'avg_exposure、time_in_market、regime_entries、stops 等' },
    notes: { type: 'string' },
  },
  required: ['id', 'ok'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['pass', 'fail'] },
    severity: { type: 'string', enum: ['none', 'low', 'medium', 'high'] },
    reasons: { type: 'string' },
    evidence: { type: 'string', description: '你實際執行或閱讀到的證據，含數字' },
    fixes: { type: 'string' },
  },
  required: ['verdict', 'severity', 'reasons'],
}

const SYNTH_SCHEMA = {
  type: 'object',
  properties: {
    champion: { type: 'string', description: '本輪應成為新基準的變體 id；沒有就填現任冠軍的 id' },
    champion_reason: { type: 'string' },
    champion_cells: { type: 'object', description: '若換冠軍：六格 {sharpe, max_dd, net_return, gate}，供下一輪當 baseline.cells' },
    champion_mean_sharpe: { type: 'number' },
    ranking: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, gates: { type: 'number' }, mean_sharpe: { type: 'number' }, verified: { type: 'string' }, summary: { type: 'string' } }, required: ['id', 'summary'] } },
    engine_proposals: { type: 'array', items: { type: 'object', properties: { title: { type: 'string' }, detail: { type: 'string' }, expected_benefit: { type: 'string' } }, required: ['title', 'detail'] } },
    next_round_ideas: { type: 'array', items: { type: 'object', properties: { key: { type: 'string' }, prompt: { type: 'string' } }, required: ['key', 'prompt'] } },
    report_path: { type: 'string' },
  },
  required: ['champion', 'ranking', 'engine_proposals', 'next_round_ideas'],
}

phase('Design')
log(`第 ${round} 輪：${directions.length} 個設計方向`)

const results = await pipeline(
  directions,
  d => agent(`${COMMON}
本方向：${d.prompt}
請提出 ${d.n || 2} 個變體。每個變體要能只靠一個變體檔（PARAMS + build(df, params) 回傳 in_regime 或 {enter, exit}、RiskParams、warmup）實作，不需要改引擎。規則寫到能直接翻成 numpy/pandas 程式碼。參數必須事前決定並說明依據，設計階段不要跑回測。id 用小寫英數底線，加上方向前綴 ${d.key}_。不要重複已試過的變體；若要在冠軍上疊加，明說疊加的是哪一段。`,
    { label: `design:${d.key}`, phase: 'Design', schema: DESIGN_SCHEMA }),
  (design, d) => {
    if (!design || !design.variants) { log(`${d.key} 設計失敗，略過`); return [] }
    const vs = design.variants.slice(0, d.n || 2).map(v => ({ ...v, id: v.id.replace(/[^a-z0-9_]/g, '_'), lens: d.key }))
    log(`${d.key} 設計完成：${vs.map(v => v.id).join(', ')}`)
    return vs
  },
  vs => parallel(vs.map(v => () =>
    agent(`${COMMON}
你要實作並評估變體「${v.name}」（id: ${v.id}）。
規則：${v.rules}
參數：${JSON.stringify(v.params)}
理由：${v.rationale}

步驟：
1. 建立 ${ROOT}/experiments/variants/${v.id}.py，遵守變體檔契約（NAME、DESCRIPTION、PARAMS dict、build(df, params) 回傳 {"in_regime": 布林陣列 或 {"enter":..., "exit":...}, "risk": RiskParams(...), "warmup": int}）。只用 abgt.indicators 與 numpy/pandas；warmup 設成所有指標都有效的第一根索引。絕對不能用未來資料。
2. 在 ${ROOT} 下執行：PYTHONIOENCODING=utf-8 ${PY} evaluate_variant.py --variant ${v.id} --alt-split --sensitivity --baseline ${baseline.id} --stress
3. 若報錯，修你自己的變體檔再跑，最多修 3 次。不要改引擎、評估器或別人的變體檔。不可以看 OOS 結果回頭調參數；發現規則本身有 bug 才能改。
4. 讀 ${ROOT}/results/variants/${v.id}.json：symbols[<標的>].windows[<區間>] 有 strategy.{sharpe,calmar,max_dd,net_return,round_trips}、buy_and_hold.sharpe、gate.passed；symbols[<標的>].vs_baseline[<區間>] 有 d_sharpe、t_daily；sensitivity.summary[<區間>] 有 pass_ratio/sharpe_min/sharpe_max 與 n_effective；stress[<情境>][<區間>].pass。把十格（BTC/ETH × IS/OOS/FULL/IS2/OOS2）填進 cells，其餘填 sens / stress / extra。
5. notes 寫下你觀察到的行為（換手、曝險、哪個區間為什麼失敗、壓力情境哪個翻盤）。
只回傳結構化結果。`,
      { label: `impl:${v.id}`, phase: 'Implement', schema: RESULT_SCHEMA }).then(r => r ? { ...r, design: v } : null)
  )),
)

const all = results.flat().filter(Boolean)
const okResults = all.filter(r => r.ok && r.cells)
log(`實作完成：${okResults.length}/${all.length} 個變體有結果` + (all.length - okResults.length ? `（失敗：${all.filter(r => !r.ok).map(r => r.id).join(', ')}）` : ''))

function num(x) { return typeof x === 'number' && !Number.isNaN(x) ? x : null }
function score(r) {
  const c = r.cells || {}
  const gates10 = CELLS10.filter(k => c[k] && c[k].gate).length
  const gates6 = CELLS6.filter(k => c[k] && c[k].gate).length
  const s6 = CELLS6.map(k => c[k] && num(c[k].sharpe)).filter(x => x !== null)
  const meanSharpe = s6.length ? s6.reduce((a, b) => a + b, 0) / s6.length : -9
  const dOOS = OOS_CELLS.map(k => c[k] && num(c[k].d_sharpe_vs_baseline)).filter(x => x !== null)
  const minDOOS = dOOS.length ? Math.min(...dOOS) : -9
  const sensOOS = OOS_CELLS.map(k => r.sens && r.sens[k] && num(r.sens[k].pass_ratio)).filter(x => x !== null)
  const minSens = sensOOS.length ? Math.min(...sensOOS) : 0
  return { gates10, gates6, meanSharpe, minDOOS, minSens }
}
const scored = okResults.map(r => ({ ...r, score: score(r) })).sort((a, b) => (b.score.gates10 - a.score.gates10) || (b.score.meanSharpe - a.score.meanSharpe))
for (const r of scored) log(`${r.id}: 閘門 ${r.score.gates10}/10, 平均 Sharpe ${r.score.meanSharpe.toFixed(2)}, OOS 對冠軍最低 Δ ${r.score.minDOOS.toFixed(2)}, OOS 敏感度最低 ${r.score.minSens.toFixed(2)}`)

let candidates = scored.filter(r => r.score.gates10 >= 9 && r.score.meanSharpe > baseline.meanSharpe && r.score.minDOOS >= -0.05)
if (candidates.length === 0) {
  candidates = scored.slice(0, 2)
  log(`沒有變體達到取代門檻，驗證前 2 名以累積學習：${candidates.map(c => c.id).join(', ')}`)
} else {
  candidates = candidates.slice(0, 4)
  log(`進入驗證：${candidates.map(c => c.id).join(', ')}`)
}

phase('Verify')
const VERIFY_LENSES = [
  { key: 'lookahead', prompt: '視角：前視偏差與實作 bug。逐行讀變體檔，對照 indicators.py 與 regime.py，找任何用到未來資料的地方（rolling 視窗越界、shift 方向、resample 對齊、以整段序列算出的常數當閾值），以及 in_regime/enter/exit 與規則描述不一致之處。務必寫一小段 Python 做截斷測試（把資料截到第 i 根重算 build()，比對 in_regime[:i+1] 是否與全序列前綴一致，取多個 i）。預設懷疑：找不到明確證據才 pass。' },
  { key: 'overfit', prompt: '視角：過擬合與穩健性。結果 json 已含替換切分（IS2/OOS2）、對冠軍的 Δ 與 t 值、±25% 敏感度。你要另外跑 --sens-factors 0.5,1.5，並檢查：(a) 超額 Sharpe（策略 − 同期持有）在 OOS→OOS2 的變化是否掉超過 0.3（注意：不是絕對 Sharpe，持有本身會變）；(b) 參數在 OOS 是否為寬平台（用 evaluate_variant.py 對關鍵參數多跑幾個值）；(c) 對冠軍的日報酬差 t 值是否 ≥ 1（< 1 代表差異在雜訊內）；(d) 參數數量與自由度是否跟改善幅度相稱。任一項嚴重不符判 fail。' },
  { key: 'executable', prompt: '視角：可執行性與經濟合理性。看 round_trips、avg_exposure、四組壓力情境（哪個情境翻盤代表對什麼敏感）、規則是否依賴只在 2020–2026 出現的市況、實盤能否在日線收盤後決策並在下一根開盤成交、有無隱含槓桿、狀態機的路徑依賴（從不同起點起算多久收斂）。給出是否值得上線的判斷與前提。' },
  { key: 'holdout', prompt: `視角：沒碰過的標的。在 ${ROOT} 下執行：PYTHONIOENCODING=utf-8 ${PY} evaluate_variant.py --variant <id> --symbols SOL/USDT:USDT,BNB/USDT:USDT --alt-split --baseline ${baseline.id} --out results/holdout。判準：兩個 holdout 標的合計十格中閘門通過 ≥ 5 格，且 holdout 各格對冠軍的 ΔSharpe 平均 ≥ −0.05（冠軍在 holdout 也可能不過閘門，重點是相對表現）。把每格數字寫進 evidence。` },
]
const verified = await pipeline(
  candidates,
  c => parallel(VERIFY_LENSES.map(vl => () =>
    agent(`${COMMON}
你是驗證者。變體 id ${c.id}，檔案 ${ROOT}/experiments/variants/${c.id}.py，結果 ${ROOT}/results/variants/${c.id}.json。
設計規則：${c.design ? c.design.rules : ''}
評估結果摘要：${JSON.stringify({ cells: c.cells, sens: c.sens, stress: c.stress, extra: c.extra })}
${vl.prompt.replace('<id>', c.id)}
執行 Python 時在 ${ROOT} 下用：PYTHONIOENCODING=utf-8 ${PY} ...。不要修改任何既有檔案（holdout 結果寫到 results/holdout 可以）。回傳 verdict/severity/reasons/evidence/fixes。`,
      { label: `verify:${vl.key}:${c.id}`, phase: 'Verify', schema: VERDICT_SCHEMA }).then(v => ({ lens: vl.key, ...(v || { verdict: 'fail', severity: 'high', reasons: 'agent 未回傳' }) }))
  )).then(vs => ({ ...c, verdicts: vs.filter(Boolean) })),
)

for (const c of verified) log(`${c.id}: ${c.verdicts.map(v => `${v.lens}=${v.verdict}(${v.severity})`).join(' ')}`)

phase('Synthesize')
const synth = await agent(`${COMMON}
你是本輪的綜合者。下面是第 ${round} 輪所有變體的設計、評估結果與驗證結論（JSON）。
基準：${JSON.stringify(baseline)}
所有結果：${JSON.stringify(scored.map(r => ({ id: r.id, lens: r.lens || (r.design && r.design.lens), name: r.design && r.design.name, description: r.design && r.design.description, params: r.design && r.design.params, score: r.score, cells: r.cells, sens: r.sens, stress: r.stress, extra: r.extra, notes: r.notes })))}
失敗的實作：${JSON.stringify(all.filter(r => !r.ok).map(r => ({ id: r.id, error: r.error })))}
驗證：${JSON.stringify(verified.map(c => ({ id: c.id, verdicts: c.verdicts })))}

任務：
1. 決定 champion：只有「十格 ≥ 9 且四個 OOS 格全過、六格平均 Sharpe 高於冠軍、四個 OOS 格對冠軍 ΔSharpe 都 ≥ −0.05、lookahead / overfit / holdout 三個驗證都 pass」的變體才能取代；否則 champion 填 ${baseline.id}。若換冠軍，填 champion_cells（六格）與 champion_mean_sharpe。
2. 排名所有有結果的變體，每個一句話總結它教了我們什麼（跟上一輪報告的結論對照，指出哪些假設被證實或推翻）。
3. 提出 2–4 個引擎/風控模型的改進提案（需要改 regime.py 才能做的事），要具體到欄位與語意；已在本輪引擎實作的東西不要再提。
4. 提出下一輪 4–6 個設計方向，每個是 {key, prompt}：key 小寫英數底線，prompt 要寫給設計者看、包含假設與判準；避開已證明無效的方向。
5. 把完整回合報告（繁體中文、含表格：變體 × 十格 Sharpe/回撤/閘門、對冠軍 Δ 與 t、敏感度、壓力、驗證結論、結論與建議）寫到 ${ROOT}/experiments/rounds/round_${round}.md（用 Bash mkdir -p 再寫檔），report_path 回傳該路徑。`,
  { label: 'synthesize', phase: 'Synthesize', schema: SYNTH_SCHEMA })

return {
  round,
  n_designed: all.length,
  n_ok: okResults.length,
  scored: scored.map(r => ({ id: r.id, gates10: r.score.gates10, gates6: r.score.gates6, meanSharpe: r.score.meanSharpe, minDOOS: r.score.minDOOS, minSens: r.score.minSens, file: r.file })),
  failed: all.filter(r => !r.ok).map(r => ({ id: r.id, error: r.error })),
  verified: verified.map(c => ({ id: c.id, verdicts: c.verdicts.map(v => ({ lens: v.lens, verdict: v.verdict, severity: v.severity, reasons: v.reasons })) })),
  synthesis: synth,
}
