export const meta = {
  name: 'abgt-optimize-round',
  description: 'Design, implement, evaluate and adversarially verify exposure-based trend strategy variants and running-gate guards for ABGT (round N)',
  phases: [
    { title: 'Design', detail: 'directions × variants, params chosen a priori' },
    { title: 'Implement', detail: 'one agent per variant: write file, run evaluate_variant.py with alt-split / sensitivity / baseline / stress' },
    { title: 'Verify', detail: 'candidates vs 4 lenses: lookahead, overfit (placebo + excess Sharpe), executability, holdout SOL/BNB' },
    { title: 'Synthesize', detail: 'ranking, champion (Sharpe path or risk path), engine proposals, round report' },
  ],
}

const ROOT = 'D:/feihong/tradingView/ABGT'
const PY = `${ROOT}/.venv/Scripts/python.exe`
const round = args.round
const baseline = args.baseline
const ledger = args.ledger || []
const directions = args.directions || []
const ACCEPT = Object.assign({ oos_avg_delta: 0.05, t_min: 1.5, oos_min_delta: -0.05, risk_path_dd_pp: 0.03, holdout_avg_delta: 0.0 }, args.accept || {})
const CELLS6 = ['BTC_IS', 'BTC_OOS', 'BTC_FULL', 'ETH_IS', 'ETH_OOS', 'ETH_FULL']
const CELLS10 = CELLS6.concat(['BTC_IS2', 'BTC_OOS2', 'ETH_IS2', 'ETH_OOS2'])
const OOS_CELLS = ['BTC_OOS', 'ETH_OOS', 'BTC_OOS2', 'ETH_OOS2']

const COMMON = `
你在幫一個加密貨幣趨勢跟隨策略專案做研究（第 ${round} 輪）。專案根目錄 ${ROOT}（Windows，Bash 裡用正斜線）。
先讀 ${ROOT}/CLAUDE.md 與 ${ROOT}/experiments/rounds/round_${round - 1}.md（上一輪報告：哪些方向有效、哪些已證明無效、安慰劑檢定的結論），再讀 ${ROOT}/abgt/strategies/regime.py（引擎與 RiskParams 全部欄位，含 scale_mode / scale_floor / lag_stop 運行閘門）、${ROOT}/abgt/indicators.py、${ROOT}/experiments/variants/ema50_plain.py 與 graded_exposure_dist_ratchet.py（變體檔範例，後者含 scale 陣列）。
核心事實：
- 引擎只做多。策略提供趨勢判定：布林陣列 in_regime，或 dict(enter=..., exit=..., scale=...) 讓引擎維護狀態機與分級曝險。
- RiskParams 欄位：max_exposure、scale_mode(track|ratchet_up)、scale_floor、vol_target 系列、atr_stop_mult / stop_trigger / reentry / reentry_min_bars、dd_pause 系列、lag_stop / lag_window / lag_action(hold|flat) / lag_resume(regime_reset|bars|ratio_high) / lag_resume_bars、funding_rate_8h / funding、exec_delay_bars、rebalance_threshold。
- 評估：BTC 與 ETH 日線；區間 IS 2020-01→2022-12、OOS 2023-01→今、FULL、IS2 2020-07→2023-12、OOS2 2024-01→今；閘門 = Sharpe、Calmar 高於同期買入持有且最大回撤更小；每格另有「相對落後」= 策略/持有比值自高點的最深回撤；對基準逐格 ΔSharpe 與配對 t；四組壓力 + 歷史資金費率；extra 有 fills_by_reason、notional_turnover、lag_triggers / lag_events、rel_dd_vs_hold。
- 現任冠軍/基準 ${baseline.id}：${baseline.note || ''}。六格平均 Sharpe ${baseline.meanSharpe}。各格：${JSON.stringify(baseline.cells)}
- 取代冠軍的門檻（第 4 輪起，因為第 3 輪安慰劑證明舊門檻沒鑑別力）：十格 ≥ 9 且四個 OOS 格全過、四個 OOS 格對冠軍 ΔSharpe 都 ≥ ${ACCEPT.oos_min_delta}，然後二擇一 —— Sharpe 路徑：四 OOS 格 Δ 平均 ≥ +${ACCEPT.oos_avg_delta} 且至少一格 |t| ≥ ${ACCEPT.t_min}；風險路徑：四 OOS 格最大回撤全部比冠軍淺 ≥ ${ACCEPT.risk_path_dd_pp * 100} pp 且四 OOS 格 Δ 平均 ≥ −0.02。再加 lookahead / overfit（含安慰劑：把變體的關鍵訊號隨機置換 ≥ 100 次，變體要高於安慰劑分布第 90 百分位）/ holdout（SOL+BNB 平均 Δ ≥ ${ACCEPT.holdout_avg_delta}、冠軍過閘格不翻掛）三道驗證全過。
- 運行閘門（guard）類方向不是要贏 Sharpe，是要在「幾乎不傷 Sharpe」的前提下把相對持有的最壞落後鎖住；判準寫在該方向的 prompt 裡，驗證者依那些判準判 pass/fail。
- 已試過的變體（不要重複）：${ledger.map(l => `${l.id}：${l.note}`).join('；')}
- 不可用未來資料：任何在第 i 根用到的值只能來自 ≤ i 根的收盤資料。日線調倉每次成本約 0.07%。參數一律事前登記，不得看 OOS 調。
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
          rules: { type: 'string', description: '精確到可直接實作的規則與偽碼，含 enter/exit/scale 或 in_regime 定義與 RiskParams 設定' },
          params: { type: 'object', description: '數值參數與預設值；事前決定，不得看 OOS 調' },
          rationale: { type: 'string' },
          failure_modes: { type: 'string' },
          pre_registered_criteria: { type: 'string', description: '事前登記的成功/關閉判準（數字）' },
        },
        required: ['id', 'name', 'description', 'rules', 'params', 'rationale', 'pre_registered_criteria'],
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
    cells: { type: 'object', description: '鍵 BTC_IS/BTC_OOS/BTC_FULL/BTC_IS2/BTC_OOS2/ETH_… → {sharpe, calmar, max_dd, net_return, round_trips, bh_sharpe, gate, rel_dd_vs_hold, d_sharpe_vs_baseline, d_max_dd_vs_baseline, t_daily}' },
    sens: { type: 'object', description: '鍵 BTC_IS/…/ETH_OOS2 → {pass_ratio, sharpe_min, sharpe_max}；另加 n_effective' },
    stress: { type: 'object', description: '五組壓力情境 → 各區間 pass 布林（兩標的）' },
    extra: { type: 'object', description: 'avg_exposure、time_in_market、regime_entries、n_fills、notional_turnover_per_year、lag_triggers、rel_dd_vs_hold 等' },
    criteria_check: { type: 'string', description: '對照自己事前登記的判準，逐條寫過/不過與數字' },
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
    champion_path: { type: 'string', description: 'sharpe | risk | none' },
    champion_cells: { type: 'object', description: '若換冠軍：六格 {sharpe, max_dd, net_return, gate}，供下一輪當 baseline.cells' },
    champion_mean_sharpe: { type: 'number' },
    guard_recommendation: { type: 'object', description: '運行閘門的結論：{variant_id, lag_stop, lag_window, lag_action, lag_resume, lag_resume_bars, verdict, reason}；沒有 guard 方向就空物件' },
    ranking: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, kind: { type: 'string' }, gates: { type: 'number' }, mean_sharpe: { type: 'number' }, verified: { type: 'string' }, summary: { type: 'string' } }, required: ['id', 'summary'] } },
    engine_proposals: { type: 'array', items: { type: 'object', properties: { title: { type: 'string' }, detail: { type: 'string' }, expected_benefit: { type: 'string' } }, required: ['title', 'detail'] } },
    next_round_ideas: { type: 'array', items: { type: 'object', properties: { key: { type: 'string' }, prompt: { type: 'string' } }, required: ['key', 'prompt'] } },
    report_path: { type: 'string' },
  },
  required: ['champion', 'champion_path', 'ranking', 'engine_proposals', 'next_round_ideas'],
}

phase('Design')
log(`第 ${round} 輪：${directions.length} 個設計方向`)

const results = await pipeline(
  directions,
  d => agent(`${COMMON}
本方向（kind=${d.kind || 'strategy'}）：${d.prompt}
請提出 ${d.n || 2} 個變體。每個變體要能只靠一個變體檔（PARAMS + build(df, params) 回傳 in_regime 或 {enter, exit, scale}、RiskParams、warmup）實作，不需要改引擎。規則寫到能直接翻成 numpy/pandas 程式碼。參數與成功/關閉判準必須事前登記並說明依據，設計階段不要跑回測。id 用小寫英數底線，加上方向前綴 ${d.key}_。不要重複已試過的變體。`,
    { label: `design:${d.key}`, phase: 'Design', schema: DESIGN_SCHEMA }),
  (design, d) => {
    if (!design || !design.variants) { log(`${d.key} 設計失敗，略過`); return [] }
    const vs = design.variants.slice(0, d.n || 2).map(v => ({ ...v, id: v.id.replace(/[^a-z0-9_]/g, '_'), lens: d.key, kind: d.kind || 'strategy', direction_prompt: d.prompt }))
    log(`${d.key} 設計完成：${vs.map(v => v.id).join(', ')}`)
    return vs
  },
  vs => parallel(vs.map(v => () =>
    agent(`${COMMON}
你要實作並評估變體「${v.name}」（id: ${v.id}，kind=${v.kind}）。
規則：${v.rules}
參數：${JSON.stringify(v.params)}
理由：${v.rationale}
事前登記的判準：${v.pre_registered_criteria}

步驟：
1. 建立 ${ROOT}/experiments/variants/${v.id}.py，遵守變體檔契約（NAME、DESCRIPTION、PARAMS dict、build(df, params) 回傳 {"in_regime": 布林陣列 或 {"enter":..., "exit":..., "scale":...}, "risk": RiskParams(...), "warmup": int}）。只用 abgt.indicators 與 numpy/pandas；warmup 設成所有指標都有效的第一根索引。絕對不能用未來資料。
2. 在 ${ROOT} 下執行：PYTHONIOENCODING=utf-8 ${PY} evaluate_variant.py --variant ${v.id} --alt-split --sensitivity --baseline ${baseline.id} --stress
3. 若報錯，修你自己的變體檔再跑，最多修 3 次。不要改引擎、評估器或別人的變體檔。不可以看 OOS 結果回頭調參數；發現規則本身有 bug 才能改。
4. 讀 ${ROOT}/results/variants/${v.id}.json：symbols[<標的>].windows[<區間>] 有 strategy.{sharpe,calmar,max_dd,net_return,round_trips}、buy_and_hold.sharpe、gate.passed、rel_dd_vs_hold；symbols[<標的>].vs_baseline[<區間>] 有 d_sharpe、d_max_dd、t_daily；sensitivity.summary[<區間>]；stress[<情境>][<區間>].pass；extra。把十格填進 cells，其餘填 sens / stress / extra。
5. criteria_check：對照你事前登記的判準逐條寫過/不過與數字。notes 寫觀察（換手、曝險、哪個區間為什麼失敗、壓力情境哪個翻盤、guard 的觸發次數與時點）。
只回傳結構化結果。`,
      { label: `impl:${v.id}`, phase: 'Implement', schema: RESULT_SCHEMA }).then(r => r ? { ...r, design: v, kind: v.kind } : null)
  )),
)

const all = results.flat().filter(Boolean)
const okResults = all.filter(r => r.ok && r.cells)
log(`實作完成：${okResults.length}/${all.length} 個變體有結果` + (all.length - okResults.length ? `（失敗：${all.filter(r => !r.ok).map(r => r.id).join(', ')}）` : ''))

function num(x) { return typeof x === 'number' && !Number.isNaN(x) ? x : null }
function score(r) {
  const c = r.cells || {}
  const gates10 = CELLS10.filter(k => c[k] && c[k].gate).length
  const oosAllPass = OOS_CELLS.every(k => c[k] && c[k].gate)
  const s6 = CELLS6.map(k => c[k] && num(c[k].sharpe)).filter(x => x !== null)
  const meanSharpe = s6.length ? s6.reduce((a, b) => a + b, 0) / s6.length : -9
  const dOOS = OOS_CELLS.map(k => c[k] && num(c[k].d_sharpe_vs_baseline)).filter(x => x !== null)
  const minDOOS = dOOS.length ? Math.min(...dOOS) : -9
  const avgDOOS = dOOS.length ? dOOS.reduce((a, b) => a + b, 0) / dOOS.length : -9
  const tOOS = OOS_CELLS.map(k => c[k] && num(c[k].t_daily)).filter(x => x !== null)
  const maxAbsT = tOOS.length ? Math.max(...tOOS.map(Math.abs)) : 0
  const ddOOS = OOS_CELLS.map(k => c[k] && num(c[k].d_max_dd_vs_baseline)).filter(x => x !== null)
  const minDDImprove = ddOOS.length ? Math.min(...ddOOS) : -9   // 正值 = 回撤更淺
  const sensOOS = OOS_CELLS.map(k => r.sens && r.sens[k] && num(r.sens[k].pass_ratio)).filter(x => x !== null)
  const minSens = sensOOS.length ? Math.min(...sensOOS) : 0
  const sharpePath = gates10 >= 9 && oosAllPass && minDOOS >= ACCEPT.oos_min_delta && avgDOOS >= ACCEPT.oos_avg_delta && maxAbsT >= ACCEPT.t_min
  const riskPath = gates10 >= 9 && oosAllPass && minDOOS >= ACCEPT.oos_min_delta && avgDOOS >= -0.02 && minDDImprove >= ACCEPT.risk_path_dd_pp
  return { gates10, oosAllPass, meanSharpe, minDOOS, avgDOOS, maxAbsT, minDDImprove, minSens, sharpePath, riskPath }
}
const scored = okResults.map(r => ({ ...r, score: score(r) })).sort((a, b) => (b.score.gates10 - a.score.gates10) || (b.score.avgDOOS - a.score.avgDOOS))
for (const r of scored) log(`${r.id} [${r.kind}]: 閘門 ${r.score.gates10}/10, 平均 Sharpe ${r.score.meanSharpe.toFixed(2)}, OOS Δ 平均 ${r.score.avgDOOS.toFixed(3)} 最低 ${r.score.minDOOS.toFixed(3)}, |t|max ${r.score.maxAbsT.toFixed(2)}, 回撤改善最小 ${(r.score.minDDImprove * 100).toFixed(1)}pp, 路徑 ${r.score.sharpePath ? 'Sharpe' : r.score.riskPath ? '風險' : '無'}`)

const guards = scored.filter(r => r.kind === 'guard')
let strategyCands = scored.filter(r => r.kind !== 'guard' && (r.score.sharpePath || r.score.riskPath))
if (strategyCands.length === 0) {
  strategyCands = scored.filter(r => r.kind !== 'guard').slice(0, 2)
  log(`沒有策略變體達到取代門檻，驗證前 2 名以累積學習：${strategyCands.map(c => c.id).join(', ')}`)
}
const candidates = strategyCands.slice(0, 4).concat(guards)
log(`進入驗證：${candidates.map(c => `${c.id}[${c.kind}]`).join(', ') || '無'}`)

phase('Verify')
const VERIFY_LENSES = [
  { key: 'lookahead', prompt: '視角：前視偏差與實作 bug。逐行讀變體檔，對照 indicators.py 與 regime.py（含 scale_mode / lag_stop 的語意），找任何用到未來資料的地方（rolling 視窗越界、shift 方向、resample 對齊、以整段序列算出的常數當閾值），以及 in_regime/enter/exit/scale 與規則描述不一致之處。務必寫一小段 Python 做截斷測試（把資料截到第 i 根重算 build()，比對 in_regime[:i+1] 與 scale[:i+1] 是否與全序列前綴一致，取多個 i）。預設懷疑：找不到明確證據才 pass。' },
  { key: 'overfit', prompt: '視角：過擬合與穩健性，含安慰劑。結果 json 已含替換切分、對冠軍的 Δ 與 t、±25% 敏感度。你要另外做：(a) 安慰劑 —— 把變體的關鍵訊號（進場濾網 / scale 級別 / 觸發時點）在保留分布下隨機置換 ≥ 100 次，重跑六格與四 OOS 格，報告實際變體落在安慰劑分布的哪個百分位（六格平均與四 OOS Δ 平均都要 ≥ 第 90 百分位才算有內容）；(b) 去掉貢獻最大的 3 個事件段後四 OOS 格 Δ 是否仍 ≥ −0.05；(c) 對冠軍的日報酬差 t 值（< 1 代表差異在雜訊內）；(d) 參數平台（關鍵參數 ±50% 與相鄰值）。guard 類變體改查：X 與恢復規則的擾動是否改變結論、觸發是否集中在少數事件。任一項嚴重不符判 fail。' },
  { key: 'executable', prompt: '視角：可執行性與經濟合理性。看 round_trips、n_fills、notional_turnover、fills_by_reason、五組壓力情境（哪個翻盤代表對什麼敏感）、規則是否依賴只在 2020–2026 出現的市況、實盤能否在日線收盤後決策並在下一根開盤成交、有無隱含槓桿、狀態機從不同起點起算多久收斂。guard 類變體另查：觸發後轉持有/空手在實務上的意義（例如 hold 在熊市觸發等於買在半山腰）、最壞相對落後是否真的被鎖在 −(X + 成本) 內。給出是否值得上線的判斷與前提。' },
  { key: 'holdout', prompt: `視角：沒碰過的標的。在 ${ROOT} 下執行：PYTHONIOENCODING=utf-8 ${PY} evaluate_variant.py --variant <id> --symbols SOL/USDT:USDT,BNB/USDT:USDT --alt-split --baseline ${baseline.id} --stress --out results/holdout。判準：holdout 各標的對冠軍的 ΔSharpe 平均 ≥ ${ACCEPT.holdout_avg_delta}，且冠軍過閘的格沒有一格被翻掛；guard 類另看 holdout 上的觸發次數與最壞相對落後是否同樣被鎖住。把每格數字寫進 evidence。` },
]
const verified = await pipeline(
  candidates,
  c => parallel(VERIFY_LENSES.map(vl => () =>
    agent(`${COMMON}
你是驗證者。變體 id ${c.id}（kind=${c.kind}），檔案 ${ROOT}/experiments/variants/${c.id}.py，結果 ${ROOT}/results/variants/${c.id}.json。
設計規則：${c.design ? c.design.rules : ''}
事前登記的判準：${c.design ? c.design.pre_registered_criteria : ''}
方向說明：${c.design ? c.design.direction_prompt : ''}
評估結果摘要：${JSON.stringify({ cells: c.cells, sens: c.sens, stress: c.stress, extra: c.extra, criteria_check: c.criteria_check })}
${vl.prompt.replace('<id>', c.id)}
執行 Python 時在 ${ROOT} 下用：PYTHONIOENCODING=utf-8 ${PY} ...。不要修改任何既有檔案（結果寫到 results/holdout 或 results/verify 可以）。回傳 verdict/severity/reasons/evidence/fixes。`,
      { label: `verify:${vl.key}:${c.id}`, phase: 'Verify', schema: VERDICT_SCHEMA }).then(v => ({ lens: vl.key, ...(v || { verdict: 'fail', severity: 'high', reasons: 'agent 未回傳' }) }))
  )).then(vs => ({ ...c, verdicts: vs.filter(Boolean) })),
)

for (const c of verified) log(`${c.id}: ${c.verdicts.map(v => `${v.lens}=${v.verdict}(${v.severity})`).join(' ')}`)

phase('Synthesize')
const synth = await agent(`${COMMON}
你是本輪的綜合者。下面是第 ${round} 輪所有變體的設計、評估結果與驗證結論（JSON）。
基準：${JSON.stringify(baseline)}
接受門檻：${JSON.stringify(ACCEPT)}
所有結果：${JSON.stringify(scored.map(r => ({ id: r.id, kind: r.kind, lens: r.lens || (r.design && r.design.lens), name: r.design && r.design.name, description: r.design && r.design.description, params: r.design && r.design.params, criteria: r.design && r.design.pre_registered_criteria, criteria_check: r.criteria_check, score: r.score, cells: r.cells, sens: r.sens, stress: r.stress, extra: r.extra, notes: r.notes })))}
失敗的實作：${JSON.stringify(all.filter(r => !r.ok).map(r => ({ id: r.id, error: r.error })))}
驗證：${JSON.stringify(verified.map(c => ({ id: c.id, kind: c.kind, verdicts: c.verdicts })))}

任務：
1. 決定 champion（策略類）：Sharpe 路徑或風險路徑數值達標，且 lookahead / overfit（含安慰劑 ≥ P90）/ holdout 三道驗證都 pass 才能取代；否則 champion 填 ${baseline.id}，champion_path 填 none。若換冠軍，填 champion_cells（六格）與 champion_mean_sharpe。
2. guard_recommendation：若本輪有運行閘門方向，依其事前判準與四道驗證，給出「建議採用的參數組」或「不採用」與理由；引擎欄位名要對得上 RiskParams。
3. 排名所有有結果的變體，每個一句話總結它教了我們什麼（對照上一輪報告的假設，指出哪些被證實或推翻）。
4. 提出 2–4 個引擎/風控模型的改進提案（已在本輪引擎實作的東西不要再提）。
5. 提出下一輪 3–5 個設計方向，每個是 {key, prompt}，避開已證明無效的方向。
6. 把完整回合報告（繁體中文、含表格：變體 × 十格 Sharpe/回撤/閘門、相對落後、對冠軍 Δ 與 t、安慰劑百分位、敏感度、壓力、驗證結論、guard 結論、結論與建議）寫到 ${ROOT}/experiments/rounds/round_${round}.md（用 Bash mkdir -p 再寫檔），report_path 回傳該路徑。`,
  { label: 'synthesize', phase: 'Synthesize', schema: SYNTH_SCHEMA })

return {
  round,
  n_designed: all.length,
  n_ok: okResults.length,
  scored: scored.map(r => ({ id: r.id, kind: r.kind, gates10: r.score.gates10, meanSharpe: r.score.meanSharpe, avgDOOS: r.score.avgDOOS, minDOOS: r.score.minDOOS, maxAbsT: r.score.maxAbsT, minDDImprove: r.score.minDDImprove, minSens: r.score.minSens, sharpePath: r.score.sharpePath, riskPath: r.score.riskPath, file: r.file })),
  failed: all.filter(r => !r.ok).map(r => ({ id: r.id, error: r.error })),
  verified: verified.map(c => ({ id: c.id, kind: c.kind, verdicts: c.verdicts.map(v => ({ lens: v.lens, verdict: v.verdict, severity: v.severity, reasons: v.reasons })) })),
  synthesis: synth,
}
