# CLAUDE.md — ABGT / AAPEX 自動化交易框架

BGT（ETH 15 分）與 APEX（BTC 日線）的 Python 回測、買入持有基準、上線閘門，之後接實盤。本文件是專案的單一事實來源；每次回測後更新第 5 節。

---

## 0. 一句話定位

把 Pine 策略搬到 Python，用同一套引擎對「同期買入持有」做嚴格比較，只有通過閘門的策略才准接實盤。目的不是證明策略會贏，是**用數據決定放不放行**。

## 1. 關於「保證比純持有高」

做不到，數學上沒有任何策略能保證。本框架能給的是兩道閘門：

| 閘門 | 內容 | 狀態 |
|------|------|------|
| 上線閘門（風險調整，現行） | Sharpe > 持有 **且** Calmar > 持有 **且** 最大回撤 < 持有；每個評估區間（IS / OOS / FULL）各自判定。季度表只當參考 | `abgt/benchmark.py gate(mode="risk")`，`--gate risk` |
| 上線閘門（絕對，保留） | 全期淨報酬 > 持有且回撤更小且每季都過 | `--gate absolute`；對只做多不可達，見下 |
| 運行閘門 | 原設計：落後 X% 自動轉持有/空手。**第 4 輪否定 hold/flat 型自動閘門**：觸發時點是成本模型與起算日的函數（五個壓力情境五套觸發集合、成交價 σ=0.05% 雜訊 20 種子 13 種觸發集合）、累計落後 20% 在牛市是趨勢跟隨的正常區間（14 個季度上線日 BTC 9 / ETH 11 觸發）、hold 在頂部觸發等於關掉唯一的風控（BTC 2021-11→2022-05 −50% vs plain −30%）、恢復重設基期讓「鎖定」變成 0.8^k 向下棘輪。**現行：只記錄不動作** — 引擎 `lag_action="record"`、`lag_base_mode="hwm"`；紙上前向每日記錄比值高水位回撤與滾動比值，累積 ≥ 6 個月後以 cohort 分布再議 X，動作只能是通知 / 人工覆核 | `regime.py` 引擎 v4；`live/paper.py` |

**季度閘門的結構性問題**（2026-09-23 發現）：在 ETH/BTC 單季漲 50–70% 的季度，任何不是 >100% 做多的策略都不可能「報酬更高且回撤更小」。所以「每季都過」對只做多趨勢策略等於不可能，見第 5 節四種曝險型變體最多也只過 6/15 季。要有意義的閘門是全期版本或風險調整版本（Sharpe / Calmar 高於持有）。

## 2. 架構

```
ABGT/
├── run_backtest.py          入口：--strategy bgt|apex [--mode kf|multi] [--require-kf-dir] [--offline] [--since] [--gate risk|absolute]
├── evaluate_variant.py      曝險型變體評估：--variant <name> [--symbols ...] [--alt-split] [--sensitivity --sens-factors 0.5,0.75,1.25,1.5] [--baseline <id>] [--stress]
├── experiments/variants/    變體檔（PARAMS + build(df, params) → in_regime 或 {enter, exit}, RiskParams, warmup）
├── evaluate_portfolio.py    雙標的組合評估：--variant / --variants、--weights、對「持有不再平衡」與「持有月再平衡」兩種基準、--stress、組合層 --lag-stop
├── build_dashboard.py       產生 dashboard/index.html 與 status.json（即時狀態寫進 artifact 資料庫 status/current）
├── live/paper.py            現貨紙上前向：init / run / report / selftest；live/README.md 有事前登記與排程指令
├── pine/ema50_plain.pine    冠軍的 TradingView 版（目視 + 警報 + 相對持有落後面板）
├── experiments/workflows/   optimize_round.js：每輪的多代理工作流腳本（設計 → 實作評估 → 四道驗證 → 綜合）
├── experiments/rounds/      round_N.md 回合報告、loop_state.json 迴圈狀態（中斷後先讀這個）、rounds_meta.json
├── abgt/
│   ├── data.py              ccxt 抓 Binance USDT-M K 線，CSV 快取，之後只補新資料
│   ├── indicators.py        對應 Pine ta.*：ema/rma（SMA 起算）、atr、rsi、macd、dmi、supertrend_pine、pivot、HTF 對齊
│   ├── broker.py            Book：進出場、手續費、每筆出場紀錄、平倉→平倉回合
│   ├── metrics.py           淨報酬 / CAGR / 最大回撤 / Sharpe(日) / Calmar / 勝率 / 獲利因子
│   ├── benchmark.py         買入持有、季度分段表、閘門
│   └── strategies/
│       ├── bgt.py           BGT V5.1.3 信號 + 模擬器（process_orders_on_close=true 語意）
│       ├── apex.py          APEX v4.0 只做多 信號 + 模擬器（下一根開盤成交、雙腿 exit、追蹤止損、熔斷）
│   ├── portfolio.py         雙標的組合引擎：共用權益、每標的權重 × regime × scale、組合層 dd_pause / lag_stop、同權重持有基準（不再平衡 / 月再平衡）
│       ├── regime.py        曝險型引擎 v4：in_regime 或 {enter, exit, scale} + RiskParams（曝險上限、分級曝險 track/ratchet_up + floor、絕對/相對/下行波動目標、吊燈停損 low/close 觸發、重進規則、回撤暫停與恢復、運行閘門 hold/flat/record、資金費率、執行延遲、調倉門檻）
│       └── atr_channel.py   正式策略：EMA50 ± ATR 不對稱通道（第 1 輪最佳變體，BTC 回撤較小的替代品）
├── experiments/regime.py    對照：趨勢內 100% 持倉 / 趨勢外空手
├── tests/test_smoke.py      python tests/test_smoke.py（不需 pytest）
├── data/                    K 線快取（.gitignore）
└── results/                 每次回測的 equity / trades / segments / report（.gitignore）
```

環境：`.venv`（Python 3.14），`pip install -r requirements.txt`。Windows 終端要 `PYTHONIOENCODING=utf-8`。

### 執行

```bash
.venv/Scripts/python.exe run_backtest.py --strategy apex
.venv/Scripts/python.exe run_backtest.py --strategy bgt                      # BGT-S2 多信號
.venv/Scripts/python.exe run_backtest.py --strategy bgt --mode kf            # BGT-S1 只用 KF 主信號
.venv/Scripts/python.exe run_backtest.py --strategy bgt --require-kf-dir     # 聯動路徑也要 45 分方向一致
.venv/Scripts/python.exe experiments/regime.py                               # 曝險型對照
```

資料：ETH/USDT:USDT 15m 從 2022-12-02（暖機 30 天）、BTC/USDT:USDT 1d 從 2021-11-07（暖機 420 天，EMA200 用）。起算日固定 2023-01-01，買入持有也從同一根起算。

## 3. 與 Pine 的對應（parity 注意事項）

| 項目 | 做法 | 差異 |
|------|------|------|
| `request.security(..., lookahead_off)` | 每根 15m 拿「收盤時間 ≤ 本根收盤時間」的最後一根已完成 HTF K 線 | 與 TradingView 歷史 K 線語意一致；即時 K 線 TradingView 會用未完成值，Python 版不會 |
| 45m / 4H K 線 | 從 15m 用 `origin=epoch` 重採樣，邊界落 UTC 整點 | Binance 沒有原生 45m，TradingView 也是自己合成 |
| `ta.ema` / `ta.rma` | 前 n−1 根 NaN，第 n 根 SMA 起算 | 同 Pine |
| `ta.pivotlow/high` | 中心嚴格小於/大於左右各 n 根 | Pine 平手處理不確定，實價幾乎不會平手 |
| BGT 市價單 | 信號 K 收盤價 ± 2 tick | `process_orders_on_close=true` |
| BGT KC 止損 | 前一根掛、本根觸價成交，跳空用開盤價 | 同 Pine stop 單 |
| BGT 走人減倉 | 對每筆 entry 各減該筆剩餘量的 30% / 50% | `strategy.close(qty_percent)` |
| APEX 進場 | 下一根開盤價 + 2 tick | `process_orders_on_close=false` |
| APEX 兩腿 exit | K 線內路徑假設：開盤離高點近 → O→H→L→C，否則 O→L→H→C；停損/追蹤含滑點，限價不含 | TradingView 券商模擬器的假設 |
| APEX 追蹤止損 | 觸 TP1 啟動，stop = 最高點 − 0.35×止損距離；可在啟動同根被掃 | 近似 |
| 保證金 | 不模擬 | Pine v6 預設 margin 100% |
| 交易計數 | 「出場筆數」= TradingView closedtrades 口徑（分批各算一筆）；「進出回合」= 平倉到平倉 | 回合數才是「< 30 筆無意義」該看的 |

驗證過的：ADX 手算（Wilder）與 `ta.dmi` 末值一致到小數第二位；HTF 對齊無前視（`test_htf_merge_no_lookahead`）；帳本損益算術（`test_book_math`）。

**沒驗證過的**：跟 TradingView 策略測試器逐筆比對。要比對請把 TradingView 回測區間設 2023-01-01 起，看回合數和淨報酬的量級是否接近。

## 4. 策略參數

BGT：`abgt/strategies/bgt.py` `BGTParams`，與 `../BGT/CLAUDE.md` 的 V5.1.3 一致（Buffer 在價格側、`kfBufMult=0.5`、分層冷卻）。`entry_mode="kf"` = BGT-S1，`"multi"` = BGT-S2。`require_kf_dir_all=False` 是 Pine v2 的原樣（聯動路徑沒有 45 分方向閘）。

APEX：`abgt/strategies/apex.py` `APEXParams`，與 `../APEX/CLAUDE.md` §1.4 一致。只做多（`i_longOnly=true`），做空未移植。

## 5. 回測紀錄

區間 2023-01-01 → 2026-09-22/23，初始 10,000，手續費 0.05%/邊，滑點 2 tick。

| 策略 | 標的 | 淨報酬 | 最大回撤 | CAGR | Sharpe | 回合 | 勝率 | PF | 季度通過 | 閘門 |
|------|------|--------|---------|------|--------|------|------|----|---------|------|
| 買入持有 | BTC 1d | +418.7% | −53.0% | +55.6% | 1.18 | — | — | — | — | — |
| APEX v4.0 只做多 | BTC 1d | −0.7% | −5.4% | −0.2% | −0.07 | 11 | 36% | 0.91 | 6/15 | ✗ |
| 買入持有 | ETH 15m | +128.9% | −69.3% | +24.9% | 0.66 | — | — | — | — | — |
| BGT-S2 多信號 | ETH 15m | −25.4% | −34.8% | −7.6% | −0.65 | 993 | 33% | 0.86 | 6/15 | ✗ |
| BGT-S1 KF 單信號 | ETH 15m | +1.8% | −2.0% | +0.5% | 0.47 | 28 | 29% | 1.66 | 7/15 | ✗ |
| BGT-S2 + 45 分方向閘 | ETH 15m | −27.5% | −36.0% | −8.3% | −0.72 | 936 | 33% | 0.85 | 6/15 | ✗ |

曝險型對照（`experiments/regime.py`，BTC 1d 2023→，趨勢內 100% 持倉）：

| 變體 | 淨報酬 | 最大回撤 | CAGR | Sharpe | 季度通過 |
|------|--------|---------|------|--------|---------|
| R1 close > EMA200 | +193.1% | −34.5% | +33.5% | 0.95 | 3/15 |
| R2 三線多頭排列 | +164.6% | −37.1% | +29.9% | 0.96 | 4/15 |
| R3 三線排列 + ADX ≥ 28 | +103.4% | −21.0% | +21.0% | 0.95 | 6/15 |
| R4 close > EMA50 | +394.4% | −30.0% | +53.6% | **1.41** | 5/15 |

### 曝險型基準變體（`evaluate_variant.py`，風險調整閘門，2026-09-23）

區間：IS 2020-01→2022-12（牛轉熊）、OOS 2023-01→今、FULL。✓ = 該區間過閘門。

| 變體 | 標的 | IS 淨報酬/回撤/Sharpe | OOS | FULL | 敏感度 OOS 通過率 |
|------|------|----------------------|-----|------|------------------|
| 持有 | BTC | +130% / −77% / 0.76 | +419% / −53% / 1.18 | +1097% / −77% / 0.92 | — |
| **ema50_plain** | BTC | +211% / −56% / 1.01 ✓ | +387% / −30% / 1.40 ✓ | +1414% / −56% / 1.17 ✓ | 100% |
| ema50_atr（3×ATR 吊燈，停損後等重置） | BTC | +5% / −53% / 0.23 ✗ | +372% / −29% / 1.52 ✓ | +395% / −53% / 0.89 ✗ | 100% |
| turtle_55_20（+2×ATR） | BTC | +49% / −21% / 0.69 ✗ | +123% / −18% / 1.14 ✗ | +232% / −21% / 0.92 ✓ | 33% |
| ema50_voltarget（波動目標 60% + 停損 + 回撤暫停 25%） | BTC | +19% / −50% / 0.34 ✗ | +363% / −28% / 1.52 ✓ | +452% / −50% / 0.96 ✗ | 100%（引擎回撤暫停高水位 bug 修正後） |
| 持有 | ETH | +618% / −79% / 1.18 | +129% / −68% / 0.67 | +1552% / −79% / 0.93 | — |
| **ema50_plain** | ETH | +570% / −61% / 1.26 ✓ | +122% / −51% / 0.71 ✓ | +1387% / −61% / 0.99 ✓ | 100% |
| ema50_atr | ETH | +142% / −55% / 0.85 ✗ | +75% / −50% / 0.59 ✗ | +324% / −55% / 0.72 ✗ | 67% |
| turtle_55_20 | ETH | +284% / −15% / 1.53 ✓ | +72% / −22% / 0.75 ✓ | +560% / −22% / 1.15 ✓ | 67% |

讀法：最簡單的 `close > EMA50` 在兩個標的三個區間全過，是目前的冠軍基準。吊燈停損 + 「停損後等趨勢重置」反而傷（高波動牛市被掃出後錯過續漲）。海龜回撤最小（−21%）但曝險只 17%，報酬低。

### 優化迴圈 第 1 輪（2026-09-23，`experiments/rounds/round_1.md`）

29 個代理：4 視角 × 3 變體 → 12 個全部實作評估 → 前 4 名各三道驗證 → 綜合。十格 = BTC/ETH × IS/OOS/FULL/IS2(2020-07→2023-12)/OOS2(2024-01→)。

| 變體 | 六格閘門 | 平均 Sharpe | 一句話 |
|------|---------|------------|--------|
| ema50_plain（基準） | 6/6 | 1.09 | 十格全過；SOL 5/5、BNB 3/5 |
| vol_atr_channel：EMA50 + 1.0 ATR 進 / − 0.25 ATR 出 | 6/6 | 1.19 | 回撤六格全淺 6–15 pp、換手 1/2.5、三道驗證 pass；**但 holdout SOL 3/5、BNB 0/5，對基準 ΔSharpe SOL +0.13 / BNB −0.13**；資金費率 0.01%/8h 或延遲一根時 BTC OOS 掉到閘門下 |
| vol_downside_gate：EMA50 + 下行/上行半變異比值崩盤閘 | 6/6 | 1.18 | 改善全在 ETH，BTC 三區間全劣；ETH 優勢集中在前 10 個單日，延遲一根就消失 |
| vol_shock_budget：EMA50 + 絕對波動預算 0.70 | 6/6 | 1.15 | 0.70 是 IS 校準，BTC 2025 後死碼；替換切分 ETH OOS 失敗 |
| ensemble_ema_ladder_vote | 6/6 | 1.11 | 等價於「close > 第 K 低的 EMA」，回撤更深，對基準 t < 0.5 |
| exit_ema50_stall / exit_ema50_band | 6/6、5/6 | 1.06、1.20 | 時間停滯標記的是盤整低點不是頂；對稱帶 BTC OOS 差 0.03 |
| classic_*（Faber 10 月、黃金交叉、TSMOM）、ensemble_donchian/family、chandelier_immediate | ≤ 4/6 | — | 慢訊號在 2023 後 V 型市場付的是轉折延遲；low 觸價停損 + 立即重進 = 付費買下影線保險 |

**決定**：第 1 輪報告把 `vol_atr_channel` 定為冠軍，但事後補的 holdout（SOL/BNB，round 1 沒有這道檢查）顯示它在沒碰過的標的上輸給純 EMA50 — 驗證者擔心的「同一個 OOS 已被 16 個變體選擇過」成立。**基準維持 `ema50_plain`**；通道版正式化為 `abgt/strategies/atr_channel.py`（`run_backtest.py --strategy channel`），定位是「BTC 回撤較小、換手較低的替代品」，不是升級。第 1 輪計為無新冠軍。

第 1 輪促成的引擎升級（`regime.py` v2，回歸測試與第 1 輪數字逐位一致）：enter/exit 狀態機由引擎維護；停損可收盤確認（`stop_trigger="close"`）；重進規則加 `close_above_stop` 與 `reentry_min_bars`；波動目標加尺度無關的 `relative` 模式與只算下行的 `downside`；回撤暫停加 `new_high` / `bars` 恢復；資金費率與執行延遲可模擬。評估器加替換切分、`--sens-factors`、對基準逐格 Δ 與配對 t、四組壓力情境、敏感度去重（映射到相同 in_regime 的擾動不計）。

### 優化迴圈 第 2 輪（2026-09-23，`experiments/rounds/round_2.md`）

24 個代理：5 方向 × 2 變體 → 10 個全部評估（十格 + 敏感度 + 壓力 + 對基準 Δ）→ 前 2 名四道驗證 → 綜合。**無新冠軍（連續 2 輪）**。

| 變體 | 十格 | 六格平均 Sharpe | 對基準 OOS 最低 Δ | 一句話 |
|------|------|----------------|------------------|--------|
| fastreentry_channel_or_20d_high | 10/10 | 1.155 | −0.098 | 快速腿（close > EMA50 且 20 日新高）是「突破次日押續漲」的對稱賭注：提早進場段由構造必勝、通道不會進的多餘回合 BTC 0/12 勝；overfit / executable 判 fail |
| fastreentry_channel_or_3bar_confirm | 10/10 | 1.134 | −0.093 | 同上，連續 3 日確認版 |
| relvol_ema50_downside / _total | 10/10 | 1.089 / 1.086 | −0.013 / +0.008 | 尺度無關波動預算「單獨不傷」但也沒加分，downside 版 BTC t ≈ −2.25（本輪唯一顯著、方向為負） |
| closestop_ema50_c3atr_hold7 | 9/10 | 1.094 | −0.157 | 收盤確認停損 + 重進門檻：停損截到的是回檔不是崩盤，仍是低賣高買 |
| crashgate_ret5_atr_abs / _dvol_ratio_abs | 9/10 | 1.02 | −0.09 / −0.03 | 加了絕對門檻的崩盤閘在 BTC 是死碼、在 ETH 回檔低賣 |
| crossasset_ema50_majority_vote / _run_hysteresis | 8/10 | 1.12 / 1.07 | −0.14 / −0.18 | 時間域濾波（5 日多數決、連續根遲滯）在出場側吃掉整根崩盤日 |
| closestop_channel_c3atr_hold7 | 7/10 | 1.137 | −0.271 | 通道 + 停損：回撤優勢被還回去 |

結論：第 1 輪路線圖的四個方向全部否定，共同原因是日線 EMA50 本身已是約一週尺度的崩盤偵測器 — 疊加保護型規則是死碼或回檔低賣高買，放鬆型規則還回通道的回撤與換手優勢，時間域濾波在出場側付整根崩盤日。壓力測試用歷史資金費率重跑基準：BTC IS 掉閘門、ETH 五格全掛 → 永續執行不可行，只能現貨。

第 2 輪促成的引擎 v3：修正「收盤觸發停損 × 延遲成交」時出場排隊中會再排進場的 bug（重進規則改從成交根起算、`stop_events` 逐次紀錄）；`in_regime` dict 可帶 `scale` 陣列做分級曝險；`extra` 加 `exposure_key` / `fill_key` / `n_fills`，敏感度去重改看「regime 與曝險都相同」；`RiskParams.funding` 可放逐日歷史費率陣列，`data.fetch_funding()` 抓 Binance 費率歷史，壓力測試多一組「歷史資金費率」。

### 優化迴圈 第 3 輪（2026-09-23，`experiments/rounds/round_3.md`）— 停止規則觸發

28 個代理：5 方向 × 2 變體 → 10 個全部評估 → 3 個候選四道驗證 → 綜合。**無新冠軍（連續 3 輪），迴圈依規則停止。**

| 變體 | 十格 | 六格平均 | OOS 最低 Δ | 驗證 | 一句話 |
|------|------|---------|-----------|------|--------|
| volume_confirmed_entry_q1_ema_exit | 10/10 | 1.19 | −0.023 | lookahead ✓ overfit ✗ executable ✗ holdout ✓ | 三輪最接近的候選；增量全在 IS 鞭打段，四 OOS |t| < 0.4；安慰劑：25% 隨機濾網同樣達標；換現貨量 7/80 個決策就翻 |
| btc_leader_gate_ema50_and | 10/10 | 1.17 | −0.022 | ✓ ✗ ✗ ✓ | 三輪首個同時改善 SOL/BNB、ETH 回撤六格全淺；但 ETH 增量來自 IS 三個崩盤日，OOS 四種機制互抵 |
| volume_confirmed_entry_q1_band_exit | 10/10 | 1.17 | −0.026 | ✓ ✗ ✗ ✓(踩線) | fastreentry 的鏡像，多餘回合 0/21 勝 |
| slow_vol_scaled_band_atr50 / _median365 | 10/10 | 1.22 / 1.18 | −0.057 / −0.154 | holdout ✗ | 帶寬換慢尺度：ATR50 幾乎無操作；「V 型提早 2–5 根」假設不成立；證明通道 2023 後落後的是 +1 ATR 門檻本身 |
| context_adaptive_entry_exit_cooldown / _whipsaw_count | 10/10、9/10 | 1.15 / 1.04 | −0.073 / −0.176 | — | 出場後 5 根內的再穿越 100% 負期望（BTC IS t 2.31），但 2023 後大回合都在假出場後 1–3 根 |
| graded_exposure_dist_ratchet / _ema_ladder_mean | 10/10 | 1.12 / 1.14 | −0.065 / −0.07 | — | 鋸齒成本減半成立，但半倉日 IS +0.15 / OOS −0.18 正負翻轉；**消融 A3（分級 + 棘輪 + EMA50 出場）數值全達門檻、holdout SOL 5/5 BNB 3/5，未走驗證，是最值得續做的線索** |
| btc_leader_gate_relstrength_ratio | 7/10 | 0.92 | — | — | 相對強度閘賣出仍在漲的相對弱勢幣；關閉 |

三輪總結（32 個變體、81 個代理、約 2.2 小時工作流）：
1. **`ema50_plain` 仍是冠軍。** 沒有任何疊加規則在 OOS 對它有統計上可分辨的改善；改善只出現在 2020–2022 的熊市反彈鞭打段，而那正是任何「讓進場變慢」的規則都能賺到的。
2. **現行取代門檻沒有鑑別力**：第 3 輪安慰劑檢定顯示 25% 的隨機進場濾網也滿足「十格全過 + 六格平均 > 冠軍 + 四 OOS Δ ≥ −0.05」。若重開迴圈，門檻改為「四個 OOS 格對冠軍 ΔSharpe 平均 ≥ +0.05、至少一格 |t| ≥ 1.5、holdout 平均 Δ ≥ 0」。
3. **有內容的發現**：不對稱遲滯（進場確認、出場不拖延）降回撤與換手（通道）；BTC 趨勢狀態對 ETH/SOL/BNB 的回撤有真實避險價值（btc_leader，尾部事件）；分級曝險把鋸齒成本減半（A3 線索）。這三條可以組合成下一階段的候選，但要走完整驗證。
4. **永續執行不可行**（歷史資金費率下基準 ETH 全掛、BTC IS 掉閘）→ 現貨。

### 優化迴圈 第 4 輪（2026-09-23，`experiments/rounds/round_4.md`）— 使用者指示重開：a 分級曝險 + b 運行閘門

19 個代理：2 方向 × 2 變體 → 4 個全部評估 → 3 個四道驗證 → 綜合。**無新冠軍；graded_exposure 與 hold 型運行閘門兩個方向都依事前判準關閉。**

| 變體 | 類型 | 十格 | 六格平均 | 四 OOS Δ 平均 | 驗證 | 一句話 |
|------|------|------|---------|--------------|------|--------|
| graded_exposure_ema_exit_ratchet_a3 | strategy | 10/10 | 1.148 | +0.012 | 安慰劑 fail | 第 3 輪 A3 數字逐位重現、holdout 兩標的不劣，但把「哪個回合拿到哪條升階路徑」隨機置換 150 次它只在第 83 百分位，10.7% 隨機路徑滿足全部門檻；bars_to_full 中位數 2 根 = 分級實際是「進場後 1–2 根要嘛滿倉要嘛整回合半倉」 |
| graded_exposure_ema_exit_clip_d05 | strategy | 9/10 | 1.109 | +0.071（t 1.85） | lookahead ✓ overfit ✗ executable ✗ holdout ✗ | BTC OOS +0.135/+0.176 是四輪最強的 BTC 證據，但假設來自同一段 2024–26 資料、ETH 機制反向（半倉根次日反而漲）、ETH IS2 掛、BNB 五格全負；62% 的減半只持續 1 根 = EMA50+0.5 ATR 軟出場帶的變形 |
| running_gate_lag_stop_cum20_hold60 | guard | 4/10 | 0.95 | — | ✓ ✗ ✗ ✗ | 十格翻掛 6 格，最深相對落後 −61.6% → −76.3%；11 筆觸發 6 筆在翻多日（策略空手、進場已排隊）→ 等於「重進後 60 根禁止出場」 |
| running_gate_lag_stop_roll180_20_hold60 | guard | 5/10 | 0.84 | — | ✓ ✗ ✗ ✗ | 頂部觸發抱住整段第一波熊市；觸發集合隨成本模型 / 起算日 / 成交價雜訊變動；安慰劑（同長度隨機放置的 hold）在兩標的都優於實際時點 |

### c：BTC/ETH 50/50 組合（`abgt/portfolio.py`、`evaluate_portfolio.py`，2026-09-23）

| 區間 | 組合 Sharpe | 單標的平均 | 50/50 持有不再平衡 | 50/50 持有月再平衡 | 組合回撤 / 持有回撤 |
|------|------------|-----------|------------------|------------------|-------------------|
| IS | 1.24 | 1.14 | 1.04 | 1.02 | −59% / −78% |
| OOS | 1.13 | 1.06 | 0.79 | 0.93 | −39% / −62% |
| FULL | 1.17 | 1.08 | 0.90 | 0.96 | −59% / −78% |
| IS2 | 1.39 | 1.27 | 1.15 | 1.19 | |
| OOS2 | 0.77 | 0.71 | 0.49 | 0.58 | |

十格對兩種基準全過；組合 Sharpe 五個區間都高於單標的平均（分散紅利 +0.06～+0.12），ETH 的薄邊距被 BTC 吸收。壓力：費 / 滑 / 常數資金費率 / 延遲全過，歷史資金費率翻 IS、IS2 → 現貨。通道版組合 FULL 1.29、回撤 −46%、換手 9.9×/年（ema50_plain 24.3×）。組合層 lag_stop 0.20 hold 的探索是反例（2021-05、2022-02 觸發抱住熊市，回撤 −80%）。

### 現貨資料來源檢查（2026-09-23）

用 Binance 現貨日線重跑 ema50_plain：BTC 十格全過，Sharpe 1.03 / 1.39 / 1.18 / 1.31 / 0.97（永續 1.01 / 1.40 / 1.17 / 1.32 / 0.97）；ETH 十格全過，1.32 / 0.76 / 1.05 / 1.25 / 0.51（永續略低）。策略搬到現貨成立。現貨真實成本（費 0.10% + 滑 0.20%）下 ETH 五格全掛、BTC 全過 — ETH 對持有的邊距薄，是紙上前向要盯的點。

### 現貨紙上前向（`live/paper.py`，上線日 2026-10-01）

ema50_plain、BTC/USDT 與 ETH/USDT 現貨各 50%、費 0.10%/邊、滑 0.05%、00:00 UTC 收盤決策次日開盤成交；帳本用 2026-06-01→09-22 自我驗證與 `simulate_portfolio` 逐位一致（114 日、14 筆成交、NAV 差 0.00000%）。運行指標只記錄。事前登記的評估視窗、可證偽預測與停止條件在 `live/README.md`。

四輪總結（36 個變體、100 個代理）：**`ema50_plain` 是冠軍，且沒有任何疊加規則在 OOS 對它有可分辨的改善。** 兩條有內容的線索（不對稱遲滯降回撤換手、BTC 領先閘的尾部避險）都不足以取代。優化到此為止，資源轉到前向。

取代冠軍的門檻（第 2 輪起）：十格 ≥ 9 且四個 OOS 格全過、六格平均 Sharpe 高於冠軍、四個 OOS 格對冠軍 ΔSharpe ≥ −0.05、OOS 敏感度通過率 ≥ 0.67、前視 / 過擬合（超額 Sharpe 口徑）/ holdout（SOL+BNB 十格 ≥ 5 過且平均 Δ ≥ −0.05）三道驗證全過。

### 四個資本槓桿與槓桿系統（2026-09-23/24，`experiments/rounds/levers.md`、`results/levers/`）

引擎：借款利息 `borrow_rate_annual`、強平模擬 `liq_level`/`liq_penalty`（Binance 現貨全倉 1.1）、回撤降槓桿 `dd_lever_full`、組合層 `vol_target`（原本被忽略）、`redeploy_idle`。
修正：組合引擎先賣後買、1× 買量以現金為上限（之前會透支到 1.07×）；`live/paper.py` 同步，自我驗證 2,091 日 / 295 筆逐位一致。
**基準修正**：`evaluate_portfolio` 的持有改成「視窗起點重新等權買入」（`fresh_benches`）；舊版是起算日買入後漂移（2023 起 ETH 佔七成以上），報酬對比偏樂觀。

| 區間（現貨、費 0.10% 滑 0.05%） | 組合 淨報酬 / 回撤 / Sharpe | 持有不再平衡（起點重設） | 持有月再平衡 |
|---|---|---|---|
| IS | +437% / −59% / 1.30 | +472% / −78% / 1.12 | +435% / −76% / 1.10 |
| OOS | +231% / −38% / 1.10 | +274% / −58% / 0.96 | +267% / −60% / 0.93 |
| FULL | +1675% / −60% / 1.18 | +1552% / −78% / 0.95 | +1870% / −76% / 1.00 |
| OOS2 | +70% / −38% / 0.74 | +56% / −59% / 0.57 | +60% / −60% / 0.58 |

十格仍全過。**風險調整贏持有；絕對報酬 OOS 輸持有**（靠躲熊市贏，沒有深熊的牛市會落後）。

- 分散：4 標的等權 OOS Sharpe 1.33 vs BTC 單獨 1.34（t 0.19）— 分散不加 Sharpe，換的是逐年穩定；IS 的增量含 SOL/BNB 存活者偏誤。
- 成本：每邊 0.15% → 可達 ≈ 0.106%（BNB 抵扣 + 先掛限價 60 分轉市價；BTC 賣單 60 分成交僅 84%）組合 OOS Sharpe +0.03、CAGR ≈ +1.6 pp；現貨 maker 0.02% 拿不到；ETH 損益兩平只有 0.21%/邊。
- 本金：線性無上限；每月 +1,000 的損益增量是 EMA ±25% 參數範圍的 6.7–15.5 倍；回撤加碼無效（回撤時多半空手）。
- 槓桿（`experiments/levers/leverage_policy.py`，104 組事前登記網格 × 兩組切分）：OOS Sharpe 全網格 1.03–1.18（1× 1.10）→ 槓桿只是同一個賭注押大；「同風險層」兩切分驗證都失敗；「持有風險層」兩切分都選 `max_exposure 2.0 / vol_target 0.9 / dd_lever_full 0.15`，OOS CAGR 55.9% vs 37.9%、回撤 −54.5% vs −37.6%、Sharpe 相同、零強平，但鄰居只有 2/5 可行。固定 1.5× 延遲一根十格掉到 3/10。redeploy_idle 全面變差。
- **政策：現在 1×、不借錢、不掛波動目標**；重開條件見 levers.md §5.3（前向 ≥ 12 個月、實際成本對得上、以當日借款利率重跑、延遲一根 ≥ 8/10、四道驗證）。

### 測試網執行器（`live/executor.py`，2026-09-24）

只連 `testnet.binance.vision`（URL 硬檢查、無正式網下單路徑）；決策用正式網公開日線、成交在測試網；子帳記帳、冪等 clientOrderId、只在交易所明確回「找不到」時才重下、先賣後買、`--maker-first N`、`live/KILL` 停單 / `--flatten`、對帳。沒有金鑰自動 dry-run。2026-10-01 上線（`.env` 解析容忍全形「＝」）；帳本 `live/state/executor.json`。

### 結論（2026-09-23）

1. **APEX 輸給持有不是 bug，是設計**：每筆風險 1%、3.7 年 11 回合，全贏也只有 +27%。ATR 風險式倉位在 5 倍牛市裡結構上追不上持有；它的價值在回撤（−5.4% vs −53%），不在報酬。
2. **BGT-S2 的 −25% 幾乎全是手續費**：毛損益 −88，手續費 −2,450（993 回合、每回合中位數只持 37 根 ≈ 9 小時）。按出場原因拆：三層走人合計 +8,478（有效），趨勢切換平倉 1,105 次合計 −10,489（真正的止損，也是出血點），KC 止損只觸發 62 次（−527，如預期幾乎是死碼）。加 45 分方向閘沒有改善（−27.5%），因為聯動信號本來就大多順著 45 分方向，問題不在方向，在 45 分 Supertrend 每 19 小時翻一次、每翻一次就平倉。KF 主信號 3.7 年只 42 次（S1 版 +1.8%、PF 1.66、回撤 −2%）— 有正期望但曝險太低。
3. **季度閘門「報酬更高且回撤更小」對只做多策略不可達**：四個曝險型變體最多 6/15。R4（close > EMA50）全期 +394% / −30% / Sharpe 1.41 是最接近的，報酬略輸持有但回撤少一半、風險調整後贏。
4. 要「贏過持有」只有兩條路：曝險型趨勢跟隨（趨勢內滿倉，靠躲過熊市回撤贏），或改用風險調整後的閘門。1% 風險式的 APEX 和 15 分短線的 BGT 都不是這條路上的東西。

## 6. 路線圖

- [x] 框架：資料 / 指標 / 帳本 / 模擬 / 基準 / 閘門 / 報告
- [x] BGT-S2、APEX 首輪回測
- [x] BGT-S1、BGT-S2+方向閘 回測
- [ ] 跟 TradingView 策略測試器比對回合數與淨報酬量級
- [x] 閘門改風險調整（使用者決定，2026-09-23）
- [x] 曝險型引擎 + 變體評估器 + 四個基準變體
- [x] 多代理設計 / 對抗驗證迴圈 第 1 輪（`experiments/workflows/optimize_round.js`，狀態在 `experiments/rounds/loop_state.json`）
- [x] 第 2 輪：崩盤閘、收盤確認停損、快速重進、相對下行波動預算、跨標的時間濾波 — 全部否定
- [x] 第 3 輪：情境自適應、BTC 領先閘、慢波動帶寬、量能確認、分級曝險 — 無新冠軍，迴圈停止（`loop_state.json` status = stopped）
- [x] 第 4 輪（使用者指示重開）：A3 分級曝險正式化 + 2×2 缺格、hold 型運行閘門 — 兩方向關閉；引擎 v4（lag_stop 記錄型、ratchet_up、帳目輸出）
- [x] c：BTC/ETH 50/50 組合引擎與評估（十格全過、Sharpe 高於單標的平均）
- [x] 通道版正式化 `abgt/strategies/atr_channel.py`；ema50_plain Pine 版 `pine/ema50_plain.pine`
- [x] 現貨資料來源檢查；`live/paper.py` 紙上前向（上線 2026-10-01）；Windows 排程任務 `ABGT-paper` 每天 08:05 台北執行 `live/run_daily.bat`（已建立並測試，結果碼 0）
- [x] ETH 決定：留在 50/50 組合跑前向，6 個月檢查點見 `live/README.md`
- [ ] 前向累積 ≥ 6 個月後：guard_cohort_calibration（純量測）決定 X 是否存在；session_close_robustness（1h 合成不同收盤時點）
- [x] 第 5 輪：使用者決定不開（2026-09-23）
- [x] 資金費率歷史資料接進 data.py；結論：永續執行成本讓基準 ETH 全掛、BTC IS 掉閘門，上線走現貨
- [x] 四個資本槓桿分析 + 槓桿系統（強平、回撤降槓桿、組合 vol_target）+ 事前登記槓桿網格 → 現在 1×
- [x] 持有基準改視窗起點重設；組合引擎現貨不透支
- [x] Binance 現貨測試網執行器 `live/executor.py`（dry-run 已驗證）
- [x] 測試網執行器上線（2026-10-01）：子帳 9,900 USDT、首筆 BTC LIMIT_MAKER 成交 / ETH 30 秒未成交轉市價、對帳 OK、重跑冪等；加入 `run_daily.bat`（ABGT-paper 排程每天 08:05 台北）。測試網手續費為 0，真實成本要在正式網量
- [ ] 4 × 25% 記錄型分帳

## 7. 與 Claude 協作

1. 回測只回報表格那幾個數字，明細在 `results/`。
2. 改策略邏輯先改 Pine 再同步 Python，兩邊 CLAUDE.md 一起更新。
3. 參數一次只動一個。
4. 不寫「保證獲利」，寫閘門有沒有過。
