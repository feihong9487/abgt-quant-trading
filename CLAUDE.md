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
| 運行閘門 | 實盤每日算策略淨值 vs 持有淨值，落後超過 X% 自動停止交易轉持有。最壞情況鎖在「持有 − X% − 手續費」 | 未做（要先有過上線閘門的策略） |

**季度閘門的結構性問題**（2026-09-23 發現）：在 ETH/BTC 單季漲 50–70% 的季度，任何不是 >100% 做多的策略都不可能「報酬更高且回撤更小」。所以「每季都過」對只做多趨勢策略等於不可能，見第 5 節四種曝險型變體最多也只過 6/15 季。要有意義的閘門是全期版本或風險調整版本（Sharpe / Calmar 高於持有）。

## 2. 架構

```
ABGT/
├── run_backtest.py          入口：--strategy bgt|apex [--mode kf|multi] [--require-kf-dir] [--offline] [--since] [--gate risk|absolute]
├── evaluate_variant.py      曝險型變體評估：--variant <name> [--symbols ...] [--sensitivity]；多區間（IS 2020–2022 含熊市 / OOS 2023– / FULL）× 風險調整閘門 × 參數 ±25% 敏感度
├── experiments/variants/    變體檔（PARAMS + build(df, params) → in_regime, RiskParams, warmup）
├── abgt/
│   ├── data.py              ccxt 抓 Binance USDT-M K 線，CSV 快取，之後只補新資料
│   ├── indicators.py        對應 Pine ta.*：ema/rma（SMA 起算）、atr、rsi、macd、dmi、supertrend_pine、pivot、HTF 對齊
│   ├── broker.py            Book：進出場、手續費、每筆出場紀錄、平倉→平倉回合
│   ├── metrics.py           淨報酬 / CAGR / 最大回撤 / Sharpe(日) / Calmar / 勝率 / 獲利因子
│   ├── benchmark.py         買入持有、季度分段表、閘門
│   └── strategies/
│       ├── bgt.py           BGT V5.1.3 信號 + 模擬器（process_orders_on_close=true 語意）
│       ├── apex.py          APEX v4.0 只做多 信號 + 模擬器（下一根開盤成交、雙腿 exit、追蹤止損、熔斷）
│       └── regime.py        曝險型引擎：in_regime 布林陣列 + RiskParams（曝險上限、波動目標、吊燈 ATR 停損、重進規則、回撤暫停、調倉門檻）
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
- [ ] 多代理設計 / 對抗驗證迴圈：探索趨勢過濾 × 曝險模型 × 停損 × 重進 × 回撤控制，只收「IS/OOS/FULL 都過 + 敏感度穩」的變體
- [ ] 把冠軍變體做成正式策略（`abgt/strategies/`）+ Pine 版（TradingView 目視）
- [ ] 運行閘門 + Binance 測試網下單（`live/`）

## 7. 與 Claude 協作

1. 回測只回報表格那幾個數字，明細在 `results/`。
2. 改策略邏輯先改 Pine 再同步 Python，兩邊 CLAUDE.md 一起更新。
3. 參數一次只動一個。
4. 不寫「保證獲利」，寫閘門有沒有過。
