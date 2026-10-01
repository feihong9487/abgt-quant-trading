# live/ — 現貨紙上前向測試

策略：`ema50_plain`（收盤 > EMA50 → 次日開盤買到目標曝險；收盤 < EMA50 → 次日開盤全賣），BTC/USDT 與 ETH/USDT 現貨各 50% 名目、共用權益。
不掛任何倉位覆蓋層、不掛自動動作的運行閘門（第 4 輪結論）；運行指標只記錄。

## 事前登記（2026-09-23）

| 項目 | 值 |
|------|----|
| 上線日 | 2026-10-01（UTC） |
| 資金 | 10,000 USDT 紙上 |
| 手續費 | 0.10%/邊（Binance 現貨 taker；BNB 抵扣 0.075% 另列） |
| 滑點 | 0.05% |
| 決策時點 | 00:00 UTC 日線收盤後（台北 08:00）；成交 = 次日開盤價 |
| 評估視窗 | 12 個月，或每標的 ≥ 20 回合 |
| 可證偽預測 | 前向 12 個月 regime 翻轉次數落在回測 2023→ 逐年分布 p10–p90；策略對持有比值的高水位回撤不深於回測 OOS 的 −42%（BTC）/ −50%（ETH）超過 10 pp |
| 停止條件 | 兩標的同時連續 3 個月比值創新低且策略絕對回撤 > 回測 OOS 最深回撤 + 10 pp → 人工覆核（不自動動作） |

## 指令

```bash
cd D:/feihong/tradingView/ABGT
.venv/Scripts/python.exe live/paper.py init --start 2026-10-01      # 一次
.venv/Scripts/python.exe live/paper.py run --verbose                 # 每天 00:05 UTC 後
.venv/Scripts/python.exe live/paper.py report
.venv/Scripts/python.exe live/paper.py selftest --start 2026-06-01   # 用過去區間對回測引擎驗證帳本
```

`run` 是冪等的：同一天重跑不會重複記帳；漏跑幾天再跑會依序補上（用當時的開盤價成交，與回測語意相同）。

## 排程（已建立，2026-09-23）

Windows 工作排程器任務 `ABGT-paper`：每天台北 08:05（= 00:05 UTC）執行 `live/run_daily.bat`（跑 `paper.py run --verbose`、輸出到 `live/state/run.log`、帳本有變動就 commit）。機器關機或睡眠時不會跑，但 `run` 會在下次執行時依序補上漏掉的日子（用當時的開盤價成交，與回測語意相同）。

```powershell
schtasks /query /tn "ABGT-paper" /fo list     # 看狀態與下次執行時間
schtasks /run /tn "ABGT-paper"                # 手動跑一次
schtasks /delete /tn "ABGT-paper" /f          # 移除
```

## ETH 的決定（2026-09-23）

現貨真實成本下 ETH 單獨對持有的邊距只剩 0.05 Sharpe，但 BTC/ETH 50/50 組合十格全過、組合 Sharpe 五個區間都高於單標的平均（分散紅利 +0.06～+0.12）。**前向照組合口徑跑，ETH 留著** — 前向測試的目的就是量這條薄邊距。事前登記的 6 個月檢查點：
- 用現貨日線對前向區間跑 `evaluate_variant.py --variant ema50_plain --symbols ETH/USDT`，若 ETH 單獨對持有的 Sharpe Δ < −0.10 且 BTC 那半邊 ≥ 0 → 下一期改 BTC 100%；
- 若 ETH 那半邊只是「與持有持平」而組合 Sharpe 仍高於 BTC 單獨 → 留。
判定只在檢查點做，期中不因單月表現調整。

## 4×25% 記錄型分帳（事前登記 2026-10-01）

`live/state/paper_4x25.json`：BTC/ETH/SOL/BNB 現貨各 25%，同一策略、同上線日 2026-10-01、同成本（費 0.10% 滑 0.05%）、同調倉門檻（0.15 × 權重）。
**只記錄**：不取代、不影響上面登記的 BTC/ETH 50/50 帳本，期中不因表現調整任何一本。
自我驗證：2021-01-01→2026-09-30 共 2,091 日、662 筆成交，帳本 NAV 對 `simulate_portfolio` 最大差 0.00000%。

來源是 `experiments/rounds/levers.md` §1：分散在 OOS 對 BTC 單獨沒有 Sharpe 增量，換的是逐年穩定；IS 的增量含 SOL/BNB 存活者偏誤（用 2026 市值回頭挑），所以要前向量。

| 項目 | 值 |
|------|----|
| 對照 | 50/50 帳本、BTC 單獨（以同期現貨日線回放 `simulate_regime`） |
| 可證偽預測（6 個月檢查點 2027-04-01） | ① 4×25% 對 50/50 的逐日報酬差配對 t 的 \|t\| < 2（分散不加也不減 Sharpe）；② 4×25% 的最大回撤不比 50/50 深超過 5 pp；③ 名目換手 25 ± 8 ×/年（回測 25.2） |
| 判定 | ①②③ 全成立 → 「分散換穩定、不換報酬」得到前向支持，下一期可討論改 4×25%；① 不成立且 4×25% 較差 → 結論是分散在前向有成本、關帳；其餘 → 再跑 6 個月 |
| 不做的事 | 不依 6 個月內的數字換權重或換標的；不加入第 5 個標的 |

## 狀態檔

`live/state/paper_4x25.json`：同下，記錄型分帳。

`live/state/paper.json`：設定、現金、持倉、待成交目標、買入持有份額、每日紀錄（NAV、持有 NAV、比值、比值高水位回撤、90/180/365 根滾動比值、各標的 regime 與距 EMA50）、每筆成交。每次 `run` 原子寫入。
