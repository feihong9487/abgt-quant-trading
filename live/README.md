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

## 排程（Windows 工作排程器，每天台北 08:05 = 00:05 UTC）

```powershell
schtasks /create /tn "ABGT-paper" /sc daily /st 08:05 /tr "D:\feihong\tradingView\ABGT\.venv\Scripts\python.exe D:\feihong\tradingView\ABGT\live\paper.py run --verbose >> D:\feihong\tradingView\ABGT\live\state\run.log 2>&1"
```

移除：`schtasks /delete /tn "ABGT-paper" /f`。

## 狀態檔

`live/state/paper.json`：設定、現金、持倉、待成交目標、買入持有份額、每日紀錄（NAV、持有 NAV、比值、比值高水位回撤、90/180/365 根滾動比值、各標的 regime 與距 EMA50）、每筆成交。每次 `run` 原子寫入。
