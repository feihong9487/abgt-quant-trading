@echo off
rem ABGT daily job: called by Task Scheduler at 08:05 Taipei (= 00:05 UTC)
cd /d D:\feihong\tradingView\ABGT
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
echo ===== %date% %time% ===== >> live\state\run.log
rem 1) spot paper-forward ledger
.venv\Scripts\python.exe live\paper.py run --verbose >> live\state\run.log 2>&1
rem 2) testnet executor (Binance spot TESTNET only; maker-first 60s then market)
echo ----- executor ----- >> live\state\run.log
.venv\Scripts\python.exe live\executor.py run --maker-first 60 >> live\state\run.log 2>&1
rem commit the ledgers when they changed (no push)
git add live\state\paper.json live\state\executor.json >nul 2>&1
git diff --cached --quiet || git commit -q -m "paper/testnet: daily ledger %date%" >> live\state\run.log 2>&1
