@echo off
rem ABGT daily job: called by Task Scheduler at 08:05 Taipei (= 00:05 UTC)
cd /d D:\feihong\tradingView\ABGT
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
echo ===== %date% %time% ===== >> live\state\run.log
rem 1) spot paper-forward ledger
.venv\Scripts\python.exe live\paper.py run --verbose >> live\state\run.log 2>&1
rem 1b) recording-only paper ledger: BTC/ETH/SOL/BNB 25% each
.venv\Scripts\python.exe live\paper.py run --state live\state\paper_4x25.json --verbose >> live\state\run.log 2>&1
rem 2) testnet executor (Binance spot TESTNET only; maker-first 60s then market)
echo ----- executor ----- >> live\state\run.log
.venv\Scripts\python.exe live\executor.py run --maker-first 60 >> live\state\run.log 2>&1
rem 3) rebuild local dashboard (dashboard/index.html + live.json) from the fresh ledgers
.venv\Scripts\python.exe build_dashboard.py >> live\state\run.log 2>&1
rem 4) copy live.json into the Google Drive sync folder (claude.ai dashboard reads it via the Drive connector)
.venv\Scripts\python.exe live\drive_sync.py >> live\state\run.log 2>&1
rem commit the ledgers when they changed (no push)
git add live\state\paper.json live\state\paper_4x25.json live\state\executor.json dashboard\live.json dashboard\status.json dashboard\index.html >nul 2>&1
git diff --cached --quiet || git commit -q -m "paper/testnet: daily ledger %date%" >> live\state\run.log 2>&1
