@echo off
rem ABGT daily job: called by Task Scheduler at 08:05 Taipei (= 00:05 UTC)
rem Every step runs even if an earlier one fails; failures go to live\state\failed_steps.txt and notify.py reports them.
cd /d D:\feihong\tradingView\ABGT
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set GCM_INTERACTIVE=never
set GIT_TERMINAL_PROMPT=0
set PY=.venv\Scripts\python.exe
set LOG=live\state\run.log
set FAIL=live\state\failed_steps.txt
echo ===== %date% %time% ===== >> %LOG%
rem 1) spot paper-forward ledger (registered 50/50)
%PY% live\paper.py run --verbose >> %LOG% 2>&1
if errorlevel 1 echo paper>> %FAIL%
rem 1b) recording-only paper ledger: BTC/ETH/SOL/BNB 25% each
%PY% live\paper.py run --state live\state\paper_4x25.json --verbose >> %LOG% 2>&1
if errorlevel 1 echo paper_4x25>> %FAIL%
rem 2) testnet executor (Binance spot TESTNET only; maker-first 60s then market; auto-rebuilds sleeve after a testnet reset)
echo ----- executor ----- >> %LOG%
%PY% live\executor.py run --maker-first 60 >> %LOG% 2>&1
if errorlevel 1 echo executor>> %FAIL%
rem 3) rebuild local dashboard (dashboard/index.html + live.json) from the fresh ledgers
%PY% build_dashboard.py >> %LOG% 2>&1
if errorlevel 1 echo dashboard>> %FAIL%
rem 4) copy live.json into the Google Drive sync folder (claude.ai dashboard reads it via the Drive connector)
%PY% live\drive_sync.py >> %LOG% 2>&1
if errorlevel 1 echo drive_sync>> %FAIL%
rem 5) commit the ledgers when they changed, then push to the private GitHub repo (non-interactive)
git add live\state\paper.json live\state\paper_4x25.json live\state\executor.json dashboard\live.json dashboard\status.json dashboard\index.html >nul 2>&1
git diff --cached --quiet || git commit -q -m "paper/testnet: daily ledger %date%" >> %LOG% 2>&1
git push -q >> %LOG% 2>&1
if errorlevel 1 echo git_push>> %FAIL%
rem 6) Discord notification: only on events (failures, reset, fills, trend flips), weekly summary on Mondays
%PY% live\notify.py daily >> %LOG% 2>&1
