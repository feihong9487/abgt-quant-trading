@echo off
rem ABGT spot paper-forward: called by Task Scheduler daily at 08:05 Taipei (= 00:05 UTC)
cd /d D:\feihong\tradingView\ABGT
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
echo ===== %date% %time% ===== >> live\state\run.log
.venv\Scripts\python.exe live\paper.py run --verbose >> live\state\run.log 2>&1
rem commit the ledger when it changed (no push)
git add live\state\paper.json >nul 2>&1
git diff --cached --quiet || git commit -q -m "paper: daily ledger %date%" >> live\state\run.log 2>&1
