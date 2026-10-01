"""每日排程的通知：讀帳本與失敗紀錄，有事才發 Discord（webhook），每週一發一次心跳摘要。

設定（ABGT/.env，已在 .gitignore）：
  DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
沒有設定時只把訊息印到 run.log，不會讓排程失敗。

事件：
  - 排程步驟失敗（run_daily.bat 寫的 live/state/failed_steps.txt）
  - 測試網執行器：錯誤 / 金鑰失效 / 對帳不符 / 偵測到重置 / 今天有成交
  - 趨勢翻轉：今天的決策 K 線與前一次執行相比，某個幣穿越 EMA50
  - 紙上帳本沒推進到昨天（每本各查）
心跳：每週一（UTC）發一次兩本紙上帳本與測試網子帳的摘要，證明系統還活著。

用法：
  python live/notify.py daily [--always]     # 排程最後一步；--always 沒事也發摘要
  python live/notify.py test                 # 發一則測試訊息
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(ROOT, "live", "state")
FAILED = os.path.join(STATE_DIR, "failed_steps.txt")
LEDGERS = (("paper.json", "50/50（登記）"), ("paper_4x25.json", "4×25%（記錄）"))
DASHBOARD = "https://claude.ai/artifact/Ang6gMopYsRizJ1wdShTQ7"
STEP_NAMES = {"paper": "紙上帳本 50/50", "paper_4x25": "紙上帳本 4×25%", "executor": "測試網執行器", "dashboard": "重建 dashboard",
              "drive_sync": "寫入 Google Drive", "git_commit": "git commit", "git_push": "推上 GitHub"}


def read_env() -> dict:
    out = {}
    p = os.path.join(ROOT, ".env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8-sig"):
            line = line.strip().replace("＝", "=")
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    if os.environ.get("DISCORD_WEBHOOK_URL"):
        out["DISCORD_WEBHOOK_URL"] = os.environ["DISCORD_WEBHOOK_URL"]
    return out


def jload(name: str) -> dict | None:
    p = os.path.join(STATE_DIR, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def send(text: str) -> bool:
    """Discord 一則訊息上限 2000 字；超過就分段。沒有 webhook 只印出。"""
    url = read_env().get("DISCORD_WEBHOOK_URL")
    print(text)
    if not url:
        print("notify：沒有 DISCORD_WEBHOOK_URL，只寫進 log")
        return False
    chunks, cur = [], ""
    for line in text.splitlines():
        if len(cur) + len(line) + 1 > 1900:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    chunks.append(cur)
    ok = True
    for c in chunks:
        req = urllib.request.Request(url, data=json.dumps({"content": c, "username": "ABGT"}).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "User-Agent": "ABGT-notify/1.0"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                ok &= 200 <= r.status < 300
        except urllib.error.HTTPError as e:
            print(f"notify：Discord 回 {e.code} {e.read()[:200]!r}")
            ok = False
        except Exception as e:                                      # 網路錯誤不讓排程失敗
            print(f"notify：送出失敗 {type(e).__name__}: {e}")
            ok = False
    return ok


def collect(now: datetime) -> tuple[list[str], list[str]]:
    """回傳 (警示事件, 一般事件)。"""
    alerts, events = [], []
    if os.path.exists(FAILED):
        steps = [x.strip() for x in open(FAILED, encoding="utf-8", errors="replace") if x.strip()]
        for s in dict.fromkeys(steps):
            alerts.append(f"❌ 排程步驟失敗：{STEP_NAMES.get(s, s)}（看 live/state/run.log）")
    # 測試網
    ex = jload("executor.json")
    if ex and ex.get("runs"):
        days = sorted(ex["runs"])
        day, r = days[-1], ex["runs"][days[-1]]
        st = r.get("status")
        if st == "auth_error":
            alerts.append(f"🔑 測試網金鑰失效：{r.get('error', '')[:150]} → 到 testnet.binance.vision 重產金鑰、更新 .env")
        elif st == "error":
            alerts.append(f"❌ 測試網執行錯誤（{day}）：{r.get('error', '')[:200]}")
        elif st == "running":
            alerts.append(f"⚠️ 測試網執行沒有完成（{day} 狀態 running），可能中途中斷")
        if r.get("reset"):
            last = (ex.get("resets") or [{}])[-1]
            alerts.append(f"♻️ 偵測到測試網重置，已自動重建子帳（資本 {last.get('new_capital', 0):,.0f} USDT）")
        rec = r.get("reconcile") or {}
        if st == "done" and rec.get("ok") is False:
            bad = [k for k, v in (rec.get("rows") or {}).items() if not v.get("ok")]
            alerts.append(f"⚠️ 測試網對帳不符：{', '.join(bad)}（帳戶 < 子帳）")
        fills = [f for f in r.get("fills", []) if f]
        if fills and r.get("finished", "")[:10] == now.strftime("%Y-%m-%d"):
            lines = []
            for f in fills:
                q = f" · 滑價 {f['slip_bps']:+.1f} bps" if f.get("slip_bps") is not None else ""
                lines.append(f"  {f['symbol'].split('/')[0]} {'買' if f['side'] == 'buy' else '賣'} {f['qty']:.5f} @ {f['avg']:,.2f}（{'掛單' if f.get('type') == 'limit' else '市價'}{q}）")
            events.append("🧾 測試網今天成交：\n" + "\n".join(lines))
        # 趨勢翻轉：與前一次決策比
        if len(days) >= 2:
            d0, d1 = ex["runs"][days[-2]].get("decision") or {}, r.get("decision") or {}
            for s in d1:
                if s in d0 and d0[s].get("on") != d1[s].get("on"):
                    up = d1[s]["on"]
                    events.append(f"{'📈' if up else '📉'} {s.split('/')[0]} {'站上' if up else '跌破'} EMA50（{d1[s]['bar']} 收 {d1[s]['close']:,.2f}，距離 {d1[s]['dist_pct']:+.1%}）→ {'進場' if up else '出場'}")
    # 紙上帳本是否推進到昨天（00:05 UTC 跑，最後收完的 K 線 = 昨天）
    expect = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    for fname, label in LEDGERS:
        p = jload(fname)
        if not p:
            continue
        start = p["config"]["start"]
        if expect < start:
            continue
        if p.get("last_processed") != expect:
            alerts.append(f"⚠️ 紙上帳本 {label} 沒推進到 {expect}（最後處理 {p.get('last_processed')}）")
        else:
            h = p["history"][-1]
            for f in h.get("fills") or []:
                events.append(f"📝 紙上 {label}：{f['symbol'].split('/')[0]} {f['side']} {f['qty']:.5f} @ {f['price']:,.2f}（{f['reason']}）")
    return alerts, events


def summary() -> str:
    lines = ["📊 ABGT 每週摘要"]
    for fname, label in LEDGERS:
        p = jload(fname)
        if not p:
            continue
        if not p.get("history"):
            lines.append(f"• 紙上 {label}：尚未起算（上線 {p['config']['start']}）")
            continue
        h = p["history"][-1]
        lines.append(f"• 紙上 {label}：NAV {h['nav']:,.0f}（{h['nav'] / p['config']['capital'] - 1:+.1%}）· 持有 {h['bh_nav']:,.0f} · 比值 {h['ratio']:.3f} · 回撤 {h['nav_dd']:+.1%} · {len(p['history'])} 天")
    ex = jload("executor.json")
    if ex and ex.get("runs"):
        day = sorted(ex["runs"])[-1]
        r = ex["runs"][day]
        rec = r.get("reconcile") or {}
        q = [f["slip_bps"] for f in ex.get("fills", []) if f.get("slip_bps") is not None]
        mk = [f for f in ex.get("fills", []) if f.get("leg") in ("maker", "market")]
        maker_share = (sum(1 for f in mk if f["leg"] == "maker") / len(mk)) if mk else None
        lines.append(f"• 測試網子帳：NAV {rec.get('nav', 0):,.0f} · 最後執行 {day} {r.get('status')} · 對帳 {'OK' if rec.get('ok') else '—'}"
                     + (f" · 平均滑價 {sum(q) / len(q):+.1f} bps（{len(q)} 筆）" if q else "")
                     + (f" · 掛單成交佔比 {maker_share:.0%}" if maker_share is not None else ""))
        if ex.get("resets"):
            lines.append(f"• 測試網重置過 {len(ex['resets'])} 次（最近 {ex['resets'][-1]['day']}）")
    lines.append(DASHBOARD)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("daily")
    p.add_argument("--always", action="store_true")
    sub.add_parser("test")
    args = ap.parse_args()
    now = datetime.now(timezone.utc)
    if args.cmd == "test":
        ok = send(f"✅ ABGT 通知測試 {now:%Y-%m-%d %H:%M} UTC — 之後每天 08:05（台北）有事才會發，每週一發摘要。\n{DASHBOARD}")
        return 0 if ok else 1
    alerts, events = collect(now)
    weekly = now.weekday() == 0
    if alerts or events or args.always or weekly:
        parts = [f"**ABGT {now:%Y-%m-%d}**"]
        if alerts:
            parts += alerts
        if events:
            parts += events
        if weekly or args.always:
            parts.append(summary())
        elif alerts or events:
            parts.append(DASHBOARD)
        send("\n".join(parts))
    else:
        print(f"notify：{now:%Y-%m-%d} 沒有事件，不發")
    if os.path.exists(FAILED):
        os.remove(FAILED)
    return 0


if __name__ == "__main__":
    sys.exit(main())
