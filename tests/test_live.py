"""live/ 的行為測試（假交易所、暫存狀態檔，不連網）：python tests/test_live.py"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "live"))

import executor   # noqa: E402
import notify     # noqa: E402


class FakeEx:
    def __init__(self, bal):
        self.bal = bal

    def fetch_balance(self):
        return {k: {"total": v} for k, v in self.bal.items()}


def _sleeve_state(btc=0.05, eth=1.5, usdt=10.0):
    return {"capital": 9900.0, "sleeve": {"usdt": usdt, "qty": {"BTC/USDT": btc, "ETH/USDT": eth}}, "runs": {}, "fills": []}


def test_detect_reset_and_rebuild():
    st = _sleeve_state()
    assert executor.detect_reset(FakeEx({"USDT": 100.0, "BTC": 1.05, "ETH": 2.5}), st) is None
    # 重置後帳戶回到預設：BTC/ETH 各 1、USDT 10000 → 子帳 ETH 1.5 > 帳戶 1.0
    info = executor.detect_reset(FakeEx({"USDT": 10000.0, "BTC": 1.0, "ETH": 1.0}), st)
    assert info is not None and any("ETH" in x for x in info["short"])
    executor.rebuild_after_reset(st, info, "2026-11-01")
    assert st["sleeve"] == {"usdt": 9900.0, "qty": {"BTC/USDT": 0.0, "ETH/USDT": 0.0}}
    assert len(st["resets"]) == 1 and st["resets"][0]["old_sleeve"]["qty"]["ETH/USDT"] == 1.5
    # 帳戶 USDT 不足資本 → 用帳戶 × 0.99
    st2 = _sleeve_state()
    info2 = executor.detect_reset(FakeEx({"USDT": 5000.0, "BTC": 0.0, "ETH": 0.0}), st2)
    executor.rebuild_after_reset(st2, info2, "2026-11-01")
    assert abs(st2["sleeve"]["usdt"] - 4950.0) < 1e-9


def test_apply_fill_slippage_sign():
    st = _sleeve_state(btc=0.0, eth=0.0, usdt=10000.0)
    buy = {"side": "buy", "filled": 0.1, "cost": 0.1 * 101.0, "average": 101.0, "type": "market", "id": "1"}
    r = executor.apply_fill(st, "BTC/USDT", buy, {"leg": "market", "ref_mid": 100.0})
    assert abs(r["slip_bps"] - 100.0) < 1e-9          # 買貴 1% = +100 bps 成本
    sell = {"side": "sell", "filled": 0.1, "cost": 0.1 * 99.0, "average": 99.0, "type": "market", "id": "2"}
    r2 = executor.apply_fill(st, "BTC/USDT", sell, {"leg": "market", "ref_mid": 100.0})
    assert abs(r2["slip_bps"] - 100.0) < 1e-9         # 賣便宜 1% = +100 bps 成本
    assert abs(st["sleeve"]["qty"]["BTC/USDT"]) < 1e-12


def test_notify_collect_events():
    with tempfile.TemporaryDirectory() as d:
        notify.STATE_DIR, notify.FAILED = d, os.path.join(d, "failed_steps.txt")
        dec = lambda on: {"BTC/USDT": {"on": on, "bar": "2026-11-02", "close": 90000.0, "dist_pct": 0.01 if on else -0.01}}
        ex = {"runs": {"2026-11-01": {"status": "done", "decision": dec(True), "reconcile": {"ok": True}},
                       "2026-11-02": {"status": "done", "decision": dec(False), "finished": "2026-11-03T00:06:00+00:00",
                                      "reconcile": {"ok": False, "rows": {"BTC": {"ok": False}}},
                                      "fills": [{"symbol": "BTC/USDT", "side": "sell", "qty": 0.05, "avg": 90000.0, "type": "market", "slip_bps": 2.0}]}}}
        json.dump(ex, open(os.path.join(d, "executor.json"), "w", encoding="utf-8"))
        paper = {"config": {"start": "2026-10-01", "capital": 10000.0}, "last_processed": "2026-10-30", "history": []}
        json.dump(paper, open(os.path.join(d, "paper.json"), "w", encoding="utf-8"))
        open(notify.FAILED, "w").write("drive_sync\ngit_push\n")
        alerts, events = notify.collect(datetime(2026, 11, 3, 0, 6, tzinfo=timezone.utc))
        text = "\n".join(alerts + events)
        assert "寫入 Google Drive" in text and "推上 GitHub" in text
        assert "對帳不符" in text
        assert "沒推進到 2026-11-02" in text
        assert "跌破 EMA50" in text and "測試網今天成交" in text and "+2.0 bps" in text


def test_notify_quiet_when_nothing_happens():
    with tempfile.TemporaryDirectory() as d:
        notify.STATE_DIR, notify.FAILED = d, os.path.join(d, "failed_steps.txt")
        dec = {"BTC/USDT": {"on": True, "bar": "2026-11-02", "close": 1.0, "dist_pct": 0.1}}
        ex = {"runs": {"2026-11-01": {"status": "done", "decision": dec, "reconcile": {"ok": True}},
                       "2026-11-02": {"status": "done", "decision": dec, "finished": "2026-11-03T00:06:00+00:00", "reconcile": {"ok": True}, "fills": []}}}
        json.dump(ex, open(os.path.join(d, "executor.json"), "w", encoding="utf-8"))
        paper = {"config": {"start": "2026-10-01", "capital": 10000.0}, "last_processed": "2026-11-02", "history": [{"fills": []}]}
        json.dump(paper, open(os.path.join(d, "paper.json"), "w", encoding="utf-8"))
        alerts, events = notify.collect(datetime(2026, 11, 3, 0, 6, tzinfo=timezone.utc))
        assert alerts == [] and events == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    n = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            n += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"{n}/{len(tests)} passed")
    sys.exit(0 if n == len(tests) else 1)
