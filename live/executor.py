"""現貨執行器（只連 Binance 現貨測試網 testnet.binance.vision；程式裡沒有任何正式網下單路徑）。

策略與紙上前向相同：ema50_plain，BTC/USDT、ETH/USDT 各 50% 名目（1×，不借錢）。
- 決策：用「正式網公開日線」（唯讀、不需金鑰）最後一根已收盤 K 線判定 close > EMA50；測試網價格與深度和正式網不同，只拿來成交。
- 帳本：測試網帳戶預設有一堆假幣，所以執行器只管自己的「子帳」（sleeve）：起始 USDT 資本、每個標的自己買進的數量；
  成交後依實際成交回報（數量、均價、手續費幣別）更新子帳；對帳時確認帳戶餘額 ≥ 子帳持倉。
- 冪等：每天每個標的最多一張單，clientOrderId = abgt-<YYYYMMDD>-<BASE>-<b|s>；同一天重跑會先查這張單，已成交就不重下。
- 下單：預設市價；--maker-first N 會先掛 LIMIT_MAKER（買在買一、賣在賣一）等 N 秒，沒成交的部分撤單改市價。
- 保護：live/KILL 檔存在就不下單（--flatten 時撤掉未成交單並把子帳持倉賣光）；單筆名目上限 = 子帳 NAV × 1.05；
  最小名目（交易所 minNotional）以下不下單；任何步驟失敗都記錄並以非零碼結束，不重試到把錢花光。

金鑰：ABGT/.env（已在 .gitignore）
  BINANCE_TESTNET_KEY=...
  BINANCE_TESTNET_SECRET=...
沒有金鑰時自動 dry-run：印出會下的單，不連任何私有端點。

用法（ABGT 目錄）：
  python live/executor.py init --capital 10000            # 建子帳（testnet 帳戶要有 ≥ capital 的 USDT）
  python live/executor.py run [--dry-run] [--maker-first 60]
  python live/executor.py status                          # 子帳、帳戶餘額、對帳
  python live/executor.py check                           # 連線 / 金鑰 / 市場規格檢查（不下單）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from abgt import indicators as ta          # noqa: E402

STATE_DIR = os.path.join(ROOT, "live", "state")
STATE = os.path.join(STATE_DIR, "executor.json")
KILL = os.path.join(ROOT, "live", "KILL")
ENV = os.path.join(ROOT, ".env")
TESTNET_HOST = "testnet.binance.vision"
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
WEIGHTS = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
EMA_LEN = 50
REBALANCE_THRESHOLD = 0.15


# ---------- 基礎 ----------

def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}Z] {msg}", flush=True)


def read_env() -> dict:
    out = {}
    if os.path.exists(ENV):
        for line in open(ENV, encoding="utf-8-sig"):
            line = line.strip().replace("＝", "=")         # 中文輸入法的全形等號
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    for k in ("BINANCE_TESTNET_KEY", "BINANCE_TESTNET_SECRET"):
        if os.environ.get(k):
            out[k] = os.environ[k]
    return out


def load_state() -> dict | None:
    if not os.path.exists(STATE):
        return None
    with open(STATE, encoding="utf-8") as f:
        return json.load(f)


def save_state(st: dict) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE)


def public_mainnet():
    """正式網公開行情（唯讀）：只用來判定趨勢，不帶金鑰。"""
    import ccxt
    return ccxt.binance({"enableRateLimit": True})


def testnet(env: dict | None):
    """測試網連線。沒有金鑰時只能用公開端點。硬性檢查 URL 一定是測試網。"""
    import ccxt
    cfg = {"enableRateLimit": True, "options": {"defaultType": "spot"}}
    if env and env.get("BINANCE_TESTNET_KEY") and env.get("BINANCE_TESTNET_SECRET"):
        cfg["apiKey"] = env["BINANCE_TESTNET_KEY"]
        cfg["secret"] = env["BINANCE_TESTNET_SECRET"]
    ex = ccxt.binance(cfg)
    ex.set_sandbox_mode(True)
    for k in ("public", "private"):
        url = ex.urls["api"][k]
        if TESTNET_HOST not in url:
            raise SystemExit(f"拒絕執行：{k} 端點不是測試網（{url}）")
    return ex


def has_keys(env: dict) -> bool:
    return bool(env.get("BINANCE_TESTNET_KEY") and env.get("BINANCE_TESTNET_SECRET"))


# ---------- 決策 ----------

def decide(pub) -> dict:
    """最後一根「已收盤」日線的 close vs EMA50（正式網公開資料）。"""
    out = {}
    now_ms = pub.milliseconds()
    for s in SYMBOLS:
        rows = pub.fetch_ohlcv(s, "1d", limit=400)
        rows = [r for r in rows if r[0] + 86_400_000 <= now_ms]          # 丟掉還沒收完的當日 K 線
        c = np.array([r[4] for r in rows], dtype="float64")
        ema = ta.ema(c, EMA_LEN)
        bar_date = datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        out[s] = {"bar": bar_date, "close": float(c[-1]), "ema": float(ema[-1]), "on": bool(c[-1] > ema[-1]),
                  "dist_pct": float(c[-1] / ema[-1] - 1)}
    return out


def targets_from(decision: dict) -> dict:
    return {s: (WEIGHTS[s] if decision[s]["on"] else 0.0) for s in SYMBOLS}


def plan_orders(st: dict, prices: dict, targets: dict, markets: dict | None) -> list[dict]:
    """子帳 NAV（測試網中間價）→ 目標數量 → 差額；門檻與紙上前向相同。"""
    sl = st["sleeve"]
    nav = sl["usdt"] + sum(sl["qty"][s] * prices[s] for s in SYMBOLS)
    orders = []
    for s in SYMBOLS:
        tgt = targets[s]
        cur = sl["qty"][s] * prices[s] / nav if nav > 0 else 0.0
        need = (tgt == 0 and cur > 1e-9) or (tgt > 0 and cur <= 1e-9) or (tgt > 0 and abs(tgt - cur) > REBALANCE_THRESHOLD * WEIGHTS[s])
        if not need:
            continue
        diff_qty = tgt * nav / prices[s] - sl["qty"][s]
        if tgt == 0:
            diff_qty = -sl["qty"][s]                                      # 出場一律賣光子帳持倉
        side = "buy" if diff_qty > 0 else "sell"
        qty = abs(diff_qty)
        if markets:
            m = markets[s]
            qty = float(m["_amount_to_precision"](qty))
            min_cost = (m["limits"].get("cost") or {}).get("min") or 0.0
            if qty * prices[s] < min_cost:
                orders.append({"symbol": s, "side": side, "qty": qty, "skip": f"名目 {qty * prices[s]:.2f} < 最小 {min_cost}"})
                continue
        if side == "buy":
            qty = min(qty, sl["usdt"] / (prices[s] * 1.003))                # 不用子帳外的 USDT
        if qty * prices[s] > nav * 1.05:
            raise SystemExit(f"保護：{s} 單筆名目 {qty * prices[s]:.2f} > 子帳 NAV × 1.05（{nav:.2f}）")
        orders.append({"symbol": s, "side": side, "qty": qty, "target": tgt, "current": cur, "nav": nav, "price_ref": prices[s]})
    return orders


# ---------- 下單 ----------

def client_id(day: str, symbol: str, side: str, suffix: str = "") -> str:
    return f"abgt-{day.replace('-', '')}-{symbol.split('/')[0]}-{side[0]}{suffix}"


def find_order(ex, symbol: str, cid: str):
    """查今天這張 clientOrderId 的單；只有交易所明確回「找不到」才當沒有，其他錯誤往上丟（避免網路錯誤時重複下單）。"""
    import ccxt
    try:
        return ex.fetch_order(None, symbol, {"origClientOrderId": cid})
    except ccxt.OrderNotFound:
        return None


def apply_fill(st: dict, symbol: str, order: dict) -> dict:
    """依成交回報更新子帳；手續費依幣別扣在對應資產。回傳成交紀錄。"""
    sl = st["sleeve"]
    base = symbol.split("/")[0]
    filled = float(order.get("filled") or 0.0)
    cost = float(order.get("cost") or 0.0)
    if filled <= 0:
        return {}
    fee_base = fee_quote = fee_other = 0.0
    for f in (order.get("fees") or ([order["fee"]] if order.get("fee") else [])):
        if not f or f.get("cost") is None:
            continue
        cur, amt = f.get("currency"), float(f["cost"])
        if cur == base:
            fee_base += amt
        elif cur == "USDT":
            fee_quote += amt
        else:
            fee_other += amt
    if order["side"] == "buy":
        sl["qty"][symbol] += filled - fee_base
        sl["usdt"] -= cost + fee_quote
    else:
        sl["qty"][symbol] = max(0.0, sl["qty"][symbol] - filled - fee_base)
        sl["usdt"] += cost - fee_quote
    rec = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "symbol": symbol, "side": order["side"], "type": order.get("type"),
           "qty": filled, "avg": float(order.get("average") or (cost / filled)), "cost": cost, "fee_base": fee_base, "fee_quote": fee_quote,
           "fee_other": fee_other, "id": order.get("id"), "client_id": order.get("clientOrderId")}
    st["fills"].append(rec)
    return rec


def execute(ex, st: dict, day: str, o: dict, maker_first: int) -> list[dict]:
    s, side = o["symbol"], o["side"]
    recs = []
    remaining = o["qty"]
    if maker_first > 0:
        cid = client_id(day, s, side, "m")
        prev = find_order(ex, s, cid)
        if prev is None:
            ob = ex.fetch_order_book(s, 5)
            px = ob["bids"][0][0] if side == "buy" else ob["asks"][0][0]
            amt = float(ex.amount_to_precision(s, remaining))
            log(f"{s} 掛 LIMIT_MAKER {side} {amt} @ {px}（等 {maker_first}s）")
            try:
                prev = ex.create_order(s, "limit", side, amt, px, {"newClientOrderId": cid, "type": "LIMIT_MAKER"})
            except Exception as e:                       # 會吃單時交易所拒絕 LIMIT_MAKER → 直接走市價
                log(f"{s} LIMIT_MAKER 被拒：{type(e).__name__} {str(e)[:120]}")
                prev = None
            if prev is not None:
                deadline = time.time() + maker_first
                while time.time() < deadline:
                    time.sleep(5)
                    prev = ex.fetch_order(prev["id"], s)
                    if prev["status"] in ("closed", "canceled", "expired", "rejected"):
                        break
                if prev["status"] == "open":
                    ex.cancel_order(prev["id"], s)
                    prev = ex.fetch_order(prev["id"], s)
        if prev is not None and float(prev.get("filled") or 0) > 0 and not any(f.get("id") == prev["id"] for f in st["fills"]):
            recs.append(apply_fill(st, s, prev))
            save_state(st)
        if prev is not None:
            remaining -= float(prev.get("filled") or 0.0)
    if remaining <= 0:
        return recs
    cid = client_id(day, s, side)
    prev = find_order(ex, s, cid)
    if prev is not None:
        log(f"{s} 今天的市價單 {cid} 已存在（{prev['status']}），不重下")
        if prev["status"] == "closed" and not any(f.get("id") == prev["id"] for f in st["fills"]):
            recs.append(apply_fill(st, s, prev))
        return recs
    if side == "sell":
        remaining = min(remaining, st["sleeve"]["qty"][s])
    amt = float(ex.amount_to_precision(s, remaining))
    m = ex.market(s)
    min_cost = (m["limits"].get("cost") or {}).get("min") or 0.0
    if amt * o["price_ref"] < min_cost:
        log(f"{s} 剩餘 {amt} 低於最小名目，略過")
        return recs
    log(f"{s} 市價 {side} {amt}（clientOrderId {cid}）")
    order = ex.create_order(s, "market", side, amt, None, {"newClientOrderId": cid})
    for _ in range(6):
        if order.get("status") == "closed":
            break
        time.sleep(2)
        order = ex.fetch_order(order["id"], s)
    recs.append(apply_fill(st, s, order))
    return recs


# ---------- 對帳 ----------

def reconcile(ex, st: dict, prices: dict) -> dict:
    bal = ex.fetch_balance()
    sl = st["sleeve"]
    rows = {}
    ok = True
    for s in SYMBOLS:
        base = s.split("/")[0]
        have = float(bal.get(base, {}).get("total") or 0.0)
        need = sl["qty"][s]
        rows[base] = {"account": have, "sleeve": need, "ok": have + 1e-9 >= need}
        ok &= rows[base]["ok"]
    usdt = float(bal.get("USDT", {}).get("total") or 0.0)
    rows["USDT"] = {"account": usdt, "sleeve": sl["usdt"], "ok": usdt + 1e-6 >= sl["usdt"]}
    ok &= rows["USDT"]["ok"]
    nav = sl["usdt"] + sum(sl["qty"][s] * prices[s] for s in SYMBOLS)
    expo = {s: sl["qty"][s] * prices[s] / nav if nav > 0 else 0.0 for s in SYMBOLS}
    return {"ok": ok, "rows": rows, "nav": nav, "exposure": expo}


def mid_prices(ex) -> dict:
    out = {}
    for s in SYMBOLS:
        t = ex.fetch_ticker(s)
        out[s] = (t["bid"] + t["ask"]) / 2 if t.get("bid") and t.get("ask") else t["last"]
    return out


# ---------- 指令 ----------

def cmd_check(args) -> int:
    env = read_env()
    ex = testnet(env)
    ex.load_markets()
    log(f"測試網端點：{ex.urls['api']['public']}")
    for s in SYMBOLS:
        m = ex.market(s)
        t = ex.fetch_ticker(s)
        log(f"{s} 測試網 bid {t['bid']} ask {t['ask']} | 最小名目 {m['limits']['cost']['min']} 數量精度 {m['precision']['amount']}")
    pub = public_mainnet()
    d = decide(pub)
    for s, r in d.items():
        log(f"{s} 正式網日線 {r['bar']} close {r['close']:,.2f} EMA{EMA_LEN} {r['ema']:,.2f}（{r['dist_pct']:+.2%}）→ {'趨勢內' if r['on'] else '趨勢外'}")
    if has_keys(env):
        bal = ex.fetch_balance()
        log("金鑰可用；測試網餘額：" + ", ".join(f"{k} {bal[k]['total']}" for k in ("USDT", "BTC", "ETH", "BNB") if k in bal))
    else:
        log("沒有 BINANCE_TESTNET_KEY / SECRET（ABGT/.env）→ run 只會 dry-run")
    return 0


def cmd_init(args) -> int:
    if os.path.exists(STATE) and not args.force:
        raise SystemExit(f"{STATE} 已存在（要重建加 --force）")
    env = read_env()
    if has_keys(env):
        ex = testnet(env)
        usdt = float(ex.fetch_balance().get("USDT", {}).get("total") or 0.0)
        if usdt < args.capital:
            raise SystemExit(f"測試網 USDT {usdt} < 子帳資本 {args.capital}")
    st = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "venue": f"binance spot testnet ({TESTNET_HOST})",
          "strategy": "ema50_plain", "weights": WEIGHTS, "capital": args.capital, "rebalance_threshold": REBALANCE_THRESHOLD,
          "sleeve": {"usdt": args.capital, "qty": {s: 0.0 for s in SYMBOLS}}, "runs": {}, "fills": []}
    save_state(st)
    log(f"子帳已建立：資本 {args.capital} USDT → {STATE}")
    return 0


def cmd_run(args) -> int:
    env = read_env()
    dry = args.dry_run or not has_keys(env)
    st = load_state()
    if st is None:
        if not dry:
            raise SystemExit("先跑 init")
        st = {"capital": 10000.0, "sleeve": {"usdt": 10000.0, "qty": {s: 0.0 for s in SYMBOLS}}, "runs": {}, "fills": []}
    pub = public_mainnet()
    d = decide(pub)
    day = max(r["bar"] for r in d.values())
    targets = targets_from(d)
    for s, r in d.items():
        log(f"{s} {r['bar']} close {r['close']:,.2f} EMA{EMA_LEN} {r['ema']:,.2f}（{r['dist_pct']:+.2%}）→ 目標 {targets[s]:.0%}")
    ex = testnet(env if not dry else None)
    ex.load_markets()
    markets = {s: dict(ex.market(s), _amount_to_precision=(lambda q, s=s: ex.amount_to_precision(s, q))) for s in SYMBOLS}
    prices = mid_prices(ex)
    run_rec = st["runs"].get(day)
    if run_rec and run_rec.get("status") == "done" and not args.force:
        log(f"{day} 已執行過（{run_rec['finished']}），不重跑（--force 可強制）")
        return 0
    if os.path.exists(KILL):
        log("偵測到 live/KILL：不下新單")
        if args.flatten and not dry:
            for o in ex.fetch_open_orders():
                if o["symbol"] in SYMBOLS:
                    ex.cancel_order(o["id"], o["symbol"])
            targets = {s: 0.0 for s in SYMBOLS}
        else:
            return 3
    orders = plan_orders(st, prices, targets, markets)
    if not orders:
        log("今天不用下單")
    for o in orders:
        log(("[dry-run] " if dry else "") + (f"{o['symbol']} 略過：{o['skip']}" if o.get("skip") else
            f"{o['symbol']} {o['side']} {o['qty']:.6f}（目前 {o['current']:.1%} → 目標 {o['target']:.0%}，參考價 {o['price_ref']:,.2f}）"))
    if dry:
        return 0
    prev = st["runs"].get(day) or {}
    st["runs"][day] = {"status": "running", "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "decision": d,
                       "targets": targets, "orders": [{k: v for k, v in o.items() if not callable(v)} for o in orders],
                       "prev_fills": [f for f in prev.get("fills", []) + prev.get("prev_fills", []) if f],   # --force 重跑不丟前一次的成交
                       "attempt": int(prev.get("attempt", 0)) + 1}
    save_state(st)
    fills = []
    try:
        for o in orders:                                   # 先賣後買，釋放 USDT
            if o.get("skip") or o["side"] != "sell":
                continue
            fills += execute(ex, st, day, o, args.maker_first)
            save_state(st)
        prices = mid_prices(ex)
        for o in plan_orders(st, prices, targets, markets):
            if o.get("skip") or o["side"] != "buy":
                continue
            fills += execute(ex, st, day, o, args.maker_first)
            save_state(st)
    except Exception as e:
        st["runs"][day]["status"] = "error"
        st["runs"][day]["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        save_state(st)
        log(f"錯誤：{st['runs'][day]['error']}")
        return 2
    rec = reconcile(ex, st, mid_prices(ex))
    st["runs"][day].update({"status": "done", "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "fills": st["runs"][day].pop("prev_fills") + [f for f in fills if f], "reconcile": rec})
    save_state(st)
    log(f"完成：成交 {len([f for f in fills if f])} 筆 | 子帳 NAV {rec['nav']:,.2f} | 曝險 " + ", ".join(f"{s.split('/')[0]} {v:.1%}" for s, v in rec["exposure"].items())
        + f" | 對帳 {'OK' if rec['ok'] else '不符'}")
    return 0 if rec["ok"] else 4


def cmd_status(args) -> int:
    st = load_state()
    if st is None:
        raise SystemExit("尚未 init")
    env = read_env()
    ex = testnet(env if has_keys(env) else None)
    prices = mid_prices(ex)
    sl = st["sleeve"]
    nav = sl["usdt"] + sum(sl["qty"][s] * prices[s] for s in SYMBOLS)
    log(f"子帳 NAV {nav:,.2f}（{nav / st['capital'] - 1:+.2%}）| USDT {sl['usdt']:,.2f} | " + ", ".join(f"{s} {sl['qty'][s]:.6f}" for s in SYMBOLS))
    if st["runs"]:
        last = sorted(st["runs"])[-1]
        log(f"最後執行 {last}：{st['runs'][last]['status']}")
    if has_keys(env):
        rec = reconcile(ex, st, prices)
        log("對帳：" + "; ".join(f"{k} 帳戶 {v['account']:.6f} / 子帳 {v['sleeve']:.6f} {'OK' if v['ok'] else '不足'}" for k, v in rec["rows"].items()))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p = sub.add_parser("init")
    p.add_argument("--capital", type=float, default=10000.0)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--maker-first", type=int, default=0, help="先掛 LIMIT_MAKER 等 N 秒，剩下的轉市價")
    p.add_argument("--force", action="store_true")
    p.add_argument("--flatten", action="store_true", help="配合 live/KILL：撤單並賣光子帳持倉")
    sub.add_parser("status")
    args = ap.parse_args()
    return {"check": cmd_check, "init": cmd_init, "run": cmd_run, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
