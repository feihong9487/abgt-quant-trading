"""現貨紙上前向測試：ema50_plain，BTC/USDT 與 ETH/USDT 現貨各 50% 名目，每天 00:00 UTC 收盤後跑一次。

規則（與回測逐位相同）：收盤 > EMA50 → 下一根開盤買到目標曝險；收盤 < EMA50 → 下一根開盤全賣。
記錄型運行指標（不自動動作）：份額法 NAV、同起點買入持有 NAV、比值、比值高水位回撤、90/180/365 根滾動比值、策略絕對回撤。

用法（在 ABGT 目錄）：
  python live/paper.py init --start 2026-10-01                 # 建立帳本（可加 --symbols/--weights/--capital/--fee/--slip）
  python live/paper.py run                                     # 每日執行：補資料、成交昨天排的單、收盤決策、記錄
  python live/paper.py report                                  # 看目前狀態
  python live/paper.py selftest --start 2026-06-01             # 用過去區間跑一遍並與回測引擎比對
排程：Windows 工作排程器每天 08:05（台北）= 00:05 UTC 執行 run；見 live/README.md。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from abgt import data as dt                      # noqa: E402
from abgt import indicators as ta                # noqa: E402

STATE_DIR = os.path.join(ROOT, "live", "state")
DEFAULT_STATE = os.path.join(STATE_DIR, "paper.json")
DATA_SINCE = "2019-09-01"


def load_state(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_state(st: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def fetch_all(symbols: list[str], offline: bool) -> dict[str, pd.DataFrame]:
    out = {}
    for s in symbols:
        df = dt.load_cached("data", s, "1d", DATA_SINCE) if offline else dt.fetch_ohlcv(s, "1d", DATA_SINCE, "data", exchange_id="binance", verbose=False)
        if df is None:
            raise SystemExit(f"{s} 沒有現貨快取，去掉 --offline")
        out[s] = df
    return out


def regime_today(df: pd.DataFrame, upto: pd.Timestamp, ema_len: int) -> dict:
    """只用 ≤ upto 的收盤重算（不沿用任何快取狀態），回傳當日 EMA、狀態與距離。"""
    c = df.loc[:upto, "close"].to_numpy(dtype="float64")
    if len(c) < ema_len + 1:
        raise SystemExit(f"歷史不足 {ema_len} 根")
    ema = ta.ema(c, ema_len)
    return {"close": float(c[-1]), "ema": float(ema[-1]), "on": bool(c[-1] > ema[-1]), "dist_pct": float(c[-1] / ema[-1] - 1)}


def init(args) -> None:
    symbols = args.symbols.split(",")
    weights = [float(x) for x in args.weights.split(",")]
    st = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"symbols": symbols, "weights": dict(zip(symbols, weights)), "capital": args.capital, "fee": args.fee, "slip": args.slip,
                   "ema_len": args.ema_len, "rebalance_threshold": 0.15, "strategy": "ema50_plain", "exchange": "binance spot",
                   "start": args.start, "note": "收盤決策、次日開盤成交；運行閘門只記錄不動作（第 4 輪結論）"},
        "cash": args.capital, "positions": {s: 0.0 for s in symbols}, "pending": {}, "bh_qty": {}, "started": False,
        "last_processed": None, "history": [], "fills": [], "ratio_hwm": 1.0, "nav_hwm": args.capital,
    }
    save_state(st, args.state)
    print(f"已建立 {args.state}：{symbols} 權重 {weights}，資金 {args.capital}，上線日 {args.start}（費 {args.fee:.2%}/邊、滑 {args.slip:.2%}）")


def step(st: dict, dfs: dict[str, pd.DataFrame], day: pd.Timestamp) -> dict:
    """處理一根已收完的日線：開盤成交昨天排的單 → 收盤決策 → 記錄。"""
    cfg = st["config"]
    symbols, weights, fee, slip = cfg["symbols"], cfg["weights"], cfg["fee"], cfg["slip"]
    rows = {s: dfs[s].loc[day] for s in symbols}
    start = pd.Timestamp(cfg["start"], tz="UTC")
    if day < start:
        return {}
    # 1) 開盤：成交昨天排的目標曝險
    fills_today = []
    if st["started"]:
        # 先賣後買；買量以現金為上限（現貨不能透支）—— 與 simulate_portfolio 同規則
        nav0 = st["cash"] + sum(st["positions"][s] * rows[s]["open"] for s in symbols)
        todo = sorted(st["pending"].items(), key=lambda kv: kv[1] * nav0 / float(rows[kv[0]]["open"]) - st["positions"][kv[0]] > 0)
        for s, target in todo:
            o = float(rows[s]["open"])
            nav_open = st["cash"] + sum(st["positions"][x] * rows[x]["open"] for x in symbols)
            target_qty = target * nav_open / o
            diff = target_qty - st["positions"][s]
            if diff > 0:
                diff = min(diff, max(0.0, st["cash"]) / (o * (1 + slip) * (1 + fee)))
            if abs(diff) * o < 1e-6 * nav_open:
                continue
            px = o * (1 + slip) if diff > 0 else o * (1 - slip)
            cost = abs(diff) * px
            if diff > 0:
                st["cash"] -= cost * (1 + fee)
            else:
                st["cash"] += cost * (1 - fee)
            st["positions"][s] = max(0.0, st["positions"][s] + diff)
            fills_today.append({"date": str(day.date()), "symbol": s, "side": "買" if diff > 0 else "賣", "qty": abs(diff), "price": px,
                                "notional": cost, "fee": cost * fee, "reason": "進場" if target > 0 and diff > 0 else ("趨勢結束" if target == 0 else "調整曝險")})
        st["pending"] = {}
        st["fills"].extend(fills_today)
    # 2) 收盤：起算 / 決策
    closes = {s: float(rows[s]["close"]) for s in symbols}
    if not st["started"]:
        st["started"] = True
        st["bh_qty"] = {s: cfg["capital"] * weights[s] * (1 - fee) / closes[s] for s in symbols}
    nav = st["cash"] + sum(st["positions"][s] * closes[s] for s in symbols)
    bh_nav = sum(st["bh_qty"][s] * closes[s] for s in symbols)
    ratio = nav / bh_nav if bh_nav > 0 else 1.0
    st["ratio_hwm"] = max(st["ratio_hwm"], ratio)
    st["nav_hwm"] = max(st["nav_hwm"], nav)
    regimes = {}
    for s in symbols:
        r = regime_today(dfs[s], day, cfg["ema_len"])
        regimes[s] = r
        target = weights[s] if r["on"] else 0.0
        current = st["positions"][s] * closes[s] / nav if nav > 0 else 0.0
        if (target == 0 and current > 0) or (target > 0 and current == 0) or (target > 0 and abs(target - current) > cfg["rebalance_threshold"] * weights[s]):
            st["pending"][s] = target
    hist = st["history"]
    def roll(n):
        if len(hist) >= n:
            return ratio / hist[-n]["ratio"] - 1
        return None
    rec = {"date": str(day.date()), "nav": nav, "bh_nav": bh_nav, "ratio": ratio, "ratio_dd": ratio / st["ratio_hwm"] - 1,
           "nav_dd": nav / st["nav_hwm"] - 1, "roll90": roll(90), "roll180": roll(180), "roll365": roll(365),
           "cash": st["cash"], "positions": dict(st["positions"]),
           "regime": {s: {"on": regimes[s]["on"], "close": regimes[s]["close"], "ema": regimes[s]["ema"], "dist_pct": regimes[s]["dist_pct"]} for s in symbols},
           "pending": dict(st["pending"]), "fills": fills_today}
    hist.append(rec)
    st["last_processed"] = str(day.date())
    return rec


def run(args) -> None:
    st = load_state(args.state)
    cfg = st["config"]
    dfs = fetch_all(cfg["symbols"], args.offline)
    idx = None
    for df in dfs.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    upto = pd.Timestamp(args.upto, tz="UTC") if args.upto else idx[-1]
    start = pd.Timestamp(cfg["start"], tz="UTC")
    last = pd.Timestamp(st["last_processed"], tz="UTC") if st["last_processed"] else None
    days = [d for d in idx if d <= upto and d >= start and (last is None or d > last)]
    if not days:
        print(f"沒有新的已收盤日線（最後處理 {st['last_processed']}，資料到 {idx[-1].date()}，上線日 {cfg['start']}）")
        print_report(st)
        return
    for d in days:
        rec = step(st, dfs, d)
        if rec and args.verbose:
            print(f"{rec['date']} NAV {rec['nav']:.2f} 持有 {rec['bh_nav']:.2f} 比值 {rec['ratio']:.4f} " + " ".join(f"{s.split('/')[0]}:{'在' if rec['regime'][s]['on'] else '空'}({rec['regime'][s]['dist_pct'] * 100:+.1f}%)" for s in cfg["symbols"]) + (f" 成交 {len(rec['fills'])}" if rec["fills"] else "") + (f" 明日 {rec['pending']}" if rec["pending"] else ""))
    save_state(st, args.state)
    print_report(st)


def print_report(st: dict) -> None:
    cfg = st["config"]
    if not st["history"]:
        print(f"尚未起算（上線日 {cfg['start']}）")
        return
    h = st["history"][-1]
    print(f"=== 紙上前向 {cfg['strategy']} | {', '.join(cfg['symbols'])} | 上線 {cfg['start']} | 最後處理 {h['date']} | 共 {len(st['history'])} 日 ===")
    print(f"NAV {h['nav']:,.2f}（{h['nav'] / cfg['capital'] - 1:+.2%}）| 持有 {h['bh_nav']:,.2f}（{h['bh_nav'] / cfg['capital'] - 1:+.2%}）| 比值 {h['ratio']:.4f}（{h['ratio'] - 1:+.2%}）")
    print(f"比值高水位回撤 {h['ratio_dd']:+.2%} | NAV 回撤 {h['nav_dd']:+.2%} | 滾動 90/180/365：" + " / ".join('—' if h[k] is None else f"{h[k]:+.1%}" for k in ("roll90", "roll180", "roll365")))
    for s in cfg["symbols"]:
        r = h["regime"][s]
        pos = h["positions"][s]
        print(f"  {s}: {'趨勢內' if r['on'] else '趨勢外'} close {r['close']:,.2f} EMA{cfg['ema_len']} {r['ema']:,.2f}（{r['dist_pct']:+.2%}）| 持倉 {pos:.6f}（{pos * r['close'] / h['nav']:.1%} NAV）")
    if h["pending"]:
        print("明天 00:00 UTC 開盤要做：" + "；".join(f"{s} → 目標曝險 {t:.0%}" for s, t in h["pending"].items()))
    else:
        print("明天沒有單。")
    if st["fills"]:
        f = st["fills"][-1]
        print(f"最近成交：{f['date']} {f['side']} {f['symbol']} {f['qty']:.6f} @ {f['price']:,.2f}（{f['reason']}）| 累計成交 {len(st['fills'])} 筆、手續費 {sum(x['fee'] for x in st['fills']):,.2f}")


def selftest(args) -> None:
    """用過去區間跑紙上帳本，與回測引擎（simulate_portfolio）比對 NAV 路徑。"""
    from abgt.portfolio import simulate_portfolio, buy_and_hold_portfolio, align
    from abgt.strategies.regime import RiskParams
    tmp = os.path.join(STATE_DIR, "selftest.json")
    args.state = tmp
    args.capital, args.fee, args.slip, args.ema_len = 10000.0, 0.001, 0.0005, 50
    init(args)
    st = load_state(tmp)
    dfs = fetch_all(st["config"]["symbols"], True)
    idx = align(dfs)
    start = pd.Timestamp(args.start, tz="UTC")
    days = [d for d in idx if d >= start]
    for d in days:
        step(st, dfs, d)
    save_state(st, tmp)
    print_report(st)
    # 回測引擎同口徑：起算根 = start 那根，regime = close > EMA50
    s0 = int(idx.searchsorted(start))
    regimes = {}
    for s in dfs:
        c = dfs[s]["close"].to_numpy(dtype="float64")
        regimes[s] = c > ta.ema(c, 50)
    rp = RiskParams(commission_pct=0.001, slippage_pct=0.0005, rebalance_threshold=0.15)
    res = simulate_portfolio(dfs, regimes, rp, st["config"]["weights"], s0)
    eng = res.equity
    paper = pd.Series([h["nav"] for h in st["history"]], index=pd.to_datetime([h["date"] for h in st["history"]], utc=True))
    common = eng.index.intersection(paper.index)
    diff = (paper.reindex(common) / eng.reindex(common) - 1)
    print(f"\n自我驗證：{len(common)} 日 | 紙上 NAV / 引擎權益 最大相對差 {diff.abs().max():.5%}（期末 {diff.iloc[-1]:+.5%}）| 引擎成交 {res.extra['n_fills']} 筆 vs 紙上 {len(st['fills'])} 筆")
    os.remove(tmp)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--symbols", default="BTC/USDT,ETH/USDT")
    p.add_argument("--weights", default="0.5,0.5")
    p.add_argument("--capital", type=float, default=10000.0)
    p.add_argument("--fee", type=float, default=0.001, help="現貨 taker 0.10%%/邊（BNB 抵扣 0.075%%）")
    p.add_argument("--slip", type=float, default=0.0005)
    p.add_argument("--ema-len", type=int, default=50)
    p.add_argument("--start", required=True, help="上線日 YYYY-MM-DD（UTC）")
    p.add_argument("--state", default=DEFAULT_STATE)
    p = sub.add_parser("run")
    p.add_argument("--state", default=DEFAULT_STATE)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--upto", help="只處理到這個日期（測試用）")
    p.add_argument("--verbose", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--state", default=DEFAULT_STATE)
    p = sub.add_parser("selftest")
    p.add_argument("--start", required=True)
    p.add_argument("--symbols", default="BTC/USDT,ETH/USDT")
    p.add_argument("--weights", default="0.5,0.5")
    args = ap.parse_args()
    if args.cmd == "init":
        init(args)
    elif args.cmd == "run":
        run(args)
    elif args.cmd == "report":
        print_report(load_state(args.state))
    elif args.cmd == "selftest":
        selftest(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
