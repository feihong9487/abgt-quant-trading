"""槓桿「成本」：ema50_plain 在現貨資料上的 手續費 × 滑點 網格、損益兩平成本、換手成本拖累、maker-first 執行層估計。

重跑：
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/levers/costs.py
輸出：
    results/levers/costs.json、results/levers/costs.md

口徑（全部可獨立重算）：
- 資料：Binance 現貨日線 BTC/USDT、ETH/USDT（data/ 快取，2019-09-01 起）；執行層 15 分交叉檢查用 ETH/USDT:USDT 永續 15m（2022-12 起，只有這一組 15m 快取）
- 策略：experiments/variants/ema50_plain.py 原樣（close > EMA50 滿倉否則空手；收盤判定、次日開盤成交；EMA50 SMA 起算，warmup = 第 50 根）
- 引擎：abgt.strategies.regime.simulate_regime（單標的）、abgt.portfolio.simulate_portfolio（50/50，rebalance_threshold 0.15，與 evaluate_portfolio 預設一致）
- 區間：IS 2020-01→2022-12、OOS 2023-01→今、FULL 2020-01→今、IS2 2020-07→2023-12、OOS2 2024-01→今；每格起點 = max(warmup, 區間起點)，權益在格內重新歸一
- 閘門：abgt.benchmark.gate(mode="risk")：Sharpe > 同期持有 且 Calmar > 持有 且 最大回撤 < 持有；持有 = 同格起點買入、付一次進場手續費
- 單標的「五格」× BTC/ETH = 十格；組合「十格」= 五格 × 兩種持有基準（不再平衡 / 月再平衡）
- 損益兩平：每邊總成本 c 以 commission_pct = c、slippage_pct = 0 施加（比例手續費與比例滑點對權益的一階效果相同：每邊都扣 c × 成交名目），
  粗掃 0→1.2% 步 0.02% 找第一個掛格，再在 [最後過, 第一掛] 之間二分到 0.001%；同時記錄非單調翻轉次數
- 換手：名目換手/年 = 格內成交名目（買 + 賣）/ 格內平均權益 / 格內年數；估計拖累 = 換手 × 每邊成本；實測拖累 = (零成本 log 成長 − 有成本 log 成長)/年
- 執行層：對每個實際成交日（買 = 進場日、賣 = 出場日）用當日 open/high/low/close 估「開盤價掛限價當日成交」機率：
  買單掛 O 視為成交若 low ≤ O×(1−δ)、賣單若 high ≥ O×(1+δ)，δ ∈ {0（嚴格穿過）, 0.02%, 0.05%, 0.10%, 0.20%}；δ 越大越保守
  期望每邊成本 = p×maker + (1−p)×(taker + 市價滑點) + (1−p)×E[延遲成本 | 未成交]；延遲成本用「未成交那部分」的同日 open→close（買：C/O−1；賣：1−C/O）
  ETH 另用永續 15m 估「前 N 根（15/30/60/120 分）內穿過」的機率與 N 分鐘後轉市價的漂移成本
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

from abgt import data as dt                                                   # noqa: E402
from abgt.benchmark import buy_and_hold, gate, segment_table                  # noqa: E402
from abgt.metrics import summarize                                            # noqa: E402
from abgt.portfolio import align, buy_and_hold_portfolio, simulate_portfolio  # noqa: E402
from abgt.strategies.regime import RiskParams, simulate_regime                # noqa: E402
from evaluate_variant import ALT_WINDOWS, DATA_SINCE, DEFAULT_WINDOWS, load_variant, parse_windows  # noqa: E402

KEY = "costs"
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
SHORT = {s: s.split("/")[0] for s in SYMBOLS}
WEIGHTS = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
FEES = [0.0002, 0.00075, 0.001]                 # maker / BNB 抵扣 / taker
SLIPS = [0.0002, 0.0005, 0.001, 0.002]
WINDOWS = parse_windows(DEFAULT_WINDOWS + "," + ALT_WINDOWS)
WNAMES = [w[0] for w in WINDOWS]
INITIAL = 10_000.0
REB_THRESHOLD = 0.15
MAKER, BNB, TAKER, MKT_SLIP = 0.0002, 0.00075, 0.001, 0.0005
LIVE_FEE, LIVE_SLIP = 0.001, 0.0005             # 上線登記口徑
DELTAS = [0.0, 0.0002, 0.0005, 0.001, 0.002]
BENCH_NAMES = {"nr": "持有不再平衡", "mr": "持有月再平衡"}
OUT_DIR = os.path.join(ROOT, "results", "levers")
ETH_15M = "ETH/USDT:USDT"


def pct(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{nd}f}%"


def num(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def window_slices(index: pd.DatetimeIndex, warmup: int):
    out = []
    for wname, wstart, wend in WINDOWS:
        s0 = max(warmup, int(index.searchsorted(pd.Timestamp(wstart, tz="UTC"))))
        s1 = int(index.searchsorted(pd.Timestamp(wend, tz="UTC"), side="right")) if wend else len(index)
        if s1 - s0 >= 30:
            out.append((wname, s0, s1))
    return out


# ---------------- 單標的 ----------------

def run_single(df: pd.DataFrame, on: np.ndarray, warmup: int, rp: RiskParams):
    res = simulate_regime(df, on, rp, warmup)
    out = {}
    for wname, s0, s1 in window_slices(df.index, warmup):
        eq = res.equity.loc[df.index[s0]:df.index[s1 - 1]]
        eq = eq / eq.iloc[0] * rp.initial_capital
        bh = buy_and_hold(df["close"].iloc[s0:s1], rp.initial_capital, rp.commission_pct)
        s, b = summarize(eq), summarize(bh)
        g = gate(s, b, segment_table(eq, bh), mode="risk")
        out[wname] = {"sharpe": s["sharpe"], "calmar": s["calmar"], "max_dd": s["max_dd"], "cagr": s["cagr"], "net_return": s["net_return"],
                      "bh_sharpe": b["sharpe"], "bh_calmar": b["calmar"], "bh_max_dd": b["max_dd"],
                      "pass": bool(g["passed"]), "sharpe_pass": bool(g["sharpe_pass"]), "calmar_pass": bool(g["calmar_pass"]), "dd_pass": bool(g["overall_dd_pass"]),
                      "years": s["years"], "log_growth": float(math.log(eq.iloc[-1] / eq.iloc[0])), "avg_equity": float(eq.mean()),
                      "start": str(eq.index[0].date()), "end": str(eq.index[-1].date())}
    return res, out


def turnover_single(res, df: pd.DataFrame, warmup: int, wm: dict) -> dict:
    """格內名目換手/年與成交數（從 trades 反推：每筆 closed trade = 一買一賣）。"""
    tr = res.trades
    buys = pd.Series((tr.qty * tr.entry_price).to_numpy(), index=pd.DatetimeIndex(tr.entry_time))
    sells = pd.Series((tr.qty * tr.exit_price).to_numpy(), index=pd.DatetimeIndex(tr.exit_time))
    out = {}
    for wname, s0, s1 in window_slices(df.index, warmup):
        t0, t1 = df.index[s0], df.index[s1 - 1]
        b = buys.loc[(buys.index >= t0) & (buys.index <= t1)]
        s = sells.loc[(sells.index >= t0) & (sells.index <= t1)]
        eq = res.equity.loc[t0:t1]
        years = wm[wname]["years"]
        notional = float(b.sum() + s.sum())
        out[wname] = {"turnover_per_year": notional / float(eq.mean()) / years, "fills": int(len(b) + len(s)), "fills_per_year": (len(b) + len(s)) / years}
    return out


# ---------------- 組合 ----------------

def run_portfolio(dfs, regimes, idx, warmup, rp: RiskParams):
    res = simulate_portfolio(dfs, regimes, rp, WEIGHTS, warmup)
    benches = {k: buy_and_hold_portfolio(dfs, WEIGHTS, idx, warmup, rp.initial_capital, rp.commission_pct, reb)
               for k, reb in (("nr", None), ("mr", "M"))}
    out = {}
    for wname, s0, s1 in window_slices(idx, warmup):
        e = res.equity.loc[idx[s0]:idx[s1 - 1]]
        e = e / e.iloc[0] * rp.initial_capital
        s = summarize(e)
        row = {"sharpe": s["sharpe"], "calmar": s["calmar"], "max_dd": s["max_dd"], "cagr": s["cagr"], "years": s["years"],
               "log_growth": float(math.log(e.iloc[-1] / e.iloc[0])), "start": str(e.index[0].date()), "end": str(e.index[-1].date())}
        for bk, b in benches.items():
            bb = b.loc[idx[s0]:idx[s1 - 1]]
            bb = bb / bb.iloc[0] * rp.initial_capital
            bs = summarize(bb)
            g = gate(s, bs, segment_table(e, bb), mode="risk")
            row[bk] = {"bh_sharpe": bs["sharpe"], "bh_calmar": bs["calmar"], "bh_max_dd": bs["max_dd"], "pass": bool(g["passed"]),
                       "sharpe_pass": bool(g["sharpe_pass"]), "calmar_pass": bool(g["calmar_pass"]), "dd_pass": bool(g["overall_dd_pass"])}
        row["pass_count"] = int(row["nr"]["pass"]) + int(row["mr"]["pass"])
        out[wname] = row
    return res, out


def turnover_portfolio(res, idx, warmup, wm) -> dict:
    tr = res.trades
    fills = pd.Series((tr.qty * tr.price).to_numpy(), index=pd.DatetimeIndex(tr.exit_time)) if len(tr) else pd.Series(dtype="float64")
    out = {}
    for wname, s0, s1 in window_slices(idx, warmup):
        t0, t1 = idx[s0], idx[s1 - 1]
        f = fills.loc[(fills.index >= t0) & (fills.index <= t1)]
        eq = res.equity.loc[t0:t1]
        years = wm[wname]["years"]
        out[wname] = {"turnover_per_year": float(f.sum()) / float(eq.mean()) / years, "fills": int(len(f)), "fills_per_year": len(f) / years}
    return out


# ---------------- 損益兩平 ----------------

def breakeven(eval_fn, wnames, coarse_step=0.0002, c_max=0.012, tol=0.00001) -> dict:
    """eval_fn(c) -> {wname: pass(bool)}。回傳每格 {c_star, last_pass, first_fail, flips, fail_component}。"""
    grid = [round(i * coarse_step, 8) for i in range(int(round(c_max / coarse_step)) + 1)]
    rows = {c: eval_fn(c) for c in grid}
    out = {}
    for w in wnames:
        seq = [(c, rows[c][w]["pass"]) for c in grid if w in rows[c]]
        flips = sum(1 for (c0, p0), (c1, p1) in zip(seq, seq[1:]) if p0 != p1)
        first_fail = next((c for c, p in seq if not p), None)
        if first_fail is None:
            out[w] = {"c_star": None, "note": f"掃到 {c_max:.2%} 仍全過", "flips": flips, "pass_at_zero": seq[0][1]}
            continue
        if first_fail == seq[0][0]:
            out[w] = {"c_star": 0.0, "note": "零成本已掛", "flips": flips, "pass_at_zero": False,
                      "fail_component": _fail_component(rows[first_fail][w])}
            continue
        lo = max(c for c, p in seq if c < first_fail and p)
        hi = first_fail
        while hi - lo > tol:
            mid = (lo + hi) / 2
            r = eval_fn(mid)
            if r[w]["pass"]:
                lo = mid
            else:
                hi = mid
        r_hi = eval_fn(hi)
        out[w] = {"c_star": (lo + hi) / 2, "last_pass": lo, "first_fail": hi, "flips": flips, "pass_at_zero": seq[0][1],
                  "fail_component": _fail_component(r_hi[w]),
                  "sharpe_at_fail": r_hi[w].get("sharpe"), "bh_sharpe": r_hi[w].get("bh_sharpe")}
    return out


def _fail_component(cell: dict) -> str:
    parts = []
    if not cell.get("sharpe_pass", True):
        parts.append("Sharpe")
    if not cell.get("calmar_pass", True):
        parts.append("Calmar")
    if not cell.get("dd_pass", True):
        parts.append("回撤")
    return "+".join(parts) or "?"


# ---------------- 執行層 ----------------

def fill_days(res) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    tr = res.trades
    return pd.DatetimeIndex(tr.entry_time).unique().sort_values(), pd.DatetimeIndex(tr.exit_time).unique().sort_values()


def _stats(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"n": 0, "mean": None, "median": None, "p75": None, "p90": None, "max": None}
    return {"n": int(len(x)), "mean": float(np.mean(x)), "median": float(np.median(x)), "p75": float(np.percentile(x, 75)),
            "p90": float(np.percentile(x, 90)), "max": float(np.max(x))}


def exec_daily(df: pd.DataFrame, buys: pd.DatetimeIndex, sells: pd.DatetimeIndex, since: str | None) -> dict:
    out = {}
    for side, times in (("buy", buys), ("sell", sells)):
        if since:
            times = times[times >= pd.Timestamp(since, tz="UTC")]
        sub = df.loc[times]
        o, h, l, c = (sub[k].to_numpy() for k in ("open", "high", "low", "close"))
        row = {"n": int(len(sub)), "oc_all": _stats((c / o - 1) if side == "buy" else (1 - c / o)), "by_delta": {}}
        for d in DELTAS:
            if side == "buy":
                filled = (l < o) if d == 0 else (l <= o * (1 - d))
                opp = c / o - 1
            else:
                filled = (h > o) if d == 0 else (h >= o * (1 + d))
                opp = 1 - c / o
            row["by_delta"][str(d)] = {"p": float(filled.mean()) if len(filled) else None, "n_filled": int(filled.sum()),
                                       "opp_unfilled": _stats(opp[~filled]), "opp_filled": _stats(opp[filled])}
        out[side] = row
    # 無條件（所有日）作對照
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    if since:
        m = df.index >= pd.Timestamp(since, tz="UTC")
        o, h, l, c = o[m], h[m], l[m], c[m]
    out["all_days"] = {"n": int(len(o)),
                       "buy_p": {str(d): float(((l < o) if d == 0 else (l <= o * (1 - d))).mean()) for d in DELTAS},
                       "sell_p": {str(d): float(((h > o) if d == 0 else (h >= o * (1 + d))).mean()) for d in DELTAS}}
    return out


def exec_15m(df15: pd.DataFrame, buys: pd.DatetimeIndex, sells: pd.DatetimeIndex, n_bars=(1, 2, 4, 8, 96)) -> dict:
    """永續 15m：成交日 00:00 UTC 起前 N 根內是否穿過開盤價；N 根後轉市價的漂移。"""
    out = {"n_bars": list(n_bars), "since": str(df15.index[0].date()), "until": str(df15.index[-1].date())}
    day_start = df15.index.normalize()
    groups = {k: v for k, v in df15.groupby(day_start)}
    for side, times in (("buy", buys), ("sell", sells)):
        days = [t for t in times if t in groups and len(groups[t]) >= 96]
        row = {"n": len(days), "by_n": {}}
        for N in n_bars:
            per_delta = {}
            for d in DELTAS:
                filled, drift = [], []
                for t in days:
                    g = groups[t]
                    o = float(g["open"].iloc[0])
                    lo = float(g["low"].iloc[:N].min())
                    hi = float(g["high"].iloc[:N].max())
                    px_n = float(g["close"].iloc[N - 1])
                    if side == "buy":
                        f = (lo < o) if d == 0 else (lo <= o * (1 - d))
                        dr = px_n / o - 1
                    else:
                        f = (hi > o) if d == 0 else (hi >= o * (1 + d))
                        dr = 1 - px_n / o
                    filled.append(f)
                    drift.append(dr)
                filled = np.array(filled)
                drift = np.array(drift)
                per_delta[str(d)] = {"p": float(filled.mean()) if len(filled) else None, "n_filled": int(filled.sum()),
                                     "drift_unfilled": _stats(drift[~filled]), "drift_all": _stats(drift)}
            row["by_n"][str(N)] = per_delta
        out[side] = row
    return out


def policy_cost(p: float, opp_unfilled_mean: float | None, maker: float, taker: float = TAKER, slip: float = MKT_SLIP) -> dict:
    """maker-first：p 成交付 maker、0 滑點；(1−p) 轉市價付 taker + 滑點 + 延遲漂移。"""
    base = p * maker + (1 - p) * (taker + slip)
    opp = (1 - p) * (opp_unfilled_mean or 0.0)
    return {"p": p, "no_drift": base, "drift_term": opp, "with_drift": base + opp}


# ---------------- 主流程 ----------------

def main() -> int:
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    mod = load_variant("ema50_plain")
    dfs = {s: dt.load_cached("data", s, "1d", DATA_SINCE) for s in SYMBOLS}
    if any(v is None for v in dfs.values()):
        print("缺現貨快取")
        return 2
    built = {s: mod.build(dfs[s], copy.deepcopy(mod.PARAMS)) for s in SYMBOLS}
    on = {s: built[s]["in_regime"] for s in SYMBOLS}
    warm = {s: int(built[s]["warmup"]) for s in SYMBOLS}
    idx = align(dfs)
    warm_port = max(int(idx.searchsorted(dfs[s].index[warm[s]])) for s in SYMBOLS)
    print(f"資料：" + ", ".join(f"{SHORT[s]} {dfs[s].index[0].date()}→{dfs[s].index[-1].date()} ({len(dfs[s])} 根, warmup {warm[s]})" for s in SYMBOLS))

    def rp_of(fee, slip, **kw):
        return RiskParams(commission_pct=fee, slippage_pct=slip, rebalance_threshold=REB_THRESHOLD, initial_capital=INITIAL, **kw)

    def rp_port(fee, slip):
        return rp_of(fee, slip)

    # ---- 1) 網格 ----
    grid_rows = []
    cache_single: dict = {}
    for fee in FEES:
        for slip in SLIPS:
            row = {"fee": fee, "slip": slip, "total": fee + slip}
            for s in SYMBOLS:
                _, wm = run_single(dfs[s], on[s], warm[s], rp_of(fee, slip))
                cache_single[(s, fee, slip)] = wm
                row[SHORT[s]] = {"gates": sum(int(w["pass"]) for w in wm.values()),
                                 "windows": {w: {"sharpe": x["sharpe"], "bh_sharpe": x["bh_sharpe"], "max_dd": x["max_dd"], "pass": x["pass"], "fail": None if x["pass"] else _fail_component(x)} for w, x in wm.items()}}
            _, pm = run_portfolio(dfs, on, idx, warm_port, rp_port(fee, slip))
            row["PORT"] = {"gates": sum(w["pass_count"] for w in pm.values()),
                           "windows": {w: {"sharpe": x["sharpe"], "bh_sharpe_nr": x["nr"]["bh_sharpe"], "bh_sharpe_mr": x["mr"]["bh_sharpe"], "max_dd": x["max_dd"],
                                           "pass_nr": x["nr"]["pass"], "pass_mr": x["mr"]["pass"]} for w, x in pm.items()}}
            row["ten_cells_single"] = row["BTC"]["gates"] + row["ETH"]["gates"]
            grid_rows.append(row)
            print(f"  網格 費 {fee:.3%} 滑 {slip:.2%} → BTC {row['BTC']['gates']}/5 ETH {row['ETH']['gates']}/5 組合 {row['PORT']['gates']}/10  ({time.time() - t_start:.0f}s)")

    # ---- 2) 損益兩平（每邊總成本 c 以 commission 施加）----
    be = {}
    for s in SYMBOLS:
        be[SHORT[s]] = breakeven(lambda c, s=s: run_single(dfs[s], on[s], warm[s], rp_of(c, 0.0))[1], WNAMES)
        print(f"  兩平 {SHORT[s]}: " + ", ".join(f"{w} {pct(v['c_star'], 3) if v.get('c_star') is not None else v.get('note')}" for w, v in be[SHORT[s]].items()) + f"  ({time.time() - t_start:.0f}s)")
    port_memo: dict = {}

    def port_at(c):
        key = round(c, 10)
        if key not in port_memo:
            port_memo[key] = run_portfolio(dfs, on, idx, warm_port, rp_port(c, 0.0))[1]
        return port_memo[key]

    for bk in ("nr", "mr"):
        def eval_port(c, bk=bk):
            pm = port_at(c)
            return {w: {"pass": x[bk]["pass"], "sharpe": x["sharpe"], "bh_sharpe": x[bk]["bh_sharpe"],
                        "sharpe_pass": x[bk]["sharpe_pass"], "calmar_pass": x[bk]["calmar_pass"], "dd_pass": x[bk]["dd_pass"]} for w, x in pm.items()}
        be[f"PORT_{bk}"] = breakeven(eval_port, WNAMES)
        print(f"  兩平 組合 vs {BENCH_NAMES[bk]}: " + ", ".join(f"{w} {pct(v['c_star'], 3) if v.get('c_star') is not None else v.get('note')}" for w, v in be[f'PORT_{bk}'].items()) + f"  ({time.time() - t_start:.0f}s)")

    # ---- 3) 換手與拖累 ----
    drag = {}
    ref_levels = {"taker+滑0.05%": LIVE_FEE + LIVE_SLIP, "BNB+滑0.05%": BNB + LIVE_SLIP, "maker+滑0.02%": MAKER + 0.0002}
    res0, wm0 = {}, {}
    for s in SYMBOLS:
        res0[s], wm0[s] = run_single(dfs[s], on[s], warm[s], rp_of(0.0, 0.0))
        _, wm_live = run_single(dfs[s], on[s], warm[s], rp_of(LIVE_FEE, LIVE_SLIP))
        to = turnover_single(res0[s], dfs[s], warm[s], wm0[s])
        drag[SHORT[s]] = {"_engine_full_turnover_per_year": float(res0[s].extra["notional_turnover_per_year"]), "_engine_full_fills": int(res0[s].extra["n_fills"])}
        for w in wm0[s]:
            drag[SHORT[s]][w] = {"turnover_per_year": to[w]["turnover_per_year"], "fills": to[w]["fills"], "fills_per_year": to[w]["fills_per_year"],
                                 "est_drag": {k: to[w]["turnover_per_year"] * v for k, v in ref_levels.items()},
                                 "measured_drag_live": (wm0[s][w]["log_growth"] - wm_live[w]["log_growth"]) / wm0[s][w]["years"],
                                 "zero_cost_sharpe": wm0[s][w]["sharpe"], "live_sharpe": wm_live[w]["sharpe"], "bh_sharpe": wm0[s][w]["bh_sharpe"],
                                 "zero_cost_cagr": wm0[s][w]["cagr"], "live_cagr": wm_live[w]["cagr"]}
    resp0, pm0 = run_portfolio(dfs, on, idx, warm_port, rp_port(0.0, 0.0))
    _, pm_live = run_portfolio(dfs, on, idx, warm_port, rp_port(LIVE_FEE, LIVE_SLIP))
    top = turnover_portfolio(resp0, idx, warm_port, pm0)
    drag["PORT"] = {w: {"turnover_per_year": top[w]["turnover_per_year"], "fills": top[w]["fills"], "fills_per_year": top[w]["fills_per_year"],
                        "est_drag": {k: top[w]["turnover_per_year"] * v for k, v in ref_levels.items()},
                        "measured_drag_live": (pm0[w]["log_growth"] - pm_live[w]["log_growth"]) / pm0[w]["years"],
                        "zero_cost_sharpe": pm0[w]["sharpe"], "live_sharpe": pm_live[w]["sharpe"], "bh_sharpe": pm0[w]["nr"]["bh_sharpe"],
                        "zero_cost_cagr": pm0[w]["cagr"], "live_cagr": pm_live[w]["cagr"]} for w in pm0}
    print(f"  換手/拖累完成 ({time.time() - t_start:.0f}s)")

    # ---- 4) 執行層 ----
    execu = {}
    for s in SYMBOLS:
        buys, sells = fill_days(res0[s])
        execu[SHORT[s]] = {"FULL": exec_daily(dfs[s], buys, sells, None), "OOS": exec_daily(dfs[s], buys, sells, "2023-01-01"),
                           "n_buy_days": int(len(buys)), "n_sell_days": int(len(sells))}
    df15 = dt.load_cached("data", ETH_15M, "15m", "2022-12-01")
    if df15 is not None:
        buys, sells = fill_days(res0["ETH/USDT"])
        execu["ETH"]["perp15m"] = exec_15m(df15, buys, sells)
    # 政策期望成本：以 OOS 成交日、買賣各半（實際買賣數幾乎相等）
    policy = {}
    for s in SYMBOLS:
        k = SHORT[s]
        policy[k] = {}
        for d in ("0.0002", "0.0005", "0.001"):
            pb = execu[k]["OOS"]["buy"]["by_delta"][d]
            ps = execu[k]["OOS"]["sell"]["by_delta"][d]
            p_avg = (pb["p"] + ps["p"]) / 2
            opp_close = ((1 - pb["p"]) * (pb["opp_unfilled"]["mean"] or 0) + (1 - ps["p"]) * (ps["opp_unfilled"]["mean"] or 0)) / 2  # 每邊平均的 (1-p)×E[opp]
            entry = {"p_buy": pb["p"], "p_sell": ps["p"], "p_avg": p_avg,
                     "opp_unfilled_mean_buy": pb["opp_unfilled"]["mean"], "opp_unfilled_mean_sell": ps["opp_unfilled"]["mean"],
                     "daily_bounds": {}}
            for mk_name, mk in (("maker0.02", MAKER), ("maker0.075", BNB)):
                base = p_avg * mk + (1 - p_avg) * (TAKER + MKT_SLIP)
                entry["daily_bounds"][mk_name] = {"formula_no_slip": p_avg * mk + (1 - p_avg) * TAKER,   # 任務原式 p×maker + (1−p)×taker
                                                  "fallback_immediate": base, "fallback_at_close": base + opp_close}
            policy[k][d] = entry
    if "perp15m" in execu["ETH"]:
        p15 = execu["ETH"]["perp15m"]
        policy["ETH"]["perp15m"] = {}
        for N in ("1", "2", "4", "8", "96"):
            policy["ETH"]["perp15m"][N] = {}
            for d in ("0.0002", "0.0005", "0.001"):
                b = p15["buy"]["by_n"][N][d]
                sl = p15["sell"]["by_n"][N][d]
                p_avg = (b["p"] + sl["p"]) / 2
                drift = ((1 - b["p"]) * (b["drift_unfilled"]["mean"] or 0) + (1 - sl["p"]) * (sl["drift_unfilled"]["mean"] or 0)) / 2
                cell = {"p_buy": b["p"], "p_sell": sl["p"], "p_avg": p_avg, "drift_unfilled_buy": b["drift_unfilled"]["mean"], "drift_unfilled_sell": sl["drift_unfilled"]["mean"],
                        "drift_term": drift}
                for mk_name, mk in (("maker0.02", MAKER), ("maker0.075", BNB)):
                    base = p_avg * mk + (1 - p_avg) * (TAKER + MKT_SLIP)
                    cell[mk_name] = {"no_drift": base, "with_drift": base + drift}
                policy["ETH"]["perp15m"][N][d] = cell

    # ---- 5) 用政策期望成本跑閘門（每邊成本以 commission 施加）----
    # 代表值：ETH 15m N=4（60 分）δ=0.05% 含漂移；BTC 沒有 15m，用日線 δ=0.05% 上下界；另加純 taker 與 BNB 對照
    scen = {"純taker 0.10%+滑0.05%": LIVE_FEE + LIVE_SLIP, "BNB 0.075%+滑0.05%": BNB + LIVE_SLIP}
    if "perp15m" in policy.get("ETH", {}):
        c4 = policy["ETH"]["perp15m"]["4"]["0.0005"]
        scen["maker-first 60分轉市價(ETH 15m δ0.05% 含漂移, maker0.02)"] = c4["maker0.02"]["with_drift"]
        scen["maker-first 60分轉市價(ETH 15m δ0.05% 含漂移, maker0.075)"] = c4["maker0.075"]["with_drift"]
    scen["maker-first 日線上界(δ0.05% 立即轉市價, maker0.02)"] = float(np.mean([policy[k]["0.0005"]["daily_bounds"]["maker0.02"]["fallback_immediate"] for k in ("BTC", "ETH")]))
    scen["maker-first 日線下界(δ0.05% 收盤才轉市價, maker0.02)"] = float(np.mean([policy[k]["0.0005"]["daily_bounds"]["maker0.02"]["fallback_at_close"] for k in ("BTC", "ETH")]))
    policy_runs = {}
    for name, c in scen.items():
        r = {"cost_per_side": c}
        for s in SYMBOLS:
            _, wm = run_single(dfs[s], on[s], warm[s], rp_of(c, 0.0))
            r[SHORT[s]] = {"gates": sum(int(w["pass"]) for w in wm.values()), "sharpe": {w: x["sharpe"] for w, x in wm.items()}, "pass": {w: x["pass"] for w, x in wm.items()}}
        _, pm = run_portfolio(dfs, on, idx, warm_port, rp_port(c, 0.0))
        r["PORT"] = {"gates": sum(x["pass_count"] for x in pm.values()), "sharpe": {w: x["sharpe"] for w, x in pm.items()}, "pass_nr": {w: x["nr"]["pass"] for w, x in pm.items()}}
        policy_runs[name] = r
        print(f"  政策 {name} c={c:.4%}: BTC {r['BTC']['gates']}/5 ETH {r['ETH']['gates']}/5 組合 {r['PORT']['gates']}/10")

    report = {
        "key": KEY, "generated": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "elapsed_s": round(time.time() - t_start, 1),
        "assumptions": {
            "data": {SHORT[s]: {"file": dt.cache_path("data", s, "1d"), "start": str(dfs[s].index[0].date()), "end": str(dfs[s].index[-1].date()), "bars": int(len(dfs[s])), "warmup_idx": warm[s]} for s in SYMBOLS},
            "perp15m": None if df15 is None else {"file": dt.cache_path("data", ETH_15M, "15m"), "start": str(df15.index[0].date()), "end": str(df15.index[-1].date()), "bars": int(len(df15))},
            "strategy": "experiments/variants/ema50_plain.py（close > EMA50 滿倉否則空手；收盤判定次日開盤成交）",
            "windows": {w: [a, b] for w, a, b in WINDOWS}, "gate": "Sharpe > 持有 且 Calmar > 持有 且 最大回撤 < 持有（abgt.benchmark.gate mode=risk）",
            "portfolio": {"weights": WEIGHTS, "rebalance_threshold": REB_THRESHOLD, "warmup_idx": warm_port, "benches": BENCH_NAMES},
            "fee_grid": FEES, "slip_grid": SLIPS, "maker": MAKER, "bnb": BNB, "taker": TAKER, "market_slip": MKT_SLIP, "live": {"fee": LIVE_FEE, "slip": LIVE_SLIP},
            "breakeven_method": "每邊總成本 c 以 commission_pct=c、slippage_pct=0 施加（持有基準只付一次 c）；粗掃 0→1.2% 步 0.02%，再二分到 0.001%",
            "turnover_method": "格內成交名目(買+賣)/格內平均權益/格內年數；估計拖累 = 換手 × 每邊成本；實測拖累 = (零成本 log 成長 − 費0.10%+滑0.05% log 成長)/年",
            "execution_method": "買單掛開盤價 O 視為成交若 low ≤ O(1−δ)（δ=0 用嚴格 low<O），賣單 high ≥ O(1+δ)；延遲成本 = 未成交子集的同日 open→close；15m 版用前 N 根的 low/high 與第 N 根收盤",
            "deltas": DELTAS,
        },
        "grid": grid_rows, "breakeven": be, "drag": drag, "execution": execu, "policy": policy, "policy_runs": policy_runs,
    }
    with open(os.path.join(OUT_DIR, f"{KEY}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)
    md = render_md(report)
    with open(os.path.join(OUT_DIR, f"{KEY}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)
    print(f"\n→ {os.path.join(OUT_DIR, KEY)}.json / .md（{time.time() - t_start:.0f}s）")
    return 0


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


# ---------------- Markdown ----------------

def render_md(rep: dict) -> str:
    L = []
    a = rep["assumptions"]
    L.append(f"# 槓桿：成本（ema50_plain，現貨日線）\n")
    L.append(f"產生 {rep['generated']}｜資料 " + "、".join(f"{k} {v['start']}→{v['end']}" for k, v in a["data"].items()) + f"｜閘門 = {a['gate']}｜重跑 `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/levers/costs.py`\n")

    # 1 網格
    L.append("## 1. 手續費 × 滑點網格（每格 Sharpe，✓/✗ = 該格閘門）\n")
    L.append("| 費/邊 | 滑/邊 | 總/邊 | BTC 五格 | BTC OOS | BTC OOS2 | ETH 五格 | ETH OOS | ETH OOS2 | 單標的十格 | 組合十格 | 組合 OOS | 組合 OOS2 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rep["grid"]:
        def cell(sym, w):
            x = r[sym]["windows"][w]
            return f"{x['sharpe']:.2f}{'✓' if x['pass'] else '✗'}"
        def pcell(w):
            x = r["PORT"]["windows"][w]
            return f"{x['sharpe']:.2f}{'✓' if x['pass_nr'] else '✗'}{'✓' if x['pass_mr'] else '✗'}"
        L.append(f"| {pct(r['fee'], 3)} | {pct(r['slip'], 2)} | {pct(r['total'], 3)} | {r['BTC']['gates']}/5 | {cell('BTC', 'OOS')} | {cell('BTC', 'OOS2')} | {r['ETH']['gates']}/5 | {cell('ETH', 'OOS')} | {cell('ETH', 'OOS2')} | {r['ten_cells_single']}/10 | {r['PORT']['gates']}/10 | {pcell('OOS')} | {pcell('OOS2')} |")
    g0 = rep["grid"][0]
    L.append("")
    L.append("同期持有 Sharpe（費 0.02% 格，持有只付一次進場費）：BTC " + " / ".join(f"{w} {g0['BTC']['windows'][w]['bh_sharpe']:.2f}" for w in WNAMES) + "；ETH " + " / ".join(f"{w} {g0['ETH']['windows'][w]['bh_sharpe']:.2f}" for w in WNAMES) + "；組合不再平衡 " + " / ".join(f"{w} {g0['PORT']['windows'][w]['bh_sharpe_nr']:.2f}" for w in WNAMES) + "；組合月再平衡 " + " / ".join(f"{w} {g0['PORT']['windows'][w]['bh_sharpe_mr']:.2f}" for w in WNAMES) + "\n")
    L.append("組合欄兩個符號 = 對「持有不再平衡」/「持有月再平衡」。\n")

    # 2 兩平
    L.append("## 2. 損益兩平每邊總成本 c*（閘門由過翻掛的成本；commission=c、slip=0 施加）\n")
    L.append("| 標的 | " + " | ".join(WNAMES) + " |")
    L.append("|---|" + "---|" * len(WNAMES))
    for k, v in rep["breakeven"].items():
        label = {"PORT_nr": "組合 vs 不再平衡", "PORT_mr": "組合 vs 月再平衡"}.get(k, k)
        cells = []
        for w in WNAMES:
            x = v.get(w)
            if x is None:
                cells.append("—")
            elif x.get("c_star") is None:
                cells.append(x.get("note", "—"))
            else:
                cells.append(f"{pct(x['c_star'], 3)}（{x.get('fail_component', '?')}）" + (f" 翻轉{x['flips']}" if x.get("flips", 0) > 1 else ""))
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("讀法：c* 是每邊總成本（手續費 + 滑點）；上線口徑 taker 0.10% + 滑 0.05% = 0.15%/邊，BNB 抵扣 0.125%，maker-first 見第 4 節。括號 = 掛的是哪個條件。「翻轉」>1 表示掃描中閘門非單調（成本略高反而過）。\n")

    # 3 換手拖累
    L.append("## 3. 名目換手與成本拖累（%/年，log 口徑）\n")
    L.append("| 標的 | 區間 | 換手×/年 | 成交/年 | 估計拖累 @0.15% | @0.125% | @0.04%(純maker) | 實測拖累 @費0.10%+滑0.05% | Sharpe 零成本→0.15% | 持有 Sharpe | CAGR 零成本→0.15% |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, v in rep["drag"].items():
        for w, x in v.items():
            if w.startswith("_"):
                continue
            e = x["est_drag"]
            L.append(f"| {k} | {w} | {x['turnover_per_year']:.1f} | {x['fills_per_year']:.1f} | {pct(e['taker+滑0.05%'])} | {pct(e['BNB+滑0.05%'])} | {pct(e['maker+滑0.02%'])} | {pct(x['measured_drag_live'])} | {x['zero_cost_sharpe']:.2f}→{x['live_sharpe']:.2f} | {x['bh_sharpe']:.2f} | {pct(x['zero_cost_cagr'], 1)}→{pct(x['live_cagr'], 1)} |")
    L.append("")

    # 4 執行層
    L.append("## 4. 執行層：先掛限價（開盤價）當 maker、未成交轉市價\n")
    L.append("### 4.1 日線估計：成交日「開盤價限價當日成交」機率 p 與未成交子集的 open→close 延遲成本（OOS 2023→ 成交日）\n")
    L.append("| 標的 | 邊 | n | δ=0(嚴格穿過) | δ=0.02% | δ=0.05% | δ=0.10% | δ=0.20% | 未成交 open→close 均值 @δ0.05% (n) | @δ0.10% (n) | 全部成交日 open→close 均值 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in ("BTC", "ETH"):
        for side in ("buy", "sell"):
            x = rep["execution"][k]["OOS"][side]
            bd = x["by_delta"]
            L.append(f"| {k} | {'買(進場)' if side == 'buy' else '賣(出場)'} | {x['n']} | " + " | ".join(pct(bd[str(d)]['p'], 1) for d in DELTAS) +
                     f" | {pct(bd['0.0005']['opp_unfilled']['mean'])} ({bd['0.0005']['opp_unfilled']['n']}) | {pct(bd['0.001']['opp_unfilled']['mean'])} ({bd['0.001']['opp_unfilled']['n']}) | {pct(x['oc_all']['mean'])} |")
    L.append("")
    L.append("無條件（所有日）對照 p@δ0.05%：" + "；".join(f"{k} 買 {pct(rep['execution'][k]['OOS']['all_days']['buy_p']['0.0005'], 1)} / 賣 {pct(rep['execution'][k]['OOS']['all_days']['sell_p']['0.0005'], 1)}" for k in ("BTC", "ETH")) + "\n")
    L.append("FULL 期（2020→）成交日 p@δ0.05%：" + "；".join(f"{k} 買 {pct(rep['execution'][k]['FULL']['buy']['by_delta']['0.0005']['p'], 1)} (n {rep['execution'][k]['FULL']['buy']['n']}) / 賣 {pct(rep['execution'][k]['FULL']['sell']['by_delta']['0.0005']['p'], 1)} (n {rep['execution'][k]['FULL']['sell']['n']})" for k in ("BTC", "ETH")) + "\n")

    if "perp15m" in rep["execution"].get("ETH", {}):
        p15 = rep["execution"]["ETH"]["perp15m"]
        L.append(f"### 4.2 ETH 永續 15m 交叉檢查（{p15['since']}→{p15['until']}，成交日 00:00 UTC 起前 N 根內穿過開盤價；買 n={p15['buy']['n']}、賣 n={p15['sell']['n']}）\n")
        L.append("| N 根(分) | 邊 | p@δ0 | p@δ0.02% | p@δ0.05% | p@δ0.10% | 未成交漂移均值 @δ0.05% (n) | 未成交漂移 p90 @δ0.05% | 全部成交日 N 根後漂移均值 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for N in p15["n_bars"]:
            for side in ("buy", "sell"):
                b = p15[side]["by_n"][str(N)]
                L.append(f"| {N} ({N * 15}) | {'買' if side == 'buy' else '賣'} | {pct(b['0.0']['p'], 1)} | {pct(b['0.0002']['p'], 1)} | {pct(b['0.0005']['p'], 1)} | {pct(b['0.001']['p'], 1)} | {pct(b['0.0005']['drift_unfilled']['mean'])} ({b['0.0005']['drift_unfilled']['n']}) | {pct(b['0.0005']['drift_unfilled']['p90'])} | {pct(b['0.0005']['drift_all']['mean'])} |")
        L.append("")

    L.append("### 4.3 政策期望每邊成本 = p×maker + (1−p)×(taker 0.10% + 滑 0.05%) + (1−p)×E[延遲成本|未成交]\n")
    L.append("| 來源 | p(買/賣/均) | 原式 p×0.02%+(1−p)×0.10% | maker 0.02%：不計漂移(含市價滑點) | 含漂移 | maker 0.075%：不計漂移 | 含漂移 |")
    L.append("|---|---|---|---|---|---|---|")
    for k in ("BTC", "ETH"):
        for d in ("0.0002", "0.0005", "0.001"):
            x = rep["policy"][k][d]
            db = x["daily_bounds"]
            L.append(f"| {k} 日線 δ={pct(float(d), 2)}（漂移 = 等到收盤） | {pct(x['p_buy'], 0)}/{pct(x['p_sell'], 0)}/{pct(x['p_avg'], 0)} | {pct(db['maker0.02']['formula_no_slip'], 3)} | {pct(db['maker0.02']['fallback_immediate'], 3)} | {pct(db['maker0.02']['fallback_at_close'], 3)} | {pct(db['maker0.075']['fallback_immediate'], 3)} | {pct(db['maker0.075']['fallback_at_close'], 3)} |")
    if "perp15m" in rep["policy"].get("ETH", {}):
        for N in ("1", "2", "4", "8", "96"):
            for d in ("0.0002", "0.0005", "0.001"):
                x = rep["policy"]["ETH"]["perp15m"][N][d]
                L.append(f"| ETH 15m N={N}（{int(N) * 15} 分後轉市價）δ={pct(float(d), 2)} | {pct(x['p_buy'], 0)}/{pct(x['p_sell'], 0)}/{pct(x['p_avg'], 0)} | {pct(x['p_avg'] * MAKER + (1 - x['p_avg']) * TAKER, 3)} | {pct(x['maker0.02']['no_drift'], 3)} | {pct(x['maker0.02']['with_drift'], 3)} | {pct(x['maker0.075']['no_drift'], 3)} | {pct(x['maker0.075']['with_drift'], 3)} |")
    L.append("")
    L.append("### 4.4 用政策期望成本跑閘門（每邊成本以 commission 施加）\n")
    L.append("| 情境 | 每邊成本 | BTC 五格 | BTC OOS/OOS2 | ETH 五格 | ETH OOS/OOS2 | 組合十格 | 組合 OOS/OOS2 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, r in rep["policy_runs"].items():
        L.append(f"| {name} | {pct(r['cost_per_side'], 3)} | {r['BTC']['gates']}/5 | {r['BTC']['sharpe']['OOS']:.2f}{'✓' if r['BTC']['pass']['OOS'] else '✗'} / {r['BTC']['sharpe']['OOS2']:.2f}{'✓' if r['BTC']['pass']['OOS2'] else '✗'} | {r['ETH']['gates']}/5 | {r['ETH']['sharpe']['OOS']:.2f}{'✓' if r['ETH']['pass']['OOS'] else '✗'} / {r['ETH']['sharpe']['OOS2']:.2f}{'✓' if r['ETH']['pass']['OOS2'] else '✗'} | {r['PORT']['gates']}/10 | {r['PORT']['sharpe']['OOS']:.2f}{'✓' if r['PORT']['pass_nr']['OOS'] else '✗'} / {r['PORT']['sharpe']['OOS2']:.2f}{'✓' if r['PORT']['pass_nr']['OOS2'] else '✗'} |")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
