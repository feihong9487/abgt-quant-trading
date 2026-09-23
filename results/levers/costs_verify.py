"""獨立重算 experiments/levers/costs.py 的關鍵數字（懷疑者驗證）。

不 import abgt.*：EMA、regime、單標的 / 組合模擬、持有基準、Sharpe / 回撤 / Calmar、閘門全部自寫（pandas / numpy）。

重跑：
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe results/levers/costs_verify.py [--no-net]
輸出：
    results/levers/costs_verify.json（並在 stdout 印比對摘要）

口徑：
- 資料：data/BTCUSDT_1d.csv、data/ETHUSDT_1d.csv（Binance 現貨日線），2019-09-01 起
- EMA50：前 50 根 SMA 當種子（index 49），之後 alpha = 2/51 遞迴；regime = close > EMA50；warmup = 49
- 收盤決策、次日開盤成交；買價 open×(1+slip)、賣價 open×(1−slip)，兩邊都付 fee×成交名目
- 單標的：買量以現金為上限（qty = cash / (px(1+fee))）；整段從 warmup 跑一次，視窗內切片後重新歸一
- 組合：共用權益 50/50，趨勢內目標 = 0.5，|目標 − 目前| > 0.15×0.5 調倉；同根先賣後買、買量以現金為上限
- 持有基準：單標的 = 視窗起點收盤買入付一次 fee；組合四種：
    fresh_nr：視窗起點重新 50/50 買入、不再平衡（正確口徑）
    fresh_mr：視窗起點 50/50、每月第一根再平衡（付 fee×|調整名目|）
    drift_nr / drift_mr：從組合 warmup（2019-10）起算再切片（= 分析者用的 abgt.portfolio.buy_and_hold_portfolio 口徑）
- 閘門：Sharpe > 持有 且 Calmar > 持有 且 最大回撤比持有淺
- 損益兩平 c*：每邊總成本 c 以 fee=c、slip=0 施加（持有也付一次 c），粗掃 0→1.2% 步 0.01%，在 [最後過, 第一掛] 間二分到 1e-6
- 限價成交率：只看策略實際成交日（OOS = 2023-01-01 起），買 low ≤ open(1−δ)、賣 high ≥ open(1+δ)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results", "levers", "costs_verify.json")
ANALYST = os.path.join(ROOT, "results", "levers", "costs.json")
SINCE = "2019-09-01"
WINDOWS = [("IS", "2020-01-01", "2022-12-31"), ("OOS", "2023-01-01", None), ("FULL", "2020-01-01", None),
           ("IS2", "2020-07-01", "2023-12-31"), ("OOS2", "2024-01-01", None)]
WN = [w[0] for w in WINDOWS]
EMA_N = 50
THR = 0.15
W = (0.5, 0.5)
COST_LEVELS = [0.0004, 0.00125, 0.0015, 0.002, 0.003]   # 每邊總成本
DELTAS = [0.0, 0.0002, 0.0005, 0.001, 0.002]


# ---------------- 資料與訊號 ----------------

def load(fname: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA, fname))
    df = df[df["timestamp"] >= int(pd.Timestamp(SINCE, tz="UTC").timestamp() * 1000)]
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df[["open", "high", "low", "close"]].astype("float64")


def ema_sma_seed(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    a = 2.0 / (n + 1)
    out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def regime(df: pd.DataFrame) -> tuple[np.ndarray, int]:
    c = df["close"].to_numpy()
    e = ema_sma_seed(c, EMA_N)
    on = np.where(np.isnan(e), False, c > e)
    return on, EMA_N - 1


# ---------------- 模擬 ----------------

def sim_single(o, c, on, w0, fee, slip):
    n = len(c)
    cash, q = 1.0, 0.0
    pend = None
    eq = np.full(n, np.nan)
    notional = np.zeros(n)
    side = np.zeros(n, dtype=int)
    for i in range(w0, n):
        if pend is not None and pend[0] == i:
            tgt = pend[1]
            pend = None
            if tgt > 0 and q == 0:
                px = o[i] * (1 + slip)
                q = cash / (px * (1 + fee))
                notional[i] += q * px
                cash -= q * px * (1 + fee)
                side[i] = 1
            elif tgt == 0 and q > 0:
                px = o[i] * (1 - slip)
                cash += q * px * (1 - fee)
                notional[i] += q * px
                q = 0.0
                side[i] = -1
        eq[i] = cash + q * c[i]
        ref = 1.0 if q > 0 else 0.0
        if on[i] and ref == 0:
            pend = (i + 1, 1.0)
        elif (not on[i]) and ref > 0:
            pend = (i + 1, 0.0)
    return eq, notional, side


def sim_port(O, C, ON, w0, fee, slip):
    K = len(O)
    n = len(C[0])
    cash = 1.0
    q = [0.0] * K
    pend = [None] * K
    eq = np.full(n, np.nan)
    notional = np.zeros(n)
    fills = []                                # (i, k, side)
    for i in range(w0, n):
        todo = [k for k in range(K) if pend[k] is not None and pend[k][0] == i]
        e0 = cash + sum(q[j] * O[j][i] for j in range(K))
        todo.sort(key=lambda k: pend[k][1] * e0 / O[k][i] - q[k] > 0)       # 賣先
        for k in todo:
            tgt = pend[k][1]
            pend[k] = None
            eo = cash + sum(q[j] * O[j][i] for j in range(K))
            d = tgt * eo / O[k][i] - q[k]
            if d > 0:
                d = min(d, max(0.0, cash) / (O[k][i] * (1 + slip) * (1 + fee)))
            if abs(d) * O[k][i] < 1e-6 * eo:
                continue
            if d > 0:
                px = O[k][i] * (1 + slip)
                cash -= d * px * (1 + fee)
            else:
                px = O[k][i] * (1 - slip)
                cash += -d * px * (1 - fee)
            q[k] += d
            if q[k] <= 1e-12:
                q[k] = 0.0
            notional[i] += abs(d) * px
            fills.append((i, k, 1 if d > 0 else -1))
        eqv = cash + sum(q[k] * C[k][i] for k in range(K))
        eq[i] = eqv
        for k in range(K):
            tgt = W[k] if ON[k][i] else 0.0
            cur = q[k] * C[k][i] / eqv if eqv > 0 else 0.0
            if tgt == 0 and cur > 0:
                pend[k] = (i + 1, 0.0)
            elif tgt > 0 and cur == 0:
                pend[k] = (i + 1, tgt)
            elif tgt > 0 and cur > 0 and abs(tgt - cur) > THR * W[k]:
                pend[k] = (i + 1, tgt)
    return eq, notional, fills


def bh_port(C, idx, s0, fee, monthly: bool):
    n = len(C[0])
    K = len(C)
    eq = np.full(n, np.nan)
    q = [W[k] * (1 - fee) / C[k][s0] for k in range(K)]
    last_m = idx[s0].month
    for i in range(s0, n):
        v = sum(q[k] * C[k][i] for k in range(K))
        if monthly and idx[i].month != last_m:
            last_m = idx[i].month
            cost = sum(abs(v * W[k] - q[k] * C[k][i]) for k in range(K)) * fee
            v -= cost
            q = [v * W[k] / C[k][i] for k in range(K)]
        eq[i] = v
    return eq


# ---------------- 指標 ----------------

def metrics(eq: np.ndarray, idx: pd.DatetimeIndex) -> dict:
    e = np.asarray(eq, dtype="float64")
    e = e / e[0]
    r = e[1:] / e[:-1] - 1
    sh = float(r.mean() / r.std(ddof=1) * math.sqrt(365)) if len(r) > 2 and r.std(ddof=1) > 0 else float("nan")
    peak = np.maximum.accumulate(e)
    mdd = float((e / peak - 1).min())
    yrs = (idx[-1] - idx[0]).total_seconds() / (365.25 * 86400)
    cagr = float(e[-1] ** (1 / yrs) - 1)
    return {"sharpe": sh, "max_dd": mdd, "cagr": cagr, "calmar": cagr / abs(mdd) if mdd < 0 else float("nan"),
            "years": yrs, "log_growth": float(math.log(e[-1]))}


def gate(s: dict, b: dict) -> dict:
    sp, cp, dp = s["sharpe"] > b["sharpe"], s["calmar"] > b["calmar"], s["max_dd"] > b["max_dd"]
    return {"pass": bool(sp and cp and dp), "sharpe": bool(sp), "calmar": bool(cp), "dd": bool(dp)}


def win_slices(idx: pd.DatetimeIndex, w0: int):
    out = {}
    for name, a, b in WINDOWS:
        s0 = max(w0, int(idx.searchsorted(pd.Timestamp(a, tz="UTC"))))
        s1 = int(idx.searchsorted(pd.Timestamp(b, tz="UTC"), side="right")) if b else len(idx)
        out[name] = (s0, s1)
    return out


# ---------------- 評估 ----------------

class Ctx:
    def __init__(self):
        self.dfs = {"BTC": load("BTCUSDT_1d.csv"), "ETH": load("ETHUSDT_1d.csv")}
        self.idx = self.dfs["BTC"].index.intersection(self.dfs["ETH"].index)
        self.reg = {}
        for k, df in self.dfs.items():
            self.reg[k] = regime(df)
        self.al = {k: df.reindex(self.idx) for k, df in self.dfs.items()}
        self.on_al = {k: pd.Series(self.reg[k][0], index=self.dfs[k].index).reindex(self.idx).fillna(False).to_numpy(bool) for k in self.dfs}
        self.w_port = max(int(self.idx.searchsorted(self.dfs[k].index[self.reg[k][1]])) for k in self.dfs)
        self.ws_single = {k: win_slices(self.dfs[k].index, self.reg[k][1]) for k in self.dfs}
        self.ws_port = win_slices(self.idx, self.w_port)

    def single(self, k, fee, slip, bh_fee=None):
        df = self.dfs[k]
        o, c = df["open"].to_numpy(), df["close"].to_numpy()
        on, w0 = self.reg[k]
        eq, notional, side = sim_single(o, c, on, w0, fee, slip)
        bh_fee = fee if bh_fee is None else bh_fee
        out = {}
        for w, (s0, s1) in self.ws_single[k].items():
            ix = df.index[s0:s1]
            m = metrics(eq[s0:s1], ix)
            b = metrics(c[s0:s1] * (1 - bh_fee) / c[s0], ix)
            out[w] = {**m, "bh": b, "gate": gate(m, b),
                      "turnover": float(notional[s0:s1].sum() / np.mean(eq[s0:s1]) / m["years"]),
                      "fills_per_year": float((side[s0:s1] != 0).sum() / m["years"])}
        return out, side

    def port(self, fee, slip, bh_fee=None, benches=("fresh_nr", "fresh_mr", "drift_nr", "drift_mr")):
        O = [self.al[k]["open"].to_numpy() for k in ("BTC", "ETH")]
        C = [self.al[k]["close"].to_numpy() for k in ("BTC", "ETH")]
        ON = [self.on_al[k] for k in ("BTC", "ETH")]
        eq, notional, fills = sim_port(O, C, ON, self.w_port, fee, slip)
        bh_fee = fee if bh_fee is None else bh_fee
        drift = {}
        if "drift_nr" in benches:
            drift["drift_nr"] = bh_port(C, self.idx, self.w_port, bh_fee, False)
        if "drift_mr" in benches:
            drift["drift_mr"] = bh_port(C, self.idx, self.w_port, bh_fee, True)
        out = {}
        fill_i = np.array([f[0] for f in fills])
        for w, (s0, s1) in self.ws_port.items():
            ix = self.idx[s0:s1]
            m = metrics(eq[s0:s1], ix)
            row = {**m, "turnover": float(notional[s0:s1].sum() / np.mean(eq[s0:s1]) / m["years"]),
                   "fills_per_year": float(((fill_i >= s0) & (fill_i < s1)).sum() / m["years"])}
            for bk in benches:
                if bk.startswith("fresh"):
                    beq = bh_port(C, self.idx, s0, bh_fee, bk.endswith("mr"))[s0:s1]
                else:
                    beq = drift[bk][s0:s1]
                b = metrics(beq, ix)
                row[bk] = {"bh": b, "gate": gate(m, b)}
            out[w] = row
        return out, fills


def breakeven(pass_fn, step=0.0001, cmax=0.012, tol=1e-6):
    grid = [round(i * step, 8) for i in range(int(round(cmax / step)) + 1)]
    seq = [(c, pass_fn(c)) for c in grid]
    flips = sum(1 for (a, pa), (b, pb) in zip(seq, seq[1:]) if pa != pb)
    ff = next((c for c, p in seq if not p), None)
    if ff is None:
        return {"c_star": None, "note": "全過", "flips": flips}
    if ff == 0:
        return {"c_star": 0.0, "note": "零成本已掛", "flips": flips}
    lo, hi = max(c for c, p in seq if c < ff and p), ff
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if pass_fn(mid):
            lo = mid
        else:
            hi = mid
    return {"c_star": (lo + hi) / 2, "flips": flips}


def fill_rates(df: pd.DataFrame, days: pd.DatetimeIndex, side: str) -> dict:
    sub = df.loc[days]
    o, h, l, c = (sub[x].to_numpy() for x in ("open", "high", "low", "close"))
    out = {"n": int(len(sub))}
    for d in DELTAS:
        if side == "buy":
            f = (l < o) if d == 0 else (l <= o * (1 - d))
        else:
            f = (h > o) if d == 0 else (h >= o * (1 + d))
        out[str(d)] = float(f.mean()) if len(f) else None
    return out


def fetch_spot_15m(sym: str, day: pd.Timestamp, nbars: int = 96):
    url = f"https://api.binance.com/api/v3/klines?symbol={sym}&interval=15m&startTime={int(day.timestamp() * 1000)}&limit={nbars}"
    with urllib.request.urlopen(url, timeout=15) as r:
        rows = json.loads(r.read())
    return np.array([[float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in rows])


def intraday_fill(bars_by_day: dict, days, side, N, d):
    """回傳 (p, n, 未成交子集第 N 根收盤相對開盤的不利漂移均值)。"""
    res, drift = [], []
    for t in days:
        b = bars_by_day.get(t)
        if b is None or len(b) < N:
            continue
        o = b[0, 0]
        if side == "buy":
            f = b[:N, 2].min() <= o * (1 - d) if d > 0 else b[:N, 2].min() < o
            dr = b[N - 1, 3] / o - 1
        else:
            f = b[:N, 1].max() >= o * (1 + d) if d > 0 else b[:N, 1].max() > o
            dr = 1 - b[N - 1, 3] / o
        res.append(bool(f))
        if not f:
            drift.append(dr)
    return (float(np.mean(res)) if res else None), len(res), (float(np.mean(drift)) if drift else 0.0)


def policy_cost(intr_k: dict, N=4, d=0.0005, maker=0.0002, fallback=0.0015) -> float:
    """每邊期望成本（買賣平均）= p×maker + (1−p)×(fallback + E[漂移|未成交])。"""
    out = []
    for s in ("buy", "sell"):
        p, _, dr = intr_k[f"{s}_N{N}_d{d}"]
        out.append(p * maker + (1 - p) * (fallback + dr))
    return float(np.mean(out))


# ---------------- 主流程 ----------------

def main() -> int:
    t0 = time.time()
    no_net = "--no-net" in sys.argv
    ctx = Ctx()
    ana = json.load(open(ANALYST, encoding="utf-8")) if os.path.exists(ANALYST) else None
    rep = {"generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
           "data_end": {k: str(df.index[-1].date()) for k, df in ctx.dfs.items()},
           "warmup": {"BTC": str(ctx.dfs["BTC"].index[ctx.reg["BTC"][1]].date()), "ETH": str(ctx.dfs["ETH"].index[ctx.reg["ETH"][1]].date()),
                      "PORT": str(ctx.idx[ctx.w_port].date())}}

    # (a) 成本網格（fee = 總成本 − 0.05% 滑點，0.04% 格用 0.02+0.02、0.30% 用 0.10+0.20，與分析者同組合）
    split = {0.0004: (0.0002, 0.0002), 0.00125: (0.00075, 0.0005), 0.0015: (0.001, 0.0005), 0.002: (0.001, 0.001), 0.003: (0.001, 0.002)}
    grid = {}
    for tot in COST_LEVELS:
        fee, slip = split[tot]
        row = {"fee": fee, "slip": slip}
        for k in ("BTC", "ETH"):
            r, _ = ctx.single(k, fee, slip)
            row[k] = {w: {"sharpe": r[w]["sharpe"], "bh_sharpe": r[w]["bh"]["sharpe"], "pass": r[w]["gate"]["pass"], "cagr": r[w]["cagr"]} for w in WN}
        p, _ = ctx.port(fee, slip)
        row["PORT"] = {w: {"sharpe": p[w]["sharpe"], "cagr": p[w]["cagr"], "max_dd": p[w]["max_dd"],
                           **{bk: {"bh_sharpe": p[w][bk]["bh"]["sharpe"], "bh_calmar": p[w][bk]["bh"]["calmar"], "bh_max_dd": p[w][bk]["bh"]["max_dd"],
                                   "pass": p[w][bk]["gate"]["pass"]} for bk in ("fresh_nr", "fresh_mr", "drift_nr", "drift_mr")},
                           "calmar": p[w]["calmar"]} for w in WN}
        grid[f"{tot:.5f}"] = row
    rep["grid"] = grid
    # 補：政策成本點的組合 Sharpe
    rep["policy_points"] = {}
    for c in (0.0, 0.0002, 0.00044, 0.00096, 0.00125, 0.0015):
        p, _ = ctx.port(c, 0.0, benches=("fresh_mr",))
        rep["policy_points"][f"{c:.5f}"] = {w: {"sharpe": p[w]["sharpe"], "cagr": p[w]["cagr"]} for w in ("OOS", "OOS2", "FULL")}
    print(f"網格完成 {time.time() - t0:.0f}s")

    # (b) 損益兩平
    be = {}
    for k in ("BTC", "ETH"):
        memo = {}

        def run(c, k=k, memo=memo):
            key = round(c, 9)
            if key not in memo:
                memo[key] = ctx.single(k, c, 0.0)[0]
            return memo[key]
        be[k] = {w: breakeven(lambda c, w=w: run(c)[w]["gate"]["pass"]) for w in ("OOS", "OOS2")}
        for w in ("OOS", "OOS2"):
            be[k][w]["fail_at"] = {kk: v for kk, v in run(be[k][w]["c_star"] + 2e-6)[w]["gate"].items()} if be[k][w].get("c_star") else None
    pmemo = {}

    def prun(c):
        key = round(c, 9)
        if key not in pmemo:
            pmemo[key] = ctx.port(c, 0.0)[0]
        return pmemo[key]
    for bk in ("fresh_nr", "fresh_mr", "drift_nr", "drift_mr"):
        be[f"PORT_{bk}"] = {w: breakeven(lambda c, w=w, bk=bk: prun(c)[w][bk]["gate"]["pass"], step=0.0002) for w in ("OOS", "OOS2")}
    rep["breakeven"] = be
    print(f"兩平完成 {time.time() - t0:.0f}s")

    # (c) 換手與拖累
    drag = {}
    for k in ("BTC", "ETH"):
        z, _ = ctx.single(k, 0.0, 0.0)
        live, _ = ctx.single(k, 0.001, 0.0005)
        drag[k] = {w: {"turnover": z[w]["turnover"], "fills_per_year": z[w]["fills_per_year"], "est_drag_0.15": z[w]["turnover"] * 0.0015,
                       "measured_drag": (z[w]["log_growth"] - live[w]["log_growth"]) / z[w]["years"],
                       "sharpe_0": z[w]["sharpe"], "sharpe_live": live[w]["sharpe"]} for w in WN}
    z, _ = ctx.port(0.0, 0.0, benches=())
    live, _ = ctx.port(0.001, 0.0005, benches=())
    drag["PORT"] = {w: {"turnover": z[w]["turnover"], "fills_per_year": z[w]["fills_per_year"], "est_drag_0.15": z[w]["turnover"] * 0.0015,
                        "measured_drag": (z[w]["log_growth"] - live[w]["log_growth"]) / z[w]["years"],
                        "sharpe_0": z[w]["sharpe"], "sharpe_live": live[w]["sharpe"]} for w in WN}
    rep["drag"] = drag

    # (d) 開盤價限價當日成交率（策略實際成交日）
    oos0 = pd.Timestamp("2023-01-01", tz="UTC")
    fr, days_by = {}, {}
    for k in ("BTC", "ETH"):
        df = ctx.dfs[k]
        _, side = ctx.single(k, 0.0, 0.0)
        bd = df.index[side == 1]
        sd = df.index[side == -1]
        days_by[k] = {"buy": bd[bd >= oos0], "sell": sd[sd >= oos0]}
        fr[k] = {"OOS": {s: fill_rates(df, days_by[k][s], s) for s in ("buy", "sell")},
                 "FULL": {s: fill_rates(df, (bd if s == "buy" else sd)[(bd if s == "buy" else sd) >= pd.Timestamp("2020-01-01", tz="UTC")], s) for s in ("buy", "sell")},
                 "open_position_at_end": bool(side[side != 0][-1] == 1) if (side != 0).any() else False}
    # 組合實際成交日（含調倉）
    _, pf = ctx.port(0.0015, 0.0, benches=())
    for k_i, k in enumerate(("BTC", "ETH")):
        df = ctx.al[k]
        for s_, sname in ((1, "buy"), (-1, "sell")):
            dd = pd.DatetimeIndex(sorted({ctx.idx[i] for i, kk, sd_ in pf if kk == k_i and sd_ == s_ and ctx.idx[i] >= oos0}))
            fr[k].setdefault("PORT_OOS", {})[sname] = fill_rates(df, dd, sname)
    # 永續 vs 現貨偏差：同一批成交日，用永續日線算成交率、比較開盤價
    perp_d = load("ETHUSDT_USDT_1d.csv")
    pb = {}
    for s in ("buy", "sell"):
        dd = days_by["ETH"][s].intersection(perp_d.index)
        pb[s] = {"perp_daily": fill_rates(perp_d, dd, s), "spot_daily": fill_rates(ctx.dfs["ETH"], dd, s),
                 "open_gap_bps_mean": float(((perp_d.loc[dd, "open"] / ctx.dfs["ETH"].loc[dd, "open"] - 1) * 1e4).mean()),
                 "open_gap_bps_absmean": float(((perp_d.loc[dd, "open"] / ctx.dfs["ETH"].loc[dd, "open"] - 1).abs() * 1e4).mean())}
    # 永續 15m（分析者口徑）vs 現貨 15m（線上抓）
    p15 = pd.read_csv(os.path.join(DATA, "ETHUSDT_USDT_15m.csv"))
    p15.index = pd.to_datetime(p15["timestamp"], unit="ms", utc=True)
    perp_by = {t: g[["open", "high", "low", "close"]].to_numpy(float) for t, g in p15.groupby(p15.index.normalize()) if len(g) >= 96}
    intr = {"ETH_perp15m": {}, "ETH_spot15m": {}, "BTC_spot15m": {}}
    for s in ("buy", "sell"):
        for N in (1, 4, 8):
            for d in (0.0005, 0.001):
                intr["ETH_perp15m"][f"{s}_N{N}_d{d}"] = intraday_fill(perp_by, days_by["ETH"][s], s, N, d)
    if not no_net:
        try:
            spot_by = {"ETH": {}, "BTC": {}}
            for k in ("ETH", "BTC"):
                for t in sorted(set(days_by[k]["buy"]) | set(days_by[k]["sell"])):
                    spot_by[k][t] = fetch_spot_15m(f"{k}USDT", t, 8)
            for k in ("ETH", "BTC"):
                for s in ("buy", "sell"):
                    for N in (1, 4, 8):
                        for d in (0.0005, 0.001):
                            intr[f"{k}_spot15m"][f"{s}_N{N}_d{d}"] = intraday_fill(spot_by[k], days_by[k][s], s, N, d)
            pol = {}
            for k in ("ETH", "BTC"):
                for mk, fb, lab in ((0.0002, 0.0015, "maker0.02_fb0.15"), (0.00075, 0.0015, "maker0.075_fb0.15"), (0.00075, 0.00125, "maker0.075_fbBNB0.125"),
                                    (0.001, 0.0015, "VIP0無BNB_maker0.10_fb0.15")):
                    pol[f"{k}_{lab}"] = policy_cost(intr[f"{k}_spot15m"], 4, 0.0005, mk, fb)
            pol["ETH_perp_maker0.02_fb0.15"] = policy_cost(intr["ETH_perp15m"], 4, 0.0005, 0.0002, 0.0015)
            pol["ETH_perp_maker0.075_fb0.15"] = policy_cost(intr["ETH_perp15m"], 4, 0.0005, 0.00075, 0.0015)
            intr["policy_cost_N4_d0.05"] = pol
            # 永續與現貨 00:00 開盤價差
            gaps = [perp_by[t][0, 0] / spot_by["ETH"][t][0, 0] - 1 for t in spot_by["ETH"] if t in perp_by]
            intr["ETH_perp_vs_spot_open_gap_bps"] = {"mean": float(np.mean(gaps) * 1e4), "absmean": float(np.mean(np.abs(gaps)) * 1e4), "n": len(gaps)}
        except Exception as ex:          # 無網路時略過
            intr["spot_fetch_error"] = repr(ex)
    rep["fill_rates"] = fr
    rep["perp_vs_spot_daily"] = pb
    rep["intraday"] = intr

    # ---------------- 與分析者比對 ----------------
    cmp = []
    if ana:
        def ag(fee, slip):
            return next(r for r in ana["grid"] if abs(r["fee"] - fee) < 1e-9 and abs(r["slip"] - slip) < 1e-9)
        for tot in COST_LEVELS:
            fee, slip = split[tot]
            a = ag(fee, slip)
            g = grid[f"{tot:.5f}"]
            for k in ("BTC", "ETH"):
                for w in ("OOS", "OOS2"):
                    cmp.append({"item": f"{k} {w} Sharpe @{tot:.3%}", "mine": g[k][w]["sharpe"], "analyst": a[k]["windows"][w]["sharpe"],
                                "mine_pass": g[k][w]["pass"], "analyst_pass": a[k]["windows"][w]["pass"]})
            for w in ("OOS", "OOS2"):
                cmp.append({"item": f"PORT {w} Sharpe @{tot:.3%}", "mine": g["PORT"][w]["sharpe"], "analyst": a["PORT"]["windows"][w]["sharpe"],
                            "mine_pass_fresh_nr/mr": [g["PORT"][w]["fresh_nr"]["pass"], g["PORT"][w]["fresh_mr"]["pass"]],
                            "analyst_pass_nr/mr": [a["PORT"]["windows"][w]["pass_nr"], a["PORT"]["windows"][w]["pass_mr"]],
                            "bh_sharpe_fresh_nr/drift_nr/fresh_mr/drift_mr": [g["PORT"][w][b]["bh_sharpe"] for b in ("fresh_nr", "drift_nr", "fresh_mr", "drift_mr")],
                            "analyst_bh_nr/mr": [a["PORT"]["windows"][w]["bh_sharpe_nr"], a["PORT"]["windows"][w]["bh_sharpe_mr"]]})
        for k, ak in (("BTC", "BTC"), ("ETH", "ETH"), ("PORT_drift_nr", "PORT_nr"), ("PORT_drift_mr", "PORT_mr"), ("PORT_fresh_nr", "PORT_nr"), ("PORT_fresh_mr", "PORT_mr")):
            for w in ("OOS", "OOS2"):
                cmp.append({"item": f"c* {k} {w}", "mine": be[k][w].get("c_star"), "analyst": ana["breakeven"][ak][w].get("c_star")})
        for k in ("BTC", "ETH", "PORT"):
            for w in ("OOS", "OOS2", "FULL"):
                cmp.append({"item": f"turnover {k} {w}", "mine": drag[k][w]["turnover"], "analyst": ana["drag"][k][w]["turnover_per_year"]})
                cmp.append({"item": f"measured drag {k} {w}", "mine": drag[k][w]["measured_drag"], "analyst": ana["drag"][k][w]["measured_drag_live"]})
        for k in ("BTC", "ETH"):
            for s in ("buy", "sell"):
                for d in ("0.0005", "0.001"):
                    cmp.append({"item": f"fill {k} {s} OOS δ={d}", "mine": fr[k]["OOS"][s][d], "analyst": ana["execution"][k]["OOS"][s]["by_delta"][d]["p"],
                                "n_mine": fr[k]["OOS"][s]["n"], "n_analyst": ana["execution"][k]["OOS"][s]["n"]})
    rep["compare"] = cmp
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # ---------------- 摘要 ----------------
    def f(x, nd=3):
        return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))
    print(f"資料到 {rep['data_end']}｜warmup {rep['warmup']}")
    for c in cmp:
        extra = {k: v for k, v in c.items() if k not in ("item", "mine", "analyst")}
        d = (c["mine"] - c["analyst"]) if isinstance(c["mine"], float) and isinstance(c["analyst"], float) else None
        print(f"{c['item']:<34} mine {f(c['mine'], 4):>8}  ana {f(c['analyst'], 4):>8}  Δ {f(d, 4):>8}  {extra if extra else ''}")
    print("政策點組合 Sharpe（fee=c, slip 0）：", {k: {w: round(v[w]['sharpe'], 3) for w in v} for k, v in rep["policy_points"].items()})
    print("政策點組合 CAGR：", {k: {w: round(v[w]['cagr'], 4) for w in v} for k, v in rep["policy_points"].items()})
    print("永續 vs 現貨日線：", json.dumps(pb, ensure_ascii=False))
    print("日內：", json.dumps(intr, ensure_ascii=False))
    print("組合成交日（含調倉）成交率：", {k: fr[k]["PORT_OOS"] for k in ("BTC", "ETH")})
    print("兩平：", json.dumps(be, ensure_ascii=False))
    print(f"→ {OUT}（{time.time() - t0:.0f}s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
