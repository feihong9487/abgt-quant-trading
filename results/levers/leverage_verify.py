"""懷疑者獨立重算：leverage（槓桿）。

不 import abgt.*；直接讀 data/BTCUSDT_1d.csv、ETHUSDT_1d.csv（現貨），自寫 EMA50（SMA 起算）、regime、
共用權益組合模擬（固定槓桿 / 波動目標、借貸計息、逐日全倉保證金率）、持有基準、Sharpe/Calmar/DD/閘門，
再與 results/levers/leverage.json 逐項比對。另做分析者沒做的檢查：
  (a) 「Sharpe 不低於 1×」對任何借貸成本 > 0 的固定槓桿是否在代數上就不可能（Sharpe_L = Sharpe_1 − 借貸成本/(L·σ)），
      用 0% 借貸重跑 1.5× / 2× 看 Sharpe 是否回到 1× 水準；
  (b) Kelly 表的區間 Sharpe（IS2 1.39 / OOS2 0.78）與網格表（1.37 / 0.74）為何不同 —— slice_window 的 wend+1 天是否多算一天；
  (c) 借款利率盈虧平衡：固定 1.5× 的 CAGR 優勢在多高的借款利率下歸零；
  (d) 用 Binance 全倉規則檢查 2.5× 是否超過 3× 上限之外的另一個限制：單標的在場時的 sleeve 曝險。

重跑：PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe results/levers/leverage_verify.py
輸出：results/levers/leverage_verify.json
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FEE, SLIP, TH, INIT = 0.001, 0.0005, 0.15, 10_000.0
BORROW_TOL = 0.005
FILES = {"BTC": "BTCUSDT_1d.csv", "ETH": "ETHUSDT_1d.csv"}
FUND = {"BTC": "BTCUSDT_USDT_funding.csv", "ETH": "ETHUSDT_USDT_funding.csv"}
W = {"BTC": 0.5, "ETH": 0.5}
WINDOWS = [("IS", "2020-01-01", "2022-12-31"), ("OOS", "2023-01-01", None), ("FULL", "2020-01-01", None),
           ("IS2", "2020-07-01", "2023-12-31"), ("OOS2", "2024-01-01", None)]
REF = json.load(open(os.path.join(ROOT, "results", "levers", "leverage.json"), encoding="utf-8"))


def load(name: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(ROOT, "data", FILES[name]))
    d = d[d["timestamp"] >= int(pd.Timestamp("2019-09-01", tz="UTC").timestamp() * 1000)]
    d.index = pd.to_datetime(d["timestamp"], unit="ms", utc=True)
    return d[["open", "high", "low", "close"]].astype("float64")


def ema_sma_seed(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    out[n - 1] = x[:n].mean()
    a = 2.0 / (n + 1)
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def sharpe(eq: pd.Series) -> float:
    r = eq.resample("1D").last().dropna().pct_change().dropna()
    return float(r.mean() / r.std() * math.sqrt(365))


def max_dd(eq: pd.Series) -> float:
    a = eq.to_numpy()
    return float(((a - np.maximum.accumulate(a)) / np.maximum.accumulate(a)).min())


def summ(eq: pd.Series) -> dict:
    yrs = (eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 86400)
    ratio = eq.iloc[-1] / eq.iloc[0]
    dd = max_dd(eq)
    cagr = ratio ** (1 / yrs) - 1
    return {"net": ratio - 1, "cagr": cagr, "dd": dd, "sharpe": sharpe(eq), "calmar": cagr / abs(dd)}


def gate(s: dict, b: dict) -> bool:
    return bool(s["sharpe"] > b["sharpe"] and s["calmar"] > b["calmar"] and s["dd"] > b["dd"])


def win_slice(eq: pd.Series, ws: str, we: str | None) -> pd.Series:
    """與 evaluate_portfolio.window_metrics 同語意：searchsorted 起點、wend 含當日（side=right）。"""
    a = pd.Timestamp(ws, tz="UTC")
    if we:
        b = pd.Timestamp(we, tz="UTC")
        return eq.loc[a:b]
    return eq.loc[a:]


dfs = {k: load(k) for k in FILES}
idx = dfs["BTC"].index.intersection(dfs["ETH"].index)
O = {k: dfs[k]["open"].reindex(idx).to_numpy() for k in W}
H = {k: dfs[k]["high"].reindex(idx).to_numpy() for k in W}
Lo = {k: dfs[k]["low"].reindex(idx).to_numpy() for k in W}
C = {k: dfs[k]["close"].reindex(idx).to_numpy() for k in W}
ON = {}
EMA = {}
for k in W:
    e = ema_sma_seed(dfs[k]["close"].to_numpy(), 50)
    EMA[k] = pd.Series(e, index=dfs[k].index).reindex(idx).to_numpy()
    ON[k] = pd.Series(dfs[k]["close"].to_numpy() > e, index=dfs[k].index).reindex(idx).fillna(False).to_numpy(dtype=bool)
N = len(idx)
# 暖機：兩標的 EMA50 都有值的第一根（與引擎 warmup = max(各標的第一個非 NaN) 相同）
START = int(max(np.argmax(~np.isnan(EMA[k])) for k in W))
RVOL = {}
for k in W:
    r = np.log(pd.Series(C[k]) / pd.Series(C[k]).shift(1))
    RVOL[k] = (r.rolling(20).std() * math.sqrt(365)).to_numpy()


def simulate(L: float = 1.0, borrow: float = 0.0, vt: float = 0.0, delay: int = 0, fee: float = FEE, slip: float = SLIP):
    """共用權益；每 sleeve 目標 = w × L × min(1, vt/(rvol20×L))；收盤決策、次日開盤成交；先賣後買；
    L ≤ 1 時買量以現金為上限（不透支）；借款利息 = max(0, 部位 − 權益×1.005) × b / 365 每日收盤扣；
    保證金率 = 部位(以當日最低價) / (−現金)，只記錄不強平。"""
    keys = list(W)
    cash = INIT
    qty = {k: 0.0 for k in keys}
    pend = {k: [] for k in keys}
    eq = np.full(N, np.nan)
    expo = np.zeros(N)
    ml_low = np.full(N, np.inf)
    ml_close = np.full(N, np.inf)
    lev_prev = np.zeros(N)
    borrow_paid = 0.0
    fills = 0
    notional = 0.0
    for i in range(START, N):
        # 開盤成交
        todo = []
        for k in keys:
            due = [p for p in pend[k] if p[0] <= i]
            if due:
                pend[k] = [p for p in pend[k] if p[0] > i]
                todo.append((k, due[-1][1]))
        eq0 = cash + sum(qty[j] * O[j][i] for j in keys)
        todo.sort(key=lambda x: x[1] * eq0 / O[x[0]][i] - qty[x[0]] > 0)
        for k, target in todo:
            eq_open = cash + sum(qty[j] * O[j][i] for j in keys)
            diff = target * eq_open / O[k][i] - qty[k]
            if diff > 0 and L <= 1.0:
                diff = min(diff, max(0.0, cash) / (O[k][i] * (1 + slip) * (1 + fee)))
            if abs(diff) * O[k][i] < 1e-6 * eq_open:
                continue
            if diff > 0:
                px = O[k][i] * (1 + slip)
                cash -= diff * px * (1 + fee)
            else:
                px = O[k][i] * (1 - slip)
                cash += -diff * px * (1 - fee)
            qty[k] += diff
            if qty[k] < 1e-12:
                qty[k] = 0.0
            fills += 1
            notional += abs(diff) * px
        debt = -cash
        if debt > 1e-9 and any(qty[k] > 0 for k in keys):
            ml_low[i] = sum(qty[k] * Lo[k][i] for k in keys) / debt
            ml_close[i] = sum(qty[k] * C[k][i] for k in keys) / debt
        # 收盤
        pos_val = sum(qty[k] * C[k][i] for k in keys)
        eqv = cash + pos_val
        if borrow > 0 and eqv > 0:
            bor = max(0.0, pos_val - eqv * (1 + BORROW_TOL))
            if bor > 0:
                cost = bor * borrow / 365.0
                cash -= cost
                borrow_paid += cost
                eqv -= cost
        for k in keys:
            target = 0.0
            if ON[k][i]:
                s = 1.0
                if vt > 0 and RVOL[k][i] == RVOL[k][i] and RVOL[k][i] > 0:
                    s = min(1.0, vt / (RVOL[k][i] * L))
                target = W[k] * L * s
            cur = qty[k] * C[k][i] / eqv
            ref = pend[k][-1][1] if pend[k] else cur
            if target == 0 and ref > 0:
                pend[k].append((i + 1 + delay, 0.0))
            elif target > 0 and ref == 0:
                pend[k].append((i + 1 + delay, target))
            elif target > 0 and ref > 0 and abs(target - ref) > TH * W[k]:
                pend[k].append((i + 1 + delay, target))
        eq[i] = eqv
        expo[i] = pos_val / eqv
        if i + 1 < N:
            lev_prev[i + 1] = pos_val / eqv
    e = pd.Series(eq, index=idx).iloc[START:]
    return {"eq": e, "expo": expo[START:], "ml_low": ml_low[START:], "ml_close": ml_close[START:], "lev_prev": lev_prev[START:],
            "borrow_paid": borrow_paid, "fills": fills, "final": float(e.iloc[-1]), "min_eq": float(e.min()),
            "turnover_py": notional / float(e.mean()) / ((e.index[-1] - e.index[0]).days / 365.25)}


def hold(rebalance: str | None):
    q = {k: INIT * (1 - FEE) * W[k] / C[k][START] for k in W}
    eq = np.full(N, np.nan)
    last_m = idx[START].month
    for i in range(START, N):
        val = sum(q[k] * C[k][i] for k in W)
        if rebalance == "M" and idx[i].month != last_m:
            last_m = idx[i].month
            cost = sum(abs(val * W[k] - q[k] * C[k][i]) * FEE for k in W)
            val -= cost
            q = {k: val * W[k] / C[k][i] for k in W}
        eq[i] = val
    return pd.Series(eq, index=idx).iloc[START:]


BENCH = {"NR": hold(None), "M": hold("M")}


def windows_of(eq: pd.Series) -> dict:
    out = {}
    for wn, ws, we in WINDOWS:
        e = win_slice(eq, ws, we)
        e = e / e.iloc[0] * INIT
        s = summ(e)
        row = {"sharpe": s["sharpe"], "dd": s["dd"], "cagr": s["cagr"], "calmar": s["calmar"], "gates": 0, "bench": {}}
        for bn, b in BENCH.items():
            bb = win_slice(b, ws, we)
            bb = bb / bb.iloc[0] * INIT
            bs = summ(bb)
            row["bench"][bn] = {"sharpe": bs["sharpe"], "dd": bs["dd"], "calmar": bs["calmar"]}
            row["gates"] += int(gate(s, bs))
        out[wn] = row
    return out


out = {"portfolio_start": str(idx[START].date()), "data_last": str(idx[-1].date()), "n_days_after_start": int(N - START)}

# ---------- 1. 固定槓桿網格（我的引擎 vs 分析者 JSON） ----------
grid_cmp = {}
sims = {}
for tag, L, b in [("L1.0_b0%", 1.0, 0.0), ("L1.25_b5%", 1.25, 0.05), ("L1.5_b10%", 1.5, 0.10), ("L2.0_b10%", 2.0, 0.10), ("L2.5_b10%", 2.5, 0.10)]:
    r = simulate(L, b)
    sims[tag] = r
    wm = windows_of(r["eq"])
    ref = REF["fixed_grid"][tag]
    grid_cmp[tag] = {
        "mine": {w: {"sharpe": round(x["sharpe"], 4), "dd": round(x["dd"], 4), "gates": x["gates"]} for w, x in wm.items()},
        "ref": {w: {"sharpe": round(x["sharpe"], 4), "dd": round(x["max_dd"], 4), "gates": sum(int(bb["gate"]) for bb in x["bench"].values())} for w, x in ref["windows"].items()},
        "max_abs_sharpe_diff": max(abs(wm[w]["sharpe"] - ref["windows"][w]["sharpe"]) for w in wm),
        "max_abs_dd_diff": max(abs(wm[w]["dd"] - ref["windows"][w]["max_dd"]) for w in wm),
        "gates_mine_vs_ref": [sum(x["gates"] for x in wm.values()), ref["gates_passed"]],
        "final_equity_mine_vs_ref": [round(r["final"]), round(ref["final_equity"])],
        "borrow_paid_mine_vs_ref": [round(r["borrow_paid"]), round(ref["borrow_paid"])],
        "min_equity_mine_vs_ref": [round(r["min_eq"]), round(ref["min_equity_full_run"])],
        "fills_mine_vs_ref": [r["fills"], ref["n_fills"]],
        "min_margin_level_low_mine_vs_ref": [None if not np.isfinite(r["ml_low"].min()) else round(float(r["ml_low"].min()), 3), ref["engine_min_margin_level"]],
    }
out["fixed_grid_cmp"] = grid_cmp
out["hold_dd"] = {w: {bn: round(x["bench"][bn]["dd"], 4) for bn in ("NR", "M")} for w, x in windows_of(sims["L1.0_b0%"]["eq"]).items()}

# ---------- 2. 波動目標型 ----------
vt_cmp = {}
for tag, L, b, vt in [("vt0.6_L1.5_b10%", 1.5, 0.10, 0.6), ("vt0.6_L2.0_b10%", 2.0, 0.10, 0.6), ("vt0.6_L1.0_b0%", 1.0, 0.0, 0.6), ("vt0.4_L1.0_b0%", 1.0, 0.0, 0.4)]:
    r = simulate(L, b, vt)
    sims[tag] = r
    wm = windows_of(r["eq"])
    ref = REF["vol_target_grid"][tag]
    borrow_days = float(np.mean(r["expo"] > 1 + BORROW_TOL))
    vt_cmp[tag] = {
        "mine": {w: {"sharpe": round(x["sharpe"], 4), "dd": round(x["dd"], 4), "gates": x["gates"]} for w, x in wm.items()},
        "ref": {w: {"sharpe": round(x["sharpe"], 4), "dd": round(x["max_dd"], 4)} for w, x in ref["windows"].items()},
        "max_abs_sharpe_diff": max(abs(wm[w]["sharpe"] - ref["windows"][w]["sharpe"]) for w in wm),
        "max_abs_dd_diff": max(abs(wm[w]["dd"] - ref["windows"][w]["max_dd"]) for w in wm),
        "final_equity_mine_vs_ref": [round(r["final"]), round(ref["final_equity"])],
        "FULL_cagr_mine_vs_ref": [round(wm["FULL"]["cagr"], 4), round(ref["windows"]["FULL"]["cagr"], 4)],
        "borrow_days_share_mine_vs_ref": [round(borrow_days, 4), ref["engine_borrow_days_frac"]],
        "max_lev_prev_close_mine": round(float(r["lev_prev"].max()), 3),
        "min_margin_level_low_mine_vs_ref": [None if not np.isfinite(r["ml_low"].min()) else round(float(r["ml_low"].min()), 3), ref["engine_min_margin_level"]],
        "fills_mine_vs_ref": [r["fills"], ref["n_fills"]],
    }
out["vol_target_cmp"] = vt_cmp

# ---------- 3. Kelly（組合 FULL 等） + bootstrap + Kelly 表 vs 網格表 Sharpe 差異來源 ----------
eq1 = sims["L1.0_b0%"]["eq"]
daily = eq1.resample("1D").last().dropna().pct_change().dropna()
kelly = {}
rng = np.random.default_rng(7)
for wn, ws, we in WINDOWS:
    # 分析者 slice_window：wend + 1 天、loc 含端點 → 多含 wend 次日一根
    a = pd.Timestamp(ws, tz="UTC")
    r_analyst = daily.loc[a:(pd.Timestamp(we, tz="UTC") + pd.Timedelta(days=1))] if we else daily.loc[a:]
    # 網格表口徑：權益從區間起點正規化 → 第一筆報酬是起點次日
    e = win_slice(eq1, ws, we)
    r_grid = e.pct_change().dropna()
    mu_d, sd_d = float(r_analyst.mean()), float(r_analyst.std())
    f = mu_d / sd_d ** 2
    arr = r_analyst.to_numpy()
    n = len(arr)
    nb = math.ceil(n / 30)
    fs = np.empty(2000)
    for k in range(2000):
        starts = rng.integers(0, n, nb)
        sample = np.concatenate([arr[(s + np.arange(30)) % n] for s in starts])[:n]
        fs[k] = sample.mean() / sample.var(ddof=1)
    dd_real = -float((e / e.cummax() - 1).min())
    kelly[wn] = {"n_days_analyst_slice": n, "n_days_grid_slice": int(len(r_grid)),
                 "extra_days_in_analyst_slice": [str(t.date()) for t in r_analyst.index.difference(r_grid.index)],
                 "mu_annual": round(mu_d * 365, 4), "sigma_annual": round(sd_d * math.sqrt(365), 4),
                 "sharpe_analyst_slice": round(mu_d / sd_d * math.sqrt(365), 4), "sharpe_grid_slice": round(float(r_grid.mean() / r_grid.std() * math.sqrt(365)), 4),
                 "f_star": round(f, 4), "f_star_grid_slice": round(float(r_grid.mean() / r_grid.var()), 4),
                 "boot_p05": round(float(np.percentile(fs, 5)), 3), "boot_p50": round(float(np.percentile(fs, 50)), 3), "boot_p95": round(float(np.percentile(fs, 95)), 3),
                 "P_f_below_1.5": round(float(np.mean(fs < 1.5)), 3),
                 "realized_dd_1x": round(dd_real, 4), "model_P_dd_ge_realized_1x": round((1 - dd_real) ** (2 * f - 1), 4),
                 "ref": {x: REF["kelly"]["portfolio"]["windows"][wn][x] for x in ("n_days", "mu_arith_annual", "sigma_annual", "sharpe", "kelly", "kelly_boot_p05", "kelly_boot_prob_below_1.5")} | {"model_p_1x": REF["kelly"]["portfolio"]["windows"][wn]["model_p_dd_ge_realized"]["1.0"]}}
out["kelly_portfolio"] = kelly

# ---------- 4. 「Sharpe ≥ 1×」是否對任何借貸成本 > 0 都不可能：借 0% 重跑 + 代數預測 ----------
alg = {}
s1 = {w: windows_of(eq1)[w] for w in ("FULL",)}
for L in (1.25, 1.5, 2.0):
    r0 = simulate(L, 0.0)
    r10 = simulate(L, 0.10)
    w0 = windows_of(r0["eq"])
    w10 = windows_of(r10["eq"])
    # 代數：r_L ≈ L·r_1 − c_t，Sharpe_L ≈ Sharpe_1 − mean(c)/(L·σ_1)；c = 借貸成本佔權益/日
    d1 = win_slice(eq1, "2020-01-01", None).pct_change().dropna()
    yrs = (d1.index[-1] - d1.index[0]).days / 365.25
    borrow_pct_py = r10["borrow_paid"] / float(r10["eq"].mean()) / ((r10["eq"].index[-1] - r10["eq"].index[0]).days / 365.25)
    pred = float(d1.mean() / d1.std() * math.sqrt(365)) - borrow_pct_py / (L * float(d1.std() * math.sqrt(365)))
    alg[f"L{L}"] = {"FULL_sharpe_1x": round(s1["FULL"]["sharpe"], 4), "FULL_sharpe_borrow0": round(w0["FULL"]["sharpe"], 4), "FULL_sharpe_borrow10": round(w10["FULL"]["sharpe"], 4),
                    "FULL_sharpe_borrow10_predicted_from_1x_minus_cost": round(pred, 4), "borrow_cost_pct_equity_per_year": round(borrow_pct_py, 4),
                    "FULL_dd_borrow0": round(w0["FULL"]["dd"], 4), "FULL_cagr_borrow0": round(w0["FULL"]["cagr"], 4), "FULL_cagr_borrow10": round(w10["FULL"]["cagr"], 4),
                    "FULL_calmar_1x": round(s1["FULL"]["calmar"], 3), "FULL_calmar_borrow10": round(w10["FULL"]["calmar"], 3),
                    "sharpe_ge_1x_windows_borrow0": sum(int(w0[w]["sharpe"] >= windows_of(eq1)[w]["sharpe"]) for w in w0),
                    "gates_borrow0": sum(x["gates"] for x in w0.values())}
out["sharpe_vs_1x_algebra"] = alg

# ---------- 5. 借款利率盈虧平衡（1.5×：FULL CAGR 對 1× 的優勢在多高利率下歸零；並列 Calmar） ----------
be = {}
for b in (0.10, 0.20, 0.30, 0.40, 0.50):
    r = simulate(1.5, b)
    w = windows_of(r["eq"])
    be[f"b{int(b * 100)}%"] = {"FULL_cagr": round(w["FULL"]["cagr"], 4), "FULL_sharpe": round(w["FULL"]["sharpe"], 4), "FULL_dd": round(w["FULL"]["dd"], 4),
                              "FULL_calmar": round(w["FULL"]["calmar"], 3), "OOS_cagr": round(w["OOS"]["cagr"], 4), "OOS2_cagr": round(w["OOS2"]["cagr"], 4), "gates10": sum(x["gates"] for x in w.values())}
w1 = windows_of(eq1)
be["1x"] = {"FULL_cagr": round(w1["FULL"]["cagr"], 4), "FULL_sharpe": round(w1["FULL"]["sharpe"], 4), "FULL_dd": round(w1["FULL"]["dd"], 4), "FULL_calmar": round(w1["FULL"]["calmar"], 3),
            "OOS_cagr": round(w1["OOS"]["cagr"], 4), "OOS2_cagr": round(w1["OOS2"]["cagr"], 4)}
out["breakeven_borrow_1.5x"] = be

# ---------- 6. 強平距離、最差在場日、2020-03-12、資金費率 ----------
def r_liq(L, m=1.1):
    return m * (L - 1) / L - 1


out["r_liq"] = {str(L): {"r_liq": round(r_liq(L), 4), "r_liq_drift_L+0.15": round(r_liq(L + 0.15), 4), "ref": next(x for x in REF["liquidation"]["table"] if x["L"] == L)["r_liq"]} for L in (1.25, 1.5, 2.0, 2.5)}
asset = {}
for k in W:
    on = ON[k]
    held = np.zeros(N, dtype=bool)
    held[1:] = on[:-1]
    wu = int(np.argmax(~np.isnan(EMA[k])))
    held[:wu + 1] = False
    pl = Lo[k][1:] / C[k][:-1] - 1
    cc = C[k][1:] / C[k][:-1] - 1
    hd = held[1:]
    dates = idx[1:]
    j = int(np.argmin(np.where(hd, pl, np.inf)))
    i0312 = idx.get_loc(pd.Timestamp("2020-03-12", tz="UTC"))
    plh = pl[hd]
    f = pd.read_csv(os.path.join(ROOT, "data", FUND[k]))
    fs = pd.Series(f["rate"].to_numpy(), index=pd.to_datetime(f["timestamp"], unit="ms", utc=True)).resample("1D").sum()
    hs = pd.Series(held, index=idx)
    ff = fs.reindex(idx).fillna(0.0)
    asset[k] = {"n_held_days": int(hd.sum()), "worst_held_prevclose_to_low": [round(float(pl[j]), 4), str(dates[j].date())],
                "worst_all_close_to_close": [round(float(cc.min()), 4), str(dates[cc.argmin()].date())],
                "p99_p999_min_held": [round(float(np.percentile(plh, 1)), 4), round(float(np.percentile(plh, 0.1)), 4), round(float(plh.min()), 4)],
                "held_on_2020_03_12": bool(held[i0312]), "prev_close_vs_ema50_0311": [round(float(C[k][i0312 - 1]), 2), round(float(EMA[k][i0312 - 1]), 2)],
                "ret_2020_03_12_c2c_and_to_low": [round(float(C[k][i0312] / C[k][i0312 - 1] - 1), 4), round(float(Lo[k][i0312] / C[k][i0312 - 1] - 1), 4)],
                "funding_held_days_annual": round(float(ff[hs].mean() * 365), 4), "funding_held_share_positive": round(float((ff[hs] > 0).mean()), 4),
                "funding_held_OOS_annual": round(float(ff[hs].loc["2023-01-01":].mean() * 365), 4),
                "ref": {"worst_held": REF["liquidation"]["assets"][f"{k}/USDT"]["worst"]["prevclose_to_low_held"], "funding_held": REF["funding_vs_borrow"][f"{k}/USDT"]["avg_annual_funding_held_days"]}}
    # 趨勢段數
    ep = 0
    i = 0
    while i < N:
        if held[i]:
            ep += 1
            while i + 1 < N and held[i + 1]:
                i += 1
        i += 1
    asset[k]["n_episodes"] = ep
    asset[k]["ref_n_episodes"] = REF["liquidation"]["assets"][f"{k}/USDT"]["n_episodes"]
out["assets"] = asset

# ---------- 7. 組合最差日保證金率（1.5× / 2× / 2.5× 借 10%），與單標的在場時 sleeve 曝險 ----------
mm = {}
for tag in ("L1.5_b10%", "L2.0_b10%", "L2.5_b10%"):
    r = sims[tag]
    ml = r["ml_low"]
    fin = np.isfinite(ml)
    d = idx[START:][fin]
    mm[tag] = {"min_ml_low": round(float(ml[fin].min()), 3), "date": str(d[np.argmin(ml[fin])].date()), "days_ml_le_1.3": int((ml[fin] <= 1.3).sum()),
               "share_days_borrow": round(float(fin.mean()), 4), "max_expo_close": round(float(r["expo"].max()), 3),
               "ref_min_ml": REF["liquidation"]["portfolio_margin"][tag.split("_")[0][1:]]["min_margin_level_intraday"]}
out["portfolio_margin"] = mm

# ---------- 8. 延遲一根（分析者沒對槓桿格做）：1× vs vt0.6_L1.5 vs L1.5 的 FULL/OOS Sharpe 變化 ----------
dl = {}
for tag, L, b, vt in [("L1.0", 1.0, 0.0, 0.0), ("L1.5_b10", 1.5, 0.10, 0.0), ("vt0.6_L1.5_b10", 1.5, 0.10, 0.6)]:
    w0 = windows_of(simulate(L, b, vt, delay=0)["eq"])
    w1_ = windows_of(simulate(L, b, vt, delay=1)["eq"])
    dl[tag] = {w: {"sharpe_d0": round(w0[w]["sharpe"], 3), "sharpe_d1": round(w1_[w]["sharpe"], 3), "dd_d0": round(w0[w]["dd"], 3), "dd_d1": round(w1_[w]["dd"], 3), "gates_d1": w1_[w]["gates"]} for w in w0}
out["delay_one_bar"] = dl

with open(os.path.join(ROOT, "results", "levers", "leverage_verify.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=2, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
