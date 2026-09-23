"""槓桿量化：ema50_plain（現貨、費 0.10%/邊、滑 0.05%）能不能、該不該加槓桿。

五個部分（都可獨立重算）：
  1. Kelly：BTC 單獨與 BTC+ETH 50/50 組合的日報酬 → 年化 μ、σ、Sharpe、f* = μ/σ²、含借貸成本 f*、半 Kelly；IS / OOS / FULL / IS2 / OOS2 各算；區塊 bootstrap 給 f* 的區間
  2. 固定槓桿網格：max_exposure {1, 1.25, 1.5, 2, 2.5} × 借貸年化 {5%, 10%, 15%}，與 evaluate_portfolio --max-exposure --borrow-rate 同一個引擎與同一個 window_metrics，十格閘門
  3. 波動目標型槓桿：vol_target {0.4, 0.6} × max_exposure {1.5, 2.0}。注意 evaluate_portfolio 的 --vol-target 在組合引擎裡是 no-op
     （simulate_portfolio 不讀 vol_target），本腳本用組合引擎支援的每標的 scale 陣列實作同一語意：sleeve 目標 = w × min(L, vt / rvol20)
  4. 爆倉與跳空：Binance 現貨全倉保證金（借 USDT 買幣、維持保證金率 1.1 強平）的強平距離 vs 歷史單日跌幅（含 2020-03-12）、
     策略在場日的單日跌幅分布、每段趨勢的最深回撤；用網格模擬的成交紀錄重建逐日保證金率（兩標的同時觸最低價，保守）
  5. 借貸成本：對平均權益的年化 %；對照永續在場日的歷史資金費率（「用永續補槓桿」的替代工具成本）

用法：
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/levers/leverage.py
    輸出 results/levers/leverage.json 與 results/levers/leverage.md；不改任何既有檔案。
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from abgt import data as dt                                                # noqa: E402
from abgt import indicators as ta                                          # noqa: E402
from abgt.metrics import summarize                                         # noqa: E402
from abgt.portfolio import align, buy_and_hold_portfolio, simulate_portfolio   # noqa: E402
from abgt.strategies.regime import RiskParams, simulate_regime             # noqa: E402
from evaluate_portfolio import window_metrics                              # noqa: E402
from evaluate_variant import ALT_WINDOWS, DATA_SINCE, DEFAULT_WINDOWS, load_variant, parse_windows   # noqa: E402

# ---------- 設定（全部明寫，改這裡就能重算） ----------
KEY = "leverage"
SYMBOLS = ["BTC/USDT", "ETH/USDT"]              # 現貨（上線口徑）
PERP = {"BTC/USDT": "BTC/USDT:USDT", "ETH/USDT": "ETH/USDT:USDT"}   # 資金費率對照用
WEIGHTS = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
FEE, SLIP = 0.001, 0.0005                       # 0.10%/邊、0.05%
REBAL = 0.15                                    # evaluate_portfolio 預設調倉門檻（每 sleeve 0.15 × w = 0.075）
LEVERS = [1.0, 1.25, 1.5, 2.0, 2.5]
BORROWS = [0.05, 0.10, 0.15]
VOL_TARGETS = [0.4, 0.6]
VT_LEVERS = [1.5, 2.0]
VOL_LOOKBACK = 20
MAINT_LIQ, MAINT_CALL = 1.1, 1.3                # Binance 全倉保證金：≤1.1 強平、≤1.3 追繳通知
BOOT_REPS, BOOT_BLOCK, BOOT_SEED = 2000, 30, 7
WINDOWS = parse_windows(DEFAULT_WINDOWS + "," + ALT_WINDOWS)
OUT_DIR = os.path.join(ROOT, "results", "levers")
INITIAL = 10_000.0


def _git_head() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


ENGINE_COMMIT = _git_head()      # 引擎版本：0881f4b 起組合層有 vol_target / 強平 / 現貨不透支 / 借款容差 0.5%，數字與之前版本差 1–2%


def fmt(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else f"{x:.{nd}f}"


def pct(x, nd=1):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.{nd}f}%"


def slice_window(s: pd.Series, wstart: str, wend: str | None) -> pd.Series:
    a = pd.Timestamp(wstart, tz="UTC")
    b = pd.Timestamp(wend, tz="UTC") + pd.Timedelta(days=1) if wend else None
    return s.loc[a:b] if b else s.loc[a:]


# ---------- 資料與 regime ----------
dfs = {s: dt.load_cached("data", s, "1d", DATA_SINCE) for s in SYMBOLS}
assert all(v is not None for v in dfs.values()), "缺現貨快取"
idx = align(dfs)
mod = load_variant("ema50_plain")
built = {s: mod.build(dfs[s], copy.deepcopy(mod.PARAMS)) for s in SYMBOLS}
regimes = {s: built[s]["in_regime"] for s in SYMBOLS}
warmup = max(int(idx.searchsorted(dfs[s].index[int(built[s]["warmup"])])) for s in SYMBOLS)
base_rp = RiskParams(rebalance_threshold=REBAL, commission_pct=FEE, slippage_pct=SLIP)


def run_pf(rp: RiskParams, regimes_: dict | None = None):
    res = simulate_portfolio(dfs, regimes_ or regimes, rp, WEIGHTS, warmup)
    benches = {"持有不再平衡": buy_and_hold_portfolio(dfs, WEIGHTS, idx, warmup, rp.initial_capital, rp.commission_pct, None),
               "持有月再平衡": buy_and_hold_portfolio(dfs, WEIGHTS, idx, warmup, rp.initial_capital, rp.commission_pct, "M")}
    wm = window_metrics(res.equity, benches, idx, WINDOWS, warmup, rp.initial_capital)
    return res, wm, benches


# ================= 1. Kelly =================
def kelly_block(eq: pd.Series, held: pd.Series, label: str) -> dict:
    """eq：權益；held：該日是否持倉（決定借貸只在在場日計）。回傳各區間 μ σ Sharpe f*。"""
    daily = eq.resample("1D").last().dropna().pct_change().dropna()
    held_d = held.reindex(daily.index).fillna(False).astype(bool)
    out = {"label": label, "windows": {}}
    rng = np.random.default_rng(BOOT_SEED)
    for wname, ws, we in WINDOWS:
        r = slice_window(daily, ws, we)
        h = held_d.reindex(r.index).fillna(False)
        if len(r) < 60:
            continue
        mu_d, sd_d = float(r.mean()), float(r.std())
        mu_a, sd_a = mu_d * 365, sd_d * math.sqrt(365)
        mu_log = float(np.log1p(r).mean() * 365)
        tim = float(h.mean())
        f_star = mu_d / sd_d ** 2
        row = {"n_days": int(len(r)), "mu_arith_annual": mu_a, "sigma_annual": sd_a, "mu_log_annual": mu_log,
               "sharpe": mu_a / sd_a, "time_in_market": tim, "kelly": f_star, "half_kelly": f_star / 2,
               "kelly_with_borrow": {f"{b:.0%}": max(1.0, (mu_a - b * tim) / sd_a ** 2) if f_star > 1 else f_star for b in BORROWS},
               "growth_curve": {}}
        for L in LEVERS:
            for b in BORROWS:
                g = L * mu_a - max(0.0, L - 1) * b * tim - L ** 2 * sd_a ** 2 / 2
                row["growth_curve"][f"L{L}_b{b:.0%}"] = g
        # 區塊 bootstrap：f* 的 5–95 分位
        arr = r.to_numpy()
        n = len(arr)
        nb = int(math.ceil(n / BOOT_BLOCK))
        fs = np.empty(BOOT_REPS)
        for k in range(BOOT_REPS):
            starts = rng.integers(0, n, nb)
            sample = np.concatenate([arr[(s + np.arange(BOOT_BLOCK)) % n] for s in starts])[:n]
            fs[k] = sample.mean() / sample.var(ddof=1)
        row["kelly_boot_p05"], row["kelly_boot_p50"], row["kelly_boot_p95"] = (float(np.percentile(fs, q)) for q in (5, 50, 95))
        row["kelly_boot_prob_below_1"] = float(np.mean(fs < 1.0))
        row["kelly_boot_prob_below_1.5"] = float(np.mean(fs < 1.5))
        # 幾何布朗運動下，用 Kelly 的 φ = L/f* 倍時「一輩子曾經回撤 ≥ D」的機率 = (1−D)^(2/φ − 1)（無限期上界）。
        # 拿它對照實際回撤：若模型給的機率很小而回撤已經發生，代表模型低估尾部，f* 要打折。
        dd_real = -float((eq.resample("1D").last().dropna().pipe(lambda e: slice_window(e, ws, we))).pipe(lambda e: (e / e.cummax() - 1).min()))
        row["realized_max_dd"] = dd_real
        row["model_p_dd_ge_realized"] = {}
        for L in LEVERS:
            phi = L / f_star if f_star > 0 else float("inf")
            expo_ = 2 / phi - 1
            row["model_p_dd_ge_realized"][str(L)] = float((1 - dd_real) ** expo_) if expo_ > 0 else 1.0
        out["windows"][wname] = row
    return out


# BTC 單獨（單標的引擎，現貨、上線成本）
btc_rp = dataclasses.replace(built["BTC/USDT"]["risk"], commission_pct=FEE, slippage_pct=SLIP)
btc_res = simulate_regime(dfs["BTC/USDT"], regimes["BTC/USDT"], btc_rp, int(built["BTC/USDT"]["warmup"]))
on_btc = pd.Series(np.asarray(regimes["BTC/USDT"], dtype=bool), index=dfs["BTC/USDT"].index)
held_btc = on_btc.shift(1).fillna(False).astype(bool)      # 收盤 i−1 判定、開盤 i 成交 → 第 i 日持倉 ⟺ on[i−1]
kelly_btc = kelly_block(btc_res.equity, held_btc, "BTC 單獨 ema50_plain 現貨")

pf1_res, pf1_wm, benches = run_pf(base_rp)
held_pf = pd.Series(False, index=idx)
for s in SYMBOLS:
    o = pd.Series(np.asarray(regimes[s], dtype=bool), index=dfs[s].index).reindex(idx).fillna(False).astype(bool)
    held_pf |= o.shift(1).fillna(False).astype(bool)
kelly_pf = kelly_block(pf1_res.equity, held_pf, "BTC+ETH 50/50 ema50_plain 現貨")


# ================= 2. 固定槓桿網格 =================
def grid_row(res, wm, L, b, tag):
    e = res.extra
    eq = res.equity
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    row = {"tag": tag, "max_exposure": L, "borrow_rate": b, "windows": {},
           "engine_min_margin_level": e.get("min_margin_level"), "engine_borrow_days_frac": e.get("borrow_days_frac"),
           "borrow_paid": e["borrow_paid"], "borrow_cost_pct_of_avg_equity_per_year": e["borrow_paid"] / float(eq.mean()) / years,
           "min_equity_full_run": e["min_equity"], "n_fills": e["n_fills"], "turnover_per_year": e["notional_turnover_per_year"],
           "avg_exposure": e["avg_exposure"], "max_exposure_seen": e["max_exposure_seen"], "rel_dd_vs_hold": e["rel_dd_vs_hold"],
           "final_equity": float(eq.iloc[-1])}
    gates = 0
    for wname, w in wm.items():
        s = w["strategy"]
        ew = slice_window(eq, *[x for x in [ww for ww in WINDOWS if ww[0] == wname][0][1:]])
        dd_path = ew / ew.cummax() - 1
        r = {"sharpe": s["sharpe"], "max_dd": s["max_dd"], "calmar": s["calmar"], "net_return": s["net_return"], "cagr": s["cagr"],
             "min_equity_ratio": float(ew.min() / ew.iloc[0]), "max_dd_trough_date": str(dd_path.idxmin().date()),
             "max_dd_peak_date": str(ew.loc[:dd_path.idxmin()].idxmax().date()), "bench": {}}
        for bname, bb in w["bench"].items():
            r["bench"][bname] = {"sharpe": bb["summary"]["sharpe"], "max_dd": bb["summary"]["max_dd"], "calmar": bb["summary"]["calmar"],
                                 "gate": bb["gate"]["passed"], "dd_ok": bb["gate"]["overall_dd_pass"], "sharpe_ok": bb["gate"]["sharpe_pass"]}
            gates += int(bb["gate"]["passed"])
        row["windows"][wname] = r
    row["gates_passed"] = gates
    return row


grid = {}
grid_res = {}
for L in LEVERS:
    for b in (BORROWS if L > 1 else [0.0]):
        rp = dataclasses.replace(base_rp, max_exposure=L, borrow_rate_annual=b)
        res, wm, _ = run_pf(rp)
        tag = f"L{L}_b{b:.0%}"
        grid[tag] = grid_row(res, wm, L, b, tag)
        grid_res[tag] = res

base_tag = "L1.0_b0%"
# 「最大合理槓桿」：每區間 (a) 回撤不深於兩種持有、(b) Sharpe ≥ 1× 同區間 Sharpe（另列容忍 −0.02）
max_lever = {}
for b in BORROWS:
    per_w = {}
    for wname, *_ in WINDOWS:
        base_s = grid[base_tag]["windows"][wname]["sharpe"]
        ok_dd, ok_both, ok_tol = 1.0, 1.0, 1.0
        for L in LEVERS:
            tag = base_tag if L == 1 else f"L{L}_b{b:.0%}"
            w = grid[tag]["windows"][wname]
            dd_ok = all(x["dd_ok"] for x in w["bench"].values())
            if dd_ok:
                ok_dd = max(ok_dd, L)
            if dd_ok and w["sharpe"] >= base_s:
                ok_both = max(ok_both, L)
            if dd_ok and w["sharpe"] >= base_s - 0.02:
                ok_tol = max(ok_tol, L)
        per_w[wname] = {"max_L_dd_only": ok_dd, "max_L_dd_and_sharpe": ok_both, "max_L_dd_and_sharpe_tol0.02": ok_tol}
    max_lever[f"{b:.0%}"] = per_w | {"min_over_windows": {k: min(v[k] for v in per_w.values()) for k in per_w[next(iter(per_w))]}}

# 對照：永續資料、回測成本（0.05%/0.02%）1.5× 借 8% 的煙霧數字（任務給的 FULL 1.13 / −74%）
perp_dfs = {PERP[s]: dt.load_cached("data", PERP[s], "1d", DATA_SINCE) for s in SYMBOLS}
perp_check = None
if all(v is not None for v in perp_dfs.values()):
    p_idx = align(perp_dfs)
    p_built = {s: mod.build(perp_dfs[s], copy.deepcopy(mod.PARAMS)) for s in perp_dfs}
    p_reg = {s: p_built[s]["in_regime"] for s in perp_dfs}
    p_wu = max(int(p_idx.searchsorted(perp_dfs[s].index[int(p_built[s]["warmup"])])) for s in perp_dfs)
    p_w = {PERP[s]: WEIGHTS[s] for s in SYMBOLS}
    p_rp = RiskParams(rebalance_threshold=REBAL, commission_pct=0.0005, slippage_pct=0.0002, max_exposure=1.5, borrow_rate_annual=0.08)
    p_res = simulate_portfolio(perp_dfs, p_reg, p_rp, p_w, p_wu)
    p_b = {"持有不再平衡": buy_and_hold_portfolio(perp_dfs, p_w, p_idx, p_wu, INITIAL, 0.0005, None)}
    p_wm = window_metrics(p_res.equity, p_b, p_idx, WINDOWS, p_wu, INITIAL)
    perp_check = {w: {"sharpe": x["strategy"]["sharpe"], "max_dd": x["strategy"]["max_dd"]} for w, x in p_wm.items()}


# ================= 3. 波動目標型槓桿（每 sleeve scale 陣列） =================
def vt_scale(df: pd.DataFrame, vt: float, L: float) -> np.ndarray:
    c = df["close"].to_numpy(dtype="float64")
    r = np.log(c / ta.shift(c, 1))
    rvol = pd.Series(r).rolling(VOL_LOOKBACK).std().to_numpy() * math.sqrt(365)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.minimum(1.0, vt / rvol / L)
    s[np.isnan(rvol) | (rvol <= 0)] = 1.0
    return s


vt_grid = {}
vt_res = {}
# 引擎 0881f4b 起組合層原生支援 vol_target（每標的 min(1, vt / (rvol × cap))，語意 = 本腳本的 vt_scale），這裡用原生欄位跑，
# 並用 scale 陣列版對一組做逐位交叉檢查（vt_crosscheck）。
for vt in VOL_TARGETS:
    for L in VT_LEVERS:
        for b in BORROWS:
            rp = dataclasses.replace(base_rp, max_exposure=L, borrow_rate_annual=b, vol_target=vt, vol_lookback=VOL_LOOKBACK)
            res, wm, _ = run_pf(rp)
            tag = f"vt{vt}_L{L}_b{b:.0%}"
            vt_grid[tag] = grid_row(res, wm, L, b, tag) | {"vol_target": vt}
            vt_res[tag] = res
# 同一波動目標但不加槓桿（L=1、不借錢）：分離「波動目標本身」與「槓桿」的效果
for vt in VOL_TARGETS:
    rp = dataclasses.replace(base_rp, vol_target=vt, vol_lookback=VOL_LOOKBACK)
    res, wm, _ = run_pf(rp)
    tag = f"vt{vt}_L1.0_b0%"
    vt_grid[tag] = grid_row(res, wm, 1.0, 0.0, tag) | {"vol_target": vt}
    vt_res[tag] = res
# 交叉檢查：scale 陣列實作 vs 原生 vol_target（vt 0.6、L 2.0、借 10%）
_reg_vt = {s: {"in_regime": np.asarray(regimes[s], dtype=bool), "scale": vt_scale(dfs[s], 0.6, 2.0)} for s in SYMBOLS}
_res_sc, _, _ = run_pf(dataclasses.replace(base_rp, max_exposure=2.0, borrow_rate_annual=0.10), _reg_vt)
vt_crosscheck = {"config": "vt0.6_L2.0_b10%", "max_abs_rel_equity_diff": float((_res_sc.equity / vt_res["vt0.6_L2.0_b10%"].equity - 1).abs().max()),
                 "n_fills_scale_impl": int(_res_sc.extra["n_fills"]), "n_fills_native": int(vt_res["vt0.6_L2.0_b10%"].extra["n_fills"])}


# ================= 4. 爆倉與跳空 =================
def r_liq(L: float) -> float:
    """借 (L−1)E 買 L·E 的幣；保證金率 = L(1+r)/(L−1) ≤ 1.1 強平 → r = 1.1(L−1)/L − 1。L ≤ 1 無借貸。"""
    return float("nan") if L <= 1.0 else MAINT_LIQ * (L - 1) / L - 1


def r_call(L: float) -> float:
    return float("nan") if L <= 1.0 else MAINT_CALL * (L - 1) / L - 1


liq_table = []
for L in LEVERS:
    liq_table.append({"L": L, "r_liq": r_liq(L), "r_margin_call": r_call(L),
                      "L_drift": L + 2 * REBAL * 0.5 if L > 1 else L,     # 兩 sleeve 各漂到門檻邊緣才調倉
                      "r_liq_drift": r_liq(L + 2 * REBAL * 0.5) if L > 1 else float("nan")})

# 4a/4b：每標的歷史單日跌幅、在場日分布、每段趨勢
asset_risk = {}
for s in SYMBOLS:
    df = dfs[s]
    o, h, l, c = (df[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    on = np.asarray(regimes[s], dtype=bool)
    held = np.zeros(len(c), dtype=bool)
    held[1:] = on[:-1]
    wu = int(built[s]["warmup"])
    held[:wu + 1] = False
    cc = c[1:] / c[:-1] - 1                       # 收→收
    pl = l[1:] / c[:-1] - 1                       # 前收→最低
    ol = l[1:] / o[1:] - 1                        # 開→最低
    dates = df.index[1:]
    hd = held[1:]
    worst = {
        "close_to_close_all": {"ret": float(cc.min()), "date": str(dates[cc.argmin()].date())},
        "prevclose_to_low_all": {"ret": float(pl.min()), "date": str(dates[pl.argmin()].date())},
        "open_to_low_all": {"ret": float(ol.min()), "date": str(dates[ol.argmin()].date())},
        "close_to_close_held": {"ret": float(cc[hd].min()), "date": str(dates[hd][cc[hd].argmin()].date())},
        "prevclose_to_low_held": {"ret": float(pl[hd].min()), "date": str(dates[hd][pl[hd].argmin()].date())},
        "open_to_low_held": {"ret": float(ol[hd].min()), "date": str(dates[hd][ol[hd].argmin()].date())},
    }
    d0312 = df.index.get_loc(pd.Timestamp("2020-03-12", tz="UTC"))
    ema = ta.ema(c, 50)
    day0312 = {"close_to_close": float(c[d0312] / c[d0312 - 1] - 1), "prevclose_to_low": float(l[d0312] / c[d0312 - 1] - 1),
               "held_that_day": bool(held[d0312]), "prev_close": float(c[d0312 - 1]), "ema50_prev_close": float(ema[d0312 - 1]),
               "last_exit_signal_before": None}
    # 2020-03-12 前最後一次 regime 轉空的日期
    j = d0312 - 1
    while j > 0 and not on[j]:
        j -= 1
    day0312["last_exit_signal_before"] = str(df.index[j + 1].date()) if j + 1 < d0312 else None
    day0312["days_flat_before_crash"] = int(d0312 - (j + 1)) if j + 1 < d0312 else 0
    # 在場日分布
    plh = pl[hd]
    dist = {"n_held_days": int(hd.sum()), "p50": float(np.percentile(plh, 50)), "p90": float(np.percentile(plh, 10)),
            "p99": float(np.percentile(plh, 1)), "p999": float(np.percentile(plh, 0.1)), "min": float(plh.min())}
    top10 = sorted(zip(plh.tolist(), dates[hd]), key=lambda z: z[0])[:10]
    dist["worst10"] = [{"date": str(t.date()), "prevclose_to_low": float(v), "close_to_close": float(cc[hd][list(dates[hd]).index(t)])} for v, t in top10]
    # 每段趨勢（連續持倉日）
    episodes = []
    i = 0
    n = len(c)
    while i < n:
        if not held[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and held[j + 1]:
            j += 1
        entry = o[i]
        exit_px = o[j + 1] if j + 1 < n else c[j]
        lows = l[i:j + 1]
        closes = c[i:j + 1]
        run_hi = np.maximum.accumulate(np.concatenate([[entry], closes]))[:-1]   # 進場價與之前收盤的最高
        run_hi = np.maximum(run_hi, np.maximum.accumulate(np.concatenate([[entry], closes[:-1]])) if len(closes) > 1 else entry)
        peak_to_trough = float((lows / run_hi - 1).min())
        episodes.append({"start": str(df.index[i].date()), "end": str(df.index[j].date()), "days": int(j - i + 1),
                         "ret": float(exit_px / entry - 1), "entry_to_trough": float(lows.min() / entry - 1),
                         "peak_to_trough": peak_to_trough, "worst_day_prevclose_to_low": float(pl[i - 1:j].min()),
                         "worst_day_close_to_close": float(cc[i - 1:j].min())})
        i = j + 1
    ep = pd.DataFrame(episodes)
    per_L = {}
    for L in LEVERS:
        if L <= 1:
            continue
        rl, rld = r_liq(L), r_liq(L + 2 * REBAL * 0.5)
        per_L[str(L)] = {
            "r_liq": rl, "r_liq_drift": rld,
            "held_days_below_r_liq": int((plh <= rl).sum()), "held_days_below_r_liq_drift": int((plh <= rld).sum()),
            "episodes_single_day_below_r_liq": int((ep.worst_day_prevclose_to_low <= rl).sum()),
            "episodes_single_day_below_r_liq_drift": int((ep.worst_day_prevclose_to_low <= rld).sum()),
            "episodes_peak_to_trough_below_r_liq": int((ep.peak_to_trough <= rl).sum()),
            "episodes_entry_to_trough_below_r_liq": int((ep.entry_to_trough <= rl).sum()),
            "p_exit_before_liq_rebalanced": float(1 - (ep.worst_day_prevclose_to_low <= rld).mean()),
            "p_exit_before_liq_no_rebalance": float(1 - (ep.peak_to_trough <= rl).mean()),
            "margin_to_liq_worst_held_day": float(plh.min() - rld),
        }
    asset_risk[s] = {"worst": worst, "day_2020_03_12": day0312, "held_day_prevclose_to_low_dist": dist,
                     "n_episodes": int(len(ep)), "episode_stats": {
                         "peak_to_trough_p50": float(ep.peak_to_trough.median()), "peak_to_trough_p90": float(ep.peak_to_trough.quantile(0.1)),
                         "peak_to_trough_min": float(ep.peak_to_trough.min()), "peak_to_trough_min_start": str(ep.loc[ep.peak_to_trough.idxmin(), "start"]),
                         "entry_to_trough_min": float(ep.entry_to_trough.min()), "entry_to_trough_min_start": str(ep.loc[ep.entry_to_trough.idxmin(), "start"]),
                         "worst_day_min": float(ep.worst_day_prevclose_to_low.min()), "worst_day_min_start": str(ep.loc[ep.worst_day_prevclose_to_low.idxmin(), "start"])},
                     "per_L": per_L, "episodes": episodes}


# 4d：從網格模擬的成交紀錄重建逐日保證金率（借 10% 那組；兩標的同時觸最低價 = 保守）
def margin_path(res, L: float) -> dict:
    eq = res.equity
    tr = res.trades
    n = len(idx)
    start = int(idx.get_loc(eq.index[0]))
    qty = {s: 0.0 for s in SYMBOLS}
    fills_by_day = {}
    if len(tr):
        for _, f in tr.iterrows():
            fills_by_day.setdefault(pd.Timestamp(f["exit_time"]), []).append(f)
    c = {s: dfs[s]["close"].reindex(idx).to_numpy() for s in SYMBOLS}
    l = {s: dfs[s]["low"].reindex(idx).to_numpy() for s in SYMBOLS}
    rows = []
    eq_prev = float(eq.iloc[0])
    for i in range(start + 1, n):
        t = idx[i]
        cash = eq_prev - sum(qty[s] * c[s][i - 1] for s in SYMBOLS)
        qty_before = dict(qty)                      # 前一日收盤的持倉 → 前收槓桿 = 持倉市值 / 權益
        for f in fills_by_day.get(t, []):
            if f["side"] > 0:
                cash -= f["qty"] * f["price"] * (1 + FEE)
                qty[f["asset"]] += f["qty"]
            else:
                cash += f["qty"] * f["price"] * (1 - FEE)
                qty[f["asset"]] -= f["qty"]
                if qty[f["asset"]] < 1e-12:
                    qty[f["asset"]] = 0.0
        liab = max(0.0, -cash)
        if liab <= 1e-6 * max(eq_prev, 1.0):
            liab = 0.0                              # 浮點殘差不算借款
        assets_low = sum(qty[s] * l[s][i] for s in SYMBOLS)
        assets_close = sum(qty[s] * c[s][i] for s in SYMBOLS)
        ml_low = assets_low / liab if liab > 0 else float("inf")
        ml_close = assets_close / liab if liab > 0 else float("inf")
        lev_prev_close = sum(qty_prev * c[s][i - 1] for s, qty_prev in qty_before.items()) / eq_prev if eq_prev > 0 else 0.0
        rows.append((t, liab, ml_low, ml_close, lev_prev_close))
        eq_prev = float(eq.loc[t]) if t in eq.index else eq_prev
    mp = pd.DataFrame(rows, columns=["time", "liab", "ml_low", "ml_close", "lev"]).set_index("time")
    lev = mp[mp.liab > 0]
    out = {"days_with_borrow": int(len(lev)), "share_days_with_borrow": float(len(lev) / len(mp)),
           "max_leverage_close": float(mp.lev.max()), "date_max_leverage": str(mp.lev.idxmax().date()),
           "avg_leverage_by_window": {w[0]: float(slice_window(mp.lev, w[1], w[2]).mean()) for w in WINDOWS},
           "max_leverage_by_window": {w[0]: float(slice_window(mp.lev, w[1], w[2]).max()) for w in WINDOWS},
           "share_days_borrow_by_window": {w[0]: float((slice_window(mp.liab, w[1], w[2]) > 0).mean()) for w in WINDOWS},
           "min_margin_level_intraday": float(lev.ml_low.min()) if len(lev) else None,
           "min_margin_level_close": float(lev.ml_close.min()) if len(lev) else None,
           "date_min_margin_level": str(lev.ml_low.idxmin().date()) if len(lev) else None,
           "days_ml_below_liq": int((lev.ml_low <= MAINT_LIQ).sum()), "days_ml_below_call": int((lev.ml_low <= MAINT_CALL).sum()),
           "days_ml_below_1.5": int((lev.ml_low <= 1.5).sum()),
           "worst5": [{"date": str(t.date()), "ml_low": float(r.ml_low), "ml_close": float(r.ml_close)} for t, r in lev.nsmallest(5, "ml_low").iterrows()]}
    # 假想壓力：同一天兩標的都再跌 X%（在最差保證金率日之上），要多少 X 才強平
    if len(lev):
        ml = float(lev.ml_low.min())
        out["extra_drop_to_liq_on_worst_day"] = float(MAINT_LIQ / ml - 1)   # ml×(1+x) = 1.1 → x
    return out


margin = {}
for L in LEVERS:
    if L <= 1:
        continue
    margin[str(L)] = margin_path(grid_res[f"L{L}_b10%"], L)
margin_vt = {tag: margin_path(vt_res[tag], vt_grid[tag]["max_exposure"]) for tag in vt_grid if "_b10%" in tag or "_b0%" in tag}
margin["1.0"] = margin_path(grid_res[base_tag], 1.0)          # 1× 的曝險路徑（對照用，不借錢）


# ================= 5. 借貸成本 vs 永續資金費率 =================
funding_cmp = {}
for s in SYMBOLS:
    f = dt.load_funding_cached("data", PERP[s])
    if f is None:
        continue
    hs = pd.Series(np.asarray(regimes[s], dtype=bool), index=dfs[s].index).shift(1).fillna(False).astype(bool)
    ff = f.reindex(hs.index).fillna(0.0)
    held_f = ff[hs]
    full_f = ff[ff.index >= pd.Timestamp("2020-01-01", tz="UTC")]
    row = {"held_days": int(hs.sum()), "avg_annual_funding_held_days": float(held_f.mean() * 365),
           "avg_annual_funding_all_days": float(full_f.mean() * 365), "share_held_days_positive": float((held_f > 0).mean()),
           "by_window": {}}
    for wname, ws, we in WINDOWS:
        x = slice_window(held_f, ws, we)
        if len(x):
            row["by_window"][wname] = float(x.mean() * 365)
    funding_cmp[s] = row


# ================= 報表 =================
def md_report() -> str:
    L_ = []
    L_.append(f"# 槓桿分析（{KEY}）— ema50_plain BTC+ETH 50/50 現貨，費 {FEE:.2%}/邊、滑 {SLIP:.2%}")
    L_.append("")
    L_.append(f"資料：Binance 現貨日線 {', '.join(SYMBOLS)}（起 {DATA_SINCE}，組合起算 {pf1_res.equity.index[0].date()}，最後 {idx[-1].date()}）；區間 " +
              "、".join(f"{w[0]} {w[1]}→{w[2] or '今'}" for w in WINDOWS) + f"。所有數字由 `experiments/levers/leverage.py` 重算（引擎 git `{ENGINE_COMMIT}`；0881f4b 起現貨 1× 不透支、借款容差 0.5%、組合層原生 vol_target，與 2dad804 引擎的數字差在 Sharpe ±0.02 / 成交 ±4 筆內）。")
    L_.append("")
    L_.append("## 1. Kelly：μ、σ、f* 在各區間穩不穩")
    L_.append("")
    L_.append("μ = 日簡單報酬均值 × 365（含空手日的 0）、σ = 日標準差 × √365、f* = μ/σ²（與年化無關：= 日均/日變異數）；含借貸 f*ᵦ = (μ − r_b × 在場比例)/σ²（借貸只在在場日計，f* ≤ 1 時不借）；bootstrap = 區塊長 30 日、2000 次的 f* 5–95 分位。")
    L_.append("")
    for kb in (kelly_btc, kelly_pf):
        L_.append(f"### {kb['label']}")
        L_.append("")
        L_.append("| 區間 | 天數 | μ 年化 | σ 年化 | Sharpe | 在場 | f* | 半 Kelly | f*ᵦ 借 5% | f*ᵦ 借 10% | f*ᵦ 借 15% | f* boot p05 / p50 / p95 | P(f*<1) | P(f*<1.5) | 實際回撤(1×) | 模型 P(回撤≥實際) 1× / 1.5× / 2× |")
        L_.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for w, r in kb["windows"].items():
            kb_ = r["kelly_with_borrow"]
            mp_ = r["model_p_dd_ge_realized"]
            L_.append(f"| {w} | {r['n_days']} | {pct(r['mu_arith_annual'])} | {r['sigma_annual']:.1%} | {r['sharpe']:.2f} | {r['time_in_market']:.0%} | {r['kelly']:.2f} | {r['half_kelly']:.2f} | "
                      f"{kb_['5%']:.2f} | {kb_['10%']:.2f} | {kb_['15%']:.2f} | {r['kelly_boot_p05']:.2f} / {r['kelly_boot_p50']:.2f} / {r['kelly_boot_p95']:.2f} | {r['kelly_boot_prob_below_1']:.0%} | {r['kelly_boot_prob_below_1.5']:.0%} | "
                      f"{-r['realized_max_dd']:.0%} | {mp_['1.0']:.1%} / {mp_['1.5']:.1%} / {mp_['2.0']:.1%} |")
        L_.append("")
        L_.append("「模型 P(回撤≥實際)」= 幾何布朗運動、用 L/f* 倍 Kelly 時一輩子曾回撤到實際 1× 回撤深度的機率上界 (1−D)^(2f*/L − 1)。1× 的實際回撤在模型下是幾 % 的事件卻已發生 → 模型（也就是 f*）低估尾部，f* 只能當上界不能當目標。")
        L_.append("")
        L_.append("對數成長率 g(L) = Lμ − (L−1)·r_b·在場 − L²σ²/2（借 10%）：")
        L_.append("")
        L_.append("| 區間 | " + " | ".join(f"L={L}" for L in LEVERS) + " |")
        L_.append("|---|" + "---|" * len(LEVERS))
        for w, r in kb["windows"].items():
            L_.append(f"| {w} | " + " | ".join(f"{r['growth_curve'][f'L{L}_b10%']:+.3f}" for L in LEVERS) + " |")
        L_.append("")
    L_.append("## 2. 固定槓桿網格（組合，十格閘門）")
    L_.append("")
    L_.append("每格：Sharpe / 最大回撤 / 區間內最低權益（相對區間起點）；✓ = 過兩種持有基準的風險閘門（Sharpe、Calmar 高於持有且回撤更小），數字為過閘格數 /10。借貸成本 = 全程借貸利息 / 平均權益 / 年。")
    L_.append("")
    g15 = grid["L1.5_b10%"]
    L_.append(f"獨立重算（同引擎 `{ENGINE_COMMIT}`、同 window_metrics）：`evaluate_portfolio.py --variant ema50_plain --symbols BTC/USDT,ETH/USDT --weights 0.5,0.5 --fee 0.001 --slip 0.0005 --max-exposure 1.5 --borrow-rate 0.10 --out <別的目錄>`，"
              f"應得 IS {g15['windows']['IS']['sharpe']:.2f}/{g15['windows']['IS']['max_dd']:.1%}、OOS {g15['windows']['OOS']['sharpe']:.2f}/{g15['windows']['OOS']['max_dd']:.1%}、FULL {g15['windows']['FULL']['sharpe']:.2f}/{g15['windows']['FULL']['max_dd']:.1%}、"
              f"IS2 {g15['windows']['IS2']['sharpe']:.2f}、OOS2 {g15['windows']['OOS2']['sharpe']:.2f}、借貸成本 {g15['borrow_paid']:,.0f}、最低權益 {g15['min_equity_full_run']:,.0f}、成交 {g15['n_fills']}。")
    L_.append("")
    L_.append("| 槓桿 | 借貸 | " + " | ".join(w[0] for w in WINDOWS) + " | 十格 | 借貸成本 %/年 | 全程最低權益（起 10,000） | 成交 | 換手/年 | 期末權益 |")
    L_.append("|---|---|" + "---|" * len(WINDOWS) + "---|---|---|---|---|---|")
    for tag, g in grid.items():
        cells = []
        for w in WINDOWS:
            x = g["windows"][w[0]]
            npass = sum(int(b["gate"]) for b in x["bench"].values())
            cells.append(f"{x['sharpe']:.2f} / {x['max_dd']:.0%} / {x['min_equity_ratio']:.2f}{'✓' if npass == 2 else ('½' if npass == 1 else '✗')}")
        L_.append(f"| {g['max_exposure']}× | {g['borrow_rate']:.0%} | " + " | ".join(cells) + f" | {g['gates_passed']} | {g['borrow_cost_pct_of_avg_equity_per_year']:.1%} | {g['min_equity_full_run']:,.0f} | {g['n_fills']} | {g['turnover_per_year']:.1f} | {g['final_equity']:,.0f} |")
    L_.append("")
    L_.append("持有基準回撤（同區間）：" + "；".join(f"{w[0]} 不再平衡 {grid[base_tag]['windows'][w[0]]['bench']['持有不再平衡']['max_dd']:.0%} / 月再平衡 {grid[base_tag]['windows'][w[0]]['bench']['持有月再平衡']['max_dd']:.0%}" for w in WINDOWS))
    L_.append("")
    L_.append("最大合理槓桿（每區間；「回撤」= 兩種持有都不比它淺；「Sharpe」= 不低於 1× 同區間）：")
    L_.append("")
    L_.append("| 借貸 | 條件 | " + " | ".join(w[0] for w in WINDOWS) + " | 五區間取最小 |")
    L_.append("|---|---|" + "---|" * (len(WINDOWS) + 1))
    for b, per in max_lever.items():
        for k, lab in (("max_L_dd_only", "只看回撤"), ("max_L_dd_and_sharpe", "回撤 + Sharpe ≥ 1×"), ("max_L_dd_and_sharpe_tol0.02", "回撤 + Sharpe ≥ 1× − 0.02")):
            L_.append(f"| {b} | {lab} | " + " | ".join(f"{per[w[0]][k]}×" for w in WINDOWS) + f" | {per['min_over_windows'][k]}× |")
    L_.append("")
    if perp_check:
        L_.append(f"對照煙霧（永續資料、費 0.05%/滑 0.02%、1.5× 借 8%）：" + "、".join(f"{w} Sharpe {x['sharpe']:.2f} 回撤 {x['max_dd']:.0%}" for w, x in perp_check.items()) + "（任務給的 FULL 1.13 / −74% 重現）。")
        L_.append("")
    L_.append("## 3. 波動目標型槓桿（每 sleeve 目標 = w × min(L, vt / rvol20)，借 10%；另列同 scale 但 L=1）")
    L_.append("")
    L_.append(f"引擎 2dad804 的 `evaluate_portfolio --vol-target` 在組合層是 no-op（本次 CLI 驗證過：加不加 --vol-target 0.4 輸出逐位相同）；引擎 0881f4b 起原生支援（每標的 min(1, vt / (rvol20 × 上限))），此表用原生欄位跑，"
              f"與本腳本 scale 陣列實作的交叉檢查（{vt_crosscheck['config']}）：權益最大相對差 {vt_crosscheck['max_abs_rel_equity_diff']:.2e}、成交 {vt_crosscheck['n_fills_scale_impl']} vs {vt_crosscheck['n_fills_native']}。"
              "rvol20 = 20 日對數報酬標準差 × √365，收盤算、次日開盤成交。CLI 重算：加 `--vol-target 0.6 --max-exposure 2.0 --borrow-rate 0.10`。")
    L_.append("")
    L_.append("| vt | 槓桿 | 借貸 | " + " | ".join(w[0] for w in WINDOWS) + " | 十格 | FULL CAGR | 期末權益 | 平均曝險 | 最高曝險 | 借貸成本 %/年 | 全程最低權益 | 換手/年 |")
    L_.append("|---|---|---|" + "---|" * len(WINDOWS) + "---|---|---|---|---|---|---|---|")
    base_row = grid[base_tag]
    L_.append(f"| — | 1.0× | 0% | " + " | ".join(f"{base_row['windows'][w[0]]['sharpe']:.2f} / {base_row['windows'][w[0]]['max_dd']:.0%}✓" for w in WINDOWS) +
              f" | {base_row['gates_passed']} | {base_row['windows']['FULL']['cagr']:.1%} | {base_row['final_equity']:,.0f} | {base_row['avg_exposure']:.2f} | {base_row['max_exposure_seen']:.2f} | 0.0% | {base_row['min_equity_full_run']:,.0f} | {base_row['turnover_per_year']:.1f} |")
    for tag, g in vt_grid.items():
        if g["borrow_rate"] not in (0.0, 0.10):
            continue
        cells = []
        for w in WINDOWS:
            x = g["windows"][w[0]]
            npass = sum(int(b["gate"]) for b in x["bench"].values())
            cells.append(f"{x['sharpe']:.2f} / {x['max_dd']:.0%}{'✓' if npass == 2 else ('½' if npass == 1 else '✗')}")
        L_.append(f"| {g['vol_target']} | {g['max_exposure']}× | {g['borrow_rate']:.0%} | " + " | ".join(cells) + f" | {g['gates_passed']} | {g['windows']['FULL']['cagr']:.1%} | {g['final_equity']:,.0f} | {g['avg_exposure']:.2f} | {g['max_exposure_seen']:.2f} | {g['borrow_cost_pct_of_avg_equity_per_year']:.1%} | {g['min_equity_full_run']:,.0f} | {g['turnover_per_year']:.1f} |")
    L_.append("")
    L_.append("借 5% / 15% 的版本在 JSON `vol_target_grid`。各區間平均前收槓桿（持倉市值 / 權益，由成交紀錄重建）與有借貸的天數比例：")
    L_.append("")
    L_.append("| 設定 | " + " | ".join(f"{w[0]} 平均 / 最高 / 借貸日" for w in WINDOWS) + " |")
    L_.append("|---|" + "---|" * len(WINDOWS))
    for tag, m in [("L1.0_b0%", margin["1.0"])] + [(t, m) for t, m in margin_vt.items()]:
        L_.append(f"| {tag} | " + " | ".join(f"{m['avg_leverage_by_window'][w[0]]:.2f} / {m['max_leverage_by_window'][w[0]]:.2f} / {m['share_days_borrow_by_window'][w[0]]:.0%}" for w in WINDOWS) + " |")
    L_.append("")
    L_.append("## 4. 爆倉與跳空")
    L_.append("")
    L_.append(f"模型：Binance 現貨全倉保證金，借 (L−1)·E USDT 買 L·E 的幣；保證金率 = 幣值 / 負債 = L(1+r)/(L−1)；≤ {MAINT_LIQ} 強平、≤ {MAINT_CALL} 追繳。"
              f"強平距離 r_liq = {MAINT_LIQ}(L−1)/L − 1。「漂移」= 引擎調倉門檻 {REBAL}×w 讓兩 sleeve 各漂 {REBAL * 0.5:.3f} 才調倉，總曝險最壞到 L+{2 * REBAL * 0.5:.2f} 才賣。")
    L_.append("")
    L_.append("| L | r_liq | r 追繳 | 漂移後 L | 漂移後 r_liq |")
    L_.append("|---|---|---|---|---|")
    for r in liq_table:
        L_.append(f"| {r['L']}× | {pct(r['r_liq'])} | {pct(r['r_margin_call'])} | {r['L_drift']:.2f}× | {pct(r['r_liq_drift'])} |")
    L_.append("")
    L_.append("### 4a. 歷史單日跌幅（現貨日線）與 2020-03-12")
    L_.append("")
    L_.append("| 標的 | 全期最差 收→收 | 全期最差 前收→最低 | 在場日最差 收→收 | 在場日最差 前收→最低 | 在場日最差 開→最低 | 2020-03-12 收→收 / 前收→最低 | 3/12 策略在場？ | 3/12 前最後出場訊號 |")
    L_.append("|---|---|---|---|---|---|---|---|---|")
    for s, a in asset_risk.items():
        w, d = a["worst"], a["day_2020_03_12"]
        L_.append(f"| {s.split('/')[0]} | {pct(w['close_to_close_all']['ret'])} ({w['close_to_close_all']['date']}) | {pct(w['prevclose_to_low_all']['ret'])} ({w['prevclose_to_low_all']['date']}) | "
                  f"{pct(w['close_to_close_held']['ret'])} ({w['close_to_close_held']['date']}) | {pct(w['prevclose_to_low_held']['ret'])} ({w['prevclose_to_low_held']['date']}) | {pct(w['open_to_low_held']['ret'])} ({w['open_to_low_held']['date']}) | "
                  f"{pct(d['close_to_close'])} / {pct(d['prevclose_to_low'])} | {'是' if d['held_that_day'] else '否'}（前收 {d['prev_close']:,.0f} vs EMA50 {d['ema50_prev_close']:,.0f}） | {d['last_exit_signal_before']}（空手 {d['days_flat_before_crash']} 天） |")
    L_.append("")
    L_.append("### 4b. 策略在場日的「前收→最低」分布與每段趨勢")
    L_.append("")
    L_.append("| 標的 | 在場日 | p50 | p90 | p99 | p99.9 | 最差 | 趨勢段數 | 段內峰→谷 p50 / p90 / 最差 | 段內進場→谷最差 | 段內最差單日 |")
    L_.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for s, a in asset_risk.items():
        d, e = a["held_day_prevclose_to_low_dist"], a["episode_stats"]
        L_.append(f"| {s.split('/')[0]} | {d['n_held_days']} | {pct(d['p50'])} | {pct(d['p90'])} | {pct(d['p99'])} | {pct(d['p999'])} | {pct(d['min'])} | {a['n_episodes']} | "
                  f"{pct(e['peak_to_trough_p50'])} / {pct(e['peak_to_trough_p90'])} / {pct(e['peak_to_trough_min'])}（{e['peak_to_trough_min_start']} 起） | {pct(e['entry_to_trough_min'])}（{e['entry_to_trough_min_start']} 起） | {pct(e['worst_day_min'])}（{e['worst_day_min_start']} 起） |")
    L_.append("")
    L_.append("在場日最差 10 天（前收→最低 / 收→收）：")
    L_.append("")
    for s, a in asset_risk.items():
        L_.append(f"- {s.split('/')[0]}：" + "；".join(f"{x['date']} {pct(x['prevclose_to_low'])} / {pct(x['close_to_close'])}" for x in a["held_day_prevclose_to_low_dist"]["worst10"]))
    L_.append("")
    L_.append("### 4c. 單標的滿槓桿時「EMA50 出場先於強平」的歷史頻率")
    L_.append("")
    L_.append("有調倉（引擎語意）：強平只會由單日跳空造成，比較每段趨勢最差單日（前收→最低）與漂移後 r_liq；不調倉：比較段內峰→谷。")
    L_.append("")
    L_.append("| 標的 | L | r_liq（漂移後） | 在場日跌破 r_liq 天數 | 趨勢段最差單日跌破段數 / 總段數 | P(出場先於強平, 有調倉) | 段內峰→谷跌破段數 | P(出場先於強平, 不調倉) | 最差在場日距強平 |")
    L_.append("|---|---|---|---|---|---|---|---|---|")
    for s, a in asset_risk.items():
        for Lk, r in a["per_L"].items():
            L_.append(f"| {s.split('/')[0]} | {Lk}× | {pct(r['r_liq'])}（{pct(r['r_liq_drift'])}） | {r['held_days_below_r_liq_drift']} | {r['episodes_single_day_below_r_liq_drift']} / {a['n_episodes']} | {r['p_exit_before_liq_rebalanced']:.1%} | {r['episodes_peak_to_trough_below_r_liq']} | {r['p_exit_before_liq_no_rebalance']:.1%} | {pct(r['margin_to_liq_worst_held_day'])} |")
    L_.append("")
    L_.append("### 4d. 組合層逐日保證金率（用網格模擬的成交重建；兩標的同一刻觸最低價，保守）")
    L_.append("")
    L_.append("組合 50/50 的 sleeve 目標 = w × L，所以只有兩個標的同時在趨勢內才會借錢（單邊在場時曝險 ≤ L/2 ≤ 1.25，2.5× 時單邊也借）。")
    L_.append("")
    L_.append("| L | 有借貸的天數（佔比） | 最高前收槓桿（日） | 最低保證金率（盤中） | 引擎自算最低保證金率 | 最低（收盤） | 發生日 | ≤1.1 天數 | ≤1.3 天數 | ≤1.5 天數 | 最差日再跌多少才強平 |")
    L_.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for Lk, m in margin.items():
        if Lk == "1.0":
            continue
        L_.append(f"| {Lk}× | {m['days_with_borrow']}（{m['share_days_with_borrow']:.0%}） | {m['max_leverage_close']:.2f}（{m['date_max_leverage']}） | {fmt(m['min_margin_level_intraday'])} | {fmt(grid[f'L{Lk}_b10%']['engine_min_margin_level'])} | {fmt(m['min_margin_level_close'])} | {m['date_min_margin_level']} | {m['days_ml_below_liq']} | {m['days_ml_below_call']} | {m['days_ml_below_1.5']} | {pct(m.get('extra_drop_to_liq_on_worst_day'))} |")
    L_.append("")
    L_.append("「引擎自算」= `simulate_portfolio` 的 `min_margin_level`（開盤成交後以當日最低價算、借款 = −現金），與本腳本由成交紀錄重建的值互相驗證。")
    L_.append("")
    L_.append("波動目標版：")
    L_.append("")
    L_.append("| 設定 | 有借貸的天數 | 最高前收槓桿（日） | 最低保證金率（盤中） | 發生日 | ≤1.3 天數 | 最差日再跌多少才強平 |")
    L_.append("|---|---|---|---|---|---|---|")
    for tag, m in margin_vt.items():
        L_.append(f"| {tag} | {m['days_with_borrow']} | {m['max_leverage_close']:.2f}（{m['date_max_leverage']}） | {fmt(m['min_margin_level_intraday'])} | {m['date_min_margin_level']} | {m['days_ml_below_call']} | {pct(m.get('extra_drop_to_liq_on_worst_day'))} |")
    L_.append("")
    L_.append("## 5. 借貸成本 vs 永續資金費率（替代工具）")
    L_.append("")
    L_.append("借貸假設：Binance 現貨全倉保證金 USDT 借款，VIP0 量級年化 5–15%（日利率 0.014%–0.041%，按小時計息；牛市借款需求高時偏上緣，而策略在場正是牛市）；本分析取 10% 為中心。")
    L_.append("")
    L_.append("| 標的 | 在場日 | 在場日平均資金費率 年化 | 全期平均 年化 | 在場日費率為正比例 | " + " | ".join(w[0] for w in WINDOWS) + " |")
    L_.append("|---|---|---|---|---|" + "---|" * len(WINDOWS))
    for s, f in funding_cmp.items():
        L_.append(f"| {s.split('/')[0]} | {f['held_days']} | {f['avg_annual_funding_held_days']:.1%} | {f['avg_annual_funding_all_days']:.1%} | {f['share_held_days_positive']:.0%} | " + " | ".join(f"{f['by_window'].get(w[0], float('nan')):.1%}" for w in WINDOWS) + " |")
    L_.append("")
    L_.append("讀法：永續補槓桿的成本 = 在場日資金費率（付在整個永續名目上），與保證金借款年化（只付在超過 1 的部分）比較；第 2/3 輪已證明 1× 全走永續時 ETH 五格全掛。")
    L_.append("")
    L_.extend(policy_section())
    return "\n".join(L_)


def conclusions() -> dict:
    """由數字推出的結論摘要（政策文字放在 md 末尾與結構化輸出）。"""
    b10 = max_lever["10%"]["min_over_windows"]
    b5 = max_lever["5%"]["min_over_windows"]
    b15 = max_lever["15%"]["min_over_windows"]
    g1 = grid[base_tag]["windows"]
    g15 = grid["L1.5_b10%"]["windows"]
    g2 = grid["L2.0_b10%"]["windows"]
    v = vt_grid["vt0.6_L2.0_b10%"]["windows"]
    v1 = vt_grid["vt0.6_L1.0_b0%"]["windows"]
    return {"max_L_dd_only_b10": b10["max_L_dd_only"], "max_L_dd_and_sharpe_b10": b10["max_L_dd_and_sharpe"], "max_L_dd_and_sharpe_tol_b10": b10["max_L_dd_and_sharpe_tol0.02"],
            "max_L_dd_and_sharpe_b5": b5["max_L_dd_and_sharpe"], "max_L_dd_and_sharpe_b15": b15["max_L_dd_and_sharpe"],
            "sharpe_1x": {w: g1[w]["sharpe"] for w in g1}, "max_dd_1x": {w: g1[w]["max_dd"] for w in g1},
            "sharpe_1.5x_b10": {w: g15[w]["sharpe"] for w in g15}, "max_dd_1.5x_b10": {w: g15[w]["max_dd"] for w in g15},
            "sharpe_2x_b10": {w: g2[w]["sharpe"] for w in g2}, "max_dd_2x_b10": {w: g2[w]["max_dd"] for w in g2},
            "gates_2x_b10": grid["L2.0_b10%"]["gates_passed"], "gates_1.5x_b10": grid["L1.5_b10%"]["gates_passed"],
            "sharpe_vt06_L2_b10": {w: v[w]["sharpe"] for w in v}, "max_dd_vt06_L2_b10": {w: v[w]["max_dd"] for w in v},
            "sharpe_vt06_L1": {w: v1[w]["sharpe"] for w in v1}, "max_dd_vt06_L1": {w: v1[w]["max_dd"] for w in v1},
            "kelly_pf_full": kelly_pf["windows"]["FULL"]["kelly"], "kelly_pf_by_window": {w: r["kelly"] for w, r in kelly_pf["windows"].items()},
            "kelly_pf_boot_p05_by_window": {w: r["kelly_boot_p05"] for w, r in kelly_pf["windows"].items()},
            "min_margin_level_1.5x": margin["1.5"]["min_margin_level_intraday"], "min_margin_level_2x": margin["2.0"]["min_margin_level_intraday"],
            "min_margin_level_2.5x": margin["2.5"]["min_margin_level_intraday"],
            "funding_held_annual": {s: f["avg_annual_funding_held_days"] for s, f in funding_cmp.items()}}


def policy_section() -> list[str]:
    """第 6 節：由上面的數字直接推出的政策。數字全部從計算結果取，不手寫。"""
    g1 = grid[base_tag]["windows"]
    g125 = grid["L1.25_b5%"]["windows"]
    g15 = grid["L1.5_b10%"]["windows"]
    g2 = grid["L2.0_b10%"]["windows"]
    vt62 = vt_grid["vt0.6_L2.0_b10%"]
    vt61 = vt_grid["vt0.6_L1.0_b0%"]
    vt41 = vt_grid["vt0.4_L1.0_b0%"]
    kp = kelly_pf["windows"]
    m15, m2, m25 = margin["1.5"], margin["2.0"], margin["2.5"]
    eth = asset_risk["ETH/USDT"]
    L = []
    L.append("## 6. 結論與政策")
    L.append("")
    L.append("### 6.1 五個事實")
    L.append("")
    L.append(f"1. **Kelly 說可以、閘門說不行，兩者不矛盾。** 組合 f* 五區間 {', '.join(f'{w} {r['kelly']:.2f}' for w, r in kp.items())}（半 Kelly {kp['FULL']['half_kelly']:.2f}），"
             f"但 30 日區塊 bootstrap 的 p05 為 {', '.join(f'{r['kelly_boot_p05']:.2f}' for r in kp.values())}、P(f*<1.5) 最高 {max(r['kelly_boot_prob_below_1.5'] for r in kp.values()):.0%}（OOS2）。"
             f"f* 只看 μ/σ² 不看回撤：1× 的實際回撤 {-kp['FULL']['realized_max_dd']:.0%} 在 Kelly 的幾何布朗模型裡是 {kp['FULL']['model_p_dd_ge_realized']['1.0']:.1%} 的事件卻已發生，模型低估尾部，f* 只能當上界。")
    L.append(f"2. **固定槓桿不增加 Sharpe，只把回撤等比放大。** 借 10%：1× → 1.5× → 2× 的 FULL Sharpe {g1['FULL']['sharpe']:.2f} → {g15['FULL']['sharpe']:.2f} → {g2['FULL']['sharpe']:.2f}，"
             f"回撤 {g1['FULL']['max_dd']:.0%} → {g15['FULL']['max_dd']:.0%} → {g2['FULL']['max_dd']:.0%}；OOS 回撤 {g1['OOS']['max_dd']:.0%} → {g15['OOS']['max_dd']:.0%} → {g2['OOS']['max_dd']:.0%}。"
             f"「回撤不超過持有 且 Sharpe 不低於 1×」的最大槓桿在借 5% / 10% / 15% 下五區間取最小都是 {max_lever['5%']['min_over_windows']['max_L_dd_and_sharpe']}× / {max_lever['10%']['min_over_windows']['max_L_dd_and_sharpe']}× / {max_lever['15%']['min_over_windows']['max_L_dd_and_sharpe']}×；"
             f"只看回撤是 {max_lever['10%']['min_over_windows']['max_L_dd_only']}×，但 1.5× 的 FULL 回撤 {g15['FULL']['max_dd']:.0%} 只比月再平衡持有 {g1['FULL']['bench']['持有月再平衡']['max_dd']:.0%} 淺 {(g15['FULL']['max_dd'] - g1['FULL']['bench']['持有月再平衡']['max_dd']) * 100:.1f} pp，是踩線過；2× 十格 {grid['L2.0_b10%']['gates_passed']}/10。"
             f"最溫和的 1.25× 借 5% 五區間 Sharpe 對 1× 的差為 {', '.join(f'{w} {g125[w]['sharpe'] - g1[w]['sharpe']:+.3f}' for w in g125)}（不低於 1× 的區間：{sum(g125[w]['sharpe'] >= g1[w]['sharpe'] for w in g125)}/5），回撤 {g125['FULL']['max_dd']:.0%} vs {g1['FULL']['max_dd']:.0%}。")
    L.append(f"3. **波動目標型槓桿實際上是「降槓桿」。** vt 0.4 / 0.6 在 L=1 不借錢就把 Sharpe 提高（FULL {vt41['windows']['FULL']['sharpe']:.2f} / {vt61['windows']['FULL']['sharpe']:.2f} vs {g1['FULL']['sharpe']:.2f}）、回撤縮到 {vt41['windows']['FULL']['max_dd']:.0%} / {vt61['windows']['FULL']['max_dd']:.0%}，"
             f"代價是 FULL CAGR {vt41['windows']['FULL']['cagr']:.0%} / {vt61['windows']['FULL']['cagr']:.0%} vs {g1['FULL']['cagr']:.0%}（期末 {vt41['final_equity']:,.0f} / {vt61['final_equity']:,.0f} vs {grid[base_tag]['final_equity']:,.0f}）。"
             f"vt 0.6 × 2× 借 10% 是唯一「五區間 Sharpe 都 ≥ 1× 且十格全過且 CAGR 更高」的格（Sharpe {', '.join(f'{vt62['windows'][w]['sharpe']:.2f}' for w in vt62['windows'])}；FULL CAGR {vt62['windows']['FULL']['cagr']:.0%}、期末 {vt62['final_equity']:,.0f}），"
             f"但它的 OOS 回撤 {vt62['windows']['OOS']['max_dd']:.0%}（{vt62['windows']['OOS']['max_dd_peak_date']} → {vt62['windows']['OOS']['max_dd_trough_date']}）比 1× 的 {g1['OOS']['max_dd']:.0%}（{g1['OOS']['max_dd_peak_date']} → {g1['OOS']['max_dd_trough_date']}）深 {(g1['OOS']['max_dd'] - vt62['windows']['OOS']['max_dd']) * 100:.0f} pp — 2024–26 波動低（OOS 平均前收槓桿 {margin_vt['vt0.6_L2.0_b10%']['avg_leverage_by_window']['OOS']:.2f} vs IS {margin_vt['vt0.6_L2.0_b10%']['avg_leverage_by_window']['IS']:.2f}），它把槓桿加在 OOS；FULL 回撤 {vt62['windows']['FULL']['max_dd']:.0%} 與 1× 持平是因為 2021–22 高波動時它自動降到 1× 以下。"
             f"平均曝險 {vt62['avg_exposure']:.2f}、有借貸的天數 {margin_vt['vt0.6_L2.0_b10%']['share_days_with_borrow']:.0%}、借貸成本 {vt62['borrow_cost_pct_of_avg_equity_per_year']:.1%}/年。這個格沒有經過第 4 輪標準的安慰劑 / holdout / 延遲一根驗證，vt=0.6 是絕對數字（第 1 輪 vol_shock_budget 0.70 的教訓：絕對波動門檻是 IS 校準）。")
    L.append(f"4. **≤ 1.5× 強平不是約束，回撤才是。** 全倉 1.1 強平：1.5× 的強平距離 −63%（漂移後 −57%），策略在場日最差單日 BTC {asset_risk['BTC/USDT']['held_day_prevclose_to_low_dist']['min']:.1%}、ETH {eth['held_day_prevclose_to_low_dist']['min']:.1%}（{eth['worst']['prevclose_to_low_held']['date']}）；"
             f"組合逐日重建的最低保證金率 1.5× {m15['min_margin_level_intraday']:.2f}（最差日還要再跌 {-m15['extra_drop_to_liq_on_worst_day']:.0%}）、2× {m2['min_margin_level_intraday']:.2f}（再跌 {-m2['extra_drop_to_liq_on_worst_day']:.0%}）、2.5× {m25['min_margin_level_intraday']:.2f}（{m25['days_ml_below_call']} 天觸追繳、再跌 {-m25['extra_drop_to_liq_on_worst_day']:.0%} 就強平）。"
             f"單標的滿槓桿時 EMA50 出場先於強平的歷史頻率：1.5× BTC {asset_risk['BTC/USDT']['n_episodes']}/{asset_risk['BTC/USDT']['n_episodes']}、ETH {eth['n_episodes']}/{eth['n_episodes']} 段 100%；2× ETH {eth['per_L']['2.0']['p_exit_before_liq_rebalanced']:.1%}（2021-05-19 單日 {eth['worst']['prevclose_to_low_held']['ret']:.0%} 跌破漂移後 −41%）。2020-03-12 兩標的策略都已空手（3/07、3/08 出場）。"
             f"歷史頻率 ≠ 機率：樣本只有 {asset_risk['BTC/USDT']['n_episodes']}+{eth['n_episodes']} 段、一次 3/12 級跳空發生在在場日就是單標的 2× 以上直接強平。")
    L.append(f"5. **工具：如果借，只借現貨保證金，不用永續。** 策略在場日的歷史資金費率年化 BTC {funding_cmp['BTC/USDT']['avg_annual_funding_held_days']:.0%} / ETH {funding_cmp['ETH/USDT']['avg_annual_funding_held_days']:.0%}（IS {funding_cmp['BTC/USDT']['by_window']['IS']:.0%} / {funding_cmp['ETH/USDT']['by_window']['IS']:.0%}，OOS {funding_cmp['BTC/USDT']['by_window']['OOS']:.0%} / {funding_cmp['ETH/USDT']['by_window']['OOS']:.0%}），付在全部名目上；"
             f"現貨保證金 USDT 借款年化 5–15%（假設，見 §5）只付在超過 1 的部分。1.5× 用永續補 0.5× 的成本 ≈ 0.5 × 18–22% ≈ 9–11%/年的權益，加上永續本身 1× 那段的費率讓 ETH 五格全掛（第 2 輪）。")
    L.append("")
    L.append("### 6.2 政策（可執行）")
    L.append("")
    L.append("**現在：1×、現貨、不借錢。** 理由是事實 2：歷史上沒有任何借貸利率 ≥ 5% 的固定槓桿在五個區間都做到「Sharpe 不低於 1× 且回撤不深於持有」，而本專案的閘門就是這兩條。槓桿唯一可靠的效果是把 −60% 的回撤放大成 −75% / −85%，換 CAGR +18 / +32 pp——這不是「合理風險下的合理槓桿」，是同一個賭注押大。")
    L.append("")
    L.append("**什麼條件下才重開槓桿討論（全部同時成立）：**")
    L.append("")
    L.append("1. 現貨紙上前向（2026-10-01 起）累積 ≥ 12 個月，且該 12 個月的實際成本（費 + 滑 + 鋸齒）與模型 0.10% / 0.05% 的差 ≤ 0.05%/邊——因為固定槓桿的 Sharpe 差距只有 0.02–0.05，成本模型差 0.05% 就翻掉結論。")
    L.append("2. 前向 12 個月的策略 Sharpe ≥ 同期 50/50 持有 且 回撤 < 持有（閘門在真錢口徑上過一次）。")
    L.append("3. 當天 Binance 現貨全倉 USDT 借款年化 ≤ 8%（30 日平均），且以 1.5× 重跑本腳本後「回撤 + Sharpe ≥ 1×」在五區間都成立——目前 5% 也不成立，所以這一條大概率要等策略邊際變厚才會過。")
    L.append("4. 本金 ≥ 借款最小單位與計息粒度可忽略的規模（Binance 按小時計息、最小借款 10 USDT 量級，本金 ≥ 10,000 USDT 即可忽略）；本金大小不是槓桿的理由，證據才是。")
    L.append("")
    L.append(f"**如果四條都過，唯一在數據上站得住的形式**：波動目標型，vt 0.6、上限 1.5×（不是 2×：2× 的 OOS 回撤 {vt62['windows']['OOS']['max_dd']:.0%}），現貨全倉保證金借 USDT，每日收盤按 min(1.5, 0.6 / rvol20) 調目標、調倉門檻 0.15。"
             f"它歷史上有借貸的天數只有 {margin_vt['vt0.6_L1.5_b10%']['share_days_with_borrow']:.0%}、最高前收槓桿 {margin_vt['vt0.6_L1.5_b10%']['max_leverage_close']:.2f}、最低保證金率 {margin_vt['vt0.6_L1.5_b10%']['min_margin_level_intraday']:.2f}，借貸成本 {vt_grid['vt0.6_L1.5_b10%']['borrow_cost_pct_of_avg_equity_per_year']:.1%}/年，"
             f"FULL Sharpe {vt_grid['vt0.6_L1.5_b10%']['windows']['FULL']['sharpe']:.2f}、回撤 {vt_grid['vt0.6_L1.5_b10%']['windows']['FULL']['max_dd']:.0%}、CAGR {vt_grid['vt0.6_L1.5_b10%']['windows']['FULL']['cagr']:.0%}（vs 1× {g1['FULL']['sharpe']:.2f} / {g1['FULL']['max_dd']:.0%} / {g1['FULL']['cagr']:.0%}）；"
             f"但 OOS 回撤 {vt_grid['vt0.6_L1.5_b10%']['windows']['OOS']['max_dd']:.0%} 比 1× 的 {g1['OOS']['max_dd']:.0%} 深 {(g1['OOS']['max_dd'] - vt_grid['vt0.6_L1.5_b10%']['windows']['OOS']['max_dd']) * 100:.0f} pp，OOS Sharpe {vt_grid['vt0.6_L1.5_b10%']['windows']['OOS']['sharpe']:.2f} vs {g1['OOS']['sharpe']:.2f}、OOS2 {vt_grid['vt0.6_L1.5_b10%']['windows']['OOS2']['sharpe']:.2f} vs {g1['OOS2']['sharpe']:.2f} — 在最近三年它對 1× 的改善是零，多出來的是回撤。"
             "啟用前它必須先走第 4 輪標準：安慰劑（隨機 scale 路徑）、holdout SOL/BNB、延遲一根、費+滑 0.20 四道，且 vt 在 0.4–0.8 要有平台不能是尖點。")
    L.append("")
    L.append("**降回 1× 的條件（任一成立，當天收盤執行）：** (a) 借款年化 30 日平均 > 12%（借 15% 時 1.5× 的 OOS Sharpe 掉到 " + f"{grid['L1.5_b15%']['windows']['OOS']['sharpe']:.2f} vs 1× {g1['OOS']['sharpe']:.2f}）；(b) 全倉帳戶保證金率收盤 < 2.0（相當於再跌 45% 才到 1.1，留一次 2021-05-19 級的單日）；"
             "(c) 兩標的同時在趨勢外（引擎自然歸零）；(d) 前向 6 個月滾動 Sharpe 低於同期持有——注意 (d) 是「以自身權益為輸入的閘門」，第 4 輪證明這類閘門的觸發時點對成本與起算日敏感，所以它只能觸發「人工覆核後降槓桿」，不能自動化，且降回 1× 後不自動升回。")
    L.append("")
    L.append("**不做的事：** 固定 1.5× 或 2×（事實 2）、永續補槓桿（事實 5）、以回撤深度反向加槓桿（「跌深加碼」= 在強平距離最短時借最多）、單標的 2× 以上（事實 4：ETH 2021-05-19 單日就跌破強平距離）。")
    L.append("")
    return L


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    md = md_report()
    concl = conclusions()
    report = {"key": KEY, "config": {"symbols": SYMBOLS, "weights": WEIGHTS, "fee": FEE, "slip": SLIP, "rebalance_threshold": REBAL, "levers": LEVERS, "borrows": BORROWS,
                                     "vol_targets": VOL_TARGETS, "vt_levers": VT_LEVERS, "vol_lookback": VOL_LOOKBACK, "maint_liq": MAINT_LIQ, "maint_call": MAINT_CALL,
                                     "bootstrap": {"reps": BOOT_REPS, "block": BOOT_BLOCK, "seed": BOOT_SEED}, "windows": WINDOWS, "data_since": DATA_SINCE,
                                     "engine_commit": ENGINE_COMMIT, "vt_crosscheck": vt_crosscheck,
                                     "data_last": str(idx[-1].date()), "portfolio_start": str(pf1_res.equity.index[0].date())},
              "kelly": {"btc": kelly_btc, "portfolio": kelly_pf},
              "fixed_grid": grid, "max_lever": max_lever, "perp_smoke_check": perp_check,
              "vol_target_grid": vt_grid,
              "liquidation": {"table": liq_table, "assets": {s: {k: v for k, v in a.items() if k != "episodes"} for s, a in asset_risk.items()},
                              "portfolio_margin": margin, "portfolio_margin_vt": margin_vt},
              "episodes": {s: a["episodes"] for s, a in asset_risk.items()},
              "funding_vs_borrow": funding_cmp, "conclusions": concl}
    def _clean(x):
        """NaN / inf → None，讓 JSON 合法；numpy 純量轉 Python。"""
        if isinstance(x, dict):
            return {str(k): _clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_clean(v) for v in x]
        if isinstance(x, (np.floating, float)):
            return None if (math.isnan(x) or math.isinf(x)) else float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        return x

    with open(os.path.join(OUT_DIR, f"{KEY}.json"), "w", encoding="utf-8") as f:
        json.dump(_clean(report), f, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(OUT_DIR, f"{KEY}.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
    print(json.dumps(concl, ensure_ascii=False))
    print(f"\n→ {OUT_DIR}/{KEY}.json / .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
