"""本金曲線（capital lever）：同一條 BTC/ETH 50/50 ema50_plain 現貨曲線，換本金與投入節奏會差多少。

重跑：
    cd D:/feihong/tradingView/ABGT
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/levers/capital.py
輸出：results/levers/capital.json、results/levers/capital.md（本檔不改任何既有檔案）

方法（每一步都能用 abgt 引擎或一支試算表獨立重算）：
1. 策略日權益 E_t：abgt.portfolio.simulate_portfolio，現貨 BTC/USDT + ETH/USDT 日線（Binance spot，2019-09-01 起、EMA50 暖機後起算），
   權重 0.5/0.5、close > EMA50 滿倉否則空手、收盤決策次日開盤成交、費 0.10%/邊、滑 0.05%、調倉門檻 0.15、起始 10,000。
   同權重買入持有（不再平衡）B_t：abgt.portfolio.buy_and_hold_portfolio（同起點、同費率）。
2. 定期投入用「份額法」：投入日 d = 起始日 + 之後每月第一根日 K 線；以 d 收盤的單位淨值買份額。
   投入前先扣一次進場成本 c = (費 + 滑) × 當日在場比例（策略：0.5×on_BTC[d] + 0.5×on_ETH[d]，用 d 收盤的 regime，因為新錢會在次日開盤部署；持有：1.0）。
   帳戶價值 V_t = 份額 × 單位淨值、投入總額 I_t、損益 P_t = V_t − I_t。
   - 金錢加權 IRR：現金流 −投入（各投入日）、+V_end（期末），年化 XIRR，二分法求解
   - 時間加權 TWR：單位淨值期末 / 期初 − 1，年化 = CAGR（與投入節奏無關，就是策略自身在該區間的報酬）
   - 最深美元回撤：P_t 自其高水位的最大跌幅（把入金排除在外，只量市場造成的損失）；帳戶 % = 美元回撤 / 高水位當日 V；單位淨值 % = 策略自身路徑回撤
   - 水下最長天數：P_t 低於前高的最長連續日數（含入金的帳戶經驗）
3. 「本金 × 年化」：NAV = 本金 × (1 + CAGR)^年，是 CAGR 複利的點估計（≈ 中位路徑，不是平均）；最壞回撤金額 = NAV × |該情境的歷史最大回撤|。
   另附 30 日區塊 bootstrap（固定種子）給 p10 / p50 / p90 分布。
4. 到 30 萬要幾年：月複利 m = (1 + CAGR)^(1/12) − 1，月末投入 X，解 FV(n) = P(1+m)^n + X((1+m)^n − 1)/m ≥ 300,000。
5. 回撤加碼對照：投入日若單位淨值距高水位回撤 ≥ 20%（用當日收盤，無未來資料）該月投入 2×；「預算配對」組把 2× 版的總投入平均攤回每月，
   把「時點」與「多投的錢」分開。策略看策略自己的回撤、持有看持有自己的回撤。
6. 參數 vs 本金：同一引擎換 ema_len ∈ {30, 38, 50, 62, 70}（±25% 是專案敏感度慣例、±40% 更寬），比較「參數造成的期末 NAV 範圍」與
   「本金 / 投入節奏造成的範圍」。
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

from abgt import data as dt                                                   # noqa: E402
from abgt.metrics import max_drawdown, summarize                              # noqa: E402
from abgt.portfolio import align, buy_and_hold_portfolio, simulate_portfolio  # noqa: E402
from abgt.strategies.regime import RiskParams, regime_flags                   # noqa: E402
from evaluate_variant import load_variant                                     # noqa: E402

SYMBOLS = ["BTC/USDT", "ETH/USDT"]
WEIGHTS = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
FEE, SLIP, REBAL = 0.001, 0.0005, 0.15
COST = FEE + SLIP
DATA_SINCE = "2019-09-01"
INITIAL = 10_000.0
WINDOWS = [("IS", "2020-01-01", "2022-12-31"), ("OOS", "2023-01-01", None), ("FULL", "2020-01-01", None),
           ("IS2", "2020-07-01", "2023-12-31"), ("OOS2", "2024-01-01", None)]
STARTS = [("2020-01", "2020-01-01"), ("2021-11（頂部）", "2021-11-01"), ("2023-01", "2023-01-01")]
CAPITALS = [10_000, 50_000, 300_000]
MONTHLY = [0, 1000, 3000]
EMA_GRID = [30, 38, 50, 62, 70]
TABLE_CAPITALS = [10_000, 50_000, 100_000, 300_000, 1_000_000]
TABLE_YEARS = [1, 3, 5, 10]
TARGET = 300_000
TARGET_MONTHLY = [0, 500, 1000, 2000, 3000, 5000]
DD_THRESHOLDS = [0.2, 0.3, 0.4]
BOOT_PATHS, BOOT_BLOCK, BOOT_SEED = 1000, 30, 42
OUT_DIR = os.path.join(ROOT, "results", "levers")


# ---------------------------------------------------------------- 引擎曲線

def build_portfolio(dfs: dict, idx: pd.DatetimeIndex, ema_len: int) -> dict:
    mod = load_variant("ema50_plain")
    regimes, warm_times, on = {}, [], {}
    for s in SYMBOLS:
        built = mod.build(dfs[s], {"ema_len": ema_len})
        regimes[s] = built["in_regime"]
        warm_times.append(dfs[s].index[int(built["warmup"])])
        on_s, _ = regime_flags(built["in_regime"])
        on[s] = pd.Series(on_s, index=dfs[s].index).reindex(idx).fillna(False).astype(bool)
    warmup = max(int(idx.searchsorted(t)) for t in warm_times)
    rp = RiskParams(commission_pct=FEE, slippage_pct=SLIP, rebalance_threshold=REBAL, initial_capital=INITIAL)
    res = simulate_portfolio(dfs, regimes, rp, WEIGHTS, warmup)
    bh = buy_and_hold_portfolio(dfs, WEIGHTS, idx, warmup, INITIAL, FEE, None)
    in_frac = sum(WEIGHTS[s] * on[s].astype(float) for s in SYMBOLS).iloc[warmup:]
    return {"equity": res.equity, "bh": bh, "in_frac": in_frac, "warmup": warmup, "ema_len": ema_len,
            "n_fills": int(res.extra["n_fills"]), "round_trips": int(len(res.round_trips)),
            "time_in_market": float(res.extra["time_in_market"]), "avg_exposure": float(res.extra["avg_exposure"])}


def slice_window(eq: pd.Series, a: str, b: str | None) -> pd.Series:
    e = eq.loc[a:] if b is None else eq.loc[a:b]
    return e / e.iloc[0] * INITIAL


def window_stats(eq: pd.Series) -> dict:
    out = {}
    for name, a, b in WINDOWS:
        s = summarize(slice_window(eq, a, b))
        out[name] = {"start": str(s["start"].date()), "end": str(s["end"].date()), "years": float(s["years"]),
                     "net_return": float(s["net_return"]), "cagr": float(s["cagr"]), "max_dd": float(s["max_dd"]),
                     "sharpe": float(s["sharpe"]), "calmar": float(s["calmar"])}
    return out


# ---------------------------------------------------------------- 定期投入（份額法）

def contribution_days(index: pd.DatetimeIndex) -> np.ndarray:
    ym = np.array([d.year * 12 + d.month for d in index])
    flag = np.zeros(len(index), dtype=bool)
    flag[0] = True
    flag[1:] = ym[1:] != ym[:-1]
    return flag


def xirr(dates: list, cfs: list) -> float:
    t0 = dates[0]
    yrs = np.array([(d - t0).days / 365.25 for d in dates], dtype="float64")
    cf = np.array(cfs, dtype="float64")

    def npv(r: float) -> float:
        return float(np.sum(cf / (1.0 + r) ** yrs))

    lo, hi = -0.9999, 50.0
    if npv(lo) < 0 or npv(hi) > 0:
        return math.nan
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return int(best)


def simulate_dca(unit: pd.Series, in_frac: pd.Series | float, start: str, p0: float, monthly: float,
                 rule: str = "fixed", dd_thresh: float = 0.2, mult: float = 2.0) -> dict:
    """unit：單位淨值（任何正的日序列）；in_frac：投入日在場比例（Series 或常數）。"""
    u = unit.loc[start:]
    u = u / u.iloc[0]
    ua = u.to_numpy(dtype="float64")
    n = len(ua)
    f = (in_frac.reindex(u.index).fillna(0.0).to_numpy(dtype="float64") if isinstance(in_frac, pd.Series)
         else np.full(n, float(in_frac)))
    days = contribution_days(u.index)
    hwm = np.maximum.accumulate(ua)
    unit_dd = ua / hwm - 1.0
    units = 0.0
    invested = np.zeros(n)
    value = np.zeros(n)
    cf_dates, cf_amts = [], []
    boosted, boosted_frac = 0, []
    for i in range(n):
        amt = 0.0
        if i == 0:
            amt = p0
        elif days[i] and monthly > 0:
            amt = monthly
            if rule == "dd2x" and unit_dd[i] <= -dd_thresh:
                amt = monthly * mult
                boosted += 1
                boosted_frac.append(f[i])
        if amt > 0:
            units += amt * (1.0 - COST * f[i]) / ua[i]
            cf_dates.append(u.index[i])
            cf_amts.append(-amt)
        invested[i] = (invested[i - 1] if i else 0.0) + amt
        value[i] = units * ua[i]
    pnl = value - invested
    pnl_peak = np.maximum.accumulate(pnl)
    dd_usd = pnl_peak - pnl
    i_trough = int(np.argmax(dd_usd))
    i_peak = int(np.argmax(pnl[: i_trough + 1])) if i_trough > 0 else 0
    years = (u.index[-1] - u.index[0]).days / 365.25
    n_contrib = int(days[1:].sum()) if monthly > 0 else 0
    return {
        "start": str(u.index[0].date()), "end": str(u.index[-1].date()), "years": float(years), "days": int(n),
        "p0": float(p0), "monthly": float(monthly), "rule": rule, "dd_thresh": float(dd_thresh) if rule == "dd2x" else None,
        "n_contrib_months": n_contrib, "boosted_months": int(boosted),
        "boosted_in_market_frac": float(np.mean(boosted_frac)) if boosted_frac else None,
        "invested": float(invested[-1]), "final_nav": float(value[-1]), "profit": float(pnl[-1]),
        "multiple_on_invested": float(value[-1] / invested[-1]) if invested[-1] > 0 else math.nan,
        "irr": xirr(cf_dates + [u.index[-1]], cf_amts + [float(value[-1])]),
        "twr_total": float(ua[-1] - 1.0), "twr_cagr": float(ua[-1] ** (1.0 / years) - 1.0) if years > 0 else math.nan,
        "max_dd_usd": float(dd_usd[i_trough]), "max_dd_usd_peak_date": str(u.index[i_peak].date()),
        "max_dd_usd_trough_date": str(u.index[i_trough].date()),
        "max_dd_pct_account": float(-dd_usd[i_trough] / value[i_peak]) if value[i_peak] > 0 else math.nan,
        "max_dd_pct_unit": float(unit_dd.min()),
        "underwater_days_pnl": longest_run(dd_usd > 1e-9), "underwater_days_unit": longest_run(unit_dd < -1e-12),
    }


# ---------------------------------------------------------------- 本金 × 年化、bootstrap、到 30 萬

def years_to_target(p0: float, monthly: float, g: float, target: float = TARGET) -> float:
    if p0 >= target:
        return 0.0
    if g <= 0 and monthly <= 0:
        return math.inf
    m = (1.0 + g) ** (1.0 / 12.0) - 1.0
    if monthly == 0:
        n = math.log(target / p0) / math.log1p(m)
    elif abs(m) < 1e-12:
        n = (target - p0) / monthly
    else:
        n = math.log((target + monthly / m) / (p0 + monthly / m)) / math.log1p(m)
    return n / 12.0


def block_bootstrap(rets: np.ndarray, years_list: list, n_paths: int, block: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    L = len(rets)
    out = {}
    for T in years_list:
        n = int(round(T * 365))
        nb = math.ceil(n / block)
        mults = np.empty(n_paths)
        mdds = np.empty(n_paths)
        for p in range(n_paths):
            starts = rng.integers(0, L - block + 1, size=nb)
            path = np.concatenate([rets[s:s + block] for s in starts])[:n]
            eqp = np.cumprod(1.0 + path)
            mults[p] = eqp[-1]
            mdds[p] = max_drawdown(np.concatenate([[1.0], eqp]))
        # mdd 是負數：mdd_p90_deep = 第 10 百分位（90% 的路徑回撤比它淺）
        out[str(T)] = {"mult_p10": float(np.percentile(mults, 10)), "mult_p50": float(np.percentile(mults, 50)),
                       "mult_p90": float(np.percentile(mults, 90)), "prob_loss": float(np.mean(mults < 1.0)),
                       "mdd_p50": float(np.percentile(mdds, 50)), "mdd_p90_deep": float(np.percentile(mdds, 10)),
                       "mdd_worst": float(mdds.min())}
    return out


# ---------------------------------------------------------------- 格式

def money(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.0f}"


def pct(x: float, nd: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else f"{x * 100:+.{nd}f}%"


def num(x: float, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    return f"{x:.{nd}f}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------- 主流程

def main() -> int:
    os.chdir(ROOT)
    dfs = {s: dt.load_cached("data", s, "1d", DATA_SINCE) for s in SYMBOLS}
    if any(v is None for v in dfs.values()):
        print("缺現貨快取 data/BTCUSDT_1d.csv / data/ETHUSDT_1d.csv")
        return 2
    idx = align(dfs)
    base = build_portfolio(dfs, idx, 50)
    eq, bh, in_frac = base["equity"], base["bh"], base["in_frac"]
    ws_strat, ws_bh = window_stats(eq), window_stats(bh)

    out: dict = {
        "key": "capital",
        "setup": {"symbols": SYMBOLS, "weights": WEIGHTS, "fee": FEE, "slip": SLIP, "rebalance_threshold": REBAL,
                  "data_since": DATA_SINCE, "engine_start": str(eq.index[0].date()), "engine_end": str(eq.index[-1].date()),
                  "engine_initial": INITIAL, "engine_final_equity": float(eq.iloc[-1]), "bh_final_equity": float(bh.iloc[-1]),
                  "n_fills": base["n_fills"], "round_trips": base["round_trips"], "time_in_market": base["time_in_market"],
                  "avg_exposure": base["avg_exposure"], "contribution_cost_rate": COST,
                  "contribution_timing": "起始日與之後每月第一根日 K 線，以當日收盤單位淨值買份額，先扣 (費+滑)×在場比例"},
        "windows": {"strategy": ws_strat, "buy_and_hold": ws_bh},
    }

    lines = []
    lines.append(f"# 本金曲線（capital lever）— BTC/ETH 50/50 ema50_plain 現貨\n")
    lines.append(f"引擎：現貨 BTC/USDT + ETH/USDT 日線、費 {FEE:.2%}/邊、滑 {SLIP:.2%}、調倉門檻 {REBAL}、起算 {eq.index[0].date()} → {eq.index[-1].date()}（10,000 起始 → 策略 {money(float(eq.iloc[-1]))}、持有 {money(float(bh.iloc[-1]))}）；"
                 f"成交 {base['n_fills']} 筆、回合 {base['round_trips']}、在場 {base['time_in_market']:.0%}、平均曝險 {base['avg_exposure']:.2f}。定期投入以份額法計、投入日扣 (費+滑) × 在場比例（見腳本 docstring）。\n")

    # 1. 視窗指標
    lines.append("## 1. 組合曲線視窗指標（策略 vs 同權重持有不再平衡）\n")
    rows = []
    for w, _, _ in WINDOWS:
        s, b = ws_strat[w], ws_bh[w]
        rows.append([w, f"{s['start']}→{s['end']}", pct(s["cagr"]), pct(s["max_dd"]), num(s["sharpe"]), num(s["calmar"]),
                     pct(b["cagr"]), pct(b["max_dd"]), num(b["sharpe"]), num(b["calmar"])])
    lines.append(md_table(["區間", "起訖", "策略 CAGR", "策略回撤", "Sharpe", "Calmar", "持有 CAGR", "持有回撤", "Sharpe", "Calmar"], rows) + "\n")

    # 2. 定期投入 54 格
    lines.append("## 2. 定期投入模擬：起始本金 × 每月投入 × 起點，策略 vs 持有\n")
    lines.append("IRR = 金錢加權年化；TWR = 時間加權年化（= 該區間策略自身 CAGR，與投入無關）；最深 $ 回撤 = 損益曲線（NAV − 累計投入）自高水位的最大跌幅；帳戶 % = 該金額 / 高水位當日 NAV；單位 % = 策略自身路徑最大回撤；水下 = 損益低於前高的最長連續天數。\n")
    dca_rows = []
    for sname, sdate in STARTS:
        for p0 in CAPITALS:
            for m in MONTHLY:
                r_s = simulate_dca(eq, in_frac, sdate, p0, m)
                r_b = simulate_dca(bh, 1.0, sdate, p0, m)
                dca_rows.append({"start_label": sname, "start": sdate, "p0": p0, "monthly": m, "strategy": r_s, "buy_and_hold": r_b,
                                 "nav_ratio_strategy_over_hold": r_s["final_nav"] / r_b["final_nav"]})
    out["dca_grid"] = dca_rows
    for sname, _ in STARTS:
        rows = []
        for r in dca_rows:
            if r["start_label"] != sname:
                continue
            for lab, key in (("策略", "strategy"), ("持有", "buy_and_hold")):
                x = r[key]
                rows.append([money(r["p0"]), money(r["monthly"]), lab, money(x["invested"]), money(x["final_nav"]), money(x["profit"]),
                             pct(x["irr"]), pct(x["twr_cagr"]), money(x["max_dd_usd"]), pct(x["max_dd_pct_account"]), pct(x["max_dd_pct_unit"]),
                             str(x["underwater_days_pnl"])])
        r0 = next(r for r in dca_rows if r["start_label"] == sname)
        lines.append(f"### 起點 {sname}（{r0['strategy']['start']} → {r0['strategy']['end']}，{r0['strategy']['years']:.2f} 年）\n")
        lines.append(md_table(["本金", "每月", "對象", "投入總額", "期末 NAV", "損益", "IRR", "TWR 年化", "最深 $ 回撤", "帳戶 %", "單位 %", "水下天"], rows) + "\n")

    # 3. 本金 × 年化
    cagr_base = ws_strat["OOS"]["cagr"]
    cagr_cons = cagr_base / 2.0
    cagr_opt = ws_strat["IS"]["cagr"]
    scenarios = {"保守（OOS CAGR 的一半）": (cagr_cons, ws_strat["FULL"]["max_dd"], "FULL"),
                 "基準（OOS 2023→ CAGR）": (cagr_base, ws_strat["OOS"]["max_dd"], "OOS"),
                 "樂觀（IS 2020–22 CAGR）": (cagr_opt, ws_strat["IS"]["max_dd"], "IS")}
    out["scenarios"] = {k: {"cagr": v[0], "max_dd": v[1], "max_dd_window": v[2]} for k, v in scenarios.items()}
    lines.append("## 3. 本金 × 年化：期望 NAV 與可能的最壞回撤金額\n")
    lines.append(f"CAGR：保守 {pct(cagr_cons)}（基準的一半）、基準 {pct(cagr_base)}（OOS 2023→ 策略）、樂觀 {pct(cagr_opt)}（IS 2020–22 策略）。NAV = 本金 × (1+CAGR)^年 是複利點估計（≈ 中位路徑）；最壞回撤金額 = 該 NAV × |歷史最大回撤|（保守用 FULL {pct(ws_strat['FULL']['max_dd'])}、基準用 OOS {pct(ws_strat['OOS']['max_dd'])}、樂觀用 IS {pct(ws_strat['IS']['max_dd'])}），意思是「到了那個水位之後可能吐回去的錢」。\n")
    table = {}
    for scn, (g, mdd, _) in scenarios.items():
        rows = []
        table[scn] = {}
        for c in TABLE_CAPITALS:
            row = [money(c)]
            table[scn][str(c)] = {}
            for T in TABLE_YEARS:
                nav = c * (1 + g) ** T
                table[scn][str(c)][str(T)] = {"nav": nav, "worst_dd_usd": nav * abs(mdd)}
                row.append(f"{money(nav)} / −{money(nav * abs(mdd))}")
            rows.append(row)
        lines.append(f"### {scn}：CAGR {pct(g)}，回撤 {pct(mdd)}\n")
        lines.append(md_table(["本金", *[f"{T} 年 NAV / 最壞回撤" for T in TABLE_YEARS]], rows) + "\n")
    out["capital_x_cagr"] = table

    # 3b. bootstrap
    boot = {}
    for w in ("OOS", "FULL"):
        _, a, b = next(x for x in WINDOWS if x[0] == w)
        rets = slice_window(eq, a, b).pct_change().dropna().to_numpy(dtype="float64")
        boot[w] = block_bootstrap(rets, TABLE_YEARS, BOOT_PATHS, BOOT_BLOCK, BOOT_SEED)
    out["bootstrap"] = {"paths": BOOT_PATHS, "block_days": BOOT_BLOCK, "seed": BOOT_SEED, "by_window": boot}
    lines.append(f"### 3b. 分布（{BOOT_BLOCK} 日區塊 bootstrap × {BOOT_PATHS} 條，種子 {BOOT_SEED}；本金 1 的倍數，乘上本金即可）\n")
    rows = []
    for w in boot:
        for T in TABLE_YEARS:
            x = boot[w][str(T)]
            rows.append([w, str(T), num(x["mult_p10"]), num(x["mult_p50"]), num(x["mult_p90"]), f"{x['prob_loss'] * 100:.0f}%", pct(x["mdd_p50"]), pct(x["mdd_p90_deep"]), pct(x["mdd_worst"])])
    lines.append(md_table(["抽樣來源", "年", "倍數 p10", "p50", "p90", "虧損機率", "最大回撤 p50", "p90（較深）", "最壞"], rows) + "\n")

    # 4. 到 30 萬
    lines.append(f"## 4. 從 10,000 + 每月 X 到 {money(TARGET)} 要幾年（月複利、月末投入）\n")
    rows, t2t = [], {}
    for X in TARGET_MONTHLY:
        row = [money(X)]
        t2t[str(X)] = {}
        for scn, (g, _, _) in scenarios.items():
            y = years_to_target(10_000, X, g)
            t2t[str(X)][scn] = y
            row.append(num(y, 1) if math.isfinite(y) else "∞")
        y0 = years_to_target(10_000, X, 0.0)
        t2t[str(X)]["0%（純存錢）"] = y0
        row.append(num(y0, 1) if math.isfinite(y0) else "∞")
        rows.append(row)
    out["years_to_300k"] = t2t
    lines.append(md_table(["每月投入", *[f"{k.split('（')[0]} {pct(v[0])}" for k, v in scenarios.items()], "0%（純存錢）"], rows) + "\n")

    # 5. 參數 vs 本金
    lines.append("## 5. 本金曲線 vs 策略參數：同一條引擎換 ema_len，與換本金 / 投入節奏比\n")
    param_runs = {}
    for n in EMA_GRID:
        pr = base if n == 50 else build_portfolio(dfs, idx, n)
        ws = window_stats(pr["equity"])
        lump = {}
        for sname, sdate in STARTS:
            r = simulate_dca(pr["equity"], pr["in_frac"], sdate, 10_000, 0)
            lump[sname] = {"final_nav": r["final_nav"], "cagr": r["twr_cagr"], "max_dd_unit": r["max_dd_pct_unit"]}
        param_runs[str(n)] = {"windows": {w: {"cagr": ws[w]["cagr"], "max_dd": ws[w]["max_dd"], "sharpe": ws[w]["sharpe"]} for w in ws},
                              "lump_10k": lump, "n_fills": pr["n_fills"]}
    out["param_runs"] = param_runs
    rows = []
    for n in EMA_GRID:
        pr = param_runs[str(n)]
        rows.append([str(n), *[f"{money(pr['lump_10k'][s]['final_nav'])}（{pct(pr['lump_10k'][s]['cagr'])}）" for s, _ in STARTS],
                     num(pr["windows"]["OOS"]["sharpe"]), pct(pr["windows"]["OOS"]["max_dd"]), num(pr["windows"]["FULL"]["sharpe"]), pct(pr["windows"]["FULL"]["max_dd"])])
    lines.append("10,000 一次投入、不追加，期末 NAV（TWR 年化）：\n")
    lines.append(md_table(["ema_len", *[f"起點 {s}" for s, _ in STARTS], "OOS Sharpe", "OOS 回撤", "FULL Sharpe", "FULL 回撤"], rows) + "\n")

    comp, rows = {}, []
    for sname, _ in STARTS:
        navs25 = [param_runs[str(n)]["lump_10k"][sname]["final_nav"] for n in (38, 50, 62)]
        navs40 = [param_runs[str(n)]["lump_10k"][sname]["final_nav"] for n in EMA_GRID]
        g = {(r["p0"], r["monthly"]): r for r in dca_rows if r["start_label"] == sname}
        base_nav = g[(10_000, 0)]["strategy"]["final_nav"]
        p25 = max(navs25) - min(navs25)
        p40 = max(navs40) - min(navs40)
        m1 = g[(10_000, 1000)]["strategy"]["final_nav"] - base_nav
        m3 = g[(10_000, 3000)]["strategy"]["final_nav"] - base_nav
        c5 = g[(50_000, 0)]["strategy"]["final_nav"] - base_nav
        base_profit = g[(10_000, 0)]["strategy"]["profit"]
        pm1 = g[(10_000, 1000)]["strategy"]["profit"] - base_profit
        pm3 = g[(10_000, 3000)]["strategy"]["profit"] - base_profit
        pc5 = g[(50_000, 0)]["strategy"]["profit"] - base_profit
        # 參數範圍以「損益」量（10k 一次投入，損益 = NAV − 10k，範圍與 NAV 範圍相同）
        comp[sname] = {"param_range_usd_pm25": p25, "param_range_usd_pm40": p40, "param_ratio_pm25": max(navs25) / min(navs25), "param_ratio_pm40": max(navs40) / min(navs40),
                       "monthly1000_delta_nav": m1, "monthly3000_delta_nav": m3, "capital50k_delta_nav": c5,
                       "monthly1000_delta_profit": pm1, "monthly3000_delta_profit": pm3, "capital50k_delta_profit": pc5,
                       "monthly1000_profit_over_param25": pm1 / p25 if p25 > 0 else math.inf, "monthly3000_profit_over_param25": pm3 / p25 if p25 > 0 else math.inf,
                       "capital50k_profit_over_param25": pc5 / p25 if p25 > 0 else math.inf,
                       "monthly1000_profit_over_param40": pm1 / p40 if p40 > 0 else math.inf, "monthly3000_profit_over_param40": pm3 / p40 if p40 > 0 else math.inf,
                       "capital50k_profit_over_param40": pc5 / p40 if p40 > 0 else math.inf,
                       "invested_monthly1000": g[(10_000, 1000)]["strategy"]["invested"], "invested_monthly3000": g[(10_000, 3000)]["strategy"]["invested"]}
        rows.append([sname, money(base_nav), f"{money(p25)}（×{num(max(navs25) / min(navs25))}）", f"{money(p40)}（×{num(max(navs40) / min(navs40))}）",
                     f"+{money(pm1)}（投入 +{money(g[(10_000, 1000)]['strategy']['invested'] - 10_000)}）", f"+{money(pm3)}（投入 +{money(g[(10_000, 3000)]['strategy']['invested'] - 10_000)}）",
                     f"+{money(pc5)}（投入 +40,000）", num(pm1 / p25 if p25 > 0 else math.inf, 1), num(pm3 / p25 if p25 > 0 else math.inf, 1), num(pc5 / p25 if p25 > 0 else math.inf, 1),
                     num(pm1 / p40 if p40 > 0 else math.inf, 1), num(pm3 / p40 if p40 > 0 else math.inf, 1), num(pc5 / p40 if p40 > 0 else math.inf, 1)])
    out["param_vs_capital"] = comp
    lines.append("同一起點、同一條策略，「參數」與「本金 / 節奏」各自能改變多少期末**損益**（NAV − 投入總額，美元；把投入的本金本身扣掉，只比市場賺到的錢）。參數範圍 = 10k 一次投入下 ema_len 極值間的 NAV 差（= 損益差）：\n")
    lines.append(md_table(["起點", "EMA50 10k 期末", "EMA 38–62 範圍", "EMA 30–70 範圍", "每月 +1,000 損益增量", "每月 +3,000 損益增量", "本金 10k→50k 損益增量", "+1,000 / ±25%", "+3,000 / ±25%", "50k / ±25%", "+1,000 / ±40%", "+3,000 / ±40%", "50k / ±40%"], rows) + "\n")

    # 6. 回撤加碼
    lines.append("## 6. 回撤中加碼：投入日單位淨值距高水位回撤 ≥ 門檻 → 該月投入 2×（策略看策略的回撤、持有看持有的回撤）\n")
    lines.append("「預算配對」= 把 2× 版的總投入平均攤回每月固定投入（同樣的錢、不挑時點），用來把「時點」與「多投的錢」分開；比較看 IRR 與每投入 1 元的期末價值。\n")
    dd_rows = []
    for sname, sdate in STARTS:
        for m in (1000, 3000):
            for th in DD_THRESHOLDS:
                rec = {"start_label": sname, "start": sdate, "p0": 10_000, "monthly": m, "dd_thresh": th}
                for lab, key, unit, frac in (("策略", "strategy", eq, in_frac), ("持有", "buy_and_hold", bh, 1.0)):
                    fixed = simulate_dca(unit, frac, sdate, 10_000, m)
                    boost = simulate_dca(unit, frac, sdate, 10_000, m, rule="dd2x", dd_thresh=th)
                    extra = boost["invested"] - fixed["invested"]
                    matched = simulate_dca(unit, frac, sdate, 10_000, (boost["invested"] - 10_000) / max(1, fixed["n_contrib_months"])) if extra > 0 else fixed
                    rec[key] = {"fixed": fixed, "dd2x": boost, "budget_matched": matched,
                                "delta_irr_dd2x_vs_matched": boost["irr"] - matched["irr"],
                                "delta_multiple_dd2x_vs_matched": boost["multiple_on_invested"] - matched["multiple_on_invested"]}
                dd_rows.append(rec)
    out["dd_boost"] = dd_rows
    rows = []
    for rec in dd_rows:
        if rec["dd_thresh"] != 0.2:
            continue
        for lab, key in (("策略", "strategy"), ("持有", "buy_and_hold")):
            x = rec[key]
            f_, b_, m_ = x["fixed"], x["dd2x"], x["budget_matched"]
            rows.append([rec["start_label"], money(rec["monthly"]), lab, f"{b_['boosted_months']}/{f_['n_contrib_months']}",
                         "—" if b_["boosted_in_market_frac"] is None else f"{b_['boosted_in_market_frac']:.0%}",
                         f"{money(f_['invested'])} → {money(f_['final_nav'])}", pct(f_["irr"]), num(f_["multiple_on_invested"]),
                         f"{money(b_['invested'])} → {money(b_['final_nav'])}", pct(b_["irr"]), num(b_["multiple_on_invested"]),
                         f"{money(m_['invested'])} → {money(m_['final_nav'])}", pct(m_["irr"]), num(m_["multiple_on_invested"]),
                         pct(x["delta_irr_dd2x_vs_matched"], 2)])
    lines.append(md_table(["起點", "每月", "對象", "加碼月數/總月數", "加碼時在場", "固定：投入→NAV", "IRR", "倍數", "回撤 2×：投入→NAV", "IRR", "倍數", "預算配對：投入→NAV", "IRR", "倍數", "IRR 差（2× − 配對）"], rows) + "\n")
    rows = []
    for th in DD_THRESHOLDS:
        for lab, key in (("策略", "strategy"), ("持有", "buy_and_hold")):
            d_irr = [rec[key]["delta_irr_dd2x_vs_matched"] for rec in dd_rows if rec["dd_thresh"] == th and rec[key]["dd2x"]["boosted_months"] > 0]
            d_mul = [rec[key]["delta_multiple_dd2x_vs_matched"] for rec in dd_rows if rec["dd_thresh"] == th and rec[key]["dd2x"]["boosted_months"] > 0]
            n_b = [rec[key]["dd2x"]["boosted_months"] for rec in dd_rows if rec["dd_thresh"] == th]
            rows.append([f"{th:.0%}", lab, f"{sum(1 for x in n_b if x > 0)}/{len(n_b)}", pct(float(np.mean(d_irr)), 2) if d_irr else "—", pct(float(np.min(d_irr)), 2) if d_irr else "—", pct(float(np.max(d_irr)), 2) if d_irr else "—", num(float(np.mean(d_mul)), 3) if d_mul else "—"])
    lines.append("門檻敏感度（只算有觸發加碼的格；IRR 差 = 回撤 2× − 預算配對，正 = 挑時點有幫助）：\n")
    lines.append(md_table(["門檻", "對象", "有觸發格數", "IRR 差平均", "最小", "最大", "倍數差平均"], rows) + "\n")

    # 7. 結論（數字由上面產生，文字在腳本內固定寫法但引用計算值）
    g2020 = {(r["p0"], r["monthly"]): r for r in dca_rows if r["start"] == "2020-01-01"}
    g2021 = {(r["p0"], r["monthly"]): r for r in dca_rows if r["start"] == "2021-11-01"}
    g2023 = {(r["p0"], r["monthly"]): r for r in dca_rows if r["start"] == "2023-01-01"}
    wins = sum(1 for r in dca_rows if r["strategy"]["final_nav"] > r["buy_and_hold"]["final_nav"])
    dd_wins = sum(1 for r in dca_rows if r["strategy"]["max_dd_usd"] < r["buy_and_hold"]["max_dd_usd"])
    lines.append("## 7. 讀法\n")
    lines.append(f"- 27 個投入格中，策略期末 NAV 高於持有 {wins}/27、最深美元回撤比持有淺 {dd_wins}/27。\n"
                 f"- 起點 2021-11（頂部）10k 一次投入：策略 {money(g2021[(10_000, 0)]['strategy']['final_nav'])} vs 持有 {money(g2021[(10_000, 0)]['buy_and_hold']['final_nav'])}；"
                 f"最深 $ 回撤 {money(g2021[(10_000, 0)]['strategy']['max_dd_usd'])} vs {money(g2021[(10_000, 0)]['buy_and_hold']['max_dd_usd'])}；水下 {g2021[(10_000, 0)]['strategy']['underwater_days_pnl']} vs {g2021[(10_000, 0)]['buy_and_hold']['underwater_days_pnl']} 天。\n"
                 f"- 同起點加每月 3,000：策略 {money(g2021[(10_000, 3000)]['strategy']['final_nav'])}（IRR {pct(g2021[(10_000, 3000)]['strategy']['irr'])}）vs 持有 {money(g2021[(10_000, 3000)]['buy_and_hold']['final_nav'])}（IRR {pct(g2021[(10_000, 3000)]['buy_and_hold']['irr'])}）；"
                 f"水下最長 {g2021[(10_000, 3000)]['strategy']['underwater_days_pnl']} vs {g2021[(10_000, 3000)]['buy_and_hold']['underwater_days_pnl']} 天。\n"
                 f"- 起點 2023-01 10k：策略 {money(g2023[(10_000, 0)]['strategy']['final_nav'])} vs 持有 {money(g2023[(10_000, 0)]['buy_and_hold']['final_nav'])}；300k：{money(g2023[(300_000, 0)]['strategy']['final_nav'])} vs {money(g2023[(300_000, 0)]['buy_and_hold']['final_nav'])}，最深 $ 回撤 {money(g2023[(300_000, 0)]['strategy']['max_dd_usd'])} vs {money(g2023[(300_000, 0)]['buy_and_hold']['max_dd_usd'])}。\n"
                 f"- 參數 vs 本金（損益口徑，第 5 節）：起點 2023-01 每月 +1,000 的損益增量是 ema_len ±25% 範圍的 {num(comp['2023-01']['monthly1000_profit_over_param25'], 1)}×、"
                 f"本金 10k→50k 是 {num(comp['2023-01']['capital50k_profit_over_param25'], 1)}×；起點 2021-11 分別 {num(comp['2021-11（頂部）']['monthly1000_profit_over_param25'], 1)}× / {num(comp['2021-11（頂部）']['capital50k_profit_over_param25'], 1)}×；"
                 f"起點 2020-01 分別 {num(comp['2020-01']['monthly1000_profit_over_param25'], 1)}× / {num(comp['2020-01']['capital50k_profit_over_param25'], 1)}×。\n"
                 f"- 回撤加碼（第 6 節，門檻 20%）：對策略曲線「回撤 2×」比「同總額平均攤」的 IRR 差平均 {pct(float(np.mean([rec['strategy']['delta_irr_dd2x_vs_matched'] for rec in dd_rows if rec['dd_thresh'] == 0.2])), 2)}、"
                 f"對持有 {pct(float(np.mean([rec['buy_and_hold']['delta_irr_dd2x_vs_matched'] for rec in dd_rows if rec['dd_thresh'] == 0.2])), 2)}；"
                 f"策略加碼月只有 {np.mean([rec['strategy']['dd2x']['boosted_in_market_frac'] for rec in dd_rows if rec['dd_thresh'] == 0.2 and rec['strategy']['dd2x']['boosted_in_market_frac'] is not None]):.0%} 的時間在場（回撤 ≥ 20% 時策略多半已空手，加碼的錢先躺現金等下次進場）。\n")
    text = "\n".join(lines)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "capital.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(OUT_DIR, "capital.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(text)
    print(f"\n→ {os.path.join(OUT_DIR, 'capital.json')} / capital.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
