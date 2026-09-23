"""懷疑者獨立重算 capital 分析的關鍵數字（不用 abgt 引擎，純 numpy/pandas）。

重跑：cd D:/feihong/tradingView/ABGT && PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe results/levers/capital_verify.py
只讀 data/BTCUSDT_1d.csv、data/ETHUSDT_1d.csv、results/levers/capital.json；不寫任何既有檔案（只印）。
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

ROOT = "D:/feihong/tradingView/ABGT"
FEE, SLIP, W, REBAL = 0.001, 0.0005, 0.5, 0.15
COST = FEE + SLIP


def load(sym: str) -> pd.DataFrame:
    df = pd.read_csv(f"{ROOT}/data/{sym}_1d.csv")
    df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df[df.index >= "2019-09-01"][["open", "high", "low", "close"]].astype("float64")


def ema_sma_seed(c: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(c), np.nan)
    a = 2.0 / (n + 1)
    out[n - 1] = c[:n].mean()
    for i in range(n, len(c)):
        out[i] = a * c[i] + (1 - a) * out[i - 1]
    return out


def strategy_curve(btc: pd.DataFrame, eth: pd.DataFrame, n: int = 50, initial: float = 10_000.0):
    """純 numpy 重寫：收盤 close > EMA(n) 判定，次日開盤成交，進場目標 = 0.5 × 開盤權益；費 0.10%/邊、滑 0.05%。
    在場期間若實際權重偏離 0.5 超過 0.15×0.5 = 7.5 pp，次日開盤調回 0.5 × 權益（引擎 pending 清空後 ref = 當前權重）。"""
    idx = btc.index.intersection(eth.index)
    o = {"B": btc["open"].reindex(idx).to_numpy(), "E": eth["open"].reindex(idx).to_numpy()}
    c = {"B": btc["close"].reindex(idx).to_numpy(), "E": eth["close"].reindex(idx).to_numpy()}
    on = {k: c[k] > ema_sma_seed(c[k], n) for k in c}
    start = n - 1  # 兩標的 EMA 同日暖機完
    cash, qty = initial, {"B": 0.0, "E": 0.0}
    eq = np.full(len(idx), np.nan)
    in_frac = np.zeros(len(idx))
    fills = 0
    pend = {"B": None, "E": None}
    for i in range(start, len(idx)):
        for k in ("B", "E"):
            if pend[k] is not None:
                target = pend[k]
                pend[k] = None
                eq_open = cash + sum(qty[j] * o[j][i] for j in ("B", "E"))
                tq = target * eq_open / o[k][i]
                diff = tq - qty[k]
                if diff > 0:
                    px = o[k][i] * (1 + SLIP)
                    cash -= diff * px * (1 + FEE)
                else:
                    px = o[k][i] * (1 - SLIP)
                    cash += -diff * px * (1 - FEE)
                qty[k] += diff
                fills += 1
        eqv = cash + sum(qty[j] * c[j][i] for j in ("B", "E"))
        eq[i] = eqv
        for k in ("B", "E"):
            holding = qty[k] > 1e-12
            cur = qty[k] * c[k][i] / eqv
            if on[k][i] and not holding:
                pend[k] = W
            elif not on[k][i] and holding:
                pend[k] = 0.0
            elif on[k][i] and holding and abs(W - cur) > REBAL * W:
                pend[k] = W  # 引擎「調整曝險」：在場時實際權重偏離目標 > 0.15 × 0.5 就調回 0.5 × 權益
        in_frac[i] = W * on["B"][i] + W * on["E"][i]
    s = pd.Series(eq, index=idx).iloc[start:]
    return s, pd.Series(in_frac, index=idx).iloc[start:], fills


def hold_curve(btc, eth, start_ts, initial=10_000.0, rebalance_monthly=False):
    """同權重 50/50 買入持有，從 start_ts 當日收盤起、付一次 0.10% 手續費。"""
    idx = btc.index.intersection(eth.index)
    idx = idx[idx >= start_ts]
    cb, ce = btc["close"].reindex(idx).to_numpy(), eth["close"].reindex(idx).to_numpy()
    qb = initial * (1 - FEE) * W / cb[0]
    qe = initial * (1 - FEE) * W / ce[0]
    eq = np.empty(len(idx))
    last_m = idx[0].month
    for i in range(len(idx)):
        v = qb * cb[i] + qe * ce[i]
        if rebalance_monthly and idx[i].month != last_m:
            last_m = idx[i].month
            cost = (abs(v * W - qb * cb[i]) + abs(v * W - qe * ce[i])) * FEE
            v -= cost
            qb, qe = v * W / cb[i], v * W / ce[i]
        eq[i] = v
    return pd.Series(eq, index=idx)


def stats(eq: pd.Series) -> dict:
    eq = eq.dropna()
    yrs = (eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 86400)
    r = eq.iloc[-1] / eq.iloc[0]
    peak = np.maximum.accumulate(eq.to_numpy())
    mdd = float(((eq.to_numpy() - peak) / peak).min())
    d = eq.pct_change().dropna()
    return {"cagr": r ** (1 / yrs) - 1, "max_dd": mdd, "sharpe": float(d.mean() / d.std() * math.sqrt(365)),
            "calmar": (r ** (1 / yrs) - 1) / abs(mdd), "years": yrs, "final": float(eq.iloc[-1])}


def xirr_newton(dates, cfs) -> float:
    t = np.array([(d - dates[0]).days / 365.25 for d in dates])
    cf = np.array(cfs)
    r = 0.1
    for _ in range(200):
        f = np.sum(cf / (1 + r) ** t)
        df = np.sum(-t * cf / (1 + r) ** (t + 1))
        step = f / df
        r -= step
        if abs(step) < 1e-12:
            break
    return r


def dca(unit: pd.Series, in_frac, start: str, p0: float, monthly: float, dd2x: float | None = None):
    """份額法（與分析者定義相同，但獨立實作）：起始日 + 之後每月第一根 K 線收盤買份額；投入前扣 0.15% × 在場比例。"""
    u = unit.loc[start:]
    u = u / u.iloc[0]
    ua = u.to_numpy()
    f = in_frac.reindex(u.index).fillna(0.0).to_numpy() if isinstance(in_frac, pd.Series) else np.full(len(ua), in_frac)
    months = np.array([d.year * 12 + d.month for d in u.index])
    first = np.r_[True, months[1:] != months[:-1]]
    hwm = np.maximum.accumulate(ua)
    units, inv, val = 0.0, np.zeros(len(ua)), np.zeros(len(ua))
    dts, cfs = [], []
    boosted, boosted_in = 0, []
    for i in range(len(ua)):
        amt = p0 if i == 0 else (monthly if first[i] else 0.0)
        if i > 0 and amt > 0 and dd2x is not None and ua[i] / hwm[i] - 1 <= -dd2x:
            amt *= 2
            boosted += 1
            boosted_in.append(f[i])
        if amt > 0:
            units += amt * (1 - COST * f[i]) / ua[i]
            dts.append(u.index[i]); cfs.append(-amt)
        inv[i] = (inv[i - 1] if i else 0) + amt
        val[i] = units * ua[i]
    pnl = val - inv
    dd = np.maximum.accumulate(pnl) - pnl
    it = int(np.argmax(dd)); ip = int(np.argmax(pnl[: it + 1]))
    # 水下最長天數
    best = cur = 0
    for x in dd > 1e-9:
        cur = cur + 1 if x else 0; best = max(best, cur)
    yrs = (u.index[-1] - u.index[0]).days / 365.25
    return {"invested": inv[-1], "final": val[-1], "irr": xirr_newton(dts + [u.index[-1]], cfs + [val[-1]]),
            "twr_cagr": ua[-1] ** (1 / yrs) - 1, "max_dd_usd": dd[it], "acct_pct": -dd[it] / val[ip] if val[ip] > 0 else np.nan,
            "underwater": best, "boosted": boosted, "boosted_in": float(np.mean(boosted_in)) if boosted_in else None,
            "n_months": int(first[1:].sum())}


def years_to(p0, X, g, T=300_000):
    m = (1 + g) ** (1 / 12) - 1
    if X == 0:
        return math.log(T / p0) / math.log1p(m) / 12
    if g == 0:
        return (T - p0) / X / 12
    return math.log((T + X / m) / (p0 + X / m)) / math.log1p(m) / 12


def main():
    J = json.load(open(f"{ROOT}/results/levers/capital.json", encoding="utf-8"))
    btc, eth = load("BTCUSDT"), load("ETHUSDT")
    eq, in_frac, fills = strategy_curve(btc, eth, 50)
    bh_full = hold_curve(btc, eth, eq.index[0])
    print("== A. 引擎曲線（純 numpy 重寫 vs 分析者 abgt 引擎）")
    print(f"起算 {eq.index[0].date()} → {eq.index[-1].date()}；策略期末 {eq.iloc[-1]:,.0f}（分析者 {J['setup']['engine_final_equity']:,.0f}）；"
          f"持有 {bh_full.iloc[-1]:,.0f}（分析者 {J['setup']['bh_final_equity']:,.0f}）；成交 {fills}（分析者 {J['setup']['n_fills']}）；"
          f"在場 {float((in_frac > 0).mean()):.3f}（分析者 {J['setup']['time_in_market']:.3f}）")

    print("\n== B. 視窗指標（策略：切片正規化；持有：分析者用 2019-10 起算的漂移權重切片 vs 我用視窗起點重新 50/50）")
    WIN = {"IS": ("2020-01-01", "2022-12-31"), "OOS": ("2023-01-01", None), "FULL": ("2020-01-01", None),
           "IS2": ("2020-07-01", "2023-12-31"), "OOS2": ("2024-01-01", None)}
    for w, (a, b) in WIN.items():
        s = stats(eq.loc[a:b] if b else eq.loc[a:])
        js = J["windows"]["strategy"][w]
        bh_sl = bh_full.loc[a:b] if b else bh_full.loc[a:]
        hb_slice = stats(bh_sl)
        hb_fresh = stats(hold_curve(btc, eth, pd.Timestamp(a, tz="UTC")).loc[:b] if b else hold_curve(btc, eth, pd.Timestamp(a, tz="UTC")))
        hb_reb = stats(hold_curve(btc, eth, pd.Timestamp(a, tz="UTC"), rebalance_monthly=True).loc[:b] if b else hold_curve(btc, eth, pd.Timestamp(a, tz="UTC"), rebalance_monthly=True))
        jb = J["windows"]["buy_and_hold"][w]
        print(f"{w:5} 策略 CAGR {s['cagr']:+.1%} MDD {s['max_dd']:+.1%} Sh {s['sharpe']:.2f} Cal {s['calmar']:.2f} | 分析者 {js['cagr']:+.1%} {js['max_dd']:+.1%} {js['sharpe']:.2f} {js['calmar']:.2f}")
        print(f"      持有(切片漂移) CAGR {hb_slice['cagr']:+.1%} MDD {hb_slice['max_dd']:+.1%} Sh {hb_slice['sharpe']:.2f} Cal {hb_slice['calmar']:.2f} | 分析者 {jb['cagr']:+.1%} {jb['max_dd']:+.1%} {jb['sharpe']:.2f} {jb['calmar']:.2f}")
        print(f"      持有(視窗起點 50/50 不再平衡) CAGR {hb_fresh['cagr']:+.1%} MDD {hb_fresh['max_dd']:+.1%} Sh {hb_fresh['sharpe']:.2f} Cal {hb_fresh['calmar']:.2f}"
              f" | 月再平衡 CAGR {hb_reb['cagr']:+.1%} MDD {hb_reb['max_dd']:+.1%} Sh {hb_reb['sharpe']:.2f} Cal {hb_reb['calmar']:.2f}"
              f" | 閘門(Sh&Cal&MDD) vs 50/50: {'過' if s['sharpe'] > hb_fresh['sharpe'] and s['calmar'] > hb_fresh['calmar'] and s['max_dd'] > hb_fresh['max_dd'] else '不過'}"
              f" vs 月再平衡: {'過' if s['sharpe'] > hb_reb['sharpe'] and s['calmar'] > hb_reb['calmar'] and s['max_dd'] > hb_reb['max_dd'] else '不過'}")

    print("\n== C. 漂移權重：分析者的『持有』在各起點實際的 BTC/ETH 權重")
    for d in ("2020-01-01", "2021-11-01", "2023-01-01"):
        t = pd.Timestamp(d, tz="UTC")
        i0 = eq.index[0]
        vb = W * btc["close"][t] / btc["close"][i0]
        ve = W * eth["close"][t] / eth["close"][i0]
        print(f"  {d}: BTC {vb / (vb + ve):.0%} / ETH {ve / (vb + ve):.0%}")

    print("\n== D. 定期投入格（策略 / 分析者持有(漂移) / 我的持有(起點 50/50 不再平衡) / 月再平衡）")
    grid = {(r["start"], r["p0"], r["monthly"]): r for r in J["dca_grid"]}
    wins_nav, wins_dd, wins_nav_reb, wins_dd_reb, n = 0, 0, 0, 0, 0
    for d in ("2020-01-01", "2021-11-01", "2023-01-01"):
        t = pd.Timestamp(d, tz="UTC")
        h50 = hold_curve(btc, eth, t)
        hreb = hold_curve(btc, eth, t, rebalance_monthly=True)
        for p0 in (10_000, 50_000, 300_000):
            for m in (0, 1000, 3000):
                rs = dca(eq, in_frac, d, p0, m)
                rh = dca(h50, 1.0, d, p0, m)
                rr = dca(hreb, 1.0, d, p0, m)
                js, jh = grid[(d, p0, m)]["strategy"], grid[(d, p0, m)]["buy_and_hold"]
                n += 1
                wins_nav += rs["final"] > rh["final"]; wins_dd += rs["max_dd_usd"] < rh["max_dd_usd"]
                wins_nav_reb += rs["final"] > rr["final"]; wins_dd_reb += rs["max_dd_usd"] < rr["max_dd_usd"]
                if p0 == 10_000 or (p0 == 300_000 and m == 0):
                    print(f"  {d} {p0:>7,} +{m:>5,}: 策略 NAV {rs['final']:,.0f} IRR {rs['irr']:+.1%} $DD {rs['max_dd_usd']:,.0f} 水下 {rs['underwater']}"
                          f"（分析者 {js['final_nav']:,.0f} {js['irr']:+.1%} {js['max_dd_usd']:,.0f} {js['underwater_days_pnl']}）"
                          f" | 分析者持有 {jh['final_nav']:,.0f} IRR {jh['irr']:+.1%} $DD {jh['max_dd_usd']:,.0f}"
                          f" | 50/50 持有 {rh['final']:,.0f} IRR {rh['irr']:+.1%} $DD {rh['max_dd_usd']:,.0f} 水下 {rh['underwater']}"
                          f" | 月再平衡 {rr['final']:,.0f} IRR {rr['irr']:+.1%} $DD {rr['max_dd_usd']:,.0f}")
    print(f"  27 格：策略 NAV > 起點 50/50 持有 {wins_nav}/27、$回撤更淺 {wins_dd}/27；vs 月再平衡 NAV {wins_nav_reb}/27、$回撤 {wins_dd_reb}/27"
          f"（分析者：27/27、27/27）")

    print("\n== E. 到 30 萬年數（閉式解重算）")
    g_base = stats(eq.loc["2023-01-01":])["cagr"]; g_cons = g_base / 2; g_opt = stats(eq.loc["2020-01-01":"2022-12-31"])["cagr"]
    for X in (0, 1000, 3000, 5000):
        print(f"  每月 {X:>5}: 保守 {years_to(10_000, X, g_cons):.1f} / 基準 {years_to(10_000, X, g_base):.1f} / 樂觀 {years_to(10_000, X, g_opt):.1f} / 0% {years_to(10_000, X, 0.0) if X else float('inf'):.1f}"
              f"  （分析者 {J['years_to_300k'][str(X)]}）")
    # 蒙地卡羅式驗證：直接逐月累加
    m = (1 + g_base) ** (1 / 12) - 1; v, k = 10_000, 0
    while v < 300_000:
        v = v * (1 + m) + 2000; k += 1
    print(f"  逐月累加驗證：基準 +2000/月 需 {k} 個月 = {k / 12:.1f} 年（分析者 4.6）")

    print("\n== F. 本金 × 年化點估計")
    print(f"  基準 CAGR {g_base:+.2%}；300k 1 年 {300_000 * (1 + g_base):,.0f}、5 年 {300_000 * (1 + g_base) ** 5:,.0f}（分析者 416,011 / 1,538,289）；"
          f"最壞回撤 1 年 = {300_000 * (1 + g_base) * 0.383:,.0f}")
    print(f"  保守 CAGR {g_cons:+.2%}；300k 1 年 {300_000 * (1 + g_cons):,.0f} × 0.598 = {300_000 * (1 + g_cons) * 0.598:,.0f}（分析者 214,107）")

    print("\n== G. 回撤 ≥20% 加碼（起點 2023-01 / 2020-01，每月 1000）：2× vs 預算配對")
    for d in ("2020-01-01", "2021-11-01", "2023-01-01"):
        for lab, unit, fr in (("策略", eq, in_frac), ("50/50持有", hold_curve(btc, eth, pd.Timestamp(d, tz="UTC")), 1.0)):
            fx = dca(unit, fr, d, 10_000, 1000)
            bo = dca(unit, fr, d, 10_000, 1000, dd2x=0.2)
            mt = dca(unit, fr, d, 10_000, (bo["invested"] - 10_000) / fx["n_months"])
            print(f"  {d} {lab:8}: 加碼 {bo['boosted']}/{fx['n_months']} 月 在場 {bo['boosted_in'] if bo['boosted_in'] is None else f'{bo['boosted_in']:.0%}'} | 固定 IRR {fx['irr']:+.2%} | 2× {bo['invested']:,.0f}→{bo['final']:,.0f} IRR {bo['irr']:+.2%} | 配對 {mt['invested']:,.0f}→{mt['final']:,.0f} IRR {mt['irr']:+.2%} | 差 {bo['irr'] - mt['irr']:+.2%}")
    jd = [r for r in J["dd_boost"] if r["dd_thresh"] == 0.2 and r["monthly"] == 1000]
    print("  分析者：" + "; ".join(f"{r['start']} 策略 {r['strategy']['delta_irr_dd2x_vs_matched']:+.2%} / 持有 {r['buy_and_hold']['delta_irr_dd2x_vs_matched']:+.2%}" for r in jd))

    print("\n== H. 參數 vs 本金：ema_len 30/38/50/62/70 10k 一次投入 NAV（我的引擎）")
    navs = {}
    for nn in (30, 38, 50, 62, 70):
        e2, f2, _ = strategy_curve(btc, eth, nn)
        navs[nn] = {d: dca(e2, f2, d, 10_000, 0)["final"] for d in ("2020-01-01", "2021-11-01", "2023-01-01")}
    for d in ("2020-01-01", "2021-11-01", "2023-01-01"):
        r25 = max(navs[k][d] for k in (38, 50, 62)) - min(navs[k][d] for k in (38, 50, 62))
        r40 = max(navs[k][d] for k in navs) - min(navs[k][d] for k in navs)
        base = dca(eq, in_frac, d, 10_000, 0)["final"]
        m1 = dca(eq, in_frac, d, 10_000, 1000); c5 = dca(eq, in_frac, d, 50_000, 0)
        pm1 = (m1["final"] - m1["invested"]) - (base - 10_000); pc5 = (c5["final"] - 50_000) - (base - 10_000)
        jc = J["param_vs_capital"][{"2020-01-01": "2020-01", "2021-11-01": "2021-11（頂部）", "2023-01-01": "2023-01"}[d]]
        print(f"  {d}: NAV {[round(navs[k][d]) for k in navs]} ±25% 範圍 {r25:,.0f}（分析者 {jc['param_range_usd_pm25']:,.0f}） ±40% {r40:,.0f}（{jc['param_range_usd_pm40']:,.0f}）"
              f" | +1000 損益增量 {pm1:,.0f} → {pm1 / r25:.1f}×（分析者 {jc['monthly1000_profit_over_param25']:.1f}×） | 50k {pc5:,.0f} → {pc5 / r25:.1f}×（{jc['capital50k_profit_over_param25']:.1f}×）")
        # 同尺度比較：每投入 1 美元的損益增量 vs 參數
        print(f"      口徑檢查：+1000/月 多投 {m1['invested'] - 10_000:,.0f}，多賺 {pm1:,.0f}（每多投 1 元多賺 {pm1 / (m1['invested'] - 10_000):.2f}）；"
              f"10k→50k 多投 40,000 多賺 {pc5:,.0f}（每 1 元 {pc5 / 40_000:.2f}）；參數 ±25% 在 10k 上的範圍 = {r25 / base:.1%} of NAV → 在 50k 本金上等於 {r25 * 5:,.0f}")

    print("\n== I. bootstrap 種子敏感度（OOS 30 日區塊、1000 條）")
    rets = (eq.loc["2023-01-01":].pct_change().dropna()).to_numpy()
    for seed in (42, 7, 123):
        rng = np.random.default_rng(seed)
        for T in (1, 5):
            n_ = T * 365; nb = math.ceil(n_ / 30)
            mults, mdds = [], []
            for _ in range(1000):
                st = rng.integers(0, len(rets) - 30 + 1, size=nb)
                p = np.concatenate([rets[s:s + 30] for s in st])[:n_]
                e_ = np.cumprod(1 + p); mults.append(e_[-1])
                e_ = np.r_[1.0, e_]; mdds.append(((e_ - np.maximum.accumulate(e_)) / np.maximum.accumulate(e_)).min())
            mults, mdds = np.array(mults), np.array(mdds)
            print(f"  seed {seed} {T}y: p10/p50/p90 {np.percentile(mults, 10):.2f}/{np.percentile(mults, 50):.2f}/{np.percentile(mults, 90):.2f} 虧損 {np.mean(mults < 1):.0%} MDD p50 {np.percentile(mdds, 50):+.1%} p90深 {np.percentile(mdds, 10):+.1%} 最壞 {mdds.min():+.1%}")
    jb = J["bootstrap"]["by_window"]["OOS"]
    print(f"  分析者 seed42 1y {jb['1']['mult_p10']:.2f}/{jb['1']['mult_p50']:.2f}/{jb['1']['mult_p90']:.2f} 虧損 {jb['1']['prob_loss']:.0%}；5y {jb['5']['mult_p10']:.2f}/{jb['5']['mult_p50']:.2f}/{jb['5']['mult_p90']:.2f} 虧損 {jb['5']['prob_loss']:.0%}")

    print("\n== J. 未來資料檢查：把 regime 往後平移一天（用前一日收盤決策）看差異；若分析者的曲線與 shift(+1) 版更接近則有 lookahead")
    e_lag, _, _ = strategy_curve(btc, eth, 50)
    print(f"  （策略曲線用 close[i] > EMA[i] 在 i 收盤決策、i+1 開盤成交，與引擎一致；期末 {e_lag.iloc[-1]:,.0f}）")
    # 假設同日收盤成交（lookahead 版）會多賺多少：
    idx = btc.index.intersection(eth.index)
    cb, ce = btc["close"].reindex(idx).to_numpy(), eth["close"].reindex(idx).to_numpy()
    onb, one = cb > ema_sma_seed(cb, 50), ce > ema_sma_seed(ce, 50)
    rb, re_ = np.r_[0, np.diff(cb) / cb[:-1]], np.r_[0, np.diff(ce) / ce[:-1]]
    la = np.cumprod(1 + W * onb * rb + W * one * re_)  # 同日收盤成交、無成本 = lookahead 上限
    print(f"  同日收盤成交、零成本的曲線期末 {10_000 * la[-1] / la[49]:,.0f} vs 引擎 {eq.iloc[-1]:,.0f}（差距 = 一日延遲 + 成本；分析者曲線不含同日成交）")


if __name__ == "__main__":
    main()
