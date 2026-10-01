"""獨立重算 close_timing（懷疑者驗證）。不 import abgt.* / evaluate_*.py，只用 pandas / numpy。

用法：python results/robustness/close_timing_verify.py
只印結果，不寫任何檔案。
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEE, SLIP, THR, INIT = 0.001, 0.0005, 0.15, 10_000.0
W = {"BTC": 0.5, "ETH": 0.5}
WINDOWS = [("IS", "2020-01-01", "2022-12-31"), ("OOS", "2023-01-01", None), ("FULL", "2020-01-01", None),
           ("IS2", "2020-07-01", "2023-12-31"), ("OOS2", "2024-01-01", None)]
MIN_HOURS = 18


def load_hourly(sym: str) -> pd.DataFrame:
    h = pd.read_csv(os.path.join(ROOT, "data", f"{sym}USDT_1h.csv"))
    h.index = pd.to_datetime(h["timestamp"], unit="ms", utc=True)     # K 線開始時刻
    return h[["open", "high", "low", "close", "volume"]].astype(float).sort_index()


def synth_daily(h: pd.DataFrame, off: int) -> tuple[pd.DataFrame, int]:
    """日界 = off:00 UTC。key = 該 1h 所屬日 K 的開始時刻（不用 resample，自己算）。"""
    sh = pd.Timedelta(hours=off)
    key = (h.index - sh).floor("D") + sh
    g = h.groupby(key)
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "n": g["close"].count(),
                      "first_h": g.apply(lambda x: x.index[0]), "last_h": g.apply(lambda x: x.index[-1])})
    end_avail = h.index[-1] + pd.Timedelta(hours=1)
    d = d[d.index + pd.Timedelta(days=1) <= end_avail]
    n_short = int((d["n"] < MIN_HOURS).sum())
    d = d[d["n"] >= MIN_HOURS]
    assert (d.index.hour == off).all() and (d.index.minute == 0).all()
    # 無前視：每根日 K 用到的 1h 都在 [start, start+24h) 內
    assert (d["first_h"] >= d.index).all() and (d["last_h"] < d.index + pd.Timedelta(days=1)).all()
    return d, n_short


def ema_sma_seed(x: np.ndarray, n: int = 50) -> np.ndarray:
    out = np.full(len(x), np.nan)
    a = 2.0 / (n + 1)
    out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def simulate(dd: dict[str, pd.DataFrame], weights: dict[str, float]):
    keys = list(dd)
    idx = dd[keys[0]].index
    for k in keys[1:]:
        idx = idx.intersection(dd[k].index)
    o = {k: dd[k]["open"].reindex(idx).to_numpy() for k in keys}
    c = {k: dd[k]["close"].reindex(idx).to_numpy() for k in keys}
    on, w0 = {}, 0
    for k in keys:
        cl = dd[k]["close"].to_numpy()
        e = ema_sma_seed(cl)
        sig = pd.Series(cl > np.where(np.isnan(e), np.inf, e), index=dd[k].index)
        on[k] = sig.reindex(idx).fillna(False).to_numpy()
        w0 = max(w0, int(idx.searchsorted(dd[k].index[49])))
    n = len(idx)
    cash, qty = INIT, {k: 0.0 for k in keys}
    pend: dict[str, float | None] = {k: None for k in keys}
    eq = np.full(n, np.nan)
    # 每根「開盤成交後」的持倉與現金（供 00:00 估值），以及「開盤成交前」的持倉與現金
    post_cash, post_qty = np.full(n, np.nan), {k: np.full(n, np.nan) for k in keys}
    pre_cash, pre_qty = np.full(n, np.nan), {k: np.full(n, np.nan) for k in keys}
    fills = 0
    for i in range(w0, n):
        pre_cash[i] = cash
        for k in keys:
            pre_qty[k][i] = qty[k]
        orders = [(k, pend[k]) for k in keys if pend[k] is not None]
        for k in keys:
            pend[k] = None
        if orders:
            eqo = cash + sum(qty[k] * o[k][i] for k in keys)
            sells = [(k, t) for k, t in orders if t * eqo / o[k][i] < qty[k]]
            buys = [(k, t) for k, t in orders if (k, t) not in sells]
            for k, t in sells + buys:
                eqo = cash + sum(qty[j] * o[j][i] for j in keys)
                d = t * eqo / o[k][i] - qty[k]
                if d > 0:
                    d = min(d, max(cash, 0.0) / (o[k][i] * (1 + SLIP) * (1 + FEE)))
                    if d * o[k][i] < 1e-6 * eqo:
                        continue
                    cash -= d * o[k][i] * (1 + SLIP) * (1 + FEE)
                    qty[k] += d
                else:
                    if -d * o[k][i] < 1e-6 * eqo:
                        continue
                    cash += -d * o[k][i] * (1 - SLIP) * (1 - FEE)
                    qty[k] += d
                    if qty[k] < 1e-12:
                        qty[k] = 0.0
                fills += 1
        post_cash[i] = cash
        for k in keys:
            post_qty[k][i] = qty[k]
        e = cash + sum(qty[k] * c[k][i] for k in keys)
        eq[i] = e
        for k in keys:
            tgt = weights[k] if on[k][i] else 0.0
            cur = qty[k] * c[k][i] / e
            if tgt == 0.0 and cur > 0:
                pend[k] = 0.0
            elif tgt > 0 and cur == 0:
                pend[k] = tgt
            elif tgt > 0 and abs(tgt - cur) > THR * weights[k]:
                pend[k] = tgt
    return dict(idx=idx, w0=w0, eq=pd.Series(eq, index=idx), c=c, keys=keys, fills=fills,
                post_cash=post_cash, post_qty=post_qty, pre_cash=pre_cash, pre_qty=pre_qty)


def stats(e: pd.Series) -> dict:
    e = e.dropna()
    r = e.pct_change().dropna()
    yrs = (e.index[-1] - e.index[0]).total_seconds() / (365.25 * 86400)
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1
    dd = float((e / e.cummax() - 1).min())
    return {"sharpe": float(r.mean() / r.std(ddof=1) * math.sqrt(365)), "cagr": cagr, "dd": dd,
            "calmar": cagr / abs(dd) if dd < 0 else math.nan, "n": len(e)}


def hold(c: dict, idx, s0: int, s1: int, weights: dict, monthly: bool, rebal_fee: bool) -> pd.Series:
    keys = list(c)
    q = {k: INIT * (1 - FEE) * weights[k] / c[k][s0] for k in keys}
    out, m = [], idx[s0].month
    for i in range(s0, s1):
        v = sum(q[k] * c[k][i] for k in keys)
        if monthly and idx[i].month != m:
            m = idx[i].month
            if rebal_fee:
                v -= sum(abs(v * weights[k] - q[k] * c[k][i]) for k in keys) * FEE
            q = {k: v * weights[k] / c[k][i] for k in keys}
        out.append(v)
    return pd.Series(out, index=idx[s0:s1])


def window_bounds(idx, w0, ws, we, conv: str):
    t0 = pd.Timestamp(ws, tz="UTC")
    s0 = max(w0, int(idx.searchsorted(t0)))
    if we is None:
        s1 = len(idx)
    elif conv == "orig":      # 原作者：K 線開始時刻 ≤ 結束日 00:00
        s1 = int(idx.searchsorted(pd.Timestamp(we, tz="UTC"), side="right"))
    else:                     # 自然：K 線開始日期 ≤ 結束日
        s1 = int(idx.searchsorted(pd.Timestamp(we, tz="UTC") + pd.Timedelta(days=1)))
    return s0, s1


def evaluate(sim, weights, conv="orig", rebal_fee=True) -> dict:
    idx, w0, eq, c = sim["idx"], sim["w0"], sim["eq"], sim["c"]
    row = {}
    for name, ws, we in WINDOWS:
        s0, s1 = window_bounds(idx, w0, ws, we, conv)
        st = stats(eq.iloc[s0:s1])
        hs = {b: stats(hold(c, idx, s0, s1, weights, b == "M", rebal_fee)) for b in ("N", "M")}
        cells = sum(int(st["sharpe"] > h["sharpe"] and st["calmar"] > h["calmar"] and st["dd"] > h["dd"]) for h in hs.values())
        row[name] = {**st, "cells": cells, "hold_sharpe": max(h["sharpe"] for h in hs.values()), "hold": hs}
    return row


def mark_midnight(sim, hourly: dict, stale: bool = False) -> pd.Series:
    """每天 00:00 UTC 以真實 1h 價格估值。00:00 落在 [start, start+24h) 的那根：
    若 00:00 == start（offset 0）→ 用開盤成交前的持倉（= 前一根收盤權益，精確）；否則用成交後持倉。
    stale=True：用最後一次收盤權益往前帶（原作者第一版的錯誤做法，用來量差距）。"""
    idx, keys, w0 = sim["idx"], sim["keys"], sim["w0"]
    starts = idx[w0:]
    mids = pd.date_range(starts[0].ceil("D"), (starts[-1] + pd.Timedelta(days=1)).floor("D"), freq="D")
    if stale:
        ends = starts + pd.Timedelta(days=1)
        e = pd.Series(sim["eq"].iloc[w0:].to_numpy(), index=ends)
        return e.reindex(e.index.union(mids)).ffill().reindex(mids).dropna()
    vals = []
    px = {k: hourly[k]["close"] for k in keys}
    for t in mids:
        p = int(starts.searchsorted(t, side="right")) - 1
        if p < 0:
            vals.append(np.nan)
            continue
        i = w0 + p
        if t - idx[i] >= pd.Timedelta(days=1):   # 缺日（不應發生）
            vals.append(np.nan)
            continue
        if idx[i] == t:
            cash, q = sim["pre_cash"][i], {k: sim["pre_qty"][k][i] for k in keys}
        else:
            cash, q = sim["post_cash"][i], {k: sim["post_qty"][k][i] for k in keys}
        hp = t - pd.Timedelta(hours=1)
        vals.append(cash + sum(q[k] * px[k].get(hp, np.nan) for k in keys))
    return pd.Series(vals, index=mids).dropna()


def main():
    hourly = {"BTC": load_hourly("BTC"), "ETH": load_hourly("ETH")}
    # 交易所日線對照（offset 0）
    for k in hourly:
        ref = pd.read_csv(os.path.join(ROOT, "data", f"{k}USDT_1d.csv")) if os.path.exists(os.path.join(ROOT, "data", f"{k}USDT_1d.csv")) else None
        if ref is not None:
            ref.index = pd.to_datetime(ref["timestamp"], unit="ms", utc=True)
            syn, _ = synth_daily(hourly[k], 0)
            cm = syn.index.intersection(ref.index)
            print(f"[對照] {k} offset0 vs 1d：共 {len(cm)} 天，close 最大差 {(syn.loc[cm,'close']/ref.loc[cm,'close']-1).abs().max():.2e}，"
                  f"open 最大差 {(syn.loc[cm,'open']/ref.loc[cm,'open']-1).abs().max():.2e}")
    orig = json.load(open(os.path.join(ROOT, "results", "robustness", "close_timing.json"), encoding="utf-8"))
    rows, alt_conv, alt_fee, sims, info = {}, {}, {}, {}, {}
    for h in range(24):
        dd, nshort = {}, {}
        for k in hourly:
            dd[k], nshort[k] = synth_daily(hourly[k], h)
        sim = simulate(dd, W)
        sims[h] = (sim, dd)
        rows[h] = evaluate(sim, W)
        alt_conv[h] = evaluate(sim, W, conv="natural")
        alt_fee[h] = evaluate(sim, W, rebal_fee=False)
        idx = sim["idx"]
        gaps = int((pd.Series(idx).diff().dropna() != pd.Timedelta(days=1)).sum())
        part = {k: int((dd[k]["n"] < 24).sum()) for k in dd}
        oos_n = rows[h]["OOS"]["n"]
        info[h] = (nshort, gaps, part, oos_n, sim["fills"], str(idx[sim["w0"]]))
    print("\n[樣本可比性] offset: 丟掉的短日(BTC,ETH) / 日K 間隔≠24h 次數 / 不滿 24 根的日K數 / OOS 根數 / 成交筆數 / 暖機後第一根")
    for h in range(24):
        print(f"  {h:02d}: {info[h]}")

    print("\n[逐時點] 我的 Sharpe（原作者）  十格 我的/原")
    diffs = []
    for h in range(24):
        o = orig["rows"][str(h)]["port"]
        mine = rows[h]
        cells_m = sum(mine[w]["cells"] for w in mine)
        cells_o = sum(o[w]["gate_cells"] for w in o)
        for w in mine:
            diffs.append(abs(mine[w]["sharpe"] - o[w]["sharpe"]))
        print(f"  {h:02d}:00 " + " ".join(f"{w} {mine[w]['sharpe']:.4f}({o[w]['sharpe']:.4f})" for w in mine)
              + f" | 十格 {cells_m}/{cells_o} | 自然視窗十格 {sum(alt_conv[h][w]['cells'] for w in mine)} | 月再平衡不付費十格 {sum(alt_fee[h][w]['cells'] for w in mine)}")
    print(f"  全部 24×5 Sharpe 最大絕對差 {max(diffs):.2e}")

    for h in (0, 5, 12, 21):
        print(f"\n[重點時點 {h:02d}:00] OOS {rows[h]['OOS']['sharpe']:.4f}  OOS2 {rows[h]['OOS2']['sharpe']:.4f}  十格 {sum(rows[h][w]['cells'] for w in rows[h])}")
        for w in rows[h]:
            r = rows[h][w]
            print(f"   {w}: S {r['sharpe']:.3f} Calmar {r['calmar']:.3f} DD {r['dd']:.3f} | 持有N S {r['hold']['N']['sharpe']:.3f} C {r['hold']['N']['calmar']:.3f} DD {r['hold']['N']['dd']:.3f}"
                  f" | 持有M S {r['hold']['M']['sharpe']:.3f} C {r['hold']['M']['calmar']:.3f} DD {r['hold']['M']['dd']:.3f} | cells {r['cells']}")

    def summary(rr, tag):
        oos = [rr[h]["OOS"]["sharpe"] for h in range(24)]
        oos2 = [rr[h]["OOS2"]["sharpe"] for h in range(24)]
        rank = lambda v, x: 1 + sum(y > x for y in v)
        rank_w = lambda v, x: sum(y >= x for y in v)
        R1 = sum(sum(rr[h][w]["cells"] for w in rr[h]) == 10 for h in range(24))
        edge = {w: float(np.median([rr[h][w]["sharpe"] - rr[h][w]["hold_sharpe"] for h in range(24)])) for w in rr[0]}
        print(f"\n[{tag}] R1 十格全過 {R1}/24；OOS 中位 {np.median(oos):.4f} 全距 {min(oos):.3f}-{max(oos):.3f} std {np.std(oos):.3f}；"
              f"00:00 OOS 名次 {rank(oos, oos[0])}-{rank_w(oos, oos[0])}/24、OOS2 名次 {rank(oos2, oos2[0])}-{rank_w(oos2, oos2[0])}/24；"
              f"OOS2 中位 {np.median(oos2):.4f}；R3 中位 edge " + " ".join(f"{w} {v:+.3f}" for w, v in edge.items()))
        return R1, edge

    summary(rows, "原口徑")
    summary(alt_conv, "自然視窗結束日")
    summary(alt_fee, "月再平衡不付費")

    # 時間分散：真實 00:00 估值 vs 陳舊收盤往前帶
    for stale in (False, True):
        curves = {h: mark_midnight(sims[h][0], hourly, stale) for h in range(24)}
        grid = pd.date_range(max(c.index[0] for c in curves.values()), min(c.index[-1] for c in curves.values()), freq="D")
        combo = sum(curves[h].reindex(grid) / curves[h].reindex(grid).iloc[0] for h in curves) / 24 * INIT
        out = []
        for name, ws, we in WINDOWS:
            seg = combo.loc[ws:we] if we else combo.loc[ws:]
            out.append(f"{name} {stats(seg)['sharpe']:.3f}/{stats(seg)['dd']:.3f}")
        print(f"\n[時間分散 {'陳舊往前帶' if stale else '真實 00:00 估值'}] Sharpe/DD：" + "  ".join(out)
              + ("" if stale else f"  (原作者 OOS {orig['time_diversified']['OOS']['sharpe']:.3f} OOS2 {orig['time_diversified']['OOS2']['sharpe']:.3f})"))
        if not stale:
            e0 = sims[0][0]["eq"].iloc[sims[0][0]["w0"]:]
            e0.index = e0.index + pd.Timedelta(days=1)
            cm = e0.index.intersection(curves[0].index)
            print(f"   offset0 00:00 估值 vs 自身收盤權益 最大相對差 {(curves[0].reindex(cm)/e0.reindex(cm)-1).abs().max():.2e}（原作者 {orig['mark_check_offset0']:.2e}）")

    # 單一時點 Sharpe 抽樣誤差量級
    for h in (0,):
        r = sims[h][0]["eq"].loc["2023-01-01":].pct_change().dropna()
        sr = r.mean() / r.std() * math.sqrt(365)
        se = math.sqrt((1 + 0.5 * (sr / math.sqrt(365)) ** 2) / len(r)) * math.sqrt(365)
        print(f"\n[抽樣誤差] 00:00 OOS Sharpe {sr:.2f}，近似 SE ≈ {se:.2f}（{len(r)} 天）")
    # 相鄰時點相關
    rets = pd.DataFrame({h: pd.Series(mark_midnight(sims[h][0], hourly).loc['2023-01-01':].pct_change().dropna()) for h in range(24)})
    cc = rets.corr().to_numpy()
    print(f"[時點間 00:00 估值日報酬相關] 平均 {cc[np.triu_indices(24, 1)].mean():.3f}，最小 {cc.min():.3f}")


if __name__ == "__main__":
    main()
