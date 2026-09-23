"""懷疑者獨立重算：diversification（標的分散）。

不 import abgt.*；直接讀 data/*.csv，自寫 EMA50（SMA 起算）、regime、共用權益組合模擬、持有基準、Sharpe/Calmar/DD、閘門，
再與 results/levers/diversification.json 逐格比對。另做分析者沒做的檢查：
  (a) OOS 用「2023-01-01 等權買入不再平衡」的新鮮基準（而非從 2020-09-29 漂移過來的持有 NR），看閘門與逐年「贏持有」是否仍成立；
  (b) 成交原因拆解（進場 / 趨勢結束 / 調整曝險）看漂移調倉佔了多少換手。

重跑：PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe results/levers/diversification_verify.py
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FEE, SLIP, TH, INIT = 0.001, 0.0005, 0.15, 10_000.0
FILES = {"BTC": "BTCUSDT_1d.csv", "ETH": "ETHUSDT_1d.csv", "SOL": "SOLUSDT_1d.csv", "BNB": "BNBUSDT_1d.csv"}
WINDOWS = [("IS", "2020-01-01", "2022-12-31"), ("OOS", "2023-01-01", None), ("FULL", "2020-01-01", None),
           ("IS2", "2020-07-01", "2023-12-31"), ("OOS2", "2024-01-01", None)]
COMBOS = [("BTC",), ("BTC", "ETH"), ("BTC", "ETH", "SOL"), ("BTC", "ETH", "SOL", "BNB")]


def load(name: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(ROOT, "data", FILES[name]))
    d = d[d["timestamp"] >= int(pd.Timestamp("2019-09-01", tz="UTC").timestamp() * 1000)]
    d.index = pd.to_datetime(d["timestamp"], unit="ms", utc=True)
    return d[["open", "high", "low", "close"]].astype("float64")


def ema_sma_seed(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = x[:n].mean()
    a = 2.0 / (n + 1)
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def sharpe(eq: pd.Series) -> float:
    r = eq.pct_change().dropna()
    return float(r.mean() / r.std() * math.sqrt(365)) if len(r) > 2 and r.std() > 0 else float("nan")


def max_dd(eq: pd.Series) -> float:
    a = eq.to_numpy()
    return float(((a - np.maximum.accumulate(a)) / np.maximum.accumulate(a)).min())


def summ(eq: pd.Series) -> dict:
    yrs = (eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 86400)
    ratio = eq.iloc[-1] / eq.iloc[0]
    dd = max_dd(eq)
    cagr = ratio ** (1 / yrs) - 1
    return {"net": ratio - 1, "cagr": cagr, "dd": dd, "sharpe": sharpe(eq), "calmar": cagr / abs(dd) if dd < 0 else float("nan")}


def gate(s: dict, b: dict) -> bool:
    return bool(s["sharpe"] > b["sharpe"] and s["calmar"] > b["calmar"] and s["dd"] > b["dd"])


def simulate(dfs: dict[str, pd.DataFrame], w: dict[str, float], start: pd.Timestamp, delay: int = 0, fee=FEE, slip=SLIP):
    """共用權益、每標的目標曝險 = 權重 × (close > EMA50)、收盤決策、下一根(+delay)開盤成交、費/滑每邊、只做多。"""
    keys = list(w)
    idx = None
    for k in keys:
        idx = dfs[k].index if idx is None else idx.intersection(dfs[k].index)
    o = {k: dfs[k]["open"].reindex(idx).to_numpy() for k in keys}
    c = {k: dfs[k]["close"].reindex(idx).to_numpy() for k in keys}
    on = {}
    for k in keys:
        e = ema_sma_seed(dfs[k]["close"].to_numpy(), 50)
        on[k] = pd.Series(dfs[k]["close"].to_numpy() > e, index=dfs[k].index).reindex(idx).fillna(False).to_numpy(dtype=bool)
    n = len(idx)
    s0 = int(idx.searchsorted(start))
    cash, qty = INIT, {k: 0.0 for k in keys}
    pending = {k: [] for k in keys}          # (fill_bar, target, reason)
    eq = np.full(n, np.nan)
    expo = {k: np.zeros(n) for k in keys}
    fills, notional = [], 0.0
    for i in range(s0, n):
        for k in keys:
            due = [p for p in pending[k] if p[0] <= i]
            if not due:
                continue
            pending[k] = [p for p in pending[k] if p[0] > i]
            _, target, reason = due[-1]
            eq_open = cash + sum(qty[j] * o[j][i] for j in keys)
            diff = target * eq_open / o[k][i] - qty[k]
            if abs(diff) * o[k][i] < 1e-6 * eq_open:
                continue
            if diff > 0:
                px = o[k][i] * (1 + slip)
                cash -= diff * px * (1 + fee)
            else:
                px = o[k][i] * (1 - slip)
                cash += -diff * px * (1 - fee)
            qty[k] += diff
            if qty[k] <= 1e-12:
                qty[k] = 0.0
            notional += abs(diff) * px
            fills.append((idx[i], k, reason, abs(diff) * px))
        eqv = cash + sum(qty[k] * c[k][i] for k in keys)
        for k in keys:
            target = w[k] if on[k][i] else 0.0
            cur = qty[k] * c[k][i] / eqv
            ref = pending[k][-1][1] if pending[k] else cur
            if target == 0 and ref > 0:
                pending[k].append((i + 1 + delay, 0.0, "趨勢結束"))
            elif target > 0 and ref == 0:
                pending[k].append((i + 1 + delay, target, "進場"))
            elif target > 0 and ref > 0 and abs(target - ref) > TH * w[k]:
                pending[k].append((i + 1 + delay, target, "調整曝險"))
            expo[k][i] = cur
        eq[i] = eqv
    equity = pd.Series(eq, index=idx).iloc[s0:]
    years = (idx[-1] - idx[s0]).days / 365.25
    total = sum(expo[k][s0:] for k in keys)
    f = pd.DataFrame(fills, columns=["time", "asset", "reason", "notional"])
    return {"equity": equity, "idx": idx, "s0": s0, "closes": c, "n_fills": len(fills),
            "turnover": float(notional / np.nanmean(eq[s0:]) / years), "time_in_market": float(np.mean(total > 0)),
            "avg_expo": {k: float(np.nanmean(expo[k][s0:])) for k in keys},
            "fills_by_reason": f.groupby("reason")["notional"].agg(["count", "sum"]).to_dict("index") if len(f) else {}}


def hold(dfs, w, idx, s0, rebalance: str | None):
    keys = list(w)
    c = {k: dfs[k]["close"].reindex(idx).to_numpy() for k in keys}
    qty = {k: INIT * (1 - FEE) * w[k] / c[k][s0] for k in keys}
    eq = np.full(len(idx), np.nan)
    last_m = idx[s0].month
    for i in range(s0, len(idx)):
        val = sum(qty[k] * c[k][i] for k in keys)
        if rebalance == "M" and idx[i].month != last_m:
            last_m = idx[i].month
            cost = sum(abs(val * w[k] - qty[k] * c[k][i]) * FEE for k in keys)
            val -= cost
            qty = {k: val * w[k] / c[k][i] for k in keys}
        eq[i] = val
    return pd.Series(eq, index=idx).iloc[s0:]


def window_slice(eq: pd.Series, wstart, wend, floor_start):
    a = max(pd.Timestamp(wstart, tz="UTC"), floor_start)
    e = eq.loc[a:] if wend is None else eq.loc[a:pd.Timestamp(wend, tz="UTC") + pd.Timedelta(hours=23)]
    return e / e.iloc[0] * INIT


def run_combo(dfs, assets, start, delay=0, fee=FEE, slip=SLIP):
    w = {a: 1.0 / len(assets) for a in assets}
    r = simulate(dfs, w, start, delay, fee, slip)
    idx, s0 = r["idx"], r["s0"]
    nr, mr = hold(dfs, w, idx, s0, None), hold(dfs, w, idx, s0, "M")
    out = {"start": str(idx[s0].date()), "n_fills": r["n_fills"], "turnover": r["turnover"], "time_in_market": r["time_in_market"],
           "avg_expo": r["avg_expo"], "fills_by_reason": r["fills_by_reason"], "windows": {}, "_eq": r["equity"], "_nr": nr, "_mr": mr}
    gates = 0
    for wn, ws, we in WINDOWS:
        e = window_slice(r["equity"], ws, we, idx[s0])
        if len(e) < 30:
            continue
        s = summ(e)
        bn, bm = summ(window_slice(nr, ws, we, idx[s0])), summ(window_slice(mr, ws, we, idx[s0]))
        # 新鮮基準：在該視窗起始日等權買入不再平衡
        fs = int(idx.searchsorted(e.index[0]))
        bf = summ(hold(dfs, w, idx, fs, None).loc[:e.index[-1]])
        gn, gm, gf = gate(s, bn), gate(s, bm), gate(s, bf)
        gates += int(gn) + int(gm)
        ratio = (e / window_slice(nr, ws, we, idx[s0])).to_numpy()
        out["windows"][wn] = {"start": str(e.index[0].date()), "sharpe": s["sharpe"], "dd": s["dd"], "calmar": s["calmar"], "net": s["net"],
                              "nr": bn, "mr": bm, "fresh_nr": bf, "gate_nr": gn, "gate_mr": gm, "gate_fresh_nr": gf,
                              "rel_dd_nr": float(np.nanmin(ratio / np.maximum.accumulate(ratio) - 1))}
    out["gates"] = gates
    return out


def paired_t(ea: pd.Series, eb: pd.Series, start: str, end: str | None):
    d = (ea.pct_change() - eb.pct_change()).loc[start:end].dropna()
    return float(d.mean() / d.std() * math.sqrt(len(d))), int(len(d))


def yearly(eq: pd.Series, dfs, w, idx):
    out = {}
    for y in (2023, 2024, 2025, 2026):
        e = eq.loc[f"{y}-01-01":f"{y}-12-31"]
        if len(e) < 30:
            continue
        fs = int(idx.searchsorted(e.index[0]))
        fresh = hold(dfs, w, idx, fs, None).loc[:e.index[-1]]
        out[y] = {"sharpe": sharpe(e), "dd": max_dd(e), "fresh_hold_sharpe": sharpe(fresh), "fresh_hold_dd": max_dd(fresh)}
    return out


def main():
    dfs = {k: load(k) for k in FILES}
    common = pd.Timestamp("2020-09-29", tz="UTC")
    ref = json.load(open(os.path.join(ROOT, "results", "levers", "diversification.json"), encoding="utf-8"))
    sp = ref["markets"]["spot"]["variants"]["ema50_plain"]
    # 確認 SOL EMA50 首個可用日
    e_sol = ema_sma_seed(dfs["SOL"]["close"].to_numpy(), 50)
    first_sol = dfs["SOL"].index[int(np.argmax(~np.isnan(e_sol)))]
    print("SOL EMA50 首個可用日（自算）:", first_sol.date(), "| 分析者:", sp["common_start"])

    res, report = {}, {"common_start_check": str(first_sol.date()), "combos": {}, "delta_vs_json": {}}
    max_abs_diff = 0.0
    for assets in COMBOS:
        nm = "+".join(assets)
        r = run_combo(dfs, assets, common)
        res[nm] = r
        j = sp["common"][nm]
        rows = {}
        for wn in r["windows"]:
            mine, ref_c = r["windows"][wn], j["windows"][wn]
            diffs = {"sharpe": mine["sharpe"] - ref_c["sharpe"], "dd": mine["dd"] - ref_c["max_dd"], "calmar": mine["calmar"] - ref_c["calmar"],
                     "nr_sharpe": mine["nr"]["sharpe"] - ref_c["hold_nr"]["sharpe"], "mr_sharpe": mine["mr"]["sharpe"] - ref_c["hold_mr"]["sharpe"],
                     "nr_dd": mine["nr"]["dd"] - ref_c["hold_nr"]["max_dd"], "rel_dd_nr": mine["rel_dd_nr"] - ref_c["hold_nr"]["rel_dd"]}
            max_abs_diff = max(max_abs_diff, max(abs(v) for v in diffs.values()))
            rows[wn] = {"mine": {"sharpe": round(mine["sharpe"], 3), "dd": round(mine["dd"], 4), "calmar": round(mine["calmar"], 3), "net": round(mine["net"], 3),
                                 "nr_sharpe": round(mine["nr"]["sharpe"], 3), "nr_dd": round(mine["nr"]["dd"], 4), "nr_net": round(mine["nr"]["net"], 3),
                                 "mr_sharpe": round(mine["mr"]["sharpe"], 3), "mr_dd": round(mine["mr"]["dd"], 4),
                                 "fresh_nr_sharpe": round(mine["fresh_nr"]["sharpe"], 3), "fresh_nr_dd": round(mine["fresh_nr"]["dd"], 4),
                                 "fresh_nr_calmar": round(mine["fresh_nr"]["calmar"], 3), "fresh_nr_net": round(mine["fresh_nr"]["net"], 3),
                                 "gate_nr": mine["gate_nr"], "gate_mr": mine["gate_mr"], "gate_fresh_nr": mine["gate_fresh_nr"]},
                        "json": {"sharpe": round(ref_c["sharpe"], 3), "dd": round(ref_c["max_dd"], 4), "nr_sharpe": round(ref_c["hold_nr"]["sharpe"], 3),
                                 "gate_nr": ref_c["hold_nr"]["gate"], "gate_mr": ref_c["hold_mr"]["gate"]},
                        "max_abs_diff": round(max(abs(v) for v in diffs.values()), 5)}
            print(f"{nm:18}{wn:5} S {mine['sharpe']:.3f} (json {ref_c['sharpe']:.3f})  DD {mine['dd']:.4f} (json {ref_c['max_dd']:.4f})  "
                  f"NR S {mine['nr']['sharpe']:.3f} (json {ref_c['hold_nr']['sharpe']:.3f})  freshNR S/DD/Calmar {mine['fresh_nr']['sharpe']:.2f}/{mine['fresh_nr']['dd']:.3f}/{mine['fresh_nr']['calmar']:.2f}  "
                  f"gates NR/MR/fresh {int(mine['gate_nr'])}/{int(mine['gate_mr'])}/{int(mine['gate_fresh_nr'])}")
        yr = yearly(r["_eq"], dfs, {a: 1 / len(assets) for a in assets}, r["idx"] if "idx" in r else None) if False else None
        report["combos"][nm] = {"start": r["start"], "gates_10": r["gates"], "json_gates_10": j["gates"], "n_fills": r["n_fills"], "json_n_fills": j["n_fills"],
                                "turnover": round(r["turnover"], 2), "json_turnover": round(j["notional_turnover_per_year"], 2),
                                "time_in_market": round(r["time_in_market"], 3), "avg_expo": {k: round(v, 3) for k, v in r["avg_expo"].items()},
                                "fills_by_reason": {k: {"count": int(v["count"]), "notional_share": round(v["sum"] / sum(x["sum"] for x in r["fills_by_reason"].values()), 3)} for k, v in r["fills_by_reason"].items()},
                                "windows": rows}
        print(f"  十格 {r['gates']}/10 (json {j['gates']}) | 成交 {r['n_fills']} (json {j['n_fills']}) | 換手 {r['turnover']:.2f} (json {j['notional_turnover_per_year']:.2f}) | 原因 {report['combos'][nm]['fills_by_reason']}")

    # 配對 t 對 BTC 單獨（共同起算）
    print("\n配對 t（日報酬差，對 BTC 單獨）")
    report["paired_t_vs_btc"] = {}
    for nm in ("BTC+ETH", "BTC+ETH+SOL", "BTC+ETH+SOL+BNB"):
        report["paired_t_vs_btc"][nm] = {}
        for wn, ws, we in WINDOWS:
            a = res[nm]["windows"][wn]["start"]
            t, n = paired_t(res[nm]["_eq"], res["BTC"]["_eq"], a, we)
            jt = sp["vs_btc_common"][nm][wn]["t_daily"]
            report["paired_t_vs_btc"][nm][wn] = {"t": round(t, 3), "n": n, "json_t": round(jt, 3)}
            print(f"  {nm:18}{wn:5} t {t:+.3f} n {n} (json {jt:+.3f})")

    # 逐年（OOS 年），持有用「該年 1/1 等權買入」的新鮮基準，對照分析者用 2020-09-29 漂移持有
    print("\n逐年 Sharpe（括號：新鮮等權持有）/ 回撤")
    report["yearly"] = {}
    for assets in COMBOS:
        nm = "+".join(assets)
        w = {a: 1 / len(assets) for a in assets}
        idx = res[nm]["_eq"].index
        full_idx = None
        for k in assets:
            full_idx = dfs[k].index if full_idx is None else full_idx.intersection(dfs[k].index)
        yr = yearly(res[nm]["_eq"], dfs, w, full_idx)
        jy = sp["oos_stability"][nm]["yearly"]
        beat_fresh = sum(int(v["sharpe"] > v["fresh_hold_sharpe"]) for v in yr.values())
        beat_json = sp["oos_stability"][nm]["years_beat_hold"]
        report["yearly"][nm] = {str(y): {"sharpe": round(v["sharpe"], 3), "json_sharpe": round(jy[str(y)]["sharpe"], 3), "fresh_hold_sharpe": round(v["fresh_hold_sharpe"], 3),
                                          "json_drifted_hold_sharpe": round(jy[str(y)]["hold_sharpe"], 3), "dd": round(v["dd"], 4)} for y, v in yr.items()}
        report["yearly"][nm]["worst_year_sharpe"] = round(min(v["sharpe"] for v in yr.values()), 3)
        report["yearly"][nm]["years_beat_fresh_hold"] = beat_fresh
        report["yearly"][nm]["json_years_beat_drifted_hold"] = beat_json
        print(f"  {nm:18}", "  ".join(f"{y}: {v['sharpe']:.2f} ({v['fresh_hold_sharpe']:.2f}; json漂移持有 {jy[str(y)]['hold_sharpe']:.2f}) / {v['dd']:.3f}" for y, v in yr.items()),
              f"| 最差年 {min(v['sharpe'] for v in yr.values()):.2f} | 贏新鮮持有 {beat_fresh}/{len(yr)} (json 贏漂移持有 {beat_json}/4)")

    # 4 標的漂移持有在 2023-01-01 的實際權重（說明「持有 NR」在 OOS 是什麼東西）
    print("\n漂移持有 NR 在各視窗起點的實際權重（共同起算 2020-09-29 買入）")
    report["drifted_weights"] = {}
    for assets in COMBOS[1:]:
        nm = "+".join(assets)
        full_idx = None
        for k in assets:
            full_idx = dfs[k].index if full_idx is None else full_idx.intersection(dfs[k].index)
        s0 = int(full_idx.searchsorted(common))
        q = {k: (1 / len(assets)) / dfs[k]["close"].reindex(full_idx).iloc[s0] for k in assets}
        report["drifted_weights"][nm] = {}
        for d in ("2023-01-01", "2024-01-01"):
            i = int(full_idx.searchsorted(pd.Timestamp(d, tz="UTC")))
            vals = {k: q[k] * dfs[k]["close"].reindex(full_idx).iloc[i] for k in assets}
            tot = sum(vals.values())
            report["drifted_weights"][nm][d] = {k: round(v / tot, 3) for k, v in vals.items()}
        print(f"  {nm:18}", report["drifted_weights"][nm])

    # 壓力：延遲一根、費 0.10% + 滑 0.20%（原生起算 = BTC/BTC+ETH 2019-10-20；含 SOL 2020-09-29）
    print("\n壓力（原生起算）")
    report["stress"] = {}
    for sname, kw in (("延遲1根", {"delay": 1}), ("費0.10%+滑0.20%", {"slip": 0.002})):
        report["stress"][sname] = {}
        for assets in COMBOS:
            nm = "+".join(assets)
            native = pd.Timestamp("2019-10-20", tz="UTC") if "SOL" not in assets else common
            r = run_combo(dfs, assets, native, **kw)
            j = sp["stress"][sname][nm]
            report["stress"][sname][nm] = {"gates_10": r["gates"], "json_gates_10": j["gates"],
                                           "OOS_sharpe": round(r["windows"]["OOS"]["sharpe"], 3), "json_OOS_sharpe": round(j["OOS"]["sharpe"], 3),
                                           "OOS2_sharpe": round(r["windows"]["OOS2"]["sharpe"], 3), "json_OOS2_sharpe": round(j["OOS2"]["sharpe"], 3)}
            print(f"  {sname:14}{nm:18} 十格 {r['gates']}/10 (json {j['gates']}) OOS S {r['windows']['OOS']['sharpe']:.3f} (json {j['OOS']['sharpe']:.3f}) OOS2 S {r['windows']['OOS2']['sharpe']:.3f} (json {j['OOS2']['sharpe']:.3f})")

    # 原生起算 BTC / BTC+ETH（2019-10-20 起、IS 從 2020-01-01）
    print("\n原生起算 BTC、BTC+ETH")
    report["native"] = {}
    for assets in COMBOS[:2]:
        nm = "+".join(assets)
        r = run_combo(dfs, assets, pd.Timestamp("2019-10-20", tz="UTC"))
        j = sp["native"][nm]
        report["native"][nm] = {wn: {"sharpe": round(r["windows"][wn]["sharpe"], 3), "json_sharpe": round(j["windows"][wn]["sharpe"], 3)} for wn in r["windows"]}
        report["native"][nm]["gates_10"] = r["gates"]
        report["native"][nm]["turnover"] = round(r["turnover"], 2)
        report["native"][nm]["n_fills"] = r["n_fills"]
        print(f"  {nm:10}", "  ".join(f"{wn} {r['windows'][wn]['sharpe']:.3f} (json {j['windows'][wn]['sharpe']:.3f})" for wn in r["windows"]), f"| 十格 {r['gates']} | 成交 {r['n_fills']} (json {j['n_fills']}) 換手 {r['turnover']:.2f}")

    report["max_abs_diff_common_panel"] = round(max_abs_diff, 6)
    print(f"\n共同起算面板 vs JSON 最大絕對差：{max_abs_diff:.6f}")
    with open(os.path.join(ROOT, "results", "levers", "diversification_verify.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print("→ results/levers/diversification_verify.json")


if __name__ == "__main__":
    main()
