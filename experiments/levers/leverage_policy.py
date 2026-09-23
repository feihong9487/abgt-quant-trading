"""槓桿政策優化（事前登記的網格 + 兩組獨立切分的選擇/驗證）。

問題：在「合理風險」下，槓桿能不能把 ema50_plain 組合的風險調整優勢換成更多報酬？
工具（引擎已支援）：max_exposure（上限）× vol_target（每標的波動目標，低波動時才放大）× dd_lever_full（回撤降槓桿），
全倉強平模擬（liq_level 1.1、罰 2%）、借款利息 borrow_rate_annual。

口徑：BTC/USDT + ETH/USDT 現貨 50/50、費 0.10%/邊、滑 0.05%、借款 10%/年（壓力 15%）。

事前登記的選擇規則（寫在跑之前，不看 OOS）：
  在選擇視窗（切分 A = IS 2020→2022；切分 B = IS2 2020-07→2023）上，候選要同時滿足
    (1) 對兩種持有基準閘門全過（Sharpe、Calmar 高於持有且回撤更淺）
    (2) 零強平
    (3) Sharpe ≥ 1× 基準 Sharpe − 0.05（槓桿不能明顯稀釋風險調整報酬）
    (4) 風險預算：
        同風險層 = 最大回撤不比 1× 基準深超過 2 個百分點
        持有風險層 = 最大回撤比持有淺至少 5 個百分點
  在可行集內取 CAGR 最高者；另報「高原檢查」：它在網格上的鄰居有幾個也可行。
  然後在驗證視窗（A → OOS 2023→；B → OOS2 2024→）報同樣的數字與閘門，外加壓力情境。

用法：python experiments/levers/leverage_policy.py
輸出：results/levers/leverage_policy.json / .md
"""
from __future__ import annotations

import copy
import dataclasses
import itertools
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from abgt import data as dt                                                   # noqa: E402
from abgt.portfolio import align, buy_and_hold_portfolio, simulate_portfolio  # noqa: E402
from abgt.strategies.regime import RiskParams                                 # noqa: E402
from evaluate_portfolio import window_metrics                                 # noqa: E402
from evaluate_variant import DATA_SINCE, load_variant, parse_windows          # noqa: E402

SYMBOLS = ["BTC/USDT", "ETH/USDT"]
WEIGHTS = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
WINDOWS = parse_windows("IS:2020-01-01:2022-12-31,OOS:2023-01-01:,FULL:2020-01-01:,IS2:2020-07-01:2023-12-31,OOS2:2024-01-01:")
SPLITS = {"A": ("IS", "OOS"), "B": ("IS2", "OOS2")}
BASE = dict(commission_pct=0.001, slippage_pct=0.0005, rebalance_threshold=0.15, borrow_rate_annual=0.10,
            liq_level=1.1, liq_penalty=0.02, vol_lookback=20)
GRID_X = [1.0, 1.25, 1.5, 2.0, 2.5]
GRID_VT = [0.0, 0.5, 0.7, 0.9]
GRID_DDL = [0.0, 0.15, 0.30]
GRID_RD = [False, True]            # redeploy_idle：閒置權重分給趨勢內標的（不借錢的「集中」槓桿）
STRESS = {"借款15%": {"borrow_rate_annual": 0.15}, "費0.10%+滑0.20%": {"slippage_pct": 0.002},
          "延遲1根": {"exec_delay_bars": 1}, "強平罰10%": {"liq_penalty": 0.10}}
BENCHES = ("持有不再平衡", "持有月再平衡")


def load():
    dfs = {s: dt.load_cached("data", s, "1d", DATA_SINCE) for s in SYMBOLS}
    idx = align(dfs)
    mod = load_variant("ema50_plain")
    regimes, warm = {}, []
    for s in SYMBOLS:
        b = mod.build(dfs[s], copy.deepcopy(mod.PARAMS))
        regimes[s] = b["in_regime"]
        warm.append(int(idx.searchsorted(dfs[s].index[int(b["warmup"])])))
    return dfs, idx, regimes, max(warm)


def configs():
    out = []
    for rd, x, vt, ddl in itertools.product(GRID_RD, GRID_X, GRID_VT, GRID_DDL):
        if x == 1.0 and ddl > 0:
            continue                      # 1× 時回撤降槓桿不作用
        out.append({"max_exposure": x, "vol_target": vt, "dd_lever_full": ddl, "redeploy_idle": rd})
    return out


def key(c):
    return f"x{c['max_exposure']}_vt{c['vol_target']}_ddl{c['dd_lever_full']}" + ("_rd" if c.get("redeploy_idle") else "")


def neighbors(c):
    def adj(lst, v):
        i = lst.index(v)
        return [lst[j] for j in (i - 1, i + 1) if 0 <= j < len(lst)]
    out = []
    for f, lst in (("max_exposure", GRID_X), ("vol_target", GRID_VT), ("dd_lever_full", GRID_DDL)):
        for v in adj(lst, c[f]):
            n = dict(c, **{f: v})
            if n["max_exposure"] == 1.0 and n["dd_lever_full"] > 0:
                continue
            out.append(n)
    return out


def run_one(dfs, idx, regimes, warm, benches, cfg, overrides=None):
    rp = RiskParams(**BASE, **cfg)
    if overrides:
        rp = dataclasses.replace(rp, **overrides)
    res = simulate_portfolio(dfs, regimes, rp, WEIGHTS, warm)
    wm = window_metrics(res.equity, benches, idx, WINDOWS, warm, rp.initial_capital)
    row = {"cfg": cfg, "key": key(cfg), "overrides": overrides or {}, "windows": {}}
    for w, m in wm.items():
        s = m["strategy"]
        row["windows"][w] = {"cagr": s["cagr"], "net": s["net_return"], "dd": s["max_dd"], "sharpe": s["sharpe"], "calmar": s["calmar"],
                             "gate": all(m["bench"][b]["gate"]["passed"] for b in BENCHES),
                             "hold_dd": min(m["bench"][b]["summary"]["max_dd"] for b in BENCHES),
                             "hold_sharpe": max(m["bench"][b]["summary"]["sharpe"] for b in BENCHES),
                             "hold_cagr": max(m["bench"][b]["summary"]["cagr"] for b in BENCHES)}
    e = res.extra
    # 強平只算落在該視窗內的
    for w, (_, ws, we) in zip(wm.keys(), [x for x in WINDOWS if x[0] in wm]):
        evs = [ev for ev in e["liq_events"] if ev["time"] >= ws and (not we or ev["time"] <= we)]
        row["windows"][w]["liqs"] = len(evs)
    row["extra"] = {"avg_exposure": e["avg_exposure"], "max_exposure_seen": e["max_exposure_seen"], "borrow_paid": e["borrow_paid"],
                    "min_margin_level": e["min_margin_level"], "liquidations": e["liquidations"], "liq_events": e["liq_events"],
                    "turnover": e["notional_turnover_per_year"], "n_fills": e["n_fills"],
                    "borrow_days_frac": e["borrow_days_frac"]}
    return row


def feasible(row, base_row, w, tier):
    r, b = row["windows"][w], base_row["windows"][w]
    if not r["gate"] or r["liqs"] > 0 or r["sharpe"] < b["sharpe"] - 0.05:
        return False
    if tier == "same_risk":
        return r["dd"] >= b["dd"] - 0.02
    return r["dd"] >= r["hold_dd"] + 0.05


def fmt(x, p=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x * 100:+.1f}%" if p else f"{x:.2f}"


def main():
    dfs, idx, regimes, warm = load()
    benches = {"持有不再平衡": buy_and_hold_portfolio(dfs, WEIGHTS, idx, warm, 10_000.0, BASE["commission_pct"], None),
               "持有月再平衡": buy_and_hold_portfolio(dfs, WEIGHTS, idx, warm, 10_000.0, BASE["commission_pct"], "M")}
    rows = {}
    for cfg in configs():
        r = run_one(dfs, idx, regimes, warm, benches, cfg)
        rows[r["key"]] = r
        f = r["windows"]["FULL"]
        print(f"{r['key']:25} FULL CAGR {fmt(f['cagr'])} DD {fmt(f['dd'])} Sharpe {fmt(f['sharpe'], False)} 強平 {r['extra']['liquidations']} "
              f"平均曝險 {r['extra']['avg_exposure']:.2f} 最低保證金率 {fmt(r['extra']['min_margin_level'], False)}", flush=True)
    base = rows[key({"max_exposure": 1.0, "vol_target": 0.0, "dd_lever_full": 0.0, "redeploy_idle": False})]

    picks = {}
    for split, (sel, val) in SPLITS.items():
        for tier in ("same_risk", "hold_risk"):
            feas = [r for r in rows.values() if feasible(r, base, sel, tier)]
            if not feas:
                picks[f"{split}_{tier}"] = None
                continue
            best = max(feas, key=lambda r: r["windows"][sel]["cagr"])
            nb = neighbors(best["cfg"])
            nb_ok = sum(1 for n in nb if key(n) in rows and feasible(rows[key(n)], base, sel, tier))
            stress = {}
            for sname, ov in STRESS.items():
                s = run_one(dfs, idx, regimes, warm, benches, best["cfg"], ov)
                sb = run_one(dfs, idx, regimes, warm, benches, base["cfg"], ov) if sname != "借款15%" and sname != "強平罰10%" else base
                stress[sname] = {w: {"cagr": s["windows"][w]["cagr"], "dd": s["windows"][w]["dd"], "sharpe": s["windows"][w]["sharpe"],
                                     "gate": s["windows"][w]["gate"], "liqs": s["windows"][w]["liqs"],
                                     "base_cagr": sb["windows"][w]["cagr"], "base_sharpe": sb["windows"][w]["sharpe"]}
                                 for w in (sel, val)}
            picks[f"{split}_{tier}"] = {"key": best["key"], "cfg": best["cfg"], "sel": sel, "val": val,
                                        "n_feasible": len(feas), "neighbors_feasible": f"{nb_ok}/{len(nb)}",
                                        "sel_row": best["windows"][sel], "val_row": best["windows"][val],
                                        "val_feasible": feasible(best, base, val, tier),
                                        "base_sel": base["windows"][sel], "base_val": base["windows"][val],
                                        "full": best["windows"]["FULL"], "base_full": base["windows"]["FULL"],
                                        "extra": best["extra"], "stress": stress}

    os.makedirs("results/levers", exist_ok=True)
    with open("results/levers/leverage_policy.json", "w", encoding="utf-8") as f:
        json.dump({"base": BASE, "grid": {"max_exposure": GRID_X, "vol_target": GRID_VT, "dd_lever_full": GRID_DDL, "redeploy_idle": GRID_RD},
                   "rows": rows, "picks": picks}, f, ensure_ascii=False, indent=1, default=str)

    L = ["# 槓桿政策優化（事前登記網格）", "",
         f"口徑：BTC/ETH 現貨 50/50 ema50_plain，費 0.10% 滑 0.05%，借款 10%/年，強平 1.1（罰 2%）。網格 {len(rows)} 組。", "",
         "## 全網格（FULL 2020→）", "",
         "| 設定 | CAGR | 回撤 | Sharpe | Calmar | 閘門 | 平均曝險 | 最高曝險 | 借款天數 | 最低保證金率 | 強平 | 借款利息 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows.values(), key=lambda r: -r["windows"]["FULL"]["cagr"]):
        f, e = r["windows"]["FULL"], r["extra"]
        L.append(f"| {r['key']} | {fmt(f['cagr'])} | {fmt(f['dd'])} | {fmt(f['sharpe'], False)} | {fmt(f['calmar'], False)} | {'✓' if f['gate'] else '✗'} | "
                 f"{e['avg_exposure']:.2f} | {e['max_exposure_seen']:.2f} | {e['borrow_days_frac']:.0%} | {fmt(e['min_margin_level'], False)} | {e['liquidations']} | {e['borrow_paid']:,.0f} |")
    f = base["windows"]["FULL"]
    L += ["", f"持有（FULL）：CAGR {fmt(f['hold_cagr'])}、回撤 {fmt(f['hold_dd'])}、Sharpe {fmt(f['hold_sharpe'], False)}", "",
          "## 選擇 → 驗證", "",
          "| 切分/風險層 | 選中 | 可行數 | 鄰居可行 | 選擇窗 CAGR / 回撤 / Sharpe | 1× 同窗 | 驗證窗 CAGR / 回撤 / Sharpe | 1× 同窗 | 驗證窗仍可行 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, p in picks.items():
        if p is None:
            L.append(f"| {name} | 無可行 | 0 | — | — | — | — | — | — |")
            continue
        s, v, bs, bv = p["sel_row"], p["val_row"], p["base_sel"], p["base_val"]
        L.append(f"| {name} | {p['key']} | {p['n_feasible']} | {p['neighbors_feasible']} | {fmt(s['cagr'])} / {fmt(s['dd'])} / {fmt(s['sharpe'], False)} | "
                 f"{fmt(bs['cagr'])} / {fmt(bs['dd'])} / {fmt(bs['sharpe'], False)} | {fmt(v['cagr'])} / {fmt(v['dd'])} / {fmt(v['sharpe'], False)} | "
                 f"{fmt(bv['cagr'])} / {fmt(bv['dd'])} / {fmt(bv['sharpe'], False)} | {'✓' if p['val_feasible'] else '✗'} |")
    L += ["", "## 壓力（選中設定，驗證窗）", "", "| 切分/風險層 | 情境 | CAGR | 1× CAGR | Sharpe | 1× Sharpe | 回撤 | 閘門 | 強平 |", "|---|---|---|---|---|---|---|---|---|"]
    for name, p in picks.items():
        if p is None:
            continue
        for sname, st in p["stress"].items():
            x = st[p["val"]]
            L.append(f"| {name} | {sname} | {fmt(x['cagr'])} | {fmt(x['base_cagr'])} | {fmt(x['sharpe'], False)} | {fmt(x['base_sharpe'], False)} | {fmt(x['dd'])} | {'✓' if x['gate'] else '✗'} | {x['liqs']} |")
    with open("results/levers/leverage_policy.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[-(len(picks) * 6 + 12):]))
    print("→ results/levers/leverage_policy.md")


if __name__ == "__main__":
    main()
