"""把迴圈狀態、變體結果、權益曲線、回合進度彙整成 dashboard/index.html（純靜態，資料內嵌）。

用法：python build_dashboard.py            # 讀快取與 results/，寫 dashboard/index.html
之後用 Artifact 工具把 dashboard/index.html 重新發佈到同一個 URL。
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from abgt import data as dt                              # noqa: E402
from abgt.benchmark import buy_and_hold                  # noqa: E402
from abgt.strategies.regime import simulate_regime       # noqa: E402
from evaluate_variant import DATA_SINCE, load_variant    # noqa: E402

ASSETS = {"BTC": "BTC/USDT:USDT", "ETH": "ETH/USDT:USDT", "SOL": "SOL/USDT:USDT", "BNB": "BNB/USDT:USDT"}
CELLS = [f"{a}_{w}" for a in ("BTC", "ETH") for w in ("IS", "OOS", "FULL", "IS2", "OOS2")]
CURVE_VARIANTS = ["ema50_plain", "vol_atr_channel"]
SUBAGENT_DIR = os.path.expanduser("~/.claude/projects/C--Users-feihong/287fbefb-730e-4409-9cf4-4b83870f5428/subagents/workflows")


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def num(x):
    return None if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else (round(float(x), 4) if isinstance(x, (int, float)) else x)


def equity_curves() -> tuple[dict, dict]:
    out, windows = {}, {}
    for key in ("BTC", "ETH"):
        df = dt.load_cached("data", ASSETS[key], "1d", DATA_SINCE)
        if df is None:
            continue
        s0 = int(df.index.searchsorted(pd.Timestamp("2020-01-01", tz="UTC")))
        series = {}
        for vid in CURVE_VARIANTS:
            mod = load_variant(vid)
            built = mod.build(df, dict(mod.PARAMS))
            res = simulate_regime(df, built["in_regime"], built["risk"], int(built["warmup"]))
            eq = res.equity.iloc[s0 - int(built["warmup"]):] if s0 >= int(built["warmup"]) else res.equity
            eq = eq / eq.iloc[0] * 100
            series[vid] = eq
        bh = buy_and_hold(df["close"].iloc[s0:], 100.0, 0.0005)
        idx = series[CURVE_VARIANTS[0]].index
        out[key] = {"dates": [d.strftime("%Y-%m-%d") for d in idx],
                    **{vid: [round(float(v), 2) for v in s.reindex(idx).to_numpy()] for vid, s in series.items()},
                    "bh": [round(float(v), 2) for v in bh.reindex(idx).to_numpy()]}
    windows = {"IS": ["2020-01-01", "2022-12-31"], "OOS": ["2023-01-01", None], "IS2": ["2020-07-01", "2023-12-31"], "OOS2": ["2024-01-01", None]}
    return out, windows


def variants_table(ledger: list[dict]) -> list[dict]:
    round_of = {}
    for i, item in enumerate(ledger):
        round_of[item["id"]] = 0 if i < 4 else (1 if i < 16 else 2)
    notes = {item["id"]: item["note"] for item in ledger}
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "variants", "*.json"))):
        d = jload(path)
        vid = d["variant"]
        cells, sharpes, doos, sens = {}, [], [], []
        for a in ("BTC", "ETH"):
            sym = d["symbols"].get(ASSETS[a], {})
            for w, x in sym.get("windows", {}).items():
                s = x["strategy"]
                cells[f"{a}_{w}"] = {"sharpe": num(s["sharpe"]), "dd": num(s["max_dd"]), "ret": num(s["net_return"]), "gate": bool(x["gate"]["passed"]),
                                     "bh_sharpe": num(x["buy_and_hold"]["sharpe"])}
                if w in ("IS", "OOS", "FULL"):
                    sharpes.append(s["sharpe"])
                if w in ("OOS", "OOS2") and sym.get("vs_baseline", {}).get(w):
                    doos.append(sym["vs_baseline"][w]["d_sharpe"])
                ss = sym.get("sensitivity", {}).get("summary", {}).get(w)
                if w in ("OOS", "OOS2") and ss and ss.get("pass_ratio") is not None:
                    sens.append(ss["pass_ratio"])
        gates = sum(1 for k in CELLS if cells.get(k, {}).get("gate"))
        evaluated = sum(1 for k in CELLS if k in cells)
        rows.append({
            "id": vid, "name": d.get("name", vid), "description": d.get("description", ""), "params": d.get("params", {}),
            "round": round_of.get(vid, 3), "note": notes.get(vid, ""),
            "mean_sharpe": num(np.mean(sharpes)) if sharpes else None, "gates": gates, "evaluated": evaluated,
            "min_doos": num(min(doos)) if doos else None, "min_sens": num(min(sens)) if sens else None,
            "cells": cells, "updated": datetime.fromtimestamp(os.path.getmtime(path)).strftime("%m-%d %H:%M"),
        })
    rows.sort(key=lambda r: (-r["gates"], -(r["mean_sharpe"] or -9)))
    return rows


def legacy_results() -> list[dict]:
    rows = []
    for name, label in (("apex", "APEX v4.0 只做多"), ("bgt_multi", "BGT-S2 多信號"), ("bgt_multi_kfdir", "BGT-S2 + 45 分方向閘"), ("bgt_kf", "BGT-S1 KF 單信號"), ("channel", "ATR 通道遲滯（正式策略）")):
        p = os.path.join(ROOT, "results", f"{name}_summary.json")
        if not os.path.exists(p):
            continue
        d = jload(p)
        s, b = d["strategy"], d["buy_and_hold"]
        rows.append({"label": d.get("label", label), "symbol": d["symbol"], "tf": d["tf"],
                     "net_return": num(s["net_return"]), "max_dd": num(s["max_dd"]), "sharpe": num(s["sharpe"]),
                     "round_trips": s.get("round_trips"), "win_rate": num(s.get("win_rate")), "pf": num(s.get("profit_factor")),
                     "bh_net": num(b["net_return"]), "bh_dd": num(b["max_dd"]), "bh_sharpe": num(b["sharpe"]),
                     "gate": d["gate"].get("passed"), "gate_mode": d["gate"].get("mode", "absolute")})
    return rows


def stress_table() -> dict:
    out = {}
    p = os.path.join(ROOT, "results", "variants", "ema50_plain.json")
    if not os.path.exists(p):
        return out
    d = jload(p)
    for a in ("BTC", "ETH"):
        st = d["symbols"].get(ASSETS[a], {}).get("stress")
        if st:
            out[a] = {name: {w: {"sharpe": num(x["sharpe"]), "pass": bool(x["pass"])} for w, x in per.items() if not w.startswith("_")} for name, per in st.items()}
            for name, per in st.items():
                if per.get("_funding_cost_pct_per_year") is not None:
                    out[a][name]["_cost"] = num(per["_funding_cost_pct_per_year"])
    return out


def coverage() -> list[dict]:
    rows = []
    for key, sym in ASSETS.items():
        df = dt.load_cached("data", sym, "1d", DATA_SINCE)
        fd = dt.load_funding_cached("data", sym)
        if df is None:
            continue
        rows.append({"asset": key, "symbol": sym, "tf": "1d", "start": df.index[0].strftime("%Y-%m-%d"), "end": df.index[-1].strftime("%Y-%m-%d"),
                     "bars": int(len(df)), "funding_days": int(len(fd)) if fd is not None else 0,
                     "funding_annual": num(fd.mean() * 365) if fd is not None else None})
    eth15 = dt.load_cached("data", "ETH/USDT:USDT", "15m", "2022-12-01")
    if eth15 is not None:
        rows.append({"asset": "ETH", "symbol": "ETH/USDT:USDT", "tf": "15m", "start": eth15.index[0].strftime("%Y-%m-%d"), "end": eth15.index[-1].strftime("%Y-%m-%d"),
                     "bars": int(len(eth15)), "funding_days": 0, "funding_annual": None})
    return rows


def workflow_progress(run_id: str | None) -> dict:
    if not run_id:
        return {}
    path = os.path.join(SUBAGENT_DIR, run_id, "journal.jsonl")
    if not os.path.exists(path):
        return {"run_id": run_id, "journal": False}
    labels_done, labels_started = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "result" and r.get("label"):
                labels_done.append(r["label"])
            elif r.get("type") in ("started", "launched") and r.get("label"):
                labels_started.append(r["label"])
    phases = {}
    for lab in labels_done:
        ph = lab.split(":")[0]
        phases[ph] = phases.get(ph, 0) + 1
    running = [x for x in labels_started if x not in labels_done]
    return {"run_id": run_id, "journal": True, "done": len(labels_done), "started": len(set(labels_started)), "phases": phases, "running": running[-8:]}


def main() -> int:
    state = jload(os.path.join(ROOT, "experiments", "rounds", "loop_state.json"))
    meta_path = os.path.join(ROOT, "experiments", "rounds", "rounds_meta.json")
    rounds = jload(meta_path) if os.path.exists(meta_path) else []
    curves, windows = equity_curves()
    data = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "loop": {k: state.get(k) for k in ("round", "status", "rounds_without_champion", "stop_rule", "updated")},
        "baseline": state["baseline"],
        "directions": [{"key": d["key"], "n": d.get("n", 2), "prompt": d["prompt"][:160]} for d in state.get(f"directions_round_{state['round']}", [])],
        "rounds": rounds,
        "progress": workflow_progress(state.get("workflow", {}).get("run_id")),
        "curves": curves, "windows": windows,
        "variants": variants_table(state["ledger"]),
        "legacy": legacy_results(),
        "stress": stress_table(),
        "coverage": coverage(),
        "cells": CELLS,
    }
    tpl = open(os.path.join(ROOT, "dashboard", "template.html"), encoding="utf-8").read()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = tpl.replace("/*__DATA__*/null", payload)
    os.makedirs(os.path.join(ROOT, "dashboard"), exist_ok=True)
    out = os.path.join(ROOT, "dashboard", "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"dashboard → {out}（{len(html) / 1024:.0f} KB，{len(data['variants'])} 個變體，{len(curves)} 條曲線標的）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
