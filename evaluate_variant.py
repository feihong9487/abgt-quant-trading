"""評估一個曝險型變體：多標的 × 多區間（含 2022 熊市）× 風險調整閘門 × 參數敏感度 × 對基準的 Δ × 壓力測試。

變體檔放在 experiments/variants/<name>.py，需提供：
    NAME, DESCRIPTION
    PARAMS: dict            # 數值參數；敏感度分析會逐一擾動
    build(df, params) -> {"in_regime": np.ndarray[bool] 或 {"enter":..., "exit":...}, "risk": RiskParams, "warmup": int}

用法：
    python evaluate_variant.py --variant vol_atr_channel
    python evaluate_variant.py --variant vol_atr_channel --sensitivity --baseline ema50_plain --stress
    python evaluate_variant.py --variant vol_atr_channel --symbols BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT
區間預設：IS 2020-01→2022-12、OOS 2023-01→今、FULL 2020-01→今；--alt-split 再加 IS2 2020-07→2023-12、OOS2 2024-01→今。
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abgt import data as dt                              # noqa: E402
from abgt.benchmark import buy_and_hold, gate, segment_table   # noqa: E402
from abgt.metrics import num, pct, summarize             # noqa: E402
from abgt.strategies.regime import simulate_regime, state_machine   # noqa: E402

DATA_SINCE = "2019-09-01"
DEFAULT_WINDOWS = "IS:2020-01-01:2022-12-31,OOS:2023-01-01:,FULL:2020-01-01:"
ALT_WINDOWS = "IS2:2020-07-01:2023-12-31,OOS2:2024-01-01:"
STRESS = {
    "費0.10%+滑0.05%": {"commission_pct": 0.001, "slippage_pct": 0.0005},
    "費0.10%+滑0.20%": {"commission_pct": 0.001, "slippage_pct": 0.002},
    "資金費率0.01%/8h": {"funding_rate_8h": 0.0001},
    "延遲1根": {"exec_delay_bars": 1},
}


def load_variant(name: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "variants", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"variant_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr in ("PARAMS", "build"):
        if not hasattr(mod, attr):
            raise SystemExit(f"變體 {name} 缺少 {attr}")
    return mod


def parse_windows(spec: str) -> list[tuple[str, str, str | None]]:
    out = []
    for part in spec.split(","):
        name, start, end = part.split(":")
        out.append((name, start, end or None))
    return out


def regime_key(in_regime) -> str:
    arr = state_machine(in_regime["enter"], in_regime["exit"]) if isinstance(in_regime, dict) else np.asarray(in_regime, dtype=bool)
    return hashlib.md5(np.packbits(arr).tobytes()).hexdigest()


def run_once(df: pd.DataFrame, mod, params: dict, windows, risk_override: dict | None = None) -> dict:
    built = mod.build(df, params)
    in_regime, rp, warmup = built["in_regime"], built["risk"], int(built.get("warmup", 0))
    if risk_override:
        rp = dataclasses.replace(rp, **risk_override)
    res = simulate_regime(df, in_regime, rp, warmup)
    extra = {k: v for k, v in res.extra.items() if k != "stop_events"}
    extra["n_stop_events"] = len(res.extra.get("stop_events", []))
    out = {"windows": {}, "extra": extra, "regime_key": regime_key(in_regime), "exposure_key": res.extra["exposure_key"],
           "warmup": warmup, "equity": res.equity}
    for wname, wstart, wend in windows:
        s0 = max(warmup, int(df.index.searchsorted(pd.Timestamp(wstart, tz="UTC"))))
        s1 = int(df.index.searchsorted(pd.Timestamp(wend, tz="UTC"), side="right")) if wend else len(df)
        if s1 - s0 < 30:
            continue
        eq = res.equity.loc[df.index[s0]:df.index[s1 - 1]]
        eq = eq / eq.iloc[0] * rp.initial_capital
        bh = buy_and_hold(df["close"].iloc[s0:s1], rp.initial_capital, rp.commission_pct)
        tr = res.trades[(res.trades.exit_time >= df.index[s0]) & (res.trades.exit_time <= df.index[s1 - 1])]
        rt = res.round_trips[(res.round_trips.end_time >= df.index[s0]) & (res.round_trips.end_time <= df.index[s1 - 1])]
        s, b = summarize(eq, tr, rt), summarize(bh)
        g = gate(s, b, segment_table(eq, bh), mode="risk")
        out["windows"][wname] = {"strategy": s, "buy_and_hold": b, "gate": g, "start": str(eq.index[0].date()), "end": str(eq.index[-1].date())}
    return out


def compare_to_baseline(df: pd.DataFrame, r: dict, rb: dict, windows) -> dict:
    """逐格 ΔSharpe / Δ回撤 / Δ淨報酬 + 日報酬差的配對 t 值。"""
    out = {}
    d_s = r["equity"].resample("1D").last().pct_change()
    d_b = rb["equity"].resample("1D").last().pct_change()
    for wname, w in r["windows"].items():
        if wname not in rb["windows"]:
            continue
        s, b = w["strategy"], rb["windows"][wname]["strategy"]
        diff = (d_s - d_b).loc[w["start"]:w["end"]].dropna()
        t = float(diff.mean() / diff.std() * math.sqrt(len(diff))) if len(diff) > 10 and diff.std() > 0 else float("nan")
        out[wname] = {"d_sharpe": s["sharpe"] - b["sharpe"], "d_max_dd": s["max_dd"] - b["max_dd"],
                      "d_net_return": s["net_return"] - b["net_return"], "t_daily": t, "n_days": int(len(diff))}
    return out


def sensitivity(df: pd.DataFrame, mod, params: dict, windows, factors: list[float], base_key: tuple) -> dict:
    """base_key = (regime_key, exposure_key)：兩者都相同的擾動才視為「沒有改變任何東西」而去重。"""
    rows = []
    seen = {base_key}
    for k, v in params.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        for f in factors:
            p2 = copy.deepcopy(params)
            p2[k] = max(1, int(round(v * f))) if isinstance(v, int) else v * f
            if p2[k] == v:
                continue
            try:
                r = run_once(df, mod, p2, windows)
            except Exception as ex:  # noqa: BLE001
                rows.append({"param": k, "factor": f, "value": p2[k], "error": str(ex)})
                continue
            key = (r["regime_key"], r["exposure_key"])
            row = {"param": k, "factor": f, "value": p2[k], "same_regime": key in seen}
            seen.add(key)
            for wname, w in r["windows"].items():
                row[f"{wname}_sharpe"] = w["strategy"]["sharpe"]
                row[f"{wname}_pass"] = w["gate"]["passed"]
            rows.append(row)
    df_s = pd.DataFrame(rows)
    eff = df_s[~df_s.get("same_regime", pd.Series(False, index=df_s.index)).fillna(False)] if len(df_s) else df_s
    summary = {"n": int(len(df_s)), "n_effective": int(len(eff))}
    for wname, *_ in windows:
        col = f"{wname}_sharpe"
        if col in eff:
            vals = eff[col].dropna()
            summary[wname] = {"sharpe_min": float(vals.min()) if len(vals) else None,
                              "sharpe_median": float(vals.median()) if len(vals) else None,
                              "sharpe_max": float(vals.max()) if len(vals) else None,
                              "pass_ratio": float(eff[f"{wname}_pass"].mean()) if f"{wname}_pass" in eff and len(eff) else None}
    return {"summary": summary, "rows": rows}


def stress(df: pd.DataFrame, mod, params: dict, windows, funding: pd.Series | None = None) -> dict:
    out = {}
    configs = dict(STRESS)
    if funding is not None:
        arr = funding.reindex(df.index).fillna(0.0).to_numpy(dtype="float64")
        configs["歷史資金費率"] = {"funding": arr}
    for name, ov in configs.items():
        r = run_once(df, mod, params, windows, risk_override=ov)
        out[name] = {w: {"sharpe": x["strategy"]["sharpe"], "max_dd": x["strategy"]["max_dd"], "pass": x["gate"]["passed"]} for w, x in r["windows"].items()}
        if name == "歷史資金費率":
            years = (df.index[-1] - df.index[max(r["warmup"], 0)]).days / 365.25
            avg_equity = float(r["equity"].mean())
            out[name]["_funding_paid"] = r["extra"]["funding_paid"]
            out[name]["_funding_cost_pct_per_year"] = r["extra"]["funding_paid"] / avg_equity / years if years > 0 and avg_equity > 0 else None
    return out


def fmt_windows(sym: str, r: dict, cmp_: dict | None) -> str:
    lines = [f"--- {sym} ---",
             f"{'區間':6}{'':6}{'淨報酬':>9}{'回撤':>8}{'CAGR':>8}{'Sharpe':>8}{'Calmar':>8}{'回合':>5}{'閘門':>5}" + (f"{'ΔSharpe':>9}{'Δ回撤':>8}{'t':>7}" if cmp_ else "")]
    for wname, w in r["windows"].items():
        s, b, g = w["strategy"], w["buy_and_hold"], w["gate"]
        extra = ""
        if cmp_ and wname in cmp_:
            cc = cmp_[wname]
            extra = f"{cc['d_sharpe']:>+9.2f}{cc['d_max_dd'] * 100:>+7.1f}%{num(cc['t_daily']):>7}"
        lines.append(f"{wname:6}{'策略':6}{pct(s['net_return']):>9}{pct(s['max_dd']):>8}{pct(s['cagr']):>8}{num(s['sharpe']):>8}{num(s['calmar']):>8}{s.get('round_trips', 0):>5}{'✓' if g['passed'] else '✗':>5}{extra}")
        lines.append(f"{'':6}{'持有':6}{pct(b['net_return']):>9}{pct(b['max_dd']):>8}{pct(b['cagr']):>8}{num(b['sharpe']):>8}{num(b['calmar']):>8}")
    e = r["extra"]
    lines.append(f"平均曝險 {e['avg_exposure']:.2f}（趨勢內 {e.get('avg_exposure_in_regime', 0):.2f}，部分曝險 {e.get('partial_days', 0)} 天）| 在場 {e['time_in_market']:.0%} | 進/出 {e.get('regime_entries', 0)}/{e.get('regime_exits', 0)} | 成交 {e.get('n_fills', 0)} | ATR 停損 {e['stops']} | 回撤暫停 {e['dd_pauses']}" + (f" | 資金費率 {e['funding_paid']:.0f}" if e.get("funding_paid") else ""))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--symbols", default="BTC/USDT:USDT,ETH/USDT:USDT")
    ap.add_argument("--tf", default="1d")
    ap.add_argument("--windows", default=DEFAULT_WINDOWS)
    ap.add_argument("--alt-split", action="store_true", help="加上替換切分 IS2 2020-07→2023-12 / OOS2 2024-01→")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--sens-factors", default="0.75,1.25", help="敏感度擾動倍數，例如 0.5,0.75,1.25,1.5")
    ap.add_argument("--baseline", help="對照變體 id，輸出逐格 Δ 與配對 t 值")
    ap.add_argument("--stress", action="store_true", help="四組壓力情境：高手續費、高滑點、資金費率、延遲一根")
    ap.add_argument("--fetch", action="store_true", help="連網補資料（預設用快取）")
    ap.add_argument("--out", default="results/variants")
    args = ap.parse_args()

    mod = load_variant(args.variant)
    windows = parse_windows(args.windows + ("," + ALT_WINDOWS if args.alt_split else ""))
    factors = [float(x) for x in args.sens_factors.split(",")]
    params = copy.deepcopy(mod.PARAMS)
    base_mod = load_variant(args.baseline) if args.baseline else None
    report = {"variant": args.variant, "name": getattr(mod, "NAME", args.variant), "description": getattr(mod, "DESCRIPTION", ""),
              "params": params, "baseline": args.baseline, "windows": [w[0] for w in windows], "symbols": {}}
    text = [f"=== 變體 {report['name']}：{report['description']}", f"參數 {json.dumps(params, ensure_ascii=False)}" + (f" | 對照 {args.baseline}" if args.baseline else "")]
    for sym in args.symbols.split(","):
        df = dt.fetch_ohlcv(sym, args.tf, DATA_SINCE, "data", verbose=False) if args.fetch else dt.load_cached("data", sym, args.tf, DATA_SINCE)
        if df is None:
            print(f"{sym} 沒有快取，加 --fetch")
            return 2
        r = run_once(df, mod, params, windows)
        cmp_ = None
        if base_mod is not None:
            rb = run_once(df, base_mod, copy.deepcopy(base_mod.PARAMS), windows)
            cmp_ = compare_to_baseline(df, r, rb, windows)
            r["vs_baseline"] = cmp_
        if args.sensitivity:
            r["sensitivity"] = sensitivity(df, mod, params, windows, factors, (r["regime_key"], r["exposure_key"]))
        if args.stress:
            r["stress"] = stress(df, mod, params, windows, dt.load_funding_cached("data", sym))
        r.pop("equity", None)
        report["symbols"][sym] = r
        text.append(fmt_windows(sym, r, cmp_))
        if args.sensitivity:
            ss = r["sensitivity"]["summary"]
            parts = [f"{w}: Sharpe {num(v['sharpe_min'])}~{num(v['sharpe_max'])} 通過率 {v['pass_ratio']:.0%}"
                     for w, v in ss.items() if w not in ("n", "n_effective") and v.get("sharpe_min") is not None]
            text.append(f"敏感度（{ss['n']} 組擾動，{ss['n_effective']} 組真的改變 in_regime；倍數 {args.sens_factors}）" + (" | " + " | ".join(parts) if parts else ""))
        if args.stress:
            for name, per_w in r["stress"].items():
                cells = "  ".join(f"{w}:{num(x['sharpe'])}{'✓' if x['pass'] else '✗'}" for w, x in per_w.items() if not w.startswith("_"))
                tail = f"  （資金費率成本 ≈ 平均權益的 {per_w['_funding_cost_pct_per_year'] * 100:.1f}%/年）" if per_w.get("_funding_cost_pct_per_year") else ""
                text.append(f"壓力 {name:14} {cells}{tail}")
    out_text = "\n".join(text)
    print(out_text)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.variant}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(args.out, f"{args.variant}.txt"), "w", encoding="utf-8") as f:
        f.write(out_text + "\n")
    print(f"\n→ {args.out}/{args.variant}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
