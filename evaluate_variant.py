"""評估一個曝險型變體：多標的 × 多區間（含 2022 熊市）× 風險調整閘門 × 參數敏感度。

變體檔放在 experiments/variants/<name>.py，需提供：
    NAME, DESCRIPTION
    PARAMS: dict            # 數值參數；敏感度分析會逐一 ±25% 擾動
    build(df, params) -> {"in_regime": np.ndarray[bool], "risk": RiskParams, "warmup": int}

用法：
    python evaluate_variant.py --variant ema50_atr
    python evaluate_variant.py --variant ema50_atr --symbols BTC/USDT:USDT,ETH/USDT:USDT --sensitivity
區間預設：IS 2020-01-01→2022-12-31（含牛轉熊）、OOS 2023-01-01→今、FULL 2020-01-01→今。
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abgt import data as dt                              # noqa: E402
from abgt.benchmark import buy_and_hold, gate, segment_table   # noqa: E402
from abgt.metrics import num, pct, summarize             # noqa: E402
from abgt.strategies.regime import simulate_regime       # noqa: E402

DATA_SINCE = "2019-09-01"
DEFAULT_WINDOWS = "IS:2020-01-01:2022-12-31,OOS:2023-01-01:,FULL:2020-01-01:"


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


def run_once(df: pd.DataFrame, mod, params: dict, windows, fee: float) -> dict:
    built = mod.build(df, params)
    in_regime, rp, warmup = built["in_regime"], built["risk"], int(built.get("warmup", 0))
    res = simulate_regime(df, in_regime, rp, warmup)
    out = {"windows": {}, "extra": res.extra}
    for wname, wstart, wend in windows:
        s0 = max(warmup, int(df.index.searchsorted(pd.Timestamp(wstart, tz="UTC"))))
        s1 = int(df.index.searchsorted(pd.Timestamp(wend, tz="UTC"), side="right")) if wend else len(df)
        if s1 - s0 < 30:
            continue
        eq = res.equity.loc[df.index[s0]:df.index[s1 - 1]]
        eq = eq / eq.iloc[0] * rp.initial_capital
        bh = buy_and_hold(df["close"].iloc[s0:s1], rp.initial_capital, fee)
        tr = res.trades[(res.trades.exit_time >= df.index[s0]) & (res.trades.exit_time <= df.index[s1 - 1])]
        rt = res.round_trips[(res.round_trips.end_time >= df.index[s0]) & (res.round_trips.end_time <= df.index[s1 - 1])]
        s, b = summarize(eq, tr, rt), summarize(bh)
        seg = segment_table(eq, bh)
        g = gate(s, b, seg, mode="risk")
        out["windows"][wname] = {"strategy": s, "buy_and_hold": b, "gate": g, "start": str(eq.index[0].date()), "end": str(eq.index[-1].date())}
    return out


def sensitivity(df: pd.DataFrame, mod, params: dict, windows, fee: float) -> dict:
    rows = []
    for k, v in params.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        for f in (0.75, 1.25):
            p2 = copy.deepcopy(params)
            p2[k] = max(1, int(round(v * f))) if isinstance(v, int) else v * f
            if p2[k] == v:
                continue
            try:
                r = run_once(df, mod, p2, windows, fee)
            except Exception as ex:  # noqa: BLE001
                rows.append({"param": k, "value": p2[k], "error": str(ex)})
                continue
            row = {"param": k, "value": p2[k]}
            for wname, w in r["windows"].items():
                row[f"{wname}_sharpe"] = w["strategy"]["sharpe"]
                row[f"{wname}_pass"] = w["gate"]["passed"]
            rows.append(row)
    df_s = pd.DataFrame(rows)
    summary = {"n": len(df_s)}
    for wname, *_ in windows:
        col = f"{wname}_sharpe"
        if col in df_s:
            vals = df_s[col].dropna()
            summary[wname] = {"sharpe_min": float(vals.min()) if len(vals) else None,
                              "sharpe_median": float(vals.median()) if len(vals) else None,
                              "sharpe_max": float(vals.max()) if len(vals) else None,
                              "pass_ratio": float(df_s[f"{wname}_pass"].mean()) if f"{wname}_pass" in df_s else None}
    return {"summary": summary, "rows": rows}


def fmt_windows(sym: str, r: dict) -> str:
    lines = [f"--- {sym} ---",
             f"{'區間':6}{'':10}{'淨報酬':>9}{'回撤':>8}{'CAGR':>8}{'Sharpe':>8}{'Calmar':>8}{'回合':>5}{'閘門':>5}"]
    for wname, w in r["windows"].items():
        s, b, g = w["strategy"], w["buy_and_hold"], w["gate"]
        lines.append(f"{wname:6}{'策略':10}{pct(s['net_return']):>9}{pct(s['max_dd']):>8}{pct(s['cagr']):>8}{num(s['sharpe']):>8}{num(s['calmar']):>8}{s.get('round_trips', 0):>5}{'✓' if g['passed'] else '✗':>5}")
        lines.append(f"{'':6}{'持有':10}{pct(b['net_return']):>9}{pct(b['max_dd']):>8}{pct(b['cagr']):>8}{num(b['sharpe']):>8}{num(b['calmar']):>8}{'':>5}{'':>5}")
    e = r["extra"]
    lines.append(f"平均曝險 {e['avg_exposure']:.2f} | 在場時間 {e['time_in_market']:.0%} | ATR 停損 {e['stops']} 次 | 回撤暫停 {e['dd_pauses']} 次")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--symbols", default="BTC/USDT:USDT,ETH/USDT:USDT")
    ap.add_argument("--tf", default="1d")
    ap.add_argument("--windows", default=DEFAULT_WINDOWS)
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--offline", action="store_true", default=True)
    ap.add_argument("--fetch", action="store_true", help="連網補資料（預設用快取）")
    ap.add_argument("--out", default="results/variants")
    args = ap.parse_args()

    mod = load_variant(args.variant)
    windows = parse_windows(args.windows)
    params = copy.deepcopy(mod.PARAMS)
    fee = 0.0005
    report = {"variant": args.variant, "name": getattr(mod, "NAME", args.variant), "description": getattr(mod, "DESCRIPTION", ""),
              "params": params, "symbols": {}}
    text = [f"=== 變體 {report['name']}：{report['description']}", f"參數 {json.dumps(params, ensure_ascii=False)}"]
    for sym in args.symbols.split(","):
        df = dt.fetch_ohlcv(sym, args.tf, DATA_SINCE, "data", verbose=False) if args.fetch else dt.load_cached("data", sym, args.tf, DATA_SINCE)
        if df is None:
            print(f"{sym} 沒有快取，加 --fetch")
            return 2
        r = run_once(df, mod, params, windows, fee)
        if args.sensitivity:
            r["sensitivity"] = sensitivity(df, mod, params, windows, fee)
        report["symbols"][sym] = r
        text.append(fmt_windows(sym, r))
        if args.sensitivity:
            ss = r["sensitivity"]["summary"]
            parts = [f"{w}: Sharpe {num(v['sharpe_min'])}~{num(v['sharpe_max'])}（中位 {num(v['sharpe_median'])}）通過率 {v['pass_ratio']:.0%}"
                     for w, v in ss.items() if w != "n" and v.get("sharpe_min") is not None]
            text.append(f"敏感度（{ss['n']} 組 ±25% 擾動）" + (" | " + " | ".join(parts) if parts else ""))
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
