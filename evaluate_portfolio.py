"""評估雙標的組合：每個標的跑同一個變體（或各自指定），共用權益，對兩種同權重買入持有基準。

用法：
    python evaluate_portfolio.py --variant ema50_plain --symbols BTC/USDT:USDT,ETH/USDT:USDT --weights 0.5,0.5
    python evaluate_portfolio.py --variants ema50_plain,vol_atr_channel --symbols BTC/USDT:USDT,ETH/USDT:USDT
    加 --stress 跑成本情境；--lag-stop 0.2 --lag-window 0 --lag-action hold 試組合層運行閘門
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abgt import data as dt                                   # noqa: E402
from abgt.benchmark import gate, segment_table                # noqa: E402
from abgt.metrics import num, pct, summarize                  # noqa: E402
from abgt.portfolio import align, buy_and_hold_portfolio, simulate_portfolio   # noqa: E402
from abgt.strategies.regime import RiskParams, simulate_regime   # noqa: E402
from evaluate_variant import DATA_SINCE, DEFAULT_WINDOWS, ALT_WINDOWS, STRESS, load_variant, parse_windows   # noqa: E402


def window_metrics(eq: pd.Series, benches: dict[str, pd.Series], df_idx: pd.DatetimeIndex, windows, warmup: int, initial: float) -> dict:
    out = {}
    for wname, wstart, wend in windows:
        s0 = max(warmup, int(df_idx.searchsorted(pd.Timestamp(wstart, tz="UTC"))))
        s1 = int(df_idx.searchsorted(pd.Timestamp(wend, tz="UTC"), side="right")) if wend else len(df_idx)
        if s1 - s0 < 30:
            continue
        e = eq.loc[df_idx[s0]:df_idx[s1 - 1]]
        e = e / e.iloc[0] * initial
        s = summarize(e)
        row = {"strategy": s, "bench": {}, "start": str(e.index[0].date()), "end": str(e.index[-1].date())}
        for bname, b in benches.items():
            bb = b.loc[df_idx[s0]:df_idx[s1 - 1]]
            bb = bb / bb.iloc[0] * initial
            bs = summarize(bb)
            ratio = (e / bb).to_numpy()
            row["bench"][bname] = {"summary": bs, "gate": gate(s, bs, segment_table(e, bb), mode="risk"),
                                   "rel_dd_vs_hold": float(np.nanmin(ratio / np.maximum.accumulate(ratio) - 1))}
        out[wname] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="ema50_plain")
    ap.add_argument("--variants", help="逗號分隔，與 symbols 一一對應")
    ap.add_argument("--symbols", default="BTC/USDT:USDT,ETH/USDT:USDT")
    ap.add_argument("--weights", default="0.5,0.5")
    ap.add_argument("--windows", default=DEFAULT_WINDOWS + "," + ALT_WINDOWS)
    ap.add_argument("--rebalance-threshold", type=float, default=0.15)
    ap.add_argument("--fee", type=float, default=0.0005, help="每邊手續費（0.0005 = 0.05%%）")
    ap.add_argument("--slip", type=float, default=0.0002)
    ap.add_argument("--max-exposure", type=float, default=1.0, help="趨勢內總曝險上限（>1 = 槓桿，會計借貸成本）")
    ap.add_argument("--borrow-rate", type=float, default=0.0, help="年化借貸成本，對曝險超過 1 的部分計息")
    ap.add_argument("--vol-target", type=float, default=0.0)
    ap.add_argument("--vol-lookback", type=int, default=20)
    ap.add_argument("--lag-stop", type=float, default=0.0)
    ap.add_argument("--lag-window", type=int, default=0)
    ap.add_argument("--lag-action", default="hold")
    ap.add_argument("--lag-resume", default="ratio_high")
    ap.add_argument("--lag-resume-bars", type=int, default=60)
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--out", default="results/portfolio")
    args = ap.parse_args()

    symbols = args.symbols.split(",")
    weights = dict(zip(symbols, [float(x) for x in args.weights.split(",")]))
    variant_ids = args.variants.split(",") if args.variants else [args.variant] * len(symbols)
    dfs = {s: dt.load_cached("data", s, "1d", DATA_SINCE) for s in symbols}
    if any(v is None for v in dfs.values()):
        print("缺快取，先跑 evaluate_variant --fetch")
        return 2
    idx = align(dfs)
    regimes, warmups, singles = {}, [], {}
    for s, vid in zip(symbols, variant_ids):
        mod = load_variant(vid)
        built = mod.build(dfs[s], copy.deepcopy(mod.PARAMS))
        regimes[s] = built["in_regime"]
        wu_time = dfs[s].index[int(built["warmup"])]
        warmups.append(int(idx.searchsorted(wu_time)))
        r = simulate_regime(dfs[s], built["in_regime"], built["risk"], int(built["warmup"]))
        singles[s] = r.equity
    warmup = max(warmups)
    rp = RiskParams(rebalance_threshold=args.rebalance_threshold, lag_stop=args.lag_stop, lag_window=args.lag_window,
                    lag_action=args.lag_action, lag_resume=args.lag_resume, lag_resume_bars=args.lag_resume_bars,
                    commission_pct=args.fee, slippage_pct=args.slip, max_exposure=args.max_exposure, borrow_rate_annual=args.borrow_rate,
                    vol_target=args.vol_target, vol_lookback=args.vol_lookback)
    windows = parse_windows(args.windows)

    def run(rp_: RiskParams, fundings=None):
        res = simulate_portfolio(dfs, regimes, rp_, weights, warmup, fundings)
        benches = {"持有不再平衡": buy_and_hold_portfolio(dfs, weights, idx, warmup, rp_.initial_capital, rp_.commission_pct, None),
                   "持有月再平衡": buy_and_hold_portfolio(dfs, weights, idx, warmup, rp_.initial_capital, rp_.commission_pct, "M")}
        return res, window_metrics(res.equity, benches, idx, windows, warmup, rp_.initial_capital)

    res, wm = run(rp)
    # 單標的平均（同視窗）供對照
    single_avg = {}
    for wname, wstart, wend in windows:
        vals = []
        for s in symbols:
            e = singles[s]
            s0 = max(int(e.index.searchsorted(pd.Timestamp(wstart, tz="UTC"))), 0)
            s1 = int(e.index.searchsorted(pd.Timestamp(wend, tz="UTC"), side="right")) if wend else len(e)
            ee = e.iloc[s0:s1]
            if len(ee) > 30:
                vals.append(summarize(ee)["sharpe"])
        single_avg[wname] = float(np.mean(vals)) if vals else None

    label = " + ".join(f"{s.split('/')[0]}×{vid}" for s, vid in zip(symbols, variant_ids))
    lines = [f"=== 組合 {label} | 權重 {args.weights} | 起算 {res.equity.index[0].date()} | 調倉門檻 {args.rebalance_threshold} | 費 {args.fee:.3%}/邊 滑 {args.slip:.2%}" + (f" | 曝險上限 {args.max_exposure}× 借貸 {args.borrow_rate:.1%}/年" if args.max_exposure != 1.0 else "") + (f" | 波動目標 {args.vol_target}" if args.vol_target else "") + (f" | 運行閘門 lag_stop={args.lag_stop} window={args.lag_window} action={args.lag_action} resume={args.lag_resume}/{args.lag_resume_bars}" if args.lag_stop else ""),
             f"{'區間':6}{'':12}{'淨報酬':>9}{'回撤':>8}{'Sharpe':>8}{'Calmar':>8}{'相對落後':>9}{'閘門':>5}{'單標的平均Sharpe':>16}"]
    for wname, w in wm.items():
        s = w["strategy"]
        lines.append(f"{wname:6}{'組合':12}{pct(s['net_return']):>9}{pct(s['max_dd']):>8}{num(s['sharpe']):>8}{num(s['calmar']):>8}{'':>9}{'':>5}{num(single_avg.get(wname)):>16}")
        for bname, b in w["bench"].items():
            bs = b["summary"]
            lines.append(f"{'':6}{bname:12}{pct(bs['net_return']):>9}{pct(bs['max_dd']):>8}{num(bs['sharpe']):>8}{num(bs['calmar']):>8}{pct(b['rel_dd_vs_hold']):>9}{'✓' if b['gate']['passed'] else '✗':>5}")
    e = res.extra
    lines.append(f"平均曝險 {e['avg_exposure']:.2f}（{', '.join(f'{k.split('/')[0]} {v:.2f}' for k, v in e['avg_exposure_by_asset'].items())}，最高 {e.get('max_exposure_seen', 0):.2f}）| 在場 {e['time_in_market']:.0%} | 成交 {e['n_fills']}（名目換手 {e['notional_turnover_per_year']:.1f}×/年）| 回合 {len(res.round_trips)} | 落後閘門觸發 {e['lag_triggers']} | 相對持有最深落後 {pct(e['rel_dd_vs_hold'])}" + (f" | 借貸成本 {e['borrow_paid']:,.0f} | 最低權益 {e.get('min_equity', 0):,.0f}" if e.get("borrow_paid") else ""))
    if e["lag_events"]:
        lines.append("觸發：" + "; ".join(f"{ev['time']} lag {ev['lag'] * 100:.1f}% → {ev['action']}，{('恢復於 +' + str(ev['resume_bar'] - ev['trigger_bar']) + ' 根') if ev['resume_bar'] is not None else '未恢復'}" for ev in e["lag_events"]))
    report = {"label": label, "symbols": symbols, "variants": variant_ids, "weights": weights, "risk": dataclasses.asdict(rp) | {"funding": None},
              "windows": wm, "single_avg_sharpe": single_avg, "extra": {k: v for k, v in e.items() if k not in ("bench_equity",)}}
    if args.stress:
        report["stress"] = {}
        configs = dict(STRESS)
        funds = {s: dt.load_funding_cached("data", s) for s in symbols}
        for name, ov in configs.items():
            rp2 = dataclasses.replace(rp, **{k: v for k, v in ov.items() if k != "funding"})
            r2, wm2 = run(rp2)
            report["stress"][name] = {w: {b: {"sharpe": x["bench"][b]["summary"]["sharpe"], "pass": x["bench"][b]["gate"]["passed"]} for b in x["bench"]} | {"_sharpe": x["strategy"]["sharpe"]} for w, x in wm2.items()}
            lines.append(f"壓力 {name:14} " + "  ".join(f"{w}:{num(x['strategy']['sharpe'])}{'✓' if x['bench']['持有不再平衡']['gate']['passed'] else '✗'}" for w, x in wm2.items()))
        if all(f is not None for f in funds.values()):
            fundings = {s: funds[s].reindex(dfs[s].index).fillna(0.0).to_numpy(dtype="float64") for s in symbols}
            r3, wm3 = run(rp, fundings)
            report["stress"]["歷史資金費率"] = {w: {"_sharpe": x["strategy"]["sharpe"], "pass": x["bench"]["持有不再平衡"]["gate"]["passed"]} for w, x in wm3.items()}
            lines.append(f"壓力 {'歷史資金費率':14} " + "  ".join(f"{w}:{num(x['strategy']['sharpe'])}{'✓' if x['bench']['持有不再平衡']['gate']['passed'] else '✗'}" for w, x in wm3.items()) + f"  （資金費率 {r3.extra['funding_paid']:.0f}）")
    text = "\n".join(lines)
    print(text)
    os.makedirs(args.out, exist_ok=True)
    tag = "_".join(s.split("/")[0] for s in symbols) + "_" + "_".join(dict.fromkeys(variant_ids)) + (f"_lag{args.lag_stop}" if args.lag_stop else "")
    with open(os.path.join(args.out, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(args.out, f"{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n→ {args.out}/{tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
