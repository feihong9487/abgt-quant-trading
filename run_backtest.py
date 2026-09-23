"""ABGT 回測入口：抓資料 → 算信號 → 模擬 → 對買入持有 → 季度閘門 → 報告。

用法：
  python run_backtest.py --strategy bgt            # BGT-S2 多信號，ETH/USDT 永續 15m
  python run_backtest.py --strategy bgt --mode kf  # BGT-S1 只用 KF 主信號
  python run_backtest.py --strategy apex           # APEX v4.0 只做多，BTC/USDT 永續 1d
  加 --offline 用快取不連網；--since 改起算日（預設 2023-01-01）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abgt import data as dt                      # noqa: E402
from abgt.benchmark import buy_and_hold, gate, segment_table   # noqa: E402
from abgt.metrics import num, pct, summarize     # noqa: E402
from abgt.strategies import apex, atr_channel, bgt   # noqa: E402

STRATS = {
    "bgt": {"symbol": "ETH/USDT:USDT", "tf": "15m", "warmup_days": 30, "tick": 0.01},
    "apex": {"symbol": "BTC/USDT:USDT", "tf": "1d", "warmup_days": 420, "tick": 0.1},
    "channel": {"symbol": "BTC/USDT:USDT", "tf": "1d", "warmup_days": 120, "tick": 0.1},
}


def report(label: str, symbol: str, tf: str, s: dict, b: dict, seg: pd.DataFrame, g: dict, extra: dict) -> str:
    lines = []
    lines.append(f"=== ABGT 回測：{label} | {symbol} {tf} | {s['start']:%Y-%m-%d} → {s['end']:%Y-%m-%d}（{s['years']:.2f} 年）===")
    lines.append(f"{'':14}{'策略':>12}{'買入持有':>12}")
    lines.append(f"{'淨報酬':14}{pct(s['net_return']):>12}{pct(b['net_return']):>12}")
    lines.append(f"{'CAGR':14}{pct(s['cagr']):>12}{pct(b['cagr']):>12}")
    lines.append(f"{'最大回撤':14}{pct(s['max_dd']):>12}{pct(b['max_dd']):>12}")
    lines.append(f"{'Sharpe(日)':14}{num(s['sharpe']):>12}{num(b['sharpe']):>12}")
    lines.append(f"{'Calmar':14}{num(s['calmar']):>12}{num(b['calmar']):>12}")
    lines.append(f"{'出場筆數':14}{s.get('trades', 0):>12}{'—':>12}")
    lines.append(f"{'進出回合':14}{s.get('round_trips', 0):>12}{'—':>12}")
    lines.append(f"{'勝率(出場筆)':14}{pct(s.get('win_rate')):>12}{'—':>12}")
    lines.append(f"{'勝率(回合)':14}{pct(s.get('trip_win_rate')):>12}{'—':>12}")
    lines.append(f"{'獲利因子':14}{num(s.get('profit_factor')):>12}{'—':>12}")
    for k, v in extra.items():
        lines.append(f"{k:14}{str(v):>12}")
    lines.append("")
    lines.append("--- 季度閘門：策略需同時「報酬更高」且「回撤更小」---")
    lines.append(f"{'季度':8}{'策略報酬':>10}{'持有報酬':>10}{'策略回撤':>10}{'持有回撤':>10}{'通過':>6}")
    for _, r in seg.iterrows():
        lines.append(f"{r['segment']:8}{pct(r['strat_ret']):>10}{pct(r['bh_ret']):>10}{pct(r['strat_dd']):>10}{pct(r['bh_dd']):>10}{'✓' if r['passed'] else '✗':>6}")
    lines.append("")
    lines.append(f"全期報酬 {'✓' if g['overall_ret_pass'] else '✗'}  全期回撤 {'✓' if g['overall_dd_pass'] else '✗'}  "
                 f"Sharpe {'✓' if g['sharpe_pass'] else '✗'}  Calmar {'✓' if g['calmar_pass'] else '✗'}  "
                 f"季度 {g['segments_passed']}/{g['segments_total']}（參考）")
    lines.append(f"上線閘門（{'風險調整：Sharpe、Calmar 高於持有且回撤更小' if g['mode'] == 'risk' else '絕對：報酬更高、回撤更小、季度全過'}）："
                 f"{'通過' if g['passed'] else '未通過'}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=list(STRATS), required=True)
    ap.add_argument("--mode", choices=["multi", "kf"], default="multi", help="bgt 專用：multi=BGT-S2，kf=BGT-S1")
    ap.add_argument("--require-kf-dir", action="store_true", help="bgt 專用：聯動/百分比路徑也要求 45 分方向一致（簡報規則）")
    ap.add_argument("--since", default="2023-01-01")
    ap.add_argument("--symbol")
    ap.add_argument("--tf")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--min-pass-ratio", type=float, default=1.0)
    ap.add_argument("--gate", choices=["risk", "absolute"], default="risk")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = STRATS[args.strategy]
    symbol, tf = args.symbol or cfg["symbol"], args.tf or cfg["tf"]
    fetch_since = (pd.Timestamp(args.since) - pd.Timedelta(days=cfg["warmup_days"])).strftime("%Y-%m-%d")
    print(f"[1/4] 資料 {symbol} {tf} 自 {fetch_since}（含暖機）")
    if args.offline:
        df = dt.load_cached("data", symbol, tf, fetch_since)
        if df is None:
            print("沒有快取，拿掉 --offline 先抓一次")
            return 2
    else:
        df = dt.fetch_ohlcv(symbol, tf, fetch_since, "data")

    print("[2/4] 計算信號")
    if args.strategy == "bgt":
        p = bgt.BGTParams(entry_mode=args.mode, require_kf_dir_all=args.require_kf_dir)
        ep = bgt.BGTEngineParams(tick_size=cfg["tick"])
        sig = bgt.compute_signals(df, p)
        fv = bgt.first_valid_index(sig)
        label = "BGT-S1 KF單信號" if args.mode == "kf" else "BGT-S2 多信號"
        name = f"bgt_{args.mode}"
        if args.require_kf_dir:
            label += "+45分方向閘"
            name += "_kfdir"
    elif args.strategy == "channel":
        built = atr_channel.build(df)
        sig = df
        fv = built["warmup"]
        ep = built["risk"]
        label = "ATR 通道遲滯（第 1 輪冠軍）"
        name = "channel"
    else:
        p = apex.APEXParams()
        ep = apex.APEXEngineParams(tick_size=cfg["tick"])
        sig = apex.compute_signals(df, p)
        fv = apex.first_valid_index(sig)
        label = "APEX v4.0 只做多"
        name = "apex"
    start_idx = max(fv, int(sig.index.searchsorted(pd.Timestamp(args.since, tz="UTC"))))
    if start_idx >= len(sig):
        print("資料不足，無法起算")
        return 2

    print(f"[3/4] 模擬（起算 {sig.index[start_idx]:%Y-%m-%d}，{len(sig) - start_idx} 根）")
    if args.strategy == "bgt":
        res = bgt.simulate(sig, ep, start_idx)
        w = sig.iloc[start_idx:]
        extra = {
            "KF主信號(多/空)": f"{int(w.long_kf.sum())}/{int(w.short_kf.sum())}",
            "聯動(多/空)": f"{int(w.linkage_long.sum())}/{int(w.linkage_short.sum())}",
            "百分比(多/空)": f"{int(w.prob_long.sum())}/{int(w.prob_short.sum())}",
            "45分翻轉次數": int(w.kf_flip.sum()),
        }
    elif args.strategy == "channel":
        res = atr_channel.run(df, start_idx)
        extra = {"平均曝險": f"{res.extra['avg_exposure']:.2f}", "在場時間": f"{res.extra['time_in_market']:.0%}",
                 "進/出": f"{res.extra['regime_entries']}/{res.extra['regime_exits']}"}
    else:
        res = apex.simulate(sig, ep, start_idx, p)
        kt = res.extra.get("kill_switch_time")
        extra = {
            "信號 A/B": f"{res.extra['signals']['A']}/{res.extra['signals']['B']}",
            "熔斷觸發": f"{kt:%Y-%m-%d}" if kt is not None else "無",
        }

    print("[4/4] 基準與閘門")
    bh = buy_and_hold(sig["close"].iloc[start_idx:], ep.initial_capital, ep.commission_pct)
    s_sum = summarize(res.equity, res.trades, res.round_trips)
    b_sum = summarize(bh)
    seg = segment_table(res.equity, bh)
    g = gate(s_sum, b_sum, seg, args.min_pass_ratio, mode=args.gate)

    text = report(label, symbol, tf, s_sum, b_sum, seg, g, extra)
    print()
    print(text)

    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame({"strategy": res.equity, "buy_and_hold": bh}).to_csv(os.path.join(args.out, f"{name}_equity.csv"))
    res.trades.to_csv(os.path.join(args.out, f"{name}_trades.csv"), index=False)
    res.round_trips.to_csv(os.path.join(args.out, f"{name}_round_trips.csv"), index=False)
    seg.to_csv(os.path.join(args.out, f"{name}_segments.csv"), index=False)
    with open(os.path.join(args.out, f"{name}_report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    summary = {"label": label, "symbol": symbol, "tf": tf, "strategy": {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in s_sum.items()},
               "buy_and_hold": {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in b_sum.items()},
               "gate": g, "extra": {k: str(v) for k, v in extra.items()}}
    with open(os.path.join(args.out, f"{name}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果存到 {args.out}/{name}_*.csv|json|txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
