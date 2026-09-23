"""對照實驗：趨勢內 100% 持倉、趨勢外空手（曝險型趨勢跟隨），看閘門在這個區間是否可達。

不是正式策略，用來回答「1% 風險式倉位追不上牛市，那換成曝險式呢？」
用法：python experiments/regime.py [--symbol BTC/USDT:USDT --tf 1d --since 2023-01-01]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abgt import data as dt                      # noqa: E402
from abgt import indicators as ta                # noqa: E402
from abgt.benchmark import buy_and_hold, gate, segment_table   # noqa: E402
from abgt.metrics import num, pct, summarize     # noqa: E402

FEE = 0.0005


def regime_equity(df: pd.DataFrame, in_trend: np.ndarray, start_idx: int, initial: float = 10_000.0) -> pd.Series:
    """收盤判定、下一根開盤成交、每邊 0.05% 手續費。"""
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    n = len(c)
    eq = np.full(n, np.nan)
    cash, qty = initial, 0.0
    for i in range(start_idx, n):
        want = bool(in_trend[i - 1]) if i > 0 else False
        if want and qty == 0:
            qty = cash * (1 - FEE) / o[i]
            cash = 0.0
        elif not want and qty > 0:
            cash = qty * o[i] * (1 - FEE)
            qty = 0.0
        eq[i] = cash + qty * c[i]
    return pd.Series(eq, index=df.index, name="regime").iloc[start_idx:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--tf", default="1d")
    ap.add_argument("--since", default="2023-01-01")
    ap.add_argument("--warmup-days", type=int, default=420)
    args = ap.parse_args()

    fetch_since = (pd.Timestamp(args.since) - pd.Timedelta(days=args.warmup_days)).strftime("%Y-%m-%d")
    df = dt.load_cached("data", args.symbol, args.tf, fetch_since)
    if df is None:
        df = dt.fetch_ohlcv(args.symbol, args.tf, fetch_since, "data")
    c, h, l = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
    ema200, ema50, ema21 = ta.ema(c, 200), ta.ema(c, 50), ta.ema(c, 21)
    _, _, adx = ta.dmi(h, l, c, 14, 14)
    bull3 = (c > ema200) & (ema21 > ema50) & (ema50 > ema200)
    variants = {
        "R1 close>EMA200": c > ema200,
        "R2 三線多頭排列": bull3,
        "R3 三線排列+ADX≥28": bull3 & (adx >= 28),
        "R4 close>EMA50": c > ema50,
    }
    start_idx = max(int(np.argmax(~np.isnan(ema200) & ~np.isnan(adx))), int(df.index.searchsorted(pd.Timestamp(args.since, tz="UTC"))))
    bh = buy_and_hold(df["close"].iloc[start_idx:], 10_000.0, FEE)
    b = summarize(bh)
    print(f"=== 曝險型趨勢跟隨對照 | {args.symbol} {args.tf} | {bh.index[0]:%Y-%m-%d} → {bh.index[-1]:%Y-%m-%d} ===")
    print(f"{'變體':22}{'淨報酬':>10}{'最大回撤':>10}{'CAGR':>9}{'Sharpe':>8}{'季度通過':>9}{'閘門':>5}")
    print(f"{'買入持有':22}{pct(b['net_return']):>10}{pct(b['max_dd']):>10}{pct(b['cagr']):>9}{num(b['sharpe']):>8}{'—':>9}{'—':>5}")
    for name, flag in variants.items():
        eq = regime_equity(df, flag, start_idx)
        s = summarize(eq)
        seg = segment_table(eq, bh)
        g = gate(s, b, seg)
        print(f"{name:22}{pct(s['net_return']):>10}{pct(s['max_dd']):>10}{pct(s['cagr']):>9}{num(s['sharpe']):>8}{g['segments_passed']:>4}/{g['segments_total']:<4}{'✓' if g['passed'] else '✗':>5}")


if __name__ == "__main__":
    main()
