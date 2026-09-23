"""EMA50 ± ATR 通道 + 20 日收盤新高快速進場（OR 分支）。

進場：close > EMA50 + k_in×ATR（原通道腿）**或**（close > EMA50 且 close > 前 hi_len 根收盤最高，快速腿）。
出場：close < EMA50 − k_out×ATR（與 vol_atr_channel 逐位相同）。通道內維持前一狀態，由引擎 state_machine 維護。

設計：通道的 OOS 淨報酬落後 ema50_plain（BTC +323% vs +387%）主要是進場延遲成本。崩盤後 ≥ hi_len 根的 V 型回升，
hi_len 日窗內只剩崩盤後的收盤，站回 EMA50 幾乎必定同時創 hi_len 日新高 → 快速腿觸發、少付 1 ATR；
盤整市 EMA50 位在區間中段而 hi_len 日高在區間頂，站回 EMA50 不創新高 → 快速腿沉默，保留通道濾鋸齒能力。
ema_len / atr_len / k_in / k_out 原封不動繼承 vol_atr_channel，任何 Δ 都可歸因到快速腿。

無前視：EMA / ATR 因果遞迴（TR 用 close[i−1]）；prev_hi = highest(close, hi_len) 再 shift(1)，只含 i−hi_len..i−1；
enter / exit 每根只讀 ≤ i 的 OHLC；引擎在 i+1 開盤成交。
enter 與 exit 不可能同根同時為真（enter 要 c > ema，exit 要 c < ema − k_out×atr，atr ≥ 0）。

診斷：環境變數 ABGT_DEBUG=1 時印出各視窗 entries 筆數與「只靠快速腿」的 entries 筆數與日期（不影響評估）。
"""
import os

import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams, state_machine

NAME = "EMA50 ± ATR 通道 + 20 日收盤新高快速進場"
DESCRIPTION = "close > EMA50 + 1.0×ATR14 或（close > EMA50 且 close > 前 20 日收盤最高）進入趨勢、close < EMA50 − 0.25×ATR14 離開；無風控"
PARAMS = {"ema_len": 50, "atr_len": 14, "k_in": 1.0, "k_out": 0.25, "hi_len": 20}

_DEBUG_WINDOWS = (("IS", "2020-01-01", "2022-12-31"), ("OOS", "2023-01-01", None), ("FULL", "2020-01-01", None),
                  ("IS2", "2020-07-01", "2023-12-31"), ("OOS2", "2024-01-01", None))


def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    n = len(c)
    ema = ta.ema(c, int(p["ema_len"]))                          # Pine 語意：前 n−1 根 NaN，SMA 起算後遞迴；第 i 根只用 ≤ i 的收盤
    atr = ta.atr(h, l, c, int(p["atr_len"]))                    # RMA(TR)，TR 用 close[i−1]
    prev_hi = ta.shift(ta.highest(c, int(p["hi_len"])), 1)      # max(c[i−hi_len .. i−1])，不含第 i 根；前 hi_len 根 NaN
    chan_enter = c > ema + float(p["k_in"]) * atr               # 原通道進場（NaN 比較 → False）
    fast_enter = (c > ema) & (c > prev_hi)                      # 快速腿：站上 EMA50 且收盤嚴格高於前 hi_len 根任一收盤
    enter = chan_enter | fast_enter
    exit_ = c < ema - float(p["k_out"]) * atr                   # 出場腿與 vol_atr_channel 逐位相同
    valid = ~np.isnan(ema) & ~np.isnan(atr)
    warmup = max(int(np.argmax(valid)) if valid.any() else n, int(p["hi_len"]))   # 預設 = 49，與通道相同視窗

    if os.environ.get("ABGT_DEBUG"):
        _debug(df, enter, exit_, fast_enter, chan_enter, warmup)

    return {"in_regime": {"enter": enter, "exit": exit_}, "risk": RiskParams(), "warmup": warmup}


def _debug(df, enter, exit_, fast_enter, chan_enter, warmup):
    """各視窗：entries 筆數、只靠快速腿觸發的 entries 筆數與日期。"""
    import pandas as pd

    on = state_machine(enter, exit_)
    entries = on & ~ta.shift_bool(on, 1)
    fast_only = entries & fast_enter & ~chan_enter
    idx = df.index
    print(f"[fastreentry] warmup={warmup} ({idx[warmup].date() if warmup < len(idx) else 'n/a'})")
    for wname, ws, we in _DEBUG_WINDOWS:
        s0 = max(warmup, int(idx.searchsorted(pd.Timestamp(ws, tz="UTC"))))
        s1 = len(idx) if we is None else int(idx.searchsorted(pd.Timestamp(we, tz="UTC"), side="right"))
        m = np.zeros(len(idx), dtype=bool)
        m[s0:s1] = True
        n_e = int((entries & m).sum())
        n_f = int((fast_only & m).sum())
        dates = [str(d.date()) for d in idx[fast_only & m]]
        print(f"[fastreentry] {wname:5} entries={n_e:3d} fast_only={n_f:3d} dates={dates}")
