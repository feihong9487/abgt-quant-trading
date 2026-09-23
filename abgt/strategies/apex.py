"""APEX v4.0 的 Python 版（對應 APEX_v4.0.pine，只做多）。

模擬器貼著 Pine strategy() 語意：
- process_orders_on_close=false：市價單（進場、熔斷平倉）在下一根開盤成交
- 進場同時掛兩腿 exit：TP1 腿 50%（limit TP1 / stop SL）、TP2 腿（limit TP2 / stop SL / 追蹤：TP1 啟動，offset 0.35×止損距離）
- 停損/追蹤/限價在 K 線內用 TradingView 的路徑假設判定：開盤離高點近就 O→H→L→C，否則 O→L→H→C
- 熔斷：權益從高點回撤 ≥ 20% → 平倉且不再進場（永久）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import indicators as ta
from ..broker import BacktestResult, Book


@dataclass
class APEXParams:
    ema_slow: int = 200
    ema_mid: int = 50
    ema_fast: int = 21
    adx_len: int = 14
    adx_trend: float = 28.0
    adx_strong: float = 40.0
    touch_bars: int = 4
    ema_tol: float = 0.003
    rsi_len: int = 14
    rsi_lo_long: float = 38.0
    rsi_hi_long: float = 65.0
    dc_len: int = 20
    vol_len: int = 20
    vol_mult: float = 1.5
    atr_len: int = 14
    sl_mult_a: float = 1.5
    sl_mult_b: float = 2.0
    tp1_mult: float = 2.5
    tp2_mult: float = 4.0
    tp1_pct: float = 50.0
    risk_pct: float = 1.0
    max_dd_pct: float = 20.0
    trail_ratio: float = 0.35
    need_bull_bar: bool = True
    vol_confirm_a: bool = False
    vol_confirm_b: bool = True


@dataclass
class APEXEngineParams:
    initial_capital: float = 10_000.0
    commission_pct: float = 0.0005
    slippage_ticks: int = 2
    tick_size: float = 0.1             # BTCUSDT 永續 0.1
    use_kill_switch: bool = True


def compute_signals(df: pd.DataFrame, p: APEXParams) -> pd.DataFrame:
    o, h, l, c, v = (df[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close", "volume"))
    n = len(c)
    ema200, ema50, ema21 = ta.ema(c, p.ema_slow), ta.ema(c, p.ema_mid), ta.ema(c, p.ema_fast)
    atr = ta.atr(h, l, c, p.atr_len)
    rsi = ta.rsi(c, p.rsi_len)
    _, _, adx = ta.dmi(h, l, c, p.adx_len, p.adx_len)
    dc_hi = ta.shift(ta.highest(h, p.dc_len), 1)
    high_vol = v >= ta.sma(v, p.vol_len) * p.vol_mult
    bull = (c > ema200) & (ema21 > ema50) & (ema50 > ema200)

    touched = l <= ema21 * (1 + p.ema_tol)
    since_touch = np.full(n, 999, dtype=int)
    cnt = 999
    for i in range(n):
        cnt = 0 if touched[i] else cnt + 1
        since_touch[i] = cnt
    valid_pb = since_touch <= p.touch_bars

    candle = (c > o) if p.need_bull_bar else np.ones(n, bool)
    mode_a = valid_pb & (c > ema21) & candle & (rsi >= p.rsi_lo_long) & (rsi <= p.rsi_hi_long) & bull & (adx >= p.adx_trend)
    if p.vol_confirm_a:
        mode_a &= high_vol
    mode_b = (c > dc_hi) & bull & (adx >= p.adx_strong) & candle
    if p.vol_confirm_b:
        mode_b &= high_vol

    out = df.copy()
    out["ema200"], out["ema50"], out["ema21"], out["atr"], out["rsi"], out["adx"] = ema200, ema50, ema21, atr, rsi, adx
    out["dc_hi"], out["bull"], out["mode_a"], out["mode_b"] = dc_hi, bull, mode_a, mode_b
    return out


def first_valid_index(sig: pd.DataFrame) -> int:
    ok = sig[["ema200", "atr", "rsi", "adx", "dc_hi"]].notna().all(axis=1).to_numpy()
    return int(np.argmax(ok)) if ok.any() else len(sig)


def _path(o: float, h: float, l: float, c: float) -> list[float]:
    return [o, h, l, c] if (h - o) < (o - l) else [o, l, h, c]


def simulate(sig: pd.DataFrame, ep: APEXEngineParams, start_idx: int, p: APEXParams | None = None) -> BacktestResult:
    p = p or APEXParams()
    idx = sig.index
    o, h, l, c = (sig[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    atr = sig["atr"].to_numpy()
    mode_a, mode_b = sig["mode_a"].to_numpy(), sig["mode_b"].to_numpy()
    n = len(c)
    book = Book(ep.initial_capital, ep.commission_pct)
    slip = ep.slippage_ticks * ep.tick_size
    eq = np.full(n, np.nan)

    pending_entry: dict | None = None
    pending_close = False
    legs: list[dict] = []
    entry_obj = None
    peak = ep.initial_capital
    kill = False
    kill_time = None
    n_signals = {"A": 0, "B": 0}

    for i in range(start_idx, n):
        t = idx[i]
        # 1) 上一根的市價單在本根開盤成交
        if pending_close and book.entries:
            book.close_all(o[i] - slip, i, t, "熔斷")
            legs = []
            entry_obj = None
        pending_close = False
        if pending_entry is not None:
            pe = pending_entry
            pending_entry = None
            entry_obj = book.open(1, pe["qty"], o[i] + slip, i, t, eid="Long")
            q1 = pe["qty"] * p.tp1_pct / 100.0
            legs = [
                {"name": "TP1", "qty": q1, "limit": pe["tp1"], "stop": pe["sl"], "trail": None},
                {"name": "TP2", "qty": pe["qty"] - q1, "limit": pe["tp2"], "stop": pe["sl"],
                 "trail": {"activate": pe["tp1"], "offset": pe["trail_offset"], "active": False, "stop": np.nan}},
            ]

        # 2) 兩腿 exit 在 K 線內依路徑假設判定
        if legs and entry_obj is not None:
            path = _path(o[i], h[i], l[i], c[i])
            for k, px in enumerate(path):
                for leg in list(legs):
                    eff_stop = leg["stop"]
                    tr = leg["trail"]
                    if tr is not None and tr["active"]:
                        eff_stop = max(eff_stop, tr["stop"])
                    if px <= eff_stop:
                        fill = (o[i] if k == 0 else eff_stop) - slip
                        reason = "追蹤止損" if (tr is not None and tr["active"] and tr["stop"] > leg["stop"]) else "止損"
                        book.close(entry_obj, leg["qty"], fill, i, t, reason)
                        legs.remove(leg)
                        continue
                    if px >= leg["limit"]:
                        fill = o[i] if k == 0 else leg["limit"]
                        book.close(entry_obj, leg["qty"], fill, i, t, leg["name"])
                        legs.remove(leg)
                        continue
                    if tr is not None:
                        if not tr["active"] and px >= tr["activate"]:
                            tr["active"] = True
                            tr["stop"] = px - tr["offset"]
                        elif tr["active"]:
                            tr["stop"] = max(tr["stop"], px - tr["offset"])
            if not book.entries:
                legs = []
                entry_obj = None

        # 3) 收盤評估：熔斷 → 信號 → 下一根開盤的市價單
        equity_now = book.equity(c[i])
        if ep.use_kill_switch:
            peak = max(peak, equity_now)
            dd_pct = (peak - equity_now) / peak * 100.0 if peak > 0 else 0.0
            kill = dd_pct >= p.max_dd_pct
            if kill and kill_time is None:
                kill_time = t
        pos = book.position()
        if kill and pos != 0:
            pending_close = True
        if pos == 0 and not kill and pending_entry is None:
            a_prev = mode_a[i - 1] if i > 0 else False
            b_prev = mode_b[i - 1] if i > 0 else False
            sig_a = bool(mode_a[i] and not a_prev)
            sig_b = bool(mode_b[i] and not b_prev and not mode_a[i])
            if sig_a or sig_b:
                sl_mult = p.sl_mult_a if sig_a else p.sl_mult_b
                stop_dist = atr[i] * sl_mult
                if stop_dist > 0 and not np.isnan(stop_dist):
                    qty = equity_now * (p.risk_pct / 100.0) / stop_dist
                    pending_entry = {
                        "qty": qty, "mode": "A" if sig_a else "B",
                        "sl": c[i] - stop_dist, "tp1": c[i] + atr[i] * p.tp1_mult, "tp2": c[i] + atr[i] * p.tp2_mult,
                        "trail_offset": round(stop_dist / ep.tick_size * p.trail_ratio) * ep.tick_size,
                    }
                    n_signals["A" if sig_a else "B"] += 1
        eq[i] = book.equity(c[i])

    equity = pd.Series(eq, index=idx, name="strategy").iloc[start_idx:]
    return BacktestResult(equity=equity, trades=book.trades_df(), round_trips=book.trips_df(), start_idx=start_idx,
                          extra={"kill_switch_time": kill_time, "signals": n_signals})
