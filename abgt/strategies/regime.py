"""曝險型趨勢跟隨引擎：趨勢內持倉、趨勢外空手，風控模型可組合。

策略只需要提供一個布林陣列 in_regime（每根收盤判定，不可用未來資料）；風控由 RiskParams 決定：
- max_exposure：趨勢內的目標曝險（1.0 = 100% 權益）
- vol_target：年化波動目標，>0 時曝險 = min(max_exposure, vol_target / 已實現波動)
- atr_stop_mult：吊燈式追蹤停損（進場後最高收盤 − mult × ATR），K 線內觸價成交；0 = 關
- reentry：停損後 "reset" = 等趨勢先轉空再轉多才重進；"immediate" = 下一根就能重進
- dd_pause：權益從高水位回撤 ≥ X 時空手，等趨勢重置才恢復，高水位也在趨勢重置時重新起算；0 = 關
- rebalance_threshold：曝險差距超過此值才調倉（限制波動目標的換手）

執行語意：收盤判定、下一根開盤成交（± 滑點）、每邊手續費；只做多。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import indicators as ta
from ..broker import BacktestResult, Book


@dataclass
class RiskParams:
    max_exposure: float = 1.0
    vol_target: float = 0.0
    vol_lookback: int = 20
    atr_len: int = 14
    atr_stop_mult: float = 0.0
    reentry: str = "reset"
    dd_pause: float = 0.0
    rebalance_threshold: float = 0.10
    initial_capital: float = 10_000.0
    commission_pct: float = 0.0005
    slippage_pct: float = 0.0002     # 日線用比例滑點（0.02%），比 tick 更通用


# ---------- 幾個內建的 regime 判定，變體可以直接用或自己算 ----------

def regime_close_over_ema(df: pd.DataFrame, n: int) -> np.ndarray:
    c = df["close"].to_numpy()
    return c > ta.ema(c, n)


def regime_ema_cross(df: pd.DataFrame, fast: int, slow: int) -> np.ndarray:
    c = df["close"].to_numpy()
    return ta.ema(c, fast) > ta.ema(c, slow)


def regime_apex3(df: pd.DataFrame, fast: int = 21, mid: int = 50, slow: int = 200) -> np.ndarray:
    c = df["close"].to_numpy()
    e1, e2, e3 = ta.ema(c, fast), ta.ema(c, mid), ta.ema(c, slow)
    return (c > e3) & (e1 > e2) & (e2 > e3)


def regime_donchian(df: pd.DataFrame, entry_len: int, exit_len: int) -> np.ndarray:
    """突破 entry_len 日高進、跌破 exit_len 日低出的狀態機（海龜式）。"""
    c = df["close"].to_numpy()
    hi = ta.shift(ta.highest(c, entry_len), 1)
    lo = ta.shift(ta.lowest(c, exit_len), 1)
    out = np.zeros(len(c), dtype=bool)
    on = False
    for i in range(len(c)):
        if not on and c[i] > hi[i]:
            on = True
        elif on and c[i] < lo[i]:
            on = False
        out[i] = on
    return out


def adx_filter(df: pd.DataFrame, n: int, min_adx: float) -> np.ndarray:
    _, _, adx = ta.dmi(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), n, n)
    return adx >= min_adx


# ---------- 引擎 ----------

def simulate_regime(df: pd.DataFrame, in_regime: np.ndarray, rp: RiskParams, start_idx: int) -> BacktestResult:
    idx = df.index
    o, h, l, c = (df[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    n = len(c)
    in_regime = np.asarray(in_regime, dtype=bool)
    atr = ta.atr(h, l, c, rp.atr_len)
    logret = np.log(c / ta.shift(c, 1))
    bars_per_year = 365.0 * 86400.0 / pd.Series(idx).diff().dropna().median().total_seconds()
    rvol = pd.Series(logret).rolling(rp.vol_lookback).std().to_numpy() * np.sqrt(bars_per_year)

    book = Book(rp.initial_capital, rp.commission_pct)
    eq = np.full(n, np.nan)
    exposure = np.zeros(n)
    pending: tuple[float, str] | None = None
    stop = np.nan
    hi_close = np.nan
    armed = True
    paused = False
    peak = rp.initial_capital
    n_stop = n_pause = 0

    for i in range(start_idx, n):
        t = idx[i]
        # 1) 前一根收盤決定的目標曝險，本根開盤成交
        if pending is not None:
            target, reason = pending
            pending = None
            eq_open = book.equity(o[i])
            pos = book.position()
            target_qty = target * eq_open / o[i]
            diff = target_qty - pos
            if diff > 1e-12:
                book.open(1, diff, o[i] * (1 + rp.slippage_pct), i, t)
                if pos == 0:
                    hi_close = np.nan
                    stop = np.nan
            elif diff < -1e-12:
                remain = -diff
                for e in list(book.entries):
                    q = min(e.qty, remain)
                    book.close(e, q, o[i] * (1 - rp.slippage_pct), i, t, reason)
                    remain -= q
                    if remain <= 1e-12:
                        break
        # 2) 吊燈停損：前一根收盤算好的 stop，本根觸價成交
        if rp.atr_stop_mult > 0 and book.entries and not np.isnan(stop) and l[i] <= stop:
            book.close_all(min(o[i], stop) * (1 - rp.slippage_pct), i, t, "ATR停損")
            n_stop += 1
            stop = np.nan
            if rp.reentry == "reset":
                armed = False
        # 3) 收盤：更新追蹤停損、回撤、目標曝險
        pos = book.position()
        eqv = book.equity(c[i])
        if pos > 0 and rp.atr_stop_mult > 0 and not np.isnan(atr[i]):
            hi_close = c[i] if np.isnan(hi_close) else max(hi_close, c[i])
            new_stop = hi_close - rp.atr_stop_mult * atr[i]
            stop = new_stop if np.isnan(stop) else max(stop, new_stop)
        peak = max(peak, eqv)
        dd = 1 - eqv / peak if peak > 0 else 0.0
        on = bool(in_regime[i])
        if not on:
            armed = True
            paused = False
            peak = eqv        # 趨勢重置：回撤暫停的高水位從這裡重新起算，否則暫停後永遠回不到舊高點、一重進就再觸發
        if rp.dd_pause > 0 and dd >= rp.dd_pause and not paused and pos > 0:
            paused = True
            n_pause += 1
        want = on and armed and not paused
        if not want:
            target = 0.0
        elif rp.vol_target > 0 and not np.isnan(rvol[i]) and rvol[i] > 0:
            target = min(rp.max_exposure, rp.vol_target / rvol[i])
        else:
            target = rp.max_exposure
        current = pos * c[i] / eqv if eqv > 0 else 0.0
        if (target == 0 and pos > 0):
            pending = (0.0, "回撤暫停" if paused else "趨勢結束")
        elif target > 0 and pos == 0:
            pending = (target, "進場")
        elif target > 0 and abs(target - current) > rp.rebalance_threshold:
            pending = (target, "調整曝險")
        eq[i] = eqv
        exposure[i] = current

    equity = pd.Series(eq, index=idx, name="strategy").iloc[start_idx:]
    return BacktestResult(equity=equity, trades=book.trades_df(), round_trips=book.trips_df(), start_idx=start_idx,
                          extra={"stops": n_stop, "dd_pauses": n_pause,
                                 "avg_exposure": float(np.nanmean(exposure[start_idx:])),
                                 "time_in_market": float(np.mean(exposure[start_idx:] > 0))})
