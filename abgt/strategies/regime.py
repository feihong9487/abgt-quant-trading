"""曝險型趨勢跟隨引擎：趨勢內持倉、趨勢外空手，風控模型可組合。

策略提供趨勢判定（每根收盤判定，不可用未來資料），兩種形式：
- np.ndarray[bool] in_regime：直接給狀態
- dict(enter=np.ndarray[bool], exit=np.ndarray[bool])：引擎維護狀態機 —— 趨勢外且 enter → 進、趨勢內且 exit → 出、其餘維持

風控由 RiskParams 決定：
- max_exposure：趨勢內的目標曝險（1.0 = 100% 權益）
- vol_target / vol_target_mode / vol_side / vol_lookback / vol_ref_lookback：波動目標倉位
    absolute：曝險 = min(max_exposure, vol_target / rvol)
    relative：曝險 = min(max_exposure, vol_target × ref / rvol)，ref = rvol 過去 vol_ref_lookback 根的中位數（尺度無關）
    vol_side total = 全部報酬的標準差；downside = 只算負報酬的半變異
- atr_stop_mult / stop_trigger：吊燈式追蹤停損（進場後最高收盤 − mult × ATR）
    stop_trigger "low" = K 線內觸價成交；"close" = 收盤跌破才觸發，下一根開盤成交
- reentry：停損後何時准重進 —— "reset" 等趨勢先轉空再轉多；"immediate" 立刻；"close_above_stop" 收盤高於停損出場價才准
- reentry_min_bars：停損後至少隔 N 根才准重進
- dd_pause / dd_pause_resume：權益從高水位回撤 ≥ X 時空手；恢復規則 "regime_reset"（趨勢重置）/ "new_high"（收盤創 N 日新高）/ "bars"（隔 N 根）
- funding_rate_8h：永續資金費率（每根日線扣 曝險 × 費率 × 3）
- exec_delay_bars：收盤決策後延遲幾根才在開盤成交（0 = 下一根開盤）
- rebalance_threshold：曝險差距超過此值才調倉

執行語意：收盤判定、下一根（+ 延遲）開盤成交（± 比例滑點）、每邊手續費；只做多。
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
    vol_target_mode: str = "absolute"     # absolute | relative
    vol_side: str = "total"               # total | downside
    vol_lookback: int = 20
    vol_ref_lookback: int = 365
    atr_len: int = 14
    atr_stop_mult: float = 0.0
    stop_trigger: str = "low"             # low | close
    reentry: str = "reset"                # reset | immediate | close_above_stop
    reentry_min_bars: int = 0
    dd_pause: float = 0.0
    dd_pause_resume: str = "regime_reset"  # regime_reset | new_high | bars
    dd_resume_high_n: int = 20
    dd_resume_bars: int = 20
    funding_rate_8h: float = 0.0
    exec_delay_bars: int = 0
    rebalance_threshold: float = 0.10
    initial_capital: float = 10_000.0
    commission_pct: float = 0.0005
    slippage_pct: float = 0.0002          # 日線用比例滑點（0.02%）


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
    return state_machine(c > hi, c < lo)


def regime_atr_channel(df: pd.DataFrame, ema_len: int = 50, atr_len: int = 14, k_in: float = 1.0, k_out: float = 0.25):
    """第 1 輪冠軍：EMA ± ATR 不對稱通道遲滯。回傳 (enter, exit, warmup)。"""
    c, h, l = (df[k].to_numpy(dtype="float64") for k in ("close", "high", "low"))
    ema = ta.ema(c, ema_len)
    atr = ta.atr(h, l, c, atr_len)
    enter = c > ema + k_in * atr
    exit_ = c < ema - k_out * atr
    valid = ~np.isnan(ema) & ~np.isnan(atr)
    warmup = int(np.argmax(valid)) if valid.any() else len(c)
    return enter, exit_, warmup


def adx_filter(df: pd.DataFrame, n: int, min_adx: float) -> np.ndarray:
    _, _, adx = ta.dmi(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), n, n)
    return adx >= min_adx


def state_machine(enter: np.ndarray, exit_: np.ndarray) -> np.ndarray:
    """趨勢外且 enter → True；趨勢內且 exit → False；其餘維持。NaN 比較已是 False，暖機期自然為 False。"""
    enter = np.asarray(enter, dtype=bool)
    exit_ = np.asarray(exit_, dtype=bool)
    out = np.zeros(len(enter), dtype=bool)
    state = False
    for i in range(len(enter)):
        if not state and enter[i]:
            state = True
        elif state and exit_[i]:
            state = False
        out[i] = state
    return out


def _realized_vol(c: np.ndarray, rp: RiskParams, bars_per_year: float) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (rvol, ref)。ref 只在 relative 模式有值。"""
    r = np.log(c / ta.shift(c, 1))
    if rp.vol_side == "downside":
        neg = np.where(r < 0, r, 0.0)
        neg[np.isnan(r)] = np.nan
        rvol = np.sqrt(pd.Series(neg ** 2).rolling(rp.vol_lookback).mean().to_numpy()) * np.sqrt(bars_per_year)
    else:
        rvol = pd.Series(r).rolling(rp.vol_lookback).std().to_numpy() * np.sqrt(bars_per_year)
    ref = np.full(len(c), np.nan)
    if rp.vol_target_mode == "relative":
        ref = pd.Series(rvol).rolling(rp.vol_ref_lookback, min_periods=max(rp.vol_lookback * 3, 30)).median().to_numpy()
    return rvol, ref


# ---------- 引擎 ----------

def simulate_regime(df: pd.DataFrame, in_regime, rp: RiskParams, start_idx: int) -> BacktestResult:
    idx = df.index
    o, h, l, c = (df[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    n = len(c)
    if isinstance(in_regime, dict):
        on_arr = state_machine(in_regime["enter"], in_regime["exit"] if in_regime.get("exit") is not None else ~np.asarray(in_regime["enter"], dtype=bool))
    else:
        on_arr = np.asarray(in_regime, dtype=bool)
    atr = ta.atr(h, l, c, rp.atr_len)
    bars_per_year = 365.0 * 86400.0 / pd.Series(idx).diff().dropna().median().total_seconds()
    rvol, vol_ref = _realized_vol(c, rp, bars_per_year)
    hi_n = ta.highest(c, rp.dd_resume_high_n)

    book = Book(rp.initial_capital, rp.commission_pct)
    eq = np.full(n, np.nan)
    exposure = np.zeros(n)
    pending: list[tuple[int, float, str]] = []      # (成交 bar, 目標曝險, 原因)
    stop = np.nan
    hi_close = np.nan
    mode = "armed"                                  # armed | wait_reset | wait_price
    stop_exit_px = np.nan
    stop_bar = -10 ** 9
    paused = False
    pause_bar = -10 ** 9
    peak = rp.initial_capital
    n_stop = n_pause = n_entries = n_exits = 0
    funding_paid = 0.0
    prev_on = False

    def fill_bar(i: int) -> int:
        return i + 1 + rp.exec_delay_bars

    for i in range(start_idx, n):
        t = idx[i]
        # 1) 到期的目標曝險在本根開盤成交（同一根多筆取最後一筆）
        due = [p for p in pending if p[0] <= i]
        if due:
            pending = [p for p in pending if p[0] > i]
            _, target, reason = due[-1]
            eq_open = book.equity(o[i])
            pos = book.position()
            target_qty = target * eq_open / o[i]
            diff = target_qty - pos
            if abs(diff) * o[i] < 1e-6 * eq_open:
                diff = 0.0
            if diff > 1e-12:
                book.open(1, diff, o[i] * (1 + rp.slippage_pct), i, t)
                if pos == 0:
                    hi_close = np.nan
                    stop = np.nan
                    n_entries += 1
            elif diff < -1e-12:
                remain = -diff
                for e in list(book.entries):
                    q = min(e.qty, remain)
                    book.close(e, q, o[i] * (1 - rp.slippage_pct), i, t, reason)
                    remain -= q
                    if remain <= 1e-12:
                        break
                if book.position() <= 1e-12:
                    n_exits += 1
        # 2) 觸價型吊燈停損：前一根收盤算好的 stop，本根觸價成交
        if rp.atr_stop_mult > 0 and rp.stop_trigger == "low" and book.entries and not np.isnan(stop) and l[i] <= stop:
            px = min(o[i], stop) * (1 - rp.slippage_pct)
            book.close_all(px, i, t, "ATR停損")
            n_stop += 1
            n_exits += 1
            stop_exit_px = px
            stop_bar = i
            stop = np.nan
            mode = {"reset": "wait_reset", "immediate": "armed", "close_above_stop": "wait_price"}[rp.reentry]
            pending = [p for p in pending if p[1] == 0.0]   # 已被停損，取消還沒成交的加碼
        # 3) 收盤：追蹤停損、資金費率、回撤、目標曝險
        pos = book.position()
        eqv = book.equity(c[i])
        if pos > 0 and rp.funding_rate_8h > 0:
            cost = pos * c[i] * rp.funding_rate_8h * 3
            book.realized -= cost
            funding_paid += cost
            eqv = book.equity(c[i])
        stop_hit_close = False
        if pos > 0 and rp.atr_stop_mult > 0 and not np.isnan(atr[i]):
            hi_close = c[i] if np.isnan(hi_close) else max(hi_close, c[i])
            new_stop = hi_close - rp.atr_stop_mult * atr[i]
            stop = new_stop if np.isnan(stop) else max(stop, new_stop)
            if rp.stop_trigger == "close" and c[i] < stop:
                stop_hit_close = True
        peak = max(peak, eqv)
        dd = 1 - eqv / peak if peak > 0 else 0.0
        on = bool(on_arr[i])
        if not on:
            mode = "armed"
            stop_exit_px = np.nan
            paused = False
            peak = eqv        # 趨勢重置：回撤高水位重新起算
        elif paused:
            if rp.dd_pause_resume == "new_high" and not np.isnan(hi_n[i]) and c[i] >= hi_n[i]:
                paused, peak = False, eqv
            elif rp.dd_pause_resume == "bars" and i - pause_bar >= rp.dd_resume_bars:
                paused, peak = False, eqv
        if rp.dd_pause > 0 and dd >= rp.dd_pause and not paused and pos > 0:
            paused = True
            pause_bar = i
            n_pause += 1
        # 重進門檻
        if mode == "wait_price" and on and not np.isnan(stop_exit_px) and c[i] > stop_exit_px:
            mode = "armed"
        can_enter = mode == "armed" and (i - stop_bar) >= rp.reentry_min_bars
        want = on and not paused and (pos > 0 or can_enter)
        if stop_hit_close:
            want = False
        if not want:
            target = 0.0
        elif rp.vol_target > 0 and not np.isnan(rvol[i]) and rvol[i] > 0:
            if rp.vol_target_mode == "relative":
                target = rp.max_exposure if np.isnan(vol_ref[i]) else min(rp.max_exposure, rp.vol_target * vol_ref[i] / rvol[i])
            else:
                target = min(rp.max_exposure, rp.vol_target / rvol[i])
        else:
            target = rp.max_exposure
        current = pos * c[i] / eqv if eqv > 0 else 0.0
        # 已排隊但還沒成交的目標視為「即將達成」，避免延遲成交時重複排同一個目標
        ref = pending[-1][1] if pending else current
        if stop_hit_close and pos > 0:
            pending = [p for p in pending if p[1] == 0.0]
            pending.append((fill_bar(i), 0.0, "ATR停損"))
            n_stop += 1
            stop_exit_px = c[i]
            stop_bar = i
            stop = np.nan
            mode = {"reset": "wait_reset", "immediate": "armed", "close_above_stop": "wait_price"}[rp.reentry]
        elif target == 0 and ref > 0:
            pending.append((fill_bar(i), 0.0, "回撤暫停" if paused else "趨勢結束"))
        elif target > 0 and ref == 0:
            pending.append((fill_bar(i), target, "進場"))
        elif target > 0 and ref > 0 and abs(target - ref) > rp.rebalance_threshold:
            pending.append((fill_bar(i), target, "調整曝險"))
        eq[i] = eqv
        exposure[i] = current
        prev_on = on

    equity = pd.Series(eq, index=idx, name="strategy").iloc[start_idx:]
    return BacktestResult(equity=equity, trades=book.trades_df(), round_trips=book.trips_df(), start_idx=start_idx,
                          extra={"stops": n_stop, "dd_pauses": n_pause, "regime_entries": n_entries, "regime_exits": n_exits,
                                 "funding_paid": funding_paid,
                                 "avg_exposure": float(np.nanmean(exposure[start_idx:])),
                                 "time_in_market": float(np.mean(exposure[start_idx:] > 0))})
