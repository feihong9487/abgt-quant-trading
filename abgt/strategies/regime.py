"""曝險型趨勢跟隨引擎：趨勢內持倉、趨勢外空手，風控模型可組合。

策略提供趨勢判定（每根收盤判定，不可用未來資料），三種形式：
- np.ndarray[bool] in_regime：直接給狀態
- dict(enter=np.ndarray[bool], exit=np.ndarray[bool])：引擎維護狀態機 —— 趨勢外且 enter → 進、趨勢內且 exit → 出、其餘維持
- dict 另可加 scale=np.ndarray[float]（每根 0–1 的目標曝險倍數，NaN 視為 1）：分級曝險，經 rebalance_threshold 才調倉

風控由 RiskParams 決定：
- max_exposure：趨勢內的目標曝險（1.0 = 100% 權益）
- scale_mode / scale_floor：分級曝險的處理 —— "track" 逐根跟隨 scale；"ratchet_up" 趨勢內取進場以來的累計最大值（只升不降，出場或趨勢重置歸零）；scale_floor 是趨勢內的曝險下限
- vol_target / vol_target_mode / vol_side / vol_lookback / vol_ref_lookback：波動目標倉位
    absolute：曝險 = min(max_exposure, vol_target / rvol)
    relative：曝險 = min(max_exposure, vol_target × ref / rvol)，ref = rvol 過去 vol_ref_lookback 根的中位數（尺度無關）
    vol_side total = 全部報酬的標準差；downside = 只算負報酬的半變異
- atr_stop_mult / stop_trigger：吊燈式追蹤停損（進場後最高收盤 − mult × ATR）
    stop_trigger "low" = K 線內觸價成交；"close" = 收盤跌破才觸發，下一根（+ 延遲）開盤成交
- reentry：停損後何時准重進 —— "reset" 等趨勢先轉空再轉多；"immediate" 立刻；"close_above_stop" 收盤高於停損成交價才准
- reentry_min_bars：停損成交後至少隔 N 根才准重進
- 出場單（停損或趨勢結束）排隊中一律不得排新進場
- dd_pause / dd_pause_resume：權益從高水位回撤 ≥ X 時空手；恢復規則 "regime_reset"（趨勢重置）/ "new_high"（收盤創 N 日新高）/ "bars"（隔 N 根）
- lag_stop / lag_window / lag_action / lag_resume / lag_resume_bars：運行閘門 —— 策略權益相對「同起點買入持有」落後 ≥ lag_stop 時觸發；
    lag_window 0 = 自上次恢復（或起算）累計、>0 = 以 N 根前為基期的滾動相對表現；lag_action "hold" = 忽略趨勢判定改 100% 持有、"flat" = 空手；
    lag_resume "regime_reset"（趨勢重置）/ "bars"（隔 N 根）/ "ratio_high"（策略/持有比值創 N 根新高）；恢復時累計基期重設
- funding_rate_8h：永續資金費率常數（每根日線扣 曝險 × 費率 × 3）；funding：逐日費率陣列（有值時覆蓋常數，負值為收入）
- exec_delay_bars：收盤決策後延遲幾根才在開盤成交（0 = 下一根開盤）
- rebalance_threshold：曝險差距超過此值才調倉

執行語意：收盤判定、下一根（+ 延遲）開盤成交（± 比例滑點）、每邊手續費；只做多。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import indicators as ta
from ..broker import BacktestResult, Book


@dataclass
class RiskParams:
    max_exposure: float = 1.0
    scale_mode: str = "track"             # track | ratchet_up
    scale_floor: float = 0.0
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
    lag_stop: float = 0.0
    lag_window: int = 0
    lag_action: str = "hold"              # hold | flat | record（只記錄，不改目標曝險）
    lag_base_mode: str = "reset"          # reset（起算根 / 上次恢復根）| hwm（對比值高水位量測，恢復不重設）| start_only（永遠對起算根）
    lag_resume: str = "regime_reset"      # regime_reset | bars | ratio_high
    lag_resume_bars: int = 60
    lag_retrigger_cooldown: int = 0       # 恢復後 N 根內不得再觸發
    funding_rate_8h: float = 0.0
    funding: np.ndarray | None = field(default=None, repr=False, compare=False)
    borrow_rate_annual: float = 0.0       # 槓桿借貸成本：每根對 max(0, 曝險 − 1) × 權益 計息（現貨保證金借 USDT 或永續資金費率之外的成本）
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
    """第 1 輪最佳變體：EMA ± ATR 不對稱通道遲滯。回傳 (enter, exit, warmup)。"""
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


def regime_flags(in_regime) -> tuple[np.ndarray, np.ndarray | None]:
    """把三種輸入形式統一成 (on 布林陣列, scale 陣列或 None)。"""
    if isinstance(in_regime, dict):
        if "enter" in in_regime:
            exit_ = in_regime.get("exit")
            enter = np.asarray(in_regime["enter"], dtype=bool)
            on = state_machine(enter, exit_ if exit_ is not None else ~enter)
        else:
            on = np.asarray(in_regime["in_regime"], dtype=bool)
        scale = in_regime.get("scale")
        return on, (np.asarray(scale, dtype="float64") if scale is not None else None)
    return np.asarray(in_regime, dtype=bool), None


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


_MODE_AFTER_STOP = {"reset": "wait_reset", "immediate": "armed", "close_above_stop": "wait_price"}


# ---------- 引擎 ----------

def simulate_regime(df: pd.DataFrame, in_regime, rp: RiskParams, start_idx: int) -> BacktestResult:
    idx = df.index
    o, h, l, c = (df[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    n = len(c)
    on_arr, scale_arr = regime_flags(in_regime)
    atr = ta.atr(h, l, c, rp.atr_len)
    bars_per_year = 365.0 * 86400.0 / pd.Series(idx).diff().dropna().median().total_seconds()
    rvol, vol_ref = _realized_vol(c, rp, bars_per_year)
    hi_n = ta.highest(c, rp.dd_resume_high_n)
    funding_arr = None
    if rp.funding is not None:
        funding_arr = np.asarray(rp.funding, dtype="float64")
        if len(funding_arr) != n:
            raise ValueError("funding 陣列長度必須等於 K 線數")

    book = Book(rp.initial_capital, rp.commission_pct)
    eq = np.full(n, np.nan)
    exposure = np.zeros(n)
    bench = np.full(n, np.nan)                       # 同起點買入持有權益
    ratio_hist = np.full(n, np.nan)                  # 策略 / 持有
    pending: list[tuple[int, float, str]] = []      # (成交 bar, 目標曝險, 原因)
    fills: list[tuple[int, int, str]] = []          # (bar, 方向, 原因)
    fills_by_reason: dict[str, int] = {}
    notional = 0.0
    stop = np.nan
    hi_close = np.nan
    mode = "armed"                                  # armed | wait_reset | wait_price | wait_fill
    stop_exit_px = np.nan
    stop_bar = -10 ** 9
    stop_events: list[dict] = []
    open_stop: dict | None = None
    regime_reset_since_stop = False
    paused = False
    pause_bar = -10 ** 9
    peak = rp.initial_capital
    scale_hi = 0.0
    scale_ups = scale_downs = 0
    lag_mode = False
    lag_bar = -10 ** 9
    lag_resume_at = -10 ** 9
    lag_base = 1.0
    ratio_hwm = 1.0
    lag_min_margin = np.inf
    lag_events: list[dict] = []
    n_stop = n_pause = n_entries = n_exits = 0
    funding_paid = 0.0
    borrow_paid = 0.0
    bh_qty = rp.initial_capital * (1 - rp.commission_pct) / c[start_idx]

    def fill_bar(i: int) -> int:
        return i + 1 + rp.exec_delay_bars

    def after_stop_fill(i: int, px: float, level: float, decision_bar: int) -> None:
        nonlocal mode, stop_exit_px, stop_bar, open_stop, regime_reset_since_stop
        mode = _MODE_AFTER_STOP[rp.reentry]
        stop_exit_px = px
        stop_bar = i
        regime_reset_since_stop = False
        if stop_events and stop_events[-1].get("fill_bar") is None:
            open_stop = stop_events[-1]          # 收盤觸發：決策時已記一筆，補上成交資訊
            open_stop.update({"fill_bar": i, "fill_px": px, "reentry_bar": None, "reentry_px": None, "path": None})
        else:
            open_stop = {"decision_bar": decision_bar, "fill_bar": i, "fill_px": px, "stop_level": level,
                         "reentry_bar": None, "reentry_px": None, "path": None}
            stop_events.append(open_stop)

    def record_fill(i: int, side: int, reason: str, qty: float, px: float) -> None:
        nonlocal notional
        fills.append((i, side, reason))
        fills_by_reason[reason] = fills_by_reason.get(reason, 0) + 1
        notional += abs(qty) * px

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
                px = o[i] * (1 + rp.slippage_pct)
                book.open(1, diff, px, i, t)
                record_fill(i, 1, reason, diff, px)
                if reason == "調整曝險":
                    scale_ups += 1
                if pos == 0:
                    hi_close = np.nan
                    stop = np.nan
                    n_entries += 1
                    if open_stop is not None and open_stop["reentry_bar"] is None:
                        open_stop["reentry_bar"] = i
                        open_stop["reentry_px"] = o[i]
                        open_stop["path"] = "regime_off" if regime_reset_since_stop else rp.reentry
                        open_stop = None
            elif diff < -1e-12:
                remain = -diff
                px = o[i] * (1 - rp.slippage_pct)
                for e in list(book.entries):
                    q = min(e.qty, remain)
                    book.close(e, q, px, i, t, reason)
                    remain -= q
                    if remain <= 1e-12:
                        break
                record_fill(i, -1, reason, -diff, px)
                if reason == "調整曝險":
                    scale_downs += 1
                if book.position() <= 1e-12:
                    n_exits += 1
                    scale_hi = 0.0
                    if reason == "ATR停損":
                        after_stop_fill(i, px, np.nan, i - 1 - rp.exec_delay_bars)
        # 2) 觸價型吊燈停損：前一根收盤算好的 stop，本根觸價成交
        if rp.atr_stop_mult > 0 and rp.stop_trigger == "low" and book.entries and not np.isnan(stop) and l[i] <= stop:
            px = min(o[i], stop) * (1 - rp.slippage_pct)
            level = stop
            qty_before = book.position()
            book.close_all(px, i, t, "ATR停損")
            record_fill(i, -1, "ATR停損", qty_before, px)
            n_stop += 1
            n_exits += 1
            stop = np.nan
            scale_hi = 0.0
            pending = [p for p in pending if p[1] == 0.0]   # 已被停損，取消還沒成交的加碼
            after_stop_fill(i, px, level, i)
        # 3) 收盤：資金費率、追蹤停損、回撤、運行閘門、目標曝險
        pos = book.position()
        eqv = book.equity(c[i])
        if pos > 0:
            rate = funding_arr[i] if funding_arr is not None else rp.funding_rate_8h * 3
            if rate == rate and rate != 0.0:
                cost = pos * c[i] * rate
                book.realized -= cost
                funding_paid += cost
                eqv = book.equity(c[i])
            if rp.borrow_rate_annual > 0 and eqv > 0:
                borrowed = max(0.0, pos * c[i] - eqv)
                if borrowed > 0:
                    cost = borrowed * rp.borrow_rate_annual / bars_per_year
                    book.realized -= cost
                    borrow_paid += cost
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
        bench[i] = bh_qty * c[i]
        ratio = eqv / bench[i] if bench[i] > 0 else 1.0
        ratio_hist[i] = ratio
        on = bool(on_arr[i])
        if not on:
            if mode != "wait_fill":
                mode = "armed"
            stop_exit_px = np.nan
            regime_reset_since_stop = True
            paused = False
            peak = eqv        # 趨勢重置：回撤高水位重新起算
            scale_hi = 0.0
            if lag_mode and rp.lag_resume == "regime_reset":
                lag_mode, lag_base = False, ratio
                lag_events[-1]["resume_bar"] = i
        elif paused:
            if rp.dd_pause_resume == "new_high" and not np.isnan(hi_n[i]) and c[i] >= hi_n[i]:
                paused, peak = False, eqv
            elif rp.dd_pause_resume == "bars" and i - pause_bar >= rp.dd_resume_bars:
                paused, peak = False, eqv
        if rp.dd_pause > 0 and dd >= rp.dd_pause and not paused and pos > 0:
            paused = True
            pause_bar = i
            n_pause += 1
        # 運行閘門：相對持有落後
        ratio_hwm = max(ratio_hwm, ratio)
        if rp.lag_stop > 0:
            if lag_mode:
                resumed = False
                if rp.lag_resume == "bars" and i - lag_bar >= rp.lag_resume_bars:
                    resumed = True
                elif rp.lag_resume == "ratio_high" and i - lag_bar >= 1:
                    win = ratio_hist[max(start_idx, i - rp.lag_resume_bars):i]
                    resumed = bool(len(win) and ratio >= np.nanmax(win))
                if resumed:
                    lag_mode = False
                    lag_resume_at = i
                    if rp.lag_base_mode == "reset":
                        lag_base = ratio
                    lag_events[-1]["resume_bar"] = i
            else:
                if rp.lag_window > 0:
                    j = i - rp.lag_window
                    lag = ratio / ratio_hist[j] - 1 if j >= start_idx and ratio_hist[j] == ratio_hist[j] and ratio_hist[j] > 0 else 0.0
                elif rp.lag_base_mode == "hwm":
                    lag = ratio / ratio_hwm - 1
                else:
                    lag = ratio / lag_base - 1
                lag_min_margin = min(lag_min_margin, lag + rp.lag_stop)
                if lag <= -rp.lag_stop and i - lag_resume_at >= rp.lag_retrigger_cooldown:
                    lag_mode, lag_bar = True, i
                    lag_events.append({"trigger_bar": i, "time": str(idx[i].date()), "lag": float(lag), "ratio": float(ratio), "resume_bar": None, "action": rp.lag_action})
        exiting = any(p[1] == 0.0 for p in pending)
        if mode == "wait_price" and on and not np.isnan(stop_exit_px) and c[i] > stop_exit_px:
            mode = "armed"
        can_enter = mode == "armed" and not exiting and (i - stop_bar) >= rp.reentry_min_bars
        want = on and not paused and ((pos > 0 and not exiting) or can_enter)
        if stop_hit_close:
            want = False
        if not want:
            target = 0.0
        else:
            target = rp.max_exposure
            if scale_arr is not None:
                s = scale_arr[i] if scale_arr[i] == scale_arr[i] else 1.0
                s = float(np.clip(s, 0.0, 1.0))
                if rp.scale_mode == "ratchet_up":
                    scale_hi = max(scale_hi, s)
                    s = scale_hi
                target *= max(s, rp.scale_floor)
            if rp.vol_target > 0 and not np.isnan(rvol[i]) and rvol[i] > 0:
                if rp.vol_target_mode == "relative":
                    factor = 1.0 if np.isnan(vol_ref[i]) else min(1.0, rp.vol_target * vol_ref[i] / rvol[i] / rp.max_exposure)
                else:
                    factor = min(1.0, rp.vol_target / rvol[i] / rp.max_exposure)
                target *= factor
        if lag_mode and not stop_hit_close and rp.lag_action != "record":
            target = rp.max_exposure if rp.lag_action == "hold" else 0.0
        current = pos * c[i] / eqv if eqv > 0 else 0.0
        # 已排隊但還沒成交的目標視為「即將達成」，避免延遲成交時重複排同一個目標
        ref = pending[-1][1] if pending else current
        if stop_hit_close and pos > 0 and not exiting:
            pending = [p for p in pending if p[1] == 0.0]
            pending.append((fill_bar(i), 0.0, "ATR停損"))
            stop_events.append({"decision_bar": i, "fill_bar": None, "stop_level": stop})
            n_stop += 1
            stop = np.nan
            mode = "wait_fill"
        elif target == 0 and ref > 0:
            acting = lag_mode and rp.lag_action != "record"
            reason = "回撤暫停" if paused else ("落後閘門" if acting else ("曝險歸零" if on else "趨勢結束"))
            pending.append((fill_bar(i), 0.0, reason))
        elif target > 0 and ref == 0:
            acting = lag_mode and rp.lag_action != "record"
            pending.append((fill_bar(i), target, "落後閘門持有" if (acting and not on) else "進場"))
        elif target > 0 and ref > 0 and abs(target - ref) > rp.rebalance_threshold:
            pending.append((fill_bar(i), target, "調整曝險"))
        eq[i] = eqv
        exposure[i] = current

    ex = exposure[start_idx:]
    on_slice = on_arr[start_idx:]
    eq_s = eq[start_idx:]
    equity = pd.Series(eq, index=idx, name="strategy").iloc[start_idx:]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_log = np.diff(np.log(eq_s)) - np.diff(np.log(c[start_idx:]))
    rel_log = np.concatenate([[0.0], rel_log])
    ratio_s = ratio_hist[start_idx:]
    rel_dd = float(np.nanmin(ratio_s / np.maximum.accumulate(np.where(np.isnan(ratio_s), -np.inf, ratio_s)) - 1)) if len(ratio_s) else 0.0
    years = max(1e-9, (idx[-1] - idx[start_idx]).days / 365.25)
    extra = {
        "stops": n_stop, "dd_pauses": n_pause, "regime_entries": n_entries, "regime_exits": n_exits,
        "n_fills": len(fills), "fills_by_reason": fills_by_reason, "funding_paid": funding_paid, "borrow_paid": borrow_paid,
        "max_exposure_seen": float(np.nanmax(ex)) if len(ex) else 0.0,
        "min_equity": float(np.nanmin(eq_s)) if len(eq_s) else 0.0,
        "notional_turnover": float(notional / np.nanmean(eq_s)) if np.nanmean(eq_s) > 0 else 0.0,
        "notional_turnover_per_year": float(notional / np.nanmean(eq_s) / years) if np.nanmean(eq_s) > 0 else 0.0,
        "avg_exposure": float(np.nanmean(ex)), "time_in_market": float(np.mean(ex > 0)),
        "avg_exposure_in_regime": float(np.mean(ex[on_slice])) if on_slice.any() else 0.0,
        "partial_days": int(np.sum((ex > 0) & (ex < rp.max_exposure * 0.99))),
        "scale_ups": scale_ups, "scale_downs": scale_downs,
        "lag_triggers": len(lag_events), "lag_events": lag_events,
        "lag_min_margin": float(lag_min_margin) if lag_min_margin != np.inf else None,
        "lag_bars": int(sum(((e["resume_bar"] if e["resume_bar"] is not None else n - 1) - e["trigger_bar"]) for e in lag_events)),
        "rel_dd_vs_hold": rel_dd, "rel_final_vs_hold": float(ratio_s[-1] - 1) if len(ratio_s) else 0.0,
        "rel_log_vs_hold": rel_log, "bench_equity": bench[start_idx:],
        "exposure_key": hashlib.md5(np.round(ex, 4).tobytes()).hexdigest(),
        "fill_key": hashlib.md5(repr(fills).encode()).hexdigest(),
        "stop_events": stop_events,
    }
    return BacktestResult(equity=equity, trades=book.trades_df(), round_trips=book.trips_df(), start_idx=start_idx, extra=extra)
