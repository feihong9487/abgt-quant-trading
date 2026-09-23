"""雙（多）標的組合引擎：一本共用權益，每個標的有自己的 regime 與權重。

支援的 RiskParams 欄位：max_exposure、rebalance_threshold、exec_delay_bars、commission_pct、slippage_pct、
funding_rate_8h / funding（每標的一個陣列，用 fundings dict）、dd_pause 系列（組合層）、
lag_stop 系列（組合層，基準 = 同權重買入持有不再平衡）。scale 陣列只支援 track 模式。
不支援：波動目標、ATR 停損、重進規則（這些是單標的語意，組合層要先證明有必要再加）。

執行語意與單標的引擎相同：收盤判定、下一根（+ 延遲）開盤成交、每邊手續費、只做多。
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .broker import BacktestResult
from .strategies.regime import RiskParams, regime_flags


def align(dfs: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    idx = None
    for df in dfs.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    return idx


def buy_and_hold_portfolio(dfs: dict[str, pd.DataFrame], weights: dict[str, float], idx: pd.DatetimeIndex, start_idx: int,
                           initial: float, fee: float, rebalance: str | None = None) -> pd.Series:
    """同權重買入持有。rebalance=None 不再平衡；"M" 每月第一根再平衡（付手續費）。"""
    closes = {k: dfs[k]["close"].reindex(idx).to_numpy(dtype="float64") for k in dfs}
    n = len(idx)
    eq = np.full(n, np.nan)
    qty = {k: initial * (1 - fee) * weights[k] / closes[k][start_idx] for k in dfs}
    last_month = idx[start_idx].month
    for i in range(start_idx, n):
        val = sum(qty[k] * closes[k][i] for k in dfs)
        if rebalance == "M" and idx[i].month != last_month:
            last_month = idx[i].month
            new = {}
            cost = 0.0
            for k in dfs:
                target_val = val * weights[k]
                cur_val = qty[k] * closes[k][i]
                cost += abs(target_val - cur_val) * fee
                new[k] = target_val / closes[k][i]
            val -= cost
            qty = {k: new[k] * (val / sum(new[j] * closes[j][i] for j in dfs)) for k in dfs}
        eq[i] = val
    return pd.Series(eq, index=idx, name="buy_and_hold").iloc[start_idx:]


def simulate_portfolio(dfs: dict[str, pd.DataFrame], regimes: dict, rp: RiskParams, weights: dict[str, float],
                       start_idx: int, fundings: dict[str, np.ndarray] | None = None) -> BacktestResult:
    idx = align(dfs)
    keys = list(dfs)
    n = len(idx)
    o = {k: dfs[k]["open"].reindex(idx).to_numpy(dtype="float64") for k in keys}
    c = {k: dfs[k]["close"].reindex(idx).to_numpy(dtype="float64") for k in keys}
    on, scale = {}, {}
    for k in keys:
        on_k, sc_k = regime_flags(regimes[k])
        src_idx = dfs[k].index
        on[k] = pd.Series(on_k, index=src_idx).reindex(idx).fillna(False).to_numpy(dtype=bool)
        scale[k] = pd.Series(sc_k, index=src_idx).reindex(idx).to_numpy(dtype="float64") if sc_k is not None else None
    fund = {k: (pd.Series(fundings[k], index=dfs[k].index).reindex(idx).fillna(0.0).to_numpy(dtype="float64") if fundings and k in fundings else None) for k in keys}

    cash = rp.initial_capital
    qty = {k: 0.0 for k in keys}
    pending: dict[str, list[tuple[int, float, str]]] = {k: [] for k in keys}
    fills: list[dict] = []
    trips: list[dict] = []
    trip_open: dict[str, dict | None] = {k: None for k in keys}
    eq = np.full(n, np.nan)
    expo = {k: np.zeros(n) for k in keys}
    bench_qty = {k: rp.initial_capital * (1 - rp.commission_pct) * weights[k] / c[k][start_idx] for k in keys}
    bench = np.full(n, np.nan)
    ratio_hist = np.full(n, np.nan)
    peak = rp.initial_capital
    paused = False
    pause_bar = -10 ** 9
    lag_mode = False
    lag_bar = -10 ** 9
    lag_base = 1.0
    lag_events: list[dict] = []
    funding_paid = 0.0
    borrow_paid = 0.0
    notional = 0.0
    n_pause = 0
    bars_per_year = 365.0 * 86400.0 / pd.Series(idx).diff().dropna().median().total_seconds()

    def fill_bar(i: int) -> int:
        return i + 1 + rp.exec_delay_bars

    for i in range(start_idx, n):
        t = idx[i]
        # 1) 到期目標在開盤成交（每標的各自）
        for k in keys:
            due = [p for p in pending[k] if p[0] <= i]
            if not due:
                continue
            pending[k] = [p for p in pending[k] if p[0] > i]
            _, target, reason = due[-1]
            eq_open = cash + sum(qty[j] * o[j][i] for j in keys)
            target_qty = target * eq_open / o[k][i]
            diff = target_qty - qty[k]
            if abs(diff) * o[k][i] < 1e-6 * eq_open:
                continue
            if diff > 0:
                px = o[k][i] * (1 + rp.slippage_pct)
                cost = diff * px
                cash -= cost * (1 + rp.commission_pct)
                if qty[k] <= 1e-12:
                    trip_open[k] = {"asset": k, "start_time": t, "start_bar": i, "entry_px": px, "n_entries": 0}
                qty[k] += diff
                trip_open[k]["n_entries"] += 1
            else:
                px = o[k][i] * (1 - rp.slippage_pct)
                proceeds = -diff * px
                cash += proceeds * (1 - rp.commission_pct)
                qty[k] += diff
                if qty[k] <= 1e-12:
                    qty[k] = 0.0
                    tr = trip_open[k]
                    if tr:
                        trips.append({**{x: tr[x] for x in ("asset", "start_time", "start_bar", "entry_px", "n_entries")},
                                      "end_time": t, "exit_px": px, "bars": i - tr["start_bar"], "log_ret": float(np.log(px / tr["entry_px"])), "last_reason": reason})
                        trip_open[k] = None
            notional += abs(diff) * px
            fills.append({"asset": k, "time": t, "side": 1 if diff > 0 else -1, "qty": abs(diff), "price": px, "reason": reason})
        # 2) 收盤：資金費率、權益、回撤、運行閘門、目標曝險
        for k in keys:
            if qty[k] > 0:
                rate = fund[k][i] if fund[k] is not None else rp.funding_rate_8h * 3
                if rate == rate and rate != 0.0:
                    cost = qty[k] * c[k][i] * rate
                    cash -= cost
                    funding_paid += cost
        eqv = cash + sum(qty[k] * c[k][i] for k in keys)
        if rp.borrow_rate_annual > 0 and eqv > 0:
            borrowed = max(0.0, sum(qty[k] * c[k][i] for k in keys) - eqv)
            if borrowed > 0:
                cost = borrowed * rp.borrow_rate_annual / bars_per_year
                cash -= cost
                borrow_paid += cost
                eqv -= cost
        bench[i] = sum(bench_qty[k] * c[k][i] for k in keys)
        ratio = eqv / bench[i] if bench[i] > 0 else 1.0
        ratio_hist[i] = ratio
        peak = max(peak, eqv)
        dd = 1 - eqv / peak if peak > 0 else 0.0
        any_on = any(on[k][i] for k in keys)
        if not any_on:
            paused = False
            peak = eqv
            if lag_mode and rp.lag_resume == "regime_reset":
                lag_mode, lag_base = False, ratio
                lag_events[-1]["resume_bar"] = i
        elif paused:
            if rp.dd_pause_resume == "bars" and i - pause_bar >= rp.dd_resume_bars:
                paused, peak = False, eqv
        if rp.dd_pause > 0 and dd >= rp.dd_pause and not paused and any(qty[k] > 0 for k in keys):
            paused, pause_bar = True, i
            n_pause += 1
        if rp.lag_stop > 0:
            if lag_mode:
                if rp.lag_resume == "bars" and i - lag_bar >= rp.lag_resume_bars:
                    lag_mode, lag_base = False, ratio
                    lag_events[-1]["resume_bar"] = i
                elif rp.lag_resume == "ratio_high" and i - lag_bar >= 1:
                    win = ratio_hist[max(start_idx, i - rp.lag_resume_bars):i]
                    if len(win) and ratio >= np.nanmax(win):
                        lag_mode, lag_base = False, ratio
                        lag_events[-1]["resume_bar"] = i
            else:
                if rp.lag_window > 0:
                    j = i - rp.lag_window
                    lag = ratio / ratio_hist[j] - 1 if j >= start_idx and ratio_hist[j] == ratio_hist[j] and ratio_hist[j] > 0 else 0.0
                else:
                    lag = ratio / lag_base - 1
                if lag <= -rp.lag_stop:
                    lag_mode, lag_bar = True, i
                    lag_events.append({"trigger_bar": i, "time": str(t.date()), "lag": float(lag), "resume_bar": None, "action": rp.lag_action})
        for k in keys:
            want = on[k][i] and not paused
            target = 0.0
            if want:
                s = 1.0
                if scale[k] is not None and scale[k][i] == scale[k][i]:
                    s = float(np.clip(scale[k][i], 0.0, 1.0))
                target = weights[k] * rp.max_exposure * s
            if lag_mode:
                target = weights[k] * rp.max_exposure if rp.lag_action == "hold" else 0.0
            current = qty[k] * c[k][i] / eqv if eqv > 0 else 0.0
            ref = pending[k][-1][1] if pending[k] else current
            if target == 0 and ref > 0:
                pending[k].append((fill_bar(i), 0.0, "回撤暫停" if paused else ("落後閘門" if lag_mode else "趨勢結束")))
            elif target > 0 and ref == 0:
                pending[k].append((fill_bar(i), target, "落後閘門持有" if (lag_mode and not on[k][i]) else "進場"))
            elif target > 0 and ref > 0 and abs(target - ref) > rp.rebalance_threshold * weights[k]:
                pending[k].append((fill_bar(i), target, "調整曝險"))
            expo[k][i] = current
        eq[i] = eqv

    equity = pd.Series(eq, index=idx, name="portfolio").iloc[start_idx:]
    eq_s = eq[start_idx:]
    ratio_s = ratio_hist[start_idx:]
    rel_dd = float(np.nanmin(ratio_s / np.maximum.accumulate(np.where(np.isnan(ratio_s), -np.inf, ratio_s)) - 1)) if len(ratio_s) else 0.0
    years = max(1e-9, (idx[-1] - idx[start_idx]).days / 365.25)
    total_expo = sum(expo[k][start_idx:] for k in keys)
    trades = pd.DataFrame(fills)
    if len(trades):
        trades = trades.rename(columns={"time": "exit_time"})
        trades["pnl"] = np.nan
    rt = pd.DataFrame(trips)
    if len(rt):
        rt["pnl"] = rt["log_ret"]
    extra = {
        "assets": keys, "weights": weights,
        "n_fills": len(fills), "funding_paid": funding_paid, "borrow_paid": borrow_paid,
        "max_exposure_seen": float(np.nanmax(total_expo)) if len(total_expo) else 0.0,
        "min_equity": float(np.nanmin(eq_s)) if len(eq_s) else 0.0,
        "notional_turnover_per_year": float(notional / np.nanmean(eq_s) / years) if np.nanmean(eq_s) > 0 else 0.0,
        "avg_exposure": float(np.nanmean(total_expo)), "time_in_market": float(np.mean(total_expo > 0)),
        "avg_exposure_by_asset": {k: float(np.nanmean(expo[k][start_idx:])) for k in keys},
        "dd_pauses": n_pause, "lag_triggers": len(lag_events), "lag_events": lag_events,
        "rel_dd_vs_hold": rel_dd, "rel_final_vs_hold": float(ratio_s[-1] - 1) if len(ratio_s) else 0.0,
        "bench_equity": bench[start_idx:],
        "exposure_key": hashlib.md5(np.round(total_expo, 4).tobytes()).hexdigest(),
    }
    return BacktestResult(equity=equity, trades=trades, round_trips=rt, start_idx=start_idx, extra=extra)
