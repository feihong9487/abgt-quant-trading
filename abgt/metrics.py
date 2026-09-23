"""績效指標：淨報酬、CAGR、最大回撤、Sharpe（日）、Calmar、交易統計。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def max_drawdown(eq) -> float:
    a = np.asarray(eq, dtype="float64")
    peak = np.maximum.accumulate(a)
    return float(((a - peak) / peak).min())


def summarize(eq: pd.Series, trades: pd.DataFrame | None = None, trips: pd.DataFrame | None = None) -> dict:
    eq = eq.dropna()
    start, end = eq.index[0], eq.index[-1]
    years = (end - start).total_seconds() / (365.25 * 86400)
    ratio = eq.iloc[-1] / eq.iloc[0]
    d = {
        "start": start, "end": end, "years": years,
        "net_return": ratio - 1,
        "cagr": ratio ** (1 / years) - 1 if years > 0 else math.nan,
        "max_dd": max_drawdown(eq.to_numpy()),
    }
    daily = eq.resample("1D").last().dropna().pct_change().dropna()
    d["sharpe"] = float(daily.mean() / daily.std() * math.sqrt(365)) if len(daily) > 2 and daily.std() > 0 else math.nan
    d["calmar"] = d["cagr"] / abs(d["max_dd"]) if d["max_dd"] < 0 else math.nan
    if trades is not None:
        n = len(trades)
        d["trades"] = n
        if n:
            gp = trades.loc[trades.pnl > 0, "pnl"].sum()
            gl = -trades.loc[trades.pnl < 0, "pnl"].sum()
            d["win_rate"] = float((trades.pnl > 0).mean())
            d["profit_factor"] = gp / gl if gl > 0 else math.inf
        else:
            d["win_rate"] = math.nan
            d["profit_factor"] = math.nan
    if trips is not None:
        d["round_trips"] = len(trips)
        d["trip_win_rate"] = float((trips.pnl > 0).mean()) if len(trips) else math.nan
    return d


def pct(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.1f}%"


def num(x: float, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return "—"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    return f"{x:.{nd}f}"
