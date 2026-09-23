"""對應 Pine Script v6 ta.* 的指標實作。

語意盡量貼近 Pine：
- ema / rma 前 n-1 根為 na，第 n 根以 SMA 起算，之後遞迴（Pine 也是 SMA 起算）
- 所有輸入輸出都是 numpy float 陣列，na 用 NaN；布林比較遇到 NaN 一律 False（跟 Pine 一樣）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype="float64")


def shift(x, n: int = 1) -> np.ndarray:
    """x[n]：往後移 n 根，前 n 根補 NaN。"""
    a = _arr(x)
    out = np.full(a.shape, np.nan)
    if n == 0:
        return a.copy()
    if n < len(a):
        out[n:] = a[:-n]
    return out


def shift_bool(b, n: int = 1) -> np.ndarray:
    a = np.asarray(b, dtype=bool)
    out = np.zeros(a.shape, dtype=bool)
    if n < len(a):
        out[n:] = a[:-n]
    return out


def _seeded_recursive(x, alpha: float, n: int) -> np.ndarray:
    s = pd.Series(_arr(x))
    out = np.full(len(s), np.nan)
    valid = s.notna().to_numpy()
    if valid.sum() < n:
        return out
    first = int(np.argmax(valid))
    seed_pos = first + n - 1
    if seed_pos >= len(s):
        return out
    seed = s.iloc[first:seed_pos + 1].mean()
    tail = s.iloc[seed_pos:].copy()
    tail.iloc[0] = seed
    out[seed_pos:] = tail.ewm(alpha=alpha, adjust=False, ignore_na=True).mean().to_numpy()
    return out


def ema(x, n: int) -> np.ndarray:
    return _seeded_recursive(x, 2.0 / (n + 1), n)


def rma(x, n: int) -> np.ndarray:
    return _seeded_recursive(x, 1.0 / n, n)


def sma(x, n: int) -> np.ndarray:
    return pd.Series(_arr(x)).rolling(n).mean().to_numpy()


def highest(x, n: int) -> np.ndarray:
    return pd.Series(_arr(x)).rolling(n).max().to_numpy()


def lowest(x, n: int) -> np.ndarray:
    return pd.Series(_arr(x)).rolling(n).min().to_numpy()


def true_range(high, low, close, handle_na: bool = True) -> np.ndarray:
    """ta.tr(handle_na)：handle_na=True 時第一根用 high-low（ta.atr 用這個），False 時第一根 na（ta.tr 用這個）。"""
    h, l, c = _arr(high), _arr(low), _arr(close)
    pc = shift(c, 1)
    tr = np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc)))
    if not handle_na:
        tr[0] = np.nan
    return tr


def atr(high, low, close, n: int) -> np.ndarray:
    return rma(true_range(high, low, close, True), n)


def rsi(close, n: int) -> np.ndarray:
    c = _arr(close)
    ch = c - shift(c, 1)
    up = np.where(ch > 0, ch, 0.0)
    dn = np.where(ch < 0, -ch, 0.0)
    nan = np.isnan(ch)
    up[nan] = np.nan
    dn[nan] = np.nan
    ru, rd = rma(up, n), rma(dn, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(rd == 0, 100.0, np.where(ru == 0, 0.0, 100.0 - 100.0 / (1.0 + ru / rd)))
    out[np.isnan(ru) | np.isnan(rd)] = np.nan
    return out


def macd(close, fast: int = 12, slow: int = 26, signal: int = 9):
    m = ema(close, fast) - ema(close, slow)
    return m, ema(m, signal)


def crossover(a, b) -> np.ndarray:
    a, b = _arr(a), _arr(b)
    return (a > b) & (shift(a) <= shift(b))


def crossunder(a, b) -> np.ndarray:
    a, b = _arr(a), _arr(b)
    return (a < b) & (shift(a) >= shift(b))


def dmi(high, low, close, di_len: int, adx_len: int):
    """ta.dmi(diLength, adxSmoothing) → (+DI, -DI, ADX)。"""
    h, l, c = _arr(high), _arr(low), _arr(close)
    up = h - shift(h)
    down = shift(l) - l
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm[np.isnan(up)] = np.nan
    minus_dm[np.isnan(down)] = np.nan
    trur = rma(true_range(h, l, c, False), di_len)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus = 100.0 * rma(plus_dm, di_len) / trur
        minus = 100.0 * rma(minus_dm, di_len) / trur
    s = plus + minus
    dx = np.abs(plus - minus) / np.where(s == 0, 1.0, s)
    adx = 100.0 * rma(dx, adx_len)
    return plus, minus, adx


def supertrend_pine(hl2, close, atr_v, mult: float):
    """對應 BGT 的 calcTrend()：回傳 (trendLine, trendDir)。

    finalUp := (upBand < nz(finalUp[1], upBand) or close[1] > nz(finalUp[1], upBand)) ? upBand : nz(finalUp[1], upBand)
    finalDn := (dnBand > nz(finalDn[1], dnBand) or close[1] < nz(finalDn[1], dnBand)) ? dnBand : nz(finalDn[1], dnBand)
    trendDir := close > nz(finalUp[1], upBand) ? 1 : close < nz(finalDn[1], dnBand) ? -1 : nz(trendDir[1], 1)
    """
    hl2, c, a = _arr(hl2), _arr(close), _arr(atr_v)
    n = len(c)
    fu = np.full(n, np.nan)
    fd = np.full(n, np.nan)
    d = np.ones(n, dtype=np.int8)
    prev_up = prev_dn = np.nan
    prev_dir = 1
    for i in range(n):
        up = hl2[i] + mult * a[i]
        dn = hl2[i] - mult * a[i]
        pu = up if np.isnan(prev_up) else prev_up
        pdn = dn if np.isnan(prev_dn) else prev_dn
        c1 = c[i - 1] if i > 0 else np.nan
        new_up = up if (up < pu or c1 > pu) else pu
        new_dn = dn if (dn > pdn or c1 < pdn) else pdn
        if c[i] > pu:
            nd = 1
        elif c[i] < pdn:
            nd = -1
        else:
            nd = prev_dir
        fu[i], fd[i], d[i] = new_up, new_dn, nd
        prev_up, prev_dn, prev_dir = new_up, new_dn, nd
    trend = np.where(d == 1, fd, fu)
    return trend, d.astype("float64")


def pivot_low(low, left: int, right: int) -> np.ndarray:
    """ta.pivotlow(low, left, right)：在樞紐後 right 根才確認，值填在確認那根。"""
    l = _arr(low)
    n = len(l)
    out = np.full(n, np.nan)
    w = left + right + 1
    if n < w or right < 1:
        return out
    win = sliding_window_view(l, w)
    center = win[:, left]
    others = np.fmin(np.min(win[:, :left], axis=1), np.min(win[:, left + 1:], axis=1))
    ok = center < others
    idx = np.arange(n - w + 1) + w - 1
    out[idx[ok]] = center[ok]
    return out


def pivot_high(high, left: int, right: int) -> np.ndarray:
    h = _arr(high)
    n = len(h)
    out = np.full(n, np.nan)
    w = left + right + 1
    if n < w or right < 1:
        return out
    win = sliding_window_view(h, w)
    center = win[:, left]
    others = np.fmax(np.max(win[:, :left], axis=1), np.max(win[:, left + 1:], axis=1))
    ok = center > others
    idx = np.arange(n - w + 1) + w - 1
    out[idx[ok]] = center[ok]
    return out


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """15m → 45min / 4h。origin=epoch 讓邊界落在 UTC 整點（1440 可被 45 與 240 整除，跟 TradingView 對齊）。"""
    o = df.resample(rule, origin="epoch", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
    )
    return o.dropna(subset=["open"])


def merge_htf(index: pd.DatetimeIndex, tf_minutes: int, htf: pd.DataFrame, htf_rule: str, cols: list[str]) -> dict[str, np.ndarray]:
    """request.security(..., lookahead_off) 的歷史語意：
    每根圖表 K 線拿到「收盤時間 ≤ 該 K 線收盤時間」的最後一根已完成 HTF K 線的值。
    """
    htf_minutes = int(pd.Timedelta(htf_rule).total_seconds() // 60)
    left = pd.DataFrame({"ct": index + pd.Timedelta(minutes=tf_minutes)})
    right = htf[cols].copy()
    right["ct"] = htf.index + pd.Timedelta(minutes=htf_minutes)
    right = right.sort_values("ct")
    m = pd.merge_asof(left, right, on="ct", direction="backward")
    return {c: m[c].to_numpy(dtype="float64") for c in cols}
