"""用 ccxt 從 Binance USDT-M 抓 K 線，存 CSV 快取，之後只補新資料。"""
from __future__ import annotations

import os

import ccxt
import pandas as pd


def cache_path(cache_dir: str, symbol: str, timeframe: str) -> str:
    fname = f"{symbol.replace('/', '').replace(':', '_')}_{timeframe}.csv"
    return os.path.join(cache_dir, fname)


def load_cached(cache_dir: str, symbol: str, timeframe: str, since: str) -> pd.DataFrame | None:
    path = cache_path(cache_dir, symbol, timeframe)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    since_ms = int(pd.Timestamp(since, tz="UTC").timestamp() * 1000)
    df = df[df["timestamp"] >= since_ms]
    return _to_frame(df) if len(df) else None


def fetch_ohlcv(symbol: str, timeframe: str, since: str, cache_dir: str = "data",
                exchange_id: str = "binanceusdm", limit: int = 1500, verbose: bool = True) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path(cache_dir, symbol, timeframe)
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    since_ms = ex.parse8601(f"{since}T00:00:00Z")
    now_ms = ex.milliseconds()

    cached = None
    cursor = since_ms
    if os.path.exists(path):
        cached = pd.read_csv(path)
        if len(cached) and int(cached["timestamp"].iloc[0]) <= since_ms:
            cursor = int(cached["timestamp"].iloc[-1]) + tf_ms
        else:
            cached = None

    rows = []
    calls = 0
    while cursor < now_ms - tf_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        calls += 1
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + tf_ms
        if nxt <= cursor:
            break
        cursor = nxt
        if verbose and calls % 10 == 0:
            print(f"  抓取 {symbol} {timeframe}: 已到 {pd.Timestamp(cursor, unit='ms', tz='UTC'):%Y-%m-%d}")

    new = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.concat([cached, new]) if cached is not None else new
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df = df[df["timestamp"] + tf_ms <= now_ms]   # 丟掉未收完的那根
    df = df[df["timestamp"] >= since_ms]
    df.to_csv(path, index=False)
    if verbose:
        print(f"  {symbol} {timeframe}: {len(df)} 根（新抓 {len(new)}），{pd.Timestamp(int(df['timestamp'].iloc[0]), unit='ms', tz='UTC'):%Y-%m-%d} → {pd.Timestamp(int(df['timestamp'].iloc[-1]), unit='ms', tz='UTC'):%Y-%m-%d}")
    return _to_frame(df)


def _to_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.pop("timestamp"), unit="ms", utc=True)
    out.index.name = "time"
    return out[["open", "high", "low", "close", "volume"]].astype("float64")
