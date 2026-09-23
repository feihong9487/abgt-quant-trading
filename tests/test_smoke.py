"""不依賴 pytest 的煙霧測試：python tests/test_smoke.py"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abgt import indicators as ta               # noqa: E402
from abgt.benchmark import buy_and_hold, gate, segment_table   # noqa: E402
from abgt.broker import Book                    # noqa: E402
from abgt.metrics import summarize              # noqa: E402
from abgt.strategies import apex, bgt           # noqa: E402


def synthetic_ohlcv(n: int, freq: str, seed: int = 7, start: str = "2024-01-01", drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, 0.004, n)
    close = 2000 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 0.003, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = rng.lognormal(5, 0.5, n)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def test_rma_ema_seed():
    x = np.full(50, 3.0)
    r = ta.rma(x, 10)
    assert np.isnan(r[8]) and np.isclose(r[9], 3.0) and np.isclose(r[-1], 3.0)
    e = ta.ema(np.arange(30, dtype=float), 5)
    assert np.isnan(e[3]) and np.isclose(e[4], 2.0)   # SMA 起算
    assert np.all(np.diff(e[4:]) > 0)


def test_supertrend_flip():
    up = np.linspace(100, 200, 200)
    down = np.linspace(200, 100, 200)
    c = np.concatenate([up, down])
    h, l = c + 1, c - 1
    a = ta.atr(h, l, c, 14)
    trend, d = ta.supertrend_pine((h + l) / 2, c, a, 2.0)
    assert d[150] == 1 and d[-1] == -1
    flips = np.sum(np.diff(d) != 0)
    assert flips == 1, flips
    assert np.all(trend[150:200] < c[150:200]) and np.all(trend[300:] > c[300:])


def test_pivots():
    low = np.array([5, 4, 3, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    pl = ta.pivot_low(low, 3, 3)
    assert np.isnan(pl[:6]).all() and pl[6] == 2 and np.isnan(pl[7:]).all()
    high = -low
    ph = ta.pivot_high(high, 3, 3)
    assert ph[6] == -2


def test_book_math():
    b = Book(1000.0, 0.001)
    e = b.open(1, 1.0, 100.0, 0, pd.Timestamp("2024-01-01", tz="UTC"))
    assert np.isclose(b.realized, -0.1)
    pnl = b.close(e, 1.0, 110.0, 1, pd.Timestamp("2024-01-02", tz="UTC"), "test")
    assert np.isclose(pnl, 10 - 0.11 - 0.1)
    assert np.isclose(b.equity(999.0), 1000 + 9.79)
    assert len(b.trips) == 1 and np.isclose(b.trips[0]["pnl"], 9.79)
    # 空單
    b2 = Book(1000.0, 0.0)
    e2 = b2.open(-1, 2.0, 100.0, 0, pd.Timestamp("2024-01-01", tz="UTC"))
    assert np.isclose(b2.open_pnl(90.0), 20.0)
    b2.close(e2, 1.0, 90.0, 1, pd.Timestamp("2024-01-02", tz="UTC"), "half")
    assert np.isclose(b2.realized, 10.0) and np.isclose(b2.position(), -1.0)


def test_htf_merge_no_lookahead():
    df = synthetic_ohlcv(3 * 96, "15min")
    htf = ta.resample_ohlcv(df, "45min")
    assert len(htf) == 3 * 32
    htf["c45"] = htf["close"]
    m = ta.merge_htf(df.index, 15, htf, "45min", ["c45"])["c45"]
    # 第 0、1 根 15m 還沒有完成的 45m → NaN；第 2 根（收在 00:45）拿到 00:00 那根 45m 的收盤 = 第 2 根 15m 收盤
    assert np.isnan(m[0]) and np.isnan(m[1]) and np.isclose(m[2], df["close"].iloc[2])
    assert np.isclose(m[3], df["close"].iloc[2]) and np.isclose(m[5], df["close"].iloc[5])


def test_bgt_pipeline():
    df = synthetic_ohlcv(6000, "15min")
    sig = bgt.compute_signals(df, bgt.BGTParams())
    fv = bgt.first_valid_index(sig)
    assert 0 < fv < 1500, fv
    res = bgt.simulate(sig, bgt.BGTEngineParams(), fv)
    assert res.equity.notna().all() and np.isfinite(res.equity).all()
    assert set(res.trades.columns) >= {"entry_id", "pnl", "reason"}
    bh = buy_and_hold(sig["close"].iloc[fv:], 10000, 0.0005)
    s, b = summarize(res.equity, res.trades, res.round_trips), summarize(bh)
    seg = segment_table(res.equity, bh)
    g = gate(s, b, seg)
    assert "passed" in g and len(seg) >= 1
    # 帳本一致性：最後權益 = 初始 + 所有出場淨損益 + 未平倉損益
    last_px = sig["close"].iloc[-1]
    open_pnl = res.equity.iloc[-1] - 10000 - res.trades.pnl.sum()
    assert abs(open_pnl) < 10000, open_pnl


def test_apex_pipeline():
    df = synthetic_ohlcv(900, "1D", drift=0.002, seed=3)
    sig = apex.compute_signals(df, apex.APEXParams())
    fv = apex.first_valid_index(sig)
    assert 199 <= fv < 260, fv
    res = apex.simulate(sig, apex.APEXEngineParams(), fv)
    assert res.equity.notna().all() and np.isfinite(res.equity).all()
    if len(res.trades):
        assert set(res.trades.reason.unique()) <= {"TP1", "TP2", "止損", "追蹤止損", "熔斷"}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as ex:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(ex).__name__}: {ex}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
