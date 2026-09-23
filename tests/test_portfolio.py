"""組合引擎測試：python tests/test_portfolio.py"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abgt.portfolio import align, buy_and_hold_portfolio, simulate_portfolio   # noqa: E402
from abgt.strategies.regime import RiskParams, simulate_regime                # noqa: E402
from tests.test_smoke import synthetic_ohlcv                                  # noqa: E402


def _two_assets():
    a = synthetic_ohlcv(500, "1D", seed=9, drift=0.002)
    b = synthetic_ohlcv(500, "1D", seed=4, drift=0.001)
    on_a = np.zeros(500, bool); on_a[40:200] = True; on_a[260:420] = True
    on_b = np.zeros(500, bool); on_b[100:300] = True
    return a, b, on_a, on_b


def test_identical_assets_reduce_to_single():
    a, _, on_a, _ = _two_assets()
    single = simulate_regime(a, on_a, RiskParams(), 20)
    port = simulate_portfolio({"A": a, "B": a.copy()}, {"A": on_a, "B": on_a}, RiskParams(), {"A": 0.5, "B": 0.5}, 20)
    assert float(np.nanmax(np.abs(port.equity.to_numpy() / single.equity.to_numpy() - 1))) < 5e-4


def test_benchmark_math():
    a, b, *_ = _two_assets()
    idx = align({"A": a, "B": b})
    bh = buy_and_hold_portfolio({"A": a, "B": b}, {"A": 0.5, "B": 0.5}, idx, 20, 10000, 0.0005)
    exp = 10000 * 0.9995 * (0.5 * a.close.iloc[-1] / a.close.iloc[20] + 0.5 * b.close.iloc[-1] / b.close.iloc[20])
    assert abs(bh.iloc[-1] - exp) < 1e-6
    bhm = buy_and_hold_portfolio({"A": a, "B": b}, {"A": 0.5, "B": 0.5}, idx, 20, 10000, 0.0005, "M")
    assert bhm.notna().all() and bhm.iloc[0] == bh.iloc[0]


def test_exposure_and_fills():
    a, b, on_a, on_b = _two_assets()
    port = simulate_portfolio({"A": a, "B": b}, {"A": on_a, "B": on_b}, RiskParams(), {"A": 0.5, "B": 0.5}, 20)
    e = port.extra
    assert 0.4 < e["avg_exposure_by_asset"]["A"] / 0.5 < 0.75 and 0.3 < e["avg_exposure_by_asset"]["B"] / 0.5 < 0.5
    assert len(port.round_trips) == 3          # A 兩回合 + B 一回合
    assert set(port.round_trips.asset) == {"A", "B"}
    assert port.equity.notna().all()


def test_lag_guard_portfolio():
    a, b, on_a, on_b = _two_assets()
    base = simulate_portfolio({"A": a, "B": b}, {"A": on_a, "B": on_b}, RiskParams(), {"A": 0.5, "B": 0.5}, 20)
    g = simulate_portfolio({"A": a, "B": b}, {"A": on_a, "B": on_b}, RiskParams(lag_stop=0.1, lag_action="hold", lag_resume="bars", lag_resume_bars=30), {"A": 0.5, "B": 0.5}, 20)
    assert base.extra["lag_triggers"] == 0 and g.extra["lag_triggers"] >= 1
    assert g.extra["rel_dd_vs_hold"] >= base.extra["rel_dd_vs_hold"] - 1e-9


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
