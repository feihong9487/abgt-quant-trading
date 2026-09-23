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


def test_portfolio_liquidation_and_vol_target():
    import pandas as pd
    n = 120
    idx = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC")
    px = np.full(n, 100.0)
    mk = lambda: pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99, "close": px, "volume": 1.0}, index=idx)
    a, b = mk(), mk()
    a.iloc[60, a.columns.get_loc("low")] = 60.0
    b.iloc[60, b.columns.get_loc("low")] = 60.0
    on = np.zeros(n, bool); on[10:110] = True
    rp = RiskParams(max_exposure=3.0, commission_pct=0.0, slippage_pct=0.0, liq_level=1.1, liq_penalty=0.0)
    r = simulate_portfolio({"A": a, "B": b}, {"A": on, "B": on}, rp, {"A": 0.5, "B": 0.5}, 5)
    assert r.extra["liquidations"] == 1
    assert abs(r.equity.iloc[-1] - 2000) < 1.0             # 與單標的 3× 案例同解
    assert r.extra["n_fills"] == 4                           # 兩筆進場 + 兩筆強平，不重進
    # 組合層與單標的一致（兩個相同標的）
    s = simulate_regime(a, on, rp, 5)
    assert abs(s.equity.iloc[-1] - r.equity.iloc[-1]) < 1.0
    # 波動目標：壓低平均曝險
    x, y, on_a, on_b = _two_assets()
    base = simulate_portfolio({"A": x, "B": y}, {"A": on_a, "B": on_b}, RiskParams(max_exposure=1.5), {"A": 0.5, "B": 0.5}, 20)
    vt = simulate_portfolio({"A": x, "B": y}, {"A": on_a, "B": on_b}, RiskParams(max_exposure=1.5, vol_target=0.05), {"A": 0.5, "B": 0.5}, 20)
    assert vt.extra["avg_exposure"] < base.extra["avg_exposure"]


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
