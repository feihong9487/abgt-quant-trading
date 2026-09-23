"""曝險型引擎的行為測試：python tests/test_regime.py"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abgt.strategies.regime import RiskParams, simulate_regime, state_machine, regime_atr_channel   # noqa: E402
from tests.test_smoke import synthetic_ohlcv   # noqa: E402


def _flag_df(n=400, seed=11):
    df = synthetic_ohlcv(n, "1D", seed=seed, drift=0.001)
    on = np.zeros(n, dtype=bool)
    on[50:150] = True
    on[200:300] = True
    return df, on


def test_state_machine_matches_manual():
    df = synthetic_ohlcv(600, "1D", seed=5)
    enter, exit_, warmup = regime_atr_channel(df)
    on = state_machine(enter, exit_)
    # 手動重算
    state, manual = False, np.zeros(len(on), bool)
    for i in range(len(on)):
        if not state and enter[i]:
            state = True
        elif state and exit_[i]:
            state = False
        manual[i] = state
    assert (on == manual).all()
    r1 = simulate_regime(df, on, RiskParams(), warmup)
    r2 = simulate_regime(df, {"enter": enter, "exit": exit_}, RiskParams(), warmup)
    assert np.allclose(r1.equity.to_numpy(), r2.equity.to_numpy())


def test_fills_next_open_and_delay():
    df, on = _flag_df()
    r0 = simulate_regime(df, on, RiskParams(), 10)
    assert r0.trades.iloc[0]["entry_time"] == df.index[51]       # on 從第 50 根開始 → 第 51 根開盤進
    assert r0.trades.iloc[0]["exit_time"] == df.index[151]       # 第 150 根轉 off → 第 151 根開盤出
    r1 = simulate_regime(df, on, RiskParams(exec_delay_bars=1), 10)
    assert r1.trades.iloc[0]["entry_time"] == df.index[52] and r1.trades.iloc[0]["exit_time"] == df.index[152]
    assert r0.extra["regime_entries"] == 2 and r0.extra["regime_exits"] == 2


def test_close_trigger_stop_and_reentry():
    df, on = _flag_df()
    # 人為製造崩盤：第 100 根起連跌
    df = df.copy()
    c = df["close"].to_numpy().copy()
    c[100:120] = c[99] * np.linspace(0.97, 0.6, 20)
    df["close"] = c
    df["open"] = np.concatenate([[c[0]], c[:-1]])
    df["high"] = np.maximum(df["open"], df["close"]) * 1.002
    df["low"] = np.minimum(df["open"], df["close"]) * 0.998
    rp = RiskParams(atr_stop_mult=2.0, stop_trigger="close", reentry="reset")
    r = simulate_regime(df, on, rp, 10)
    stops = r.trades[r.trades.reason == "ATR停損"]
    assert len(stops) >= 1 and r.extra["stops"] >= 1
    st = stops.iloc[0]
    i_exit = df.index.get_loc(st["exit_time"])
    assert abs(st["exit_price"] - df["open"].iloc[i_exit] * (1 - rp.slippage_pct)) < 1e-6   # 收盤觸發、下一根開盤成交
    # reset：崩盤後 on 仍為 True（到第 150 根），不得重進
    later = r.trades[(r.trades.entry_time > st["exit_time"]) & (r.trades.entry_time < df.index[150])]
    assert len(later) == 0
    # immediate：可以重進
    r2 = simulate_regime(df, on, RiskParams(atr_stop_mult=2.0, stop_trigger="close", reentry="immediate"), 10)
    st2 = r2.trades[r2.trades.reason == "ATR停損"].iloc[0]
    later2 = r2.trades[(r2.trades.entry_time > st2["exit_time"]) & (r2.trades.entry_time < df.index[150])]
    assert len(later2) >= 1
    # close_above_stop：只有收盤回到停損價之上才重進
    r3 = simulate_regime(df, on, RiskParams(atr_stop_mult=2.0, stop_trigger="close", reentry="close_above_stop"), 10)
    st3 = r3.trades[r3.trades.reason == "ATR停損"].iloc[0]
    later3 = r3.trades[(r3.trades.entry_time > st3["exit_time"]) & (r3.trades.entry_time < df.index[150])]
    for _, t in later3.iterrows():
        i_ent = df.index.get_loc(t["entry_time"])
        assert df["close"].iloc[i_ent - 1] > st3["exit_price"] * 0.999


def test_low_trigger_stop_unchanged():
    df, on = _flag_df(seed=3)
    r = simulate_regime(df, on, RiskParams(atr_stop_mult=0.5, stop_trigger="low"), 10)
    stops = r.trades[r.trades.reason == "ATR停損"]
    assert len(stops) >= 1
    for _, t in stops.iterrows():
        i = df.index.get_loc(t["exit_time"])
        assert t["exit_price"] <= df["open"].iloc[i]   # 觸價成交價不高於開盤


def test_vol_target_modes():
    df, on = _flag_df(n=800)
    on[:] = True
    for mode, side in (("absolute", "total"), ("relative", "total"), ("relative", "downside")):
        rp = RiskParams(vol_target=0.3 if mode == "absolute" else 1.0, vol_target_mode=mode, vol_side=side, vol_lookback=20, vol_ref_lookback=200)
        r = simulate_regime(df, on, rp, 30)
        assert 0 < r.extra["avg_exposure"] <= 1.0 + 1e-6, (mode, side, r.extra)
        assert r.equity.notna().all()


def test_funding_reduces_equity():
    df, on = _flag_df()
    r0 = simulate_regime(df, on, RiskParams(), 10)
    r1 = simulate_regime(df, on, RiskParams(funding_rate_8h=0.0003), 10)
    assert r1.extra["funding_paid"] > 0 and r1.equity.iloc[-1] < r0.equity.iloc[-1]


def test_dd_pause_resume_bars():
    df, on = _flag_df()
    df = df.copy()
    c = df["close"].to_numpy().copy()
    c[60:80] = c[59] * np.linspace(0.95, 0.5, 20)      # 趨勢內深跌 → 觸發回撤暫停
    c[80:150] = c[79] * np.linspace(1.0, 1.6, 70)      # 之後強彈
    df["close"] = c
    df["open"] = np.concatenate([[c[0]], c[:-1]])
    df["high"] = np.maximum(df["open"], df["close"]) * 1.002
    df["low"] = np.minimum(df["open"], df["close"]) * 0.998
    r_reset = simulate_regime(df, on, RiskParams(dd_pause=0.2, dd_pause_resume="regime_reset"), 10)
    r_bars = simulate_regime(df, on, RiskParams(dd_pause=0.2, dd_pause_resume="bars", dd_resume_bars=10), 10)
    assert r_reset.extra["dd_pauses"] >= 1 and r_bars.extra["dd_pauses"] >= 1
    # bars 模式會在暫停後 10 根重新進場，reset 模式要等第 150 根趨勢結束
    pause_exit = r_bars.trades[r_bars.trades.reason == "回撤暫停"].iloc[0]["exit_time"]
    re_bars = r_bars.trades[(r_bars.trades.entry_time > pause_exit) & (r_bars.trades.entry_time < df.index[150])]
    re_reset = r_reset.trades[(r_reset.trades.entry_time > pause_exit) & (r_reset.trades.entry_time < df.index[150])]
    assert len(re_bars) >= 1 and len(re_reset) == 0
    assert df.index.get_loc(re_bars.iloc[0]["entry_time"]) - df.index.get_loc(pause_exit) >= 10


def test_close_stop_with_delay_no_reentry_bypass():
    """P1：收盤觸發停損 × 延遲成交時，出場排隊中不得再排進場；重進規則從成交根起算。"""
    df, on = _flag_df()
    df = df.copy()
    c = df["close"].to_numpy().copy()
    c[100:120] = c[99] * np.linspace(0.97, 0.6, 20)
    df["close"] = c
    df["open"] = np.concatenate([[c[0]], c[:-1]])
    df["high"] = np.maximum(df["open"], df["close"]) * 1.002
    df["low"] = np.minimum(df["open"], df["close"]) * 0.998
    for reentry in ("reset", "close_above_stop", "immediate"):
        rp = RiskParams(atr_stop_mult=2.0, stop_trigger="close", reentry=reentry, exec_delay_bars=1, reentry_min_bars=5)
        r = simulate_regime(df, on, rp, 10)
        stops = r.trades[r.trades.reason == "ATR停損"]
        assert len(stops) >= 1, reentry
        st = stops.iloc[0]
        i_fill = df.index.get_loc(st["exit_time"])
        # 停損成交根之前（決策根到成交根之間）不得有任何進場
        between = r.trades[(r.trades.entry_time > df.index[i_fill - 2]) & (r.trades.entry_time <= st["exit_time"])]
        assert len(between) == 0, (reentry, between)
        later = r.trades[(r.trades.entry_time > st["exit_time"]) & (r.trades.entry_time < df.index[150])]
        if reentry == "reset":
            assert len(later) == 0
        else:
            for _, t in later.iterrows():
                assert df.index.get_loc(t["entry_time"]) - i_fill >= 5 + 1   # reentry_min_bars 從成交根起算，再加下一根成交
        ev = r.extra["stop_events"]
        assert ev and ev[0]["fill_bar"] == i_fill and ev[0]["decision_bar"] == i_fill - 2


def test_scale_graded_exposure():
    """P2：scale 陣列分級曝險，經 rebalance_threshold 才調倉。"""
    df, on = _flag_df()
    scale = np.where(on, 0.5, np.nan)
    scale[80:150] = 1.0
    r = simulate_regime(df, {"in_regime": on, "scale": scale}, RiskParams(rebalance_threshold=0.1), 10)
    ex = r.extra
    assert ex["partial_days"] > 20 and 0.4 < ex["avg_exposure_in_regime"] < 1.0
    adj = r.trades[r.trades.reason == "調整曝險"]
    assert len(adj) == 0   # 只有加碼（0.5→1.0）沒有減碼，所以沒有「調整曝險」的出場紀錄
    assert ex["n_fills"] >= 4 and ex["regime_entries"] == 2 and ex["regime_exits"] == 2
    r0 = simulate_regime(df, on, RiskParams(), 10)
    assert r0.extra["exposure_key"] != ex["exposure_key"]


def test_funding_array_income_and_cost():
    df, on = _flag_df()
    r0 = simulate_regime(df, on, RiskParams(), 10)
    neg = np.full(len(df), -0.0003)
    r_income = simulate_regime(df, on, RiskParams(funding=neg), 10)
    r_cost = simulate_regime(df, on, RiskParams(funding=-neg), 10)
    assert r_income.equity.iloc[-1] > r0.equity.iloc[-1] > r_cost.equity.iloc[-1]
    assert r_income.extra["funding_paid"] < 0 < r_cost.extra["funding_paid"]


def test_scale_ratchet_and_floor():
    """P2：ratchet_up 趨勢內只升不降；scale_floor 是趨勢內曝險下限。"""
    df, on = _flag_df()
    scale = np.where(on, 0.5, np.nan)
    scale[70:90] = 1.0      # 趨勢內先升到 1.0，之後 scale 回到 0.5
    r_track = simulate_regime(df, {"in_regime": on, "scale": scale}, RiskParams(rebalance_threshold=0.1), 10)
    r_ratch = simulate_regime(df, {"in_regime": on, "scale": scale}, RiskParams(rebalance_threshold=0.1, scale_mode="ratchet_up"), 10)
    assert r_track.extra["scale_downs"] >= 1 and r_ratch.extra["scale_downs"] == 0
    assert r_ratch.extra["avg_exposure_in_regime"] > r_track.extra["avg_exposure_in_regime"]
    r_floor = simulate_regime(df, {"in_regime": on, "scale": np.where(on, 0.2, np.nan)}, RiskParams(scale_floor=0.6), 10)
    ex_in = r_floor.extra["avg_exposure_in_regime"]
    assert 0.55 < ex_in < 0.7, ex_in


def test_lag_stop_hold_and_flat():
    """P1：策略相對同起點持有落後 ≥ X 時觸發運行閘門；hold 轉 100% 持有、flat 空手；bars 恢復。"""
    df = synthetic_ohlcv(400, "1D", seed=21, drift=0.004)   # 強勢上漲：趨勢外空手的策略必落後持有
    on = np.zeros(400, dtype=bool)
    on[30:60] = True                                       # 之後長期空手 → 落後累積
    base = simulate_regime(df, on, RiskParams(), 10)
    hold = simulate_regime(df, on, RiskParams(lag_stop=0.10, lag_action="hold", lag_resume="bars", lag_resume_bars=40), 10)
    flat = simulate_regime(df, on, RiskParams(lag_stop=0.10, lag_action="flat", lag_resume="bars", lag_resume_bars=40), 10)
    assert base.extra["lag_triggers"] == 0 and hold.extra["lag_triggers"] >= 1
    assert hold.extra["rel_dd_vs_hold"] > base.extra["rel_dd_vs_hold"]          # hold 版相對持有的最深落後較淺
    assert hold.equity.iloc[-1] > base.equity.iloc[-1]                          # 上漲市場裡轉持有賺回來
    ev = hold.extra["lag_events"][0]
    assert ev["resume_bar"] is None or ev["resume_bar"] - ev["trigger_bar"] >= 40
    assert flat.extra["lag_triggers"] >= 1 and "落後閘門持有" in hold.extra["fills_by_reason"]
    assert flat.extra["fills_by_reason"].get("落後閘門持有", 0) == 0
    assert abs(hold.extra["rel_log_vs_hold"].sum() - (np.log(hold.equity.iloc[-1] / hold.equity.iloc[0]) - np.log(df["close"].iloc[-1] / df["close"].iloc[10]))) < 1e-6


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
