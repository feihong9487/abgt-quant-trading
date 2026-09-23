"""分級曝險正式化 A：距離棘輪階梯 + 冠軍 EMA50 出場（第 3 輪消融 A3 原樣）。

regime 逐位等於冠軍 ema50_plain（enter = close > EMA50、exit = close < EMA50），只改趨勢內的倉位：
- d = (close − EMA50) / ATR14；階梯 raw = clip(floor(d / (k_full / N_STEPS)), 0, N_STEPS)
  N_STEPS = 2 → d < 0.5 → 0.5 倉、0.5 ≤ d < 1.0 → 0.75、d ≥ 1.0 → 1.0
- 進場當根即以當根距離定級（跳空到 ≥ 1 ATR 直接滿倉）；趨勢內級別只升不降（棘輪）；出場全出歸零
- 無停損、無波動目標、無回撤暫停、無運行閘門；rebalance_threshold 0.15 < 0.25 步幅，每次升階都在次日開盤以「調整曝險」成交
- 引擎 scale_mode="track"：逐根跟隨本檔已棘輪的 scale；與 scale_mode="ratchet_up" + 未棘輪即時級別逐位相同（P2 回歸）

無前視：EMA / ATR / d 在第 i 根只用 ≤ i 根 OHLC（RMA 遞迴、TR 用 close[i−1]）；棘輪由前往後遞推；
引擎 state_machine(enter, exit) 逐根等於本檔迴圈的 state。
巢狀特例：s_min = 1.0 逐位等於 ema50_plain（scale ∈ {0, 1}、exposure_key 相同）。

參數逐字沿用第 3 輪 A3 事前值（ema 50、ATR 14、k_full 1.0、s_min 0.5、N_STEPS 2、rebalance 0.15），不得調。
"""
import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams

NAME = "分級曝險 A3：距離棘輪階梯 + EMA50 出場"
DESCRIPTION = "close > EMA50 進（0.5 倉）、+0.5 ATR → 0.75、+1.0 ATR → 1.0，趨勢內只升不降；close < EMA50 全出；regime 與 ema50_plain 逐位相同"
PARAMS = {"ema_len": 50, "atr_len": 14, "k_full": 1.0, "s_min": 0.5, "rebalance_threshold": 0.15}
N_STEPS = 2   # 模組常數（結構參數，不進 PARAMS、不做 ±25%）：階梯 = N_STEPS+1 = 3 級


def _levels_step(p):
    s_min, k_full = float(p["s_min"]), float(p["k_full"])
    levels = [s_min + j * (1.0 - s_min) / N_STEPS for j in range(N_STEPS + 1)]   # [0.5, 0.75, 1.0]
    step = k_full / N_STEPS                                                      # 0.5 ATR
    return levels, step


def _signals(df, p):
    """回傳 (c, ema, atr, enter, exit_, raw)。全部只用 ≤ i 根資料。"""
    c = df["close"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    ema = ta.ema(c, int(p["ema_len"]))              # Pine 語意：前 ema_len−1 根 NaN，SMA 起算後遞迴；第 i 根只用 ≤ i 根 close
    atr = ta.atr(h, l, c, int(p["atr_len"]))        # RMA(TR)，TR 用 c[i−1]；只用 ≤ i 根
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (c - ema) / atr                          # NaN 維持 NaN
    enter = c > ema                                  # 冠軍進場（NaN 比較 → False）
    exit_ = c < ema                                  # 冠軍出場（= A3 的 k_out = 0）
    _, step = _levels_step(p)
    with np.errstate(invalid="ignore"):
        raw = np.where(np.isnan(d), 0, np.clip(np.floor(d / step), 0, N_STEPS)).astype(int)   # d<0.5→0、0.5≤d<1→1、d≥1→2
    return c, ema, atr, enter, exit_, raw


def _ratchet(ema, atr, enter, exit_, raw, levels, ratchet=True):
    """由前往後遞推，與引擎 state_machine 同規則。回傳 (scale, state_arr, lvl_arr)。ratchet=False = 即時級別（供自檢 (2) 用）。"""
    n = len(ema)
    scale = np.zeros(n)
    state_arr = np.zeros(n, dtype=bool)
    lvl_arr = np.full(n, -1, dtype=int)
    state = False
    lvl = 0
    for i in range(n):
        if np.isnan(ema[i]) or np.isnan(atr[i]):
            state, lvl = False, 0                            # 暖機期一律趨勢外
        elif (not state) and enter[i]:
            state, lvl = True, int(raw[i])                   # 進場當根即以當根距離定級（跳空到 ≥1 ATR 直接滿倉）
        elif state and exit_[i]:
            state, lvl = False, 0
        elif state:
            lvl = max(lvl, int(raw[i])) if ratchet else int(raw[i])   # 棘輪：只升不降
        scale[i] = levels[lvl] if state else 0.0
        state_arr[i] = state
        lvl_arr[i] = lvl if state else -1
    return scale, state_arr, lvl_arr


def _risk(p):
    return RiskParams(max_exposure=1.0, atr_len=int(p["atr_len"]), scale_mode="track", scale_floor=0.0,
                      vol_target=0.0, atr_stop_mult=0.0, dd_pause=0.0, lag_stop=0.0,
                      reentry="immediate", exec_delay_bars=0, rebalance_threshold=float(p["rebalance_threshold"]))


def build(df, p):
    c, ema, atr, enter, exit_, raw = _signals(df, p)
    levels, _ = _levels_step(p)
    scale, _, _ = _ratchet(ema, atr, enter, exit_, raw, levels, ratchet=True)
    valid = ~np.isnan(ema) & ~np.isnan(atr)
    warmup = int(np.argmax(valid)) if valid.any() else len(c)   # = 49，與 ema50_plain 同視窗
    return {"in_regime": {"enter": enter, "exit": exit_, "scale": scale}, "risk": _risk(p), "warmup": warmup}


if __name__ == "__main__":
    # 自我檢查（BTC、ETH 各跑，全部 assert；任一失敗即實作失敗，不得改參數）
    import dataclasses
    import os
    import sys

    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, ROOT)
    from abgt import data as dt
    from abgt.strategies.regime import simulate_regime, state_machine
    import evaluate_variant as ev

    plain_mod = ev.load_variant("ema50_plain")
    for sym in ("BTC/USDT:USDT", "ETH/USDT:USDT"):
        df = dt.load_cached("data", sym, "1d", "2019-09-01")
        assert df is not None, f"{sym} 無快取"
        r = build(df, PARAMS)
        en, ex, sc = r["in_regime"]["enter"], r["in_regime"]["exit"], r["in_regime"]["scale"]
        rp, warmup = r["risk"], r["warmup"]
        c = df["close"].to_numpy(dtype="float64")
        n = len(c)
        levels, _ = _levels_step(PARAMS)

        # (1) regime 逐位等於冠軍
        assert not np.any(en & ex), "enter 與 exit 不可同根同真"
        on = state_machine(en, ex)
        plain = c > ta.ema(c, PARAMS["ema_len"])
        assert np.array_equal(on, plain), "state_machine(enter, exit) 必須逐位等於 close > EMA50"
        assert np.array_equal(on, sc > 0), "state_machine 必須逐位等於 scale > 0"
        plain_built = plain_mod.build(df, plain_mod.PARAMS)
        assert ev.regime_key(r["in_regime"]) == ev.regime_key(plain_built["in_regime"]), "regime_key 必須與 ema50_plain 字串相等"

        # (2) 引擎棘輪逐位相同（P2 回歸）
        _, ema, atr, _, _, raw = _signals(df, PARAMS)
        raw_scale, st_raw, _ = _ratchet(ema, atr, en, ex, raw, levels, ratchet=False)   # 不棘輪的即時級別
        assert np.array_equal(st_raw, on)
        res_canon = simulate_regime(df, {"enter": en, "exit": ex, "scale": sc}, rp, warmup)
        res_eng = simulate_regime(df, {"enter": en, "exit": ex, "scale": raw_scale},
                                  dataclasses.replace(rp, scale_mode="ratchet_up"), warmup)
        assert res_eng.extra["exposure_key"] == res_canon.extra["exposure_key"], "引擎 ratchet_up 的 exposure_key 必須與檔內棘輪 track 相同"
        assert res_eng.extra["fill_key"] == res_canon.extra["fill_key"], "引擎 ratchet_up 的 fill_key 必須與檔內棘輪 track 相同"
        assert res_canon.extra["scale_downs"] == 0 and res_eng.extra["scale_downs"] == 0, "棘輪不得有降階成交"

        # (3) 巢狀特例：s_min = 1.0 逐位等於 ema50_plain
        r1 = build(df, {**PARAMS, "s_min": 1.0})
        assert np.array_equal(r1["in_regime"]["scale"], plain.astype(float)), "s_min=1.0 的 scale 應為 plain 的 0/1"
        res_1 = simulate_regime(df, r1["in_regime"], r1["risk"], r1["warmup"])
        res_plain = simulate_regime(df, plain_built["in_regime"], plain_built["risk"], plain_built["warmup"])
        assert res_1.extra["exposure_key"] == res_plain.extra["exposure_key"], "s_min=1.0 的 exposure_key 應等於 ema50_plain"

        # (4) 截斷不變（前視）
        for cut in (60, 400, 900, 1500, n - 30, n - 1):
            rc = build(df.iloc[:cut], PARAMS)
            assert np.array_equal(rc["in_regime"]["enter"], en[:cut]), f"截斷 {cut} enter 不一致"
            assert np.array_equal(rc["in_regime"]["exit"], ex[:cut]), f"截斷 {cut} exit 不一致"
            assert np.array_equal(rc["in_regime"]["scale"], sc[:cut]), f"截斷 {cut} scale 不一致"

        # (5) 統計
        inreg = sc[on]
        dist = {lv: round(float(np.mean(np.isclose(inreg, lv))), 4) for lv in levels}
        starts = np.flatnonzero(on & ~np.concatenate(([False], on[:-1])))
        ends = np.flatnonzero(on & ~np.concatenate((on[1:], [False])))
        bars_to_full, never_full = [], 0
        for s0, e0 in zip(starts, ends):
            path = sc[s0:e0 + 1]
            full = np.flatnonzero(path >= 0.99)
            if len(full):
                bars_to_full.append(int(full[0]))
            else:
                never_full += 1
        n_rounds = len(starts)
        print(f"{sym}: OK | warmup {warmup} | 在場 {on.mean():.1%} | 趨勢內級別分布 {dist} | 平均 scale(趨勢內) {inreg.mean():.3f} | "
              f"回合 {n_rounds} | bars_to_full 中位數 {np.median(bars_to_full) if bars_to_full else float('nan')} (n={len(bars_to_full)}) | "
              f"從未滿倉回合比例 {never_full / n_rounds:.1%} | 成交 {res_canon.extra['n_fills']} {res_canon.extra['fills_by_reason']} | "
              f"scale_ups {res_canon.extra['scale_ups']} scale_downs {res_canon.extra['scale_downs']}")
