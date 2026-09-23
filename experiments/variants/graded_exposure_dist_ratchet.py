"""距離分級曝險（棘輪階梯）：冠軍進場半倉、+0.5 ATR 七五、+1 ATR 滿倉、通道出場。

疊加位置：冠軍 ema50_plain 的進場段（close > EMA50 不變，只改進場當下的倉位與其後的補倉），
出場段換成 vol_atr_channel 的 close < EMA50 − k_out×ATR（k_out=0 消融即冠軍出場）。

規則：
- enter = close > EMA50；exit = close < EMA50 − k_out×ATR14（互斥：兩者不可能同根同真）
- d = (close − EMA50) / ATR14；階梯 raw = clip(floor(d / (k_full / N_STEPS)), 0, N_STEPS)
  N_STEPS = 2 → d < 0.5 → 0.5 倉、0.5 ≤ d < 1.0 → 0.75、d ≥ 1.0 → 1.0
- 棘輪（ratchet=1）：趨勢內級別只升不降；ratchet=0 = 規格原文 clip 版（回檔會降級再回補）
- 無停損、無波動目標、無回撤暫停；rebalance_threshold 0.15 < 0.25 步幅，每次升階都在次日開盤成交

無前視：EMA / ATR / d 在第 i 根只用 ≤ i 根 OHLC（RMA 遞迴、TR 用 close[i−1]）；棘輪由前往後遞推；
引擎 state_machine(enter, exit) 逐根等於本檔迴圈的 state（風控全預設，無停損會改變引擎狀態）。
巢狀特例：(s_min, k_out) = (1, 0) 逐位等於 ema50_plain；(1, 0.25) = 只加出場帶；(0.5, 0) = 只分級。
"""
import os

import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams, state_machine

NAME = "距離分級曝險（棘輪階梯）"
DESCRIPTION = "close > EMA50 進（0.5 倉）、+0.5 ATR → 0.75、+1.0 ATR → 1.0，趨勢內只升不降；close < EMA50 − 0.25 ATR 全出；無停損無波動目標"
PARAMS = {"ema_len": 50, "atr_len": 14, "k_full": 1.0, "s_min": 0.5, "k_out": 0.25, "rebalance_threshold": 0.15, "ratchet": 1}
N_STEPS = 2   # 模組常數（結構參數，不進 PARAMS、不做 ±25%）：階梯 = N_STEPS+1 級


def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    n = len(c)
    ema = ta.ema(c, int(p["ema_len"]))                 # Pine 語意：前 ema_len−1 根 NaN，SMA 起算後遞迴；只用 ≤ i 根收盤
    atr = ta.atr(h, l, c, int(p["atr_len"]))           # RMA(TR)，TR 用 c[i−1]；只用 ≤ i 根
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (c - ema) / atr                             # 距 EMA 幾個 ATR；NaN 維持 NaN
    enter = c > ema                                     # 冠軍的進場條件（NaN 比較 → False）
    exit_ = c < ema - float(p["k_out"]) * atr           # 通道的出場；k_out = 0 時就是冠軍的 close < EMA50
    s_min, k_full = float(p["s_min"]), float(p["k_full"])
    levels = [s_min + j * (1.0 - s_min) / N_STEPS for j in range(N_STEPS + 1)]   # [0.5, 0.75, 1.0]
    step = k_full / N_STEPS                                                      # 0.5 ATR
    with np.errstate(invalid="ignore"):
        raw = np.where(np.isnan(d), 0, np.clip(np.floor(d / step), 0, N_STEPS)).astype(int)   # d<0.5→0、0.5≤d<1→1、d≥1→2
    ratchet = int(p["ratchet"]) == 1
    scale = np.zeros(n)
    lvl_arr = np.zeros(n, dtype=int)
    state = False
    lvl = 0
    for i in range(n):                                  # 由前往後遞推，與引擎 state_machine 同規則
        if np.isnan(ema[i]) or np.isnan(atr[i]):
            state, lvl = False, 0                       # 暖機期一律趨勢外
        elif (not state) and enter[i]:
            state, lvl = True, int(raw[i])              # 進場當根即以當根距離定級（跳空到 +1 ATR 以上就直接滿倉）
        elif state and exit_[i]:
            state, lvl = False, 0
        elif state:
            lvl = max(lvl, int(raw[i])) if ratchet else int(raw[i])   # 棘輪：只升不降；ratchet=0 = 規格原文 clip 版
        scale[i] = levels[lvl] if state else 0.0
        lvl_arr[i] = lvl if state else -1
    valid = ~np.isnan(ema) & ~np.isnan(atr)
    warmup = int(np.argmax(valid)) if valid.any() else n    # = ema_len − 1 = 49，與基準、通道同視窗
    rp = RiskParams(max_exposure=1.0, vol_target=0.0, atr_len=int(p["atr_len"]), atr_stop_mult=0.0,
                    dd_pause=0.0, reentry="immediate", rebalance_threshold=float(p["rebalance_threshold"]))
    out = {"in_regime": {"enter": enter, "exit": exit_, "scale": scale}, "risk": rp, "warmup": warmup}
    if os.environ.get("ABGT_DEBUG"):
        out["_debug"] = {"ema": ema, "atr": atr, "d": d, "raw": raw, "lvl": lvl_arr, "levels": levels}
    return out


if __name__ == "__main__":
    # 自我檢查：enter/exit 互斥；引擎狀態機 == 本地 state；(s_min,k_out)=(1,0) 逐根等於 ema50_plain；截斷不變
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from abgt import data as dt

    for sym in ("BTC/USDT:USDT", "ETH/USDT:USDT"):
        df = dt.load_cached("data", sym, "1d", "2019-09-01")
        if df is None:
            print(sym, "無快取")
            continue
        r = build(df, PARAMS)
        en, ex, sc = r["in_regime"]["enter"], r["in_regime"]["exit"], r["in_regime"]["scale"]
        assert not np.any(en & ex), "enter 與 exit 不可同根同真"
        on = state_machine(en, ex)
        assert np.array_equal(on, sc > 0), "引擎狀態機必須逐根等於本地 state"
        c = df["close"].to_numpy(dtype="float64")
        plain = c > ta.ema(c, PARAMS["ema_len"])
        r4 = build(df, {**PARAMS, "s_min": 1.0, "k_out": 0.0})
        on4 = state_machine(r4["in_regime"]["enter"], r4["in_regime"]["exit"])
        assert np.array_equal(on4, plain), "(s_min,k_out)=(1,0) 應逐根等於 ema50_plain"
        assert np.array_equal(r4["in_regime"]["scale"], plain.astype(float)), "A4 scale 應為 plain 的 0/1"
        assert not np.any(on & ~np.concatenate(([False], on[:-1])) & ~plain), "變體只能在冠軍進場根進場（出場帶只延後出場）"
        for cut in (60, 400, 900, 1500, len(df) - 30, len(df) - 1):
            rc = build(df.iloc[:cut], PARAMS)
            assert np.array_equal(rc["in_regime"]["enter"], en[:cut]), f"截斷 {cut} enter 不一致"
            assert np.array_equal(rc["in_regime"]["exit"], ex[:cut]), f"截斷 {cut} exit 不一致"
            assert np.array_equal(rc["in_regime"]["scale"], sc[:cut]), f"截斷 {cut} scale 不一致"
        inreg = sc[on]
        dist = {lv: float(np.mean(np.isclose(inreg, lv))) for lv in (0.5, 0.75, 1.0)}
        print(f"{sym}: OK | warmup {r['warmup']} | 在場 {on.mean():.1%} | 趨勢內級別分布 {dist} | 平均 scale(趨勢內) {inreg.mean():.3f}")
