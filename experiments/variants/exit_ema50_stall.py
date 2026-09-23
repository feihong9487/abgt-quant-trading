"""EMA50 曝險 + 停滯時間停損 + R 日收盤新高重進。

三態狀態機 FLAT / LONG / STALLED：
- FLAT → LONG：close > EMA50（與 ema50_plain 相同的進場）
- LONG → FLAT：close <= EMA50（基準出場）
- LONG → STALLED：持倉 stall_bars 根未創持倉新高（收盤）且 close < EMA(fast_len)；時間維度的趨勢衰退訊號
- STALLED → LONG：close 創前 reentry_high 根的收盤新高（趨勢恢復）
- STALLED → FLAT：close <= EMA50（趨勢重置，之後走基準進場）

所有判定只用 ≤ i 根的收盤資料：EMA 為因果遞迴；prev_hi_R = highest(close, R) 再 shift(1)，
只含 i−R..i−1 根，不含本根與之後。in_regime 在第 i 根收盤判定，引擎在 i+1 開盤成交。
"""
import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams

NAME = "EMA50 + 停滯時間停損 + 10 日新高重進"
DESCRIPTION = "close > EMA50 進場；持倉 20 根未創新高且 close < EMA20 → 停滯出場；停滯後 close 創 10 日收盤新高重進；跌破 EMA50 回到基準"
PARAMS = {"ema_len": 50, "fast_len": 20, "stall_bars": 20, "reentry_high": 10}


def build(df, params):
    c = df["close"].to_numpy(dtype="float64")
    e_slow = ta.ema(c, int(params["ema_len"]))
    e_fast = ta.ema(c, int(params["fast_len"]))
    B = int(params["stall_bars"])
    R = int(params["reentry_high"])
    # 前 R 根（不含本根）的最高收盤；前 R 根為 NaN → 比較一律 False（不會在無資料時重進）
    prev_hi_R = ta.shift(ta.highest(c, R), 1)

    n = len(c)
    out = np.zeros(n, dtype=bool)
    state = "FLAT"
    hi = np.nan
    hi_bar = -1
    for i in range(n):
        if np.isnan(e_slow[i]) or np.isnan(e_fast[i]):
            out[i] = False
            continue
        above = c[i] > e_slow[i]
        if state == "LONG":
            if not above:
                state = "FLAT"                                  # 基準出場
            else:
                if c[i] > hi:
                    hi = c[i]
                    hi_bar = i                                  # 創持倉新高，計時歸零
                elif (i - hi_bar) >= B and c[i] < e_fast[i]:
                    state = "STALLED"                           # 時間停損：B 根未創新高且短期弱勢
        elif state == "FLAT":
            if above:
                state = "LONG"
                hi = c[i]
                hi_bar = i                                      # 與基準相同的進場
        else:  # STALLED
            if not above:
                state = "FLAT"                                  # 趨勢重置 → 之後用基準進場規則
            elif c[i] > prev_hi_R[i]:
                state = "LONG"
                hi = c[i]
                hi_bar = i                                      # 創 R 日收盤新高 → 趨勢恢復，重進
        out[i] = (state == "LONG")

    both_valid = ~np.isnan(e_slow) & ~np.isnan(e_fast)
    warmup = int(np.argmax(both_valid)) + R                     # 兩條 EMA 都有效的第一根 + R，確保 prev_hi_R 有值
    return {"in_regime": out, "risk": RiskParams(), "warmup": warmup}
