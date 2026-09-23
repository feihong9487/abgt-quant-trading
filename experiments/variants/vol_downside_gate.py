"""EMA50 曝險 + 下行波動狀態閘：崩盤狀態空手（遲滯），旗標解除且 close > EMA50 就立即重進。

指標（第 i 根只用 ≤ i 根收盤）：
- r = ln(close / close[1])
- dn2 = r² if r < 0 else 0；up2 = r² if r > 0 else 0
- dvol = sqrt(mean(dn2, w))、uvol = sqrt(mean(up2, w))（右對齊窗，含當根不含之後）
- ratio = dvol / max(uvol, 1e-12)
崩盤狀態機：ratio > ratio_off 進入崩盤、ratio < ratio_on 解除；NaN 不改變狀態。
in_regime = close > EMA(ema_len) and not crash。RiskParams 預設（滿倉、無停損、無波動目標、無回撤暫停）。
"""
import numpy as np
import pandas as pd

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams

NAME = "EMA50 + 下行波動狀態閘"
DESCRIPTION = "close > EMA50 且 20 日下行/上行半變異比未進入崩盤狀態（>1.5 進、<1.0 出）才滿倉；無風控"
PARAMS = {"ema_len": 50, "vol_window": 20, "ratio_off": 1.5, "ratio_on": 1.0}


def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    n = len(c)
    w = int(p["vol_window"])
    ema = ta.ema(c, int(p["ema_len"]))
    trend = c > ema

    # 對數報酬：r[i] 只用 close[i-1] 與 close[i]；r[0] = NaN
    r = np.log(c / ta.shift(c, 1))
    dn2 = np.where(r < 0, r * r, 0.0)
    up2 = np.where(r > 0, r * r, 0.0)
    dn2[np.isnan(r)] = np.nan
    up2[np.isnan(r)] = np.nan
    # rolling(w).mean() 為右對齊窗（i-w+1 .. i），前 w-1 根（含 r[0] NaN）為 NaN
    dvol = np.sqrt(pd.Series(dn2).rolling(w).mean().to_numpy())
    uvol = np.sqrt(pd.Series(up2).rolling(w).mean().to_numpy())
    ratio = dvol / np.maximum(uvol, 1e-12)

    # 崩盤狀態機：只往前遞推
    crash = np.zeros(n, dtype=bool)
    state = False
    off, on = float(p["ratio_off"]), float(p["ratio_on"])
    for i in range(n):
        x = ratio[i]
        if not np.isnan(x):
            if not state and x > off:
                state = True
            elif state and x < on:
                state = False
        crash[i] = state

    in_regime = trend & ~crash

    # warmup：EMA 與 ratio 都有效的第一根（ratio 第一個非 NaN 在索引 w，因 r[0] 為 NaN）
    ema_first = int(np.argmax(~np.isnan(ema)))
    ratio_first = int(np.argmax(~np.isnan(ratio)))
    warmup = max(ema_first, ratio_first)
    return {"in_regime": np.asarray(in_regime, dtype=bool), "risk": RiskParams(), "warmup": warmup}
