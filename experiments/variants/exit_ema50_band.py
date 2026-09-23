"""EMA50 ATR 緩衝帶（Schmitt 觸發）：close > EMA50+k×ATR 進、close < EMA50−k×ATR 出，帶內維持前一狀態。

只改 in_regime 的判定，隔離「換手控制」單一假設；風控全預設（無停損、無波動目標、無回撤暫停）。
每根收盤判定，EMA / ATR 在第 i 根只用 ≤ i 根的資料；狀態機由前往後遞推，無前視。
"""
import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams

NAME = "EMA50 ATR 緩衝帶"
DESCRIPTION = "close > EMA50+0.5×ATR 進、close < EMA50−0.5×ATR 出；遲滯帶降低鋸齒；無風控"
PARAMS = {"ema_len": 50, "band_atr": 0.5, "atr_len": 14}


def build(df, params):
    c = df["close"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    e = ta.ema(c, params["ema_len"])              # Pine 語意，前 n-1 根 NaN
    a = ta.atr(h, l, c, params["atr_len"])         # RMA(TR)
    k = params["band_atr"]
    upper = e + k * a
    lower = e - k * a
    n = len(c)
    out = np.zeros(n, dtype=bool)
    on = False
    for i in range(n):
        if np.isnan(e[i]) or np.isnan(a[i]):
            out[i] = False
            continue
        if (not on) and c[i] > upper[i]:
            on = True                                # 進場需站上上緣
        elif on and c[i] < lower[i]:
            on = False                               # 出場需跌破下緣
        # 介於 lower..upper 之間：維持前一狀態（遲滯）
        out[i] = on
    valid = ~np.isnan(e) & ~np.isnan(a)
    warmup = int(np.argmax(valid)) if valid.any() else n
    return {"in_regime": out, "risk": RiskParams(), "warmup": warmup}
