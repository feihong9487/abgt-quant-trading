"""EMA50 ± ATR 通道遲滯（不對稱）：close > EMA50 + k_in×ATR 進、close < EMA50 − k_out×ATR 出，通道內維持前一狀態。

設計：進場要確認（+1 ATR，濾掉熊市裡只探上 EMA50 幾個百分點的反彈），出場不拖延（−0.25 ATR，只吸收跌破一天就回來的假出場）。
只改 in_regime 的判定；風控全預設（無停損、無波動目標、無回撤暫停）。
無前視：EMA / ATR 在第 i 根只用 ≤ i 根的 OHLC（RMA 遞迴、TR 用 close[i−1]）；狀態機由前往後遞推；引擎在 i+1 開盤成交。
"""
import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams

NAME = "EMA50 ± ATR 通道遲滯"
DESCRIPTION = "close > EMA50 + 1.0×ATR14 進入趨勢、close < EMA50 − 0.25×ATR14 離開，其間維持；無風控"
PARAMS = {"ema_len": 50, "atr_len": 14, "k_in": 1.0, "k_out": 0.25}


def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    n = len(c)
    ema = ta.ema(c, int(p["ema_len"]))              # Pine 語意：前 n−1 根 NaN，SMA 起算後遞迴
    atr = ta.atr(h, l, c, int(p["atr_len"]))        # RMA(TR)，TR 用前一根收盤
    upper = ema + float(p["k_in"]) * atr
    lower = ema - float(p["k_out"]) * atr
    on = np.zeros(n, dtype=bool)
    state = False
    for i in range(n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            state = False                            # 暖機期一律趨勢外
        elif not state and c[i] > upper[i]:
            state = True                             # 進場：站上 EMA50 + k_in×ATR
        elif state and c[i] < lower[i]:
            state = False                            # 出場：跌破 EMA50 − k_out×ATR
        on[i] = state                                # 通道內維持前一根狀態
    valid = ~np.isnan(ema) & ~np.isnan(atr)
    warmup = int(np.argmax(valid)) if valid.any() else n
    return {"in_regime": on, "risk": RiskParams(), "warmup": warmup}
