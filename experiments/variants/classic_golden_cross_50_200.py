"""黃金交叉：SMA50 > SMA200 持倉（Brock-Lakonishok-LeBaron 1992 VMA 家族／實務黃金交叉）。"""
import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams

NAME = "黃金交叉 SMA50/200"
DESCRIPTION = "SMA50 > SMA200 滿倉，否則空手；逐日判定；無停損無波動目標"
PARAMS = {"fast_len": 50, "slow_len": 200}


def build(df, params):
    c = df["close"].to_numpy()
    fast = ta.sma(c, int(params["fast_len"]))     # rolling(n).mean()：視窗含當根、不含之後，前 n-1 根 NaN
    slow = ta.sma(c, int(params["slow_len"]))
    on = fast > slow                              # NaN 比較為 False（與 Pine 語意一致）
    warmup = int(np.argmax(~np.isnan(slow) & ~np.isnan(fast)))
    rp = RiskParams(max_exposure=1.0, vol_target=0.0, atr_stop_mult=0.0, dd_pause=0.0)
    return {"in_regime": on, "risk": rp, "warmup": warmup}
