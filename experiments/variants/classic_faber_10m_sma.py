"""Faber (2007) 10 月均線擇時：月末收盤 > 近 10 個月末收盤 SMA 才持倉；每月判定一次。"""
import numpy as np
import pandas as pd

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams

NAME = "Faber 10 月 SMA（月末判定）"
DESCRIPTION = "月末收盤 > 近 10 個月末收盤的 SMA → 下月滿倉，否則空手；每月只判定一次；無停損無波動目標"
PARAMS = {"sma_months": 10}


def build(df, params):
    m = int(params["sma_months"])
    c = df["close"].to_numpy()
    n = len(c)
    idx = df.index
    is_me = (idx.day == idx.days_in_month)            # 純日曆：本根是否為該月最後一天，不看未來 K 線
    me_pos = np.flatnonzero(is_me)
    me_close = c[me_pos]
    sma_me = ta.sma(me_close, m)                       # 只用月末收盤算 SMA；前 m-1 個月末為 NaN
    valid = ~np.isnan(sma_me)
    sig = np.full(n, np.nan)
    sig[me_pos[valid]] = (me_close[valid] > sma_me[valid]).astype(float)
    on = pd.Series(sig).ffill().fillna(0.0).to_numpy().astype(bool)   # 月末決定，持有到下一個月末
    warmup = int(me_pos[m - 1]) if len(me_pos) >= m else n
    rp = RiskParams(max_exposure=1.0, vol_target=0.0, atr_stop_mult=0.0, dd_pause=0.0)
    return {"in_regime": on, "risk": rp, "warmup": warmup}
