"""時間序列動量 1/3/12 月（Moskowitz-Ooi-Pedersen 2012；Hurst-Ooi-Pedersen 2017 等權組合）：
月末判定，三個回看期的過去報酬有 >= 2 個為正就持倉到下個月末。

無前視：月末旗標只看本根時間戳（day == days_in_month）；每個回看期的報酬 r_L[k] 只用
第 k 個月末與第 k−L 個月末的收盤（ta.shift 往後移），訊號填在月末那根，ffill 到下一個月末，
引擎於下個月第一根開盤成交。
"""
import numpy as np
import pandas as pd

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams

NAME = "TSMOM 1/3/12 月多數決"
DESCRIPTION = "月末：1、3、12 個月報酬中 >= 2 個 > 0 → 下月滿倉，否則空手；無停損無波動目標"
PARAMS = {"lb_short": 1, "lb_mid": 3, "lb_long": 12}


def build(df, params):
    lbs = [int(params["lb_short"]), int(params["lb_mid"]), int(params["lb_long"])]
    c = df["close"].to_numpy(dtype="float64")
    n = len(c)
    idx = df.index
    is_me = np.asarray(idx.day == idx.days_in_month)   # 純日曆月末旗標，只看本根時間戳
    me_pos = np.flatnonzero(is_me)
    me_close = c[me_pos]
    votes = np.zeros(len(me_pos))
    valid = np.ones(len(me_pos), dtype=bool)
    for L in lbs:
        r = me_close / ta.shift(me_close, L) - 1.0      # 與 L 個月末前的收盤比；前 L 個為 NaN
        valid &= ~np.isnan(r)
        votes += (r > 0)
    sig = np.full(n, np.nan)
    sig[me_pos[valid]] = (votes[valid] * 2 > len(lbs)).astype(float)   # 多數決：3 個中 >= 2
    on = pd.Series(sig).ffill().fillna(0.0).to_numpy().astype(bool)
    L_max = max(lbs)
    warmup = int(me_pos[L_max]) if len(me_pos) > L_max else n   # 第一個三個回看期都有值的月末
    rp = RiskParams(max_exposure=1.0, vol_target=0.0, atr_stop_mult=0.0, dd_pause=0.0)
    return {"in_regime": on, "risk": rp, "warmup": warmup}
