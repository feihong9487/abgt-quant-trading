"""基準變體：close > EMA50 就 100% 持倉，否則空手。沒有停損、沒有波動目標。"""
import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams, regime_close_over_ema

NAME = "EMA50 純曝險"
DESCRIPTION = "close > EMA50 滿倉，否則空手；無風控"
PARAMS = {"ema_len": 50}


def build(df, params):
    n = params["ema_len"]
    on = regime_close_over_ema(df, n)
    warmup = int(np.argmax(~np.isnan(ta.ema(df["close"].to_numpy(), n))))
    return {"in_regime": on, "risk": RiskParams(), "warmup": warmup}
