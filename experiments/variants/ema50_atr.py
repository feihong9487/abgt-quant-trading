"""EMA50 曝險 + 吊燈式 ATR 追蹤停損（停損後等趨勢重置才重進）。"""
import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams, regime_close_over_ema

NAME = "EMA50 + 3×ATR 吊燈停損"
DESCRIPTION = "close > EMA50 滿倉；最高收盤 − 3×ATR(14) 追蹤停損；停損後等趨勢重置"
PARAMS = {"ema_len": 50, "atr_stop_mult": 3.0, "atr_len": 14}


def build(df, params):
    n = params["ema_len"]
    on = regime_close_over_ema(df, n)
    warmup = int(np.argmax(~np.isnan(ta.ema(df["close"].to_numpy(), n))))
    rp = RiskParams(atr_stop_mult=params["atr_stop_mult"], atr_len=params["atr_len"], reentry="reset")
    return {"in_regime": on, "risk": rp, "warmup": warmup}
