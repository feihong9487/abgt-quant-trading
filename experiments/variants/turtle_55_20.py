"""海龜式：突破 55 日高進場、跌破 20 日低出場，加 2×ATR 吊燈停損。"""
import numpy as np

from abgt.strategies.regime import RiskParams, regime_donchian

NAME = "海龜 55/20 + 2×ATR"
DESCRIPTION = "close 突破 55 日高進、跌破 20 日低出；最高收盤 − 2×ATR 追蹤停損"
PARAMS = {"entry_len": 55, "exit_len": 20, "atr_stop_mult": 2.0}


def build(df, params):
    on = regime_donchian(df, params["entry_len"], params["exit_len"])
    rp = RiskParams(atr_stop_mult=params["atr_stop_mult"], reentry="reset")
    return {"in_regime": on, "risk": rp, "warmup": int(params["entry_len"]) + 1}
