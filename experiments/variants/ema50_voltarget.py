"""EMA50 曝險 + 波動目標倉位（年化 60%）+ 3×ATR 停損 + 回撤 25% 暫停。風控全開的版本。"""
import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams, regime_close_over_ema

NAME = "EMA50 + 波動目標 60% + 停損 + 回撤暫停"
DESCRIPTION = "close > EMA50；曝險 = min(1, 60% / 已實現年化波動)；3×ATR 吊燈停損；權益回撤 25% 暫停到趨勢重置"
PARAMS = {"ema_len": 50, "vol_target": 0.60, "vol_lookback": 20, "atr_stop_mult": 3.0, "dd_pause": 0.25}


def build(df, params):
    n = params["ema_len"]
    on = regime_close_over_ema(df, n)
    warmup = int(np.argmax(~np.isnan(ta.ema(df["close"].to_numpy(), n))))
    rp = RiskParams(vol_target=params["vol_target"], vol_lookback=params["vol_lookback"],
                    atr_stop_mult=params["atr_stop_mult"], dd_pause=params["dd_pause"], reentry="reset")
    return {"in_regime": on, "risk": rp, "warmup": warmup}
