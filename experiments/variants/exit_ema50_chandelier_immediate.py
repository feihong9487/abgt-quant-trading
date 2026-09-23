"""EMA50 曝險 + 吊燈式 ATR 追蹤停損 + 立即重進。

與 ema50_atr 的唯一差異是 reentry="immediate"：停損後若該根收盤仍 close > EMA50，
下一根開盤就重進（hi_close / stop 從重進那根收盤重新起算），不用等趨勢先轉空再轉多。
in_regime 與基準 ema50_plain 逐根相同，停損參數與 ema50_atr 相同，差異可完整歸因於重進規則。
"""
import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams, regime_close_over_ema

NAME = "EMA50 + 3×ATR 吊燈停損 + 立即重進"
DESCRIPTION = "close > EMA50 滿倉；最高收盤 − 3×ATR(14) 追蹤停損；停損後若仍 close > EMA50 下一根開盤立即重進"
PARAMS = {"ema_len": 50, "atr_stop_mult": 3.0, "atr_len": 14}


def build(df, params):
    n = params["ema_len"]
    on = regime_close_over_ema(df, n)              # c > ema(c, n)，NaN 比較 = False
    warmup = int(np.argmax(~np.isnan(ta.ema(df["close"].to_numpy(), n))))
    rp = RiskParams(max_exposure=1.0, vol_target=0.0,
                    atr_stop_mult=params["atr_stop_mult"], atr_len=params["atr_len"],
                    reentry="immediate", dd_pause=0.0)
    return {"in_regime": on, "risk": rp, "warmup": warmup}
