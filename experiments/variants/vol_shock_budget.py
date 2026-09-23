"""EMA50 + 高目標短窗波動預算（震盪吸收器，無停損、無回撤暫停）。

隔離「波動目標」本身的效果：ema50_voltarget / ema50_atr 的傷害幾乎全來自 ATR 停損 + 等重置，
這個變體只保留波動目標，趨勢結束（close ≤ EMA50）才空手。
- 目標 0.70 ≈ BTC 趨勢內 20 日已實現年化波動 p65：正常牛市滿倉，只在波動爆發時砍
- 回看 10 日：降曝險 1–2 天到位、10 天後自動回復滿倉
- 調倉門檻 0.15：吸收 10 日波動估計的雜訊，換手極低

無前視：EMA 與引擎 rvol 都只用第 i 根含以前的收盤；目標曝險在第 i 根收盤決定、第 i+1 根開盤成交。
"""
import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams

NAME = "EMA50 + 波動預算 70%/10 日（無停損）"
DESCRIPTION = "close > EMA50；曝險 = min(1, 0.70 / 10 日已實現年化波動)；調倉門檻 15%；無停損無回撤暫停"
PARAMS = {"ema_len": 50, "vol_target": 0.70, "vol_lookback": 10, "rebalance_threshold": 0.15}


def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    ema = ta.ema(c, int(p["ema_len"]))          # 前 ema_len−1 根 NaN，只用 ≤ i 根收盤
    in_regime = c > ema                          # NaN 比較 → False
    warmup = int(np.argmax(~np.isnan(ema)))      # = ema_len − 1
    rp = RiskParams(max_exposure=1.0,
                    vol_target=float(p["vol_target"]),
                    vol_lookback=int(p["vol_lookback"]),
                    atr_stop_mult=0.0, dd_pause=0.0, reentry="immediate",
                    rebalance_threshold=float(p["rebalance_threshold"]))
    return {"in_regime": np.asarray(in_regime, dtype=bool), "risk": rp, "warmup": warmup}
