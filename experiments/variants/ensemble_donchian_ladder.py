"""Donchian 階梯多數決：20/55/100 三階突破狀態機，出場長度 = 進場長度一半，三票至少兩票才在趨勢內。

每階都是 regime.py 內建的 regime_donchian（海龜式狀態機）：
    hi[i] = max(close[i-n_in .. i-1])、lo[i] = min(close[i-n_out .. i-1])   ← 只用過去資料
    close > hi 開、close < lo 關；hi/lo 為 NaN 時比較為 False → 狀態機保持關
三階：20/10、55/28、100/50。多數決（k_min=2）：
    - 100/50 開（成熟牛市）：55/28 決定 → 跌破 28 日低出場，20 日新高即重進
    - 100/50 關（熊市或崩後）：20/10 與 55/28 都要開 → 進場 = 55 日新高，出場 = 10 日新低（較緊）
風控全部預設：滿倉 0/1、無吊燈停損、無波動目標、無回撤暫停。
"""
import numpy as np

from abgt.strategies.regime import RiskParams, regime_donchian

NAME = "Donchian 階梯多數決 20/55/100"
DESCRIPTION = "三階 Donchian 突破狀態機（出場 = 進場一半：20/10、55/28、100/50），三票至少兩票開就滿倉；無風控"
PARAMS = {"entry_1": 20, "entry_2": 55, "entry_3": 100, "exit_ratio": 0.5, "k_min": 2}


def build(df, p):
    states = []
    for key in ("entry_1", "entry_2", "entry_3"):
        n_in = int(p[key])
        n_out = max(2, int(round(n_in * p["exit_ratio"])))   # 20→10、55→28、100→50
        states.append(regime_donchian(df, n_in, n_out))
    votes = np.stack(states, axis=1)
    in_regime = votes.sum(axis=1) >= int(p["k_min"])
    warmup = int(p["entry_2"]) + 1                            # 56，與 turtle_55_20 相同；100 階在第 101 根前恆為關
    return {"in_regime": np.asarray(in_regime, dtype=bool), "risk": RiskParams(), "warmup": warmup}
