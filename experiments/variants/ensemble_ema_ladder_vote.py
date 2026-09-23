"""EMA 階梯半數決：close 高於階梯 25/50/100/200 中至少半數的 EMA 才算趨勢內（自適應門檻）。

等價封閉式：in_regime[i] ⇔ close[i] > 第 k_min 低的 EMA 值（NaN 視為 +∞）。
- 熊市排列 EMA25<EMA50<EMA100<EMA200：第 2 低 = EMA50 → 進出場都在 EMA50（同基準 ema50_plain）。
- 多頭排列 EMA200<EMA100<EMA50<EMA25：第 2 低 = EMA100 → 容忍回檔到 EMA100 才出場。
長階 EMA 尚未有效（NaN）時比較為 False = 反對票（Pine 語意），不用未來資料。
暖機取第二階（EMA50）有效的第一根 = 49，與 ema50_plain 相同窗口。
"""
import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams

NAME = "EMA 階梯半數決"
DESCRIPTION = "close > 至少半數的 EMA(25/50/100/200) 滿倉，否則空手；無風控"
PARAMS = {"base_len": 25, "ratio": 2.0, "n_rungs": 4, "vote_frac": 0.5}


def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    N = max(1, int(p["n_rungs"]))
    rungs = [max(2, int(round(p["base_len"] * p["ratio"] ** k))) for k in range(N)]   # 預設 [25, 50, 100, 200]
    emas = [ta.ema(c, L) for L in rungs]
    # 每根 i 的票：close[i] > EMA_L[i]；EMA_L[i] 只用 ≤ i 根的收盤，NaN 比較為 False（反對票）
    votes = np.stack([c > e for e in emas], axis=1)
    k_min = max(1, int(np.ceil(N * p["vote_frac"])))                                  # 4 × 0.5 → 2
    in_regime = votes.sum(axis=1) >= k_min
    # 暖機：第二階（預設 EMA50）有效的第一根；階數不足 2 時取最後一階
    warm_rung = emas[min(1, N - 1)]
    warmup = int(np.argmax(~np.isnan(warm_rung)))
    return {"in_regime": np.asarray(in_regime, dtype=bool), "risk": RiskParams(), "warmup": warmup}
