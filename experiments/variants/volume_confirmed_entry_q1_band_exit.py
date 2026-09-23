"""量能確認進場（q = 1.0）+ EMA50 − 0.25 ATR 出場。

疊加對象：vol_atr_channel 的出場腿（close < EMA50 − k_out×ATR14）原封不動；
進場腿由「close > EMA50 + k_in×ATR」換成「close > EMA50 且 vr ≥ q」，
vr = 當根 volume / median(volume[i−W..i−1])（不含當根）。不是 OR 腿，是完整替換。
enter 與 exit 互斥（enter 要 c > ema，exit 要 c < ema − k_out×atr，atr ≥ 0），狀態機由引擎維護；
「穿越當根量不足時之後任一根滿足即進」自動成立。
無前視：ema / atr 因果遞迴（TR 用 close[i−1]）；med 只用 i−W..i−1；引擎在 i+1 開盤成交。
q = 0 時退化為「plain 進場 + −0.25 ATR 出場」（消融對照 ablation）。
RiskParams 全部預設（無停損、無波動目標、無回撤暫停、費 0.05%/邊、滑點 0.02%）。
"""
import numpy as np
import pandas as pd

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams

NAME = "量能確認進場（q=1.0）+ EMA50 − 0.25 ATR 出場"
DESCRIPTION = "close > EMA50 且 volume ≥ q×median(volume[i−W..i−1]) 進；close < EMA50 − k_out×ATR14 出，其間維持；無風控"
PARAMS = {"ema_len": 50, "vol_win": 20, "q": 1.0, "atr_len": 14, "k_out": 0.25}


def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    v = df["volume"].to_numpy(dtype="float64")
    L, W, q = int(p["ema_len"]), int(p["vol_win"]), float(p["q"])
    A, k_out = int(p["atr_len"]), float(p["k_out"])
    ema = ta.ema(c, L)                                # 前 L−1 根 NaN
    atr = ta.atr(h, l, c, A)                          # RMA(TR)，TR 用 close[i−1]；前 A−1 根 NaN
    med = ta.shift(pd.Series(v).rolling(W).median().to_numpy(), 1)   # median(v[i−W..i−1])，不含當根
    safe = np.where(med > 0, med, np.nan)
    vr = v / safe
    enter = (c > ema) & (vr >= q)                     # NaN 比較 → False
    exit_ = c < ema - k_out * atr                     # 與 vol_atr_channel 出場逐位相同；NaN → False
    valid = ~np.isnan(ema) & ~np.isnan(atr)
    warmup = max(int(np.argmax(valid)) if valid.any() else len(c), W)   # 預設 max(49, 20) = 49
    return {"in_regime": {"enter": enter, "exit": exit_}, "risk": RiskParams(), "warmup": warmup}
