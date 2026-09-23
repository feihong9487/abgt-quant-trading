"""EMA50 量能確認進場（q = 1.0）／ 出場同基準。

疊加對象：ema50_plain 的進場腿（close > EMA50）；出場腿不動（close ≤ EMA50 出）。
enter = close > EMA50 且 volume[i] ≥ q × median(volume[i−W .. i−1])（不含當根）
exit  = close ≤ EMA50（含 EMA NaN）
狀態機由引擎維護：趨勢外的每一根都重新評估 enter，所以「穿越當根量不足時之後任一根滿足即進」自動成立。
q = 0 時 enter = close > EMA50、exit = ~enter，狀態機逐位等於 ema50_plain（D1 對位檢查）。

無前視：ema 只用 ≤ i 的收盤；med 只用 i−W..i−1 的量；v[i] 為第 i 根收盤時已確定的量；引擎在 i+1 開盤成交。
風控：RiskParams 全部預設（無停損、無波動目標、無回撤暫停、無資金費率、無延遲），讓 Δ 全部歸因於進場腿。
warmup = max(L−1, W) = 49（與基準相同，ETH IS 起點維持 2020-01-15，不造成視窗錯位）。
"""
import numpy as np
import pandas as pd

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams

NAME = "EMA50 量能確認進場（q=1.0）/ EMA50 出場"
DESCRIPTION = "close > EMA50 且 volume ≥ q×median(volume[i−W..i−1]) 進；close ≤ EMA50 出；無風控"
PARAMS = {"ema_len": 50, "vol_win": 20, "q": 1.0}


def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    v = df["volume"].to_numpy(dtype="float64")      # Binance USDT-M 永續 volume 欄（合約數量，同標的內一致）
    L, W, q = int(p["ema_len"]), int(p["vol_win"]), float(p["q"])
    ema = ta.ema(c, L)                                # 前 L−1 根 NaN，SMA 起算後遞迴；第 i 根只用 ≤ i 收盤
    med = ta.shift(pd.Series(v).rolling(W).median().to_numpy(), 1)   # = median(v[i−W .. i−1])，不含當根；前 W 根 NaN
    safe = np.where(med > 0, med, np.nan)             # med ≤ 0 / NaN → NaN（比較為 False → 不進）
    with np.errstate(invalid="ignore", divide="ignore"):
        vr = v / safe                                 # 量能比值，無量綱
    above = c > ema                                   # NaN → False
    enter = above & (vr >= q)
    exit_ = ~above                                    # close ≤ EMA50（含 EMA NaN）
    warmup = max(L - 1, W)                            # 預設 49 = 基準 warmup
    return {"in_regime": {"enter": enter, "exit": exit_}, "risk": RiskParams(), "warmup": warmup}
