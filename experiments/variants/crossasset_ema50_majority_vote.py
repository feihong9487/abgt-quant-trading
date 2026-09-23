"""EMA50 五日多數決：過去 window 根收盤中 ≥ min_above 根收在 EMA50 上方即在趨勢內，否則趨勢外。

設計：基準 ema50_plain 的鋸齒主要是 ≤ 2 天的短暫穿越（以天數而非幅度定義），3 of 5 多數決是最小的能忽略 2 日事件的規則；
同時忽略幅度很大的單日假穿越（價格帶做不到）並容忍交替雜訊（狀態機遇 A B A B 會來回翻）。
乾淨穿越的延遲是 2 根（計數 1,2,3 → 第 3 根判定，第 4 根開盤成交）。
只改 in_regime 的判定；風控全預設（無停損、無波動目標、無回撤暫停）。
無前視：cnt[i] = sum(1[c[i-j] > ema[i-j]], j=0..w-1)，每一項只用 ≤ i 根的收盤與當根 EMA；引擎在 i+1 開盤成交。
恆等性：window = min_above = 1 時與 ema50_plain 逐根相同。
"""
import numpy as np
import pandas as pd

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams

NAME = "EMA50 五日多數決"
DESCRIPTION = "過去 5 根收盤中 ≥ 3 根 > EMA50 即在趨勢內，否則趨勢外；無風控"
PARAMS = {"ema_len": 50, "window": 5, "min_above": 3}


def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    n = len(c)
    ema = ta.ema(c, int(p["ema_len"]))                  # Pine 語意：前 n−1 根 NaN，SMA 起算後遞迴
    w = max(1, int(p["window"]))
    k = min(w, max(1, int(p["min_above"])))             # 約束 1 ≤ k ≤ w，避免退化成永遠 False
    above = (c > ema).astype(np.int64)                  # NaN 比較為 False → 0
    cnt = pd.Series(above).rolling(w).sum().to_numpy()  # cnt[i] = sum(above[i-w+1 .. i])，只用 ≤ i
    on = np.where(np.isnan(cnt), False, cnt >= k)
    valid = ~np.isnan(ema)
    first = int(np.argmax(valid)) if valid.any() else n
    warmup = min(n, first + w - 1)                      # 視窗內 w 根 EMA 都有效後才起算
    on[:warmup] = False
    return {"in_regime": on.astype(bool), "risk": RiskParams(), "warmup": warmup}
