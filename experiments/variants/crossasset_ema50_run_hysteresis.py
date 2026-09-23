"""EMA50 單日免疫遲滯（時間域 Schmitt 觸發）：連續 n_in 根收盤 > EMA50 才進、連續 n_out 根收盤 < EMA50 才出。

設計：把基準 ema50_plain 的 close vs EMA50 判定加上「時間域遲滯」——單日對側收盤（一天的假突破 / 一天的洗盤）
不改變狀態，要連續 n 根站在同一側才換狀態。跟價格域的 ATR 通道不同，這個閾值不依賴各標的的波動幅度，
所以理論上可以跨標的搬（SOL/BNB holdout 是它的主要考題）。只改 in_regime 判定；風控全預設。

無前視：EMA 在第 i 根只用 ≤ i 根的收盤；run 長度是「以第 i 根結尾的連續根數」，只用 ≤ i；
狀態機交給引擎（regime.state_machine），成交在 i+1 開盤，與基準相同。
恆等性：n_in = n_out = 1 時 in_regime 與 ema50_plain 逐根相同。
"""
import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams

NAME = "EMA50 單日免疫遲滯"
DESCRIPTION = "連續 2 根收盤 > EMA50 進、連續 2 根收盤 < EMA50 出；單日對側不改變狀態；無風控"
PARAMS = {"ema_len": 50, "n_in": 2, "n_out": 2}


def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    n = len(c)
    ema = ta.ema(c, int(p["ema_len"]))              # Pine 語意：前 n−1 根 NaN，SMA 起算後遞迴
    n_in, n_out = max(1, int(p["n_in"])), max(1, int(p["n_out"]))
    above = c > ema                                  # NaN 比較為 False（暖機期兩條 run 都是 0）
    below = c < ema                                  # c == ema 時兩者皆 False，兩條 run 都歸零
    idx = np.arange(n)
    # run_up[i] = 以第 i 根結尾、連續 above 的根數（above[i] 為 False 時為 0）；只用 ≤ i 的資料
    run_up = idx - np.maximum.accumulate(np.where(~above, idx, -1))
    run_dn = idx - np.maximum.accumulate(np.where(~below, idx, -1))
    enter = run_up >= n_in                           # 第 i 根收盤時，含本根已連續 n_in 根在 EMA 上方
    exit_ = run_dn >= n_out                          # 含本根已連續 n_out 根在 EMA 下方
    valid = ~np.isnan(ema)
    first = int(np.argmax(valid)) if valid.any() else n
    warmup = min(n, first + max(n_in, n_out) - 1)    # 第一根 run 有可能達標的索引
    enter[:warmup] = False
    exit_[:warmup] = False
    return {"in_regime": {"enter": enter, "exit": exit_}, "risk": RiskParams(), "warmup": warmup}
