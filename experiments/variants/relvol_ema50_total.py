"""EMA50 + 相對「總」波動預算（對照組：vol_side=total，主變體 relvol_ema50_downside 的唯一差異）。

目的：把 relvol_ema50_downside 的兩個假設拆開歸因 ——
  (a) 尺度無關的 ref（rvol 對自身 365 日中位數）治好 vol_shock_budget 的「IS 校準 + 2025 死碼 + 替換切分掛」
  (b) 只算下行波動治好「牛市噴出段被砍」
本檔只含 (a)：與主變體逐字相同，唯 vol_side 讀到 "total"。

引擎在第 i 根收盤實際算的：
  r[i]    = ln(c[i]/c[i-1])
  rvol[i] = std(r[i-19..i], ddof=1) × sqrt(365)          （總波動 = 上行 + 下行）
  ref[i]  = median(rvol[i-364..i])，min_periods 60 → 第一個有效值 index 79
  target  = 0（趨勢外）；1.0（ref/rvol 無效）；否則 min(1, 1.4 × ref / rvol)
  曝險差 > 0.15 才調倉，i+1 開盤成交。有效首砍點 rvol > 1.4×ref；曝險 < 0.85 需 rvol > 1.647×ref。

無前視：EMA 與引擎 rvol/ref 都只用 ≤ i 根收盤；目標在第 i 根收盤決定、第 i+1 根開盤成交。
"""
import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams

NAME = "EMA50 + 相對總波動預算 1.4×（對照：total）"
DESCRIPTION = "close > EMA50；曝險 = min(1, 1.4 × 365 日中位數(rvol20) / rvol20)，rvol 用總波動；調倉門檻 15%；無停損無回撤暫停"
PARAMS = {"ema_len": 50, "vol_target": 1.4, "vol_lookback": 20, "vol_ref_lookback": 365,
          "rebalance_threshold": 0.15, "vol_side": "total", "vol_target_mode": "relative"}


def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    ema = ta.ema(c, int(p["ema_len"]))          # 前 ema_len−1 根 NaN，只用 ≤ i 根收盤
    in_regime = c > ema                          # NaN 比較 → False
    warmup = int(np.argmax(~np.isnan(ema)))      # = ema_len − 1 = 49
    rp = RiskParams(max_exposure=1.0,
                    vol_target=float(p["vol_target"]),
                    vol_target_mode=str(p["vol_target_mode"]),
                    vol_side=str(p["vol_side"]),
                    vol_lookback=int(p["vol_lookback"]),
                    vol_ref_lookback=int(p["vol_ref_lookback"]),
                    atr_stop_mult=0.0, dd_pause=0.0, reentry="immediate",
                    rebalance_threshold=float(p["rebalance_threshold"]))
    return {"in_regime": np.asarray(in_regime, dtype=bool), "risk": rp, "warmup": warmup}
