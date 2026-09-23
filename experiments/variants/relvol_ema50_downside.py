"""EMA50 + 相對下行波動預算（1.4× 自身一年中位數）。

趨勢判定與 ema50_plain 逐根相同（close > EMA50 = 趨勢內），只改 sizing：
- 引擎 vol_target_mode="relative"、vol_side="downside"：
    dvol[i] = sqrt(mean(min(r,0)^2 over 20 根)) × sqrt(365)      # 20 日下行半變異，只用負報酬日
    ref[i]  = median(dvol over 365 根，min_periods=60)            # 自身一年中位數（尺度無關）
    target  = min(1.0, 1.4 × ref / dvol)                          # dvol 高於自身中位數 1.4 倍才開始砍
- 調倉門檻 0.15 同時是遲滯帶：滿倉狀態下第一次砍倉要 dvol > 1.4/0.85 = 1.647×ref；
  從曝險 e 回補要 target > e + 0.15，再砍要 target < e − 0.15。
- ref 未就緒（資料起點後 79 根內）或 20 日內無負報酬 → 引擎自動退回滿倉 = 基準。
- 無停損、無回撤暫停；funding / delay / 費用用引擎預設（壓力情境由評估器 risk_override 覆蓋）。

與 vol_shock_budget（絕對 0.70 / 10 日總波動）的差別：
  (a) relative → 同一參數在 BTC/ETH、2020/2026 有相同語意，不是 IS 校準的絕對數；
  (b) downside → 牛市噴出段（上行波動）不砍倉，只對下跌波動反應。

無前視：EMA 只用 ≤ i 根收盤；引擎 dvol / ref 都是右對齊 rolling（含當根不含之後）；
目標曝險在第 i 根收盤決定、第 i+1 根開盤成交。in_regime 無狀態機、無路徑依賴。

固定常數不進 PARAMS（避免敏感度工具把 max_exposure 擾動成 1.25 的槓桿）：
max_exposure=1.0、atr_stop_mult=0、dd_pause=0、reentry="immediate"。
"""
import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams

NAME = "EMA50 + 相對下行波動預算 1.4×"
DESCRIPTION = "close > EMA50 趨勢內；曝險 = min(1, 1.4 × 20日下行半變異一年中位數 / 20日下行半變異)；調倉門檻 15%；無停損無回撤暫停"
PARAMS = {"ema_len": 50, "vol_target": 1.4, "vol_lookback": 20, "vol_ref_lookback": 365,
          "rebalance_threshold": 0.15, "vol_side": "downside", "vol_target_mode": "relative"}


def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    ema = ta.ema(c, int(p["ema_len"]))            # Pine 語意：前 ema_len−1 根 NaN，SMA 起算後遞迴，只用 ≤ i 根收盤
    in_regime = c > ema                            # NaN 比較 → False；與 ema50_plain 逐根相同
    warmup = int(np.argmax(~np.isnan(ema)))        # = ema_len − 1；與基準同視窗（ref 未就緒時引擎自動退回滿倉）
    rp = RiskParams(max_exposure=1.0,
                    vol_target=float(p["vol_target"]),
                    vol_target_mode=str(p["vol_target_mode"]),   # "relative"
                    vol_side=str(p["vol_side"]),                 # "downside"
                    vol_lookback=int(p["vol_lookback"]),
                    vol_ref_lookback=int(p["vol_ref_lookback"]),
                    atr_stop_mult=0.0, dd_pause=0.0, reentry="immediate",
                    rebalance_threshold=float(p["rebalance_threshold"]))
    return {"in_regime": np.asarray(in_regime, dtype=bool), "risk": rp, "warmup": warmup}
