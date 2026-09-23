"""EMA50 曝險 + 收盤確認 3×ATR 吊燈停損 + 收盤回升重進（7 根冷卻）。

regime 與基準 ema50_plain 逐根相同（close > EMA50 滿倉、否則空手），只疊加 RiskParams 的停損區塊，
所以「變體 − 基準」全部歸因給 停損 + 重進規則：
- stop_trigger="close"：第 i 根收盤 c[i] < stop[i] 才觸發，i+1 開盤成交；不吃下影線（第 1 輪 ema50_atr 的 2020-11-26 元兇）。
- stop[i] = max(stop[i-1], hi_close − 3×ATR14[i])，hi_close 從進場成交那根收盤起算（棘輪，賣壓期 ATR 放大不放寬停損）。
- reentry="close_above_stop" + reentry_min_bars=7：停損後要「曾經收在觸發收盤價之上」且距停損 ≥ 7 根才重進；
  趨勢重置（close < EMA50）則清空門檻，之後照基準規則進場（冷卻仍算）。
無前視：EMA / ATR 第 i 根只用 ≤ i 根的 OHLC；停損比較用第 i 根收盤、成交在 i+1 開盤；stop_exit_px 是已收盤的值。
"""
import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams

NAME = "EMA50 + 收盤確認 3×ATR 吊燈停損 + 收盤回升重進（7 根冷卻）"
DESCRIPTION = "close > EMA50 滿倉；最高收盤 − 3×ATR14 追蹤停損（收盤跌破才觸發、下一根開盤出）；停損後收盤回到觸發收盤價之上且隔 ≥ 7 根才重進"
PARAMS = {"ema_len": 50, "atr_len": 14, "atr_stop_mult": 3.0, "reentry_min_bars": 7}


def build(df, p):
    n_ema = int(p["ema_len"])                 # 評估器對 int 參數用 max(1, int(round(v*f))) 擾動，仍加 int() 保險
    n_atr = int(p["atr_len"])
    k_stop = float(p["atr_stop_mult"])
    hold = int(round(p["reentry_min_bars"]))
    c = df["close"].to_numpy(dtype="float64")
    ema = ta.ema(c, n_ema)                    # Pine 語意：前 n−1 根 NaN，SMA 起算後遞迴；第 i 根只用 ≤ i 的收盤
    on = c > ema                              # NaN 比較 = False；與 ema50_plain 的 regime_close_over_ema 逐根相同
    valid = ~np.isnan(ema)
    warmup = int(np.argmax(valid)) if valid.any() else len(c)   # ATR14 暖機 13 根短於 EMA50 的 49 根，取 EMA
    rp = RiskParams(max_exposure=1.0, vol_target=0.0,
                    atr_len=n_atr, atr_stop_mult=k_stop, stop_trigger="close",
                    reentry="close_above_stop", reentry_min_bars=hold,
                    dd_pause=0.0)             # 其餘全預設：funding 0、delay 0、rebalance 0.10、手續費 0.05%、滑點 0.02%
    return {"in_regime": on, "risk": rp, "warmup": warmup}
