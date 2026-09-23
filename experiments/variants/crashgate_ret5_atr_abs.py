"""EMA50 曝險 + 報酬衝擊崩盤閘：m 日報酬跌破 −k_trig × 衝擊前 ATR%（附絕對下限）進入崩盤、報酬回到 −k_rel × scale 以內或收盤收復旗標價解除。

疊加對象：基準 ema50_plain 的 trend = close > EMA50。只改 in_regime，RiskParams 全預設。

指標（第 i 根只用 ≤ i 根 OHLC）：
- atrp     = ATR(atr_len) / close                     ATR 佔價比（RMA(TR)，TR 用 close[i−1]）
- atrp_pre = atrp[i − m]                              衝擊前 m 根的日常波幅（避免崩盤自己把門檻抬高）
- ref      = median(atrp, ref_len, min_periods=60)    右對齊含第 i 根；絕對下限（自身一年中位數）
- scale    = max(atrp_pre, ref)                       NaN 傳播（np.maximum，不用 fmax）
- ret      = close / close[i − m] − 1                 m 日簡單報酬
- trig     = ret < −k_trig × scale；normal = ret > −k_rel × scale
崩盤狀態機（與 crashgate_dvol_ratio_abs 相同骨架）：
- 崩盤中：normal → 解除並重新武裝；否則 PRICE_RELEASE 且 close > 旗標價 → 解除、記住旗標價（lfc）
- 非崩盤：normal → 清除 lfc（重新武裝）；trig 且（lfc 為 NaN 或 close < lfc）→ 進入崩盤、旗標價 = close
  價格解除後只有再收在旗標價以下才准重觸發，避免「解除→隔天重旗標→再解除」的兩日來回。
in_regime = trend & ~crash。
無前視：ATR 遞迴、shift 只往後看、rolling 右對齊含當根、ret 用 close[i−m]；狀態機由前往後遞推；引擎在 i+1 開盤成交。
約束：k_rel < k_trig（同根不會既觸發又解除）。
"""
import numpy as np
import pandas as pd

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams

NAME = "EMA50 + 報酬衝擊崩盤閘（5 日報酬 vs ATR%，絕對下限）"
DESCRIPTION = ("close > EMA50 且未處於崩盤狀態才滿倉；5 日報酬 < −3.0 × max(5 日前 ATR20%, 250 日 ATR% 中位數) 進入崩盤，"
               "報酬回到 −1.0 × scale 以內或收盤收復旗標價解除；價格解除後需再收在旗標價以下才可重觸發；無風控")
PARAMS = {"ema_len": 50, "ret_len": 5, "atr_len": 20, "k_trig": 3.0, "k_rel": 1.0, "ref_len": 250}
REF_MIN_PERIODS = 60      # 模組常數，不進 PARAMS：ref 中位數的最少樣本數
PRICE_RELEASE = True      # 模組常數，False 可消融價格收復解除


def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    n = len(c)
    m = int(p["ret_len"])
    L = int(p["ref_len"])
    k_trig = float(p["k_trig"])
    k_rel = float(p["k_rel"])
    if not k_rel < k_trig:
        raise ValueError(f"約束 k_rel < k_trig 不成立：k_rel={k_rel}, k_trig={k_trig}")

    ema = ta.ema(c, int(p["ema_len"]))              # Pine 語意：前 n−1 根 NaN，SMA 起算後遞迴
    trend = c > ema                                  # NaN 比較為 False

    atrp = ta.atr(h, l, c, int(p["atr_len"])) / c    # ATR 佔價比；第一個有效值 i = atr_len − 1
    atrp_pre = ta.shift(atrp, m)                     # 衝擊前 m 根的日常波幅；有效值 i ≥ atr_len − 1 + m
    ref = pd.Series(atrp).rolling(L, min_periods=REF_MIN_PERIODS).median().to_numpy()   # 右對齊含第 i 根
    scale = np.maximum(atrp_pre, ref)                # NaN 傳播：任一邊 NaN 則 scale NaN
    ret = c / ta.shift(c, m) - 1.0                   # m 日簡單報酬，ret[i] 用 c[i−m]

    with np.errstate(invalid="ignore"):
        trig = ret < -k_trig * scale
        normal = ret > -k_rel * scale
    valid = ~np.isnan(ret) & ~np.isnan(scale)

    # 崩盤狀態機：只往前遞推
    crash = np.zeros(n, dtype=bool)
    state = False
    flag_close = np.nan     # 進入崩盤那根的收盤（旗標價）
    lfc = np.nan            # 價格解除後記住的旗標價；NaN = 已武裝
    for i in range(n):
        if valid[i]:
            if state:
                if normal[i]:
                    state, lfc = False, np.nan                 # 衝擊被吸收：解除並重新武裝
                elif PRICE_RELEASE and c[i] > flag_close:
                    state, lfc = False, flag_close             # 價格收復旗標價：解除，記住旗標價
            else:
                if normal[i]:
                    lfc = np.nan                               # 重新武裝
                if trig[i] and (np.isnan(lfc) or c[i] < lfc):  # 價格解除後只有再創旗標價以下的收盤才重觸發
                    state, flag_close = True, c[i]
        crash[i] = state

    in_regime = trend & ~crash

    # warmup：EMA、ref、atrp_pre（以及 ret）都有效的第一根
    ema_first = int(np.argmax(~np.isnan(ema))) if (~np.isnan(ema)).any() else n
    valid_first = int(np.argmax(valid)) if valid.any() else n
    warmup = max(ema_first, valid_first)
    return {"in_regime": np.asarray(in_regime, dtype=bool), "risk": RiskParams(), "warmup": warmup}
