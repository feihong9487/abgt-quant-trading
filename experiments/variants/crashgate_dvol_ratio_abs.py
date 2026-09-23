"""EMA50 + 下行波動崩盤閘（比值 × 絕對門檻 × 價格收復解除）。

疊加對象：基準 ema50_plain 的 trend = close > EMA50（不是通道版）。只改 in_regime，RiskParams 全預設。

指標（第 i 根只用 ≤ i 根收盤，全部右對齊）：
- r = ln(close / close[1])；dn2 = r² if r < 0 else 0；up2 = r² if r > 0 else 0
- dvol / uvol = sqrt(mean(dn2 / up2, w)) × sqrt(365)（年化，窗 i-w+1..i）
- ratio = dvol / max(uvol, 1e-12)（年化係數抵銷）
- ref = median(dvol, ref_len, min_periods=REF_MIN_PERIODS)（含第 i 根、不含之後）
- trig   = ratio > ratio_trig 且 dvol > abs_trig × ref（兩者皆要）
- normal = ratio < ratio_rel 或 dvol < abs_rel × ref（任一回常態即解除）

崩盤狀態機（只往前遞推）：
- 旗標中：波動回常態 → 解除並重新武裝；否則若 PRICE_RELEASE 且 close > 旗標日收盤 → 解除但記住旗標價 lfc
- 未旗標：波動回常態 → 重新武裝（lfc = NaN）；trig 且（已武裝 或 close < lfc）→ 旗標、記旗標價
- NaN 根狀態不變（同 vol_downside_gate）
in_regime = trend & ~crash。

約束：ratio_rel ≤ ratio_trig、abs_rel ≤ abs_trig（build 內把解除門檻夾到觸發門檻，等號時遲滯帶為 0 但仍互斥）。
無前視：EMA 遞迴、r 用 close[i-1]、rolling 皆右對齊含當根；引擎在 i+1 開盤成交。
"""
import numpy as np
import pandas as pd

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams

NAME = "EMA50 + 下行波動崩盤閘（比值×絕對門檻×價格解除）"
DESCRIPTION = ("close > EMA50 且未在崩盤旗標中才滿倉；旗標 = 20 日下行/上行半變異比 > 1.5 且下行波動 > 250 日中位數 × 1.25，"
               "比值 < 1.0 或下行波動 < 中位數解除，或收盤收復旗標日收盤解除（之後要再收在旗標價以下才可重觸發）；無風控")
PARAMS = {"ema_len": 50, "vol_window": 20, "ratio_trig": 1.5, "ratio_rel": 1.0, "abs_trig": 1.25, "abs_rel": 1.0, "ref_len": 250}
REF_MIN_PERIODS = 60      # 模組常數，不進 PARAMS；= 引擎 _realized_vol 的 max(3×vol_lookback, 30)
PRICE_RELEASE = True      # 模組常數；False = 消融「只有絕對門檻、無價格解除」
BARS_PER_YEAR = 365.0


def compute_indicators(df, p):
    """回傳 dict(ema, trend, dvol, uvol, ratio, ref, trig, normal, valid)。全部第 i 根只用 ≤ i 根收盤。"""
    c = df["close"].to_numpy(dtype="float64")
    w = int(p["vol_window"])
    L = int(p["ref_len"])
    ratio_trig = float(p["ratio_trig"])
    ratio_rel = min(float(p["ratio_rel"]), ratio_trig)      # 約束：解除 ≤ 觸發
    abs_trig = float(p["abs_trig"])
    abs_rel = min(float(p["abs_rel"]), abs_trig)

    ema = ta.ema(c, int(p["ema_len"]))                     # Pine 語意，前 ema_len-1 根 NaN
    trend = c > ema                                          # NaN 比較 → False

    r = np.log(c / ta.shift(c, 1))                           # r[0] = NaN
    dn2 = np.where(r < 0, r * r, 0.0)
    up2 = np.where(r > 0, r * r, 0.0)
    dn2[np.isnan(r)] = np.nan
    up2[np.isnan(r)] = np.nan
    dvol = np.sqrt(pd.Series(dn2).rolling(w).mean().to_numpy()) * np.sqrt(BARS_PER_YEAR)   # 窗 i-w+1..i
    uvol = np.sqrt(pd.Series(up2).rolling(w).mean().to_numpy()) * np.sqrt(BARS_PER_YEAR)
    ratio = dvol / np.maximum(uvol, 1e-12)
    ref = pd.Series(dvol).rolling(L, min_periods=REF_MIN_PERIODS).median().to_numpy()      # 窗 i-L+1..i 含當根

    with np.errstate(invalid="ignore"):
        trig = (ratio > ratio_trig) & (dvol > abs_trig * ref)
        normal = (ratio < ratio_rel) | (dvol < abs_rel * ref)
    valid = ~np.isnan(ratio) & ~np.isnan(ref)
    return {"close": c, "ema": ema, "trend": trend, "dvol": dvol, "uvol": uvol, "ratio": ratio, "ref": ref,
            "trig": trig, "normal": normal, "valid": valid}


def crash_flag(c, trig, normal, valid, price_release=True):
    """崩盤狀態機。回傳 (crash 布林陣列, 事件列表 [(i, 'flag'|'release_vol'|'release_price', close)])。"""
    n = len(c)
    crash = np.zeros(n, dtype=bool)
    events = []
    state = False
    flag_close = np.nan
    lfc = np.nan                                             # 價格解除後記住的旗標價；NaN = 已重新武裝
    for i in range(n):
        if valid[i]:
            if state:
                if normal[i]:
                    state, lfc = False, np.nan               # 波動回常態：解除並重新武裝
                    events.append((i, "release_vol", c[i]))
                elif price_release and c[i] > flag_close:
                    state, lfc = False, flag_close           # 價格收復旗標價：解除，但記住旗標價
                    events.append((i, "release_price", c[i]))
            else:
                if normal[i]:
                    lfc = np.nan                             # 重新武裝
                if trig[i] and (np.isnan(lfc) or c[i] < lfc):   # 價格解除後要再收在旗標價以下才准重觸發
                    state, flag_close = True, c[i]
                    events.append((i, "flag", c[i]))
        crash[i] = state                                     # NaN 根狀態不變
    return crash, events


def build(df, p):
    ind = compute_indicators(df, p)
    crash, _ = crash_flag(ind["close"], ind["trig"], ind["normal"], ind["valid"], PRICE_RELEASE)
    in_regime = ind["trend"] & ~crash
    ema_ok = ~np.isnan(ind["ema"])
    ref_ok = ind["valid"]
    n = len(ind["close"])
    ema_first = int(np.argmax(ema_ok)) if ema_ok.any() else n
    ref_first = int(np.argmax(ref_ok)) if ref_ok.any() else n
    warmup = max(ema_first, ref_first)
    return {"in_regime": np.asarray(in_regime, dtype=bool), "risk": RiskParams(), "warmup": warmup}
