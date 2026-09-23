"""五家族多數決：水平／排列／突破／動能／APEX 強度，五票中 >= k_min 票才持倉。

五票刻意選五個不同機制（而非同一族的五個回看期，那只是中位數濾波）：
  1 level : close > EMA50                                   （冠軍基準）
  2 order : EMA20 > EMA50                                   （均線多頭排列，比票 1 慢 1–2 週，濾單根刺穿）
  3 break : Donchian 55 進 / 20 出 狀態機                   （海龜 S2；regime_donchian 內 hi/lo 已 shift(1)）
  4 mom   : close > close[90]                               （一季的時間序列動能）
  5 apex  : close > EMA200 且 EMA21 > EMA50 > EMA200 且 ADX(14,14) >= 28   （APEX v4.0 / R3 原規格，強趨勢票）

無前視：每票都只用第 i 根及之前的收盤資料；任一指標為 NaN 時該票為 False（Pine 的 na 比較語意）。
apex 票在 EMA200 有效前恆為 False，該期間等於「其餘四票要 3 票」；warmup 不含 apex_slow。
風控全部預設（滿倉 0/1，無停損、無波動目標、無回撤暫停），與基準同條件，差異可歸因於信號。
"""
import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams, adx_filter, regime_apex3, regime_donchian

NAME = "五家族多數決"
DESCRIPTION = "EMA50 水平 / EMA20>EMA50 排列 / Donchian 55-20 突破 / 90 日動能 / APEX 三線+ADX>=28，五票中 >= 3 票滿倉；無風控"
PARAMS = {"ema_len": 50, "ema_fast": 20, "don_entry": 55, "don_exit": 20, "mom_len": 90,
          "apex_fast": 21, "apex_mid": 50, "apex_slow": 200, "adx_len": 14, "adx_min": 28.0, "k_min": 3}


def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    e_mid = ta.ema(c, int(p["ema_len"]))
    e_fast = ta.ema(c, int(p["ema_fast"]))
    with np.errstate(invalid="ignore"):
        v_level = c > e_mid                                                    # 票 1：價格在 EMA50 上
        v_order = e_fast > e_mid                                               # 票 2：EMA20 > EMA50
        v_break = regime_donchian(df, int(p["don_entry"]), int(p["don_exit"]))  # 票 3：突破狀態機
        v_mom = c > ta.shift(c, int(p["mom_len"]))                             # 票 4：90 日時間序列動能
        v_apex = (regime_apex3(df, int(p["apex_fast"]), int(p["apex_mid"]), int(p["apex_slow"]))
                  & adx_filter(df, int(p["adx_len"]), float(p["adx_min"])))    # 票 5：APEX 三線 + ADX
    votes = np.stack([v_level, v_order, v_break, v_mom, v_apex], axis=1).astype(bool)   # NaN 比較 → False
    n_votes = votes.sum(axis=1)
    in_regime = n_votes >= int(p["k_min"])
    warmup = int(max(p["ema_len"], p["ema_fast"], p["don_entry"] + 1, p["mom_len"], 2 * p["adx_len"]))   # 不含 apex_slow
    return {"in_regime": in_regime, "risk": RiskParams(), "warmup": warmup,
            "votes": n_votes}   # 診斷用：票數分佈，評估器忽略
