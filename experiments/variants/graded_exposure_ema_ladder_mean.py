"""EMA 長度分散平均曝險：三條不同長度的 EMA（36 / 50 / 70）各自做 plain 規則（close > EMA_n），曝險 = 站上幾條 / 3。

設計（round 2 §11 方向 5(b)）：對 ema_len 做「貝氏平均」而不是點估計 —— 等權平均三個不同長度的 plain 策略，
報酬序列 ≈ 三腿報酬的平均，Sharpe 對 ema_len 的曲面被平滑；部位是分數曝險（0 / ⅓ / ⅔ / 1），
與 round 1 的 ensemble_ema_ladder_vote（二元投票，等價 close > 第 K 低 EMA）本質不同。
腿的長度由單一 ema_ratio 決定：lens = {round(L/r), L, round(L×r)}，r → 1 時退化為 [L] = ema50_plain（巢狀空假設）。

引擎：in_regime = {enter: scale > 0, exit: scale == 0, scale}；on = state_machine(enter, exit) 逐根等於 scale > 0；
趨勢內 target = scale[i]，階梯步幅 ⅓ > rebalance_threshold 0.15 所以每次變級都在次日開盤成交。
三腿未全就緒的根只平均已就緒的腿（ETH 快取 2019-11-27 起，warmup 沿用基準的 L−1 = 49，
ETH IS 前 20 根 2020-01-15 → 02-03 是 EMA36 + EMA50 兩腿版，之後三腿）。
無停損、無波動目標、無回撤暫停。

無前視：每條 EMA 在第 i 根只用 ≤ i 根的收盤（Pine 語意，SMA 起算後遞迴）；逐根平均無狀態；
截斷測試 build(df[:k])["scale"] 必須等於 build(df)["scale"][:k]。
"""
import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams

NAME = "EMA 長度分散平均曝險"
DESCRIPTION = "曝險 = mean(close > EMA_n for n in {36, 50, 70})，四級 0/⅓/⅔/1；in_regime = 曝險 > 0；調倉門檻 0.15；無停損無波動目標"
PARAMS = {"ema_len": 50, "ema_ratio": 1.4, "rebalance_threshold": 0.15}


def leg_lengths(p) -> list[int]:
    L = int(p["ema_len"])
    r = float(p["ema_ratio"])
    return sorted(set([max(1, int(round(L / r))), L, max(1, int(round(L * r)))]))   # [36, 50, 70]；r→1 時 [50]


def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    n = len(c)
    L = int(p["ema_len"])
    lens = leg_lengths(p)
    legs = []
    for m in lens:
        e = ta.ema(c, m)                                                    # 前 m−1 根 NaN，只用 ≤ i 根收盤
        legs.append(np.where(np.isnan(e), np.nan, (c > e).astype(np.float64)))   # 1.0 / 0.0 / NaN（未就緒）
    A = np.vstack(legs)                                                     # k × n
    ready = ~np.isnan(A)
    cnt = ready.sum(axis=0)
    with np.errstate(invalid="ignore"):
        scale = np.where(cnt > 0, np.nansum(A, axis=0) / np.maximum(cnt, 1), 0.0)   # 只平均已就緒的腿
    scale = np.clip(scale, 0.0, 1.0)
    enter = scale > 0
    exit_ = scale == 0                                                      # 與 enter 互斥 → on 逐根等於 scale > 0
    warmup = max(0, min(L - 1, n))                                          # 49：與 ema50_plain 同視窗
    rp = RiskParams(max_exposure=1.0, vol_target=0.0, atr_stop_mult=0.0, dd_pause=0.0,
                    reentry="immediate", rebalance_threshold=float(p["rebalance_threshold"]))
    return {"in_regime": {"enter": enter, "exit": exit_, "scale": scale}, "risk": rp, "warmup": warmup}
