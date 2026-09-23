"""EMA50 ± ATR 通道 + 「連續 N 根收在 EMA50 上」快速進場（OR 腿）；出場腿與 vol_atr_channel 逐位相同。

進場：close > EMA50 + k_in×ATR（通道腿）**或** 連續 confirm_bars 根（含本根）收盤 > EMA50（時間確認腿）。
出場：close < EMA50 − k_out×ATR。通道內維持前一狀態（引擎 {enter, exit} 狀態機）。

設計：vol_atr_channel 的 +1 ATR 進場門檻在崩盤後 ATR 膨脹時要付 6–10% 的延遲成本；
時間確認腿只付 N−1 根的漂移（N=3 約 2–3%），讓「緩慢磨上去」的 V 型反轉（2020-04、2023-01）提早進場。
通道出場（c < ema − k_out×ATR）必然使 above=False → run 歸零，所以出場後重進至少要再連續 N 根站上 EMA50 或直接越過 +k_in ATR。

無前視：EMA / ATR 在第 i 根只用 ≤ i 根的 OHLC（RMA 遞迴、TR 用 close[i−1]）；
run 是 above 的前綴累計（maximum.accumulate 只看 ≤ i）；引擎在 i+1 開盤成交。
風控全預設（無停損、無波動目標、無回撤暫停）。
"""
import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams

NAME = "EMA50 ATR 通道 + 3 根確認快速進場"
DESCRIPTION = "close > EMA50 + 1.0×ATR14 或 連續 3 根收在 EMA50 上 進入趨勢、close < EMA50 − 0.25×ATR14 離開，其間維持；無風控"
PARAMS = {"ema_len": 50, "atr_len": 14, "k_in": 1.0, "k_out": 0.25, "confirm_bars": 3}


def _run_length(above: np.ndarray) -> np.ndarray:
    """以第 i 根結尾的「連續 above==True」根數；above[i]==False 時為 0。只用 ≤ i 的資料。"""
    n = len(above)
    idx = np.arange(n)
    last_below = np.maximum.accumulate(np.where(above, -1, idx))   # 最近一根 above==False 的索引（沒有則 −1）
    return idx - last_below


def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    n = len(c)
    ema = ta.ema(c, int(p["ema_len"]))              # Pine 語意：前 n−1 根 NaN，SMA 起算後遞迴
    atr = ta.atr(h, l, c, int(p["atr_len"]))        # RMA(TR)，TR 用前一根收盤
    N = int(p["confirm_bars"])
    above = c > ema                                 # NaN 比較 → False，暖機期自然為 False
    run = _run_length(above)
    fast_enter = run >= N                           # 連續 N 根（含本根）收盤 > EMA50
    chan_enter = c > ema + float(p["k_in"]) * atr   # 通道腿
    enter = chan_enter | fast_enter
    exit_ = c < ema - float(p["k_out"]) * atr       # 出場腿與 vol_atr_channel 逐位相同
    valid = ~np.isnan(ema) & ~np.isnan(atr)
    warmup = max(int(np.argmax(valid)) if valid.any() else n, N)
    return {"in_regime": {"enter": enter, "exit": exit_}, "risk": RiskParams(), "warmup": warmup}


if __name__ == "__main__":
    # 自我檢查：run 的算法
    ab = np.array([False, True, True, False, True, True, True])
    r = _run_length(ab)
    assert r.tolist() == [0, 1, 2, 0, 1, 2, 3], r
    cnt, loop = 0, []
    for a in ab:
        cnt = cnt + 1 if a else 0
        loop.append(cnt)
    assert loop == r.tolist()
    print("run_length OK", r.tolist())
