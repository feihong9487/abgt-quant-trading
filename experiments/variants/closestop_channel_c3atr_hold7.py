"""EMA50 ± ATR 通道 + 收盤確認 3×ATR 吊燈停損 + 收盤回升重進（7 根冷卻）。

regime：與正式策略 atr_channel.py / 第 1 輪 vol_atr_channel 逐根相同 —— close > EMA50 + k_in×ATR 進、
close < EMA50 − k_out×ATR 出、通道內維持；enter/exit 交給引擎狀態機。
疊加的只有 RiskParams 停損區塊：
- 吊燈停損 stop = max(前值, 進場後最高收盤 − 3×ATR14)，第 i 根「收盤」跌破才觸發（stop_trigger="close"），i+1 開盤成交
- 重進 close_above_stop：停損後要有一根收盤高於停損觸發收盤，且距停損 ≥ 7 根，且仍在趨勢內才重進；
  若崩盤把趨勢打斷（c < EMA50 − 0.25 ATR），重進改走通道的 enter（c > EMA50 + 1.0 ATR），價格門檻比純 EMA50 嚴格
假設：停損只截斷 2021-05、2022-05 那類快速崩盤（停損比 −0.25 ATR 出場早），非崩盤回檔不重進更高。
無前視：EMA / ATR 只用 ≤ i 根 OHLC；狀態機與停損由前往後遞推；引擎在 i+1 開盤成交。
"""
from abgt.strategies.regime import RiskParams, regime_atr_channel

NAME = "EMA50±ATR 通道 + 收盤確認 3×ATR 吊燈停損 + 收盤回升重進（7 根冷卻）"
DESCRIPTION = ("close > EMA50 + 1.0×ATR14 進、< EMA50 − 0.25×ATR14 出（通道內維持）；"
               "進場後最高收盤 − 3×ATR14 收盤跌破即停損，收盤回到停損價之上且隔 ≥ 7 根才重進")
PARAMS = {"ema_len": 50, "atr_len": 14, "k_in": 1.0, "k_out": 0.25, "atr_stop_mult": 3.0, "reentry_min_bars": 7}


def build(df, p):
    n_ema = int(p["ema_len"])
    n_atr = int(p["atr_len"])
    k_in = float(p["k_in"])
    k_out = float(p["k_out"])
    k_stop = float(p["atr_stop_mult"])
    hold = int(round(p["reentry_min_bars"]))
    enter, exit_, warmup = regime_atr_channel(df, n_ema, n_atr, k_in, k_out)   # 引擎內建：EMA/ATR 只用 ≤ i 根
    rp = RiskParams(max_exposure=1.0, vol_target=0.0,
                    atr_len=n_atr, atr_stop_mult=k_stop, stop_trigger="close",
                    reentry="close_above_stop", reentry_min_bars=hold,
                    dd_pause=0.0)
    return {"in_regime": {"enter": enter, "exit": exit_}, "risk": rp, "warmup": warmup}
