"""BTC 領先閘（AND）：非 BTC 標的 in_regime = (自身 close > EMA50) & (BTC close > BTC EMA50)；BTC 本身 = close > EMA50。

設計：第 1、2 輪 22 個變體全是單標的濾波；本變體第一次把另一個標的（BTC）的趨勢狀態帶進來。
BTC 閘直接沿用冠軍 ema50_plain 的 BTC 規則（close > EMA50），零個新調校參數；btc_ema_len 只是為了讓敏感度能單獨擾動 BTC 側。
結構假設：加密幣系統性崩盤由 BTC 帶動、山寨幣 beta > 1，ETH 在「自身 > EMA50 但 BTC < EMA50」的日子是脫離大盤的獨行上漲，
事後多半是熊市反彈或盤整鋸齒。對稱 AND 是唯一無狀態且自洽的形式：進場側 BTC-on 是確認、出場側 BTC-off 是提前。
只改 in_regime 的判定；風控全預設（無停損、無波動目標、無回撤暫停）。
無前視：第 i 根只用 ≤ i 根的自身收盤與 ≤ i 根的 BTC 收盤（兩者都是 00:00 UTC 同時收的日線，決策時點相同）；
EMA 為因果遞迴，build 內把 BTC 序列截到 df.index[-1]，所以 build(df.iloc[:T]) 的 in_regime[:T] 與全量 build 的前 T 根逐位相同。
"""
import os

if __name__ == "__main__":                            # 直接執行診斷時才需要：把專案根目錄加進 sys.path（評估器 import 不走這段）
    import sys as _sys
    _root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)

import numpy as np
import pandas as pd

from abgt import data as dt
from abgt import indicators as ta
from abgt.strategies.regime import RiskParams

NAME = "BTC 領先閘（AND）"
DESCRIPTION = "非 BTC 標的：close > EMA50 且 BTC close > BTC EMA50 才在趨勢內；BTC 本身 = close > EMA50（與基準逐根相同）；無風控"
PARAMS = {"ema_len": 50, "btc_ema_len": 50}
BTC_SYMBOL, TF, DATA_SINCE = "BTC/USDT:USDT", "1d", "2019-09-01"   # 與 evaluate_variant.DATA_SINCE 相同


def _load_btc():
    # 評估器從專案根目錄跑（cache_dir="data"）；找不到時退回「變體檔往上兩層/data」，兩處都沒有就報錯
    here = os.path.dirname(os.path.abspath(__file__))
    for d in ("data", os.path.normpath(os.path.join(here, "..", "..", "data"))):
        btc = dt.load_cached(d, BTC_SYMBOL, TF, DATA_SINCE)
        if btc is not None:
            return btc
    raise FileNotFoundError("缺 BTC 日線快取 data/BTCUSDT_USDT_1d.csv")


def _btc_view(df):
    """回傳 (btc 截到 ≤ df.index[-1], aligned_close, is_btc)。"""
    btc = _load_btc()
    btc = btc.loc[btc.index <= df.index[-1]]          # 截斷：build 內看不到 df 最後一根之後的任何 BTC 資料（前視截斷測試時 BTC 一起截斷）
    if btc.index[-1] < df.index[-1]:
        raise ValueError(f"BTC 快取落後：BTC 最後一根 {btc.index[-1].date()} < 標的最後一根 {df.index[-1].date()}，"
                         f"先用 --variant ema50_plain --symbols BTC/USDT:USDT --fetch 補 BTC")
    aligned = btc["close"].reindex(df.index).to_numpy("float64")   # df 有、BTC 沒有的日期 → NaN
    c = df["close"].to_numpy("float64")
    same = np.isclose(aligned, c, rtol=1e-9, atol=0.0)             # NaN → False
    is_btc = bool(same.mean() >= 0.99)                             # df 本身就是 BTC（同一份快取，100% 相同；其他標的 ≈ 0%）
    return btc, aligned, is_btc


def build(df, p):
    c = df["close"].to_numpy("float64")
    n = len(c)
    ema = ta.ema(c, int(p["ema_len"]))               # Pine 語意：前 n−1 根 NaN，SMA 起算後遞迴
    own_on = c > ema                                  # == ema50_plain 的 in_regime（NaN 比較為 False）
    valid = ~np.isnan(ema)
    warmup = int(np.argmax(valid)) if valid.any() else n
    btc, aligned, is_btc = _btc_view(df)
    if is_btc:                                        # BTC 本身：基準規則，regime_key 與 ema50_plain 相同
        return {"in_regime": own_on, "risk": RiskParams(), "warmup": warmup}
    btc_c = btc["close"].to_numpy("float64")
    btc_ema = ta.ema(btc_c, int(p["btc_ema_len"]))   # 在 BTC 自己的完整（已截斷）序列上算：seed 用 BTC 2019-09-08 起的前 n 根，與 df 起點無關
    btc_on = pd.Series(btc_c > btc_ema, index=btc.index).reindex(df.index, fill_value=False).to_numpy(bool)   # 缺值視為 False
    btc_ema_on_df = pd.Series(btc_ema, index=btc.index).reindex(df.index).to_numpy("float64")
    ok = ~np.isnan(btc_ema_on_df)
    first_btc_valid = int(np.argmax(ok)) if ok.any() else n
    warmup = max(warmup, first_btc_valid)             # 現有快取下 = 自身 warmup（ETH 第 49 根 2020-01-15；BTC EMA50 自 2019-10-27 有效）→ 視窗與基準完全相同
    on = own_on & btc_on
    on[:warmup] = False
    return {"in_regime": on, "risk": RiskParams(), "warmup": warmup}


if __name__ == "__main__":
    # 診斷（只在直接執行時跑，評估器 import 不觸發）：列印與基準 in_regime 不同的根數與那些日子的相對報酬
    import sys
    os.chdir(_root)
    sym = sys.argv[1] if len(sys.argv) > 1 else "ETH/USDT:USDT"
    df = dt.load_cached("data", sym, TF, DATA_SINCE)
    r = build(df, PARAMS)
    on = r["in_regime"]
    w = r["warmup"]
    c_ = df["close"].to_numpy("float64")
    base = c_ > ta.ema(c_, PARAMS["ema_len"])
    base[:w] = False
    o = df["open"].to_numpy("float64")
    r_next = np.full(len(o), np.nan)
    r_next[:-2] = np.log(o[2:] / o[1:-1])                # 第 i 根判定差異 → 持倉差異在 open[i+1]→open[i+2]
    avoided = base & ~on
    avoided[:w] = False                                   # 基準持有、變體空手的根（變體 ⊂ 基準，無反向）
    reverse = on & ~base                                  # 應恆為 0
    btc_last = _load_btc().index[-1]
    n_eval = len(on) - w
    idx = df.index
    print(f"=== {NAME} | {sym} | warmup={w} ({idx[w].date()}) | 評估根數 {n_eval}")
    print(f"BTC 快取最後日期 {btc_last.date()} vs df 最後日期 {idx[-1].date()}")
    print(f"變體 ⊂ 基準檢查：on & ~base = {int(reverse.sum())} 根（預期 0）")
    print(f"avoided（基準持有、變體空手）= {int(avoided.sum())} 根，佔評估期 {avoided.sum() / n_eval:.1%}，佔基準持有日 {avoided.sum() / max(1, base.sum()):.1%}")
    ra = r_next[avoided]
    ra_ok = ra[~np.isnan(ra)]
    print(f"avoided 日次日開盤→開盤對數報酬：sum {ra_ok.sum():+.4f}，mean {ra_ok.mean():+.5f}（{ra_ok.mean() * 100:+.3f}%/日），n={len(ra_ok)}，勝率（>0）{(ra_ok > 0).mean():.1%}")
    years = pd.Series(np.where(avoided, r_next, np.nan), index=idx).groupby(idx.year)
    cnt = pd.Series(avoided.astype(int), index=idx).groupby(idx.year).sum()
    print("按年：年 | avoided 根數 | sum r_next")
    for y, s in years.sum().items():
        if cnt[y] > 0:
            print(f"  {y} | {int(cnt[y]):4d} | {s:+.4f}")
    order = np.argsort(-np.abs(np.where(avoided & ~np.isnan(r_next), r_next, 0.0)))[:10]
    print("|r_next| 最大的 10 個 avoided 日（判定日 → 次日開盤→再次日開盤）：")
    for i in order:
        if avoided[i]:
            print(f"  {idx[i].date()}  r_next {r_next[i]:+.4f}  close {c_[i]:.2f}")
    def _trips(a):
        a = a.astype(int)
        return int(((a[1:] - a[:-1]) == 1).sum() + (1 if a[0] else 0))
    print(f"回合（0→1 轉換）：基準 {_trips(base[w:])}，變體 {_trips(on[w:])}")
    print(f"在場率：基準 {base[w:].mean():.1%}，變體 {on[w:].mean():.1%}")
