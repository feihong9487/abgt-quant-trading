"""相對強度閘（規則 B，規則 A「BTC 領先閘」的對照）：非 BTC 標的要同時滿足 自身 close > EMA50 且 自身/BTC 比值 > 比值的 EMA50 才在趨勢內。

設計目的：規則 A（自身 close > EMA50 AND BTC close > BTC EMA50）若改善 ETH，有兩種互斥解釋——
  (i) BTC 是崩盤領先者，BTC 的絕對趨勢狀態帶有 ETH 自身價格沒有的資訊；
  (ii) 只是「ETH 在跑輸 BTC 的日子表現差」這種橫斷面動能效應。
本變體完全不看 BTC 的絕對趨勢，只看 ETH 相對 BTC 的強弱：A 好 B 不好 → (i)；B 與 A 一樣好或更好 → (ii)；兩個都不好 → 方向關閉。
結構刻意與 A 一致（同 AND、同 RiskParams()、同 warmup 邏輯、同兩個整數參數），差異只剩閘門內容。

規則：
- BTC 本身（is_btc）：in_regime = close > EMA50，與 ema50_plain 逐根相同。
- 其他標的：ratio = close / BTC close（同日對齊），rs_on = ratio > EMA(ratio, ratio_ema_len)；in_regime = (close > EMA50) & rs_on。
- 比值用 EMA50：與自身 EMA50 同尺度，不新增調校自由度；橫斷面加密幣動能文獻常用 1–3 個月回看，50 日落在中間。
- 無風控（RiskParams 全預設）。

BTC 資料：build 內以 abgt.data 讀 BTC 日線快取（不連網），以 df.index 對齊並截到 ≤ df.index[-1]（截斷測試時 BTC 序列一起截斷），
BTC 快取落後於標的時直接報錯（否則尾端 NaN 會被靜默當作 False）。缺值（對齊不到）視為 False。

無前視：ratio[i] 只用第 i 根的自身與 BTC 收盤；EMA 為因果遞迴（SMA 起算）；build(df.iloc[:T]) 的前 T 根與全量逐位相同；引擎在 i+1 開盤成交。
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from abgt import data as dt                        # noqa: E402
from abgt import indicators as ta                  # noqa: E402
from abgt.strategies.regime import RiskParams      # noqa: E402

NAME = "相對強度閘（對照）"
DESCRIPTION = "非 BTC 標的：close > EMA50 且 close/BTC 比值 > 比值的 EMA50 才在趨勢內；BTC 本身 = close > EMA50；無風控"
PARAMS = {"ema_len": 50, "ratio_ema_len": 50}
BTC_SYMBOL, TF, DATA_SINCE = "BTC/USDT:USDT", "1d", "2019-09-01"
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")


# ---------- BTC 輔助（與規則 A 相同的兩個函式） ----------

@lru_cache(maxsize=1)
def _load_btc() -> pd.DataFrame:
    """讀 BTC 日線快取（與評估器同一份 CSV、同一個 DATA_SINCE），不連網；快取不存在直接報錯。"""
    btc = dt.load_cached(_DATA_DIR, BTC_SYMBOL, TF, DATA_SINCE)
    if btc is None:
        raise FileNotFoundError(f"找不到 {BTC_SYMBOL} {TF} 快取：{dt.cache_path(_DATA_DIR, BTC_SYMBOL, TF)}（先跑 evaluate_variant --fetch）")
    return btc


def _btc_view(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, bool]:
    """回傳 (截到 ≤ df.index[-1] 的 BTC df, 對齊到 df.index 的 BTC close 陣列（缺值 NaN）, 是否就是 BTC 本身)。

    - 截斷：BTC 只保留時間 ≤ df.index[-1] 的根，build(df.iloc[:T]) 看到的 BTC 與 build(df) 前 T 根完全一致（無前視）。
    - 落後檢查：BTC 快取最後一根必須 ≥ df.index[-1]，否則尾端對齊值為 NaN 會被靜默當作 False → 直接報錯。
    - is_btc：對齊後兩邊 close 在重疊區逐根相等（容忍浮點）即視為 BTC 本身（不靠符號名稱）。
    """
    btc_full = _load_btc()
    if btc_full.index[-1] < df.index[-1]:
        raise ValueError(f"BTC 快取落後：BTC 最後一根 {btc_full.index[-1]} < 標的最後一根 {df.index[-1]}，先補 BTC 快取")
    btc = btc_full.loc[:df.index[-1]]
    aligned = btc["close"].reindex(df.index).to_numpy(dtype="float64")
    own = df["close"].to_numpy(dtype="float64")
    overlap = ~np.isnan(aligned)
    is_btc = bool(overlap.any() and np.allclose(own[overlap], aligned[overlap], rtol=1e-9, atol=0.0))
    return btc, aligned, is_btc


# ---------- 變體 ----------

def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    n = len(c)
    ema = ta.ema(c, int(p["ema_len"]))              # Pine 語意：前 n−1 根 NaN，SMA 起算後遞迴
    own_on = c > ema                                 # == ema50_plain（NaN 比較為 False）
    valid = ~np.isnan(ema)
    warmup = int(np.argmax(valid)) if valid.any() else n
    btc, aligned, is_btc = _btc_view(df)             # aligned = BTC close 對齊到 df.index，缺值 NaN
    if is_btc:
        return {"in_regime": own_on, "risk": RiskParams(), "warmup": warmup}
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = c / aligned                          # BTC 缺值 → NaN
    sd = np.nanstd(ratio) if np.isfinite(ratio).any() else np.nan
    if not np.isfinite(sd) or sd == 0:
        raise ValueError("ratio 恆為常數或全缺值：is_btc 判定失敗或資料異常")
    ratio_ema = ta.ema(ratio, int(p["ratio_ema_len"]))   # 以第一個非 NaN 起算、內部 NaN 用 ignore_na；現有快取下 ratio 無 NaN
    with np.errstate(invalid="ignore"):
        rs_on = ratio > ratio_ema                    # NaN → False（缺值視為 False）
    ok = ~np.isnan(ratio_ema)
    first_rs_valid = int(np.argmax(ok)) if ok.any() else n
    warmup = max(warmup, first_rs_valid)             # 兩個長度都 50 → 都在第 49 根有效 → warmup 49，與基準視窗相同
    on = own_on & rs_on
    on[:warmup] = False
    return {"in_regime": on, "risk": RiskParams(), "warmup": warmup}


# ---------- 診斷（__main__）：與規則 A 相同的統計，讓兩個變體的 avoided 日可直接比較 ----------

_WINDOWS = {"IS": ("2020-01-01", "2022-12-31"), "OOS": ("2023-01-01", None), "FULL": ("2020-01-01", None),
            "IS2": ("2020-07-01", "2023-12-31"), "OOS2": ("2024-01-01", None)}
_RISK_SEG = ("2020-10-01", "2020-12-31")     # 事前指定的主要風險段：BTC 領漲、ETH 絕對上漲但相對走弱


def _window_mask(idx: pd.DatetimeIndex, start: str, end: str | None) -> np.ndarray:
    m = idx >= pd.Timestamp(start, tz="UTC")
    if end:
        m &= idx <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return np.asarray(m)


def _truncation_check(df: pd.DataFrame, p: dict, full_on: np.ndarray, full_warmup: int, step: int = 97) -> tuple[int, int]:
    """build(df.iloc[:T]) 前 T 根必須與全量逐位相同（BTC 在 _btc_view 內同步截斷）。回傳 (切點數, 不一致筆數)。"""
    n = len(df)
    cuts = list(range(full_warmup + 1, n, step)) + [n - 1, n]
    bad = 0
    for T in cuts:
        b = build(df.iloc[:T], p)
        if int(b["warmup"]) != full_warmup or not np.array_equal(np.asarray(b["in_regime"], dtype=bool), full_on[:T]):
            bad += 1
    return len(cuts), bad


def diagnostics(df: pd.DataFrame, sym: str, p: dict) -> dict:
    c = df["close"].to_numpy(dtype="float64")
    o = df["open"].to_numpy(dtype="float64")
    n = len(c)
    idx = df.index
    built = build(df, p)
    on = np.asarray(built["in_regime"], dtype=bool)
    warmup = int(built["warmup"])
    ema = ta.ema(c, int(p["ema_len"]))
    own_on = c > ema
    own_on[:warmup] = False
    btc, aligned, is_btc = _btc_view(df)
    out = {"symbol": sym, "is_btc": is_btc, "n": n, "warmup": warmup, "start": str(idx[warmup].date()), "end": str(idx[-1].date())}
    cuts, bad = _truncation_check(df, p, on, warmup)
    out["truncation"] = {"cuts": cuts, "mismatch": bad}
    if is_btc:
        out["identical_to_baseline"] = bool(np.array_equal(on, own_on))
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = c / aligned
    ratio_ema = ta.ema(ratio, int(p["ratio_ema_len"]))
    with np.errstate(invalid="ignore"):
        rs_on = ratio > ratio_ema
    rs_on[:warmup] = False
    # 引擎實現的「第 i 根決策」報酬：i+1 開盤進、i+2 開盤出（不含費用）；次日收→收作為對照口徑
    r_oo = np.full(n, np.nan)
    r_oo[:-2] = np.log(o[2:] / o[1:-1])
    r_cc = np.full(n, np.nan)
    r_cc[:-1] = np.log(c[1:] / c[:-1])
    avoided = own_on & ~rs_on                       # 基準在場、本變體因相對強度不在場（AND 結構下 added 恆為 0）
    added = on & ~own_on
    assert not added.any()
    ev = np.zeros(n, dtype=bool)
    ev[warmup:] = True

    def _flips(a: np.ndarray, m: np.ndarray) -> int:
        s = a[m].astype(np.int8)
        return int(np.abs(np.diff(s)).sum()) if len(s) > 1 else 0

    per_w = {}
    for w, (s0, s1) in _WINDOWS.items():
        m = _window_mask(idx, s0, s1) & ev
        av = avoided & m
        vals = r_oo[av]
        vals = vals[~np.isnan(vals)]
        per_w[w] = {
            "bars": int(m.sum()),
            "own_on_rate": float(own_on[m].mean()) if m.any() else None,
            "rs_on_rate": float(rs_on[m].mean()) if m.any() else None,
            "on_rate": float(on[m].mean()) if m.any() else None,
            "avoided_days": int(av.sum()),
            "avoided_sum_log_oo": float(vals.sum()) if len(vals) else 0.0,
            "avoided_sum_log_cc": float(np.nansum(r_cc[av])),
            "avoided_pos_days": int((vals > 0).sum()),
            "avoided_neg_days": int((vals < 0).sum()),
            "flips_baseline": _flips(own_on, m),
            "flips_variant": _flips(on, m),
        }
    out["windows"] = per_w
    # 最大的 avoided 日（依 |次日開→開| 排序）與集中度
    av_idx = np.where(avoided & ev & ~np.isnan(r_oo))[0]
    order = av_idx[np.argsort(-np.abs(r_oo[av_idx]))]
    top = [{"date": str(idx[i].date()), "r_oo": round(float(r_oo[i]), 4), "own_over_ema": round(float(c[i] / ema[i] - 1), 4),
            "ratio_over_ema": round(float(ratio[i] / ratio_ema[i] - 1), 4)} for i in order[:10]]
    out["avoided_top10"] = top
    tot = float(np.nansum(r_oo[av_idx]))
    out["avoided_top10_share"] = (float(sum(r_oo[i] for i in order[:10])) / tot) if tot != 0 else None
    # 事前指定的風險段：ETH 2020-10-01→2020-12-31（own_on & ~rs_on）根數與 ETH 累積對數報酬
    m = _window_mask(idx, *_RISK_SEG) & ev
    seg_av = avoided & m
    out["risk_segment"] = {
        "range": f"{_RISK_SEG[0]}→{_RISK_SEG[1]}",
        "bars": int(m.sum()),
        "own_on_days": int((own_on & m).sum()),
        "avoided_days": int(seg_av.sum()),
        "avoided_sum_log_oo": float(np.nansum(r_oo[seg_av])),
        "avoided_sum_log_cc": float(np.nansum(r_cc[seg_av])),
        "segment_log_return_close": float(np.log(c[np.where(m)[0][-1]] / c[np.where(m)[0][0]])) if m.any() else None,
        "segment_ratio_log_change": float(np.log(ratio[np.where(m)[0][-1]] / ratio[np.where(m)[0][0]])) if m.any() else None,
    }
    # 逐年 avoided 合計（看改善 / 傷害集中在哪一年）
    years = {}
    for y in sorted(set(idx[ev].year)):
        m = np.asarray(idx.year == y) & ev
        av = avoided & m
        years[int(y)] = {"avoided_days": int(av.sum()), "avoided_sum_log_oo": round(float(np.nansum(r_oo[av])), 4),
                         "rs_on_rate": round(float(rs_on[m].mean()), 3) if m.any() else None}
    out["by_year"] = years
    return out


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,BNB/USDT:USDT")
    ap.add_argument("--out", default=os.path.join(_PROJECT_ROOT, "results", "variants", "btc_leader_gate_relstrength_ratio_diag.json"))
    args = ap.parse_args()
    report = {"variant": "btc_leader_gate_relstrength_ratio", "params": PARAMS, "symbols": {}}
    for sym in args.symbols.split(","):
        df = dt.load_cached(_DATA_DIR, sym, TF, DATA_SINCE)
        if df is None:
            print(f"{sym} 沒有快取")
            continue
        d = diagnostics(df, sym, PARAMS)
        report["symbols"][sym] = d
        print(f"--- {sym} --- is_btc={d['is_btc']} warmup={d['warmup']} {d['start']}→{d['end']} 截斷測試 {d['truncation']['cuts']} 切點 不一致 {d['truncation']['mismatch']}")
        if d["is_btc"]:
            print(f"  與 ema50_plain 逐根相同：{d['identical_to_baseline']}")
            continue
        print(f"  {'區間':5}{'根數':>6}{'own_on':>8}{'rs_on':>8}{'on':>8}{'avoided':>9}{'Σlog(開→開)':>13}{'Σlog(收→收)':>13}{'正/負':>9}{'翻轉 基準/變體':>16}")
        for w, x in d["windows"].items():
            print(f"  {w:5}{x['bars']:>6}{x['own_on_rate']:>8.2%}{x['rs_on_rate']:>8.2%}{x['on_rate']:>8.2%}{x['avoided_days']:>9}"
                  f"{x['avoided_sum_log_oo']:>+13.3f}{x['avoided_sum_log_cc']:>+13.3f}{x['avoided_pos_days']:>4}/{x['avoided_neg_days']:<4}"
                  f"{x['flips_baseline']:>8}/{x['flips_variant']:<6}")
        rs = d["risk_segment"]
        print(f"  風險段 {rs['range']}：{rs['bars']} 根，own_on {rs['own_on_days']} 根，avoided {rs['avoided_days']} 根，"
              f"avoided Σlog 開→開 {rs['avoided_sum_log_oo']:+.3f} / 收→收 {rs['avoided_sum_log_cc']:+.3f}；"
              f"段內自身 log 報酬 {rs['segment_log_return_close']:+.3f}、比值 log 變化 {rs['segment_ratio_log_change']:+.3f}")
        print(f"  avoided 前 10 大（|開→開|）佔合計 {d['avoided_top10_share'] if d['avoided_top10_share'] is None else round(d['avoided_top10_share'], 2)}：")
        for t in d["avoided_top10"]:
            print(f"    {t['date']}  r_oo {t['r_oo']:+.4f}  close/EMA50 {t['own_over_ema']:+.3f}  ratio/EMA {t['ratio_over_ema']:+.3f}")
        print("  逐年：" + "  ".join(f"{y}: {v['avoided_days']}日 {v['avoided_sum_log_oo']:+.3f} (rs_on {v['rs_on_rate']:.0%})" for y, v in d["by_year"].items()))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n→ {args.out}")
