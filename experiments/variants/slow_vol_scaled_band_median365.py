"""EMA50 通道，帶寬 = 一年中位數 ATR%（慢速穩健尺度）。

進場 close > EMA50 + k_in × band、出場 close < EMA50 − k_out × band，通道內維持前一狀態（引擎狀態機）。
band = median(ATR14 / close, 過去 365 根) × close：無量綱的慢速自校準百分比帶，跨標的語意相同。
ema_len / atr_len / k_in / k_out 原封繼承 vol_atr_channel，對母體的每一格 Δ 都可歸因到「帶寬尺度」這一個改動。

設計（round 2 §11.3）：ATR14 在崩盤後膨脹讓 +1 ATR 在 V 型反轉時是 6–10% 的距離（OOS 延遲來源），盤整低波動時縮到 2–3%
（鋸齒來源）。一年中位數 ATR% 對尖峰免疫：崩盤後慢帶比 ATR14 窄 → 進場較早；低波動盤整慢帶比 ATR14 寬 → 門檻反而更高。

無前視：EMA / ATR 因果遞迴（TR 用 close[i−1]）；rolling median 右對齊、含當根不含之後；band[i] = mref[i] × c[i]；
enter / exit 每根只讀 ≤ i 的 OHLC；引擎在 i+1 開盤成交。band ≥ 0 ⇒ upper > lower ⇒ enter 與 exit 不會同根為真。

REF_MIN_PERIODS = 30 是固定常數（不進 PARAMS）：讓 warmup 維持 49 與基準 / 通道同視窗（round 2 §12.1 ETH 視窗教訓）。
資料不足 365 根時 mref 是「≥ 30 根的擴張視窗中位數」（BTC 2019-10→2020-11、ETH 2020-01→2020-11）。

診斷：環境變數 ABGT_DEBUG=1 時印出對 vol_atr_channel（參照固定，不跟著擾動）的回合配對：paired / split / extra / skipped、
進出場提早 / 延後直方圖、依 ATR14/band 比值分桶的平均 entry_lead、extra / split / skipped 回合明細（不影響回傳值）。
"""
import os

import numpy as np
import pandas as pd

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams, regime_atr_channel, state_machine

NAME = "EMA50 通道，帶寬 = 一年中位數 ATR%"
DESCRIPTION = "close > EMA50 + 1.0×band 進、close < EMA50 − 0.25×band 出，band = median(ATR14/close, 365 根) × close；無風控"
PARAMS = {"ema_len": 50, "atr_len": 14, "ref_len": 365, "k_in": 1.0, "k_out": 0.25}
REF_MIN_PERIODS = 30      # 固定常數，不進 PARAMS：讓 warmup 維持 49 與基準 / 通道同視窗（round 2 §12.1 ETH 視窗教訓）

_DEBUG_WINDOWS = (("IS", "2020-01-01", "2022-12-31"), ("OOS", "2023-01-01", None), ("FULL", "2020-01-01", None),
                  ("IS2", "2020-07-01", "2023-12-31"), ("OOS2", "2024-01-01", None))
_ROUND_COST = 2 * (0.0005 + 0.0002)


def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    n = len(c)
    ema = ta.ema(c, int(p["ema_len"]))                  # Pine 語意：前 ema_len−1 根 NaN，SMA 起算後遞迴；第 i 根只用 ≤ i 收盤
    atr14 = ta.atr(h, l, c, int(p["atr_len"]))          # RMA(TR)，TR 用 close[i−1]；第 atr_len−1 根起有值
    atrp = atr14 / c                                    # ATR 佔價格的比例（無量綱）
    mref = pd.Series(atrp).rolling(int(p["ref_len"]), min_periods=REF_MIN_PERIODS).median().to_numpy()
    #  右對齊 rolling：第 i 根只含 i−ref_len+1 .. i；資料不足 ref_len 根時是「≥ REF_MIN_PERIODS 根的擴張視窗中位數」
    band = mref * c                                     # 回到價格單位；第 i 根只用 ≤ i 的資料
    upper = ema + float(p["k_in"]) * band
    lower = ema - float(p["k_out"]) * band
    enter = c > upper                                   # NaN 比較 → False
    exit_ = c < lower                                   # band ≥ 0 ⇒ upper > lower ⇒ enter 與 exit 不會同根為真
    valid = ~np.isnan(ema) & ~np.isnan(band)
    warmup = int(np.argmax(valid)) if valid.any() else n   # 預設 = 49（mref 從第 42 根起有值，EMA 決定 warmup）
    if os.environ.get("ABGT_DEBUG"):
        _debug(df, enter, exit_, atr14, band, p, warmup)
    return {"in_regime": {"enter": enter, "exit": exit_}, "risk": RiskParams(), "warmup": warmup}


# ---------------------------------------------------------------------------------------------------------------------
# 診斷（ABGT_DEBUG=1）：對 vol_atr_channel 的回合配對與提早 / 延後分布
# ---------------------------------------------------------------------------------------------------------------------

def _rounds(on):
    """[(e, x)]：e = 第一根 on[i] & ~on[i−1]，x = 其後第一根 ~on[j]（仍在場則 x = n）。"""
    on = np.asarray(on, dtype=bool)
    n = len(on)
    out = []
    i = 0
    while i < n:
        if on[i] and (i == 0 or not on[i - 1]):
            j = i + 1
            while j < n and on[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _ret(o, c, e, x):
    """open[x+1] / open[e+1] − 1 − 往返成本（x+1 ≥ n 用 close[n−1]）。"""
    n = len(o)
    px_in = o[e + 1] if e + 1 < n else c[n - 1]
    px_out = o[x + 1] if x + 1 < n else c[n - 1]
    return px_out / px_in - 1 - _ROUND_COST


def _bin(v):
    if v <= -3:
        return "≤−3"
    if v in (-2, -1, 0, 1, 2):
        return {-2: "−2", -1: "−1", 0: "0", 1: "+1", 2: "+2"}[v]
    if 3 <= v <= 5:
        return "3–5"
    return ">5"


_BINS = ["≤−3", "−2", "−1", "0", "+1", "+2", "3–5", ">5"]


def _hist(vals):
    cnt = {b: 0 for b in _BINS}
    for v in vals:
        cnt[_bin(int(v))] += 1
    return " ".join(f"{b}:{cnt[b]}" for b in _BINS)


def _debug(df, enter, exit_, atr14, band, p, warmup):
    o = df["open"].to_numpy(dtype="float64")
    c = df["close"].to_numpy(dtype="float64")
    idx = df.index
    n = len(c)
    chan_enter, chan_exit, _ = regime_atr_channel(df, ema_len=int(p["ema_len"]), atr_len=14, k_in=1.0, k_out=0.25)
    on_v = state_machine(enter, exit_)
    on_c = state_machine(chan_enter, chan_exit)
    rv = [r for r in _rounds(on_v) if r[0] >= warmup]
    rc = [r for r in _rounds(on_c) if r[0] >= warmup]
    d = lambda i: str(idx[min(i, n - 1)].date())        # noqa: E731
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = atr14 / band

    # --- 配對 ---
    status = []                 # 每個變體回合：("paired", ci) / ("split", ci, prev_x_v) / ("extra",)
    taken = {}                  # ci → 最後一個配到它的變體回合的 x_v（split 的 gap 用）
    overlapped = set()
    for (e_v, x_v) in rv:
        ov = [ci for ci, (e_c, x_c) in enumerate(rc) if e_v < x_c and e_c < x_v]
        overlapped.update(ov)
        if not ov:
            status.append(("extra",))
            continue
        free = [ci for ci in ov if ci not in taken]
        if free:
            ci = free[0]
            status.append(("paired", ci))
        else:
            ci = ov[0]
            status.append(("split", ci, taken[ci]))
        taken[ci] = x_v
    skipped = [ci for ci in range(len(rc)) if ci not in overlapped]

    print(f"[slowband] warmup={warmup} ({d(warmup)}) n={n} | 變體回合 {len(rv)} / 通道回合 {len(rc)} | "
          f"mref(ATR%) 中位 {np.nanmedian(band / c) * 100:.2f}% 範圍 {np.nanmin((band / c)[warmup:]) * 100:.2f}–{np.nanmax((band / c)[warmup:]) * 100:.2f}% | "
          f"ATR14/band 中位 {np.nanmedian(ratio[warmup:]):.2f}")
    ratio_at_v = [ratio[e] for e, _ in rv]
    print(f"[slowband] 變體進場日 ATR14/band：中位 {np.median(ratio_at_v):.2f}，<0.8 {sum(r < 0.8 for r in ratio_at_v)}，"
          f"0.8–1.2 {sum(0.8 <= r <= 1.2 for r in ratio_at_v)}，>1.2 {sum(r > 1.2 for r in ratio_at_v)}")

    for wname, ws, we in _DEBUG_WINDOWS:
        s0 = max(warmup, int(idx.searchsorted(pd.Timestamp(ws, tz="UTC"))))
        s1 = n if we is None else int(idx.searchsorted(pd.Timestamp(we, tz="UTC"), side="right"))
        in_w = lambda e: s0 <= e < s1      # noqa: E731
        vi = [k for k, (e, _) in enumerate(rv) if in_w(e)]
        ci_w = [ci for ci, (e, _) in enumerate(rc) if in_w(e)]
        paired = [k for k in vi if status[k][0] == "paired"]
        split = [k for k in vi if status[k][0] == "split"]
        extra = [k for k in vi if status[k][0] == "extra"]
        skip_w = [ci for ci in ci_w if ci in skipped]
        print(f"[slowband] {wname:5} n_rounds_v={len(vi)} n_rounds_c={len(ci_w)} paired={len(paired)} split={len(split)} "
              f"extra={len(extra)} skipped={len(skip_w)}")
        entry_lead = [rc[status[k][1]][0] - rv[k][0] for k in paired]
        exit_lead = [rc[status[k][1]][1] - rv[k][1] for k in paired]
        if paired:
            print(f"[slowband] {wname:5} entry_lead(e_c−e_v, 正=變體早) {_hist(entry_lead)} | 平均 {np.mean(entry_lead):+.2f} 中位 {np.median(entry_lead):+.1f}")
            print(f"[slowband] {wname:5} exit_lead (x_c−x_v, 正=變體早) {_hist(exit_lead)} | 平均 {np.mean(exit_lead):+.2f} 中位 {np.median(exit_lead):+.1f}")
            buckets = {"<0.8": [], "0.8–1.2": [], ">1.2": []}
            for k in paired:
                e_c = rc[status[k][1]][0]
                r = ratio[e_c]
                key = "<0.8" if r < 0.8 else (">1.2" if r > 1.2 else "0.8–1.2")
                buckets[key].append(rc[status[k][1]][0] - rv[k][0])
            print("[slowband] " + f"{wname:5} 依通道進場日 ATR14/band 分桶：" +
                  " | ".join(f"{b}: paired={len(v)} 平均 entry_lead={np.mean(v):+.2f}" if v else f"{b}: paired=0" for b, v in buckets.items()))
            # 配對回合的變體 vs 通道報酬（同一配對）
            lv = sum(np.log1p(_ret(o, c, *rv[k])) for k in paired)
            lc = sum(np.log1p(_ret(o, c, *rc[status[k][1]])) for k in paired)
            print(f"[slowband] {wname:5} paired 回合 Σlog(1+ret)：變體 {lv:+.3f} 通道 {lc:+.3f} 差 {lv - lc:+.3f}")
        if extra:
            rets = [_ret(o, c, *rv[k]) for k in extra]
            print(f"[slowband] {wname:5} extra 回合 {len(extra)}：勝率 {np.mean([r > 0 for r in rets]):.0%} Σlog {sum(np.log1p(rets)):+.3f} | " +
                  ", ".join(f"{d(rv[k][0])}→{d(rv[k][1])} {r * 100:+.1f}%" for k, r in zip(extra, rets)))
        if split:
            gaps = [o[min(rv[k][0] + 1, n - 1)] / o[min(status[k][2] + 1, n - 1)] - 1 for k in split]
            print(f"[slowband] {wname:5} split 回合 {len(split)}：Σlog(1+gap) {sum(np.log1p(gaps)):+.3f} | " +
                  ", ".join(f"{d(rv[k][0])}→{d(rv[k][1])} gap {g * 100:+.1f}% (通道 {d(rc[status[k][1]][0])}→{d(rc[status[k][1]][1])})"
                            for k, g in zip(split, gaps)))
        if skip_w:
            rets = [_ret(o, c, *rc[ci]) for ci in skip_w]
            losses = [r for r in rets if r <= 0]
            wins = [r for r in rets if r > 0]
            print(f"[slowband] {wname:5} skipped 通道回合 {len(skip_w)}：虧損 {len(losses)} 獲利 {len(wins)} Σlog虧損 {sum(np.log1p(losses)):+.3f} "
                  f"Σlog獲利 {sum(np.log1p(wins)):+.3f} | " +
                  ", ".join(f"{d(rc[ci][0])}→{d(rc[ci][1])} {r * 100:+.1f}% (ATR14/band {ratio[rc[ci][0]]:.2f})" for ci, r in zip(skip_w, rets)))
