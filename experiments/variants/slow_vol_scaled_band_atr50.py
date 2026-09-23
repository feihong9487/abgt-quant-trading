"""EMA50 通道，帶寬 = ATR50（中速平滑尺度，對照組）。

進場 close > EMA50 + k_in×ATR50、出場 close < EMA50 − k_out×ATR50，其間維持；由引擎 state_machine 維護狀態。
與 vol_atr_channel 唯一差別是帶寬的估計量：RMA(TR, 50)（alpha = 1/50、半衰期 ≈ 34 根）取代 RMA(TR, 14)（半衰期 ≈ 9 根）。
ATR50 與 ATR14 的長期均值相同（同一個 TR 的均值），所以帶寬平均水準不變、只有時間剖面改變：崩盤後 ATR50 仍「記著」崩盤
（60 根前的 TR 仍佔 0.98^60 ≈ 30% 權重），V 型反轉時帶寬不會比 ATR14 更早收斂。這是「慢波動尺度帶寬」方向的對照組（A = 365 日中位數）。

ema_len / k_in / k_out 原封不動繼承通道；RiskParams 全預設（max_exposure 1.0、無波動目標、無停損、無回撤暫停、funding 0、delay 0、
rebalance_threshold 0.10），與 vol_atr_channel 逐位相同。

無前視：EMA 與 RMA 都是因果遞迴（SMA 起算），TR 用 close[i−1]；enter / exit 每根只讀 ≤ i 的 OHLC；引擎在 i+1 開盤成交。
enter 與 exit 不可能同根同時為真（atr50 ≥ 0 ⇒ upper ≥ lower）。

暖機：ema 第 49 根起有值、atr50 第 49 根起有值 → warmup = 49，與基準 / 通道同視窗。敏感度擾動 ema_len 62 或 atr_len 62 時 warmup = 61，
ETH 快取（2019-11-27 起）的 IS / FULL 起點會後移 12 根（2020-01-15 → 2020-01-27），那些列是跨視窗數字。

診斷（ABGT_DEBUG=1）：以 regime_atr_channel(df, ema_len, 14, 1.0, 0.25)（= vol_atr_channel）為參照，逐回合配對：
paired（1:1 重疊）/ split（1:N 或 N:1 重疊）/ extra（本變體有、通道沒有）/ skipped（通道有、本變體沒有）；
entry_lead = 通道進場根 − 本變體進場根（> 0 = 本變體提早）、exit_lead 同理；依 ratio = ATR14 / ATR50（在通道進場根）分桶
<0.8 / 0.8–1.2 / >1.2 的 paired 數與平均 entry_lead；崩盤（單日 ≤ −10% 或 5 日 ≤ −20%）後 60 根內的 paired 進 / 出場 lead。
判準：FULL 每標的 extra ≤ 3 且 extra 勝率 > 0；預先登記：ratio > 1.2 桶 paired 很少且平均 entry_lead ≤ 0；崩盤後出場 exit_lead < 0。
"""
import os

import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams, regime_atr_channel, state_machine

NAME = "EMA50 通道，帶寬 = ATR50"
DESCRIPTION = "close > EMA50 + 1.0×ATR50 進、close < EMA50 − 0.25×ATR50 出，其間維持；無風控"
PARAMS = {"ema_len": 50, "atr_len": 50, "k_in": 1.0, "k_out": 0.25}

_DEBUG_WINDOWS = (("IS", "2020-01-01", "2022-12-31"), ("OOS", "2023-01-01", None), ("FULL", "2020-01-01", None),
                  ("IS2", "2020-07-01", "2023-12-31"), ("OOS2", "2024-01-01", None))
_REF_ATR_LEN = 14          # 參照固定為 vol_atr_channel：EMA50 ± ATR14，k_in 1.0 / k_out 0.25
_REF_K_IN, _REF_K_OUT = 1.0, 0.25
_CRASH_1D, _CRASH_5D, _POST_CRASH_BARS = -0.10, -0.20, 60


def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    n = len(c)
    ema = ta.ema(c, int(p["ema_len"]))                 # Pine 語意：前 n−1 根 NaN，SMA 起算後遞迴；第 i 根只用 ≤ i 的收盤
    atr50 = ta.atr(h, l, c, int(p["atr_len"]))         # RMA(TR, atr_len)：前 n−1 根 NaN，第 n 根 SMA(TR) 起算，之後 alpha = 1/n；TR 用 close[i−1]
    upper = ema + float(p["k_in"]) * atr50
    lower = ema - float(p["k_out"]) * atr50
    enter = c > upper                                  # NaN 比較 → False（暖機期不進）
    exit_ = c < lower                                  # atr50 ≥ 0 ⇒ 不會與 enter 同根為真
    valid = ~np.isnan(ema) & ~np.isnan(atr50)
    warmup = int(np.argmax(valid)) if valid.any() else n   # 預設 = 49（與基準 / 通道同視窗）

    if os.environ.get("ABGT_DEBUG"):
        _debug(df, enter, exit_, ta.atr(h, l, c, _REF_ATR_LEN), atr50, p, warmup)

    return {"in_regime": {"enter": enter, "exit": exit_}, "risk": RiskParams(), "warmup": warmup}


# ---------- 診斷（不影響評估） ----------

def _rounds(on: np.ndarray) -> list[tuple[int, int]]:
    """回合 = [進場根, 出場根)；最後一回合若仍在場，出場根 = len(on)。"""
    n = len(on)
    prev = ta.shift_bool(on, 1)
    entries = np.flatnonzero(on & ~prev)
    exits = np.flatnonzero(~on & prev)
    out = []
    for e in entries:
        later = exits[exits > e]
        out.append((int(e), int(later[0]) if len(later) else n))
    return out


def _round_ret(o: np.ndarray, c: np.ndarray, e: int, x: int, fee: float = 0.0005, slip: float = 0.0002) -> float:
    """次日開盤成交的回合報酬（含費 / 滑點）；未平倉回合以最後收盤估。"""
    n = len(o)
    px_in = o[e + 1] * (1 + slip) if e + 1 < n else c[-1]
    px_out = o[x + 1] * (1 - slip) if x + 1 < n else c[-1]
    return px_out / px_in * (1 - fee) ** 2 - 1


def _hist(vals: list[int]) -> str:
    if not vals:
        return "{}"
    u, cnt = np.unique(np.asarray(vals), return_counts=True)
    return "{" + ", ".join(f"{int(k):+d}:{int(v)}" for k, v in zip(u, cnt)) + "}"


def _debug(df, enter, exit_, atr14, atr50, p, warmup):
    import pandas as pd

    idx = df.index
    n = len(idx)
    o = df["open"].to_numpy(dtype="float64")
    c = df["close"].to_numpy(dtype="float64")
    tag = "[atr50]"

    # 逐位驗證：與引擎既有 regime_atr_channel(df, ema_len, atr_len, k_in, k_out) 相同
    e2, x2, w2 = regime_atr_channel(df, ema_len=int(p["ema_len"]), atr_len=int(p["atr_len"]), k_in=float(p["k_in"]), k_out=float(p["k_out"]))
    same = bool(np.array_equal(enter, e2) and np.array_equal(exit_, x2) and warmup == w2)
    on_b = state_machine(enter, exit_)
    ref_e, ref_x, ref_w = regime_atr_channel(df, ema_len=int(p["ema_len"]), atr_len=_REF_ATR_LEN, k_in=_REF_K_IN, k_out=_REF_K_OUT)
    on_r = state_machine(ref_e, ref_x)
    print(f"{tag} n={n} {idx[0].date()}→{idx[-1].date()} close[-1]={c[-1]:.2f} params={p} warmup={warmup} ({idx[warmup].date() if warmup < n else 'n/a'})"
          f" | 與 regime_atr_channel 逐位相同={same} | 參照通道 warmup={ref_w}")

    # 崩盤根：單日 log 報酬 ≤ −10% 或 5 日 ≤ −20%
    lr = np.log(c / ta.shift(c, 1))
    lr5 = np.log(c / ta.shift(c, 5))
    crash = np.flatnonzero((lr <= _CRASH_1D) | (lr5 <= _CRASH_5D))
    post_crash = np.zeros(n, dtype=bool)
    for k in crash:
        post_crash[k + 1:min(n, k + 1 + _POST_CRASH_BARS)] = True

    # ATR14 / ATR50 比值：全樣本與「上穿 EMA50 日」的分位數（IS）
    ema = ta.ema(c, int(p["ema_len"]))
    ratio = atr14 / atr50
    cross_up = (c > ema) & ~(ta.shift(c, 1) > ta.shift(ema, 1))
    s_is = max(warmup, int(idx.searchsorted(pd.Timestamp("2020-01-01", tz="UTC"))))
    e_is = int(idx.searchsorted(pd.Timestamp("2022-12-31", tz="UTC"), side="right"))
    r_all = ratio[s_is:e_is]
    r_all = r_all[~np.isnan(r_all)]
    r_x = ratio[s_is:e_is][cross_up[s_is:e_is]]
    r_x = r_x[~np.isnan(r_x)]
    q = lambda a, qs: "/".join(f"{np.quantile(a, x):.2f}" for x in qs) if len(a) else "n/a"
    print(f"{tag} IS ATR14/ATR50 全樣本 q10/50/90={q(r_all, (0.1, 0.5, 0.9))} | 上穿 EMA50 日 n={len(r_x)} q50/90={q(r_x, (0.5, 0.9))}"
          f" | 上穿日 ATR14>ATR50 佔比={np.mean(r_x > 1):.0%}" if len(r_x) else f"{tag} IS 無上穿日")

    rb, rr = _rounds(on_b), _rounds(on_r)
    ov_b = [[j for j, (er, xr) in enumerate(rr) if eb < xr and er < xb] for (eb, xb) in rb]
    ov_r = [[i for i, (eb, xb) in enumerate(rb) if eb < xr and er < xb] for (er, xr) in rr]
    kinds_b = []
    for i, js in enumerate(ov_b):
        if not js:
            kinds_b.append("extra")
        elif len(js) == 1 and len(ov_r[js[0]]) == 1:
            kinds_b.append("paired")
        else:
            kinds_b.append("split")
    skipped = [j for j, is_ in enumerate(ov_r) if not is_]
    diff_bars = int(np.sum(on_b[warmup:] != on_r[warmup:]))
    print(f"{tag} 回合 本變體={len(rb)} 通道={len(rr)} | 在場 本變體={np.mean(on_b[warmup:]):.1%} 通道={np.mean(on_r[warmup:]):.1%}"
          f" | in_regime 不同根數={diff_bars}/{n - warmup} | 崩盤根={len(crash)}")

    for wname, ws, we in _DEBUG_WINDOWS:
        s0 = max(warmup, int(idx.searchsorted(pd.Timestamp(ws, tz="UTC"))))
        s1 = n if we is None else int(idx.searchsorted(pd.Timestamp(we, tz="UTC"), side="right"))
        in_w = lambda bar: s0 <= bar < s1
        paired, split, extra = [], [], []
        for i, (eb, xb) in enumerate(rb):
            if not in_w(eb):
                continue
            if kinds_b[i] == "paired":
                er, xr = rr[ov_b[i][0]]
                paired.append((eb, xb, er, xr))
            elif kinds_b[i] == "split":
                split.append((eb, xb, len(ov_b[i])))
            else:
                extra.append((eb, xb))
        skip_w = [rr[j] for j in skipped if in_w(rr[j][0])]
        entry_lead = [er - eb for eb, xb, er, xr in paired]
        exit_lead = [xr - xb for eb, xb, er, xr in paired if xr < n and xb < n]
        ex_rets = [_round_ret(o, c, eb, xb) for eb, xb in extra]
        ex_win = np.mean([r > 0 for r in ex_rets]) if ex_rets else float("nan")
        print(f"{tag} {wname:5} paired={len(paired):3d} split={len(split):2d} extra={len(extra):2d} skipped={len(skip_w):2d}"
              f" | entry_lead 均值={np.mean(entry_lead) if entry_lead else float('nan'):+.2f} 直方圖={_hist(entry_lead)}"
              f" | exit_lead 均值={np.mean(exit_lead) if exit_lead else float('nan'):+.2f} 直方圖={_hist(exit_lead)}")
        if extra:
            print(f"{tag} {wname:5}   extra 勝率={ex_win:.0%} 回合="
                  + ", ".join(f"{idx[eb].date()}→{idx[xb].date() if xb < n else 'open'} {r:+.1%}" for (eb, xb), r in zip(extra, ex_rets)))
        if split:
            print(f"{tag} {wname:5}   split=" + ", ".join(f"{idx[eb].date()}→{idx[xb].date() if xb < n else 'open'}×{k}" for eb, xb, k in split))
        if skip_w:
            print(f"{tag} {wname:5}   skipped（通道有、本變體無）=" + ", ".join(f"{idx[er].date()}→{idx[xr].date() if xr < n else 'open'} {_round_ret(o, c, er, xr):+.1%}" for er, xr in skip_w))
        # ratio 分桶（在通道進場根）
        buckets = {"<0.8": [], "0.8–1.2": [], ">1.2": []}
        for eb, xb, er, xr in paired:
            rt = ratio[er]
            if np.isnan(rt):
                continue
            key = "<0.8" if rt < 0.8 else (">1.2" if rt > 1.2 else "0.8–1.2")
            buckets[key].append(er - eb)
        print(f"{tag} {wname:5}   ratio 桶 paired 數 / 平均 entry_lead: "
              + " | ".join(f"{k}: {len(v)} / {np.mean(v) if v else float('nan'):+.2f}" for k, v in buckets.items()))
        # 崩盤後 60 根內
        pc_entry = [er - eb for eb, xb, er, xr in paired if post_crash[er]]
        pc_exit = [xr - xb for eb, xb, er, xr in paired if xr < n and xb < n and post_crash[xr]]
        print(f"{tag} {wname:5}   崩盤後 {_POST_CRASH_BARS} 根內：paired 進場 n={len(pc_entry)} 平均 entry_lead={np.mean(pc_entry) if pc_entry else float('nan'):+.2f}"
              f" | 出場 n={len(pc_exit)} 平均 exit_lead={np.mean(pc_exit) if pc_exit else float('nan'):+.2f}")
