"""BGT 牛熊三變 V5.1.3 的 Python 版（對應 bgt_strategy_v2.pine / bgt_strategy.pine）。

信號計算和 Pine 版一一對應；模擬器貼著 Pine strategy() 的語意：
- process_orders_on_close=true：市價單在信號 K 線收盤價成交（± 滑點）
- KC 止損是前一根掛的 stop 單，本根觸價即成交
- 走人減倉對每筆 entry 各減 qty_percent
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import indicators as ta
from ..broker import BacktestResult, Book


@dataclass
class BGTParams:
    kf_atr_len: int = 14
    kf_atr_mult: float = 2.0
    kf_htf: str = "45min"
    kf_buf_mult: float = 0.5
    kc_len: int = 80
    kc_mult_up: float = 10.5
    kc_mult_dn: float = 9.5
    kc_smooth: int = 50
    prob_threshold: int = 65
    swing_len: int = 10
    max_boxes: int = 5
    div_len: int = 5
    walk_cooldown: int = 100
    mtf_tf: str = "4h"
    mtf_ema_len: int = 50
    use_fear_greed: bool = True
    use_eth_opt: bool = True
    use_linkage: bool = True
    use_add_on: bool = True
    entry_mode: str = "multi"          # "kf" = BGT-S1 只用 KF 主信號開倉；"multi" = BGT-S2 三選一
    require_kf_dir_all: bool = False   # True：聯動路徑也要求 45 分方向一致（簡報規則；Pine v2 沒有這道閘）


@dataclass
class BGTEngineParams:
    initial_capital: float = 10_000.0
    commission_pct: float = 0.0005     # 0.05% 每邊
    slippage_ticks: int = 2
    tick_size: float = 0.01            # ETHUSDT 永續 0.01
    entry_frac: float = 0.25           # 每批 25% 權益
    pyramiding: int = 4
    walk_low_reduce: float = 0.30
    walk_mid_reduce: float = 0.50
    use_kc_stop: bool = True


def _tf_minutes(index: pd.DatetimeIndex) -> int:
    return int(pd.Series(index).diff().dropna().median().total_seconds() // 60)


def compute_signals(df: pd.DataFrame, p: BGTParams) -> pd.DataFrame:
    o, h, l, c, v = (df[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close", "volume"))
    n = len(c)
    tfm = _tf_minutes(df.index)
    hl2 = (h + l) / 2

    # === V1 當前週期趨勢 ===
    atr_v1 = ta.atr(h, l, c, p.kf_atr_len)
    v1_trend, v1_dir = ta.supertrend_pine(hl2, c, atr_v1, p.kf_atr_mult)
    v1_buf = np.where(v1_dir == 1, v1_trend + atr_v1 * p.kf_buf_mult, v1_trend - atr_v1 * p.kf_buf_mult)

    # === KF45 主週期趨勢（在 45 分序列上算，再以「最後一根已完成」對齊回 15 分）===
    htf = ta.resample_ohlcv(df, p.kf_htf)
    a45 = ta.atr(htf["high"].to_numpy(), htf["low"].to_numpy(), htf["close"].to_numpy(), p.kf_atr_len)
    t45, d45 = ta.supertrend_pine(((htf["high"] + htf["low"]) / 2).to_numpy(), htf["close"].to_numpy(), a45, p.kf_atr_mult)
    htf["kf_trend"], htf["kf_dir"], htf["kf_atr"] = t45, d45, a45
    m = ta.merge_htf(df.index, tfm, htf, p.kf_htf, ["kf_trend", "kf_dir", "kf_atr"])
    kf_trend, kf_dir, kf_atr = m["kf_trend"], m["kf_dir"], m["kf_atr"]
    kf_buf = np.where(kf_dir == 1, kf_trend + kf_atr * p.kf_buf_mult,
                      np.where(kf_dir == -1, kf_trend - kf_atr * p.kf_buf_mult, np.nan))
    kf_prev = ta.shift(kf_dir)
    kf_flip = (kf_dir != kf_prev) & ~np.isnan(kf_dir) & ~np.isnan(kf_prev)

    # === KC 通道（極寬版）===
    kc_mid = ta.ema(c, p.kc_len)
    kc_atr = ta.rma(ta.true_range(h, l, c, False), p.kc_len)
    kc_upper = ta.ema(kc_mid + p.kc_mult_up * kc_atr, p.kc_smooth)
    kc_lower = ta.ema(kc_mid - p.kc_mult_dn * kc_atr, p.kc_smooth)
    atr14 = ta.atr(h, l, c, 14)

    # === 供需區（swing box，最多 5 個，跌破/突破即刪）===
    piv_low = ta.pivot_low(l, p.swing_len, p.swing_len)
    piv_high = ta.pivot_high(h, p.swing_len, p.swing_len)
    in_demand = np.zeros(n, dtype=bool)
    in_supply = np.zeros(n, dtype=bool)
    demand: list[tuple[float, float]] = []
    supply: list[tuple[float, float]] = []
    for i in range(n):
        if not np.isnan(piv_low[i]) and not np.isnan(atr14[i]):
            demand.append((piv_low[i] - atr14[i] * 0.5, piv_low[i] + atr14[i] * 0.5))
            if len(demand) > p.max_boxes:
                demand.pop(0)
        if not np.isnan(piv_high[i]) and not np.isnan(atr14[i]):
            supply.append((piv_high[i] - atr14[i] * 0.5, piv_high[i] + atr14[i] * 0.5))
            if len(supply) > p.max_boxes:
                supply.pop(0)
        ci = c[i]
        if demand:
            demand = [b for b in demand if not (ci < b[0])]
            in_demand[i] = any(b[0] <= ci <= b[1] for b in demand)
        if supply:
            supply = [b for b in supply if not (ci > b[1])]
            in_supply[i] = any(b[0] <= ci <= b[1] for b in supply)

    # === 百分比概率評分 ===
    rsi = ta.rsi(c, 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        trend_strength = np.abs(c - kc_mid) / kc_atr
    base = np.minimum(60.0, 40.0 + trend_strength * 4)
    ph, rh = ta.highest(c, p.div_len), ta.highest(rsi, p.div_len)
    pl, rl = ta.lowest(c, p.div_len), ta.lowest(rsi, p.div_len)
    bear_div = (ph > ta.shift(ph, p.div_len)) & (rh < ta.shift(rh, p.div_len))
    bull_div = (pl < ta.shift(pl, p.div_len)) & (rl > ta.shift(rl, p.div_len))
    div_score = np.where(bear_div | bull_div, 20.0, 0.0)
    rsi_score = np.where((rsi > 70) | (rsi < 30), 8.0, 0.0)
    vol_score = np.where(v > ta.sma(v, 20) * 1.5, 5.0, 0.0)

    h4 = ta.resample_ohlcv(df, p.mtf_tf)
    h4["trend"] = np.sign(h4["close"].to_numpy() - ta.ema(h4["close"].to_numpy(), p.mtf_ema_len))
    htf_trend = ta.merge_htf(df.index, tfm, h4, p.mtf_tf, ["trend"])["trend"]
    mtf_score = np.where(np.sign(c - kf_trend) == htf_trend, 12.0, 0.0)
    sd_score = np.where(in_demand | in_supply, 10.0, 0.0)
    in_kf_band = ((kf_dir == 1) & (c >= kf_trend) & (c <= kf_buf)) | ((kf_dir == -1) & (c <= kf_trend) & (c >= kf_buf))
    kf_score = np.where(in_kf_band, 8.0, 0.0)
    total = base + div_score + rsi_score + vol_score + mtf_score + sd_score + kf_score

    prob_long = bull_div & (rsi < 35) & (total >= p.prob_threshold) & (kf_dir == 1)
    prob_short = bear_div & (rsi > 65) & (total >= p.prob_threshold) & (kf_dir == -1)

    # === KF 精準入場主信號 ===
    long_kf = (kf_dir == 1) & (v1_dir == 1) & in_kf_band & (c > o)
    short_kf = (kf_dir == -1) & (v1_dir == -1) & in_kf_band & (c < o)

    # === 聯動信號（百分比 / 恐懼貪婪 / MACD 三選二）===
    macd_line, sig_line = ta.macd(c)
    eth_long, eth_short = ta.crossover(macd_line, sig_line), ta.crossunder(macd_line, sig_line)
    prob_long_cond = (total >= p.prob_threshold) & (kf_dir == 1)
    prob_short_cond = (total >= p.prob_threshold) & (kf_dir == -1)
    link_long = prob_long_cond.astype(int) + ((rsi < 30) & p.use_fear_greed).astype(int) + (eth_long & p.use_eth_opt).astype(int)
    link_short = prob_short_cond.astype(int) + ((rsi > 70) & p.use_fear_greed).astype(int) + (eth_short & p.use_eth_opt).astype(int)
    linkage_long = p.use_linkage & (link_long >= 2)
    linkage_short = p.use_linkage & (link_short >= 2)

    # === 順勢加倉 ===
    long_add = p.use_add_on & (kf_dir == 1) & in_kf_band & (rsi < 35)
    short_add = p.use_add_on & (kf_dir == -1) & in_kf_band & (rsi > 65)

    # === 走人三層 + 分層冷卻（V5.1.3）===
    l1u, l2u, l3u = kc_upper + atr14, kc_upper + atr14 * 2, kc_upper + atr14 * 3
    l1d, l2d, l3d = kc_lower - atr14, kc_lower - atr14 * 2, kc_lower - atr14 * 3
    hit_long = np.where(h > l3u, 3, np.where(h > l2u, 2, np.where(h > l1u, 1, 0)))
    hit_short = np.where(l < l3d, 3, np.where(l < l2d, 2, np.where(l < l1d, 1, 0)))
    wll = np.zeros(n, bool); wml = np.zeros(n, bool); whl = np.zeros(n, bool)
    wls = np.zeros(n, bool); wms = np.zeros(n, bool); whs = np.zeros(n, bool)
    cd = p.walk_cooldown
    last_l = last_s = 0
    lvl_l = lvl_s = 0
    for i in range(n):
        if kf_flip[i]:
            last_l = i - cd
            last_s = i - cd
        if i - last_l >= cd:
            lvl_l = 0
        if i - last_s >= cd:
            lvl_s = 0
        hl_, hs_ = int(hit_long[i]), int(hit_short[i])
        wll[i] = hl_ == 1 and lvl_l < 1
        wml[i] = hl_ == 2 and lvl_l < 2
        whl[i] = hl_ == 3 and lvl_l < 3
        wls[i] = hs_ == 1 and lvl_s < 1
        wms[i] = hs_ == 2 and lvl_s < 2
        whs[i] = hs_ == 3 and lvl_s < 3
        if wll[i] or wml[i] or whl[i]:
            lvl_l, last_l = hl_, i
        if wls[i] or wms[i] or whs[i]:
            lvl_s, last_s = hs_, i

    # === 開倉信號合成 ===
    if p.entry_mode == "kf":
        long_entry, short_entry = long_kf, short_kf
    else:
        long_entry = long_kf | linkage_long | prob_long
        short_entry = short_kf | linkage_short | prob_short
    if p.require_kf_dir_all:
        long_entry = long_entry & (kf_dir == 1)
        short_entry = short_entry & (kf_dir == -1)

    out = df.copy()
    out["v1_trend"], out["v1_dir"], out["v1_buf"] = v1_trend, v1_dir, v1_buf
    out["kf_trend"], out["kf_dir"], out["kf_buf"], out["kf_flip"] = kf_trend, kf_dir, kf_buf, kf_flip
    out["kc_upper"], out["kc_lower"], out["kc_mid"], out["atr14"], out["rsi"] = kc_upper, kc_lower, kc_mid, atr14, rsi
    out["in_kf_band"], out["in_demand"], out["in_supply"], out["total_score"] = in_kf_band, in_demand, in_supply, total
    out["long_kf"], out["short_kf"], out["linkage_long"], out["linkage_short"] = long_kf, short_kf, linkage_long, linkage_short
    out["prob_long"], out["prob_short"], out["long_add"], out["short_add"] = prob_long, prob_short, long_add, short_add
    out["long_entry"], out["short_entry"] = long_entry, short_entry
    out["walk_low_long"], out["walk_mid_long"], out["walk_high_long"] = wll, wml, whl
    out["walk_low_short"], out["walk_mid_short"], out["walk_high_short"] = wls, wms, whs
    return out


def first_valid_index(sig: pd.DataFrame) -> int:
    ok = sig[["kf_buf", "v1_trend", "kc_upper", "kc_lower", "atr14", "rsi"]].notna().all(axis=1).to_numpy()
    return int(np.argmax(ok)) if ok.any() else len(sig)


def simulate(sig: pd.DataFrame, ep: BGTEngineParams, start_idx: int) -> BacktestResult:
    idx = sig.index
    o, h, l, c = (sig[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    g = {k: sig[k].to_numpy() for k in (
        "long_entry", "short_entry", "long_add", "short_add", "kf_flip",
        "walk_low_long", "walk_mid_long", "walk_high_long", "walk_low_short", "walk_mid_short", "walk_high_short")}
    kc_upper, kc_lower = sig["kc_upper"].to_numpy(), sig["kc_lower"].to_numpy()
    n = len(c)
    book = Book(ep.initial_capital, ep.commission_pct)
    slip = ep.slippage_ticks * ep.tick_size
    stop: tuple[float, int] | None = None
    eq = np.full(n, np.nan)

    def exit_px(side: int, px: float) -> float:
        return px - slip if side > 0 else px + slip

    for i in range(start_idx, n):
        t = idx[i]
        # 1) 前一根掛的 KC 止損，本根觸價即成交（跳空就用開盤價）
        if stop is not None and book.entries:
            lvl, side = stop
            if side > 0 and l[i] <= lvl:
                book.close_all(exit_px(1, min(o[i], lvl)), i, t, "KC止損")
            elif side < 0 and h[i] >= lvl:
                book.close_all(exit_px(-1, max(o[i], lvl)), i, t, "KC止損")
        stop = None

        # 2) 收盤時依「成交前」的部位狀態決定訂單（Pine 的 strategy.position_size 就是這個）
        pos = book.position()
        n_open = len(book.entries)
        equity_now = book.equity(c[i])
        orders: list[tuple] = []
        next_stop = None
        if pos == 0:
            if g["long_entry"][i]:
                orders.append(("entry", 1))
            elif g["short_entry"][i]:
                orders.append(("entry", -1))
        else:
            side = 1 if pos > 0 else -1
            if side > 0 and g["long_add"][i] and n_open < ep.pyramiding:
                orders.append(("entry", 1))
            if side < 0 and g["short_add"][i] and n_open < ep.pyramiding:
                orders.append(("entry", -1))
            if ep.use_kc_stop:
                next_stop = (kc_lower[i], 1) if side > 0 else (kc_upper[i], -1)
            if (side > 0 and g["walk_low_long"][i]) or (side < 0 and g["walk_low_short"][i]):
                orders.append(("reduce", ep.walk_low_reduce, "低概率走人"))
            if (side > 0 and g["walk_mid_long"][i]) or (side < 0 and g["walk_mid_short"][i]):
                orders.append(("reduce", ep.walk_mid_reduce, "中概率走人"))
            if (side > 0 and g["walk_high_long"][i]) or (side < 0 and g["walk_high_short"][i]):
                orders.append(("close_all", "高概率走人"))
            if g["kf_flip"][i]:
                orders.append(("close_all", "趨勢切換"))

        # 3) 全部在本根收盤成交
        for od in orders:
            if od[0] == "entry":
                side = od[1]
                qty = equity_now * ep.entry_frac / c[i]
                if qty > 0:
                    book.open(side, qty, c[i] + slip * side, i, t)
            elif od[0] == "reduce":
                for e in list(book.entries):
                    book.close(e, e.qty * od[1], exit_px(e.side, c[i]), i, t, od[2])
            elif od[0] == "close_all":
                for e in list(book.entries):
                    book.close(e, e.qty, exit_px(e.side, c[i]), i, t, od[1])
        stop = next_stop if (next_stop is not None and book.entries) else None
        eq[i] = book.equity(c[i])

    equity = pd.Series(eq, index=idx, name="strategy").iloc[start_idx:]
    return BacktestResult(equity=equity, trades=book.trades_df(), round_trips=book.trips_df(), start_idx=start_idx)
