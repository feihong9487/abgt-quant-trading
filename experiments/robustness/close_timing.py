"""收盤時點穩健性：日線收盤從 00:00 UTC 換成其他 23 個整點，ema50_plain 的結論還成立嗎？

做法：Binance 現貨 1h K 線（2019-09 起）→ 以 h:00 UTC 為日界合成日線（h = 0..23，開=首根開、高=最高、低=最低、收=末根收；
少於 18 根 1h 的日子丟掉）→ 同一個 ema50_plain、同一個組合引擎（BTC/ETH 50/50、費 0.10% 滑 0.05%、調倉門檻 0.15）、
同一套視窗與「視窗起點重新等權買入」持有基準（不再平衡只付進場費；月再平衡另付每月調整的手續費）。另外報 BTC、ETH 單獨。
資料：1h 共 17 個缺口、最長 5 小時，沒有任何一天因不足 18 根被丟；只有資料起點那根不完整日 K —— offset 1–17 丟掉，
offset 18–23 會保留一根只有 18–23 根 1h 的開頭日 K（只影響 2019 年的 EMA 起算，2020 年後可忽略）。

事前登記的判準（寫在跑之前；結果不得回頭改）：
  R1 廣度：組合十格（5 視窗 × 2 持有基準）全過的收盤時點數 ≥ 20/24 → 廣度穩健；< 12/24 → 脆弱
  R2 非挑選：00:00 的 OOS 與 OOS2 Sharpe 在 24 個時點中都不在前 2 名（若都在前 2 名 → 「00:00 剛好是好時點」，記為警示）
  R3 邊際：24 個時點「策略 Sharpe − 持有 Sharpe（兩基準較高者）」的中位數在 OOS 與 OOS2 都 > 0
  判定：R1 ≥ 20 且 R3 → 穩健；R1 < 12 或 R3 不成立 → 脆弱；其餘 → 混合。R2 只當警示，不改判定。
  另報「時點運氣」量級：24 個時點 OOS Sharpe 的全距與標準差，對照策略相對持有的 Sharpe 差。

資訊性（不參與判定、不是新策略）：24 個時點各 1/24 資本的「時間分散」組合。

用法：python experiments/robustness/close_timing.py
輸出：results/robustness/close_timing.json / .md
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from abgt import data as dt                                                   # noqa: E402
from abgt.metrics import summarize                                            # noqa: E402
from abgt.portfolio import align, simulate_portfolio                          # noqa: E402
from abgt.strategies.regime import RiskParams                                 # noqa: E402
from evaluate_portfolio import fresh_benches, window_metrics                  # noqa: E402
from evaluate_variant import load_variant, parse_windows                      # noqa: E402

SYMBOLS = ["BTC/USDT", "ETH/USDT"]
WINDOWS = parse_windows("IS:2020-01-01:2022-12-31,OOS:2023-01-01:,FULL:2020-01-01:,IS2:2020-07-01:2023-12-31,OOS2:2024-01-01:")
RP = dict(commission_pct=0.001, slippage_pct=0.0005, rebalance_threshold=0.15)
BENCHES = ("持有不再平衡", "持有月再平衡")
MIN_HOURS = 18


def daily_from_hourly(h: pd.DataFrame, offset: int) -> pd.DataFrame:
    """以 offset:00 UTC 為日界合成日線；索引 = 該日開始時刻。丟掉最後一根沒收完的與資料太少的日子。"""
    # 注意：freq="1D" 會忽略 offset（pandas 只對固定長度頻率套用 offset）→ 用 "24h" + epoch 起點
    g = h.resample("24h", origin="epoch", offset=f"{offset}h", label="left", closed="left")
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "volume": g["volume"].sum(), "n": g["close"].count()})
    last_hour = h.index[-1]
    d = d[d.index + pd.Timedelta(days=1) <= last_hour + pd.Timedelta(hours=1)]      # 收完的才算
    d = d[d["n"] >= MIN_HOURS].drop(columns="n").dropna()
    assert (d.index.hour == offset).all(), f"日界錯誤：offset {offset} 的 K 線開始時刻不是 {offset}:00"
    return d


def run(dfs: dict, weights: dict) -> tuple:
    mod = load_variant("ema50_plain")
    idx = align(dfs)
    regimes, warm = {}, []
    for s, df in dfs.items():
        b = mod.build(df, copy.deepcopy(mod.PARAMS))
        regimes[s] = b["in_regime"]
        warm.append(int(idx.searchsorted(df.index[int(b["warmup"])])))
    w0 = max(warm)
    rp = RiskParams(**RP)
    res = simulate_portfolio(dfs, regimes, rp, weights, w0)
    wm = window_metrics(res.equity, fresh_benches(dfs, weights, idx, rp.initial_capital, rp.commission_pct), idx, WINDOWS, w0, rp.initial_capital)
    row = {}
    for w, m in wm.items():
        s = m["strategy"]
        row[w] = {"sharpe": s["sharpe"], "cagr": s["cagr"], "dd": s["max_dd"], "calmar": s["calmar"],
                  "gate_cells": sum(int(m["bench"][b]["gate"]["passed"]) for b in BENCHES),
                  "hold_sharpe": max(m["bench"][b]["summary"]["sharpe"] for b in BENCHES),
                  "hold_dd": min(m["bench"][b]["summary"]["max_dd"] for b in BENCHES)}
    return row, res


def mark_at_midnight(res, dfs: dict, hourly: dict) -> pd.Series:
    """把一個收盤時點的組合在每天 00:00 UTC 以「當下真實價格」重新估值（不能用最後一次收盤值往前帶：
    最多 23 小時的陳舊價格會壓低量測到的波動、高估時間分散的 Sharpe）。
    做法：K 線 i 內（開盤成交後到收盤）持倉 qty_i 與現金 cash_i 不變，cash_i = 收盤權益 − Σ qty_i × 收盤價；
    00:00 落在 K 線 i 內 → 價值 = cash_i + Σ qty_i × 00:00 那一刻的 1h 收盤價（[23:00, 00:00) 那根的 close）。"""
    idx = align(dfs)
    eq = res.equity
    tr = res.trades
    qty = {s: pd.Series(0.0, index=idx) for s in dfs}
    if len(tr):
        for s in dfs:
            t = tr[tr["asset"] == s]
            delta = (t["qty"] * t["side"]).groupby(t["exit_time"]).sum()
            qty[s] = delta.reindex(idx, fill_value=0.0).cumsum()
    close = {s: dfs[s]["close"].reindex(idx) for s in dfs}
    cash = eq - sum(qty[s].reindex(eq.index) * close[s].reindex(eq.index) for s in dfs)
    starts = eq.index
    mids = pd.date_range(starts[0].ceil("1D"), (starts[-1] + pd.Timedelta(days=1)).floor("1D"), freq="1D")
    pos = starts.searchsorted(mids, side="left") - 1                # 00:00 之前開始的那根 K 線；00:00 剛好是成交時點時取成交前持倉（不把當天交易成本提前算進來）
    ok = pos >= 0
    mids, pos = mids[ok], pos[ok]
    val = cash.to_numpy()[pos].copy()
    for s in dfs:
        px = hourly[s]["close"].reindex(mids - pd.Timedelta(hours=1)).ffill().to_numpy()   # [23:00, 00:00) 的收盤 = 00:00 的價格
        val += qty[s].reindex(starts).to_numpy()[pos] * px
    return pd.Series(val, index=mids).dropna()


def fmt(x, p=False, d=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x * 100:+.1f}%" if p else f"{x:.{d}f}"


def main() -> int:
    hourly = {s: dt.load_cached("data", s, "1h", "2019-09-01") for s in SYMBOLS}
    if any(v is None for v in hourly.values()):
        print("缺 1h 快取：先跑 abgt.data.fetch_ohlcv(sym, '1h', '2019-09-01', 'data', exchange_id='binance')")
        return 2
    # 對照：offset 0 合成日線 vs 交易所日線
    sanity = {}
    for s in SYMBOLS:
        syn = daily_from_hourly(hourly[s], 0)
        ref = dt.load_cached("data", s, "1d", "2019-09-01")
        common = syn.index.intersection(ref.index)
        sanity[s] = {"days": int(len(common)), "max_close_diff_pct": float((syn.loc[common, "close"] / ref.loc[common, "close"] - 1).abs().max() * 100)}
    print("對照（offset 0 合成 vs 交易所日線）：", sanity)

    rows, curves = {}, {}
    for h in range(24):
        dfs = {s: daily_from_hourly(hourly[s], h) for s in SYMBOLS}
        port, res = run(dfs, {"BTC/USDT": 0.5, "ETH/USDT": 0.5})
        btc, _ = run({"BTC/USDT": dfs["BTC/USDT"]}, {"BTC/USDT": 1.0})
        eth, _ = run({"ETH/USDT": dfs["ETH/USDT"]}, {"ETH/USDT": 1.0})
        rows[h] = {"port": port, "BTC": btc, "ETH": eth, "turnover": res.extra["notional_turnover_per_year"], "fills": res.extra["n_fills"]}
        curves[h] = mark_at_midnight(res, dfs, hourly)
        if h == 0:
            rows_eq0 = res.equity
        print(f"{h:02d}:00  組合 Sharpe " + " ".join(f"{w} {fmt(port[w]['sharpe'])}" for w in ("IS", "OOS", "FULL", "IS2", "OOS2"))
              + f" | 十格 {sum(port[w]['gate_cells'] for w in port)}/10 | BTC OOS {fmt(btc['OOS']['sharpe'])} ETH OOS {fmt(eth['OOS']['sharpe'])}", flush=True)

    # ---- 判準 ----
    full_pass = [h for h in rows if sum(rows[h]["port"][w]["gate_cells"] for w in rows[h]["port"]) == 10]
    def rank_of_zero(w):
        vals = sorted(((rows[h]["port"][w]["sharpe"], h) for h in rows), reverse=True)
        return [h for _, h in vals].index(0) + 1
    r2 = {w: rank_of_zero(w) for w in ("OOS", "OOS2")}
    edge = {w: float(np.median([rows[h]["port"][w]["sharpe"] - rows[h]["port"][w]["hold_sharpe"] for h in rows])) for w in ("IS", "OOS", "FULL", "IS2", "OOS2")}
    R1, R3 = len(full_pass), edge["OOS"] > 0 and edge["OOS2"] > 0
    R2_warn = r2["OOS"] <= 2 and r2["OOS2"] <= 2
    verdict = "穩健" if (R1 >= 20 and R3) else ("脆弱" if (R1 < 12 or not R3) else "混合")
    spread = {w: {"min": float(min(rows[h]["port"][w]["sharpe"] for h in rows)), "max": float(max(rows[h]["port"][w]["sharpe"] for h in rows)),
                  "median": float(np.median([rows[h]["port"][w]["sharpe"] for h in rows])), "std": float(np.std([rows[h]["port"][w]["sharpe"] for h in rows]))}
              for w in ("IS", "OOS", "FULL", "IS2", "OOS2")}
    single = {a: {w: {"median": float(np.median([rows[h][a][w]["sharpe"] for h in rows])), "min": float(min(rows[h][a][w]["sharpe"] for h in rows)),
                      "max": float(max(rows[h][a][w]["sharpe"] for h in rows)),
                      "gate_pass_offsets": int(sum(rows[h][a][w]["gate_cells"] == 2 for h in rows)),
                      "median_edge": float(np.median([rows[h][a][w]["sharpe"] - rows[h][a][w]["hold_sharpe"] for h in rows]))}
                  for w in ("OOS", "OOS2", "FULL")} for a in ("BTC", "ETH")}

    # ---- 資訊性：時間分散（24 份各 1/24 資本）----
    grid = pd.date_range(max(c.index[0] for c in curves.values()).ceil("1D"), min(c.index[-1] for c in curves.values()).floor("1D"), freq="1D")
    combo = sum(curves[h].reindex(grid) / curves[h].reindex(grid).iloc[0] for h in curves) / 24 * 10000
    # 自我檢查：offset 0 在 00:00 的重新估值必須等於它自己的收盤權益（K 線 i 的收盤 = 隔天 00:00）
    e0 = rows_eq0.copy()
    e0.index = e0.index + pd.Timedelta(days=1)
    common = e0.index.intersection(curves[0].index)
    mark_check = float((curves[0].reindex(common) / e0.reindex(common) - 1).abs().max())
    print(f"重新估值自我檢查（offset 0）：最大相對差 {mark_check:.2e}")
    assert mark_check < 1e-9, "offset 0 的 00:00 估值必須等於它自己的收盤權益"
    tdiv = {}
    for name, ws, we in WINDOWS:
        seg = combo.loc[ws:we] if we else combo.loc[ws:]
        if len(seg) > 30:
            sm = summarize(seg / seg.iloc[0] * 10000)
            tdiv[name] = {"sharpe": sm["sharpe"], "dd": sm["max_dd"], "cagr": sm["cagr"]}

    out = {"sanity": sanity, "rows": rows, "criteria": {"R1_full_pass_offsets": R1, "R1_list": full_pass, "R2_rank_of_00": r2, "R2_warning": R2_warn,
           "R3_median_edge": edge, "R3": R3, "verdict": verdict}, "spread": spread, "single": single, "time_diversified": tdiv, "mark_check_offset0": mark_check}
    os.makedirs("results/robustness", exist_ok=True)
    with open("results/robustness/close_timing.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)

    L = ["# 收盤時點穩健性（ema50_plain，BTC/ETH 50/50 現貨）", "",
         f"對照：offset 0 合成日線 vs 交易所日線，收盤價最大差 BTC {sanity['BTC/USDT']['max_close_diff_pct']:.4f}%、ETH {sanity['ETH/USDT']['max_close_diff_pct']:.4f}%。", "",
         f"## 判定：**{verdict}**", "",
         f"- R1 十格全過的收盤時點：**{R1}/24**（≥ 20 穩健、< 12 脆弱）",
         f"- R2 00:00 的名次：OOS 第 {r2['OOS']}、OOS2 第 {r2['OOS2']}（/24）" + ("　⚠ 兩者都在前 2" if R2_warn else ""),
         f"- R3 策略 − 持有 Sharpe 的中位數：OOS {edge['OOS']:+.2f}、OOS2 {edge['OOS2']:+.2f}（IS {edge['IS']:+.2f}、FULL {edge['FULL']:+.2f}、IS2 {edge['IS2']:+.2f}）", "",
         "## 時點運氣的量級（組合 Sharpe 在 24 個收盤時點的分布）", "",
         "| 視窗 | 最低 | 中位數 | 最高 | 標準差 | 00:00 | 持有（00:00） |", "|---|---|---|---|---|---|---|"]
    for w in ("IS", "OOS", "FULL", "IS2", "OOS2"):
        sp = spread[w]
        L.append(f"| {w} | {fmt(sp['min'])} | {fmt(sp['median'])} | {fmt(sp['max'])} | {fmt(sp['std'])} | {fmt(rows[0]['port'][w]['sharpe'])} | {fmt(rows[0]['port'][w]['hold_sharpe'])} |")
    L += ["", "## 每個收盤時點（組合）", "", "| UTC | 台北 | IS | OOS | FULL | IS2 | OOS2 | 十格 | OOS 回撤 | 換手 ×/年 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for h in range(24):
        p = rows[h]["port"]
        L.append(f"| {h:02d}:00 | {(h + 8) % 24:02d}:00 | " + " | ".join(fmt(p[w]["sharpe"]) for w in ("IS", "OOS", "FULL", "IS2", "OOS2"))
                 + f" | {sum(p[w]['gate_cells'] for w in p)}/10 | {fmt(p['OOS']['dd'], True)} | {rows[h]['turnover']:.1f} |")
    L += ["", "## 單標的（24 個時點）", "", "| 標的 | 視窗 | Sharpe 最低 / 中位 / 最高 | 兩基準都過的時點 | 對持有 Sharpe 差中位數 |", "|---|---|---|---|---|"]
    for a in ("BTC", "ETH"):
        for w in ("OOS", "OOS2", "FULL"):
            x = single[a][w]
            L.append(f"| {a} | {w} | {fmt(x['min'])} / {fmt(x['median'])} / {fmt(x['max'])} | {x['gate_pass_offsets']}/24 | {x['median_edge']:+.2f} |")
    L += ["", "## 資訊性：時間分散（24 個收盤時點各 1/24 資本，每天 00:00 以真實價格重新估值；不參與判定）", "", "| 視窗 | Sharpe | 回撤 | CAGR |", "|---|---|---|---|"]
    for w, x in tdiv.items():
        L.append(f"| {w} | {fmt(x['sharpe'])} | {fmt(x['dd'], True)} | {fmt(x['cagr'], True)} |")
    with open("results/robustness/close_timing.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    print("→ results/robustness/close_timing.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
