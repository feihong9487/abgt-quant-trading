"""槓桿分析 — 標的分散（diversification）。

問題：在 ema50_plain（close > EMA50 滿倉否則空手、只做多、收盤決策次日開盤成交）之上，
等權加入 ETH / SOL / BNB 各能換到多少 Sharpe？回撤有沒有變深？哪個組合在 OOS 最穩？
BTC 加權（事前用市值排名，不用回測挑）有沒有理由取代等權？

設定（上線口徑）：現貨日線為主（永續為對照）、費 0.10%/邊、滑 0.05%、調倉門檻 0.15、初始 10,000。
組合引擎 abgt.portfolio.simulate_portfolio：一本共用權益、每標的目標曝險 = 權重 × regime；
基準 = 同權重買入持有「不再平衡」與「月再平衡」（月再平衡付同樣手續費）。
十格 = IS 2020-01→2022-12 / OOS 2023-01→今 / FULL / IS2 2020-07→2023-12 / OOS2 2024-01→今 × 兩種持有基準；
閘門 = Sharpe > 持有 且 Calmar > 持有 且 最大回撤 < 持有（abgt.benchmark.gate mode="risk"）。

視窗注意：SOL 現貨 2020-08-11 起、永續 2020-09-14 起；BNB 永續 2020-02-10 起（現貨 2019-09 起）。
含 SOL 的組合其 IS / IS2 / FULL 起點被 SOL 的 EMA50 暖機拉到 2020-09 末（現貨）/ 2020-11 初（永續），
所以另做「共同起算」面板：四個組合都從最晚暖機那天起算，讓 BTC 單獨 / BTC+ETH 跟含 SOL/BNB 的組合同視窗比較。

沒有未來資料：regime 由各標的自己的收盤價計算（EMA50 SMA 起算），組合權重在進場那根用當時權益決定。

重跑：
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/levers/diversification.py
輸出：results/levers/diversification.json、results/levers/diversification.md
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from abgt import data as dt                                                      # noqa: E402
from abgt.benchmark import buy_and_hold                                          # noqa: E402
from abgt.metrics import num, pct, summarize                                     # noqa: E402
from abgt.portfolio import align, buy_and_hold_portfolio, simulate_portfolio     # noqa: E402
from abgt.strategies.regime import RiskParams, simulate_regime                   # noqa: E402
from evaluate_portfolio import window_metrics                                    # noqa: E402
from evaluate_variant import ALT_WINDOWS, DATA_SINCE, DEFAULT_WINDOWS, load_variant, parse_windows   # noqa: E402

# ---------- 設定（全部事前固定） ----------
FEE = 0.001          # 0.10%/邊
SLIP = 0.0005        # 0.05%
REBAL_TH = 0.15
INITIAL = 10_000.0
WINDOWS = parse_windows(DEFAULT_WINDOWS + "," + ALT_WINDOWS)
WNAMES = [w[0] for w in WINDOWS]
MARKETS = {
    "spot": {"BTC": "BTC/USDT", "ETH": "ETH/USDT", "SOL": "SOL/USDT", "BNB": "BNB/USDT"},
    "perp": {"BTC": "BTC/USDT:USDT", "ETH": "ETH/USDT:USDT", "SOL": "SOL/USDT:USDT", "BNB": "BNB/USDT:USDT"},
}
COMBOS = [("BTC",), ("BTC", "ETH"), ("BTC", "ETH", "SOL"), ("BTC", "ETH", "SOL", "BNB")]
# BTC 加權：只用「市值排名」這種非績效理由事前定，不用回測挑。60/40 與 50/30/20 是任務給的例子；四標的用排名線性權重 4:3:2:1。
WEIGHTED = {
    "BTC60/ETH40": {"BTC": 0.6, "ETH": 0.4},
    "BTC50/ETH30/SOL20": {"BTC": 0.5, "ETH": 0.3, "SOL": 0.2},
    "BTC40/ETH30/SOL20/BNB10": {"BTC": 0.4, "ETH": 0.3, "SOL": 0.2, "BNB": 0.1},
}
VARIANTS = ["ema50_plain", "vol_atr_channel"]
BENCH = {"hold_nr": "持有不再平衡", "hold_mr": "持有月再平衡"}
STRESS = {
    "費0.10%+滑0.20%": {"commission_pct": 0.001, "slippage_pct": 0.002},
    "延遲1根": {"exec_delay_bars": 1},
}
OOS_YEARS = [2023, 2024, 2025, 2026]
OUT_DIR = os.path.join("results", "levers")
KEY = "diversification"


def combo_name(assets) -> str:
    return "+".join(assets)


def equal_weights(assets) -> dict[str, float]:
    return {a: 1.0 / len(assets) for a in assets}


def rp_live(**over) -> RiskParams:
    base = dict(commission_pct=FEE, slippage_pct=SLIP, rebalance_threshold=REBAL_TH, initial_capital=INITIAL)
    base.update(over)
    return RiskParams(**base)


# ---------- 資料與 regime ----------
def build_market(market: str, variant_id: str):
    mod = load_variant(variant_id)
    dfs, regimes, warm_time = {}, {}, {}
    for a, sym in MARKETS[market].items():
        df = dt.load_cached("data", sym, "1d", DATA_SINCE)
        if df is None:
            raise SystemExit(f"缺快取 {sym}，先跑 evaluate_variant --fetch")
        built = mod.build(df, copy.deepcopy(mod.PARAMS))
        dfs[a] = df
        regimes[a] = built["in_regime"]
        warm_time[a] = df.index[int(built["warmup"])]
    return dfs, regimes, warm_time


# ---------- 組合 ----------
def cell(w: dict) -> dict:
    s = w["strategy"]
    out = {"start": w["start"], "end": w["end"], "net_return": s["net_return"], "cagr": s["cagr"],
           "max_dd": s["max_dd"], "sharpe": s["sharpe"], "calmar": s["calmar"]}
    for bkey, bname in BENCH.items():
        b = w["bench"][bname]
        bs, g = b["summary"], b["gate"]
        out[bkey] = {"sharpe": bs["sharpe"], "calmar": bs["calmar"], "max_dd": bs["max_dd"], "net_return": bs["net_return"],
                     "rel_dd": b["rel_dd_vs_hold"], "gate": g["passed"],
                     "sharpe_pass": g["sharpe_pass"], "calmar_pass": g["calmar_pass"], "dd_pass": g["overall_dd_pass"]}
    return out


def run_portfolio(dfs, regimes, weights: dict[str, float], start_time, rp: RiskParams) -> dict:
    sub = {a: dfs[a] for a in weights}
    reg = {a: regimes[a] for a in weights}
    idx = align(sub)
    start_idx = int(idx.searchsorted(start_time))
    res = simulate_portfolio(sub, reg, rp, weights, start_idx)
    benches = {BENCH["hold_nr"]: buy_and_hold_portfolio(sub, weights, idx, start_idx, rp.initial_capital, rp.commission_pct, None),
               BENCH["hold_mr"]: buy_and_hold_portfolio(sub, weights, idx, start_idx, rp.initial_capital, rp.commission_pct, "M")}
    wm = window_metrics(res.equity, benches, idx, WINDOWS, start_idx, rp.initial_capital)
    e = res.extra
    windows = {w: cell(wm[w]) for w in WNAMES if w in wm}
    gates = sum(int(c[b]["gate"]) for c in windows.values() for b in BENCH)
    return {
        "weights": weights, "start": str(idx[start_idx].date()), "end": str(idx[-1].date()),
        "windows": windows, "gates": gates, "gates_max": 2 * len(windows),
        "n_fills": int(e["n_fills"]), "round_trips": int(len(res.round_trips)),
        "notional_turnover_per_year": e["notional_turnover_per_year"],
        "time_in_market": e["time_in_market"], "avg_exposure": e["avg_exposure"],
        "avg_exposure_by_asset": e["avg_exposure_by_asset"], "max_exposure_seen": e["max_exposure_seen"],
        "rel_dd_vs_hold_nr_full": e["rel_dd_vs_hold"],
        "_equity": res.equity, "_hold_nr": benches[BENCH["hold_nr"]], "_hold_mr": benches[BENCH["hold_mr"]],
    }


def run_single(df: pd.DataFrame, regime, rp: RiskParams, start_time) -> dict:
    start_idx = int(df.index.searchsorted(start_time))
    res = simulate_regime(df, regime, rp, start_idx)
    eq = res.equity
    out = {"start": str(df.index[start_idx].date()), "windows": {}, "_equity": eq}
    for wname, wstart, wend in WINDOWS:
        s0 = max(start_idx, int(df.index.searchsorted(pd.Timestamp(wstart, tz="UTC"))))
        s1 = int(df.index.searchsorted(pd.Timestamp(wend, tz="UTC"), side="right")) if wend else len(df)
        if s1 - s0 < 30:
            continue
        e = eq.loc[df.index[s0]:df.index[s1 - 1]]
        e = e / e.iloc[0] * rp.initial_capital
        bh = buy_and_hold(df["close"].iloc[s0:s1], rp.initial_capital, rp.commission_pct)
        s, b = summarize(e), summarize(bh)
        out["windows"][wname] = {"sharpe": s["sharpe"], "max_dd": s["max_dd"], "calmar": s["calmar"], "net_return": s["net_return"],
                                 "hold_sharpe": b["sharpe"], "hold_max_dd": b["max_dd"], "start": str(e.index[0].date())}
    return out


def yearly(eq: pd.Series, hold: pd.Series) -> dict:
    out = {}
    for y in OOS_YEARS:
        e = eq.loc[f"{y}-01-01":f"{y}-12-31"].dropna()
        h = hold.loc[f"{y}-01-01":f"{y}-12-31"].dropna()
        if len(e) < 30:
            continue
        s, b = summarize(e), summarize(h)
        out[str(y)] = {"sharpe": s["sharpe"], "max_dd": s["max_dd"], "net_return": s["net_return"],
                       "hold_sharpe": b["sharpe"], "hold_max_dd": b["max_dd"], "hold_net_return": b["net_return"],
                       "beat_hold_sharpe": bool(s["sharpe"] == s["sharpe"] and s["sharpe"] > b["sharpe"]),
                       "n_days": int(len(e))}
    return out


def daily_returns(eq: pd.Series) -> pd.Series:
    return eq.resample("1D").last().dropna().pct_change().dropna()


def paired_t(eq_a: pd.Series, eq_b: pd.Series, start: str, end: str) -> dict:
    """a − b 的日報酬差：均值、t 值（均值 / 標準誤）、天數。同一段日期、同一本引擎，配對檢定。"""
    d = (daily_returns(eq_a) - daily_returns(eq_b)).loc[start:end].dropna()
    t = float(d.mean() / d.std() * math.sqrt(len(d))) if len(d) > 10 and d.std() > 0 else float("nan")
    return {"mean_daily_diff_bp": float(d.mean() * 1e4), "t_daily": t, "n_days": int(len(d))}


def corr_matrix(series: dict[str, pd.Series], start: str | None = None) -> dict:
    df = pd.concat({k: daily_returns(v) for k, v in series.items()}, axis=1).dropna()
    if start:
        df = df.loc[start:]
    c = df.corr()
    return {"n_days": int(len(df)), "corr": {a: {b: float(c.loc[a, b]) for b in c.columns} for a in c.index}}


# ---------- 表格 ----------
def md_table(headers: list[str], rows: list[list]) -> str:
    def f(x):
        if x is None:
            return "—"
        if isinstance(x, bool):
            return "✓" if x else "✗"
        if isinstance(x, float):
            return "—" if math.isnan(x) else f"{x:.2f}"
        return str(x)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(f(x) for x in r) + " |")
    return "\n".join(lines)


def ten_cell_rows(name: str, r: dict) -> list[list]:
    rows = []
    for w, c in r["windows"].items():
        nr, mr = c["hold_nr"], c["hold_mr"]
        rows.append([name, w, c["start"], pct(c["net_return"]), pct(c["max_dd"]), num(c["sharpe"]), num(c["calmar"]),
                     f"{num(nr['sharpe'])} / {num(nr['calmar'])} / {pct(nr['max_dd'])}", pct(nr["rel_dd"]), nr["gate"],
                     f"{num(mr['sharpe'])} / {num(mr['calmar'])} / {pct(mr['max_dd'])}", pct(mr["rel_dd"]), mr["gate"]])
    return rows


TEN_HEADERS = ["組合", "區間", "起", "淨報酬", "回撤", "Sharpe", "Calmar", "持有NR S/C/DD", "落後NR", "閘NR", "持有MR S/C/DD", "落後MR", "閘MR"]


def compact_rows(name: str, r: dict) -> list:
    """一列：每個區間 Sharpe / 回撤 / 閘門數(0–2)。"""
    row = [name, r["start"]]
    for w in WNAMES:
        c = r["windows"].get(w)
        if c is None:
            row.append("—")
            continue
        g = int(c["hold_nr"]["gate"]) + int(c["hold_mr"]["gate"])
        row.append(f"{num(c['sharpe'])} / {pct(c['max_dd'])} / {g}")
    row.append(f"{r['gates']}/{r['gates_max']}")
    return row


COMPACT_HEADERS = ["組合", "起算"] + [f"{w} S/DD/閘" for w in WNAMES] + ["十格"]


def strip_private(d):
    if isinstance(d, dict):
        return {k: strip_private(v) for k, v in d.items() if not str(k).startswith("_")}
    if isinstance(d, list):
        return [strip_private(v) for v in d]
    if isinstance(d, (np.floating,)):
        return float(d)
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, (np.bool_,)):
        return bool(d)
    return d


# ---------- 主流程 ----------
def main() -> int:
    rp = rp_live()
    report = {"key": KEY, "date": pd.Timestamp.utcnow().strftime("%Y-%m-%d"), "config": {
        "fee_per_side": FEE, "slippage": SLIP, "rebalance_threshold": REBAL_TH, "initial": INITIAL,
        "windows": {w: [s, e] for w, s, e in WINDOWS}, "markets": MARKETS, "combos": [list(c) for c in COMBOS],
        "weighted": WEIGHTED, "variants": VARIANTS, "gate": "Sharpe > 持有 且 Calmar > 持有 且 最大回撤 < 持有（每格各自判定）",
        "engine": "abgt.portfolio.simulate_portfolio（共用權益、每標的目標曝險 = 權重 × regime、收盤判定次日開盤成交、只做多）",
        "risk_params": dataclasses.asdict(rp) | {"funding": None}}, "markets": {}}
    md = [f"# 槓桿分析：標的分散（{KEY}）", "",
          f"日期 {report['date']}｜費 {FEE:.2%}/邊 滑 {SLIP:.2%}｜調倉門檻 {REBAL_TH}｜十格 = IS / OOS / FULL / IS2 / OOS2 × 持有不再平衡（NR）/ 持有月再平衡（MR）｜閘門 = Sharpe、Calmar 高於同期持有且回撤更小",
          "", "重跑：`PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/levers/diversification.py`", ""]

    for market in ("spot", "perp"):
        mk = {"variants": {}}
        report["markets"][market] = mk
        for vid in VARIANTS:
            dfs, regimes, warm = build_market(market, vid)
            common_start = max(warm.values())
            vk = {"warmup_first_bar": {a: str(t.date()) for a, t in warm.items()},
                  "data_start": {a: str(dfs[a].index[0].date()) for a in dfs},
                  "common_start": str(common_start.date()), "native": {}, "common": {}, "weighted_native": {}, "weighted_common": {},
                  "single_native": {}, "single_common": {}}
            mk["variants"][vid] = vk
            equities_common, holds_common = {}, {}
            for assets in COMBOS:
                nm = combo_name(assets)
                w = equal_weights(assets)
                native_start = max(warm[a] for a in assets)
                vk["native"][nm] = run_portfolio(dfs, regimes, w, native_start, rp)
                vk["common"][nm] = run_portfolio(dfs, regimes, w, common_start, rp)
                equities_common[nm] = vk["common"][nm]["_equity"]
                holds_common[nm] = vk["common"][nm]["_hold_nr"]
            for nm, w in WEIGHTED.items():
                native_start = max(warm[a] for a in w)
                vk["weighted_native"][nm] = run_portfolio(dfs, regimes, w, native_start, rp)
                vk["weighted_common"][nm] = run_portfolio(dfs, regimes, w, common_start, rp)
            for a in dfs:
                vk["single_native"][a] = run_single(dfs[a], regimes[a], rp, warm[a])
                vk["single_common"][a] = run_single(dfs[a], regimes[a], rp, common_start)
            # 相關（共同起算後的單標的策略日報酬；另列持有日報酬）
            strat_ser = {a: vk["single_common"][a]["_equity"] for a in dfs}
            hold_ser = {a: dfs[a]["close"].loc[common_start:] for a in dfs}
            vk["corr"] = {"strategy_full": corr_matrix(strat_ser), "strategy_oos": corr_matrix(strat_ser, "2023-01-01"),
                          "hold_full": corr_matrix(hold_ser), "hold_oos": corr_matrix(hold_ser, "2023-01-01")}
            # 分散紅利：組合 Sharpe − 成分單標的 Sharpe 平均（共同起算）
            bonus = {}
            for assets in COMBOS:
                nm = combo_name(assets)
                bonus[nm] = {}
                for wname in WNAMES:
                    c = vk["common"][nm]["windows"].get(wname)
                    singles = [vk["single_common"][a]["windows"][wname]["sharpe"] for a in assets if wname in vk["single_common"][a]["windows"]]
                    if c is None or not singles:
                        continue
                    bonus[nm][wname] = {"combo_sharpe": c["sharpe"], "single_mean": float(np.mean(singles)), "bonus": c["sharpe"] - float(np.mean(singles)),
                                        "single_max": float(np.max(singles)), "single_min": float(np.min(singles))}
            vk["diversification_bonus"] = bonus
            # 增量（共同起算，等權，逐一加標的）
            inc = {}
            for i in range(1, len(COMBOS)):
                prev, cur = combo_name(COMBOS[i - 1]), combo_name(COMBOS[i])
                inc[f"{prev} → {cur}"] = {}
                for wname in WNAMES:
                    a, b = vk["common"][prev]["windows"].get(wname), vk["common"][cur]["windows"].get(wname)
                    if a is None or b is None:
                        continue
                    inc[f"{prev} → {cur}"][wname] = {"d_sharpe": b["sharpe"] - a["sharpe"], "d_max_dd_pp": (b["max_dd"] - a["max_dd"]) * 100,
                                                     "d_calmar": b["calmar"] - a["calmar"],
                                                     "d_hold_nr_sharpe": b["hold_nr"]["sharpe"] - a["hold_nr"]["sharpe"],
                                                     "d_hold_nr_dd_pp": (b["hold_nr"]["max_dd"] - a["hold_nr"]["max_dd"]) * 100}
            vk["increments"] = inc
            # 對 BTC 單獨（共同起算）的逐格 Δ 與配對 t：回答「多加標的比只做 BTC 好多少」
            vs_btc = {}
            btc_r = vk["common"]["BTC"]
            for assets in COMBOS[1:]:
                nm = combo_name(assets)
                r = vk["common"][nm]
                vs_btc[nm] = {}
                for wname in WNAMES:
                    a, b = r["windows"].get(wname), btc_r["windows"].get(wname)
                    if a is None or b is None:
                        continue
                    vs_btc[nm][wname] = {"d_sharpe": a["sharpe"] - b["sharpe"], "d_max_dd_pp": (a["max_dd"] - b["max_dd"]) * 100,
                                         "d_calmar": a["calmar"] - b["calmar"], "d_rel_dd_nr_pp": (a["hold_nr"]["rel_dd"] - b["hold_nr"]["rel_dd"]) * 100}
                    vs_btc[nm][wname].update(paired_t(r["_equity"], btc_r["_equity"], a["start"], a["end"]))
            vk["vs_btc_common"] = vs_btc
            # OOS 穩定度（共同起算面板；OOS/OOS2 起點與原生相同）
            stab = {}
            for nm, eq in equities_common.items():
                r = vk["common"][nm]
                yr = yearly(eq, holds_common[nm])
                oos_cells = [r["windows"][w] for w in ("OOS", "OOS2") if w in r["windows"]]
                stab[nm] = {"yearly": yr,
                            "oos_gates": sum(int(c[b]["gate"]) for c in oos_cells for b in BENCH),
                            "oos_min_sharpe": min(c["sharpe"] for c in oos_cells),
                            "oos_max_dd": r["windows"]["OOS"]["max_dd"],
                            "oos_rel_dd_nr": r["windows"]["OOS"]["hold_nr"]["rel_dd"],
                            "worst_year_sharpe": min(v["sharpe"] for v in yr.values()),
                            "years_beat_hold": sum(int(v["beat_hold_sharpe"]) for v in yr.values()), "n_years": len(yr)}
            vk["oos_stability"] = stab
            # 壓力（只做現貨 ema50_plain 等權、原生起算）
            if market == "spot" and vid == "ema50_plain":
                st = {}
                for sname, ov in STRESS.items():
                    st[sname] = {}
                    for assets in COMBOS:
                        nm = combo_name(assets)
                        r = run_portfolio(dfs, regimes, equal_weights(assets), max(warm[a] for a in assets), rp_live(**ov))
                        st[sname][nm] = {w: {"sharpe": c["sharpe"], "max_dd": c["max_dd"], "gate_nr": c["hold_nr"]["gate"], "gate_mr": c["hold_mr"]["gate"]}
                                         for w, c in r["windows"].items()} | {"gates": r["gates"], "turnover": r["notional_turnover_per_year"]}
                vk["stress"] = st

    # ---------- Markdown ----------
    sp = report["markets"]["spot"]["variants"]["ema50_plain"]
    pp = report["markets"]["perp"]["variants"]["ema50_plain"]
    spc = report["markets"]["spot"]["variants"]["vol_atr_channel"]

    md += ["## 1. 資料與視窗", "",
           md_table(["標的", "現貨資料起", "現貨 EMA50 首個可用日", "永續資料起", "永續 EMA50 首個可用日"],
                    [[a, sp["data_start"][a], sp["warmup_first_bar"][a], pp["data_start"][a], pp["warmup_first_bar"][a]] for a in ("BTC", "ETH", "SOL", "BNB")]),
           "",
           f"原生起算：每個組合從成分裡最晚暖機那天起算（現貨 BTC、BTC+ETH 的 IS 從 2020-01-01；含 SOL 的組合 IS 從 {sp['native']['BTC+ETH+SOL']['start']}）。",
           f"共同起算：四個組合都從 {sp['common_start']}（現貨）/ {pp['common_start']}（永續）起算，IS / IS2 / FULL 因此變短，OOS / OOS2 起點不變。表中「起」欄是該格實際起始日。", ""]

    md += ["## 2. 主表：現貨 ema50_plain 等權，原生起算（十格）", "",
           md_table(TEN_HEADERS, sum((ten_cell_rows(nm, sp["native"][nm]) for nm in sp["native"]), [])), "",
           "讀法：持有欄為 Sharpe / Calmar / 回撤；「落後」= 策略/持有比值自高點最深回撤；閘 = 該格對該基準是否過閘門。", ""]

    md += ["## 3. 共同起算面板：現貨 ema50_plain 等權（四組合同視窗，公平比較）", "",
           md_table(TEN_HEADERS, sum((ten_cell_rows(nm, sp["common"][nm]) for nm in sp["common"]), [])), ""]

    inc_rows = []
    for step, per_w in sp["increments"].items():
        for w, d in per_w.items():
            inc_rows.append([step, w, f"{d['d_sharpe']:+.2f}", f"{d['d_max_dd_pp']:+.1f} pp", f"{d['d_calmar']:+.2f}", f"{d['d_hold_nr_sharpe']:+.2f}", f"{d['d_hold_nr_dd_pp']:+.1f} pp"])
    md += ["## 4. 逐一加標的的增量（共同起算，等權）", "",
           md_table(["加入", "區間", "ΔSharpe", "Δ回撤", "ΔCalmar", "Δ持有NR Sharpe", "Δ持有NR 回撤"], inc_rows), "",
           "符號：Δ回撤 = 新組合回撤 − 舊組合回撤（pp），**負值 = 回撤變深**，正值 = 變淺。持有欄同。", ""]
    vb_rows = []
    for nm, per_w in sp["vs_btc_common"].items():
        for w, d in per_w.items():
            vb_rows.append([nm, w, f"{d['d_sharpe']:+.2f}", f"{d['d_max_dd_pp']:+.1f} pp", f"{d['d_calmar']:+.2f}", f"{d['d_rel_dd_nr_pp']:+.1f} pp",
                            f"{d['mean_daily_diff_bp']:+.1f} bp", num(d["t_daily"]), d["n_days"]])
    md += ["### 4b. 每個組合對 BTC 單獨（共同起算）：逐格 Δ 與日報酬差配對 t", "",
           md_table(["組合", "區間", "ΔSharpe", "Δ回撤", "ΔCalmar", "Δ落後NR", "日報酬差均值", "t", "天數"], vb_rows), "",
           "讀法：t = 日報酬差均值 / 標準誤；|t| < 1.5 視為在 OOS 上與 BTC 單獨不可分辨（沿用第 4 輪門檻）。Δ回撤負值 = 比 BTC 單獨深。", ""]

    bonus_rows = []
    for nm, per_w in sp["diversification_bonus"].items():
        for w, d in per_w.items():
            bonus_rows.append([nm, w, num(d["combo_sharpe"]), num(d["single_mean"]), f"{d['bonus']:+.2f}", num(d["single_min"]), num(d["single_max"])])
    single_rows = []
    for a in ("BTC", "ETH", "SOL", "BNB"):
        row = [a, sp["single_common"][a]["start"]]
        for w in WNAMES:
            c = sp["single_common"][a]["windows"].get(w)
            row.append("—" if c is None else f"{num(c['sharpe'])} / {pct(c['max_dd'])}（持有 {num(c['hold_sharpe'])} / {pct(c['hold_max_dd'])}）")
        single_rows.append(row)
    cf, co = sp["corr"]["strategy_full"], sp["corr"]["strategy_oos"]
    hf, ho = sp["corr"]["hold_full"], sp["corr"]["hold_oos"]
    assets4 = ["BTC", "ETH", "SOL", "BNB"]
    corr_rows = [[f"{a}–{b}", num(cf["corr"][a][b]), num(co["corr"][a][b]), num(hf["corr"][a][b]), num(ho["corr"][a][b])]
                 for i, a in enumerate(assets4) for b in assets4[i + 1:]]
    md += ["## 5. 單標的（共同起算、同成本）、分散紅利、相關", "",
           md_table(["標的", "起算"] + [f"{w} 策略 S/DD（持有）" for w in WNAMES], single_rows), "",
           md_table(["組合", "區間", "組合 Sharpe", "成分單標的平均", "紅利", "成分最低", "成分最高"], bonus_rows), "",
           md_table(["配對", f"策略日報酬相關 FULL（n={cf['n_days']}）", f"策略 OOS（n={co['n_days']}）", "持有 FULL", "持有 OOS"], corr_rows), ""]

    turn_rows = []
    for panel, label in (("native", "原生"), ("common", "共同")):
        for nm, r in sp[panel].items():
            turn_rows.append([f"{nm}（{label}）", r["start"], r["n_fills"], r["round_trips"], num(r["notional_turnover_per_year"], 1), f"{r['time_in_market']:.0%}",
                              num(r["avg_exposure"]), ", ".join(f"{a} {v:.2f}" for a, v in r["avg_exposure_by_asset"].items()), pct(r["rel_dd_vs_hold_nr_full"])])
    md += ["## 6. 換手與曝險（現貨 ema50_plain 等權）", "",
           md_table(["組合", "起算", "成交", "回合", "名目換手/年", "在場", "平均曝險", "各標的平均曝險", "相對持有NR最深落後（全期）"], turn_rows), ""]

    w_rows = []
    for panel, label in (("weighted_native", "原生"), ("weighted_common", "共同")):
        eq_panel = "native" if panel == "weighted_native" else "common"
        for nm, r in sp[panel].items():
            eq_nm = combo_name(list(r["weights"]))
            eqr = sp[eq_panel][eq_nm]
            w_rows.append(compact_rows(f"{nm}（{label}）", r))
            w_rows.append(compact_rows(f"等權 {eq_nm}（{label}）", eqr))
            d = [f"{(r['windows'][w]['sharpe'] - eqr['windows'][w]['sharpe']):+.2f}" if w in r["windows"] and w in eqr["windows"] else "—" for w in WNAMES]
            w_rows.append([f"　Δ（加權 − 等權）", ""] + d + [f"{r['gates'] - eqr['gates']:+d}"])
    md += ["## 7. BTC 加權（事前市值排名權重）對等權", "",
           md_table(COMPACT_HEADERS, w_rows), "",
           "權重理由：市值排名（BTC > ETH > SOL ≈ BNB），60/40 與 50/30/20 為任務給定、四標的 4:3:2:1 線性排名；不得依本表挑權重。", ""]

    md += ["## 8. 對照：現貨 vol_atr_channel 等權", "",
           md_table(COMPACT_HEADERS, [compact_rows(f"{nm}（原生）", r) for nm, r in spc["native"].items()] + [compact_rows(f"{nm}（共同）", r) for nm, r in spc["common"].items()]), ""]

    md += ["## 9. 對照：永續 ema50_plain 等權（同成本模型，不含資金費率）", "",
           md_table(COMPACT_HEADERS, [compact_rows(f"{nm}（原生）", r) for nm, r in pp["native"].items()] + [compact_rows(f"{nm}（共同）", r) for nm, r in pp["common"].items()]), ""]

    stab_rows = []
    for nm, s in sp["oos_stability"].items():
        yr = s["yearly"]
        stab_rows.append([nm, f"{s['oos_gates']}/4", num(s["oos_min_sharpe"]), pct(s["oos_max_dd"]), pct(s["oos_rel_dd_nr"])]
                         + [f"{num(yr[str(y)]['sharpe'])}（{num(yr[str(y)]['hold_sharpe'])}）/ {pct(yr[str(y)]['max_dd'])}" if str(y) in yr else "—" for y in OOS_YEARS]
                         + [num(s["worst_year_sharpe"]), f"{s['years_beat_hold']}/{s['n_years']}"])
    md += ["## 10. OOS 穩定度（現貨 ema50_plain 等權；逐年 Sharpe（括號為持有NR）/ 回撤）", "",
           md_table(["組合", "OOS+OOS2 閘門", "OOS 最低 Sharpe", "OOS 回撤", "OOS 落後NR"] + [f"{y}{'YTD' if y == 2026 else ''}" for y in OOS_YEARS] + ["最差年 Sharpe", "年 Sharpe 贏持有"], stab_rows), ""]

    st_rows = []
    for sname, per in sp["stress"].items():
        for nm, r in per.items():
            st_rows.append([sname, nm] + [f"{num(r[w]['sharpe'])} / {pct(r[w]['max_dd'])} / {int(r[w]['gate_nr']) + int(r[w]['gate_mr'])}" if w in r else "—" for w in WNAMES] + [f"{r['gates']}/10", num(r["turnover"], 1)])
    md += ["## 11. 壓力（現貨 ema50_plain 等權、原生起算）", "",
           md_table(["情境", "組合"] + [f"{w} S/DD/閘" for w in WNAMES] + ["十格", "換手/年"], st_rows), ""]

    # ---------- 自動摘要（數字由上面表格直接引用） ----------
    def g(panel, nm, w, k="sharpe"):
        return sp[panel][nm]["windows"][w][k]
    md += ["## 12. 摘要（自動產生，數字全部來自上表）", ""]
    for nm in sp["native"]:
        r = sp["native"][nm]
        md.append(f"- {nm}（原生 {r['start']}）：十格 {r['gates']}/{r['gates_max']}；Sharpe IS {num(g('native', nm, 'IS'))} / OOS {num(g('native', nm, 'OOS'))} / FULL {num(g('native', nm, 'FULL'))} / IS2 {num(g('native', nm, 'IS2'))} / OOS2 {num(g('native', nm, 'OOS2'))}；回撤 FULL {pct(g('native', nm, 'FULL', 'max_dd'))}、OOS {pct(g('native', nm, 'OOS', 'max_dd'))}；換手 {num(r['notional_turnover_per_year'], 1)}×/年")
    md.append("")
    for nm in sp["common"]:
        r = sp["common"][nm]
        md.append(f"- {nm}（共同 {r['start']}）：十格 {r['gates']}/{r['gates_max']}；Sharpe IS {num(g('common', nm, 'IS'))} / OOS {num(g('common', nm, 'OOS'))} / FULL {num(g('common', nm, 'FULL'))} / IS2 {num(g('common', nm, 'IS2'))} / OOS2 {num(g('common', nm, 'OOS2'))}；回撤 FULL {pct(g('common', nm, 'FULL', 'max_dd'))}、OOS {pct(g('common', nm, 'OOS', 'max_dd'))}")
    md.append("")
    text = "\n".join(md)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{KEY}.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(OUT_DIR, f"{KEY}.json"), "w", encoding="utf-8") as f:
        json.dump(strip_private(report), f, ensure_ascii=False, indent=2, default=str)
    print(f"\n→ {OUT_DIR}/{KEY}.json, {OUT_DIR}/{KEY}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
