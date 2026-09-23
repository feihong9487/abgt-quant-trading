"""運行閘門 B：180 根滾動落後 20% → 轉持有 60 根（與 A 只差量測視窗：A 累計、B 滾動）。

不是新策略：趨勢判定逐位等於冠軍 ema50_plain（close > EMA50），只加引擎 P1 的運行閘門 RiskParams：
- 每根收盤 i ≥ warmup：ratio[i] = 策略權益 / 同起點買入持有權益
- 未在閘門內：j = i − lag_window；lag = ratio[i] / ratio[j] − 1（j ≥ warmup 且 ratio[j] 有效，否則 0 → 起算後前 180 根不可能觸發）
  lag ≤ −lag_stop → 進入閘門（lag_mode），動作 hold = 忽略趨勢判定、目標曝險 100%
- 在閘門內：i − trigger_bar ≥ lag_resume_bars → 恢復（當根不檢查觸發，下一根起可再觸發）；恢復當根若趨勢 off → 次日開盤賣出
- 鏈式再觸發（B 特有）：恢復後 ratio[i−180] 若仍在下跌前高位，下一根即再觸發；hold 期間比值持平，所以最多鏈 2 次、單一連續 hold ≤ 180 根；
  每個鏈接點若趨勢 off 付一次「恢復根賣出 → 再觸發買回」的一根來回（0.14% + 一日跳空）
- 週期定義：trigger_bar ≤ 前一筆 resume_bar + 2 的 lag_events 併為同一週期（原始筆數與併後週期數都印）

無前視：ratio[j] 是 j 根收盤時已算出的值（引擎 ratio_hist 不回填）；EMA 只用 ≤ i 根收盤；閘門決策在收盤、下一根開盤成交。
巢狀特例：lag_stop = 0 逐位等於 ema50_plain（引擎跳過閘門）。
模組常數 LAG_RESUME = "bars"（ratio_high / regime_reset 排除，理由同 A）；字串參數（lag_action、base）不進敏感度。
"""
import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams, regime_atr_channel

NAME = "運行閘門 B：roll180 落後 20% → 持有 60 根"
DESCRIPTION = "close > EMA50 滿倉否則空手（= ema50_plain）；策略/持有比值 180 根滾動落後 ≥ 20% → 忽略趨勢改 100% 持有 60 根後恢復；hold 期間可鏈式再觸發（最多 2 次）"
PARAMS = {"ema_len": 50, "lag_window": 180, "lag_stop": 0.20, "lag_resume_bars": 60, "lag_action": "hold", "base": "ema50_plain"}
LAG_RESUME = "bars"   # 模組常數；ratio_high / regime_reset 排除，理由同 A


def _regime(df, p):
    """回傳 (in_regime, warmup)。base=ema50_plain 與冠軍逐位相同（warmup 49）；base=vol_atr_channel 為移植對照。"""
    c = df["close"].to_numpy(dtype="float64")
    base = str(p.get("base", "ema50_plain"))
    if base == "ema50_plain":
        ema = ta.ema(c, int(p["ema_len"]))          # Pine 語意：前 n−1 根 NaN，SMA 起算後遞迴；第 i 根只用 ≤ i 根 close
        in_regime = c > ema                          # NaN 比較 → False
        valid = ~np.isnan(ema)
        warmup = int(np.argmax(valid)) if valid.any() else len(c)
        return in_regime, warmup
    if base == "vol_atr_channel":
        enter, exit_, warmup = regime_atr_channel(df, int(p["ema_len"]), 14, 1.0, 0.25)
        return {"enter": enter, "exit": exit_}, warmup
    raise ValueError(f"未知 base：{base}")


def _risk(p):
    return RiskParams(lag_stop=float(p["lag_stop"]), lag_window=int(p["lag_window"]), lag_action=str(p["lag_action"]),
                      lag_resume=LAG_RESUME, lag_resume_bars=int(p["lag_resume_bars"]))   # 其餘預設 = ema50_plain


def build(df, p):
    in_regime, warmup = _regime(df, p)
    return {"in_regime": in_regime, "risk": _risk(p), "warmup": warmup}


# ---------- 分析工具（自檢與驗證者共用；不影響 build） ----------

def merge_cycles(events, gap=2):
    """trigger_bar ≤ 前一筆 resume_bar + gap → 併為同一週期。回傳 [{'trigger_bar','resume_bar','events':[...], 'links': n}]。"""
    cycles = []
    for e in events:
        if cycles and cycles[-1]["resume_bar"] is not None and e["trigger_bar"] <= cycles[-1]["resume_bar"] + gap:
            cycles[-1]["events"].append(e)
            cycles[-1]["resume_bar"] = e["resume_bar"]
            cycles[-1]["links"] += 1
        else:
            cycles.append({"trigger_bar": e["trigger_bar"], "resume_bar": e["resume_bar"], "events": [e], "links": 0})
    return cycles


def cycle_table(df, res_var, res_plain, on, warmup, p, horizon=60):
    """逐次觸發表：每個 lag_event 與併後週期的時點、lag、j 根、regime、段相對表現、鏈接成本、hold 內漂移。"""
    idx = df.index
    c = df["close"].to_numpy(dtype="float64")
    o = df["open"].to_numpy(dtype="float64")
    n = len(c)
    eq_v = res_var.equity.reindex(idx).to_numpy(dtype="float64")
    eq_p = res_plain.equity.reindex(idx).to_numpy(dtype="float64")
    bench = np.full(n, np.nan)
    bench[warmup:] = res_var.extra["bench_equity"]
    ratio = eq_v / bench
    W = int(p["lag_window"])
    events = res_var.extra["lag_events"]
    rows = []
    for e in events:
        t = e["trigger_bar"]
        r = e["resume_bar"] if e["resume_bar"] is not None else n - 1
        j = t - W
        h = min(n - 1, t + horizon)
        seg = ratio[t + 1:r + 1]                       # 成交後 hold 期間的比值（t+1 起，含恢復根）
        drift = float(np.nanmax(np.abs(seg / ratio[t + 1] - 1))) if len(seg) > 1 else 0.0
        rows.append({
            "trigger": str(idx[t].date()), "trigger_bar": int(t), "resume": str(idx[r].date()) if e["resume_bar"] is not None else "(未恢復)",
            "hold_bars": int(r - t), "lag": float(e["lag"]), "overshoot": float(-e["lag"] - float(p["lag_stop"])),
            "in_band": bool(-0.25 <= e["lag"] <= -float(p["lag_stop"])),
            "regime_on_at_trigger": bool(on[t]), "regime_on_at_resume": bool(on[r]),
            "j": str(idx[j].date()), "j_bench_1d": float(c[j] / c[j - 1] - 1) if j > 0 else float("nan"),
            "j_ratio_1d": float(ratio[j] / ratio[j - 1] - 1) if j > 0 and ratio[j - 1] == ratio[j - 1] else float("nan"),
            "j_regime_on": bool(on[j]),
            "seg_bench": float(c[r] / c[t] - 1), "seg_plain": float(eq_p[r] / eq_p[t] - 1), "seg_var": float(eq_v[r] / eq_v[t] - 1),
            "h60_bench": float(c[h] / c[t] - 1), "h60_plain": float(eq_p[h] / eq_p[t] - 1), "h60_var": float(eq_v[h] / eq_v[t] - 1),
            "hold_drift": drift,
        })
    cycles = merge_cycles(events)
    crows = []
    for cy in cycles:
        t = cy["trigger_bar"]
        r = cy["resume_bar"] if cy["resume_bar"] is not None else n - 1
        link_cost = 0.0
        link_details = []
        for a, b in zip(cy["events"][:-1], cy["events"][1:]):
            ra = a["resume_bar"]
            if ra is None:
                continue
            # 鏈接點：恢復根 ra（趨勢 off → ra+1 開盤賣出）、再觸發 b（b+1 開盤買回）。比值跨鏈接點的變化 = 來回成本 + 跳空
            k1 = min(n - 1, b["trigger_bar"] + 1)
            cost = float(ratio[k1] / ratio[ra] - 1) if ratio[ra] == ratio[ra] else 0.0
            link_cost += cost
            link_details.append({"resume": str(idx[ra].date()), "retrigger": str(idx[b["trigger_bar"]].date()),
                                 "regime_on_at_resume": bool(on[ra]), "ratio_change": cost,
                                 "gap_open": float(o[min(n - 1, b["trigger_bar"] + 1)] / o[min(n - 1, ra + 1)] - 1) if not on[ra] else 0.0})
        seg_var, seg_plain, seg_bench = eq_v[r] / eq_v[t] - 1, eq_p[r] / eq_p[t] - 1, c[r] / c[t] - 1
        crows.append({
            "trigger": str(idx[t].date()), "trigger_bar": int(t), "resume": str(idx[r].date()) if cy["resume_bar"] is not None else "(未恢復)",
            "resume_bar": int(r), "hold_bars": int(r - t), "n_events": len(cy["events"]), "links": cy["links"],
            "regime_on_at_trigger": bool(on[t]), "lag": float(cy["events"][0]["lag"]),
            "seg_bench": float(seg_bench), "seg_plain": float(seg_plain), "seg_var": float(seg_var),
            "var_vs_plain": float((1 + seg_var) / (1 + seg_plain) - 1), "var_vs_bench": float((1 + seg_var) / (1 + seg_bench) - 1),
            "false_trigger": bool(seg_var < seg_plain),          # 誤觸發 = 該週期 hold 輸給 plain
            "top_trigger": bool(seg_bench < 0),                   # 頂部觸發 = 週期內持有為負
            "link_cost": float(link_cost), "link_details": link_details,
        })
    return rows, crows


if __name__ == "__main__":
    # 自我檢查（BTC、ETH 必跑；SOL、BNB 有快取則一併印）
    import dataclasses
    import json
    import os
    import sys

    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, ROOT)
    from abgt import data as dt
    from abgt.strategies.regime import simulate_regime
    import evaluate_variant as ev

    plain_mod = ev.load_variant("ema50_plain")
    windows = ev.parse_windows(ev.DEFAULT_WINDOWS + "," + ev.ALT_WINDOWS)
    dump = {}
    for sym in ("BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT"):
        df = dt.load_cached("data", sym, "1d", "2019-09-01")
        if df is None:
            print(sym, "無快取，略過")
            continue
        n = len(df)
        r = build(df, PARAMS)
        on, rp, warmup = np.asarray(r["in_regime"], dtype=bool), r["risk"], r["warmup"]
        c = df["close"].to_numpy(dtype="float64")
        plain_built = plain_mod.build(df, plain_mod.PARAMS)
        assert warmup == plain_built["warmup"] == 49, "warmup 必須與冠軍相同（49）"
        assert np.array_equal(on, np.asarray(plain_built["in_regime"], dtype=bool)), "in_regime 必須逐位等於 ema50_plain"
        assert ev.regime_key(r["in_regime"]) == ev.regime_key(plain_built["in_regime"])
        res_plain = simulate_regime(df, plain_built["in_regime"], plain_built["risk"], plain_built["warmup"])
        res_var = simulate_regime(df, r["in_regime"], rp, warmup)

        # (1) lag_stop = 0 逐位等於冠軍
        r0 = build(df, {**PARAMS, "lag_stop": 0.0})
        res0 = simulate_regime(df, r0["in_regime"], r0["risk"], r0["warmup"])
        assert res0.extra["exposure_key"] == res_plain.extra["exposure_key"] and res0.extra["fill_key"] == res_plain.extra["fill_key"], "lag_stop=0 必須逐位等於 ema50_plain"
        assert np.allclose(res0.equity.to_numpy(), res_plain.equity.to_numpy()), "lag_stop=0 權益必須逐位等於 ema50_plain"
        assert res_var.extra["exposure_key"] != res_plain.extra["exposure_key"], "變體應與冠軍不同（有觸發）"

        # (2) 截斷不變（前視）：in_regime、曝險、lag_events 在截斷資料上逐位相同
        ev_full = res_var.extra["lag_events"]
        for cut in (300, 600, 1000, 1500, 2000, n - 30, n - 1):
            rc = build(df.iloc[:cut], PARAMS)
            assert np.array_equal(np.asarray(rc["in_regime"], dtype=bool), on[:cut]), f"截斷 {cut} in_regime 不一致"
            resc = simulate_regime(df.iloc[:cut], rc["in_regime"], rc["risk"], rc["warmup"])
            ex_full = res_var.equity.to_numpy()[:cut - warmup]
            assert np.allclose(resc.equity.to_numpy(), ex_full), f"截斷 {cut} 權益不一致"
            trig_full = [e["trigger_bar"] for e in ev_full if e["trigger_bar"] < cut]
            trig_cut = [e["trigger_bar"] for e in resc.extra["lag_events"]]
            assert trig_full == trig_cut, f"截斷 {cut} lag_events 不一致 {trig_full} vs {trig_cut}"

        # (3)(6)(7) 逐次觸發表、併鏈週期、鏈接成本、j 根日期
        rows, crows = cycle_table(df, res_var, res_plain, on, warmup, PARAMS)
        # (4) lag_bars；(5) 每格 rel_dd 與冠軍差
        rv = ev.run_once(df, sys.modules[__name__], PARAMS, windows)
        rb = ev.run_once(df, plain_mod, plain_mod.PARAMS, windows)
        cmp_ = ev.compare_to_baseline(df, rv, rb, windows)
        print(f"\n===== {sym} =====  warmup {warmup} | lag_events 原始 {len(rows)} 筆 → 併鏈後 {len(crows)} 週期 | lag_bars {res_var.extra['lag_bars']} "
              f"| 成交 {res_var.extra['n_fills']} {res_var.extra['fills_by_reason']} | 冠軍成交 {res_plain.extra['n_fills']}")
        print("--- 逐次觸發（原始 lag_events）---")
        print(f"{'trigger':11}{'resume':11}{'bars':>5}{'lag':>8}{'over':>7}{'band':>5}{'reg@T':>6}{'reg@R':>6}  {'j(i-180)':11}{'j_bench1d':>10}{'j_ratio1d':>10}{'reg@j':>6}{'seg_bench':>10}{'seg_plain':>10}{'seg_var':>9}{'h60_bench':>10}{'h60_plain':>10}{'h60_var':>9}{'drift':>7}")
        for x in rows:
            print(f"{x['trigger']:11}{x['resume']:11}{x['hold_bars']:>5}{x['lag']:>+8.3f}{x['overshoot']:>+7.3f}{'Y' if x['in_band'] else 'N':>5}{'on' if x['regime_on_at_trigger'] else 'off':>6}{'on' if x['regime_on_at_resume'] else 'off':>6}  "
                  f"{x['j']:11}{x['j_bench_1d']:>+10.3f}{x['j_ratio_1d']:>+10.3f}{'on' if x['j_regime_on'] else 'off':>6}{x['seg_bench']:>+10.3f}{x['seg_plain']:>+10.3f}{x['seg_var']:>+9.3f}{x['h60_bench']:>+10.3f}{x['h60_plain']:>+10.3f}{x['h60_var']:>+9.3f}{x['hold_drift']:>7.4f}")
        print("--- 併鏈後週期 ---")
        print(f"{'trigger':11}{'resume':11}{'bars':>5}{'ev':>3}{'links':>6}{'reg@T':>6}{'lag':>8}{'seg_bench':>10}{'seg_plain':>10}{'seg_var':>9}{'v-plain':>9}{'v-bench':>9}{'false':>6}{'top':>5}{'linkcost':>9}")
        for x in crows:
            print(f"{x['trigger']:11}{x['resume']:11}{x['hold_bars']:>5}{x['n_events']:>3}{x['links']:>6}{'on' if x['regime_on_at_trigger'] else 'off':>6}{x['lag']:>+8.3f}{x['seg_bench']:>+10.3f}{x['seg_plain']:>+10.3f}{x['seg_var']:>+9.3f}{x['var_vs_plain']:>+9.3f}{x['var_vs_bench']:>+9.3f}{'Y' if x['false_trigger'] else 'N':>6}{'Y' if x['top_trigger'] else 'N':>5}{x['link_cost']:>+9.4f}")
            for ld in x["link_details"]:
                print(f"    鏈接：恢復 {ld['resume']}（趨勢 {'on' if ld['regime_on_at_resume'] else 'off'}）→ 再觸發 {ld['retrigger']} | 比值變化 {ld['ratio_change']:+.4f} | 開盤跳空 {ld['gap_open']:+.4f}")
        print("--- 每格：變體 vs 冠軍 ---")
        print(f"{'格':6}{'Sharpe':>8}{'plainS':>8}{'ΔS':>7}{'t':>7}{'maxDD':>8}{'plainDD':>8}{'rel_dd':>8}{'plainRel':>9}{'Δrel':>7}{'閘門':>5}{'plain閘':>7}{'觸發數':>7}{'net':>8}{'plainNet':>9}")
        for wname, w in rv["windows"].items():
            wb = rb["windows"][wname]
            s0 = df.index.searchsorted(w["start"]); s1 = df.index.searchsorted(w["end"], side="right")
            ntrig = sum(1 for x in crows if s0 <= x["trigger_bar"] < s1)
            print(f"{wname:6}{w['strategy']['sharpe']:>8.3f}{wb['strategy']['sharpe']:>8.3f}{cmp_[wname]['d_sharpe']:>+7.3f}{cmp_[wname]['t_daily']:>+7.2f}{w['strategy']['max_dd']:>8.3f}{wb['strategy']['max_dd']:>8.3f}"
                  f"{w['rel_dd_vs_hold']:>8.3f}{wb['rel_dd_vs_hold']:>9.3f}{(w['rel_dd_vs_hold'] - wb['rel_dd_vs_hold']):>+7.3f}{'✓' if w['gate']['passed'] else '✗':>5}{'✓' if wb['gate']['passed'] else '✗':>7}{ntrig:>7}{w['strategy']['net_return']:>+8.3f}{wb['strategy']['net_return']:>+9.3f}")
        max_drift = max((x["hold_drift"] for x in rows), default=0.0)
        print(f"hold 內比值最大漂移 {max_drift:.4%} | 落在 [−0.25, −0.20] 的觸發 {sum(x['in_band'] for x in rows)}/{len(rows)} | 鏈接點合計來回成本 {sum(x['link_cost'] for x in crows):+.4f}")
        dump[sym] = {"events": rows, "cycles": crows, "lag_bars": res_var.extra["lag_bars"], "fills_by_reason": res_var.extra["fills_by_reason"],
                     "cells": {w: {"sharpe": x["strategy"]["sharpe"], "d_sharpe": cmp_[w]["d_sharpe"], "t": cmp_[w]["t_daily"], "rel_dd": x["rel_dd_vs_hold"],
                                   "plain_rel_dd": rb["windows"][w]["rel_dd_vs_hold"], "gate": x["gate"]["passed"], "plain_gate": rb["windows"][w]["gate"]["passed"]}
                               for w, x in rv["windows"].items()}}
    os.makedirs(os.path.join(ROOT, "results", "variants"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "variants", "running_gate_lag_stop_roll180_20_hold60_selfcheck.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=1, default=str)
    print("\n自檢全部通過 → results/variants/running_gate_lag_stop_roll180_20_hold60_selfcheck.json")
