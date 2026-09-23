"""運行閘門 A：自基期累計落後 20% → 轉持有 60 根（CLAUDE.md §1 字面版，有界恢復）。kind = guard。

不是新策略：趨勢判定與冠軍 ema50_plain 逐位相同（close > EMA50 滿倉否則空手），只加引擎 v3 的運行閘門 RiskParams：
- 每根收盤 i ≥ warmup：bench[i] = initial×(1−comm)/c[warmup] × c[i]；ratio[i] = 權益[i] / bench[i]
- 未在閘門模式：lag = ratio[i]/lag_base − 1（lag_base 起算時 1.0）；lag ≤ −lag_stop（0.20）→ 進入閘門模式（hold：忽略趨勢判定、目標曝險 = max_exposure）
- 閘門模式滿 lag_resume_bars（60）根 → 恢復，lag_base 重設為當下 ratio（每個週期最多落後 X + 單日跳空 + 成本，不是自上線累計）
- 目標變動在下一根開盤成交（±0.02% 滑點、0.05%/邊）。觸發時已在場 → 無成交，只是 60 根內 close < EMA50 的出場被壓住；
  觸發時空手 → 下一根開盤買進（fills_by_reason「落後閘門持有」）；恢復根 regime 為 off → 下一根開盤賣出（「趨勢結束」）

模組常數（結構參數，不進 PARAMS、不做 ±25%）：
- LAG_WINDOW = 0：累計自基期（CLAUDE.md §1 定義、實盤上線日起算的可操作語意）。評估器對 int 0 做 ±25% 會變成 max(1, round(0)) = 1
  （滾動 1 根 = 閘門死碼），所以不放 PARAMS。
- LAG_RESUME = "bars"：hold + ratio_high 在熊市裡永不恢復（持有時 ratio 隨 c 單調）、hold + regime_reset 在空手觸發時下一根即恢復；
  bars 是唯一有界且不退化的恢復。

對照與移植（同檔案改字串參數，字串不進敏感度）：
- flat 對照：PARAMS | {"lag_action": "flat"}（觸發後空手 60 根）。只作對照，不是候選。
- 通道移植：PARAMS | {"base": "vol_atr_channel"}，對照組為 vol_atr_channel 本身（判準 C4）。

無前視：EMA 在第 i 根只用 ≤ i 根收盤；ratio[i] 只用 ≤ i 根收盤與已成交部位；bench 分母固定為 c[warmup]；
in_regime 與冠軍逐位相同；warmup = 49 與冠軍相同 → 引擎 bench / ratio 基期與冠軍逐位相同。
巢狀特例：lag_stop = 0.0 → 閘門關閉，regime_key 與 exposure_key 逐位等於 ema50_plain。
參數事前登記（lag_stop 0.20 / 60 根 / hold / bars / 累計），不得依 OOS 或已知壞結果回調。
"""
import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams, regime_atr_channel

NAME = "運行閘門 A：累計落後 20% → 轉持有 60 根"
DESCRIPTION = "close > EMA50 滿倉否則空手（= ema50_plain）；策略/持有比值自基期累計落後 ≥ 20% 時忽略趨勢判定改 100% 持有 60 根，之後基期重設回到 EMA50 規則"
PARAMS = {"ema_len": 50, "lag_stop": 0.20, "lag_resume_bars": 60, "lag_action": "hold", "base": "ema50_plain"}
LAG_WINDOW = 0        # 模組常數（結構參數）：累計自基期；不放 PARAMS，避免 ±25% 把 0 變成 1（滾動 1 根 = 死碼）污染敏感度
LAG_RESUME = "bars"   # 模組常數：ratio_high 與 regime_reset 明確排除（見模組 docstring）
CHANNEL_ATR_LEN, CHANNEL_K_IN, CHANNEL_K_OUT = 14, 1.0, 0.25   # 通道移植固定用 vol_atr_channel 的事前值


def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    base = str(p.get("base", "ema50_plain"))
    if base == "ema50_plain":
        ema = ta.ema(c, int(p["ema_len"]))          # Pine 語意：前 n−1 根 NaN，SMA 起算後遞迴；第 i 根只用 ≤ i 根收盤
        in_regime = c > ema                            # 與 ema50_plain 逐位相同（NaN 比較 → False）
        valid = ~np.isnan(ema)
        warmup = int(np.argmax(valid)) if valid.any() else len(c)   # = ema_len − 1 = 49，起算根與冠軍相同
    elif base == "vol_atr_channel":                    # 移植檢查用（判準 C4），非候選
        enter, exit_, warmup = regime_atr_channel(df, int(p["ema_len"]), CHANNEL_ATR_LEN, CHANNEL_K_IN, CHANNEL_K_OUT)
        in_regime = {"enter": enter, "exit": exit_}
    else:
        raise ValueError(f"未知 base: {base}")
    rp = RiskParams(lag_stop=float(p["lag_stop"]), lag_window=LAG_WINDOW, lag_action=str(p["lag_action"]),
                    lag_resume=LAG_RESUME, lag_resume_bars=int(p["lag_resume_bars"]))   # 其餘全部預設 = ema50_plain 的 RiskParams()
    return {"in_regime": in_regime, "risk": rp, "warmup": warmup}


if __name__ == "__main__":
    # 自我檢查（實作者必跑，輸出貼進報告）。任一 assert 失敗即實作失敗，不得改參數。
    import json
    import os
    import sys

    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    from abgt import data as dt
    from abgt.strategies.regime import simulate_regime
    import evaluate_variant as ev

    plain_mod = ev.load_variant("ema50_plain")
    chan_mod = ev.load_variant("vol_atr_channel")
    windows = ev.parse_windows(ev.DEFAULT_WINDOWS + "," + ev.ALT_WINDOWS)
    X = float(PARAMS["lag_stop"])
    report = {}
    for sym in ("BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT"):
        df = dt.load_cached("data", sym, "1d", ev.DATA_SINCE)
        assert df is not None, f"{sym} 無快取"
        c = df["close"].to_numpy(dtype="float64")
        n = len(c)
        idx = df.index
        r = build(df, PARAMS)
        on, rp, warmup = r["in_regime"], r["risk"], r["warmup"]
        plain_built = plain_mod.build(df, plain_mod.PARAMS)

        # (1) 閘門關閉 = 冠軍：regime_key 與 exposure_key 逐位相同
        r0 = build(df, {**PARAMS, "lag_stop": 0.0})
        assert ev.regime_key(r0["in_regime"]) == ev.regime_key(plain_built["in_regime"]), "lag_stop=0 的 regime_key 必須等於 ema50_plain"
        assert ev.regime_key(on) == ev.regime_key(plain_built["in_regime"]), "變體 in_regime 必須逐位等於 ema50_plain"
        assert r0["warmup"] == plain_built["warmup"] == warmup, "warmup 必須與冠軍相同（bench 基期）"
        res0 = simulate_regime(df, r0["in_regime"], r0["risk"], r0["warmup"])
        res_plain = simulate_regime(df, plain_built["in_regime"], plain_built["risk"], plain_built["warmup"])
        assert res0.extra["exposure_key"] == res_plain.extra["exposure_key"], "lag_stop=0 的 exposure_key 必須等於 ema50_plain"
        assert res0.extra["fill_key"] == res_plain.extra["fill_key"], "lag_stop=0 的 fill_key 必須等於 ema50_plain"
        assert res0.extra["lag_triggers"] == 0
        # 通道移植的 in_regime 逐位等於 vol_atr_channel
        rc = build(df, {**PARAMS, "base": "vol_atr_channel"})
        chan_built = chan_mod.build(df, chan_mod.PARAMS)
        assert ev.regime_key(rc["in_regime"]) == ev.regime_key(chan_built["in_regime"]), "base=vol_atr_channel 的 regime 必須逐位等於 vol_atr_channel"
        assert rc["warmup"] == chan_built["warmup"]

        # (2) 截斷不變（前視）：build(df.iloc[:cut]) 的 in_regime 等於全序列前綴
        for cut in (60, 400, 900, 1500, n - 30, n - 1):
            rcut = build(df.iloc[:cut], PARAMS)
            assert np.array_equal(rcut["in_regime"], on[:cut]), f"{sym} 截斷 {cut} in_regime 不一致"
            assert rcut["warmup"] == warmup

        # (3)(4) 逐次觸發表：trigger 日期、lag、觸發根 regime、resume 日期、持有根數、fills 原因、段相對表現、overshoot
        res = simulate_regime(df, on, rp, warmup)
        eq_v = res.equity.to_numpy()          # 從 warmup 起
        eq_p = res_plain.equity.to_numpy()
        bench = res.extra["bench_equity"]
        trades = res.trades
        events = []
        for e in res.extra["lag_events"]:
            tb, rb = e["trigger_bar"], e["resume_bar"]
            rb_eff = rb if rb is not None else n - 1
            a = min(tb + 1 - warmup, len(eq_v) - 1)
            b = min(rb_eff + 1 - warmup, len(eq_v) - 1)
            seg = float(np.log(eq_v[b] / eq_v[a]) - np.log(eq_p[b] / eq_p[a]))
            hold_ret = float(c[min(rb_eff + 1, n - 1)] / c[min(tb + 1, n - 1)] - 1)
            plain_ret = float(eq_p[b] / eq_p[a] - 1)
            # 該段內的成交（進場日 / 出場日落在 (tb, rb+1]）
            t0, t1 = idx[min(tb + 1, n - 1)], idx[min(rb_eff + 1, n - 1)]
            fills = []
            if len(trades):
                ent = trades[(trades.entry_time >= t0) & (trades.entry_time <= t1)]
                ex_ = trades[(trades.exit_time >= t0) & (trades.exit_time <= t1)]
                fills += [f"買 {t.date()}" for t in sorted(set(ent.entry_time))]
                fills += [f"賣 {t.date()} {rs}" for t, rs in sorted(set(zip(ex_.exit_time, ex_.reason)))]
            # C3(d)：hold 週期內（成交根之後至恢復根，排除成交日）變體比值對持有的漂移
            ratio_seg = eq_v[tb + 1 - warmup: rb_eff + 1 - warmup] / bench[tb + 1 - warmup: rb_eff + 1 - warmup]
            fill_days = set()
            if len(trades):
                for t in list(ent.entry_time) + list(ex_.exit_time):
                    fill_days.add(int(idx.get_loc(t)))
            bars_seg = list(range(tb + 1, rb_eff + 1))                      # (trigger, resume]：閘門控制的持有段
            buy_days = {int(idx.get_loc(t)) for t in ent.entry_time} if len(trades) else set()
            start_bar = (max(buy_days) + 1) if buy_days else tb + 1           # 進場成交根之後
            keep = [j for j, bar in enumerate(bars_seg) if bar >= start_bar and bar not in fill_days]
            drift = float(np.max(np.abs(ratio_seg[keep] / ratio_seg[keep][0] - 1))) if len(keep) > 1 else 0.0
            events.append({
                "trigger": str(idx[tb].date()), "trigger_bar": int(tb), "lag": round(float(e["lag"]), 4),
                "overshoot_pp": round((-float(e["lag"]) - X) * 100, 2), "regime_at_trigger": "on" if bool(on[tb]) else "off",
                "resume": str(idx[rb].date()) if rb is not None else None, "hold_bars": int(rb_eff - tb),
                "fills": fills, "seg_rel_perf": round(seg, 4), "misfire": seg < 0,
                "hold_ret": round(hold_ret, 4), "plain_ret": round(plain_ret, 4), "ratio_drift_max": round(drift, 5),
                "in_IS": "2020-01-01" <= str(idx[tb].date()) <= "2022-12-31",
            })
        # (5) lag_bars / 總根數、每格 rel_dd_vs_hold 與冠軍差（評估器口徑）
        rv = ev.run_once(df, sys.modules[__name__], PARAMS, windows)
        rpl = ev.run_once(df, plain_mod, plain_mod.PARAMS, windows)
        cells = {w: {"rel_dd": round(rv["windows"][w]["rel_dd_vs_hold"], 4), "plain_rel_dd": round(rpl["windows"][w]["rel_dd_vs_hold"], 4),
                     "d_rel_dd_pp": round((rv["windows"][w]["rel_dd_vs_hold"] - rpl["windows"][w]["rel_dd_vs_hold"]) * 100, 2),
                     "sharpe": round(rv["windows"][w]["strategy"]["sharpe"], 4), "plain_sharpe": round(rpl["windows"][w]["strategy"]["sharpe"], 4),
                     "gate": rv["windows"][w]["gate"]["passed"], "plain_gate": rpl["windows"][w]["gate"]["passed"]}
                 for w in rv["windows"]}
        total = n - warmup
        n_is = sum(1 for e in events if e["in_IS"])
        n_mis_is = sum(1 for e in events if e["in_IS"] and e["misfire"])
        in_band = sum(1 for e in events if -0.25 <= e["lag"] <= -0.20)
        report[sym] = {"warmup": int(warmup), "start": str(idx[warmup].date()), "n_bars": int(total), "lag_triggers": len(events),
                       "lag_bars": int(res.extra["lag_bars"]), "lag_bars_ratio": round(res.extra["lag_bars"] / total, 4),
                       "fills_by_reason": res.extra["fills_by_reason"], "n_fills": res.extra["n_fills"], "plain_n_fills": res_plain.extra["n_fills"],
                       "engine_rel_dd_vs_hold": round(res.extra["rel_dd_vs_hold"], 4), "plain_engine_rel_dd_vs_hold": round(res_plain.extra["rel_dd_vs_hold"], 4),
                       "rel_final_vs_hold": round(res.extra["rel_final_vs_hold"], 4), "plain_rel_final_vs_hold": round(res_plain.extra["rel_final_vs_hold"], 4),
                       "IS_cycles": n_is, "IS_misfires": n_mis_is, "misfires_total": sum(1 for e in events if e["misfire"]),
                       "lag_in_band_ratio": round(in_band / len(events), 3) if events else None,
                       "max_ratio_drift": max([e["ratio_drift_max"] for e in events], default=0.0),
                       "events": events, "cells": cells}
        print(f"=== {sym} | 起算 {idx[warmup].date()} (warmup {warmup}) | 總根數 {total} | 觸發 {len(events)} 次 | lag_bars {res.extra['lag_bars']} ({res.extra['lag_bars'] / total:.1%}) | "
              f"成交 {res.extra['n_fills']} (plain {res_plain.extra['n_fills']}) {res.extra['fills_by_reason']} | 引擎累計 rel_dd {res.extra['rel_dd_vs_hold']:+.3f} (plain {res_plain.extra['rel_dd_vs_hold']:+.3f}) | "
              f"期末相對持有 {res.extra['rel_final_vs_hold']:+.3f} (plain {res_plain.extra['rel_final_vs_hold']:+.3f})")
        print(f"  自檢 (1) lag_stop=0 ≡ ema50_plain（regime/exposure/fill key 相同）OK；(2) 截斷 6 個切點 in_regime 不變 OK；通道移植 regime ≡ vol_atr_channel OK")
        print(f"  {'#':>2} {'trigger':10} {'lag':>7} {'over(pp)':>8} {'reg':>3} {'resume':10} {'hold':>4} {'seg_rel':>8} {'hold_ret':>8} {'plain_ret':>9} {'drift':>7} 誤觸發 fills")
        for k, e in enumerate(events, 1):
            print(f"  {k:>2} {e['trigger']:10} {e['lag']:>+7.3f} {e['overshoot_pp']:>8.2f} {e['regime_at_trigger']:>3} {str(e['resume']):10} {e['hold_bars']:>4} {e['seg_rel_perf']:>+8.3f} {e['hold_ret']:>+8.3f} {e['plain_ret']:>+9.3f} {e['ratio_drift_max']:>7.4f} {'誤' if e['misfire'] else '　'} {'; '.join(e['fills'])}")
        print(f"  IS 週期 {n_is}（≤6?）、IS 誤觸發 {n_mis_is}（≤3?）、lag∈[-0.25,-0.20] 比例 {report[sym]['lag_in_band_ratio']}、hold 週期內比值最大漂移 {report[sym]['max_ratio_drift']:.4%}")
        for w, cv in cells.items():
            print(f"  {w:5} rel_dd {cv['rel_dd']:+.3f} (plain {cv['plain_rel_dd']:+.3f}, Δ {cv['d_rel_dd_pp']:+.1f} pp) | Sharpe {cv['sharpe']:.3f} (plain {cv['plain_sharpe']:.3f}, Δ {cv['sharpe'] - cv['plain_sharpe']:+.3f}) | 閘門 {'✓' if cv['gate'] else '✗'} (plain {'✓' if cv['plain_gate'] else '✗'})")
    os.makedirs("results/variants", exist_ok=True)
    with open("results/variants/running_gate_lag_stop_cum20_hold60_selfcheck.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print("\n→ results/variants/running_gate_lag_stop_cum20_hold60_selfcheck.json")
