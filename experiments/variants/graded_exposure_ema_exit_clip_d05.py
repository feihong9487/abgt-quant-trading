"""分級曝險正式化 B：二元滿倉進場 + 趨勢內近均線減半（2×2 缺格 graded_exit_clip_ema50）。

疊加位置：冠軍 ema50_plain 的 regime 完全不變（enter = close > EMA50、exit = close < EMA50，滿倉進場），
只在「趨勢內、價格曾離開均線 ≥ d_clip ATR 之後又回到均線 d_clip ATR 內」把曝險降到 s_clip；d 回到 ≥ d_clip 即升回滿倉。

規則：
- enter = close > EMA50；exit = close < EMA50（冠軍的進出場，互斥）
- d = (close − EMA50) / ATR14
- 上膛（armed）：回合內第一次 d ≥ d_clip 後永久上膛，直到出場；進場當根若 d ≥ d_clip 則直接上膛
- scale：趨勢外 0；趨勢內「已上膛且 d < d_clip」→ s_clip（半倉），其餘（未上膛或 d ≥ d_clip）→ 1.0
- 無棘輪（scale_mode="track"，逐根跟隨）、無停損、無波動目標、無回撤暫停；rebalance_threshold 0.15 < 步幅 0.5，每次級別翻轉都在次日開盤成交

為什麼要上膛：穿越進場當根 d 幾乎都 < d_clip，若允許立即降級，幾乎每個回合都會在第二個開盤賣掉一半 —— 那是「延遲一根的分級進場」
外加雙倍成交，不是 2×2 的「二元進場」那一格。上膛規則只用同一個 d_clip，沒有新增數字；它把減半限制在
「價格曾離開均線又回到均線附近」的回檔／出場尾段。不設寬限根數、不設回升門檻（未登記的自由度）。

無前視：EMA / ATR / d 在第 i 根只用 ≤ i 根 OHLC（RMA 遞迴、TR 用 close[i−1]）；state / armed 由前往後遞推；
引擎 state_machine(enter, exit) 逐根等於本檔迴圈的 state（風控全預設，無停損不會改變引擎狀態）。
巢狀特例：s_clip = 1.0 逐位等於 ema50_plain（scale 為 plain 的 0/1、exposure_key 相同）。
"""
import os

import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams, state_machine

NAME = "分級曝險 B：二元進場 + 近均線減半"
DESCRIPTION = "close > EMA50 滿倉進、close < EMA50 全出（= 冠軍）；趨勢內曾離開 EMA50 ≥ 0.5 ATR 後回到 0.5 ATR 內 → 半倉，d ≥ 0.5 升回滿倉；無棘輪無停損"
PARAMS = {"ema_len": 50, "atr_len": 14, "d_clip": 0.5, "s_clip": 0.5, "rebalance_threshold": 0.15}
# d_clip / s_clip 進 PARAMS 只為了讓評估器印 ±25% 敏感度；登記值 0.5 / 0.5 不得改。


def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    n = len(c)
    ema_len, atr_len = int(p["ema_len"]), int(p["atr_len"])
    d_clip, s_clip = float(p["d_clip"]), float(p["s_clip"])
    ema = ta.ema(c, ema_len)                            # Pine 語意：前 ema_len−1 根 NaN，SMA 起算後遞迴；只用 ≤ i 根收盤
    atr = ta.atr(h, l, c, atr_len)                      # RMA(TR)，TR 用 c[i−1]；只用 ≤ i 根
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (c - ema) / atr                             # 距 EMA 幾個 ATR；NaN 維持 NaN
    enter = c > ema                                     # 冠軍的進場（NaN 比較 → False）
    exit_ = c < ema                                     # 冠軍的出場
    scale = np.zeros(n)
    armed_arr = np.zeros(n, dtype=bool)
    state = False
    armed = False
    for i in range(n):                                  # 由前往後遞推，與引擎 state_machine 同規則
        if np.isnan(ema[i]) or np.isnan(atr[i]):
            state, armed = False, False                 # 暖機期一律趨勢外
        elif (not state) and enter[i]:
            state, armed = True, bool(d[i] >= d_clip)   # 進場當根滿倉；若當根已 ≥ d_clip ATR 則直接上膛
        elif state and exit_[i]:
            state, armed = False, False
        elif state:
            armed = armed or bool(d[i] >= d_clip)       # 回合內第一次 d ≥ d_clip 後永久上膛，直到出場
        if not state:
            scale[i] = 0.0
        elif armed and d[i] < d_clip:
            scale[i] = s_clip                           # 已上膛且回到均線 d_clip ATR 內 → 半倉
        else:
            scale[i] = 1.0                              # 未上膛（剛進場還沒離開均線）或 d ≥ d_clip → 滿倉
        armed_arr[i] = armed and state
    valid = ~np.isnan(ema) & ~np.isnan(atr)
    warmup = int(np.argmax(valid)) if valid.any() else n    # = ema_len − 1 = 49，與基準同視窗
    rp = RiskParams(max_exposure=1.0, atr_len=atr_len, scale_mode="track", scale_floor=0.0,
                    vol_target=0.0, atr_stop_mult=0.0, dd_pause=0.0, lag_stop=0.0,
                    reentry="immediate", exec_delay_bars=0, rebalance_threshold=float(p["rebalance_threshold"]))
    out = {"in_regime": {"enter": enter, "exit": exit_, "scale": scale}, "risk": rp, "warmup": warmup}
    if os.environ.get("ABGT_DEBUG"):
        out["_debug"] = {"ema": ema, "atr": atr, "d": d, "armed": armed_arr}
    return out


if __name__ == "__main__":
    # 自我檢查（BTC、ETH 各跑，全部 assert）：
    # (1) state_machine(enter, exit) == (c > EMA50) == (scale > 0)；regime_key 與 ema50_plain 相等
    # (2) 未上膛的根一律 scale == 1.0；scale == s_clip 的根必滿足 d < d_clip 且該回合先前已出現 d ≥ d_clip
    # (3) 巢狀特例：s_clip = 1.0 時 scale == plain.astype(float)，simulate 的 exposure_key == ema50_plain 的 exposure_key
    # (4) 截斷不變：cut ∈ {60, 400, 900, 1500, len−30, len−1}，enter/exit/scale 前綴逐位相同
    # (5) 印出：半倉根佔趨勢內根數比例、每回合減半次數的分布、FULL n_fills 與 ema50_plain 的比值
    import hashlib
    import importlib.util
    import sys

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, root)
    os.environ["ABGT_DEBUG"] = "1"
    from abgt import data as dt
    from abgt.strategies.regime import simulate_regime

    spec = importlib.util.spec_from_file_location("variant_ema50_plain", os.path.join(root, "experiments", "variants", "ema50_plain.py"))
    plain_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plain_mod)

    def rkey(in_regime):
        arr = state_machine(in_regime["enter"], in_regime["exit"]) if isinstance(in_regime, dict) else np.asarray(in_regime, dtype=bool)
        return hashlib.md5(np.packbits(arr).tobytes()).hexdigest()

    for sym in ("BTC/USDT:USDT", "ETH/USDT:USDT"):
        df = dt.load_cached("data", sym, "1d", "2019-09-01")
        if df is None:
            print(sym, "無快取")
            continue
        r = build(df, PARAMS)
        en, ex, sc = r["in_regime"]["enter"], r["in_regime"]["exit"], r["in_regime"]["scale"]
        d, armed = r["_debug"]["d"], r["_debug"]["armed"]
        c = df["close"].to_numpy(dtype="float64")
        plain = c > ta.ema(c, PARAMS["ema_len"])
        # (1)
        assert not np.any(en & ex), "enter 與 exit 不可同根同真"
        on = state_machine(en, ex)
        assert np.array_equal(on, plain), "state_machine(enter, exit) 必須逐根等於 close > EMA50"
        assert np.array_equal(on, sc > 0), "scale > 0 必須逐根等於 state"
        pb = plain_mod.build(df, plain_mod.PARAMS)
        assert rkey(r["in_regime"]) == rkey(pb["in_regime"]), "regime_key 必須與 ema50_plain 相等"
        assert r["warmup"] == pb["warmup"] == PARAMS["ema_len"] - 1, "warmup 必須與基準相同"
        # (2)
        assert np.all(sc[on & ~armed] == 1.0), "未上膛的根一律 scale == 1.0"
        half = np.isclose(sc, PARAMS["s_clip"]) & on
        assert np.all(d[half] < PARAMS["d_clip"]), "半倉根必須 d < d_clip"
        assert np.all(armed[half]), "半倉根必須已上膛"
        # 上膛的根：該回合先前（含當根）已出現 d ≥ d_clip
        seen = False
        prev_on = False
        for i in range(len(c)):
            if on[i] and not prev_on:
                seen = False
            if on[i]:
                seen = seen or bool(d[i] >= PARAMS["d_clip"])
                assert armed[i] == seen, f"第 {i} 根上膛旗標與回合內 d ≥ d_clip 歷史不一致"
                if half[i]:
                    assert seen, f"第 {i} 根半倉但回合內尚未出現 d ≥ d_clip"
            prev_on = on[i]
        assert np.all(np.isin(sc[on], [PARAMS["s_clip"], 1.0])), "趨勢內 scale 只能是 s_clip 或 1.0"
        # (3)
        r1 = build(df, {**PARAMS, "s_clip": 1.0})
        assert np.array_equal(r1["in_regime"]["scale"], plain.astype(float)), "s_clip=1 時 scale 應為 plain 的 0/1"
        res1 = simulate_regime(df, r1["in_regime"], r1["risk"], r1["warmup"])
        resp = simulate_regime(df, pb["in_regime"], pb["risk"], pb["warmup"])
        assert res1.extra["exposure_key"] == resp.extra["exposure_key"], "s_clip=1 的 exposure_key 必須等於 ema50_plain"
        assert res1.extra["n_fills"] == resp.extra["n_fills"], "s_clip=1 的成交數必須等於 ema50_plain"
        # (4)
        for cut in (60, 400, 900, 1500, len(df) - 30, len(df) - 1):
            rc = build(df.iloc[:cut], PARAMS)
            assert np.array_equal(rc["in_regime"]["enter"], en[:cut]), f"截斷 {cut} enter 不一致"
            assert np.array_equal(rc["in_regime"]["exit"], ex[:cut]), f"截斷 {cut} exit 不一致"
            assert np.array_equal(rc["in_regime"]["scale"], sc[:cut]), f"截斷 {cut} scale 不一致"
        # (5)
        res = simulate_regime(df, r["in_regime"], r["risk"], r["warmup"])
        half_share = float(half.sum() / on.sum())
        # 每回合減半次數：回合內 scale 由 1.0 → s_clip 的翻轉次數
        counts = []
        i = 0
        n = len(c)
        while i < n:
            if on[i]:
                j = i
                k = 0
                prev = 1.0
                while j < n and on[j]:
                    if np.isclose(sc[j], PARAMS["s_clip"]) and prev == 1.0:
                        k += 1
                    prev = sc[j]
                    j += 1
                counts.append(k)
                i = j
            else:
                i += 1
        counts = np.array(counts)
        dist = {int(k): int((counts == k).sum()) for k in np.unique(counts)}
        ratio = res.extra["n_fills"] / resp.extra["n_fills"]
        print(f"{sym}: OK | warmup {r['warmup']} | 在場 {on.mean():.1%} | 半倉根/趨勢內根 {half_share:.1%} ({int(half.sum())}/{int(on.sum())}) "
              f"| 回合數 {len(counts)} 減半次數分布 {dist} 中位數 {np.median(counts):.0f} 最大 {counts.max()} "
              f"| FULL 成交 {res.extra['n_fills']} vs plain {resp.extra['n_fills']} = {ratio:.2f}× "
              f"| fills_by_reason {res.extra['fills_by_reason']} | 平均曝險 {res.extra['avg_exposure']:.3f}（趨勢內 {res.extra['avg_exposure_in_regime']:.3f}）")
