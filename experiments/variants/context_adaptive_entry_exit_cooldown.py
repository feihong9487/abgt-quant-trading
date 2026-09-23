"""情境自適應進場：出場冷卻（純 EMA50 出場）。

出場固定 close ≤ EMA50（與 ema50_plain 逐根相同）。進場門檻依「距上次出場多近」切換：
- 上次出場訊號根後 1..cooldown_bars 根內（冷卻窗）：只准以 close > EMA50 + k_in_strict×ATR14 重進（= vol_atr_channel 進場）
- 冷卻過期後：恢復 close > EMA50（= ema50_plain 進場）

設計：第 2 輪證明降低門檻的 OR 腿在 BTC 是零期望、通道 +1 ATR 是 BTC 的 alpha，但通道在 V 型市場付 1 ATR 延遲。
本變體無記憶、只看「距上次出場多近」：鋸齒再穿越（出場後 1–5 根就站回 EMA50）要多付 1 ATR 的確認，
崩盤後首次回升距上次出場通常數十根 → 冷卻早已過期 → 與冠軍同根進場，V 型市場零延遲。
純 EMA50 出場讓「plain 模式」逐根等於冠軍：冠軍每個回合不是原樣保留就是被冷卻擋掉 / 延後，Δ 可逐回合歸因。

無前視：EMA / ATR 在第 i 根只用 ≤ i 根的 OHLC；ctx[i] 只用 ≤ i−1 根決定的 last_exit 與第 i 根收盤；
狀態機由前往後遞推；引擎在 i+1 開盤成交。截斷不變。
風控全預設（無停損、無波動目標、無回撤暫停、延遲 0），所以引擎 state_machine(enter, exit_) 的軌跡與 build 內的本地副本逐根相同，
last_exit 定義自洽；壓力測試的「延遲一根」只改成交根，不改訊號根，冷卻計數不受影響。

邊界：cooldown_bars = 0 時 ctx[i] 只可能在 i == last_exit 為真，而該根 state 已 True、enter 被忽略 → 逐根等於 ema50_plain。
enter 與 exit_ 同根同真只當 c ≤ ema 且 c > ema + k×atr，不可能。
"""
import os

import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams, state_machine

NAME = "情境自適應進場：出場冷卻（純 EMA50 出場）"
DESCRIPTION = "出場 close ≤ EMA50 固定（= ema50_plain）；上次出場後 5 根內只准以 close > EMA50 + 1.0×ATR14 重進，之後恢復 close > EMA50；無風控"
PARAMS = {"ema_len": 50, "atr_len": 14, "k_in_strict": 1.0, "cooldown_bars": 5}


def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    n = len(c)
    ema = ta.ema(c, int(p["ema_len"]))              # Pine 語意：前 n−1 根 NaN，SMA 起算後遞迴
    atr = ta.atr(h, l, c, int(p["atr_len"]))        # RMA(TR)，TR 用前一根收盤
    valid = ~np.isnan(ema) & ~np.isnan(atr)
    warmup = int(np.argmax(valid)) if valid.any() else n      # 預設 49，與冠軍同視窗
    plain_enter = c > ema                                     # = ema50_plain 進場
    strict_enter = c > ema + float(p["k_in_strict"]) * atr   # = vol_atr_channel 進場
    exit_ = ~(c > ema)                                        # c ≤ ema（NaN 亦為 True）
    CD = int(p["cooldown_bars"])
    state = False
    last_exit = -10 ** 9                                      # 上次出場「訊號根」（state True→False 的那根），不是成交根
    enter = np.zeros(n, dtype=bool)
    ctx = np.zeros(n, dtype=bool)
    for i in range(n):
        if not valid[i]:
            state = False                                     # enter[i] False；exit_ True 但趨勢外被忽略
            continue
        ctx[i] = (i - last_exit) <= CD                        # 出場訊號根後 1..CD 根為冷卻（先算情境再更新狀態）
        enter[i] = strict_enter[i] if ctx[i] else plain_enter[i]
        # 與引擎 state_machine 相同語意的本地副本，只為了取得 last_exit
        if not state and enter[i]:
            state = True
        elif state and exit_[i]:
            state = False
            last_exit = i
    if os.environ.get("ABGT_DEBUG"):
        _debug(df, enter, exit_, plain_enter, strict_enter, ctx, warmup, ema, atr, valid)
    return {"in_regime": {"enter": enter, "exit": exit_}, "risk": RiskParams(), "warmup": warmup}


# ---------------------------------------------------------------------------
# 診斷（ABGT_DEBUG=1）：不影響回傳值
# ---------------------------------------------------------------------------

_DEBUG_WINDOWS = (("IS", "2020-01-01", "2022-12-31"), ("OOS", "2023-01-01", None), ("FULL", "2020-01-01", None),
                  ("IS2", "2020-07-01", "2023-12-31"), ("OOS2", "2024-01-01", None))
_COST_PER_ROUND = 0.0014


def _rounds(on: np.ndarray) -> list[tuple[int, int | None]]:
    """回傳 [(e, x)]：e = 進場訊號根（on 由 False→True 的那根），x = 出場訊號根（on 由 True→False 的那根），期末仍在場 x=None。"""
    out = []
    e = None
    for i in range(len(on)):
        if on[i] and e is None:
            e = i
        elif not on[i] and e is not None:
            out.append((e, i))
            e = None
    if e is not None:
        out.append((e, None))
    return out


def _debug(df, enter, exit_, plain_enter, strict_enter, ctx, warmup, ema, atr, valid):
    import pandas as pd

    c = df["close"].to_numpy(dtype="float64")
    o = df["open"].to_numpy(dtype="float64")
    n = len(c)
    idx = df.index
    on_var = state_machine(enter, exit_)
    on_champ = c > ema
    on_chan = state_machine(strict_enter, c < ema - 0.25 * atr)
    assert not np.any(on_var & ~on_champ), "變體 regime 必須是冠軍 regime 的子集合"

    def fill_px(sig_bar):
        """訊號根 sig_bar 的成交價：下一根開盤；期末仍在場用最後收盤代替。"""
        return o[sig_bar + 1] if sig_bar is not None and sig_bar + 1 < n else c[n - 1]

    champ_rounds = _rounds(on_champ)
    var_rounds = _rounds(on_var)
    print(f"\n[debug] {NAME} | n={n} | warmup={warmup} | 冠軍回合 {len(champ_rounds)} | 變體回合 {len(var_rounds)} | 通道回合 {len(_rounds(on_chan))}")
    for wname, ws, we in _DEBUG_WINDOWS:
        s0 = max(warmup, int(idx.searchsorted(pd.Timestamp(ws, tz="UTC"))))
        s1 = int(idx.searchsorted(pd.Timestamp(we, tz="UTC"), side="right")) if we else n
        if s1 - s0 < 30:
            continue
        sl = slice(s0, s1)
        # a. 冷卻根數占比、被擋訊號根
        n_bars = s1 - s0
        n_ctx = int(ctx[sl].sum())
        blocked = ctx & plain_enter & ~strict_enter
        prev_on = np.concatenate(([False], on_var[:-1]))
        blocked_flat = blocked & ~prev_on                      # 變體當時空手 → 真的被擋（會延後或略過）
        print(f"  [{wname}] {idx[s0].date()}→{idx[s1 - 1].date()} 根數 {n_bars} | 在場 冠軍 {on_champ[sl].mean():.1%} 變體 {on_var[sl].mean():.1%} 通道 {on_chan[sl].mean():.1%}")
        print(f"    a. 冷卻根占比 {n_ctx / n_bars:.1%} ({n_ctx}) | 冷卻窗內 plain 真/strict 假 訊號根 {int(blocked[sl].sum())} | 其中變體空手（實際被擋） {int(blocked_flat[sl].sum())}")
        # b. 冠軍回合分類 A / D / S
        cls = {"A": [], "D": [], "S": []}
        for e, x in champ_rounds:
            if not (s0 <= e < s1):
                continue
            xe = x if x is not None else n
            seg = on_var[e:xe]
            r_champ = float(np.log(fill_px(x) / fill_px(e)))
            if seg[0]:
                cls["A"].append((e, x, r_champ, 0.0, r_champ))
            elif seg.any():
                e2 = e + int(np.argmax(seg))
                delay_cost = float(np.log(fill_px(e2) / fill_px(e)))
                r_var = float(np.log(fill_px(x) / fill_px(e2)))
                cls["D"].append((e, x, r_champ, delay_cost, r_var, e2))
            else:
                cls["S"].append((e, x, r_champ, 0.0, 0.0))
        n_rounds = sum(len(v) for v in cls.values())
        print(f"    b. 冠軍回合 {n_rounds}（A 同根進 / D 延後進 / S 整段略過）")
        for k, rows in cls.items():
            if not rows:
                print(f"       {k}: 0 回合")
                continue
            rs = np.array([r[2] for r in rows])
            wins = int((rs > 0).sum())
            line = f"       {k}: {len(rows)} 回合 | 勝 {wins} ({wins / len(rows):.0%}) | 冠軍 Σlog {rs.sum():+.3f} − 費 {_COST_PER_ROUND * len(rows):.3f} = {rs.sum() - _COST_PER_ROUND * len(rows):+.3f} | 中位持有 {np.median([((r[1] if r[1] is not None else n) - r[0]) for r in rows]):.0f} 根"
            if k == "D":
                dc = np.array([r[3] for r in rows])
                rv = np.array([r[4] for r in rows])
                line += f" | 延後成本 Σlog(o[e'+1]/o[e+1]) {dc.sum():+.3f}（中位 {np.median(dc):+.4f}，延後根數中位 {np.median([r[5] - r[0] for r in rows]):.0f}）| 變體實得 Σlog {rv.sum():+.3f}"
            if k == "S":
                line += f" | 變體避掉的損益 = {rs.sum():+.3f}（{'負期望 ✓' if rs.sum() < 0 else '正期望 ✗'}，勝率 {'<40% ✓' if wins / len(rows) < 0.4 else '≥40% ✗'}）"
            print(line)
            if k in ("S", "D") and len(rows) <= 40:
                for r in rows:
                    e, x = r[0], r[1]
                    hold = (x if x is not None else n) - e
                    tail = f" 延後 {r[5] - e} 根 成本 {r[3]:+.3%}" if k == "D" else ""
                    print(f"          {idx[e].date()}→{idx[x].date() if x is not None else 'open'} 持 {hold:>3} 根 冠軍 {r[2]:+.3%}{tail}")
        # c. 冠軍出場→下次進場的間隔分布
        gaps = []
        for (e1, x1), (e2, _x2) in zip(champ_rounds, champ_rounds[1:]):
            if x1 is not None and s0 <= e2 < s1:
                gaps.append(e2 - x1)
        if gaps:
            g = np.array(gaps)
            print(f"    c. 冠軍出場→下次進場間隔 n={len(g)} | 分位 25/50/75 = {np.percentile(g, 25):.0f}/{np.percentile(g, 50):.0f}/{np.percentile(g, 75):.0f} | ≤5 根 {np.mean(g <= 5):.1%} | ≤3 根 {np.mean(g <= 3):.1%} | >20 根 {np.mean(g > 20):.1%}")
        # d. 對 vol_atr_channel 全程空手的變體回合
        cc = []
        for e, x in var_rounds:
            if not (s0 <= e < s1):
                continue
            xe = x if x is not None else n
            if not on_chan[e:xe].any():
                cc.append(float(np.log(fill_px(x) / fill_px(e))))
        if cc:
            a = np.array(cc)
            print(f"    d. 對通道全程空手的變體回合 {len(a)} | 勝 {int((a > 0).sum())} ({(a > 0).mean():.0%}) | Σlog {a.sum():+.3f} − 費 {_COST_PER_ROUND * len(a):.3f}")
        else:
            print("    d. 對通道全程空手的變體回合 0")


if __name__ == "__main__":
    # 自我檢查：cooldown_bars=0 逐根等於 ema50_plain；enter/exit 不會同根同真
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from abgt import data as dt

    for sym in ("BTC/USDT:USDT", "ETH/USDT:USDT"):
        df = dt.load_cached("data", sym, "1d", "2019-09-01")
        if df is None:
            print(sym, "無快取")
            continue
        r0 = build(df, {**PARAMS, "cooldown_bars": 0})
        on0 = state_machine(r0["in_regime"]["enter"], r0["in_regime"]["exit"])
        plain = df["close"].to_numpy(dtype="float64") > ta.ema(df["close"].to_numpy(dtype="float64"), PARAMS["ema_len"])
        assert np.array_equal(on0, plain), "cooldown_bars=0 應逐根等於 ema50_plain"
        r = build(df, PARAMS)
        en, ex = r["in_regime"]["enter"], r["in_regime"]["exit"]
        assert not np.any(en & ex), "enter 與 exit 不可同根同真"
        on = state_machine(en, ex)
        assert not np.any(on & ~plain), "變體 regime 必須是冠軍 regime 的子集合"
        # 截斷測試：前綴截斷後 enter / exit 不變
        for cut in (400, 900, 1500, len(df) - 30):
            rc = build(df.iloc[:cut], PARAMS)
            assert np.array_equal(rc["in_regime"]["enter"], en[:cut]) and np.array_equal(rc["in_regime"]["exit"], ex[:cut]), f"截斷 {cut} 不一致"
        print(f"{sym}: OK | 回合 冠軍 {len(_rounds(plain))} → 變體 {len(_rounds(on))} | 在場 {plain.mean():.1%} → {on.mean():.1%} | warmup {r['warmup']}")
