"""情境自適應進場：影子鞭打計數（通道出場 −0.25 ATR）。

出場固定 close < EMA50 − k_out×ATR14（= vol_atr_channel 的出場）。
進場門檻依情境二選一：
    plain  : close > EMA50                      （= ema50_plain 的進場）
    strict : close > EMA50 + k_in_strict×ATR14  （= vol_atr_channel 的進場）
情境由一個「影子狀態機」決定：影子永遠用 plain 門檻與同一出場跑一個假想部位，
每次影子出場時若「持有 ≤ whip_hold_max 根 且 收盤對收盤虧損」就記一次鞭打。
    ctx_strict[i] = 過去 whip_window 根內（出場根落在 (i−W, i]）鞭打回合數 ≥ whip_min_count
                    且 最近一次鞭打之後影子沒有持有過 ≥ trend_hold_min 根
ctx_strict 為真時本根用 strict 門檻，否則用 plain；引擎用 state_machine(enter, exit_) 維護實際部位。

為何用影子而不是自己的回合：strict 模式本身會壓掉鞭打樣本，若計自己的回合會在 W 根後自動切回 plain
再被掃一次（極限環）；影子是純價格函數，截斷前視檢查與向量化重算都簡單。

巢狀關係：whip_min_count → ∞（或 k_in_strict = 0）= 「k_in = 0、k_out = 0.25 的通道」；ctx_strict 恆真 = vol_atr_channel 逐位相同。

無前視：EMA / ATR 因果（RMA 遞迴、TR 用 close[i−1]）；影子迴圈只由前往後、只讀 ≤ i 的收盤；
cnt 只數出場根 ≤ i 的回合；enter / exit 每根只讀 ≤ i 的 OHLC；引擎在 i+1 開盤成交。
enter 與 exit 不可能同根同時為真（任一 enter 要 c > ema，exit 要 c < ema − k_out×atr，atr ≥ 0）。

診斷：環境變數 ABGT_DEBUG=1 時印五視窗的情境占比 / 翻轉 / 影子回合分布 / 進場模式 / 回合 A-B-C 歸因（不影響評估）。
"""
import os

import numpy as np

from abgt import indicators as ta
from abgt.strategies.regime import RiskParams, state_machine

NAME = "情境自適應進場：影子鞭打計數（通道出場）"
DESCRIPTION = ("出場 close < EMA50 − 0.25×ATR14 固定；進場 close > EMA50（plain）或 close > EMA50 + 1.0×ATR14（strict）；"
               "影子 plain 狀態機過去 60 根內 ≥ 2 個持有 ≤ 5 根且虧損的回合 → strict，影子持有 ≥ 20 根或視窗衰減 → plain；無風控")
PARAMS = {"ema_len": 50, "atr_len": 14, "k_in_strict": 1.0, "k_out": 0.25,
          "whip_hold_max": 5, "whip_window": 60, "whip_min_count": 2, "trend_hold_min": 20}

_DEBUG_WINDOWS = (("IS", "2020-01-01", "2022-12-31"), ("OOS", "2023-01-01", None), ("FULL", "2020-01-01", None),
                  ("IS2", "2020-07-01", "2023-12-31"), ("OOS2", "2024-01-01", None))
_COST_LOG = 0.0014      # 一個回合（進 + 出）的手續費 + 滑點，下一根開盤成交口徑


def build(df, p):
    c = df["close"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    n = len(c)
    ema = ta.ema(c, int(p["ema_len"]))                          # Pine 語意：第 i 根只用 ≤ i 的收盤
    atr = ta.atr(h, l, c, int(p["atr_len"]))                    # RMA(TR)，TR 用 close[i−1]
    valid = ~np.isnan(ema) & ~np.isnan(atr)
    warmup = int(np.argmax(valid)) if valid.any() else n        # 預設 = 49，與 ema50_plain / vol_atr_channel 同視窗
    plain_enter = c > ema                                       # 寬鬆門檻（NaN 比較 → False）
    strict_enter = c > ema + float(p["k_in_strict"]) * atr      # 嚴格門檻 = vol_atr_channel 的進場
    exit_ = c < ema - float(p["k_out"]) * atr                   # 固定出場 = vol_atr_channel 的出場
    W, HMAX, KMIN, TMIN = int(p["whip_window"]), int(p["whip_hold_max"]), int(p["whip_min_count"]), int(p["trend_hold_min"])

    # 影子狀態機：永遠用 plain 門檻與同一出場的假想部位；只用來產生情境，不受實際部位影響
    sh_on = False
    e0 = -1
    whip_exits: list[int] = []          # 鞭打回合的出場根（只留 > i − W 的）
    last_whip = -1
    last_trend = -1
    ctx_strict = np.zeros(n, dtype=bool)
    enter = np.zeros(n, dtype=bool)
    shadow_rounds: list[tuple[int, int, int, float, bool]] = []   # (e0, x, hold, ret, is_whip) 診斷用
    for i in range(n):
        if not valid[i]:
            sh_on = False
            continue                                            # enter[i] 保持 False（暖機期趨勢外）
        # (1) 趨勢成立標記：影子目前持有 ≥ TMIN 根（先於本根的出場判定，讓恰好 TMIN 根出場的回合也算）
        if sh_on and (i - e0) >= TMIN:
            last_trend = i
        # (2) 影子狀態機吃第 i 根收盤
        if not sh_on and plain_enter[i]:
            sh_on = True
            e0 = i
        elif sh_on and exit_[i]:
            sh_on = False
            hold = i - e0                                       # 訊號根計的持有根數
            ret = c[i] / c[e0] - 1.0                            # 收盤對收盤，兩者皆 ≤ i
            is_whip = hold <= HMAX and ret < 0.0
            if is_whip:
                whip_exits.append(i)
                last_whip = i
            shadow_rounds.append((e0, i, hold, ret, is_whip))
        # (3) 情境：出場根落在 (i−W, i] 的鞭打回合數 ≥ KMIN，且最近一次鞭打之後影子沒有持有過 ≥ TMIN 根
        while whip_exits and whip_exits[0] <= i - W:
            whip_exits.pop(0)
        cnt = len(whip_exits)
        ctx_strict[i] = (cnt >= KMIN) and (last_whip > last_trend)
        # (4) 本根進場門檻
        enter[i] = bool(strict_enter[i]) if ctx_strict[i] else bool(plain_enter[i])

    if os.environ.get("ABGT_DEBUG"):
        _debug(df, enter, exit_, strict_enter, ctx_strict, warmup, p, shadow_rounds)

    return {"in_regime": {"enter": enter, "exit": exit_}, "risk": RiskParams(), "warmup": warmup}


# ---------------------------------------------------------------- 診斷 ----------------------------------------------------------------

def _rounds(on: np.ndarray) -> list[tuple[int, int | None]]:
    """把 on 陣列切成 [e, x) 回合：e = 進場訊號根（on 由 False 轉 True），x = 出場訊號根（on 由 True 轉 False），None = 期末仍持有。"""
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


def _debug(df, enter, exit_, strict_enter, ctx_strict, warmup, p, shadow_rounds):
    import pandas as pd

    idx = df.index
    o = df["open"].to_numpy(dtype="float64")
    c = df["close"].to_numpy(dtype="float64")
    n = len(c)
    on_var = state_machine(enter, exit_)
    on_chan = state_machine(strict_enter, exit_)                # = vol_atr_channel
    chan_entry = on_chan & ~ta.shift_bool(on_chan, 1)
    rounds = _rounds(on_var)
    tag = "[ctxwhip]"
    print(f"{tag} params={p} warmup={warmup} ({idx[warmup].date() if warmup < n else 'n/a'}) bars={n} "
          f"{idx[0].date()}→{idx[-1].date()} last_close={c[-1]:.2f} rounds_var={len(rounds)} rounds_chan={len(_rounds(on_chan))} "
          f"shadow_rounds={len(shadow_rounds)} shadow_whips={sum(1 for r in shadow_rounds if r[4])}")

    def pnl(e, x):
        """下一根開盤成交：進 o[e+1]、出 o[x+1]；期末未平倉或出場成交根超出資料時用最後收盤代替並標記。"""
        px_in = o[e + 1] if e + 1 < n else c[n - 1]
        if x is None or x + 1 >= n:
            return float(np.log(c[n - 1] / px_in)), True
        return float(np.log(o[x + 1] / px_in)), False

    for wname, ws, we in _DEBUG_WINDOWS:
        s0 = max(warmup, int(idx.searchsorted(pd.Timestamp(ws, tz="UTC"))))
        s1 = n if we is None else int(idx.searchsorted(pd.Timestamp(we, tz="UTC"), side="right"))
        if s1 - s0 < 30:
            continue
        seg = slice(s0, s1)
        # a. 情境
        cs = ctx_strict[seg]
        share = float(cs.mean())
        flips = int(np.sum(cs[1:] != cs[:-1]))
        sh_w = [r for r in shadow_rounds if s0 <= r[1] < s1]
        holds = np.array([r[2] for r in sh_w], dtype=float)
        q = np.percentile(holds, [25, 50, 75]) if len(holds) else [np.nan] * 3
        n_whip = sum(1 for r in sh_w if r[4])
        print(f"{tag} {wname:5} a. strict_share={share:.3f} ctx_flips={flips} shadow_rounds={len(sh_w)} shadow_whips={n_whip} "
              f"hold_q25/50/75={q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}")
        # b. 進場模式
        rw = [(e, x) for e, x in rounds if s0 <= e < s1]
        n_plain = sum(1 for e, _ in rw if not ctx_strict[e])
        n_strict = len(rw) - n_plain
        print(f"{tag} {wname:5} b. entries={len(rw)} plain={n_plain} strict={n_strict}")
        # c. 回合 A / B / C 歸因（以進場訊號根落在視窗內計；on_var ⊇ on_chan 且出場相同，所以通道只可能在 [e, x) 內晚進、同根出）
        cls = {"A": [], "B": [], "C": []}
        early_inc = 0.0
        open_rounds = 0
        for e, x in rw:
            r_log, is_open = pnl(e, x)
            open_rounds += int(is_open)
            x_eff = n if x is None else x
            if chan_entry[e]:
                cls["A"].append(r_log)
            else:
                later = np.flatnonzero(chan_entry[e + 1:x_eff])
                if len(later):
                    e2 = e + 1 + int(later[0])
                    cls["B"].append(r_log)
                    px_in = o[e + 1] if e + 1 < n else c[n - 1]
                    px_e2 = o[e2 + 1] if e2 + 1 < n else c[n - 1]
                    early_inc += float(np.log(px_e2 / px_in))
                else:
                    cls["C"].append(r_log)
        parts = []
        for k in ("A", "B", "C"):
            v = np.array(cls[k], dtype=float)
            net = v - _COST_LOG
            wins = int(np.sum(net > 0))
            parts.append(f"{k}: n={len(v)} wins={wins} sum_net_log={float(net.sum()):+.4f}")
        print(f"{tag} {wname:5} c. " + " | ".join(parts) + f" | B_early_inc_log={early_inc:+.4f} | open_rounds={open_rounds}")
        # d. 判準：C 類勝率、plain 模式進場的回合勝率與對數合計
        vC = np.array(cls["C"], dtype=float) - _COST_LOG
        c_wr = float(np.mean(vC > 0)) if len(vC) else float("nan")
        plain_logs = np.array([pnl(e, x)[0] for e, x in rw if not ctx_strict[e]], dtype=float) - _COST_LOG
        p_wr = float(np.mean(plain_logs > 0)) if len(plain_logs) else float("nan")
        print(f"{tag} {wname:5} d. C_win_rate={c_wr:.3f} (n={len(vC)}) | plain_mode_rounds={len(plain_logs)} "
              f"plain_win_rate={p_wr:.3f} plain_sum_net_log={float(plain_logs.sum()):+.4f}")
