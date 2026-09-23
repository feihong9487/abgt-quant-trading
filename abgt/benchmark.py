"""買入持有基準 + 分段閘門。

閘門定義（使用者選定）：策略淨報酬 > 同期持有，且策略最大回撤 < 持有；
按季切段，每一段都要過（min_pass_ratio 可調）。
"""
from __future__ import annotations

import pandas as pd

from .metrics import max_drawdown


def buy_and_hold(close: pd.Series, initial: float, fee_pct: float) -> pd.Series:
    qty = initial * (1 - fee_pct) / close.iloc[0]
    eq = qty * close
    eq.name = "buy_and_hold"
    return eq


def segment_table(strat: pd.Series, bench: pd.Series, freq: str = "QS") -> pd.DataFrame:
    strat = strat.dropna()
    bench = bench.reindex(strat.index)
    rows = []
    prev_s = strat.shift(1)
    prev_b = bench.shift(1)
    for key, s in strat.groupby(pd.Grouper(freq=freq)):
        if len(s) < 2:
            continue
        b = bench.loc[s.index]
        base_s = prev_s.loc[s.index[0]]
        base_b = prev_b.loc[s.index[0]]
        if pd.isna(base_s):
            base_s, base_b = s.iloc[0], b.iloc[0]
        s_path = pd.concat([pd.Series([base_s]), s.reset_index(drop=True)]).to_numpy()
        b_path = pd.concat([pd.Series([base_b]), b.reset_index(drop=True)]).to_numpy()
        rs = s.iloc[-1] / base_s - 1
        rb = b.iloc[-1] / base_b - 1
        ds = max_drawdown(s_path)
        db = max_drawdown(b_path)
        rows.append({
            "segment": f"{key.year}Q{(key.month - 1) // 3 + 1}",
            "start": s.index[0], "end": s.index[-1], "bars": len(s),
            "strat_ret": rs, "bh_ret": rb, "strat_dd": ds, "bh_dd": db,
            "ret_pass": bool(rs > rb), "dd_pass": bool(ds >= db),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df["passed"] = df["ret_pass"] & df["dd_pass"]
    return df


def _gt(a: float, b: float) -> bool:
    return bool(a == a and b == b and a > b)   # NaN 一律不過


def gate(strat_summary: dict, bh_summary: dict, segments: pd.DataFrame | None = None,
         min_pass_ratio: float = 1.0, mode: str = "risk") -> dict:
    """mode="risk"（預設）：Sharpe > 持有 且 Calmar > 持有 且 最大回撤比持有小；季度只當參考。
    mode="absolute"：全期淨報酬 > 持有 且 回撤更小 且 季度通過率 ≥ min_pass_ratio（原始定義，對只做多不可達）。
    """
    s, b = strat_summary, bh_summary
    overall_ret = _gt(s["net_return"], b["net_return"])
    overall_dd = _gt(s["max_dd"], b["max_dd"])
    sharpe_ok = _gt(s["sharpe"], b["sharpe"])
    calmar_ok = _gt(s["calmar"], b["calmar"])
    ratio = float(segments["passed"].mean()) if segments is not None and len(segments) else 0.0
    n_pass = int(segments["passed"].sum()) if segments is not None and len(segments) else 0
    if mode == "risk":
        passed = sharpe_ok and calmar_ok and overall_dd
    else:
        passed = overall_ret and overall_dd and ratio >= min_pass_ratio
    return {
        "mode": mode,
        "overall_ret_pass": overall_ret,
        "overall_dd_pass": overall_dd,
        "sharpe_pass": sharpe_ok,
        "calmar_pass": calmar_ok,
        "segments_total": int(len(segments)) if segments is not None else 0,
        "segments_passed": n_pass,
        "segment_pass_ratio": ratio,
        "min_pass_ratio": min_pass_ratio,
        "passed": bool(passed),
    }
