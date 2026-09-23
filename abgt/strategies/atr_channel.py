"""正式策略：EMA ± ATR 不對稱通道遲滯（第 1 輪冠軍 vol_atr_channel）。

規則：close > EMA50 + k_in×ATR14 → 進入趨勢；close < EMA50 − k_out×ATR14 → 離開；通道內維持前一狀態。
定位（第 1 輪驗證共識）：OOS 風險調整報酬 ≈ 純 EMA50，優勢在回撤（六格全淺 6–15 pp）與換手（基準的 1/2.5）。
上線前提：先只上 BTC 現貨；狀態機有路徑依賴，實盤要從 2019-09 重放完整歷史初始化。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .regime import RiskParams, regime_atr_channel, simulate_regime


@dataclass
class ATRChannelParams:
    ema_len: int = 50
    atr_len: int = 14
    k_in: float = 1.0
    k_out: float = 0.25


def build(df: pd.DataFrame, p: ATRChannelParams | None = None, risk: RiskParams | None = None) -> dict:
    p = p or ATRChannelParams()
    enter, exit_, warmup = regime_atr_channel(df, p.ema_len, p.atr_len, p.k_in, p.k_out)
    return {"in_regime": {"enter": enter, "exit": exit_}, "risk": risk or RiskParams(), "warmup": warmup}


def run(df: pd.DataFrame, start_idx: int | None = None, p: ATRChannelParams | None = None, risk: RiskParams | None = None):
    built = build(df, p, risk)
    s0 = built["warmup"] if start_idx is None else max(built["warmup"], start_idx)
    return simulate_regime(df, built["in_regime"], built["risk"], s0)
