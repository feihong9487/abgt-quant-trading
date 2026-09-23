"""部位帳本：進出場、手續費、每筆出場紀錄、平倉→平倉回合。策略模擬器共用。"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Entry:
    id: str
    side: int          # +1 多 / -1 空
    qty: float
    price: float
    bar: int
    time: pd.Timestamp
    fee_paid: float
    orig_qty: float


@dataclass
class BacktestResult:
    equity: pd.Series                 # 每根 K 收盤的權益（從 start_idx 起）
    trades: pd.DataFrame              # 每筆出場紀錄（TradingView closedtrades 口徑：分批出場各算一筆）
    round_trips: pd.DataFrame         # 平倉→平倉回合
    start_idx: int
    extra: dict = field(default_factory=dict)


class Book:
    def __init__(self, initial: float, fee_pct: float):
        self.initial = initial
        self.fee = fee_pct
        self.entries: list[Entry] = []
        self.realized = 0.0           # 已實現損益，含所有已付手續費
        self.trades: list[dict] = []
        self.trips: list[dict] = []
        self._trip: dict | None = None
        self._seq = 0

    # --- 狀態 ---
    def position(self) -> float:
        return sum(e.side * e.qty for e in self.entries)

    def open_pnl(self, price: float) -> float:
        return sum(e.side * (price - e.price) * e.qty for e in self.entries)

    def equity(self, price: float) -> float:
        return self.initial + self.realized + self.open_pnl(price)

    # --- 動作 ---
    def open(self, side: int, qty: float, price: float, bar: int, time: pd.Timestamp, eid: str | None = None) -> Entry:
        if not self.entries:
            self._trip = {"start_time": time, "start_bar": bar, "side": side, "n_entries": 0, "_base": self.realized}
        fee = qty * price * self.fee
        self.realized -= fee
        self._seq += 1
        e = Entry(id=eid or f"{'L' if side > 0 else 'S'}{len(self.entries) + 1}", side=side, qty=qty, price=price,
                  bar=bar, time=time, fee_paid=fee, orig_qty=qty)
        self.entries.append(e)
        self._trip["n_entries"] += 1
        return e

    def close(self, e: Entry, qty: float, price: float, bar: int, time: pd.Timestamp, reason: str) -> float:
        qty = min(qty, e.qty)
        if qty <= 0:
            return 0.0
        gross = e.side * (price - e.price) * qty
        fee = qty * price * self.fee
        entry_fee_part = e.fee_paid * (qty / e.orig_qty)
        pnl = gross - fee - entry_fee_part
        self.realized += gross - fee
        self.trades.append({
            "entry_id": e.id, "side": e.side, "qty": qty, "entry_price": e.price, "exit_price": price,
            "entry_time": e.time, "exit_time": time, "bars_held": bar - e.bar, "pnl": pnl, "reason": reason,
        })
        e.qty -= qty
        if e.qty <= 1e-12:
            self.entries.remove(e)
        if not self.entries and self._trip is not None:
            t = self._trip
            self.trips.append({
                "start_time": t["start_time"], "end_time": time, "side": t["side"], "n_entries": t["n_entries"],
                "bars": bar - t["start_bar"], "pnl": self.realized - t["_base"], "last_reason": reason,
            })
            self._trip = None
        return pnl

    def close_all(self, price: float, bar: int, time: pd.Timestamp, reason: str) -> None:
        for e in list(self.entries):
            self.close(e, e.qty, price, bar, time, reason)

    # --- 輸出 ---
    def trades_df(self) -> pd.DataFrame:
        cols = ["entry_id", "side", "qty", "entry_price", "exit_price", "entry_time", "exit_time", "bars_held", "pnl", "reason"]
        return pd.DataFrame(self.trades, columns=cols)

    def trips_df(self) -> pd.DataFrame:
        cols = ["start_time", "end_time", "side", "n_entries", "bars", "pnl", "last_reason"]
        return pd.DataFrame(self.trips, columns=cols)
