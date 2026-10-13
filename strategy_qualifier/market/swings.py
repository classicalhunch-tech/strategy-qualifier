"""Confirmed pivot swings; each pivot is emitted only after right-side bars exist."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class SwingType(str, Enum):
    HIGH = "high"
    LOW = "low"


@dataclass(frozen=True)
class Swing:
    swing_type: SwingType
    price: float
    formed_at: pd.Timestamp
    confirmed_at: pd.Timestamp


def find_swings(bars: pd.DataFrame, left: int = 2, right: int = 2) -> list[Swing]:
    """Find strict local high/low pivots in time order.

    A pivot at row i is not knowable until row i + right has closed.
    Ties are deliberately not classified as strict pivots.
    """
    if left < 1 or right < 1:
        raise ValueError("left and right must both be >= 1")
    data = bars.copy()
    data.columns = [str(c).lower() for c in data.columns]
    missing = {"high", "low"} - set(data.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("bars must use a DatetimeIndex")
    if data[["high", "low"]].isna().any().any():
        raise ValueError("high/low columns cannot contain NaN")
    data = data.sort_index()
    if data.index.has_duplicates:
        raise ValueError("bar timestamps must be unique")

    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    times = data.index
    found: list[Swing] = []
    for i in range(left, len(data) - right):
        hwin = highs[i-left:i+right+1]
        lwin = lows[i-left:i+right+1]
        if highs[i] == hwin.max() and (hwin == highs[i]).sum() == 1:
            found.append(Swing(SwingType.HIGH, float(highs[i]), times[i], times[i+right]))
        if lows[i] == lwin.min() and (lwin == lows[i]).sum() == 1:
            found.append(Swing(SwingType.LOW, float(lows[i]), times[i], times[i+right]))
    return sorted(found, key=lambda s: (s.confirmed_at, s.formed_at, s.swing_type.value))
