"""ATR features with explicit availability timing."""
from __future__ import annotations

import pandas as pd


def true_range(bars: pd.DataFrame) -> pd.Series:
    """Return true range per bar. Requires high, low, close columns."""
    required = {"high", "low", "close"}
    missing = required - {str(c).lower() for c in bars.columns}
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    data = bars.copy()
    data.columns = [str(c).lower() for c in data.columns]
    previous_close = data["close"].shift(1)
    ranges = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr_series(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Simple rolling ATR, including the current bar's range."""
    if period < 1:
        raise ValueError("period must be >= 1")
    return true_range(bars).rolling(period, min_periods=period).mean()


def known_atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR known at the OPEN of each bar; excludes that bar's range."""
    return atr_series(bars, period).shift(1)
