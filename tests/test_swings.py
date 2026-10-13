import pandas as pd

from strategy_qualifier.market.swings import find_swings


def test_confirmed_swings_are_timestamped_no_earlier_than_formation():
    idx = pd.date_range("2025-01-01", periods=12, freq="h")
    df = pd.DataFrame({
        "open":  [10, 11, 12, 11, 10, 9, 10, 11, 12, 11, 10, 9],
        "high":  [11, 12, 13, 12, 11, 10, 11, 12, 13, 12, 11, 10],
        "low":   [9, 10, 11, 10, 9, 8, 9, 10, 11, 10, 9, 8],
        "close": [10, 11, 12, 10, 9, 9, 10, 11, 12, 10, 9, 9],
    }, index=idx)
    swings = find_swings(df)
    assert all(s.confirmed_at >= s.formed_at for s in swings)


def test_empty_input_returns_no_swings():
    df = pd.DataFrame(columns=["open", "high", "low", "close"])
    assert find_swings(df) == []
