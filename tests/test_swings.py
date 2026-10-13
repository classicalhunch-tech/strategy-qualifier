import pandas as pd
import pytest

from strategy_qualifier.market.swings import SwingType, find_swings


def test_pivot_is_timestamped_when_confirmed_not_when_formed():
    idx = pd.date_range("2025-01-01", periods=5, freq="h")
    df = pd.DataFrame({"high":[1,2,5,3,2], "low":[0,1,2,1,0]}, index=idx)
    found = find_swings(df, left=1, right=1)
    high = next(s for s in found if s.swing_type is SwingType.HIGH)
    assert high.formed_at == idx[2]
    assert high.confirmed_at == idx[3]


def test_swings_require_datetime_index():
    with pytest.raises(ValueError):
        find_swings(pd.DataFrame({"high":[1,2,1], "low":[0,1,0]}))


def test_swings_reject_tied_timestamps():
    idx = pd.DatetimeIndex(["2025-01-01", "2025-01-01", "2025-01-02"])
    df = pd.DataFrame({"high":[1,2,1], "low":[0,1,0]}, index=idx)
    with pytest.raises(ValueError):
        find_swings(df, 1, 1)
