import pandas as pd
import pytest

from strategy_qualifier.market.atr import atr_series, known_atr, true_range


def make_bars():
    idx = pd.date_range("2025-01-01", periods=5, freq="h")
    return pd.DataFrame(
        {"open": [1, 2, 3, 2, 4], "high": [2, 4, 4, 5, 6],
         "low": [0, 1, 2, 1, 3], "close": [1, 3, 2, 4, 5]},
        index=idx,
    )


def test_true_range_includes_gap_from_previous_close():
    values = true_range(make_bars())
    assert values.iloc[1] == 3


def test_known_atr_lags_current_bar():
    values = known_atr(make_bars(), period=2)
    assert pd.isna(values.iloc[1])
    assert values.iloc[2] == pytest.approx(2.5)


def test_atr_rejects_invalid_period():
    with pytest.raises(ValueError):
        atr_series(make_bars(), period=0)
