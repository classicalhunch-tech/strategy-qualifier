import pandas as pd

from strategy_qualifier.backtest import Signal, run_backtest


def data(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


def test_limit_entry_waits_for_touch_and_uses_planned_price():
    bars = data([
        (100, 101, 99, 100),
        (101, 103, 100, 102),
        (102, 104, 98, 103),
        (103, 106, 102, 105),
    ])
    trade = run_backtest(
        bars,
        [Signal(0, 1, 97, 105, "limit-demo", entry_price=100, entry_expiry_bars=3)],
    )[0]
    assert trade.entry_bar == 2
    assert trade.entry_price == 100
    assert trade.exit_reason == "TARGET"
    assert trade.net_pnl == 5


def test_limit_entry_expires_without_a_fill():
    bars = data([
        (100, 101, 99, 100),
        (101, 104, 101, 103),
        (103, 105, 102, 104),
        (104, 106, 103, 105),
    ])
    trades = run_backtest(
        bars,
        [Signal(0, 1, 95, 110, "expired", entry_price=98, entry_expiry_bars=2)],
    )
    assert trades == []
