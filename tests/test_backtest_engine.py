import pandas as pd
import pytest

from strategy_qualifier.backtest import BacktestConfig, Signal, run_backtest


def bars(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


def test_signal_enters_on_next_bar_not_signal_bar():
    data = bars([(100, 101, 99, 100), (101, 104, 100, 103), (103, 104, 102, 103)])
    trades = run_backtest(data, [Signal(0, 1, 98, 105, "H1")])
    assert len(trades) == 1
    assert trades[0].entry_bar == 1
    assert trades[0].entry_price == 101


def test_stop_wins_when_stop_and_target_touch_same_bar():
    data = bars([(100, 101, 99, 100), (100, 106, 94, 100)])
    trades = run_backtest(data, [Signal(0, 1, 95, 105)])
    assert trades[0].exit_reason == "STOP"
    assert trades[0].net_pnl == -5


def test_spread_slippage_and_commission_reduce_pnl():
    data = bars([(100, 101, 99, 100), (100, 104, 99, 103)])
    signal = [Signal(0, 1, 95, 103)]
    plain = run_backtest(data, signal)[0]
    costly = run_backtest(
        data, signal,
        BacktestConfig(spread=0.2, slippage_per_side=0.1,
                       commission_per_unit_per_side=0.25),
    )[0]
    assert costly.net_pnl < plain.net_pnl
    assert costly.costs > 0


def test_rejects_bad_ohlc_and_duplicate_signals():
    data = bars([(100, 99, 98, 100), (100, 101, 99, 100)])
    with pytest.raises(ValueError):
        run_backtest(data, [Signal(0, 1, 95, 105)])
    valid = bars([(100, 101, 99, 100), (100, 101, 99, 100)])
    with pytest.raises(ValueError):
        run_backtest(valid, [Signal(0, 1, 95, 105), Signal(0, -1, 105, 95)])


def test_invalid_signal_direction_is_rejected():
    data = bars([(100, 101, 99, 100), (100, 101, 99, 100)])
    with pytest.raises(ValueError):
        run_backtest(data, [Signal(0, 0, 95, 105)])


def test_rejects_non_finite_prices_and_costs():
    data = bars([(100, 101, 99, 100), (100, 101, 99, 100)])
    bad = bars([(100, float("inf"), 99, 100), (100, 101, 99, 100)])
    with pytest.raises(ValueError):
        run_backtest(bad, [Signal(0, 1, 95, 105)])
    with pytest.raises(ValueError):
        run_backtest(data, [Signal(0, 1, 95, 105)],
                     BacktestConfig(spread=float("nan")))


def test_skips_entry_beyond_target_after_gap_open():
    data = bars([(100, 101, 99, 100), (106, 108, 105, 107)])
    trades = run_backtest(data, [Signal(0, 1, 95, 105)])
    assert trades == []
