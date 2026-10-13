import pandas as pd
import pytest

from strategy_qualifier.backtest import (
    BacktestConfig,
    Signal,
    run_strategy_backtest,
    summarize_trades,
)


def make_bars(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


def test_strategy_callback_only_receives_past_and_current_bars():
    data = make_bars([
        (100, 101, 99, 100),
        (101, 102, 100, 101),
        (102, 104, 101, 103),
    ])
    seen_lengths = []

    def strategy(history):
        seen_lengths.append(len(history))
        assert len(history) < len(data)
        if len(history) == 1:
            return Signal(bar_index=0, side=1, stop_loss=95, take_profit=105, hypothesis_id="demo")
        return None

    trades = run_strategy_backtest(data, strategy)
    assert seen_lengths == [1, 2]
    assert len(trades) == 1
    assert trades[0].entry_bar == 1


def test_strategy_cannot_emit_signal_for_a_future_or_old_bar():
    data = make_bars([(100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100)])

    def invalid_strategy(history):
        return Signal(bar_index=0, side=1, stop_loss=95, take_profit=105)

    with pytest.raises(ValueError, match="current bar"):
        run_strategy_backtest(data, invalid_strategy)


def test_summary_metrics_are_strategy_agnostic():
    data = make_bars([
        (100, 101, 99, 100),
        (100, 106, 99, 105),
        (105, 106, 103, 104),
        (104, 105, 100, 101),
    ])
    trades = run_strategy_backtest(
        data,
        lambda history: Signal(len(history) - 1, 1, 95, 105, "demo")
        if len(history) == 1 else None,
        BacktestConfig(),
    )
    summary = summarize_trades(trades)
    assert summary["trade_count"] == 1
    assert summary["net_pnl"] == 5
    assert summary["win_rate"] == 1
    assert summary["expectancy"] == 5
    assert summary["profit_factor"] == float("inf")
    assert summary["max_drawdown"] == 0
    assert summary["average_r"] > 0


def test_empty_trade_summary_is_defined():
    summary = summarize_trades([])
    assert summary["trade_count"] == 0
    assert summary["net_pnl"] == 0
    assert summary["win_rate"] == 0
    assert summary["expectancy"] == 0
