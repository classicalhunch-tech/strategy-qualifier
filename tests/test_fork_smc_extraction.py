import pandas as pd

from strategy_qualifier.strategies.fork_smc import (
    SignalType,
    TradeSignal,
    generate_flip_zone_signals,
    run_backtest,
)


def test_extracted_smc_api_imports():
    assert callable(generate_flip_zone_signals)
    assert callable(run_backtest)
    assert SignalType.LONG.value == "LONG"


def test_extracted_smc_replay_accepts_short_input():
    # The original replay engine intentionally returns an empty result
    # when there is insufficient history to establish a setup.
    bars = pd.DataFrame({
        "open": [100, 101, 100, 102],
        "high": [102, 103, 102, 103],
        "low": [99, 100, 99, 101],
        "close": [101, 100, 102, 102],
    }, index=pd.date_range("2024-01-01", periods=4, freq="5min"))
    result = run_backtest(bars)
    assert result.trades == []
    assert result.errors == []
