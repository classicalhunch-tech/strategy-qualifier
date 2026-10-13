import json

import pandas as pd
import pytest

from strategy_qualifier.backtest import BacktestConfig, Signal, Trade
from strategy_qualifier.qualification import QualificationCriteria, qualify_strategy
from strategy_qualifier.qualification_data import qualify_bars


def trade(net_pnl, r_multiple):
    return Trade(
        hypothesis_id="H1", side=1, signal_bar=0, entry_bar=1, exit_bar=2,
        entry_price=100.0, exit_price=100.0 + net_pnl, stop_loss=95.0,
        take_profit=110.0, exit_reason="TARGET" if net_pnl > 0 else "STOP",
        bars_held=2, gross_pnl=net_pnl, costs=0.0, net_pnl=net_pnl,
        initial_risk=5.0, r_multiple=r_multiple,
    )


def criteria():
    # Explicit test-only gates; production users must choose their own.
    return QualificationCriteria(min_trades=3, min_expectancy=0.0, max_drawdown=1000.0)


def test_development_failure_never_calls_holdout(tmp_path):
    called = []
    result = qualify_strategy(
        hypothesis_id="weak-v1",
        development_trades=[trade(-1, -0.2), trade(-2, -0.4)],
        holdout_runner=lambda: called.append(True) or [trade(5, 1)],
        criteria=criteria(),
        dev_ledger_path=tmp_path / "dev.csv",
        holdout_ledger_path=tmp_path / "holdout.json",
        code_version="test-commit",
        dev_data_version="dev-fixture-v1",
        holdout_data_version="holdout-fixture-v1",
    )
    assert result.stage == "development"
    assert result.decision == "INCONCLUSIVE"
    assert called == []
    assert not (tmp_path / "holdout.json").exists()


def test_survivor_runs_holdout_once_and_records_verdict(tmp_path):
    calls = []
    dev = [trade(2, 0.4), trade(3, 0.6), trade(1, 0.2), trade(4, 0.8)]
    holdout = [trade(2, 0.4), trade(3, 0.6), trade(1, 0.2), trade(4, 0.8)]

    def run_holdout():
        calls.append(True)
        return holdout

    kwargs = dict(
        hypothesis_id="strong-v1", development_trades=dev,
        holdout_runner=run_holdout, criteria=criteria(),
        dev_ledger_path=tmp_path / "dev.csv",
        holdout_ledger_path=tmp_path / "holdout.json",
        code_version="test-commit", dev_data_version="dev-fixture-v1",
        holdout_data_version="holdout-fixture-v1",
    )
    result = qualify_strategy(**kwargs)
    assert result.stage == "holdout"
    assert result.holdout_decision == "PASS"
    assert len(calls) == 1
    ledger = json.loads((tmp_path / "holdout.json").read_text())
    assert ledger[0]["status"] == "COMPLETED"
    assert ledger[0]["decision"] == "PASS"

    with pytest.raises(ValueError, match="holdout already used"):
        qualify_strategy(**kwargs)
    assert len(calls) == 1


def test_failed_holdout_callback_is_still_consumed(tmp_path):
    kwargs = dict(
        hypothesis_id="error-v1",
        development_trades=[trade(2, 0.4), trade(3, 0.6), trade(1, 0.2), trade(4, 0.8)],
        holdout_runner=lambda: (_ for _ in ()).throw(RuntimeError("fixture failure")),
        criteria=criteria(), dev_ledger_path=tmp_path / "dev.csv",
        holdout_ledger_path=tmp_path / "holdout.json",
        code_version="test-commit", dev_data_version="dev-v1",
        holdout_data_version="holdout-v1",
    )
    with pytest.raises(RuntimeError, match="fixture failure"):
        qualify_strategy(**kwargs)
    ledger = json.loads((tmp_path / "holdout.json").read_text())
    assert ledger[0]["status"] == "ERROR_HOLDOUT_CONSUMED"
    with pytest.raises(ValueError, match="holdout already used"):
        qualify_strategy(**kwargs)


def synthetic_bars(n=24):
    # Every entry candle reaches the long target; OHLC relationships are valid.
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "open": [100.0] * n,
        "high": [103.0] * n,
        "low": [99.0] * n,
        "close": [100.5] * n,
    })


def winning_strategy(history):
    i = len(history) - 1
    return Signal(
        bar_index=i, side=1, stop_loss=95.0, take_profit=102.0,
        hypothesis_id="synthetic-winner-v1",
    )


winning_strategy.hypothesis_id = "synthetic-winner-v1"


def losing_strategy(history):
    i = len(history) - 1
    return Signal(
        bar_index=i, side=1, stop_loss=99.5, take_profit=102.0,
        hypothesis_id="synthetic-loser-v1",
    )


losing_strategy.hypothesis_id = "synthetic-loser-v1"


def test_ohlc_to_holdout_qualification_end_to_end(tmp_path):
    result = qualify_bars(
        bars=synthetic_bars(),
        strategy=winning_strategy,
        split_index=12,
        criteria=criteria(),
        dev_ledger_path=tmp_path / "dev.csv",
        holdout_ledger_path=tmp_path / "holdout.json",
        code_version="synthetic-test",
        data_version="synthetic-ohlc-v1",
        config=BacktestConfig(spread=0.1, slippage_per_side=0.01),
    )
    assert result.stage == "holdout"
    assert result.decision == "PASS"
    assert result.metrics["trade_count"] >= 3
    assert result.holdout_metrics["trade_count"] >= 3
    ledger = json.loads((tmp_path / "holdout.json").read_text())
    assert ledger[0]["data_version"] == "synthetic-ohlc-v1:holdout[12:24]"
    assert ledger[0]["status"] == "COMPLETED"


def test_ohlc_to_qualification_rejects_weak_strategy_without_holdout(tmp_path):
    result = qualify_bars(
        bars=synthetic_bars(),
        strategy=losing_strategy,
        split_index=12,
        criteria=criteria(),
        dev_ledger_path=tmp_path / "dev.csv",
        holdout_ledger_path=tmp_path / "holdout.json",
        code_version="synthetic-test",
        data_version="synthetic-ohlc-v1",
    )
    assert result.stage == "development"
    assert result.decision == "RULED OUT"
    assert not (tmp_path / "holdout.json").exists()


def test_qualification_split_rejects_unsorted_timestamp_data(tmp_path):
    bars = synthetic_bars()
    bars.loc[[0, 1], "timestamp"] = bars.loc[[1, 0], "timestamp"].to_numpy()
    with pytest.raises(ValueError, match="chronological"):
        qualify_bars(
            bars=bars, strategy=winning_strategy, split_index=12,
            criteria=criteria(), dev_ledger_path=tmp_path / "dev.csv",
            holdout_ledger_path=tmp_path / "holdout.json",
            code_version="test", data_version="unordered",
        )
