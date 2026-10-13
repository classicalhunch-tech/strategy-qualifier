"""Run a strategy through a chronological development/holdout qualification.

The split index is the first holdout bar. The strategy receives only historical
bars available at each decision. During holdout, pre-split bars remain available
as warm-up context, but no signals are accepted before the split index.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from strategy_qualifier.backtest import BacktestConfig, Signal, Trade, run_strategy_backtest
from strategy_qualifier.qualification import (
    QualificationCriteria,
    QualificationResult,
    qualify_strategy,
)


def _validate_chronology(bars: pd.DataFrame) -> None:
    """Reject non-monotonic timestamps when the input exposes them."""
    if "timestamp" in [str(column).strip().lower() for column in bars.columns]:
        timestamp_column = next(
            column for column in bars.columns
            if str(column).strip().lower() == "timestamp"
        )
        timestamps = pd.to_datetime(bars[timestamp_column], errors="raise")
        if not timestamps.is_monotonic_increasing:
            raise ValueError("bars must be in chronological order")
    elif isinstance(bars.index, pd.DatetimeIndex) and not bars.index.is_monotonic_increasing:
        raise ValueError("bars must be in chronological order")


def qualify_bars(
    *,
    bars: pd.DataFrame,
    strategy: Callable,
    split_index: int,
    criteria: QualificationCriteria,
    dev_ledger_path: str | Path,
    holdout_ledger_path: str | Path,
    code_version: str,
    data_version: str,
    config: BacktestConfig | None = None,
) -> QualificationResult:
    """Backtest a generic strategy and qualify it on a positional time split.

    split_index is the first holdout candle. Development trades are simulated
    only on the development slice. For holdout, the strategy receives the full
    past-through-current history (including development candles for warm-up), but
    signals before split_index are discarded. A holdout signal is entered no
    earlier than the next candle under the generic OHLC backtest rules.

    Input bars must already be sorted oldest-to-newest. If a timestamp column or
    DatetimeIndex is present, ordering is checked. This function cannot establish
    that the chosen holdout period was never previously inspected; the single-use
    hypothesis ledger provides that guardrail.
    """
    if not isinstance(bars, pd.DataFrame):
        raise TypeError("bars must be a pandas DataFrame")
    if not callable(strategy):
        raise TypeError("strategy must be callable")
    if not isinstance(split_index, int) or isinstance(split_index, bool):
        raise TypeError("split_index must be an integer")
    if split_index < 2 or split_index > len(bars) - 2:
        raise ValueError("split_index must leave at least two bars in each period")
    if bars.empty:
        raise ValueError("bars must not be empty")
    _validate_chronology(bars)

    dev_bars = bars.iloc[:split_index].copy()
    dev_trades = run_strategy_backtest(dev_bars, strategy, config)

    def run_holdout() -> list[Trade]:
        def holdout_only(history: pd.DataFrame):
            # Run the strategy on its full history for correct warm-up context.
            signal = strategy(history)
            current_bar = len(history) - 1
            if current_bar < split_index:
                return None
            return signal

        return run_strategy_backtest(bars.copy(), holdout_only, config)

    hypothesis_id = getattr(strategy, "hypothesis_id", None)
    if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
        hypothesis_id = getattr(strategy, "__name__", "")
    if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
        raise ValueError("strategy must define a non-empty __name__ or hypothesis_id attribute")

    return qualify_strategy(
        hypothesis_id=hypothesis_id.strip(),
        development_trades=dev_trades,
        holdout_runner=run_holdout,
        criteria=criteria,
        dev_ledger_path=dev_ledger_path,
        holdout_ledger_path=holdout_ledger_path,
        code_version=code_version,
        dev_data_version=f"{data_version}:development[0:{split_index}]",
        holdout_data_version=f"{data_version}:holdout[{split_index}:{len(bars)}]",
    )
