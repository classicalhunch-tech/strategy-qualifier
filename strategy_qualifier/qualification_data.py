"""Run a strategy through a chronological, strictly gated development/holdout split."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from strategy_qualifier.backtest import BacktestConfig, Signal, Trade, run_strategy_backtest
from strategy_qualifier.qualification import (
    QualificationCriteria, QualificationResult, qualify_strategy,
)
from strategy_qualifier.qualification_gates import (
    DevelopmentEvidence, HoldoutEvidence, PreRegistration,
)


def _validate_chronology(bars: pd.DataFrame) -> None:
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
    preregistration: PreRegistration,
    preregistration_path: str | Path,
    development_evidence: DevelopmentEvidence,
    holdout_evidence_builder: Callable[[list[Trade], pd.DataFrame], HoldoutEvidence],
    criteria: QualificationCriteria,
    dev_ledger_path: str | Path,
    holdout_ledger_path: str | Path,
    code_version: str,
    data_version: str,
    config: BacktestConfig | None = None,
) -> QualificationResult:
    """Strictly qualify a frozen strategy on chronological OHLC bars.

    All mandatory statistical evidence must be computed from the matching
    development/holdout trades and provided explicitly. Missing evidence blocks
    qualification. The holdout evidence builder is called only once, after the
    single-use holdout ledger is reserved.
    """
    if not isinstance(bars, pd.DataFrame):
        raise TypeError("bars must be a pandas DataFrame")
    if not callable(strategy):
        raise TypeError("strategy must be callable")
    if not callable(holdout_evidence_builder):
        raise TypeError("holdout_evidence_builder must be callable")
    if not isinstance(split_index, int) or isinstance(split_index, bool):
        raise TypeError("split_index must be an integer")
    if split_index < 2 or split_index > len(bars) - 2:
        raise ValueError("split_index must leave at least two bars in each period")
    if bars.empty:
        raise ValueError("bars must not be empty")
    _validate_chronology(bars)

    dev_bars = bars.iloc[:split_index].copy()
    dev_trades = run_strategy_backtest(dev_bars, strategy, config)

    def run_holdout() -> tuple[list[Trade], HoldoutEvidence]:
        def holdout_only(history: pd.DataFrame):
            signal = strategy(history)
            current_bar = len(history) - 1
            if current_bar < split_index:
                return None
            return signal

        trades = run_strategy_backtest(bars.copy(), holdout_only, config)
        evidence = holdout_evidence_builder(trades, bars.iloc[split_index:].copy())
        if not isinstance(evidence, HoldoutEvidence):
            raise TypeError("holdout_evidence_builder must return HoldoutEvidence")
        return trades, evidence

    hypothesis_id = getattr(strategy, "hypothesis_id", None)
    if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
        hypothesis_id = getattr(strategy, "__name__", "")
    if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
        raise ValueError("strategy must define a non-empty __name__ or hypothesis_id attribute")

    return qualify_strategy(
        hypothesis_id=hypothesis_id.strip(),
        preregistration=preregistration,
        preregistration_path=preregistration_path,
        development_trades=dev_trades,
        development_evidence=development_evidence,
        holdout_runner=run_holdout,
        criteria=criteria,
        dev_ledger_path=dev_ledger_path,
        holdout_ledger_path=holdout_ledger_path,
        code_version=code_version,
        dev_data_version=f"{data_version}:development[0:{split_index}]",
        holdout_data_version=f"{data_version}:holdout[{split_index}:{len(bars)}]",
    )
