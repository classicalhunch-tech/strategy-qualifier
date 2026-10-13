"""Run a strategy through a chronological, strictly gated development/holdout split."""
from __future__ import annotations

from dataclasses import replace
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


def _validate_inputs(bars: pd.DataFrame, strategy: Callable, split_index: int) -> None:
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


def _hypothesis_id(strategy: Callable) -> str:
    value = getattr(strategy, "hypothesis_id", None)
    if not isinstance(value, str) or not value.strip():
        value = getattr(strategy, "__name__", "")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("strategy must define a non-empty __name__ or hypothesis_id attribute")
    return value.strip()


def _holdout_trades_only(bars: pd.DataFrame, strategy: Callable, split_index: int,
                         config: BacktestConfig | None) -> list[Trade]:
    def holdout_only(history: pd.DataFrame):
        signal = strategy(history)
        current_bar = len(history) - 1
        if current_bar < split_index:
            return None
        return signal

    full_run = run_strategy_backtest(bars.copy(), holdout_only, config)
    # The evidence builder receives only holdout bars. Convert the bar indices
    # to that slice's coordinate system so timestamps cannot accidentally index
    # into the development period.
    return [
        replace(
            trade,
            signal_bar=trade.signal_bar - split_index,
            entry_bar=trade.entry_bar - split_index,
            exit_bar=trade.exit_bar - split_index,
        )
        for trade in full_run
        if trade.signal_bar >= split_index and trade.entry_bar >= split_index
    ]


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
    """Qualification API for callers that already computed evidence independently.

    Prefer qualify_bars_automatic() for the integrated, research-tool-backed path.
    """
    _validate_inputs(bars, strategy, split_index)
    if not callable(holdout_evidence_builder):
        raise TypeError("holdout_evidence_builder must be callable")
    dev_bars = bars.iloc[:split_index].copy()
    dev_trades = run_strategy_backtest(dev_bars, strategy, config)

    def run_holdout() -> tuple[list[Trade], HoldoutEvidence]:
        trades = _holdout_trades_only(bars, strategy, split_index, config)
        evidence = holdout_evidence_builder(trades, bars.iloc[split_index:].copy())
        if not isinstance(evidence, HoldoutEvidence):
            raise TypeError("holdout_evidence_builder must return HoldoutEvidence")
        return trades, evidence

    return qualify_strategy(
        hypothesis_id=_hypothesis_id(strategy),
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


def qualify_bars_automatic(
    *,
    bars: pd.DataFrame,
    strategy: Callable,
    split_index: int,
    preregistration: PreRegistration,
    preregistration_path: str | Path,
    candidate_pool_builder: Callable[[pd.DataFrame, list[Trade]], pd.DataFrame],
    criteria: QualificationCriteria,
    dev_ledger_path: str | Path,
    holdout_ledger_path: str | Path,
    code_version: str,
    data_version: str,
    config: BacktestConfig | None = None,
    session_classifier: Callable | None = None,
    variants_tested: int = 1,
    variant_dev_scores: list[float] | None = None,
    variant_holdout_scores: list[float] | None = None,
    seed: int = 20261013,
) -> QualificationResult:
    """Run the integrated qualification flow using measured research utilities.

    The caller must provide candidate_pool_builder because random-entry testing
    needs every eligible alternative trade under the same exit/cost model. It
    receives development bars only. No holdout statistics are calculated until
    the mandatory development gates pass and the single-use holdout is reserved.
    """
    _validate_inputs(bars, strategy, split_index)
    if not callable(candidate_pool_builder):
        raise TypeError("candidate_pool_builder is required; it must build all eligible alternatives")
    from strategy_qualifier.automatic_evidence import (
        build_development_evidence, build_holdout_evidence,
    )

    dev_bars = bars.iloc[:split_index].copy()
    holdout_bars = bars.iloc[split_index:].copy()
    dev_trades = run_strategy_backtest(dev_bars, strategy, config)
    trials = 1
    # Count all prior distinct dev hypotheses, including ruled-out and exploratory
    # candidates. The current hypothesis is added by the qualification gate.
    from strategy_qualifier.research.ledger import count_trials
    trials += count_trials(str(dev_ledger_path), "dev")
    dev_evidence = build_development_evidence(
        trades=dev_trades, bars=dev_bars,
        candidate_pool_builder=candidate_pool_builder,
        n_trials=trials, alpha=preregistration.alpha,
        session_classifier=session_classifier, seed=seed,
    )

    def build_holdout(trades: list[Trade], isolated_bars: pd.DataFrame) -> HoldoutEvidence:
        return build_holdout_evidence(
            development_trades=dev_trades,
            holdout_trades=trades,
            holdout_bars=isolated_bars,
            n_trials=trials,
            session_classifier=session_classifier,
            variants_tested=variants_tested,
            variant_dev_scores=variant_dev_scores,
            variant_holdout_scores=variant_holdout_scores,
            seed=seed + 1,
        )

    return qualify_bars(
        bars=bars, strategy=strategy, split_index=split_index,
        preregistration=preregistration, preregistration_path=preregistration_path,
        development_evidence=dev_evidence,
        holdout_evidence_builder=build_holdout,
        criteria=criteria, dev_ledger_path=dev_ledger_path,
        holdout_ledger_path=holdout_ledger_path, code_version=code_version,
        data_version=data_version, config=config,
    )
