"""Automatic, strategy-agnostic research evidence for the qualification gate.

This module orchestrates the repository's research utilities. A strategy must
provide a candidate-pool builder for random-entry testing because a valid pool
must use the same entry/exit/cost model as the frozen strategy; the qualifier
will not fabricate candidate trades.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd

from strategy_qualifier.backtest import Trade
from strategy_qualifier.qualification_gates import (
    DevelopmentEvidence,
    HoldoutEvidence,
    PreRegistration,
)
from strategy_qualifier.research.bif import bif
from strategy_qualifier.research.block_bootstrap import block_bootstrap_ci
from strategy_qualifier.research.deflated_sharpe import deflated_sharpe
from strategy_qualifier.research.random_entry_baseline import baseline as random_baseline
from strategy_qualifier.research.session_vs_outside import ols_cluster
from strategy_qualifier.research.walk_forward import run as walk_forward_run


def _timestamps(bars: pd.DataFrame) -> pd.Series:
    timestamp_col = next(
        (c for c in bars.columns if str(c).strip().lower() == "timestamp"), None
    )
    values = bars[timestamp_col] if timestamp_col is not None else bars.index
    ts = pd.to_datetime(values, errors="raise", utc=True)
    return pd.Series(ts).reset_index(drop=True)


def trades_frame(trades: list[Trade], bars: pd.DataFrame) -> pd.DataFrame:
    """Represent backtest trades using entry timestamps and net R."""
    times = _timestamps(bars)
    rows = []
    for trade in trades:
        if trade.entry_bar < 0 or trade.entry_bar >= len(times):
            raise ValueError("trade entry_bar is outside the supplied period")
        rows.append({
            "entry_time": times.iloc[trade.entry_bar],
            "direction": "long" if trade.side == 1 else "short",
            "net_r": float(trade.r_multiple),
            "is_signal": 1,
            "signal_bar": int(trade.signal_bar),
            "entry_bar": int(trade.entry_bar),
            "exit_bar": int(trade.exit_bar),
        })
    return pd.DataFrame(rows, columns=[
        "entry_time", "direction", "net_r", "is_signal",
        "signal_bar", "entry_bar", "exit_bar",
    ])


def _session_flag(times: pd.Series, classifier: Callable | None = None) -> np.ndarray:
    if classifier is not None:
        flags = np.asarray([bool(classifier(ts)) for ts in times], dtype=float)
    else:
        # Default frozen reference window from the fork: weekdays 13:00–17:00 UTC.
        utc = pd.to_datetime(times, utc=True)
        flags = np.asarray([
            float(ts.dayofweek < 5 and 13 <= ts.hour < 17) for ts in utc
        ], dtype=float)
    if len(flags) != len(times) or not np.isfinite(flags).all():
        raise ValueError("session classifier returned invalid flags")
    if len(np.unique(flags)) < 2:
        raise ValueError("session-vs-outside test needs trades in both groups")
    return flags


def _session_effect(frame: pd.DataFrame, classifier: Callable | None = None) -> tuple[float, float, float]:
    if len(frame) < 4:
        raise ValueError("session-vs-outside test needs at least four trades")
    d = frame.copy().sort_values("entry_time").reset_index(drop=True)
    d["in_window"] = _session_flag(d["entry_time"], classifier)
    y = d["net_r"].to_numpy(float)
    window = d["in_window"].to_numpy(float)
    x = np.column_stack([np.ones(len(d)), window])
    clusters = pd.to_datetime(d["entry_time"], utc=True).dt.strftime("%Y-%m-%d")
    fitted = ols_cluster(y, x, clusters)
    if fitted is None:
        raise ValueError("session-vs-outside model is singular")
    beta, se = fitted
    coef, std_err = float(beta[1]), float(se[1])
    if not math.isfinite(std_err) or std_err <= 0:
        raise ValueError("session cluster-robust standard error is not estimable")
    # Two-sided normal approximation; the registered rule uses p <= alpha.
    z = coef / std_err
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return coef, std_err, p


def _validate_pool(pool: pd.DataFrame, signal_count: int) -> pd.DataFrame:
    required = {"entry_time", "direction", "net_r", "is_signal"}
    missing = required - set(pool.columns)
    if missing:
        raise ValueError(f"candidate pool missing required columns: {sorted(missing)}")
    d = pool.copy()
    d["entry_time"] = pd.to_datetime(d["entry_time"], errors="raise", utc=True)
    d["net_r"] = pd.to_numeric(d["net_r"], errors="raise")
    d["is_signal"] = pd.to_numeric(d["is_signal"], errors="raise")
    if not np.isfinite(d["net_r"].to_numpy(float)).all():
        raise ValueError("candidate pool net_r values must be finite")
    actual = int((d["is_signal"] > 0).sum())
    if actual != signal_count:
        raise ValueError(
            f"candidate pool labels {actual} signal trades but backtest produced {signal_count}"
        )
    if d.empty:
        raise ValueError("candidate pool is empty")
    return d


def build_development_evidence(
    *,
    trades: list[Trade],
    bars: pd.DataFrame,
    candidate_pool_builder: Callable[[pd.DataFrame, list[Trade]], pd.DataFrame],
    n_trials: int,
    alpha: float,
    session_classifier: Callable | None = None,
    seed: int = 20261013,
) -> DevelopmentEvidence:
    """Compute every development statistic from frozen backtest outputs.

    candidate_pool_builder receives only development bars and development
    signal trades. It must return ALL eligible candidate entries under the same
    exits and costs, with columns entry_time, direction, net_r, is_signal.
    """
    if not callable(candidate_pool_builder):
        raise TypeError("candidate_pool_builder is required for random-entry testing")
    frame = trades_frame(trades, bars)
    if len(frame) < 10:
        raise ValueError("too few development trades for automatic research evidence")
    pool = candidate_pool_builder(bars.copy(), list(trades))
    if not isinstance(pool, pd.DataFrame):
        raise TypeError("candidate_pool_builder must return a pandas DataFrame")
    pool = _validate_pool(pool, len(trades))

    random = random_baseline(
        pool, signal_col="is_signal", returns_col="net_r", time_col="entry_time",
        strata_cols=["direction"], n_reps=5000, seed=seed,
    )
    session_coef, session_se, session_p = _session_effect(frame, session_classifier)
    frame["in_window"] = _session_flag(frame["entry_time"], session_classifier)
    folds = walk_forward_run(
        frame, returns_col="net_r", window_col="in_window",
        time_col="entry_time", n_folds=5, label_horizon=5, embargo_days=5,
    )
    coefficients = tuple(float(v) for v in folds["coef"].dropna().tolist())
    lower, _ = block_bootstrap_ci(
        frame["net_r"].to_numpy(float), block_size=5, n_boot=10000,
        alpha=0.10, seed=seed,
    )
    dsr = float(deflated_sharpe(frame["net_r"].to_numpy(float), n_trials=n_trials)["dsr"])
    return DevelopmentEvidence(
        walk_forward_coefficients=coefficients,
        session_coefficient=session_coef,
        session_cluster_robust_se=session_se,
        session_p_value=session_p,
        random_entry_p_value=float(random["p_value"]),
        observed_mean_r=float(random["observed_mean"]),
        random_entry_90th_percentile_r=float(random["baseline_hi90"]),
        dsr=dsr,
        bootstrap_90ci_lower=float(lower),
        signal_trade_count=len(trades),
        random_entry_repetitions=5000,
        bootstrap_block_size=5,
        bootstrap_repetitions=10000,
        dsr_trials=n_trials,
    )


def build_holdout_evidence(
    *,
    development_trades: list[Trade],
    holdout_trades: list[Trade],
    holdout_bars: pd.DataFrame,
    n_trials: int,
    session_classifier: Callable | None = None,
    variants_tested: int = 1,
    variant_dev_scores: list[float] | None = None,
    variant_holdout_scores: list[float] | None = None,
    seed: int = 20261014,
) -> HoldoutEvidence:
    """Compute holdout replication metrics after the single-use gate is reserved."""
    dev_r = np.asarray([float(t.r_multiple) for t in development_trades], dtype=float)
    hold_r = np.asarray([float(t.r_multiple) for t in holdout_trades], dtype=float)
    if len(dev_r) < 2 or len(hold_r) < 2:
        raise ValueError("development and holdout each need at least two trades")
    dev_mean, hold_mean = float(dev_r.mean()), float(hold_r.mean())
    dev_se = float(dev_r.std(ddof=1) / math.sqrt(len(dev_r)))
    hold_se = float(hold_r.std(ddof=1) / math.sqrt(len(hold_r)))
    if dev_se <= 0 or hold_se <= 0 or not math.isfinite(dev_se + hold_se):
        raise ValueError("BIF requires non-zero, finite standard errors")
    bif_result = bif(dev_mean / dev_se, hold_mean / hold_se)
    lower, _ = block_bootstrap_ci(
        hold_r, block_size=5, n_boot=10000, alpha=0.10, seed=seed,
    )
    dsr = float(deflated_sharpe(hold_r, n_trials=n_trials)["dsr"])

    rho = percentile = None
    if variants_tested > 1:
        if variant_dev_scores is None or variant_holdout_scores is None:
            raise ValueError("variant scores are required for holdout rank-stability testing")
        if len(variant_dev_scores) != variants_tested or len(variant_holdout_scores) != variants_tested:
            raise ValueError("variant score lengths must match variants_tested")
        from strategy_qualifier.research.rank_stability import rank_stability
        rank = rank_stability(variant_dev_scores, variant_holdout_scores)
        rho = float(rank["spearman"])
        percentile = float(rank["winner_oos_percentile"])

    return HoldoutEvidence(
        bif=float(bif_result["bif"]),
        dev_mean_r=dev_mean,
        holdout_mean_r=hold_mean,
        bootstrap_90ci_lower=float(lower),
        dsr=dsr,
        variants_tested=variants_tested,
        spearman_rho=rho,
        winner_oos_percentile=percentile,
        bootstrap_block_size=5,
        bootstrap_repetitions=10000,
        dsr_trials=n_trials,
    )
