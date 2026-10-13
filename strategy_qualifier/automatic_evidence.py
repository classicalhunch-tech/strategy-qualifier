"""Automatic, strategy-agnostic research evidence for the qualification gate.

This module orchestrates the repository's research utilities. A strategy must
provide a candidate-pool builder for random-entry testing because a valid pool
must use the same entry/exit/cost model as the frozen strategy; the qualifier
will not fabricate candidate trades.

Fail-closed contract: any input that is missing, malformed, non-finite or
internally inconsistent raises EvidenceError (a ValueError subclass). Callers
must treat that as INCONCLUSIVE, never as a pass.
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

RANDOM_ENTRY_REPETITIONS = 5000
BOOTSTRAP_BLOCK_SIZE = 5
BOOTSTRAP_REPETITIONS = 10000


class EvidenceError(ValueError):
    """Evidence is missing, invalid, inconsistent or cannot be computed.

    Subclasses ValueError so existing callers keep working. It must be treated
    as INCONCLUSIVE and never as a passing result.
    """


def _require_positive_int(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise EvidenceError(f"{name} must be a positive integer")


def _require_finite(**values: float) -> None:
    bad = [name for name, value in values.items() if not math.isfinite(float(value))]
    if bad:
        raise EvidenceError(f"evidence could not be computed (non-finite): {', '.join(bad)}")


def _finite_array(values, name: str) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{name} must be numeric: {exc}") from exc
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise EvidenceError(f"{name} must be a one-dimensional array of finite numbers")
    return arr


def _timestamps(bars: pd.DataFrame) -> pd.Series:
    """Return strictly increasing UTC bar timestamps or fail closed."""
    if not isinstance(bars, pd.DataFrame):
        raise EvidenceError("bars must be a pandas DataFrame")
    timestamp_col = next(
        (c for c in bars.columns if str(c).strip().lower() == "timestamp"), None
    )
    if timestamp_col is not None:
        values = bars[timestamp_col]
    elif isinstance(bars.index, pd.DatetimeIndex):
        values = bars.index
    else:
        raise EvidenceError(
            "bars need a 'timestamp' column or a DatetimeIndex; "
            "integer positions are not timestamps"
        )
    try:
        ts = pd.Series(pd.to_datetime(values, errors="raise", utc=True)).reset_index(drop=True)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"invalid bar timestamps: {exc}") from exc
    if ts.isna().any():
        raise EvidenceError("bar timestamps must not contain missing values")
    if len(ts) > 1 and not (ts.diff().iloc[1:] > pd.Timedelta(0)).all():
        raise EvidenceError("bar timestamps must be strictly increasing")
    return ts


def trades_frame(trades: list[Trade], bars: pd.DataFrame) -> pd.DataFrame:
    """Represent backtest trades using entry timestamps and net R."""
    times = _timestamps(bars)
    rows = []
    seen: set[tuple[int, int, int]] = set()
    previous_entry = -1
    for trade in trades:
        try:
            signal_bar = int(trade.signal_bar)
            entry_bar = int(trade.entry_bar)
            exit_bar = int(trade.exit_bar)
            side = int(trade.side)
            net_r = float(trade.r_multiple)
        except (AttributeError, TypeError, ValueError) as exc:
            raise EvidenceError(f"malformed trade record: {exc}") from exc
        if entry_bar < 0 or entry_bar >= len(times):
            raise EvidenceError("trade entry_bar is outside the supplied period")
        if exit_bar < entry_bar or exit_bar >= len(times):
            raise EvidenceError(
                "trade exit_bar must satisfy entry_bar <= exit_bar < number of bars"
            )
        if signal_bar < 0 or signal_bar >= entry_bar:
            raise EvidenceError(
                "trade signal_bar must be non-negative and strictly before entry_bar"
            )
        if side not in (1, -1):
            raise EvidenceError("trade side must be 1 or -1")
        if not math.isfinite(net_r):
            raise EvidenceError("trade r_multiple must be finite")
        if entry_bar < previous_entry:
            raise EvidenceError("trades must be in chronological order by entry_bar")
        key = (signal_bar, entry_bar, side)
        if key in seen:
            raise EvidenceError("duplicate trade record")
        seen.add(key)
        previous_entry = entry_bar
        rows.append({
            "entry_time": times.iloc[entry_bar],
            "direction": "long" if side == 1 else "short",
            "net_r": net_r,
            "is_signal": 1,
            "signal_bar": signal_bar,
            "entry_bar": entry_bar,
            "exit_bar": exit_bar,
        })
    return pd.DataFrame(rows, columns=[
        "entry_time", "direction", "net_r", "is_signal",
        "signal_bar", "entry_bar", "exit_bar",
    ])


def _session_flag(times: pd.Series, classifier: Callable | None = None) -> np.ndarray:
    # OPEN DEFECT C1: the default window below is not taken from the frozen
    # preregistration. It will be removed once the PreRegistration fields are known.
    if classifier is not None:
        flags = np.asarray([bool(classifier(ts)) for ts in times], dtype=float)
    else:
        # Default reference window from the fork: weekdays 13:00-17:00 UTC.
        utc = pd.to_datetime(times, utc=True)
        flags = np.asarray([
            float(ts.dayofweek < 5 and 13 <= ts.hour < 17) for ts in utc
        ], dtype=float)
    if len(flags) != len(times) or not np.isfinite(flags).all():
        raise EvidenceError("session classifier returned invalid flags")
    if len(np.unique(flags)) < 2:
        raise EvidenceError("session-vs-outside test needs trades in both groups")
    return flags


def _session_effect(frame: pd.DataFrame, classifier: Callable | None = None) -> tuple[float, float, float]:
    if len(frame) < 4:
        raise EvidenceError("session-vs-outside test needs at least four trades")
    d = frame.copy().sort_values("entry_time").reset_index(drop=True)
    d["in_window"] = _session_flag(d["entry_time"], classifier)
    y = d["net_r"].to_numpy(float)
    window = d["in_window"].to_numpy(float)
    x = np.column_stack([np.ones(len(d)), window])
    clusters = pd.to_datetime(d["entry_time"], utc=True).dt.strftime("%Y-%m-%d")
    fitted = ols_cluster(y, x, clusters)
    if fitted is None:
        raise EvidenceError("session-vs-outside model is singular")
    beta, se = fitted
    coef, std_err = float(beta[1]), float(se[1])
    if not math.isfinite(std_err) or std_err <= 0:
        raise EvidenceError("session cluster-robust standard error is not estimable")
    # Two-sided normal approximation; the registered rule uses p <= alpha.
    z = coef / std_err
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return coef, std_err, p


def _validate_pool(
    pool: pd.DataFrame,
    signal_count: int,
    *,
    signal_frame: pd.DataFrame | None = None,
    period: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Validate a candidate pool; signal_frame/period enable cross-checks."""
    if not isinstance(pool, pd.DataFrame):
        raise EvidenceError("candidate pool must be a pandas DataFrame")
    required = {"entry_time", "direction", "net_r", "is_signal"}
    missing = required - set(pool.columns)
    if missing:
        raise EvidenceError(f"candidate pool missing required columns: {sorted(missing)}")
    d = pool.copy().reset_index(drop=True)
    try:
        d["entry_time"] = pd.to_datetime(d["entry_time"], errors="raise", utc=True)
        d["net_r"] = pd.to_numeric(d["net_r"], errors="raise")
        d["is_signal"] = pd.to_numeric(d["is_signal"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"candidate pool has malformed values: {exc}") from exc
    if d["entry_time"].isna().any():
        raise EvidenceError("candidate pool entry_time must not contain missing values")
    if not np.isfinite(d["net_r"].to_numpy(float)).all():
        raise EvidenceError("candidate pool net_r values must be finite")
    flags = d["is_signal"].to_numpy(float)
    if not np.isin(flags, (0.0, 1.0)).all():
        raise EvidenceError("candidate pool is_signal must be exactly 0 or 1")
    actual = int((flags == 1.0).sum())
    if actual != signal_count:
        raise EvidenceError(
            f"candidate pool labels {actual} signal trades but backtest produced {signal_count}"
        )
    if d.empty:
        raise EvidenceError("candidate pool is empty")

    d["direction"] = d["direction"].astype(str).str.strip().str.lower()
    if not d["direction"].isin(("long", "short")).all():
        raise EvidenceError("candidate pool direction must be 'long' or 'short'")
    if d.duplicated(["entry_time", "direction"]).any():
        raise EvidenceError("candidate pool contains duplicate entry_time/direction observations")
    if period is not None:
        start, end = period
        if (d["entry_time"] < start).any() or (d["entry_time"] > end).any():
            raise EvidenceError("candidate pool contains entries outside the development period")

    is_signal = flags == 1.0
    if signal_frame is not None:
        keys = ["entry_time", "direction", "net_r"]
        ref = signal_frame[keys].copy()
        ref["entry_time"] = pd.to_datetime(ref["entry_time"], utc=True)
        ref = ref.sort_values(keys).reset_index(drop=True)
        got = d.loc[is_signal, keys].sort_values(keys).reset_index(drop=True)
        same_times = bool((got["entry_time"] - ref["entry_time"]).eq(pd.Timedelta(0)).all())
        same_dirs = bool(got["direction"].eq(ref["direction"]).all())
        if not (same_times and same_dirs):
            raise EvidenceError(
                "candidate pool signal rows do not match the strategy's actual trades"
            )
        if not np.allclose(
            got["net_r"].to_numpy(float), ref["net_r"].to_numpy(float), rtol=0.0, atol=1e-9
        ):
            raise EvidenceError(
                "candidate pool signal net_r differs from the backtest's net R (costs must match)"
            )

    if is_signal.all():
        raise EvidenceError("candidate pool has no non-signal alternatives")
    for side in ("long", "short"):
        in_side = (d["direction"] == side).to_numpy()
        n_signal = int((is_signal & in_side).sum())
        n_other = int(((~is_signal) & in_side).sum())
        if n_other < n_signal:
            raise EvidenceError(
                f"candidate pool has fewer non-signal {side} candidates ({n_other}) "
                f"than strategy signals ({n_signal})"
            )
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
    exits and costs, with columns entry_time, direction, net_r, is_signal. Its
    signal rows must reproduce the backtest's trades exactly.
    """
    if not callable(candidate_pool_builder):
        raise TypeError("candidate_pool_builder is required for random-entry testing")
    _require_positive_int(n_trials, "n_trials")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float, np.floating))
        or not 0.0 < float(alpha) < 1.0
    ):
        raise EvidenceError("alpha must be strictly between 0 and 1")
    times = _timestamps(bars)
    frame = trades_frame(trades, bars)
    if len(frame) < 10:
        raise EvidenceError("too few development trades for automatic research evidence")
    pool = candidate_pool_builder(bars.copy(), list(trades))
    if not isinstance(pool, pd.DataFrame):
        raise TypeError("candidate_pool_builder must return a pandas DataFrame")
    pool = _validate_pool(
        pool, len(trades), signal_frame=frame, period=(times.iloc[0], times.iloc[-1])
    )

    random = random_baseline(
        pool, signal_col="is_signal", returns_col="net_r", time_col="entry_time",
        strata_cols=["direction"], n_reps=RANDOM_ENTRY_REPETITIONS, seed=seed,
    )
    random_p = float(random["p_value"])
    observed_mean = float(random["observed_mean"])
    random_hi90 = float(random["baseline_hi90"])
    _require_finite(
        random_entry_p_value=random_p, observed_mean_r=observed_mean,
        random_entry_90th_percentile_r=random_hi90,
    )

    session_coef, session_se, session_p = _session_effect(frame, session_classifier)
    _require_finite(session_coefficient=session_coef, session_se=session_se,
                    session_p_value=session_p)
    frame["in_window"] = _session_flag(frame["entry_time"], session_classifier)
    folds = walk_forward_run(
        frame, returns_col="net_r", window_col="in_window",
        time_col="entry_time", n_folds=5, label_horizon=5, embargo_days=5,
    )
    coefficients = tuple(float(v) for v in folds["coef"].dropna().tolist())
    if not coefficients:
        raise EvidenceError("walk-forward produced no usable fold coefficients")
    if not all(math.isfinite(c) for c in coefficients):
        raise EvidenceError("walk-forward produced non-finite fold coefficients")

    lower, _ = block_bootstrap_ci(
        frame["net_r"].to_numpy(float), block_size=BOOTSTRAP_BLOCK_SIZE,
        n_boot=BOOTSTRAP_REPETITIONS, alpha=0.10, seed=seed,
    )
    dsr = float(deflated_sharpe(frame["net_r"].to_numpy(float), n_trials=n_trials)["dsr"])
    _require_finite(bootstrap_90ci_lower=float(lower), dsr=dsr)
    return DevelopmentEvidence(
        walk_forward_coefficients=coefficients,
        session_coefficient=session_coef,
        session_cluster_robust_se=session_se,
        session_p_value=session_p,
        random_entry_p_value=random_p,
        observed_mean_r=observed_mean,
        random_entry_90th_percentile_r=random_hi90,
        dsr=dsr,
        bootstrap_90ci_lower=float(lower),
        signal_trade_count=len(trades),
        random_entry_repetitions=RANDOM_ENTRY_REPETITIONS,
        bootstrap_block_size=BOOTSTRAP_BLOCK_SIZE,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        dsr_trials=int(n_trials),
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
    """Compute holdout replication metrics after the single-use gate is reserved.

    holdout_bars is used to validate every holdout trade (indices, timestamps,
    ordering). session_classifier is accepted for API compatibility and unused.
    """
    _require_positive_int(n_trials, "n_trials")
    _require_positive_int(variants_tested, "variants_tested")
    hold_frame = trades_frame(holdout_trades, holdout_bars)
    dev_r = np.asarray([float(t.r_multiple) for t in development_trades], dtype=float)
    hold_r = hold_frame["net_r"].to_numpy(float)
    if len(dev_r) < 2 or len(hold_r) < 2:
        raise EvidenceError("development and holdout each need at least two trades")
    if not np.isfinite(dev_r).all():
        raise EvidenceError("development trade R values must be finite")

    dev_scores = hold_scores = None
    if variants_tested > 1:
        if variant_dev_scores is None or variant_holdout_scores is None:
            raise EvidenceError("variant scores are required for holdout rank-stability testing")
        dev_scores = _finite_array(variant_dev_scores, "variant_dev_scores")
        hold_scores = _finite_array(variant_holdout_scores, "variant_holdout_scores")
        if len(dev_scores) != variants_tested or len(hold_scores) != variants_tested:
            raise EvidenceError("variant score lengths must match variants_tested")

    dev_mean, hold_mean = float(dev_r.mean()), float(hold_r.mean())
    dev_se = float(dev_r.std(ddof=1) / math.sqrt(len(dev_r)))
    hold_se = float(hold_r.std(ddof=1) / math.sqrt(len(hold_r)))
    if dev_se <= 0 or hold_se <= 0 or not math.isfinite(dev_se + hold_se):
        raise EvidenceError("BIF requires non-zero, finite standard errors")
    bif_result = bif(dev_mean / dev_se, hold_mean / hold_se)
    lower, _ = block_bootstrap_ci(
        hold_r, block_size=BOOTSTRAP_BLOCK_SIZE, n_boot=BOOTSTRAP_REPETITIONS,
        alpha=0.10, seed=seed,
    )
    dsr = float(deflated_sharpe(hold_r, n_trials=n_trials)["dsr"])
    _require_finite(bif=float(bif_result["bif"]), bootstrap_90ci_lower=float(lower), dsr=dsr)

    rho = percentile = None
    if variants_tested > 1:
        from strategy_qualifier.research.rank_stability import rank_stability
        rank = rank_stability(dev_scores.tolist(), hold_scores.tolist())
        rho = float(rank["spearman"])
        percentile = float(rank["winner_oos_percentile"])
        _require_finite(spearman_rho=rho, winner_oos_percentile=percentile)

    return HoldoutEvidence(
        bif=float(bif_result["bif"]),
        dev_mean_r=dev_mean,
        holdout_mean_r=hold_mean,
        bootstrap_90ci_lower=float(lower),
        dsr=dsr,
        variants_tested=variants_tested,
        spearman_rho=rho,
        winner_oos_percentile=percentile,
        bootstrap_block_size=BOOTSTRAP_BLOCK_SIZE,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        dsr_trials=int(n_trials),
    )