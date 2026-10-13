"""Fail-closed integrity tests for strategy_qualifier.automatic_evidence."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from strategy_qualifier import automatic_evidence as ae
from strategy_qualifier.automatic_evidence import EvidenceError
from strategy_qualifier.qualification_gates import HoldoutEvidence


def make_bars(n: int = 200) -> pd.DataFrame:
    ts = pd.date_range("2025-01-06 00:00", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"timestamp": ts, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0}
    )


def make_trade(signal_bar, entry_bar, exit_bar, side=1, r=0.5):
    return SimpleNamespace(
        signal_bar=signal_bar, entry_bar=entry_bar, exit_bar=exit_bar,
        side=side, r_multiple=r,
    )


def make_trades(count: int = 12):
    return [
        make_trade(10 + 10 * i, 11 + 10 * i, 14 + 10 * i,
                   side=1 if i % 2 == 0 else -1, r=0.8 if i % 3 else -0.4)
        for i in range(count)
    ]


def make_holdout_trades(count: int = 30):
    return [
        make_trade(3 + 6 * i, 4 + 6 * i, 6 + 6 * i, side=1, r=0.8 if i % 3 else -0.4)
        for i in range(count)
    ]


def valid_pool(bars, trades) -> pd.DataFrame:
    frame = ae.trades_frame(trades, bars)
    signals = frame[["entry_time", "direction", "net_r"]].assign(is_signal=1)
    times = ae._timestamps(bars)
    used = set(frame["entry_time"])
    rows = [
        {"entry_time": t, "direction": d, "net_r": 0.1, "is_signal": 0}
        for t in times.iloc[::3] if t not in used
        for d in ("long", "short")
    ]
    return pd.concat([signals, pd.DataFrame(rows)], ignore_index=True)


@pytest.fixture()
def ctx():
    bars = make_bars()
    trades = make_trades()
    frame = ae.trades_frame(trades, bars)
    times = ae._timestamps(bars)
    return bars, trades, frame, (times.iloc[0], times.iloc[-1])


# ---- timestamps -----------------------------------------------------------

def test_timestamps_reject_integer_positions():
    bars = pd.DataFrame({"open": [1.0] * 3, "high": [1.0] * 3, "low": [1.0] * 3, "close": [1.0] * 3})
    with pytest.raises(EvidenceError, match="timestamp"):
        ae._timestamps(bars)


def test_timestamps_reject_duplicates():
    bars = make_bars(5)
    bars.loc[3, "timestamp"] = bars.loc[2, "timestamp"]
    with pytest.raises(EvidenceError, match="strictly increasing"):
        ae._timestamps(bars)


def test_timestamps_reject_missing_values():
    bars = make_bars(5)
    bars.loc[1, "timestamp"] = pd.NaT
    with pytest.raises(EvidenceError, match="missing"):
        ae._timestamps(bars)


def test_timestamps_accept_datetime_index():
    assert len(ae._timestamps(make_bars(5).set_index("timestamp"))) == 5


# ---- trades_frame ----------------------------------------------------------

@pytest.mark.parametrize("kwargs, message", [
    (dict(signal_bar=10, entry_bar=500, exit_bar=501), "outside the supplied period"),
    (dict(signal_bar=10, entry_bar=-3, exit_bar=5), "outside the supplied period"),
    (dict(signal_bar=10, entry_bar=20, exit_bar=19), "exit_bar"),
    (dict(signal_bar=20, entry_bar=20, exit_bar=25), "strictly before"),
    (dict(signal_bar=10, entry_bar=11, exit_bar=12, side=0), "side"),
    (dict(signal_bar=10, entry_bar=11, exit_bar=12, r=float("nan")), "finite"),
    (dict(signal_bar=10, entry_bar=11, exit_bar=12, r=float("inf")), "finite"),
])
def test_trades_frame_rejects_invalid_trade(kwargs, message):
    with pytest.raises(EvidenceError, match=message):
        ae.trades_frame([make_trade(**kwargs)], make_bars())


def test_trades_frame_rejects_out_of_order_and_duplicate_trades():
    a, b = make_trade(10, 11, 14), make_trade(30, 31, 34)
    with pytest.raises(EvidenceError, match="chronological"):
        ae.trades_frame([b, a], make_bars())
    with pytest.raises(EvidenceError, match="duplicate"):
        ae.trades_frame([a, a], make_bars())


# ---- candidate pool --------------------------------------------------------

def _validate(pool, ctx):
    _, trades, frame, period = ctx
    return ae._validate_pool(pool, len(trades), signal_frame=frame, period=period)


def test_valid_pool_is_accepted(ctx):
    bars, trades, _, _ = ctx
    pool = valid_pool(bars, trades)
    out = _validate(pool, ctx)
    assert len(out) == len(pool)
    assert set(out["direction"]) == {"long", "short"}


def _with(pool, changes):
    p = pool.copy()
    for (idx, col), value in changes.items():
        p.loc[idx, col] = value
    return p


CASES = [
    ("missing_column", lambda p: p.drop(columns=["direction"]), "missing required columns"),
    ("nan_return", lambda p: _with(p, {(p.index[-1], "net_r"): np.nan}), "must be finite"),
    ("inf_return", lambda p: _with(p, {(p.index[-1], "net_r"): np.inf}), "must be finite"),
    ("flag_two", lambda p: _with(p, {(p.index[-1], "is_signal"): 2}), "is_signal"),
    ("flag_nan", lambda p: _with(p, {(p.index[-1], "is_signal"): np.nan}), "is_signal"),
    ("nat_time", lambda p: _with(p, {(p.index[-1], "entry_time"): pd.NaT}), "entry_time"),
    ("bad_direction", lambda p: _with(p, {(p.index[-1], "direction"): "sideways"}), "direction"),
    ("duplicate_row", lambda p: pd.concat([p, p.iloc[[-1]]], ignore_index=True), "duplicate"),
    ("outside_period",
     lambda p: _with(p, {(p.index[-1], "entry_time"): pd.Timestamp("2030-01-01", tz="UTC")}),
     "outside the development period"),
    ("signal_return_changed", lambda p: _with(p, {(0, "net_r"): p.loc[0, "net_r"] + 0.5}),
     "differs from the backtest"),
    ("signal_time_shifted",
     lambda p: _with(p, {(0, "entry_time"): p.loc[0, "entry_time"] + pd.Timedelta(minutes=1)}),
     "do not match"),
    ("signal_unlabelled", lambda p: _with(p, {(0, "is_signal"): 0}),
     "signal trades but backtest produced"),
    ("no_alternatives", lambda p: p[p["is_signal"] == 1], "no non-signal alternatives"),
    ("few_short_alternatives",
     lambda p: p.drop(p[(p.is_signal == 0) & (p.direction == "short")].index[1:]),
     "fewer non-signal"),
]


@pytest.mark.parametrize("name, mutate, message", CASES, ids=[c[0] for c in CASES])
def test_invalid_pool_is_rejected(ctx, name, mutate, message):
    bars, trades, _, _ = ctx
    with pytest.raises(EvidenceError, match=message):
        _validate(mutate(valid_pool(bars, trades)), ctx)


# ---- development builder wiring -------------------------------------------

def test_development_evidence_rejects_pool_with_wrong_signal_returns():
    bars, trades = make_bars(), make_trades()

    def builder(b, t):
        pool = valid_pool(b, t)
        pool.loc[pool.is_signal == 1, "net_r"] += 0.25
        return pool

    with pytest.raises(EvidenceError, match="differs from the backtest"):
        ae.build_development_evidence(
            trades=trades, bars=bars, candidate_pool_builder=builder,
            n_trials=1, alpha=0.05,
        )


@pytest.mark.parametrize("kwargs, message", [
    (dict(n_trials=0, alpha=0.05), "n_trials"),
    (dict(n_trials=1, alpha=1.5), "alpha"),
    (dict(n_trials=1, alpha=0.0), "alpha"),
])
def test_development_evidence_rejects_bad_parameters(kwargs, message):
    with pytest.raises(EvidenceError, match=message):
        ae.build_development_evidence(
            trades=make_trades(), bars=make_bars(),
            candidate_pool_builder=lambda b, t: valid_pool(b, t), **kwargs,
        )


def test_development_evidence_rejects_non_dataframe_pool():
    with pytest.raises(TypeError):
        ae.build_development_evidence(
            trades=make_trades(), bars=make_bars(),
            candidate_pool_builder=lambda b, t: [1, 2, 3], n_trials=1, alpha=0.05,
        )


# ---- holdout builder -------------------------------------------------------

@pytest.mark.parametrize("bad_trade, message", [
    (make_trade(40, 41, 80), "exit_bar"),
    (make_trade(10, -3, 5), "outside the supplied period"),
])
def test_holdout_evidence_validates_trades_against_holdout_bars(bad_trade, message):
    with pytest.raises(EvidenceError, match=message):
        ae.build_holdout_evidence(
            development_trades=make_trades(12),
            holdout_trades=[bad_trade] + make_holdout_trades(5),
            holdout_bars=make_bars(50), n_trials=1,
        )


def _holdout(**overrides):
    kwargs = dict(
        development_trades=make_trades(12),
        holdout_trades=make_holdout_trades(30),
        holdout_bars=make_bars(200), n_trials=4,
    )
    kwargs.update(overrides)
    return ae.build_holdout_evidence(**kwargs)


def test_holdout_evidence_rejects_bad_variant_inputs():
    with pytest.raises(EvidenceError, match="variants_tested"):
        _holdout(variants_tested=0)
    with pytest.raises(EvidenceError, match="variant scores are required"):
        _holdout(variants_tested=3)
    with pytest.raises(EvidenceError, match="must match variants_tested"):
        _holdout(variants_tested=3, variant_dev_scores=[1, 2], variant_holdout_scores=[1, 2])
    with pytest.raises(EvidenceError, match="finite"):
        _holdout(variants_tested=3, variant_dev_scores=[1, 2, float("nan")],
                 variant_holdout_scores=[1, 2, 3])


def test_holdout_evidence_is_computed_from_holdout_trades():
    evidence = _holdout()
    assert isinstance(evidence, HoldoutEvidence)
    assert evidence.dsr_trials == 4 and evidence.variants_tested == 1
    assert evidence.spearman_rho is None
    assert math.isfinite(evidence.bif)
    expected = np.mean([t.r_multiple for t in make_holdout_trades(30)])
    assert evidence.holdout_mean_r == pytest.approx(expected)
