import numpy as np
import pytest

from strategy_qualifier.statistics.bootstrap import block_bootstrap_ci
from strategy_qualifier.statistics.sharpe import deflated_sharpe


def test_bootstrap_is_reproducible():
    x = np.arange(30, dtype=float)
    assert block_bootstrap_ci(x, seed=4) == block_bootstrap_ci(x, seed=4)


def test_bootstrap_empty_returns_nan_pair():
    lo, hi = block_bootstrap_ci([])
    assert np.isnan(lo) and np.isnan(hi)


def test_dsr_rejects_invalid_trial_count():
    with pytest.raises(ValueError):
        deflated_sharpe([1, 2, 3], n_trials=0)


def test_dsr_handles_too_few_trades():
    assert np.isnan(deflated_sharpe([1.0, 2.0])["dsr"])


def test_dsr_returns_probability_for_variable_series():
    result = deflated_sharpe(np.random.default_rng(2).normal(0.1, 1, 100), n_trials=10)
    assert 0 <= result["dsr"] <= 1
