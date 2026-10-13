import pytest

from strategy_qualifier.protocol import (
    DataUse, Decision, HypothesisRun, record_holdout_once
)


def test_dev_must_be_exploratory():
    run = HypothesisRun("H1", DataUse.DEV, Decision.PASS, 20, "abc", "bars-v1")
    with pytest.raises(ValueError):
        run.validate()


def test_holdout_is_single_use(tmp_path):
    path = tmp_path / "ledger.json"
    record_holdout_once(path, "H1", {"decision": "PASS"})
    with pytest.raises(ValueError):
        record_holdout_once(path, "h1", {"decision": "PASS"})


def test_valid_dev_run():
    HypothesisRun("H1", DataUse.DEV, Decision.EXPLORATORY, 0, "abc", "bars-v1").validate()
