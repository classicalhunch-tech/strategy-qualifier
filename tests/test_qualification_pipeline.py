import json

import pandas as pd
import pytest

from strategy_qualifier.backtest import BacktestConfig, Signal, Trade
from strategy_qualifier.qualification import QualificationCriteria, qualify_strategy
from strategy_qualifier.qualification_data import qualify_bars
from strategy_qualifier.qualification_gates import (
    DevelopmentEvidence, HoldoutEvidence, PreRegistration, register_preregistration,
    mark_preregistration_status, verify_preregistration,
)
from strategy_qualifier.research.deflated_sharpe import deflated_sharpe


def prereg(hypothesis_id="H-001"):
    return PreRegistration(
        hypothesis_id=hypothesis_id,
        question="Does the frozen test signal produce positive net R?",
        hypothesis="The signal has positive expectancy after costs.",
        expected_sign="positive", market="TEST", timeframe="1H",
        entry_rules="Signal at current close; enter next open.",
        exit_rules="Fixed stop and target.", stop_rule="Five units.",
        target_rule="Ten units.", session_filter="All sessions.",
        direction_filter="Long only.", dev_start="2020-01-01",
        dev_end="2021-01-01", holdout_start="2021-01-01",
        holdout_end="2022-01-01", min_signal_trades=30, min_walk_forward_mean_coefficient=0.01, alpha=0.05,
        min_trades=20, min_expectancy=0.0, max_drawdown=1000.0,
        falsification_criteria="Any mandatory gate fails.",
        registered_before_data=True,
    )


def expected_dsr(n=40, n_trials=1):
    returns = [1.6 + (i % 2) * 0.2 for i in range(n)]
    return float(deflated_sharpe(returns, n_trials=n_trials)["dsr"])


def evidence(n=40):
    return DevelopmentEvidence(
        walk_forward_coefficients=(0.2, 0.3, 0.25, 0.15, 0.22),
        session_coefficient=0.2, session_cluster_robust_se=0.05,
        session_p_value=0.01, random_entry_p_value=0.01,
        observed_mean_r=1.7, random_entry_90th_percentile_r=0.5,
        dsr=expected_dsr(n), bootstrap_90ci_lower=0.8, signal_trade_count=n,
        random_entry_repetitions=5000, bootstrap_block_size=5,
        bootstrap_repetitions=10000, dsr_trials=1,
    )


def holdout_evidence():
    return HoldoutEvidence(
        bif=1.1, dev_mean_r=1.7, holdout_mean_r=1.7,
        bootstrap_90ci_lower=0.4, dsr=expected_dsr(),
    )


def trade(net_pnl, r_multiple, hypothesis_id="H-001"):
    return Trade(
        hypothesis_id=hypothesis_id, side=1, signal_bar=0, entry_bar=1, exit_bar=2,
        entry_price=100.0, exit_price=100.0 + net_pnl, stop_loss=95.0,
        take_profit=110.0, exit_reason="TARGET" if net_pnl > 0 else "STOP",
        bars_held=2, gross_pnl=net_pnl, costs=0.0, net_pnl=net_pnl,
        initial_risk=5.0, r_multiple=r_multiple,
    )


def trades(n=40, hypothesis_id="H-001"):
    return [trade(8.0 + (i % 2), 1.6 + (i % 2) * 0.2, hypothesis_id) for i in range(n)]


def criteria():
    return QualificationCriteria(min_trades=20, min_expectancy=0.0, max_drawdown=1000.0)


def run_args(tmp_path, preregistration=None, dev_trades=None, dev_evidence=None, runner=None):
    p = preregistration or prereg()
    prereg_path = tmp_path / "prereg.json"
    if not prereg_path.exists():
        register_preregistration(p, prereg_path)
    return dict(
        hypothesis_id=p.hypothesis_id, preregistration=p,
        preregistration_path=prereg_path,
        development_trades=dev_trades if dev_trades is not None else trades(40, p.hypothesis_id),
        development_evidence=dev_evidence if dev_evidence is not None else evidence(),
        holdout_runner=runner or (lambda: (trades(40, p.hypothesis_id), holdout_evidence())),
        criteria=criteria(), dev_ledger_path=tmp_path / "dev.csv",
        holdout_ledger_path=tmp_path / "holdout.json",
        code_version="test-commit", dev_data_version="dev-fixture-v1",
        holdout_data_version="holdout-fixture-v1",
    )


def test_all_mandatory_gates_pass_before_single_use_holdout(tmp_path):
    calls = []
    args = run_args(
        tmp_path,
        runner=lambda: calls.append(True) or (trades(40), holdout_evidence()),
    )
    result = qualify_strategy(**args)
    assert result.stage == "holdout"
    assert result.decision == "PASS"
    assert not result.reasons
    assert calls == [True]
    ledger = json.loads((tmp_path / "holdout.json").read_text())
    assert ledger[0]["status"] == "COMPLETED"
    assert ledger[0]["decision"] == "PASS"
    assert ledger[0]["research_evidence"]["bif"] == 1.1

    with pytest.raises(ValueError, match="terminal|already used|cannot be reset"):
        qualify_strategy(**args)
    assert calls == [True]




def test_development_dsr_evidence_must_match_recomputed_trades(tmp_path):
    bad = evidence()
    bad = DevelopmentEvidence(**{**bad.__dict__, "dsr": 0.95})
    result = qualify_strategy(**run_args(tmp_path, dev_evidence=bad))
    assert result.stage == "development"
    assert result.decision == "INCONCLUSIVE"
    assert any("does not match the recomputed DSR" in reason for reason in result.reasons)
    assert not (tmp_path / "holdout.json").exists()


def test_holdout_dsr_evidence_must_match_recomputed_trades(tmp_path):
    good = holdout_evidence()
    bad = HoldoutEvidence(**{**good.__dict__, "dsr": 0.95})
    result = qualify_strategy(**run_args(
        tmp_path, runner=lambda: (trades(40), bad),
    ))
    assert result.stage == "holdout"
    assert result.decision == "INCONCLUSIVE"
    assert any("does not match the recomputed DSR" in reason for reason in result.reasons)


def test_dsr_evidence_outside_probability_range_is_rejected(tmp_path):
    bad = DevelopmentEvidence(**{**evidence().__dict__, "dsr": 1.2})
    result = qualify_strategy(**run_args(tmp_path, dev_evidence=bad))
    assert result.decision == "INCONCLUSIVE"
    assert any("within [0, 1]" in reason for reason in result.reasons)

def test_any_development_gate_failure_prevents_holdout(tmp_path):
    calls = []
    bad = DevelopmentEvidence(
        walk_forward_coefficients=(0.2, -0.3, -0.1, 0.15, 0.22),
        session_coefficient=0.2, session_cluster_robust_se=0.05,
        session_p_value=0.01, random_entry_p_value=0.01,
        observed_mean_r=1.7, random_entry_90th_percentile_r=0.5,
        dsr=expected_dsr(), bootstrap_90ci_lower=-0.1, signal_trade_count=40,
        random_entry_repetitions=5000, bootstrap_block_size=5,
        bootstrap_repetitions=10000, dsr_trials=1,
    )
    result = qualify_strategy(**run_args(
        tmp_path, dev_evidence=bad,
        runner=lambda: calls.append(True) or (trades(40), holdout_evidence()),
    ))
    assert result.stage == "development"
    assert result.decision == "RULED OUT"
    assert any("walk-forward" in reason for reason in result.reasons)
    assert any("bootstrap" in reason for reason in result.reasons)
    assert calls == []
    assert not (tmp_path / "holdout.json").exists()


def test_n_star_shortfall_is_inconclusive_and_never_peeks(tmp_path):
    calls = []
    result = qualify_strategy(**run_args(
        tmp_path, dev_trades=trades(10),
        dev_evidence=evidence(10),
        runner=lambda: calls.append(True) or (trades(40), holdout_evidence()),
    ))
    assert result.decision == "INCONCLUSIVE"
    assert any("N*" in reason for reason in result.reasons)
    assert calls == []


def test_missing_preregistration_fails_closed(tmp_path):
    p = prereg()
    args = run_args(tmp_path, preregistration=p)
    args["preregistration_path"] = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="pre-registration ledger missing"):
        qualify_strategy(**args)




def test_overlapping_previously_exposed_holdout_window_is_rejected(tmp_path):
    path = tmp_path / "prereg.json"
    first = prereg("H-001")
    register_preregistration(first, path)
    mark_preregistration_status(str(path), "H-001", "PASS", {"decision": "PASS"})

    second = prereg("H-002")
    register_preregistration(second, path)
    with pytest.raises(ValueError, match="overlaps previously exposed window"):
        verify_preregistration(second, path)

def test_preregistration_is_frozen_and_cannot_be_overwritten(tmp_path):
    p = prereg()
    path = tmp_path / "prereg.json"
    register_preregistration(p, path)
    with pytest.raises(ValueError, match="already exists"):
        register_preregistration(p, path)
    altered = prereg()
    altered = PreRegistration(**{**altered.__dict__, "min_signal_trades": 31})
    with pytest.raises(ValueError, match="differs from the frozen"):
        from strategy_qualifier.qualification_gates import verify_preregistration
        verify_preregistration(altered, path)


def test_holdout_gate_failure_is_ruled_out_and_consumed(tmp_path):
    bad_holdout = HoldoutEvidence(
        bif=2.1, dev_mean_r=1.7, holdout_mean_r=-0.1,
        bootstrap_90ci_lower=-0.5, dsr=0.7,
    )
    args = run_args(tmp_path, runner=lambda: (trades(40), bad_holdout))
    result = qualify_strategy(**args)
    assert result.decision == "RULED OUT"
    assert any("BIF" in reason for reason in result.reasons)
    assert any("sign flip" in reason for reason in result.reasons)
    assert any("bootstrap" in reason for reason in result.reasons)
    ledger = json.loads((tmp_path / "holdout.json").read_text())
    assert ledger[0]["decision"] == "RULED OUT"
    with pytest.raises(ValueError, match="terminal|already used|cannot be reset"):
        qualify_strategy(**args)


def synthetic_bars(n=24):
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "open": [100.0] * n, "high": [103.0] * n,
        "low": [99.0] * n, "close": [100.5] * n,
    })


def winning_strategy(history):
    i = len(history) - 1
    return Signal(bar_index=i, side=1, stop_loss=95.0, take_profit=102.0,
                  hypothesis_id="H-001")


winning_strategy.hypothesis_id = "H-001"


def test_ohlc_pipeline_requires_research_evidence_and_preregistration(tmp_path):
    p = prereg()
    register_preregistration(p, tmp_path / "prereg.json")
    result = qualify_bars(
        bars=synthetic_bars(), strategy=winning_strategy, split_index=12,
        preregistration=p, preregistration_path=tmp_path / "prereg.json",
        development_evidence=evidence(30),
        holdout_evidence_builder=lambda rows, bars: holdout_evidence(),
        criteria=criteria(), dev_ledger_path=tmp_path / "dev.csv",
        holdout_ledger_path=tmp_path / "holdout.json", code_version="synthetic-test",
        data_version="synthetic-ohlc-v1",
        config=BacktestConfig(spread=0.1, slippage_per_side=0.01),
    )
    # The fixture's constant-price exits do not establish DSR; it must not pass.
    assert result.decision in ("RULED OUT", "INCONCLUSIVE")


def test_qualification_split_rejects_unsorted_timestamp_data(tmp_path):
    bars = synthetic_bars()
    bars.loc[[0, 1], "timestamp"] = bars.loc[[1, 0], "timestamp"].to_numpy()
    p = prereg()
    register_preregistration(p, tmp_path / "prereg.json")
    with pytest.raises(ValueError, match="chronological"):
        qualify_bars(
            bars=bars, strategy=winning_strategy, split_index=12,
            preregistration=p, preregistration_path=tmp_path / "prereg.json",
            development_evidence=evidence(),
            holdout_evidence_builder=lambda rows, frame: holdout_evidence(),
            criteria=criteria(), dev_ledger_path=tmp_path / "dev.csv",
            holdout_ledger_path=tmp_path / "holdout.json",
            code_version="test", data_version="unordered",
        )
