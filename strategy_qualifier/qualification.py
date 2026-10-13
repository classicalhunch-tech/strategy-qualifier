"""Strict, research-only qualification gate for frozen strategy hypotheses.

A PASS requires pre-registration plus every mandatory development and holdout
research gate. Missing evidence fails closed. This package never places orders.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from strategy_qualifier.backtest import Trade, summarize_trades
from strategy_qualifier.protocol import record_holdout_once
from strategy_qualifier.research.deflated_sharpe import deflated_sharpe
from strategy_qualifier.research.ledger import add_entry, count_trials
from strategy_qualifier.qualification_gates import (
    DevelopmentEvidence, HoldoutEvidence, PreRegistration,
    mark_preregistration_status, verify_preregistration,
)


@dataclass(frozen=True)
class QualificationCriteria:
    """Base risk/performance gates. Statistical gates are mandatory and fixed."""

    min_trades: int
    min_expectancy: float
    max_drawdown: float
    min_dsr: float = 0.90

    def validate(self) -> None:
        if self.min_trades < 1:
            raise ValueError("min_trades must be >= 1")
        if not math.isfinite(self.min_expectancy):
            raise ValueError("min_expectancy must be finite")
        if not math.isfinite(self.max_drawdown) or self.max_drawdown < 0:
            raise ValueError("max_drawdown must be finite and >= 0")
        if not math.isfinite(self.min_dsr) or self.min_dsr < 0.90 or self.min_dsr > 1:
            raise ValueError("mandatory DSR threshold cannot be lower than 0.90")


@dataclass(frozen=True)
class QualificationResult:
    hypothesis_id: str
    decision: str
    stage: str
    metrics: dict
    reasons: tuple[str, ...]
    dsr: float | None = None
    holdout_metrics: dict | None = None
    holdout_decision: str | None = None


def _metrics(trades: Iterable[Trade]) -> tuple[list[Trade], dict, list[float]]:
    rows = list(trades)
    summary = summarize_trades(rows)
    returns = [float(t.r_multiple) for t in rows]
    return rows, summary, returns


def _failed_gates(summary: dict, criteria: QualificationCriteria,
                  dsr: float | None) -> list[str]:
    failures = []
    if summary["trade_count"] < criteria.min_trades:
        failures.append(f"trade_count {summary['trade_count']} < {criteria.min_trades}")
    if summary["expectancy"] < criteria.min_expectancy:
        failures.append(f"expectancy {summary['expectancy']:.8g} < {criteria.min_expectancy:.8g}")
    if summary["max_drawdown"] > criteria.max_drawdown:
        failures.append(f"max_drawdown {summary['max_drawdown']:.8g} > {criteria.max_drawdown:.8g}")
    if dsr is None or not math.isfinite(dsr) or dsr <= 0.90 or dsr < criteria.min_dsr:
        failures.append(f"DSR {dsr!r} must be > 0.90 and >= configured floor {criteria.min_dsr:.8g}")
    return failures


def _close(left: float, right: float) -> bool:
    """Strictly compare independently recomputed research statistics."""
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)


def _update_reserved_holdout(path: str | Path, hypothesis_id: str, payload: dict) -> None:
    file_path = Path(path)
    rows = json.loads(file_path.read_text(encoding="utf-8"))
    for row in reversed(rows):
        if row.get("hypothesis_id", "").strip().casefold() == hypothesis_id.strip().casefold():
            row.update(payload)
            file_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
            return
    raise RuntimeError("reserved holdout record disappeared")


def qualify_strategy(
    *,
    hypothesis_id: str,
    preregistration: PreRegistration,
    preregistration_path: str | Path,
    development_trades: Iterable[Trade],
    development_evidence: DevelopmentEvidence,
    holdout_runner: Callable[[], tuple[Iterable[Trade], HoldoutEvidence]],
    criteria: QualificationCriteria,
    dev_ledger_path: str | Path,
    holdout_ledger_path: str | Path,
    code_version: str,
    dev_data_version: str,
    holdout_data_version: str,
) -> QualificationResult:
    """Require every development gate before reserving a single-use holdout.

    holdout_runner must return (holdout trades, computed HoldoutEvidence). It is
    invoked only after frozen preregistration, trade gates, DSR, all development
    evidence, and N* pass. The attempt is consumed before callback execution.
    """
    hypothesis_id = hypothesis_id.strip()
    if not hypothesis_id or hypothesis_id.casefold() != preregistration.hypothesis_id.strip().casefold():
        raise ValueError("hypothesis_id must match the frozen pre-registration")
    if not callable(holdout_runner):
        raise TypeError("holdout_runner must be callable")
    if not code_version.strip() or not dev_data_version.strip() or not holdout_data_version.strip():
        raise ValueError("code_version and both data versions are required")
    criteria.validate()
    prereg_hash = verify_preregistration(preregistration, preregistration_path)
    if criteria.min_trades != preregistration.min_trades:
        raise ValueError("qualification min_trades must exactly match the frozen pre-registration")
    if criteria.min_expectancy != preregistration.min_expectancy or criteria.max_drawdown != preregistration.max_drawdown:
        raise ValueError("criteria differ from frozen pre-registration")

    dev_rows, dev_summary, dev_returns = _metrics(development_trades)
    trials = max(1, count_trials(str(dev_ledger_path), "dev") + 1)
    dev_dsr_result = deflated_sharpe(dev_returns, n_trials=trials)
    dev_dsr = float(dev_dsr_result["dsr"])
    dev_failures = _failed_gates(dev_summary, criteria, dev_dsr if math.isfinite(dev_dsr) else None)
    dev_failures.extend(development_evidence.failures(preregistration, trials))
    dev_integrity_problems: list[str] = []
    if development_evidence.signal_trade_count != dev_summary["trade_count"]:
        problem = "development evidence signal_trade_count does not match the actual development trades"
        dev_failures.append(problem)
        dev_integrity_problems.append(problem)
    actual_dev_mean_r = float(np.mean(dev_returns)) if dev_returns else float("nan")
    if (not math.isfinite(development_evidence.observed_mean_r)
            or not _close(development_evidence.observed_mean_r, actual_dev_mean_r)):
        problem = "development evidence observed_mean_r does not match the actual development trade returns"
        dev_failures.append(problem)
        dev_integrity_problems.append(problem)
    if (not math.isfinite(development_evidence.dsr)
            or not 0.0 <= development_evidence.dsr <= 1.0):
        problem = "development evidence DSR must be finite and within [0, 1]"
        dev_failures.append(problem)
        dev_integrity_problems.append(problem)
    elif math.isfinite(dev_dsr) and not _close(development_evidence.dsr, dev_dsr):
        problem = "development evidence DSR does not match the recomputed DSR from development trades"
        dev_failures.append(problem)
        dev_integrity_problems.append(problem)
    if dev_summary["trade_count"] < preregistration.min_trades:
        dev_failures.append(
            f"minimum dev trades not met: {dev_summary['trade_count']} < {preregistration.min_trades}"
        )
    add_entry(
        str(dev_ledger_path), hypothesis_id, "dev",
        float(dev_summary["expectancy"]),
        (float(np.std([t.net_pnl for t in dev_rows], ddof=1) / math.sqrt(len(dev_rows)))
         if len(dev_rows) > 1 else 0.0),
        int(dev_summary["trade_count"]), "EXPLORATORY",
        notes=f"code={code_version}; data={dev_data_version}; trials_used={trials}; prereg_sha256={prereg_hash}; dev_failures={len(dev_failures)}",
    )
    if dev_failures:
        enough_data = dev_summary["trade_count"] >= max(criteria.min_trades, preregistration.min_signal_trades)
        decision = (
            "INCONCLUSIVE" if dev_integrity_problems
            else ("RULED OUT" if enough_data else "INCONCLUSIVE")
        )
        status = "DEV_RULED_OUT" if decision == "RULED OUT" else "DEV_INCONCLUSIVE"
        mark_preregistration_status(str(preregistration_path), hypothesis_id, status, {
            "decision": decision, "stage": "development", "reasons": dev_failures,
            "metrics": dev_summary, "dsr": dev_dsr if math.isfinite(dev_dsr) else None,
        })
        add_entry(
            str(dev_ledger_path), hypothesis_id, "dev",
            float(dev_summary["expectancy"]), 0.0, int(dev_summary["trade_count"]),
            decision, notes="mandatory development gate failure; " + "; ".join(dev_failures),
        )
        return QualificationResult(
            hypothesis_id, decision, "development", dev_summary, tuple(dev_failures),
            dev_dsr if math.isfinite(dev_dsr) else None,
        )

    mark_preregistration_status(str(preregistration_path), hypothesis_id, "DEV_PASSED", {
        "decision": "DEV PASSED", "stage": "development", "metrics": dev_summary,
        "dsr": dev_dsr if math.isfinite(dev_dsr) else None,
    })
    record_holdout_once(holdout_ledger_path, hypothesis_id, {
        "status": "STARTED", "decision": "INCONCLUSIVE",
        "code_version": code_version, "data_version": holdout_data_version,
        "preregistration_sha256": prereg_hash, "dev_trials_used": trials,
    })
    try:
        holdout_result = holdout_runner()
        if not isinstance(holdout_result, tuple) or len(holdout_result) != 2:
            raise TypeError("holdout_runner must return (trades, HoldoutEvidence)")
        holdout_trades, holdout_evidence = holdout_result
        if not isinstance(holdout_evidence, HoldoutEvidence):
            raise TypeError("holdout_runner must provide HoldoutEvidence computed from holdout data")
        holdout_rows, holdout_summary, holdout_returns = _metrics(holdout_trades)
        holdout_dsr_result = deflated_sharpe(holdout_returns, n_trials=trials)
        holdout_dsr = float(holdout_dsr_result["dsr"])
        holdout_failures = _failed_gates(
            holdout_summary, criteria,
            holdout_dsr if math.isfinite(holdout_dsr) else None,
        )
        holdout_failures.extend(holdout_evidence.failures(trials))
        holdout_integrity_problems: list[str] = []
        actual_holdout_mean_r = float(np.mean(holdout_returns)) if holdout_returns else float("nan")
        actual_dev_mean_r = float(np.mean(dev_returns)) if dev_returns else float("nan")
        if (not math.isfinite(holdout_evidence.dev_mean_r)
                or not _close(holdout_evidence.dev_mean_r, actual_dev_mean_r)):
            problem = "holdout evidence dev_mean_r does not match the actual development trade returns"
            holdout_failures.append(problem)
            holdout_integrity_problems.append(problem)
        if (not math.isfinite(holdout_evidence.holdout_mean_r)
                or not _close(holdout_evidence.holdout_mean_r, actual_holdout_mean_r)):
            problem = "holdout evidence holdout_mean_r does not match the actual holdout trade returns"
            holdout_failures.append(problem)
            holdout_integrity_problems.append(problem)
        if (not math.isfinite(holdout_evidence.dsr)
                or not 0.0 <= holdout_evidence.dsr <= 1.0):
            problem = "holdout evidence DSR must be finite and within [0, 1]"
            holdout_failures.append(problem)
            holdout_integrity_problems.append(problem)
        elif math.isfinite(holdout_dsr) and not _close(holdout_evidence.dsr, holdout_dsr):
            problem = "holdout evidence DSR does not match the recomputed DSR from holdout trades"
            holdout_failures.append(problem)
            holdout_integrity_problems.append(problem)
        holdout_decision = (
            "INCONCLUSIVE" if holdout_integrity_problems
            else ("PASS" if not holdout_failures else (
                "RULED OUT" if holdout_summary["trade_count"] >= criteria.min_trades
                else "INCONCLUSIVE"
            ))
        )
        mark_preregistration_status(str(preregistration_path), hypothesis_id, holdout_decision.replace(" ", "_") if holdout_decision == "RULED OUT" else holdout_decision, {
            "decision": holdout_decision, "stage": "holdout", "reasons": holdout_failures,
            "metrics": holdout_summary,
        })
        add_entry(
            str(dev_ledger_path), hypothesis_id, "holdout",
            float(holdout_summary["expectancy"]), 0.0, int(holdout_summary["trade_count"]),
            holdout_decision, notes="final holdout verdict; " + "; ".join(holdout_failures),
        )
        _update_reserved_holdout(holdout_ledger_path, hypothesis_id, {
            "status": "COMPLETED", "decision": holdout_decision,
            "metrics": holdout_summary,
            "dsr": holdout_dsr if math.isfinite(holdout_dsr) else None,
            "research_evidence": {
                "bif": holdout_evidence.bif,
                "dev_mean_r": holdout_evidence.dev_mean_r,
                "holdout_mean_r": holdout_evidence.holdout_mean_r,
                "bootstrap_90ci_lower": holdout_evidence.bootstrap_90ci_lower,
                "dsr": holdout_evidence.dsr,
                "variants_tested": holdout_evidence.variants_tested,
                "spearman_rho": holdout_evidence.spearman_rho,
                "winner_oos_percentile": holdout_evidence.winner_oos_percentile,
                "bootstrap_block_size": holdout_evidence.bootstrap_block_size,
                "bootstrap_repetitions": holdout_evidence.bootstrap_repetitions,
                "dsr_trials": holdout_evidence.dsr_trials,
            },
            "reasons": holdout_failures,
        })
        return QualificationResult(
            hypothesis_id, holdout_decision, "holdout", dev_summary,
            tuple(holdout_failures), dev_dsr if math.isfinite(dev_dsr) else None,
            holdout_summary, holdout_decision,
        )
    except Exception as exc:
        try:
            mark_preregistration_status(str(preregistration_path), hypothesis_id, "INCONCLUSIVE", {
                "decision": "INCONCLUSIVE", "stage": "holdout",
                "error": f"{type(exc).__name__}: {exc}",
            })
        except Exception:
            pass
        _update_reserved_holdout(holdout_ledger_path, hypothesis_id, {
            "status": "ERROR_HOLDOUT_CONSUMED",
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
