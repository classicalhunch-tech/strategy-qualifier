"""End-to-end, research-only qualification gate for frozen strategy hypotheses.

This module orchestrates existing backtest outputs; it does not generate signals,
change strategy rules, or place orders. All thresholds are supplied by the caller.
Development runs are always logged as EXPLORATORY. Holdout is reserved in a JSON
ledger before the holdout callback is invoked, so failed callbacks still consume it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Callable, Iterable

from strategy_qualifier.backtest import Trade, summarize_trades
from strategy_qualifier.protocol import record_holdout_once
from strategy_qualifier.research.deflated_sharpe import deflated_sharpe
from strategy_qualifier.research.ledger import add_entry, count_trials


@dataclass(frozen=True)
class QualificationCriteria:
    """Explicit gates; expectancy and drawdown are measured in net PnL units."""

    min_trades: int
    min_expectancy: float
    max_drawdown: float
    min_dsr: float | None = None

    def validate(self) -> None:
        if self.min_trades < 0:
            raise ValueError("min_trades must be >= 0")
        if not math.isfinite(self.min_expectancy):
            raise ValueError("min_expectancy must be finite")
        if not math.isfinite(self.max_drawdown) or self.max_drawdown < 0:
            raise ValueError("max_drawdown must be finite and >= 0")
        if self.min_dsr is not None and (
            not math.isfinite(self.min_dsr) or not 0 <= self.min_dsr <= 1
        ):
            raise ValueError("min_dsr must be between 0 and 1")


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
        failures.append(
            f"expectancy {summary['expectancy']:.8g} < {criteria.min_expectancy:.8g}"
        )
    if summary["max_drawdown"] > criteria.max_drawdown:
        failures.append(
            f"max_drawdown {summary['max_drawdown']:.8g} > {criteria.max_drawdown:.8g}"
        )
    if criteria.min_dsr is not None and (
        dsr is None or not math.isfinite(dsr) or dsr < criteria.min_dsr
    ):
        failures.append(f"DSR {dsr!r} is below required {criteria.min_dsr:.8g}")
    return failures


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
    development_trades: Iterable[Trade],
    holdout_runner: Callable[[], Iterable[Trade]],
    criteria: QualificationCriteria,
    dev_ledger_path: str | Path,
    holdout_ledger_path: str | Path,
    code_version: str,
    dev_data_version: str,
    holdout_data_version: str,
) -> QualificationResult:
    """Evaluate development trades, then run holdout only if every dev gate passes.

    The holdout callback must use the already-frozen strategy and untouched
    chronological holdout data. This function cannot verify that caller-side fact.
    A holdout attempt is reserved before callback execution and cannot be repeated
    for the same case-insensitive hypothesis ID.
    """
    hypothesis_id = hypothesis_id.strip()
    if not hypothesis_id:
        raise ValueError("hypothesis_id is required")
    if not callable(holdout_runner):
        raise TypeError("holdout_runner must be callable")
    if not code_version.strip() or not dev_data_version.strip() or not holdout_data_version.strip():
        raise ValueError("code_version and both data versions are required")
    criteria.validate()

    dev_rows, dev_summary, dev_returns = _metrics(development_trades)
    trials = max(1, count_trials(str(dev_ledger_path)) + 1)
    dev_dsr_result = deflated_sharpe(dev_returns, n_trials=trials)
    dev_dsr = float(dev_dsr_result["dsr"])
    dev_failures = _failed_gates(
        dev_summary, criteria, dev_dsr if math.isfinite(dev_dsr) else None
    )
    add_entry(
        str(dev_ledger_path), hypothesis_id, "dev",
        float(dev_summary["expectancy"]),
        (float(__import__("numpy").std([t.net_pnl for t in dev_rows], ddof=1)
               / math.sqrt(len(dev_rows))) if len(dev_rows) > 1 else 0.0),
        int(dev_summary["trade_count"]), "EXPLORATORY",
        notes=f"code={code_version}; data={dev_data_version}; trials_used={trials}",
    )
    if dev_failures:
        return QualificationResult(
            hypothesis_id, "RULED OUT" if dev_summary["trade_count"] >= criteria.min_trades
            else "INCONCLUSIVE", "development", dev_summary, tuple(dev_failures),
            dev_dsr if math.isfinite(dev_dsr) else None,
        )

    # Reserve the single-use holdout BEFORE the callback can inspect its data.
    record_holdout_once(holdout_ledger_path, hypothesis_id, {
        "status": "STARTED",
        "decision": "INCONCLUSIVE",
        "code_version": code_version,
        "data_version": holdout_data_version,
    })
    try:
        holdout_rows, holdout_summary, holdout_returns = _metrics(holdout_runner())
        holdout_dsr_result = deflated_sharpe(holdout_returns, n_trials=trials)
        holdout_dsr = float(holdout_dsr_result["dsr"])
        holdout_failures = _failed_gates(
            holdout_summary, criteria,
            holdout_dsr if math.isfinite(holdout_dsr) else None,
        )
        holdout_decision = "PASS" if not holdout_failures else (
            "RULED OUT" if holdout_summary["trade_count"] >= criteria.min_trades
            else "INCONCLUSIVE"
        )
        _update_reserved_holdout(holdout_ledger_path, hypothesis_id, {
            "status": "COMPLETED",
            "decision": holdout_decision,
            "metrics": holdout_summary,
            "dsr": holdout_dsr if math.isfinite(holdout_dsr) else None,
            "reasons": holdout_failures,
        })
        return QualificationResult(
            hypothesis_id, holdout_decision, "holdout", dev_summary, (),
            dev_dsr if math.isfinite(dev_dsr) else None,
            holdout_summary, holdout_decision,
        )
    except Exception as exc:
        _update_reserved_holdout(holdout_ledger_path, hypothesis_id, {
            "status": "ERROR_HOLDOUT_CONSUMED",
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
