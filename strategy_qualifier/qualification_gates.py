"""Mandatory pre-registration and research gates for the qualification pipeline.

All statistics in evidence objects must be computed from the corresponding
stage's trade ledger using the research utilities. Missing/invalid evidence fails
closed; the holdout callback is never called unless every development gate passes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreRegistration:
    hypothesis_id: str
    question: str
    hypothesis: str
    expected_sign: str
    market: str
    timeframe: str
    entry_rules: str
    exit_rules: str
    stop_rule: str
    target_rule: str
    session_filter: str
    direction_filter: str
    dev_start: str
    dev_end: str
    holdout_start: str
    holdout_end: str
    min_signal_trades: int
    min_walk_forward_mean_coefficient: float
    alpha: float
    min_trades: int
    min_expectancy: float
    max_drawdown: float
    falsification_criteria: str
    registered_before_data: bool

    def validate(self) -> None:
        required = (
            "hypothesis_id", "question", "hypothesis", "expected_sign", "market",
            "timeframe", "entry_rules", "exit_rules", "stop_rule", "target_rule",
            "session_filter", "direction_filter", "dev_start", "dev_end",
            "holdout_start", "holdout_end", "falsification_criteria",
        )
        for field in required:
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"pre-registration field {field} is required")
        if self.expected_sign not in ("positive", "negative"):
            raise ValueError("expected_sign must be 'positive' or 'negative'")
        if not self.registered_before_data:
            raise ValueError("pre-registration must be frozen before data is inspected")
        if self.min_signal_trades < 1 or self.min_trades < 1:
            raise ValueError("N* and min_trades must be >= 1")
        if not math.isfinite(self.min_walk_forward_mean_coefficient) or self.min_walk_forward_mean_coefficient <= 0:
            raise ValueError("min_walk_forward_mean_coefficient must be finite and > 0, fixed before testing")
        if not math.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if not math.isfinite(self.min_expectancy):
            raise ValueError("min_expectancy must be finite")
        if not math.isfinite(self.max_drawdown) or self.max_drawdown < 0:
            raise ValueError("max_drawdown must be finite and >= 0")
        try:
            ds, de = datetime.fromisoformat(self.dev_start), datetime.fromisoformat(self.dev_end)
            hs, he = datetime.fromisoformat(self.holdout_start), datetime.fromisoformat(self.holdout_end)
        except ValueError as exc:
            raise ValueError("data window boundaries must be ISO-8601 dates/timestamps") from exc
        if not ds < de or not hs < he or de > hs:
            raise ValueError("dev and holdout windows must be ordered and non-overlapping")


def _payload(preregistration: PreRegistration) -> dict[str, Any]:
    preregistration.validate()
    return asdict(preregistration)


def register_preregistration(preregistration: PreRegistration, path: str | Path) -> str:
    """Freeze a hypothesis once; refuses to overwrite an existing registration."""
    payload = _payload(preregistration)
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.exists():
        rows = json.loads(file_path.read_text(encoding="utf-8"))
    else:
        rows = []
    key = preregistration.hypothesis_id.strip().casefold()
    if any(str(row.get("hypothesis_id", "")).strip().casefold() == key for row in rows):
        raise ValueError(f"pre-registration already exists for '{preregistration.hypothesis_id}'")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    rows.append({
        **payload,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": digest,
        "status": "REGISTERED",
    })
    file_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return digest


def verify_preregistration(preregistration: PreRegistration, path: str | Path) -> str:
    """Require the exact frozen registration to exist and match; never auto-register."""
    payload = _payload(preregistration)
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError("pre-registration ledger missing; register hypothesis before testing")
    rows = json.loads(file_path.read_text(encoding="utf-8"))
    key = preregistration.hypothesis_id.strip().casefold()
    matches = [row for row in rows if str(row.get("hypothesis_id", "")).strip().casefold() == key]
    if len(matches) != 1:
        raise ValueError("pre-registration must exist exactly once in the ledger")
    row = matches[0]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if row.get("sha256") != digest:
        raise ValueError("pre-registration differs from the frozen ledger entry")
    if row.get("status") != "REGISTERED":
        raise ValueError(
            f"hypothesis is already in terminal/active state {row.get('status')!r}; "
            "do not rerun it under the same hypothesis ID"
        )
    return digest


def mark_preregistration_status(path: str | Path, hypothesis_id: str,
                                status: str, details: dict | None = None) -> None:
    """Record a one-way status transition; never resets a hypothesis to REGISTERED."""
    allowed = {"DEV_RULED_OUT", "DEV_INCONCLUSIVE", "DEV_PASSED", "PASS", "RULED_OUT", "INCONCLUSIVE"}
    if status not in allowed:
        raise ValueError(f"unsupported pre-registration status: {status}")
    file_path = Path(path)
    rows = json.loads(file_path.read_text(encoding="utf-8"))
    key = hypothesis_id.strip().casefold()
    matches = [row for row in rows if str(row.get("hypothesis_id", "")).strip().casefold() == key]
    if len(matches) != 1:
        raise ValueError("pre-registration must exist exactly once to update status")
    current = matches[0].get("status")
    terminal = {"DEV_RULED_OUT", "DEV_INCONCLUSIVE", "PASS", "RULED_OUT", "INCONCLUSIVE"}
    if current in terminal:
        raise ValueError("pre-registration status is terminal and cannot be reset or overwritten")
    if current == "DEV_PASSED" and status not in {"PASS", "RULED_OUT", "INCONCLUSIVE"}:
        raise ValueError("after DEV_PASSED, only a final holdout verdict may be recorded")
    if current not in {"REGISTERED", "DEV_PASSED"}:
        raise ValueError(f"invalid pre-registration status transition from {current!r}")
    matches[0]["status"] = status
    matches[0]["result"] = details or {}
    file_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class DevelopmentEvidence:
    walk_forward_coefficients: tuple[float, ...]
    session_coefficient: float
    session_cluster_robust_se: float
    session_p_value: float
    random_entry_p_value: float
    observed_mean_r: float
    random_entry_90th_percentile_r: float
    dsr: float
    bootstrap_90ci_lower: float
    signal_trade_count: int
    random_entry_repetitions: int
    bootstrap_block_size: int
    bootstrap_repetitions: int
    dsr_trials: int

    def failures(self, prereg: PreRegistration, expected_trials: int) -> list[str]:
        fail: list[str] = []
        coeffs = self.walk_forward_coefficients
        if len(coeffs) < 3 or not all(math.isfinite(x) for x in coeffs):
            fail.append("walk-forward evidence missing or non-finite (at least 3 folds required)")
        else:
            positive = sum(x > 0 for x in coeffs)
            negative = sum(x < 0 for x in coeffs)
            mean_coef = sum(coeffs) / len(coeffs)
            if positive <= len(coeffs) / 2:
                fail.append("walk-forward: not a majority of folds positive")
            if negative >= 2:
                fail.append("walk-forward: 2 or more folds flip sign")
            if mean_coef <= prereg.min_walk_forward_mean_coefficient:
                fail.append("walk-forward: mean coefficient does not exceed the frozen stability floor")
        numeric = {
            "session_coefficient": self.session_coefficient,
            "session_cluster_robust_se": self.session_cluster_robust_se,
            "session_p_value": self.session_p_value,
            "random_entry_p_value": self.random_entry_p_value,
            "observed_mean_r": self.observed_mean_r,
            "random_entry_90th_percentile_r": self.random_entry_90th_percentile_r,
            "dsr": self.dsr,
            "bootstrap_90ci_lower": self.bootstrap_90ci_lower,
        }
        for name, value in numeric.items():
            if not math.isfinite(value):
                fail.append(f"{name} is missing or non-finite")
        if math.isfinite(self.session_coefficient) and self.session_coefficient <= 0:
            fail.append("session-vs-outside: session effect is not positive")
        if not math.isfinite(self.session_cluster_robust_se) or self.session_cluster_robust_se <= 0:
            fail.append("session-vs-outside: cluster-robust standard error is invalid")
        if not 0 <= self.session_p_value <= prereg.alpha:
            fail.append(f"session-vs-outside: p-value must be <= alpha ({prereg.alpha})")
        if not 0 <= self.random_entry_p_value <= prereg.alpha:
            fail.append(f"random-entry baseline: p-value must be <= alpha ({prereg.alpha})")
        if self.observed_mean_r <= self.random_entry_90th_percentile_r:
            fail.append("random-entry baseline: observed mean R is not above the random 90th percentile")
        if not self.dsr > 0.90:
            fail.append("development DSR must be > 0.90")
        if not self.bootstrap_90ci_lower > 0:
            fail.append("development 90% block-bootstrap CI lower bound must be > 0")
        if self.random_entry_repetitions != 5000:
            fail.append("random-entry baseline must use exactly 5000 repetitions")
        if self.bootstrap_block_size != 5 or self.bootstrap_repetitions < 1000:
            fail.append("development block bootstrap must use block size 5 and at least 1000 repetitions")
        if self.dsr_trials != expected_trials:
            fail.append(f"development DSR trial count {self.dsr_trials} does not match ledger count {expected_trials}")
        if self.signal_trade_count < prereg.min_signal_trades:
            fail.append(
                f"N* not met: {self.signal_trade_count} signal trades < {prereg.min_signal_trades}"
            )
        return fail


@dataclass(frozen=True)
class HoldoutEvidence:
    bif: float
    dev_mean_r: float
    holdout_mean_r: float
    bootstrap_90ci_lower: float
    dsr: float
    variants_tested: int = 1
    spearman_rho: float | None = None
    winner_oos_percentile: float | None = None
    bootstrap_block_size: int = 5
    bootstrap_repetitions: int = 10000
    dsr_trials: int = 1

    def failures(self, expected_trials: int) -> list[str]:
        fail: list[str] = []
        values = (self.bif, self.dev_mean_r, self.holdout_mean_r,
                  self.bootstrap_90ci_lower, self.dsr)
        if not all(math.isfinite(x) for x in values):
            fail.append("holdout evidence contains missing or non-finite required metrics")
            return fail
        if self.dev_mean_r == 0 or self.holdout_mean_r == 0 or self.dev_mean_r * self.holdout_mean_r < 0:
            fail.append("holdout mean R sign flip or zero effect")
        if not 0.5 < self.bif < 1.5:
            fail.append("BIF must be strictly between 0.5 and 1.5")
        if not self.bootstrap_90ci_lower > 0:
            fail.append("holdout 90% block-bootstrap CI lower bound must be > 0")
        if self.bootstrap_block_size != 5 or self.bootstrap_repetitions < 1000:
            fail.append("holdout block bootstrap must use block size 5 and at least 1000 repetitions")
        if self.dsr_trials != expected_trials:
            fail.append(f"holdout DSR trial count {self.dsr_trials} does not match development ledger count {expected_trials}")
        if not self.dsr > 0.90:
            fail.append("holdout DSR must be > 0.90 using the same dev trial count")
        if self.variants_tested > 1:
            if self.spearman_rho is None or not math.isfinite(self.spearman_rho) or self.spearman_rho <= 0.5:
                fail.append("rank stability requires Spearman rho > 0.5 when variants were tested")
            if (self.winner_oos_percentile is None or
                    not math.isfinite(self.winner_oos_percentile) or
                    self.winner_oos_percentile <= 0.5):
                fail.append("rank stability requires IS winner OOS percentile > 0.5 when variants were tested")
        return fail
