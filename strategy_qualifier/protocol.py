"""Small, explicit research-data guardrails."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path


class DataUse(str, Enum):
    DEV = "dev"
    HOLDOUT = "holdout"


class Decision(str, Enum):
    PASS = "PASS"
    RULED_OUT = "RULED OUT"
    INCONCLUSIVE = "INCONCLUSIVE"
    EXPLORATORY = "EXPLORATORY"


@dataclass(frozen=True)
class HypothesisRun:
    hypothesis_id: str
    data_used: DataUse
    decision: Decision
    n_trades: int
    code_version: str
    data_version: str

    def validate(self) -> None:
        if not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id is required")
        if self.n_trades < 0:
            raise ValueError("n_trades must be non-negative")
        if self.data_used is DataUse.DEV and self.decision is not Decision.EXPLORATORY:
            raise ValueError("development-data runs must be marked EXPLORATORY")
        if not self.code_version.strip() or not self.data_version.strip():
            raise ValueError("code_version and data_version are required")


def record_holdout_once(path: str | Path, hypothesis_id: str, payload: dict) -> None:
    """Append a holdout record only if this hypothesis has never used holdout.

    This local JSON ledger is a guardrail, not a security boundary. Keep it under
    version control or otherwise back it up so the single-use history is preserved.
    """
    if not hypothesis_id.strip():
        raise ValueError("hypothesis_id is required")
    file_path = Path(path)
    rows = []
    if file_path.exists():
        rows = json.loads(file_path.read_text(encoding="utf-8"))
    if any(row.get("hypothesis_id", "").strip().casefold() == hypothesis_id.strip().casefold()
           for row in rows):
        raise ValueError(f"holdout already used for hypothesis '{hypothesis_id}'")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    rows.append({"hypothesis_id": hypothesis_id, **payload})
    file_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
