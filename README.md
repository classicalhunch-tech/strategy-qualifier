# Strategy Qualifier

A standalone research and qualification engine for testing trading hypotheses before they are considered for a separate trading system.

**This project does not place live orders.** Its purpose is to measure whether a frozen, explicitly defined strategy has evidence of an edge after realistic costs and statistical scrutiny.

## Current status

Initial scaffold. The first migration brings in selected, inspectable research primitives and statistical tools from the separate source repository. No claim is made that the migrated code has passed tests until the test suite is actually run.

## Design principles

- **Causality first:** a signal may use only information available at its decision time.
- **Separate development and holdout data:** holdout data is not for iterative tuning.
- **Log every hypothesis:** selection bias grows with the number of variants tried.
- **Realistic evaluation:** include transaction costs, trade counts, uncertainty, stability, and baselines.
- **No live execution:** this repository is a research/qualification tool, not an order-routing bot.

## Planned layers

1. **Market primitives:** ATR, confirmed swings, market structure, supply/demand zones, liquidity.
2. **Backtest adapter:** a small, explicit interface that consumes frozen signals and models fills/costs without importing a live bot.
3. **Statistical judge:** deflated Sharpe, block bootstrap, walk-forward checks, stability and random-entry baselines.
4. **Research guardrails:** dev/holdout separation, single-use holdout checks, hypothesis ledger, reproducible configs.
5. **Candidate generator (later):** propose bounded hypotheses; it must not silently tune on holdout data.

## Quick start

Python 3.11 is the initial target.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

## Source provenance

Selected code is being adapted from [classicalhunch-tech/A-stratergy-fork](https://github.com/classicalhunch-tech/A-stratergy-fork). The original repository is kept separate and is not modified by this project. Review the source repository's license before redistributing or reusing its code.

See [docs/architecture.md](docs/architecture.md) and [docs/research_protocol.md](docs/research_protocol.md).
