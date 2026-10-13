# Implementation roadmap

## Phase 1 — foundation
- [x] Create a standalone package and test configuration.
- [x] Add ATR with a lagged, decision-time-safe variant.
- [x] Add confirmed pivot swings with explicit confirmation timestamps.
- [x] Add a first development/holdout protocol guard.
- [x] Add bootstrap and Deflated Sharpe screening utilities.
- [ ] Add tests and run the complete suite in CI.

## Phase 2 — market structure
- [ ] Add BOS/CHOCH structure state machine.
- [ ] Add supply/demand zone lifecycle and mitigation.
- [ ] Add BSL/SSL liquidity lifecycle and sweep detection.
- [ ] Add fixture-based causality tests for each primitive.

## Phase 3 — backtest and judge
- [ ] Define a frozen signal contract.
- [ ] Build deterministic backtesting with spread, slippage, commissions, and clear fill timing.
- [ ] Add walk-forward evaluation, baseline comparisons, and stability summaries.
- [ ] Add an append-only hypothesis ledger and report bundle.

## Phase 4 — candidate generation
- [ ] Generate bounded candidates from explicit parameter ranges.
- [ ] Count and record all variants before ranking.
- [ ] Keep candidate search restricted to development data.
- [ ] Require a frozen candidate before any holdout evaluation.
