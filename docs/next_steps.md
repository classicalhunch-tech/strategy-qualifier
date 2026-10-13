# Implementation roadmap

## Phase 1 — foundation (mostly complete)
- [x] Create a standalone package and test configuration.
- [x] Add ATR with a lagged, decision-time-safe variant.
- [x] Add confirmed pivot swings with explicit confirmation timestamps.
- [x] Add BOS/CHOCH structure, supply/demand zones, and BSL/SSL liquidity components.
- [x] Add a first development/holdout protocol guard.
- [x] Add bootstrap and Deflated Sharpe screening utilities.
- [x] Add a deterministic OHLC backtester for frozen signals, with costs and conservative same-bar stop handling.
- [x] Run the package test suite in CI (see the latest GitHub Actions result).

## Phase 2 — strategy contract and causal integration (current priority)
- [x] Audit the source strategy paths and document their behavior separately.
- [ ] Confirm code reuse/license permissions before further wholesale migration.
- [ ] Freeze one initial hypothesis configuration (recommended: liquidity sweep → structure break → flip-zone retest).
- [ ] Extend the backtest contract to support the strategy's planned retest/limit entry rather than silently substituting next-bar market entry.
- [ ] Implement a causal bar-by-bar replay adapter that feeds only confirmed events into the strategy.
- [ ] Preserve pending retests, expiry, invalidation, and zone-consumption state.
- [ ] Add optional MTF context gating and tests for no-lookahead behavior.
- [ ] Test gap, entry/stop/target same-bar ambiguity, and realistic costs.

## Phase 3 — qualification report
- [ ] Produce a reproducible trade ledger and summary report.
- [ ] Add walk-forward evaluation and development-only baseline comparisons.
- [ ] Add block-bootstrap uncertainty and multiple-testing-aware statistics with clear limitations.
- [ ] Add rank/stability and random-entry baselines.
- [ ] Strengthen the hypothesis ledger and prevent accidental repeated holdout use.

## Phase 4 — bounded candidate generation (later)
- [ ] Generate candidates only from explicitly allowed parameter ranges.
- [ ] Count and record every attempted variant before ranking.
- [ ] Restrict candidate search to development data.
- [ ] Require a frozen candidate before any holdout evaluation.

## Scope boundary

Strategy Qualifier is a research tool. It must not connect to a broker or place live orders. Keep the original bot unchanged and keep the qualifier in its own repository.
