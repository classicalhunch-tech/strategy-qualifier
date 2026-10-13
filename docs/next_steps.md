# Implementation status

## Strategy testing foundation
- [x] Keep the qualifier in a separate repository; do not modify the source trading bot.
- [x] Provide a generic callback backtester for strategies that implement the qualifier's signal interface.
- [x] Extract the fork's core SMC market components: ATR, swings, structure, structure state, zones, liquidity and MTF context.
- [x] Extract the fork's SMC signal generator and pending-retest state machine.
- [x] Extract the fork's event-driven SMC replay engine, including its entry/fill options, costs, retest expiry and target handling.
- [x] Extract reusable research utilities: hypothesis ledger, Deflated Sharpe, block bootstrap, walk-forward analysis, random-entry baseline, rank stability, Monte Carlo, session comparisons, fill-mode/stop/target/pattern reports and Z-vs-room analyses.
- [x] Add CI smoke tests for extracted modules and the SMC replay API.

## Remaining work needed for strategy testing
- [ ] Wait for and fix any failures in the latest CI run.
- [ ] Review copied modules for source-specific paths, dataset assumptions and any behavior that needs adaptation to the qualifier.
- [ ] Add focused tests around each strategy/research module before relying on its numeric results.
- [ ] Keep each strategy's entry, exit and lifecycle rules distinct; use the generic callback interface for strategies not implemented by the SMC engine.

## Scope boundary
- This repository is for research and qualification only; it must not place live orders.
- The source trading bot remains unchanged.
- MT5 execution, dashboard runtime, credentials, historical datasets and one-off patch/debug scripts are not imported as strategy-testing tools.
- Do not add candidate generation or unrelated features unless explicitly requested.
