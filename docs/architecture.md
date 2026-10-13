# Architecture

## Boundary
This repository evaluates frozen hypotheses. It does not connect to MetaTrader, submit orders, or share runtime state with the trading bot.

## Layers
- `strategy_qualifier.market`: causal market features and structure primitives.
- `strategy_qualifier.backtest` (planned): deterministic simulation of entries, exits, spread, slippage, and fees.
- `strategy_qualifier.statistics` (planned): uncertainty estimates and multiple-testing adjustments.
- `strategy_qualifier.protocol`: hypothesis metadata and development/holdout guardrails.
- `tools/` (planned): command-line research reports.

## Data contract
Input bars must be time-sorted OHLC data. A feature must document when it becomes knowable. Confirmed swings are only available at their confirmation timestamp, not their formation timestamp. Backtest fills must not use future bars to decide an entry.

## Migration policy
The source repository has not exposed a license file during this review. Before copying its implementation wholesale, confirm the applicable license and attribution requirements. This project starts with small independent primitives; selected upstream components can be ported once licensing and dependency contracts are clear.
