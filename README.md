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

## Test any strategy

Implement a callable that receives a DataFrame containing only bars available through the current candle close. Return `None` or a `Signal` for that current bar. The qualifier enters on the following bar and applies the configured costs and OHLC exit rules. This interface is strategy-agnostic: your strategy rules belong in your own function/module, not in the qualifier.

```python
import pandas as pd
from strategy_qualifier.backtest import (
    BacktestConfig, Signal, run_strategy_backtest, summarize_trades,
)

bars = pd.read_csv("your_ohlc.csv")  # requires open, high, low, close columns

def my_strategy(history: pd.DataFrame):
    i = len(history) - 1
    # Replace this example condition with your own strategy rules.
    if i >= 1 and history.close.iloc[i] > history.close.iloc[i - 1]:
        entry = float(history.close.iloc[i])
        return Signal(
            bar_index=i,
            side=1,
            stop_loss=entry - 2.0,
            take_profit=entry + 4.0,
            hypothesis_id="my-strategy-v1",
        )
    return None

trades = run_strategy_backtest(
    bars, my_strategy,
    BacktestConfig(spread=0.2, slippage_per_side=0.05),
)
print(summarize_trades(trades))
pd.DataFrame([trade.__dict__ for trade in trades]).to_csv("trade_ledger.csv", index=False)
```

Signals are generated from past/current bars only. The last bar is not queried because it has no next bar for entry. Set `entry_price=...` on a `Signal` to model a resting limit/retest entry; `entry_expiry_bars` controls how long it waits. The engine is an OHLC research simulator, not a tick-accurate fill model.

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
