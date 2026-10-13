# Strategy Qualifier

A standalone research and qualification engine for testing trading hypotheses before they are considered for a separate trading system.

**This project does not place live orders.** Its purpose is to measure whether a frozen, explicitly defined strategy has evidence of an edge after realistic costs and statistical scrutiny.

## Current status

The qualifier now includes two testing paths: a generic callback backtester for strategies you supply, and an extracted SMC replay engine preserving the fork's sweep → structure-break → flip-zone retest logic. Reusable statistical/research utilities are also being moved into this repository. CI is used to validate each migration.

## Design principles

- **Causality first:** a signal may use only information available at its decision time.
- **Separate development and holdout data:** holdout data is not for iterative tuning.
- **Log every hypothesis:** selection bias grows with the number of variants tried.
- **Realistic evaluation:** include transaction costs, trade counts, uncertainty, stability, and baselines.
- **No live execution:** this repository is a research/qualification tool, not an order-routing bot.

## Extracted strategy and research tools

- `strategy_qualifier.strategies.fork_smc`: the fork's SMC signal generator, retest state machine, original event-driven replay/backtest, session rules, MTF confluence/tagging, structure targets and causal MTF structure integration.
- `strategy_qualifier.market`: ATR, swings, structure, structure state, zones, liquidity and MTF context.
- `strategy_qualifier.research`: hypothesis ledger, deflated Sharpe, block bootstrap, walk-forward analysis, random-entry baseline, rank stability, Monte Carlo trade resampling, session-vs-outside analysis and backtest-inflation factor.
- `strategy_qualifier.backtest`: separate generic callback engine for other strategies, including market and resting-limit entries.

The source fork also contains dashboards, MT5/live execution, data-specific diagnostics and one-off patch scripts. Those are intentionally not imported into this research-only package; the qualifier should not need a live trading runtime to evaluate a strategy.

## Test any strategy

For the extracted SMC strategy, use the fork-compatible replay engine:

```python
import pandas as pd
from strategy_qualifier.strategies.fork_smc import run_backtest

bars = pd.read_csv("your_ohlc.csv", parse_dates=["timestamp"]).set_index("timestamp")
result = run_backtest(
    bars,
    reward_multiple=2.0,
    max_bars_to_retest=20,
    spread=0.2,
    slippage=0.05,
    commission=0.0,
)
print(result.win_rate, result.expectancy, len(result.trades))
```

This preserves the SMC engine's own setup/retest lifecycle. Other strategies should use the generic callback interface below, rather than being forced into SMC rules.


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


## Development/holdout qualification pipeline

The generic OHLC path can also run a complete research qualification: it backtests
the development segment, applies caller-supplied gates, and evaluates the frozen
strategy on the holdout segment only if development passes. The split is positional
and chronological: `split_index` is the first holdout candle. The strategy sees
only history available through each candle close; earlier candles remain available
as holdout warm-up context, while pre-split signals are excluded from holdout
trades.

```python
import pandas as pd
from strategy_qualifier.backtest import BacktestConfig, Signal
from strategy_qualifier.qualification import QualificationCriteria
from strategy_qualifier.qualification_data import qualify_bars

bars = pd.read_csv("your_ohlc.csv", parse_dates=["timestamp"])
# Ensure rows are sorted oldest-to-newest before choosing the split.

def my_strategy(history: pd.DataFrame):
    i = len(history) - 1
    # Replace this example condition with your frozen strategy rules.
    if i >= 1 and history.close.iloc[i] > history.close.iloc[i - 1]:
        close = float(history.close.iloc[i])
        return Signal(
            bar_index=i, side=1, stop_loss=close - 2.0,
            take_profit=close + 4.0, hypothesis_id="my-strategy-v1",
        )
    return None

my_strategy.hypothesis_id = "my-strategy-v1"

result = qualify_bars(
    bars=bars,
    strategy=my_strategy,
    split_index=int(len(bars) * 0.7),  # first 70% development, remaining holdout
    criteria=QualificationCriteria(
        min_trades=30,
        min_expectancy=0.0,
        max_drawdown=100.0,
        # Set min_dsr only if you have chosen and justified a threshold.
    ),
    dev_ledger_path="research/dev_ledger.csv",
    holdout_ledger_path="research/holdout_ledger.json",
    code_version="your-frozen-commit-or-version",
    data_version="symbol-timeframe-provider-and-dataset-version",
    config=BacktestConfig(spread=0.2, slippage_per_side=0.05),
)
print(result.decision, result.metrics, result.holdout_metrics)
```

The criteria above are an example only, not recommended trading thresholds. Use a
unique, stable `hypothesis_id` for each materially different strategy variant.
Keep the holdout ledger and do not reset it to retry a hypothesis. The ledger is a
local research guardrail, not a tamper-proof security boundary. This synthetic/CI
integration proves the pipeline wiring and decision flow; real-market performance
still requires your versioned OHLC data, frozen strategy rules, and justified
costs and thresholds.

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
