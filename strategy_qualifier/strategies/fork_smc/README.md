# Extracted SMC strategy engine

This package contains the source fork's core SMC strategy implementation, kept separate from the qualifier's generic backtest engine:

- swing, structure, zone and liquidity detectors in `strategy_qualifier.market`
- `signals.py`: liquidity sweep → structure break → flip-zone setup generation
- `retest_engine.py`: pending-retest lifecycle and trigger logic
- `rules.py`: session and zone-selection rules
- `confluence.py`: optional higher-timeframe confluence filters
- `replay.py`: the fork's event-driven SMC replay/backtest, including costs, retest handling and configurable fills/targets

Use `strategy_qualifier.strategies.fork_smc.run_backtest(...)` to run the extracted SMC strategy on OHLC data. Use `strategy_qualifier.backtest.run_strategy_backtest(...)` for other strategies implementing the qualifier callback interface. These are different engines because strategies may have different rules.

Source modules were adapted only to import from the independent qualifier package. This extraction does not import MT5 execution or dashboard runtime code. Check the source repository's reuse/licensing terms before redistributing the extracted code beyond your own project.
