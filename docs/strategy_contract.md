# Strategy Contract (Initial Audit)

## Purpose

This document records observable behavior in the source repository so the standalone qualifier can implement and test a clearly frozen hypothesis. It is a behavior specification, not copied source code. The source repository currently declares no license, so further code reuse must wait until reuse rights and attribution requirements are clear.

Source reviewed:
- `strategy/signals.py` — `generate_flip_zone_signals`
- `strategy/rules.py` — `TradingRules.evaluate_setup`
- `strategy/confluence.py` — MTF direction gate
- `strategy/backtest.py` — existing event-driven simulation entry point

## Hypothesis A: Liquidity sweep → structure break → flip-zone retest

This appears to be the more complete SMC/ICT candidate flow. Treat these as source defaults, not yet as a final frozen research configuration.

1. **Liquidity event:** consider liquidity levels marked swept. Sell-side liquidity (SSL / sell_side / low) creates a LONG candidate; buy-side liquidity (BSL / buy_side / high) creates a SHORT candidate.
2. **Structure confirmation:** find a matching structure break no more than 15 bars after the sweep by default.
3. **Flip zone:** find a matching flip zone associated with the structure break no more than 5 bars from the break by default. A consumed zone should not create another candidate.
4. **Entry:** default entry level is the zone midpoint. An alternative `extreme` mode uses the zone's lower boundary for LONG or upper boundary for SHORT.
5. **Stop:** LONG stop is below the zone's lower boundary; SHORT stop is above its upper boundary. Default stop buffer is zero. In `extreme` mode the source enforces a small nonzero buffer when configured buffer is zero.
6. **Target:** default target is 2.5 times initial risk from the planned entry.
7. **Retest expiry:** default maximum is 50 bars from the structure-break bar. A setup that does not trigger within its window should expire.
8. **Session:** signal generation receives a session label from its caller. Do not assume the signal generator itself enforces London/New York hours.
9. **Higher-timeframe context:** an optional MTF filter can gate candidates using 1H macro and 15M internal structure. In the filter's default gated mode, macro direction must agree; internal direction must also agree unless the configured neutral/soft-conflict options permit otherwise. With direction gating disabled, MTF labels are tags rather than entry gates.
10. **Costs and fills:** research runs must model spread, slippage and commission. Entry/exit behavior must follow a documented, conservative fill model; it must not use future bars to decide a signal.

## Hypothesis B: Session + trend + supply/demand mitigation

The separate `TradingRules.evaluate_setup` path describes a simpler rule set and should not be silently combined with Hypothesis A:

- Only London or New York session candles are allowed (configured hours are UTC by assumption).
- In bullish structure, select the nearest unmitigated demand zone and buy when the candle intersects the zone according to the source's boundary checks.
- In bearish structure, select the nearest unmitigated supply zone and sell on the corresponding intersection.
- The source marks a zone mitigated when it emits the setup.
- The stop is beyond the relevant zone boundary by a configurable price buffer (default 2.0 in the source units).
- This rule path does not define a target in the returned setup, so an exit/target policy must be explicitly chosen before it can be qualified.

## Integration hazards to resolve before trusting results

- The standalone qualifier's current backtester enters at the next bar's open. Hypothesis A plans a zone midpoint/extreme retest entry, so the two are **not yet execution-equivalent**. Do not report its current backtest as a faithful result for Hypothesis A.
- The strategy pipeline needs a causal bar-by-bar replay: at each decision time, only already-confirmed swings, structure breaks, zones, and sweeps may be visible.
- A full integration must preserve pending retests, expiry, invalidation, zone consumption, and optional MTF filtering.
- OHLC bars cannot reveal the intrabar order of all events. Define and test conservative assumptions for a bar that touches an entry and stop/target in the same candle.
- The default parameters above are code defaults, not proof that these settings are the user's intended final strategy.
- Dataset, symbol, timeframe, timezone/session handling, position sizing, and final entry/exit configuration must be recorded with each hypothesis run.

## Recommended first research version

Start with Hypothesis A only, using a frozen configuration and a causal replay. Keep Hypothesis B as a separate candidate until its target/exit rule is specified. Before using a holdout set, lock the strategy parameters and the fill model, then record the code/data versions and every development trial.
