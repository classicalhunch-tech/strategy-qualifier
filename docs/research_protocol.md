# Research protocol

1. Write down the hypothesis, market, timeframe, signal, entry, stop, target, costs, and invalidation criteria before running the experiment.
2. Split data chronologically into development and holdout segments. Never randomly shuffle time-series bars.
3. Use development data for exploration only. Record every variant, including failed ones.
4. Freeze the exact rules and code before evaluating on holdout data.
5. Use holdout once per frozen hypothesis. If rules change after seeing holdout results, the holdout is no longer untouched for that revised hypothesis.
6. Report trade count, net expectancy, drawdown, uncertainty intervals, costs, regime/time stability, and a suitable baseline. A high win rate alone is not evidence of an edge.
7. Do not call a strategy qualified solely because one metric passes a threshold. Small samples and correlated trades can make estimates misleading.
8. Preserve the data version, code commit, config, random seed, and all attempted variants.
9. This engine provides research evidence, not a guarantee of future profitability.
