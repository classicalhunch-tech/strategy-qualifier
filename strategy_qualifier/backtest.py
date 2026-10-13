"""Small cost-aware OHLC backtester for frozen, precomputed signals.

Signals are evaluated after bar i closes and may enter no earlier than bar i+1's
open. This deliberately small engine does not generate signals or place live orders.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class Signal:
    """A frozen signal emitted at the close of bar_index."""

    bar_index: int
    side: int  # +1 long, -1 short
    stop_loss: float
    take_profit: float
    hypothesis_id: str = ""

    def validate(self, n_bars: int) -> None:
        if self.side not in (-1, 1):
            raise ValueError("signal side must be +1 (long) or -1 (short)")
        if not math.isfinite(self.stop_loss) or not math.isfinite(self.take_profit):
            raise ValueError("stop_loss and take_profit must be finite")
        if not 0 <= self.bar_index < n_bars - 1:
            raise ValueError("signal bar_index must have a following entry bar")
        if self.side == 1 and not self.stop_loss < self.take_profit:
            raise ValueError("long signal requires stop_loss < take_profit")
        if self.side == -1 and not self.take_profit < self.stop_loss:
            raise ValueError("short signal requires take_profit < stop_loss")


@dataclass(frozen=True)
class BacktestConfig:
    spread: float = 0.0
    slippage_per_side: float = 0.0
    commission_per_unit_per_side: float = 0.0
    max_holding_bars: int | None = None

    def validate(self) -> None:
        values = (self.spread, self.slippage_per_side, self.commission_per_unit_per_side)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("spread, slippage and commission must be finite")
        if min(values) < 0:
            raise ValueError("spread, slippage and commission must be non-negative")
        if self.max_holding_bars is not None and self.max_holding_bars < 1:
            raise ValueError("max_holding_bars must be >= 1 or None")


@dataclass(frozen=True)
class Trade:
    hypothesis_id: str
    side: int
    signal_bar: int
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    exit_reason: str
    bars_held: int
    gross_pnl: float
    costs: float
    net_pnl: float
    initial_risk: float
    r_multiple: float


def run_backtest(
    bars: pd.DataFrame,
    signals: Iterable[Signal],
    config: BacktestConfig | None = None,
) -> list[Trade]:
    """Run non-overlapping trades against OHLC bars.

    Entry is next-bar open. If stop and target both touch within one bar, stop
    wins (conservative ambiguity handling). Gap-through stops fill at the open;
    targets fill at their target price. Spread/slippage are charged on each side,
    and commission is charged per unit on each side. PnL is per one unit.
    """
    cfg = config or BacktestConfig()
    cfg.validate()
    if not isinstance(bars, pd.DataFrame):
        raise ValueError("bars must be a pandas DataFrame")
    frame = bars.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    required = {"open", "high", "low", "close"}
    if not required.issubset(frame.columns):
        raise ValueError(f"bars missing columns: {sorted(required - set(frame.columns))}")
    if frame.empty:
        return []
    frame = frame.loc[:, ["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if frame.isna().any().any() or not frame.map(math.isfinite).all().all():
        raise ValueError("OHLC bars must contain finite numeric values")
    if not ((frame["high"] >= frame[["open", "close", "low"]].max(axis=1)).all()
            and (frame["low"] <= frame[["open", "close", "high"]].min(axis=1)).all()):
        raise ValueError("invalid OHLC relationships")

    prepared = sorted(list(signals), key=lambda s: s.bar_index)
    for signal in prepared:
        signal.validate(len(frame))
    if len({s.bar_index for s in prepared}) != len(prepared):
        raise ValueError("only one signal per bar is supported")

    half_cost = cfg.spread / 2.0 + cfg.slippage_per_side
    commission = cfg.commission_per_unit_per_side
    trades: list[Trade] = []
    next_available_bar = 0

    for signal in prepared:
        entry_bar = signal.bar_index + 1
        if entry_bar < next_available_bar:
            continue
        entry_open = float(frame.iloc[entry_bar]["open"])
        entry_price = entry_open + signal.side * half_cost
        if signal.side == 1 and not signal.stop_loss < entry_price < signal.take_profit:
            continue
        if signal.side == -1 and not signal.take_profit < entry_price < signal.stop_loss:
            continue
        risk = abs(entry_price - signal.stop_loss)
        if risk <= 0:
            continue

        final_bar = len(frame) - 1
        if cfg.max_holding_bars is not None:
            final_bar = min(final_bar, entry_bar + cfg.max_holding_bars - 1)
        exit_bar, raw_exit, reason = final_bar, float(frame.iloc[final_bar]["close"]), "TIME"
        for j in range(entry_bar, final_bar + 1):
            bar = frame.iloc[j]
            o, h, l = float(bar.open), float(bar.high), float(bar.low)
            if signal.side == 1:
                if o <= signal.stop_loss:
                    exit_bar, raw_exit, reason = j, o, "STOP_GAP"
                    break
                if l <= signal.stop_loss:
                    exit_bar, raw_exit, reason = j, signal.stop_loss, "STOP"
                    break
                if h >= signal.take_profit:
                    exit_bar, raw_exit, reason = j, signal.take_profit, "TARGET"
                    break
            else:
                if o >= signal.stop_loss:
                    exit_bar, raw_exit, reason = j, o, "STOP_GAP"
                    break
                if h >= signal.stop_loss:
                    exit_bar, raw_exit, reason = j, signal.stop_loss, "STOP"
                    break
                if l <= signal.take_profit:
                    exit_bar, raw_exit, reason = j, signal.take_profit, "TARGET"
                    break

        exit_price = raw_exit - signal.side * half_cost
        gross = signal.side * (raw_exit - entry_open)
        costs = 2.0 * half_cost + 2.0 * commission
        net = signal.side * (exit_price - entry_price) - 2.0 * commission
        trades.append(Trade(
            hypothesis_id=signal.hypothesis_id,
            side=signal.side,
            signal_bar=signal.bar_index,
            entry_bar=entry_bar,
            exit_bar=exit_bar,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            exit_reason=reason,
            bars_held=exit_bar - entry_bar + 1,
            gross_pnl=gross,
            costs=costs,
            net_pnl=net,
            initial_risk=risk,
            r_multiple=net / risk,
        ))
        next_available_bar = exit_bar + 1

    return trades

def run_strategy_backtest(
    bars: pd.DataFrame,
    strategy,
    config: BacktestConfig | None = None,
) -> list[Trade]:
    """Generate signals chronologically, then evaluate them with the OHLC engine.

    strategy is a callable receiving a copy of bars available through the
    current candle close. It may return None, one Signal, or an iterable of
    Signals. Every emitted signal must have bar_index == len(history) - 1.
    The final candle is not queried because a signal there cannot enter on a
    following candle. This interface works with any strategy; it contains no
    strategy-specific trading rules.
    """
    if not callable(strategy):
        raise TypeError("strategy must be callable: strategy(history)")

    if not isinstance(bars, pd.DataFrame):
        raise ValueError("bars must be a pandas DataFrame")
    if len(bars) < 2:
        return []

    generated: list[Signal] = []
    for end in range(1, len(bars)):
        history = bars.iloc[:end].copy()
        result = strategy(history)
        if result is None:
            continue
        if isinstance(result, Signal):
            current = [result]
        else:
            try:
                current = list(result)
            except TypeError as exc:
                raise TypeError("strategy must return None, Signal, or an iterable of Signals") from exc
        current_bar = end - 1
        for signal in current:
            if not isinstance(signal, Signal):
                raise TypeError("strategy outputs must be Signal instances")
            if signal.bar_index != current_bar:
                raise ValueError(
                    f"strategy may only emit a signal for the current bar ({current_bar})"
                )
            generated.append(signal)

    return run_backtest(bars, generated, config)


def summarize_trades(trades: Iterable[Trade]) -> dict[str, float | int]:
    """Return compact, strategy-agnostic performance metrics for a trade list."""
    rows = list(trades)
    pnl = [float(trade.net_pnl) for trade in rows]
    wins = [value for value in pnl if value > 0]
    losses = [value for value in pnl if value < 0]
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in pnl:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0)
    return {
        "trade_count": len(rows),
        "net_pnl": sum(pnl),
        "win_rate": len(wins) / len(rows) if rows else 0.0,
        "expectancy": sum(pnl) / len(rows) if rows else 0.0,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "average_r": sum(trade.r_multiple for trade in rows) / len(rows) if rows else 0.0,
    }

