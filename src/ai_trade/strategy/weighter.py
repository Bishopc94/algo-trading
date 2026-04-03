"""Adaptive strategy weighting based on historical trade performance.

All strategies start with equal weight (1.0).  After a configurable
burn-in period, the weighter recalculates weights using a composite
score of win rate, profit factor, average P&L, and recency-weighted
performance.

The weight acts as a conviction multiplier in the SignalAggregator:
  adjusted_conviction = raw_conviction * strategy_weight

This means better-performing strategies naturally get higher conviction
scores, which gives them priority in the execution queue and (through
market regime/sentiment modifiers) larger position sizes.

Weight range: [min_weight, max_weight] (default 0.3 to 2.0)
  - A weight of 0.3 means the strategy's conviction is reduced to 30%
  - A weight of 2.0 means the strategy's conviction is doubled (clamped to 1.0)
  - A weight of 1.0 means no adjustment (default during burn-in)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ai_trade.monitoring.logger import get_logger

if TYPE_CHECKING:
    from ai_trade.monitoring.database import Database

logger = get_logger(__name__)


class StrategyWeighter:
    """Compute and apply per-strategy conviction multipliers based on
    historical trade performance.

    Usage:
        weighter = StrategyWeighter(database, config)
        weighter.maybe_recalculate()  # Call periodically
        w = weighter.get_weight("momentum")  # Returns float multiplier
    """

    def __init__(self, database: Database, config) -> None:
        self._db = database
        self._min_weight: float = getattr(config, "min_weight", 0.3)
        self._max_weight: float = getattr(config, "max_weight", 2.0)
        self._burn_in_trades: int = getattr(config, "burn_in_trades", 10)
        self._recalc_interval: int = getattr(config, "recalc_interval_trades", 5)
        self._recency_halflife: float = getattr(config, "recency_halflife_trades", 20.0)

        self._weights: dict[str, float] = {}
        self._trade_count_at_last_recalc: int = 0

    def get_weight(self, strategy_name: str) -> float:
        """Return the current weight for a strategy.  Default 1.0 (no adjustment)."""
        return self._weights.get(strategy_name, 1.0)

    def get_all_weights(self) -> dict[str, float]:
        """Return a copy of all current weights for logging/display."""
        return dict(self._weights)

    def maybe_recalculate(self) -> None:
        """Recalculate weights if enough new trades have completed since last recalc."""
        try:
            all_trades = self._db.get_all_trades()
            closed = [t for t in all_trades if t.get("status") == "closed" and t.get("pnl") is not None]
            total_closed = len(closed)

            if total_closed - self._trade_count_at_last_recalc < self._recalc_interval:
                return

            self._recalculate(closed)
            self._trade_count_at_last_recalc = total_closed
        except Exception:
            logger.exception("weighter_recalc_failed")

    def _recalculate(self, closed_trades: list[dict]) -> None:
        """Compute new weights from closed trade history."""
        # Group trades by strategy
        by_strategy: dict[str, list[dict]] = {}
        for trade in closed_trades:
            name = trade.get("strategy", "unknown")
            by_strategy.setdefault(name, []).append(trade)

        # Compute composite scores for strategies with enough data
        scores: dict[str, float] = {}
        avg_pnls: dict[str, float] = {}

        for name, trades in by_strategy.items():
            if len(trades) < self._burn_in_trades:
                continue

            win_rate = self._compute_win_rate(trades)
            profit_factor = self._compute_profit_factor(trades)
            avg_pnl = self._compute_avg_pnl(trades)
            recency = self._compute_recency_score(trades)

            avg_pnls[name] = avg_pnl
            scores[name] = {
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "avg_pnl_raw": avg_pnl,
                "recency": recency,
            }

        if not scores:
            logger.debug("weighter_no_eligible_strategies", reason="all below burn-in")
            return

        # Normalize avg P&L across strategies
        max_avg_pnl = max(avg_pnls.values()) if avg_pnls else 1.0
        min_avg_pnl = min(avg_pnls.values()) if avg_pnls else 0.0
        pnl_range = max_avg_pnl - min_avg_pnl if max_avg_pnl != min_avg_pnl else 1.0

        new_weights: dict[str, float] = {}

        for name, components in scores.items():
            # Normalize avg P&L to [0, 1]
            if pnl_range > 0 and max_avg_pnl > 0:
                avg_pnl_norm = max(0.0, (avg_pnls[name] - min_avg_pnl) / pnl_range)
            else:
                avg_pnl_norm = 0.5

            # Composite score: weighted combination
            composite = (
                0.35 * components["win_rate"]
                + 0.25 * components["profit_factor"]
                + 0.25 * avg_pnl_norm
                + 0.15 * components["recency"]
            )

            # Map composite [0, 1] → weight [min_weight, max_weight]
            weight = self._min_weight + (self._max_weight - self._min_weight) * composite
            weight = max(self._min_weight, min(self._max_weight, weight))
            new_weights[name] = round(weight, 3)

        self._weights = new_weights

        logger.info(
            "strategy_weights_updated",
            weights=new_weights,
            strategy_count=len(new_weights),
        )

    @staticmethod
    def _compute_win_rate(trades: list[dict]) -> float:
        """Win count / total count."""
        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        return wins / len(trades) if trades else 0.0

    @staticmethod
    def _compute_profit_factor(trades: list[dict]) -> float:
        """Gross profit / gross loss, capped at 3.0, normalized to [0, 1]."""
        gross_profit = sum(t["pnl"] for t in trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t["pnl"] for t in trades if t.get("pnl", 0) < 0))

        if gross_loss <= 0:
            pf = 3.0  # Cap at max if no losses
        else:
            pf = min(3.0, gross_profit / gross_loss)

        return pf / 3.0  # Normalize to [0, 1]

    @staticmethod
    def _compute_avg_pnl(trades: list[dict]) -> float:
        """Average P&L per trade (raw, not normalized)."""
        total = sum(t.get("pnl", 0) for t in trades)
        return total / len(trades) if trades else 0.0

    def _compute_recency_score(self, trades: list[dict]) -> float:
        """Exponential-decay weighted win rate.  Recent trades matter more.

        Trade weight = exp(-age / halflife) where age is position in the
        list (most recent = 0, oldest = len-1).
        """
        if not trades:
            return 0.0

        # Sort by trade date (most recent last) if dates are available
        sorted_trades = sorted(trades, key=lambda t: t.get("entry_date", ""))

        total_weight = 0.0
        weighted_wins = 0.0

        for i, trade in enumerate(sorted_trades):
            age = len(sorted_trades) - 1 - i  # Most recent = 0
            w = math.exp(-age / self._recency_halflife)
            total_weight += w
            if trade.get("pnl", 0) > 0:
                weighted_wins += w

        return weighted_wins / total_weight if total_weight > 0 else 0.0
