"""Performance tracking — P&L, Sharpe, win rate, drawdown calculations.

WHAT THIS MODULE DOES:
    Computes aggregate trading performance metrics from the trade history
    stored in SQLite.  Produces a metrics dictionary and a human-readable
    daily summary string.

WHY IT EXISTS:
    You can't improve what you don't measure.  This module answers:
    - Are we profitable overall?  (total P&L)
    - Are we consistently profitable?  (win rate, profit factor)
    - Are our risk-adjusted returns good?  (Sharpe ratio)
    - What's the worst equity decline we've experienced?  (max drawdown)
    - How does each strategy perform individually?

KEY TRADING METRICS EXPLAINED:

    Win Rate:
        Percentage of closed trades that made money.  A 40-60% win rate
        is typical for trend-following strategies.  Win rate alone doesn't
        tell you much — a 30% win rate is fine if the average winner is
        3x the average loser.

    Profit Factor:
        gross_wins / gross_losses.  A profit factor > 1.0 means you're
        making more than you're losing.  > 2.0 is excellent.  A profit
        factor of infinity means you have no losing trades (unlikely to
        last).

    Sharpe Ratio:
        (mean_return / std_dev_of_returns) × sqrt(252).  Measures
        risk-adjusted returns — how much return you get per unit of
        volatility.  Higher is better:
        - < 0:   losing money
        - 0-1:   poor to mediocre
        - 1-2:   good
        - > 2:   excellent
        The sqrt(252) factor annualizes the ratio (252 trading days/year).

    Max Drawdown:
        The largest peak-to-trough decline in account equity, expressed
        as a percentage.  A 10% max drawdown means at some point your
        account fell 10% from its highest value.  Lower is better —
        drawdowns are psychologically painful and hard to recover from
        (a 50% loss requires a 100% gain to break even).

KEY DESIGN DECISIONS:
    - Metrics are computed from the database on-demand, not cached.  This
      ensures they're always up-to-date, and the computation is fast enough
      for the modest data volumes of a single account.
    - Max drawdown uses daily equity snapshots (not trade-by-trade), which
      is standard in the industry.
    - The Sharpe ratio computation is simplified: it uses per-trade P&L
      rather than daily returns.  This is an approximation but sufficient
      for monitoring purposes.
"""

from __future__ import annotations

import math
from datetime import datetime

from ai_trade.monitoring.database import Database
from ai_trade.monitoring.logger import get_logger
from ai_trade.monitoring import console as con

log = get_logger(__name__)


class PerformanceTracker:
    """Computes and logs trading performance metrics.

    All metrics are derived from the trade and snapshot data in the
    database.  No state is cached between calls.
    """

    def __init__(self, database: Database):
        self._db = database

    def calculate_metrics(self) -> dict:
        """Calculate aggregate performance metrics from closed trades.

        Only considers trades where status="closed" AND pnl is not None
        (i.e., trades where we know the final P&L).

        Returns:
            Dictionary with keys: total_trades, win_rate, total_pnl,
            avg_pnl, profit_factor, avg_win, avg_loss, max_win, max_loss,
            sharpe_ratio, max_drawdown_pct.

            Returns zeroed-out metrics if there are no closed trades.
        """
        trades = self._db.get_all_trades()

        # PYTHON PATTERN — list comprehension with multiple conditions:
        # Filters the list to only include trades that are both "closed"
        # AND have a non-None P&L value.
        closed = [t for t in trades if t["status"] == "closed" and t["pnl"] is not None]

        # Return zeros if no closed trades exist yet.
        if not closed:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "max_win": 0.0,
                "max_loss": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0,
            }

        # Separate winners and losers.
        # Trades with exactly $0 P&L are counted as losses (conservative).
        wins = [t for t in closed if t["pnl"] > 0]
        losses = [t for t in closed if t["pnl"] <= 0]

        # Aggregate P&L calculations.
        total_pnl = sum(t["pnl"] for t in closed)
        gross_wins = sum(t["pnl"] for t in wins) if wins else 0.0
        # abs() converts gross losses to a positive number for display
        # and for the profit factor calculation.
        gross_losses = abs(sum(t["pnl"] for t in losses)) if losses else 0.0

        avg_win = gross_wins / len(wins) if wins else 0.0
        avg_loss = gross_losses / len(losses) if losses else 0.0

        # ── Sharpe Ratio calculation ─────────────────────────
        # Sharpe = (mean_return / std_dev) × sqrt(annualization_factor)
        #
        # Step 1: Calculate mean P&L per trade.
        pnl_values = [t["pnl"] for t in closed]
        mean_pnl = total_pnl / len(closed)

        # Step 2: Calculate variance (average of squared deviations from mean).
        # This is the population variance (dividing by N, not N-1).
        variance = sum((p - mean_pnl) ** 2 for p in pnl_values) / len(closed)

        # Step 3: Standard deviation = square root of variance.
        std_pnl = math.sqrt(variance) if variance > 0 else 0.0

        # Step 4: Annualize.  sqrt(252) converts per-trade Sharpe to
        # approximate annualized Sharpe (assuming ~252 trading days/year).
        sharpe = (mean_pnl / std_pnl * math.sqrt(252)) if std_pnl > 0 else 0.0

        # Max drawdown from daily equity snapshots (not from trade P&Ls).
        max_dd = self._calculate_max_drawdown()

        metrics = {
            "total_trades": len(closed),
            "win_rate": len(wins) / len(closed) if closed else 0.0,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(mean_pnl, 2),
            # TRADING CONCEPT — Profit Factor:
            # gross_wins / gross_losses.  If gross_losses is 0 (no losers),
            # profit factor is technically infinite.  float("inf") represents
            # positive infinity in Python.
            "profit_factor": round(gross_wins / gross_losses, 2) if gross_losses > 0 else float("inf"),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "max_win": round(max(pnl_values), 2),    # Best single trade
            "max_loss": round(min(pnl_values), 2),    # Worst single trade (negative number)
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 2),
        }
        return metrics

    def _calculate_max_drawdown(self) -> float:
        """Max drawdown % from daily equity snapshots.

        TRADING CONCEPT — Max Drawdown:
            Walk through the equity curve chronologically, tracking the
            highest equity seen so far ("peak").  At each point, the
            drawdown is: (peak - current) / peak.  The maximum of all
            these drawdowns is the "max drawdown."

            Example:
                Day 1: $10,000 (peak = $10,000, dd = 0%)
                Day 2: $10,500 (peak = $10,500, dd = 0%)
                Day 3: $9,800  (peak = $10,500, dd = 6.7%)  ← max drawdown
                Day 4: $10,200 (peak = $10,500, dd = 2.9%)

        Returns:
            Max drawdown as a percentage (e.g., 6.7 means 6.7%).
        """
        snapshots = self._db.get_snapshots(limit=365)
        if len(snapshots) < 2:
            return 0.0

        # Snapshots come from the DB newest-first, so reverse for
        # chronological order (oldest first) to walk the equity curve.
        equities = [s["equity"] for s in reversed(snapshots)]
        peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq  # New high-water mark
            # Calculate current drawdown from peak.
            dd = (peak - eq) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd * 100  # Convert decimal to percentage

    def strategy_performance(self, strategy_name: str) -> dict:
        """Metrics filtered to a single strategy.

        Useful for comparing which strategies are working and which
        should be disabled or re-tuned.

        Args:
            strategy_name: Name of the strategy to filter by (must match
                           the "strategy" field in the trades table).

        Returns:
            Dictionary with total_trades, win_rate, and total_pnl.
        """
        trades = self._db.get_all_trades()
        closed = [
            t for t in trades
            if t["status"] == "closed" and t["pnl"] is not None and t["strategy"] == strategy_name
        ]
        if not closed:
            return {"total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0}

        wins = [t for t in closed if t["pnl"] > 0]
        total_pnl = sum(t["pnl"] for t in closed)
        return {
            "total_trades": len(closed),
            "win_rate": round(len(wins) / len(closed), 2),
            "total_pnl": round(total_pnl, 2),
        }

    def daily_summary(self, equity: float, cash: float, open_positions: int, day_trades_used: int) -> str:
        """Generate end-of-day summary string.

        Combines current account state with historical performance metrics
        into a formatted report that's logged and optionally displayed.

        Args:
            equity:          Current total account value.
            cash:            Current available cash.
            open_positions:  Number of currently open positions.
            day_trades_used: Day trades used in the rolling 5-day window.

        Returns:
            A multi-line formatted string suitable for logging or display.
        """
        metrics = self.calculate_metrics()
        today = datetime.now().strftime("%Y-%m-%d")

        # Log the summary metrics as structured key-value pairs for
        # machine-parseable analysis.
        log.info("daily_summary", equity=equity, cash=cash, **metrics)

        return con.daily_summary(
            today=today, equity=equity, cash=cash,
            open_positions=open_positions,
            day_trades_used=day_trades_used,
            metrics=metrics,
        )
