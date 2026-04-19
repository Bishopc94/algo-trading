"""Decision audit trail for the V2 self-learning trading system.

Every decision the bot makes -- scanning, evaluating, signaling, ranking,
sizing, approving, rejecting, executing, exiting -- is logged here with
human-readable reasoning and structured factors for ML training.

Usage:
    dl = DecisionLogger(db)

    # During scanning
    dl.log_scan("AAPL", "momentum_scanner", "consider",
                "price $185, RVOL 2.3x, gap +3.2% -- passes momentum scanner")

    # During strategy evaluation
    dl.log_evaluate("AAPL", "momentum", "reject",
                    "RSI 82 exceeds 80 max -- too overbought",
                    factors={"rsi": 82, "threshold": 80, "miss_by": 2})

    # Near-miss logging
    dl.log_near_miss("AAPL", "pullback", "RSI 54 (needs <53)",
                     miss_pct=1.9)

    # Signal generation
    dl.log_signal("AAPL", "momentum", 0.78,
                  "entry $185, stop $181.50, target $195.50, R:R 1:2.9",
                  factors={...})

    # Batch flush at end of cycle
    dl.flush()
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from ai_trade.monitoring.database import Database
from ai_trade.monitoring.logger import get_logger

log = get_logger(__name__)


class DecisionLogger:
    """Accumulates decisions during a scan cycle, then batch-inserts them.

    Buffering decisions and flushing once per cycle (instead of one INSERT
    per decision) reduces SQLite write pressure from ~200 writes/cycle to 1.
    """

    def __init__(self, db: Database):
        self._db = db
        self._buffer: list[dict] = []

    def _add(
        self,
        decision_type: str,
        symbol: str,
        strategy: str | None,
        action: str,
        conviction: float | None = None,
        reasoning: str = "",
        factors: dict | None = None,
    ) -> None:
        self._buffer.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision_type": decision_type,
            "symbol": symbol,
            "strategy": strategy or "",
            "action": action,
            "conviction": conviction,
            "reasoning": reasoning,
            "factors": json.dumps(factors) if factors else None,
        })

    # ── Scan phase ──

    def log_scan(
        self, symbol: str, scanner: str, action: str, reasoning: str = "",
        factors: dict | None = None,
    ) -> None:
        """Log a scanning decision (consider/reject a candidate)."""
        self._add("scan", symbol, scanner, action, reasoning=reasoning, factors=factors)

    # ── Evaluate phase ──

    def log_evaluate(
        self, symbol: str, strategy: str, action: str, reasoning: str = "",
        conviction: float | None = None, factors: dict | None = None,
    ) -> None:
        """Log a strategy evaluation (signal/reject with reason)."""
        self._add(
            "evaluate", symbol, strategy, action,
            conviction=conviction, reasoning=reasoning, factors=factors,
        )

    def log_near_miss(
        self, symbol: str, strategy: str, reasoning: str,
        miss_pct: float | None = None, factors: dict | None = None,
    ) -> None:
        """Log a near-miss: strategy almost fired but missed by a small margin."""
        f = factors or {}
        if miss_pct is not None:
            f["miss_pct"] = miss_pct
        self._add(
            "evaluate", symbol, strategy, "near_miss",
            reasoning=reasoning, factors=f,
        )

    # ── Signal phase ──

    def log_signal(
        self, symbol: str, strategy: str, conviction: float,
        reasoning: str = "", factors: dict | None = None,
    ) -> None:
        """Log a generated signal."""
        self._add(
            "signal", symbol, strategy, "signal",
            conviction=conviction, reasoning=reasoning, factors=factors,
        )

    # ── Ranking phase ──

    def log_rank(
        self, symbol: str, strategy: str, conviction: float,
        rank: int, reasoning: str = "", factors: dict | None = None,
    ) -> None:
        """Log a signal's rank position after weighting and modifiers."""
        f = factors or {}
        f["rank"] = rank
        self._add(
            "rank", symbol, strategy, "ranked",
            conviction=conviction, reasoning=reasoning, factors=f,
        )

    # ── Sizing phase ──

    def log_size(
        self, symbol: str, strategy: str, shares: int,
        cost: float, risk_amount: float, reasoning: str = "",
        factors: dict | None = None,
    ) -> None:
        """Log position sizing calculation."""
        f = factors or {}
        f.update({"shares": shares, "cost": cost, "risk_amount": risk_amount})
        self._add(
            "size", symbol, strategy, "sized",
            reasoning=reasoning, factors=f,
        )

    # ── Approval/rejection phase ──

    def log_approve(
        self, symbol: str, strategy: str, conviction: float,
        reasoning: str = "", factors: dict | None = None,
    ) -> None:
        """Log trade approval by risk manager."""
        self._add(
            "approve", symbol, strategy, "approve",
            conviction=conviction, reasoning=reasoning, factors=factors,
        )

    def log_reject(
        self, symbol: str, strategy: str, reason: str,
        conviction: float | None = None, factors: dict | None = None,
    ) -> None:
        """Log trade rejection by risk manager or other gate."""
        self._add(
            "approve", symbol, strategy, "reject",
            conviction=conviction, reasoning=reason, factors=factors,
        )

    # ── Execution phase ──

    def log_execute(
        self, symbol: str, strategy: str, conviction: float,
        reasoning: str = "", factors: dict | None = None,
    ) -> None:
        """Log order submission."""
        self._add(
            "execute", symbol, strategy, "execute",
            conviction=conviction, reasoning=reasoning, factors=factors,
        )

    # ── Exit phase ──

    def log_exit(
        self, symbol: str, strategy: str, reasoning: str = "",
        factors: dict | None = None,
    ) -> None:
        """Log a trade exit."""
        self._add(
            "exit", symbol, strategy, "exit",
            reasoning=reasoning, factors=factors,
        )

    # ── Review phase ──

    def log_review(
        self, symbol: str, strategy: str, reasoning: str = "",
        factors: dict | None = None,
    ) -> None:
        """Log a post-trade review insight."""
        self._add(
            "review", symbol, strategy, "review",
            reasoning=reasoning, factors=factors,
        )

    # ── Buffer management ──

    def flush(self) -> int:
        """Write all buffered decisions to the database in one transaction.

        Returns the number of decisions flushed.
        """
        count = len(self._buffer)
        if count == 0:
            return 0
        try:
            self._db.log_decisions_batch(self._buffer)
            log.debug("decisions_flushed", count=count)
        except Exception as e:
            log.error("decisions_flush_failed", count=count, error=str(e))
            # Fall back to individual inserts
            for d in self._buffer:
                try:
                    self._db.log_decision(**d)
                except Exception:
                    pass
        self._buffer.clear()
        return count

    @property
    def pending_count(self) -> int:
        """Number of decisions buffered but not yet flushed."""
        return len(self._buffer)
