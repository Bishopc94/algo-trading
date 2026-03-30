"""Signal ranking, aggregation, and trade-queue construction.

This module contains the ``SignalAggregator`` — the "brain" of the system.
It collects signals from all strategies, ranks them by priority, and builds
an execution queue subject to risk and PDT constraints.

The key insight in the ranking logic is the **swing-first philosophy**:
  - Swing trades are FREE under PDT rules (no day-trade slot consumed).
  - Day trades cost 1 of the precious 3 PDT slots.
  - Therefore, swing signals are ALWAYS evaluated first, sorted by
    conviction (highest confidence = highest priority).
  - Day signals are only considered if:
    1. There are remaining PDT slots available, AND
    2. The signal's conviction >= 0.80 (high confidence only)

The execution queue is then built top-down: for each ranked signal, the
system checks cash availability, position count limits, portfolio heat,
and risk approval before including it in the final queue.

Python-specific notes:
  - ``list.sort(key=lambda s: s.conviction, reverse=True)`` sorts the
    list in-place by the ``conviction`` attribute in descending order.
    ``lambda`` is Python's anonymous function syntax — similar to arrow
    functions in JavaScript (``s => s.conviction``).
  - ``getattr(obj, "attr", default)`` safely accesses an attribute with
    a fallback value, similar to ``obj?.attr ?? default`` in other
    languages.
"""

from __future__ import annotations

import pandas as pd

from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import BaseStrategy, HoldType, Signal

logger = get_logger(__name__)


class SignalAggregator:
    """Collects signals from all strategies, ranks them, and builds an
    execution queue subject to risk and PDT constraints.

    This is the central decision-making component.  It doesn't make
    trading decisions itself — it coordinates the strategies, risk manager,
    PDT manager, and position sizer to produce a prioritised list of
    trades to execute.
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        pdt_manager,
        risk_manager,
        position_sizer,
        weighter=None,
    ) -> None:
        self.strategies = strategies
        self.pdt_manager = pdt_manager
        self.risk_manager = risk_manager
        self.position_sizer = position_sizer
        self._weighter = weighter

    # ------------------------------------------------------------------

    def collect_and_rank(
        self,
        candidates: list[str],
        daily_bars_dict: dict[str, pd.DataFrame],
        intraday_bars_dict: dict[str, pd.DataFrame],
        account_equity: float,
        available_cash: float,
        held_symbols: set[str] | None = None,
    ) -> list[dict]:
        """Run all strategies on all candidates and return a prioritised
        execution queue.

        This is the main entry point — called by the TradingBot orchestrator
        during each evaluation window (9:35 AM, 12:00 PM, 3:00 PM).

        Args:
            held_symbols: Set of ticker symbols already held in open positions.
                          Signals for these symbols are skipped to prevent
                          duplicate entries.

        Returns:
            A list of ``{"signal": Signal, "shares": int}`` dicts, ordered
            by execution priority.  The caller iterates this list and
            submits orders for each entry.
        """
        held = held_symbols or set()

        # Recalculate adaptive weights if enough new trades have completed
        if self._weighter is not None:
            self._weighter.maybe_recalculate()

        all_signals: list[Signal] = []

        # ── Step 1: Evaluate every strategy × every candidate ──
        for symbol in candidates:
            # Skip symbols we already hold — prevents duplicate positions
            if symbol in held:
                logger.debug("skip_held_symbol", symbol=symbol)
                continue
            daily = daily_bars_dict.get(symbol)
            intraday = intraday_bars_dict.get(symbol)

            if daily is None or daily.empty:
                continue

            for strategy in self.strategies:
                if not strategy.enabled:
                    continue
                try:
                    sig = strategy.evaluate(symbol, daily, intraday)
                except Exception:
                    logger.exception(
                        "strategy_evaluate_error",
                        strategy=type(strategy).__name__,
                        symbol=symbol,
                    )
                    continue

                if sig is not None:
                    # Apply adaptive strategy weight to conviction
                    if self._weighter is not None:
                        w = self._weighter.get_weight(sig.strategy_name)
                        sig.conviction = max(0.0, min(1.0, sig.conviction * w))
                    all_signals.append(sig)
                    # Log every generated signal to the database for analysis
                    self._log_signal(sig)

        if not all_signals:
            logger.info("no_signals_generated", candidates_evaluated=len(candidates), strategies_count=len(self.strategies))
            return []

        logger.info(
            "signals_collected",
            total=len(all_signals),
            symbols=[s.symbol for s in all_signals],
            strategies=[s.strategy_name for s in all_signals],
            convictions=[round(s.conviction, 2) for s in all_signals],
        )

        # ── Step 2: Separate signals by hold type ──
        swing_signals: list[Signal] = []
        day_signals: list[Signal] = []

        for sig in all_signals:
            if sig.hold_type == HoldType.SWING:
                swing_signals.append(sig)
            else:
                day_signals.append(sig)

        # ── Step 3: Sort swing signals by conviction (best first) ──
        # Swing trades are free (no PDT cost) so they get priority.
        swing_signals.sort(key=lambda s: s.conviction, reverse=True)

        # ── Step 4: Filter day signals by PDT budget and conviction ──
        min_conviction: float = getattr(
            self.pdt_manager.config, "min_conviction_for_day_trade", 0.80
        )
        can_day = self.pdt_manager.can_day_trade()

        qualifying_day: list[Signal] = []
        if can_day:
            # Only day-trade signals with conviction >= 0.80 qualify
            qualifying_day = [
                s for s in day_signals if s.conviction >= min_conviction
            ]
            qualifying_day.sort(key=lambda s: s.conviction, reverse=True)
        else:
            logger.info("day_trade_blocked", reason="PDT limit")

        # ── Step 5: Merge — swing first, then qualifying day signals ──
        ranked = swing_signals + qualifying_day

        # ── Step 6: Build execution queue with resource constraints ──
        execution_queue: list[dict] = []
        remaining_cash = available_cash
        # Start open_count from existing positions to respect max_positions
        open_count = len(held) if held else 0
        max_positions: int = getattr(
            self.risk_manager.config, "max_open_positions", 4
        )

        queued_symbols: set[str] = set()
        for sig in ranked:
            # Stop if we're out of cash or at max positions
            if remaining_cash <= 0 or open_count >= max_positions:
                break

            # Skip duplicate symbols within the same execution queue
            if sig.symbol in queued_symbols:
                logger.debug("skip_duplicate_in_queue", symbol=sig.symbol,
                             strategy=sig.strategy_name)
                continue

            # Calculate position size using the fixed-fractional sizer
            shares = self.position_sizer.calculate_shares(
                sig, account_equity, remaining_cash
            )
            if shares <= 0:
                continue

            # Get open trades for portfolio-heat check
            open_trades = getattr(self.risk_manager, "_database", None)
            open_trades_list: list[dict] = []
            if open_trades is not None:
                try:
                    open_trades_list = open_trades.get_open_trades()
                except Exception:
                    pass

            # Run the full risk approval gate (daily loss, concentration,
            # portfolio heat, affordability)
            approved, reason = self.risk_manager.approve_trade(
                signal=sig,
                shares=shares,
                current_equity=account_equity,
                available_cash=remaining_cash,
                open_positions_count=open_count,
                open_trades=open_trades_list,
            )

            if not approved:
                logger.info(
                    "trade_rejected",
                    symbol=sig.symbol,
                    strategy=sig.strategy_name,
                    reason=reason,
                )
                continue

            # Trade approved — add to execution queue
            execution_queue.append({"signal": sig, "shares": shares})
            remaining_cash -= shares * sig.entry_price
            open_count += 1
            queued_symbols.add(sig.symbol)

            logger.info(
                "trade_queued",
                symbol=sig.symbol,
                strategy=sig.strategy_name,
                shares=shares,
                conviction=sig.conviction,
            )

        # ── Step 7: Return the final queue ──
        logger.info(
            "execution_queue_built",
            total_signals=len(all_signals),
            queued_trades=len(execution_queue),
        )
        return execution_queue

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _log_signal(self, sig: Signal) -> None:
        """Best-effort log of a generated signal to the database.

        This creates a record in the ``signals`` table for post-hoc
        analysis (e.g. "how many signals did each strategy generate?",
        "what was the average conviction of rejected signals?").
        """
        try:
            db = getattr(self.risk_manager, "_database", None)
            if db is not None:
                db.log_signal(
                    symbol=sig.symbol,
                    strategy=sig.strategy_name,
                    conviction=sig.conviction,
                    hold_type=sig.hold_type.value,
                    direction=sig.direction,
                )
        except Exception:
            logger.debug("signal_log_failed", symbol=sig.symbol)
