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

from typing import TYPE_CHECKING

import pandas as pd

from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import BaseStrategy, HoldType, Rejection, Signal

if TYPE_CHECKING:
    from ai_trade.monitoring.decision_logger import DecisionLogger
    from ai_trade.ml.predictor import SignalQualityPredictor
    from ai_trade.risk.dynamic_risk import DynamicRiskController
    from ai_trade.risk.smart_pdt import SmartPDTPlanner

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
        decision_logger: DecisionLogger | None = None,
        ml_predictor: SignalQualityPredictor | None = None,
        dynamic_risk: DynamicRiskController | None = None,
        smart_pdt: SmartPDTPlanner | None = None,
    ) -> None:
        self.strategies = strategies
        self.pdt_manager = pdt_manager
        self.risk_manager = risk_manager
        self.position_sizer = position_sizer
        self._weighter = weighter
        self._dl = decision_logger
        self._ml = ml_predictor
        self._dynamic = dynamic_risk
        self._smart_pdt = smart_pdt
        self._near_misses: list[Rejection] = []
        self.market_context = None

    def set_market_context(self, ctx) -> None:
        """Thread the current MarketContext to every strategy for exit planning."""
        self.market_context = ctx
        for strategy in self.strategies:
            strategy.market_context = ctx

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

        # V2 Phase 7: refresh the dynamic-risk state once per cycle so
        # every signal in the cycle reads a consistent streak + drawdown
        # view (instead of hitting the DB for every candidate).
        if self._dynamic is not None:
            try:
                self._dynamic.refresh_streak()
                self._dynamic.refresh_drawdown(account_equity)
            except Exception:
                logger.exception("dynamic_risk_refresh_failed")

        all_signals: list[Signal] = []
        self._near_misses: list[Rejection] = []

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

                # Drain rejections (whether sig is None or not)
                rejections = strategy.drain_rejections()
                if rejections and self._dl is not None:
                    self._log_rejections(rejections)

                if sig is not None:
                    # Apply adaptive strategy weight to conviction
                    if self._weighter is not None:
                        w = self._weighter.get_weight(sig.strategy_name)
                        sig.conviction = max(0.0, min(1.0, sig.conviction * w))
                    # Apply ML signal-quality prediction (pass-through in
                    # cold start — predictor returns None when no model).
                    self._apply_ml_prediction(sig)
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

        # ── Step 3: Filter and sort swing signals by conviction ──
        # Swing trades are free (no PDT cost) so they get priority, but
        # they still need a minimum conviction to avoid noise trades.
        min_swing_conviction: float = getattr(
            self.risk_manager.config, "min_conviction_for_swing", 0.55
        )
        swing_signals = [
            s for s in swing_signals if s.conviction >= min_swing_conviction
        ]
        swing_signals.sort(key=lambda s: s.conviction, reverse=True)

        # ── Step 4: Filter day signals by PDT budget and conviction ──
        # V2 Phase 8: SmartPDTPlanner produces a dynamic threshold from
        # day-of-week + slots_remaining + recent-EV.  Day signals that
        # miss the dynamic floor get one rescue attempt via day->swing
        # conversion if their strategy's thesis holds overnight.
        can_day = self.pdt_manager.can_day_trade()
        pdt_plan = None
        if self._smart_pdt is not None:
            try:
                pdt_plan = self._smart_pdt.plan_cycle(self.pdt_manager)
                logger.info(
                    "smart_pdt_plan",
                    slots_remaining=pdt_plan.slots_remaining,
                    day=pdt_plan.day_name,
                    stance=pdt_plan.stance,
                    threshold=round(pdt_plan.dynamic_threshold, 3),
                    reasons=pdt_plan.reasons,
                )
            except Exception:
                logger.exception("smart_pdt_plan_failed")
                pdt_plan = None

        if pdt_plan is not None:
            min_conviction = pdt_plan.dynamic_threshold
        else:
            min_conviction = getattr(
                self.pdt_manager.config, "min_conviction_for_day_trade", 0.80
            )

        qualifying_day: list[Signal] = []
        converted_swing: list[Signal] = []
        if can_day:
            for s in day_signals:
                if s.conviction >= min_conviction:
                    qualifying_day.append(s)
                    continue
                # Below the day-trade floor -- try to rescue it as a swing.
                if (
                    self._smart_pdt is not None
                    and self._smart_pdt.is_swing_compatible(s.strategy_name)
                    and s.conviction >= min_swing_conviction
                ):
                    self._smart_pdt.convert_day_to_swing(s)
                    converted_swing.append(s)
                    logger.info(
                        "pdt_day_to_swing",
                        symbol=s.symbol,
                        strategy=s.strategy_name,
                        conviction=round(s.conviction, 3),
                        threshold=round(min_conviction, 3),
                    )
            qualifying_day.sort(key=lambda s: s.conviction, reverse=True)
        else:
            logger.info("day_trade_blocked", reason="PDT limit")
            # Even when day trading is frozen, high-conviction
            # swing-compatible setups still get the rescue path.
            if self._smart_pdt is not None:
                for s in day_signals:
                    if (
                        self._smart_pdt.is_swing_compatible(s.strategy_name)
                        and s.conviction >= min_swing_conviction
                    ):
                        self._smart_pdt.convert_day_to_swing(s)
                        converted_swing.append(s)
                        logger.info(
                            "pdt_day_to_swing_frozen",
                            symbol=s.symbol,
                            strategy=s.strategy_name,
                            conviction=round(s.conviction, 3),
                        )

        # Converted signals join the swing pool (re-sorted below).
        if converted_swing:
            swing_signals.extend(converted_swing)
            swing_signals.sort(key=lambda s: s.conviction, reverse=True)

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

        regime_value: str | None = None
        if self.market_context is not None:
            try:
                regime_value = self.market_context.regime.value
            except Exception:
                regime_value = None

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

            # V2 Phase 7: ask the dynamic-risk controller for a runtime
            # snapshot blending conviction + streak + regime + drawdown
            # into one risk_scale and one concentration cap.
            risk_scale = 1.0
            max_pct_override: float | None = None
            max_positions_override: int | None = None
            dyn_snapshot = None
            if self._dynamic is not None:
                try:
                    dyn_snapshot = self._dynamic.snapshot_for_signal(
                        conviction=sig.conviction,
                        regime=regime_value,
                        current_equity=account_equity,
                    )
                except Exception:
                    logger.exception("dynamic_risk_snapshot_failed", symbol=sig.symbol)
                    dyn_snapshot = None

            if dyn_snapshot is not None:
                if not dyn_snapshot.allow_new_entries:
                    logger.info(
                        "trade_rejected",
                        symbol=sig.symbol,
                        strategy=sig.strategy_name,
                        reason=f"drawdown tier {dyn_snapshot.drawdown_tier}",
                    )
                    break
                if sig.direction == "long" and not dyn_snapshot.allow_new_longs:
                    logger.info(
                        "trade_rejected",
                        symbol=sig.symbol,
                        strategy=sig.strategy_name,
                        reason="regime blocks new longs",
                    )
                    continue
                risk_scale = dyn_snapshot.total_risk_scale
                max_pct_override = dyn_snapshot.max_position_pct
                max_positions_override = dyn_snapshot.max_open_positions
                if dyn_snapshot.reasons:
                    logger.debug(
                        "dynamic_risk_snapshot",
                        symbol=sig.symbol,
                        conviction=round(sig.conviction, 3),
                        risk_scale=round(risk_scale, 3),
                        max_pct=max_pct_override,
                        tier=dyn_snapshot.drawdown_tier,
                        reasons=dyn_snapshot.reasons,
                    )

            if risk_scale <= 0:
                continue

            # Calculate position size using the fixed-fractional sizer
            shares = self.position_sizer.calculate_shares(
                sig,
                account_equity,
                remaining_cash,
                risk_scale=risk_scale,
                max_position_pct_override=max_pct_override,
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
                max_positions_override=max_positions_override,
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

    def get_near_misses(self) -> list[dict]:
        """Return near-miss rejections from the last collect_and_rank call.

        Returns dicts matching the cycle_summary near_misses format:
            {symbol, strategy, reason, miss_pct}
        """
        return [
            {
                "symbol": r.symbol,
                "strategy": r.strategy,
                "reason": r.to_reasoning(),
                "miss_pct": r.miss_pct,
            }
            for r in self._near_misses
        ]

    def _apply_ml_prediction(self, sig: Signal) -> None:
        """Blend ML probability into the signal conviction (pass-through cold start).

        The predictor returns None when no model is loaded, in which
        case the rule conviction is left untouched and we log nothing
        — no audit spam during bootstrap.
        """
        if self._ml is None:
            return
        try:
            ml_prob = self._ml.predict(sig, self.market_context)
        except Exception:
            logger.exception("ml_predict_failed", symbol=sig.symbol)
            return

        if ml_prob is None:
            return

        rule_conviction = sig.conviction
        blended, trace = self._ml.apply_to_conviction(rule_conviction, ml_prob)
        sig.conviction = blended
        if sig.metadata is None:
            sig.metadata = {}
        sig.metadata["ml_trace"] = trace

        if self._dl is not None:
            try:
                self._dl.log_evaluate(
                    symbol=sig.symbol,
                    strategy=sig.strategy_name,
                    action="ml_predict",
                    reasoning=(
                        f"ml={ml_prob:.3f} rule={rule_conviction:.3f} "
                        f"w={trace['blend_weight']} blended={blended:.3f}"
                    ),
                    factors=trace,
                )
            except Exception:
                logger.exception("ml_predict_log_failed", symbol=sig.symbol)

    def _log_rejections(self, rejections: list[Rejection]) -> None:
        """Log rejections to DecisionLogger and collect near-misses."""
        for r in rejections:
            # Only log the first rejection per strategy×symbol (the one that
            # actually killed the evaluation — subsequent ones didn't run).
            self._dl.log_evaluate(
                symbol=r.symbol,
                strategy=r.strategy,
                action="near_miss" if r.is_near_miss else "reject",
                reasoning=r.to_reasoning(),
                factors={
                    "filter": r.filter_name,
                    "actual": r.actual,
                    "threshold": r.threshold,
                    "direction": r.direction,
                    "miss_pct": r.miss_pct,
                },
            )
            if r.is_near_miss:
                self._near_misses.append(r)

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
