"""Portfolio-level risk manager.

WHAT THIS MODULE DOES:
    Acts as a centralized gatekeeper that must approve every trade before
    it is submitted to the broker.  It enforces three independent safety
    checks plus a basic affordability check.

WHY IT EXISTS:
    Individual position sizing (see position_sizer.py) limits risk per
    trade, but you also need PORTFOLIO-level controls.  Without them, you
    could have 10 perfectly-sized trades that collectively put 50% of your
    account at risk.  This module prevents that.

THE THREE RISK CHECKS:
    1. Daily Loss Limit — If the account has already lost X% today, stop
       all new trading.  This prevents "revenge trading" (emotionally
       chasing losses) and limits damage on bad days.

    2. Position Concentration — Limits the total number of open positions.
       Even if each position is small, too many simultaneous positions
       become hard to monitor and manage.

    3. Portfolio Heat — The total dollar amount at risk across ALL open
       trades (sum of entry-to-stop distances × shares), expressed as a
       percentage of equity.  This is the most important aggregate risk
       metric: it answers "if every open trade hits its stop-loss right
       now, what percentage of my account do I lose?"

KEY DESIGN DECISIONS:
    - Each check returns a tuple of (bool, str) — the bool says pass/fail,
      and the string explains why.  This makes it easy to log the specific
      reason a trade was rejected.
    - The `approve_trade` method runs all checks sequentially and
      short-circuits (stops checking) on the first failure.
    - Starting equity must be set once at market open each day so the
      daily-loss calculation has a reference point.
"""

from __future__ import annotations

from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import Signal

logger = get_logger(__name__)


class RiskManager:
    """Centralised gate-keeper that must approve every trade before execution.

    Enforces daily loss limits, position concentration, and portfolio heat.

    TRADING CONCEPT — Risk Gates:
        Before any order is sent to the broker, it must pass through a
        series of "gates."  If ANY gate says no, the trade is rejected.
        This is a defensive programming pattern common in trading systems
        to prevent catastrophic losses from bugs, bad data, or unusual
        market conditions.
    """

    def __init__(self, config, database, dynamic_controller=None) -> None:
        self.config = config
        self._database = database

        # PYTHON PATTERN — float | None (union type):
        # This variable starts as None (no value set yet) and later gets
        # assigned a float.  The `|` syntax is a union type hint meaning
        # "this can be either a float or None."  It requires Python 3.10+
        # or `from __future__ import annotations`.
        self._starting_equity: float | None = None

        # V2 Phase 7: optional DynamicRiskController.  When wired, the
        # approve_trade path consults it for the tiered drawdown breaker
        # and the per-regime max-positions override so Phase 7's runtime
        # risk posture composes with the portfolio gates.
        self._dynamic = dynamic_controller

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_starting_equity(self, equity: float) -> None:
        """Cache today's opening equity for daily-loss calculations.

        This must be called once at market open each day.  All subsequent
        calls to `check_daily_loss_limit` compare the current equity
        against this reference value.  If a DynamicRiskController is
        wired, it shares the same reference so tiered drawdown
        calculations stay in sync with the hard daily loss limit.
        """
        self._starting_equity = equity
        logger.info("starting_equity_set", equity=equity)
        if self._dynamic is not None:
            try:
                self._dynamic.set_starting_equity(equity)
            except Exception:
                logger.exception("dynamic_risk_set_starting_equity_failed")

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_daily_loss_limit(
        self, current_equity: float
    ) -> tuple[bool, str]:
        """Return ``(False, reason)`` if the daily loss limit has been hit.

        TRADING CONCEPT — Daily Loss Limit:
            A hard cap on how much the account can lose in a single day.
            For example, a 5% daily limit on a $10,000 account means: if
            equity drops below $9,500, no new trades are allowed for the
            rest of the day.  This is the most important "circuit breaker"
            in the system.

        PYTHON PATTERN — tuple[bool, str] return:
            Python functions can return multiple values as a tuple.  The
            caller typically unpacks them with: `ok, reason = check_...()`
            This is idiomatic Python for functions that need to return both
            a success/failure flag and additional context.
        """
        # If starting equity was never set (e.g. first run), we allow
        # trading — we can't calculate a loss without a reference point.
        if self._starting_equity is None:
            return True, "no starting equity set"

        # Calculate how much of today's starting equity has been lost.
        # A positive loss_pct means the account is down.
        loss_pct = (
            (self._starting_equity - current_equity) / self._starting_equity
        )

        # Default daily loss limit is 5% (0.05).
        limit: float = getattr(self.config, "daily_loss_limit_pct", 0.05)

        if loss_pct > limit:
            # Format as percentage for human-readable logging.
            # The `:.2%` format spec multiplies by 100 and adds a % sign.
            msg = (
                f"daily loss limit hit: {loss_pct:.2%} "
                f"(limit {limit:.2%})"
            )
            logger.warning("daily_loss_limit", loss_pct=loss_pct, limit=limit)
            return False, msg

        return True, "ok"

    def check_concentration(
        self,
        current_positions_count: int,
        max_positions_override: int | None = None,
    ) -> tuple[bool, str]:
        """Return ``(False, reason)`` if max open positions reached.

        TRADING CONCEPT — Position Concentration:
            Limiting the number of simultaneous open positions forces
            diversification and keeps the portfolio manageable.  A common
            limit for small accounts is 3-5 positions.

        V2 Phase 7: ``max_positions_override`` comes from the
        DynamicRiskController's regime bonus (fewer positions in bear,
        more in strong bull).
        """
        max_pos: int = (
            max_positions_override
            if max_positions_override is not None
            else getattr(self.config, "max_open_positions", 4)
        )
        if current_positions_count >= max_pos:
            msg = f"max positions reached: {current_positions_count}/{max_pos}"
            logger.warning("max_positions", count=current_positions_count)
            return False, msg
        return True, "ok"

    def check_drawdown_breaker(self, current_equity: float) -> tuple[bool, str]:
        """Tiered drawdown circuit breaker (Phase 7).

        Consults the DynamicRiskController's most recent drawdown tier.
        Tier 0-1 allow new entries (tier 1 still halves position size
        via the risk scale); tier 2-3 halt new entries entirely.  This
        is STRICTER than ``check_daily_loss_limit`` — it fires at 5%
        drawdown rather than the config default, as specified in the
        V2 brief.
        """
        if self._dynamic is None:
            return True, "ok"
        try:
            tier = self._dynamic.refresh_drawdown(current_equity)
        except Exception:
            logger.exception("drawdown_refresh_failed")
            return True, "ok"
        if not tier.allow_new_entries:
            return False, f"drawdown circuit breaker: {tier.reason}"
        return True, "ok"

    def compute_portfolio_heat(
        self,
        open_trades: list[dict],
        current_equity: float,
    ) -> float:
        """Total open-trade risk as a fraction of equity (``0.06`` == 6%).

        Single source of truth for portfolio heat — both the entry gate
        (``check_portfolio_heat``) and the console/email display call this
        so the two readings can never drift. Returns ``0.0`` when equity is
        non-positive.

        TRADING CONCEPT — Portfolio Heat:
            "Heat" is the total dollar amount at risk across all open
            positions, expressed as a fraction of total equity.

            For each open trade:
                trade_risk = |entry_price - stop_loss| × shares

            Portfolio heat = sum(all trade_risk) / current_equity

            The distance is read directionally off the trade's ``side``
            (long: entry - stop, short: stop - entry).  A negative reading
            means the stop is on the profit side of entry — a trailing
            stop that locked in gains, or a short mishandled as a long.
            It is logged as ``trade_risk_sign_mismatch`` and still counted
            at its positive magnitude so it can never reduce total heat.

            For example, if you have 3 trades each risking $200 on a
            $10,000 account, your heat is $600 / $10,000 = 0.06.

        Args:
            open_trades:    List of trade dictionaries from the database,
                            each containing "entry_price", "stop_loss",
                            and "shares" keys.
            current_equity: Current total account value.
        """
        if current_equity <= 0:
            return 0.0

        total_risk = 0.0
        for trade in open_trades:
            # Use .get() with fallback to 0 for safety — some trades might
            # have missing/null fields if they were inserted with incomplete data.
            entry = trade.get("entry_price", 0) or 0
            stop = trade.get("stop_loss", 0) or 0
            shares = trade.get("shares", 0) or 0
            side = (trade.get("side") or "long").lower()

            # Directional risk-per-share: for a long the protective stop sits
            # BELOW entry (entry - stop > 0); for a short it sits ABOVE
            # (stop - entry > 0). A negative value means the stop is on the
            # profit side of the entry.
            directional = (stop - entry) if side == "short" else (entry - stop)

            # Risk always enters the heat sum as a positive magnitude — a
            # negative directional reading must never silently reduce total
            # heat (that would wave a trade through the cap). See §3.3.
            total_risk += abs(directional) * shares

            if directional < 0:
                # Two benign-to-malign causes, distinguishable from the values:
                #   - a winning long whose trailing stop has ratcheted above
                #     entry to lock in profit (JRSH 2026-06-15: long, entry
                #     3.86, trailed stop 4.17 after price hit 4.44) — common,
                #     not corruption; the position simply has no downside left
                #   - a short mislabeled/handled as long, where the stop above
                #     entry is real risk the unsigned formula would hide
                # Either way we surface it rather than trusting the sign.
                logger.warning(
                    "trade_risk_sign_mismatch",
                    symbol=trade.get("symbol"),
                    side=side,
                    entry_price=entry,
                    stop_loss=stop,
                    shares=shares,
                    directional_risk_per_share=round(directional, 4),
                    counted_as=round(abs(directional) * shares, 2),
                )

        return total_risk / current_equity

    def check_portfolio_heat(
        self,
        open_trades: list[dict],
        current_equity: float,
    ) -> tuple[bool, str]:
        """Return ``(False, reason)`` if total risk of open trades exceeds
        the portfolio heat limit.

        Heat itself is computed by ``compute_portfolio_heat`` (the shared
        definition); this method only applies the cap. A typical limit is
        6-10%. If heat exceeds it, no new trades are allowed until existing
        trades are closed or their stops are tightened.
        """
        max_heat: float = getattr(self.config, "max_portfolio_heat_pct", 0.06)

        # Can't calculate heat with zero equity (would cause division by zero).
        if current_equity <= 0:
            return False, "equity is zero"

        heat_pct = self.compute_portfolio_heat(open_trades, current_equity)
        if heat_pct > max_heat:
            msg = (
                f"portfolio heat {heat_pct:.2%} exceeds "
                f"limit {max_heat:.2%}"
            )
            logger.warning("portfolio_heat", heat_pct=heat_pct, limit=max_heat)
            return False, msg

        return True, "ok"

    # ------------------------------------------------------------------
    # Combined approval
    # ------------------------------------------------------------------

    def approve_trade(
        self,
        signal: Signal,
        shares: int,
        current_equity: float,
        available_cash: float,
        open_positions_count: int,
        open_trades: list[dict],
        max_positions_override: int | None = None,
    ) -> tuple[bool, str]:
        """Run all risk checks and return ``(True, 'approved')`` or
        ``(False, reason)``.

        This is the single entry point for trade approval.  The checks
        are run in order and short-circuit on the first failure — if the
        daily loss limit is hit, we don't bother checking concentration
        or portfolio heat.

        Args:
            signal:               The trade signal to evaluate.
            shares:               Number of shares the position sizer wants to buy.
            current_equity:       Current total account value.
            available_cash:       Cash available for new trades.
            open_positions_count: How many positions are currently open.
            open_trades:          List of open trade dicts from the database.

        Returns:
            A tuple of (approved: bool, reason: str).
        """

        # Check 1: Daily loss limit
        ok, reason = self.check_daily_loss_limit(current_equity)
        if not ok:
            return False, reason

        # Check 1b: V2 Phase 7 tiered drawdown breaker (fires before the
        # hard daily-loss cap so we halve size at -3% and halt at -5%).
        ok, reason = self.check_drawdown_breaker(current_equity)
        if not ok:
            return False, reason

        # Check 2: Position concentration (regime-adjusted when the
        # dynamic controller provides an override).
        ok, reason = self.check_concentration(
            open_positions_count,
            max_positions_override=max_positions_override,
        )
        if not ok:
            return False, reason

        # Check 3: Portfolio heat
        ok, reason = self.check_portfolio_heat(open_trades, current_equity)
        if not ok:
            return False, reason

        # Check 4: Basic affordability — can we actually pay for the shares?
        cost = shares * signal.entry_price
        if cost > available_cash:
            msg = f"insufficient cash: need ${cost:.2f}, have ${available_cash:.2f}"
            return False, msg

        logger.info(
            "trade_approved",
            symbol=signal.symbol,
            shares=shares,
            cost=cost,
        )
        return True, "approved"
