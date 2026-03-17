"""Fixed-fractional position sizing.

WHAT THIS MODULE DOES:
    Given a trade signal (with an entry price and a stop-loss price), this
    module calculates how many shares to buy so that the maximum possible
    loss on the trade is a fixed percentage of account equity.

WHY IT EXISTS:
    Position sizing is the single most important risk management tool in
    trading.  Without it, you might bet your entire account on one trade.
    The "fixed-fractional" method ensures that every trade risks the same
    dollar amount relative to your equity, regardless of share price or
    stop distance.

HOW THE MATH WORKS (example):
    Account equity   = $10,000
    Max risk per trade = 2% → risk budget = $200
    Entry price      = $50.00
    Stop-loss price  = $48.00 → risk per share = $2.00
    Shares from risk = floor($200 / $2) = 100 shares
    Dollar value     = 100 × $50 = $5,000

    Then we apply additional constraints:
    - Concentration limit: don't put more than 30% of equity in one stock
    - Cash limit: don't spend more cash than you actually have

KEY DESIGN DECISIONS:
    - Uses math.floor() everywhere to always round DOWN.  We never want to
      buy more shares than the risk budget allows.
    - If all constraints result in 0 shares but you can afford at least 1,
      we buy 1 share.  This prevents the bot from being completely idle on
      small accounts.
"""

# "from __future__ import annotations" enables modern type-hint syntax
# (e.g. `float | None`) on all Python 3.7+ versions by deferring
# evaluation of type annotations until they're explicitly inspected.
from __future__ import annotations

import math

from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import Signal

logger = get_logger(__name__)


class PositionSizer:
    """Calculate the number of shares for a trade using fixed-fractional
    risk sizing, subject to concentration and cash constraints.

    TRADING CONCEPT — Fixed-Fractional Sizing:
        You risk a fixed percentage (e.g. 2%) of your total account equity
        on every trade.  The number of shares is determined by dividing
        the dollar risk budget by the per-share risk (distance from entry
        to stop-loss).  This means:
        - Tight stops → more shares (small risk per share)
        - Wide stops → fewer shares (large risk per share)
        - The dollar amount at risk stays constant either way.
    """

    def __init__(self, config) -> None:
        # `config` holds parameters like max_risk_per_trade_pct (e.g. 0.02
        # for 2%) and max_position_pct (e.g. 0.30 for 30%).
        self.config = config

    def calculate_shares(
        self,
        signal: Signal,
        account_equity: float,
        available_cash: float,
    ) -> int:
        """Return the number of whole shares to buy (>= 0).

        The calculation follows a 5-step pipeline, each step potentially
        reducing the share count:

        1. Dollar risk budget    — How much money can we afford to lose?
        2. Per-share risk        — How much do we lose per share if stopped out?
        3. Shares from risk      — Budget / per-share risk
        4. Concentration cap     — Don't exceed X% of equity in one position
        5. Cash cap              — Don't spend more than available cash

        Args:
            signal:          The trade signal containing entry_price,
                             stop_loss_price, and symbol.
            account_equity:  Total account value (cash + positions).
            available_cash:  Cash available for new trades right now.

        Returns:
            An integer number of shares, always >= 0.
        """

        # PYTHON PATTERN — getattr with default:
        # Safely reads config attributes, falling back to sensible defaults
        # if the attribute is missing.  This avoids crashes from incomplete
        # config files.
        max_risk_pct: float = getattr(
            self.config, "max_risk_per_trade_pct", 0.02  # Default: risk 2% per trade
        )
        max_position_pct: float = getattr(
            self.config, "max_position_pct", 0.30  # Default: max 30% in one stock
        )

        # ── Step 1: Dollar risk budget ──────────────────────────
        # This is the maximum dollar amount we're willing to lose on this
        # single trade.  For a $10,000 account at 2%, that's $200.
        risk_amount = account_equity * max_risk_pct

        # ── Step 2: Per-share risk ──────────────────────────────
        # The distance between entry price and stop-loss price.  If we buy
        # at $50 with a stop at $48, the per-share risk is $2.  We use
        # abs() because stop_loss could theoretically be above entry for
        # a short trade (though this bot currently only goes long).
        risk_per_share = abs(signal.entry_price - signal.stop_loss_price)

        # Guard against division by zero: if entry == stop, the signal is
        # invalid and we refuse to size it.
        if risk_per_share == 0:
            logger.warning(
                "zero_risk_per_share",
                symbol=signal.symbol,
                entry=signal.entry_price,
                stop=signal.stop_loss_price,
            )
            return 0

        # ── Step 3: Shares from risk budget ─────────────────────
        # Core fixed-fractional formula: risk_budget / risk_per_share.
        # math.floor() rounds DOWN to a whole number — we never buy a
        # fractional share that would push us over the risk budget.
        shares = math.floor(risk_amount / risk_per_share)

        # ── Step 4: Enforce concentration limit ─────────────────
        # TRADING CONCEPT — Concentration Risk:
        # Even if the per-trade risk is small, putting too much capital
        # into one stock exposes us to gap risk (the stock opens 20% lower
        # than expected and blows through our stop).  The concentration
        # limit caps the total dollar value of any single position.
        max_dollar_value = account_equity * max_position_pct
        if shares * signal.entry_price > max_dollar_value:
            shares = math.floor(max_dollar_value / signal.entry_price)

        # ── Step 5: Enforce available cash ──────────────────────
        # Can't spend money we don't have.  `min()` picks the smaller of
        # our calculated shares and what cash actually allows.
        if signal.entry_price > 0:
            shares = min(shares, math.floor(available_cash / signal.entry_price))

        # ── Floor at 0, but allow at least 1 if affordable ─────
        # max(0, shares) prevents negative values.  Then, if shares ended
        # up at 0 but we can actually afford 1 share, we buy 1.  This
        # keeps small accounts active instead of completely idle.
        shares = max(0, shares)
        if shares == 0 and available_cash >= signal.entry_price > 0:
            shares = 1

        logger.debug(
            "position_sized",
            symbol=signal.symbol,
            shares=shares,
            risk_amount=risk_amount,
            risk_per_share=risk_per_share,
            dollar_value=shares * signal.entry_price,
        )

        return shares
