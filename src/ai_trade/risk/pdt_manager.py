"""Pattern Day Trade (PDT) tracker for cash / small accounts.

WHAT THIS MODULE DOES:
    Tracks how many "day trades" (buying and selling the same stock on the
    same calendar day) the account has made within a rolling 5-business-day
    window, and decides whether the bot is allowed to make another one.

WHY IT EXISTS:
    The SEC's Pattern Day Trade (PDT) rule applies to margin accounts with
    less than $25,000 in equity.  If you make 4 or more day trades within
    any rolling 5-business-day period, the broker will flag your account as
    a "pattern day trader" and may freeze it for 90 days.  This module
    keeps us safely below that threshold.

KEY DESIGN DECISIONS:
    - We default to a maximum of 3 day trades per 5-day window (the legal
      limit before the PDT flag triggers is 4, but we use 3 as the default
      to give a margin of safety).
    - A configurable "reserve" further reduces the budget.  For example,
      with max_day_trades=3 and day_trade_reserve=1, we will only use 2
      of the 3 slots automatically, leaving 1 for manual/emergency use.
    - Day-trade records are persisted to SQLite via the Database class so
      they survive bot restarts.
    - Weekend days (Saturday/Sunday) are skipped when calculating the
      5-business-day lookback window (but holidays are NOT accounted for).
"""

# "from __future__ import annotations" makes ALL type hints in this file
# lazy / string-based.  This lets you write things like "float | None"
# (union syntax) even on Python versions older than 3.10, because the
# annotation is never evaluated at runtime — it stays as a string.
from __future__ import annotations

from datetime import date, timedelta

from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import HoldType

# Create a module-level logger.  In Python, __name__ evaluates to the
# fully-qualified module path (e.g. "ai_trade.risk.pdt_manager"), which
# makes it easy to see where log messages originate.
logger = get_logger(__name__)


class PDTManager:
    """Track day-trade usage over the rolling 5-business-day window
    and enforce the PDT budget configured in ``config.pdt``.

    TRADING CONCEPT — Pattern Day Trade (PDT) Rule:
        A "day trade" is any round trip (buy then sell, or sell-short then
        cover) of the same security completed within a single trading day.
        The PDT rule says: if you execute 4 or more day trades in a 5-
        business-day period on a margin account under $25k, your broker
        must restrict the account.

    Typical usage:
        >>> pdt = PDTManager(config, database)
        >>> if pdt.can_day_trade():
        ...     # safe to submit a day trade
        ...     pdt.record_day_trade("AAPL", "2025-06-01", buy_id, sell_id)
    """

    def __init__(self, config, database) -> None:
        # `config` is a configuration object (likely a dataclass or namespace)
        # that holds settings like max_day_trades and day_trade_reserve.
        self.config = config

        # `database` is our SQLite persistence layer (see monitoring/database.py).
        # We use it to store and retrieve day-trade records so they persist
        # across bot restarts.
        self._database = database

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_day_trades_used(self) -> int:
        """Count day trades recorded in the last 5 business days.

        Queries the database for any day-trade records with a trade_date
        on or after the date that is 5 business days ago.  Returns the
        count of matching records.
        """
        cutoff = self._five_business_days_ago()

        # .isoformat() converts a Python date object to a string like
        # "2025-06-01", which matches the TEXT format stored in SQLite.
        trades = self._database.get_day_trades_since(cutoff.isoformat())
        return len(trades)

    def can_day_trade(self) -> bool:
        """Return ``True`` if we still have budget (respecting reserve).

        The check is:  used_trades < max_day_trades - reserve

        For example, with max=3 and reserve=1:
            - 0 used → 0 < 2 → True  (can trade)
            - 1 used → 1 < 2 → True  (can trade)
            - 2 used → 2 < 2 → False (budget exhausted)

        PYTHON PATTERN — getattr with default:
            `getattr(obj, "attr_name", default)` safely reads an attribute
            from an object, returning `default` if the attribute doesn't
            exist.  This avoids AttributeError if the config doesn't have
            that field.
        """
        max_trades: int = getattr(self.config, "max_day_trades", 3)
        reserve: int = getattr(self.config, "day_trade_reserve", 1)
        used = self.get_day_trades_used()
        allowed = used < max_trades - reserve

        # Structured logging: key=value pairs make logs easy to filter/parse.
        logger.debug(
            "pdt_check",
            used=used,
            max=max_trades,
            reserve=reserve,
            allowed=allowed,
        )
        return allowed

    def day_trades_remaining(self) -> int:
        """How many day trades we could still make (ignoring reserve).

        This is the raw count without the safety reserve subtracted.
        Useful for display/reporting purposes.

        PYTHON PATTERN — max(0, ...):
            Clamps the result to a minimum of 0 so we never return a
            negative number, even if the database somehow has more trades
            than the configured maximum.
        """
        max_trades: int = getattr(self.config, "max_day_trades", 3)
        return max(0, max_trades - self.get_day_trades_used())

    def record_day_trade(
        self,
        symbol: str,
        trade_date: str,
        buy_order_id: str = "",
        sell_order_id: str = "",
    ) -> None:
        """Persist a day-trade record to the database.

        Called after a same-day round trip is completed.  The record is
        used by `get_day_trades_used` to enforce the rolling window limit.

        Args:
            symbol:        Ticker symbol (e.g. "AAPL").
            trade_date:    ISO date string (e.g. "2025-06-01").
            buy_order_id:  Alpaca order ID for the buy leg.
            sell_order_id: Alpaca order ID for the sell leg.
        """
        self._database.record_day_trade(
            symbol=symbol,
            trade_date=trade_date,
            buy_order_id=buy_order_id,
            sell_order_id=sell_order_id,
        )
        logger.info(
            "day_trade_recorded",
            symbol=symbol,
            trade_date=trade_date,
            remaining=self.day_trades_remaining(),
        )

    @staticmethod
    def would_be_day_trade(hold_type: HoldType) -> bool:
        """Return ``True`` if the hold type implies a same-day round trip.

        PYTHON PATTERN — @staticmethod:
            A static method doesn't receive `self` — it's a plain function
            that lives inside the class for organizational purposes.  You
            can call it as `PDTManager.would_be_day_trade(...)` without
            needing an instance.

        TRADING CONCEPT — Hold Types:
            - DAY:      explicitly a same-day trade (always a day trade).
            - ADAPTIVE: may be closed same-day depending on conditions, so
                        we conservatively treat it as a day trade.
            - SWING:    held overnight, NOT a day trade.
        """
        return hold_type in (HoldType.DAY, HoldType.ADAPTIVE)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _five_business_days_ago() -> date:
        """Return the date 5 business days before today, skipping weekends.

        We walk backwards from today one calendar day at a time.  Each
        time we land on a weekday (Monday–Friday), we increment our
        business-day counter.  Once we've counted 5 weekdays, we return
        that date.

        NOTE: This does NOT account for market holidays (e.g. Christmas,
        MLK Day).  For a small-account PDT tracker, this is conservative
        — we may count a slightly wider window than necessary, which is
        the safer direction (over-counting day trades, not under-counting).

        PYTHON DETAIL — date.weekday():
            Returns 0 for Monday, 1 for Tuesday, ... 4 for Friday,
            5 for Saturday, 6 for Sunday.  So `weekday() < 5` means
            Monday through Friday.
        """
        today = date.today()
        biz_days = 0
        cursor = today
        while biz_days < 5:
            cursor -= timedelta(days=1)  # Move one calendar day backward
            if cursor.weekday() < 5:     # Mon=0 ... Fri=4 are business days
                biz_days += 1
        return cursor
