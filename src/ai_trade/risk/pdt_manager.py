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

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import HoldType

# PDT is an ET-based calendar rule. Entry/exit dates must be compared in
# ET, not UTC — a trade entered 16:05 ET and exited 00:30 UTC the next day
# is the same ET day (and therefore a round-trip day trade).
ET = ZoneInfo("America/New_York")

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

        # Regulatory framework.  FINRA retired the PDT rule on 2026-06-04,
        # replacing the 3-trades/5-days designation with an intraday margin
        # model (Intraday Buying Power + pre-trade margin checks).  Under
        # "intraday_margin" the day-trade COUNT limit no longer exists, so
        # this manager stops gating on it and defers entirely to Alpaca's
        # pre-trade checks + our buying_power-based sizing.  Default stays
        # "legacy" until the operator flips it on/after the cutover date.
        self._framework: str = getattr(config, "framework", "legacy")

        # Alpaca's server-side PDT count — synced at startup and before trades.
        # This is the authoritative count; our local DB is a backup.
        # NOTE: deprecated by Alpaca 2026-06-04, fully removed 2026-07-06 —
        # reads are guarded with getattr() so a missing field can't crash us.
        self._alpaca_daytrade_count: int | None = None

        # Track the last mismatch pair so we only warn when it changes —
        # otherwise a stale local row spams a warning every sync cycle.
        self._last_mismatch_pair: tuple[int, int] | None = None

    def framework_is_legacy(self) -> bool:
        """True while the old PDT count rule is in force (pre-2026-06-04)."""
        return self._framework == "legacy"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync_with_alpaca(self) -> None:
        """Sync our PDT count with Alpaca's server-side count.

        Alpaca independently tracks day trades.  If the bot was restarted,
        or trades were made outside the bot, our local DB count may be
        stale.  Alpaca's count is authoritative — if it's higher than
        ours, we trust Alpaca.
        """
        # Under the intraday-margin framework there is no day-trade count to
        # sync — the field is a deprecated placeholder (always 0) and the
        # count limit no longer exists.  Skip the sync entirely.
        if not self.framework_is_legacy():
            self._alpaca_daytrade_count = 0
            return
        try:
            from ai_trade.clients import get_account
            account = get_account()
            # getattr guard: field is removed from the API after 2026-07-06.
            self._alpaca_daytrade_count = int(getattr(account, "daytrade_count", 0) or 0)
            local_count = self._get_local_day_trades_used()
            if self._alpaca_daytrade_count != local_count:
                pair = (self._alpaca_daytrade_count, local_count)
                if pair != self._last_mismatch_pair:
                    logger.warning(
                        "pdt_count_mismatch",
                        alpaca_count=self._alpaca_daytrade_count,
                        local_count=local_count,
                        using="alpaca (authoritative)",
                    )
                    self._last_mismatch_pair = pair
                else:
                    logger.debug(
                        "pdt_count_mismatch_repeat",
                        alpaca_count=self._alpaca_daytrade_count,
                        local_count=local_count,
                    )
            else:
                self._last_mismatch_pair = None
                logger.debug(
                    "pdt_synced",
                    count=self._alpaca_daytrade_count,
                )
        except Exception as e:
            logger.warning("pdt_sync_failed", error=str(e))

    def get_day_trades_used(self) -> int:
        """Return the higher of Alpaca's count and our local count.

        Alpaca's server-side count is authoritative.  We take the max
        of both to be conservative — if either source says we've used
        a slot, we respect that.
        """
        local = self._get_local_day_trades_used()
        alpaca = self._alpaca_daytrade_count
        if alpaca is not None:
            return max(local, alpaca)
        return local

    def _get_local_day_trades_used(self) -> int:
        """Count day trades recorded in our local DB in the last 5 business days."""
        cutoff = self._rolling_window_start()
        trades = self._database.get_day_trades_since(cutoff.isoformat())
        return len(trades)

    def in_flight_day_trades(self) -> int:
        """Count open positions that will round-trip into day trades today.

        Day trades are recorded at EXIT time (see ``record_if_day_trade``),
        so positions opened earlier in the same session don't yet appear
        in the historical count.  The PDT gate has to add this number on
        top of the historical count or the bot will oversubscribe within
        a single trading day.

        We count only positions where:
          * status is open
          * entry_time falls on today's ET date
          * hold_type is DAY or ADAPTIVE (would_be_day_trade)
        """
        try:
            open_trades = self._database.get_open_trades()
        except Exception:
            return 0
        today_et = datetime.now(ET).date()
        count = 0
        for t in open_trades:
            hold = (t.get("hold_type") or "").lower()
            if hold not in ("day", "adaptive"):
                continue
            entry_time = t.get("entry_time")
            if not entry_time:
                continue
            try:
                entry_et = datetime.fromisoformat(entry_time).astimezone(ET)
            except (TypeError, ValueError):
                continue
            if entry_et.date() == today_et:
                count += 1
        return count

    def can_day_trade(self) -> bool:
        """Return ``True`` if we still have budget (respecting reserve).

        The check is:  (used + in_flight) < max_day_trades - reserve

        ``used`` counts completed round-trips in the rolling 5-day window;
        ``in_flight`` counts open day-mode positions opened today that
        will become day trades when the EOD close-out fires.  Without
        ``in_flight`` we'd allow the gate to issue 4 same-day day-mode
        entries before the first one closed and the count caught up.

        Example with max=3, reserve=0:
            - 0 used, 0 in-flight → 0 < 3 → True  (can trade)
            - 0 used, 2 in-flight → 2 < 3 → True  (can trade)
            - 0 used, 3 in-flight → 3 < 3 → False (slots booked)

        Under the intraday-margin framework (post-2026-06-04) there is no
        day-trade count limit — Alpaca's pre-trade margin checks and our
        buying_power sizing are the only constraints — so this always
        returns True and lets those layers do the gating.
        """
        if not self.framework_is_legacy():
            return True
        max_trades: int = getattr(self.config, "max_day_trades", 3)
        reserve: int = getattr(self.config, "day_trade_reserve", 1)
        used = self.get_day_trades_used()
        in_flight = self.in_flight_day_trades()
        allowed = (used + in_flight) < max_trades - reserve

        logger.debug(
            "pdt_check",
            used=used,
            in_flight=in_flight,
            max=max_trades,
            reserve=reserve,
            allowed=allowed,
        )
        return allowed

    def day_trades_remaining(self) -> int:
        """How many day trades we could still make right now (ignores reserve).

        Subtracts both completed round-trips in the rolling window and
        open same-day day-mode positions that will become day trades on
        close.  Useful for display and operator reporting.
        """
        max_trades: int = getattr(self.config, "max_day_trades", 3)
        return max(
            0,
            max_trades - self.get_day_trades_used() - self.in_flight_day_trades(),
        )

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

    def record_if_day_trade(self, trade: dict) -> bool:
        """Record a day-trade row iff the trade round-tripped within a
        single ET calendar day.

        This is the EXIT-TIME hook: called after a trade transitions to
        ``status='closed'``.  We compare ``entry_time`` and ``exit_time``
        (both stored as UTC ISO strings) in Eastern Time and only record
        when they fall on the same ET date.

        Previously the bot recorded day trades at ENTRY time based on the
        signal's intended ``hold_type``.  That over-counted whenever a
        day-typed trade got held overnight (e.g. late-session fill that
        couldn't close before 16:00 ET) and silently froze the PDT
        budget on subsequent days.  Recording at exit is the only way
        to match FINRA's same-day-round-trip definition.

        Idempotent: skips if a matching ``(symbol, trade_date,
        buy_order_id)`` row already exists.

        Returns ``True`` if a new row was written.
        """
        entry_time = trade.get("entry_time")
        exit_time = trade.get("exit_time")
        if not entry_time or not exit_time:
            return False
        try:
            entry_et = datetime.fromisoformat(entry_time).astimezone(ET)
            exit_et = datetime.fromisoformat(exit_time).astimezone(ET)
        except (TypeError, ValueError):
            logger.debug(
                "record_if_day_trade_parse_failed",
                entry_time=entry_time,
                exit_time=exit_time,
            )
            return False
        if entry_et.date() != exit_et.date():
            return False

        trade_date = entry_et.date().isoformat()
        symbol = trade.get("symbol") or ""
        buy_order_id = str(trade.get("buy_order_id") or "")
        sell_order_id = str(trade.get("sell_order_id") or "")

        # Idempotency — scan the rolling window for a pre-existing row
        # matching this trade.  Reconcile jobs may fire twice on the same
        # close; without this guard we would double-count.
        existing = self._database.get_day_trades_since(trade_date)
        for row in existing:
            if (
                row.get("symbol") == symbol
                and row.get("trade_date") == trade_date
                and (not buy_order_id or row.get("buy_order_id") == buy_order_id)
            ):
                return False

        self.record_day_trade(
            symbol=symbol,
            trade_date=trade_date,
            buy_order_id=buy_order_id,
            sell_order_id=sell_order_id,
        )
        return True

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
    def _rolling_window_start() -> date:
        """Return the earliest date in the rolling 5-business-day window.

        FINRA's window is "5 rolling business days" — that's TODAY plus
        the 4 most-recent prior weekdays.  Example with today = Fri 5/1:
        the window is {Mon 4/27, Tue 4/28, Wed 4/29, Thu 4/30, Fri 5/1}.
        The cutoff (earliest in-window date) is Mon 4/27.

        Previously this walked back 5 weekdays, returning Fri 4/24 — that
        gave a 6-business-day window and silently held trades from one
        week ago in the count.  Holiday-aware? No (e.g. MLK Day still
        counts as a business day here); that's the conservative direction.

        PYTHON DETAIL — date.weekday():
            Returns 0 for Monday, 1 for Tuesday, ... 4 for Friday,
            5 for Saturday, 6 for Sunday.  `weekday() < 5` means Mon-Fri.
        """
        today = date.today()
        biz_days = 0
        cursor = today
        while biz_days < 4:
            cursor -= timedelta(days=1)
            if cursor.weekday() < 5:
                biz_days += 1
        return cursor
