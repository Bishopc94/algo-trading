"""Order management — submit, close, cancel, and sync positions via Alpaca.

WHAT THIS MODULE DOES:
    Handles the full lifecycle of stock orders: submitting new bracket
    orders, closing positions, cancelling orders, and reconciling what
    the broker (Alpaca) knows with what our local database knows.

WHY IT EXISTS:
    The trading bot needs a clean interface between its decision-making
    logic (strategies, risk checks) and the broker API.  This module is
    that interface.  It also maintains a local database mirror of all
    trades for performance tracking and auditing.

KEY CONCEPTS:
    - Bracket Order: A single order that creates THREE legs at once:
      (1) a market buy to enter the position, (2) a stop-loss order to
      limit downside, and (3) a take-profit limit order to lock in gains.
      The broker manages legs 2 and 3 automatically — when one fills, the
      other is cancelled (this is called "OCO" — One Cancels Other).

    - Time In Force (TIF): How long the order stays active.
      * DAY = expires at market close today.
      * GTC = Good Till Cancelled — stays active until filled or you cancel it.

    - Position Sync / Reconciliation: The broker is the source of truth
      for what positions you actually hold.  Our local database can drift
      (e.g. a stop-loss fills while the bot is offline).  The sync process
      compares both sides and fixes discrepancies.

KEY DESIGN DECISIONS:
    - The Alpaca client is accessed via a @property that calls a factory
      function each time.  This ensures we always get a properly-configured
      client (useful if credentials rotate or for testing).
    - All broker API calls are wrapped in try/except to prevent one failed
      order from crashing the entire bot.
    - The database is updated immediately after order submission, not after
      fill confirmation.  This is "optimistic" recording — we know the
      order was accepted by Alpaca, even if it hasn't filled yet.
"""

from __future__ import annotations

from datetime import datetime, timezone

# Alpaca SDK imports for interacting with the brokerage.
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, OrderStatus, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from ai_trade.clients import get_trading_client
from ai_trade.monitoring.database import Database
from ai_trade.monitoring.logger import get_logger

log = get_logger(__name__)


class OrderManager:
    """Manages order lifecycle: submission, closure, cancellation, and DB sync.

    This class is the primary interface between the bot's strategy logic
    and the Alpaca brokerage API for stock (non-options) orders.
    """

    def __init__(self, database: Database) -> None:
        # We store a reference to the database, not the trading client.
        # The client is fetched on-demand via the _client property below.
        self._db = database

    @property
    def _client(self) -> TradingClient:
        """Lazily fetch the Alpaca TradingClient.

        PYTHON PATTERN — @property:
            The @property decorator turns a method into something that
            looks like an attribute access.  Instead of calling
            `self._client()`, you just write `self._client`.  This is
            useful for computed or lazily-fetched values.

            Here, every access calls `get_trading_client()` to get a
            fresh client instance.  This ensures we always use the
            currently-configured credentials.
        """
        return get_trading_client()

    # ── Order submission ─────────────────────────────────────

    def submit_bracket_order(self, signal, shares: int) -> str | None:
        """Submit a bracket order (market entry + stop-loss + take-profit).

        TRADING CONCEPT — Bracket Order:
            A bracket order is a three-part order that enters a position
            and simultaneously sets up protective exit orders:

            1. ENTRY (market order) — buy shares immediately at the
               current market price.
            2. STOP-LOSS — a stop order placed BELOW entry price.  If the
               stock drops to this price, the position is sold to limit
               losses.  This is your "worst case" exit.
            3. TAKE-PROFIT — a limit order placed ABOVE entry price.  If
               the stock rises to this price, the position is sold to lock
               in gains.

            The stop-loss and take-profit are linked as OCO (One Cancels
            Other): when one fills, the broker automatically cancels the
            other.

        Args:
            signal: A Signal object containing symbol, entry_price,
                    stop_loss_price, take_profit_price, hold_type, and
                    strategy_name.
            shares: Number of shares to buy (from the position sizer).

        Returns:
            The Alpaca order ID as a string on success, or None on failure.
        """
        try:
            # Choose Time In Force based on hold type:
            # - DAY trades use TIF.DAY (auto-cancel at close if unfilled)
            # - Swing trades use TIF.GTC (stay active until filled/cancelled)
            tif = (
                TimeInForce.DAY
                if signal.hold_type.value == "day"
                else TimeInForce.GTC
            )

            # Build the bracket order request.
            # OrderClass.BRACKET tells Alpaca this is a 3-legged order.
            # round(..., 2) ensures prices have at most 2 decimal places,
            # which is required by the exchange for stocks.
            request = MarketOrderRequest(
                symbol=signal.symbol,
                qty=shares,
                side=OrderSide.BUY,
                time_in_force=tif,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=round(signal.stop_loss_price, 2)),
                take_profit=TakeProfitRequest(limit_price=round(signal.take_profit_price, 2)),
            )

            # Submit to Alpaca — this sends the order to the exchange.
            order = self._client.submit_order(order_data=request)

            log.info(
                "bracket_order_submitted",
                symbol=signal.symbol,
                shares=shares,
                order_id=str(order.id),
                stop_loss=round(signal.stop_loss_price, 2),
                take_profit=round(signal.take_profit_price, 2),
                hold_type=signal.hold_type.value,
            )

            # Record the trade in our local database immediately.
            # We use datetime.now(timezone.utc) to get the current time
            # in UTC (Coordinated Universal Time) — the standard for
            # storing timestamps in databases.
            self._db.insert_trade(
                symbol=signal.symbol,
                strategy=signal.strategy_name,
                side="long",
                shares=shares,
                entry_price=signal.entry_price,
                entry_time=datetime.now(timezone.utc).isoformat(),
                stop_loss=round(signal.stop_loss_price, 2),
                take_profit=round(signal.take_profit_price, 2),
                hold_type=signal.hold_type.value,
                status="open",
                buy_order_id=str(order.id),
            )

            return str(order.id)

        except Exception:
            # PYTHON PATTERN — log.exception():
            # Like log.error() but automatically includes the full stack
            # trace (traceback) of the exception.  Essential for debugging
            # why an order failed (e.g. insufficient buying power, invalid
            # symbol, API timeout).
            log.exception(
                "bracket_order_failed",
                symbol=signal.symbol,
                shares=shares,
            )
            return None

    # ── Position closing ─────────────────────────────────────

    def close_position(self, symbol: str) -> bool:
        """Close an open position for *symbol*. Returns True on success.

        This tells Alpaca to liquidate the entire position (sell all shares)
        at the current market price.  Alpaca handles the mechanics —
        submitting a market sell order and cancelling any open bracket legs.
        """
        try:
            self._client.close_position(symbol_or_asset_id=symbol)
            log.info("position_closed", symbol=symbol)
            return True
        except Exception:
            log.exception("close_position_failed", symbol=symbol)
            return False

    def close_all_day_trades(self, open_trades: list[dict]) -> None:
        """Close all open day-trade positions and update DB status.

        Called near market close (e.g. 3:45 PM ET) to ensure we don't
        accidentally hold day-trade positions overnight, which would still
        count as a day trade but defeat the purpose of the day-trade
        strategy.

        TRADING CONCEPT — Day Trade Closure:
            Day trades MUST be closed before the market closes at 4:00 PM
            ET.  Holding them overnight converts them into swing trades,
            which may expose the account to overnight gap risk and changes
            the PDT accounting.

        Args:
            open_trades: List of trade dicts from the database.
        """
        for trade in open_trades:
            # Only close trades that are both marked as "day" hold_type
            # AND still in "open" status.
            if trade.get("hold_type") == "day" and trade.get("status") == "open":
                symbol = trade["symbol"]
                success = self.close_position(symbol)
                if success:
                    self._db.update_trade(
                        trade["id"],
                        status="closed",
                        exit_time=datetime.now(timezone.utc).isoformat(),
                    )
                    log.info("day_trade_closed", symbol=symbol, trade_id=trade["id"])
                else:
                    log.warning(
                        "day_trade_close_failed",
                        symbol=symbol,
                        trade_id=trade["id"],
                    )

    # ── Queries ──────────────────────────────────────────────

    def get_open_positions(self) -> list:
        """Return all open positions from Alpaca.

        Each position object contains: symbol, qty, market_value,
        unrealized_pl, etc.  This queries the broker directly — it's the
        ground truth of what we actually hold.
        """
        return self._client.get_all_positions()

    def get_open_orders(self) -> list:
        """Return all open (unfilled/partially-filled) orders from Alpaca."""
        return self._client.get_orders()

    # ── Bulk operations ──────────────────────────────────────

    def cancel_all_open_orders(self) -> None:
        """Cancel every open order on Alpaca.

        Typically called at end-of-day or when the bot shuts down to
        ensure no stale orders linger.  Alpaca returns a list of the
        cancelled order objects.
        """
        try:
            canceled = self._client.cancel_orders()
            log.info("all_orders_canceled", count=len(canceled))
        except Exception:
            log.exception("cancel_all_orders_failed")

    # ── Reconciliation ───────────────────────────────────────

    def sync_positions(self) -> dict:
        """Reconcile Alpaca positions with the local DB.

        This is critical because the broker is the source of truth.  Our
        database can become stale when:
        - A stop-loss or take-profit fills while the bot is offline.
        - A manual trade is made via the Alpaca dashboard.
        - Network issues cause a missed fill notification.

        The reconciliation logic:
        1. Fetch all positions from Alpaca → these are what we ACTUALLY hold.
        2. Fetch all "open" trades from our database → these are what we
           THINK we hold.
        3. Positions on Alpaca but NOT in our DB → "untracked" (log a warning).
        4. Trades in our DB but NOT on Alpaca → "stale" (mark as closed in DB).

        Returns a summary dict with counts of each reconciliation action,
        useful for monitoring and alerting.
        """
        summary: dict[str, int] = {
            "alpaca_positions": 0,
            "db_open_trades": 0,
            "untracked_positions": 0,
            "stale_trades_closed": 0,
        }

        try:
            # Step 1: Get the broker's ground truth.
            positions = self._client.get_all_positions()

            # PYTHON PATTERN — set comprehension:
            # `{pos.symbol for pos in positions}` creates a set of all
            # symbols we hold on Alpaca.  Sets provide O(1) lookup time,
            # which is much faster than scanning a list for membership.
            position_symbols = {pos.symbol for pos in positions}
            summary["alpaca_positions"] = len(positions)

            # Step 2: Get our database's view.
            db_trades = self._db.get_open_trades()
            summary["db_open_trades"] = len(db_trades)

            db_symbols = {t["symbol"] for t in db_trades}

            # Step 3: Untracked positions (on Alpaca, not in DB).
            # Set subtraction: position_symbols - db_symbols gives symbols
            # that exist on Alpaca but aren't tracked in our database.
            for sym in position_symbols - db_symbols:
                log.warning("untracked_position", symbol=sym)
                summary["untracked_positions"] += 1

            # Step 4: Stale trades (in DB, not on Alpaca).
            # These are trades our DB thinks are open, but the broker says
            # we don't hold them anymore.  Most commonly, this means a
            # stop-loss or take-profit order filled and closed the position.
            for trade in db_trades:
                if trade["symbol"] not in position_symbols:
                    self._db.update_trade(
                        trade["id"],
                        status="closed",
                        exit_time=datetime.now(timezone.utc).isoformat(),
                    )
                    log.info(
                        "stale_trade_closed",
                        symbol=trade["symbol"],
                        trade_id=trade["id"],
                    )
                    summary["stale_trades_closed"] += 1

        except Exception:
            log.exception("sync_positions_failed")

        return summary
