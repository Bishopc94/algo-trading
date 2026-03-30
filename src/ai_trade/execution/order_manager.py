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

import time
from datetime import datetime, timezone

# Alpaca SDK imports for interacting with the brokerage.
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, OrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
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

            stop_price = round(signal.stop_loss_price, 2)
            target_price = round(signal.take_profit_price, 2)
            entry_price = signal.entry_price

            # Adapt bracket legs to current market price.
            # Strategies compute entry/stop/target from daily bar close, which
            # can be hours stale by execution time.  Alpaca requires
            # stop < current_price < target.  Instead of rejecting when prices
            # diverge, we recalculate stop/target at the same risk/reward
            # RATIO relative to the current price.
            try:
                from ai_trade.data.historical import fetch_snapshots
                snaps = fetch_snapshots([signal.symbol])
                snap = snaps.get(signal.symbol)
                if snap and hasattr(snap, "latest_trade") and snap.latest_trade:
                    current_price = float(snap.latest_trade.price)
                elif snap and hasattr(snap, "daily_bar") and snap.daily_bar:
                    current_price = float(snap.daily_bar.close)
                else:
                    current_price = entry_price

                if entry_price > 0 and current_price > 0 and current_price != entry_price:
                    # Reject if price has diverged too far — the thesis is stale
                    divergence = abs(current_price - entry_price) / entry_price
                    if divergence > 0.50:
                        log.warning(
                            "bracket_rejected_extreme_divergence",
                            symbol=signal.symbol,
                            signal_entry=entry_price,
                            current_price=current_price,
                            divergence_pct=round(divergence * 100, 1),
                        )
                        return None

                    # Calculate the risk% and reward% from the original signal
                    stop_pct = (entry_price - signal.stop_loss_price) / entry_price
                    target_pct = (signal.take_profit_price - entry_price) / entry_price

                    # Recalculate bracket legs around current price
                    new_stop = round(current_price * (1 - stop_pct), 2)
                    new_target = round(current_price * (1 + target_pct), 2)

                    # Final sanity check: stop must be below price, target above
                    if new_stop >= current_price:
                        new_stop = round(current_price * 0.97, 2)  # 3% fallback
                    if new_target <= current_price:
                        new_target = round(current_price * 1.06, 2)  # 6% fallback

                    # Recalculate position size to preserve dollar risk.
                    # Original risk = shares * (signal_entry - signal_stop).
                    # New risk per share = current_price - new_stop.
                    # Scale shares so total dollar risk stays the same.
                    original_risk_per_share = signal.entry_price - signal.stop_loss_price
                    new_risk_per_share = current_price - new_stop
                    original_shares = shares
                    if original_risk_per_share > 0 and new_risk_per_share > 0:
                        total_dollar_risk = shares * original_risk_per_share
                        shares = max(1, int(total_dollar_risk / new_risk_per_share))

                    if new_stop != stop_price or new_target != target_price:
                        log.info(
                            "bracket_adapted_to_current_price",
                            symbol=signal.symbol,
                            signal_entry=entry_price,
                            current_price=current_price,
                            original_stop=stop_price,
                            new_stop=new_stop,
                            original_target=target_price,
                            new_target=new_target,
                            stop_pct=round(stop_pct * 100, 1),
                            target_pct=round(target_pct * 100, 1),
                            original_shares=original_shares,
                            adapted_shares=shares,
                            original_cost=round(original_shares * entry_price, 2),
                            adapted_cost=round(shares * current_price, 2),
                        )
                        print(f"    Price adapted: {signal.symbol} signal@${entry_price:.2f}"
                              f" -> now@${current_price:.2f}"
                              f" | stop ${stop_price}->${new_stop}"
                              f" | target ${target_price}->${new_target}"
                              f" | shares {original_shares}->{shares}"
                              f" (cost ${original_shares * entry_price:.0f}"
                              f" -> ${shares * current_price:.0f})")
                    stop_price = new_stop
                    target_price = new_target
                    entry_price = current_price
            except Exception as e:
                log.debug("bracket_price_adapt_failed", symbol=signal.symbol, error=str(e))

            # Build the bracket order request.
            # OrderClass.BRACKET tells Alpaca this is a 3-legged order.
            request = MarketOrderRequest(
                symbol=signal.symbol,
                qty=shares,
                side=OrderSide.BUY,
                time_in_force=tif,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=stop_price),
                take_profit=TakeProfitRequest(limit_price=target_price),
            )

            # Submit to Alpaca — this sends the order to the exchange.
            order = self._client.submit_order(order_data=request)

            log.info(
                "bracket_order_submitted",
                symbol=signal.symbol,
                shares=shares,
                order_id=str(order.id),
                stop_loss=stop_price,
                take_profit=target_price,
                entry_price=entry_price,
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
                entry_price=entry_price,  # Use adapted price, not stale signal price
                entry_time=datetime.now(timezone.utc).isoformat(),
                stop_loss=stop_price,
                take_profit=target_price,
                hold_type=signal.hold_type.value,
                status="open",
                buy_order_id=str(order.id),
            )

            return str(order.id)

        except Exception as e:
            error_msg = str(e).lower()
            error_raw = str(e)
            symbol = signal.symbol

            # Classify the error for clear operator feedback
            if "pattern day trading" in error_msg or "40310100" in error_msg:
                log.error("bracket_order_pdt_blocked", symbol=symbol, shares=shares, error=error_raw)
                print(f"    ORDER BLOCKED: {symbol} — Alpaca PDT protection triggered."
                      f" All day-trade slots used on Alpaca's side.")
            elif "insufficient" in error_msg or "buying power" in error_msg or "40110000" in error_msg:
                log.error("bracket_order_insufficient_funds", symbol=symbol, shares=shares, error=error_raw)
                print(f"    ORDER BLOCKED: {symbol} — Insufficient buying power for {shares} shares.")
            elif "forbidden" in error_msg or "403" in error_msg:
                log.error("bracket_order_forbidden", symbol=symbol, shares=shares, error=error_raw)
                print(f"    ORDER BLOCKED: {symbol} — Account restriction (403). Check Alpaca dashboard.")
            elif "not found" in error_msg or "asset" in error_msg and "not" in error_msg:
                log.error("bracket_order_invalid_symbol", symbol=symbol, error=error_raw)
                print(f"    ORDER FAILED: {symbol} — Symbol not found or not tradeable.")
            elif "halt" in error_msg or "suspended" in error_msg:
                log.error("bracket_order_halted", symbol=symbol, error=error_raw)
                print(f"    ORDER BLOCKED: {symbol} — Trading halted/suspended.")
            elif "rate" in error_msg or "429" in error_msg or "too many" in error_msg:
                log.error("bracket_order_rate_limited", symbol=symbol, error=error_raw)
                print(f"    ORDER DELAYED: {symbol} — Rate limited by Alpaca. Try again next window.")
            elif "timeout" in error_msg or "timed out" in error_msg or "connect" in error_msg:
                log.error("bracket_order_network_error", symbol=symbol, error=error_raw)
                print(f"    ORDER FAILED: {symbol} — Network error: {error_raw[:80]}")
            else:
                log.exception("bracket_order_failed", symbol=symbol, shares=shares)
                print(f"    ORDER FAILED: {symbol} — {error_raw[:120]}")
            return None

    # ── Position closing ─────────────────────────────────────

    def close_position(self, symbol: str) -> bool:
        """Close an open position for *symbol*. Returns True on success.

        This tells Alpaca to liquidate the entire position (sell all shares)
        at the current market price.  Alpaca handles the mechanics —
        submitting a market sell order and cancelling any open bracket legs.

        If shares are held by pending orders (e.g. bracket stop/target legs),
        we cancel those orders first, then retry the close.
        """
        try:
            self._client.close_position(symbol_or_asset_id=symbol)
            log.info("position_closed", symbol=symbol)
            return True
        except Exception as e:
            error_msg = str(e).lower()
            if "no position" in error_msg or "not found" in error_msg:
                log.info("position_already_closed", symbol=symbol)
                return True

            # Shares held by open orders — cancel them and retry once
            if "insufficient qty" in error_msg or "held_for_orders" in error_msg:
                log.warning("close_position_held_by_orders", symbol=symbol)
                print(f"    Shares held by open orders for {symbol} — cancelling orders and retrying...")
                try:
                    self._cancel_orders_for_symbol(symbol)
                    time.sleep(1)  # Brief pause for Alpaca to process cancellations
                    self._client.close_position(symbol_or_asset_id=symbol)
                    log.info("position_closed_after_cancel", symbol=symbol)
                    return True
                except Exception as retry_err:
                    retry_msg = str(retry_err).lower()
                    if "no position" in retry_msg or "not found" in retry_msg:
                        log.info("position_already_closed_after_cancel", symbol=symbol)
                        return True
                    log.exception("close_position_retry_failed", symbol=symbol)
                    print(f"    WARNING: Retry close {symbol} failed — {str(retry_err)[:100]}")
                    return False

            log.exception("close_position_failed", symbol=symbol)
            print(f"    WARNING: Failed to close {symbol} — {str(e)[:100]}")
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

                # Get current price before closing so we can record P&L
                exit_price = None
                try:
                    positions = self._client.get_all_positions()
                    for pos in positions:
                        if pos.symbol == symbol:
                            exit_price = float(pos.current_price)
                            break
                except Exception:
                    pass

                success = self.close_position(symbol)
                if success:
                    entry_price = trade.get("entry_price")
                    shares = trade.get("shares", 0)
                    pnl = None
                    pnl_pct = None
                    if exit_price and entry_price and entry_price > 0:
                        pnl = round((exit_price - entry_price) * shares, 2)
                        pnl_pct = round((exit_price - entry_price) / entry_price * 100, 2)

                    update_fields = {
                        "status": "closed",
                        "exit_time": datetime.now(timezone.utc).isoformat(),
                    }
                    if exit_price:
                        update_fields["exit_price"] = exit_price
                    if pnl is not None:
                        update_fields["pnl"] = pnl
                        update_fields["pnl_pct"] = pnl_pct

                    self._db.update_trade(trade["id"], **update_fields)
                    log.info("day_trade_closed", symbol=symbol, trade_id=trade["id"],
                             exit_price=exit_price, pnl=pnl)
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
        try:
            return self._client.get_all_positions()
        except Exception:
            log.exception("get_open_positions_failed")
            return []

    def get_open_orders(self) -> list:
        """Return all open (unfilled/partially-filled) orders from Alpaca."""
        try:
            return self._client.get_orders()
        except Exception:
            log.exception("get_open_orders_failed")
            return []

    # ── Bulk operations ──────────────────────────────────────

    def _cancel_orders_for_symbol(self, symbol: str) -> int:
        """Cancel all open orders for a specific symbol. Returns count cancelled."""
        try:
            request = GetOrdersRequest(
                status="open",
                symbols=[symbol],
            )
            orders = self._client.get_orders(filter=request)
            for order in orders:
                try:
                    self._client.cancel_order_by_id(order.id)
                except Exception:
                    pass  # Order may have already filled/cancelled
            log.info("orders_cancelled_for_symbol", symbol=symbol, count=len(orders))
            return len(orders)
        except Exception:
            log.exception("cancel_orders_for_symbol_failed", symbol=symbol)
            return 0

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
        4. Trades in our DB but NOT on Alpaca → "stale" (mark as closed in DB
           with exit price and P&L computed from the entry price).
        5. Trades still open → update unrealized P&L from live position data.

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

            # Build a lookup dict so we can get price data for any symbol
            position_map = {pos.symbol: pos for pos in positions}
            position_symbols = set(position_map.keys())
            summary["alpaca_positions"] = len(positions)

            # Step 2: Get our database's view.
            db_trades = self._db.get_open_trades()
            summary["db_open_trades"] = len(db_trades)

            db_symbols = {t["symbol"] for t in db_trades}

            # Step 3: Untracked positions (on Alpaca, not in DB).
            for sym in position_symbols - db_symbols:
                log.warning("untracked_position", symbol=sym)
                summary["untracked_positions"] += 1

            # Step 4: Stale trades (in DB, not on Alpaca) — closed by
            # stop-loss/take-profit fill.  Try to fetch the last trade
            # price to compute P&L.
            for trade in db_trades:
                sym = trade["symbol"]
                if sym not in position_symbols:
                    entry_price = trade.get("entry_price")
                    shares = trade.get("shares", 0)

                    # Attempt to get exit price from closed orders
                    exit_price = self._get_fill_price(trade.get("buy_order_id"), sym)

                    pnl = None
                    pnl_pct = None
                    if exit_price and entry_price and entry_price > 0:
                        pnl = round((exit_price - entry_price) * shares, 2)
                        pnl_pct = round((exit_price - entry_price) / entry_price * 100, 2)

                    update_fields = {
                        "status": "closed",
                        "exit_time": datetime.now(timezone.utc).isoformat(),
                    }
                    if exit_price:
                        update_fields["exit_price"] = exit_price
                    if pnl is not None:
                        update_fields["pnl"] = pnl
                        update_fields["pnl_pct"] = pnl_pct

                    self._db.update_trade(trade["id"], **update_fields)
                    log.info(
                        "stale_trade_closed",
                        symbol=sym,
                        trade_id=trade["id"],
                        exit_price=exit_price,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                    )
                    summary["stale_trades_closed"] += 1

        except Exception:
            log.exception("sync_positions_failed")

        return summary

    def _get_fill_price(self, buy_order_id: str | None, symbol: str) -> float | None:
        """Try to determine exit price from Alpaca order history.

        Checks closed orders for this symbol to find the most recent sell
        fill price.  Falls back to None if unavailable.
        """
        if not buy_order_id:
            return None
        try:
            # Get the parent bracket order — its legs contain the fill info
            order = self._client.get_order_by_id(buy_order_id)
            if order.legs:
                for leg in order.legs:
                    # A filled sell leg is our exit
                    if (leg.side == OrderSide.SELL
                            and leg.filled_avg_price is not None):
                        return float(leg.filled_avg_price)
            return None
        except Exception:
            log.debug("fill_price_lookup_failed", order_id=buy_order_id, symbol=symbol)
            return None
