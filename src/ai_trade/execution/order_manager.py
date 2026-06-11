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

from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import ConnectTimeout, ReadTimeout

# Alpaca SDK imports for interacting with the brokerage.
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderClass, OrderSide, OrderStatus, QueryOrderStatus, TimeInForce,
)
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    ReplaceOrderRequest,
    StopLossRequest,
    StopOrderRequest,
    TakeProfitRequest,
)

from ai_trade.clients import get_trading_client
from ai_trade.monitoring.database import Database
from ai_trade.monitoring.logger import get_logger
from ai_trade.utils import retry_api_call
from ai_trade.monitoring import console as con

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
        # Maps symbol -> error category for the most recent failed order.
        # Checked by the caller to decide whether to blacklist the symbol.
        self._last_order_error: dict[str, str] = {}

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

            # ---- Limit-entry path ----
            # If the strategy supplied a `limit_price`, the entry should be
            # a LIMIT at that level (waiting for a pullback to support),
            # NOT a market order at current price.  The stop and target
            # are re-anchored to the limit so the R:R ratio is preserved
            # at the actual fill price.  We skip the price-adaptation
            # block below entirely — adapting to current price would
            # defeat the purpose of waiting for a dip.
            limit_entry_price: float | None = None
            if getattr(signal, "limit_price", None) and signal.limit_price > 0:
                limit_entry_price = round(float(signal.limit_price), 2)
                if signal.entry_price > 0:
                    stop_pct = (signal.entry_price - signal.stop_loss_price) / signal.entry_price
                    target_pct = (signal.take_profit_price - signal.entry_price) / signal.entry_price
                    stop_price = round(limit_entry_price * (1 - stop_pct), 2)
                    target_price = round(limit_entry_price * (1 + target_pct), 2)
                entry_price = limit_entry_price

            # Adapt bracket legs to current market price (MARKET ENTRIES ONLY).
            # Strategies compute entry/stop/target from daily bar close, which
            # can be hours stale by execution time.  Alpaca requires
            # stop < current_price < target.  Instead of rejecting when prices
            # diverge, we recalculate stop/target at the same risk/reward
            # RATIO relative to the current price.  Limit entries skip this
            # because we WANT a specific entry level (the support), not the
            # current ask.
            if limit_entry_price is None:
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
                            print(con.price_adapted(
                                symbol=signal.symbol,
                                signal_entry=entry_price,
                                current=current_price,
                                old_stop=stop_price, new_stop=new_stop,
                                old_target=target_price, new_target=new_target,
                                old_shares=original_shares, new_shares=shares,
                            ))
                        stop_price = new_stop
                        target_price = new_target
                        entry_price = current_price
                except Exception as e:
                    log.debug("bracket_price_adapt_failed", symbol=signal.symbol, error=str(e))

            # Alpaca requires stop_price <= base_price - 0.01 and
            # take_profit >= base_price + 0.01.  We use a 0.05 buffer (not
            # just 0.01) because market orders fill at the ask, which can be
            # 1-2 cents above our snapshot price — a 0.01 gap becomes 0.00
            # relative to the actual fill and Alpaca rejects with 42210000.
            clamp_buffer = 0.05
            max_stop = round(entry_price - clamp_buffer, 2)
            min_target = round(entry_price + clamp_buffer, 2)
            if stop_price >= entry_price - clamp_buffer + 0.01:
                log.warning("stop_too_close_clamped", symbol=signal.symbol,
                            stop=stop_price, entry=entry_price, new_stop=max_stop)
                stop_price = max_stop
            if target_price <= entry_price + clamp_buffer - 0.01:
                log.warning("target_too_close_clamped", symbol=signal.symbol,
                            target=target_price, entry=entry_price, new_target=min_target)
                target_price = min_target

            # Build the bracket order request.  OrderClass.BRACKET tells
            # Alpaca this is a 3-legged order: entry + stop + take-profit.
            # The entry leg is a LIMIT when the strategy asked for a
            # pullback-entry (signal.limit_price set), otherwise MARKET.
            if limit_entry_price is not None:
                request = LimitOrderRequest(
                    symbol=signal.symbol,
                    qty=shares,
                    side=OrderSide.BUY,
                    time_in_force=tif,
                    order_class=OrderClass.BRACKET,
                    limit_price=limit_entry_price,
                    stop_loss=StopLossRequest(stop_price=stop_price),
                    take_profit=TakeProfitRequest(limit_price=target_price),
                )
            else:
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
                entry_type="limit" if limit_entry_price is not None else "market",
                limit_price=limit_entry_price,
                stop_loss=stop_price,
                take_profit=target_price,
                entry_price=entry_price,
                hold_type=signal.hold_type.value,
            )

            # Record the trade in our local database immediately.
            # We use datetime.now(timezone.utc) to get the current time
            # in UTC (Coordinated Universal Time) — the standard for
            # storing timestamps in databases.
            meta = getattr(signal, "metadata", {}) or {}
            atr_val = meta.get("atr")
            stop_method = meta.get("stop_method")
            target_method = meta.get("target_method")

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
                atr=atr_val,
                conviction=float(getattr(signal, "conviction", 0.0) or 0.0),
                stop_method=stop_method,
                target_method=target_method,
                high_since_entry=entry_price,
                low_since_entry=entry_price,
            )

            return str(order.id)

        except Exception as e:
            error_msg = str(e).lower()
            error_raw = str(e)
            symbol = signal.symbol

            # Classify the error. "account" errors are NOT symbol-specific —
            # don't let callers blacklist the symbol for these.
            if "pattern day trading" in error_msg or "40310100" in error_msg:
                log.error("bracket_order_pdt_blocked", symbol=symbol, shares=shares, error=error_raw)
                print(con.error(f"ORDER BLOCKED {symbol} — PDT protection triggered. All day-trade slots used."))
                self._last_order_error[symbol] = "account_pdt"
            elif "insufficient" in error_msg or "buying power" in error_msg or "40110000" in error_msg:
                log.error("bracket_order_insufficient_funds", symbol=symbol, shares=shares, error=error_raw)
                print(con.error(f"ORDER BLOCKED {symbol} — Insufficient buying power for {shares} shares."))
                self._last_order_error[symbol] = "account_funds"
            elif "forbidden" in error_msg or "403" in error_msg:
                log.error("bracket_order_forbidden", symbol=symbol, shares=shares, error=error_raw)
                print(con.error(f"ORDER BLOCKED {symbol} — Account restriction (403). Check Alpaca dashboard."))
                self._last_order_error[symbol] = "account_forbidden"
            elif "not found" in error_msg or "asset" in error_msg and "not" in error_msg:
                log.error("bracket_order_invalid_symbol", symbol=symbol, error=error_raw)
                print(con.error(f"ORDER FAILED {symbol} — Symbol not found or not tradeable."))
                self._last_order_error[symbol] = "symbol_invalid"
            elif "halt" in error_msg or "suspended" in error_msg:
                log.error("bracket_order_halted", symbol=symbol, error=error_raw)
                print(con.error(f"ORDER BLOCKED {symbol} — Trading halted/suspended."))
                self._last_order_error[symbol] = "symbol_halted"
            elif "rate" in error_msg or "429" in error_msg or "too many" in error_msg:
                log.error("bracket_order_rate_limited", symbol=symbol, error=error_raw)
                print(con.warning(f"ORDER DELAYED {symbol} — Rate limited by Alpaca. Try again next window."))
                self._last_order_error[symbol] = "account_rate_limit"
            elif "timeout" in error_msg or "timed out" in error_msg or "connect" in error_msg:
                log.error("bracket_order_network_error", symbol=symbol, error=error_raw)
                print(con.error(f"ORDER FAILED {symbol} — Network error: {error_raw[:80]}"))
                self._last_order_error[symbol] = "network"
            else:
                log.exception("bracket_order_failed", symbol=symbol, shares=shares)
                print(con.error(f"ORDER FAILED {symbol} — {error_raw[:120]}"))
                self._last_order_error[symbol] = "unknown"
            return None

    def pop_order_error(self, symbol: str) -> str | None:
        """Return and clear the last error type for *symbol*, or None."""
        return self._last_order_error.pop(symbol, None)

    # ── Position closing ─────────────────────────────────────

    def close_position(self, symbol: str) -> bool:
        """Close an open position for *symbol*. Returns True on success.

        This tells Alpaca to liquidate the entire position (sell all shares)
        at the current market price.  Alpaca handles the mechanics —
        submitting a market sell order and cancelling any open bracket legs.

        If shares are held by pending orders (e.g. bracket stop/target legs),
        we cancel those orders first, wait for Alpaca to release the shares,
        then retry the close.
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

            # Shares held by open orders — cancel them and retry
            if "insufficient qty" in error_msg or "held_for_orders" in error_msg:
                log.warning("close_position_held_by_orders", symbol=symbol)
                print(con.warning(f"Shares held by open orders for {symbol} — cancelling and retrying..."))
                return self._cancel_and_retry_close(symbol)

            log.exception("close_position_failed", symbol=symbol)
            print(con.warning(f"Failed to close {symbol} — {str(e)[:100]}"))
            return False

    def _cancel_and_retry_close(self, symbol: str, max_attempts: int = 4) -> bool:
        """Cancel open orders for *symbol*, wait for shares to be released, retry close.

        Alpaca's paper API can take 2-5 seconds to fully process bracket leg
        cancellations and release the held shares.  This method:
        1. Cancels per-symbol orders (bracket legs, stop-losses, etc.)
        2. If no orders found, falls back to cancelling ALL open orders
        3. Polls the position up to *max_attempts* times (2s apart) waiting
           for ``qty_available > 0``
        4. Retries the close once shares are available
        """
        # Step 1: Cancel orders holding the shares
        cancelled = self._cancel_orders_for_symbol(symbol)
        if cancelled == 0:
            # Bracket child legs may not show up in per-symbol query —
            # fall back to cancelling all open orders as a last resort.
            log.warning("no_orders_found_for_symbol_trying_cancel_all", symbol=symbol)
            try:
                self._client.cancel_orders()
            except Exception:
                pass

        # Step 2: Poll until shares are released or we give up
        for attempt in range(1, max_attempts + 1):
            time.sleep(2)
            try:
                pos = self._client.get_open_position(symbol)
                qty_available = int(pos.qty_available or 0)
                qty = int(pos.qty or 0)
                log.info(
                    "close_position_poll",
                    symbol=symbol, attempt=attempt,
                    qty=qty, qty_available=qty_available,
                )
                if qty_available > 0:
                    break
            except Exception as pe:
                pe_msg = str(pe).lower()
                if "not found" in pe_msg or "no position" in pe_msg:
                    log.info("position_gone_during_poll", symbol=symbol)
                    return True
                log.debug("position_poll_error", symbol=symbol, error=str(pe))
        else:
            # Exhausted all attempts — shares never freed
            log.error(
                "close_position_shares_never_released",
                symbol=symbol, attempts=max_attempts,
            )
            print(con.error(f"Could not release held shares for {symbol} after {max_attempts} attempts."))
            return False

        # Step 3: Retry the close
        try:
            self._client.close_position(symbol_or_asset_id=symbol)
            log.info("position_closed_after_cancel", symbol=symbol)
            return True
        except Exception as retry_err:
            retry_msg = str(retry_err).lower()
            if "no position" in retry_msg or "not found" in retry_msg:
                log.info("position_already_closed_after_cancel", symbol=symbol)
                return True
            log.error("close_position_retry_failed", symbol=symbol, error=str(retry_err))
            print(con.warning(f"Retry close {symbol} failed — {str(retry_err)[:100]}"))
            return False

    def close_all_day_trades(self, open_trades: list[dict]) -> list[dict]:
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

        Returns:
            List of trade dicts that were successfully closed (with
            ``exit_time`` populated).  The caller uses this to run the
            PDT exit-time hook.
        """
        closed: list[dict] = []
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

                    exit_time_iso = datetime.now(timezone.utc).isoformat()
                    update_fields = {
                        "status": "closed",
                        "exit_time": exit_time_iso,
                    }
                    if exit_price:
                        update_fields["exit_price"] = exit_price
                    if pnl is not None:
                        update_fields["pnl"] = pnl
                        update_fields["pnl_pct"] = pnl_pct

                    self._db.update_trade(trade["id"], **update_fields)
                    log.info("day_trade_closed", symbol=symbol, trade_id=trade["id"],
                             exit_price=exit_price, pnl=pnl)
                    # Return the merged dict so the caller can run the
                    # PDT exit-time hook with both entry_time and exit_time.
                    closed.append({**trade, **update_fields})
                else:
                    log.warning(
                        "day_trade_close_failed",
                        symbol=symbol,
                        trade_id=trade["id"],
                    )
        return closed

    # ── Queries ──────────────────────────────────────────────

    def get_open_positions(self) -> list:
        """Return all open positions from Alpaca.

        Each position object contains: symbol, qty, market_value,
        unrealized_pl, etc.  This queries the broker directly — it's the
        ground truth of what we actually hold.
        """
        try:
            return retry_api_call(self._client.get_all_positions)
        except (ConnectTimeout, ReadTimeout, RequestsConnectionError) as exc:
            log.warning("get_open_positions_network_error", error=str(exc))
            return []
        except Exception:
            log.exception("get_open_positions_failed")
            return []

    def get_open_orders(self) -> list:
        """Return all open (unfilled/partially-filled) orders from Alpaca."""
        try:
            return retry_api_call(self._client.get_orders)
        except (ConnectTimeout, ReadTimeout, RequestsConnectionError) as exc:
            log.warning("get_open_orders_network_error", error=str(exc))
            return []
        except Exception:
            log.exception("get_open_orders_failed")
            return []

    # ── Bulk operations ──────────────────────────────────────

    def _cancel_orders_for_symbol(self, symbol: str) -> int:
        """Cancel all open orders for a specific symbol. Returns count cancelled."""
        cancelled = 0
        try:
            request = GetOrdersRequest(
                status="open",
                symbols=[symbol],
            )
            orders = self._client.get_orders(filter=request)
            if not orders:
                log.info("no_open_orders_for_symbol", symbol=symbol)
                return 0
            for order in orders:
                try:
                    self._client.cancel_order_by_id(order.id)
                    cancelled += 1
                    log.info(
                        "order_cancelled",
                        symbol=symbol,
                        order_id=str(order.id),
                        order_type=str(getattr(order, "order_class", "unknown")),
                        side=str(getattr(order, "side", "unknown")),
                    )
                except Exception as ce:
                    # Order may have already filled/cancelled — not fatal
                    log.debug("cancel_single_order_skipped", symbol=symbol,
                              order_id=str(order.id), error=str(ce))
            log.info("orders_cancelled_for_symbol", symbol=symbol,
                     found=len(orders), cancelled=cancelled)
            return cancelled
        except (ConnectTimeout, ReadTimeout, RequestsConnectionError) as exc:
            log.warning("cancel_orders_for_symbol_network_error", symbol=symbol, error=str(exc))
            return cancelled
        except Exception:
            log.exception("cancel_orders_for_symbol_failed", symbol=symbol)
            return cancelled

    def cancel_all_open_orders(self) -> None:
        """Cancel every open order on Alpaca.

        Typically called at end-of-day or when the bot shuts down to
        ensure no stale orders linger.  Alpaca returns a list of the
        cancelled order objects.
        """
        try:
            canceled = self._client.cancel_orders()
            log.info("all_orders_canceled", count=len(canceled))
        except (ConnectTimeout, ReadTimeout, RequestsConnectionError) as exc:
            log.warning("cancel_all_orders_network_error", error=str(exc))
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
        summary: dict = {
            "alpaca_positions": 0,
            "db_open_trades": 0,
            "untracked_positions": 0,
            "stale_trades_closed": 0,
            "closed_trades": [],
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
            # Alpaca is the source of truth — adopt these back into the DB so
            # the bot resumes managing them (trailing stops, exits, risk
            # accounting).  Adoption reads the position's existing protective
            # orders; if there's no stop-loss leg, it attaches one so the
            # position isn't left unprotected.  See _adopt_untracked_position.
            #
            # ALSO catch DB-tracked-but-unprotected positions: a trade can sit
            # in the DB with stop_loss=0/NULL after an earlier adoption failure
            # or a botched bracket.  When this happens the heat calc treats
            # the WHOLE position value as at-risk and the bot stops opening
            # new entries (ONDS-style lockup, found 2026-06-05).  Treat these
            # the same way — query orders, re-protect, fix the DB row.
            untracked = position_symbols - db_symbols
            naked_tracked: list[dict] = []
            for t in db_trades:
                if (
                    t["symbol"] in position_symbols
                    and (not t.get("stop_loss") or float(t.get("stop_loss") or 0) <= 0)
                ):
                    naked_tracked.append(t)

            open_orders_by_symbol: dict = {}
            if untracked or naked_tracked:
                try:
                    all_open = self._client.get_orders(
                        filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
                    )
                    for o in all_open:
                        open_orders_by_symbol.setdefault(o.symbol, []).append(o)
                except Exception as e:
                    log.warning("untracked_orders_fetch_failed", error=str(e))

            for t in naked_tracked:
                sym = t["symbol"]
                try:
                    self._reprotect_tracked_position(
                        t, position_map[sym], open_orders_by_symbol.get(sym, [])
                    )
                except Exception:
                    log.exception("reprotect_tracked_failed", symbol=sym)

            for sym in untracked:
                summary["untracked_positions"] += 1
                try:
                    self._adopt_untracked_position(
                        position_map[sym], open_orders_by_symbol.get(sym, [])
                    )
                except Exception as e:
                    log.exception("adopt_untracked_failed", symbol=sym)

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
                    summary["closed_trades"].append({
                        "trade_id": trade["id"],
                        "symbol": sym,
                        "strategy": trade.get("strategy", ""),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "shares": shares,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "hold_type": trade.get("hold_type", ""),
                        "stop_loss": trade.get("stop_loss"),
                        "take_profit": trade.get("take_profit"),
                        "high_since_entry": trade.get("high_since_entry"),
                        "low_since_entry": trade.get("low_since_entry"),
                        # Needed by pdt.record_if_day_trade() to determine
                        # whether the round-trip fell within a single ET day.
                        "entry_time": trade.get("entry_time"),
                        "exit_time": update_fields["exit_time"],
                        "buy_order_id": trade.get("buy_order_id"),
                        "sell_order_id": trade.get("sell_order_id"),
                    })

        except (ConnectTimeout, ReadTimeout, RequestsConnectionError) as exc:
            log.warning("sync_positions_network_error", error=str(exc))
        except Exception:
            log.exception("sync_positions_failed")

        return summary

    def _reprotect_tracked_position(
        self, trade: dict, position, open_orders: list,
    ) -> None:
        """Re-attach a stop to a DB-tracked position that lost its protection.

        Symmetric to ``_adopt_untracked_position`` for the opposite drift:
        the DB row exists but ``stop_loss`` is 0/NULL — typically because an
        earlier adoption couldn't place the stop (shares held by a lone TP)
        and never recovered.  Heat then counts the FULL position value as
        at-risk and the bot stops opening new entries (the ONDS-style lockup
        found 2026-06-05).

        Strategy:
          1. If Alpaca shows a live stop leg, just sync the DB to it.
          2. Otherwise cancel any lone sell legs, submit an OCO (stop +
             take-profit).  Falls back to a plain stop if OCO is rejected.
          3. Update the DB row so the heat calc reflects real risk.
        """
        sym = position.symbol
        qty = abs(int(float(position.qty)))
        if qty <= 0:
            return
        entry_price = float(trade.get("entry_price") or position.avg_entry_price)
        try:
            current_price = float(position.current_price)
        except Exception:
            current_price = entry_price

        # Existing protective legs on Alpaca.
        stop_leg = target_leg = None
        for o in open_orders:
            otype = str(getattr(o, "order_type", "")).lower()
            if o.side == OrderSide.SELL and "stop" in otype and o.stop_price:
                stop_leg = o
            elif o.side == OrderSide.SELL and otype.endswith("limit") and o.limit_price:
                target_leg = o

        # Case 1: Alpaca already has a stop — just sync the DB.
        if stop_leg is not None:
            self._db.update_trade(
                trade["id"], stop_loss=float(stop_leg.stop_price),
                take_profit=float(target_leg.limit_price) if target_leg else trade.get("take_profit"),
            )
            log.warning(
                "tracked_position_db_synced_to_alpaca",
                symbol=sym, stop=float(stop_leg.stop_price),
            )
            return

        # Case 2: no stop on Alpaca — protect now.
        stop_pct = 0.03
        candidate = round(entry_price * (1 - stop_pct), 2)
        ceiling = round(current_price * 0.99, 2)  # must sit below market
        new_stop = min(candidate, ceiling)
        new_target = (
            float(target_leg.limit_price) if target_leg
            else float(trade.get("take_profit") or 0)
            or round(entry_price * 1.10, 2)
        )
        if new_stop <= 0:
            return

        # Free any held shares so the OCO can reserve them.
        for o in open_orders:
            if o.side == OrderSide.SELL:
                try:
                    self._client.cancel_order_by_id(o.id)
                except Exception as e:
                    log.debug("reprotect_cancel_leg_failed", symbol=sym, error=str(e))
        import time
        time.sleep(1)

        placed_stop = placed_target = None
        try:
            self._client.submit_order(order_data=LimitOrderRequest(
                symbol=sym, qty=qty, side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.OCO,
                take_profit=TakeProfitRequest(limit_price=new_target),
                stop_loss=StopLossRequest(stop_price=new_stop),
            ))
            placed_stop, placed_target = new_stop, new_target
            log.warning(
                "tracked_position_reprotected", method="oco",
                symbol=sym, qty=qty, new_stop=new_stop, new_target=new_target,
            )
        except Exception as e:
            log.warning("reprotect_oco_failed_trying_plain_stop", symbol=sym, error=str(e))
            try:
                self._client.submit_order(order_data=StopOrderRequest(
                    symbol=sym, qty=qty, side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC, stop_price=new_stop,
                ))
                placed_stop = new_stop
                log.warning(
                    "tracked_position_reprotected", method="plain_stop",
                    symbol=sym, new_stop=new_stop,
                )
            except Exception as e2:
                log.error("reprotect_stop_submit_failed", symbol=sym, error=str(e2))

        if placed_stop:
            self._db.update_trade(
                trade["id"], stop_loss=placed_stop,
                take_profit=placed_target if placed_target else trade.get("take_profit"),
            )

    def _adopt_untracked_position(self, position, open_orders: list) -> None:
        """Re-create a DB trade row for a position Alpaca holds but the DB lost.

        Alpaca is the source of truth.  When a position drifts out of the DB
        (broken bracket, missed fill, manual trade), we adopt it back:

          1. Read the symbol's existing protective orders — a sell STOP leg
             gives the stop-loss, a sell LIMIT leg gives the take-profit.
          2. If there's NO stop-loss leg, attach a protective stop so the
             position isn't left naked.  The standalone stop coexists with
             any lone take-profit; whichever fills first leaves the other to
             be rejected harmlessly (we only hold the shares once).
          3. Insert an `open` trade row (strategy='adopted', hold_type='swing'
             so it's never force-closed as a day trade) so the trailing-stop
             and exit jobs manage it from now on.

        long-only (the bot never shorts), so this assumes a long position.
        """
        sym = position.symbol
        qty = abs(int(float(position.qty)))
        if qty <= 0:
            return
        entry_price = float(position.avg_entry_price)
        try:
            current_price = float(position.current_price)
        except Exception:
            current_price = entry_price

        # Classify existing protective legs.
        stop_leg = None
        target_leg = None
        for o in open_orders:
            if o.side != OrderSide.BUY and o.side != OrderSide.SELL:
                continue
            otype = str(getattr(o, "order_type", "")).lower()
            if o.side == OrderSide.SELL and "stop" in otype and o.stop_price:
                stop_leg = o
            elif o.side == OrderSide.SELL and otype.endswith("limit") and o.limit_price:
                target_leg = o

        stop_loss = float(stop_leg.stop_price) if stop_leg else None
        take_profit = float(target_leg.limit_price) if target_leg else None

        # No stop protection → attach one.  The position's shares are usually
        # already reserved by a lone take-profit leg (held_for_orders == qty),
        # so a standalone stop is rejected for "insufficient qty".  The fix is
        # to cancel the existing sell leg(s) to free the shares, then submit a
        # proper OCO (one-cancels-other: stop + target reserve the shares once
        # and auto-cancel each other).  Falls back to a plain stop if the OCO
        # is rejected — downside protection matters more than the target.
        if stop_loss is None:
            stop_pct = getattr(getattr(self, "_risk_cfg", None), "stop_loss_pct", 0.03) or 0.03
            candidate = round(entry_price * (1 - stop_pct), 2)
            ceiling = round(current_price * 0.99, 2)  # sell-stop must sit below market
            new_stop = min(candidate, ceiling)
            new_target = take_profit or round(entry_price * 1.10, 2)

            if new_stop > 0:
                # Free the shares: cancel any existing sell legs.
                for o in open_orders:
                    if o.side == OrderSide.SELL:
                        try:
                            self._client.cancel_order_by_id(o.id)
                        except Exception as e:
                            log.debug("adopt_cancel_leg_failed", symbol=sym, error=str(e))
                import time
                time.sleep(1)  # let Alpaca release the held shares

                placed = False
                try:
                    self._client.submit_order(order_data=LimitOrderRequest(
                        symbol=sym, qty=qty, side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC,
                        order_class=OrderClass.OCO,
                        take_profit=TakeProfitRequest(limit_price=new_target),
                        stop_loss=StopLossRequest(stop_price=new_stop),
                    ))
                    stop_loss, take_profit, placed = new_stop, new_target, True
                    log.warning(
                        "adopted_position_reprotected", method="oco",
                        symbol=sym, qty=qty, entry=entry_price,
                        current=current_price, new_stop=new_stop, new_target=new_target,
                    )
                except Exception as e:
                    log.warning("adopt_oco_failed_trying_plain_stop", symbol=sym, error=str(e))

                if not placed:
                    # Fallback: at least a plain stop (shares freed by cancel above).
                    try:
                        self._client.submit_order(order_data=StopOrderRequest(
                            symbol=sym, qty=qty, side=OrderSide.SELL,
                            time_in_force=TimeInForce.GTC, stop_price=new_stop,
                        ))
                        stop_loss, take_profit = new_stop, None
                        log.warning(
                            "adopted_position_reprotected", method="plain_stop",
                            symbol=sym, qty=qty, new_stop=new_stop,
                        )
                    except Exception as e:
                        log.error("adopt_stop_submit_failed", symbol=sym, error=str(e))
                        stop_loss = None  # don't claim protection we failed to place

        self._db.insert_trade(
            symbol=sym,
            strategy="adopted",
            side="long",
            shares=qty,
            entry_price=entry_price,
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=stop_loss if stop_loss else 0.0,
            take_profit=take_profit if take_profit else round(entry_price * 1.10, 2),
            hold_type="swing",
            status="open",
            buy_order_id="",  # original entry order unknown
            high_since_entry=max(entry_price, current_price),
            low_since_entry=min(entry_price, current_price),
            stop_method="adopted",
            target_method="adopted",
        )
        log.info(
            "position_adopted",
            symbol=sym, qty=qty, entry=entry_price,
            stop_loss=stop_loss, take_profit=take_profit,
            had_stop=stop_leg is not None, had_target=target_leg is not None,
        )

    def find_stop_leg_id(self, buy_order_id: str) -> str | None:
        """Locate the stop-loss child order id for a bracket parent order.

        Used by the trailing-stop job to know which leg to replace. Returns
        None if the parent or the stop leg can't be found.
        """
        try:
            order = self._client.get_order_by_id(buy_order_id)
            for leg in order.legs or []:
                if leg.order_type is None:
                    continue
                if "stop" in str(leg.order_type).lower() and leg.status not in (
                    OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED,
                ):
                    return str(leg.id)
        except Exception:
            log.debug("stop_leg_lookup_failed", order_id=buy_order_id)
        return None

    def replace_stop_price(self, stop_order_id: str, new_stop_price: float) -> bool:
        """Replace a live stop-loss order with a new stop price.

        Used by the trailing-stop job to tighten stops as trades move in
        our favor. Returns True on success.
        """
        try:
            req = ReplaceOrderRequest(stop_price=round(new_stop_price, 2))
            self._client.replace_order_by_id(order_id=stop_order_id, order_data=req)
            log.info(
                "stop_price_replaced",
                order_id=stop_order_id,
                new_stop=round(new_stop_price, 2),
            )
            return True
        except Exception as e:
            log.warning(
                "stop_replace_failed",
                order_id=stop_order_id,
                error=str(e),
            )
            return False

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
