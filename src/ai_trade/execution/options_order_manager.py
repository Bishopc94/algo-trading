"""Options order execution — single-leg and multi-leg orders via Alpaca.

WHAT THIS MODULE DOES:
    Manages the submission and lifecycle of options orders, including both
    single-leg orders (e.g. buying a call) and multi-leg spread orders
    (e.g. a credit put spread with two legs).

WHY IT EXISTS:
    Options orders are structurally different from stock orders:
    - They use special OCC-format symbols (e.g. "AAPL  250620C00200000").
    - Multi-leg ("spread") orders must be submitted as a single atomic
      unit so both legs fill together or not at all.
    - Position intents (buy_to_open, sell_to_close, etc.) matter for
      margin and assignment tracking.
    - Expiration management is critical — options near expiration carry
      assignment/exercise risk and should be closed proactively.

KEY TRADING CONCEPTS:

    Option Symbol (OCC Format):
        Options use a standardized symbol format from the Options Clearing
        Corporation (OCC).  Example: "AAPL  250620C00200000" means:
        - AAPL     = underlying stock (padded to 6 chars with spaces)
        - 250620   = expiration date (YYMMDD → June 20, 2025)
        - C        = Call (P would be Put)
        - 00200000 = strike price × 1000 ($200.00)

    Credit Spread vs Debit Spread:
        - Credit spread: you RECEIVE money upfront (sell the expensive leg,
          buy the cheap leg).  limit_price is NEGATIVE (a credit to you).
        - Debit spread: you PAY money upfront (buy the expensive leg, sell
          the cheap leg).  limit_price is POSITIVE (a cost to you).

    Position Intent:
        - buy_to_open:  Opening a new long options position.
        - sell_to_open: Opening a new short (written) options position.
        - buy_to_close: Closing an existing short position.
        - sell_to_close: Closing an existing long position.

    Assignment/Exercise Risk:
        If you hold a short option through expiration and it's in-the-money,
        you may be "assigned" — forced to buy/sell 100 shares of the
        underlying stock.  This module proactively closes positions near
        expiration to avoid this risk.

KEY DESIGN DECISIONS:
    - The OptionLegRequest import is wrapped in try/except because not all
      versions of the alpaca-py SDK include it.  If unavailable, spread
      orders are gracefully rejected instead of crashing.
    - Expiration dates are parsed from the OCC symbol string directly,
      avoiding the need for a separate API call.
    - All options trades are recorded in the same "trades" table as stocks
      for simplicity, with the strategy field distinguishing them.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import ConnectTimeout, ReadTimeout

# Alpaca SDK imports for options order submission.
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, PositionIntent, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

# PYTHON PATTERN — try/except ImportError:
# Some classes may not exist in all versions of a library.  Wrapping the
# import in try/except lets the code load without crashing, and we check
# for None before using it.  "pragma: no cover" tells the test coverage
# tool to ignore this line (since we can't reliably test both import paths).
try:
    from alpaca.trading.requests import OptionLegRequest
except ImportError:  # pragma: no cover
    OptionLegRequest = None  # type: ignore[assignment,misc]

from ai_trade.clients import get_trading_client
from ai_trade.monitoring.database import Database
from ai_trade.monitoring.logger import get_logger
from ai_trade.utils import retry_api_call

log = get_logger(__name__)

# Regular expression to identify OCC-format option symbols.
# Pattern breakdown:
#   ^[A-Z]{1,6}  — 1-6 uppercase letters (the underlying ticker, e.g. "AAPL")
#   \s*          — optional whitespace (OCC pads to 6 chars with spaces)
#   \d{6}        — 6 digits for the expiration date (YYMMDD)
#   [CP]         — "C" for Call or "P" for Put
#   \d{8}$       — 8 digits for the strike price (price × 1000, zero-padded)
_OPTION_SYMBOL_RE = re.compile(r"^[A-Z]{1,6}\s*\d{6}[CP]\d{8}$")


class OptionsOrderManager:
    """Manages options order lifecycle: submission, closure, and position queries.

    Supports both single-leg orders (e.g. buy a call) and multi-leg spread
    orders (e.g. credit put spread).
    """

    def __init__(self, database: Database) -> None:
        self._db = database

    @property
    def _client(self) -> TradingClient:
        """Lazily fetch the Alpaca TradingClient on each access.

        PYTHON PATTERN — @property:
            See order_manager.py for a full explanation.  The key idea is
            that `self._client` looks like an attribute but actually calls
            a function each time, ensuring a fresh client instance.
        """
        return get_trading_client()

    # ── Order submission ─────────────────────────────────────

    def submit_spread_order(self, signal, qty: int = 1) -> str | None:
        """Submit a multi-leg spread order (credit put spread, debit call spread, etc.).

        TRADING CONCEPT — Spread Orders:
            A "spread" combines two or more options into one trade.  For
            example, a credit put spread sells a high-strike put and buys
            a lower-strike put.  The multi-leg order ensures both legs
            execute atomically — you never end up with just one leg filled,
            which would leave you with an unhedged position.

            The order_class="mleg" tells Alpaca this is a multi-leg order.

        Args:
            signal: An options signal object containing:
                    - legs: list of dicts with "symbol", "side", "qty", "position_intent"
                    - strategy_type: enum with a .value like "credit_put_spread"
                    - min_credit / max_cost: pricing constraints
                    - symbol: underlying ticker
            qty:    Number of spread contracts to trade (default 1).
                    Each contract typically controls 100 shares of stock.

        Returns:
            The Alpaca order ID as a string on success, or None on failure.
        """
        try:
            # Guard: if the SDK doesn't support OptionLegRequest, we can't
            # submit multi-leg orders.
            if OptionLegRequest is None:
                log.error("option_leg_request_unavailable",
                          msg="OptionLegRequest not found in alpaca.trading.requests")
                return None

            # Build the leg specifications for the spread order.
            # Each leg specifies which option contract to buy/sell and the
            # position intent (open vs close).
            legs = []
            for leg in signal.legs:
                legs.append(OptionLegRequest(
                    symbol=leg["symbol"],          # OCC-format option symbol
                    side=OrderSide.BUY if leg["side"] == "buy" else OrderSide.SELL,
                    ratio_qty=leg.get("qty", 1),   # Ratio for unbalanced spreads
                    position_intent=self._map_intent(leg["position_intent"]),
                ))

            # Determine the limit price based on spread type.
            # For credit spreads (you receive money), the limit price is
            # NEGATIVE — Alpaca convention for representing a net credit.
            # For debit spreads (you pay money), the limit price is POSITIVE.
            if signal.strategy_type.value in ("credit_put_spread",):
                limit_price = -abs(signal.min_credit)  # Negative = credit received
            else:
                limit_price = abs(signal.max_cost)      # Positive = cost paid

            # Build and submit the multi-leg limit order.
            # TimeInForce.DAY means the order expires at market close if
            # not filled — options orders are typically day-only.
            request = LimitOrderRequest(
                qty=qty,
                order_class="mleg",                 # Multi-leg order class
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 2),  # Round to 2 decimals (penny pricing)
                legs=legs,
            )

            order = self._client.submit_order(order_data=request)

            log.info(
                "spread_order_submitted",
                strategy=signal.strategy_type.value,
                symbol=signal.symbol,
                qty=qty,
                order_id=str(order.id),
                limit_price=round(limit_price, 2),
                num_legs=len(legs),
            )

            # Record in the trades table.  For spreads, the "side" is
            # recorded as "spread" and entry_price stores the net credit/debit.
            self._db.insert_trade(
                symbol=signal.symbol,
                strategy=signal.strategy_type.value,
                side="spread",
                shares=qty,
                entry_price=round(limit_price, 2),
                entry_time=datetime.now(timezone.utc).isoformat(),
                hold_type="day",
                status="open",
                buy_order_id=str(order.id),
            )

            return str(order.id)

        except Exception as e:
            error_msg = str(e).lower()
            symbol = getattr(signal, "symbol", "unknown")
            if "insufficient" in error_msg or "buying power" in error_msg:
                log.error("spread_order_insufficient_funds", symbol=symbol, qty=qty, error=str(e))
                print(f"    OPTIONS BLOCKED: {symbol} — Insufficient buying power for spread.")
            elif "forbidden" in error_msg or "403" in error_msg:
                log.error("spread_order_forbidden", symbol=symbol, error=str(e))
                print(f"    OPTIONS BLOCKED: {symbol} — Account restriction (403).")
            elif "not found" in error_msg or "invalid" in error_msg:
                log.error("spread_order_invalid_contract", symbol=symbol, error=str(e))
                print(f"    OPTIONS FAILED: {symbol} — Invalid contract or symbol not found.")
            else:
                log.exception("spread_order_failed", symbol=symbol, qty=qty)
                print(f"    OPTIONS FAILED: {symbol} — {str(e)[:100]}")
            return None

    def submit_single_leg_order(self, signal, qty: int = 1) -> str | None:
        """Submit a single-leg options order (long call, cash-secured put, etc.).

        TRADING CONCEPT — Single-Leg Options:
            Unlike a spread, a single-leg order trades just one option
            contract.  Examples:
            - Long call: buy a call option (bullish bet, limited risk).
            - Cash-secured put: sell a put option (bullish, obligation to
              buy shares if the stock drops below the strike price).

        Args:
            signal: Signal object with legs (list of 1 dict), max_cost,
                    min_credit, strategy_type, and symbol.
            qty:    Number of contracts (each controls 100 shares).

        Returns:
            Alpaca order ID string on success, or None on failure.
        """
        try:
            # Single-leg order uses the first (and only) leg.
            leg = signal.legs[0]
            side = OrderSide.BUY if leg["side"] == "buy" else OrderSide.SELL

            # For buys, we set a max price (max_cost) we're willing to pay.
            # For sells, we set a minimum price (min_credit) we'll accept.
            limit_price = round(
                signal.max_cost if side == OrderSide.BUY else signal.min_credit, 2
            )

            request = LimitOrderRequest(
                symbol=leg["symbol"],          # OCC-format option symbol
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY, # Expire at close if unfilled
                limit_price=limit_price,
            )

            order = self._client.submit_order(order_data=request)

            log.info(
                "single_leg_order_submitted",
                strategy=signal.strategy_type.value,
                symbol=leg["symbol"],
                side=leg["side"],
                qty=qty,
                order_id=str(order.id),
                limit_price=limit_price,
            )

            self._db.insert_trade(
                symbol=signal.symbol,
                strategy=signal.strategy_type.value,
                side=leg["side"],
                shares=qty,
                entry_price=limit_price,
                entry_time=datetime.now(timezone.utc).isoformat(),
                hold_type="day",
                status="open",
                buy_order_id=str(order.id),
            )

            return str(order.id)

        except Exception as e:
            error_msg = str(e).lower()
            symbol = getattr(signal, "symbol", "unknown")
            if "insufficient" in error_msg or "buying power" in error_msg:
                log.error("single_leg_insufficient_funds", symbol=symbol, qty=qty, error=str(e))
                print(f"    OPTIONS BLOCKED: {symbol} — Insufficient buying power.")
            elif "forbidden" in error_msg or "403" in error_msg:
                log.error("single_leg_forbidden", symbol=symbol, error=str(e))
                print(f"    OPTIONS BLOCKED: {symbol} — Account restriction (403).")
            else:
                log.exception("single_leg_order_failed", symbol=symbol, qty=qty)
                print(f"    OPTIONS FAILED: {symbol} — {str(e)[:100]}")
            return None

    def submit_options_order(self, signal, qty: int = 1) -> str | None:
        """Route to the appropriate submission method based on the number of legs.

        This is the main entry point for options order submission.  It
        inspects signal.legs and delegates to either submit_spread_order
        (for multi-leg) or submit_single_leg_order (for single-leg).
        """
        if not signal or not getattr(signal, "legs", None):
            log.error("options_order_no_legs", symbol=getattr(signal, "symbol", "unknown"))
            return None
        if len(signal.legs) > 1:
            return self.submit_spread_order(signal, qty=qty)
        return self.submit_single_leg_order(signal, qty=qty)

    # ── Position management ──────────────────────────────────

    def close_options_position(self, option_symbol: str) -> bool:
        """Close an open options position. Returns True on success.

        Alpaca's close_position works for options the same way as stocks —
        it liquidates the entire position at market price.
        """
        try:
            self._client.close_position(symbol_or_asset_id=option_symbol)
            log.info("options_position_closed", symbol=option_symbol)
            return True
        except Exception:
            log.exception("close_options_position_failed", symbol=option_symbol)
            return False

    def get_options_positions(self) -> list:
        """Return all current positions that are options contracts.

        Filters the full position list by checking if the symbol matches
        the OCC format regex or is longer than 10 characters (stock tickers
        are typically 1-5 characters, while OCC option symbols are 21+).

        PYTHON PATTERN — list comprehension with filter:
            `[x for x in items if condition]` creates a new list containing
            only items that pass the condition.  This is more concise and
            often faster than a for-loop with an if-statement and append.
        """
        try:
            positions = retry_api_call(self._client.get_all_positions)
            return [
                pos for pos in positions
                if _OPTION_SYMBOL_RE.match(pos.symbol) or len(pos.symbol) > 10
            ]
        except (ConnectTimeout, ReadTimeout, RequestsConnectionError) as exc:
            log.warning("get_options_positions_network_error", error=str(exc))
            return []
        except Exception:
            log.exception("get_options_positions_failed")
            return []

    def close_expiring_positions(self, days_until_expiration: int = 1) -> list[str]:
        """Close any options positions expiring within *days_until_expiration* days.

        TRADING CONCEPT — Expiration Risk Management:
            Options that are near expiration ("expiring") carry several
            risks:
            1. Assignment risk: short options that are in-the-money at
               expiration will be assigned, forcing you to buy/sell 100
               shares per contract.
            2. Gamma risk: option prices become extremely sensitive to
               stock price moves near expiration.
            3. Liquidity risk: bid-ask spreads widen as expiration nears.

            To avoid these risks, this method is called daily (typically
            around 3:00 PM ET) to close any positions expiring within the
            specified window.

        The expiration date is parsed directly from the OCC symbol string:
            Symbol:     "AAPL  250620C00200000"
            Stripped:   "AAPL250620C00200000"
            Last 15:    "250620C00200000"
            Chars [0:6]: "250620" → YYMMDD → June 20, 2025

        Args:
            days_until_expiration: Close positions expiring within this
                                   many days (default: 1 = tomorrow or today).

        Returns:
            List of option symbols that were successfully closed.
        """
        from datetime import timedelta

        # Calculate the cutoff: any option expiring on or before this
        # datetime should be closed.
        cutoff = datetime.now(timezone.utc) + timedelta(days=days_until_expiration)
        closed: list[str] = []

        for pos in self.get_options_positions():
            # Parse the expiration date from the OCC symbol.
            # First, strip any spaces (OCC symbols are padded).
            sym = pos.symbol.replace(" ", "")
            try:
                # The date portion is 6 characters starting 15 from the end.
                # For "AAPL250620C00200000": [-15:-9] → "250620"
                date_part = sym[-15:-9]  # YYMMDD format
                exp_dt = datetime.strptime(date_part, "%y%m%d").replace(tzinfo=timezone.utc)
            except (ValueError, IndexError):
                # If we can't parse the date, skip this position.
                continue

            if exp_dt <= cutoff:
                log.info(
                    "closing_expiring_option",
                    symbol=pos.symbol,
                    expiration=exp_dt.strftime("%Y-%m-%d"),
                    days_left=(exp_dt - datetime.now(timezone.utc)).days,
                )
                if self.close_options_position(pos.symbol):
                    closed.append(pos.symbol)

        if closed:
            log.info("expiring_positions_closed", count=len(closed), symbols=closed)
        return closed

    # ── Helpers ───────────────────────────────────────────────

    def _map_intent(self, intent_str: str) -> PositionIntent:
        """Map a snake_case intent string to the corresponding PositionIntent enum.

        TRADING CONCEPT — Position Intent:
            Options orders require specifying whether you're opening or
            closing a position, and whether you're buying or selling:
            - buy_to_open:   start a new long position
            - sell_to_open:  start a new short (written) position
            - buy_to_close:  close an existing short position
            - sell_to_close: close an existing long position

            This matters for the broker's margin calculations and regulatory
            reporting.

        PYTHON PATTERN — dict lookup with validation:
            Instead of a chain of if/elif statements, we use a dictionary
            for O(1) lookup.  If the key isn't found, .get() returns None,
            and we raise a descriptive error.

        Raises:
            ValueError: If the intent string doesn't match any known intent.
        """
        mapping: dict[str, PositionIntent] = {
            "buy_to_open": PositionIntent.BUY_TO_OPEN,
            "buy_to_close": PositionIntent.BUY_TO_CLOSE,
            "sell_to_open": PositionIntent.SELL_TO_OPEN,
            "sell_to_close": PositionIntent.SELL_TO_CLOSE,
        }
        intent = mapping.get(intent_str.lower())
        if intent is None:
            log.error("unknown_position_intent", intent=intent_str)
            # `!r` in f-strings adds repr() quotes around the value,
            # making it clear if there are hidden spaces or special chars.
            raise ValueError(f"Unknown position intent: {intent_str!r}")
        return intent
