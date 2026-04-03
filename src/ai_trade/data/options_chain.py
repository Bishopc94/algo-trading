"""Options chain data fetching via the Alpaca-py SDK.

This module provides functions to fetch options contract data:
  - ``get_options_chain()`` — fetches all active option contracts for a stock.
  - ``get_options_snapshot()`` — fetches real-time pricing and Greeks for
    specific option symbols.
  - ``find_contracts()`` — combined convenience function that fetches the
    chain, gets snapshots, and filters by delta range.

Options terminology for non-options traders:
  - **Chain**: The full list of available option contracts for a stock,
    across different strike prices and expiration dates.
  - **Greeks**: Mathematical measures of an option's sensitivity to
    various factors (delta, theta, gamma, vega).
  - **Delta**: How much the option price moves per $1 move in the stock.
    A delta of 0.30 means the option gains $0.30 for every $1 the stock
    rises.  Also roughly the probability of expiring in-the-money.
  - **Snapshot**: A point-in-time view of an option's bid/ask/Greeks.

Python-specific notes:
  - ``@lru_cache(maxsize=1)`` decorates ``_get_option_data_client()``
    to cache its return value.  The first call creates the client; all
    subsequent calls return the cached instance.  This is a simpler
    alternative to the explicit singleton pattern used in ``clients.py``.
  - ``{**dict1, **dict2}`` merges two dicts — if both have the same key,
    dict2's value wins.  This is used to merge chain contract info with
    snapshot greeks data.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from functools import lru_cache

from alpaca.data.requests import OptionSnapshotRequest
from alpaca.trading.requests import GetOptionContractsRequest

from ai_trade.clients import get_trading_client
from ai_trade.monitoring.logger import get_logger
from ai_trade.utils import retry_api_call as _retry, extract_greeks

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_option_data_client():
    """Lazily create and cache the options data client.

    Uses @lru_cache as a simple singleton — the client is created on first
    call and reused thereafter.  This is separate from the stock data client
    because options data requires the OptionHistoricalDataClient class.
    """
    from alpaca.data.historical.option import OptionHistoricalDataClient

    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    return OptionHistoricalDataClient(api_key, secret_key)


def get_options_chain(
    underlying_symbol: str,
    expiration_date: str | None = None,
    min_dte: int = 7,
    max_dte: int = 45,
    option_type: str | None = None,
) -> list[dict]:
    """Fetch the active options chain for a stock.

    Queries Alpaca for all option contracts that match the criteria and
    returns them as a list of plain dicts (not SDK objects).

    Args:
        underlying_symbol: The stock ticker (e.g. "AAPL").
        expiration_date:   Optional specific expiration date to filter to.
        min_dte:           Minimum days to expiration (default 7).
        max_dte:           Maximum days to expiration (default 45).
        option_type:       "call" or "put" to filter, or None for both.

    Returns:
        A list of dicts, each with keys: symbol, strike_price,
        expiration_date, type, root_symbol.  Sorted by expiration
        then strike price.
    """
    today = date.today()
    min_date = today + timedelta(days=min_dte)
    max_date = today + timedelta(days=max_dte)

    # Build the request parameters for Alpaca's option contracts endpoint
    request_params: dict = {
        "underlying_symbols": [underlying_symbol],
        "status": "active",                     # Only active (tradeable) contracts
        "expiration_date_gte": min_date,        # Expiring on or after min_date
        "expiration_date_lte": max_date,        # Expiring on or before max_date
    }
    if option_type is not None:
        request_params["type"] = option_type
    if expiration_date is not None:
        request_params["expiration_date"] = expiration_date

    request = GetOptionContractsRequest(**request_params)

    log.info(
        "get_options_chain",
        underlying=underlying_symbol,
        min_date=str(min_date),
        max_date=str(max_date),
        option_type=option_type,
    )

    try:
        response = _retry(get_trading_client().get_option_contracts, request)
    except Exception:
        log.error("get_options_chain_failed", underlying=underlying_symbol)
        return []

    # Convert SDK contract objects to plain dicts for easier manipulation
    contracts: list[dict] = []
    if response and response.option_contracts:
        for contract in response.option_contracts:
            contracts.append(
                {
                    "symbol": contract.symbol,
                    "strike_price": float(contract.strike_price),
                    "expiration_date": str(contract.expiration_date),
                    "type": str(contract.type),
                    "root_symbol": contract.root_symbol,
                }
            )

    # Sort by expiration date first, then by strike price (ascending).
    # This makes it easy to browse contracts chronologically.
    contracts.sort(key=lambda c: (c["expiration_date"], c["strike_price"]))

    log.info("get_options_chain_done", underlying=underlying_symbol, count=len(contracts))
    return contracts


def get_options_snapshot(option_symbols: list[str]) -> dict[str, dict]:
    """Fetch real-time pricing and Greeks for specific option symbols.

    This is the key function for options strategy evaluation — it provides
    the bid/ask prices, implied volatility, and Greeks (delta, gamma,
    theta, vega) needed to assess option contracts.

    Args:
        option_symbols: List of OCC-format option symbols
                        (e.g. "AAPL250620C00200000").

    Returns:
        A dict keyed by option symbol, each value containing:
        bid, ask, mid_price, last_trade, volume, open_interest,
        delta, gamma, theta, vega, implied_volatility.
    """
    if not option_symbols:
        return {}

    client = _get_option_data_client()

    log.info("get_options_snapshot", count=len(option_symbols))

    try:
        request = OptionSnapshotRequest(symbol_or_symbols=option_symbols)
        raw_snapshots = _retry(client.get_option_snapshot, request)
    except Exception:
        log.error("get_options_snapshot_failed", count=len(option_symbols))
        return {}

    result: dict[str, dict] = {}
    for sym, snap in raw_snapshots.items():
        # Extract bid/ask from the latest quote
        bid = 0.0
        ask = 0.0
        last_trade = 0.0
        volume = 0
        open_interest = 0

        if snap.latest_quote is not None:
            bid = float(snap.latest_quote.bid_price or 0)
            ask = float(snap.latest_quote.ask_price or 0)

        if snap.latest_trade is not None:
            last_trade = float(snap.latest_trade.price or 0)

        # Extract Greeks using the normalisation helper from utils.py
        g = extract_greeks(snap)

        # Volume and OI may be on the snapshot or a nested trade object
        if hasattr(snap, "volume"):
            volume = int(snap.volume or 0)
        if hasattr(snap, "open_interest"):
            open_interest = int(snap.open_interest or 0)

        # Mid-price = average of bid and ask (standard options pricing).
        # Fall back to last trade price if no bid/ask available.
        mid_price = (bid + ask) / 2 if (bid + ask) > 0 else last_trade

        # Merge all data into a flat dict
        result[sym] = {
            "bid": bid,
            "ask": ask,
            "mid_price": mid_price,
            "last_trade": last_trade,
            "volume": volume,
            "open_interest": open_interest,
            **g,  # Spread Greeks dict into the result (delta, gamma, theta, vega, IV)
        }

    log.info("get_options_snapshot_done", count=len(result))
    return result


def get_option_quote(option_symbol: str) -> dict:
    """Get the latest bid/ask quote for a single option symbol.

    A simpler version of ``get_options_snapshot()`` for when you only
    need the price of one specific contract.

    Returns:
        A dict with keys: bid, ask, mid_price.
    """
    client = _get_option_data_client()

    log.info("get_option_quote", symbol=option_symbol)

    try:
        request = OptionSnapshotRequest(symbol_or_symbols=[option_symbol])
        raw_snapshots = _retry(client.get_option_snapshot, request)
    except Exception:
        log.error("get_option_quote_failed", symbol=option_symbol)
        return {"bid": 0.0, "ask": 0.0, "mid_price": 0.0}

    snap = raw_snapshots.get(option_symbol)
    if snap is None:
        log.warning("get_option_quote_no_data", symbol=option_symbol)
        return {"bid": 0.0, "ask": 0.0, "mid_price": 0.0}

    bid = float(snap.latest_quote.bid_price or 0) if snap.latest_quote else 0.0
    ask = float(snap.latest_quote.ask_price or 0) if snap.latest_quote else 0.0
    mid_price = (bid + ask) / 2 if (bid + ask) > 0 else 0.0

    return {"bid": bid, "ask": ask, "mid_price": mid_price}


def find_contracts(
    underlying: str,
    option_type: str,
    min_delta: float,
    max_delta: float,
    min_dte: int = 7,
    max_dte: int = 45,
) -> list[dict]:
    """Find option contracts filtered by delta range — a convenience wrapper.

    Combines three steps into one call:
      1. Fetch the full options chain (``get_options_chain``).
      2. Fetch snapshots with Greeks for all contracts.
      3. Filter to contracts whose absolute delta falls within the range.

    This is used by strategies that want a specific delta profile
    (e.g. "give me puts with delta between 0.20 and 0.35").

    Args:
        underlying:  Stock ticker (e.g. "AAPL").
        option_type: "call" or "put".
        min_delta:   Minimum absolute delta.
        max_delta:   Maximum absolute delta.
        min_dte:     Minimum days to expiration.
        max_dte:     Maximum days to expiration.

    Returns:
        List of dicts with contract info + Greeks merged, sorted by
        absolute delta descending (highest delta first).
    """
    chain = get_options_chain(
        underlying,
        option_type=option_type,
        min_dte=min_dte,
        max_dte=max_dte,
    )
    if not chain:
        log.info("find_contracts_empty_chain", underlying=underlying)
        return []

    symbols = [c["symbol"] for c in chain]
    snapshots = get_options_snapshot(symbols)

    # Build a lookup so we can merge chain info with snapshot Greeks
    chain_lookup = {c["symbol"]: c for c in chain}

    filtered: list[dict] = []
    for sym, snap_data in snapshots.items():
        abs_delta = abs(snap_data.get("delta", 0.0))
        if min_delta <= abs_delta <= max_delta:
            # Merge contract info (strike, expiration) with Greeks (delta, theta)
            merged = {**chain_lookup.get(sym, {}), **snap_data}
            filtered.append(merged)

    # Sort by absolute delta descending — highest delta first for
    # directional trades (closer to ATM = more responsive to stock moves)
    filtered.sort(key=lambda c: abs(c.get("delta", 0.0)), reverse=True)

    log.info(
        "find_contracts_done",
        underlying=underlying,
        option_type=option_type,
        total_chain=len(chain),
        matched=len(filtered),
    )
    return filtered
