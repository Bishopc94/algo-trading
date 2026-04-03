"""Historical market data fetching via the Alpaca-py SDK.

This module provides functions to fetch OHLCV (Open, High, Low, Close,
Volume) bar data from Alpaca's market data API.  It supports both single
and multi-symbol requests and is the primary data source for all
strategy evaluations.

Key design decisions:
  - All requests use the **IEX data feed** (free tier) rather than SIP
    (paid).  IEX covers most US equities and is sufficient for a $500
    paper-trading account.
  - Every API call is wrapped in ``retry_api_call()`` for resilience
    against transient failures (rate limits, network issues).
  - Multi-symbol requests are batched into a single API call rather than
    one call per symbol — this is much faster and more efficient.

Python-specific notes:
  - ``pd.MultiIndex``: When fetching bars for multiple symbols, Alpaca
    returns a DataFrame with a two-level index: (symbol, timestamp).
    We split this back into per-symbol DataFrames using ``.xs()``
    (cross-section).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame

from ai_trade.clients import get_data_client
from ai_trade.monitoring.logger import get_logger
from ai_trade.utils import retry_api_call as _retry

logger = get_logger(__name__)


def fetch_bars(
    symbol: str,
    timeframe: TimeFrame,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch historical OHLCV bars for a single symbol.

    Returns an empty DataFrame on any failure (network, rate limit, etc.)
    rather than propagating exceptions — the caller can check df.empty.
    """
    columns = ["open", "high", "low", "close", "volume", "vwap", "trade_count"]

    try:
        client = get_data_client()
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        logger.info("fetch_bars", symbol=symbol, timeframe=str(timeframe), start=str(start), end=str(end))
        bars = _retry(client.get_stock_bars, request)

        df = bars.df
        if df is None or df.empty:
            logger.warning("fetch_bars_empty", symbol=symbol)
            return pd.DataFrame(columns=columns)

        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel("symbol")

        available = [c for c in columns if c in df.columns]
        df = df[available]

        logger.info("fetch_bars_done", symbol=symbol, rows=len(df))
        return df

    except Exception as e:
        logger.warning("fetch_bars_failed", symbol=symbol, error=str(e))
        return pd.DataFrame(columns=columns)


def fetch_bars_multi(
    symbols: list[str],
    timeframe: TimeFrame,
    start: datetime,
    end: datetime,
) -> dict[str, pd.DataFrame]:
    """Fetch bars for multiple symbols in a single API call.

    Returns a dict mapping each symbol to its DataFrame.  On complete
    failure, returns empty DataFrames for all symbols rather than raising.
    """
    columns = ["open", "high", "low", "close", "volume", "vwap", "trade_count"]

    if not symbols:
        return {}

    try:
        client = get_data_client()
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        logger.info("fetch_bars_multi", symbols_count=len(symbols), timeframe=str(timeframe))
        bars = _retry(client.get_stock_bars, request)

        result: dict[str, pd.DataFrame] = {}
        df = bars.df

        if df is None or df.empty:
            logger.warning("fetch_bars_multi_empty_response", symbols_count=len(symbols))
            return {s: pd.DataFrame(columns=columns) for s in symbols}

        if isinstance(df.index, pd.MultiIndex):
            for symbol in symbols:
                try:
                    symbol_df = df.xs(symbol, level="symbol")
                    available = [c for c in columns if c in symbol_df.columns]
                    result[symbol] = symbol_df[available]
                except KeyError:
                    result[symbol] = pd.DataFrame(columns=columns)
        else:
            available = [c for c in columns if c in df.columns]
            if symbols:
                result[symbols[0]] = df[available]

        # Ensure every requested symbol has an entry (even if empty)
        for s in symbols:
            if s not in result:
                result[s] = pd.DataFrame(columns=columns)

        logger.info("fetch_bars_multi_done", symbols_with_data=sum(1 for v in result.values() if not v.empty),
                     total_rows=sum(len(v) for v in result.values()))
        return result

    except Exception as e:
        logger.warning("fetch_bars_multi_failed", error=str(e), symbols_count=len(symbols))
        return {s: pd.DataFrame(columns=columns) for s in symbols}


def fetch_snapshots(symbols: list[str]) -> dict:
    """Fetch current market snapshots (latest price, volume, etc.) for symbols.

    Returns an empty dict on failure rather than raising.
    """
    if not symbols:
        return {}
    try:
        client = get_data_client()
        request = StockSnapshotRequest(symbol_or_symbols=symbols, feed=DataFeed.IEX)
        logger.info("fetch_snapshots", symbols_count=len(symbols))
        snapshots = _retry(client.get_stock_snapshot, request)
        logger.info("fetch_snapshots_done", count=len(snapshots))
        return snapshots
    except Exception as e:
        logger.warning("fetch_snapshots_failed", error=str(e), symbols_count=len(symbols))
        return {}


def fetch_latest_quotes(symbols: list[str]) -> dict:
    """Fetch the latest bid/ask quotes for a list of symbols.

    Returns an empty dict on failure rather than raising.
    """
    if not symbols:
        return {}
    try:
        client = get_data_client()
        request = StockLatestQuoteRequest(symbol_or_symbols=symbols, feed=DataFeed.IEX)
        logger.info("fetch_latest_quotes", symbols_count=len(symbols))
        quotes = _retry(client.get_stock_latest_quote, request)
        logger.info("fetch_latest_quotes_done", count=len(quotes))
        return quotes
    except Exception as e:
        logger.warning("fetch_latest_quotes_failed", error=str(e), symbols_count=len(symbols))
        return {}
