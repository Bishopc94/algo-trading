"""Alpaca client factory — singleton instances for trading, data, and streaming.

This module implements the **Singleton pattern**: Alpaca API clients are
created once at startup (``init_clients()``) and then retrieved via
accessor functions (``get_trading_client()``, ``get_data_client()``).

Why singletons?  Alpaca clients manage HTTP connections and authentication.
Creating one per call would waste resources and could hit rate limits.
Instead we create them once and reuse them everywhere.

Python-specific notes:
  - ``global`` keyword: In Python, a function can *read* module-level
    variables freely, but to *reassign* them it must declare ``global``.
  - Module-level variables (``_trading_client``, etc.) persist for the
    lifetime of the process — they act like static fields in other languages.

There are four types of Alpaca clients:
  1. **TradingClient** — submits/cancels orders, fetches account info and
     positions.
  2. **StockHistoricalDataClient** — fetches historical OHLCV bars, quotes,
     and snapshots.
  3. **NewsClient** — fetches news articles for sentiment analysis.
  4. **StockDataStream** — WebSocket connection for real-time streaming bars.
     Unlike the other two, streaming clients are NOT cached because each
     WebSocket connection is a separate session.
"""

from __future__ import annotations

from types import SimpleNamespace

from alpaca.data import NewsClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.live.stock import StockDataStream
from alpaca.trading.client import TradingClient
from requests.adapters import HTTPAdapter

from ai_trade.monitoring.logger import get_logger


class _TimeoutAdapter(HTTPAdapter):
    """Injects a default (connect, read) timeout on every request.

    The Alpaca SDK does not expose a timeout parameter, so without this the
    socket will block until the OS TCP timeout fires (~2 minutes on Windows).
    Mounting this adapter ensures transient API outages fail fast instead of
    stalling the trading cycle.
    """

    def __init__(self, *args, timeout: tuple[int, int] = (10, 30), **kwargs) -> None:
        self._default_timeout = timeout
        super().__init__(*args, **kwargs)

    def send(self, *args, **kwargs):  # type: ignore[override]
        kwargs.setdefault("timeout", self._default_timeout)
        return super().send(*args, **kwargs)

log = get_logger(__name__)

# Module-level singletons — set by init_clients(), read by get_*() functions.
# Starting as None ensures a clear error if someone forgets to initialise.
_trading_client: TradingClient | None = None
_data_client: StockHistoricalDataClient | None = None
_news_client: NewsClient | None = None
_cfg: SimpleNamespace | None = None  # Cached config for creating stream clients


def init_clients(cfg: SimpleNamespace) -> None:
    """Create and cache the Alpaca API client singletons.

    Must be called exactly once at startup (from ``TradingBot.start()``).
    After this call, ``get_trading_client()`` and ``get_data_client()``
    will return the cached instances.

    Args:
        cfg: The application configuration (from ``load_config()``).
             Must have ``cfg.alpaca.api_key``, ``cfg.alpaca.secret_key``,
             and ``cfg.alpaca.paper`` (True for paper trading).
    """
    global _trading_client, _data_client, _news_client, _cfg
    _cfg = cfg

    # TradingClient handles order submission, account queries, position queries.
    # The ``paper=True`` flag routes all requests to Alpaca's paper-trading
    # environment so no real money is at risk.
    _trading_client = TradingClient(
        cfg.alpaca.api_key, cfg.alpaca.secret_key, paper=cfg.alpaca.paper,
    )

    # StockHistoricalDataClient handles fetching historical bars, quotes,
    # and snapshots.  It does NOT require the paper flag because market data
    # is the same for paper and live accounts.
    _data_client = StockHistoricalDataClient(
        cfg.alpaca.api_key, cfg.alpaca.secret_key,
    )

    # NewsClient handles fetching news articles for sentiment analysis.
    # Separate from StockHistoricalDataClient — news lives on its own endpoint.
    _news_client = NewsClient(
        cfg.alpaca.api_key, cfg.alpaca.secret_key,
    )

    # Mount timeout adapters on all three clients.  The Alpaca SDK leaves
    # connect_timeout=None by default, which means the OS TCP timeout (~2 min
    # on Windows) is the only circuit breaker.  10s connect / 30s read gives
    # the API a reasonable window while failing fast on outages.
    _adapter = _TimeoutAdapter(timeout=(10, 30))
    for _client in (_trading_client, _data_client, _news_client):
        _client._session.mount("https://", _adapter)
        _client._session.mount("http://", _adapter)

    log.info("clients_initialized", paper=cfg.alpaca.paper)


def get_trading_client() -> TradingClient:
    """Return the cached TradingClient (for orders, account info, positions)."""
    if _trading_client is None:
        raise RuntimeError("Call init_clients(cfg) before using get_trading_client()")
    return _trading_client


def get_data_client() -> StockHistoricalDataClient:
    """Return the cached StockHistoricalDataClient (for bars, quotes, snapshots)."""
    if _data_client is None:
        raise RuntimeError("Call init_clients(cfg) before using get_data_client()")
    return _data_client


def get_news_client() -> NewsClient:
    """Return the cached NewsClient (for fetching news articles)."""
    if _news_client is None:
        raise RuntimeError("Call init_clients(cfg) before using get_news_client()")
    return _news_client


def get_stream_client(cfg: SimpleNamespace | None = None) -> StockDataStream:
    """Create and return a NEW StockDataStream (WebSocket connection).

    Unlike the other clients, stream clients are NOT cached — each call
    creates a fresh WebSocket connection.  This is because WebSocket
    connections are stateful and each one subscribes to different symbols.

    Args:
        cfg: Optional config override.  If None, uses the config from
             ``init_clients()``.
    """
    c = cfg or _cfg
    if c is None:
        raise RuntimeError("Call init_clients(cfg) before using get_stream_client()")
    return StockDataStream(c.alpaca.api_key, c.alpaca.secret_key)


def get_account():
    """Fetch current account information (equity, cash, buying power, etc.).

    Wrapped in ``retry_api_call`` so a transient ``ConnectTimeout`` gets up
    to two exponential-backoff retries before bubbling up.  This is called
    by the main scan loop and the PDT manager — a single network hiccup
    should not skip a trading cycle.
    """
    from ai_trade.utils import retry_api_call
    return retry_api_call(get_trading_client().get_account)
