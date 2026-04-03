"""Real-time market data streaming via Alpaca's ``StockDataStream`` WebSocket.

This module provides a ``StreamManager`` class that maintains a persistent
WebSocket connection to Alpaca's real-time data feed, receiving minute-bar
updates as they happen.

How it works:
  1. You create a ``StreamManager`` and call ``subscribe(symbols, callback)``.
  2. Call ``start()`` to launch the WebSocket in a **daemon thread** — a
     background thread that automatically dies when the main program exits.
  3. As new minute bars arrive, your callback is called with the symbol
     and bar data.
  4. Call ``stop()`` for graceful shutdown.

This is used for real-time intraday monitoring (e.g. the VWAP strategy
needs live minute bars to detect VWAP reclaims as they happen).

Python-specific notes:
  - ``threading.Thread(daemon=True)``: A daemon thread runs in the
    background and is automatically killed when the main thread exits.
    This prevents the streaming thread from keeping the program alive
    after a Ctrl+C shutdown.
  - ``async def _bar_handler``: Alpaca's SDK expects an async callback
    for WebSocket events.  The handler converts the SDK bar object into
    a plain dict before passing it to your synchronous callback.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Callable

from ai_trade.clients import get_stream_client
from ai_trade.monitoring.logger import get_logger

logger = get_logger(__name__)


class StreamManager:
    """Manages a live WebSocket connection for real-time minute bars.

    Usage::

        manager = StreamManager(cfg)
        manager.subscribe(["AAPL", "MSFT"], on_bar=my_callback)
        manager.start()   # Runs in background daemon thread
        # ... later ...
        manager.stop()     # Graceful shutdown

    Args:
        cfg: Application config with Alpaca API credentials.
    """

    def __init__(self, cfg: SimpleNamespace) -> None:
        self._stream = get_stream_client(cfg)
        # Map of symbol → callback function for dispatching incoming bars
        self._callbacks: dict[str, Callable[[str, dict], None]] = {}
        self._thread: threading.Thread | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(
        self,
        symbols: list[str],
        on_bar: Callable[[str, dict], None],
    ) -> None:
        """Subscribe to minute bars for the given symbols.

        The ``on_bar`` callback receives two arguments:
          - ``symbol``: The stock ticker (e.g. "AAPL").
          - ``bar``:    A dict with keys: open, high, low, close, volume,
                        vwap, timestamp.

        Args:
            symbols: List of stock tickers to subscribe to.
            on_bar:  Callback function called for each new minute bar.
        """
        # Register the callback for each symbol
        for symbol in symbols:
            self._callbacks[symbol] = on_bar

        # Define the async handler that Alpaca's SDK will call when a
        # new bar arrives on the WebSocket.  This converts the SDK's bar
        # object into a plain dict and dispatches to our callback.
        async def _bar_handler(bar):  # noqa: ANN001
            symbol = bar.symbol
            bar_dict = {
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume),
                "vwap": float(bar.vwap),
                "timestamp": bar.timestamp.isoformat() if bar.timestamp else None,
            }
            callback = self._callbacks.get(symbol)
            if callback:
                try:
                    callback(symbol, bar_dict)
                except Exception:
                    logger.exception("bar_callback_error", symbol=symbol)

        # Tell the Alpaca SDK to route bars through our handler
        self._stream.subscribe_bars(_bar_handler, *symbols)
        logger.info("stream_subscribed", symbols=symbols)

    def unsubscribe(self, symbols: list[str]) -> None:
        """Remove bar subscriptions for the given symbols."""
        self._stream.unsubscribe_bars(*symbols)
        for symbol in symbols:
            self._callbacks.pop(symbol, None)
        logger.info("stream_unsubscribed", symbols=symbols)

    def start(self) -> None:
        """Launch the WebSocket connection in a background daemon thread.

        A daemon thread is used so that if the main program exits (e.g.
        via Ctrl+C), the streaming thread is automatically cleaned up
        without needing explicit shutdown logic.
        """
        if self._running:
            logger.warning("stream_already_running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("stream_started")

    def stop(self) -> None:
        """Stop the WebSocket connection gracefully.

        Signals the stream to disconnect, then waits up to 5 seconds for
        the background thread to finish.
        """
        if not self._running:
            return

        self._running = False
        try:
            self._stream.stop()
        except Exception:
            logger.exception("stream_stop_error")
        if self._thread is not None:
            self._thread.join(timeout=5)  # Wait up to 5s for clean shutdown
            self._thread = None
        logger.info("stream_stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Entry point for the daemon thread — runs the WebSocket event loop.

        Automatically reconnects on connection drops with exponential backoff.
        This ensures the streaming connection survives transient network issues.
        """
        max_reconnects = 10
        reconnect_delay = 2  # seconds, doubles each attempt

        for attempt in range(max_reconnects):
            if not self._running:
                break
            try:
                self._stream.run()  # Blocks until the stream is stopped
                break  # Clean exit — stop() was called
            except Exception as e:
                if not self._running:
                    break  # We're shutting down, don't reconnect

                wait = min(reconnect_delay * (2 ** attempt), 60)  # Cap at 60s
                logger.warning(
                    "stream_disconnected_reconnecting",
                    attempt=attempt + 1,
                    max_attempts=max_reconnects,
                    wait_seconds=wait,
                    error=str(e),
                )
                import time
                time.sleep(wait)

                # Get a fresh stream client for the reconnect attempt
                try:
                    from ai_trade.clients import get_stream_client
                    self._stream = get_stream_client()
                    # Re-subscribe all symbols with their callbacks
                    if self._callbacks:
                        async def _bar_handler(bar):
                            symbol = bar.symbol
                            bar_dict = {
                                "open": float(bar.open), "high": float(bar.high),
                                "low": float(bar.low), "close": float(bar.close),
                                "volume": int(bar.volume), "vwap": float(bar.vwap),
                                "timestamp": bar.timestamp.isoformat() if bar.timestamp else None,
                            }
                            callback = self._callbacks.get(symbol)
                            if callback:
                                try:
                                    callback(symbol, bar_dict)
                                except Exception:
                                    logger.exception("bar_callback_error", symbol=symbol)
                        self._stream.subscribe_bars(_bar_handler, *self._callbacks.keys())
                except Exception as re_err:
                    logger.error("stream_reconnect_setup_failed", error=str(re_err))
        else:
            logger.error("stream_max_reconnects_exceeded", max_attempts=max_reconnects)

        self._running = False
