"""Shared utilities used across the ai_trade package.

Contains helper functions that don't belong to any single module:
  - ``retry_api_call()`` — resilient API calls with exponential backoff
  - ``extract_greeks()`` — normalises Alpaca option snapshot data
"""

from __future__ import annotations

import time

from ai_trade.monitoring.logger import get_logger

log = get_logger(__name__)

# Retry configuration
_MAX_RETRIES = 3       # Total attempts before giving up
_BACKOFF_BASE = 2      # Base for exponential delay: 2^1=2s, 2^2=4s, 2^3=8s


def retry_api_call(func, *args, **kwargs):
    """Execute *func* with exponential-backoff retry on transient API errors.

    If the call fails, waits ``2^attempt`` seconds before retrying.  After
    3 failed attempts the exception is re-raised to the caller.

    Rate-limit aware: if Alpaca returns a 429 (rate limited), we respect
    the Retry-After header or wait longer than the default backoff.

    Non-retryable errors (auth failures, invalid requests) are raised
    immediately without wasting retry attempts.

    Args:
        func:    The callable to execute (e.g. ``client.get_account``).
        *args:   Positional arguments forwarded to *func*.
        **kwargs: Keyword arguments forwarded to *func*.

    Returns:
        Whatever *func* returns on a successful call.

    Raises:
        The original exception from *func* after all retries are exhausted.
    """
    func_name = getattr(func, "__name__", str(func))

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            error_msg = str(exc).lower()

            # Non-retryable errors — fail immediately, don't waste retries
            if any(kw in error_msg for kw in [
                "forbidden", "unauthorized", "invalid",
                "pattern day trading", "insufficient",
                "not found", "40110000", "40310100",
            ]):
                log.error("api_call_non_retryable", func=func_name,
                          error=str(exc), attempt=attempt)
                raise

            # On the last attempt, don't retry — propagate the error
            if attempt == _MAX_RETRIES:
                log.error("api_call_failed_after_retries", func=func_name,
                          error=str(exc), attempts=_MAX_RETRIES)
                raise

            # Rate-limit handling: wait longer for 429s
            if "429" in error_msg or "rate" in error_msg or "too many" in error_msg:
                wait = max(30, _BACKOFF_BASE ** attempt * 5)
                log.warning("api_rate_limited", func=func_name, attempt=attempt,
                            wait=wait)
            else:
                # Standard exponential backoff: 2s → 4s → 8s
                wait = _BACKOFF_BASE ** attempt

            log.warning(
                "api_retry",
                func=func_name,
                attempt=attempt,
                wait=wait,
                error=str(exc),
            )
            time.sleep(wait)


def extract_greeks(snapshot: dict | object) -> dict:
    """Extract options Greeks from an Alpaca snapshot into a flat dictionary.

    Alpaca returns option snapshots in two possible formats depending on
    context:
      1. **Raw dict** (from the REST API or backtest engine):
         ``{"delta": 0.45, "theta": -0.03, ...}``
      2. **SDK object** (from the ``alpaca-py`` SDK):
         ``snapshot.greeks.delta``, ``snapshot.greeks.theta``, etc.

    This function normalises both formats into a consistent dict::

        {"delta": 0.45, "gamma": 0.02, "theta": -0.03, "vega": 0.08,
         "implied_volatility": 0.35}

    Args:
        snapshot: Either a raw dict or an Alpaca SDK snapshot object.

    Returns:
        A flat dict with keys: delta, gamma, theta, vega, implied_volatility.
        Missing values default to 0.0.
    """
    # Case 1: Raw dict (common in backtesting and when using raw API responses)
    if isinstance(snapshot, dict):
        return {
            "delta": snapshot.get("delta", 0.0),
            "gamma": snapshot.get("gamma", 0.0),
            "theta": snapshot.get("theta", 0.0),
            "vega": snapshot.get("vega", 0.0),
            "implied_volatility": snapshot.get("implied_volatility", 0.0),
        }

    # Case 2: SDK object — greeks are nested under a .greeks attribute.
    # getattr() safely accesses attributes that may not exist, returning
    # None as a default (similar to optional chaining ?. in other languages).
    greeks = getattr(snapshot, "greeks", None)
    return {
        "delta": float(greeks.delta) if greeks and getattr(greeks, "delta", None) is not None else 0.0,
        "gamma": float(greeks.gamma) if greeks and getattr(greeks, "gamma", None) is not None else 0.0,
        "theta": float(greeks.theta) if greeks and getattr(greeks, "theta", None) is not None else 0.0,
        "vega": float(greeks.vega) if greeks and getattr(greeks, "vega", None) is not None else 0.0,
        "implied_volatility": float(getattr(snapshot, "implied_volatility", 0) or 0),
    }
