"""Structured logging setup using structlog.

WHAT THIS MODULE DOES:
    Configures the application-wide logging system with three output
    destinations:
    1. Console — human-readable colored output for development.
    2. ai_trade.log — DEBUG-level JSON lines for full diagnostic detail.
    3. ai_trade_run.log — INFO-level JSON lines as a "decision journal"
       for post-session review.

WHY IT EXISTS:
    In a trading bot, logging is not optional — it's essential for:
    - Debugging: understanding why a trade was taken or rejected.
    - Auditing: reviewing all decisions after the trading day ends.
    - Monitoring: detecting errors or unexpected behavior in real-time.

    We use "structured logging" (key=value pairs) instead of plain text
    messages.  Structured logs are both human-readable AND machine-parseable,
    making it easy to filter and search (e.g. "show me all logs where
    symbol=AAPL and event=trade_approved").

KEY CONCEPTS:

    structlog:
        A Python library that wraps the standard `logging` module with
        structured logging capabilities.  Instead of:
            logger.info("Trade approved for AAPL, 100 shares at $150")
        You write:
            logger.info("trade_approved", symbol="AAPL", shares=100, price=150)
        The output format (JSON, key=value, colored console) is configured
        separately from the log call itself.

    Log Levels (from most to least verbose):
        - DEBUG:   Detailed diagnostic info (position sizing math, API responses)
        - INFO:    Normal operations (trade submitted, scan complete)
        - WARNING: Something unexpected but recoverable (API retry, stale trade)
        - ERROR:   Something failed (order rejected, connection lost)
        - CRITICAL: System is unusable (should trigger alerts)

    Processors:
        structlog uses a "processor chain" — a pipeline of functions that
        transform log entries before they're output.  Each processor adds
        or modifies fields (e.g., adding a timestamp, adding the log level).

KEY DESIGN DECISIONS:
    - Console output uses colored rendering in TTY mode (interactive
      terminal) and JSON in non-TTY mode (e.g., when piped to a file or
      running in Docker).
    - APScheduler's verbose "Running job... / executed successfully"
      messages are silenced to WARNING level to keep logs clean.
    - Log files are stored at <project_root>/logs/.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import structlog

from ai_trade._version import __version__

_ET = ZoneInfo("America/New_York")


def _add_et_timestamp(logger, method_name, event_dict):
    """Stamp each log entry with an Eastern Time ISO-8601 timestamp.

    We deliberately use America/New_York (not UTC and not machine local
    time) so logs read naturally against the market clock regardless of
    where the bot is running. The offset suffix (-05:00 / -04:00) keeps
    each line unambiguous across the DST boundary.
    """
    event_dict["timestamp"] = datetime.now(_ET).isoformat(timespec="milliseconds")
    return event_dict

# Resolve the log directory: 3 levels up from this file, then into logs/.
_LOG_DIR = Path(__file__).resolve().parents[3] / "logs"


def setup_logging(level: str = "INFO") -> None:
    """Configure structlog with JSON file output and human-readable console.

    This function should be called ONCE at application startup, before any
    logging is done.  It sets up the standard library logging handlers and
    the structlog processor pipeline.

    Creates two log files:
    - ai_trade.log     — DEBUG-level JSON lines (full diagnostic detail)
    - ai_trade_run.log — INFO-level JSON lines (decision journal for review)

    Args:
        level: Console log level as a string (e.g. "INFO", "DEBUG").
               File handlers always log at their configured levels regardless
               of this parameter.
    """
    # Create the logs directory if it doesn't exist.
    _LOG_DIR.mkdir(exist_ok=True)

    # ── Standard library logging handlers ────────────────────
    # These are the "output destinations" for log messages.

    # File handler — full DEBUG JSON lines for diagnostics.
    # Every log message at every level is written here.
    file_handler = logging.FileHandler(_LOG_DIR / "ai_trade.log")
    file_handler.setLevel(logging.DEBUG)

    # Run log — INFO-level decision journal for post-run analysis.
    # Only significant events (not debug noise) go here.
    run_handler = logging.FileHandler(_LOG_DIR / "ai_trade_run.log")
    run_handler.setLevel(logging.INFO)

    # Console handler — human-readable output to stdout.
    # Level is configurable (default INFO, can be set to DEBUG for development).
    console_handler = logging.StreamHandler(sys.stdout)
    # getattr(logging, "INFO") returns the integer constant 20.
    # This converts the string "INFO" to the logging module's constant.
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    # PYTHON PATTERN — logging.basicConfig:
    # Configures the root logger (the parent of all loggers).  All loggers
    # created with get_logger() inherit this configuration.
    # - format="%(message)s": structlog handles formatting, so we just
    #   pass through the message as-is.
    # - level=DEBUG: the root logger accepts ALL messages; individual
    #   handlers filter by their own levels.
    # - handlers: list of destinations for log output.
    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG,
        handlers=[file_handler, run_handler, console_handler],
    )

    # Quiet the noisy APScheduler loggers.  Without this, APScheduler
    # prints "Running job..." and "executed successfully" for EVERY job
    # execution, which clutters the logs every 60 seconds.
    logging.getLogger("apscheduler.executors.default").setLevel(logging.WARNING)
    logging.getLogger("apscheduler.scheduler").setLevel(logging.WARNING)

    # ── structlog configuration ──────────────────────────────
    # PYTHON PATTERN — structlog.configure:
    # Sets up the global structlog behavior.  The `processors` list defines
    # a pipeline that each log entry passes through before being output.
    def _add_version(logger, method_name, event_dict):
        """Stamp every log entry with the application version."""
        event_dict["version"] = __version__
        return event_dict

    structlog.configure(
        processors=[
            # merge_contextvars: merges any "context variables" (thread-local
            # key-value pairs) into each log entry.  Useful for adding
            # request IDs or session IDs to all logs within a scope.
            structlog.contextvars.merge_contextvars,

            # add_log_level: adds a "level" key (e.g., "info", "debug")
            # to each log entry.
            structlog.processors.add_log_level,

            # Eastern-time ISO-8601 timestamp (market clock).
            _add_et_timestamp,

            # Add application version to every log entry for traceability.
            _add_version,

            # Final renderer: choose format based on whether stdout is a
            # terminal (TTY) or not.
            # - TTY (interactive terminal): colored, human-readable output.
            # - Non-TTY (piped/file/Docker): JSON for machine parsing.
            #
            # PYTHON PATTERN — ternary expression:
            # `a if condition else b` is Python's inline if/else.
            # sys.stdout.isatty() returns True if stdout is a terminal.
            structlog.dev.ConsoleRenderer() if sys.stdout.isatty() else structlog.processors.JSONRenderer(),
        ],
        # LoggerFactory: creates loggers that bridge to Python's standard
        # logging module, so structlog and standard logging share the same
        # handlers (file, console, etc.).
        logger_factory=structlog.stdlib.LoggerFactory(),

        # BoundLogger: the class used for logger instances.  "Bound" means
        # you can add persistent key-value pairs to a logger instance that
        # appear in every subsequent log call from that logger.
        wrapper_class=structlog.stdlib.BoundLogger,

        # Performance optimization: after the first log call on a logger,
        # the processor chain is cached and reused.
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named logger.

    The `name` parameter is typically `__name__`, which Python automatically
    sets to the fully-qualified module path (e.g., "ai_trade.risk.pdt_manager").
    This makes it easy to identify which module produced a log message.

    Usage:
        log = get_logger(__name__)
        log.info("trade_submitted", symbol="AAPL", shares=100)

    Args:
        name: Logger name (conventionally the module's __name__).

    Returns:
        A structlog BoundLogger instance.
    """
    return structlog.get_logger(name)
