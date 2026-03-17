"""APScheduler cron job definitions — all scheduled trading activities.

WHAT THIS MODULE DOES:
    Defines the daily schedule of automated trading activities using
    APScheduler's cron triggers.  Each job runs at a specific time
    (Eastern Time) on weekdays and calls a method on the TradingBot.

WHY IT EXISTS:
    Trading follows a strict daily cadence dictated by market hours:
    - Pre-market (before 9:30 AM ET): scan for candidates
    - Market open (9:30 AM ET): sync positions, cache equity
    - Morning (10:00 AM ET): evaluate signals and enter trades
    - Midday (12:00 PM ET): review open positions
    - Power hour (3:00 PM ET): look for late-day opportunities
    - End of day (3:45 PM ET): close day trades
    - After close (4:00 PM ET): generate daily summary

    This module translates that cadence into automated cron jobs.

KEY CONCEPTS:

    APScheduler:
        "Advanced Python Scheduler" — a library that runs functions at
        scheduled times, similar to Unix cron but within a Python process.
        It supports several trigger types:
        - CronTrigger: runs at specific times (like Unix crontab entries)
        - IntervalTrigger: runs every N seconds/minutes/hours
        - DateTrigger: runs once at a specific datetime

    CronTrigger:
        Fires at specified times, defined by hour, minute, day_of_week, etc.
        Examples:
        - CronTrigger(hour=9, minute=30) → fires at 9:30 AM every day
        - CronTrigger(hour="9-15", minute="*") → fires every minute from 9:00-3:59
        - day_of_week="mon-fri" → weekdays only (skip weekends)

    BackgroundScheduler:
        Runs the scheduler in a background thread, so the main thread
        remains free.  Jobs execute in the scheduler's thread pool.

    Market Hours (Eastern Time):
        - Pre-market:   4:00 AM - 9:30 AM  (limited trading, thinner liquidity)
        - Regular:      9:30 AM - 4:00 PM  (full trading hours)
        - After-hours:  4:00 PM - 8:00 PM  (limited trading)
        - Power Hour:   3:00 PM - 4:00 PM  (historically high volume/volatility)

KEY DESIGN DECISIONS:
    - All times are configured externally (in config.schedule) rather than
      hardcoded.  This makes it easy to adjust timing without code changes.
    - The position sync job runs every minute during market hours (9:00 AM
      to 3:59 PM) to keep the local database in sync with Alpaca.
    - TYPE_CHECKING import: the TradingBot import is only used for type
      hints, not at runtime.  This avoids circular import issues.

PYTHON PATTERNS:

    TYPE_CHECKING:
        `if TYPE_CHECKING:` is a block that only executes during static
        type analysis (e.g. mypy, IDE autocomplete) — NOT at runtime.
        This is used here to import TradingBot for type hints without
        creating a circular import (jobs.py → main.py → jobs.py).

    Tuple unpacking:
        `h, m = _parse_time("09:30")` assigns 9 to h and 30 to m in a
        single line.  This is Python's way of destructuring a tuple return.
"""

from __future__ import annotations

from datetime import datetime

# PYTHON PATTERN — TYPE_CHECKING:
# This constant is True during static analysis (mypy, IDE type checking)
# and False at runtime.  Imports inside this block are ONLY used for type
# annotations — they don't actually load the module.  This prevents
# circular imports: if main.py imports jobs.py, and jobs.py tried to
# import TradingBot from main.py at runtime, Python would fail with a
# circular import error.
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ai_trade.monitoring.logger import get_logger

if TYPE_CHECKING:
    from ai_trade.main import TradingBot

log = get_logger(__name__)

# Eastern Time — all market-related scheduling uses this timezone.
ET = ZoneInfo("America/New_York")


def create_scheduler(bot: TradingBot) -> BackgroundScheduler:
    """Create and configure the APScheduler with all trading jobs.

    This function builds the complete daily schedule.  Each job is
    registered with a CronTrigger that fires at a specific time on
    weekdays.  The `bot` parameter provides the callback methods that
    each job invokes.

    PYTHON PATTERN — BackgroundScheduler:
        The scheduler runs in its own background thread.  When you call
        scheduler.start(), it spawns a thread that sleeps until the next
        job is due, wakes up, runs the job, and goes back to sleep.  The
        main thread continues running normally.

    Args:
        bot: The TradingBot instance whose methods will be called by
             each scheduled job.

    Returns:
        A configured (but not yet started) BackgroundScheduler.
        The caller must call scheduler.start() to begin executing jobs.
    """
    # Create the scheduler with Eastern Time as the default timezone.
    # All CronTrigger times will be interpreted in ET.
    scheduler = BackgroundScheduler(timezone=ET)

    # Read schedule times from the bot's configuration.
    cfg = bot.cfg.schedule

    def _parse_time(t: str) -> tuple[int, int]:
        """Parse a "HH:MM" time string into (hour, minute) integers.

        PYTHON PATTERN — tuple return:
            Python functions can return multiple values as a tuple.
            The caller unpacks with: `h, m = _parse_time("09:30")`
            This assigns h=9 and m=30.

        Args:
            t: Time string in "HH:MM" format (e.g. "09:30", "15:45").

        Returns:
            Tuple of (hour, minute) as integers.
        """
        parts = t.split(":")
        return int(parts[0]), int(parts[1])

    # ── Job 1: Pre-market scan ───────────────────────────────
    # Scans the stock universe for trading candidates.  Runs before
    # market open (typically 8:00 AM ET) to give the bot time to
    # analyze results before the opening bell at 9:30 AM.
    h, m = _parse_time(cfg.premarket_scan)
    scheduler.add_job(
        bot.job_premarket_scan,      # The function to call
        CronTrigger(hour=h, minute=m, day_of_week="mon-fri", timezone=ET),
        id="premarket_scan",         # Unique identifier for this job
        name="Pre-market stock scan",  # Human-readable description
    )

    # ── Job 2: Market open setup ─────────────────────────────
    # Runs at market open (9:30 AM ET).  Caches the starting equity
    # (for daily loss limit calculation) and syncs positions with
    # Alpaca (in case stop-losses filled overnight).
    h, m = _parse_time(cfg.market_open)
    scheduler.add_job(
        bot.job_market_open,
        CronTrigger(hour=h, minute=m, day_of_week="mon-fri", timezone=ET),
        id="market_open",
        name="Market open setup",
    )

    # ── Job 3: Entry window ──────────────────────────────────
    # The main trading job.  Evaluates all strategy signals against
    # the morning's scan results and submits bracket orders for any
    # trades that pass risk checks.  Typically runs ~30 minutes after
    # open (10:00 AM ET) to let the opening volatility settle.
    h, m = _parse_time(cfg.entry_window)
    scheduler.add_job(
        bot.job_entry_window,
        CronTrigger(hour=h, minute=m, day_of_week="mon-fri", timezone=ET),
        id="entry_window",
        name="Strategy evaluation & trade entry",
    )

    # ── Job 4: Midday check ──────────────────────────────────
    # Reviews open positions around noon.  May tighten stops, take
    # partial profits, or close underperforming trades.
    h, m = _parse_time(cfg.midday_check)
    scheduler.add_job(
        bot.job_midday_check,
        CronTrigger(hour=h, minute=m, day_of_week="mon-fri", timezone=ET),
        id="midday_check",
        name="Midday position review",
    )

    # ── Job 5: Power hour scan ───────────────────────────────
    # TRADING CONCEPT — Power Hour:
    # The last hour of trading (3:00–4:00 PM ET) is called "power hour"
    # because trading volume and volatility typically spike as
    # institutional investors make end-of-day portfolio adjustments.
    # This creates additional trading opportunities.
    h, m = _parse_time(cfg.power_hour_scan)
    scheduler.add_job(
        bot.job_power_hour,
        CronTrigger(hour=h, minute=m, day_of_week="mon-fri", timezone=ET),
        id="power_hour_scan",
        name="Power hour scan",
    )

    # ── Job 6: End-of-day — close day trades ─────────────────
    # Closes all positions marked as "day" hold_type.  Runs 15 minutes
    # before market close (typically 3:45 PM ET) to ensure fills before
    # the 4:00 PM closing bell.  See execution/order_manager.py for the
    # close_all_day_trades logic.
    h, m = _parse_time(cfg.eod_close_day_trades)
    scheduler.add_job(
        bot.job_eod_close_day_trades,
        CronTrigger(hour=h, minute=m, day_of_week="mon-fri", timezone=ET),
        id="eod_close_day_trades",
        name="Close day trade positions",
    )

    # ── Job 7: End-of-day review ─────────────────────────────
    # Generates the daily performance summary, saves an equity snapshot,
    # and logs metrics.  Runs at or shortly after market close (4:00 PM ET).
    h, m = _parse_time(cfg.eod_review)
    scheduler.add_job(
        bot.job_eod_review,
        CronTrigger(hour=h, minute=m, day_of_week="mon-fri", timezone=ET),
        id="eod_review",
        name="End-of-day review",
    )

    # ── Job 8: Position sync (every minute during market hours) ──
    # Reconciles Alpaca positions with the local database every 60
    # seconds.  This catches stop-loss and take-profit fills that happen
    # between the scheduled check-in jobs.
    #
    # CronTrigger with hour="9-15" runs during hours 9:00 through 15:59
    # (3:59 PM).  minute="*" means every minute within those hours.
    scheduler.add_job(
        bot.job_sync_positions,
        CronTrigger(
            hour="9-15", minute="*", day_of_week="mon-fri", timezone=ET,
        ),
        id="position_sync",
        name="Sync positions with Alpaca",
    )

    log.info("scheduler_configured", job_count=len(scheduler.get_jobs()))
    return scheduler
