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

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ai_trade.monitoring.logger import get_logger
from ai_trade.monitoring import console as con

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
    # misfire_grace_time=300 means a job can fire up to 5 minutes late
    # (e.g. if the system clock drifts or a previous job runs long)
    # rather than being silently skipped.
    scheduler = BackgroundScheduler(
        timezone=ET,
        job_defaults={"misfire_grace_time": 300, "coalesce": True},
    )

    # Error listener — logs job failures without crashing the scheduler.
    # Without this, an unhandled exception in a job would be silently
    # swallowed by APScheduler's default handler.
    def _job_error_listener(event):
        if event.exception:
            log.error(
                "scheduled_job_failed",
                job_id=event.job_id,
                error=str(event.exception),
                error_type=type(event.exception).__name__,
            )
            print(con.error(f"Scheduler job '{event.job_id}' FAILED: {event.exception}"))
        else:
            log.warning("scheduled_job_missed", job_id=event.job_id)

    scheduler.add_listener(_job_error_listener, EVENT_JOB_ERROR | EVENT_JOB_MISSED)

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

    # ── Job 3: Continuous scan & evaluate (every 15 min during market hours) ──
    # Replaces the old fixed-time entry windows (9:35, 11:00, 12:00, 3:00).
    # Scans for fresh candidates and evaluates all strategies every 15 minutes
    # from 9:45 AM through 3:45 PM.  The algo decides when setups are
    # worth entering — not the schedule.
    #
    # Starts at 9:45 (15 min after open) to let opening volatility settle.
    # Stops at 3:45 to leave time for EOD close at 3:50.
    scan_interval = getattr(cfg, "scan_interval_minutes", 15)
    scheduler.add_job(
        bot.job_scan_and_evaluate,
        CronTrigger(
            hour="9-15",
            minute=f"*/{scan_interval}",
            day_of_week="mon-fri",
            timezone=ET,
        ),
        id="scan_and_evaluate",
        name=f"Scan & evaluate (every {scan_interval}min)",
    )

    # ── Job 9: Options expiry management ──────────────────────
    # Closes options positions expiring today or tomorrow.  Avoids
    # assignment risk on short options and gamma/liquidity risk on
    # all options near expiration.
    h, m = _parse_time(cfg.options_expiry_check)
    scheduler.add_job(
        bot.job_options_expiry_check,
        CronTrigger(hour=h, minute=m, day_of_week="mon-fri", timezone=ET),
        id="options_expiry_check",
        name="Close expiring options",
    )

    # ── Job 10: End-of-day — close day trades ─────────────────
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

    # ── Job 14: Trailing stop update (every 5 min during market hours) ──
    # Walks open stock trades and tightens bracket-order stops when price
    # has moved in our favor (breakeven trigger or chandelier trail).
    # See main.py::job_update_trailing_stops for the logic.
    scheduler.add_job(
        bot.job_update_trailing_stops,
        CronTrigger(
            hour="9-15", minute="*/5", day_of_week="mon-fri", timezone=ET,
        ),
        id="trailing_stops",
        name="Advance trailing stops on open winners",
    )

    # ── Job 13: Options position sync (every 5 min during market hours) ──
    # Reconciles options positions with Alpaca's broker-side state.
    # Less frequent than stock sync (every minute) since options fills
    # are rarer but still need timely detection.
    scheduler.add_job(
        bot.job_options_position_sync,
        CronTrigger(
            hour="9-15", minute="*/5", day_of_week="mon-fri", timezone=ET,
        ),
        id="options_position_sync",
        name="Sync options positions with Alpaca",
    )

    # ── Job 16: Self-learning analysis sweep (V2 Phase 6) ──
    # Post-close loss-pattern scan + parameter optimizer review.  Runs
    # after eod_review so the freshly-closed trades and today's snapshot
    # are already committed to the database.  Proposals are logged
    # regardless; they only become active on next restart when the
    # ``apply_parameter_changes`` toggle is enabled in config.
    eod_analysis_time = getattr(cfg, "eod_analysis", "16:10")
    h, m = _parse_time(eod_analysis_time)
    scheduler.add_job(
        bot.job_eod_analysis,
        CronTrigger(hour=h, minute=m, day_of_week="mon-fri", timezone=ET),
        id="eod_analysis",
        name="Self-learning trade analysis sweep",
    )

    # ── Job 15: ML training (post-close, V2 Phase 5) ──
    # Retrains the signal-quality classifier after the market closes
    # using all closed-trade feature snapshots captured during the
    # session.  Cold-start safe — returns insufficient_data until the
    # training set has at least 30 labelled trades.  The predictor
    # does NOT swap models mid-session; the next bot restart picks up
    # the newly active version.
    ml_training_time = getattr(cfg, "ml_training", "17:00")
    h, m = _parse_time(ml_training_time)
    scheduler.add_job(
        bot.job_train_ml_models,
        CronTrigger(hour=h, minute=m, day_of_week="mon-fri", timezone=ET),
        id="ml_training",
        name="Retrain ML signal-quality classifier",
    )

    # ── Job 17: 0DTE position monitor (every 2 min during market hours) ──
    # Manages intraday lifecycle of 0DTE options: hard time exit at 15:30,
    # loss cut at -50%, trailing stop at 50% of peak profit.
    scheduler.add_job(
        bot.job_monitor_zero_dte,
        CronTrigger(
            hour="9-15", minute="*/2", day_of_week="mon-fri", timezone=ET,
        ),
        id="zero_dte_monitor",
        name="Monitor 0DTE options positions",
    )

    # ── Job 18: Database cleanup (weekly Sunday midnight) ──
    # Purges high-volume tables (decisions, scanner_results, ml_features,
    # ml_predictions) older than 90 days to keep the DB lean.
    scheduler.add_job(
        bot.job_cleanup_database,
        CronTrigger(
            hour=0, minute=0, day_of_week="sun", timezone=ET,
        ),
        id="db_cleanup",
        name="Database cleanup (90-day retention)",
    )

    log.info("scheduler_configured", job_count=len(scheduler.get_jobs()))
    return scheduler
