"""Main entry point — the TradingBot orchestrator.

This is the central nervous system of the trading bot.  The ``TradingBot``
class owns all components (scanner, strategies, risk manager, order manager,
etc.) and defines the scheduled jobs that run throughout the trading day.

Lifecycle:
  1. ``TradingBot.__init__()`` — loads config, creates all components
  2. ``TradingBot.start()`` — authenticates with Alpaca, starts the
     APScheduler, and blocks the main thread until Ctrl+C
  3. Scheduled jobs fire throughout the day (scan, evaluate, trade, close)
  4. ``TradingBot.stop()`` — graceful shutdown

The main evaluation pipeline runs twice per job invocation:
  - ``_evaluate_and_trade()`` — runs stock strategies on candidates
  - ``_evaluate_options()`` — runs options strategies on candidates

Both pipelines follow the same pattern:
  1. Gate: check market regime (is it safe to trade?)
  2. Gate: check daily loss limit (haven't lost too much today?)
  3. Fetch price data for all candidates
  4. Run every enabled strategy on every candidate
  5. Rank signals, apply sentiment modifiers
  6. Submit orders for approved trades

Python-specific notes for non-Python readers:
  - ``from __future__ import annotations`` makes type hints like
    ``str | None`` work as strings at runtime (needed for older Python).
  - ``signal.signal(signal.SIGINT, handler)`` registers a Unix signal
    handler — the Python equivalent of trapping Ctrl+C.
  - ``ZoneInfo("America/New_York")`` creates a timezone object for
    Eastern Time.  All schedule times are in ET because the US stock
    market operates on Eastern Time.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.timeframe import TimeFrame

from ai_trade._version import __version__
from ai_trade.config import load_config
from ai_trade.clients import init_clients, get_trading_client, get_account
from ai_trade.data.historical import fetch_bars_multi
from ai_trade.data.indicators import add_all
from ai_trade.monitoring.database import Database
from ai_trade.state_persistence import (
    apply_parameter_overrides,
    detect_offline_gap,
    get_current_regime,
    log_boot_summary,
    record_current_regime,
    record_shutdown,
    record_startup,
)
from ai_trade.ml.predictor import SignalQualityPredictor
from ai_trade.ml.features import extract_features
from ai_trade.ml.trainer import train_signal_quality_model
from ai_trade.analysis.post_trade import analyze_and_persist as analyze_closed_trade_and_persist
from ai_trade.analysis.loss_patterns import scan_loss_patterns
from ai_trade.analysis.parameter_optimizer import review_and_adjust as review_parameter_adjustments
from ai_trade.monitoring.logger import setup_logging, get_logger
from ai_trade.monitoring import console as con
from ai_trade.monitoring.notifier import (
    notify_high_conviction_signal,
    notify_options_order,
    notify_stock_order,
    notify_stock_order_failed,
    notify_trade_exit,
    notify_trailing_stop_update,
)
from ai_trade.monitoring.performance import PerformanceTracker
from ai_trade.risk.dynamic_risk import DynamicRiskController
from ai_trade.risk.pdt_manager import PDTManager
from ai_trade.risk.position_sizer import PositionSizer
from ai_trade.risk.risk_manager import RiskManager
from ai_trade.risk.smart_pdt import SmartPDTPlanner
from ai_trade.execution.order_manager import OrderManager
from ai_trade.scanner.screener import StockScreener
from ai_trade.strategy.base import HoldType
from ai_trade.strategy.bb_squeeze import BBSqueezeStrategy
from ai_trade.strategy.ema_crossover import EMACrossoverStrategy
from ai_trade.strategy.macd_divergence import MACDDivergenceStrategy
from ai_trade.strategy.mean_reversion import MeanReversionStrategy
from ai_trade.strategy.momentum import MomentumStrategy
from ai_trade.strategy.orb import ORBStrategy
from ai_trade.strategy.pullback import PullbackStrategy
from ai_trade.strategy.vwap import VWAPStrategy
from ai_trade.strategy.signal import SignalAggregator
from ai_trade.strategy.weighter import StrategyWeighter
from ai_trade.scheduler.jobs import create_scheduler

# Options imports
from ai_trade.data.options_chain import get_options_chain, get_options_snapshot
from ai_trade.execution.options_order_manager import OptionsOrderManager
from ai_trade.strategy.options.credit_put_spread import CreditPutSpreadStrategy
from ai_trade.strategy.options.debit_call_spread import DebitCallSpreadStrategy
from ai_trade.strategy.options.long_call import LongCallStrategy
from ai_trade.strategy.options.long_put import LongPutStrategy
from ai_trade.strategy.options.cash_secured_put import CashSecuredPutStrategy
from ai_trade.strategy.options.covered_call import CoveredCallStrategy
from ai_trade.strategy.options.covered_straddle import CoveredStraddleStrategy
from ai_trade.strategy.options.momentum_options import MomentumOptionsStrategy
from ai_trade.strategy.options.zero_dte import ZeroDTEStrategy

# Sentiment imports
from ai_trade.sentiment.market_regime import MarketRegimeAnalyzer, MarketRegime
from ai_trade.sentiment.news_sentiment import NewsSentimentScanner
from ai_trade.sentiment.earnings_guard import EarningsGuard
from ai_trade.sentiment.economic_calendar import (
    get_events_for_date,
    get_high_impact_events,
    is_high_impact_day,
    conviction_modifier_for_events,
)

# V2: Decision audit trail
from ai_trade.monitoring.decision_logger import DecisionLogger

# V2 Phase 13: Cycle timing
from ai_trade.monitoring.cycle_timer import CycleTimer

# V2 Phase 12: Market prediction
from ai_trade.analysis.market_prediction import (
    compute_momentum_score,
    momentum_conviction_modifier,
)

log = get_logger(__name__)

# All schedule times and market hours use Eastern Time
ET = ZoneInfo("America/New_York")

# Minimum daily bars needed for indicators (Bollinger needs 20, RSI needs 14)
_MIN_BARS = 21


class TradingBot:
    """Central orchestrator — owns all components and defines scheduled jobs.

    This class is the "glue" that connects the scanner, strategies, risk
    manager, order manager, and scheduler into a cohesive trading system.
    Each component is independently testable, but this class coordinates
    them into the full pipeline.
    """

    def __init__(self, config_path: str | None = None, dry_run: bool = False):
        self.cfg = load_config(config_path)
        self.dry_run = dry_run  # If True, log signals without submitting orders

        # Some components need fields from both account and risk configs.
        # Merge them into a single namespace so components don't need to
        # know about the config structure.
        merged_cfg = self.cfg.account
        for attr in vars(self.cfg.risk):
            if not hasattr(merged_cfg, attr):
                setattr(merged_cfg, attr, getattr(self.cfg.risk, attr))

        # ── Core components ──
        self.db = Database()                                    # SQLite persistence
        self.decisions = DecisionLogger(self.db)                # V2: Decision audit trail

        # V2 Phase 4: Restart-safe state loading.
        # Parameter overrides must be applied BEFORE strategies are
        # instantiated so each strategy sees the patched config values.
        # The applied list is cached so `start()` can include it in the
        # boot summary log.
        boot_regime = get_current_regime(self.db) or None
        self._applied_overrides = apply_parameter_overrides(self.cfg, self.db, regime=boot_regime)
        self._offline_gap = detect_offline_gap(self.db)

        self.screener = StockScreener(self.cfg.scanner)         # Pre-market scanner
        self.pdt = PDTManager(self.cfg.pdt, self.db)            # Day-trade tracking
        # V2 Phase 8: smart PDT planner -- dynamic day-trade threshold
        # + day->swing conversion for eligible strategies.
        self.smart_pdt = SmartPDTPlanner(self.cfg.pdt, self.db)
        self.sizer = PositionSizer(merged_cfg)                  # Position sizing
        # V2 Phase 7: dynamic risk tolerance — composes conviction +
        # streak + regime + drawdown into runtime sizing multipliers that
        # RiskManager and SignalAggregator consume.
        self.dynamic_risk = DynamicRiskController(merged_cfg, self.db)
        self.risk = RiskManager(
            merged_cfg, self.db, dynamic_controller=self.dynamic_risk
        )                                                        # Risk gate
        self.orders = OrderManager(self.db)                     # Stock order execution
        self.options_orders = OptionsOrderManager(self.db)      # Options order execution
        self.perf = PerformanceTracker(self.db)                 # P&L metrics

        # ── Stock strategies ──
        # Each strategy is instantiated only if enabled in config.
        self.strategies = []
        strat_cfg = self.cfg.strategies
        if strat_cfg.mean_reversion.enabled:
            self.strategies.append(MeanReversionStrategy(strat_cfg.mean_reversion))
        if strat_cfg.momentum.enabled:
            self.strategies.append(MomentumStrategy(strat_cfg.momentum))
        if strat_cfg.vwap.enabled:
            self.strategies.append(VWAPStrategy(strat_cfg.vwap))
        if getattr(strat_cfg, "ema_crossover", None) and strat_cfg.ema_crossover.enabled:
            self.strategies.append(EMACrossoverStrategy(strat_cfg.ema_crossover))
        if getattr(strat_cfg, "macd_divergence", None) and strat_cfg.macd_divergence.enabled:
            self.strategies.append(MACDDivergenceStrategy(strat_cfg.macd_divergence))
        if getattr(strat_cfg, "bb_squeeze", None) and strat_cfg.bb_squeeze.enabled:
            self.strategies.append(BBSqueezeStrategy(strat_cfg.bb_squeeze))
        if getattr(strat_cfg, "orb", None) and strat_cfg.orb.enabled:
            self.strategies.append(ORBStrategy(strat_cfg.orb))
        if getattr(strat_cfg, "pullback", None) and strat_cfg.pullback.enabled:
            self.strategies.append(PullbackStrategy(strat_cfg.pullback))

        # Adaptive strategy weighting — adjusts conviction multipliers
        # based on historical trade performance per strategy.
        weighter_cfg = getattr(self.cfg, "strategy_weighting", None)
        self.weighter = None
        if weighter_cfg and getattr(weighter_cfg, "enabled", False):
            self.weighter = StrategyWeighter(self.db, weighter_cfg)

        # V2 Phase 5: ML signal-quality predictor.  Cold-start safe —
        # returns None from predict() when no model is loaded, which
        # leaves the rule-based conviction untouched.
        self.ml_predictor = SignalQualityPredictor(self.db)

        # The signal aggregator is the "brain" — it collects signals from
        # all strategies, ranks them, and builds the execution queue.
        self.aggregator = SignalAggregator(
            strategies=self.strategies,
            pdt_manager=self.pdt,
            risk_manager=self.risk,
            position_sizer=self.sizer,
            weighter=self.weighter,
            decision_logger=self.decisions,
            ml_predictor=self.ml_predictor,
            dynamic_risk=self.dynamic_risk,
            smart_pdt=self.smart_pdt,
        )

        # ── Options strategies ──
        self.options_strategies = []
        self.options_enabled = getattr(self.cfg.options, "enabled", False)
        if self.options_enabled:
            if strat_cfg.credit_put_spread.enabled:
                self.options_strategies.append(CreditPutSpreadStrategy(strat_cfg.credit_put_spread))
            if strat_cfg.debit_call_spread.enabled:
                self.options_strategies.append(DebitCallSpreadStrategy(strat_cfg.debit_call_spread))
            if strat_cfg.long_call.enabled:
                self.options_strategies.append(LongCallStrategy(strat_cfg.long_call))
            if strat_cfg.long_put.enabled:
                self.options_strategies.append(LongPutStrategy(strat_cfg.long_put))
            if strat_cfg.cash_secured_put.enabled:
                self.options_strategies.append(CashSecuredPutStrategy(strat_cfg.cash_secured_put))
            if strat_cfg.covered_call.enabled:
                self.options_strategies.append(CoveredCallStrategy(strat_cfg.covered_call))
            if strat_cfg.covered_straddle.enabled:
                self.options_strategies.append(CoveredStraddleStrategy(strat_cfg.covered_straddle))
            if getattr(strat_cfg, "momentum_options", None) and strat_cfg.momentum_options.enabled:
                self.options_strategies.append(MomentumOptionsStrategy(strat_cfg.momentum_options))
            if getattr(strat_cfg, "zero_dte", None) and strat_cfg.zero_dte.enabled:
                self.options_strategies.append(ZeroDTEStrategy(strat_cfg.zero_dte))

        # ── Sentiment layer ──
        self.regime_analyzer = MarketRegimeAnalyzer()
        news_lookback = getattr(self.cfg.sentiment, "news_lookback_hours", 24)
        news_max = getattr(self.cfg.sentiment, "news_max_articles", 10)
        news_cache_ttl = getattr(self.cfg.sentiment, "news_cache_ttl_seconds", 900)
        self.news_scanner = NewsSentimentScanner(
            lookback_hours=news_lookback,
            max_articles=news_max,
            cache_ttl_seconds=news_cache_ttl,
        )
        self.earnings_guard = EarningsGuard(
            block_days_before=getattr(self.cfg.sentiment, "earnings_block_days_before", 1),
            block_days_after=getattr(self.cfg.sentiment, "earnings_block_days_after", 1),
        )
        self._market_context = None  # Set at market open by _analyze_market_regime()

        # ── State ──
        self._candidates: list[dict] = []          # Today's stock scanner candidates
        self._options_candidates: list[dict] = []  # Today's options-eligible candidates
        self._running = False
        self._failed_symbols: dict[str, str] = {}  # symbol -> reason (untradable, halted, etc.)
        self._failed_symbol_counts: dict[str, int] = {}  # symbol -> consecutive fail count
        self._scan_counts: dict[str, int] = {"momentum": 0, "mean_reversion": 0, "vwap": 0}

        # V2 Phase 13: Per-cycle timing budget (10s default)
        self._cycle_timer = CycleTimer()

    # ══════════════════════════════════════════════════════════
    # Lifecycle
    # ══════════════════════════════════════════════════════════

    def start(self) -> None:
        """Initialize Alpaca clients, sync state, start the scheduler.

        This method blocks the main thread (via a sleep loop) until
        the user presses Ctrl+C or the bot is stopped programmatically.
        """
        setup_logging()

        # Initialize Alpaca API clients — retry up to 3 times on transient
        # network failures.  If this fails, the bot cannot operate at all.
        try:
            init_clients(self.cfg)
        except Exception as e:
            log.error("client_init_failed", error=str(e))
            print(con.error(f"FATAL: Could not initialize Alpaca clients — {e}"))
            print(con.error("Check your API keys in .env and internet connectivity."))
            sys.exit(1)

        # Verify we can connect to Alpaca and the credentials are valid
        try:
            account = get_account()
        except Exception as e:
            log.error("account_fetch_failed", error=str(e))
            print(con.error(f"FATAL: Could not connect to Alpaca — {e}"))
            print(con.error("Check: API keys valid? Internet connected? Alpaca status page?"))
            sys.exit(1)

        log.info(
            "connected",
            equity=float(account.equity),
            cash=float(account.cash),
            day_trade_count=account.daytrade_count,
            paper=self.cfg.alpaca.paper,
        )

        # V2 Phase 4: emit structured boot summary of restored state.
        weighter_rows_loaded = len(self.weighter._weights) if self.weighter else 0
        log_boot_summary(
            weighter_rows=weighter_rows_loaded,
            overrides_applied=self._applied_overrides,
            gap=self._offline_gap,
        )
        record_startup(self.db)

        # Reconcile any positions from a previous run (e.g. if the bot
        # was restarted mid-day, positions may still be open on Alpaca).
        # Non-fatal: if sync fails, we log and continue — positions will
        # be reconciled on the next scheduled sync.
        try:
            self.orders.sync_positions()
        except Exception as e:
            log.warning("startup_sync_failed", error=str(e))
            print(con.warning(f"Position sync failed on startup — {e}"))

        # Sync PDT count with Alpaca's server-side count — our local DB
        # may be stale after a restart or if trades happened outside the bot.
        # Non-fatal: local DB count is a fallback.
        try:
            self.pdt.sync_with_alpaca()
        except Exception as e:
            log.warning("startup_pdt_sync_failed", error=str(e))

        # Start the APScheduler — it fires cron jobs throughout the day
        self._scheduler = create_scheduler(self)
        self._scheduler.start()
        self._running = True

        mode = "DRY RUN" if self.dry_run else "LIVE PAPER" if self.cfg.alpaca.paper else "LIVE REAL"
        now_et = datetime.now(ET)
        stock_strat_names = [type(s).__name__ for s in self.strategies]
        options_strat_names = [type(s).__name__ for s in self.options_strategies]
        log.info(
            "bot_started",
            version=__version__,
            mode=mode,
            current_time_et=now_et.strftime("%Y-%m-%d %H:%M:%S"),
            stock_strategies=stock_strat_names,
            options_strategies=options_strat_names,
            scanner_config={"min_price": getattr(self.cfg.scanner, "min_price", "?"),
                            "max_price": getattr(self.cfg.scanner, "max_price", "?"),
                            "min_gap_pct": getattr(self.cfg.scanner, "min_gap_pct", "?"),
                            "min_relative_volume": getattr(self.cfg.scanner, "min_relative_volume", "?")},
            schedule={"premarket_scan": self.cfg.schedule.premarket_scan,
                      "market_open": self.cfg.schedule.market_open,
                      "scan_interval_minutes": getattr(self.cfg.schedule, "scan_interval_minutes", 15)},
        )
        print(con.banner(
            version=__version__, mode=mode,
            equity=float(account.equity), cash=float(account.cash),
            stock_strats=len(self.strategies),
            options_strats=len(self.options_strategies),
            pdt_remaining=self.pdt.day_trades_remaining(),
        ))

        # If the bot starts during market hours, don't wait for the next
        # scheduled window — scan and evaluate immediately
        self._catchup_missed_jobs(now_et)

        # Block the main thread until shutdown.
        # The scheduler runs jobs in background threads.
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        """Graceful shutdown — stops the scheduler but leaves positions open.

        Server-side bracket orders (stop-loss + take-profit) remain active
        on Alpaca's servers even after the bot shuts down.  This is a key
        safety feature — your stops protect you even if the bot crashes.

        V2 Phase 4: flushes the decision logger and writes a shutdown
        timestamp to `bot_state` so the next startup can measure the
        offline gap.
        """
        log.info("bot_stopping", version=__version__)
        self._running = False
        if hasattr(self, "_scheduler"):
            self._scheduler.shutdown(wait=False)
        try:
            self.decisions.flush()
        except Exception:
            log.exception("shutdown_decisions_flush_failed")
        try:
            record_shutdown(self.db)
        except Exception:
            log.exception("shutdown_record_failed")
        print(con.stopped())

    # ══════════════════════════════════════════════════════════
    # Catch-up logic — handle mid-day startups
    # ══════════════════════════════════════════════════════════

    def _catchup_missed_jobs(self, now_et: datetime) -> None:
        """If the bot starts during market hours, scan and evaluate immediately.

        Without this, starting the bot at 10:30 AM would mean waiting until
        12:00 PM (midday_check) for the first evaluation — missing 2 hours
        of trading opportunities.
        """
        weekday = now_et.weekday()
        if weekday >= 5:  # Saturday=5, Sunday=6
            log.info("catchup_skip", reason="weekend")
            print(con.catchup("Weekend detected — no market activity. Waiting for Monday."))
            return

        hour_min = now_et.hour * 60 + now_et.minute
        market_open = 9 * 60 + 30   # 9:30 AM ET
        market_close = 16 * 60       # 4:00 PM ET

        if hour_min < market_open:
            log.info("catchup_skip", reason="pre_market",
                      current_time=now_et.strftime("%H:%M"))
            print(con.catchup(f"Market hasn't opened yet ({now_et.strftime('%I:%M %p')} ET). Waiting for 9:00 AM pre-market scan."))
            return

        if hour_min >= market_close:
            log.info("catchup_skip", reason="after_hours",
                      current_time=now_et.strftime("%H:%M"))
            print(con.catchup(f"Market is closed ({now_et.strftime('%I:%M %p')} ET). Bot will resume at next market open."))
            return

        try:
            open_positions = self.orders.get_open_positions()
        except Exception as e:
            log.warning("catchup_position_fetch_failed", error=str(e))
            open_positions = []
        has_positions = len(open_positions) > 0

        log.info("catchup_start", current_time=now_et.strftime("%H:%M"),
                  open_positions=len(open_positions))
        print(con.catchup(f"Bot started mid-session at {now_et.strftime('%I:%M %p')} ET. Running missed jobs..."))

        # Always scan for candidates on startup
        log.info("catchup_running", job="premarket_scan")
        print(con.catchup("Scanning stock universe for today's candidates..."))
        self.job_premarket_scan()
        print(con.catchup(f"Scan complete: {len(self._candidates)} candidates found."))

        # Always set up market context (regime analysis)
        if self._market_context is None:
            log.info("catchup_running", job="market_open")
            print(con.catchup("Analyzing market regime (SPY/QQQ/VIX)..."))
            self.job_market_open()
            if self._market_context:
                regime = self._market_context.regime.value
                biases = ", ".join(self._market_context.allowed_options_biases) or "none"
                print(con.catchup(f"Regime: {regime.upper()} | conviction={self._market_context.conviction_modifier}x | options: [{biases}]"))
            else:
                print(con.catchup("Market regime analysis failed — using defaults."))

        # Only evaluate strategies if we have NO open positions.
        # If we already have positions, we wait for the next scheduled
        # window to avoid doubling down accidentally.
        if not has_positions:
            log.info("catchup_running", job="scan_and_evaluate", reason="no_open_positions")
            print(con.catchup("No open positions — running scan & evaluate now..."))
            self.job_scan_and_evaluate()
        else:
            log.info("catchup_skip_entry", reason="has_open_positions",
                      count=len(open_positions))
            print(con.catchup(f"{len(open_positions)} open position(s) held — skipping evaluation. Will re-evaluate at next window."))

        log.info("catchup_complete", scanned=True, evaluated=not has_positions)
        print(con.catchup("Done. Scheduler is now running — next jobs fire on schedule.\n"))

    # ══════════════════════════════════════════════════════════
    # Scheduled Jobs — called by APScheduler at specific times
    # ══════════════════════════════════════════════════════════

    def job_premarket_scan(self) -> None:
        """9:00 AM ET — Scan the stock universe for today's candidates.

        The scanner filters ~8000 stocks down to ~20 candidates based on
        price, gap%, and relative volume.  These candidates are then
        evaluated by all strategies during the entry window.
        """
        now = datetime.now(ET)
        log.info("job_start", job="premarket_scan", time=now.strftime("%H:%M:%S"))
        print(con.section("Pre-market Scan"))
        # Reset daily blacklists — symbols halted yesterday may be tradable today
        self._failed_symbols.clear()
        self._failed_symbol_counts.clear()
        try:
            counts = self._run_full_scan()
            symbols = [c["symbol"] for c in self._candidates]
            log.info("scan_complete",
                      total=len(self._candidates),
                      momentum=counts["momentum"],
                      mean_reversion=counts["mean_reversion"],
                      vwap=counts["vwap"],
                      symbols=symbols[:10])
            if self._candidates:
                print(con.scan_result(
                    total=len(self._candidates),
                    momentum=counts["momentum"],
                    mean_rev=counts["mean_reversion"],
                    vwap=counts["vwap"],
                    symbols=symbols,
                ))
            else:
                log.warning("scan_returned_zero_candidates",
                             hint="Check scanner filters: min_gap_pct, min_relative_volume, price range")
                print(con.warning("0 candidates found. Check scanner filters if this persists."))

            # Scan a separate, broader universe for options-eligible stocks
            if self.options_enabled:
                self._options_candidates = self.screener.scan_options_universe()
                opt_syms = [c["symbol"] for c in self._options_candidates]
                top_opts = ", ".join(opt_syms[:6])
                more_opts = f" +{len(opt_syms)-6} more" if len(opt_syms) > 6 else ""
                print(con.info(f"Options universe: {len(self._options_candidates)} candidates — {top_opts}{more_opts}"))
        except Exception as e:
            log.error("scan_failed", error=str(e), error_type=type(e).__name__)
            self._candidates = []
            self._options_candidates = []
            print(con.error(f"Scan FAILED: {e}"))

    def job_market_open(self) -> None:
        """9:30 AM ET — Cache starting equity, sync positions, analyze market.

        The starting equity is saved so the daily loss limit can be
        calculated.  Market regime analysis (SPY/QQQ/VIX) determines
        whether it's safe to take new positions today.
        """
        now = datetime.now(ET)
        log.info("job_start", job="market_open")
        print(con.section("Market Open"))
        try:
            account = get_account()
            self.risk.set_starting_equity(float(account.equity))

            try:
                self.orders.sync_positions()
            except Exception as e:
                log.warning("market_open_sync_failed", error=str(e))
                print(con.warning(f"Position sync failed — {e}"))

            self._analyze_market_regime()

            log.info("market_open", equity=float(account.equity), cash=float(account.cash))
            print(con.info(f"Account: ${float(account.equity):,.2f} equity  |  ${float(account.cash):,.2f} cash"))
        except Exception as e:
            log.error("market_open_failed", error=str(e))
            print(con.error(f"Market open FAILED: {e} — will retry on next scheduled job."))

    def _analyze_market_regime(self) -> None:
        """Fetch SPY/QQQ/VIX bars and classify the market regime.

        The regime (Strong Bull → Strong Bear) determines:
          - Whether new longs are allowed
          - Whether options trades are allowed
          - Conviction and position size multipliers
        """
        try:
            end = datetime.now(ET)
            start = end - timedelta(days=250)  # Need 200+ days for EMA-200

            bars = fetch_bars_multi(["SPY", "QQQ"], TimeFrame.Day, start, end)
            spy_bars = bars.get("SPY")
            qqq_bars = bars.get("QQQ")

            if spy_bars is None or spy_bars.empty or qqq_bars is None or qqq_bars.empty:
                log.warning("market_regime_skip", reason="missing SPY/QQQ data")
                return

            # Try to fetch VIX (may not be available on all Alpaca plans)
            vix_bars = None
            try:
                vix_data = fetch_bars_multi(["VIX", "VIXY"], TimeFrame.Day, start, end)
                vix_bars = vix_data.get("VIX") or vix_data.get("VIXY")
            except Exception as e:
                log.debug("vix_fetch_failed", error=str(e))

            self._market_context = self.regime_analyzer.analyze(spy_bars, qqq_bars, vix_bars)
            record_current_regime(self.db, self._market_context.regime.value)
            log.info(
                "market_regime_result",
                regime=self._market_context.regime.value,
                allow_new_longs=self._market_context.allow_new_longs,
                allowed_options_biases=self._market_context.allowed_options_biases,
                conviction_modifier=self._market_context.conviction_modifier,
                position_size_modifier=self._market_context.position_size_modifier,
            )
            ctx = self._market_context
            biases = ", ".join(ctx.allowed_options_biases) if ctx.allowed_options_biases else "none"
            print(con.regime_line(
                regime=ctx.regime.value, spy_rsi=ctx.spy_rsi,
                spy_trend=ctx.spy_trend, vix_level=ctx.vix_level,
                vix_trend=ctx.vix_trend,
            ))
            print(con.regime_modifiers(
                conviction_mod=ctx.conviction_modifier,
                size_mod=ctx.position_size_modifier,
                longs_allowed=ctx.allow_new_longs,
                options_biases=biases,
            ))

        except Exception as e:
            log.warning("market_regime_failed", error=str(e))

    def job_scan_and_evaluate(self) -> None:
        """Every 15 min during market hours — fresh scan + strategy evaluation.

        Replaces the old fixed-time windows (9:35, 11:00, 12:00, 3:00).
        The algo decides when setups are worth entering, not the clock.

        Each cycle:
          1. Sync positions with Alpaca
          2. Invalidate snapshot cache and run fresh scanner
          3. Evaluate all stock strategies on fresh candidates
          4. Evaluate options strategies (if enabled)
        """
        now = datetime.now(ET)

        # Skip if before 9:45 AM (let opening volatility settle)
        market_start = now.replace(hour=9, minute=45, second=0, microsecond=0)
        if now < market_start:
            return

        # Skip if after 3:45 PM (EOD close handles the rest)
        market_end = now.replace(hour=15, minute=45, second=0, microsecond=0)
        if now > market_end:
            return

        log.info("job_start", job="scan_and_evaluate", time=now.strftime("%H:%M"))
        print(con.section("Scan & Evaluate"))

        try:
            self.orders.sync_positions()
        except Exception as e:
            log.warning("scan_cycle_sync_failed", error=str(e))

        try:
            self.screener.invalidate_snapshot_cache()
            counts = self._run_full_scan()
            print(con.info(f"Scan: {len(self._candidates)} candidates"
                  f" (momentum={counts['momentum']}, mr={counts['mean_reversion']}, vwap={counts['vwap']})"))
        except Exception as e:
            log.warning("scan_cycle_scan_failed", error=str(e))
            print(con.warning(f"Scan failed: {e} — evaluating existing candidates..."))

        self._run_evaluation_cycle()

    def job_options_expiry_check(self) -> None:
        """3:30 PM ET — Close options positions expiring today or tomorrow.

        Options near expiration carry assignment risk (short options) and
        gamma risk (all options).  Closing them before EOD avoids surprise
        assignments and illiquid fills.
        """
        if not self.options_enabled:
            return
        now = datetime.now(ET)
        log.info("job_start", job="options_expiry_check")
        print(con.section("Options Expiry Check"))
        try:
            closed = self.options_orders.close_expiring_positions(days_until_expiration=1)
            if closed:
                print(con.info(f"Closed {len(closed)} expiring option(s): {', '.join(closed)}"))
                # Fetch open options ONCE, then search for each closed symbol
                open_opts = self.db.get_open_options_trades()
                for sym in closed:
                    for t in open_opts:
                        if sym in str(t.get("legs", "")):
                            self.db.update_options_trade(
                                t["id"], status="closed",
                                exit_time=now.isoformat(),
                                close_reason="expiry_management",
                            )
            else:
                print(con.info("No expiring options to close."))
        except Exception as e:
            log.error("options_expiry_check_failed", error=str(e))
            print(con.error(f"Options expiry check FAILED: {e}"))

    def job_options_position_sync(self) -> None:
        """Every 5 minutes — Sync options positions with Alpaca.

        Options fills can happen at any time (limit orders filling,
        assignments, exercises).  This periodic sync keeps the local DB
        accurate without the overhead of per-minute checks.
        """
        if not self.options_enabled:
            return
        try:
            broker_positions = self.options_orders.get_options_positions()
            open_db_trades = self.db.get_open_options_trades()

            # Find DB trades that are no longer in broker positions
            broker_syms = {pos.symbol for pos in broker_positions}
            any_closed = False
            for trade in open_db_trades:
                trade_legs = trade.get("legs", "")
                # Check if any leg symbol is still held at the broker
                still_open = any(sym in trade_legs for sym in broker_syms)
                if not still_open and trade_legs:
                    log.info("options_position_closed_externally",
                             trade_id=trade["id"],
                             underlying=trade.get("underlying", "?"))
                    self.db.update_options_trade(
                        trade["id"], status="closed",
                        exit_time=datetime.now(ET).isoformat(),
                        close_reason="filled_or_expired",
                    )
                    self._log_options_trade_exit(trade, close_reason="filled_or_expired")
                    any_closed = True
            if any_closed:
                self.decisions.flush()
        except Exception as e:
            log.debug("options_position_sync_failed", error=str(e))

    def _log_options_trade_exit(self, trade: dict, close_reason: str) -> None:
        """Emit exit + review decisions for a closed options trade."""
        underlying = trade.get("underlying", "")
        strategy = trade.get("strategy", "")
        entry_debit = trade.get("entry_debit") or 0
        entry_credit = trade.get("entry_credit") or 0
        max_loss = trade.get("max_loss") or 0
        max_profit = trade.get("max_profit") or 0

        self.decisions.log_exit(
            underlying, strategy,
            reasoning=f"options closed via {close_reason}: debit=${entry_debit:.2f} credit=${entry_credit:.2f} max_loss=${max_loss:.2f} max_profit=${max_profit:.2f}",
            factors={
                "trade_id": trade.get("id"),
                "close_reason": close_reason,
                "entry_debit": entry_debit,
                "entry_credit": entry_credit,
                "max_loss": max_loss,
                "max_profit": max_profit,
                "expiration": trade.get("expiration"),
                "strikes": trade.get("strikes"),
            },
        )
        self.decisions.log_review(
            underlying, strategy,
            reasoning=f"options trade closed via {close_reason}; review whether legs behaved as expected",
            factors={"outcome": close_reason},
        )

    def job_monitor_zero_dte(self) -> None:
        """Every 2 min during market hours -- manage open 0DTE options.

        For each open 0DTE position:
          - Hard time exit at 15:30 ET (30 min before close).
          - Loss cut at -50% of premium paid.
          - Trailing stop: once profit exceeds 100% of premium, trail at
            50% of peak profit (lock in at least half the gains).
        """
        if not self.options_enabled:
            return
        try:
            open_opts = self.db.get_open_options_trades()
            zero_dte_trades = [
                t for t in open_opts
                if "zero_dte" in (t.get("strategy") or "")
            ]
            if not zero_dte_trades:
                return

            now_et = datetime.now(ET)
            time_exit = now_et.hour * 60 + now_et.minute >= 15 * 60 + 30

            from ai_trade.data.options_chain import get_options_snapshot

            for trade in zero_dte_trades:
                legs_str = trade.get("legs", "")
                if not legs_str:
                    continue

                leg_symbols = [s.strip() for s in legs_str.split(",") if s.strip()]
                if not leg_symbols:
                    continue

                if time_exit:
                    log.info(
                        "zero_dte_time_exit",
                        trade_id=trade["id"],
                        underlying=trade.get("underlying", ""),
                    )
                    for sym in leg_symbols:
                        self.options_orders.close_options_position(sym)
                    self.db.update_options_trade(
                        trade["id"], status="closed",
                        exit_time=now_et.isoformat(),
                        close_reason="zero_dte_time_exit",
                    )
                    self._log_options_trade_exit(trade, close_reason="zero_dte_time_exit")
                    continue

                snaps = get_options_snapshot(leg_symbols)
                if not snaps:
                    continue

                entry_cost = abs(trade.get("entry_debit", 0) or 0) * 100
                if entry_cost <= 0:
                    entry_cost = abs(trade.get("max_loss", 0) or 0)
                if entry_cost <= 0:
                    continue

                current_value = 0.0
                for sym in leg_symbols:
                    snap = snaps.get(sym, {})
                    mid = snap.get("mid_price", 0.0)
                    current_value += mid * 100

                pnl_pct = (current_value - entry_cost) / entry_cost if entry_cost > 0 else 0.0

                loss_cut_pct = getattr(self.cfg, "options", None)
                loss_cut_pct = getattr(loss_cut_pct, "zero_dte_loss_cut_pct", -0.50) if loss_cut_pct else -0.50

                if pnl_pct <= loss_cut_pct:
                    log.info(
                        "zero_dte_loss_cut",
                        trade_id=trade["id"],
                        underlying=trade.get("underlying", ""),
                        pnl_pct=round(pnl_pct, 3),
                    )
                    for sym in leg_symbols:
                        self.options_orders.close_options_position(sym)
                    self.db.update_options_trade(
                        trade["id"], status="closed",
                        exit_time=now_et.isoformat(),
                        close_reason="zero_dte_loss_cut",
                    )
                    self._log_options_trade_exit(trade, close_reason="zero_dte_loss_cut")
                    continue

                trail_threshold = getattr(
                    getattr(self.cfg, "options", None), "zero_dte_trail_threshold", 1.0,
                )
                trail_pct = getattr(
                    getattr(self.cfg, "options", None), "zero_dte_trail_pct", 0.50,
                )

                if pnl_pct >= trail_threshold:
                    peak_key = f"zero_dte.peak.{trade['id']}"
                    prev_peak = float(self.db.get_state(peak_key, "0") or "0")
                    peak = max(prev_peak, current_value)
                    if current_value > prev_peak:
                        self.db.set_state(peak_key, str(current_value))

                    drawdown_from_peak = (peak - current_value) / peak if peak > 0 else 0.0
                    if drawdown_from_peak >= trail_pct and prev_peak > 0:
                        log.info(
                            "zero_dte_trail_exit",
                            trade_id=trade["id"],
                            underlying=trade.get("underlying", ""),
                            pnl_pct=round(pnl_pct, 3),
                            peak=round(peak, 2),
                            current=round(current_value, 2),
                        )
                        for sym in leg_symbols:
                            self.options_orders.close_options_position(sym)
                        self.db.update_options_trade(
                            trade["id"], status="closed",
                            exit_time=now_et.isoformat(),
                            close_reason="zero_dte_trail_exit",
                        )
                        self._log_options_trade_exit(trade, close_reason="zero_dte_trail_exit")

        except Exception as e:
            log.debug("zero_dte_monitor_failed", error=str(e))

    def job_cleanup_database(self) -> None:
        """Weekly (Sunday midnight) — purge old rows from high-volume tables.

        Keeps the SQLite database lean by removing decisions, scanner results,
        and ML tables older than 90 days.  Core tables (trades, trade_analysis,
        parameter_history) are kept indefinitely for the self-learning loop.
        """
        try:
            deleted = self.db.cleanup_old_data(retention_days=90)
            total = sum(deleted.values())
            if total > 0:
                log.info("db_cleanup_done", deleted=deleted, total_rows=total)
                print(con.detail(f"DB cleanup: {total} old rows removed."))
            else:
                log.info("db_cleanup_nothing_to_do")
        except Exception as e:
            log.warning("db_cleanup_failed", error=str(e))

    def job_eod_close_day_trades(self) -> None:
        """3:50 PM ET — Force-close any open day-trade positions.

        Day trades MUST be closed before market close (4:00 PM).  We close
        at 3:50 to allow time for order execution.  This is critical for
        PDT compliance — holding a day trade overnight would still count
        as a day trade but with overnight risk.
        """
        now = datetime.now(ET)
        log.info("job_start", job="eod_close_day_trades")
        print(con.section("EOD Close Day Trades"))
        try:
            open_trades = self.db.get_open_trades()
            day_trades = [t for t in open_trades if t.get("hold_type") == "day"]
            if day_trades:
                symbols = [t.get("symbol", "?") for t in day_trades]
                print(con.info(f"Closing {len(day_trades)} day trade(s): {', '.join(symbols)}"))
            else:
                print(con.info("No open day trades to close."))
            self.orders.close_all_day_trades(open_trades)
        except Exception as e:
            log.error("eod_close_failed", error=str(e))
            print(con.error(f"EOD close FAILED: {e}"))

    def job_eod_review(self) -> None:
        """4:05 PM ET — Daily P&L summary, save snapshot to database.

        This runs 5 minutes after market close to capture the final
        settlement prices.  The snapshot is used for the equity curve
        and performance metrics.
        """
        now = datetime.now(ET)
        log.info("job_start", job="eod_review")
        print(con.section("Daily Summary"))
        try:
            account = get_account()
            positions = self.orders.get_open_positions()

            # Save daily snapshot for the equity curve
            today = datetime.now(ET).strftime("%Y-%m-%d")

            # Calculate today's realized P&L from closed trades
            realized_pnl = 0.0
            try:
                closed_today = self.db.get_trades_closed_on(today)
                realized_pnl = sum(t.get("pnl", 0) or 0 for t in closed_today)
            except Exception as e:
                log.debug("realized_pnl_fetch_failed", error=str(e))

            self.db.save_snapshot(
                date=today,
                equity=float(account.equity),
                cash=float(account.cash),
                open_positions=len(positions),
                day_trades_used=self.pdt.get_day_trades_used(),
                realized_pnl=realized_pnl,
                unrealized_pnl=float(account.equity) - float(account.cash) if positions else 0.0,
            )

            # Print human-readable summary to console
            summary = self.perf.daily_summary(
                equity=float(account.equity),
                cash=float(account.cash),
                open_positions=len(positions),
                day_trades_used=self.pdt.get_day_trades_used(),
            )
            print(summary)
        except Exception as e:
            log.error("eod_review_failed", error=str(e))

    def job_eod_analysis(self) -> None:
        """Post-close self-learning sweep — loss patterns + parameter review.

        Runs after eod_review.  Scans recent closed trades for loss
        clusters across strategy/hour/regime/stop-quality axes and asks
        the parameter optimizer to propose bounded adjustments.  When
        ``analysis.apply_parameter_changes`` is true in config, the
        optimizer writes new ``parameter_overrides`` rows which take
        effect on the next bot restart (never mid-session).
        """
        log.info("job_start", job="eod_analysis")
        try:
            clusters = scan_loss_patterns(self.db, lookback=50)
            self.db.set_state(
                "analysis.last_pattern_scan",
                datetime.now(timezone.utc).isoformat(),
            )
            self.db.set_state(
                "analysis.last_pattern_clusters", str(len(clusters))
            )
            for c in clusters[:5]:
                log.info("loss_cluster", **c)

            apply_changes = bool(
                getattr(
                    getattr(self.cfg, "analysis", None),
                    "apply_parameter_changes",
                    False,
                )
            )
            proposals = review_parameter_adjustments(
                database=self.db,
                cfg=self.cfg,
                apply_changes=apply_changes,
            )
            self.db.set_state(
                "analysis.last_optimizer_run",
                datetime.now(timezone.utc).isoformat(),
            )
            self.db.set_state(
                "analysis.last_optimizer_proposals", str(len(proposals))
            )
            if proposals:
                log.info(
                    "parameter_review_complete",
                    proposals=len(proposals),
                    applied=sum(1 for p in proposals if p.get("applied")),
                    apply_changes=apply_changes,
                )
        except Exception:
            log.exception("eod_analysis_failed")

    def job_sync_positions(self) -> None:
        """Every 60 seconds — Reconcile Alpaca positions with local DB.

        This is a safety net: if a bracket order's stop-loss or take-profit
        fills on Alpaca's side, the local database needs to be updated.
        Without this sync, the DB would show stale "open" trades.
        """
        try:
            summary = self.orders.sync_positions()
            closed = summary.get("closed_trades", []) if summary else []
            for t in closed:
                self._log_stock_trade_exit(t)
            if closed:
                self.decisions.flush()
        except Exception as e:
            log.warning("sync_failed", error=str(e))

    def job_update_trailing_stops(self) -> None:
        """Every 5 minutes — advance stop-losses on open winners.

        For each open stock trade:
          1. Fetch current price via snapshot
          2. Look up stored ATR (or fall back to a % move)
          3. Ask ExitPlanner.compute_trailing_stop_long for a tighter stop
          4. Replace the Alpaca stop-loss leg if the proposal improves it
        """
        try:
            from ai_trade.data.historical import fetch_snapshots
            from ai_trade.strategy.exit_planner import compute_trailing_stop_long

            open_trades = self.db.get_open_trades()
            open_trades = [t for t in open_trades if t.get("side") == "long"]
            if not open_trades:
                return

            symbols = sorted({t.get("symbol", "") for t in open_trades if t.get("symbol")})
            if not symbols:
                return
            snaps = fetch_snapshots(symbols)

            updated = 0
            for t in open_trades:
                sym = t.get("symbol", "")
                buy_order_id = t.get("buy_order_id")
                if not sym or not buy_order_id:
                    continue

                snap = snaps.get(sym)
                if not snap:
                    continue
                current_price = None
                if hasattr(snap, "latest_trade") and snap.latest_trade:
                    current_price = float(snap.latest_trade.price)
                elif hasattr(snap, "daily_bar") and snap.daily_bar:
                    current_price = float(snap.daily_bar.close)
                if current_price is None or current_price <= 0:
                    continue

                entry_price = float(t.get("entry_price") or 0)
                current_stop = float(t.get("stop_loss") or 0)
                if entry_price <= 0 or current_stop <= 0:
                    continue

                atr = t.get("atr")
                if atr is None or atr <= 0:
                    # NOTE: do NOT use (entry - current_stop) as a proxy --
                    # the stop may itself have been set tight (e.g. VWAP
                    # reclaim), which would circularly yield a tiny ATR
                    # and trigger a hair-trigger breakeven ratchet.
                    # 2% of entry price is a conservative floor.
                    atr = max(0.01, entry_price * 0.02)
                else:
                    atr = float(atr)

                try:
                    conviction = float(t.get("conviction") or 0.0)
                except (TypeError, ValueError):
                    conviction = 0.0

                hi = t.get("high_since_entry")
                hi = float(hi) if hi not in (None, 0) else None
                if hi is None or current_price > hi:
                    hi = current_price

                lo = t.get("low_since_entry")
                lo = float(lo) if lo not in (None, 0) else None
                if lo is None or current_price < lo:
                    lo = current_price

                new_stop, mode = compute_trailing_stop_long(
                    entry_price=entry_price,
                    current_price=current_price,
                    current_stop=current_stop,
                    atr=atr,
                    high_since_entry=hi,
                    conviction=conviction,
                )

                # Time-based rule: day trades that aren't profitable after
                # 60 min tighten to just under entry. Exit happens naturally
                # if price ticks down into the new stop.
                if t.get("hold_type") == "day" and current_price <= entry_price:
                    entry_time_str = t.get("entry_time", "")
                    try:
                        from datetime import datetime as _dt, timezone as _tz
                        entry_dt = _dt.fromisoformat(entry_time_str.replace("Z", "+00:00"))
                        if entry_dt.tzinfo is None:
                            entry_dt = entry_dt.replace(tzinfo=_tz.utc)
                        age_min = (_dt.now(_tz.utc) - entry_dt).total_seconds() / 60
                    except Exception:
                        age_min = 0
                    if age_min >= 60:
                        time_stop = round(entry_price * 0.9995, 2)
                        if time_stop > current_stop and (new_stop is None or time_stop > new_stop):
                            if time_stop < current_price - 0.01:
                                new_stop, mode = time_stop, "time_breakeven"

                if new_stop is None:
                    if hi != t.get("high_since_entry") or lo != t.get("low_since_entry"):
                        try:
                            self.db.update_trade(
                                t["id"], high_since_entry=hi, low_since_entry=lo,
                            )
                        except Exception:
                            pass
                    continue

                stop_order_id = self.orders.find_stop_leg_id(str(buy_order_id))
                if not stop_order_id:
                    continue

                if self.orders.replace_stop_price(stop_order_id, new_stop):
                    try:
                        self.db.update_trade(
                            t["id"], stop_loss=new_stop,
                            high_since_entry=hi, low_since_entry=lo,
                        )
                    except Exception as e:
                        log.debug("trailing_db_update_failed", trade_id=t["id"], error=str(e))
                    self.decisions.log_evaluate(
                        sym, "trailing_stop", "tightened",
                        reasoning=f"{mode}: stop {current_stop:.2f} -> {new_stop:.2f} (price {current_price:.2f})",
                        factors={
                            "mode": mode,
                            "old_stop": current_stop,
                            "new_stop": new_stop,
                            "current_price": current_price,
                            "high_since_entry": hi,
                            "atr": atr,
                            "conviction": conviction,
                        },
                    )
                    try:
                        notify_trailing_stop_update(
                            symbol=sym,
                            strategy=str(t.get("strategy") or ""),
                            old_stop=current_stop,
                            new_stop=new_stop,
                            entry_price=entry_price,
                            current_price=current_price,
                            high_since_entry=hi,
                            mode=mode,
                            conviction=conviction,
                            atr=atr,
                            take_profit=float(t.get("take_profit") or 0.0) or None,
                        )
                    except Exception as e:
                        log.debug("trailing_email_failed", symbol=sym, error=str(e))
                    updated += 1
            if updated:
                self.decisions.flush()
                log.info("trailing_stops_updated", count=updated)
        except Exception as e:
            log.warning("trailing_stops_job_failed", error=str(e))

    def job_train_ml_models(self) -> None:
        """Retrain the signal-quality classifier on all closed trades.

        Runs after market close.  Cold-start safe: if we haven't
        accumulated enough closed trades with feature snapshots, the
        trainer returns `insufficient_data` and this job is a no-op.
        Successful training writes a new row to `ml_models` with
        is_active=1 and deactivates prior versions — the next bot
        restart will pick up the new model.
        """
        try:
            result = train_signal_quality_model(self.db)
            log.info("ml_training_job_result", **result)
            # Record the last training attempt timestamp for observability.
            self.db.set_state(
                "ml.last_training_run",
                datetime.now(timezone.utc).isoformat(),
            )
            if result.get("status") == "ok":
                self.db.set_state(
                    "ml.last_training_success",
                    datetime.now(timezone.utc).isoformat(),
                )
                self.db.set_state(
                    "ml.active_model_version", str(result.get("version", ""))
                )
        except Exception:
            log.exception("ml_training_job_failed")

    def _snapshot_features_for_trade(self, sig, order_id: str) -> None:
        """Persist the ML feature vector for a just-executed trade.

        Called after a bracket order submits successfully.  Looks up
        the newly inserted trade row by `buy_order_id` so we can key
        the `ml_features` row to `trade_id` without changing the
        order manager's return signature.  Best-effort: failures are
        logged but never block the execution path.
        """
        try:
            with self.db._conn() as conn:  # noqa: SLF001
                row = conn.execute(
                    "SELECT id FROM trades WHERE buy_order_id = ? ORDER BY id DESC LIMIT 1",
                    (str(order_id),),
                ).fetchone()
            if row is None:
                log.debug("ml_feature_snapshot_no_trade_row", order_id=str(order_id))
                return
            trade_id = int(row["id"])
            feats = extract_features(sig, self._market_context)
            self.db.insert_ml_features(
                trade_id=trade_id,
                features=json.dumps(feats),
            )
        except Exception:
            log.exception("ml_feature_snapshot_failed", symbol=sig.symbol)

    def _log_stock_trade_exit(self, t: dict) -> None:
        """Emit exit + review decisions for a closed stock trade."""
        from ai_trade.strategy.exit_planner import score_stop_quality

        symbol = t.get("symbol", "")
        strategy = t.get("strategy", "")
        pnl = t.get("pnl")
        pnl_pct = t.get("pnl_pct")
        exit_price = t.get("exit_price")
        entry_price = t.get("entry_price")
        stop = t.get("stop_loss")
        target = t.get("take_profit")
        high_since = t.get("high_since_entry")
        low_since = t.get("low_since_entry")

        exit_reason = "unknown"
        if exit_price is not None and entry_price is not None:
            if stop is not None and abs(exit_price - stop) / max(stop, 0.01) < 0.01:
                exit_reason = "stop_loss"
            elif target is not None and abs(exit_price - target) / max(target, 0.01) < 0.01:
                exit_reason = "take_profit"
            elif pnl is not None:
                exit_reason = "win" if pnl > 0 else "loss"

        stop_quality = "not_hit"
        if entry_price and stop:
            try:
                stop_quality = score_stop_quality(
                    exit_reason=exit_reason,
                    entry_price=float(entry_price),
                    stop_price=float(stop),
                    max_favorable_price=float(high_since) if high_since else None,
                    max_adverse_price=float(low_since) if low_since else None,
                    direction="long",
                    stop_method=t.get("stop_method"),
                    target_price=float(target) if target else None,
                )
            except Exception:
                stop_quality = "not_hit"
            if stop_quality != "not_hit":
                try:
                    self.db.update_trade(t.get("trade_id"), stop_quality=stop_quality)
                except Exception:
                    pass

        try:
            if exit_price is not None and entry_price is not None and pnl is not None:
                notify_trade_exit(
                    symbol=symbol,
                    strategy=strategy,
                    exit_reason=exit_reason,
                    entry_price=float(entry_price),
                    exit_price=float(exit_price),
                    shares=int(t.get("shares") or 0),
                    pnl=float(pnl),
                    pnl_pct=float(pnl_pct) / 100.0 if pnl_pct is not None else 0.0,
                    hold_type=str(t.get("hold_type") or ""),
                    conviction=float(t.get("conviction")) if t.get("conviction") is not None else None,
                    stop_quality=stop_quality if stop_quality != "not_hit" else None,
                    high_since_entry=float(high_since) if high_since else None,
                    take_profit=float(target) if target else None,
                )
        except Exception as e:
            log.debug("exit_email_failed", symbol=symbol, error=str(e))

        pnl_str = f"${pnl:+.2f} ({pnl_pct:+.2f}%)" if pnl is not None and pnl_pct is not None else "pnl=unknown"
        self.decisions.log_exit(
            symbol, strategy,
            reasoning=f"closed via {exit_reason}: entry=${entry_price} exit=${exit_price} {pnl_str}",
            factors={
                "trade_id": t.get("trade_id"),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "shares": t.get("shares"),
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "exit_reason": exit_reason,
                "hold_type": t.get("hold_type"),
                "stop_quality": stop_quality,
                "max_favorable": high_since,
                "max_adverse": low_since,
            },
        )

        if pnl is not None:
            outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "flat")
            try:
                analysis = analyze_closed_trade_and_persist(
                    database=self.db,
                    trade=t,
                    market_context=self._market_context,
                )
            except Exception:
                log.exception("post_trade_analysis_failed", trade_id=t.get("trade_id"))
                analysis = None

            lesson = (
                analysis.get("lesson")
                if analysis and analysis.get("lesson")
                else self._review_trade_lesson(t, exit_reason, outcome)
            )
            review_factors = {
                "outcome": outcome,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "exit_reason": exit_reason,
            }
            if analysis:
                review_factors.update({
                    "entry_quality": analysis.get("entry_quality"),
                    "stop_quality": analysis.get("stop_quality"),
                    "exit_quality": analysis.get("exit_quality"),
                    "regime_at_entry": analysis.get("market_regime"),
                    "regime_at_exit": analysis.get("regime_at_exit"),
                })
            self.decisions.log_review(
                symbol, strategy,
                reasoning=lesson,
                factors=review_factors,
            )

    def _review_trade_lesson(self, t: dict, exit_reason: str, outcome: str) -> str:
        """Derive a short post-trade insight for the review log entry."""
        pnl_pct = t.get("pnl_pct") or 0
        if outcome == "win" and exit_reason == "take_profit":
            return f"take-profit hit cleanly (+{pnl_pct:.2f}%); target sizing worked"
        if outcome == "win":
            return f"winner +{pnl_pct:.2f}% via {exit_reason}; exit happened before target"
        if outcome == "loss" and exit_reason == "stop_loss":
            return f"stop-loss hit ({pnl_pct:+.2f}%); check if entry was late or stop too tight"
        if outcome == "loss":
            return f"loser {pnl_pct:+.2f}% via {exit_reason}; bailed before stop — review exit logic"
        return f"flat close via {exit_reason}"

    # ══════════════════════════════════════════════════════════
    # Core Logic: Stock evaluation pipeline
    # ══════════════════════════════════════════════════════════

    def _run_full_scan(self) -> dict[str, int]:
        """Run all scanners and merge candidates into self._candidates.

        Returns a dict with counts per scanner for logging.
        """
        momentum_candidates = self.screener.scan()
        mr_candidates = self.screener.scan_mean_reversion()
        vwap_candidates = self.screener.scan_vwap_universe()

        seen: dict[str, dict] = {}
        for c in momentum_candidates + mr_candidates + vwap_candidates:
            sym = c["symbol"]
            if sym not in seen or c.get("score", 0) > seen[sym].get("score", 0):
                seen[sym] = c
        self._candidates = list(seen.values())

        counts = {
            "momentum": len(momentum_candidates),
            "mean_reversion": len(mr_candidates),
            "vwap": len(vwap_candidates),
        }
        self._scan_counts = counts
        return counts

    def _run_evaluation_cycle(self) -> None:
        """Run both stock and options evaluation pipelines."""
        self._evaluate_and_trade()
        if self.options_enabled:
            print(con.info("Evaluating options strategies..."))
            self._evaluate_options()

    def _evaluate_and_trade(self) -> None:
        """Run stock strategies on candidates and submit approved trades.

        Pipeline:
          1. Gate: market regime allows new longs?
          2. Gate: daily loss limit not hit?
          3. Fetch daily bars (60 days) + intraday bars (2 hours)
          4. Add technical indicators to all bars
          5. Scan news sentiment for candidates
          6. Collect and rank signals via the SignalAggregator
          7. Apply sentiment modifiers (regime + news)
          8. Submit bracket orders for approved signals
        """
        if not self._candidates:
            log.info("no_candidates_to_evaluate")
            print(con.detail("No candidates to evaluate."))
            return

        # Refresh Alpaca's PDT count before evaluating — prevents submitting
        # orders that Alpaca will reject with a 403.
        # Non-fatal: if sync fails, we use the last known count.
        try:
            self.pdt.sync_with_alpaca()
        except Exception as e:
            log.warning("pdt_sync_failed_before_eval", error=str(e))

        # Gate: market regime check — don't go long into a strong bear market
        ctx = self._market_context
        if ctx and not ctx.allow_new_longs:
            log.warning("longs_blocked_by_regime", regime=ctx.regime.value,
                        conviction_mod=ctx.conviction_modifier,
                        position_size_mod=ctx.position_size_modifier)
            print(con.warning(f"New longs BLOCKED — market regime is {ctx.regime.value}."))
            return

        try:
            try:
                account = get_account()
                equity = float(account.equity)
                cash = float(account.cash)
            except Exception as e:
                log.error("account_fetch_failed_during_eval", error=str(e))
                print(con.error(f"Could not fetch account info — {e}. Skipping evaluation."))
                return

            try:
                open_positions = self.orders.get_open_positions()
            except Exception as e:
                log.warning("position_fetch_failed_during_eval", error=str(e))
                open_positions = []

            log.info(
                "evaluate_start",
                equity=equity,
                cash=cash,
                open_positions=len(open_positions),
                candidates=len(self._candidates),
                regime=ctx.regime.value if ctx else "unknown",
                regime_conviction_mod=ctx.conviction_modifier if ctx else 1.0,
                dry_run=self.dry_run,
            )

            # Gate: daily loss limit
            ok, reason = self.risk.check_daily_loss_limit(equity)
            if not ok:
                log.warning("trading_halted", reason=reason)
                return

            # Filter out symbols that have repeatedly failed orders (halted, untradable, etc.)
            symbols = [c["symbol"] for c in self._candidates if c["symbol"] not in self._failed_symbols]
            if len(self._failed_symbols) > 0:
                log.info("blacklisted_symbols_filtered", blacklisted=list(self._failed_symbols.keys()))
            end = datetime.now(ET)
            start = end - timedelta(days=60)

            # V2 Phase 13: Parallel data fetching — daily bars, intraday bars, and
            # news are independent data sources. Fetch them concurrently to cut the
            # data-fetch phase wall-clock time by ~2-3x.
            self._cycle_timer.reset()

            daily_bars: dict[str, pd.DataFrame] = {}
            intraday_bars: dict[str, pd.DataFrame] = {}
            news_sentiment: dict = {}

            with self._cycle_timer.phase("fetch"):
                intraday_start = end - timedelta(hours=2)

                def _fetch_daily():
                    return fetch_bars_multi(symbols, TimeFrame.Day, start, end)

                def _fetch_intraday():
                    return fetch_bars_multi(symbols, TimeFrame.Minute, intraday_start, end)

                def _fetch_news():
                    return self.news_scanner.scan_symbols(symbols)

                with ThreadPoolExecutor(max_workers=3, thread_name_prefix="fetch") as pool:
                    fut_daily = pool.submit(_fetch_daily)
                    fut_intraday = pool.submit(_fetch_intraday)
                    fut_news = pool.submit(_fetch_news)

                    # Collect results — each task is independent, so partial failures
                    # don't block the others.
                    try:
                        daily_bars = fut_daily.result(timeout=30)
                    except Exception as e:
                        log.warning("daily_bars_fetch_failed", error=str(e))
                        daily_bars = {s: pd.DataFrame() for s in symbols}

                    try:
                        intraday_bars = fut_intraday.result(timeout=30)
                    except Exception as e:
                        log.warning("intraday_bars_fetch_failed", error=str(e))

                    try:
                        news_sentiment = fut_news.result(timeout=30)
                    except Exception as e:
                        log.warning(
                            "news_scan_failed",
                            error_type=type(e).__name__,
                            error=str(e) or repr(e),
                            symbols_count=len(symbols),
                        )

            # Log data availability for debugging
            bars_empty = [s for s, df in daily_bars.items() if df.empty]
            bars_ok = [s for s, df in daily_bars.items() if not df.empty]
            log.info("bars_fetched", symbols_with_data=len(bars_ok), symbols_empty=len(bars_empty),
                      empty_symbols=bars_empty[:10] if bars_empty else [])
            intraday_ok = [s for s, df in intraday_bars.items() if not df.empty]
            if intraday_ok:
                log.info("intraday_bars_fetched", symbols_with_data=len(intraday_ok))

            # Add technical indicators to all DataFrames.
            # Need at least 21 bars for Bollinger Bands (20-period window + 1).
            with self._cycle_timer.phase("indicators"):
                insufficient = []
                for sym, df in list(daily_bars.items()):
                    if df.empty:
                        continue
                    if len(df) < _MIN_BARS:
                        insufficient.append(f"{sym}({len(df)})")
                        daily_bars[sym] = pd.DataFrame()  # Mark as empty so strategies skip
                        continue
                    try:
                        add_all(df, intraday=False)
                    except Exception as e:
                        log.warning("indicator_failed", symbol=sym, rows=len(df), error=str(e))
                        daily_bars[sym] = pd.DataFrame()
                if insufficient:
                    log.info("bars_insufficient", symbols=insufficient)

            # Log news catalysts and earnings proximity
            if news_sentiment:
                catalysts = [s for s, ns in news_sentiment.items() if ns.catalyst_detected]
                if catalysts:
                    log.info("news_catalysts_detected", symbols=catalysts)
                earnings_flagged = [
                    s for s, ns in news_sentiment.items()
                    if ns.earnings_status != "clear"
                ]
                if earnings_flagged:
                    log.info("earnings_proximity_detected", symbols=earnings_flagged)

            # V2 Phase 11: Economic calendar awareness
            econ_events = []
            econ_modifier = 1.0
            try:
                econ_events = get_events_for_date()
                if econ_events:
                    econ_modifier = conviction_modifier_for_events(econ_events)
                    event_names = [e.name for e in econ_events]
                    log.info(
                        "economic_events_today",
                        events=event_names,
                        conviction_modifier=econ_modifier,
                    )
            except Exception as e:
                log.warning("economic_calendar_failed", error=str(e))

            # Build the set of symbols we already hold to prevent duplicates
            held_symbols = set()
            for pos in open_positions:
                sym = getattr(pos, "symbol", None) or pos.get("symbol") if isinstance(pos, dict) else getattr(pos, "symbol", None)
                if sym:
                    held_symbols.add(sym)
            # Also check DB open trades (catches positions from earlier today)
            try:
                db_open = self.db.get_open_trades()
                for t in db_open:
                    held_symbols.add(t.get("symbol", ""))
            except Exception as e:
                log.debug("db_open_trades_fetch_failed", error=str(e))
            if held_symbols:
                log.info("held_symbols_excluded", symbols=list(held_symbols))

            # The "brain": collect signals from all strategies, rank them,
            # and build the execution queue
            self.aggregator.set_market_context(self._market_context)
            with self._cycle_timer.phase("evaluate"):
                execution_queue = self.aggregator.collect_and_rank(
                    candidates=symbols,
                    daily_bars_dict=daily_bars,
                    intraday_bars_dict=intraday_bars,
                    account_equity=equity,
                    available_cash=cash,
                    held_symbols=held_symbols,
                )

            if not execution_queue:
                log.info("no_stock_signals_passed_filters",
                          candidates=len(symbols),
                          bars_with_data=len(bars_ok),
                          pdt_remaining=self.pdt.day_trades_remaining())
                print(con.detail(f"No signals passed filters ({len(symbols)} candidates, {self.pdt.day_trades_remaining()} PDT slots left)."))
                self._print_cycle_summary([], equity, cash, len(open_positions), ctx)
                self.decisions.flush()
                return

            log.info(
                "execution_queue_ready",
                queue_size=len(execution_queue),
                symbols=[item["signal"].symbol for item in execution_queue],
                strategies=[item["signal"].strategy_name for item in execution_queue],
            )

            # V2: Log ranked signals
            for rank, item in enumerate(execution_queue, 1):
                s = item["signal"]
                risk = s.entry_price - s.stop_loss_price
                reward = s.take_profit_price - s.entry_price
                rr = reward / risk if risk > 0 else 0
                self.decisions.log_rank(
                    s.symbol, s.strategy_name, s.conviction, rank,
                    reasoning=f"entry=${s.entry_price:.2f} stop=${s.stop_loss_price:.2f} target=${s.take_profit_price:.2f} R:R=1:{rr:.1f}",
                    factors={
                        "entry": s.entry_price, "stop": s.stop_loss_price,
                        "target": s.take_profit_price, "shares": item.get("shares", 0),
                        "hold_type": s.hold_type.value, "rr_ratio": round(rr, 2),
                    },
                )
                print(con.signal_line(
                    symbol=s.symbol, strategy=s.strategy_name,
                    conviction=s.conviction, hold_type=s.hold_type.value,
                    entry=s.entry_price, stop=s.stop_loss_price,
                    target=s.take_profit_price,
                ))

            # V2 Phase 12: Compute momentum scores for queued symbols
            momentum_scores = {}
            try:
                for item in execution_queue:
                    sym = item["signal"].symbol
                    if sym in daily_bars and sym not in momentum_scores:
                        ms = compute_momentum_score(daily_bars[sym], sym)
                        momentum_scores[sym] = ms
            except Exception as e:
                log.debug("momentum_scoring_failed", error=str(e))

            # Submit orders — apply sentiment + momentum modifiers before execution
            with self._cycle_timer.phase("execute"):
                for item in execution_queue:
                    self._submit_stock_signal(
                        item, ctx, news_sentiment, econ_modifier, momentum_scores
                    )

            # V2: Rich cycle summary (regime, signals, near-misses, portfolio)
            self._print_cycle_summary(
                execution_queue, equity, cash, len(open_positions), ctx
            )

            # V2 Phase 13: Log cycle timing summary
            self._cycle_timer.log_summary()
            timing_line = self._cycle_timer.summary_line()
            if timing_line:
                print(con.detail(f"Cycle timing: {timing_line}"))

            # V2: Flush all buffered decisions to DB
            self.decisions.flush()

        except Exception as e:
            log.error("stock_evaluate_failed", error=str(e), error_type=type(e).__name__)
            log.debug("stock_evaluate_traceback", tb=traceback.format_exc())
            self._cycle_timer.log_summary()  # Log timing even on error
            self.decisions.flush()  # Flush even on error

    def _print_cycle_summary(
        self, execution_queue: list[dict], equity: float, cash: float,
        open_positions: int, ctx,
    ) -> None:
        """V2: Print rich cycle summary with regime, signals, near-misses."""
        try:
            signals_data: list[dict] = []
            for item in execution_queue:
                s = item["signal"]
                signals_data.append({
                    "symbol": s.symbol,
                    "strategy": s.strategy_name,
                    "conviction": s.conviction,
                    "entry": s.entry_price,
                    "stop": s.stop_loss_price,
                    "target": s.take_profit_price,
                    "hold_type": s.hold_type.value,
                    "shares": item.get("shares", 0),
                    "action": "QUEUED",
                })

            near_misses = self.aggregator.get_near_misses() if self.aggregator else []

            # Portfolio heat = total risk / equity
            heat_pct = 0.0
            try:
                open_trades = self.db.get_open_trades()
                total_risk = sum(
                    abs(t.get("entry_price", 0) - t.get("stop_loss_price", 0)) * t.get("shares", 0)
                    for t in open_trades
                )
                heat_pct = (total_risk / equity * 100) if equity > 0 else 0.0
            except Exception:
                pass

            vix = getattr(ctx, "vix", 0.0) if ctx else 0.0
            regime = ctx.regime.value if ctx else "unknown"
            pdt_used = 3 - self.pdt.day_trades_remaining()

            summary = con.cycle_summary(
                regime=regime,
                vix=vix,
                pdt_used=pdt_used,
                pdt_max=3,
                candidates=len(self._candidates),
                momentum=self._scan_counts.get("momentum", 0),
                mean_rev=self._scan_counts.get("mean_reversion", 0),
                vwap=self._scan_counts.get("vwap", 0),
                signals=signals_data,
                near_misses=near_misses,
                equity=equity,
                cash=cash,
                open_positions=open_positions,
                heat_pct=heat_pct,
            )
            print(summary)
        except Exception as e:
            log.debug("cycle_summary_failed", error=str(e))

    def _submit_stock_signal(self, item: dict, ctx, news_sentiment: dict,
                             econ_modifier: float = 1.0,
                             momentum_scores: dict | None = None) -> None:
        """Apply sentiment modifiers and submit a single stock order.

        Extracted from _evaluate_and_trade() to keep the main loop readable.
        Handles regime/news modifiers, earnings guard, economic calendar,
        momentum prediction, conviction gating, dry-run mode, and PDT tracking.
        """
        sig = item["signal"]
        shares = item["shares"]

        # Apply market regime modifier to conviction
        original_conviction = sig.conviction
        if ctx:
            sig.conviction = min(1.0, sig.conviction * ctx.conviction_modifier)
            shares = max(1, int(shares * ctx.position_size_modifier))

        # V2 Phase 11: Earnings proximity guard — block entries near earnings
        ns = news_sentiment.get(sig.symbol)
        if ns and ns.earnings_status != "clear":
            earnings_block = getattr(self.cfg.sentiment, "block_near_earnings", True)
            if earnings_block:
                reason = f"earnings {ns.earnings_status} — binary event risk"
                log.info(
                    "trade_blocked_by_earnings",
                    symbol=sig.symbol,
                    earnings_status=ns.earnings_status,
                )
                print(con.skip(f"{sig.symbol}: blocked — {reason}"))
                self.db.update_signal_action(sig.symbol, sig.strategy_name, "blocked_earnings")
                self.decisions.log_reject(
                    sig.symbol, sig.strategy_name, reason,
                    conviction=sig.conviction,
                    factors={"earnings_status": ns.earnings_status},
                )
                return

        # Apply news sentiment modifier to conviction
        if ns:
            sig.conviction = min(1.0, sig.conviction * ns.conviction_modifier)
            block_threshold = getattr(self.cfg.sentiment, "block_on_bearish_news", -0.5)
            if ns.net_score < block_threshold and not ns.catalyst_detected:
                log.info(
                    "trade_blocked_by_news",
                    symbol=sig.symbol,
                    net_score=ns.net_score,
                    headline=ns.top_headline[:80],
                )
                print(con.skip(f"{sig.symbol}: blocked by bearish news ({ns.net_score:+.2f}) — \"{ns.top_headline[:60]}\""))
                self.db.update_signal_action(sig.symbol, sig.strategy_name, "blocked_bearish_news")
                self.decisions.log_reject(
                    sig.symbol, sig.strategy_name,
                    f"blocked by bearish news (score={ns.net_score:+.2f}): {ns.top_headline[:60]}",
                    conviction=sig.conviction,
                    factors={"news_score": ns.net_score},
                )
                return

        # V2 Phase 11: Economic calendar conviction reduction
        if econ_modifier < 1.0:
            sig.conviction = min(1.0, sig.conviction * econ_modifier)

        # V2 Phase 12: Momentum prediction modifier
        if momentum_scores:
            ms = momentum_scores.get(sig.symbol)
            if ms:
                mm = momentum_conviction_modifier(ms)
                if mm != 1.0:
                    sig.conviction = min(1.0, sig.conviction * mm)

        # Final conviction gate after all modifiers
        min_post_mod = getattr(self.cfg.sentiment, "min_conviction_after_mods", 0.50)
        if sig.conviction < min_post_mod:
            log.info(
                "trade_below_min_conviction",
                symbol=sig.symbol,
                conviction=sig.conviction,
                original=original_conviction,
            )
            print(con.skip(f"{sig.symbol}: conviction too low after modifiers ({original_conviction:.2f} -> {sig.conviction:.2f})"))
            self.db.update_signal_action(sig.symbol, sig.strategy_name, "rejected_low_conviction")
            self.decisions.log_reject(
                sig.symbol, sig.strategy_name,
                f"conviction too low after modifiers: {original_conviction:.2f} -> {sig.conviction:.2f} (min={min_post_mod})",
                conviction=sig.conviction,
                factors={"original_conviction": original_conviction, "min_required": min_post_mod},
            )
            return

        # Email alert for high-conviction signals (>= 0.70)
        if sig.conviction >= 0.70:
            notify_high_conviction_signal(
                symbol=sig.symbol,
                strategy=sig.strategy_name,
                conviction=sig.conviction,
                hold_type=sig.hold_type.value,
                entry_price=sig.entry_price,
                stop_loss=sig.stop_loss_price,
                take_profit=sig.take_profit_price,
            )

        # Dry-run mode: log the signal but don't submit orders
        if self.dry_run:
            log.info(
                "dry_run_stock_signal",
                symbol=sig.symbol,
                strategy=sig.strategy_name,
                conviction=sig.conviction,
                original_conviction=original_conviction,
                hold_type=sig.hold_type.value,
                shares=shares,
                entry=sig.entry_price,
                stop=sig.stop_loss_price,
                target=sig.take_profit_price,
                regime=ctx.regime.value if ctx else "unknown",
                news_score=ns.net_score if ns else 0,
            )
            self.db.update_signal_action(sig.symbol, sig.strategy_name, "dry_run")
            return

        # Pre-check: skip if this symbol is already blacklisted
        if sig.symbol in self._failed_symbols:
            self.decisions.log_reject(
                sig.symbol, sig.strategy_name,
                f"symbol blacklisted: {self._failed_symbols[sig.symbol]}",
                conviction=sig.conviction,
            )
            return

        # Pre-check: verify PDT budget before submitting
        if self.pdt.would_be_day_trade(sig.hold_type) and not self.pdt.can_day_trade():
            log.info("order_skipped_pdt_exhausted", symbol=sig.symbol)
            print(con.skip(f"{sig.symbol}: PDT slots exhausted, skipping day trade."))
            self.decisions.log_reject(
                sig.symbol, sig.strategy_name,
                f"PDT slots exhausted ({self.pdt.get_day_trades_used()}/{self.cfg.pdt.max_day_trades} used)",
                conviction=sig.conviction,
                factors={"hold_type": sig.hold_type.value, "pdt_used": self.pdt.get_day_trades_used()},
            )
            return

        # Submit the bracket order to Alpaca
        order_id = self.orders.submit_bracket_order(sig, shares)
        if order_id:
            self.db.update_signal_action(sig.symbol, sig.strategy_name, "executed")
            cost = shares * sig.entry_price
            risk_amount = shares * abs(sig.entry_price - sig.stop_loss_price)
            self.decisions.log_execute(
                sig.symbol, sig.strategy_name, sig.conviction,
                reasoning=f"{shares} shares @ ${sig.entry_price:.2f} = ${cost:.2f}, risk=${risk_amount:.2f}, order={order_id}",
                factors={
                    "shares": shares, "entry": sig.entry_price, "cost": cost,
                    "risk_amount": risk_amount, "order_id": str(order_id),
                    "hold_type": sig.hold_type.value,
                    "stop": sig.stop_loss_price, "target": sig.take_profit_price,
                },
            )
            # V2 Phase 5: snapshot the feature vector for ML training.
            # We look up the newly inserted trade row by buy_order_id
            # instead of changing submit_bracket_order's return type.
            self._snapshot_features_for_trade(sig, order_id)
            trade_type = sig.hold_type.value.upper()
            log.info(
                "stock_order_submitted",
                symbol=sig.symbol,
                strategy=sig.strategy_name,
                shares=shares,
                order_id=order_id,
            )
            print(con.order_submitted(
                symbol=sig.symbol, shares=shares, entry=sig.entry_price,
                cost=cost, hold_type=sig.hold_type.value,
                strategy=sig.strategy_name,
                stop=sig.stop_loss_price, target=sig.take_profit_price,
                pdt_remaining=self.pdt.day_trades_remaining() if self.pdt.would_be_day_trade(sig.hold_type) else None,
            ))
            notify_stock_order(
                symbol=sig.symbol,
                strategy=sig.strategy_name,
                shares=shares,
                entry_price=sig.entry_price,
                stop_loss=sig.stop_loss_price,
                take_profit=sig.take_profit_price,
                hold_type=sig.hold_type.value,
                conviction=sig.conviction,
                order_id=order_id,
                cost=cost,
            )
            if self.pdt.would_be_day_trade(sig.hold_type):
                today = datetime.now(ET).strftime("%Y-%m-%d")
                self.pdt.record_day_trade(sig.symbol, today, buy_order_id=str(order_id))
        else:
            self.db.update_signal_action(sig.symbol, sig.strategy_name, "order_failed")
            log.warning(
                "stock_order_failed",
                symbol=sig.symbol,
                strategy=sig.strategy_name,
                shares=shares,
            )
            print(con.order_failed(sig.symbol, sig.strategy_name, shares))
            notify_stock_order_failed(
                symbol=sig.symbol,
                strategy=sig.strategy_name,
                shares=shares,
            )
            # Track consecutive failures — blacklist after 2 failures to avoid
            # retrying halted/untradable symbols every 15-minute cycle.
            count = self._failed_symbol_counts.get(sig.symbol, 0) + 1
            self._failed_symbol_counts[sig.symbol] = count
            if count >= 2:
                self._failed_symbols[sig.symbol] = "repeated_order_failure"
                log.info("symbol_blacklisted", symbol=sig.symbol, failures=count)

    # ══════════════════════════════════════════════════════════
    # Core Logic: Options evaluation pipeline
    # ══════════════════════════════════════════════════════════

    def _evaluate_options(self) -> None:
        """Run options strategies on candidates and submit approved trades.

        Similar to _evaluate_and_trade() but for options:
          1. Gate: market regime allows options?
          2. Gate: daily loss limit not hit?
          3. Gate: options position and capital limits not exceeded?
          4. For each candidate: fetch options chain + snapshots
          5. Run each options strategy
          6. Risk-check each signal
          7. Submit orders for approved signals
        """
        # Determine which options biases the current regime permits
        ctx = self._market_context
        allowed_biases = ctx.allowed_options_biases if ctx else ("bullish", "bearish", "neutral", "adaptive")
        if not allowed_biases:
            log.info("options_blocked_by_regime", regime=ctx.regime.value if ctx else "unknown")
            print(con.detail(f"Options: all biases blocked by {ctx.regime.value if ctx else 'unknown'} regime."))
            return

        # Use the options-specific universe; fall back to stock candidates
        candidates = self._options_candidates or self._candidates
        if not candidates or not self.options_strategies:
            log.info("options_skip", candidates=len(candidates or []),
                      strategies=len(self.options_strategies))
            return

        # Log which biases are active so the operator can see what's running
        regime_name = ctx.regime.value if ctx else "unknown"
        eligible = [
            type(s).__name__ for s in self.options_strategies
            if s.enabled and getattr(s, "bias", "neutral") in allowed_biases
        ]
        if eligible:
            print(con.detail(f"Options: {regime_name} regime allows [{', '.join(allowed_biases)}] — running {len(eligible)} eligible strategies."))
        else:
            print(con.detail(f"Options: no eligible strategies for {regime_name} regime (allowed biases: {', '.join(allowed_biases)})."))
            return
        log.info(
            "options_regime_filter",
            regime=regime_name,
            allowed_biases=list(allowed_biases),
            eligible_strategies=eligible,
        )

        try:
            account = get_account()
            equity = float(account.equity)
            cash = float(account.cash)

            # Gate: daily loss limit
            ok, reason = self.risk.check_daily_loss_limit(equity)
            if not ok:
                return

            # Gate: check options-specific position limits
            open_options = self.db.get_open_options_trades()
            max_opts = getattr(self.cfg.options, "max_options_positions", 3)
            if len(open_options) >= max_opts:
                log.info("max_options_positions_reached", count=len(open_options))
                return

            # Gate: check capital allocation to options
            # Track how much capital is already committed to open options
            max_cap_pct = getattr(self.cfg.options, "max_options_capital_pct", 0.40)
            options_capital_used = sum(
                abs(t.get("entry_debit", 0) or 0) + abs(t.get("max_loss", 0) or 0)
                for t in open_options
            )
            options_budget = equity * max_cap_pct - options_capital_used
            if options_budget <= 0:
                log.info("options_capital_exhausted")
                return

            max_single_risk_pct = getattr(self.cfg.options, "max_single_options_risk_pct", 0.12)
            max_single_risk = equity * max_single_risk_pct

            # Skip underlyings we already have options positions on
            held_underlyings = {t.get("underlying", "") for t in open_options}
            candidates = [c for c in candidates if c["symbol"] not in held_underlyings]
            if not candidates:
                log.info("options_all_candidates_held")
                return

            # Fetch daily bars for all candidates
            symbols = [c["symbol"] for c in candidates]
            end = datetime.now(ET)
            start = end - timedelta(days=60)

            daily_bars = fetch_bars_multi(symbols, TimeFrame.Day, start, end)
            for sym, df in list(daily_bars.items()):
                if df.empty or len(df) < _MIN_BARS:
                    daily_bars[sym] = pd.DataFrame()
                    continue
                try:
                    add_all(df, intraday=False)
                except Exception as e:
                    log.debug("options_indicator_failed", symbol=sym, error=str(e))
                    daily_bars[sym] = pd.DataFrame()

            # ── Phase 1: Collect all options signals ──────────────
            all_options_signals = self._collect_options_signals(
                symbols, daily_bars, allowed_biases
            )

            if not all_options_signals:
                log.info("no_options_signals", candidates=len(symbols))
                print(con.detail(f"Options: no signals generated from {len(symbols)} candidates."))
                return

            # ── Phase 2: Rank by expected ROI and submit ──────────
            def _options_roi(sig):
                """ROI = (max_profit / max_loss) * conviction. Higher is better."""
                if sig.max_loss <= 0:
                    return 0.0
                raw_roi = sig.max_profit / sig.max_loss if sig.max_profit > 0 else 0.0
                return raw_roi * sig.conviction

            all_options_signals.sort(key=_options_roi, reverse=True)

            log.info(
                "options_signals_ranked",
                total=len(all_options_signals),
                top_3=[(s.underlying, s.strategy_name, round(_options_roi(s), 2))
                       for s in all_options_signals[:3]],
            )
            for sig in all_options_signals[:5]:
                roi = _options_roi(sig)
                print(con.options_signal_line(
                    underlying=sig.underlying, strategy=sig.strategy_name,
                    roi=roi, conviction=sig.conviction,
                    max_loss=sig.max_loss, max_profit=sig.max_profit,
                ))

            positions_filled = 0
            for signal in all_options_signals:
                if options_budget <= 0 or len(open_options) + positions_filled >= max_opts:
                    break

                # Risk checks
                if signal.max_loss > max_single_risk:
                    log.info(
                        "options_trade_too_risky",
                        symbol=signal.underlying,
                        strategy=signal.strategy_name,
                        max_loss=signal.max_loss,
                        max_single_risk=round(max_single_risk, 2),
                    )
                    continue

                if signal.max_loss > options_budget:
                    continue

                # Dry-run mode: log but don't execute
                if self.dry_run:
                    log.info(
                        "dry_run_options_signal",
                        underlying=signal.underlying,
                        strategy=signal.strategy_name,
                        conviction=signal.conviction,
                        max_loss=signal.max_loss,
                        max_profit=signal.max_profit,
                        roi=round(_options_roi(signal), 2),
                        legs=len(signal.legs),
                        expiration=signal.expiration,
                    )
                    continue

                # Submit the options order to Alpaca
                order_id = self.options_orders.submit_options_order(signal)
                if order_id:
                    log.info(
                        "options_order_submitted",
                        underlying=signal.underlying,
                        strategy=signal.strategy_name,
                        order_id=order_id,
                        roi=round(_options_roi(signal), 2),
                    )
                    print(con.options_order(
                        underlying=signal.underlying,
                        strategy=signal.strategy_name,
                        legs=len(signal.legs),
                        max_loss=signal.max_loss,
                        max_profit=signal.max_profit,
                        roi=_options_roi(signal),
                        expiration=signal.expiration,
                    ))
                    notify_options_order(
                        underlying=signal.underlying,
                        strategy=signal.strategy_name,
                        legs=len(signal.legs),
                        max_loss=signal.max_loss,
                        max_profit=signal.max_profit,
                        roi=_options_roi(signal),
                        conviction=signal.conviction,
                        expiration=signal.expiration,
                        order_id=str(order_id),
                    )
                    # Record the trade in the options_trades table
                    self.db.insert_options_trade(
                        underlying=signal.underlying,
                        strategy=signal.strategy_name,
                        legs=json.dumps(signal.legs),
                        qty=1,
                        entry_credit=signal.min_credit if signal.min_credit > 0 else None,
                        entry_debit=signal.max_cost if signal.max_cost > 0 else None,
                        max_loss=signal.max_loss,
                        max_profit=signal.max_profit,
                        expiration=signal.expiration,
                        strikes=json.dumps(signal.strikes),
                        net_delta=signal.net_delta,
                        net_theta=signal.net_theta,
                        order_id=str(order_id),
                    )
                    self.decisions.log_execute(
                        signal.underlying, signal.strategy_name, signal.conviction,
                        reasoning=f"options order {order_id}: {len(signal.legs)} legs, max_loss=${signal.max_loss:.2f}, max_profit=${signal.max_profit:.2f}, ROI={_options_roi(signal):.2f}",
                        factors={
                            "order_id": str(order_id),
                            "legs": len(signal.legs),
                            "max_loss": signal.max_loss,
                            "max_profit": signal.max_profit,
                            "expiration": signal.expiration,
                            "strikes": signal.strikes,
                            "net_delta": signal.net_delta,
                            "net_theta": signal.net_theta,
                            "roi": _options_roi(signal),
                        },
                    )
                    # Deduct from budget and increment position count
                    options_budget -= signal.max_loss
                    positions_filled += 1

        except Exception as e:
            log.error("options_evaluate_failed", error=str(e))
            log.debug("options_evaluate_traceback", tb=traceback.format_exc())

    def _collect_options_signals(
        self, symbols: list[str], daily_bars: dict, allowed_biases: tuple
    ) -> list:
        """Iterate candidates × strategies and collect all passing options signals.

        For each symbol: fetch the options chain, run every eligible strategy,
        and collect signals that pass. Caches chain availability to skip
        symbols known to have no options.
        """
        signals = []
        for symbol in symbols:
            bars = daily_bars.get(symbol)
            if bars is None or bars.empty:
                continue

            if self.screener._options_chain_cache.get(symbol) is False:
                continue

            try:
                chain = get_options_chain(symbol)
                if not chain:
                    self.screener._options_chain_cache[symbol] = False
                    continue
                self.screener._options_chain_cache[symbol] = True
                option_symbols = [c["symbol"] for c in chain[:50]]
                snapshots = get_options_snapshot(option_symbols)
            except Exception as e:
                log.debug("options_chain_fetch_failed", symbol=symbol, error=str(e))
                continue

            for strategy in self.options_strategies:
                if not strategy.enabled:
                    continue
                strat_bias = getattr(strategy, "bias", "neutral")
                if strat_bias not in allowed_biases:
                    continue

                strat_name = type(strategy).__name__
                try:
                    signal = strategy.evaluate(symbol, bars, chain, snapshots)
                except Exception as e:
                    log.debug("options_strategy_error", strategy=strat_name,
                              symbol=symbol, error=str(e))
                    continue

                # Drain rejections (whether signal was produced or not)
                for r in strategy.drain_rejections():
                    self.decisions.log_evaluate(
                        symbol=r.symbol,
                        strategy=r.strategy,
                        action="near_miss" if r.is_near_miss else "reject",
                        reasoning=r.to_reasoning(),
                        factors={
                            "filter": r.filter_name,
                            "actual": r.actual,
                            "threshold": r.threshold,
                            "direction": r.direction,
                            "miss_pct": r.miss_pct,
                        },
                    )

                if signal is not None:
                    signals.append(signal)

        return signals


# ══════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════

def main():
    """Parse command-line arguments and start the trading bot.

    Usage:
        python -m ai_trade.main                    # Normal mode
        python -m ai_trade.main --dry-run           # Log signals, don't trade
        python -m ai_trade.main --config path.yaml  # Custom config file
    """
    parser = argparse.ArgumentParser(description="AI Trade — Automated Stock Trading Bot")
    parser.add_argument("--config", type=str, default=None, help="Path to settings.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Log signals without submitting orders")
    args = parser.parse_args()

    try:
        bot = TradingBot(config_path=args.config, dry_run=args.dry_run)
    except FileNotFoundError as e:
        print(f"\n  FATAL: Config file not found — {e}")
        print(f"  Create config/settings.yaml or pass --config <path>.")
        sys.exit(1)
    except Exception as e:
        print(f"\n  FATAL: Could not initialize TradingBot — {e}")
        sys.exit(1)

    # Register signal handlers for graceful shutdown.
    # SIGINT = Ctrl+C, SIGTERM = kill command.
    def _shutdown(signum, frame):
        bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        bot.start()
    except SystemExit:
        raise  # Let sys.exit() through
    except Exception as e:
        # Last-resort handler — if something completely unexpected crashes
        # the bot, log it and exit cleanly rather than showing a raw traceback.
        print(f"\n  FATAL UNHANDLED ERROR: {type(e).__name__}: {e}")
        try:
            log.error("fatal_unhandled_error", error=str(e), error_type=type(e).__name__)

            log.error("fatal_traceback", tb=traceback.format_exc())
        except Exception:
            pass
        try:
            bot.stop()
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
