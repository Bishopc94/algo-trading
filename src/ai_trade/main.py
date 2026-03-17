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
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.timeframe import TimeFrame

from ai_trade.config import load_config
from ai_trade.clients import init_clients, get_trading_client, get_account
from ai_trade.data.historical import fetch_bars, fetch_bars_multi
from ai_trade.data.indicators import add_all
from ai_trade.monitoring.database import Database
from ai_trade.monitoring.logger import setup_logging, get_logger
from ai_trade.monitoring.performance import PerformanceTracker
from ai_trade.risk.pdt_manager import PDTManager
from ai_trade.risk.position_sizer import PositionSizer
from ai_trade.risk.risk_manager import RiskManager
from ai_trade.execution.order_manager import OrderManager
from ai_trade.scanner.screener import StockScreener
from ai_trade.strategy.base import HoldType
from ai_trade.strategy.mean_reversion import MeanReversionStrategy
from ai_trade.strategy.momentum import MomentumStrategy
from ai_trade.strategy.vwap import VWAPStrategy
from ai_trade.strategy.signal import SignalAggregator
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

# Sentiment imports
from ai_trade.sentiment.market_regime import MarketRegimeAnalyzer, MarketRegime
from ai_trade.sentiment.news_sentiment import NewsSentimentScanner

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
        self.screener = StockScreener(self.cfg.scanner)         # Pre-market scanner
        self.pdt = PDTManager(self.cfg.pdt, self.db)            # Day-trade tracking
        self.sizer = PositionSizer(merged_cfg)                  # Position sizing
        self.risk = RiskManager(merged_cfg, self.db)            # Risk gate
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

        # The signal aggregator is the "brain" — it collects signals from
        # all strategies, ranks them, and builds the execution queue.
        self.aggregator = SignalAggregator(
            strategies=self.strategies,
            pdt_manager=self.pdt,
            risk_manager=self.risk,
            position_sizer=self.sizer,
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

        # ── Sentiment layer ──
        self.regime_analyzer = MarketRegimeAnalyzer()
        self.news_scanner = NewsSentimentScanner(lookback_hours=24, max_articles=10)
        self._market_context = None  # Set at market open by _analyze_market_regime()

        # ── State ──
        self._candidates: list[dict] = []  # Today's scanner candidates
        self._running = False

    # ══════════════════════════════════════════════════════════
    # Lifecycle
    # ══════════════════════════════════════════════════════════

    def start(self) -> None:
        """Initialize Alpaca clients, sync state, start the scheduler.

        This method blocks the main thread (via a sleep loop) until
        the user presses Ctrl+C or the bot is stopped programmatically.
        """
        setup_logging()
        init_clients(self.cfg)  # Create Alpaca API client singletons

        # Verify we can connect to Alpaca
        account = get_account()
        log.info(
            "connected",
            equity=float(account.equity),
            cash=float(account.cash),
            day_trade_count=account.daytrade_count,
            paper=self.cfg.alpaca.paper,
        )

        # Reconcile any positions from a previous run (e.g. if the bot
        # was restarted mid-day, positions may still be open on Alpaca)
        self.orders.sync_positions()

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
            mode=mode,
            current_time_et=now_et.strftime("%Y-%m-%d %H:%M:%S"),
            stock_strategies=stock_strat_names,
            options_strategies=options_strat_names,
            scanner_config={"min_price": getattr(self.cfg.scanner, "min_price", "?"),
                            "max_price": getattr(self.cfg.scanner, "max_price", "?"),
                            "min_gap_pct": getattr(self.cfg.scanner, "min_gap_pct", "?"),
                            "min_relative_volume": getattr(self.cfg.scanner, "min_relative_volume", "?")},
            schedule={"premarket_scan": self.cfg.schedule.premarket_scan,
                      "entry_window": self.cfg.schedule.entry_window,
                      "midday_check": self.cfg.schedule.midday_check},
        )
        print(f"\n  AI Trade Bot started [{mode}]")
        print(f"  Equity: ${float(account.equity):,.2f} | Cash: ${float(account.cash):,.2f}")
        print(f"  Stock strategies: {len(self.strategies)} | Options strategies: {len(self.options_strategies)}")
        print(f"  Day trades remaining: {self.pdt.day_trades_remaining()}")
        print(f"  Press Ctrl+C to stop.\n")

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
        """
        log.info("bot_stopping")
        self._running = False
        if hasattr(self, "_scheduler"):
            self._scheduler.shutdown(wait=False)
        print("\n  Bot stopped.\n")

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
            return

        hour_min = now_et.hour * 60 + now_et.minute
        market_open = 9 * 60 + 30   # 9:30 AM ET
        market_close = 16 * 60       # 4:00 PM ET

        if hour_min < market_open or hour_min >= market_close:
            log.info("catchup_skip", reason="outside_market_hours",
                      current_time=now_et.strftime("%H:%M"))
            return

        open_positions = self.orders.get_open_positions()
        has_positions = len(open_positions) > 0

        log.info("catchup_start", current_time=now_et.strftime("%H:%M"),
                  open_positions=len(open_positions))

        # Always scan for candidates on startup
        log.info("catchup_running", job="premarket_scan")
        self.job_premarket_scan()

        # Always set up market context (regime analysis)
        if self._market_context is None:
            log.info("catchup_running", job="market_open")
            self.job_market_open()

        # Only evaluate strategies if we have NO open positions.
        # If we already have positions, we wait for the next scheduled
        # window to avoid doubling down accidentally.
        if not has_positions:
            log.info("catchup_running", job="entry_window", reason="no_open_positions")
            self.job_entry_window()
        else:
            log.info("catchup_skip_entry", reason="has_open_positions",
                      count=len(open_positions))

        log.info("catchup_complete", scanned=True, evaluated=not has_positions)
        print(f"  Startup scan complete: {len(self._candidates)} candidates"
              f" | {'Evaluating strategies...' if not has_positions else f'{len(open_positions)} open positions, waiting for next window'}")

    # ══════════════════════════════════════════════════════════
    # Scheduled Jobs — called by APScheduler at specific times
    # ══════════════════════════════════════════════════════════

    def job_premarket_scan(self) -> None:
        """9:00 AM ET — Scan the stock universe for today's candidates.

        The scanner filters ~8000 stocks down to ~20 candidates based on
        price, gap%, and relative volume.  These candidates are then
        evaluated by all strategies during the entry window.
        """
        log.info("job_start", job="premarket_scan", time=datetime.now(ET).strftime("%H:%M:%S"))
        try:
            self._candidates = self.screener.scan()
            symbols = [c["symbol"] for c in self._candidates]
            log.info("scan_complete", candidates=len(self._candidates), symbols=symbols[:10])
            if not self._candidates:
                log.warning("scan_returned_zero_candidates",
                             hint="Check scanner filters: min_gap_pct, min_relative_volume, price range")
        except Exception as e:
            log.error("scan_failed", error=str(e), error_type=type(e).__name__)
            self._candidates = []

    def job_market_open(self) -> None:
        """9:30 AM ET — Cache starting equity, sync positions, analyze market.

        The starting equity is saved so the daily loss limit can be
        calculated.  Market regime analysis (SPY/QQQ/VIX) determines
        whether it's safe to take new positions today.
        """
        log.info("job_start", job="market_open")
        try:
            account = get_account()
            self.risk.set_starting_equity(float(account.equity))
            self.orders.sync_positions()

            # Analyze broad market conditions (SPY, QQQ, VIX)
            self._analyze_market_regime()

            log.info("market_open", equity=float(account.equity), cash=float(account.cash))
        except Exception as e:
            log.error("market_open_failed", error=str(e))

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
            except Exception:
                pass

            self._market_context = self.regime_analyzer.analyze(spy_bars, qqq_bars, vix_bars)
            log.info(
                "market_regime_result",
                regime=self._market_context.regime.value,
                allow_new_longs=self._market_context.allow_new_longs,
                allow_options=self._market_context.allow_options,
                conviction_modifier=self._market_context.conviction_modifier,
                position_size_modifier=self._market_context.position_size_modifier,
            )
            print(f"  Market: {self._market_context}")

        except Exception as e:
            log.warning("market_regime_failed", error=str(e))

    def job_entry_window(self) -> None:
        """9:35 AM ET — Evaluate all strategies and submit trades.

        This is the primary trading window.  Runs 5 minutes after market
        open to let the initial volatility settle.
        """
        log.info("job_start", job="entry_window")
        self._evaluate_and_trade()
        if self.options_enabled:
            self._evaluate_options()

    def job_midday_check(self) -> None:
        """12:00 PM ET — Re-evaluate positions and check for new swing setups.

        By midday, initial momentum plays have played out and new swing
        opportunities may have emerged.
        """
        log.info("job_start", job="midday_check")
        self.orders.sync_positions()
        self._evaluate_and_trade()
        if self.options_enabled:
            self._evaluate_options()

    def job_power_hour(self) -> None:
        """3:00 PM ET — Final scan for late momentum plays.

        "Power hour" (3-4 PM) often sees increased volume and momentum
        as institutional traders make their final moves before close.
        """
        log.info("job_start", job="power_hour")
        try:
            self._candidates = self.screener.scan()
            self._evaluate_and_trade()
        except Exception as e:
            log.error("power_hour_failed", error=str(e))

    def job_eod_close_day_trades(self) -> None:
        """3:50 PM ET — Force-close any open day-trade positions.

        Day trades MUST be closed before market close (4:00 PM).  We close
        at 3:50 to allow time for order execution.  This is critical for
        PDT compliance — holding a day trade overnight would still count
        as a day trade but with overnight risk.
        """
        log.info("job_start", job="eod_close_day_trades")
        try:
            open_trades = self.db.get_open_trades()
            self.orders.close_all_day_trades(open_trades)
        except Exception as e:
            log.error("eod_close_failed", error=str(e))

    def job_eod_review(self) -> None:
        """4:05 PM ET — Daily P&L summary, save snapshot to database.

        This runs 5 minutes after market close to capture the final
        settlement prices.  The snapshot is used for the equity curve
        and performance metrics.
        """
        log.info("job_start", job="eod_review")
        try:
            account = get_account()
            positions = self.orders.get_open_positions()

            # Save daily snapshot for the equity curve
            today = datetime.now(ET).strftime("%Y-%m-%d")
            self.db.save_snapshot(
                date=today,
                equity=float(account.equity),
                cash=float(account.cash),
                open_positions=len(positions),
                day_trades_used=self.pdt.get_day_trades_used(),
                realized_pnl=0.0,
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

    def job_sync_positions(self) -> None:
        """Every 60 seconds — Reconcile Alpaca positions with local DB.

        This is a safety net: if a bracket order's stop-loss or take-profit
        fills on Alpaca's side, the local database needs to be updated.
        Without this sync, the DB would show stale "open" trades.
        """
        try:
            self.orders.sync_positions()
        except Exception as e:
            log.warning("sync_failed", error=str(e))

    # ══════════════════════════════════════════════════════════
    # Core Logic: Stock evaluation pipeline
    # ══════════════════════════════════════════════════════════

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
            return

        # Gate: market regime check — don't go long into a strong bear market
        ctx = self._market_context
        if ctx and not ctx.allow_new_longs:
            log.warning("longs_blocked_by_regime", regime=ctx.regime.value,
                        conviction_mod=ctx.conviction_modifier,
                        position_size_mod=ctx.position_size_modifier)
            return

        try:
            account = get_account()
            equity = float(account.equity)
            cash = float(account.cash)
            open_positions = self.orders.get_open_positions()

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

            # Fetch 60 days of daily bars for all candidates at once
            symbols = [c["symbol"] for c in self._candidates]
            end = datetime.now(ET)
            start = end - timedelta(days=60)

            daily_bars = fetch_bars_multi(symbols, TimeFrame.Day, start, end)

            # Log data availability for debugging
            bars_empty = [s for s, df in daily_bars.items() if df.empty]
            bars_ok = [s for s, df in daily_bars.items() if not df.empty]
            log.info("bars_fetched", symbols_with_data=len(bars_ok), symbols_empty=len(bars_empty),
                      empty_symbols=bars_empty[:10] if bars_empty else [])

            # Add technical indicators to all DataFrames.
            # Need at least 21 bars for Bollinger Bands (20-period window + 1).
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

            # Fetch 2 hours of minute bars for the VWAP strategy
            intraday_bars: dict[str, pd.DataFrame] = {}
            try:
                intraday_start = end - timedelta(hours=2)
                intraday_bars = fetch_bars_multi(symbols, TimeFrame.Minute, intraday_start, end)
                intraday_ok = [s for s, df in intraday_bars.items() if not df.empty]
                log.info("intraday_bars_fetched", symbols_with_data=len(intraday_ok))
            except Exception as e:
                log.warning("intraday_bars_failed", error=str(e))

            # Scan news sentiment for all candidates
            news_sentiment = {}
            try:
                news_sentiment = self.news_scanner.scan_symbols(symbols)
                catalysts = [s for s, ns in news_sentiment.items() if ns.catalyst_detected]
                if catalysts:
                    log.info("news_catalysts_detected", symbols=catalysts)
            except Exception as e:
                log.warning("news_scan_failed", error=str(e))

            # The "brain": collect signals from all strategies, rank them,
            # and build the execution queue
            execution_queue = self.aggregator.collect_and_rank(
                candidates=symbols,
                daily_bars_dict=daily_bars,
                intraday_bars_dict=intraday_bars,
                account_equity=equity,
                available_cash=cash,
            )

            if not execution_queue:
                log.info("no_stock_signals_passed_filters",
                          candidates=len(symbols),
                          bars_with_data=len(bars_ok),
                          pdt_remaining=self.pdt.day_trades_remaining())
                return

            log.info(
                "execution_queue_ready",
                queue_size=len(execution_queue),
                symbols=[item["signal"].symbol for item in execution_queue],
                strategies=[item["signal"].strategy_name for item in execution_queue],
            )

            # Submit orders — apply sentiment modifiers before execution
            for item in execution_queue:
                sig = item["signal"]
                shares = item["shares"]

                # Apply market regime modifier to conviction
                original_conviction = sig.conviction
                if ctx:
                    sig.conviction = min(1.0, sig.conviction * ctx.conviction_modifier)
                    # Also adjust position size by regime (e.g. 0.5x in bear market)
                    shares = max(1, int(shares * ctx.position_size_modifier))

                # Apply news sentiment modifier to conviction
                ns = news_sentiment.get(sig.symbol)
                if ns:
                    sig.conviction = min(1.0, sig.conviction * ns.conviction_modifier)
                    # Block trades on very bearish news (unless a catalyst was detected)
                    if ns.net_score < -0.5 and not ns.catalyst_detected:
                        log.info(
                            "trade_blocked_by_news",
                            symbol=sig.symbol,
                            net_score=ns.net_score,
                            headline=ns.top_headline[:80],
                        )
                        continue

                # Final conviction gate after all modifiers
                if sig.conviction < 0.35:
                    log.info(
                        "trade_below_min_conviction",
                        symbol=sig.symbol,
                        conviction=sig.conviction,
                        original=original_conviction,
                    )
                    continue

                # In dry-run mode, log the signal but don't submit orders
                if self.dry_run:
                    news_info = f" | news={ns.net_score:+.2f}" if ns and ns.article_count > 0 else ""
                    regime_info = f" | regime={ctx.regime.value}" if ctx else ""
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
                    continue

                # Submit the bracket order to Alpaca
                order_id = self.orders.submit_bracket_order(sig, shares)
                if order_id:
                    log.info(
                        "stock_order_submitted",
                        symbol=sig.symbol,
                        strategy=sig.strategy_name,
                        shares=shares,
                        order_id=order_id,
                    )
                    # Record the day trade if applicable (for PDT tracking)
                    if self.pdt.would_be_day_trade(sig.hold_type):
                        today = datetime.now(ET).strftime("%Y-%m-%d")
                        self.pdt.record_day_trade(sig.symbol, today, buy_order_id=str(order_id))

        except Exception as e:
            log.error("stock_evaluate_failed", error=str(e), error_type=type(e).__name__)
            import traceback
            log.debug("stock_evaluate_traceback", tb=traceback.format_exc())

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
        # Gate: market regime must allow options
        ctx = self._market_context
        if ctx and not ctx.allow_options:
            log.info("options_blocked_by_regime", regime=ctx.regime.value)
            return

        if not self._candidates or not self.options_strategies:
            log.info("options_skip", candidates=len(self._candidates or []),
                      strategies=len(self.options_strategies))
            return

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

            max_single_risk = getattr(self.cfg.options, "max_single_options_risk", 100.0)

            # Fetch daily bars for all candidates
            symbols = [c["symbol"] for c in self._candidates]
            end = datetime.now(ET)
            start = end - timedelta(days=60)

            daily_bars = fetch_bars_multi(symbols, TimeFrame.Day, start, end)
            for sym, df in list(daily_bars.items()):
                if df.empty or len(df) < _MIN_BARS:
                    daily_bars[sym] = pd.DataFrame()
                    continue
                try:
                    add_all(df, intraday=False)
                except Exception:
                    daily_bars[sym] = pd.DataFrame()

            # Evaluate each candidate with each options strategy
            for symbol in symbols:
                # Stop if we've exhausted budget or positions
                if options_budget <= 0 or len(open_options) >= max_opts:
                    break

                bars = daily_bars.get(symbol)
                if bars is None or bars.empty:
                    continue

                # Fetch the options chain for this underlying stock
                try:
                    chain = get_options_chain(symbol)
                    if not chain:
                        continue
                    # Limit to 50 contracts to avoid excessive API calls
                    option_symbols = [c["symbol"] for c in chain[:50]]
                    snapshots = get_options_snapshot(option_symbols)
                except Exception as e:
                    log.debug("options_chain_fetch_failed", symbol=symbol, error=str(e))
                    continue

                # Run each options strategy against this candidate
                for strategy in self.options_strategies:
                    if not strategy.enabled:
                        continue

                    strat_name = type(strategy).__name__
                    try:
                        signal = strategy.evaluate(symbol, bars, chain, snapshots)
                    except Exception as e:
                        log.debug(
                            "options_strategy_error",
                            strategy=strat_name,
                            symbol=symbol,
                            error=str(e),
                        )
                        continue

                    if signal is None:
                        log.debug("options_no_signal", strategy=strat_name, symbol=symbol)
                        continue

                    # Risk checks for the options signal
                    if signal.max_loss > max_single_risk:
                        log.info(
                            "options_trade_too_risky",
                            symbol=symbol,
                            strategy=signal.strategy_name,
                            max_loss=signal.max_loss,
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
                        # Deduct from budget and increment position count
                        options_budget -= signal.max_loss
                        open_options.append({})  # Increment count

        except Exception as e:
            log.error("options_evaluate_failed", error=str(e))


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

    bot = TradingBot(config_path=args.config, dry_run=args.dry_run)

    # Register signal handlers for graceful shutdown.
    # SIGINT = Ctrl+C, SIGTERM = kill command.
    def _shutdown(signum, frame):
        bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    bot.start()


if __name__ == "__main__":
    main()
