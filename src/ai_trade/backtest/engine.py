"""Backtesting engine -- simulate strategy performance on historical data.

This is the **core simulation module** of the backtesting system. It walks
through historical price bars day-by-day, evaluates trading strategies,
simulates order fills with slippage, manages a virtual portfolio of stock
and options positions, enforces risk management rules, and produces
performance reports.

HOW IT FITS IN THE BACKTEST PIPELINE:
-------------------------------------
    runner.py  -->  engine.py  -->  options_pricing.py
    (CLI/setup)    (THIS FILE)     (synthetic pricing)

The runner fetches data and creates an engine instance. The engine then
takes over: it iterates through each trading day, asks strategies for
signals, sizes positions, manages exits, and tracks equity.

KEY DESIGN DECISIONS:
---------------------
1. **O(1) date lookups**: Rather than searching through DataFrames each day,
   the engine pre-builds a ``{symbol: {date_string: row_index}}`` dictionary
   at startup.  This makes price lookups constant-time regardless of how
   many trading days exist in the dataset.

2. **Exit-before-entry processing order**: Each day, the engine first checks
   stop-losses and take-profits on existing positions, THEN evaluates new
   entry signals.  This prevents the engine from using capital that should
   have been freed by a stop-loss hit earlier that same day.

3. **Equity caching**: The _equity() calculation (summing cash + position
   market values) is cached per day and invalidated at the start of each
   new day, since positions change intra-day.

TRADING CONCEPTS:
-----------------
- **Backtesting**: Running a strategy on historical data to measure how it
  *would have* performed.  Not a guarantee of future results.

- **OHLCV bars**: Each trading day is represented as Open, High, Low, Close,
  Volume.  The engine uses High/Low to check intraday stop/target triggers
  and Close for end-of-day valuations.

- **Stop-loss / Take-profit**: Automatic exit orders.  A stop-loss limits
  downside (sell if price drops to X).  A take-profit locks in gains (sell
  if price rises to Y).

- **Trailing stop**: A stop-loss that moves UP as the stock price rises,
  but never moves down.  It "trails" the price by a fixed percentage,
  protecting gains while allowing room for the position to run.

- **ATR (Average True Range)**: A volatility measure used by strategies to
  set stop-loss distances.  Higher ATR = wider stops = more room for the
  position to fluctuate without being stopped out.

- **Portfolio heat**: Total capital at risk across all open positions.
  If heat exceeds the limit (e.g., 6% of equity), no new positions are opened.

- **PDT (Pattern Day Trader) rule**: US regulation limiting accounts under
  $25k to 3 day trades per 5-business-day rolling window.  This engine
  simulates that constraint.

- **Slippage**: The difference between expected fill price and actual fill
  price.  In real markets, you rarely get exactly the price you expect.
  The engine simulates this by adding a small percentage to entries and
  subtracting from exits.

- **Equity curve**: A time series of daily portfolio values.  Used to compute
  metrics like max drawdown and Sharpe ratio.

- **Sharpe ratio**: Risk-adjusted return metric = (mean return) / (std dev of
  returns) * sqrt(252).  Higher is better.  Named after William Sharpe.

- **Max drawdown**: Largest peak-to-trough decline in portfolio value.
  A 10% drawdown means the portfolio fell 10% from its highest point before
  recovering.

- **Profit factor**: Gross profits / Gross losses.  Above 1.0 means the
  strategy is profitable overall.

PYTHON CONCEPTS FOR NON-PYTHON READERS:
---------------------------------------
- ``@dataclass``: A decorator that auto-generates __init__, __repr__, and
  __eq__ for simple data-holding classes.  ``field(default_factory=list)``
  means "default to a new empty list for each instance" (not a shared list).

- ``from __future__ import annotations``: Makes type hints lazy-evaluated,
  enabling syntax like ``str | None`` and forward references.

- ``dict[str, pd.DataFrame]``: A dictionary (hash map) with string keys and
  pandas DataFrame values.

- ``list comprehension``: ``[x for x in items if condition]`` is a concise
  way to build a filtered/transformed list.  Equivalent to map + filter.

- ``@property``: Makes a method accessible like an attribute (no parentheses).
  ``results.metrics`` looks like an attribute but runs a function.

- ``math.floor`` / ``math.sqrt``: Standard math functions.  ``floor`` rounds
  down (important for share counts -- you can't buy 2.7 shares).

Supports both stock strategies (momentum, mean reversion, VWAP) and options
strategies (long call, credit put spread, debit call spread, etc.) using
Black-Scholes synthetic pricing from options_pricing.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field  # field() customizes individual dataclass fields
from datetime import datetime, timedelta, timezone
from typing import Any  # 'Any' type hint means "any type is accepted"
from zoneinfo import ZoneInfo

import pandas as pd  # Pandas: tabular data library (DataFrames = tables with typed columns)

from ai_trade.backtest.options_pricing import (
    generate_synthetic_chain,   # Creates fake-but-realistic options chains
    historical_volatility,      # Computes annualized vol from price history
    price_option_at_expiration, # Intrinsic value at expiration
)
from ai_trade.data.indicators import add_all  # Adds technical indicators (RSI, Bollinger, ATR, etc.)
from ai_trade.monitoring.logger import get_logger
from ai_trade.sentiment.market_regime import MarketRegimeAnalyzer, MarketContext
from ai_trade.strategy.base import BaseStrategy, HoldType, Signal
from ai_trade.strategy.options.base import BaseOptionsStrategy, OptionsSignal

# get_logger(__name__) creates a structured logger tagged with this module's path.
# __name__ evaluates to "ai_trade.backtest.engine" at runtime.
log = get_logger(__name__)

# US Eastern timezone constant -- US stock markets trade 9:30-16:00 Eastern.
ET = ZoneInfo("America/New_York")


# ── Data structures ────────────────────────────────────────
# These dataclasses are simple containers (like structs in C or records in Java)
# that hold the state of positions, trades, and daily snapshots.  The @dataclass
# decorator auto-generates the constructor, so ``BacktestPosition(symbol="AAPL", ...)``
# works without writing an explicit __init__ method.


@dataclass
class BacktestPosition:
    """A simulated open stock position.

    Tracks everything needed to manage the position: entry details, current
    stop/target levels, and the highest price seen (for trailing stop logic).

    The ``__post_init__`` method is a special dataclass hook that runs after
    the auto-generated __init__.  Here it initializes ``highest_price`` to
    the entry price so the trailing stop has a starting reference point.
    """

    symbol: str               # Ticker symbol (e.g., "AAPL")
    shares: int               # Number of shares held
    entry_price: float        # Price paid per share (after slippage)
    entry_date: str           # ISO date string "YYYY-MM-DD"
    stop_loss: float          # Current stop-loss price (may be raised by trailing stop)
    take_profit: float        # Target price to sell for a gain
    hold_type: HoldType       # DAY or SWING (affects PDT counting and EOD close)
    strategy_name: str        # Which strategy generated this position
    highest_price: float = 0.0  # Highest price seen since entry (for trailing stop)

    def __post_init__(self):
        # Set the trailing stop baseline to entry price.  As the stock rises,
        # highest_price is updated, and the stop-loss is ratcheted up.
        self.highest_price = self.entry_price


@dataclass
class BacktestTrade:
    """A completed (closed) stock trade -- an entry that has been exited.

    Immutable record of what happened: entry/exit prices, P&L, and why
    the position was closed (stop_loss, take_profit, strategy_exit, etc.).
    """

    symbol: str
    strategy: str
    shares: int
    entry_price: float
    entry_date: str
    exit_price: float
    exit_date: str
    hold_type: str            # String version of HoldType enum ("day" or "swing")
    pnl: float = 0.0         # Profit/loss in dollars
    pnl_pct: float = 0.0     # Profit/loss as a decimal (0.05 = 5%)
    exit_reason: str = ""     # Why we exited: "stop_loss", "take_profit", "strategy_exit", etc.


@dataclass
class OptionsBacktestPosition:
    """A simulated open options position (may be multi-leg, e.g., a spread).

    Options positions are more complex than stocks because:
    - They can have multiple "legs" (e.g., a credit spread has a short and long leg)
    - They have an expiration date (time-limited, unlike stocks)
    - Cost can be positive (debit: you pay) or negative (credit: you receive money)
    - Each leg has a type (call/put), strike, and side (buy/sell)

    ``field(default_factory=list)`` is required for mutable defaults in
    dataclasses.  If we wrote ``legs: list = []``, ALL instances would share
    the SAME list object (a common Python gotcha).  ``default_factory=list``
    creates a NEW empty list for each instance.
    """

    underlying: str                                    # Stock ticker (e.g., "AAPL")
    strategy_name: str                                 # e.g., "credit_put_spread"
    entry_date: str                                    # ISO date string
    expiration: str                                    # YYYY-MM-DD when the option expires
    strikes: list[float] = field(default_factory=list) # Strike prices for each leg
    legs: list[dict] = field(default_factory=list)     # [{type, strike, side, qty}] per leg
    entry_cost: float = 0.0    # Positive = debit paid; negative = credit received
    max_loss: float = 0.0      # Maximum possible loss (used for position sizing)
    max_profit: float = 0.0    # Maximum possible profit
    contracts: int = 1         # Number of contracts (each controls 100 shares)
    net_delta: float = 0.0     # Net directional exposure
    conviction: float = 0.0    # Strategy's confidence score (0.0 to 1.0)


@dataclass
class OptionsBacktestTrade:
    """A completed (closed) options trade."""

    underlying: str
    strategy: str
    entry_date: str
    exit_date: str
    expiration: str
    strikes: list[float] = field(default_factory=list)
    contracts: int = 1
    entry_cost: float = 0.0    # What was paid (debit) or received (credit) per contract
    exit_value: float = 0.0    # What was received on exit per contract
    pnl: float = 0.0          # Net profit/loss in dollars
    pnl_pct: float = 0.0      # Return percentage relative to capital at risk
    exit_reason: str = ""


@dataclass
class DailySnapshot:
    """End-of-day portfolio snapshot -- one row in the equity curve.

    The equity curve is the time series of these snapshots.  It's used to
    compute performance metrics like max drawdown and Sharpe ratio.
    """

    date: str
    equity: float          # Total portfolio value (cash + positions market value)
    cash: float            # Cash not deployed in positions
    open_positions: int    # Number of open stock positions
    realized_pnl: float   # Cumulative realized P&L from closed trades
    day_trades_used: int = 0   # Day trades used in the trailing 5-day window (PDT tracking)
    open_options: int = 0      # Number of open options positions


@dataclass
class BacktestConfig:
    """Backtesting parameters -- risk management, position sizing, and limits.

    All monetary values are in dollars; percentages are decimals (0.05 = 5%).
    These values typically come from settings.yaml via the runner.
    """

    starting_capital: float = 500.0       # Initial cash in the simulated account
    max_position_pct: float = 0.30        # Max 30% of equity in a single stock position
    max_risk_per_trade_pct: float = 0.02  # Risk at most 2% of equity per trade (distance to stop)
    max_open_positions: int = 4           # Diversification limit
    daily_loss_limit_pct: float = 0.05    # Circuit breaker: stop trading if down 5% today
    max_portfolio_heat_pct: float = 0.06  # Max 6% of equity at risk across all positions
    commission_per_trade: float = 0.0     # Alpaca is commission-free for stocks
    slippage_pct: float = 0.001           # 0.1% slippage simulation (realistic for liquid stocks)
    max_day_trades: int = 3               # PDT limit: 3 day trades per 5-business-day window
    day_trade_reserve: int = 1            # Reserve 1 slot for emergency exits
    min_conviction_for_day_trade: float = 0.80  # Only day-trade with high confidence signals
    trailing_stop_pct: float = 0.04       # Trailing stop distance: 4% below highest price
    use_market_regime: bool = True        # Enable market regime analysis (bullish/bearish gating)
    # Options-specific limits
    max_options_positions: int = 3        # Max concurrent options positions
    max_options_capital_pct: float = 0.40 # Max 40% of equity allocated to options
    max_single_options_risk_pct: float = 0.08  # Max 8% equity at risk per options trade
    options_profit_target_pct: float = 0.50  # Close options at 50% of max profit
    options_loss_limit_pct: float = 2.0   # Close options when loss = 2x entry cost


# ── Core engine ────────────────────────────────────────────


class BacktestEngine:
    """Event-driven backtester that walks through daily bars.

    The engine is "event-driven" in the sense that it processes one day at
    a time, reacting to price events (stop hits, target hits, new signals).

    Usage:
        engine = BacktestEngine(strategies, config)
        results = engine.run(bars_dict, start_date, end_date)
        results.print_summary()

    The engine is stateful: calling run() resets internal state, so you can
    reuse the same engine instance for multiple backtests.
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        config: BacktestConfig | None = None,
        options_strategies: list[BaseOptionsStrategy] | None = None,
    ):
        self.strategies = strategies
        # ``or []`` is a Python idiom: if options_strategies is None (falsy),
        # use an empty list instead.  This avoids None-checks throughout the code.
        self.options_strategies = options_strategies or []
        self.cfg = config or BacktestConfig()

        # Market regime analyzer: determines if the broad market (SPY/QQQ) is
        # bullish, bearish, or neutral.  Used to gate entry decisions.
        self._regime_analyzer = MarketRegimeAnalyzer() if self.cfg.use_market_regime else None
        self._market_context: MarketContext | None = None

        # ---- Portfolio state (reset before each backtest run) ----
        self._cash: float = 0.0
        self._positions: list[BacktestPosition] = []          # Open stock positions
        self._trades: list[BacktestTrade] = []                # Closed stock trades
        self._options_positions: list[OptionsBacktestPosition] = []  # Open options positions
        self._options_trades: list[OptionsBacktestTrade] = []  # Closed options trades
        self._snapshots: list[DailySnapshot] = []             # End-of-day equity snapshots
        self._day_trade_dates: list[str] = []                 # Dates of day trades (for PDT counting)
        self._starting_equity_today: float = 0.0              # Equity at start of current day

    # ── Public API ──────────────────────────────────────────

    def run(
        self,
        bars_dict: dict[str, pd.DataFrame],
        start_date: str | None = None,
        end_date: str | None = None,
        market_bars: dict[str, pd.DataFrame] | None = None,
    ) -> BacktestResults:
        """Run the backtest simulation over a date range.

        This is the main entry point.  It:
        1. Resets internal state (cash, positions, trades)
        2. Enriches price data with technical indicators (RSI, Bollinger, ATR)
        3. Pre-builds O(1) date lookup indices
        4. Iterates day-by-day, calling _process_day for each
        5. Force-closes remaining positions at the end
        6. Returns a BacktestResults object with metrics and trade history

        Args:
            bars_dict: Dict mapping symbol -> DataFrame of daily OHLCV bars.
                      Each DataFrame has a DatetimeIndex and columns:
                      open, high, low, close, volume.
            start_date: ISO date string to start from (inclusive).  If None,
                       uses the earliest date across all symbols.
            end_date: ISO date string to end at (inclusive).
            market_bars: Optional dict with "SPY" and "QQQ" DataFrames for
                        market regime analysis (bullish/bearish gating).

        Returns:
            BacktestResults with all trades, snapshots, and computed metrics.
        """
        self._reset()

        # ---- Step 1: Enrich bars with technical indicators ----
        # Strategies need indicators like RSI, Bollinger Bands, ATR, etc.
        # add_all() computes and appends these as new columns to each DataFrame.
        # We need at least 21 rows of data for the indicators' lookback windows.
        enriched: dict[str, pd.DataFrame] = {}
        skipped = []
        for sym, df in bars_dict.items():
            if df.empty or len(df) < 21:
                if not df.empty:
                    skipped.append(f"{sym}({len(df)})")
                continue
            df = df.copy()  # .copy() prevents modifying the caller's DataFrame
            try:
                add_all(df, intraday=False)  # Adds RSI, Bollinger, VWAP, ATR columns
            except Exception as e:
                log.warning("backtest_indicator_failed", symbol=sym, rows=len(df), error=str(e))
                continue
            enriched[sym] = df
        if skipped:
            log.info("backtest_skipped_insufficient_bars", symbols=skipped)

        if not enriched:
            log.warning("backtest_no_data")
            return BacktestResults([], [], [], [], self.cfg)

        # ---- Step 2: Pre-build O(1) date lookup indices ----
        # Without this optimization, looking up a price for a given (symbol, date)
        # would require scanning the DataFrame (O(n) per lookup, thousands of times).
        # Instead, we build a nested dict: _date_idx[symbol][date_str] -> row_index
        # This gives O(1) lookups via dict hashing.
        #
        # df.index.strftime("%Y-%m-%d") converts the DatetimeIndex to string dates.
        # We store these as a new column "_date_str" for convenience.
        for df in enriched.values():
            df["_date_str"] = df.index.strftime("%Y-%m-%d")

        # Build the lookup dict.  This is a dict comprehension inside a loop:
        # For each symbol, create a dict mapping date_str -> integer row index.
        # ``enumerate(iterable)`` yields (index, value) pairs.
        self._date_idx: dict[str, dict[str, int]] = {}
        for sym, df in enriched.items():
            self._date_idx[sym] = {
                d: i for i, d in enumerate(df["_date_str"].values)
            }

        # ---- Step 3: Build a unified, sorted list of all trading dates ----
        # set().union(*iterables) merges multiple iterables into one set (no duplicates).
        # The * operator unpacks the generator so union() receives each array as a
        # separate argument.  sorted() returns an ascending list.
        all_dates = sorted(
            set().union(*(df["_date_str"].values for df in enriched.values()))
        )
        # Filter to the requested date range
        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]

        if not all_dates:
            log.warning("backtest_no_dates_in_range")
            return BacktestResults([], [], [], [], self.cfg)

        has_options = len(self.options_strategies) > 0
        log.info(
            "backtest_start",
            symbols=list(enriched.keys()),
            dates=len(all_dates),
            start=all_dates[0],
            end=all_dates[-1],
            capital=self.cfg.starting_capital,
            options_strategies=len(self.options_strategies) if has_options else 0,
        )

        # Enrich market regime bars (SPY/QQQ) with the same date string column
        enriched_market: dict[str, pd.DataFrame] = {}
        if market_bars:
            for sym, df in market_bars.items():
                if not df.empty:
                    df = df.copy()
                    df["_date_str"] = df.index.strftime("%Y-%m-%d")
                    enriched_market[sym] = df

        # ---- Step 4: The main simulation loop -- one iteration per trading day ----
        for date_str in all_dates:
            self._process_day(date_str, enriched, enriched_market)

        # ---- Step 5: Force-close all remaining positions at the last day's price ----
        # A real trader would still hold these, but for performance measurement
        # we need to mark them to market and realize the P&L.
        self._close_all_positions(all_dates[-1], enriched, reason="backtest_end")
        self._close_all_options(all_dates[-1], enriched, reason="backtest_end")

        results = BacktestResults(
            trades=self._trades,
            options_trades=self._options_trades,
            snapshots=self._snapshots,
            positions=[],  # All positions have been closed
            config=self.cfg,
        )
        total = len(self._trades) + len(self._options_trades)
        log.info("backtest_complete", total_trades=total,
                 stock_trades=len(self._trades), options_trades=len(self._options_trades))
        return results

    # ── Internal ────────────────────────────────────────────

    def _reset(self) -> None:
        """Reset all portfolio state for a fresh backtest run."""
        self._cash = self.cfg.starting_capital
        self._positions = []
        self._trades = []
        self._options_positions = []
        self._options_trades = []
        self._snapshots = []
        self._day_trade_dates = []
        self._starting_equity_today = self.cfg.starting_capital
        self._equity_cache: dict[str, float] = {}  # Memoization cache: date_str -> equity value
        self._open_symbols: set[str] = set()  # Fast O(1) lookup for "do we hold this symbol?"

    def _equity(self, date_str: str, bars_dict: dict[str, pd.DataFrame]) -> float:
        """Current total portfolio equity = cash + stock market value + options MTM.

        This is the fundamental portfolio valuation function.  It sums:
        1. Cash on hand
        2. Market value of all open stock positions (shares * current price)
        3. Mark-to-market value of all open options positions

        Cached per date because it's called multiple times per day (for equity
        checks, position sizing, etc.) and the result doesn't change within
        a single day's processing.  The cache is invalidated at the start of
        each new day in _process_day().
        """
        # Check memoization cache first
        cached = self._equity_cache.get(date_str)
        if cached is not None:
            return cached

        mkt_value = 0.0
        # Sum market value of all open stock positions
        for pos in self._positions:
            price = self._get_close(pos.symbol, date_str, bars_dict)
            if price is not None:
                mkt_value += pos.shares * price

        # Sum mark-to-market value of all open options positions
        for opt_pos in self._options_positions:
            stock_price = self._get_close(opt_pos.underlying, date_str, bars_dict)
            if stock_price is not None:
                current_val = self._options_mark_to_market(opt_pos, stock_price, date_str)
                mkt_value += current_val

        eq = self._cash + mkt_value
        self._equity_cache[date_str] = eq  # Store in cache
        return eq

    def _options_mark_to_market(
        self, pos: OptionsBacktestPosition, stock_price: float, date_str: str
    ) -> float:
        """Estimate the current market value of an options position.

        SIMPLIFICATION: Rather than re-running full Black-Scholes pricing every
        day (which would be computationally expensive and require volatility
        re-estimation), this uses a simplified model:

            value = intrinsic_value + time_value_remaining

        Where:
        - intrinsic_value: What the option is worth if exercised right now
          (computed per-leg, accounting for buy/sell sides)
        - time_value_remaining: A linear decay of the estimated extrinsic
          (time) value at entry.  Approximates theta decay.

        In reality, theta decay is non-linear (accelerates near expiration),
        but linear decay is adequate for backtesting purposes.

        Returns:
            Estimated market value in dollars (total across all contracts).
        """
        # Parse expiration and current dates to compute days-to-expiration (DTE)
        try:
            exp_dt = datetime.strptime(pos.expiration, "%Y-%m-%d")
            cur_dt = datetime.strptime(date_str, "%Y-%m-%d")
            dte = max(0, (exp_dt - cur_dt).days)
        except ValueError:
            dte = 0

        # time_decay_factor: 1.0 at entry, decays linearly to 0.0 at expiration
        entry_dt = datetime.strptime(pos.entry_date, "%Y-%m-%d")
        original_dte = max(1, (exp_dt - entry_dt).days)  # Avoid division by zero
        time_decay_factor = dte / original_dte

        # Calculate intrinsic value of each leg of the position.
        # For multi-leg strategies (e.g., credit spread = one sold put + one bought put),
        # we sum the intrinsic values with appropriate signs.
        total_intrinsic = 0.0
        for leg in pos.legs:
            strike = leg["strike"]
            opt_type = leg["type"]    # "call" or "put"
            side = leg["side"]        # "buy" (long) or "sell" (short)

            # Intrinsic value as if at expiration
            intrinsic = price_option_at_expiration(opt_type, strike, stock_price)

            if side == "buy":
                total_intrinsic += intrinsic   # Long legs add value
            else:
                total_intrinsic -= intrinsic   # Short legs subtract value

        # Estimate remaining time value.
        # Heuristic: ~30% of the entry cost was extrinsic (time) value.
        # This fraction decays linearly to zero by expiration.
        extrinsic_at_entry = abs(pos.entry_cost) * 0.3
        time_value = extrinsic_at_entry * time_decay_factor

        # Total value per contract, then multiply by 100 (shares per contract)
        # and number of contracts.
        value_per_contract = total_intrinsic + time_value
        return value_per_contract * 100 * pos.contracts

    def _get_close(
        self, symbol: str, date_str: str, bars_dict: dict[str, pd.DataFrame]
    ) -> float | None:
        """Get the closing price for a symbol on a given date.

        Uses the pre-built _date_idx for O(1) lookup: first look up the row
        index in the nested dict, then use iloc[] to access the DataFrame row.
        Returns None if the symbol doesn't have data for this date (e.g.,
        the stock wasn't trading yet, or it's a holiday for that exchange).
        """
        idx = self._date_idx.get(symbol, {}).get(date_str)
        if idx is None:
            return None
        return float(bars_dict[symbol].iloc[idx]["close"])

    def _get_bar(
        self, symbol: str, date_str: str, bars_dict: dict[str, pd.DataFrame]
    ) -> dict | None:
        """Get the full OHLCV bar for a symbol on a given date (O(1) lookup).

        Returns a plain dict with keys: open, high, low, close, volume.
        The High and Low are critical for intraday stop/target simulation --
        even though we only have daily bars, the High tells us the intraday
        peak and the Low tells us the intraday trough.
        """
        idx = self._date_idx.get(symbol, {}).get(date_str)
        if idx is None:
            return None
        row = bars_dict[symbol].iloc[idx]
        return {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }

    def _bars_up_to(
        self, symbol: str, date_str: str, bars_dict: dict[str, pd.DataFrame]
    ) -> pd.DataFrame | None:
        """Get all bars up to and including this date for strategy evaluation.

        Strategies need historical context (not just today's bar) to compute
        indicators and make decisions.  For example, a 20-day moving average
        needs the last 20 bars.

        Uses the O(1) date index to find the row position, then slices the
        DataFrame with iloc[:idx+1] (Python slice: from start up to but not
        including idx+1, which gives rows 0 through idx inclusive).
        """
        idx = self._date_idx.get(symbol, {}).get(date_str)
        if idx is None:
            # Date not in this symbol's data -- fall back to filtering by date string.
            # This handles cases where the symbol started trading after the backtest began.
            df = bars_dict.get(symbol)
            if df is None or df.empty:
                return None
            mask = df["_date_str"].values <= date_str
            subset = df.loc[mask]
            return subset if not subset.empty else None
        df = bars_dict[symbol]
        subset = df.iloc[: idx + 1]
        return subset if not subset.empty else None

    def _day_trades_in_window(self, date_str: str) -> int:
        """Count day trades in the last 5 business days (7 calendar days).

        The PDT (Pattern Day Trader) rule uses a rolling 5-business-day window.
        We approximate this with 7 calendar days (covers weekends).

        A "day trade" is opening and closing the same position on the same day.
        """
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        window_start = dt - timedelta(days=7)
        start_str = window_start.strftime("%Y-%m-%d")
        # sum(1 for ...) counts items matching the condition.
        # This is a generator expression -- it doesn't build a list in memory.
        return sum(1 for d in self._day_trade_dates if d >= start_str)

    def _can_day_trade(self, date_str: str) -> bool:
        """Check if we have PDT budget remaining for a day trade.

        Subtracts day_trade_reserve (default 1) from the max to keep a slot
        open for emergency exits or high-conviction opportunities.
        """
        used = self._day_trades_in_window(date_str)
        return used < (self.cfg.max_day_trades - self.cfg.day_trade_reserve)

    def _process_day(
        self, date_str: str, bars_dict: dict[str, pd.DataFrame],
        market_bars: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """Process a single trading day -- the heart of the simulation.

        PROCESSING ORDER (critical for correctness):
        0. Update market regime (weekly analysis of SPY/QQQ)
        1. Check exits on existing positions (stops, targets, trailing stops)
           -- This must happen BEFORE entries so freed capital is available
        1b. Check options exits (expiration, profit target, loss limit)
        2. Check daily loss limit (circuit breaker)
        3. Evaluate new entry signals (stock strategies)
        3b. Evaluate options entry signals
        4. Close end-of-day day-trade positions
        5. Take end-of-day snapshot (equity curve data point)
        """
        # Invalidate equity cache -- positions will change during today's processing
        self._equity_cache.clear()
        self._starting_equity_today = self._equity(date_str, bars_dict)

        # 0. Update market regime analysis (determines if we allow new longs)
        if self._regime_analyzer and market_bars:
            self._update_regime(date_str, market_bars)

        # 1. Check stop-losses, take-profits, and trailing stops on stock positions.
        #    Exits are processed BEFORE entries to free up capital and position slots.
        self._check_exits(date_str, bars_dict)

        # 1b. Check options positions for expiration or profit/loss triggers
        if self._options_positions:
            self._check_options_exits(date_str, bars_dict)

        # 2. Daily loss limit circuit breaker: if we've lost more than the threshold
        #    today (e.g., 5% of starting equity), stop all trading for the day.
        current_eq = self._equity(date_str, bars_dict)
        if self._starting_equity_today > 0:
            loss_pct = (self._starting_equity_today - current_eq) / self._starting_equity_today
            if loss_pct > self.cfg.daily_loss_limit_pct:
                self._take_snapshot(date_str, bars_dict)
                return  # Skip entries and day-trade closes for today

        # 3. Evaluate strategies for new stock entries.
        #    Skip if market regime analysis says "no new longs" (strong bear market).
        if self._market_context and not self._market_context.allow_new_longs:
            pass  # Bear regime: don't open new long positions
        else:
            self._evaluate_entries(date_str, bars_dict)

            # 3b. Evaluate options strategies for new entries
            if self.options_strategies:
                self._evaluate_options_entries(date_str, bars_dict)

        # 4. End-of-day: force-close positions that were opened today as day trades.
        #    Day trades MUST be closed by market close to comply with PDT rules.
        self._close_day_trades(date_str, bars_dict)

        # 5. Record end-of-day snapshot for the equity curve
        self._take_snapshot(date_str, bars_dict)

    def _update_regime(self, date_str: str, market_bars: dict[str, pd.DataFrame]) -> None:
        """Update market regime from SPY/QQQ bars up to this date.

        The market regime (bullish/bearish/neutral) determines whether the
        engine allows new long entries.  Requires at least 50 bars of history
        for the regime analyzer's moving averages and trend detection.
        """
        spy = self._bars_up_to("SPY", date_str, market_bars)
        qqq = self._bars_up_to("QQQ", date_str, market_bars)
        if spy is not None and len(spy) >= 50 and qqq is not None and len(qqq) >= 50:
            try:
                self._market_context = self._regime_analyzer.analyze(spy, qqq)
            except Exception:
                pass  # Silently skip if regime analysis fails (non-critical)

    # ── Stock position management ──────────────────────────

    def _check_exits(self, date_str: str, bars_dict: dict[str, pd.DataFrame]) -> None:
        """Check stop loss, take profit, and trailing stop on all open stock positions.

        Uses the day's High and Low to simulate intraday stop/target triggers:
        - If the Low <= stop_loss, we assume the stop was hit (exit at stop price)
        - If the High >= take_profit, we assume the target was hit (exit at target)
        - Otherwise, update the trailing stop (ratchet stop_loss upward if the
          stock reached a new high)

        Also checks for strategy-generated exit signals (e.g., a reversal
        indicator from the strategy that originally generated the entry).

        IMPORTANT: Closed position indices are collected and removed in reverse
        order to avoid index-shifting bugs (removing index 2 would shift index 3
        down to 2, invalidating later removals).
        """
        closed_indices = []

        for i, pos in enumerate(self._positions):
            bar = self._get_bar(pos.symbol, date_str, bars_dict)
            if bar is None:
                continue  # No data for this symbol today (e.g., trading halt)

            exit_price = None
            exit_reason = ""

            # ---- Stop loss check ----
            # If the day's low touched or breached the stop, assume we got filled
            # at the stop price.  In real markets, stops are not guaranteed to fill
            # at the stop price (gaps can occur), but this is a standard simplification.
            if bar["low"] <= pos.stop_loss:
                exit_price = pos.stop_loss
                exit_reason = "stop_loss"

            # ---- Take profit check ----
            # If the day's high reached the target, assume we got filled at the target.
            elif bar["high"] >= pos.take_profit:
                exit_price = pos.take_profit
                exit_reason = "take_profit"

            else:
                # ---- Trailing stop update ----
                # If the stock made a new high, update the tracking variable.
                # Then compute the trailing stop level and ratchet the stop_loss UP
                # (it never moves down -- "trailing" means it only follows price upward).
                if bar["high"] > pos.highest_price:
                    pos.highest_price = bar["high"]
                trailing_stop = pos.highest_price * (1 - self.cfg.trailing_stop_pct)
                if trailing_stop > pos.stop_loss:
                    pos.stop_loss = trailing_stop  # Tighten the stop (ratchet up)

                # ---- Strategy-generated exit signal ----
                # Ask the originating strategy if its own indicators say "exit."
                # This handles cases like "RSI crossed above 70, sell" that aren't
                # captured by simple stop/target levels.
                bars_to_date = self._bars_up_to(pos.symbol, date_str, bars_dict)
                if bars_to_date is not None:
                    for strat in self.strategies:
                        # Match the strategy that opened this position by name.
                        # The lowercase+replace converts "MomentumStrategy" -> "momentum"
                        # and checks if it appears in the position's strategy_name.
                        if strat.__class__.__name__.lower().replace("strategy", "") in pos.strategy_name:
                            try:
                                if strat.should_exit(pos.symbol, bars_to_date, pos.entry_price):
                                    exit_price = bar["close"]
                                    exit_reason = "strategy_exit"
                            except Exception:
                                pass
                            break  # Only check the matching strategy, not all of them

            if exit_price is not None:
                self._close_position(i, exit_price, date_str, exit_reason)
                closed_indices.append(i)

                # Track day trades: if we opened AND closed on the same day
                if pos.entry_date == date_str:
                    self._day_trade_dates.append(date_str)

        # Remove closed positions in REVERSE order to avoid index-shifting issues.
        # Example: if we close positions at indices [1, 3], removing index 1 first
        # would shift index 3 to 2, causing us to remove the wrong position.
        for i in sorted(closed_indices, reverse=True):
            self._positions.pop(i)

    def _evaluate_entries(self, date_str: str, bars_dict: dict[str, pd.DataFrame]) -> None:
        """Run all strategies on all symbols and enter new positions for qualifying signals.

        PROCESS:
        1. Pre-check: skip if at max open positions or portfolio heat limit
        2. Collect signals from all (strategy, symbol) combinations
        3. Separate into swing vs. day-trade signals
        4. Rank by conviction (highest first)
        5. Filter day-trade signals by PDT budget and minimum conviction
        6. Execute entries in ranked order until limits are hit

        PORTFOLIO HEAT:
        "Heat" = total dollars at risk = sum of (entry_price - stop_loss) * shares
        across all open positions.  If heat exceeds max_portfolio_heat_pct of equity,
        no new entries are allowed.  This prevents overexposure even when individual
        positions are small.
        """
        # Quick check: are we already at the position limit?
        if len(self._positions) >= self.cfg.max_open_positions:
            return

        # Calculate current portfolio heat (total capital at risk)
        current_eq = self._equity(date_str, bars_dict)
        total_heat = sum(
            abs(p.entry_price - p.stop_loss) * p.shares
            for p in self._positions
        )
        if current_eq > 0 and (total_heat / current_eq) > self.cfg.max_portfolio_heat_pct:
            return  # Too much risk already deployed

        # Collect signals from all strategies on all symbols
        signals: list[Signal] = []

        for symbol, df in bars_dict.items():
            bars_to_date = self._bars_up_to(symbol, date_str, bars_dict)
            if bars_to_date is None or len(bars_to_date) < 20:
                continue  # Need enough history for indicator calculations

            # Skip if we already have an open position in this symbol
            # (``any()`` is a Python builtin: returns True if any element is True)
            if any(p.symbol == symbol for p in self._positions):
                continue

            # Ask each enabled strategy to evaluate this symbol
            for strategy in self.strategies:
                if not strategy.enabled:
                    continue
                try:
                    sig = strategy.evaluate(symbol, bars_to_date)
                    if sig is not None:  # Strategy sees an opportunity
                        signals.append(sig)
                except Exception:
                    continue  # Skip failed evaluations (bad data, indicator errors)

        if not signals:
            return

        # ---- Prioritize swing trades over day trades ----
        # Swing trades (held overnight/multi-day) don't count toward the PDT limit,
        # making them "free" for small accounts.  Day trades are reserved for
        # high-conviction signals only.
        swing_signals = [s for s in signals if s.hold_type == HoldType.SWING]
        day_signals = [s for s in signals if s.hold_type != HoldType.SWING]

        # Sort by conviction descending (best opportunities first)
        # ``key=lambda s: s.conviction`` tells sort() to use the conviction
        # attribute as the sort key.  ``reverse=True`` makes it descending.
        swing_signals.sort(key=lambda s: s.conviction, reverse=True)
        day_signals.sort(key=lambda s: s.conviction, reverse=True)

        # Filter day trade signals: only allow if PDT budget remains AND
        # conviction exceeds the minimum threshold (default 0.80)
        qualifying_day = []
        if self._can_day_trade(date_str):
            qualifying_day = [
                s for s in day_signals
                if s.conviction >= self.cfg.min_conviction_for_day_trade
            ]

        # Merged, prioritized list: all swing signals first, then qualifying day trades
        ranked = swing_signals + qualifying_day

        for sig in ranked:
            if len(self._positions) >= self.cfg.max_open_positions:
                break
            if self._cash <= 0:
                break

            # ---- Position sizing ----
            shares = self._size_position(sig, current_eq)
            if shares <= 0:
                continue

            # ---- Apply market regime modifiers ----
            # In uncertain/bearish markets, reduce conviction and position size
            if self._market_context:
                sig.conviction = min(1.0, sig.conviction * self._market_context.conviction_modifier)
                shares = max(1, int(shares * self._market_context.position_size_modifier))
                if sig.conviction < 0.35:
                    continue  # Regime filter killed this signal

            # ---- Apply slippage to entry price ----
            # Slippage simulates the real-world cost of market impact: you
            # typically get filled slightly ABOVE the expected price when buying.
            slippage = sig.entry_price * self.cfg.slippage_pct
            fill_price = sig.entry_price + slippage

            # Check if we can afford this position
            cost = shares * fill_price
            if cost > self._cash:
                # Reduce share count to fit available cash
                # math.floor rounds DOWN (can't buy fractional shares)
                shares = math.floor(self._cash / fill_price)
                if shares <= 0:
                    continue
                cost = shares * fill_price

            # ---- Execute the entry ----
            self._cash -= cost
            self._positions.append(
                BacktestPosition(
                    symbol=sig.symbol,
                    shares=shares,
                    entry_price=fill_price,
                    entry_date=date_str,
                    stop_loss=sig.stop_loss_price,
                    take_profit=sig.take_profit_price,
                    hold_type=sig.hold_type,
                    strategy_name=sig.strategy_name,
                )
            )
            log.debug(
                "backtest_entry",
                symbol=sig.symbol,
                strategy=sig.strategy_name,
                shares=shares,
                price=fill_price,
                date=date_str,
            )

    def _size_position(self, signal: Signal, equity: float) -> int:
        """Fixed-fractional position sizing (mirrors the live PositionSizer).

        METHODOLOGY:
        1. Compute the dollar amount we're willing to risk (equity * max_risk_per_trade_pct)
        2. Compute the risk per share (distance from entry to stop-loss)
        3. shares = risk_amount / risk_per_share (rounded down)
        4. Cap by max_position_pct (no single position > 30% of equity)
        5. Cap by available cash
        6. Minimum 1 share if we can afford it

        This ensures that if the stop-loss is hit, the loss is approximately
        max_risk_per_trade_pct of equity regardless of stock price.

        Example:
            Equity = $500, risk_pct = 2%, entry = $50, stop = $48
            risk_amount = $500 * 0.02 = $10
            risk_per_share = $50 - $48 = $2
            shares = $10 / $2 = 5 shares
            If stopped out: loss = 5 * $2 = $10 = 2% of equity
        """
        risk_amount = equity * self.cfg.max_risk_per_trade_pct
        risk_per_share = abs(signal.entry_price - signal.stop_loss_price)
        if risk_per_share == 0:
            return 0  # Can't size a position with zero risk (stop = entry)

        shares = math.floor(risk_amount / risk_per_share)

        # Cap by maximum position size (prevent concentration risk)
        max_value = equity * self.cfg.max_position_pct
        if shares * signal.entry_price > max_value:
            shares = math.floor(max_value / signal.entry_price)

        # Cap by available cash
        if signal.entry_price > 0:
            shares = min(shares, math.floor(self._cash / signal.entry_price))

        # Ensure non-negative, but allow at least 1 share if we can afford it
        shares = max(0, shares)
        if shares == 0 and self._cash >= signal.entry_price > 0:
            shares = 1  # Minimum position: 1 share

        return shares

    def _close_position(
        self, idx: int, exit_price: float, date_str: str, reason: str
    ) -> None:
        """Record a closed stock trade and return proceeds to cash.

        Applies slippage to the exit (you get slightly LESS than expected
        when selling, just as you pay slightly MORE when buying).
        """
        pos = self._positions[idx]

        # Apply slippage: selling gets a slightly worse price
        slippage = exit_price * self.cfg.slippage_pct
        fill_price = exit_price - slippage

        # Return proceeds to cash
        proceeds = pos.shares * fill_price
        self._cash += proceeds

        # Calculate P&L
        pnl = (fill_price - pos.entry_price) * pos.shares
        pnl_pct = (fill_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0

        trade = BacktestTrade(
            symbol=pos.symbol,
            strategy=pos.strategy_name,
            shares=pos.shares,
            entry_price=pos.entry_price,
            entry_date=pos.entry_date,
            exit_price=fill_price,
            exit_date=date_str,
            hold_type=pos.hold_type.value,  # .value converts enum to its string value
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 4),
            exit_reason=reason,
        )
        self._trades.append(trade)

        log.debug(
            "backtest_exit",
            symbol=pos.symbol,
            pnl=trade.pnl,
            reason=reason,
            date=date_str,
        )

    def _close_day_trades(self, date_str: str, bars_dict: dict[str, pd.DataFrame]) -> None:
        """Close all positions opened today that are marked as day trades.

        Day trades MUST be closed by market close (end of day).  In the
        backtest, we close them at the closing price of the same day.
        """
        closed_indices = []
        for i, pos in enumerate(self._positions):
            # HoldType.DAY + entry today = must close today
            if pos.hold_type == HoldType.DAY and pos.entry_date == date_str:
                close_price = self._get_close(pos.symbol, date_str, bars_dict)
                if close_price is not None:
                    self._close_position(i, close_price, date_str, "eod_day_trade_close")
                    closed_indices.append(i)
                    self._day_trade_dates.append(date_str)  # Count toward PDT

        # Remove in reverse order (same index-safety pattern as _check_exits)
        for i in sorted(closed_indices, reverse=True):
            self._positions.pop(i)

    def _close_all_positions(
        self, date_str: str, bars_dict: dict[str, pd.DataFrame], reason: str = "force_close"
    ) -> None:
        """Close all remaining stock positions (used at backtest end).

        Iterates in reverse order so that pop(i) doesn't shift indices of
        positions we haven't processed yet.  ``range(len-1, -1, -1)`` is
        Python for "count from len-1 down to 0 inclusive" (the three args
        are start, stop-exclusive, step).
        """
        for i in range(len(self._positions) - 1, -1, -1):
            pos = self._positions[i]
            close_price = self._get_close(pos.symbol, date_str, bars_dict)
            if close_price is not None:
                self._close_position(i, close_price, date_str, reason)
            self._positions.pop(i)

    # ── Options position management ────────────────────────

    def _evaluate_options_entries(
        self, date_str: str, bars_dict: dict[str, pd.DataFrame]
    ) -> None:
        """Run options strategies on all symbols using synthetic Black-Scholes pricing.

        PROCESS:
        1. Pre-check: position limit and capital allocation limit
        2. For each symbol: compute historical volatility, generate synthetic chain
        3. Run each options strategy against the synthetic chain
        4. Rank signals by conviction
        5. Size and execute entries with risk checks

        SYNTHETIC CHAIN:
        Since we're backtesting, there are no real options market prices.
        We generate a synthetic chain using Black-Scholes (see options_pricing.py)
        from the stock's historical volatility.  The chain format matches the
        live API so strategies don't need separate backtest code paths.
        """
        if len(self._options_positions) >= self.cfg.max_options_positions:
            return

        current_eq = self._equity(date_str, bars_dict)

        # Check total options capital allocation (prevent overexposure to options)
        options_exposure = sum(abs(p.entry_cost) * 100 * p.contracts for p in self._options_positions)
        max_options_capital = current_eq * self.cfg.max_options_capital_pct
        if options_exposure >= max_options_capital:
            return

        current_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=ET)

        all_signals: list[OptionsSignal] = []

        for symbol in bars_dict:
            bars_to_date = self._bars_up_to(symbol, date_str, bars_dict)
            if bars_to_date is None or len(bars_to_date) < 30:
                continue  # Options strategies need more history than stock strategies

            # Skip if already have an options position on this underlying
            if any(p.underlying == symbol for p in self._options_positions):
                continue

            stock_price = float(bars_to_date["close"].iloc[-1])

            # Compute historical volatility for Black-Scholes pricing
            vol = historical_volatility(bars_to_date["close"])
            if vol <= 0:
                continue  # Can't price options without volatility

            # Generate synthetic options chain (calls and puts at various strikes/expirations)
            chain_data, snapshots = generate_synthetic_chain(
                underlying=symbol,
                stock_price=stock_price,
                current_date=current_date,
                volatility=vol,
            )

            if not chain_data:
                continue

            # Pre-compute days-to-expiration (_dte) on each contract so strategies
            # don't call datetime.now() (which would return the REAL current date,
            # not the simulated backtest date -- a subtle but critical bug to avoid).
            for contract in chain_data:
                exp_str = contract.get("expiration_date", "")
                if exp_str:
                    try:
                        exp_dt = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                        contract["_dte"] = max(0, (exp_dt - current_date.replace(tzinfo=timezone.utc)).days)
                    except (ValueError, TypeError):
                        contract["_dte"] = 0

            # Run each options strategy against the synthetic chain
            for strategy in self.options_strategies:
                if not strategy.enabled:
                    continue
                try:
                    sig = strategy.evaluate(symbol, bars_to_date, chain_data, snapshots)
                    if sig is not None:
                        all_signals.append(sig)
                except Exception:
                    continue

        if not all_signals:
            return

        # Rank by conviction (highest first), then process top signals
        all_signals.sort(key=lambda s: s.conviction, reverse=True)

        for sig in all_signals:
            if len(self._options_positions) >= self.cfg.max_options_positions:
                break

            # ---- Per-trade risk check ----
            max_risk_per_trade = current_eq * self.cfg.max_single_options_risk_pct
            if sig.max_loss > max_risk_per_trade:
                continue  # This trade risks too much for our account size

            # ---- Determine entry cost and cash requirement ----
            # Options strategies fall into two categories:
            # - Debit strategies (long call, debit spread): you PAY money upfront
            # - Credit strategies (credit spread, cash-secured put): you RECEIVE money
            #   upfront but must reserve collateral equal to max possible loss
            if sig.max_cost > 0:
                # Debit strategy: pay the premium
                entry_cost = sig.max_cost       # Per-contract cost (in dollars)
                cash_needed = entry_cost * 100   # Each contract = 100 shares
            elif sig.min_credit > 0:
                # Credit strategy: receive premium, reserve collateral
                entry_cost = -sig.min_credit    # Negative = credit received
                cash_needed = sig.max_loss       # Reserve max loss as collateral
            else:
                continue  # Skip signals with no clear cost structure

            # ---- Position sizing: how many contracts? ----
            max_risk_budget = current_eq * self.cfg.max_risk_per_trade_pct
            if sig.max_loss > 0:
                num_contracts = max(1, int(max_risk_budget / sig.max_loss))
            else:
                num_contracts = 1

            num_contracts = min(num_contracts, 10)  # Cap at 10 for diversification

            # Cap by available cash
            total_cash_needed = cash_needed * num_contracts
            if total_cash_needed > self._cash:
                num_contracts = max(1, int(self._cash / cash_needed))
                total_cash_needed = cash_needed * num_contracts

            if total_cash_needed > self._cash:
                continue  # Can't afford even 1 contract

            # Cap by total options capital allocation
            new_exposure = options_exposure + total_cash_needed
            if new_exposure > max_options_capital:
                num_contracts = max(1, int((max_options_capital - options_exposure) / cash_needed))
                total_cash_needed = cash_needed * num_contracts
                new_exposure = options_exposure + total_cash_needed
            if new_exposure > max_options_capital or num_contracts < 1:
                continue

            # ---- Parse leg details for position tracking ----
            # Options positions can have multiple legs (e.g., a spread has 2 legs).
            # We extract the type (call/put), strike, and side (buy/sell) for each.
            position_legs = []
            for leg in sig.legs:
                leg_info = {
                    "symbol": leg.get("symbol", ""),
                    "side": leg.get("side", "buy"),
                    "qty": leg.get("qty", 1),
                    "type": "call",   # Default; overridden below
                    "strike": 0.0,
                }
                # Try to determine call/put from the OCC symbol.
                # In OCC format, "C" or "P" appears after position 6 (the date portion).
                occ = leg.get("symbol", "")
                if "C" in occ[6:] if len(occ) > 6 else False:
                    leg_info["type"] = "call"
                elif "P" in occ[6:] if len(occ) > 6 else False:
                    leg_info["type"] = "put"
                position_legs.append(leg_info)

            # Assign strike prices from the signal to each leg
            for i, strike in enumerate(sig.strikes):
                if i < len(position_legs):
                    position_legs[i]["strike"] = strike
            # Single-leg case
            if len(sig.strikes) == 1 and len(position_legs) == 1:
                position_legs[0]["strike"] = sig.strikes[0]
            # Spread case (two legs, two strikes)
            elif len(sig.strikes) >= 2 and len(position_legs) >= 2:
                position_legs[0]["strike"] = sig.strikes[0]
                position_legs[1]["strike"] = sig.strikes[1]

            # Determine call/put from the strategy name (more reliable than OCC parsing)
            strategy_name = sig.strategy_name
            if "call" in strategy_name:
                for leg in position_legs:
                    leg["type"] = "call"
            elif "put" in strategy_name:
                for leg in position_legs:
                    leg["type"] = "put"
            # Special case: straddle has one call leg and one put leg
            if "straddle" in strategy_name and len(position_legs) >= 2:
                position_legs[0]["type"] = "call"
                position_legs[1]["type"] = "put"

            # ---- Execute: deduct cash ----
            if entry_cost > 0:
                # Debit strategy: pay the premium (cost * 100 shares/contract * num contracts)
                self._cash -= entry_cost * 100 * num_contracts
            else:
                # Credit strategy: receive premium, then reserve collateral
                self._cash += abs(entry_cost) * 100 * num_contracts  # Credit received
                self._cash -= sig.max_loss * num_contracts            # Collateral reserved

            self._options_positions.append(
                OptionsBacktestPosition(
                    underlying=sig.underlying,
                    strategy_name=sig.strategy_name,
                    entry_date=date_str,
                    expiration=sig.expiration,
                    strikes=list(sig.strikes),
                    legs=position_legs,
                    entry_cost=entry_cost,
                    max_loss=sig.max_loss,
                    max_profit=sig.max_profit,
                    net_delta=sig.net_delta,
                    conviction=sig.conviction,
                    contracts=num_contracts,
                )
            )
            options_exposure += total_cash_needed

            log.debug(
                "backtest_options_entry",
                underlying=sig.underlying,
                strategy=sig.strategy_name,
                strikes=sig.strikes,
                entry_cost=entry_cost,
                max_profit=sig.max_profit,
                max_loss=sig.max_loss,
                expiration=sig.expiration,
                date=date_str,
            )

    def _check_options_exits(self, date_str: str, bars_dict: dict[str, pd.DataFrame]) -> None:
        """Check expiration, profit targets, and loss limits on open options positions.

        EXIT TRIGGERS (checked in priority order):
        1. Close 1 day before expiration: to avoid real-world assignment/exercise
           risk.  In live trading, holding through expiration can result in
           unexpected stock assignment.
        2. Profit target: close when unrealized P&L reaches a percentage of max
           profit (default 50%).  Taking partial profits is standard practice.
        3. Loss limit: close when loss exceeds a threshold to cap downside.
           For debit strategies: loss > 2x entry cost.
           For credit strategies: loss > 80% of max loss.
        """
        closed_indices = []

        for i, pos in enumerate(self._options_positions):
            stock_price = self._get_close(pos.underlying, date_str, bars_dict)
            if stock_price is None:
                continue

            exit_value = 0.0
            exit_reason = ""

            # Compute the "close by" date: 1 day before expiration
            try:
                exp_dt = datetime.strptime(pos.expiration, "%Y-%m-%d")
                close_by = (exp_dt - timedelta(days=1)).strftime("%Y-%m-%d")
            except ValueError:
                close_by = pos.expiration

            expired = date_str >= pos.expiration
            close_before_exp = date_str >= close_by and not expired

            if expired or close_before_exp:
                # ---- Close near/at expiration ----
                # Use mark-to-market (not just intrinsic) to simulate selling
                # before expiration, which captures remaining time value.
                current_mtm = self._options_mark_to_market(pos, stock_price, date_str)
                exit_value = current_mtm / (100 * pos.contracts) if pos.contracts > 0 else 0.0
                exit_reason = "close_before_expiration" if close_before_exp else "expiration"
            else:
                # ---- Check profit target and loss limit ----
                current_mtm = self._options_mark_to_market(pos, stock_price, date_str)
                entry_total = pos.entry_cost * 100 * pos.contracts  # Total capital deployed

                if pos.entry_cost > 0:
                    # DEBIT strategy P&L: current value minus what we paid
                    current_pnl = current_mtm - entry_total
                    # Profit target: close if we've captured enough of the max profit
                    if pos.max_profit > 0 and current_pnl >= pos.max_profit * self.cfg.options_profit_target_pct:
                        exit_value = current_mtm / (100 * pos.contracts)
                        exit_reason = "profit_target"
                    # Loss limit: close if loss exceeds threshold
                    elif current_pnl <= -entry_total * self.cfg.options_loss_limit_pct:
                        exit_value = current_mtm / (100 * pos.contracts)
                        exit_reason = "loss_limit"
                else:
                    # CREDIT strategy P&L: credit received minus cost to close
                    credit = abs(entry_total)
                    cost_to_close = max(0, -current_mtm)  # Cost to buy back the short position
                    current_pnl = credit - cost_to_close
                    # Profit target: keep a percentage of the credit
                    if current_pnl >= credit * self.cfg.options_profit_target_pct:
                        exit_value = -cost_to_close / (100 * pos.contracts)
                        exit_reason = "profit_target"
                    # Loss limit: close if loss approaches max_loss
                    elif current_pnl <= -pos.max_loss * 0.8:
                        exit_value = -cost_to_close / (100 * pos.contracts)
                        exit_reason = "loss_limit"

            if exit_reason:
                self._close_options_position(i, exit_value, date_str, exit_reason)
                closed_indices.append(i)

        # Remove in reverse order (same pattern as stock exits)
        for i in sorted(closed_indices, reverse=True):
            self._options_positions.pop(i)

    def _options_expiration_value(self, pos: OptionsBacktestPosition, stock_price: float) -> float:
        """Calculate the settlement value of an options position at expiration.

        At expiration, only intrinsic value remains (time value = 0).
        Each leg contributes its intrinsic value with appropriate sign
        (long legs add, short legs subtract).
        """
        total = 0.0
        for leg in pos.legs:
            strike = leg["strike"]
            opt_type = leg["type"]
            side = leg["side"]

            # Intrinsic value at expiration: max(0, S-K) for calls, max(0, K-S) for puts
            intrinsic = price_option_at_expiration(opt_type, strike, stock_price)

            if side == "buy":
                total += intrinsic
            else:
                total -= intrinsic

        return total  # Per-contract value (multiply by 100 * contracts for total)

    def _close_options_position(
        self, idx: int, exit_value: float, date_str: str, reason: str
    ) -> None:
        """Record a closed options trade and return cash to the portfolio.

        Cash flow differs between debit and credit strategies:
        - Debit: We paid at entry, receive exit proceeds.
          P&L = exit_value * 100 * contracts - entry_cost * 100 * contracts
        - Credit: We received credit at entry (but reserved collateral).
          P&L = credit_received - cost_to_close
          Cash returned = collateral released - cost to close
        """
        pos = self._options_positions[idx]

        entry_total = pos.entry_cost * 100 * pos.contracts

        if pos.entry_cost > 0:
            # Debit strategy: we paid entry_cost, now receive exit_value
            exit_total = exit_value * 100 * pos.contracts
            pnl = exit_total - entry_total
            self._cash += exit_total  # Receive exit proceeds
        else:
            # Credit strategy: we received credit, now pay to close
            credit_received = abs(entry_total)
            close_cost = max(0, -exit_value * 100 * pos.contracts)
            pnl = credit_received - close_cost
            self._cash += pos.max_loss   # Release the reserved collateral
            self._cash -= close_cost     # Pay the cost to close the short position

        # pnl_pct relative to capital at risk (avoid division by zero)
        pnl_pct = pnl / max(abs(entry_total), 0.01)

        trade = OptionsBacktestTrade(
            underlying=pos.underlying,
            strategy=pos.strategy_name,
            entry_date=pos.entry_date,
            exit_date=date_str,
            expiration=pos.expiration,
            strikes=pos.strikes,
            contracts=pos.contracts,
            entry_cost=pos.entry_cost,
            exit_value=exit_value,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 4),
            exit_reason=reason,
        )
        self._options_trades.append(trade)

        log.debug(
            "backtest_options_exit",
            underlying=pos.underlying,
            strategy=pos.strategy_name,
            pnl=trade.pnl,
            reason=reason,
            date=date_str,
        )

    def _close_all_options(
        self, date_str: str, bars_dict: dict[str, pd.DataFrame], reason: str = "force_close"
    ) -> None:
        """Close all remaining options positions (used at backtest end)."""
        for i in range(len(self._options_positions) - 1, -1, -1):
            pos = self._options_positions[i]
            stock_price = self._get_close(pos.underlying, date_str, bars_dict)
            if stock_price is not None:
                exit_value = self._options_expiration_value(pos, stock_price)
                self._close_options_position(i, exit_value, date_str, reason)
            self._options_positions.pop(i)

    # ── Snapshots ──────────────────────────────────────────

    def _take_snapshot(self, date_str: str, bars_dict: dict[str, pd.DataFrame]) -> None:
        """Record an end-of-day portfolio snapshot for the equity curve.

        These snapshots are the raw data for computing metrics like max drawdown,
        Sharpe ratio, and the equity curve plot.
        """
        eq = self._equity(date_str, bars_dict)
        # sum(t.pnl for t in ...) is a generator expression: sums the pnl attribute
        # across all completed trades without creating an intermediate list.
        realized = sum(t.pnl for t in self._trades) + sum(t.pnl for t in self._options_trades)
        self._snapshots.append(
            DailySnapshot(
                date=date_str,
                equity=round(eq, 2),
                cash=round(self._cash, 2),
                open_positions=len(self._positions),
                realized_pnl=round(realized, 2),
                day_trades_used=self._day_trades_in_window(date_str),
                open_options=len(self._options_positions),
            )
        )


# ── Results ────────────────────────────────────────────────


class BacktestResults:
    """Holds backtest output and computes performance metrics.

    After the engine finishes, this class provides:
    - Aggregate metrics (win rate, Sharpe ratio, max drawdown, etc.)
    - Per-strategy breakdowns
    - Equity curve as a DataFrame
    - Trade logs as DataFrames
    - A formatted console summary

    The ``@property`` decorator (used on ``metrics`` and ``all_pnls``) makes
    a method callable without parentheses: ``results.metrics`` instead of
    ``results.metrics()``.  This is a Python convention for computed attributes
    that look like simple data access but involve calculation.
    """

    def __init__(
        self,
        trades: list[BacktestTrade],
        options_trades: list[OptionsBacktestTrade],
        snapshots: list[DailySnapshot],
        positions: list[BacktestPosition],
        config: BacktestConfig,
    ):
        self.trades = trades
        self.options_trades = options_trades
        self.snapshots = snapshots
        self.positions = positions
        self.config = config

    @property
    def all_pnls(self) -> list[float]:
        """Combined P&L list from both stock and options trades."""
        return [t.pnl for t in self.trades] + [t.pnl for t in self.options_trades]

    @property
    def metrics(self) -> dict[str, Any]:
        """Compute aggregate performance metrics from all completed trades.

        Returns a dict of metrics including:
        - Win rate, total P&L, return percentage
        - Sharpe ratio (risk-adjusted return)
        - Max drawdown (largest peak-to-trough decline)
        - Profit factor (gross wins / gross losses)
        - Per-strategy breakdowns
        - Win/loss streaks

        SHARPE RATIO CALCULATION:
        Sharpe = (mean_return / std_dev_of_returns) * sqrt(252)
        The sqrt(252) annualizes the ratio (252 = trading days per year).
        A Sharpe > 1.0 is generally considered good; > 2.0 is excellent.

        PROFIT FACTOR:
        profit_factor = gross_winning_dollars / gross_losing_dollars
        > 1.0 means net profitable.  Infinity means no losing trades.
        """
        all_pnls = self.all_pnls
        if not all_pnls:
            return self._empty_metrics()

        total_trades = len(self.trades) + len(self.options_trades)

        # Separate winning and losing trades
        wins_pnl = [p for p in all_pnls if p > 0]
        losses_pnl = [p for p in all_pnls if p <= 0]

        total_pnl = sum(all_pnls)
        gross_wins = sum(wins_pnl) if wins_pnl else 0
        gross_losses = abs(sum(losses_pnl)) if losses_pnl else 0

        avg_win = gross_wins / len(wins_pnl) if wins_pnl else 0
        avg_loss = gross_losses / len(losses_pnl) if losses_pnl else 0
        # Expectancy: average dollars gained per trade (positive = profitable system)
        expectancy = total_pnl / total_trades

        # ---- Sharpe ratio ----
        # Computed from per-trade P&L (not daily returns, since trade frequency varies)
        mean_pnl = total_pnl / total_trades
        variance = sum((p - mean_pnl) ** 2 for p in all_pnls) / len(all_pnls)
        std_pnl = math.sqrt(variance) if variance > 0 else 0
        # Annualize: multiply by sqrt(252) assuming ~1 trade per day on average
        sharpe = (mean_pnl / std_pnl * math.sqrt(252)) if std_pnl > 0 else 0

        # Max drawdown from equity curve (peak-to-trough decline)
        max_dd, max_dd_duration = self._max_drawdown()

        # Win streak / loss streak (consecutive wins or losses)
        max_win_streak, max_loss_streak = self._streaks()

        # ---- Average holding period ----
        holding_days = []
        for t in self.trades:
            try:
                entry = datetime.strptime(t.entry_date, "%Y-%m-%d")
                exit_ = datetime.strptime(t.exit_date, "%Y-%m-%d")
                holding_days.append((exit_ - entry).days)
            except ValueError:
                pass
        for t in self.options_trades:
            try:
                entry = datetime.strptime(t.entry_date, "%Y-%m-%d")
                exit_ = datetime.strptime(t.exit_date, "%Y-%m-%d")
                holding_days.append((exit_ - entry).days)
            except ValueError:
                pass

        # ---- Per-strategy breakdown ----
        # ``set(t.strategy for t in ...)`` collects unique strategy names.
        # Then for each, we filter trades and compute win rate and P&L.
        by_strategy = {}
        strategy_names = set(t.strategy for t in self.trades)
        for name in strategy_names:
            strat_trades = [t for t in self.trades if t.strategy == name]
            strat_wins = [t for t in strat_trades if t.pnl > 0]
            by_strategy[name] = {
                "trades": len(strat_trades),
                "win_rate": len(strat_wins) / len(strat_trades) if strat_trades else 0,
                "total_pnl": round(sum(t.pnl for t in strat_trades), 2),
                "avg_pnl": round(sum(t.pnl for t in strat_trades) / len(strat_trades), 2),
            }

        # Same breakdown for options strategies
        options_strategy_names = set(t.strategy for t in self.options_trades)
        for name in options_strategy_names:
            strat_trades = [t for t in self.options_trades if t.strategy == name]
            strat_wins = [t for t in strat_trades if t.pnl > 0]
            by_strategy[name] = {
                "trades": len(strat_trades),
                "win_rate": len(strat_wins) / len(strat_trades) if strat_trades else 0,
                "total_pnl": round(sum(t.pnl for t in strat_trades), 2),
                "avg_pnl": round(sum(t.pnl for t in strat_trades) / len(strat_trades), 2),
            }

        return {
            "total_trades": total_trades,
            "stock_trades": len(self.trades),
            "options_trades": len(self.options_trades),
            "winning_trades": len(wins_pnl),
            "losing_trades": len(losses_pnl),
            "win_rate": round(len(wins_pnl) / total_trades, 4) if total_trades else 0,
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_pnl / self.config.starting_capital * 100, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "expectancy": round(expectancy, 2),
            # float("inf") is Python for positive infinity (used when gross_losses = 0)
            "profit_factor": round(gross_wins / gross_losses, 2) if gross_losses > 0 else float("inf"),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "max_drawdown_duration_days": max_dd_duration,
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "avg_holding_days": round(sum(holding_days) / len(holding_days), 1) if holding_days else 0,
            "max_win": round(max(all_pnls), 2),
            "max_loss": round(min(all_pnls), 2),
            "starting_capital": self.config.starting_capital,
            "ending_equity": self.snapshots[-1].equity if self.snapshots else self.config.starting_capital,
            "by_strategy": by_strategy,
        }

    def _empty_metrics(self) -> dict[str, Any]:
        """Return a metrics dict with zero values (used when no trades occurred)."""
        return {
            "total_trades": 0, "stock_trades": 0, "options_trades": 0,
            "win_rate": 0, "total_pnl": 0,
            "total_return_pct": 0, "sharpe_ratio": 0, "max_drawdown_pct": 0,
            "starting_capital": self.config.starting_capital,
            "ending_equity": self.config.starting_capital, "by_strategy": {},
        }

    def _max_drawdown(self) -> tuple[float, int]:
        """Compute max drawdown percentage and duration in days from the equity curve.

        Max drawdown is the largest peak-to-trough decline in portfolio value.
        It answers: "What was the worst losing streak, and how deep did it go?"

        Algorithm:
        - Track the running peak (highest equity seen so far)
        - At each point, compute drawdown = (peak - current) / peak
        - Record the maximum drawdown and how many days it lasted

        Returns:
            (drawdown_pct, duration_days) -- e.g., (10.5, 14) means a 10.5%
            drawdown lasting 14 trading days.
        """
        if not self.snapshots:
            return 0.0, 0

        equities = [s.equity for s in self.snapshots]
        peak = equities[0]
        max_dd = 0.0
        dd_start = 0
        max_dd_duration = 0
        current_dd_start = 0

        for i, eq in enumerate(equities):
            if eq > peak:
                peak = eq                    # New all-time high
                current_dd_start = i         # Reset drawdown start
            dd = (peak - eq) / peak if peak > 0 else 0  # Current drawdown percentage
            if dd > max_dd:
                max_dd = dd
                max_dd_duration = i - current_dd_start  # Days since peak

        return max_dd * 100, max_dd_duration  # Convert to percentage

    def _streaks(self) -> tuple[int, int]:
        """Compute longest winning and losing streaks.

        A "streak" is consecutive trades with the same outcome (win or loss).
        Long win streaks suggest a strategy is in a favorable regime.
        Long loss streaks may indicate a need to pause or adjust.
        """
        all_pnls = self.all_pnls
        if not all_pnls:
            return 0, 0

        max_w = max_l = cur_w = cur_l = 0
        for p in all_pnls:
            if p > 0:
                cur_w += 1    # Extend win streak
                cur_l = 0     # Reset loss streak
            else:
                cur_l += 1    # Extend loss streak
                cur_w = 0     # Reset win streak
            max_w = max(max_w, cur_w)
            max_l = max(max_l, cur_l)
        return max_w, max_l

    def equity_curve(self) -> pd.DataFrame:
        """Return the equity curve as a pandas DataFrame.

        The equity curve is a time series of daily portfolio values.  It's
        the primary visualization for evaluating a backtest -- a rising curve
        means the strategy is making money, and dips show drawdown periods.

        Returns a DataFrame with columns: equity, cash, positions, options,
        indexed by date.
        """
        if not self.snapshots:
            return pd.DataFrame()
        # List comprehension builds a list of dicts, which pd.DataFrame()
        # converts into a table (each dict = one row).
        data = [
            {"date": s.date, "equity": s.equity, "cash": s.cash,
             "positions": s.open_positions, "options": s.open_options}
            for s in self.snapshots
        ]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])  # Convert string dates to datetime objects
        df = df.set_index("date")                 # Use date as the row index
        return df

    def trades_df(self) -> pd.DataFrame:
        """Return all stock trades as a pandas DataFrame.

        Uses ``dataclasses.asdict()`` to convert each BacktestTrade dataclass
        instance into a plain dict, which pandas can directly consume.
        """
        if not self.trades:
            return pd.DataFrame()
        from dataclasses import asdict  # Converts a dataclass instance to a dict
        return pd.DataFrame([asdict(t) for t in self.trades])

    def options_trades_df(self) -> pd.DataFrame:
        """Return all options trades as a pandas DataFrame."""
        if not self.options_trades:
            return pd.DataFrame()
        from dataclasses import asdict
        return pd.DataFrame([asdict(t) for t in self.options_trades])

    def print_summary(self) -> None:
        """Print a formatted performance summary to the console.

        Displays key metrics in a human-readable table format, including
        per-strategy breakdowns showing which strategies contributed to
        (or detracted from) overall performance.
        """
        m = self.metrics
        if m["total_trades"] == 0:
            print("\n  No trades were generated during the backtest period.\n")
            return

        stock_count = m.get("stock_trades", len(self.trades))
        opts_count = m.get("options_trades", len(self.options_trades))
        trade_label = f"{m['total_trades']}"
        if opts_count > 0:
            trade_label += f"  (stocks: {stock_count}, options: {opts_count})"

        # Triple-quoted f-string: multi-line string with embedded expressions.
        # The ``{'='*60}`` syntax evaluates '=' * 60 = a line of 60 equals signs.
        # Format specifiers:
        #   :,.2f  = comma-separated with 2 decimals (e.g., 1,234.56)
        #   :+.2f  = always show sign (e.g., +12.34 or -5.67)
        #   :.1%   = percentage with 1 decimal (0.65 -> "65.0%")
        #   :.1f   = 1 decimal place
        print(f"""
{'='*60}
  BACKTEST RESULTS
{'='*60}
  Period:         {self.snapshots[0].date} to {self.snapshots[-1].date}
  Starting:       ${m['starting_capital']:,.2f}
  Ending:         ${m['ending_equity']:,.2f}
  Total Return:   {m['total_return_pct']:+.2f}%
{'-'*60}
  Total Trades:   {trade_label}
  Win Rate:       {m['win_rate']:.1%}
  Avg Win:        ${m['avg_win']:,.2f}
  Avg Loss:       ${m['avg_loss']:,.2f}
  Expectancy:     ${m['expectancy']:,.2f} per trade
{'-'*60}
  Total P&L:      ${m['total_pnl']:+,.2f}
  Profit Factor:  {m.get('profit_factor', 0):.2f}
  Sharpe Ratio:   {m['sharpe_ratio']:.2f}
{'-'*60}
  Max Drawdown:   {m['max_drawdown_pct']:.1f}%  ({m.get('max_drawdown_duration_days', 0)} days)
  Max Win:        ${m.get('max_win', 0):+,.2f}
  Max Loss:       ${m.get('max_loss', 0):+,.2f}
  Win Streak:     {m.get('max_win_streak', 0)}
  Loss Streak:    {m.get('max_loss_streak', 0)}
  Avg Hold:       {m.get('avg_holding_days', 0)} days
{'-'*60}
  Strategy Breakdown:""")

        for name, stats in m.get("by_strategy", {}).items():
            print(
                f"    {name:20s}  "
                f"trades={stats['trades']:3d}  "
                f"WR={stats['win_rate']:.0%}  "
                f"P&L=${stats['total_pnl']:+,.2f}"
            )

        print(f"{'='*60}\n")
