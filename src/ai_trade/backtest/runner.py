"""Backtest CLI runner -- fetch historical data and run strategies.

This module is the **entry point** for the backtesting system. It handles:
1. Parsing command-line arguments (which symbols, date range, options)
2. Fetching historical price data from the Alpaca brokerage API
3. Constructing the configured strategies (stock and options)
4. Creating and running the BacktestEngine (defined in engine.py)
5. Displaying results and optionally exporting to CSV

HOW IT FITS IN THE BACKTEST PIPELINE:
-------------------------------------
    runner.py  -->  engine.py  -->  options_pricing.py
    (CLI/setup)    (simulation)    (synthetic pricing)

The runner is the orchestrator: it gathers all inputs, delegates the
simulation to BacktestEngine, and handles output formatting.

Usage examples (from the command line):
    ai-trade-backtest --symbols AAPL MSFT TSLA --days 180
    ai-trade-backtest --default-universe --start 2025-01-01 --end 2025-12-31 --options
    ai-trade-backtest --symbols-file watchlist.txt --start 2025-01-01 --end 2025-12-31

KEY PYTHON CONCEPTS FOR NON-PYTHON READERS:
--------------------------------------------
- ``argparse``: Python's standard library for building command-line interfaces.
  You define arguments (--symbols, --days, etc.), their types, and help text.
  argparse automatically generates --help output and validates user input.
  Similar to Commander.js (Node) or clap (Rust).

- ``from __future__ import annotations``: Makes type hints (like ``str | None``)
  lazy-evaluated strings, enabling newer syntax on older Python versions.

- ``Path``: From the ``pathlib`` module -- an object-oriented wrapper around
  file system paths.  ``Path("foo.txt").read_text()`` reads a file's contents.
  More portable than raw string manipulation for file paths.

- ``ZoneInfo``: Python's built-in timezone database (replaces the older ``pytz``
  library).  ``ZoneInfo("America/New_York")`` gives US Eastern time.

- ``if __name__ == "__main__"``: A Python idiom meaning "only run this code
  if the file is executed directly (not imported as a library)."  It's the
  equivalent of a ``main()`` function guard in C/Java.

TRADING CONCEPTS:
-----------------
- **Backtesting**: Simulating a trading strategy on historical data to estimate
  how it would have performed.  NOT a guarantee of future results (past
  performance does not predict the future), but useful for eliminating
  strategies that would have failed historically.

- **Universe**: The set of stocks that the backtest will evaluate.  A larger
  universe finds more opportunities but takes longer to run.

- **PDT (Pattern Day Trader) Rule**: US regulation requiring $25,000 minimum
  equity for accounts that make 4+ day trades in 5 business days.  This
  backtest simulates PDT constraints for small accounts.
"""

from __future__ import annotations

import argparse  # Standard library for CLI argument parsing
import sys       # System-level functions (sys.exit for non-zero exit codes)
from datetime import datetime, timedelta
from pathlib import Path        # Object-oriented filesystem paths
from zoneinfo import ZoneInfo   # IANA timezone database (Python 3.9+)

from ai_trade.backtest.engine import BacktestConfig, BacktestEngine
from ai_trade.config import load_config          # Loads settings.yaml into a typed config object
from ai_trade.clients import init_clients         # Initializes API clients (Alpaca, etc.)
from ai_trade.data.historical import fetch_bars_multi  # Fetches OHLCV bars for multiple symbols
from ai_trade.monitoring.logger import setup_logging, get_logger
from ai_trade.strategy.bb_squeeze import BBSqueezeStrategy
from ai_trade.strategy.ema_crossover import EMACrossoverStrategy
from ai_trade.strategy.macd_divergence import MACDDivergenceStrategy
from ai_trade.strategy.mean_reversion import MeanReversionStrategy
from ai_trade.strategy.momentum import MomentumStrategy
from ai_trade.strategy.orb import ORBStrategy
from ai_trade.strategy.pullback import PullbackStrategy
from ai_trade.strategy.vwap import VWAPStrategy

log = get_logger(__name__)  # __name__ is a Python builtin: the current module's import path

# US Eastern timezone -- US stock markets operate on Eastern time
ET = ZoneInfo("America/New_York")

# Well-known liquid stocks for quick testing when the user doesn't specify symbols.
# "Liquid" means high trading volume, ensuring realistic backtesting (fills would
# actually happen at the simulated prices in real life).
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "AMD",
    "NFLX", "BABA", "PLTR", "SOFI", "NIO", "SNAP", "ROKU", "SQ",
    "COIN", "HOOD", "RIVN", "LCID", "MARA", "RIOT", "UPST", "AFRM",
]


def _load_full_universe(cfg) -> list[str]:
    """Pull all tradeable US equities from Alpaca and filter for liquidity.

    Returns symbols that are active, tradable, and on major exchanges.
    This gives hundreds of symbols for broad ML training data generation.
    """
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus

    init_clients(cfg)
    from ai_trade.clients import get_trading_client

    valid_exchanges = {"NYSE", "NASDAQ", "ARCA", "AMEX", "BATS"}
    request = GetAssetsRequest(
        asset_class=AssetClass.US_EQUITY,
        status=AssetStatus.ACTIVE,
    )
    assets = get_trading_client().get_all_assets(filter=request)
    symbols = [
        a.symbol for a in assets
        if a.tradable and a.exchange in valid_exchanges
        and not a.symbol.isdigit()       # skip weird numeric tickers
        and "." not in a.symbol          # skip preferred shares (BRK.B etc)
        and len(a.symbol) <= 5           # skip long tickers (warrants, units)
    ]
    return sorted(symbols)


def _build_options_strategies(cfg):
    """Build options strategy instances from the application config.

    Reads the ``settings.yaml`` config to determine which options strategies
    are enabled, then instantiates them.  Only strategies whose config section
    has ``enabled: true`` are included.

    Certain strategies are intentionally excluded:
    - covered_call and covered_straddle require holding 100 shares of the
      underlying stock (one standard options contract = 100 shares), which
      isn't practical for the small ($500 default) backtest account.

    Args:
        cfg: The application config object (loaded from settings.yaml).

    Returns:
        A list of instantiated options strategy objects.

    Python notes:
        - ``getattr(obj, "name", default)``: Safe attribute access that returns
          ``default`` instead of raising an error if the attribute doesn't exist.
          Similar to ``obj?.name ?? default`` in JavaScript/TypeScript.
        - The imports are *inside* this function (not at the top of the file).
          This is intentional: it avoids importing heavy options strategy modules
          unless options mode is actually requested (lazy loading).
    """
    strategies = []

    # getattr safely accesses cfg.options, returning None if it doesn't exist.
    # This prevents a crash if the config file has no "options" section.
    opts_cfg = getattr(cfg, "options", None)
    if not opts_cfg or not getattr(opts_cfg, "enabled", False):
        return strategies

    strat_cfg = cfg.strategies

    # Lazy imports: only load these modules if options are enabled.
    # Each strategy class encapsulates the logic for one options trading approach.
    from ai_trade.strategy.options.long_call import LongCallStrategy
    from ai_trade.strategy.options.long_put import LongPutStrategy
    from ai_trade.strategy.options.credit_put_spread import CreditPutSpreadStrategy
    from ai_trade.strategy.options.debit_call_spread import DebitCallSpreadStrategy
    from ai_trade.strategy.options.cash_secured_put import CashSecuredPutStrategy
    from ai_trade.strategy.options.momentum_options import MomentumOptionsStrategy

    # strategy_map is a Python dict (hash map) mapping config names to classes.
    # This pattern replaces a long if/elif chain with a data-driven approach.
    strategy_map = {
        "long_call": LongCallStrategy,
        "long_put": LongPutStrategy,
        "credit_put_spread": CreditPutSpreadStrategy,
        "debit_call_spread": DebitCallSpreadStrategy,
        "cash_secured_put": CashSecuredPutStrategy,
        "momentum_options": MomentumOptionsStrategy,
    }

    # Iterate over the map: for each strategy, check if its config section
    # exists and is enabled, then instantiate it with its config.
    for name, cls in strategy_map.items():
        scfg = getattr(strat_cfg, name, None)  # e.g., cfg.strategies.long_call
        if scfg and getattr(scfg, "enabled", False):
            strategies.append(cls(scfg))  # cls(scfg) calls the constructor

    return strategies


def run_backtest(
    symbols: list[str],
    start: str,
    end: str,
    config_path: str | None = None,
    show_trades: bool = False,
    export_csv: str | None = None,
    include_options: bool = False,
    train_ml: bool = False,
    capital_override: float | None = None,
) -> None:
    """Fetch data, configure strategies, and run the backtest.

    This is the main orchestration function. It:
    1. Loads configuration from settings.yaml
    2. Initializes API clients for data fetching
    3. Downloads historical OHLCV (Open/High/Low/Close/Volume) bars
    4. Builds the list of enabled stock and options strategies
    5. Configures risk/position sizing parameters
    6. Runs the backtest engine
    7. Prints the summary and optionally exports CSV files

    Args:
        symbols: List of stock ticker symbols to backtest (e.g., ["AAPL", "MSFT"]).
        start: Start date as ISO string "YYYY-MM-DD" (inclusive).
        end: End date as ISO string "YYYY-MM-DD" (inclusive).
        config_path: Optional path to a custom settings.yaml file.
        show_trades: If True, print every individual trade to the console.
        export_csv: If provided, base filename for CSV export (e.g., "results"
                   produces "results.trades.csv", "results.equity.csv").
        include_options: If True, also run options strategies using synthetic
                        Black-Scholes pricing from options_pricing.py.

    Python notes:
        - ``str | None``: Union type meaning "either a string or None."
        - ``-> None``: This function doesn't return a value (like void in C/Java).
        - ``list[str]``: A list (dynamic array) containing strings.
    """
    setup_logging()
    cfg = load_config(config_path)  # Parses settings.yaml into a typed config object
    init_clients(cfg)               # Connects to Alpaca (or other data providers)

    print(f"\n  Fetching historical data for {len(symbols)} symbols...")
    print(f"  Period: {start} to {end}\n")

    # ---------- Fetch historical price data ----------
    # TimeFrame.Day means we're using daily bars (one OHLCV row per trading day).
    # For backtesting, daily resolution is standard -- intraday (minute/hour)
    # bars would be needed for high-frequency strategy testing.
    from alpaca.data.timeframe import TimeFrame

    # datetime.strptime parses a date string into a datetime object.
    # .replace(tzinfo=ET) attaches the Eastern timezone (required by the API).
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=ET)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=ET)

    # Fetch in batches of 200 to avoid overwhelming the API for large universes.
    BATCH_SIZE = 200
    bars_dict: dict[str, any] = {}
    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i : i + BATCH_SIZE]
        if len(symbols) > BATCH_SIZE:
            print(f"  Fetching batch {i // BATCH_SIZE + 1}/{(len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE} ({len(batch)} symbols)...")
        batch_bars = fetch_bars_multi(batch, TimeFrame.Day, start_dt, end_dt)
        bars_dict.update(batch_bars)

    loaded = {sym: df for sym, df in bars_dict.items() if not df.empty}
    print(f"  Loaded data for {len(loaded)}/{len(symbols)} symbols")

    if not loaded:
        print("  No data available. Check your date range and symbols.")
        return

    # ---------- Configure stock strategies from settings.yaml ----------
    # Each strategy is a class that evaluates daily price data and generates
    # buy/sell signals with conviction scores, stop-loss, and take-profit levels.
    strat_cfg = cfg.strategies
    strategies = []
    if strat_cfg.mean_reversion.enabled:
        strategies.append(MeanReversionStrategy(strat_cfg.mean_reversion))
    if strat_cfg.momentum.enabled:
        strategies.append(MomentumStrategy(strat_cfg.momentum))
    if strat_cfg.vwap.enabled:
        strategies.append(VWAPStrategy(strat_cfg.vwap))
    if getattr(strat_cfg, "ema_crossover", None) and strat_cfg.ema_crossover.enabled:
        strategies.append(EMACrossoverStrategy(strat_cfg.ema_crossover))
    if getattr(strat_cfg, "macd_divergence", None) and strat_cfg.macd_divergence.enabled:
        strategies.append(MACDDivergenceStrategy(strat_cfg.macd_divergence))
    if getattr(strat_cfg, "bb_squeeze", None) and strat_cfg.bb_squeeze.enabled:
        strategies.append(BBSqueezeStrategy(strat_cfg.bb_squeeze))
    if getattr(strat_cfg, "orb", None) and strat_cfg.orb.enabled:
        strategies.append(ORBStrategy(strat_cfg.orb))
    if getattr(strat_cfg, "pullback", None) and strat_cfg.pullback.enabled:
        strategies.append(PullbackStrategy(strat_cfg.pullback))

    if not strategies:
        print("  No stock strategies enabled in config.")

    # ---------- Configure options strategies (if requested) ----------
    options_strategies = []
    if include_options:
        options_strategies = _build_options_strategies(cfg)
        if options_strategies:
            # type(s).__name__ gets the class name of each strategy instance.
            # .join() concatenates a list of strings with a separator.
            print(f"  Options strategies: {', '.join(type(s).__name__ for s in options_strategies)}")
        else:
            print("  Warning: --options flag set but no options strategies enabled in config.")

    if not strategies and not options_strategies:
        print("  No strategies of any kind enabled.")
        return

    print(f"  Stock strategies: {', '.join(type(s).__name__ for s in strategies)}")

    # ---------- Build BacktestConfig from settings ----------
    # BacktestConfig is a dataclass that holds all risk management and position
    # sizing parameters.  These come from settings.yaml but could be overridden.
    effective_capital = capital_override or cfg.account.starting_capital

    # When capital is overridden (e.g. for ML training), scale up position
    # limits so capital constraints don't artificially suppress signal generation.
    if capital_override and capital_override >= 25000:
        max_positions = max(cfg.account.max_open_positions, 10)
        heat_pct = max(getattr(cfg.risk, "max_portfolio_heat_pct", 0.06), 0.12)
    else:
        max_positions = cfg.account.max_open_positions
        heat_pct = getattr(cfg.risk, "max_portfolio_heat_pct", 0.06)

    bt_config = BacktestConfig(
        starting_capital=effective_capital,
        max_position_pct=cfg.account.max_position_pct,
        max_risk_per_trade_pct=cfg.account.max_risk_per_trade_pct,
        max_open_positions=max_positions,
        daily_loss_limit_pct=cfg.account.daily_loss_limit_pct,
        max_portfolio_heat_pct=heat_pct,
        trailing_stop_pct=getattr(cfg.risk, "trailing_stop_pct", 0.02),
        max_day_trades=cfg.pdt.max_day_trades,
        day_trade_reserve=cfg.pdt.day_trade_reserve,
        min_conviction_for_day_trade=cfg.pdt.min_conviction_for_day_trade,
    )

    # Overlay options-specific config if options mode is enabled
    if include_options:
        opts_cfg = getattr(cfg, "options", None)
        if opts_cfg:
            bt_config.max_options_positions = getattr(opts_cfg, "max_options_positions", 3)
            bt_config.max_options_capital_pct = getattr(opts_cfg, "max_options_capital_pct", 0.50)
            bt_config.max_single_options_risk_pct = getattr(opts_cfg, "max_single_options_risk_pct", 0.12)
            bt_config.options_profit_target_pct = getattr(opts_cfg, "options_profit_target_pct", 0.50)
            bt_config.options_loss_limit_pct = getattr(opts_cfg, "options_loss_limit_pct", 2.0)
            bt_config.options_slippage_pct = getattr(opts_cfg, "options_slippage_pct", 0.003)

    # ---------- Fetch market regime data (SPY, QQQ) ----------
    # Market regime analysis uses broad market index data to determine
    # whether the overall market is bullish, bearish, or neutral.
    # The engine uses this to gate new entries (e.g., no new longs in a bear market).
    # We fetch extra history (300 days back) so the regime analyzer has enough
    # lookback data for its moving averages.
    market_bars = None
    if bt_config.use_market_regime:
        print("  Fetching SPY/QQQ for market regime analysis...")
        regime_start = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=300)
        regime_start_dt = regime_start.replace(tzinfo=ET)
        mkt_data = fetch_bars_multi(["SPY", "QQQ"], TimeFrame.Day, regime_start_dt, end_dt)
        # Dict comprehension again: filter out empty DataFrames
        market_bars = {sym: df for sym, df in mkt_data.items() if not df.empty}
        if market_bars:
            # list(dict.keys()) extracts the keys (symbol names) as a list
            print(f"  Market data loaded: {list(market_bars.keys())}")

    # ---------- Run the backtest ----------
    # f-string formatting: ${value:,.0f} formats as currency with commas, no decimals
    label = f"${bt_config.starting_capital:,.0f} starting capital"
    if include_options:
        label += ", options enabled"
    print(f"\n  Running backtest ({label})...\n")

    engine = BacktestEngine(strategies, bt_config, options_strategies=options_strategies)
    results = engine.run(loaded, start_date=start, end_date=end, market_bars=market_bars)

    # ---------- Output results ----------
    results.print_summary()

    # Optionally print every individual trade
    if show_trades:
        # Stock trades
        df = results.trades_df()  # Returns a pandas DataFrame of all closed stock trades
        if not df.empty:
            print("\n  Stock Trades:")
            print("  " + "-" * 100)
            # df.iterrows() iterates over DataFrame rows as (index, Series) pairs.
            # The underscore _ is a Python convention for "I don't need this variable"
            # (here we ignore the row index).
            for _, row in df.iterrows():
                # f-string format specifiers:
                #   :6s   = left-align string in 6-char field
                #   :18s  = left-align string in 18-char field
                #   :3.0f = 3-char wide float with 0 decimals
                #   :7.2f = 7-char wide float with 2 decimals
                #   :+7.2f = same but always show +/- sign
                #   :.1%  = format as percentage with 1 decimal (0.05 -> "5.0%")
                print(
                    f"  {row['entry_date']} -> {row['exit_date']}  "
                    f"{row['symbol']:6s}  {row['strategy']:18s}  "
                    f"{row['shares']:3.0f} sh  "
                    f"${row['entry_price']:7.2f} -> ${row['exit_price']:7.2f}  "
                    f"P&L ${row['pnl']:+7.2f} ({row['pnl_pct']:+.1%})  "
                    f"[{row['exit_reason']}]"
                )
            print()

        # Options trades
        opts_df = results.options_trades_df()
        if not opts_df.empty:
            print("\n  Options Trades:")
            print("  " + "-" * 100)
            for _, row in opts_df.iterrows():
                # Generator expression inside join(): format each strike as "$XXX"
                strikes_str = "/".join(f"${s:.0f}" for s in row["strikes"])
                # Ternary expression: condition_true_value if condition else condition_false_value
                cost_label = f"cost ${row['entry_cost']:.2f}" if row["entry_cost"] > 0 else f"credit ${abs(row['entry_cost']):.2f}"
                print(
                    f"  {row['entry_date']} -> {row['exit_date']}  "
                    f"{row['underlying']:6s}  {row['strategy']:20s}  "
                    f"{row['contracts']}x {strikes_str}  "
                    f"{cost_label}  "
                    f"P&L ${row['pnl']:+7.2f} ({row['pnl_pct']:+.1%})  "
                    f"[{row['exit_reason']}]"
                )
            print()

    # ---------- Export to CSV ----------
    if export_csv:
        trades_df = results.trades_df()
        equity_df = results.equity_curve()  # DataFrame of daily equity values (the "equity curve")
        if not trades_df.empty:
            # Path.with_suffix() replaces the file extension.
            # e.g., Path("results.csv").with_suffix(".trades.csv") -> "results.trades.csv"
            trades_path = Path(export_csv).with_suffix(".trades.csv")
            trades_df.to_csv(trades_path, index=False)  # index=False omits the row numbers
            print(f"  Trades exported to {trades_path}")
        opts_df = results.options_trades_df()
        if not opts_df.empty:
            opts_path = Path(export_csv).with_suffix(".options_trades.csv")
            opts_df.to_csv(opts_path, index=False)
            print(f"  Options trades exported to {opts_path}")
        if not equity_df.empty:
            equity_path = Path(export_csv).with_suffix(".equity.csv")
            equity_df.to_csv(equity_path)  # index=True here because date IS the index
            print(f"  Equity curve exported to {equity_path}")
        print()

    # ---------- ML training from backtest data ----------
    if train_ml:
        _train_ml_from_backtest(results)


def _train_ml_from_backtest(results) -> None:
    """Write backtest ML training data into a database and train the model.

    The backtest engine captures the same feature vectors as live trading
    (via extract_features) at each entry, then pairs them with the actual
    trade PnL at close.  This function:
      1. Creates (or reuses) a backtest-specific SQLite database
      2. Inserts each (features, outcome) pair as a trade + ml_features row
      3. Calls the standard trainer to fit a GradientBoostingClassifier
    """
    import json
    from ai_trade.monitoring.database import Database
    from ai_trade.ml.trainer import train_signal_quality_model

    ml_data = results.ml_training_data
    if not ml_data:
        print("\n  ML Training: No feature data captured during backtest.")
        print("  (Strategies may not have generated any signals.)")
        return

    print(f"\n  ML Training: {len(ml_data)} labelled trade(s) from backtest")

    # Use a dedicated backtest database so we don't pollute the live DB.
    # Located next to the live db in the data/ directory.
    db_path = Path(__file__).resolve().parents[3] / "data" / "backtest_ml.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(str(db_path))

    # Wipe prior backtest training data so we train on the fresh run only.
    # This prevents stale data from prior backtest runs from accumulating.
    with db._conn() as conn:
        conn.execute("DELETE FROM ml_features WHERE trade_id IN (SELECT id FROM trades WHERE bot_version = 'backtest')")
        conn.execute("DELETE FROM trades WHERE bot_version = 'backtest'")

    # Insert each trade + its feature snapshot
    inserted = 0
    for row in ml_data:
        trade_id = db.insert_trade(
            symbol=row["symbol"],
            strategy=row["strategy"],
            side="buy",
            shares=1,
            entry_price=row["features"].get("entry_price", 0),
            entry_time=row["entry_date"],
            exit_price=0,
            exit_time=row["exit_date"],
            stop_loss=0,
            take_profit=0,
            hold_type=row["hold_type"],
            pnl=row["pnl"],
            pnl_pct=row["pnl_pct"],
            status="closed",
            bot_version="backtest",
        )
        db.insert_ml_features(
            trade_id=trade_id,
            features=json.dumps(row["features"]),
        )
        inserted += 1

    print(f"  Inserted {inserted} labelled trades into {db_path.name}")

    # Train the model
    result = train_signal_quality_model(db, min_trades=30)
    status = result.get("status", "unknown")

    if status == "ok":
        print(f"\n  Model trained successfully!")
        print(f"    Version:          {result['version']}")
        print(f"    Trades used:      {result['trades_used']}")
        print(f"    Train accuracy:   {result['train_accuracy']:.1%}")
        print(f"    Val accuracy:     {result['val_accuracy']:.1%}")
        print(f"    Saved to:         {result['model_path']}")

        # Register in the live database so the bot picks it up on next restart
        live_db_path = Path(__file__).resolve().parents[3] / "data" / "ai_trade.db"
        if live_db_path.exists():
            live_db = Database(str(live_db_path))
            try:
                from ai_trade.ml.trainer import _deactivate_prior_versions, MODEL_NAME
                _deactivate_prior_versions(live_db, MODEL_NAME)
                live_db.insert_ml_model(
                    model_name=MODEL_NAME,
                    version=result["version"],
                    trained_at=result.get("trained_at", ""),
                    training_trades=result["trades_used"],
                    backtest_accuracy=result["val_accuracy"],
                    is_active=1,
                    model_path=result["model_path"],
                )
                print(f"    Registered in live DB (active on next bot restart)")
            except Exception as e:
                print(f"    Warning: could not register in live DB: {e}")
    elif status == "insufficient_data":
        print(f"\n  Insufficient data for training: {result['trades_available']} trades")
        print(f"  (Need at least {result['trades_required']})")
        print("  Try a longer backtest period or more symbols.")
    elif status == "single_class":
        print(f"\n  All {result['trades_used']} trades had the same outcome (class={result['class']}).")
        print("  Need both winning and losing trades to train.")
    else:
        print(f"\n  Training failed: {result}")


def main():
    """CLI entry point: parse arguments and invoke run_backtest.

    This function uses ``argparse`` to define the command-line interface.
    argparse is Python's standard library for CLI argument parsing -- similar
    to Commander.js (Node), clap (Rust), or getopt (C).

    HOW ARGPARSE WORKS:
    1. Create an ArgumentParser with a description.
    2. Call add_argument() for each CLI flag/option, specifying:
       - The flag name (e.g., "--symbols")
       - nargs="+" means "one or more values" (collects into a list)
       - action="store_true" makes it a boolean flag (present = True)
       - type=int/str specifies the expected type (argparse validates this)
       - default= sets the value when the flag is omitted
       - help= is the description shown in --help output
    3. Call parse_args() which reads sys.argv, validates, and returns
       a namespace object where args.symbols, args.days, etc. are set.
    """
    parser = argparse.ArgumentParser(
        description="AI Trade Backtester -- test strategies on historical data"
    )

    # --symbols AAPL MSFT TSLA: nargs="+" means "one or more space-separated values"
    # These are collected into a list: args.symbols = ["AAPL", "MSFT", "TSLA"]
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Stock symbols to backtest (e.g., AAPL MSFT TSLA)"
    )

    # --symbols-file path/to/file.txt: read symbols from a file (one per line)
    parser.add_argument(
        "--symbols-file", type=str, default=None,
        help="Path to a file with one symbol per line"
    )

    # --default-universe: action="store_true" means this is a boolean flag.
    # Present on command line = True, absent = False. No value needed.
    parser.add_argument(
        "--default-universe", action="store_true",
        help=f"Use the built-in universe of {len(DEFAULT_UNIVERSE)} liquid stocks"
    )
    parser.add_argument(
        "--full-universe", action="store_true",
        help="Pull all tradeable US equities from Alpaca for broad ML training"
    )
    parser.add_argument(
        "--capital", type=float, default=None,
        help="Override starting capital (default: from settings.yaml)"
    )

    parser.add_argument(
        "--days", type=int, default=90,
        help="Number of calendar days to look back (default: 90)"
    )

    # --start and --end override --days for precise date range control
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start date (YYYY-MM-DD). Overrides --days."
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date (YYYY-MM-DD). Default: today."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to settings.yaml"
    )
    parser.add_argument(
        "--options", action="store_true",
        help="Include options strategies in the backtest (uses Black-Scholes synthetic pricing)"
    )
    parser.add_argument(
        "--show-trades", action="store_true",
        help="Print all individual trades"
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export results to CSV (provide base filename)"
    )
    parser.add_argument(
        "--train-ml", action="store_true",
        help="Train the ML signal-quality model from backtest trade data"
    )

    # parse_args() reads from sys.argv (the command-line arguments passed to the script),
    # validates them against the definitions above, and returns a Namespace object.
    args = parser.parse_args()

    # ---------- Resolve symbols ----------
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    elif args.symbols_file:
        path = Path(args.symbols_file)
        if not path.exists():
            print(f"  Error: file not found: {path}")
            sys.exit(1)
        symbols = [line.strip().upper() for line in path.read_text().splitlines() if line.strip()]
    elif args.full_universe:
        setup_logging()
        cfg = load_config(args.config)
        init_clients(cfg)
        print("  Loading full tradeable universe from Alpaca...")
        symbols = _load_full_universe(cfg)
        print(f"  Found {len(symbols)} tradeable symbols")
    elif args.default_universe:
        symbols = DEFAULT_UNIVERSE
    else:
        print("  No symbols specified. Use --symbols, --symbols-file, --default-universe, or --full-universe.")
        print("  Example: ai-trade-backtest --symbols AAPL MSFT TSLA --days 90")
        sys.exit(1)

    # ---------- Resolve date range ----------
    # If --end is not specified, default to today in Eastern time.
    # datetime.now(ET) gets the current time in the US Eastern timezone.
    # .strftime("%Y-%m-%d") formats it as "2025-03-15".
    end_date = args.end or datetime.now(ET).strftime("%Y-%m-%d")

    if args.start:
        start_date = args.start
    else:
        # Compute start_date by subtracting --days calendar days from end_date.
        # timedelta(days=N) represents a duration of N days.
        start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=args.days)
        start_date = start_dt.strftime("%Y-%m-%d")

    # Delegate to the main orchestration function
    run_backtest(
        symbols=symbols,
        start=start_date,
        end=end_date,
        config_path=args.config,
        show_trades=args.show_trades,
        export_csv=args.export,
        include_options=args.options,
        train_ml=args.train_ml,
        capital_override=args.capital,
    )


# This guard ensures main() only runs when the script is executed directly
# (e.g., ``python runner.py`` or via the ``ai-trade-backtest`` console script),
# NOT when the module is imported by another Python file.
# In Python, when a file is run directly, __name__ is set to "__main__".
# When imported, __name__ is set to the module's import path (e.g., "ai_trade.backtest.runner").
if __name__ == "__main__":
    main()
