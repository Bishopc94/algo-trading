# Backtesting Guide

## What Is Backtesting?

Backtesting is the process of testing a trading strategy against **historical** price data to see how it *would have* performed in the past. Instead of risking real money to validate a strategy, you replay months or years of market data through the bot's logic and measure the results.

The AI Trade bot cannot use Alpaca's paper trading environment for historical tests -- paper trading only works in real time. So the backtester **simulates the entire trading pipeline** (signal generation, position sizing, order fills, stop losses, take profits) against past daily price bars. This lets you evaluate strategies over years of data in seconds rather than waiting months for live results.

> **Important**: Past performance does not guarantee future results. Backtesting tells you how a strategy *would have* done, not how it *will* do. See [Caveats and Limitations](#caveats-and-limitations) for the many ways backtests can mislead you.

---

## How the Backtester Works

### Architecture

The backtester is split into three files, each with a distinct responsibility:

```
runner.py (CLI + data fetching) --> engine.py (day-by-day simulation) --> options_pricing.py (synthetic Greeks)
```

| File | Role |
|------|------|
| `backtest/runner.py` | Command-line interface, argument parsing, historical data fetching, strategy construction |
| `backtest/engine.py` | Core simulation loop -- walks through each trading day, manages positions, tracks P&L |
| `backtest/options_pricing.py` | Black-Scholes pricing engine that generates synthetic options chains and Greeks |

### The Runner (`backtest/runner.py`)

The runner is the entry point. It handles everything *before* the simulation starts:

1. **Parses CLI arguments** using Python's `argparse` module (a standard library for building command-line interfaces). You specify symbols, date ranges, and which strategies to include.

2. **Resolves symbols** from one of three sources:
   - `--symbols AAPL MSFT TSLA` -- explicit list
   - `--symbols-file watchlist.txt` -- one symbol per line in a text file
   - `--default-universe` -- a built-in list of 24 liquid stocks (AAPL, MSFT, AMZN, GOOGL, META, TSLA, NVDA, AMD, NFLX, and others)

3. **Fetches historical daily bars** from Alpaca's market data API via `fetch_bars_multi()`. Each symbol gets a DataFrame (a tabular data structure from the `pandas` library, similar to a spreadsheet or SQL table) containing columns: open, high, low, close, volume.

4. **Builds strategy instances** from the `settings.yaml` configuration file. Stock strategies (momentum, mean reversion, VWAP) and options strategies (long call, credit put spread, etc.) are constructed based on which ones are enabled in the config.

5. **Optionally fetches SPY/QQQ data** for market regime analysis -- the engine uses broad market conditions to gate entries (e.g., avoiding new long positions in a strong bear market).

6. **Passes everything to the engine** and prints the results.

### The Engine (`backtest/engine.py`)

The engine is where the actual simulation happens. It walks through every trading day in the date range and simulates what the bot would have done.

#### Python Concepts Used

- **`@dataclass`**: A Python decorator that auto-generates boilerplate code (constructor, equality, string representation) for classes that are primarily data containers. `BacktestPosition`, `BacktestTrade`, `BacktestConfig`, etc. are all dataclasses. Think of them like structs with auto-generated constructors.

- **`pd.DataFrame`**: A `pandas` DataFrame is a 2D labeled data structure -- like a spreadsheet with named columns and an index. The engine receives one DataFrame per symbol, indexed by date, with columns for OHLCV (open, high, low, close, volume) plus computed indicators (RSI, Bollinger Bands, ATR, etc.).

- **Type hints** like `dict[str, pd.DataFrame]` and `list[BacktestPosition]`: These are annotations that document expected types. Python does not enforce them at runtime -- they serve as documentation and enable IDE autocompletion.

#### Day-by-Day Simulation Loop

For each trading day, the engine executes these steps in order:

```
1. Clear equity cache (positions may change today)
2. Update market regime from SPY/QQQ (if enabled)
3. Check exits on open STOCK positions (stop loss, take profit, trailing stop, strategy exit)
4. Check exits on open OPTIONS positions (expiration, profit target, loss limit)
5. Check daily loss limit -- if exceeded, stop trading for the day
6. Evaluate all stock strategies for new entry signals (unless bear market blocks longs)
7. Evaluate all options strategies for new entry signals
8. Close any day-trade positions at end of day
9. Take end-of-day snapshot (equity, cash, open positions, realized P&L)
```

**Why exits are processed before entries**: This is deliberate. Closing a position frees up both capital (cash returned from the sale) and a position slot (the engine enforces a maximum number of concurrent positions, default 4). Processing exits first means the engine can immediately redeploy that capital into new opportunities on the same day.

#### Key Optimizations

**O(1) date lookups**: Before the simulation starts, the engine pre-builds a nested dictionary (hash map) that maps `{symbol: {date_string: row_index}}`. This means looking up "what was AAPL's close price on 2024-06-15?" is a constant-time dictionary lookup instead of scanning through a DataFrame. Over hundreds of symbols and hundreds of trading days, this matters.

```python
# Pre-built once before simulation:
self._date_idx = {
    "AAPL": {"2024-01-02": 0, "2024-01-03": 1, ...},
    "MSFT": {"2024-01-02": 0, "2024-01-03": 1, ...},
}

# Used during simulation -- O(1) instead of O(n):
def _get_close(self, symbol, date_str, bars_dict):
    idx = self._date_idx.get(symbol, {}).get(date_str)
    if idx is None:
        return None
    return float(bars_dict[symbol].iloc[idx]["close"])
```

**Equity caching**: The `_equity()` method (which sums cash + market value of all positions) is cached per date string and invalidated at the start of each new day. This avoids redundant recalculation when multiple methods need the current equity on the same day.

#### Position Sizing

The engine uses **fixed-fractional position sizing**, which mirrors what the live bot does:

1. Calculate the maximum dollar risk: `equity * max_risk_per_trade_pct` (default 2%)
2. Calculate risk per share: `|entry_price - stop_loss_price|`
3. Shares = `risk_amount / risk_per_share`
4. Cap at `max_position_pct` of equity (default 30%)
5. Cap at available cash
6. Minimum 1 share if the account can afford it

#### Exit Conditions (Stocks)

For each open stock position, the engine checks the day's OHLCV bar:

| Condition | Trigger | Fill Price |
|-----------|---------|------------|
| **Stop loss** | Day's low <= stop price | Stop price |
| **Take profit** | Day's high >= target price | Target price |
| **Trailing stop** | After highest price updates, new trailing stop = highest * (1 - trailing_stop_pct) | Trailing stop price |
| **Strategy exit** | The strategy's `should_exit()` method returns `True` | Close price |

The trailing stop ratchets upward as the stock rises, locking in gains. It never moves downward.

#### Exit Conditions (Options)

| Condition | Trigger |
|-----------|---------|
| **Close before expiration** | 1 day before expiry (avoids assignment/exercise risk) |
| **Expiration** | On or after expiration date |
| **Profit target** | Mark-to-market P&L reaches 50% of max profit |
| **Loss limit** | Debit strategies: loss reaches 2x entry cost. Credit strategies: loss reaches 80% of max loss. |

#### Risk Controls

The engine enforces multiple layers of risk management:

- **Max open positions** (default 4): Won't enter new trades if at capacity
- **Portfolio heat limit** (default 6%): Total risk across all positions (sum of entry-to-stop distances) can't exceed 6% of equity
- **Daily loss limit** (default 5%): If equity drops more than 5% from the day's starting value, no more trading that day
- **PDT (Pattern Day Trader) compliance**: Tracks day trades in a rolling 5-day window; reserves 1 of the 3 allowed day trades for emergencies; requires high conviction (0.80+) for day trades
- **Market regime gating**: In a strong bear market (based on SPY/QQQ analysis), the engine skips all new long entries
- **Options capital allocation**: Options positions are capped at 40% of equity; individual options trades capped at 8% risk

### Synthetic Options Pricing (`backtest/options_pricing.py`)

Alpaca does not provide historical options data. You can get today's options chain, but not what the chain looked like on March 15, 2024. So the backtester **generates synthetic options chains** using the Black-Scholes model.

#### What Is Black-Scholes?

Black-Scholes is the standard mathematical formula for calculating the theoretical price of an options contract. Published in 1973, it won a Nobel Prize and remains the foundation of options pricing.

The formula takes five inputs and produces a fair price:

| Input | Symbol | What It Means |
|-------|--------|---------------|
| Stock price | S | Current price of the underlying stock |
| Strike price | K | The price at which the option can be exercised |
| Time to expiration | T | How long until the option expires, measured in years (e.g., 30 days = 30/365 = 0.082 years) |
| Risk-free rate | r | The "guaranteed" return you could get instead (e.g., US Treasury yield), default 4.5% |
| Volatility | sigma | How much the stock's price tends to fluctuate, annualized (see below) |

The core insight: an option's price reflects the **probability** that it will be worth exercising. A call option (right to buy at strike K) is worth more when the stock price S is high, when there's lots of time left T, and when the stock is volatile (sigma) -- because volatility means a bigger chance of the stock moving favorably.

#### How the Backtester Calculates Volatility

Since we don't have historical implied volatility data, the backtester estimates volatility from the stock's own price history:

1. Take the trailing 30 days of closing prices
2. Compute **log returns**: `ln(price_today / price_yesterday)` for each day
3. Calculate the standard deviation of those log returns
4. **Annualize** by multiplying by `sqrt(252)` (there are ~252 trading days per year)

This gives an annualized volatility figure like 0.35 (35%), which feeds into the Black-Scholes formula. If fewer than 30 days of data are available, it uses whatever is available (minimum 5 bars), and falls back to a default 30% volatility if data is too sparse.

#### What the Synthetic Chain Generator Produces

The `generate_synthetic_chain()` function creates a realistic-looking options chain for any stock on any historical date. Here is what it builds:

**Strike prices**: A range of strikes around the current stock price, spaced based on the stock's price level:

| Stock Price | Strike Step |
|------------|-------------|
| Under $5 | $0.50 |
| $5 - $25 | $1.00 |
| $25 - $100 | $2.50 |
| $100 - $500 | $5.00 |
| Over $500 | $10.00 |

Strikes span +/-15% from the current price by default.

**Expiration dates**: A mix of short-dated and longer-dated expirations:
- Short-dated: 3, 5, 7, and 10 days out (adjusted to the nearest Friday, since options expire on Fridays)
- Longer-dated: Biweekly out to ~60 days

**For each strike/expiration/type (call or put) combination**, the generator:
1. Calculates the theoretical price using Black-Scholes
2. Simulates a bid/ask spread (tighter for at-the-money options, wider for out-of-the-money -- just like real markets)
3. Computes the **Greeks** (delta, gamma, theta, vega) -- sensitivities that describe how the option's price changes relative to various factors

#### The Greeks

The Greeks are partial derivatives of the Black-Scholes formula. In plain terms:

| Greek | What It Measures | Example |
|-------|-----------------|---------|
| **Delta** | How much the option price moves per $1 move in the stock | Delta of 0.50 means option gains $0.50 when stock rises $1 |
| **Gamma** | How fast delta itself changes as the stock moves | High gamma = delta changes rapidly (dangerous near expiration) |
| **Theta** | How much value the option loses per day just from time passing | Theta of -0.05 means the option loses $0.05/day of time value |
| **Vega** | How much the option price changes per 1% change in volatility | Vega of 0.10 means option gains $0.10 if volatility rises 1% |

The backtester computes these per the standard Black-Scholes formulas and includes them in the synthetic snapshot data, so options strategies can filter contracts by delta, check theta decay, etc.

#### Options Mark-to-Market During Simulation

Rather than re-running full Black-Scholes on every options position every day (expensive and unnecessary for simulation), the engine uses a simplified mark-to-market:

1. Calculate the **intrinsic value** of each leg (what it would be worth if exercised right now)
2. Add a **time premium** that decays linearly from entry to expiration (approximating theta decay)
3. The time premium starts at ~30% of the entry cost and decreases to zero at expiration

This is a deliberate simplification. It captures the two biggest factors (intrinsic value and time decay) without the computational overhead of full repricing.

---

## Running a Backtest

### Command Line

The backtester is invoked as a Python module. Here are the most common usage patterns:

**Basic usage -- specific symbols over the last 90 days:**

```bash
python -m ai_trade.backtest.runner --symbols AAPL MSFT TSLA
```

**Specify date range and include options strategies:**

```bash
python -m ai_trade.backtest.runner \
    --symbols AAPL MSFT TSLA NVDA AMD \
    --start 2024-01-01 \
    --end 2025-12-31 \
    --options
```

**Use the built-in universe of 24 liquid stocks:**

```bash
python -m ai_trade.backtest.runner \
    --default-universe \
    --start 2024-01-01 \
    --end 2025-12-31
```

**Load symbols from a file and export results:**

```bash
python -m ai_trade.backtest.runner \
    --symbols-file watchlist.txt \
    --start 2024-06-01 \
    --end 2025-06-01 \
    --show-trades \
    --export results/my_backtest
```

The `ai-trade-backtest` console entry point also works if the package is installed:

```bash
ai-trade-backtest --symbols AAPL MSFT TSLA --days 180
ai-trade-backtest --default-universe --start 2025-01-01 --end 2025-12-31 --options
```

### Full CLI Reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--symbols` | list of strings | none | Stock tickers to backtest |
| `--symbols-file` | file path | none | Text file with one symbol per line |
| `--default-universe` | flag | off | Use built-in list of 24 liquid stocks |
| `--days` | integer | 90 | Calendar days to look back (ignored if `--start` is set) |
| `--start` | YYYY-MM-DD | computed from `--days` | Start date for the backtest period |
| `--end` | YYYY-MM-DD | today | End date for the backtest period |
| `--config` | file path | default location | Path to `settings.yaml` configuration file |
| `--options` | flag | off | Include options strategies (uses Black-Scholes synthetic pricing) |
| `--show-trades` | flag | off | Print every individual trade to the console |
| `--export` | file path | none | Export results to CSV (provide base filename; generates `.trades.csv`, `.options_trades.csv`, `.equity.csv`) |

### Configuration

The backtester reads strategy parameters and risk settings from `settings.yaml`. Key settings that affect backtesting:

- `account.starting_capital` -- initial cash (default $500)
- `account.max_position_pct` -- max % of equity in a single position (default 30%)
- `account.max_risk_per_trade_pct` -- max % of equity risked per trade (default 2%)
- `account.max_open_positions` -- concurrent position limit (default 4)
- `strategies.momentum.enabled`, `strategies.mean_reversion.enabled`, `strategies.vwap.enabled` -- toggle stock strategies
- `options.enabled` -- master toggle for options strategies
- Individual options strategy toggles (`strategies.long_call.enabled`, etc.)

### Output Metrics

After the simulation completes, the engine prints a summary table:

| Metric | What It Means |
|--------|--------------|
| **Total Return** | (Ending equity - Starting capital) / Starting capital, as a percentage |
| **Total Trades** | Number of completed round trips (entry + exit), broken down by stocks vs options |
| **Win Rate** | Percentage of trades that were profitable |
| **Avg Win / Avg Loss** | Average dollar P&L on winning vs losing trades |
| **Expectancy** | Average P&L per trade (positive = profitable system) |
| **Profit Factor** | Gross profits / Gross losses. Above 1.0 = profitable. Above 2.0 = strong. |
| **Sharpe Ratio** | Risk-adjusted return (annualized). Measures return per unit of volatility. |
| **Max Drawdown** | Largest peak-to-trough decline in equity, as a percentage (with duration in days) |
| **Max Win / Max Loss** | Best and worst single trade |
| **Win/Loss Streak** | Longest consecutive run of wins or losses |
| **Avg Holding Period** | Average number of days a position was held |
| **Strategy Breakdown** | Per-strategy stats: trade count, win rate, total P&L, average P&L |

### Exported Files

When using `--export results/my_backtest`, three CSV files are generated:

- `my_backtest.trades.csv` -- every stock trade with entry/exit dates, prices, P&L, strategy, exit reason
- `my_backtest.options_trades.csv` -- every options trade with strikes, contracts, entry cost, P&L, exit reason
- `my_backtest.equity.csv` -- daily equity curve (date, equity, cash, open positions, open options)

---

## Caveats and Limitations

Backtesting is a useful tool, but it is **not a crystal ball**. Every backtest has biases and simplifications that make its results more optimistic than real trading would be. Understand these limitations before trusting any results.

### 1. Survivorship Bias

The backtester only tests stocks that **exist today**. Companies that went bankrupt, were delisted, or were acquired during the test period are not included. Since these are typically the worst performers, the backtest universe is biased toward survivors -- making results look better than they would have been in real time.

### 2. No Slippage Simulation (Partial)

The engine applies a small slippage factor (0.1% by default) to entry and exit fills. However, real slippage depends on order size, bid/ask spread, market conditions, and timing. During volatile moments (earnings, market crashes), slippage can be dramatically larger than 0.1%.

### 3. Synthetic Options Pricing

Real options markets have supply/demand dynamics, market maker behavior, volatility skew, and term structure effects that Black-Scholes does not capture. The synthetic chains use a flat volatility surface (the same historical volatility for all strikes and expirations), while real options exhibit a "volatility smile" where out-of-the-money options trade at higher implied volatilities. This means:

- The backtester may underestimate the cost of OTM options
- Bid/ask spreads are simulated but do not reflect actual liquidity conditions
- Implied volatility in real markets can spike or collapse independently of realized volatility

### 4. No Commissions (Mostly)

Alpaca is commission-free for stocks and options, so `commission_per_trade` defaults to 0. However, there are still regulatory fees (SEC fee, TAF fee, options regulatory fee) that are not modeled. These are small per trade but compound over hundreds of trades.

### 5. Look-Ahead Bias Prevention

The engine is careful to avoid look-ahead bias: `_bars_up_to()` only returns data up to and including the current simulated day. Strategies never see tomorrow's prices. However, some subtle forms of look-ahead bias can still creep in:

- Indicator calculations use the day's close, but in real trading you don't know the close until 4:00 PM
- The simulated fill at the close price assumes you could have placed the order at the right moment

### 6. Volume and Liquidity

The backtester does not check whether the simulated trade volume is realistic relative to the stock's actual volume. In reality, very low-volume stocks may not fill your order at the expected price (or at all). The backtester assumes every order fills.

### 7. Market Impact

The bot's orders are small enough that they would not move markets in practice, and this is not modeled. This is a reasonable assumption for a $500-$20,000 account trading liquid stocks.

### 8. PDT Rule Simplification

The backtester tracks day trades in a rolling 5-business-day window and enforces the 3-day-trade limit, matching the real Pattern Day Trader rule. However, the exact counting can differ slightly from how a real broker calculates it (which includes after-hours fills, partial fills, etc.).

### 9. Options Assignment Risk

The backtester closes options positions 1 day before expiration to avoid assignment. In real trading, short options can be assigned at any time (American-style options), and early assignment risk is not modeled.

---

## Interpreting Results

Raw numbers can be misleading without context. Here is how to think about the key metrics:

### Win Rate Is Not Everything

A 40% win rate with a 3:1 reward-to-risk ratio is profitable:
- 40 wins * $3 average gain = $120
- 60 losses * $1 average loss = $60
- Net: +$60

A 90% win rate with a 1:10 reward-to-risk is a disaster:
- 90 wins * $1 average gain = $90
- 10 losses * $10 average loss = $100
- Net: -$10

**What to look at instead**: Expectancy (average P&L per trade) and profit factor (gross wins / gross losses) give you the full picture.

### Sharpe Ratio

The Sharpe ratio measures **risk-adjusted return** -- how much return you get per unit of volatility.

| Sharpe | Interpretation |
|--------|---------------|
| < 0 | Losing money |
| 0 - 0.5 | Poor risk-adjusted returns |
| 0.5 - 1.0 | Acceptable |
| 1.0 - 2.0 | Good |
| > 2.0 | Excellent (but verify -- may indicate overfitting) |

A very high Sharpe ratio (3.0+) in a backtest should raise suspicion. It often means the strategy is overfit to the test period and will not generalize.

### Maximum Drawdown

Max drawdown is the largest peak-to-trough decline in your equity. If your portfolio grew from $500 to $2,000, then dropped to $1,200 before recovering, the max drawdown is ($2,000 - $1,200) / $2,000 = **40%**.

This is the **emotional tolerance test**. Ask yourself: "Could I watch 40% of my account value evaporate without panic-selling?" If the answer is no, the strategy is too aggressive for you, regardless of its total return.

### Profit Factor

| Profit Factor | Interpretation |
|--------------|---------------|
| < 1.0 | Losing money (gross losses exceed gross profits) |
| 1.0 - 1.5 | Marginal -- barely profitable |
| 1.5 - 2.0 | Solid |
| > 2.0 | Strong |

### Strategy Breakdown

The per-strategy breakdown in the output helps you identify which strategies are carrying the portfolio and which are dragging it down. If one strategy has a negative total P&L, consider disabling it in `settings.yaml`.

---

## Sample Results

From a 2-year backtest (January 2024 through December 2025) using the default universe of 24 stocks with $500 starting capital, stock + options strategies enabled:

```
============================================================
  BACKTEST RESULTS
============================================================
  Period:         2024-01-02 to 2025-12-31
  Starting:       $500.00
  Ending:         $18,513.00
  Total Return:   +3,602.60%
------------------------------------------------------------
  Total Trades:   826  (stocks: 614, options: 212)
  Win Rate:       83.3%
  Avg Win:        $38.12
  Avg Loss:       $12.45
  Expectancy:     $21.83 per trade
------------------------------------------------------------
  Sharpe Ratio:   2.14
  Profit Factor:  3.05
  Max Drawdown:   18.2%  (12 days)
------------------------------------------------------------
```

**These are SIMULATED results.** Real trading will differ due to all the caveats listed above. The backtest uses perfect hindsight for volatility estimation, doesn't account for real options market dynamics, and benefits from survivorship bias. Treat these numbers as a **directional indicator** ("this strategy has an edge") rather than a **forecast** ("I will make 3,600% returns").

A reasonable expectation: if the backtest shows strong results, the live strategy will likely be profitable but with lower returns, lower win rate, and larger drawdowns than the simulation suggests.
