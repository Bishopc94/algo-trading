# AI Trade Bot -- Architecture Document

> **Audience**: Software engineers who may not be familiar with Python.
> Python-specific constructs are explained inline where they first appear.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Scanner](#1-scanner)
4. [Strategies](#2-strategies)
5. [Signal Aggregator](#3-signal-aggregator)
6. [Sentiment Layer](#4-sentiment-layer)
7. [Risk Manager](#5-risk-manager)
8. [Order Execution](#6-order-execution)
9. [PDT Manager](#7-pdt-manager)
10. [Data Flow](#data-flow)
11. [Persistence Layer](#persistence-layer)
12. [Configuration System](#configuration-system)
13. [Scheduling](#scheduling)
14. [Client Architecture](#client-architecture)
15. [Logging](#logging)
16. [Catch-up Logic](#catch-up-logic)

---

## System Overview

AI Trade is a **fully automated stock and options trading system** that runs
unattended during US market hours (9:00 AM -- 4:05 PM Eastern). It connects
to [Alpaca](https://alpaca.markets/) as its broker, currently operating in
**paper trading** mode with a **$500** account.

At a high level, the bot:

1. Scans the entire US equity universe each morning for tradeable candidates.
2. Evaluates 11 strategies (3 stock, 8 options) against those candidates.
3. Ranks the resulting signals, checks them against risk limits and PDT rules,
   and builds an execution queue.
4. Submits orders to Alpaca, then monitors and reconciles positions throughout
   the day.
5. Force-closes day trades before market close and logs a daily performance
   snapshot.

### Key Design Principles

| Principle | Implementation |
|---|---|
| Safety first | Every trade passes through a multi-gate risk manager before execution. |
| PDT awareness | A dedicated PDT manager budgets day trades (max 3 / 5 rolling business days). |
| Regime awareness | A market regime analyzer (SPY/QQQ/VIX) gates *all* new entries. |
| Mechanical exits | Bracket orders with server-side stop-loss and take-profit -- no manual intervention. |
| Observability | Structured logging (JSON + console), SQLite audit trail, daily snapshots. |
| Dry-run mode | `--dry-run` flag logs every signal without submitting a single order. |

---

## Pipeline Architecture

```
                        +---------------------------------------------+
                        |             Market Data (Alpaca)             |
                        +-----+-------------------+-------------------+
                              |                   |
                              v                   v
                        +-----------+     +-----------------+
                        |  Scanner  |     | Historical /    |
                        | screener  |     | Streaming Data  |
                        +-----+-----+     +--------+--------+
                              |                    |
                              v                    v
                     +------------------+   +--------------+
                     |  11 Strategies   |   |  Indicators  |
                     |  (stock+options) |   |  (RSI, EMA,  |
                     +--------+---------+   |  VWAP, ATR,  |
                              |             |  BB, MACD)   |
                              v             +--------------+
                   +--------------------+
                   | Signal Aggregator  |<--------- Sentiment Layer
                   | (the "brain")      |           (Regime + News)
                   +--------+-----------+
                            |
              +-------------+-------------+
              |                           |
              v                           v
      +---------------+          +---------------+
      |  PDT Manager  |          | Risk Manager  |
      | (day-trade    |          | (loss limits, |
      |  budget)      |          |  heat, conc.) |
      +-------+-------+          +-------+-------+
              |                           |
              +-------------+-------------+
                            |
                            v
                 +--------------------+
                 | Order Execution    |
                 | (bracket / limit / |
                 |  multi-leg)        |
                 +---------+----------+
                           |
                           v
                 +--------------------+
                 |    Alpaca API      |
                 +---------+----------+
                           |
                           v
                 +--------------------+         +------------------+
                 | Position Sync      |-------->| Performance      |
                 | (every 60 seconds) |         | Tracker          |
                 +--------------------+         +--------+---------+
                                                         |
                                                         v
                                                 +---------------+
                                                 |  SQLite DB    |
                                                 | (6 tables)    |
                                                 +---------------+
```

---

## 1. Scanner

**File**: `src/ai_trade/scanner/screener.py`
**Class**: `StockScreener`

The scanner is the first stage of the pipeline. Every morning (9:00 AM ET), it
identifies which stocks out of the entire US equity universe are worth
evaluating that day.

### Universe Loading

```python
request = GetAssetsRequest(
    asset_class=AssetClass.US_EQUITY,
    status=AssetStatus.ACTIVE,
)
assets = get_trading_client().get_all_assets(filter=request)
```

The scanner calls Alpaca's asset listing endpoint once, then caches the
result for the lifetime of the `StockScreener` instance (stored in
`self._universe`). Only symbols that are `tradable` and listed on
NYSE, NASDAQ, or AMEX are kept.

### Snapshot Fetching

Because the Alpaca snapshot API has per-request limits, the scanner
fetches snapshots in **batches of 500** symbols:

```python
for i in range(0, len(universe), _SNAPSHOT_BATCH_SIZE):
    batch = universe[i : i + _SNAPSHOT_BATCH_SIZE]
    snapshots = fetch_snapshots(batch)
```

Each snapshot contains the latest trade price, previous daily bar, and
current daily bar -- everything needed for gap and volume calculations.

### Filter Funnel

Every snapshot passes through a series of filters. The scanner tracks a
`filter_counts` dictionary for diagnostics so you can see exactly how many
symbols were eliminated at each stage.

| Filter | Condition | Default |
|---|---|---|
| Price range | `min_price <= price <= max_price` | $2 -- $50 |
| Gap % | `abs(gap_pct) >= min_gap_pct` | >= 2% |
| Relative volume | `rvol >= min_relative_volume` | >= 1.5x |

### Scoring Formula

Candidates that survive all filters are scored and ranked:

```
score = abs(gap_pct) * 0.4 + relative_volume * 0.3 + 10.0 * 0.3
```

| Component | Weight | Rationale |
|---|---|---|
| `abs(gap_pct)` | 40% | Larger gap = stronger catalyst |
| `relative_volume` | 30% | Volume confirms institutional interest |
| `10.0` (ADR placeholder) | 30% | Currently a constant; reserved for Average Daily Range |

The top **20** candidates (configurable via `max_candidates`) are returned
sorted by score descending.

---

## 2. Strategies

The bot runs **11 strategies** total: 3 stock strategies evaluated by the
`SignalAggregator`, and 8 options strategies evaluated in a separate
`_evaluate_options()` path.

### Strategy Base Classes

> **Python concept -- ABC (Abstract Base Class)**:
> In Python, `ABC` is a standard-library class that lets you define
> *abstract methods* -- methods that subclasses **must** implement.
> If a subclass forgets to implement an abstract method, Python raises
> a `TypeError` at instantiation time, catching the bug early.

> **Python concept -- @dataclass**:
> The `@dataclass` decorator automatically generates `__init__`,
> `__repr__`, and `__eq__` methods from class-level type annotations.
> It is similar to a struct/record in other languages. The `field()`
> function lets you set default factories (e.g., `default_factory=dict`
> creates a new empty dict for each instance, avoiding a shared-
> mutable-default bug).

**Stock strategies** inherit from `BaseStrategy` (`strategy/base.py`):

```python
class BaseStrategy(ABC):
    @abstractmethod
    def evaluate(self, symbol, daily_bars, intraday_bars=None) -> Signal | None:
        """Return a Signal if entry conditions met, else None."""

    @abstractmethod
    def should_exit(self, symbol, bars, entry_price) -> bool:
        """Check if exit conditions are met for an existing position."""
```

Every strategy produces a `Signal` dataclass:

```python
@dataclass
class Signal:
    symbol: str
    direction: str          # always "long" for a cash account
    conviction: float       # 0.0 to 1.0
    strategy_name: str
    hold_type: HoldType     # DAY, SWING, or ADAPTIVE
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    metadata: dict = field(default_factory=dict)
```

**Options strategies** inherit from `BaseOptionsStrategy`
(`strategy/options/base.py`) and produce an `OptionsSignal` dataclass that
includes legs, max cost/credit, max loss/profit, greeks, and contract details.

### Stock Strategies

#### Mean Reversion (Swing)

**File**: `src/ai_trade/strategy/mean_reversion.py`

| Aspect | Detail |
|---|---|
| Hold type | SWING (no PDT cost) |
| Core idea | Buy oversold dips in a short-term uptrend, targeting mean reversion |
| Entry conditions | RSI < 40 **AND** close >= 98% of 20-EMA **AND** close <= 103% of lower Bollinger Band |
| Stop loss | Entry - 1.5x ATR(14) |
| Take profit | Entry + 3.0x ATR(14) -- a 2:1 reward-to-risk |
| Exit override | RSI recovers above 60 **OR** price touches upper Bollinger Band |
| Conviction | Linear from 0.5 (at RSI threshold) to 1.0 (at RSI = 20) |

#### Momentum (Adaptive)

**File**: `src/ai_trade/strategy/momentum.py`

| Aspect | Detail |
|---|---|
| Hold type | ADAPTIVE -- defaults to SWING, switches to DAY if conviction >= 0.9 |
| Core idea | Enter on volume-confirmed breakouts above the 20-day high |
| Entry conditions | Close > 20-day rolling high **AND** relative volume > 1.5x **AND** ADR% > 2% **AND** close > 20-EMA |
| Stop loss | Entry - 1.5x ATR(14) |
| Take profit | Entry + 3.0x ATR(14) |
| Exit | Entirely mechanical -- bracket orders only; `should_exit()` returns `False` |
| Conviction | Linear interpolation: 1.5x rvol = 0.5, 3x = 0.75, 5x+ = 1.0 |

#### VWAP Reclaim (Day Trade)

**File**: `src/ai_trade/strategy/vwap.py`

| Aspect | Detail |
|---|---|
| Hold type | DAY (costs a PDT slot) |
| Core idea | Enter when price reclaims VWAP from below on elevated intraday volume |
| Entry conditions | Recent bars dipped below VWAP **AND** current bar closes above VWAP **AND** bar volume > 1.5x average |
| Stop loss | Max of (dip low, VWAP - 1%) |
| Take profit | Max of (VWAP + exit deviation %, prior intraday high) |
| Conviction | Base 0.7, adjusted +/- for dip depth and volume strength |

### Options Strategies

All 8 options strategies inherit from `BaseOptionsStrategy` and share utility
functions from `strategy/options/base.py` for contract filtering, delta
selection, and greek enrichment.

| # | Strategy | File | Type | Key Idea |
|---|---|---|---|---|
| 1 | Credit Put Spread | `credit_put_spread.py` | Credit | Sell OTM put spread for premium; bullish bias |
| 2 | Debit Call Spread | `debit_call_spread.py` | Debit | Buy call spread for defined-risk bullish exposure |
| 3 | Long Call | `long_call.py` | Debit | Directional bullish bet with capped risk |
| 4 | Long Put | `long_put.py` | Debit | Directional bearish bet with capped risk |
| 5 | Cash Secured Put | `cash_secured_put.py` | Credit | Sell put with cash collateral; neutral-to-bullish |
| 6 | Covered Call | `covered_call.py` | Credit | Sell call against stock position; income strategy |
| 7 | Covered Straddle | `covered_straddle.py` | Credit | Sell call + put against stock; high-premium income |
| 8 | Momentum Options | `momentum_options.py` | Debit | Options-based momentum (debit calls on breakouts) |

Options orders go through a separate risk budget:
- Max options positions: configurable (default 3)
- Max capital allocated to options: configurable % of equity (default 40%)
- Max single-trade risk: configurable dollar cap (default $100)

---

## 3. Signal Aggregator

**File**: `src/ai_trade/strategy/signal.py`
**Class**: `SignalAggregator`

The signal aggregator is the "brain" of the system. It collects signals from
all enabled strategies, ranks them by priority, and builds an execution
queue subject to risk and PDT constraints.

### Algorithm

```
1.  FOR each candidate symbol:
        FOR each enabled strategy:
            signal = strategy.evaluate(symbol, daily_bars, intraday_bars)
            IF signal is not None:
                collect it + log to database

2.  SEPARATE signals into two buckets:
        swing_signals  (hold_type == SWING)
        day_signals    (hold_type == DAY or ADAPTIVE)

3.  SORT swing_signals by conviction descending
    (swing trades are free -- no PDT cost -- so we process them first)

4.  FILTER day_signals:
        IF pdt_manager.can_day_trade() == False:
            discard ALL day signals
        ELSE:
            keep only signals where conviction >= 0.80

5.  SORT qualifying day signals by conviction descending

6.  MERGE: ranked = swing_signals + qualifying_day_signals

7.  BUILD execution queue (top-down):
        FOR each signal in ranked:
            IF remaining_cash <= 0 OR open_count >= max_positions: BREAK
            shares = position_sizer.calculate_shares(signal, equity, cash)
            IF shares <= 0: SKIP
            approved, reason = risk_manager.approve_trade(...)
            IF NOT approved: SKIP
            ADD {signal, shares} to execution queue
            DEDUCT cost from remaining_cash
            INCREMENT open_count

8.  RETURN execution queue
```

### Priority System

The ordering ensures optimal use of a small account:

1. **Swing signals first** -- They do not consume a PDT slot, so they are
   essentially "free" to enter. Sorted by conviction descending, the
   highest-confidence swing trade gets first priority on available cash.

2. **Day signals second** -- Only evaluated if the PDT manager has budget
   and the signal has conviction >= 0.80 (configurable). This high bar
   prevents wasting a scarce PDT slot on a marginal setup.

3. **Cash-limited top-down** -- The queue is built greedily: each approved
   trade reduces `remaining_cash`, so lower-priority signals may be skipped
   simply because funds ran out.

---

## 4. Sentiment Layer

The sentiment layer modifies strategy decisions based on broader market
conditions and stock-specific news. It operates as a **modifier**, not a
signal generator -- it adjusts conviction scores and can block trades
entirely.

### Market Regime Analyzer

**File**: `src/ai_trade/sentiment/market_regime.py`
**Class**: `MarketRegimeAnalyzer`

Runs once at market open (9:30 AM). Fetches 200+ days of SPY, QQQ, and VIX
daily bars, then computes a **breadth score** from 7 components:

| # | Component | Bullish | Bearish | Weight |
|---|---|---|---|---|
| 1 | SPY trend (EMA stack) | Above 20/50/200 EMA | Below all three | +/- 1.0 |
| 2 | QQQ trend | Above 20/50 EMA | Below both | +/- 1.0 |
| 3 | SPY RSI(14) | > 60 | < 40 | +/- 0.5 |
| 4 | VIX level | < 15 | > 30 | +/- 1.0 |
| 5 | VIX trend (5-day) | Falling > 10% | Rising > 10% | +/- 0.5 |
| 6 | SPY MACD histogram | Positive | Negative | +/- 0.5 |
| 7 | SPY 5-day return | > +2% | < -2% | +/- 0.5 |

**breadth_score** = sum of points / component count (normalized to [-1.0, +1.0])

Override rule: if VIX > 30, any Strong Bull or Bull regime is demoted to
Neutral.

The breadth score maps to one of **5 regimes**:

| Regime | Breadth Range | Conviction Modifier | Position Size Modifier | Allow Longs | Allow Options |
|---|---|---|---|---|---|
| Strong Bull | >= 0.6 | 1.30x | 1.00x | Yes | Yes |
| Bull | >= 0.2 | 1.10x | 1.00x | Yes | Yes |
| Neutral | >= -0.2 | 0.90x | 0.75x | Yes | Yes |
| Bear | >= -0.6 | 0.60x | 0.50x | Yes (high conv. only) | No |
| Strong Bear | < -0.6 | 0.30x | 0.25x | **No** | No |

> **Python concept -- @dataclass**:
> The `MarketContext` result is a `@dataclass` containing the regime enum,
> all modifiers, underlying indicator values, and a human-readable summary
> string. Dataclasses provide a clean way to bundle related data without
> writing boilerplate constructor code.

### News Sentiment Scanner

**File**: `src/ai_trade/sentiment/news_sentiment.py`
**Class**: `NewsSentimentScanner`

Fetches up to 10 articles per symbol from Alpaca's free news API (last 24
hours). Scores each article using two keyword dictionaries (~35 bullish
keywords, ~35 bearish keywords), each with a float weight (0.5 -- 3.0).

**Scoring process:**

1. Concatenate headline + summary, lowercase.
2. Sum matched keyword weights for bullish and bearish separately.
3. Apply **recency weighting** -- recent articles count more:
   `weight = max(0.3, 1.0 - age_hours / (lookback * 2))`
4. Normalize: `net_score = (bull - bear) / max(bull + bear, 1.0)`, clamped
   to [-1, +1].
5. Detect **catalyst**: if total bull > 4.0 or total bear > 4.0, flag as
   strong catalyst.

**Conviction modifier:**

| Net Score | Modifier | Effect |
|---|---|---|
| > +0.3 | 1.0 + min(score * 0.5, 0.3) | Up to +30% conviction boost |
| < -0.3 | 1.0 + max(score * 0.5, -0.5) | Up to -50% conviction penalty |
| -0.3 to +0.3 | 1.0 | No change |
| Catalyst + bullish | Additional 1.2x multiplier (capped at 1.5) | Strong boost |
| Catalyst + bearish | Additional 0.8x multiplier (floor at 0.3) | Strong penalty |

**Hard block**: If `net_score < -0.5` and no catalyst is detected, the trade
is blocked entirely.

---

## 5. Risk Manager

**File**: `src/ai_trade/risk/risk_manager.py`
**Class**: `RiskManager`

The risk manager is the final gate every trade must pass through. It runs
four independent checks; if **any** check fails, the trade is rejected.

### Risk Checks

| Check | Method | Limit | How It Works |
|---|---|---|---|
| Daily loss limit | `check_daily_loss_limit()` | 5% of starting equity | Compares current equity to opening equity cached at 9:30 AM. If drawdown exceeds 5%, all trading halts for the day. |
| Position concentration | `check_concentration()` | Max 4 open positions | Simple count check against `max_open_positions`. |
| Portfolio heat | `check_portfolio_heat()` | 6% of equity | Sums `abs(entry - stop) * shares` for all open trades. If total risk as a % of equity exceeds 6%, no new trades. |
| Affordability | inline in `approve_trade()` | `shares * entry <= cash` | Ensures the account can actually pay for the trade. |

### Position Sizing

**File**: `src/ai_trade/risk/position_sizer.py`
**Class**: `PositionSizer`

Uses **fixed-fractional risk sizing**:

```
risk_budget      = equity * max_risk_per_trade_pct   (default 2%)
risk_per_share   = abs(entry_price - stop_loss_price)
shares           = floor(risk_budget / risk_per_share)
```

Then clamped by:
- **Concentration limit**: `shares * entry <= equity * max_position_pct` (default 30%)
- **Cash limit**: `shares * entry <= available_cash`
- **Minimum**: If cash allows at least 1 share, always buy at least 1

---

## 6. Order Execution

### Stock Orders

**File**: `src/ai_trade/execution/order_manager.py`
**Class**: `OrderManager`

Stock entries use **bracket orders** -- a single API call that creates three
linked orders:

```python
request = MarketOrderRequest(
    symbol=signal.symbol,
    qty=shares,
    side=OrderSide.BUY,
    time_in_force=tif,              # DAY or GTC depending on hold_type
    order_class=OrderClass.BRACKET,
    stop_loss=StopLossRequest(stop_price=...),
    take_profit=TakeProfitRequest(limit_price=...),
)
```

| Order | Type | Purpose |
|---|---|---|
| Parent | Market | Immediate fill at market price |
| Stop loss | Stop (server-side) | Automatic risk management -- Alpaca holds and triggers this |
| Take profit | Limit (server-side) | Automatic profit capture |

This design means the bot does **not** need to be running for stops and
targets to execute -- they live on Alpaca's servers.

### EOD Force-Close

At **3:50 PM ET**, `job_eod_close_day_trades()` iterates all open trades in
the database with `hold_type == "day"` and `status == "open"`, calling
`close_position()` for each. This ensures no day trades accidentally become
overnight holds (which would violate PDT tracking).

### Position Reconciliation

Every **60 seconds**, `sync_positions()` compares Alpaca's live positions
against the local database:

- **Untracked position** (on Alpaca but not in DB): Logged as a warning.
- **Stale trade** (in DB as "open" but no matching Alpaca position): Marked
  as "closed" in the DB. This happens when a bracket order's stop-loss or
  take-profit fills while the bot is running.

### Options Orders

**File**: `src/ai_trade/execution/options_order_manager.py`
**Class**: `OptionsOrderManager`

Options use **limit orders** (never market orders -- options spreads can have
wide bid-ask spreads). The manager routes based on leg count:

| Legs | Method | Order Type |
|---|---|---|
| 1 (e.g., long call) | `submit_single_leg_order()` | Limit order on a single option contract |
| 2+ (e.g., credit spread) | `submit_spread_order()` | Multi-leg order (`order_class="mleg"`) with `OptionLegRequest` legs |

Each leg specifies a `PositionIntent` (buy_to_open, sell_to_open, etc.)
mapped from string to the Alpaca SDK enum.

The manager also handles **expiration safety**: `close_expiring_positions()`
parses the OCC symbol format to extract expiration dates and closes any
position expiring within N days.

---

## 7. PDT Manager

**File**: `src/ai_trade/risk/pdt_manager.py`
**Class**: `PDTManager`

The Pattern Day Trader (PDT) rule limits accounts under $25,000 to **3 day
trades per 5 rolling business days**. A "day trade" is any round trip
(buy + sell of the same security) on the same day.

### Budget System

```
Total budget:   3 day trades per 5 business days
Reserve:        1 (for emergency exits)
Available:      3 - 1 = 2 usable day trades
```

The `can_day_trade()` method queries the database for day trades in the last
5 business days (weekends excluded) and checks:

```python
allowed = used < max_trades - reserve   # i.e., used < 2
```

### Rolling Window

The 5-business-day window is computed by `_five_business_days_ago()`, which
walks backwards from today, skipping Saturday (weekday 5) and Sunday
(weekday 6):

```python
@staticmethod
def _five_business_days_ago() -> date:
    today = date.today()
    biz_days = 0
    cursor = today
    while biz_days < 5:
        cursor -= timedelta(days=1)
        if cursor.weekday() < 5:  # Mon=0 ... Fri=4
            biz_days += 1
    return cursor
```

> **Python concept -- @staticmethod**:
> A `@staticmethod` is a method that belongs to a class but does not
> receive the instance (`self`) or the class (`cls`) as a first argument.
> It is essentially a plain function namespaced under the class. It is used
> here because the calculation does not depend on any instance state.

### Hold Type Classification

The `would_be_day_trade()` method determines whether a signal's hold type
will consume a PDT slot:

```python
@staticmethod
def would_be_day_trade(hold_type: HoldType) -> bool:
    return hold_type in (HoldType.DAY, HoldType.ADAPTIVE)
```

`SWING` trades are not day trades (they are held overnight or longer).
`ADAPTIVE` trades *might* close same-day, so they are budgeted as day trades.

---

## Data Flow

### Historical Data

**File**: `src/ai_trade/data/historical.py`

| Function | Purpose | Returns |
|---|---|---|
| `fetch_bars(symbol, timeframe, start, end)` | Single-symbol OHLCV bars | `pd.DataFrame` |
| `fetch_bars_multi(symbols, timeframe, start, end)` | Multi-symbol OHLCV bars in one API call | `dict[str, pd.DataFrame]` |
| `fetch_snapshots(symbols)` | Current price + prev close + volume | Raw snapshot dict |
| `fetch_latest_quotes(symbols)` | Latest bid/ask quotes | Raw quote dict |

All functions use the **IEX** data feed (free tier) and include **retry
logic** with exponential backoff for API rate limits (via `retry_api_call`
from `utils.py`).

The multi-index handling is worth noting: Alpaca returns a pandas
`MultiIndex` DataFrame keyed by `(symbol, timestamp)`. The code uses
`df.xs(symbol, level="symbol")` to extract each symbol's slice.

### Streaming Data

**File**: `src/ai_trade/data/streaming.py`
**Class**: `StreamManager`

Provides real-time minute bars via WebSocket. The stream runs in a **daemon
thread** -- a background thread that automatically dies when the main
program exits.

> **Python concept -- daemon thread**:
> `threading.Thread(target=..., daemon=True)` creates a thread that the
> Python runtime will forcibly terminate when the main thread exits.
> Non-daemon threads would keep the process alive even after `main()`
> returns. The daemon flag is appropriate here because we want the stream
> to die cleanly with the bot.

The manager uses an async handler internally (Alpaca's SDK is async for
streaming), but exposes a synchronous callback API:

```python
stream_manager.subscribe(
    symbols=["AAPL", "TSLA"],
    on_bar=lambda symbol, bar: print(f"{symbol}: {bar['close']}"),
)
stream_manager.start()  # launches daemon thread
```

### Technical Indicators

**File**: `src/ai_trade/data/indicators.py`

All indicator functions follow the same pattern: take a DataFrame, add
columns, return it. They are **idempotent** -- if the target column already
exists, the function skips computation.

| Function | Output Columns | Library |
|---|---|---|
| `add_rsi(df, period=14)` | `rsi_14` | `ta.momentum.RSIIndicator` |
| `add_ema(df, periods=[9,20,50])` | `ema_9`, `ema_20`, `ema_50` | pandas `.ewm()` |
| `add_vwap(df)` | `vwap_calc` | Custom (cumulative TP*V / cumV, daily reset) |
| `add_atr(df, period=14)` | `atr_14` | `ta.volatility.AverageTrueRange` |
| `add_bollinger(df, period=20, std=2)` | `bb_upper`, `bb_middle`, `bb_lower`, `bb_width` | `ta.volatility.BollingerBands` |
| `add_macd(df)` | `macd`, `macd_signal`, `macd_hist` | `ta.trend.MACD` |
| `add_volume_profile(df, period=20)` | `relative_volume` | `volume / rolling_mean(volume)` |
| `compute_adr(df, period=14)` | Returns `float` (not a column) | `mean((high-low)/close) * 100` |
| `add_all(df, intraday=False)` | All of the above (VWAP only if `intraday=True`) | Convenience wrapper |

### Options Chain

**File**: `src/ai_trade/data/options_chain.py`

| Function | Purpose |
|---|---|
| `get_options_chain(symbol)` | Fetch available option contracts for a symbol |
| `get_options_snapshot(option_symbols)` | Fetch greeks, bid/ask, and IV for specific contracts |

---

## Persistence Layer

**File**: `src/ai_trade/monitoring/database.py`
**Class**: `Database`

SQLite database stored at `data/ai_trade.db` (relative to project root).

### Schema (6 Tables)

| Table | Purpose | Key Columns |
|---|---|---|
| `trades` | Stock trade lifecycle | symbol, strategy, side, shares, entry/exit price+time, stop_loss, take_profit, hold_type, pnl, status |
| `day_trades` | PDT tracking | symbol, trade_date, buy/sell order IDs |
| `daily_snapshots` | EOD equity curve | date (unique), equity, cash, open_positions, day_trades_used, realized/unrealized PnL |
| `signals` | Signal audit log | timestamp, symbol, strategy, conviction, hold_type, direction |
| `options_trades` | Options trade lifecycle | underlying, strategy, legs (JSON), qty, credits/debits, max_loss/profit, greeks, status |
| `scanner_results` | Scanner output archive | date, symbol, price, gap_pct, relative_volume, score, selected |

### SQL Injection Prevention

Column names are validated via regex before being interpolated into SQL:

```python
_COL_RE = re.compile(r"^[a-z_][a-z0-9_]*$", re.IGNORECASE)

def _validate_columns(columns: list[str]) -> None:
    for col in columns:
        if not _COL_RE.match(col):
            raise ValueError(f"Invalid column name: {col!r}")
```

All **values** use parameterized queries (`?` placeholders), which is the
standard protection against SQL injection. The column-name validation adds
a second layer of defense for the dynamically constructed `INSERT` and
`UPDATE` statements.

### Generic CRUD Methods

> **Python concept -- @contextmanager**:
> The `@contextmanager` decorator (from `contextlib`) turns a generator
> function into a context manager -- an object usable with Python's `with`
> statement. The code before `yield` runs on entry, the code after `yield`
> runs on exit (even if an exception occurs). Here it manages the SQLite
> connection lifecycle:

```python
@contextmanager
def _conn(self):
    conn = sqlite3.connect(str(self._path))
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    try:
        yield conn       # <-- caller uses connection here
        conn.commit()    # auto-commit on success
    finally:
        conn.close()     # always close, even on error
```

Three generic methods handle all writes:

| Method | SQL Pattern |
|---|---|
| `_insert(table, **kwargs)` | `INSERT INTO {table} ({cols}) VALUES (?, ?, ...)` |
| `_update(table, row_id, **kwargs)` | `UPDATE {table} SET col=?, ... WHERE id=?` |
| `_upsert(table, **kwargs)` | `INSERT OR REPLACE INTO {table} ...` |

> **Python concept -- **kwargs**:
> `**kwargs` collects all keyword arguments into a dictionary. For example,
> `_insert("trades", symbol="AAPL", shares=10)` receives
> `kwargs = {"symbol": "AAPL", "shares": 10}`. This allows a single
> generic function to handle any combination of columns.

---

## Configuration System

### File Layout

```
config/
  settings.yaml    # All strategy parameters, risk limits, schedule times
  .env             # Secrets (ALPACA_API_KEY, ALPACA_SECRET_KEY)
  .env.example     # Template for .env
```

### Loading Process

**File**: `src/ai_trade/config.py`

1. Load `.env` file (using `python-dotenv`) to populate environment variables.
2. Parse `settings.yaml` with `yaml.safe_load()`.
3. Recursively convert the resulting dict to a `SimpleNamespace`.
4. Inject API keys from environment variables into the namespace.

### The SimpleNamespace Pattern

> **Python concept -- SimpleNamespace**:
> `SimpleNamespace` (from Python's `types` module) converts a dictionary
> into an object with dot-access attributes. Given
> `{"scanner": {"min_price": 2.0}}`, the recursive converter produces
> an object where you can write `cfg.scanner.min_price` instead of
> `cfg["scanner"]["min_price"]`.
>
> This is purely syntactic sugar -- it makes the code more readable and
> catches typos at runtime (accessing a non-existent attribute raises
> `AttributeError`, whereas dict access would silently return `None` with
> `.get()`).

```python
def _to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to a SimpleNamespace for dot-access."""
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _to_namespace(v)  # recurse into nested dicts
    return SimpleNamespace(**d)
```

**Example usage throughout the codebase:**

```python
# Instead of:
min_price = config["scanner"]["min_price"]

# You write:
min_price = config.scanner.min_price
```

### Secret Management

Secrets are **never** stored in YAML. The `.env` file is loaded at startup,
and keys are injected programmatically:

```python
cfg.alpaca.api_key = os.environ.get("ALPACA_API_KEY", "")
cfg.alpaca.secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
```

If either key is missing, the bot raises an `EnvironmentError` immediately.

---

## Scheduling

**File**: `src/ai_trade/scheduler/jobs.py`

Uses [APScheduler](https://apscheduler.readthedocs.io/) (Advanced Python
Scheduler) with a `BackgroundScheduler` that runs all jobs in the background
on separate threads.

### Job Schedule

All jobs run Monday -- Friday, Eastern Time:

| Time (ET) | Job ID | Method | Purpose |
|---|---|---|---|
| 9:00 AM | `premarket_scan` | `job_premarket_scan()` | Scan universe, build candidate list |
| 9:30 AM | `market_open` | `job_market_open()` | Cache starting equity, sync positions, analyze market regime |
| 9:35 AM | `entry_window` | `job_entry_window()` | Run all strategies, rank signals, submit trades |
| 12:00 PM | `midday_check` | `job_midday_check()` | Re-evaluate positions, look for new swing setups |
| 3:00 PM | `power_hour_scan` | `job_power_hour()` | Fresh scan + strategy evaluation for late momentum |
| 3:50 PM | `eod_close_day_trades` | `job_eod_close_day_trades()` | Force-close all day-trade positions |
| 4:05 PM | `eod_review` | `job_eod_review()` | Save daily snapshot, print P&L summary |

**Plus one interval job:**

| Interval | Job ID | Method | Purpose |
|---|---|---|---|
| Every minute (9 AM -- 3 PM) | `position_sync` | `job_sync_positions()` | Reconcile Alpaca positions with local DB |

### Cron Trigger Configuration

Schedule times are read from `settings.yaml` and parsed into hour/minute
pairs:

```python
scheduler.add_job(
    bot.job_premarket_scan,
    CronTrigger(hour=9, minute=0, day_of_week="mon-fri", timezone=ET),
    id="premarket_scan",
    name="Pre-market stock scan",
)
```

The position sync job uses a range expression (`hour="9-15"`) and
`minute="*"` to fire every minute during market hours.

---

## Client Architecture

**File**: `src/ai_trade/clients.py`

Uses the **Singleton pattern** via module-level variables to ensure only one
instance of each Alpaca client exists.

> **Python concept -- module-level singletons**:
> In Python, a module is loaded exactly once. Variables declared at the
> top of a module (`_trading_client = None`) act as global state for that
> module. The `init_clients()` function sets them once, and
> `get_trading_client()` / `get_data_client()` return the cached instances.
> This is a common Python alternative to the class-based Singleton pattern
> found in languages like Java or C#.

```python
# Module-level singletons
_trading_client: TradingClient | None = None
_data_client: StockHistoricalDataClient | None = None

def init_clients(cfg: SimpleNamespace) -> None:
    """Must be called once at startup."""
    global _trading_client, _data_client
    _trading_client = TradingClient(
        cfg.alpaca.api_key, cfg.alpaca.secret_key, paper=cfg.alpaca.paper
    )
    _data_client = StockHistoricalDataClient(
        cfg.alpaca.api_key, cfg.alpaca.secret_key
    )

def get_trading_client() -> TradingClient:
    if _trading_client is None:
        raise RuntimeError("Call init_clients(cfg) first")
    return _trading_client
```

> **Python concept -- global keyword**:
> Inside a function, Python treats any variable you assign to as *local*
> by default. The `global` keyword tells Python that `_trading_client`
> refers to the module-level variable, not a new local one.

Three client types are used:

| Client | SDK Class | Purpose |
|---|---|---|
| Trading | `TradingClient` | Submit orders, manage positions, query account |
| Historical Data | `StockHistoricalDataClient` | Fetch bars, snapshots, quotes |
| Streaming | `StockDataStream` | Real-time WebSocket minute bars (not cached -- one per connection) |

---

## Logging

**File**: `src/ai_trade/monitoring/logger.py`

Uses [structlog](https://www.structlog.org/) for structured, key-value
logging.

### Output Modes

The logger auto-detects the output environment:

```python
structlog.dev.ConsoleRenderer() if sys.stdout.isatty() else structlog.processors.JSONRenderer()
```

- **Terminal** (`isatty()` is True): Colored, human-readable output via
  `ConsoleRenderer`.
- **Piped / redirected** (`isatty()` is False): Machine-parseable JSON
  lines via `JSONRenderer`.

### File Handlers

| File | Level | Purpose |
|---|---|---|
| `logs/ai_trade.log` | DEBUG | Full diagnostic detail -- every indicator value, every rejection reason |
| `logs/ai_trade_run.log` | INFO | Decision journal -- only signals, trades, and summaries for post-run review |

### Noise Suppression

APScheduler's internal loggers are silenced to WARNING level to prevent
"Running job... / executed successfully" messages from flooding the logs:

```python
logging.getLogger("apscheduler.executors.default").setLevel(logging.WARNING)
logging.getLogger("apscheduler.scheduler").setLevel(logging.WARNING)
```

### Logger Usage

Throughout the codebase, loggers use structured key-value pairs:

```python
log.info("scan_complete", candidates=20, symbols=["AAPL", "TSLA", ...])
log.warning("daily_loss_limit", loss_pct=0.052, limit=0.05)
```

These are automatically enriched with timestamp and log level by structlog's
processor chain.

---

## Catch-up Logic

**File**: `src/ai_trade/main.py` -- `_catchup_missed_jobs()`

If the bot starts **during market hours** (after 9:30 AM ET, before 4:00 PM
ET), it does not wait for the next scheduled window. Instead, it immediately
catches up:

### Catch-up Algorithm

```
1. IF weekend: skip entirely
2. IF outside market hours (before 9:30 AM or after 4:00 PM): skip
3. ALWAYS run:
   a. job_premarket_scan()     -- scan for candidates
   b. job_market_open()        -- cache equity, sync positions, analyze regime
4. IF no open positions:
   c. job_entry_window()       -- run strategies and submit trades immediately
   ELSE:
   (skip entry -- wait for next scheduled window)
```

This ensures that:
- A restart at 11:00 AM does not leave the bot idle until 12:00 PM (midday).
- If the account has no positions, the bot immediately looks for
  opportunities.
- If positions exist, it trusts the existing bracket orders and waits for the
  normal schedule.

---

## Project Structure

```
src/ai_trade/
  main.py                       # TradingBot orchestrator + CLI entry point
  config.py                     # YAML + .env config loader
  clients.py                    # Alpaca client singletons
  utils.py                      # Shared utilities (retry logic, etc.)

  scanner/
    screener.py                 # StockScreener -- pre-market candidate scan

  strategy/
    base.py                     # BaseStrategy ABC, Signal dataclass, HoldType enum
    signal.py                   # SignalAggregator -- ranking + queue building
    mean_reversion.py           # RSI mean-reversion (swing)
    momentum.py                 # Volume-breakout momentum (adaptive)
    vwap.py                     # VWAP reclaim (day trade)
    options/
      base.py                   # BaseOptionsStrategy ABC, OptionsSignal, shared utils
      credit_put_spread.py
      debit_call_spread.py
      long_call.py
      long_put.py
      cash_secured_put.py
      covered_call.py
      covered_straddle.py
      momentum_options.py

  sentiment/
    market_regime.py            # SPY/QQQ/VIX regime analyzer
    news_sentiment.py           # Keyword-based news sentiment scorer

  data/
    historical.py               # fetch_bars(), fetch_bars_multi(), fetch_snapshots()
    streaming.py                # StreamManager WebSocket daemon thread
    indicators.py               # RSI, EMA, VWAP, ATR, Bollinger, MACD, volume profile
    options_chain.py            # get_options_chain(), get_options_snapshot()

  risk/
    risk_manager.py             # Portfolio-level risk gate
    position_sizer.py           # Fixed-fractional position sizing
    pdt_manager.py              # Day trade budget tracker

  execution/
    order_manager.py            # Stock bracket orders, position sync
    options_order_manager.py    # Options limit/spread orders

  monitoring/
    database.py                 # SQLite persistence layer
    logger.py                   # structlog configuration
    performance.py              # P&L, Sharpe, win rate, drawdown

  scheduler/
    jobs.py                     # APScheduler cron + interval job definitions

  backtest/
    engine.py                   # Backtesting engine
    runner.py                   # Backtest runner
    options_pricing.py          # Options pricing for backtests

config/
  settings.yaml                 # All configurable parameters
  .env                          # API keys (git-ignored)

data/
  ai_trade.db                   # SQLite database (auto-created)

logs/
  ai_trade.log                  # DEBUG-level full log
  ai_trade_run.log              # INFO-level decision journal
```
