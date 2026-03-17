# AI Trade

**A fully automated stock and options trading bot built on the Alpaca API.**

Designed for small accounts ($500+), optimized around the Pattern Day Trade (PDT) rule, with built-in risk management, market sentiment analysis, and a backtesting engine.

> **Status**: Paper trading. This system is designed for Alpaca paper accounts. Only consider live trading after consistent paper profitability.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Features](#features)
- [Quick Start](#quick-start)
- [Backtesting](#backtesting)
- [Strategies Overview](#strategies-overview)
- [Risk Management](#risk-management)
- [Market Sentiment](#market-sentiment)
- [Daily Schedule](#daily-schedule)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [CLI Reference](#cli-reference)
- [In-Depth Documentation](#in-depth-documentation)
- [Dependencies](#dependencies)

---

## How It Works

At a high level, the bot follows this pipeline every trading day:

```
Pre-Market Scan          Find stocks moving on volume/gaps
        |
   Market Open           Analyze SPY/QQQ/VIX for market regime
        |
  Strategy Evaluation    Run 3 stock + 8 options strategies on candidates
        |
  Signal Ranking         Score and prioritize signals by conviction
        |
  Risk Gate              Check daily loss, portfolio heat, PDT budget
        |
  Order Execution        Submit bracket orders (with server-side stops)
        |
  Position Monitoring    Sync positions every 60 seconds as safety net
        |
  End of Day             Close day trades, log P&L, save snapshot
```

**The key insight**: The bot runs multiple uncorrelated strategies simultaneously. Mean reversion profits in choppy markets. Momentum profits in trending markets. Options strategies generate income in low-volatility environments. By diversifying across strategy types, the system stays profitable in more market conditions than any single strategy could.

**PDT awareness** is central to the design. Accounts under $25,000 are limited to 3 day trades per 5 rolling business days (this is a FINRA regulation, not an Alpaca rule). The system treats day-trade slots as a scarce resource — it prefers swing trades (which are "free" — they don't count as day trades because you hold overnight) and only uses day-trade slots for very high-conviction setups. One slot is always reserved for emergency exits (e.g., a swing position gaps down overnight and you need to exit same-day).

---

## Features

| Category | Details |
|---|---|
| **Stock Strategies** | Mean Reversion, Momentum Breakout, VWAP Reclaim |
| **Options Strategies** | Credit Put Spreads, Debit Call Spreads, Long Calls, Long Puts, Cash-Secured Puts, Covered Calls, Covered Straddles, Momentum Options |
| **PDT Management** | Swing-first philosophy, day-trade budgeting, emergency reserve |
| **Market Regime** | SPY/QQQ/VIX breadth scoring gates all entries |
| **News Sentiment** | Keyword-weighted news scoring via Alpaca News API |
| **Risk Controls** | Portfolio heat, daily loss limits, bracket orders (server-side stops) |
| **Backtesting** | Walk-forward event-driven simulator with slippage, Black-Scholes options pricing |
| **Logging** | Structured JSON logs (full diagnostic + decision journal) via structlog |
| **Persistence** | SQLite database for trades, signals, snapshots, PDT tracking |

---

## Quick Start

### 1. Prerequisites

- **Python 3.12+** — The language the bot is written in
- An **[Alpaca](https://alpaca.markets)** account with paper trading enabled
- Options trading enabled (Level 3 for multi-leg spreads)

### 2. Install

```bash
git clone <your-repo-url>
cd "Ai Trade"
pip install -e .
```

**What does `pip install -e .` do?** It installs the project in "editable" mode. This means Python links to your source directory instead of copying files into a system location. You can modify any source file and the changes take effect immediately — no need to reinstall. The `.` means "install the package defined in this directory" (it reads `pyproject.toml` to know what to install).

### 3. Configure API Keys

```bash
cp config/.env.example config/.env
```

Edit `config/.env` and add your Alpaca paper trading credentials:

```env
ALPACA_API_KEY=your_paper_key_here
ALPACA_SECRET_KEY=your_paper_secret_here
ALPACA_PAPER=true
```

> **Security**: API keys live in `.env` (which is git-ignored), never in the YAML config file. The config loader reads `.env` at startup and injects the keys into the application configuration object.

### 4. Run

```bash
# Live paper trading — connects to Alpaca and trades automatically
ai-trade

# Dry run — evaluates strategies and logs what it WOULD trade, but submits no orders
ai-trade --dry-run

# Custom config file
ai-trade --config path/to/settings.yaml
```

When the bot starts, it:
1. Connects to Alpaca and prints account info (equity, cash, day trades remaining)
2. Starts the background scheduler that fires jobs at the configured times
3. If started during market hours, immediately runs a catch-up: scans candidates, analyzes market regime, and evaluates strategies
4. Blocks the main thread (keeps the program alive) and runs until you press `Ctrl+C`

---

## Backtesting

Test strategies against historical data before risking real capital. The backtester fetches **real OHLCV data from Alpaca** and simulates trades day-by-day, applying the same strategy logic and risk rules as the live bot.

```bash
# Quick test — 3 symbols, last 90 calendar days
ai-trade-backtest --symbols AAPL MSFT TSLA --days 90

# Show every individual trade (entry, exit, P&L, reason)
ai-trade-backtest --symbols AAPL MSFT TSLA --days 90 --show-trades

# Specific date range
ai-trade-backtest --symbols NVDA AMD --start 2025-01-01 --end 2025-06-30

# Full default universe (24 liquid stocks)
ai-trade-backtest --default-universe --days 180

# Include options strategies (uses Black-Scholes synthetic pricing)
ai-trade-backtest --default-universe --start 2024-03-01 --end 2026-03-01 --options --show-trades

# Export results to CSV files for analysis in Excel/Sheets
ai-trade-backtest --default-universe --days 90 --export results
```

### How the Backtester Works (Brief)

1. **Fetches real stock data** from Alpaca (OHLCV daily bars)
2. **Walks forward day by day** — on each day, it only sees data up to that point (no future peeking)
3. **Runs all enabled strategies** on each symbol, just like the live bot would
4. **Simulates fills with slippage** — entries fill at the open of the next bar (realistic, since you wouldn't get the close price)
5. **Tracks positions, cash, and P&L** exactly like a real account
6. **For options**: generates synthetic options chains using the Black-Scholes model (since Alpaca doesn't provide historical options data)
7. **Outputs summary statistics**: total return, win rate, Sharpe ratio, max drawdown, per-strategy breakdown

> **Important caveat**: Options backtest results use synthetic Black-Scholes pricing, not real historical options data. Stock strategy results use real Alpaca data and are trustworthy. Options results are directionally correct but overly optimistic — real markets have wider bid/ask spreads, IV crush, and skew.

For the full deep dive, see [docs/BACKTESTING.md](docs/BACKTESTING.md).

---

## Strategies Overview

### Stock Strategies

| Strategy | What It Does | When It Works | Hold Period | PDT Cost |
|---|---|---|---|---|
| **Mean Reversion** | Buys when RSI drops below 40 while price stays above the 20-day EMA and near the lower Bollinger Band. Bets the stock will bounce back to its average. | Ranging/pullback markets | Swing (2-5 days) | Free |
| **Momentum** | Buys when price breaks above the 20-day high with 2x+ normal volume and high daily range. Rides the trend up. | Trending/breakout markets | Adaptive | 0 or 1 |
| **VWAP Reclaim** | Buys when price reclaims VWAP from below on elevated volume during the trading day. Intraday mean-reversion play. | Strong intraday trends | Day only | 1 |

**Why these three?** Mean reversion and momentum are anti-correlated strategies. When markets chop sideways, mean reversion profits (stocks keep bouncing between support and resistance) while momentum sits out (no breakouts to chase). When markets trend strongly, momentum profits (stocks break to new highs on volume) while mean reversion stays flat (stocks don't pull back to oversold levels). VWAP is the intraday specialist — used sparingly because it costs a precious day-trade slot.

### Options Strategies

| Strategy | Type | What It Does | Risk Profile |
|---|---|---|---|
| **Credit Put Spread** | Multi-leg | Sell higher-strike put, buy lower-strike put. Collect premium up front. Profit if stock stays above your short strike through expiration. | Defined risk, defined reward |
| **Debit Call Spread** | Multi-leg | Buy a near-the-money call, sell a further-out call. Reduces cost vs a naked call but caps your upside. | Defined risk, defined reward |
| **Long Call** | Single-leg | Buy a call option on a high-conviction breakout. Maximum leverage — small premium can yield large returns if the stock moves. | Premium paid = max loss |
| **Long Put** | Single-leg | Buy a put option on a confirmed breakdown. Profit when stocks drop. | Premium paid = max loss |
| **Cash-Secured Put** | Single-leg | Sell an out-of-the-money put. Collect premium. If the stock drops to your strike, you buy shares at a discount. | Assignment risk at strike price |
| **Covered Call** | Single-leg | Sell a call against 100 shares you already own. Generate income, but cap your upside. | Limits upside on shares |
| **Covered Straddle** | Multi-leg | Sell an ATM call AND an ATM put against 100 shares. Double premium but high risk if the stock moves sharply. | Assignment + downside risk |
| **Momentum Options** | Single-leg | Buy cheap, short-dated (2-10 day) OTM options on momentum moves. If the move hits, 2-5x returns. If not, lose the small premium. | Premium = max loss; highest risk/reward |

For detailed theory on each strategy, see [docs/STRATEGIES.md](docs/STRATEGIES.md).

For options fundamentals, see [docs/OPTIONS_GUIDE.md](docs/OPTIONS_GUIDE.md).

---

## Risk Management

Every trade must pass through the **Risk Manager** before execution. Multiple independent checks run in sequence — if any single check fails, the trade is rejected.

| Control | Default | Purpose |
|---|---|---|
| **Max risk per trade** | 2% of equity | Limits how much you can lose on any single stock trade |
| **Max single position** | 30% of equity | Prevents putting too much money in one stock |
| **Max open positions** | 4 | Forces diversification across holdings |
| **Daily loss limit** | 5% of equity | Hard stop — if you've lost 5% today, no more trading |
| **Portfolio heat** | 6% of equity | Total risk across ALL open positions combined |
| **Stop loss** | 3% or ATR-based | Every stock entry has a server-side stop loss |
| **Take profit** | 2:1 risk-reward min | Bracket orders lock in profit targets |
| **Trailing stop** | 4% from high | Protects profits as winning trades run |
| **Max options positions** | 3 | Limits concurrent options exposure |
| **Max options capital** | 50% of portfolio | Caps total money allocated to options |
| **Max single options risk** | 12% of equity | Per-trade options risk cap; scales with account size |

**Bracket orders** are the most important safety feature: every stock entry simultaneously creates a stop-loss order and take-profit order on Alpaca's servers. If the bot crashes, these orders **still execute**.

**Options expiration protection**: The bot automatically closes any options positions expiring within 1 day to avoid exercise/assignment risk.

For the full risk management deep dive, see [docs/RISK_MANAGEMENT.md](docs/RISK_MANAGEMENT.md).

---

## Market Sentiment

### Regime Analysis

Before evaluating any strategy, the bot analyzes **SPY** (S&P 500), **QQQ** (Nasdaq 100), and **VIX** (volatility index) to classify the broad market into one of 5 regimes that gate and modify all trading decisions.

| Regime | Conviction Mod | Position Size Mod | New Longs? | Options? |
|---|---|---|---|---|
| **Strong Bull** | 1.3x boost | 1.0x full | Yes | Yes |
| **Bull** | 1.1x slight boost | 1.0x full | Yes | Yes |
| **Neutral** | 0.9x slight cut | 0.75x reduced | Yes | Yes |
| **Bear** | 0.6x big cut | 0.5x half | High conviction only | No |
| **Strong Bear** | 0.3x minimal | 0.25x quarter | No | No |

### News Sentiment

For each candidate stock, the bot scans Alpaca News API articles and scores them using weighted keyword dictionaries. Bullish news boosts conviction up to +30%. Bearish news reduces conviction up to -50%. A net score below -0.5 blocks the trade entirely.

---

## Daily Schedule

All times Eastern Time (ET), Monday-Friday.

| Time | Job | Description |
|---|---|---|
| **9:00 AM** | Pre-market scan | Scan ~10,000 stocks, return top 20 candidates by gap + volume |
| **9:30 AM** | Market open | Cache equity, sync positions, analyze market regime |
| **9:35 AM** | Entry window | Run strategies, rank signals, submit bracket orders |
| **12:00 PM** | Midday check | Re-evaluate, check new swing setups |
| **3:00 PM** | Power hour | Final momentum scan |
| **3:50 PM** | EOD close | Force-close day trades + expiring options |
| **4:05 PM** | EOD review | Log P&L, save equity snapshot |
| **Every 60s** | Position sync | Reconcile Alpaca positions with local DB |

---

## Configuration

All tunable parameters live in `config/settings.yaml`. API keys come from `config/.env`.

```yaml
account:          # Starting capital, position limits, daily loss limit
pdt:              # Day trade budget (3), reserve (1), min conviction for day trades
scanner:          # Price range ($2-50), volume filters, gap threshold, max candidates
strategies:       # Enable/disable and tune each of the 11 strategies
options:          # Options position limits, capital allocation
sentiment:        # News lookback window, conviction thresholds
risk:             # Stop loss %, trailing stop %, Kelly fraction
schedule:         # Cron job times (all ET)
```

For a complete reference of every parameter, see [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

---

## Project Structure

```
ai_trade/
├── pyproject.toml                      # Package definition: name, version, dependencies
├── config/
│   ├── settings.yaml                   # All tunable parameters
│   └── .env                            # API keys (git-ignored)
├── src/ai_trade/
│   ├── main.py                         # TradingBot — central orchestrator
│   ├── config.py                       # YAML + .env config loader
│   ├── clients.py                      # Alpaca client factory (singleton pattern)
│   ├── utils.py                        # Retry logic, Greek extraction
│   ├── data/                           # Data acquisition
│   │   ├── historical.py               #   OHLCV bar fetching
│   │   ├── streaming.py                #   Real-time WebSocket streaming
│   │   ├── indicators.py               #   Technical indicators
│   │   └── options_chain.py            #   Options chain + Greeks
│   ├── scanner/
│   │   └── screener.py                 #   Pre-market stock scanner
│   ├── strategy/                       # Trading strategies
│   │   ├── base.py                     #   Signal dataclass + abstract base
│   │   ├── mean_reversion.py           #   RSI oversold dip-buying
│   │   ├── momentum.py                 #   Volume breakout
│   │   ├── vwap.py                     #   VWAP reclaim (day trade)
│   │   ├── signal.py                   #   Signal ranking + queue builder
│   │   └── options/                    #   8 options strategies
│   ├── risk/                           # Risk management
│   │   ├── pdt_manager.py              #   Day-trade tracking
│   │   ├── position_sizer.py           #   Fixed-fractional sizing
│   │   └── risk_manager.py             #   Portfolio-level gates
│   ├── execution/                      # Order execution
│   │   ├── order_manager.py            #   Stock bracket orders
│   │   └── options_order_manager.py    #   Multi-leg options orders
│   ├── sentiment/                      # Market intelligence
│   │   ├── market_regime.py            #   SPY/QQQ/VIX regime analysis
│   │   └── news_sentiment.py           #   News sentiment scoring
│   ├── monitoring/                     # Observability
│   │   ├── database.py                 #   SQLite persistence
│   │   ├── performance.py              #   P&L metrics
│   │   └── logger.py                   #   Structured logging
│   ├── scheduler/
│   │   └── jobs.py                     #   APScheduler cron jobs
│   └── backtest/                       # Backtesting
│       ├── engine.py                   #   Walk-forward simulator
│       ├── options_pricing.py          #   Black-Scholes pricing
│       └── runner.py                   #   CLI interface
├── data/                               # SQLite DB (auto-created)
├── logs/                               # Log files (auto-created)
├── docs/                               # In-depth documentation
└── tests/                              # Test suite
```

For a detailed architecture walkthrough, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## CLI Reference

### `ai-trade` — Live Trading Bot

```
ai-trade [--config CONFIG] [--dry-run]
```

### `ai-trade-backtest` — Backtester

```
ai-trade-backtest [--symbols AAPL MSFT] [--default-universe] [--days 90]
                  [--start 2024-01-01] [--end 2025-01-01] [--options]
                  [--show-trades] [--export results]
```

---

## In-Depth Documentation

| Document | What You'll Learn |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, data flow, component interactions, design patterns |
| [STRATEGIES.md](docs/STRATEGIES.md) | All 11 strategies — theory, entry/exit conditions, conviction scoring |
| [OPTIONS_GUIDE.md](docs/OPTIONS_GUIDE.md) | Options fundamentals — calls, puts, Greeks, spreads |
| [RISK_MANAGEMENT.md](docs/RISK_MANAGEMENT.md) | Position sizing, loss limits, portfolio heat, PDT, bracket orders |
| [BACKTESTING.md](docs/BACKTESTING.md) | Walk-forward simulator, Black-Scholes pricing, limitations |
| [CONFIGURATION.md](docs/CONFIGURATION.md) | Every settings.yaml parameter with defaults and tuning guidance |

---

## Dependencies

| Package | What It Does |
|---|---|
| `alpaca-py` | Official Alpaca SDK — trading and market data |
| `pandas` | Data manipulation — OHLCV bars stored as DataFrames |
| `ta` | Technical indicators (pure Python, no C dependencies) |
| `apscheduler` | Background job scheduler for the daily trading schedule |
| `pyyaml` | YAML config file parsing |
| `python-dotenv` | Loads API keys from `.env` files |
| `structlog` | Structured logging — JSON for machines, readable for humans |

---

## License

Private — not for redistribution.
