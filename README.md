# AI Trade

**A fully automated stock and options trading bot built on the Alpaca API.**

Designed for small accounts ($500+), optimized around the Pattern Day Trade (PDT) rule, with built-in risk management, market sentiment analysis, email alerts, and a backtesting engine.

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
- [Email Alerts](#email-alerts)
- [Daily Schedule](#daily-schedule)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [CLI Reference](#cli-reference)
- [In-Depth Documentation](#in-depth-documentation)
- [Dependencies](#dependencies)
- [Changelog](#changelog)

---

## How It Works

At a high level, the bot follows this pipeline every trading day:

```
Pre-Market Scan          Find stocks moving on volume/gaps
        |
   Market Open           Analyze SPY/QQQ/VIX for market regime
        |
  Strategy Evaluation    Run 8 stock + 8 options strategies on candidates
        |
  Signal Ranking         Score and prioritize signals by conviction
        |
  Risk Gate              Check daily loss, portfolio heat, PDT budget
        |
  Order Execution        Submit bracket orders (with server-side stops)
        |                  + adapt to current market price if stale
        |
  Position Monitoring    Sync positions every 5 minutes
        |
  End of Day             Close day trades, log P&L, save snapshot
```

**The key insight**: The bot runs multiple uncorrelated strategies simultaneously. Mean reversion profits in choppy markets. Momentum profits in trending markets. Options strategies generate income in low-volatility environments. By diversifying across strategy types, the system stays profitable in more market conditions than any single strategy could.

**PDT awareness** is central to the design. Accounts under $25,000 are limited to 3 day trades per 5 rolling business days (this is a FINRA regulation, not an Alpaca rule). The system treats day-trade slots as a scarce resource — it prefers swing trades (which are "free" — they don't count as day trades because you hold overnight) and only uses day-trade slots for very high-conviction setups. One slot is always reserved for emergency exits (e.g., a swing position gaps down overnight and you need to exit same-day).

**Price adaptation**: When a signal's entry price is stale (strategies compute prices from daily bar close, which can be hours old), the order manager recalculates stop-loss, take-profit, and position size around the current market price while preserving the original risk/reward ratio. Orders with >50% price divergence are rejected as stale.

---

## Features

| Category | Details |
|---|---|
| **Stock Strategies** | Mean Reversion, Momentum Breakout, VWAP Reclaim, EMA Crossover, Bollinger Band Squeeze, MACD Divergence, Pullback, Opening Range Breakout (ORB) |
| **Options Strategies** | Credit Put Spreads, Debit Call Spreads, Long Calls, Long Puts, Cash-Secured Puts, Covered Calls, Covered Straddles, Momentum Options |
| **PDT Management** | Swing-first philosophy, day-trade budgeting, emergency reserve |
| **Market Regime** | SPY/QQQ/VIX breadth scoring gates all entries |
| **News Sentiment** | Keyword-weighted news scoring via Alpaca News API |
| **Risk Controls** | Portfolio heat, daily loss limits, bracket orders (server-side stops), position size adaptation |
| **Email Alerts** | Real-time notifications for high-conviction signals and all trade submissions |
| **ML Signal Scoring** | GradientBoostingClassifier blends rule-based conviction with learned P(win) estimate — trained from backtest history |
| **Backtesting** | Walk-forward event-driven simulator with ML feature capture, slippage, Black-Scholes options pricing |
| **Logging** | Structured JSON logs (full diagnostic + decision journal) via structlog |
| **Persistence** | SQLite database for trades, signals, snapshots, PDT tracking (with performance indexes) |

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

Edit `config/.env` and add your credentials:

```env
ALPACA_API_KEY=your_paper_key_here
ALPACA_SECRET_KEY=your_paper_secret_here
ALPACA_PAPER=true

# Email alerts (optional — uses Gmail SMTP)
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_gmail_app_password
```

> **Security**: API keys and SMTP credentials live in `.env` (which is git-ignored), never in the YAML config file. The config loader reads `.env` at startup and injects the keys into the application configuration object.

> **Gmail app password**: Go to Google Account > Security > 2-Step Verification > App passwords > generate one for "Mail". Regular Gmail passwords won't work with SMTP.

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

# Bootstrap the ML model from backtest history (283-symbol liquid universe, 1 year)
# Trains a GradientBoostingClassifier and registers it in the live DB automatically
ai-trade-backtest --full-universe --days 365 --capital 100000 --train-ml
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
| **Mean Reversion** | Buys RSI-oversold dips near lower Bollinger Band with MACD turning up and volume capitulation. | Ranging/pullback markets | Swing (2-5 days) | Free |
| **Momentum** | Buys breakouts above 20-day high with stacked EMAs, MACD > 0, RSI 50-80, and pre-breakout consolidation. | Trending/breakout markets | Adaptive | 0 or 1 |
| **VWAP Reclaim** | Buys VWAP reclaim from below on elevated volume with bullish candle and meaningful dip depth (>0.5%). | Strong intraday trends | Day only | 1 |
| **EMA Crossover** | Buys fresh EMA-9/EMA-20 crossovers in established uptrends (EMA-20 > EMA-50) with MACD and volume confirmation. | Trending markets | Swing | Free |
| **BB Squeeze** | Buys breakouts from Bollinger Band squeeze (width in bottom 30%) with MACD, RSI, and volume confirmation. | Volatility compression | Adaptive | 0 or 1 |
| **MACD Divergence** | Buys bullish MACD divergence (price lower low, MACD higher low) with RSI double-confirmation and volume decline. | Reversal setups | Swing | Free |
| **Pullback** | Buys pullbacks to EMA-20/50 support with volume dry-up, MACD still positive, and bullish candle reversal. | Uptrend continuation | Swing | Free |
| **ORB** | Buys opening range breakouts (first 30 min) with volume spike, bullish candle, and sustained close above range. | Intraday breakouts | Day only | 1 |

**Why these eight?** The strategies are designed for maximum diversification across market conditions. Mean reversion and momentum are anti-correlated. EMA crossover and pullback profit in established trends. BB squeeze catches volatility expansion. MACD divergence catches reversals. VWAP and ORB are intraday specialists. By combining trend-following, mean-reversion, and breakout strategies, the system stays profitable in more conditions than any single strategy.

### Options Strategies

| Strategy | Type | What It Does | Risk Profile | $500 Account |
|---|---|---|---|---|
| **Credit Put Spread** | Multi-leg | Sell higher-strike put, buy lower-strike put. Collect premium. Profit if stock stays above short strike. | Defined risk ($150 max) | Primary income strategy |
| **Debit Call Spread** | Multi-leg | Buy a near-the-money call, sell a further-out call. Reduced cost vs naked call. | Defined risk, defined reward | Good for bullish setups |
| **Long Call** | Single-leg | Buy a call on a high-conviction breakout. Maximum leverage. | Premium paid = max loss ($75 cap) | Best asymmetric bet |
| **Long Put** | Single-leg | Buy a put on a confirmed breakdown. Profit when stocks drop. | Premium paid = max loss ($75 cap) | Bearish hedging |
| **Cash-Secured Put** | Single-leg | Sell an OTM put on a stock you'd buy. Collect premium. | Assignment at strike ($300 max collateral) | Income on cheap stocks |
| **Covered Call** | Single-leg | Sell a call against 100 shares you own. Generate income. | Limits upside on shares ($300 max position) | Income on held shares |
| **Covered Straddle** | Multi-leg | Sell ATM call + ATM put against 100 shares. Double premium. | High risk — needs >$600 capital | **Disabled** (too capital-intensive) |
| **Momentum Options** | Single-leg | Buy cheap, short-dated (5-20 day) OTM options on momentum moves. | Premium = max loss ($75 cap) | High risk/reward lottery |

For detailed theory on each strategy, see [docs/STRATEGIES.md](docs/STRATEGIES.md).

For options fundamentals, see [docs/OPTIONS_GUIDE.md](docs/OPTIONS_GUIDE.md).

---

## Risk Management

Every trade must pass through the **Risk Manager** before execution. Multiple independent checks run in sequence — if any single check fails, the trade is rejected.

| Control | Default | With $500 |
|---|---|---|
| **Max risk per trade** | 2% of equity | $10 |
| **Max single position** | 25% of equity | $125 |
| **Max open positions** | 4 | — |
| **Daily loss limit** | 5% of equity | $25 |
| **Portfolio heat** | 6% of equity | $30 total risk |
| **Stop loss** | 3% or ATR-based | Server-side bracket order |
| **Take profit** | 2:1 risk-reward min | Server-side bracket order |
| **Trailing stop** | 4% from high | Protects running winners |
| **Max options positions** | 3 | — |
| **Max options capital** | 50% of portfolio | $250 |
| **Max single options risk** | 12% of equity | $60 |

**Bracket orders** are the most important safety feature: every stock entry simultaneously creates a stop-loss order and take-profit order on Alpaca's servers. If the bot crashes, these orders **still execute**.

**Position size adaptation**: When the current market price diverges from the signal's entry price, the order manager recalculates the number of shares to maintain the same total dollar risk. This prevents over- or under-sizing when prices move between signal generation and order submission.

**1-share minimum guard**: Small accounts may compute 0 shares from the position sizer. The bot allows a 1-share minimum only when the single-share risk is within the per-trade risk budget — preventing the fallback from bypassing risk controls.

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

## Email Alerts

The bot sends real-time email notifications via Gmail SMTP for:

- **High-conviction signals** (conviction >= 0.70) — sent even in dry-run mode so you can track signal quality
- **Stock orders submitted** — includes symbol, shares, entry, stop-loss, take-profit, cost, and order ID
- **Stock orders failed** — alerts you to investigate
- **Options orders submitted** — includes underlying, strategy, legs, max loss/profit, ROI, and expiration

Emails are sent in background threads so they never block the trading pipeline.

**Setup**: Add `SMTP_USER` and `SMTP_PASS` to `config/.env`. See [Quick Start](#3-configure-api-keys) for details.

---

## Daily Schedule

All times Eastern Time (ET), Monday-Friday.

| Time | Job | Description |
|---|---|---|
| **9:00 AM** | Pre-market scan | Scan ~10,000 stocks, return top 20 candidates by gap + volume. Scan separate options universe (top 30 by volume + liquidity). |
| **9:30 AM** | Market open | Cache equity, sync positions, analyze SPY/QQQ/VIX market regime |
| **9:45 AM - 3:45 PM** | Scan & evaluate (every 15 min) | Run all 8 stock + 8 options strategies, rank signals, submit orders |
| **3:30 PM** | Options expiry check | Close options expiring today/tomorrow |
| **3:50 PM** | EOD close | Force-close day trades |
| **4:05 PM** | EOD review | Log P&L, save equity snapshot |
| **Every 5 min** | Options position sync | Reconcile broker options positions with local DB |

---

## Configuration

All tunable parameters live in `config/settings.yaml`. API keys and SMTP credentials come from `config/.env`.

```yaml
account:          # Starting capital, position limits (25% max), daily loss limit
pdt:              # Day trade budget (3), reserve (1), min conviction for day trades
scanner:          # Price range ($2-50), volume filters, gap threshold, max candidates
  options_universe: # Separate filters for options-eligible stocks ($10-500, >1M volume)
strategies:       # Enable/disable and tune each of the 16 strategies
options:          # Options position limits (3), capital allocation (50%)
sentiment:        # News lookback window, conviction thresholds
risk:             # Stop loss %, trailing stop %, Kelly fraction, portfolio heat
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
│   └── .env                            # API keys + SMTP creds (git-ignored)
├── src/ai_trade/
│   ├── main.py                         # TradingBot — central orchestrator
│   ├── _version.py                     # Single source of truth for version
│   ├── config.py                       # YAML + .env config loader
│   ├── clients.py                      # Alpaca client factory (singleton pattern)
│   ├── utils.py                        # Retry logic, Greek extraction
│   ├── data/                           # Data acquisition
│   │   ├── historical.py               #   OHLCV bar fetching
│   │   ├── streaming.py                #   Real-time WebSocket streaming
│   │   ├── indicators.py               #   Technical indicators (RSI, EMA, ATR, BB, VWAP, MACD)
│   │   └── options_chain.py            #   Options chain + Greeks
│   ├── scanner/
│   │   └── screener.py                 #   Multi-profile scanner (momentum, mean-reversion, VWAP, options)
│   ├── strategy/                       # Trading strategies
│   │   ├── base.py                     #   Signal dataclass + abstract base
│   │   ├── mean_reversion.py           #   RSI oversold dip-buying (swing)
│   │   ├── momentum.py                 #   Volume breakout (adaptive)
│   │   ├── vwap.py                     #   VWAP reclaim (day trade)
│   │   ├── ema_crossover.py            #   EMA-9/20 crossover trend-following (swing)
│   │   ├── bb_squeeze.py               #   Bollinger Band squeeze breakout (adaptive)
│   │   ├── macd_divergence.py          #   Bullish MACD divergence (swing)
│   │   ├── pullback.py                 #   Pullback to EMA support (swing)
│   │   ├── orb.py                      #   Opening range breakout (day trade)
│   │   ├── signal.py                   #   Signal ranking + queue builder (the "brain")
│   │   ├── weighter.py                 #   Adaptive strategy weighting
│   │   └── options/                    #   8 options strategies
│   │       ├── base.py                 #     Shared utilities: filter_contracts, enrich_greeks, select_by_delta
│   │       ├── credit_put_spread.py    #     Bull put spread (defined risk income)
│   │       ├── debit_call_spread.py    #     Bull call spread (defined risk directional)
│   │       ├── long_call.py            #     Directional call buying
│   │       ├── long_put.py             #     Directional put buying
│   │       ├── cash_secured_put.py     #     Premium selling on support
│   │       ├── covered_call.py         #     Income on held shares
│   │       ├── covered_straddle.py     #     Double-premium selling (disabled for small accounts)
│   │       └── momentum_options.py     #     Short-dated momentum plays
│   ├── risk/                           # Risk management
│   │   ├── pdt_manager.py             #   Day-trade tracking + budgeting
│   │   ├── position_sizer.py          #   Fixed-fractional sizing with risk-aware minimum
│   │   └── risk_manager.py            #   Portfolio-level gates (heat, concentration, daily loss)
│   ├── execution/                      # Order execution
│   │   ├── order_manager.py           #   Stock bracket orders with price adaptation
│   │   └── options_order_manager.py   #   Multi-leg options orders
│   ├── sentiment/                      # Market intelligence
│   │   ├── market_regime.py           #   SPY/QQQ/VIX regime analysis (5 regimes)
│   │   └── news_sentiment.py          #   Keyword-weighted news scoring
│   ├── monitoring/                     # Observability
│   │   ├── database.py                #   SQLite persistence (indexed tables)
│   │   ├── performance.py             #   P&L metrics (Sharpe, drawdown, win rate)
│   │   ├── console.py                 #   Pretty console output formatters
│   │   ├── notifier.py               #   Email alerts (SMTP, background threads)
│   │   └── logger.py                  #   Structured logging (JSON + console)
│   ├── scheduler/
│   │   └── jobs.py                    #   APScheduler cron jobs (18 scheduled tasks)
│   ├── ml/                             # Machine learning
│   │   ├── features.py                #   Canonical 15-column feature vector
│   │   ├── trainer.py                 #   GradientBoosting training (time-ordered 80/20 split)
│   │   └── predictor.py               #   Live inference + blend weight ramp (0→0.5 over 200 trades)
│   └── backtest/                       # Backtesting
│       ├── engine.py                  #   Walk-forward simulator (ML feature capture at fill)
│       ├── options_pricing.py         #   Black-Scholes pricing
│       └── runner.py                  #   CLI: --train-ml / --capital / --full-universe
├── data/                               # SQLite DB (auto-created)
│   └── ml_training_universe.txt       #   283 liquid stocks for ML training backtests
├── models/                             # Trained ML models (joblib, gitignored for prod)
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
ai-trade-backtest [--symbols AAPL MSFT] [--default-universe] [--full-universe]
                  [--days 90] [--start 2024-01-01] [--end 2025-01-01] [--options]
                  [--show-trades] [--export results]
                  [--train-ml] [--capital 100000]
```

---

## In-Depth Documentation

| Document | What You'll Learn |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, data flow, component interactions, design patterns |
| [STRATEGIES.md](docs/STRATEGIES.md) | All 16 strategies — theory, entry/exit conditions, conviction scoring |
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

## Changelog

### v2.1.0 — Backtest ML Training Pipeline

**ML Bootstrap from Historical Simulation**
- Backtest engine now captures ML features at the exact moment each order fills, using the same `extract_features()` call as live trading — guaranteed feature parity between training and inference
- Features are paired with realized P&L at close to build labelled training rows (no synthetic labels)
- `--train-ml` flag runs `_train_ml_from_backtest()` after the simulation: writes a fresh `backtest_ml.db`, trains a `GradientBoostingClassifier` (100 estimators, max_depth=3, lr=0.05, time-ordered 80/20 split), and auto-registers the new model in the live `ai_trade.db` — the bot picks it up on next restart with no manual steps
- `--full-universe` flag loads `data/ml_training_universe.txt` (283 liquid stocks: price $5-$2000, vol >500k/day) for production-grade training coverage
- `--capital N` overrides starting capital for the simulation — use `--capital 100000` so position-sizing constraints on small accounts don't suppress signal generation during training
- Batched symbol fetching (BATCH_SIZE=200) prevents Alpaca rate-limit errors on large universes
- **Model v3** trained on 433 backtest trades: val accuracy 60.9%, 88.8% precision when P(win)>0.5 (top features: RSI, conviction, ATR, relative volume, R:R ratio)
- Blend weight ramps from 0→0.5 as training trades accumulate (200 trade ramp); nightly retraining (`job_train_ml_models` at 17:00 ET) continuously improves the model as live data grows

### v2.0.0 — Complete 13-Phase Overhaul

See [docs/V2_PROGRESS.md](docs/V2_PROGRESS.md) for the full 13-phase breakdown. Key additions: decision audit trail, adaptive exit planner with S/R-anchored stops, conviction-aware trailing stops, state persistence across restarts, ML core infrastructure, self-learning trade analysis, dynamic risk controller, smart PDT management, strategy optimizer, 0DTE options strategy, news/event intelligence, market prediction module, and performance optimization (parallel fetching, WAL mode, cycle timing).

### v1.2.0 — Multi-Indicator Confluence & Options Strategy Optimization

**Strategy Overhaul (16 strategies rewritten)**
- All 8 stock strategies now require 5-7 independent entry conditions (price action + volume + RSI + MACD + EMA structure)
- All 8 options strategies now require multi-indicator confluence (MACD, EMA-50, volume patterns, Bollinger Bands)
- Additive conviction scoring with per-factor bonuses replaces linear scaling
- Every stock strategy enforces minimum 2:1 reward-to-risk ratio (1.5:1 for intraday)
- MACD histogram alignment required across all strategies
- Stacked EMA confirmation (close > EMA-20 > EMA-50) for bullish entries
- Pre-breakout consolidation filter on momentum and long call strategies
- Volume pattern analysis: dry-up on pullbacks, spikes on breakouts, capitulation on reversals

**Hardening**
- PDT pre-check before order submission prevents 30+ blocked order attempts per day
- Failed symbol blacklist after 2 consecutive order failures stops retrying halted/untradable symbols
- Swing conviction floor raised to 0.55 (was no minimum)
- Post-modifier conviction floor raised to 0.50 (was 0.35)
- Scanner min_price raised to $5 to filter penny stocks prone to halts and spreads

**Config Updates**
- EMA crossover RSI max tightened from 75 to 70
- BB squeeze min_relative_volume raised from 1.3 to 1.5
- ORB min_range_pct parameter added (0.3%)
- Debit call spread max_debit_pct tightened from 0.60 to 0.50
- Credit put spread max_risk reduced from $250 to $100

### v1.1.0 — Pretty Console Logging, 5 New Strategies, Adaptive Weighting

**New Features**
- 5 new stock strategies: EMA Crossover, BB Squeeze, MACD Divergence, Pullback, ORB
- Adaptive strategy weighting based on recent performance (win rate + profit factor)
- Pretty console output with Unicode box-drawing characters and consistent formatting
- 15-minute scanning interval replaces fixed entry windows (9:45 AM - 3:45 PM every 15 min)

**Bug Fixes**
- Fixed bracket order rejection for fill slippage (widened clamp buffer from 0.01 to 0.05)
- Fixed undefined variables in premarket scan job (momentum_count, mr_candidates, vwap_candidates)
- Fixed Unicode encoding error in console output (added UTF-8 encoding)

### v0.4.0 — Refactor, Tuning & Email Alerts

**Bug Fixes**
- Fixed credit put spread theta formula (was using `abs()` incorrectly on long leg theta)
- Refactored covered call and covered straddle to use shared `filter_contracts`/`enrich_greeks` utilities — fixes greeks data format mismatch where these strategies read from root-level snapshot fields instead of the nested `greeks` dict
- 1-share minimum fallback now checks risk budget before allowing — prevents bypassing position sizing on expensive stocks
- Bracket order price adaptation rejects orders with >50% price divergence instead of adapting (thesis is clearly stale)
- Fixed options expiry check fetching open trades in an inner loop (N queries → 1 query)
- Replaced 6 silent `except: pass` handlers with proper debug/warning logging

**Tuning ($500 Account Optimization)**
- Reduced `max_position_pct` from 30% to 25%
- Tightened mean reversion: RSI entry 40→35, exit 60→55 (deeper dips, earlier profits)
- Raised momentum volume threshold from 1.5x to 2.0x (fewer false breakouts)
- Credit put spread: delta 0.30→0.25, spread width $2.50→$1.50 (max loss $150 not $250)
- Cash-secured put & covered call: max stock price $5→$3 (max collateral $300)
- Disabled covered straddle (requires >$600 capital — impossible on $500)
- Momentum options: min DTE 2→5 days, min delta 0.15→0.25, cost cap $100→$75
- Debit call spread: max debit 60%→50% of width

**Conviction Recalibration**
- Mean reversion: capped at 0.85 (extreme RSI could be crash, not bounce)
- VWAP: base lowered 0.70→0.55, requires meaningful dip (>0.5% below VWAP)
- CSP and covered call: base lowered 0.60→0.50, ceiling capped at 0.80
- Credit put spread: base lowered 0.60→0.55
- Momentum options: base lowered 0.50→0.45, added theta decay penalty for DTE<7

**Code Quality**
- Extracted `_run_full_scan()` and `_run_evaluation_cycle()` — eliminated 3 duplicate patterns in main.py
- Added 9 SQLite performance indexes across all tables
- Reduced momentum options budget overage tolerance from 1.5x to 1.1x
- Widened covered call RSI range to 45-70 (was 40-65)

### v0.3.5 — Email Notifications

- Added email alert system via Gmail SMTP (background threads, non-blocking)
- Alerts on: high-conviction signals (>=0.70), stock orders, options orders, failed orders

### v0.3.4 — Position Size Adaptation

- Bracket order price adaptation now recalculates share count to preserve total dollar risk
- Logs original vs adapted shares and cost for full visibility

### v0.3.3 — Duplicate Position Fix & Schedule Expansion

- Fixed duplicate position entries (EUDA bought 3x in one day)
- Added `held_symbols` dedup from broker positions + DB open trades
- Added 4 new scheduled jobs: mid-morning options (10:15), late morning (11:00), options expiry check (3:30), options position sync (every 5 min)
- Fixed order submission logging (was in wrong branch — printed "submitted" on failure)

### v0.3.0 — Options Universe & ROI Ranking

- Separate options-aware universe scanner ($10-500 stocks with >1M volume)
- ROI-ranked options signal execution (collect-then-rank instead of first-come-first-served)
- Imported MomentumOptionsStrategy (was dead code)
- Fixed options risk sizing to use `max_single_options_risk_pct` config key

---

## License

Private — not for redistribution.
