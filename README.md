# AI Trade

**A fully automated stock and options trading bot built on the Alpaca API.**

Designed for small accounts, optimized around the Pattern Day Trade (PDT) rule, with built-in risk management, market sentiment analysis, a machine-learning signal-quality layer, email alerts, and a backtesting engine.

> **Status**: Paper trading (currently on a ~$500 Alpaca paper account). A live go-live plan targeting $2,000 is documented in [docs/ROADMAP_TO_LIVE.md](docs/ROADMAP_TO_LIVE.md) — do not take this bot live before completing Phases 0-2 of that roadmap.
>
> **Current version**: see [src/ai_trade/_version.py](src/ai_trade/_version.py) (2.1.1 at time of writing).

---

## Table of Contents

- [How It Works](#how-it-works)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation & Setup](#installation--setup)
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
  Strategy Evaluation    Run 8 stock + 9 options strategies on candidates
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
| **Options Strategies** | Credit Put Spreads, Debit Call Spreads, Long Calls, Long Puts, Cash-Secured Puts, Covered Calls, Covered Straddles (disabled by default), Momentum Options, 0-DTE Gamma Plays |
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
git clone https://github.com/Bishopc94/algo-trading.git
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

## Installation & Setup

This section walks end-to-end from a blank machine to a running paper-trading bot. Read the whole thing before starting — several steps have order dependencies.

### 1. System prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| **Python** | 3.12+ | Earlier versions will fail: modern type-hint syntax (`float \| None`) and `datetime.UTC` are used throughout. |
| **Git** | any recent | Used to clone the repo. |
| **SQLite** | bundled with Python | No separate install needed. |
| **OS** | Windows 11, macOS, Linux | Development is on Windows 11. Tested on bash/zsh shells. |
| **Disk** | ~500 MB | Repo (~50 MB) + Alpaca bar data + logs + SQLite DBs. ML training backtests can add a few hundred MB to `data/`. |
| **Network** | reliable | Required for Alpaca REST + WebSocket. Transient network errors are retried (see [src/ai_trade/utils.py](src/ai_trade/utils.py) `retry_api_call`). |

### 2. Alpaca account setup

1. Create a free account at <https://alpaca.markets>.
2. Open the **Paper Trading** dashboard (top-left account switcher). The paper account is a sandbox funded with virtual money — no risk while you validate the bot.
3. Under **Your API Keys**, generate a new key pair. **Save both values immediately** — the secret key is shown only once.
4. Apply for **options trading — Level 3** from the dashboard. Level 3 is needed for multi-leg spreads (credit put spread, debit call spread). Approval is usually instant on paper accounts. Without Level 3, only single-leg options strategies (long call, long put, CSP, covered call, momentum, 0-DTE) will execute.
5. Note that **paper and live API keys are different**. Every time you swap environments you must update both `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` and the `ALPACA_PAPER` flag in `.env`.

### 3. Clone and create a virtual environment

```bash
# Clone
git clone https://github.com/Bishopc94/algo-trading.git "Ai Trade"
cd "Ai Trade"

# Create an isolated Python environment (strongly recommended)
python -m venv .venv

# Activate it
source .venv/Scripts/activate    # Windows (bash/Git Bash)
# source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell / cmd
```

Verify you're in the venv — the prompt should be prefixed with `(.venv)` and `which python` should point into `.venv/`.

### 4. Install the package

```bash
pip install --upgrade pip
pip install -e .
```

**Why editable (`-e`) mode?** It symlinks the installed package to your source tree, so any edit takes effect on the next `ai-trade` invocation without re-installing. The ML and backtest layers also import `scikit-learn`, `joblib`, and `numpy`, which are pulled in transitively via `pandas` and `ta` — if you hit `ModuleNotFoundError: sklearn` during ML training, install it explicitly:

```bash
pip install scikit-learn joblib
```

For running the test suite and smoke scripts:

```bash
pip install -e ".[dev]"     # pytest + pytest-asyncio
```

### 5. Configure secrets

```bash
cp config/.env.example config/.env
```

Open `config/.env` in an editor and fill in the five fields:

| Variable | What it is | Required? |
|---|---|---|
| `ALPACA_API_KEY` | From the Alpaca paper dashboard | Yes |
| `ALPACA_SECRET_KEY` | From the Alpaca paper dashboard | Yes |
| `ALPACA_PAPER` | `true` for paper, `false` for live | Yes |
| `SMTP_USER` | Gmail address for alerts | No — leave blank to disable email |
| `SMTP_PASS` | Gmail **App Password** (not your login password) | Only if `SMTP_USER` is set |

**Generating a Gmail App Password:** Google Account → Security → 2-Step Verification (must be enabled) → App passwords → pick "Mail" and your device → copy the 16-character token into `SMTP_PASS`. Regular Gmail passwords will not authenticate against SMTP.

**Security notes:**
- `config/.env` is listed in `.gitignore` — never commit it.
- The `.env.example` template is safe to commit because it contains no real values.
- Keys are read by [src/ai_trade/config.py](src/ai_trade/config.py) at startup; they never appear in the YAML config or in logs.

### 6. Review `config/settings.yaml`

All non-secret tuning lives here. The repo ships a working default; the values most likely to need changing on first run:

```yaml
account:
  starting_capital: 500.0      # match this to your actual paper equity
  max_open_positions: 5

pdt:
  max_day_trades: 3            # FINRA limit is 4 — this leaves a safety margin
  day_trade_reserve: 1         # slots reserved for emergency exits

schedule:
  premarket_scan: "09:00"      # all times Eastern
  ml_training: "17:00"
```

A full parameter reference is at [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

### 7. First run — dry-run mode

Always start dry. This connects to Alpaca, runs the full strategy pipeline, and logs what it *would* do, but submits no orders.

```bash
ai-trade --dry-run
```

Expected output on a healthy setup:
- `config_loaded` event in the log
- Alpaca account info printed (equity, cash, `daytrade_count`)
- Scheduler startup with ~20 jobs registered
- If market is open: a scan + evaluate cycle within ~60 seconds
- Log files created under `logs/` (`ai_trade.log` for detail, `ai_trade_run.log` for structured events)

Leave it running through one 15-minute cycle to confirm scan → strategy → risk-gate flow. Stop with `Ctrl+C`.

### 8. First paper trade

Remove `--dry-run` and run during market hours (09:30-16:00 ET, weekdays):

```bash
ai-trade
```

The bot will submit bracket orders for any signal that passes all gates. Watch `logs/ai_trade_run.log` for `order_submitted` events and reconcile with the Alpaca dashboard.

### 9. (Optional) Bootstrap the ML model

The shipped `models/signal_quality_v3.joblib` was trained on 433 backtest trades. To train a fresh model from your own configuration:

```bash
ai-trade-backtest --full-universe --days 365 --capital 100000 --train-ml
```

This runs a one-year walk-forward backtest on the 283-symbol liquid universe, captures ML features at every fill, fits a `GradientBoostingClassifier`, and auto-registers the new model in `data/ai_trade.db`. The next `ai-trade` restart will load it. Training takes 10-30 minutes depending on machine speed.

### 10. Verify the ML pipeline is live

After the bot starts, grep for predictor events:

```bash
grep "predictor_model_loaded" logs/ai_trade.log
```

You should see the active model's version and training-trade count. If nothing is returned, see **Known ML pipeline bugs** in [docs/ROADMAP_TO_LIVE.md](docs/ROADMAP_TO_LIVE.md) — Bug A describes a logging-order issue where predictor startup events can be silently dropped.

### 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ALPACA_API_KEY must be set` on startup | `.env` not copied or key empty | Re-run step 5; confirm `cat config/.env` shows values |
| `Connection timeout` on every Alpaca call | Firewall / corporate proxy | Whitelist `api.alpaca.markets`, `data.alpaca.markets`, `paper-api.alpaca.markets` |
| `ModuleNotFoundError: sklearn` during `--train-ml` | Transitive install skipped scikit-learn | `pip install scikit-learn joblib` |
| `day_trade_blocked` on every signal | Account is small and flagged PDT | Expected on <$25k paper — swing strategies still run. Tune `pdt.min_conviction_for_day_trade` in settings.yaml |
| Unicode errors in console on Windows | Terminal codepage | Run under Windows Terminal or Git Bash, not legacy `cmd.exe` |
| Email alerts silently not sending | Gmail app-password not used / 2FA off | Regenerate per step 5. Gmail has hard-disabled "less secure apps" — app passwords are required |
| No `predictor_*` events in log | Bug A (see ROADMAP_TO_LIVE.md) | Pending fix — does not affect trading, only observability |

### 12. Keeping it running

The bot is a foreground process — closing the terminal stops it. For continuous paper operation choose one:

- **tmux / screen** (Linux/macOS):  `tmux new -s aitrade 'ai-trade'`, detach with `Ctrl+B D`, reattach with `tmux a -t aitrade`.
- **Windows Task Scheduler**: create a task that runs `ai-trade` at user logon in the venv's Python interpreter.
- **systemd** (Linux): wrap in a service unit pointing at the venv's `ai-trade` entrypoint.

State is persisted to `data/ai_trade.db` on every scheduled event, so restarts lose at most one 15-minute cycle of in-memory scan results.

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
| **0-DTE Gamma Plays** | Single-leg | Buy same-day expiration calls/puts on liquid underlyings (SPY, QQQ, etc.) for gamma leverage. Position monitor enforces -50% loss cut and 50% trail-from-peak. | Premium = max loss ($50 cap) | Intraday high-variance |

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
| **9:00 AM** | Pre-market scan | Scan ~10,000 stocks, return top 25 candidates by gap + volume. Scan separate options universe (top 30 by volume + liquidity). |
| **9:30 AM** | Market open | Cache equity, sync positions, analyze SPY/QQQ/VIX market regime |
| **9:45 AM - 3:45 PM** | Scan & evaluate (every 15 min) | Run all 8 stock + 9 options strategies, rank signals, submit orders |
| **3:30 PM** | Options expiry check | Close options expiring today/tomorrow |
| **3:50 PM** | EOD close | Force-close day trades |
| **4:05 PM** | EOD review | Log P&L, save equity snapshot |
| **4:10 PM** | EOD analysis | Loss-pattern detection + parameter-optimizer sweep (writes to `analysis/` tables) |
| **5:00 PM** | ML training | Retrain signal-quality classifier on latest trade outcomes; promote if it clears the quality gate |
| **Every 60 sec** | Position sync | Reconcile broker stock positions with local DB |
| **Every 5 min** | Options position sync | Reconcile broker options positions with local DB |
| **Every 5 min** | 0-DTE position monitor | Enforce -50% loss cut and 50% trail-from-peak on intraday options |

---

## Configuration

All tunable parameters live in `config/settings.yaml`. API keys and SMTP credentials come from `config/.env`.

```yaml
account:          # Starting capital, position limits (25% max), daily loss limit
pdt:              # Day trade budget (3), reserve (1), min conviction for day trades
scanner:          # Price range ($2-50), volume filters, gap threshold, max candidates
  options_universe: # Separate filters for options-eligible stocks ($10-500, >1M volume)
strategies:       # Enable/disable and tune each of the 17 strategies
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
│   ├── .env.example                    # Secrets template (safe to commit)
│   └── .env                            # API keys + SMTP creds (git-ignored)
├── src/ai_trade/
│   ├── main.py                         # TradingBot — central orchestrator
│   ├── _version.py                     # Single source of truth for version
│   ├── config.py                       # YAML + .env config loader
│   ├── clients.py                      # Alpaca client factory (singleton pattern)
│   ├── utils.py                        # Retry logic, Greek extraction
│   ├── state_persistence.py            # Save/restore bot state across restarts
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
│   │   ├── exit_planner.py             #   Adaptive exit plan with S/R-anchored stops
│   │   ├── weighter.py                 #   Adaptive strategy weighting
│   │   └── options/                    #   9 options strategies
│   │       ├── base.py                 #     Shared utilities: filter_contracts, enrich_greeks, select_by_delta
│   │       ├── credit_put_spread.py    #     Bull put spread (defined risk income)
│   │       ├── debit_call_spread.py    #     Bull call spread (defined risk directional)
│   │       ├── long_call.py            #     Directional call buying
│   │       ├── long_put.py             #     Directional put buying
│   │       ├── cash_secured_put.py     #     Premium selling on support
│   │       ├── covered_call.py         #     Income on held shares
│   │       ├── covered_straddle.py     #     Double-premium selling (disabled for small accounts)
│   │       ├── momentum_options.py     #     Short-dated momentum plays
│   │       └── zero_dte.py             #     Same-day expiration gamma plays (SPY/QQQ/etc.)
│   ├── risk/                           # Risk management
│   │   ├── pdt_manager.py              #   Day-trade tracking + Alpaca-authoritative count sync
│   │   ├── smart_pdt.py                #   Dynamic PDT nudges (day-of-week, slots remaining, recent WR)
│   │   ├── dynamic_risk.py             #   Runtime sizing multipliers (conviction, streak, regime, DD)
│   │   ├── position_sizer.py           #   Fixed-fractional sizing with risk-aware minimum
│   │   └── risk_manager.py             #   Portfolio-level gates (heat, concentration, daily loss)
│   ├── execution/                      # Order execution
│   │   ├── order_manager.py            #   Stock bracket orders with price adaptation
│   │   └── options_order_manager.py    #   Multi-leg options orders
│   ├── sentiment/                      # Market intelligence
│   │   ├── market_regime.py            #   SPY/QQQ/VIX regime analysis (5 regimes)
│   │   ├── news_sentiment.py           #   Keyword-weighted news scoring
│   │   ├── earnings_guard.py           #   Block entries around earnings dates
│   │   ├── economic_calendar.py        #   Macro event awareness (FOMC, CPI, NFP)
│   │   └── event_classifier.py         #   Classify news headlines into event types
│   ├── monitoring/                     # Observability
│   │   ├── database.py                 #   SQLite persistence (indexed tables)
│   │   ├── performance.py              #   P&L metrics (Sharpe, drawdown, win rate)
│   │   ├── console.py                  #   Pretty console output formatters
│   │   ├── notifier.py                 #   Email alerts (SMTP, background threads)
│   │   ├── cycle_timer.py              #   Scan-cycle performance instrumentation
│   │   ├── decision_logger.py          #   Per-signal audit trail (reject reasons)
│   │   └── logger.py                   #   Structured logging (JSON + console)
│   ├── analysis/                       # Post-trade learning
│   │   ├── loss_patterns.py            #   Cluster losing trades by feature similarity
│   │   ├── post_trade.py               #   Attribution of realized P&L to strategy/regime
│   │   ├── parameter_optimizer.py      #   Search over strategy params for performance lift
│   │   ├── parameter_specs.py          #   Tunable-parameter declarations
│   │   └── market_prediction.py        #   Short-horizon market direction model
│   ├── scheduler/
│   │   └── jobs.py                     #   APScheduler cron jobs (20+ scheduled tasks)
│   ├── ml/                             # Machine learning
│   │   ├── features.py                 #   Canonical 15-column feature vector
│   │   ├── trainer.py                  #   GradientBoosting training (time-ordered 80/20 split)
│   │   └── predictor.py                #   Live inference + blend weight ramp (0→0.5 over 200 trades)
│   └── backtest/                       # Backtesting
│       ├── engine.py                   #   Walk-forward simulator (ML feature capture at fill)
│       ├── options_pricing.py          #   Black-Scholes pricing
│       └── runner.py                   #   CLI: --train-ml / --capital / --full-universe
├── data/                               # SQLite DBs + ML universe (auto-created, git-ignored)
│   ├── ai_trade.db                     #   Live/paper trade DB
│   ├── backtest_ml.db                  #   Backtest trades used to train ML
│   └── ml_training_universe.txt        #   283 liquid stocks for ML training backtests
├── models/                             # Trained ML models (joblib); signal_quality_v3 shipped
├── logs/                               # Log files (auto-created, git-ignored)
├── docs/                               # In-depth documentation (incl. ROADMAP_TO_LIVE.md)
├── scripts/                            # Smoke tests per V2 phase (manual, not run in CI)
└── tests/                              # Pytest suite
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
| [ROADMAP_TO_LIVE.md](docs/ROADMAP_TO_LIVE.md) | **Read before going live.** Eight-phase plan covering baseline validation, guardrails, EV gate + honest sizing, PDT config-driven rewrite, barbell sleeves, prediction-quality upgrades (incl. model promotion pipeline), observability, and pilot. Also catalogs known ML pipeline bugs. |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, data flow, component interactions, design patterns |
| [STRATEGIES.md](docs/STRATEGIES.md) | All 17 strategies — theory, entry/exit conditions, conviction scoring |
| [OPTIONS_GUIDE.md](docs/OPTIONS_GUIDE.md) | Options fundamentals — calls, puts, Greeks, spreads |
| [RISK_MANAGEMENT.md](docs/RISK_MANAGEMENT.md) | Position sizing, loss limits, portfolio heat, PDT, bracket orders |
| [BACKTESTING.md](docs/BACKTESTING.md) | Walk-forward simulator, Black-Scholes pricing, limitations |
| [CONFIGURATION.md](docs/CONFIGURATION.md) | Every settings.yaml parameter with defaults and tuning guidance |
| [V2_PROGRESS.md](docs/V2_PROGRESS.md) | V2.0 13-phase refactor — what shipped, what it replaced |
| [V2_AGENT_BRIEF.md](docs/V2_AGENT_BRIEF.md) | Original V2 agent brief — useful context for incoming contributors |

---

## Dependencies

Declared in [pyproject.toml](pyproject.toml). Installed transitively via `pip install -e .`.

| Package | What It Does |
|---|---|
| `alpaca-py` | Official Alpaca SDK — trading and market data |
| `pandas` | Data manipulation — OHLCV bars stored as DataFrames |
| `numpy` | Numerical backbone (used by pandas, ta, scikit-learn) |
| `ta` | Technical indicators (pure Python, no C dependencies) |
| `scikit-learn` | GradientBoostingClassifier for signal-quality ML |
| `joblib` | Model serialization (`.joblib` payloads in `models/`) |
| `apscheduler` | Background job scheduler for the daily trading schedule |
| `pyyaml` | YAML config file parsing |
| `python-dotenv` | Loads API keys from `.env` files |
| `structlog` | Structured logging — JSON for machines, readable for humans |
| `requests` | HTTP client (news API, Alpaca REST fallbacks) |

Dev dependencies (`pip install -e ".[dev]"`): `pytest`, `pytest-asyncio`.

---

## Changelog

### v2.3.5 — Heat-lockup fix, tracked-position re-protect, PDT framework flip, filter report

Three issues + one feature, found by a "why hasn't the algo traded in days?" investigation:

**1. Heat-lockup root cause (ONDS-style)**
- The algo hadn't ranked or executed a single signal in 4 days despite 172+ ML predictions reaching the aggregator.
- Root cause: an adopted position (ONDS) sat in the DB with `stop_loss = 0` after the v2.3.2 OCO re-protection silently failed. The heat calculation treats `entry - 0 = full position value` as at-risk, so one $919 position alone was 20% of equity vs the 8% heat limit. Every new entry got `trade_rejected reason='portfolio heat 21.36% exceeds limit 8.00%'`.
- Fixed live: re-attached an OCO (stop $10.00, target $13.00) and corrected the DB row. Heat dropped 21.4% → 5.77%.

**2. Structural: `sync_positions` now re-protects tracked-but-naked positions**
- The v2.3.2 adoption only fired for *untracked* positions. A position that was already in the DB with stop_loss=0/NULL was invisible to it.
- New `_reprotect_tracked_position`: every sync cycle, any open DB trade whose stop_loss is 0/NULL is checked against Alpaca. If Alpaca has a live stop, the DB syncs to it. If not, an OCO is submitted (with plain-stop fallback) and the DB row is updated. This closes the loophole that left ONDS naked for 5 days.

**3. PDT framework: `legacy` → `intraday_margin`**
- FINRA Rule 4210 amendment + Alpaca implementation went live 2026-06-04. Flag flipped in config.
- Local PDT manager now no-ops on `can_day_trade()` and `sync_with_alpaca()`. The binding constraint is Alpaca's pre-trade margin checks + our `buying_power`-based sizing. All deprecated-field reads (`daytrade_count`) remain guarded with `getattr` for the 2026-07-06 field removal.

**4. New feature — filter-effectiveness report**
- Daily EOD job logs the execution funnel (`ml_predictions → ranked → executed → order_failed`) plus the **5 tightest near-miss clusters** with examples. Surfaces filters that are "right on the edge" — e.g. `LongCall.price_above_ema50` missing by 0.008% on SMR, `CreditPutSpread.ema_structure` missing by 0.09% on INFY. Read-only — operator decides whether to loosen. No auto-tuning (as discussed, sample is too small).
- Config: `analysis.filter_report_days` (default 7).

### v2.3.4 — Decision-table retention (cap DB growth)

Performance review found the `decisions` table at **270k rows / 110 MB, ~99% `reject`/`near_miss`** — growing 25-34k rows/day at the 5-min eval cadence. Nothing automated reads near-misses (the optimizer, loss-pattern scan, and post-trade analysis don't touch the decisions table; the live cycle summary uses in-memory near-misses), so they're pure ad-hoc-debugging weight and only recent ones matter.

- New `Database.prune_old_decisions(reject_retention_days)`: deletes `reject`/`near_miss` rows older than the window, `VACUUM`s if the delete was large. The meaningful low-volume audit rows (`execute`/`rank`/`exit`/`review`/`ml_predict`/`approve`) are **never pruned**.
- Wired into the EOD analysis job (runs daily after the optimizer sweep).
- Config: `analysis.decision_retention_days` (default 14). Lower it for a tighter cap.

Also documented in the review (not yet changed): the signal→execution funnel is leaky on a capital-constrained account — 651 signals ranked, 23 executed, 530 `order_failed` (mostly `account_funds`/`account_pdt`, mostly pre-fix). The `min_deployable_cash` gate + `buying_power` sizing (v2.3.1) should largely resolve it; **confirm after the next bot restart**.

### v2.3.3 — Backtest trailing-stop fidelity

The backtest's exit logic was using a naive `highest_price × (1 − trailing_stop_pct)` chandelier, while live uses `exit_planner.compute_trailing_stop_long` (conviction-aware breakeven + ATR chandelier + the v2.3.0 profit-tier ratchet). That meant the backtest **couldn't measure the ratchet** — the headline give-back fix — so paper results understated live exit behavior.

Fix:
- `BacktestPosition` now carries `atr` and `conviction`, populated at fill time from the signal's metadata and conviction (both entry paths: next-day fill and intraday fill).
- `_check_exits` calls the live `compute_trailing_stop_long` planner; falls back to the simple % chandelier only when ATR is unknown.
- Backtest exits now mirror live: breakeven trigger, ATR chandelier, and profit-tier ratchet all apply in simulation.

Review notes (not changed, by design):
- **Limit/pullback entries** (v2.3.0) aren't simulated — the backtest fills at next-day open. Faithful intraday-fill modeling is a larger lift; flagged for later.
- **Double indicator computation** (engine pre-enriches, strategies recompute on their slice) is wasteful but mirrors live behavior, so left as-is to avoid divergence.
- **Backtest PDT window** uses 7 calendar days vs live's corrected 5-business-day window — minor, and PDT itself retires 2026-06-04.

### v2.3.2 — Adopt untracked positions (Alpaca is source of truth)

**Problem:** `sync_positions` logged an `untracked_position` warning when Alpaca held a position the DB didn't know about — then did nothing. Untracked positions get no trailing-stop management, no exit logic, and no risk accounting. Two positions (BLIN, CLSK) had drifted out of the DB **and lost their stop-loss legs** — CLSK sat unprotected for ~a week with $2.2k exposure, carrying only a take-profit.

**Fix:** `sync_positions` now **adopts** untracked positions instead of just warning:
- Re-creates the DB trade row (strategy `adopted`, `hold_type=swing` so it's never force-closed as a day trade) so the trailing-stop / exit / risk jobs resume managing it.
- Reads the position's existing protective orders — a sell STOP leg → stop-loss, a sell LIMIT leg → take-profit.
- **If there's no stop leg, it re-protects.** The position's shares are usually 100% held by the lone take-profit, so a standalone stop is rejected (`insufficient qty available`). The fix cancels the existing sell leg(s) to free the shares, then submits a proper **OCO** (stop + target, one-cancels-other). Falls back to a plain stop if the OCO is rejected — downside protection takes priority over the target.

Runs every minute via `job_sync_positions`, so a position that drifts out of the DB (broken bracket, missed fill, manual trade, failed trailing-stop replace) self-heals within a cycle instead of going unmanaged. Live cleanup on ship: BLIN and CLSK re-adopted and re-protected with OCO brackets (CLSK stop $15.44 below its $15.92 entry, locking the +$215 gain against a round-trip).

New order-manager surface: `_adopt_untracked_position()`; imports `StopOrderRequest`, `QueryOrderStatus`.

### v2.3.1 — Holiday gate, cash-account settlement handling, stale-order cleanup

Three production bugs found during the paper run, all fixed:

**1. Holiday trading**
- The scheduler's `day_of_week="mon-fri"` cron fired on market holidays (Memorial Day, July 4th, etc.) because those are weekdays. The bot tried to trade on a closed market.
- Added Alpaca-clock-based gates (clock knows the holiday calendar):
  - `_market_open()` — intraday jobs (full_scan, evaluate, trailing_stops, pending_entries, zero_dte_monitor) skip when `clock.is_open` is False.
  - `_is_trading_day()` — pre-open jobs (premarket_scan, market_open) run before 9:30 (when `is_open` is False even on normal days), so they check "does the market open later today?" instead.
  - Clock result cached 60s so a burst of same-minute jobs makes one API call.

**2. Cash-account settlement / unsettled funds**
- The paper account is a **cash account** (`multiplier: 1`, `daytrading_buying_power: 0`). In a cash account, same-day sale proceeds are unsettled for T+1 and excluded from `buying_power`. Trying to deploy them gets rejected by Alpaca as "insufficient buying power" or "PDT protection" — the logs showed **325 + 169** such rejections.
- Position sizing now uses `min(cash, buying_power)` instead of raw `cash`. `buying_power` already excludes unsettled funds and open-order holds, so the bot stops sizing against money it can't actually deploy.
- New `account.min_deployable_cash` gate (default $50): when settled buying power is below it, the bot skips new entries for the cycle instead of firing orders it knows will bounce.

**3. Orphaned stale limit orders**
- `job_manage_pending_entries` iterated the local DB, but limit orders that desynced from the trade rows were invisible to it — two sat unfilled for **up to 4.6 days**, reserving ~$2,000 of buying power past their 30-min TTL.
- Rewrote the job to query **Alpaca's open orders directly**, so every unfilled buy-limit bracket gets managed regardless of DB state. Cancelling the parent bracket auto-frees its reserved buying power.

> **Note on account type:** PDT (3 day-trades / 5 days) applies to *margin* accounts. The current paper account is *cash*, where the real constraint is **settlement (T+1)**, not a day-trade count. The PDT manager still runs but is largely moot here — the binding limit is settled buying power. If the account is switched to margin later, the PDT logic becomes the active constraint again.

**4. PDT-rule retirement forward-compat (FINRA Rule 4210, effective 2026-06-04)**
- FINRA retired the Pattern Day Trader designation on 2026-06-04; Alpaca replaced it with an intraday-margin model (Intraday Buying Power + pre-trade margin checks; 4x intraday BP minimum equity lowered from $25k to $2k). The deprecated fields `pattern_day_trader`, `daytrade_count`, `last_daytrade_count`, `daytrading_buying_power`, `last_daytrading_buying_power` return placeholders until full removal on 2026-07-06.
- **Audit:** the codebase only read `daytrade_count` (in `pdt_manager.py` and one log line). All reads now use `getattr(account, "daytrade_count", 0)` so the 2026-07-06 field removal can't crash the bot.
- **New config `pdt.framework`** (`legacy` | `intraday_margin`, default `legacy`):
  - `legacy` — enforce the day-trade count limit as before.
  - `intraday_margin` — `can_day_trade()` always returns True (no count limit); `sync_with_alpaca()` no-ops without touching the deprecated field. The bot defers entirely to Alpaca's pre-trade margin checks + `buying_power`-based sizing.
- **Action for operator:** flip `pdt.framework: intraday_margin` on/after 2026-06-04. If switching the account to margin to use 4x intraday BP, note the new risk is **Intraday Margin Deficit (IMD) calls** — the `min(cash, buying_power)` sizing keeps the bot from over-leveraging into one.

### v2.3.0 — Pullback-entry limit orders + 5-min intraday cadence

**Better entry pricing (the headline feature)**
- Four breakout/reclaim strategies (momentum, orb, vwap, ema_crossover) now place **LIMIT entries at meaningful support** instead of market-buying at the breakout high.
  - **momentum** → EMA-20 (or close × 0.99 if EMA is too far away)
  - **orb** → OR high (the just-broken resistance, now natural support for the retest)
  - **vwap** → VWAP + 1 tick (wait for the reclaim retest)
  - **ema_crossover** → slow EMA (the trend line price tends to retest)
- Signal carries the limit price; if the support level is below the stop, the strategy falls back to market entry instead.
- Stop/target legs are re-anchored to the limit price so the R:R ratio is preserved at the actual fill price.
- `pullback` and `mean_reversion` left unchanged — they already enter on dips by design.

**Adaptive entry management (every 5 min)**
- New `job_manage_pending_entries` reviews unfilled limit-entry orders. Per-order it decides:
  - Drift ≥ +1.5% above limit → **cancel + re-submit market** ("chase" — the setup is still valid and we'd be missing it)
  - Drift ≤ −5% below limit → **cancel** (setup broke, mark trade `cancelled_breakdown`)
  - Age ≥ 30 min unfilled → **cancel** (mark `cancelled_stale`)
  - Otherwise → wait, recheck next tick
- Decision rows: `entry_chased_to_market`, `entry_cancelled` (breakdown / stale) for post-trade audit.

**Intraday data resolution: 15-min → 5-min**
- Live fetch (`main.py`) now pulls 5-min bars for ORB and VWAP. ORB's 30-min opening range gets a 6-bar lookback (vs 2 bars at 15-min) — much less noisy level computation. VWAP's dip-detection becomes tighter.
- `_INTRADAY_BAR_MINUTES = 5` in `strategy/orb.py` keeps the opening-range bar count in sync.

**Scan vs evaluate split (decoupled cadences)**
- The old `job_scan_and_evaluate` did two very different jobs each tick: (1) re-scan the whole tradeable universe for new candidates — expensive, thousands of API calls — and (2) run strategies on the resulting candidates — cheap, ~25 symbols. They now have separate jobs at separate cadences.
- **`job_full_scan` — every 15 min**: universe scan only. Rebuilds `self._candidates` from the screener filters.
- **`job_evaluate` — every 5 min**: strategy evaluation on the existing candidate list. No universe re-scan, no scanner work.
- Adds `_scan_in_progress` guard so a mid-flight scan can't be read with a torn candidate list.
- Catch-up path on bot startup still uses `job_scan_and_evaluate` (the combined original) so a freshly-restarted bot doesn't have to wait for the next 5-min evaluate.
- New config key: `schedule.evaluate_interval_minutes: 5` (default). `scan_interval_minutes` reverts to its original meaning (full-scan cadence, default 15).
- Net effect: full scan runs ~1/3 as often (~13 times/day vs 39 at 10-min combined cadence) → much cheaper, while signals fire ~2× faster than before (5 min vs 10).

**Why this matters (from the live data)**
- AHMA case: entry $1.59, MFE +17.6% to $1.87, then round-tripped to a stop at $1.57 (-1.2%). The trailing logic only had breakeven and ATR-based chandelier; chandelier never engaged above entry because 2× ATR > MFE on penny stocks. The v2.3.0 profit-tier ratchet (also shipped this version — see below) would have stopped at $1.72 (+8%) instead, turning −$11 into +$83.
- Combined with limit entries, the math is: better entry price → more shares for the same dollar risk → bigger absolute profit on winners that work.

**Profit-tier trailing-stop ratchet** (`strategy/exit_planner.py`)
- Added percentage-based MFE tiers alongside the existing breakeven and chandelier proposals. The tightest valid stop wins, so chandelier still rules where ATR is small (large-cap stocks) and the new ratchet takes over where chandelier never engages (cheap stocks).
- Tier table: MFE 3% → lock 1%, 5% → 2%, 10% → 5%, 15% → 8%, 25% → 15%.

### v2.2.0 — Pre-live hardening (PDT, options visibility, conviction integrity, backtest fidelity)

This release rolled up a session of diagnostic fixes uncovered while preparing the $5000 paper run:
- **PDT cutoff bug**: rolling window walked back 5 weekdays (6-business-day window) instead of 4 (5-business-day). Trades from the prior week stayed counted. Fixed in `risk/pdt_manager.py`.
- **In-flight day-trade gate**: `can_day_trade()` now also counts open same-day DAY/ADAPTIVE positions, not just historical round-trips. Prevents over-subscribing within a session.
- **Conviction mutation bug**: `_submit_stock_signal` was mutating `sig.conviction` through every modifier, so the trades table stored the post-regime-modifier value (often clamped to 1.0). Now uses a local `conviction` variable; `sig.conviction` keeps the strategy-level value for ML training.
- **Options bar-window**: `_evaluate_options` fetched 60 calendar days (~42 trading days), but 8 of 9 options strategies guard with `len(df) < 52`. Most strategies silently skipped every cycle. Bumped to 120 days; only ZeroDTE was previously running.
- **Order-failure visibility**: missing `decisions.log_reject` in the `order_failed` path now writes a row with `error_type` (`account_pdt`, `symbol_halted`, `account_funds`, etc). Account-level errors no longer blacklist the symbol.
- **Momentum gap filter**: rejects entries when overnight gap exceeds `max_gap_pct: 0.15` (KALV-style runaway gappers).
- **Momentum ADR floor**: hard reject below 0.5% ADR (sub-cent ATR setups like EM @ $1.19 where the stop is sub-tick).
- **VWAP tick-aware floor**: `max(vwap × 1.001, vwap + $0.01)` so the floor isn't sub-tick on cheap stocks.
- **ml_predictions outcome backfill**: trade-close now writes `actual_outcome` on the matching prediction so the trainer can compare predicted vs realized.
- **Backtest engine**: wired in the live `StrategyWeighter` via an in-memory `_BacktestDBAdapter` (no more "all strategies at full weight forever"); added 5-min intraday support for vwap/orb with per-bar evaluation; disabled bb_squeeze (22% WR / -$497 P&L across 18 trades).
- **Profit-tier ratchet** scaffolding in `exit_planner.py` (active in v2.3.0; merged here so the trail logic includes it from this version forward).

### v2.1.1 — Logging hardening + pre-live roadmap

- **PDT count-mismatch spam fix** — `pdt_count_mismatch` warnings now emit only when the (Alpaca, local) pair changes. Stale-row repeats downgraded to `pdt_count_mismatch_repeat` at debug level. Previously flooded logs on every sync cycle.
- **News-scan error visibility** — `news_scan_failed` now emits `error_type`, a fallback `repr(e)`, and `symbols_count` so `concurrent.futures.TimeoutError` (which has an empty `str`) is no longer a blank log line.
- **Alpaca read-path resilience** — transient network errors on `get_account` / position reads are retried with exponential backoff; connection-timeout log events downgraded from warning to info to reduce noise.
- **`config/.env.example` added** — template committed alongside the git-ignored real `.env` so fresh clones have the keys they need to fill in.
- **[docs/ROADMAP_TO_LIVE.md](docs/ROADMAP_TO_LIVE.md) added** — eight-phase plan for $2k live go-live, including config-driven PDT forward-compat (for the proposed FINRA $25k → $2k rule change), an EV gate with honest cost model, a three-layer position sizer (not Kelly — see the roadmap for rationale), barbell sleeves, and a full model-promotion pipeline (Gate 1 quality gate + optional Gate 2 shadow mode + rollback of the last 3 retired models).
- **Known ML pipeline bugs catalogued** in the roadmap: (A) predictor startup logs silently dropped because `setup_logging()` runs after `SignalQualityPredictor` is instantiated; (B) `insert_ml_prediction()` is orphaned — the `ml_predictions` table is empty; (C) the trainer promotes every new model to `is_active=1` with no quality check, so a regression can silently replace a working model.

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
