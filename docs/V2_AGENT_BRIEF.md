# V2 Agent Brief: Self-Learning Algorithmic Trading Bot

## Mission

Transform the current rule-based trading bot into a **self-learning, adaptive system** that maximizes profits through intelligent strategy selection, dynamic risk tolerance, predictive market analysis, and real-time news/event interpretation. The bot should be aggressive when its own analysis supports high conviction, learn from every trade (wins AND losses), and continuously improve its Sharpe ratio and risk-adjusted returns.

**Current stats (baseline to beat):**
- Sharpe ratio: 3.12
- Win rate: 43.75%
- Profit factor: 1.55
- Total P&L: +$18.76 on $500 account (10 days)
- 20 total trades, 6 day trades, 1 open position
- Only 2 of 16 strategies producing signals (ORB, VWAP)

**Target:** Higher Sharpe, higher absolute returns, ALL strategies firing (including 0DTE options), better risk/reward per trade, aggressively maximize returns. High risk is acceptable when the algo has proven to itself through its own track record that it can trust its predictions.

**Philosophy:** This is a machine-learned trading system. Every component — strategy selection, parameter tuning, risk tolerance, position sizing — should be driven by ML models trained on the bot's own trade history. Static rules are starting points that get replaced by learned behavior as data accumulates. The algo earns the right to take bigger risks by demonstrating consistent prediction accuracy.

---

## Current Architecture (What Exists)

### Tech Stack
- **Language:** Python 3.11+
- **Broker:** Alpaca (paper trading, soon live)
- **Database:** SQLite (`data/ai_trade.db`)
- **Scheduling:** APScheduler (15-min scan intervals, 9:45 AM - 3:45 PM ET)
- **Data:** Alpaca market data API (daily bars, minute bars, options chains, news)

### Project Structure
```
src/ai_trade/
  main.py                          # TradingBot orchestrator — owns all components, runs scheduled jobs
  config.py                        # Loads settings.yaml into SimpleNamespace
  clients.py                       # Alpaca API client singletons
  utils.py
  _version.py

  scanner/
    screener.py                    # 3 scanners: momentum (gap/RVOL), mean_reversion (RSI oversold), VWAP universe

  strategy/
    base.py                        # BaseStrategy ABC, Signal dataclass, HoldType enum (DAY/SWING/ADAPTIVE)
    signal.py                      # SignalAggregator — collects signals from all strategies, ranks, builds execution queue
    weighter.py                    # StrategyWeighter — adjusts conviction multipliers based on historical win rate/PF/avg PnL
    mean_reversion.py              # RSI oversold bounce (SWING)
    momentum.py                    # Breakout on volume (ADAPTIVE)
    vwap.py                        # VWAP reclaim/bounce (DAY)
    ema_crossover.py               # EMA 9/20 crossover (SWING)
    macd_divergence.py             # Bullish MACD divergence (SWING)
    bb_squeeze.py                  # Bollinger Band squeeze breakout (ADAPTIVE)
    orb.py                         # Opening Range Breakout (DAY)
    pullback.py                    # Trend pullback to EMA (SWING)
    options/
      base.py                      # BaseOptionsStrategy ABC
      long_call.py, long_put.py    # Directional options
      credit_put_spread.py         # Income strategy
      debit_call_spread.py         # Defined-risk bullish
      cash_secured_put.py          # Premium collection (disabled for $500)
      covered_call.py              # Premium on holdings (disabled for $500)
      covered_straddle.py          # Disabled
      momentum_options.py          # Short-DTE momentum play

  sentiment/
    market_regime.py               # Classifies STRONG_BULL/BULL/NEUTRAL/BEAR/STRONG_BEAR from SPY/QQQ/VIX
    news_sentiment.py              # Keyword-based news scoring (bullish/bearish keywords with weights)

  risk/
    risk_manager.py                # Daily loss limits, portfolio heat, concentration checks
    position_sizer.py              # Fractional Kelly sizing
    pdt_manager.py                 # PDT rule enforcement (3 day trades per 5 business days, with 1 reserve)

  execution/
    order_manager.py               # Stock bracket orders (entry + stop + target)
    options_order_manager.py       # Options order submission

  monitoring/
    database.py                    # SQLite: trades, signals, day_trades, daily_snapshots, options_trades, scanner_results
    logger.py                      # Structured logging (structlog)
    performance.py                 # PerformanceTracker — Sharpe, drawdown, win rate, P&L
    console.py                     # Rich console formatting
    notifier.py                    # Push notifications (not yet wired)

  data/
    historical.py                  # fetch_bars_multi() — batch historical bar fetching
    indicators.py                  # add_ema, add_rsi, add_atr, add_macd, add_bb, add_vwap, add_all
    streaming.py                   # WebSocket streaming (not actively used)
    options_chain.py               # Options chain + snapshot fetching

  scheduler/
    jobs.py                        # APScheduler job definitions
  backtest/
    engine.py, runner.py           # Backtesting framework
    options_pricing.py
```

### Config
All parameters live in `config/settings.yaml`. See the full file for current values. Key sections: `account`, `pdt`, `scanner`, `strategies` (16 strategies), `sentiment`, `options`, `strategy_weighting`, `risk`, `schedule`.

### How It Works Today (Flow)

1. **Pre-market (9:00 AM):** Analyze market regime (SPY/QQQ/VIX → breadth score → regime classification). Sets conviction/position-size modifiers for the day.
2. **Every 15 min (9:45 AM - 3:45 PM):**
   - Run 3 scanners (momentum, mean_reversion, VWAP) → ~44 candidates
   - Fetch 60 days daily bars + 2 hours intraday bars for all candidates
   - Add technical indicators (EMA, RSI, MACD, BB, ATR, VWAP)
   - Scan news sentiment (keyword-based) for all candidates
   - Run ALL 8 stock strategies on ALL candidates → collect Signals
   - Run ALL 8 options strategies on options universe candidates
   - SignalAggregator ranks signals: swing first (free), then day trades (costs PDT slot)
   - StrategyWeighter multiplies conviction by historical performance weight (0.3x - 2.0x)
   - Market regime modifier adjusts conviction (+30% strong bull to -70% strong bear)
   - News sentiment modifier adjusts conviction (+30% to -50%)
   - Risk manager gates each trade (daily loss, portfolio heat, concentration, cash)
   - Position sizer calculates shares (fractional Kelly)
   - OrderManager submits bracket orders (entry + stop-loss + take-profit)
3. **EOD (3:50 PM):** Close all day-trade positions. (4:05 PM) Review and snapshot.

### Key Limitations to Fix

1. **Static strategy parameters** — All thresholds (RSI, volume, ATR multipliers, tolerances) are hardcoded in YAML. The bot can't adapt them based on what's working.
2. **Primitive news analysis** — Simple keyword matching. Can't understand context, causality, or geopolitical implications. "War ending" and "war escalating" are just keyword matches, not understood events.
3. **No predictive capability** — The bot reacts to current indicators. It doesn't predict where price is going based on patterns, correlations, or learned behavior.
4. **PDT handling is binary** — Either you can day trade or you can't. No intelligence around timing, saving slots for better opportunities, or converting day-trade signals to swing when PDT is tight.
5. **Strategy weighter is too slow** — Needs 10 closed trades per strategy before adjusting (burn-in). With only 20 trades total across ALL strategies, most never get weighted.
6. **No loss analysis** — The bot records losses but doesn't analyze WHY trades failed or adjust behavior. Same mistakes repeat.
7. **Scanner-strategy mismatch** — Gap/momentum scanners feed swing strategies that need completely different candidate profiles. Many strategies get zero viable candidates.
8. **Too conservative** — Multiple overlapping conviction floors (0.45 post-sentiment, 0.55 for swing, 0.80 for day trade) stack with regime modifiers to kill most signals.

---

## What Needs to Be Built (V2 Requirements)

### 1. Intelligent Strategy Optimizer

**Goal:** The bot should dynamically tune its own strategy parameters based on what's working in the current market.

**Current state:** `StrategyWeighter` in `src/ai_trade/strategy/weighter.py` only adjusts a conviction multiplier (0.3x-2.0x) based on win rate, profit factor, avg PnL, and recency. It doesn't touch the actual strategy parameters.

**What V2 needs:**
- **Parameter optimization engine** — After each trade closes, analyze whether tighter or looser thresholds would have improved the outcome. For example:
  - If RSI-based entries keep getting stopped out, maybe the RSI threshold is too aggressive
  - If breakout trades keep winning when volume is 3x+ but losing at 1.5x, adjust the volume threshold upward
  - If take-profit targets are rarely hit but stop-losses are, tighten the R:R ratio
- **Rolling optimization window** — Use the last N trades (not all-time) to tune parameters. Market conditions change; what worked 3 months ago may not work now.
- **Per-regime parameter sets** — Different parameters for bull vs. bear vs. neutral markets. A momentum breakout strategy should behave differently in a strong bull vs. a choppy neutral market.
- **Strategy selection intelligence** — Don't just weight strategies; actively enable/disable them based on regime + recent performance. If pullback hasn't fired in 30 days, investigate why (is the market not pulling back, or is the threshold wrong?).
- **Preserve the current weighter's logic** as a component — it's sound, just too limited in scope.

### 2. Advanced Market Prediction

**Goal:** Move from reactive (indicators tell us what happened) to predictive (the bot anticipates what's likely to happen next).

**Current state:** `MarketRegimeAnalyzer` in `src/ai_trade/sentiment/market_regime.py` classifies the market into 5 regimes using SPY/QQQ/VIX technicals. It's purely backward-looking.

**What V2 needs:**
- **Pattern recognition** — Identify recurring price patterns (double bottoms, head & shoulders, ascending triangles, bull flags) that historically precede moves. Use the bot's own trade history to learn which patterns are most predictive in the current market.
- **Correlation analysis** — Track correlations between:
  - Sector moves (if semis are rallying, which other sectors follow?)
  - VIX changes and subsequent SPY direction
  - Pre-market gap behavior and intraday resolution
  - Options flow (unusual volume, put/call ratio) and price direction
- **Multi-timeframe analysis** — Currently only daily and 2-hour intraday. Add:
  - Weekly bars for trend context
  - 5-minute bars for entry timing
  - Pre-market/after-hours data for gap prediction
- **Momentum prediction** — Use rate of change of indicators (not just their values) to predict where they'll be in 1-3 bars. If RSI is at 55 and rising at +3/bar, predict it'll be at 64 soon and act on that.
- **Machine learning or statistical models** — Consider lightweight models (gradient boosting, logistic regression) trained on the bot's own trade outcomes to predict signal quality. Features: technical indicators, volume profile, time of day, day of week, market regime, sector strength.

### 3. Real-Time News & Event Intelligence

**Goal:** Understand the MEANING of news, not just keywords. React appropriately to geopolitical events, policy changes, earnings, and macro data.

**Current state:** `NewsSentimentScanner` in `src/ai_trade/sentiment/news_sentiment.py` does keyword matching against Alpaca's news API. "beats" = +1.5 bullish, "downgrade" = -1.5 bearish. No understanding of context.

**What V2 needs:**
- **LLM-powered news interpretation** — Use an LLM API (Claude, GPT, etc.) to:
  - Summarize news articles into actionable trading signals
  - Understand causality: "Iran peace deal collapses" → defense stocks up, oil up, broad market down
  - Understand temporal context: "Fed signals rate cut in September" → bullish now, but what about inflation data next week?
  - Understand magnitude: "CEO steps down" for a $2T company is different from a $500M company
- **Real-time event feeds** — Monitor beyond just Alpaca news:
  - Social media / X (Twitter) for breaking news (presidential tweets, company announcements)
  - Economic calendar (CPI, jobs report, FOMC decisions) — know WHEN events happen and position accordingly
  - Earnings calendar — avoid or exploit earnings dates
  - SEC filings (13F, insider transactions) for institutional signals
- **Event-driven strategy triggers** — When a major event is detected:
  - Immediately re-evaluate all open positions (should we exit before impact?)
  - Generate new signals based on the event (if war de-escalation → buy defense put spreads, buy consumer/travel calls)
  - Adjust risk parameters temporarily (increase position sizes if high conviction on direction, decrease if uncertainty spikes)
- **Geopolitical event mapping** — Build or integrate a knowledge base of how different event types historically affect different sectors and asset classes. Example mappings:
  - War de-escalation → defense stocks down, oil down, travel/consumer up, VIX down
  - Trade tariff announcement → affected sector down, domestic competitors up
  - Fed rate cut → banks mixed, REITs up, growth stocks up, dollar down
  - Tech antitrust action → target stock down, competitors up

### 4. Smart PDT Management

**Goal:** PDT slots are a scarce resource ($500 account, 3 slots per 5 days). Use them intelligently, not just first-come-first-serve.

**Current state:** `PDTManager` in `src/ai_trade/risk/pdt_manager.py` tracks usage and has a binary gate (`can_day_trade()`). `SignalAggregator` requires 0.80+ conviction for day trades. 1 slot is permanently reserved.

**What V2 needs:**
- **PDT slot value estimation** — Before spending a PDT slot, estimate the expected value of waiting. If it's Monday morning with 2 slots left, the bot should be more selective than if it's Friday afternoon with 2 slots (they'll expire unused over the weekend).
  - Track the average EV of day-trade signals by time of day and day of week
  - Set a dynamic conviction threshold: higher early in the week (save slots for better opportunities), lower later (use it or lose it)
- **Day-to-swing conversion** — When PDT is tight, automatically convert DAY/ADAPTIVE signals to SWING by:
  - Widening the stop loss (swing trades need more room)
  - Moving the take-profit target further out
  - Adjusting position size down (swing trades have more overnight risk)
  - Only convert if the setup also works as a multi-day hold (don't swing-trade a 30-minute ORB)
- **Slot reservation for high-probability setups** — If the bot knows that ORB signals at 9:45 AM have a 65% win rate but VWAP signals at 2:00 PM have an 80% win rate, reserve a slot for the afternoon.
- **Weekly PDT budget planning** — At the start of each week, plan how to allocate the 3 slots across the 5 trading days based on expected opportunity (e.g., Monday after a weekend gap often has better setups than Thursday).

### 5. Self-Learning Trade Analysis

**Goal:** After every trade closes, the bot should analyze what went right or wrong and adjust its behavior. Over time, it should make fewer of the same mistakes.

**Current state:** `PerformanceTracker` in `src/ai_trade/monitoring/performance.py` computes aggregate stats (Sharpe, drawdown, win rate). `StrategyWeighter` adjusts conviction multipliers. Neither analyzes individual trades or learns patterns.

**What V2 needs:**
- **Post-trade analysis engine** — For every closed trade, record and analyze:
  - **Entry quality:** Did price continue in our direction after entry? How far did it go before reversing? Was our entry timing optimal or could we have gotten a better fill?
  - **Stop-loss analysis:** If stopped out, was the stop too tight (price reversed and continued in our direction) or too loose (took too much loss before exiting)?
  - **Take-profit analysis:** If target hit, did price continue beyond it (left money on the table) or reverse immediately after (good exit)?
  - **Holding period:** Did the trade need more or less time than expected?
  - **Market context:** What was the regime when we entered? Did the regime change during the trade? Did a news event invalidate the thesis?
  - **Signal quality vs outcome:** Track conviction at entry vs actual P&L. Are high-conviction signals actually more profitable? (If not, the conviction model needs fixing.)
- **Pattern recognition in losses** — Cluster losing trades to find common factors:
  - Same time of day? (e.g., always losing on 9:45 entries → probably chasing the open)
  - Same sector? (e.g., tech momentum trades keep failing → sector rotation happening)
  - Same market regime? (e.g., pullback trades fail in NEUTRAL → need stronger trend)
  - Same indicator state? (e.g., losing when RSI > 65 → entering too late in the move)
- **Adaptive parameter adjustment** — Feed the post-trade analysis back into strategy parameters:
  - If stop-losses are consistently too tight (stopped out then price reverses), widen the ATR multiplier by 0.1
  - If entries are consistently too early (RSI oversold but keeps going lower), tighten the RSI threshold by 2 points
  - Changes should be small and incremental — not wild swings
  - Track every parameter change and its impact on subsequent trades (did the change help?)
- **Trade journal with reasoning** — Store human-readable summaries of each trade's thesis and outcome for review. Example: "Entered AAPL momentum breakout at $185, conviction 0.72, RVOL 2.3x, RSI 62. Stopped out at $182 (-1.6%) because broad market sold off (SPY -1.2%) despite stock-level thesis being sound. Lesson: Add market momentum check to momentum entries."

### 6. Dynamic Risk Tolerance

**Goal:** Risk tolerance should scale with the bot's own conviction and recent performance, not be a static config value.

**Current state:** Fixed values in `config/settings.yaml`: `max_risk_per_trade_pct: 0.02`, `stop_loss_pct: 0.03`, `daily_loss_limit_pct: 0.05`, `max_portfolio_heat_pct: 0.06`.

**What V2 needs:**
- **Portfolio-percentage-based position limits** — Max open positions should be determined by total portfolio allocation, not a fixed count. Each position consumes a percentage of the portfolio (based on conviction and risk). The bot can hold as many positions as the portfolio can support without exceeding total allocation limits.
  - Base max allocation per position: 25% of portfolio
  - On **>90% conviction**: the bot can **override position limits** — allocate up to 40-50% of portfolio to a single trade, even if it means exceeding the normal max position count. This is the "high conviction override" — the algo has earned this right by proving its prediction accuracy.
  - Total portfolio exposure cap: 100% in normal mode, up to 150% in strong conviction mode (using margin if available)
- **Conviction-based position sizing** — A 0.90 conviction signal should get a meaningfully larger position than a 0.55 signal. Currently conviction affects ranking but position sizing is mostly flat (fractional Kelly with a 0.25 fraction). Scale it:
  - 0.50-0.60 conviction → 0.5x base size
  - 0.60-0.75 conviction → 1.0x base size
  - 0.75-0.85 conviction → 1.5x base size
  - 0.85-0.95 conviction → 2.0x base size
  - **>0.90 conviction → can override max_open_positions and max_position_pct** — if the ML model's recent accuracy on similar setups is >65%, allow up to 50% of portfolio on a single trade
  - Cap scales with proven accuracy: 35% base, up to 50% only after the bot demonstrates consistent wins at high conviction levels
- **Performance-scaled aggression** — When the bot is on a winning streak (positive expectancy over last 10 trades), gradually increase risk tolerance. When on a losing streak, decrease it. This is basically the "anti-tilt" mechanism.
  - Winning streak (>60% win rate, last 10 trades) → increase max_risk_per_trade by 0.5%
  - Losing streak (<35% win rate, last 10 trades) → decrease max_risk_per_trade by 0.5%, minimum 1%
  - Neutral → use base values
- **Regime-aware risk** — The current regime modifier adjusts position SIZE, but it should also adjust:
  - Stop-loss width (tighter in uncertain markets, wider in trending)
  - Take-profit targets (wider in strong trends, tighter in choppy)
  - Number of concurrent positions (fewer in bear, more in strong bull)
  - Options risk allocation (more in bull, less in bear)
- **Drawdown circuit breakers** — Multiple tiers:
  - -3% daily → reduce position sizes by 50% for rest of day
  - -5% daily → stop opening new positions for the day (current behavior)
  - -8% weekly → reduce all parameters to minimum for the rest of the week
  - -15% from peak → full halt, notify user, require manual restart
- **Asymmetric risk** — Be willing to take MORE risk on setups where the R:R ratio is heavily skewed in our favor. A trade with 1:5 risk:reward at 0.65 conviction is better than 1:2 at 0.80 conviction. Factor expected R:R into sizing, not just conviction.

### 7. 0DTE and Aggressive Options Strategies

**Goal:** Add same-day expiration (0DTE) options as a core strategy. These are high-risk, high-reward plays that can return 100-500%+ in hours. The ML model should learn when 0DTE plays have the highest expected value.

**Current state:** `momentum_options.py` has `min_dte: 5`, explicitly avoiding short-dated options. No 0DTE capability exists.

**What V2 needs:**
- **0DTE strategy module** — A new strategy specifically for same-day expiration options on SPY, QQQ, and high-volume individual stocks. These are fundamentally different from longer-dated options:
  - Theta decay is massive — position must move fast or it's worthless
  - Delta moves rapidly as expiration approaches (gamma risk)
  - Tight entries are critical — timing matters more than direction
  - Ideal for high-conviction directional plays after news events, FOMC, or strong technical setups
- **0DTE entry criteria (ML-learned over time, starting points below):**
  - Strong directional signal from multiple timeframes (5-min, 15-min, daily all agree)
  - High relative volume confirming the move
  - VIX in a range that supports the play (not too high = expensive, not too low = no movement)
  - Time of day matters: best 0DTE entries are 9:45-10:30 AM and 2:00-3:00 PM (avoid midday chop)
  - Only on highly liquid underlyings (SPY, QQQ, AAPL, TSLA, AMZN, NVDA, META)
- **0DTE risk management:**
  - Max 5-10% of portfolio per 0DTE trade (these can go to zero)
  - Hard time-based exit: close by 3:30 PM regardless of P&L
  - Trailing stop at 50% of max profit (if up 200%, stop at +100%)
  - Cut losses fast: exit at -40% to -50% (don't let theta eat the rest)
  - The ML model should learn its own optimal stop/target levels based on trade outcomes
- **Scaling 0DTE with proven accuracy:**
  - Start with paper trades only, tiny size (2-3% of portfolio)
  - After 20+ 0DTE trades with >45% win rate and positive expectancy, increase to 5%
  - After 50+ trades with sustained profitability, increase to 10%
  - The bot earns the right to go bigger by proving it can be profitable at smaller sizes
- **All options strategies should be considered** — Don't pre-disable strategies. Even cash-secured puts on cheap stocks, iron condors, strangles — if the ML model identifies a profitable pattern, let it trade. The current approach of disabling strategies because they "seem impossible" for the account size is too conservative. Let the bot figure out what works.

### 8. Smarter Scanning & Candidate Selection

**Current state:** 3 scanners (momentum/gap, mean_reversion/RSI oversold, VWAP universe) produce ~44 candidates. ALL strategies evaluate ALL candidates regardless of fit.

**What V2 needs:**
- **Strategy-specific scanners** — Each strategy should have its own universe of candidates, not share a single pool. Swing strategies need stocks with multi-day patterns. Day strategies need stocks with today's action. Options strategies need high-IV, liquid-options stocks.
- **Dynamic scanner parameters** — Adjust scanner thresholds based on how many candidates they're producing. If the momentum scanner returns 0 candidates, gradually loosen the gap % and RVOL requirements until it finds some. If it returns 50+, tighten up.
- **Sector awareness** — Track which sectors are performing well and bias scanning toward them. If tech is leading, scan more tech names. If a sector rotation is happening (energy → utilities), the scanner should catch it.
- **Liquidity-first filtering** — Remove stocks that LOOK good on technicals but have poor bid-ask spreads or low options open interest. Getting in is easy; getting out at a good price is what matters.

---

### 9. Human-Readable Logging & Full Decision Audit Trail

**Goal:** Every decision the bot makes — from scanning to signal generation to trade execution to exit — must be logged in a way that a human can read and understand, AND stored in the database for ML training and post-analysis.

**Current state:** `src/ai_trade/monitoring/logger.py` uses structlog with JSON file output (`logs/ai_trade.log` at DEBUG, `logs/ai_trade_run.log` at INFO) and colored console output. Logging is sparse — many strategy rejections are silent `return None` with no log. The `signals` table only stores executed signals, not rejections or reasoning.

**What V2 needs:**

- **Decision journal in the database** — A new `decisions` table that captures EVERY decision point:
  ```
  decisions table:
    id, timestamp, decision_type, symbol, strategy, 
    action (consider/reject/signal/approve/execute/skip),
    conviction, reasoning (human-readable text),
    factors (JSON: {rsi: 62, rvol: 2.3, regime: "bull", ...}),
    outcome (filled later: pnl, held_duration, exit_reason)
  ```
  - When scanning: "Considered AAPL: price $185, RVOL 2.3x, gap +3.2% — passes momentum scanner"
  - When evaluating: "AAPL momentum: RSI 62 (pass), MACD hist +0.3 (pass), consolidation 1.8x ATR (reject — too wide)" 
  - When signaling: "AAPL momentum signal: conviction 0.72, entry $185, stop $181.50, target $195.50, R:R 1:2.9"
  - When ranking: "AAPL ranked #2 of 5 signals. Conviction after regime mod (1.1x): 0.79. After news mod (1.05x): 0.83"
  - When sizing: "AAPL: 2 shares @ $185 = $370 (74% of portfolio). Risk: $7.00 (1.4% of equity)"
  - When rejecting: "AAPL rejected by risk manager: portfolio heat 5.8% would exceed 6% limit"
  - When executing: "AAPL bracket order submitted: 2 shares, entry $185, stop $181.50, target $195.50, order_id abc123"
  - When exiting: "AAPL stopped out at $181.50, P&L -$7.00 (-1.4%), held 4 hours. Stop was too tight — price recovered to $188 within 2 hours."

- **Human-readable console output** — The current console shows minimal info. V2 should print a clear, scannable summary each cycle:
  ```
  ═══ Scan 10:00 AM ═══════════════════════════════════════
  Regime: BULL (breadth +0.42) | VIX 18.2 (stable) | PDT: 2/3 used
  Scanned: 52 candidates (23 momentum, 14 mean_rev, 15 vwap)
  
  Signals:
    1. AAPL  momentum     conv=0.83  entry=$185.00  stop=$181.50  target=$195.50  R:R=1:2.9  → QUEUED (2 shares, $370)
    2. TSLA  ema_cross     conv=0.71  entry=$242.00  stop=$235.00  target=$260.00  R:R=1:2.6  → QUEUED (1 share, $242)
    3. NVDA  bb_squeeze    conv=0.68  entry=$950.00  stop=$935.00  target=$990.00  R:R=1:2.7  → REJECTED (exceeds cash)
    4. AMD   vwap          conv=0.62  entry=$165.00  stop=$163.00  target=$170.00  R:R=1:2.5  → SKIPPED (day trade, PDT full)
  
  Near-misses (strategies that almost fired):
    - MSFT  pullback: RSI 54 (needs <53) — missed by 1 point
    - GOOG  momentum: consolidation 2.1x ATR (needs <2.0x) — missed by 5%
  
  Portfolio: $488.80 equity | $116.80 cash | 2 open positions | heat 4.2%
  ═══════════════════════════════════════════════════════════
  ```

- **"Why not?" logging** — For every strategy that evaluates a candidate and rejects it, log the FIRST filter that failed and how far off it was. This is critical for:
  - Tuning parameters (if 80% of rejections are "missed by <5%", the threshold is probably too tight)
  - ML training (the near-misses with their feature values are training data)
  - Human review (you can see at a glance why strategies aren't firing)

- **Trade lifecycle logging** — Track the full journey of every trade from signal to close:
  ```
  [10:02] SIGNAL  AAPL momentum conv=0.83 | entry=$185 stop=$181.50 target=$195.50
  [10:02] SIZED   AAPL 2 shares ($370) | risk=$7 (1.4%) | ML accuracy for similar: 68%
  [10:02] APPROVED AAPL | regime=BULL, news=neutral, portfolio_heat=4.2%→5.6%
  [10:03] FILLED  AAPL 2 shares @ $185.12 (slippage: $0.12)
  [11:30] UPDATE  AAPL +$1.88 (+1.0%) | high=$187.50 | stop untouched | target 58% away
  [14:15] EXIT    AAPL stopped out @ $181.50 | P&L=-$7.24 (-1.4%) | held 4h12m
  [14:15] REVIEW  AAPL: price recovered to $188.20 within 90 min. Stop was 1.5x ATR — consider 2.0x for momentum in BULL regime.
  ```

- **Store everything in the database** — Not just in log files that scroll away. Every signal considered, every rejection reason, every factor value at time of evaluation, every conviction modifier applied, every position sizing calculation. The database is the training data for the ML models.

### 10. Advanced Stop-Loss & Take-Profit Calculations

**Goal:** Stop-loss and take-profit levels should be intelligent, not just "N × ATR from entry." They should account for support/resistance, volatility regime, time in trade, and learned optimal exits.

**Current state:** All strategies use simple ATR-based stops: `stop = entry - atr_multiplier * ATR`, `target = entry + atr_multiplier * ATR`. Multipliers are static per strategy in `settings.yaml` (e.g., momentum: 1.5x stop, 3.5x target). No trailing stops in software (relies on Alpaca bracket orders which are fixed).

**What V2 needs:**

- **Support/Resistance-aware stops** — Instead of blindly placing stops at N×ATR:
  - Identify the nearest support level below entry (recent swing lows, EMA lines, VWAP, round numbers)
  - Place stop just below that support (e.g., $0.05 below the nearest swing low)
  - If no clear support within 2x ATR, use ATR-based stop as fallback
  - This prevents stops from sitting in "no man's land" where they get hit by normal noise but aren't at a level that actually invalidates the trade thesis

- **Volatility-adjusted stops** — ATR is good but static for the lookback period. Enhance:
  - Use intraday ATR (not just daily) for day trades — daily ATR overstates the expected move for a 2-hour hold
  - Adjust ATR multiplier based on current VIX: higher VIX → wider stops (market is swingier), lower VIX → tighter stops
  - Time-of-day adjustment: wider stops in the first 30 min (high volatility), tighter in the midday lull
  - Regime-based: trending markets need wider stops (let the trade breathe), choppy markets need tighter stops (cut losses fast)

- **Dynamic take-profit targets** — Instead of fixed ATR multiples:
  - Identify resistance levels above entry (recent swing highs, prior day high, whole numbers, Fibonacci extensions)
  - Set initial target at the first major resistance
  - Use trailing take-profit: as price moves favorably, trail the target upward (let winners run)
  - Partial exits: take 50% off at first target (lock in profit), let remaining 50% run with a trailing stop

- **Trailing stop implementation** — The bot currently submits bracket orders and walks away. V2 needs:
  - Software-side trailing stops that update the Alpaca order as price moves
  - Multiple trailing modes:
    - **Fixed trail:** Move stop up by $X for every $X price moves in our favor
    - **ATR trail:** Keep stop at entry + (max_favorable_move - 1.5×ATR)
    - **Breakeven trail:** Once price has moved 1×ATR in our favor, move stop to breakeven + $0.05
    - **Chandelier trail:** Stop at highest_high - 2×ATR (tracks the trend, not entry)
  - The ML model should learn which trailing mode works best for each strategy/regime combo

- **Time-based exit rules:**
  - Day trades: if not profitable within 60 minutes, tighten stop to breakeven or exit
  - Swing trades: if not profitable within 3 days, re-evaluate thesis
  - 0DTE options: hard exit at 3:30 PM, tighten stops after 2:00 PM
  - ML-learned hold durations: track optimal hold time per strategy and exit if exceeded

- **R:R ratio optimization** — The current 2:1 minimum R:R is a static rule. V2 should:
  - Track actual achieved R:R per strategy (are targets being hit, or are trades always exiting at stops?)
  - If a strategy consistently exits at 1.5:1 instead of hitting 3:1 targets, the target is too ambitious — lower it
  - If a strategy's winners always go way past the target, the target is too conservative — widen it or use trailing
  - ML model predicts optimal R:R given current conditions

- **Stop-loss quality scoring** — After every trade, score the stop:
  - **Too tight:** Price hit stop then reversed and would have been profitable → widen
  - **Too loose:** Price blew through stop on the way to a much bigger loss → tighten
  - **Just right:** Stop was near the actual reversal point → keep
  - Feed this back into the stop-placement model

### 11. Performance Optimization (Final Pass)

**Goal:** After all features are built, go back and optimize the entire system for speed and efficiency. The algo must be lean — fast scans, fast calculations, fast order submission. Milliseconds matter in trading.

**Current state:** The bot fetches data sequentially, evaluates strategies in a single-threaded loop, and makes many individual API calls. For ~44 candidates across 16 strategies, this is manageable but not fast.

**What V2 must optimize:**

- **Parallel data fetching** — Currently `fetch_bars_multi()` and news scanning run sequentially per symbol. Use `asyncio` or `concurrent.futures.ThreadPoolExecutor` to:
  - Fetch all daily bars in parallel (batch API call already exists, but ensure it's optimal)
  - Fetch intraday bars in parallel with daily bars (they're independent)
  - Scan news in parallel with bar fetching (independent data sources)
  - Fetch options chains in parallel with stock data
  - Target: reduce data-fetch phase from sequential to parallel, cutting wall-clock time by 3-5x

- **Vectorized indicator calculations** — `add_all()` in `indicators.py` computes indicators per-DataFrame. Ensure:
  - All indicator calculations use pandas/numpy vectorized operations (no Python for-loops over rows)
  - Pre-compute indicators once and cache — don't recalculate the same EMA every 15 minutes when only the last bar changed
  - For the ML feature extraction, batch all feature computation into a single vectorized pass

- **Strategy evaluation optimization:**
  - Strategies currently evaluate one-symbol-at-a-time. Where possible, vectorize to evaluate all candidates at once (e.g., RSI check across all symbols in one numpy operation)
  - Short-circuit evaluation: if the first filter fails, don't compute remaining indicators for that candidate
  - Cache strategy state between cycles — if a swing setup was valid 15 minutes ago and bars haven't changed, don't recompute

- **ML model inference speed:**
  - Keep trained models in memory (don't reload from disk each cycle)
  - Use lightweight models (LightGBM, not deep learning) for real-time inference
  - Batch predictions: predict on all candidates at once, not one at a time
  - Target: <10ms for predicting all candidates (should be trivial for gradient boosting)

- **Database write optimization:**
  - Batch inserts: collect all decisions/signals from a cycle and insert in one transaction, not individual INSERT per row
  - Use WAL (Write-Ahead Logging) mode for SQLite to avoid write locks during reads
  - Periodic cleanup: archive old decisions/signals to keep the main tables lean
  - Index critical query columns (symbol, strategy, timestamp, status)

- **API call minimization:**
  - Cache Alpaca account info for 30 seconds (don't fetch every cycle)
  - Batch order submissions where possible
  - Use WebSocket streaming for real-time price updates instead of polling (the streaming module exists but isn't active)
  - Cache options snapshots for 60 seconds (greeks don't change that fast)

- **Memory efficiency:**
  - Don't hold 60 days of bars for all 44 candidates in memory simultaneously — process and release
  - ML feature DataFrames should be compact (only the columns needed, appropriate dtypes)
  - Limit decision log retention in memory (write to DB, don't accumulate)

- **Profiling and benchmarks:**
  - Add timing decorators to each phase (scan, fetch, evaluate, rank, execute)
  - Log per-cycle timing: "Scan: 120ms | Fetch: 2.1s | Evaluate: 340ms | Rank: 5ms | Execute: 890ms"
  - Set performance budgets: entire cycle should complete in <10 seconds (currently likely 15-30s)
  - Identify and eliminate any blocking I/O during the evaluation phase

- **Lean code principles:**
  - Remove dead code paths and unused strategy logic
  - Minimize object allocations in hot paths (reuse DataFrames, pre-allocate arrays)
  - Profile memory usage and eliminate unnecessary copies (`.copy()` only when mutation is a real risk)
  - Keep the main evaluation loop as tight as possible — heavy computation should happen in batch before the loop, not inside it

### 12. Machine Learning Core

**Goal:** The entire system should be ML-driven. Static rules are bootstrapping logic that gets progressively replaced by learned models as the bot accumulates trade data. Every decision the bot makes should eventually be informed by its own performance history.

**Architecture:**
- **Signal quality model** — A classifier (gradient boosting / XGBoost / LightGBM) trained on the bot's own trade outcomes. Features:
  - All technical indicators at entry (RSI, MACD, EMA positions, BB width, ATR, RVOL)
  - Market regime at entry
  - Time of day, day of week
  - Sector of the stock
  - News sentiment score
  - Scanner source (which scanner found this candidate)
  - Strategy name
  - Historical win rate of this strategy in this regime
  - Target: binary (profitable / not profitable) or regression (actual P&L %)
  - This model's output BECOMES the conviction score, replacing the hand-coded conviction logic in each strategy
- **Parameter optimization model** — Bayesian optimization or genetic algorithm that tunes strategy parameters:
  - Runs nightly or weekly on historical data
  - Optimizes for Sharpe ratio, not just win rate (a strategy that wins 90% but loses big on the 10% is bad)
  - Separate parameter sets per market regime
  - Bounded search space (don't let parameters go to absurd values)
  - Backtests each parameter set before deploying
- **Risk model** — Predicts the probability and magnitude of adverse moves:
  - Given current positions and market state, what's the probability of a >3% drawdown today?
  - Should we be hedged? (Buy puts on SPY if risk model says drawdown likely)
  - Informs position sizing: high drawdown risk → smaller positions, low risk → larger
- **Reinforcement learning (stretch goal)** — Train an RL agent that learns the optimal action (buy/sell/hold/size) given the current state (portfolio, market, signals). This is the ultimate endgame — the bot learns a complete trading policy from experience.

**Cold start / bootstrap:**
- Before the ML models have enough data (first 50-100 trades), use the existing rule-based logic as the decision maker
- Track what the ML model WOULD HAVE predicted vs what the rules decided, to measure model readiness
- Gradually shift decision weight from rules → ML as model accuracy improves (ensemble approach)
- Never go 100% ML overnight — ramp from 0% → 25% → 50% → 75% → 90% ML weight based on demonstrated accuracy

**Training pipeline:**
- After each trade closes → add to training dataset
- Every N trades (configurable, start at 10) → retrain models
- Every retraining → backtest new model vs old model on held-out data
- Only deploy new model if it outperforms the old one on backtest
- Keep model version history and performance metrics in the database
- Log every prediction the model makes for future analysis

### 13. State Persistence & Graceful Restart

**Goal:** The bot must be able to go offline (crash, planned shutdown, power loss, multi-day pause) and come back online without losing any learned state. Weighting, ML models, adjusted parameters, and all accumulated intelligence must survive restarts.

**Current state (partial):**
- `StrategyWeighter` in `src/ai_trade/strategy/weighter.py` stores computed weights in-memory only (`self._weights: dict`). On restart they're empty until `maybe_recalculate()` runs, which rebuilds them from the `trades` table.
- `_trade_count_at_last_recalc` resets to 0 on restart, so recalculation cadence is lost.
- Trade history, day-trade records, signals, and daily snapshots ARE persisted in SQLite (good).
- PDT count is synced from Alpaca on startup (good).
- Market regime must be re-analyzed after restart (acceptable — it's a snapshot, not learned state).
- No mechanism for storing ML models, adjusted strategy parameters, or any V2-era learned state.

**What V2 needs:**

- **Persist all computed weights** — After `StrategyWeighter._recalculate()` runs, write the new weights to a `strategy_weights` table:
  ```
  strategy_weights table:
    strategy_name (PK), weight, composite_score, win_rate, profit_factor,
    avg_pnl, recency_score, trades_used, last_updated
  ```
  On startup, load the most recent weights from this table directly — don't force a recalculation from scratch. The weighter can optionally refresh them when enough new trades have closed.

- **Persist the recalculation cursor** — Store `_trade_count_at_last_recalc` in a key-value state table so recalculation intervals work correctly after restarts. Without this, the weighter may recompute too often (wasted work) or too rarely (stale weights).

- **General-purpose state table** — Add a `bot_state` table for all miscellaneous persistent key-value state:
  ```
  bot_state table:
    key (PK), value (JSON), updated_at
  ```
  Use it for: last recalc cursor, last trained model timestamp, current adjusted parameter overrides, drawdown circuit breaker state, winning/losing streak counters, last successful scan time, etc. Anything that shouldn't be lost on restart but doesn't warrant its own table.

- **ML model persistence** — Already covered by the `ml_models` table (Section 12), but to be explicit:
  - Serialize trained models to disk (`models/` directory using joblib or pickle)
  - Record the file path and version in the `ml_models` table
  - On startup, load the most recent `is_active=true` model for each model type
  - Never silently fall back to untrained models — if a model can't load, log loudly and use the rule-based fallback

- **Parameter override persistence** — When the strategy optimizer (Section 1) learns that a parameter should be adjusted (e.g., RSI threshold 38 → 42 for mean_reversion), store the change in `parameter_history`:
  ```
  parameter_overrides table (current active overrides):
    strategy_name, param_name, value, set_at, set_by_model_version, reason
  ```
  On startup, these overrides are applied on top of the base `settings.yaml` config. The base config is the starting point; learned overrides are the current state.

- **Graceful startup sequence** — Define a clear boot order:
  1. Load `settings.yaml` base config
  2. Connect to SQLite database
  3. Load strategy weights from `strategy_weights` table
  4. Load parameter overrides from `parameter_overrides` table
  5. Load active ML models from `ml_models` table
  6. Load `bot_state` key-value pairs (streaks, circuit breaker state, cursors)
  7. Sync PDT count from Alpaca
  8. Re-analyze market regime from fresh SPY/QQQ/VIX data
  9. Fetch open positions from Alpaca and reconcile with local DB
  10. Start scheduled jobs
  
  Log each step explicitly so you can see exactly what was restored from disk. Example:
  ```
  [BOOT] Loaded 8 strategy weights from DB (last updated 2026-04-13 15:30)
  [BOOT] Applied 3 parameter overrides (mean_reversion.rsi_oversold=42, momentum.atr_stop=1.8, pullback.tolerance=1.2)
  [BOOT] Loaded ML model v12 (accuracy 0.67, trained on 143 trades)
  [BOOT] Restored streak counter (wins=4, losses=2), circuit breaker=normal
  [BOOT] PDT count synced from Alpaca: 1/3 used
  [BOOT] Market regime: BULL (breadth +0.38, VIX 17.5)
  [BOOT] 1 open position reconciled: SHAZ bb_squeeze 1 share @ $28.00
  [BOOT] Ready — resumed from cold start in 2.1s
  ```

- **Graceful shutdown sequence** — When receiving SIGINT/SIGTERM:
  1. Stop accepting new signals
  2. Let in-flight orders complete (with timeout)
  3. Flush any pending writes to the database
  4. Write current `bot_state` snapshot
  5. Close database connection cleanly
  6. Exit
  
  Never leave the database in an inconsistent state. Use SQLite transactions for multi-step writes.

- **Test restart safety explicitly** — Add a test that:
  1. Runs the bot through a series of trades until weights/params/models are non-default
  2. Snapshots the in-memory state
  3. Shuts down the bot
  4. Starts it back up
  5. Verifies the in-memory state matches the pre-shutdown snapshot exactly
  
  This test should run in CI. Restart-safety bugs are the kind of thing that silently erodes months of learned behavior if not caught.

- **Offline gap handling** — If the bot has been offline for hours or days:
  - On startup, detect the gap (last `daily_snapshots` date vs today)
  - Log the gap clearly
  - Fetch and catch up on any closed positions (Alpaca may have filled stops/targets while offline)
  - Reconcile PnL and update trade records
  - Recompute strategy weights with the new closed trades
  - Do NOT try to "make up" for missed trading days — just resume forward
  - If the gap is >5 business days, flag for human review (something's probably wrong)

### 14. Continuous Backtesting & Algorithm Training (MANDATORY ONGOING WORK)

**THIS IS NOT OPTIONAL.** As you implement features, you must continuously backtest and train the trading algorithm. Every significant change must be validated against historical data before being considered "done." Code that compiles is not code that works — only backtested code is verified.

**Current state:** `src/ai_trade/backtest/engine.py` and `runner.py` contain the existing backtesting framework. It can replay historical bars through strategies and measure outcomes. The ML models do not yet exist and nothing is currently trained.

**What the agent must do throughout V2 work:**

- **Backtest after every significant change** — When you modify a strategy, add a new filter, change a parameter, or adjust risk logic, run a backtest before moving on. If the change makes things worse, don't proceed with it. Log the backtest result in the progress file.
  - Required cadence: at minimum once per phase (ideally after every major subtask)
  - Use a consistent baseline dataset so results are comparable across changes
  - Compare new code against the previous version on: Sharpe, win rate, profit factor, max drawdown, total return
  - If a change is worse on 2+ of those metrics, revert or redesign

- **Build a historical data corpus** — The backtest is only as good as the data it runs on. Before building ML models, assemble:
  - At least 1 year of daily bars for the top 500 most-traded US stocks
  - At least 60 days of minute bars for the scanner universe
  - Historical VIX, SPY, QQQ data for regime classification
  - Historical news data if available (for news sentiment model training)
  - Store this data locally so backtests don't thrash the Alpaca API
  - Document the exact date range and symbols used (for reproducibility)

- **Train strategies on historical data before live deployment** — Every new strategy, parameter change, or model should be trained/validated on historical data first:
  - **Walk-forward testing:** Train on months 1-6, test on month 7. Retrain on months 1-7, test on month 8. Continue. This simulates the real behavior of the bot learning over time.
  - **Out-of-sample validation:** Always hold out the most recent 20% of data for final validation. Never train on it.
  - **Regime diversity:** Ensure training data covers bull, bear, and neutral markets. A strategy that works in one regime but not others is not production-ready.
  - **Statistical significance:** Require at least 50 trades across the test period before trusting the results. 10 trades proves nothing.

- **Establish baseline metrics and track drift** — Before starting major changes, run a baseline backtest of the current v1.2.0 system. Record:
  - Sharpe ratio
  - Win rate
  - Profit factor
  - Max drawdown
  - Total return over test period
  - Trade frequency per strategy
  
  Then after each phase, re-run the same backtest with the modified code and compare. This is how you prove the system is actually improving (not just running without errors).

- **Train the ML models iteratively as data accumulates:**
  - **Phase 1 (bootstrap):** Backtest rule-based strategies to generate synthetic training data (entry conditions → outcomes). This gives the ML model something to learn from before any live trades happen.
  - **Phase 2 (early live):** Combine synthetic backtest data with real paper-trade data as it accumulates. Retrain weekly.
  - **Phase 3 (steady state):** Retrain on real paper/live data only (synthetic data phased out). Retrain after every 10-20 closed trades.
  - **Always version models:** Keep old model versions so you can roll back if a new model underperforms.

- **Continuous validation harness** — Build a `scripts/backtest_continuous.py` that:
  - Runs the full evaluation pipeline against the last N days of historical data
  - Reports metrics in a standardized format
  - Compares against the previous run (stored in a JSON file)
  - Fails loudly if metrics degrade beyond a threshold (e.g., Sharpe drops >20%)
  - Can be run manually or in a pre-commit hook
  - This is your "did my change break anything?" safety net

- **Backtest reports stored in the database:**
  ```
  backtest_runs table:
    id, timestamp, code_version (git hash), 
    date_range_start, date_range_end, symbols_count,
    total_trades, win_rate, sharpe, profit_factor, max_drawdown,
    total_return_pct, avg_hold_duration,
    config_snapshot (JSON of settings at time of run),
    notes (what was being tested)
  ```
  - Every backtest run goes in here, so you can trace the evolution of system quality over time
  - Query: "show me Sharpe ratio trend over the last 30 backtest runs" — should be going up, not down

- **Train, don't just test** — The word "train" is important. The ML models need to actually learn from data, not just be evaluated. This means:
  - Extract features from every historical trade (including near-misses from the decision audit trail)
  - Label them with outcomes (profit/loss/magnitude)
  - Fit models (LightGBM, XGBoost, etc.)
  - Measure training accuracy AND validation accuracy (watch for overfitting)
  - Save trained models with metadata (training date, data used, metrics)
  - Deploy the model into the live decision pipeline only after validation passes

- **Integration with progress tracking** — Every backtest run and every training run should be logged in `docs/V2_PROGRESS.md`:
  ```markdown
  ## Backtest Log
  | Date | Phase | Test Period | Sharpe | Win Rate | PF | Notes |
  |------|-------|-------------|--------|----------|-----|-------|
  | 2026-04-14 | Phase 1 baseline | 2025-01 to 2026-04 | 3.12 | 43.8% | 1.55 | v1.2.0 baseline |
  | 2026-04-15 | Phase 2 after logging | same | 3.12 | 43.8% | 1.55 | no functional change, just instrumentation |
  | 2026-04-16 | Phase 3 new stops | same | 3.48 | 47.2% | 1.78 | S/R-aware stops improved win rate |
  | ... | ... | ... | ... | ... | ... | ... |
  ```
  - Every row is evidence that work is actually improving the system
  - If rows go flat or negative, something is wrong — investigate before continuing

- **Don't just build — prove it works.** Shipping untested code in a trading system burns real money. Every feature, every parameter change, every new model must pass backtest validation before it's considered complete. The v1.2.0 baseline must keep improving, not just keep working.

---

## Implementation Constraints

1. **Broker:** Must use Alpaca APIs (paper and live). All market data, order submission, and account management go through Alpaca.
2. **Account size:** Currently $500. All position sizing and strategy selection must work at this scale. Don't pre-disable strategies — let the ML model figure out what's viable. If a strategy can't find valid trades at this account size, it'll naturally produce zero signals.
3. **PDT rule:** Absolute constraint. 3 day trades per 5 business days. Cannot be violated.
4. **Execution:** Bot runs on a Windows machine. Must be reliable for unattended 24/5 operation.
5. **Cost:** Minimize external API costs. Free data sources preferred. If using an LLM for news analysis, batch requests and cache results (don't call Claude for every single article).
6. **Database:** Keep using SQLite for simplicity. Add new tables as needed for t

rade analysis, parameter history, learning data, etc.
7. **Existing code:** Build on the existing architecture. Don't rewrite from scratch — extend it. The strategy pattern (BaseStrategy ABC → evaluate() → Signal) is sound. The evaluation pipeline (scan → fetch → evaluate → rank → execute) is sound. Extend, don't replace.
8. **Testing:** The backtesting framework exists in `src/ai_trade/backtest/`. Use it to validate changes before deploying. Any parameter optimization should be backtested against historical data.

---

## Priority Order

1. **Fix existing strategy issues first** (the v1.2.1 plan has 13 specific fixes — get all 16 strategies firing)
2. **Logging & decision audit trail** (Section 9 — instrument everything FIRST so all subsequent work generates training data from day one)
3. **Advanced stop-loss & take-profit** (Section 10 — support/resistance-aware stops, trailing stops, volatility-adjusted exits)
4. **State persistence & graceful restart** (Section 13 — persist weights/params/models/state so restarts don't erase learning)
5. **ML core infrastructure** (Section 12 — training pipeline, feature extraction, model versioning)
6. **Self-learning trade analysis** (Section 5 — post-trade analysis engine, loss pattern clustering)
7. **Dynamic risk tolerance + portfolio-% position limits** (Section 6 — conviction-based sizing, >90% conviction overrides)
8. **Smart PDT management** (Section 4 — ML-informed slot allocation, day-to-swing conversion)
9. **Strategy optimizer** (Section 1 — ML-driven parameter tuning, per-regime parameter sets)
10. **0DTE options + aggressive options strategies** (Section 7 — enable all strategies, add 0DTE)
11. **News/event intelligence** (Section 3 — LLM-powered, geopolitical event mapping)
12. **Market prediction** (Section 2 — pattern recognition, correlation models, RL stretch goal)
13. **Performance optimization** (Section 11 — FINAL PASS: parallelize, vectorize, profile, make it lean and fast)

**Continuous throughout all phases:** Backtesting & training (Section 13). Not a numbered phase — an ongoing activity that validates every other phase's work. Every change gets backtested. ML models get trained and retrained as data accumulates. No feature is "done" until it's backtested and proven to not degrade the baseline.

---

## Database Schema (Current)

```sql
-- Existing tables (in data/ai_trade.db)
trades           -- id, symbol, strategy, direction, status, entry_price, exit_price, pnl, shares, stop_loss, take_profit, entry_date, exit_date, hold_type, conviction, alpaca_order_id
day_trades       -- id, symbol, trade_date, buy_order_id, sell_order_id
daily_snapshots  -- id, date, equity, cash, open_positions, day_trades_used, total_trades, winning_trades, losing_trades, total_pnl, max_drawdown, sharpe_ratio, version
signals          -- id, timestamp, symbol, strategy, conviction, hold_type, direction, executed
options_trades   -- id, symbol, underlying, strategy, option_type, strike, expiration, direction, status, entry_price, exit_price, pnl, contracts, delta, gamma, theta, vega, iv, entry_date, exit_date, conviction, alpaca_order_id
scanner_results  -- id, scan_date, scan_type, symbol, price, volume, relative_volume, gap_pct, score
```

**New tables to add** (suggested — agent should design final schema):
- `decisions` — Full audit trail of every decision (timestamp, decision_type, symbol, strategy, action, conviction, reasoning_text, factors_json, outcome) — this is the primary training dataset for ML
- `trade_analysis` — Post-close analysis of each trade (entry quality score, stop_quality_score, exit quality, market context at entry/exit, lessons_text, stop_too_tight_flag, price_after_exit)
- `parameter_history` — Log of every parameter change (strategy, param_name, old_value, new_value, reason, timestamp)
- `strategy_performance` — Rolling performance by strategy × regime (win_rate, avg_pnl, sharpe, by regime)
- `events` — Significant market events detected (type, description, affected_sectors, timestamp)
- `learning_log` — What the bot learned and when (insight, action_taken, impact_measured)
- `ml_models` — Model version registry (model_name, version, trained_at, training_trades_count, backtest_sharpe, backtest_accuracy, is_active, model_path)
- `ml_predictions` — Every prediction the model makes (timestamp, signal_id, predicted_outcome, predicted_confidence, actual_outcome, model_version) — essential for measuring model accuracy
- `ml_features` — Feature snapshots at time of each trade entry (trade_id, feature_json) — enables retraining on historical data with the exact features the model saw
- `conviction_overrides` — Log of every >90% conviction override (trade_id, normal_position_size, override_position_size, conviction, ml_accuracy_at_time, outcome) — proves whether earned aggression is working
- `backtest_runs` — Every backtest run with code version, date range, metrics (sharpe, win_rate, pf, drawdown, total_return), config snapshot, and notes. This is the evidence trail that the system is improving over time.
- `training_runs` — Every ML model training run (model_name, training_date, trades_used, features_count, train_accuracy, val_accuracy, was_deployed, reason_for_reject_or_deploy)

---

## Key Files to Read

Before starting, read these files to understand the current implementation:

| File | Why |
|------|-----|
| `config/settings.yaml` | All current parameters and their values |
| `src/ai_trade/main.py` | The orchestrator — understand the full evaluation pipeline |
| `src/ai_trade/strategy/signal.py` | SignalAggregator — how signals are ranked and filtered |
| `src/ai_trade/strategy/weighter.py` | Current adaptive weighting logic |
| `src/ai_trade/strategy/base.py` | Signal dataclass, HoldType enum, BaseStrategy ABC |
| `src/ai_trade/sentiment/market_regime.py` | Current regime classification system |
| `src/ai_trade/sentiment/news_sentiment.py` | Current keyword-based news scoring |
| `src/ai_trade/risk/pdt_manager.py` | PDT tracking and enforcement |
| `src/ai_trade/risk/position_sizer.py` | Current position sizing logic |
| `src/ai_trade/risk/risk_manager.py` | Trade approval gates |
| `src/ai_trade/monitoring/database.py` | SQLite schema and data access methods |
| `src/ai_trade/monitoring/performance.py` | Performance metrics computation |
| `src/ai_trade/scanner/screener.py` | Current scanning logic (3 scanners) |
| Any strategy file (e.g., `src/ai_trade/strategy/momentum.py`) | Pattern for how strategies evaluate candidates |

---

## Success Criteria

The V2 system is successful when:

1. **ALL strategies fire** — All 16+ strategies (including 0DTE) generate signals. Every strategy should have a path to execution.
2. **Higher Sharpe ratio** — Sustain > 4.0 over a 30-day rolling window
3. **Better win rate** — > 50% (currently 43.75%), with the ML model demonstrating improving accuracy over time
4. **Self-improving** — Measurable improvement in metrics week-over-week as the bot learns. Each month should be better than the last.
5. **ML-driven decisions** — Within 100 trades, the ML models should be making >50% of conviction/sizing decisions (not static rules)
6. **News-reactive** — The bot adjusts positions within minutes of a major news event
7. **PDT-efficient** — Day-trade slots are used on the highest-EV opportunities, ML-allocated
8. **No penny stocks** — Zero trades on stocks under $5
9. **Profitable options** — Options trades execute regularly, including short-dated and 0DTE plays when conditions warrant
10. **Dynamic behavior** — Observable changes in strategy parameters, risk tolerance, and candidate selection over time based on learning
11. **Higher absolute returns** — Targeting 2-5% weekly return on the $500 account (currently ~0.4%/week)
12. **Conviction overrides work** — >90% conviction trades demonstrably outperform, validating the larger position sizing
13. **Portfolio-aware sizing** — Position limits are dynamic based on portfolio allocation %, not a fixed count. The bot can hold 2 large positions or 8 small ones depending on conviction distribution.
14. **Earned aggression** — The bot starts conservative, proves its models work, then progressively takes more risk. Risk tolerance is proportional to demonstrated accuracy.
15. **Full audit trail** — Every decision is stored in the database with human-readable reasoning. You can query "why did the bot buy AAPL at 10:02 on Tuesday?" and get a complete answer: the indicators, the conviction breakdown, the risk check, the sizing math.
16. **Smart exits** — Stop-losses are placed at support levels (not arbitrary ATR distances). Trailing stops lock in profits. Partial exits are used. Post-trade analysis shows stop quality improving over time.
17. **Lean and fast** — Full evaluation cycle completes in <10 seconds. No blocking I/O during evaluation. Parallel data fetching. Vectorized calculations. Per-phase timing logged every cycle.
18. **Continuously backtested** — Every change is validated against historical data before being considered complete. A `backtest_runs` table in the database shows a monotonic improvement in Sharpe/win-rate/PF over time. The agent never merges code that degrades the baseline.
19. **Trained models, not just code** — ML models are actually trained (not just scaffolded) on historical and live trade data. Models are versioned, validated against held-out data, and deployed only after outperforming the previous version. Training happens continuously as new trades close.
20. **Restart-safe** — The bot can be stopped and restarted (planned or crash) without losing any learned state. Strategy weights, parameter overrides, ML models, streak counters, and circuit breaker state all survive across sessions. After a multi-day offline period, the bot resumes forward with all intelligence intact.

---

## Agent Work Journal & Progress Tracking

**THIS IS CRITICAL.** The human working with you will hit usage limits, context compaction, and session breaks. You MUST maintain a living progress document so that any new session (or a new agent) can pick up exactly where you left off with zero ramp-up time.

### The Progress File: `docs/V2_PROGRESS.md`

Maintain this file throughout your work. Update it **every time you complete a task, hit a blocker, or make a significant decision.** This is not optional — it's the primary way continuity is maintained across sessions.

**Required structure:**

```markdown
# V2 Implementation Progress

## Current Status
**Last updated:** [timestamp]
**Current phase:** [e.g., "Phase 2: Logging & Decision Audit Trail"]
**Current task:** [e.g., "Adding rejection logging to options strategies"]
**Blocked on:** [anything blocking progress, or "Nothing"]

## Completed Work

### [Phase/Section Name] — [Status: COMPLETE / IN PROGRESS / NOT STARTED]
- [x] Task description — what was done, which files were changed
- [x] Task description
- [ ] Task description (not yet done)

### [Next Phase] — NOT STARTED
- [ ] ...

## Architecture Decisions Made
Decisions that affect future work. These are things a new agent needs to know.
- [Decision]: [Reasoning] — [Date]
- Example: "Used LightGBM over XGBoost for signal quality model because it's faster on small datasets and handles categorical features natively" — 2026-04-10

## Files Modified (Running List)
Keep a running list of every file created or significantly modified, grouped by phase.
This lets a new session quickly verify the current state.

| File | Phase | Change Summary |
|------|-------|---------------|
| src/ai_trade/monitoring/decision_logger.py | Phase 2 | NEW: Decision audit trail module |
| src/ai_trade/strategy/momentum.py | Phase 1 | Modified: consolidation filter → conviction modifier |
| ... | ... | ... |

## Known Issues / Tech Debt
Things that need fixing but aren't blocking current work.
- [Issue]: [Impact] — [Which phase will address it]

## Next Steps
Ordered list of what to do next when resuming work.
1. [Specific next action with file path]
2. [Next action after that]
3. ...

## Backtest Log
Every backtest run with its metrics. This is the evidence the system is improving.
| Date | Phase | Test Period | Sharpe | Win Rate | PF | Max DD | Notes |
|------|-------|-------------|--------|----------|-----|--------|-------|
| 2026-04-14 | baseline | 2025-01 to 2026-04 | 3.12 | 43.8% | 1.55 | -6.2% | v1.2.0 baseline before any V2 work |
| ... | ... | ... | ... | ... | ... | ... | ... |

## Training Log
Every ML model training run.
| Date | Model | Data Range | Trades | Train Acc | Val Acc | Deployed? | Notes |
|------|-------|-----------|--------|-----------|---------|-----------|-------|
| ... | ... | ... | ... | ... | ... | ... | ... |
```

### Update Rules

1. **Update after every completed task** — Not at the end of the session. After each meaningful unit of work (file modified, feature implemented, bug fixed), update the progress file immediately. If you get cut off mid-session, the last update should reflect where you actually are.

2. **Update before starting a new phase** — Summarize what was accomplished in the previous phase, note any loose ends, then outline the plan for the next phase.

3. **Update when hitting a blocker** — If something unexpected happens (API doesn't work as expected, design assumption was wrong, tests fail), document it immediately. The next session needs to know what went wrong and what was tried.

4. **Update with architecture decisions** — Any non-obvious choice (library selection, data structure, algorithm approach, trade-off made) should be recorded with reasoning. A new agent shouldn't have to re-derive these decisions.

5. **Keep the "Next Steps" section actionable** — Don't write "continue working on ML." Write "Implement `train_signal_model()` in `src/ai_trade/ml/trainer.py` — need to: (a) extract features from `decisions` table, (b) split train/test by date, (c) train LightGBM classifier, (d) save model to `models/` directory."

### Session Handoff Protocol

At the **start** of every session, the agent should:
1. Read `docs/V2_PROGRESS.md` to understand current state
2. Read `docs/V2_AGENT_BRIEF.md` to understand the full vision
3. Verify the "Files Modified" list matches actual file state (quick spot-check)
4. Resume from the "Next Steps" section

At the **end** of every session (or when you sense context is getting large), the agent should:
1. Update `docs/V2_PROGRESS.md` with everything completed this session
2. Write clear, specific "Next Steps" that a cold-start agent can follow
3. Commit progress (code + progress file) so nothing is lost

### Why This Matters

Without this file:
- A new session wastes 10-20 minutes re-reading code to figure out what was already done
- Decisions get re-made differently, creating inconsistencies  
- Half-finished work gets abandoned or duplicated
- The human has to manually explain "you were working on X, you got to Y, next do Z" every single time

With this file:
- New session reads one file, knows exactly where to pick up
- Zero repeated work, zero re-explanation needed
- The human can check progress at any time without asking
- Multiple agents can work on different phases without stepping on each other
