# Risk Management Guide

This document explains every layer of risk control in the AI Trade bot. It is written for someone comfortable with software engineering concepts but not necessarily familiar with Python syntax or trading terminology. Each section explains **why** a control exists, **how** it works in code, and **what happens** in dollar terms on a $500 account.

---

## Table of Contents

- [Why Risk Management Matters](#why-risk-management-matters)
- [Architecture Overview](#architecture-overview)
- [The Five Layers of Risk Control](#the-five-layers-of-risk-control)
  - [Layer 1: Position Sizing](#layer-1-position-sizing)
  - [Layer 2: Risk Manager Gate](#layer-2-risk-manager-gate)
  - [Layer 3: PDT Manager](#layer-3-pdt-manager)
  - [Layer 4: Bracket Orders](#layer-4-bracket-orders)
  - [Layer 5: Market Regime Filter](#layer-5-market-regime-filter)
- [Options Risk Controls](#options-risk-controls)
- [News Sentiment as Risk Gate](#news-sentiment-as-risk-gate)
- [Daily Loss Circuit Breaker](#daily-loss-circuit-breaker)
- [Position Reconciliation](#position-reconciliation)
- [Emergency Procedures](#emergency-procedures)
- [Risk Parameters Reference Table](#risk-parameters-reference-table)

---

## Why Risk Management Matters

With a $500 account, a single bad trade can easily wipe out 10-20% of the entire account balance. That is $50-$100 gone in minutes. Without guardrails, a string of two or three bad trades could reduce the account to a point where meaningful recovery becomes nearly impossible.

The risk system ensures **no single mistake can be catastrophic**. It does this through five independent layers of protection, each one capable of blocking a trade on its own. Even if one layer has a flaw or an edge case, the others catch it.

Think of it like the safety systems in a car: seat belts, airbags, crumple zones, ABS brakes, and traction control all work independently. You never want to rely on just one.

---

## Architecture Overview

Every trade signal flows through a pipeline of checks before any real money is committed:

```
                          +---------------------+
                          |   Strategy Signal    |
                          | (buy AAPL at $48.50) |
                          +----------+----------+
                                     |
                                     v
                    +----------------+----------------+
                    |  Layer 5: Market Regime Filter   |
                    |  Is the broad market healthy?    |
                    +----------------+----------------+
                                     |
                                     v
                    +----------------+----------------+
                    |  News Sentiment Gate             |
                    |  Is there negative news?         |
                    +----------------+----------------+
                                     |
                                     v
                    +----------------+----------------+
                    |  Layer 1: Position Sizer         |
                    |  How many shares can we afford?  |
                    +----------------+----------------+
                                     |
                                     v
                    +----------------+----------------+
                    |  Layer 2: Risk Manager Gate      |
                    |  Daily loss? Too many positions? |
                    |  Portfolio heat? Can we afford?  |
                    +----------------+----------------+
                                     |
                                     v
                    +----------------+----------------+
                    |  Layer 3: PDT Manager            |
                    |  Do we have day-trade budget?    |
                    +----------------+----------------+
                                     |
                                     v
                    +----------------+----------------+
                    |  Layer 4: Bracket Order          |
                    |  Submit with stop-loss +         |
                    |  take-profit to Alpaca           |
                    +----------------+----------------+
                                     |
                                     v
                          +----------+----------+
                          |   Trade Executed     |
                          |  (with server-side   |
                          |   safety net)        |
                          +---------------------+
```

If **any** layer says "no," the trade is rejected. The signal simply does not execute.

---

## The Five Layers of Risk Control

### Layer 1: Position Sizing

**Source file:** `src/ai_trade/risk/position_sizer.py`

**Trading concept -- Fixed-fractional sizing:** Instead of buying a fixed number of shares every time, you risk a fixed *percentage* of your account on each trade. This means position sizes automatically shrink as your account shrinks (protecting you during losing streaks) and grow as your account grows (compounding winners).

**How it works in code:**

The `PositionSizer.calculate_shares()` method runs five steps in sequence. Each step can only *reduce* the number of shares -- never increase beyond what the previous step allowed.

#### Step 1: Dollar risk budget

```python
risk_amount = account_equity * max_risk_pct
```

With a $500 account and `max_risk_per_trade_pct = 0.02` (2%), the risk budget is **$10**. This is the maximum dollar amount you are willing to lose if the trade hits your stop-loss.

#### Step 2: Per-share risk

```python
risk_per_share = abs(signal.entry_price - signal.stop_loss_price)
```

This is the gap between where you enter and where your stop-loss is. If you buy at $10.00 and your stop-loss is at $9.50, then `risk_per_share = $0.50`. If the stock drops to $9.50, you lose $0.50 for every share you hold.

If this value is zero (entry and stop are the same price), the sizer returns 0 shares -- it would mean infinite position size, which is obviously wrong. The logger emits a `zero_risk_per_share` warning so you can investigate.

#### Step 3: Shares from risk budget

```python
shares = math.floor(risk_amount / risk_per_share)
```

`math.floor()` rounds *down* to the nearest whole number (you cannot buy fractional shares through bracket orders). With $10 risk and $0.50 per-share risk: `floor(10 / 0.50) = 20 shares`.

#### Step 4: Concentration limit

```python
max_dollar_value = account_equity * max_position_pct   # 25% = $125
if shares * signal.entry_price > max_dollar_value:
    shares = math.floor(max_dollar_value / signal.entry_price)
```

Even if the risk calculation says you can buy 20 shares, the total dollar value of the position cannot exceed 25% of equity ($125). At $10/share, 20 shares = $200 which exceeds $125, so shares would be reduced to `floor(125 / 10) = 12`.

**Why 25%?** This prevents a single stock from dominating the portfolio. If it gaps down overnight (before the stop-loss can trigger), the damage is limited to 25% of the account.

#### Step 5: Cash constraint

```python
shares = min(shares, math.floor(available_cash / signal.entry_price))
```

You cannot spend more money than you actually have in the account. If you already have open positions using some of your cash, this further limits the size.

#### Step 6: Minimum 1 share

```python
if shares == 0 and available_cash >= signal.entry_price > 0:
    shares = 1
```

If all the math rounds down to zero but you can afford at least one share, buy one. This prevents the bot from doing nothing at all on cheap stocks.

#### Full Example Walkthrough

| Given | Value |
|-------|-------|
| Account equity | $500 |
| Available cash | $350 (some cash already in other positions) |
| Entry price | $12.00 |
| Stop-loss price | $11.20 |

| Step | Calculation | Result |
|------|-------------|--------|
| Risk budget | $500 * 2% | $10.00 |
| Per-share risk | \|$12.00 - $11.20\| | $0.80 |
| Shares from risk | floor($10.00 / $0.80) | 12 shares |
| Dollar value check | 12 * $12 = $144 vs $150 limit | 12 shares (under limit) |
| Cash constraint | floor($350 / $12) = 29 | 12 shares (already lower) |
| **Final** | | **12 shares ($144)** |

If the stop-loss triggers, the loss is 12 * $0.80 = **$9.60** (just under the $10 budget).

---

### Layer 2: Risk Manager Gate

**Source file:** `src/ai_trade/risk/risk_manager.py`

The `RiskManager` is a centralized gatekeeper. Every trade must pass **all four checks** via the `approve_trade()` method before the order is submitted. If any single check fails, the method short-circuits and returns `(False, reason)`.

#### Check 1: Daily Loss Limit

```python
loss_pct = (starting_equity - current_equity) / starting_equity
```

If equity has dropped more than **5%** from the day's starting equity ($25 on a $500 account), all trading halts for the rest of the day.

**Trading concept -- Revenge trading:** After a losing trade, traders often feel the urge to "make it back" by taking riskier and riskier trades. This is called revenge trading, and it almost always makes things worse. The circuit breaker eliminates this by removing the ability to trade at all.

The starting equity is cached at market open:

```python
risk.set_starting_equity(float(account.equity))
```

This method stores the value in `self._starting_equity`. If it has not been set (e.g., the bot restarted mid-day), the check passes by default with the message `"no starting equity set"` -- a safe-ish default since the other layers still protect the account.

#### Check 2: Position Concentration

```python
if current_positions_count >= max_open_positions:  # default: 5
    return False, f"max positions reached: {count}/{max}"
```

Maximum **5 open positions** at any time. With $500, this means roughly $100 per position (if equally sized).

**Why limit positions?** Two reasons:
1. **Commissions and slippage** eat into profits -- more positions means more friction.
2. **Monitoring burden** -- the bot and you can only track so many positions meaningfully.

Ironically, with $500 you usually cannot even fill 4 positions because the position sizer and cash constraint limit you first.

#### Check 3: Portfolio Heat

```python
total_risk = 0.0
for trade in open_trades:
    trade_risk = abs(entry - stop) * shares
    total_risk += trade_risk

heat_pct = total_risk / current_equity
```

**Trading concept -- Portfolio heat:** This is the *total amount of money at risk across all open positions*, expressed as a percentage of equity. If every open trade hit its stop-loss simultaneously, portfolio heat is the total loss you would take.

The limit is **6% of equity** ($30 on a $500 account). Even though each individual trade risks only 2%, having three trades open means 6% total -- so you effectively cannot open a fourth trade until one closes.

| Open Trades | Per-Trade Risk | Total Heat | Under 6%? |
|-------------|---------------|------------|-----------|
| 1 trade | $10 | $10 (2%) | Yes |
| 2 trades | $10 each | $20 (4%) | Yes |
| 3 trades | $10 each | $30 (6%) | Exactly at limit |
| 4th trade | $10 | $40 (8%) | **No -- blocked** |

#### Check 4: Affordability

```python
cost = shares * signal.entry_price
if cost > available_cash:
    return False, f"insufficient cash: need ${cost:.2f}, have ${available_cash:.2f}"
```

A final sanity check: even if everything else passes, you must actually have the cash. The position sizer already accounts for this, but this is a redundant safety net.

#### The `approve_trade()` Flow

```
approve_trade() called
    |
    +-- check_daily_loss_limit() --> FAIL? return (False, reason)
    |
    +-- check_concentration()    --> FAIL? return (False, reason)
    |
    +-- check_portfolio_heat()   --> FAIL? return (False, reason)
    |
    +-- affordability check      --> FAIL? return (False, reason)
    |
    +-- All passed               --> return (True, "approved")
```

---

### Layer 3: PDT Manager

**Source file:** `src/ai_trade/risk/pdt_manager.py`

#### The PDT Rule (Background)

FINRA (the Financial Industry Regulatory Authority) enforces the **Pattern Day Trader** rule: any brokerage account with less than $25,000 in equity is limited to **3 day trades per 5 rolling business days**. A "day trade" means buying and selling (or selling and buying) the same security on the same calendar day.

Violating this rule can result in the broker freezing the account for 90 days. This is not a soft limit -- it is a regulatory hard stop.

#### Budget System

The bot has 3 day trades available per the rule, but it **reserves 1** for emergency exits, leaving an effective budget of **2 day trades**.

```python
def can_day_trade(self) -> bool:
    max_trades = 3
    reserve = 1
    used = self.get_day_trades_used()
    return used < max_trades - reserve   # used < 2
```

The method `get_day_trades_used()` counts day trades recorded in the database within the last 5 business days:

```python
def get_day_trades_used(self) -> int:
    cutoff = self._five_business_days_ago()
    trades = self._database.get_day_trades_since(cutoff.isoformat())
    return len(trades)
```

#### Weekend-Aware Lookback

The 5-business-day window must skip weekends. The `_five_business_days_ago()` static method walks backward from today, only counting Monday-Friday:

```python
@staticmethod
def _five_business_days_ago() -> date:
    today = date.today()
    biz_days = 0
    cursor = today
    while biz_days < 5:
        cursor -= timedelta(days=1)
        if cursor.weekday() < 5:   # 0=Mon ... 4=Fri
            biz_days += 1
    return cursor
```

In Python, `date.weekday()` returns 0 for Monday through 6 for Sunday. `weekday() < 5` means "is it a weekday?" The loop walks backward one calendar day at a time, incrementing `biz_days` only on weekdays. On a Monday, this looks back to the prior Monday (skipping Saturday and Sunday).

#### Why Reserve 1 Day Trade?

Imagine this scenario: you bought a stock on Tuesday as a **swing trade** (intended to hold multiple days). On Wednesday morning, the company reports terrible earnings and the stock gaps down 15% at market open. You need to sell *immediately* to limit losses -- but selling the same day you have an open position would count as a day trade.

If you had already used all 3 day trades, you would be **trapped** in the position, unable to sell without violating the PDT rule. The reserved slot prevents this.

#### Conviction Gate for Day Trades

The signal aggregator (not in the PDT manager itself, but in the trading pipeline) enforces an additional rule: day trades are only permitted if the signal's conviction score is at least **0.80** (on a 0-1 scale). This means only the highest-confidence setups consume the scarce day-trade budget.

---

### Layer 4: Bracket Orders

**Source file:** `src/ai_trade/execution/order_manager.py`

#### What Is a Bracket Order?

A bracket order is actually three orders bundled together:

```
                   +--- Take-Profit (limit sell) ----> Profit locked in
                   |
Market Buy --------+
                   |
                   +--- Stop-Loss (stop sell)  ------> Loss capped
```

1. **Market buy**: Executes immediately at the best available price.
2. **Stop-loss** (server-side): A sell order that triggers if the price drops to a specified level.
3. **Take-profit** (server-side): A sell order that triggers if the price rises to a specified level.

When one of the exit orders fills, the other is automatically canceled by Alpaca's servers. You can never accidentally sell twice.

#### Why Server-Side Stops Are Critical

The stop-loss and take-profit orders live on **Alpaca's servers**, not in the bot's code. This is important because:

| Scenario | Software-only stop | Server-side stop |
|----------|-------------------|------------------|
| Bot crashes | **Stop disappears.** No protection. | Stop persists on Alpaca's servers. |
| Internet goes down | **Stop disappears.** No protection. | Stop persists. |
| Power outage | **Stop disappears.** No protection. | Stop persists. |
| Bot restarts mid-day | Stop must be re-created (risky gap). | Already active. |

With server-side stops, even a total bot failure leaves all positions protected.

#### Time-in-Force Settings

```python
tif = (
    TimeInForce.DAY
    if signal.hold_type.value == "day"
    else TimeInForce.GTC
)
```

- **`TimeInForce.DAY`**: The entire bracket order (entry + stops) expires at market close. Used for day trades -- this way, positions auto-close if neither the stop nor the target is hit by end of day.
- **`TimeInForce.GTC`** (Good-Till-Canceled): The order stays active indefinitely until filled or manually canceled. Used for swing trades that may be held for days or weeks.

#### ATR-Based Stop and Target Placement

The stop-loss and take-profit prices are set using the **ATR** (Average True Range), a measure of how much a stock typically moves in a day.

- **Stop-loss**: `entry_price - 1.5 * ATR`
- **Take-profit**: `entry_price + 3.0 * ATR`

This creates a **2:1 reward-to-risk ratio**: the potential profit is twice the potential loss. Even if only 40% of trades are winners, the system is profitable over time:

```
10 trades:
  4 winners * $20 profit each = $80
  6 losers  * $10 loss each   = $60
  Net profit                  = $20
```

**Why ATR instead of fixed percentages?** A stock that moves $2/day and one that moves $0.20/day need very different stop distances. ATR adapts automatically -- volatile stocks get wider stops, calm stocks get tighter ones.

#### Order Submission in Code

The `submit_bracket_order()` method constructs a `MarketOrderRequest` with the bracket class:

```python
request = MarketOrderRequest(
    symbol=signal.symbol,
    qty=shares,
    side=OrderSide.BUY,
    time_in_force=tif,
    order_class=OrderClass.BRACKET,
    stop_loss=StopLossRequest(stop_price=round(signal.stop_loss_price, 2)),
    take_profit=TakeProfitRequest(limit_price=round(signal.take_profit_price, 2)),
)
```

Key details:
- `OrderClass.BRACKET` tells Alpaca to treat this as three linked orders.
- Prices are rounded to 2 decimal places (cents) since US stock exchanges do not accept sub-penny prices.
- On success, the trade is recorded in the local SQLite database with status `"open"`.
- On failure (network error, insufficient funds, etc.), the exception is logged and the method returns `None`.

---

### Layer 5: Market Regime Filter

**Source file:** `src/ai_trade/sentiment/market_regime.py`

#### Trading Concept -- Market Regime

Most individual stocks move in the same direction as the overall market. When the S&P 500 is crashing, even "good" stocks tend to fall. The market regime filter analyzes broad market conditions and modifies (or blocks) individual trade signals accordingly.

#### How the Regime Is Determined

The `MarketRegimeAnalyzer.analyze()` method examines three market benchmarks:

1. **SPY** (S&P 500 ETF): The broadest measure of US stock market health. Analyzed using:
   - EMA stack (9, 20, 50, 200-day exponential moving averages)
   - RSI (Relative Strength Index -- measures if the market is overbought/oversold)
   - MACD histogram (momentum direction)
   - 5-day price return

2. **QQQ** (Nasdaq 100 ETF): Technology-heavy index, often leads market direction.
   - Analyzed with 20 and 50-day EMAs

3. **VIX** (Volatility Index): Often called the "fear gauge." A high VIX means options traders expect large moves (uncertainty).
   - Below 15: Low fear (complacent)
   - 15-22: Normal
   - 22-30: Elevated fear
   - Above 30: Panic

These indicators are combined into a **breadth score** ranging from -1.0 (everything bearish) to +1.0 (everything bullish). The breadth score maps to a regime:

| Breadth Score | Regime | Meaning |
|--------------|--------|---------|
| >= 0.6 | Strong Bull | Everything is trending up, low fear |
| >= 0.2 | Bull | Generally positive conditions |
| >= -0.2 | Neutral | Mixed signals |
| >= -0.6 | Bear | Most indicators negative |
| < -0.6 | Strong Bear | Broad market breakdown |

#### Regime Modifiers

Each regime applies two multipliers and two boolean gates to every trade signal:

| Regime | Conviction Modifier | Position Size Modifier | Allow Longs | Allow Options |
|--------|:-------------------:|:---------------------:|:-----------:|:-------------:|
| Strong Bull | 1.3x | 1.0x | Yes | Yes |
| Bull | 1.1x | 1.0x | Yes | Yes |
| Neutral | 0.9x | 0.75x | Yes | Yes |
| Bear | 0.6x | 0.5x | Yes (high conviction only) | No |
| Strong Bear | 0.3x | 0.25x | No | No |

**Conviction modifier**: Multiplied against the signal's raw conviction score. In a Strong Bull regime, a 0.70 conviction becomes 0.91 (0.70 * 1.3). In a Bear regime, the same 0.70 becomes 0.42 (0.70 * 0.6) -- likely below the minimum threshold to execute.

**Position size modifier**: Multiplied against the number of shares from the position sizer. In a Bear market, you only take half-sized positions. In a Strong Bear, quarter-sized.

**Allow longs / Allow options**: Boolean gates. In a Strong Bear, no new long (buy) stock positions at all. In a Bear or Strong Bear, no options trades (options are leveraged and decay in value -- too risky when the market is falling).

#### VIX Override

```python
if vix_level > 30 and regime in (MarketRegime.STRONG_BULL, MarketRegime.BULL):
    regime = MarketRegime.NEUTRAL
```

Even if every other indicator looks bullish, a VIX above 30 (panic territory) forces a downgrade to Neutral. History shows that sudden VIX spikes often precede major selloffs, even when prices have not yet dropped.

---

## Options Risk Controls

Options are contracts that give the right (but not the obligation) to buy or sell a stock at a specific price by a specific date. They are inherently leveraged instruments -- a small move in the stock can cause a 50-100% move in the option price, in either direction. They can also expire worthless (total loss of the amount paid).

Because of this higher risk, options have their own dedicated limits (configured in `config/settings.yaml` under the `options:` key):

| Control | Config Key | Default | Effect on $500 Account |
|---------|-----------|---------|----------------------|
| Max risk per options trade | `max_single_options_risk_pct` | 12% | $60 max at risk per trade |
| Max capital in options | `max_options_capital_pct` | 50% | $250 total across all options |
| Max concurrent options positions | `max_options_positions` | 3 | No more than 3 open at once |

**Auto-close before expiration**: Options positions are automatically closed 1 day before their expiration date. This prevents two dangerous scenarios:
- **Pin risk**: The stock hovers near the strike price at expiration, making it unpredictable whether the option will be exercised.
- **Unexpected assignment**: For short options (sold contracts), the holder on the other side can exercise at any time near expiration, forcing you to buy or deliver shares you may not be able to afford.

---

## News Sentiment as Risk Gate

**Source file:** `src/ai_trade/sentiment/news_sentiment.py`

News about a company can override any technical signal. A perfect chart setup is meaningless if the company just announced an SEC investigation.

The news sentiment system scores recent articles for each symbol and modifies the trade:

### Score Interpretation

| News Score Range | Action |
|-----------------|--------|
| < -0.5 | **Trade blocked entirely** (unless a positive catalyst is detected) |
| -0.5 to 0.0 | Conviction reduced by up to 50% |
| 0.0 to +0.5 | No modification |
| > +0.5 | Conviction boosted by up to 30% |

### Final Conviction Gate

After **all** modifiers have been applied (market regime modifier, news modifier, signal aggregation), a final check runs:

```
min_conviction_after_mods: 0.45
```

Any signal with a conviction below **0.45** after all modifiers is rejected. This is the last line of defense -- even if no single modifier was enough to block the trade, the cumulative dampening might push it below the threshold.

**Example flow:**

```
Raw signal conviction:    0.75
Market regime (Neutral):  * 0.9  = 0.675
News sentiment (mildly bearish): * 0.7 = 0.473
Final conviction:         0.473  (above 0.45 -- trade proceeds)

Raw signal conviction:    0.60
Market regime (Bear):     * 0.6  = 0.360
News sentiment (neutral): * 1.0  = 0.360
Final conviction:         0.360  (below 0.45 -- REJECTED)
```

---

## Additional Risk Controls (v1.2.0)

### Conviction Floors

Every strategy uses **additive conviction scoring** (0.50-0.90 range) rather than linear scaling. Two minimum thresholds apply after scoring:

| Threshold | Value | Purpose |
|-----------|-------|---------|
| `risk.min_conviction_for_swing` | 0.55 | Minimum conviction for swing trades (post-weighting). Signals below this are filtered out. |
| `sentiment.min_conviction_after_mods` | 0.45 | Absolute floor after regime + news modifiers are applied. |

### Failed Symbol Blacklist

If an order for a given symbol fails **2 consecutive times** (e.g., due to halts, insufficient liquidity, or API errors), that symbol is excluded from further trading for the rest of the day. This prevents the bot from repeatedly attempting orders on problematic stocks.

### PDT Pre-Check

Before submitting any order that would consume a day-trade slot, the bot calls `pdt.can_day_trade()` first. This prevents order submission failures that previously wasted API calls and logged confusing rejection messages.

### Adaptive Strategy Weighting

The `StrategyWeighter` adjusts each strategy's conviction multiplier based on historical performance. Strategies start at weight 1.0 and adjust after a burn-in period (default 10 closed trades). Poorly-performing strategies are weighted down (minimum 0.3x), while strong performers are weighted up (maximum 2.0x). This is an additional risk control because it automatically reduces exposure to strategies that are losing money.

---

## Daily Loss Circuit Breaker

This is arguably the single most important risk control. It works like a fuse in an electrical panel -- when the current (losses) exceeds a safe level, the fuse blows and cuts off all power (trading).

**Trigger:** If equity drops **5% or more** from the day's opening equity, all trading halts for the rest of the day.

| Account Size | 5% Loss Limit | Meaning |
|-------------|--------------|---------|
| $500 | $25 | After losing $25 in a day, trading stops |
| $475 (next day) | $23.75 | The limit automatically tightens |
| $525 (good day) | $26.25 | The limit loosens as equity grows |

**How it works in code:**

At market open, the bot caches the current equity:

```python
risk.set_starting_equity(float(account.equity))  # e.g., 500.0
```

Before every trade, `check_daily_loss_limit()` runs:

```python
loss_pct = (starting_equity - current_equity) / starting_equity
if loss_pct > 0.05:
    return False, "daily loss limit hit"
```

**Why this prevents revenge trading:** After a losing trade, the natural human (and algorithmic) impulse is to "make it back" by trading more aggressively. Studies show this almost always makes things worse. By cutting off trading entirely, the circuit breaker removes the opportunity to compound losses. The bot simply waits until the next trading day, when the limit resets.

---

## Position Reconciliation

**Source file:** `src/ai_trade/execution/order_manager.py` (method: `sync_positions()`)

The bot maintains a local SQLite database of all trades. But the *source of truth* is always Alpaca's servers -- positions can be closed by server-side stop-losses or take-profits without the bot knowing.

The `sync_positions()` method runs **every 60 seconds** via a scheduled job and reconciles the two data sources:

```
Alpaca Positions          Local Database
    |                         |
    +--- Compare symbols -----+
    |                         |
    |  In Alpaca but NOT      |  In DB but NOT
    |  in DB?                 |  in Alpaca?
    |  --> WARN "untracked"   |  --> CLOSE in DB (mark as "closed")
    |                         |     (stop or take-profit filled)
    v                         v
         Reconciled State
```

### What each case means:

**Position in Alpaca but not in DB ("untracked"):**
The bot logs a warning. This could mean a trade was placed manually through the Alpaca dashboard, or the DB failed to record the initial trade. The bot does not automatically close it -- that would be dangerous. It just alerts you.

**Trade marked "open" in DB but no Alpaca position ("stale"):**
This means the position was closed by a server-side stop-loss or take-profit. The bot updates the DB record to `status = "closed"` and logs the event. This is normal and expected -- it means the bracket order did its job.

### Reconciliation summary

The method returns a dictionary with counts:

```python
{
    "alpaca_positions": 2,      # Currently held
    "db_open_trades": 3,        # DB thinks are open
    "untracked_positions": 0,   # In Alpaca, not in DB
    "stale_trades_closed": 1,   # DB entry cleaned up
}
```

---

## Emergency Procedures

### Scenario 1: Ctrl+C (Graceful Shutdown)

The bot's scheduler stops and no new trades are placed. **All existing positions remain open** with their server-side stop-losses and take-profits active on Alpaca. This is safe -- the bracket orders continue protecting the positions even though the bot is no longer running.

### Scenario 2: Bot Crash (Unexpected Failure)

Same as above. Because all stop-losses and take-profits are server-side bracket orders, they persist on Alpaca's infrastructure regardless of whether the bot is running. There is no gap in protection.

When the bot restarts, `sync_positions()` will reconcile any positions that were closed by their stops while the bot was down.

### Scenario 3: Position Gaps Down Overnight

A swing trade bought on Monday might open 10% lower on Tuesday morning due to after-hours news. The stop-loss cannot protect against this gap (it only triggers during market hours at the specified price, and the market may open *below* the stop price).

This is where the **reserved PDT day-trade slot** comes in. You can sell the position on Tuesday (same day it gapped down) using the reserved slot, rather than being forced to hold it and hope it recovers.

### Scenario 4: Everything Goes Wrong

If you need to close *all* positions immediately (market crash, breaking news, etc.):

1. The `cancel_all_open_orders()` method cancels every pending order on Alpaca.
2. Positions can be closed through the Alpaca dashboard directly.
3. The next `sync_positions()` run will reconcile the DB automatically.

---

## Risk Parameters Reference Table

All values are configured in `config/settings.yaml`.

| Parameter | Config Key | Default | With $500 Account |
|-----------|-----------|---------|-------------------|
| Max risk per trade | `account.max_risk_per_trade_pct` | 2% | $10 |
| Max single position | `account.max_position_pct` | 25% | $125 |
| Max open positions | `account.max_open_positions` | 5 | -- |
| Daily loss limit | `account.daily_loss_limit_pct` | 5% | $25 |
| Portfolio heat cap | `risk.max_portfolio_heat_pct` | 6% | $30 |
| Stop loss | `risk.stop_loss_pct` | 3% | -- |
| Take profit | `risk.take_profit_pct` | 6% | -- |
| PDT day trades | `pdt.max_day_trades` | 3 | -- |
| PDT reserve | `pdt.day_trade_reserve` | 1 | -- |
| Min day trade conviction | `pdt.min_conviction_for_day_trade` | 0.80 | -- |
| Options max risk | `options.max_single_options_risk_pct` | 12% | $60 |
| Options capital limit | `options.max_options_capital_pct` | 50% | $250 |
| Options max positions | `options.max_options_positions` | 3 | -- |
| Min conviction for swing | `risk.min_conviction_for_swing` | 0.55 | -- |
| Min conviction after mods | `sentiment.min_conviction_after_mods` | 0.45 | -- |
| News block threshold | `sentiment.block_on_bearish_news` | -0.5 | -- |

---

### How These Parameters Interact

The parameters form a layered constraint system. Here is how they combine in a worst-case scenario:

```
Starting equity:     $500
Max risk per trade:  2% = $10
Max positions:       4
Portfolio heat cap:  6% = $30

If each trade risks $10:
  - 3 trades open = $30 heat (6%) = AT LIMIT
  - 4th trade blocked by portfolio heat check
  - Even though max_open_positions allows 4

If daily loss limit hits ($25 loss):
  - ALL checks become irrelevant
  - No trading until tomorrow
```

The tightest constraint always wins. In practice, portfolio heat (6%) is almost always the binding constraint before position count (4) on a $500 account.
