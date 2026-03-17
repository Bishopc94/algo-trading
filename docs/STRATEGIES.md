# Trading Bot Strategies Guide

This document is a comprehensive reference for every strategy the bot can execute. It is written for someone comfortable with software engineering concepts but not necessarily familiar with Python or financial markets. Python-specific constructs and trading theory are explained inline throughout.

---

## Table of Contents

- [Foundational Concepts](#foundational-concepts)
- [Stock Strategies](#stock-strategies)
  - [1. Mean Reversion](#1-mean-reversion)
  - [2. Momentum](#2-momentum)
  - [3. VWAP Reclaim](#3-vwap-reclaim)
- [Options Strategies](#options-strategies)
  - [Options Primer](#options-primer)
  - [4. Credit Put Spread](#4-credit-put-spread-bull-put-spread)
  - [5. Debit Call Spread](#5-debit-call-spread-bull-call-spread)
  - [6. Long Call](#6-long-call)
  - [7. Long Put](#7-long-put)
  - [8. Cash Secured Put](#8-cash-secured-put)
  - [9. Covered Call](#9-covered-call)
  - [10. Covered Straddle](#10-covered-straddle)
  - [11. Momentum Options](#11-momentum-options)
- [Signal Ranking: The Brain](#signal-ranking-the-brain)
- [Key Concepts Glossary](#key-concepts-glossary)

---

## Foundational Concepts

Before diving into strategies, here are the building blocks that every strategy relies on.

### Technical Indicators Used

| Indicator | What It Measures | Range / Units |
|-----------|-----------------|---------------|
| **RSI** (Relative Strength Index) | Whether a stock is overbought or oversold | 0-100 (below 30 = oversold, above 70 = overbought) |
| **EMA** (Exponential Moving Average) | Smoothed trend of recent prices, weighting recent bars more | Same units as price (dollars) |
| **VWAP** (Volume-Weighted Average Price) | The "fair price" for the day, weighted by volume at each price level | Same units as price (dollars) |
| **ATR** (Average True Range) | How much a stock typically moves in one bar (volatility) | Dollar amount |
| **Bollinger Bands** | A volatility envelope around a moving average (upper, middle, lower) | Same units as price (dollars) |
| **MACD** (Moving Average Convergence Divergence) | Trend momentum by comparing two EMAs | Dimensionless |

### Hold Types and the PDT Rule

The **Pattern Day Trader (PDT) Rule** is a U.S. regulation: if your account has less than $25,000 in equity, you are limited to **3 day trades in any rolling 5-business-day window**. A "day trade" is any position opened and closed on the same calendar day.

This bot tracks two hold types:

| Hold Type | Description | PDT Cost |
|-----------|-------------|----------|
| **SWING** | Position held overnight (at minimum). Closed on a future day. | **0** (free) |
| **DAY** | Position opened and closed on the same day. | **1 slot** (out of 3 available) |

The bot manages a "PDT budget" — it knows how many day-trade slots remain and refuses to open a DAY trade if the budget is exhausted.

### Conviction Scores

Every strategy produces a **conviction score** between 0.0 and 1.0. This score represents how strongly the strategy believes in the trade. It is used to:

1. **Rank signals** against each other (higher conviction = higher priority).
2. **Size the position** (higher conviction = larger allocation).
3. **Decide hold type** (some strategies promote to DAY only at very high conviction).

### Bracket Orders

A **bracket order** is a set of three linked orders submitted simultaneously:

1. **Entry order** — the buy (or sell-to-open) that opens the position.
2. **Stop-loss order** — a sell order placed below the entry price. If the stock drops to this level, the position is closed automatically to limit loss.
3. **Take-profit order** — a sell order placed above the entry price. If the stock rises to this level, the position is closed automatically to lock in profit.

When either the stop-loss or take-profit fills, the other is automatically canceled. This is also called an OCO (one-cancels-other) mechanism.

---

## Stock Strategies

### 1. Mean Reversion

> **Hold Type:** SWING (held overnight, no PDT cost)

#### Theory

Markets tend to **revert to a mean**. When a stock drops temporarily but its longer-term trend is still pointing upward, the drop is likely a pullback — not a reversal — and the price will bounce back.

Think of a rubber band stretched downward while someone is pulling the other end upward. The further it stretches from its resting position, the harder it snaps back. In market terms, the "resting position" is a moving average, and the "stretch" is how far below that average the stock has fallen.

Mean reversion strategies profit by buying during these temporary dips and selling once the price returns to (or overshoots) its average.

#### Entry Conditions

All three of these must be true simultaneously:

| Condition | Code Equivalent | What It Means |
|-----------|----------------|---------------|
| RSI < 40 | `rsi < 40` | The stock is **oversold** — selling pressure has been heavy recently. RSI (Relative Strength Index) measures the ratio of recent up-moves to down-moves on a 0-100 scale. Below 30 is classically "oversold"; we use 40 as a less aggressive threshold. |
| Close > EMA-20 * 0.98 | `close > ema_20 * 0.98` | The stock is **still in an uptrend**. Even though it dipped, the current price is within 2% of the 20-day Exponential Moving Average. The EMA-20 gives more weight to recent prices than a simple average. If the stock were crashing through its trend, this condition would fail and we'd stay out. |
| Close < BB_lower * 1.03 | `close < bb_lower * 1.03` | The stock is **near the lower Bollinger Band**. Bollinger Bands are an envelope drawn 2 standard deviations above and below a 20-period moving average. The lower band represents a statistically extreme low. Being within 3% of it confirms the stock is stretched to the downside. |

**In plain English:** "The stock is oversold, near statistical support, but hasn't broken its uptrend."

#### How RSI Works (Detailed)

RSI is calculated over a lookback window (typically 14 bars):

```
RS = average_gain_over_14_bars / average_loss_over_14_bars
RSI = 100 - (100 / (1 + RS))
```

| RSI Value | Interpretation |
|-----------|---------------|
| 0-30 | Strongly oversold |
| 30-40 | Mildly oversold |
| 40-60 | Neutral |
| 60-70 | Mildly overbought |
| 70-100 | Strongly overbought |

#### How Bollinger Bands Work (Detailed)

Bollinger Bands consist of three lines:

```
Middle Band = 20-period Simple Moving Average (SMA)
Upper Band  = Middle Band + 2 * standard_deviation(close, 20)
Lower Band  = Middle Band - 2 * standard_deviation(close, 20)
```

Because of how standard deviations work, roughly 95% of price action stays within the bands. When price touches or crosses the lower band, it's at a statistical extreme.

**Bollinger Band Width** (`bb_width`) measures volatility:

```
bb_width = (upper_band - lower_band) / middle_band
```

A small bb_width means tight bands (low volatility); a large bb_width means wide bands (high volatility).

#### How EMA Works (Detailed)

An Exponential Moving Average weights recent prices more heavily than older ones:

```
multiplier = 2 / (period + 1)
EMA_today = (close_today * multiplier) + (EMA_yesterday * (1 - multiplier))
```

For a 20-period EMA, the multiplier is `2 / 21 = 0.0952`. Today's close gets about 9.5% weight; yesterday's EMA gets 90.5%. This makes the EMA respond faster to recent price changes than a simple average.

#### Conviction Calculation

Conviction scales **linearly** from 0.5 to 1.0 as RSI drops from 40 to 20:

```
conviction = 0.5 + 0.5 * (40 - rsi) / (40 - 20)
```

| RSI | Conviction | Interpretation |
|-----|-----------|----------------|
| 40 | 0.50 | Minimum — barely oversold |
| 35 | 0.625 | Moderate |
| 30 | 0.75 | Strong |
| 25 | 0.875 | Very strong |
| 20 | 1.00 | Maximum — deeply oversold, likely to bounce |

The lower the RSI, the more "stretched" the rubber band is, and the more confident we are in a snapback.

#### Stop-Loss and Take-Profit (ATR-Based)

ATR (Average True Range) measures how much a stock typically moves in one bar. It is calculated as the 14-period average of:

```
true_range = max(
    high - low,
    abs(high - previous_close),
    abs(low - previous_close)
)
```

The bot uses ATR to set dynamic stops that adapt to each stock's volatility:

```
stop_loss   = entry_price - 1.5 * ATR
take_profit = entry_price + 3.0 * ATR
```

This gives a **2:1 reward-to-risk ratio**:

| Component | Formula | Example (entry=$50, ATR=$1.00) |
|-----------|---------|-------------------------------|
| Risk (distance to stop) | 1.5 * ATR | $1.50 below entry = $48.50 |
| Reward (distance to target) | 3.0 * ATR | $3.00 above entry = $53.00 |
| Reward : Risk | 3.0 / 1.5 | 2:1 |

With a 2:1 ratio, you only need to be right 34% of the time to break even (ignoring fees).

#### Exit Conditions

The bot also monitors for an early exit (before the bracket orders trigger):

- **RSI > 60** — the stock has "mean reverted." It was oversold (RSI < 40) and is now back in neutral-to-overbought territory. The thesis has played out.
- **Close > Upper Bollinger Band** — the stock has swung all the way from the lower band to the upper band, an even stronger mean reversion.

Either condition triggers a market sell, canceling the bracket orders.

---

### 2. Momentum

> **Hold Type:** Adaptive — DAY if conviction >= 0.9, else SWING

#### Theory

Stocks that **break out to new highs on heavy volume** tend to keep going. This is the opposite philosophy of mean reversion: instead of betting on a snapback, you're betting on continuation.

The key insight is **volume confirmation**. A stock can briefly touch a new high on thin volume — that's a fake-out. But when a breakout happens on 2x, 3x, or 5x normal volume, it means real money (often institutional) is pouring in. Supply has been absorbed and demand is overwhelming it.

#### Entry Conditions

All four must be true:

| Condition | Code Equivalent | What It Means |
|-----------|----------------|---------------|
| Close > 20-day high | `close > high_20d` | The stock is making a **new 20-day high** — a breakout. The `high_20d` is the highest closing price over the last 20 trading days. |
| Relative volume > 1.5x | `rel_vol > 1.5` | Today's volume is at least **1.5 times the 20-day average volume**. This confirms the breakout has real participation. |
| ADR > 2% | `adr > 0.02` | **Average Daily Range** exceeds 2%. ADR = average of `(high - low) / close` over recent bars. This filters out low-volatility stocks where a "breakout" might only be a few cents. |
| Close > EMA-20 | `close > ema_20` | The stock is **above its short-term trend**, confirming upward momentum. |

#### Relative Volume Explained

Relative volume compares current volume to what's "normal":

```
relative_volume = current_bar_volume / average_volume_20_days
```

| Relative Volume | Interpretation |
|-----------------|---------------|
| < 1.0 | Below average — quiet day |
| 1.0 - 1.5 | Normal activity |
| 1.5 - 3.0 | Elevated — something is happening |
| 3.0 - 5.0 | High — likely news or institutional activity |
| 5.0+ | Extremely high — major catalyst |

#### Conviction Calculation

Conviction scales linearly with relative volume:

```
if rel_vol <= 1.5:  conviction = 0.50
if rel_vol >= 5.0:  conviction = 1.00
else:               conviction = 0.50 + 0.50 * (rel_vol - 1.5) / (5.0 - 1.5)
```

> **Python note:** The code uses `min()` and `max()` built-in functions to clamp the result between 0.5 and 1.0. `min(1.0, max(0.5, value))` ensures the conviction never goes below 0.5 or above 1.0.

| Relative Volume | Conviction |
|-----------------|-----------|
| 1.5x | 0.50 |
| 2.0x | 0.571 |
| 3.0x | 0.714 |
| 4.0x | 0.857 |
| 5.0x+ | 1.00 |

#### Why "Adaptive" Hold Type?

This strategy picks its hold type based on conviction:

```python
if conviction >= 0.9:
    hold_type = HoldType.DAY
else:
    hold_type = HoldType.SWING
```

> **Python note:** `HoldType.DAY` and `HoldType.SWING` are members of a Python `Enum` — a special class that defines a fixed set of named constants. Think of it like a TypeScript `enum` or a C `#define`. `HoldType` can only be `DAY` or `SWING`.

**Why this matters:** Day trades cost a PDT slot (you only get 3 per week). The strategy only "spends" a PDT slot when conviction is very high (>= 0.9), meaning the volume is at least ~4.3x average. Lower-conviction breakouts are held as SWING trades (free PDT cost) at the price of overnight risk.

This is an **economic optimization** — the bot treats PDT slots as a scarce resource and allocates them only to the highest-confidence signals.

#### Stops and Exit

- **Stops:** Same ATR-based bracket as Mean Reversion (stop = entry - 1.5*ATR, target = entry + 3.0*ATR).
- **Exit logic:** The `should_exit()` method returns `False` — this strategy relies entirely on its bracket orders to manage the trade. There is no early-exit logic. Either the take-profit hits or the stop-loss hits.

---

### 3. VWAP Reclaim

> **Hold Type:** DAY (costs 1 PDT slot)

#### Theory

**VWAP (Volume-Weighted Average Price)** represents the true "fair price" for the day. It is calculated as:

```
VWAP = cumulative(price * volume) / cumulative(volume)
```

Unlike a simple average, VWAP weights each price by how much volume traded at that price. If a stock traded 1 million shares at $50 and only 100 shares at $55, the VWAP will be very close to $50.

VWAP is the benchmark that institutional traders (mutual funds, pension funds) use to judge execution quality. When a stock dips below VWAP and then **reclaims** it (crosses back above), it often signals that institutional buyers are stepping in at the "fair price" and pushing the stock higher.

#### Entry Conditions

| Condition | What It Means |
|-----------|---------------|
| Price reclaims VWAP from below | The current bar closes above VWAP, but a recent bar closed below it. This is the "reclaim" event. |
| Bar volume > 1.5x average | The reclaim bar has elevated volume, confirming institutional participation. |
| Recent dip below VWAP in last 10 bars | At least one of the last 10 bars had a close below VWAP. This ensures we're catching a real reclaim, not just a stock that's been above VWAP all day. |

#### Conviction Calculation

The conviction for VWAP Reclaim has a **base of 0.7** and is adjusted by two factors:

```
conviction = 0.7 + depth_adjustment + volume_adjustment
```

- **Depth-of-dip adjustment:** How far below VWAP did the stock fall? A deeper dip (e.g., 2% below VWAP) gets a larger bonus than a shallow dip (0.1% below). The logic: a deeper dip that gets reclaimed is a stronger signal.
- **Volume strength adjustment:** How strong is the reclaim volume relative to average? Volume at 3x average adds more conviction than volume at 1.6x.

The result is clamped to the [0.5, 1.0] range.

#### Stops

Unlike the ATR-based stops used by Mean Reversion and Momentum, VWAP Reclaim uses **structural stops**:

```
stop_loss = max(low_of_dip_below_vwap, vwap - 0.01 * close)
```

The stop is placed at the **low of the dip** that preceded the reclaim (the lowest price from the recent excursion below VWAP), or at VWAP minus 1% — whichever is higher.

This makes intuitive sense: if the stock falls back below the low of the dip, the reclaim has failed and the thesis is invalidated.

#### Why DAY Only?

VWAP resets at market open every day. It is a purely intraday indicator with no meaning overnight. A VWAP reclaim trade has no thesis after the close, so it must be closed the same day — making it a day trade that costs 1 PDT slot.

---

## Options Strategies

### Options Primer

If you're new to options, here is what you need to know before reading the strategies below.

#### What Is an Option?

An option is a **contract** that gives the holder the right (but not the obligation) to buy or sell 100 shares of a stock at a specific price (the **strike price**) before a specific date (the **expiration date**).

| Type | Right | You Buy When You Believe... |
|------|-------|----------------------------|
| **Call** | Right to **buy** 100 shares at the strike | The stock will go **up** |
| **Put** | Right to **sell** 100 shares at the strike | The stock will go **down** |

#### Key Options Terms

| Term | Definition |
|------|-----------|
| **Strike Price** | The price at which the option can be exercised |
| **Premium** | The price you pay (or receive) for the option contract |
| **Expiration (DTE)** | Days To Expiration — how many calendar days until the contract expires |
| **Delta** | How much the option price moves per $1 move in the stock. A 0.50-delta call gains ~$0.50 when the stock rises $1. Also approximates the probability of expiring in-the-money. |
| **Theta** | How much value the option loses per day just from time passing. Options are "wasting assets" — they lose value every day even if the stock doesn't move. |
| **IV (Implied Volatility)** | The market's estimate of future volatility baked into the option price. High IV = expensive options. |
| **OTM (Out of the Money)** | Call: strike > current price. Put: strike < current price. The option has no intrinsic value yet. |
| **ATM (At the Money)** | Strike is approximately equal to the current stock price. |
| **ITM (In the Money)** | Call: strike < current price. Put: strike > current price. The option has intrinsic value. |

#### Payoff Diagrams

Options payoffs are non-linear. Here are the four basic shapes:

**Long Call** (you pay premium, unlimited upside):
```
Profit
  |           /
  |          /
  |         /
  |--------/--------  <-- strike price
  |  max loss = premium paid
  +-----------------------> Stock Price
```

**Long Put** (you pay premium, profit as stock falls):
```
Profit
  \
   \
    \
     \--------  <-- strike price
               max loss = premium paid
  +-----------------------> Stock Price
```

**Short Call** (you receive premium, unlimited downside risk):
```
Profit
  max gain = premium received
  --------\
           \
            \
             \
  +-----------------------> Stock Price
```

**Short Put** (you receive premium, risk if stock drops to $0):
```
Profit
               max gain = premium received
          /--------
         /
        /
       /
  +-----------------------> Stock Price
```

#### Spread Width

A **spread** combines two options at different strikes. The **spread width** is the distance (in dollars) between the two strikes:

```
spread_width = abs(strike_of_leg_1 - strike_of_leg_2)
```

For example, selling a $48 put and buying a $45 put creates a spread width of $3.00. Since each contract covers 100 shares, the total exposure is $300.

---

### 4. Credit Put Spread (Bull Put Spread)

> **Direction:** Bullish | **Risk:** Defined | **PDT:** N/A (options positions)

#### Theory

A credit put spread profits when the stock stays **above** the short put strike by expiration. You collect premium upfront (the "credit") and keep it if the stock cooperates. The bought put limits your downside to a defined amount.

This is a **high-probability, low-reward** strategy. You win frequently (the stock just has to stay above a level) but each win is relatively small.

#### Construction

| Leg | Action | Delta | Purpose |
|-----|--------|-------|---------|
| Short put | Sell to open | ~0.30 delta | Generates premium. 0.30 delta means ~70% probability of expiring OTM (profitable). |
| Long put | Buy to open | Lower strike, within $2.50 of the short put | Limits maximum loss. Without this leg, you'd be exposed to the stock dropping to $0. |

> **Python note:** The bot selects strikes by iterating through the options chain (a `list` of `dict` objects, each representing one contract) and filtering for the target delta using a list comprehension: `[opt for opt in chain if abs(opt['delta']) >= 0.28 and abs(opt['delta']) <= 0.32]`. A **list comprehension** is a concise Python syntax for filtering/transforming lists — equivalent to `array.filter().map()` in JavaScript.

#### Entry Conditions

| Condition | Requirement | Rationale |
|-----------|------------|-----------|
| RSI | > 40 | Not oversold — the stock has bullish momentum |
| Price vs EMA-20 | Price > EMA-20 | Confirms uptrend |
| DTE | 20-45 days | Enough time for theta decay but not so much that capital is tied up too long |
| Credit received | >= 30% of spread width | Ensures adequate compensation for the risk. If the spread is $2.50 wide, the credit must be at least $0.75. |

#### Risk/Reward Profile

```
Max Profit = Credit Received
Max Loss   = Spread Width - Credit Received
Breakeven  = Short Put Strike - Credit Received
```

**Example:**

| Parameter | Value |
|-----------|-------|
| Stock price | $52.00 |
| Short put strike | $50.00 (0.30 delta) |
| Long put strike | $47.50 |
| Spread width | $2.50 |
| Credit received | $0.90 |
| Max profit | $0.90 ($90 per contract) |
| Max loss | $2.50 - $0.90 = $1.60 ($160 per contract) |
| Breakeven | $50.00 - $0.90 = $49.10 |
| Win probability | ~70% (based on short put delta) |

**Payoff diagram:**

```
Profit
  +$90  _______________
        |
        |
  $0  --|-------------- $49.10 (breakeven)
        |          |
        |          |
 -$160  |__________|
        $47.50    $50.00
        (long)    (short)
```

---

### 5. Debit Call Spread (Bull Call Spread)

> **Direction:** Bullish | **Risk:** Defined | **PDT:** N/A

#### Theory

A debit call spread is a **directional bet** that costs money upfront (the "debit") but profits if the stock rises above the long call strike. The sold call reduces cost but caps your upside.

This is used when you're moderately bullish — you want upside exposure but don't want to pay full price for a naked long call.

#### Construction

| Leg | Action | Delta | Purpose |
|-----|--------|-------|---------|
| Long call | Buy to open | ~0.60 delta | The primary directional bet. 0.60 delta is slightly ITM, giving a good probability of profit. |
| Short call | Sell to open | ~0.35 delta | Offsets part of the cost. This is further OTM, so it's cheaper. You give up gains beyond this strike. |

#### Entry Conditions

| Condition | Requirement | Rationale |
|-----------|------------|-----------|
| RSI | 50-70 | Stock has momentum but isn't extremely overbought |
| Price vs EMA-20 | Price > EMA-20 | Uptrend confirmed |
| Relative volume | > 1.2x | Some volume participation |
| DTE | 30-60 days | Time for the move to play out |
| Debit paid | <= 60% of spread width | Ensures at least a 0.67:1 reward-to-risk. If the spread is $5 wide, the debit must be $3.00 or less. |

#### Risk/Reward Profile

```
Max Profit = Spread Width - Debit Paid
Max Loss   = Debit Paid
Breakeven  = Long Call Strike + Debit Paid
```

**Example:**

| Parameter | Value |
|-----------|-------|
| Stock price | $50.00 |
| Long call strike | $49.00 (0.60 delta) |
| Short call strike | $53.00 (0.35 delta) |
| Spread width | $4.00 |
| Debit paid | $2.20 |
| Max profit | $4.00 - $2.20 = $1.80 ($180 per contract) |
| Max loss | $2.20 ($220 per contract) |
| Breakeven | $49.00 + $2.20 = $51.20 |

**Payoff diagram:**

```
Profit
  +$180 ____________
                   |
                   |
  $0  -------------|----  $51.20 (breakeven)
       |           |
       |           |
 -$220 |___________|
       $49.00     $53.00
       (long)     (short)
```

---

### 6. Long Call

> **Direction:** Bullish | **Risk:** Premium paid | **PDT:** N/A

#### Theory

A long call is the simplest bullish options bet: you pay a premium and profit if the stock goes up. Unlike a spread, there is **no cap on upside**. However, you pay full price (no offsetting short leg) and the entire premium is at risk if the stock doesn't move.

The bot uses this on **breakout setups** — when a stock clears its 20-day high on strong volume, suggesting a significant move is starting.

#### Entry Conditions

| Condition | Requirement | Rationale |
|-----------|------------|-----------|
| Price vs 20-day high | Close > 20-day high | Breakout confirmed |
| RSI | 55-75 | Momentum present but not wildly overbought |
| Relative volume | > 2.0x | Strong volume confirmation — higher bar than most strategies |
| IV | < 0.70 (70%) | Options aren't overpriced. High IV inflates premiums, making long calls expensive. |
| Delta | 0.50-0.70 | ATM to slightly ITM — good balance of cost and sensitivity to stock movement |
| Max cost | $75 per contract | Hard cap on premium paid — risk management |
| DTE | 20-60 days | Enough time for the breakout to develop |

#### Risk/Reward Profile

```
Max Profit = Unlimited (stock can rise without limit)
Max Loss   = Premium Paid (capped at $75/contract by the bot)
Breakeven  = Strike Price + Premium Paid
```

#### Why the IV Filter Matters

Implied Volatility directly affects option prices. The same 0.60-delta call might cost:

| IV | Premium (approximate) |
|----|----------------------|
| 30% | $2.50 |
| 50% | $4.00 |
| 70% | $5.50 |
| 100% | $7.50 |

By requiring IV < 70%, the bot avoids buying options when they're expensive (often right after earnings or major news). This is sometimes called avoiding "IV crush" — after a high-IV event, implied volatility drops and option prices can fall dramatically even if the stock moves in your favor.

---

### 7. Long Put

> **Direction:** Bearish | **Risk:** Premium paid | **PDT:** N/A

#### Theory

A long put is the mirror image of a long call — it profits when the stock **drops**. The bot uses this on **breakdown setups** — when a stock falls below its 20-day low, confirming a bearish trend.

#### Entry Conditions

| Condition | Requirement | Rationale |
|-----------|------------|-----------|
| Price vs 20-day low | Close < 20-day low | Breakdown confirmed |
| RSI | < 40 | Bearish momentum — stock is already weak |
| Price vs EMA-20 | Price < EMA-20 | Below short-term trend |
| Price vs EMA-50 | Price < EMA-50 | Below medium-term trend — **both** moving averages must confirm |
| Relative volume | > 1.5x | Volume confirms the breakdown |
| Delta | 0.45-0.65 (put delta is negative, but we reference absolute value) | Slightly ITM to ATM — good directional sensitivity |

#### Why Require Price Below BOTH EMA-20 and EMA-50?

A stock below its 20-day EMA might just be in a short pullback. But a stock below **both** its 20-day and 50-day EMA is in a confirmed downtrend at multiple timeframes. This dual-EMA filter reduces false signals.

```
Strong downtrend:  price < EMA-20 < EMA-50    (both averages above price, and short-term EMA
                                                has crossed below long-term EMA)
```

#### Risk/Reward Profile

```
Max Profit = Strike Price - Premium Paid (theoretically, stock falls to $0)
Max Loss   = Premium Paid
Breakeven  = Strike Price - Premium Paid
```

---

### 8. Cash Secured Put

> **Direction:** Neutral to bullish | **Risk:** Obligation to buy 100 shares | **PDT:** N/A

#### Theory

Selling a cash-secured put means you **collect premium** for agreeing to buy 100 shares at the strike price if the stock drops. You must have enough cash in your account to buy those shares (hence "cash secured").

This strategy is used on **cheap stocks ($2-$5)** that you'd actually want to own. If the stock stays above the strike, you keep the premium (free money). If it drops below the strike, you end up buying a stock you wanted at a discount (strike price minus premium received).

It's the options equivalent of placing a limit buy order below the current price — except you get **paid to wait**.

#### Entry Conditions

| Condition | Requirement | Rationale |
|-----------|------------|-----------|
| Stock price | $2.00 - $5.00 | Cheap enough that securing 100 shares doesn't tie up too much capital |
| RSI | 35-55 | Not overbought — the stock is in a neutral or mildly oversold zone |
| Price vs EMA-50 | Near EMA-50 | The stock is near its medium-term average — not extended in either direction |
| Delta | ~0.25 | ~75% chance the put expires worthless (you keep the premium) |
| Annualized return | >= 15% | The premium, annualized, must yield at least 15%. See calculation below. |
| Capital required | Cash to buy 100 shares | Example: for a $4 stock, you need $400 in cash |

#### Annualized Return Calculation

```
premium_per_share     = premium received (e.g., $0.15)
capital_at_risk       = strike_price * 100 (e.g., $3.75 * 100 = $375)
holding_period_days   = DTE (e.g., 30)

raw_return            = premium_per_share * 100 / capital_at_risk
                      = $15 / $375 = 4.0%

annualized_return     = raw_return * (365 / holding_period_days)
                      = 4.0% * (365 / 30) = 48.7%
```

The 15% minimum ensures you're being adequately compensated for tying up capital.

#### Risk/Reward Profile

```
Max Profit = Premium Received
Max Loss   = (Strike Price - Premium Received) * 100
             (if stock drops to $0, you buy 100 shares at strike minus the premium offset)
Breakeven  = Strike Price - Premium Received
```

---

### 9. Covered Call

> **Direction:** Neutral to mildly bullish | **Risk:** Capped upside on shares you already own | **PDT:** N/A

#### Theory

If you already own 100 shares of a stock, you can **sell a call** against them. You collect premium, which is income on a position you already hold. The trade-off: if the stock rockets past the call's strike, your shares get "called away" (sold at the strike price) and you miss the upside beyond that.

This is the most conservative options strategy — you're simply monetizing your existing shares.

#### Entry Conditions

| Condition | Requirement | Rationale |
|-----------|------------|-----------|
| Existing position | Must own >= 100 shares of the stock | You need shares to "cover" the short call |
| RSI | 40-65 | Neutral to mildly bullish — not a stock that's about to crash or rocket |
| Price vs EMA-20 | Near EMA-20 | Stock is trading near its short-term average, suggesting sideways action |
| Delta | ~0.30 | ~70% chance the call expires worthless and you keep both the shares and the premium |
| Annualized return | >= 12% | Minimum yield for tying up shares |

#### Risk/Reward Profile

```
Max Profit = (Strike - Current Price) * 100 + Premium Received
             (stock rises to strike + you keep the premium)
Max Loss   = (Current Price - Premium Received) * 100
             (stock drops to $0, offset slightly by premium)
Breakeven  = Current Price - Premium Received
```

**Example:**

| Parameter | Value |
|-----------|-------|
| Stock price (you own shares) | $50.00 |
| Call strike sold | $53.00 (0.30 delta) |
| Premium received | $1.20 |
| Max profit | ($53 - $50 + $1.20) * 100 = $420 |
| Breakeven | $50 - $1.20 = $48.80 |

---

### 10. Covered Straddle

> **Direction:** Neutral | **Risk:** High — exposed on both sides | **PDT:** N/A

#### Theory

A covered straddle combines a **covered call** and a **cash-secured put** at the same strike (ATM). You sell an ATM call against your 100 shares AND sell an ATM put (secured by cash). This collects **double premium** but exposes you on both sides:

- If the stock drops significantly, you're forced to buy 100 MORE shares at the strike (from the short put) while your existing shares lose value.
- If the stock rises significantly, your shares are called away at the strike (from the short call), and you miss the upside.

This strategy is used in **low-volatility environments** where the stock is expected to stay flat.

#### Entry Conditions

| Condition | Requirement | Rationale |
|-----------|------------|-----------|
| Existing position | Must own >= 100 shares | Covers the short call |
| RSI | 40-60 | Neutral — stock isn't trending in either direction |
| BB Width | < 0.10 | **Tight Bollinger Bands** — volatility is compressed. This is the key filter. |
| Delta | ~0.50 (ATM) for both legs | ATM options have the highest theta (time decay), maximizing premium collected |

#### Why Bollinger Band Width < 0.10?

Recall that `bb_width = (upper_band - lower_band) / middle_band`. A value below 0.10 means the bands are within 10% of each other relative to the stock price. This indicates:

1. The stock has been trading in a very tight range.
2. Historical volatility is low.
3. The stock is less likely to make a big move — which is exactly what this strategy needs.

> **Warning:** Low volatility can precede a big move (a "volatility squeeze"). The tight bands make the double premium attractive, but the risk of an explosive breakout is real. This is the **highest-risk** income strategy in the bot.

#### Risk/Reward Profile

```
Max Profit = Premium from Call + Premium from Put
             (stock stays exactly at the strike through expiration)
Max Loss   = Substantial on both sides:
             Downside: stock drops, you own 200 shares (100 original + 100 from put assignment) at a loss
             Upside: shares called away, you sold at the strike minus premium, missing the rally
```

---

### 11. Momentum Options

> **Direction:** Bullish or Bearish | **Risk:** Premium paid (small, defined) | **PDT:** N/A

#### Theory

This is the **highest-risk, highest-reward** strategy. It buys cheap, short-dated, out-of-the-money options on momentum moves — essentially lottery tickets on breakouts or breakdowns.

The idea: when a stock breaks its 20-day high (or low) on heavy volume, a significant move is likely. An OTM option that costs $0.50 might become $3.00 if the stock moves 1.5 ATR in the right direction. The bot estimates this expected profit and only takes the trade if the return on investment is at least 25%.

Most of these trades will expire worthless. The strategy depends on occasional large winners making up for many small losses.

#### Entry Conditions

**For bullish setups (OTM calls):**

| Condition | Requirement |
|-----------|------------|
| Close > 20-day high | Breakout |
| Relative volume | > 1.5x |
| ADR | > 2% |
| Close > EMA-20 | Uptrend |

**For bearish setups (OTM puts):**

| Condition | Requirement |
|-----------|------------|
| Close < 20-day low | Breakdown |
| Relative volume | > 1.5x |
| ADR | > 2% |
| Close < EMA-20 | Downtrend |

**For both:**

| Parameter | Requirement |
|-----------|------------|
| Delta | 0.15-0.45 | Cheap, OTM options |
| DTE | 2-20 days | Short-dated for maximum leverage |
| Max cost | $100 per contract | Hard cap on risk |
| Estimated ROI | >= 25% | See calculation below |

#### ROI Estimation

The bot estimates what the option would be worth if the stock moves 1.5 ATR in the expected direction:

```
expected_stock_move  = 1.5 * ATR
estimated_new_price  = current_option_price + (delta * expected_stock_move)
estimated_profit     = estimated_new_price - current_option_price
estimated_roi        = estimated_profit / current_option_price

# Only take the trade if:
estimated_roi >= 0.25  (25%)
```

> **Python note:** This calculation uses the option's delta as a linear approximation of price sensitivity. In reality, delta changes as the stock moves (this is called **gamma**), so the estimate is conservative for OTM options (which have positive gamma — they accelerate into the money).

**Example:**

| Parameter | Value |
|-----------|-------|
| Stock price | $48.00 |
| ATR | $1.50 |
| Expected move | 1.5 * $1.50 = $2.25 |
| OTM call strike | $50.00 |
| Call delta | 0.25 |
| Call premium | $0.40 |
| Estimated new price | $0.40 + (0.25 * $2.25) = $0.96 |
| Estimated profit | $0.96 - $0.40 = $0.56 |
| Estimated ROI | $0.56 / $0.40 = **140%** |
| Verdict | **TAKE THE TRADE** (140% > 25%) |

---

## Signal Ranking: The Brain

The `SignalAggregator` is the central decision engine. It collects signals from all 11 strategies, ranks them, and decides which ones actually get executed. Think of it as a dispatcher that takes a queue of "I want to trade" requests and filters them through resource constraints.

### Step 1: Collect Signals

Every strategy runs its `check_entry()` method against every eligible stock. Each method returns either `None` (no signal) or a signal object containing:

- The stock ticker
- The strategy name
- The conviction score (0.0 to 1.0)
- The hold type (DAY or SWING)
- Entry price, stop-loss, take-profit levels
- Position sizing details

> **Python note:** `None` is Python's null value. The method signature looks like `def check_entry(self, data) -> Optional[Signal]`. The `Optional[Signal]` type hint means the return value is either a `Signal` object or `None`. The `->` syntax is a "return type annotation" — it doesn't enforce the type at runtime but documents the developer's intent.

### Step 2: Sort by Priority

Signals are sorted into two buckets and ordered:

```
1. SWING signals  — sorted by conviction (highest first)
2. DAY signals    — sorted by conviction (highest first)
```

**SWING signals are processed first** because they are "free" — they don't consume a PDT slot. The bot always prefers free resources over scarce ones.

### Step 3: Filter DAY Signals

DAY signals face an additional gate:

```python
if signal.hold_type == HoldType.DAY:
    if signal.conviction < 0.80:
        discard(signal)  # Not confident enough to spend a PDT slot
    if pdt_budget_remaining <= 0:
        discard(signal)  # No PDT slots left
```

This means a DAY signal needs **both** high conviction (>= 0.80) **and** an available PDT slot.

### Step 4: Build Execution Queue

The remaining signals are processed top-down. For each signal, the bot checks:

| Check | Question |
|-------|----------|
| **Cash available?** | Is there enough buying power to open this position? |
| **Position count?** | Would this exceed the max number of open positions? |
| **Risk approval?** | Does the portfolio-level risk manager approve? (e.g., not too much exposure to one sector) |

If all checks pass, the signal enters the execution queue. Otherwise, it is skipped and the bot moves to the next signal.

### Step 5: Apply Sentiment Modifiers

Before final execution, each signal's conviction is modified by external factors:

| Modifier | Effect |
|----------|--------|
| **Market regime** | If the broad market (e.g., SPY) is in a strong downtrend, bullish conviction is reduced. If in a strong uptrend, bearish conviction is reduced. |
| **News sentiment** | If recent news about the ticker is strongly negative, bullish conviction is reduced (and vice versa). |

```
final_conviction = base_conviction * market_regime_modifier * news_modifier
```

A signal that had 0.85 conviction might drop to 0.68 after modifiers, potentially falling below thresholds.

### Step 6: Execute

Surviving signals are sent to the broker API as bracket orders. The bot logs every decision (taken and skipped) for audit and backtesting purposes.

### Visual Summary

```
                    All 11 Strategies
                          |
                    [check_entry() on each stock]
                          |
                    Raw Signals (0 to many per stock)
                          |
                    Split: SWING vs DAY
                          |
               +----------+----------+
               |                     |
          SWING signals         DAY signals
          (sort by conviction)  (sort by conviction)
               |                     |
               |              [conviction >= 0.80?]
               |              [PDT budget > 0?]
               |                     |
               +----------+----------+
                          |
                    Merged queue (SWING first, then DAY)
                          |
                    [Cash check]
                    [Position count check]
                    [Risk approval]
                          |
                    Approved signals
                          |
                    [Apply sentiment modifiers]
                          |
                    Final execution queue
                          |
                    [Submit bracket orders to broker]
```

---

## Key Concepts Glossary

| Term | Definition |
|------|-----------|
| **RSI (Relative Strength Index)** | A momentum oscillator (0-100) measuring the speed and magnitude of recent price changes. Calculated as `100 - 100/(1 + avg_gain/avg_loss)` over 14 periods. Values below 30 indicate oversold conditions; above 70 indicates overbought. |
| **EMA (Exponential Moving Average)** | A moving average that places exponentially more weight on recent prices. EMA-20 means a 20-period EMA. Reacts faster to price changes than a Simple Moving Average (SMA). Formula: `EMA = close * k + EMA_prev * (1-k)` where `k = 2/(period+1)`. |
| **VWAP (Volume-Weighted Average Price)** | The average price weighted by volume, calculated as `cumulative(price * volume) / cumulative(volume)`. Resets daily. Represents the "fair value" for the day and is the benchmark institutional traders use. |
| **ATR (Average True Range)** | The 14-period average of the True Range, where True Range = `max(high-low, abs(high-prev_close), abs(low-prev_close))`. Measures volatility in dollar terms. A $50 stock with ATR of $2.00 typically moves $2 per bar. |
| **Bollinger Bands** | Three lines: a 20-period SMA (middle), and upper/lower bands at +/- 2 standard deviations. ~95% of price action falls within the bands. Used to identify overbought/oversold conditions and volatility. |
| **MACD (Moving Average Convergence Divergence)** | The difference between the 12-period EMA and 26-period EMA. A "signal line" (9-period EMA of the MACD) is used for crossover signals. MACD crossing above the signal line is bullish; below is bearish. |
| **Delta** | The rate of change of an option's price per $1 change in the underlying stock. A 0.50-delta call gains ~$0.50 when the stock rises $1. Also approximates the probability the option expires in-the-money. Calls have positive delta (0 to 1.0); puts have negative delta (-1.0 to 0). |
| **Theta** | The rate at which an option loses value per day due to time decay, all else being equal. A theta of -$0.05 means the option loses $5 per contract per day. Theta accelerates as expiration approaches and is highest for ATM options. |
| **IV (Implied Volatility)** | The market's forecast of the stock's future volatility, derived from current option prices using the Black-Scholes model. Expressed as an annualized percentage. IV of 50% means the market expects the stock to move ~50%/sqrt(252) = ~3.15% per day (one standard deviation). |
| **OTM (Out of the Money)** | An option with no intrinsic value. For calls: strike price > stock price. For puts: strike price < stock price. OTM options are cheaper but less likely to be profitable. |
| **ATM (At the Money)** | An option whose strike price equals (or is nearest to) the current stock price. ATM options have ~0.50 delta and the highest theta decay. |
| **ITM (In the Money)** | An option with intrinsic value. For calls: strike price < stock price. For puts: strike price > stock price. ITM options are more expensive but move more closely with the stock. |
| **DTE (Days to Expiration)** | The number of calendar days remaining until an option contract expires. After expiration, the contract is worthless (if OTM) or automatically exercised (if ITM). |
| **Spread Width** | The dollar difference between the two strike prices in a multi-leg options spread. A spread with strikes at $50 and $47.50 has a width of $2.50. Maximum loss in a credit spread = spread width - credit received. |
| **Conviction** | A bot-internal score from 0.0 to 1.0 representing confidence in a trade signal. Calculated differently by each strategy. Used for ranking, position sizing, and hold-type decisions. Higher conviction = larger position, higher priority. |
| **Bracket Order** | A set of three linked orders: entry, stop-loss, and take-profit. When the entry fills, the stop and target are automatically submitted. When either the stop or target fills, the other is canceled (OCO — One Cancels Other). |
| **PDT Rule (Pattern Day Trader)** | SEC regulation requiring $25,000 minimum equity for accounts making 4+ day trades in 5 business days. Below this threshold, you are limited to 3 day trades per rolling 5-day window. The bot tracks PDT budget and prioritizes SWING trades to conserve day-trade slots. |
| **ADR (Average Daily Range)** | The average of `(high - low) / close` over recent bars, expressed as a percentage. Measures how much a stock typically moves intraday. ADR > 2% means the stock is volatile enough to trade. |
| **Relative Volume** | Current volume divided by the 20-day average volume. Values above 1.5 indicate unusual activity; above 3.0 suggests institutional participation or a news catalyst. |
