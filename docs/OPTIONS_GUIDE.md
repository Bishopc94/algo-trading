# Options Trading Fundamentals for the AI Trade Bot

> **Audience**: Software engineers who understand code but may have zero knowledge of
> options trading. This guide starts from absolute basics and builds up to how
> every concept maps to real code in this repository.

---

## Table of Contents

1. [What Are Options?](#what-are-options)
2. [Key Terms](#key-terms)
3. [The Greeks](#the-greeks)
4. [How Options Work in This Bot](#how-options-work-in-this-bot)
5. [The 8 Options Strategies](#the-8-options-strategies)
6. [Payoff Diagrams](#payoff-diagrams)
7. [Risk Sizing for Options](#risk-sizing-for-options)
8. [Close-Before-Expiration](#close-before-expiration)
9. [OCC Symbol Format](#occ-symbol-format)
10. [Black-Scholes in the Backtester](#black-scholes-in-the-backtester)

---

## What Are Options?

An **option** is a contract that gives you the **right** (but not the obligation) to
buy or sell a stock at a specific price by a specific date. Think of it like a
reservation: you pay a small fee now to lock in a price, and later you can decide
whether to use it or walk away.

There are two types:

| Type | Right granted | You profit when... |
|------|--------------|-------------------|
| **Call** | Right to **BUY** the stock at the strike price | Stock goes **up** |
| **Put** | Right to **SELL** the stock at the strike price | Stock goes **down** |

### The 100-share multiplier

Every single options contract controls **100 shares** of the underlying stock.
This is the most important number in options trading. When you see an option priced
at $1.50, the actual cost is:

```
$1.50 per share x 100 shares = $150.00 per contract
```

### Buyers vs. Sellers

- **Buyer** (long): Pays the premium upfront. Has the right, not the obligation.
  Maximum loss is limited to the premium paid.
- **Seller** (short / "writer"): Receives the premium. Has the **obligation** to
  fulfill the contract if the buyer exercises. Can face much larger losses.

### A concrete example

Suppose AAPL stock trades at $150. You buy 1 AAPL $160 Call expiring in 30 days
for $2.00 per share.

- You pay: $2.00 x 100 = **$200** (this is the premium)
- If AAPL rises to $170 before expiration: your call is worth at least $10.00
  per share ($170 - $160 strike). You profit: ($10.00 - $2.00) x 100 = **$800**.
- If AAPL stays below $160: the call expires worthless. You lose the **$200** you
  paid. That is your maximum loss, no matter how far the stock drops.

---

## Key Terms

### Strike Price

The price at which the option contract allows you to buy (call) or sell (put) the
underlying stock. It is fixed when the contract is created.

```
Stock price: $25.00
Call strike: $27.50   <-- you can buy at $27.50 regardless of market price
Put strike:  $22.50   <-- you can sell at $22.50 regardless of market price
```

### Expiration Date

The date when the contract ceases to exist. After this date, the option is
worthless. In this bot, options expire on Fridays (standard weekly/monthly
cycles).

### Premium

The price you pay (or receive) for an option contract. It is quoted **per share**,
so always multiply by 100 to get the real dollar cost.

```
Premium per share:  $0.75
Cost per contract:  $0.75 x 100 = $75.00
Cost for 3 contracts: $75.00 x 3 = $225.00
```

### DTE (Days to Expiration)

How many calendar days remain until the contract expires. The bot uses DTE ranges
to filter contracts. For example, the `long_call` strategy requires 20-60 DTE
(`min_dte: 20`, `max_dte: 60` in `settings.yaml`).

Why it matters:
- **High DTE** (45-60 days): More time for your thesis to play out, but higher
  premium.
- **Low DTE** (5-10 days): Cheap, but time decay accelerates rapidly. The
  `momentum_options` strategy targets these (5-20 DTE).

### ITM / ATM / OTM

These describe the relationship between the stock's current price and the
option's strike price.

**For Calls:**

```
Stock price: $50.00

  ITM (In The Money):   Strike $45   ($50 > $45 -- would be profitable to exercise)
  ATM (At The Money):   Strike $50   ($50 = $50 -- right at the boundary)
  OTM (Out of The Money): Strike $55 ($50 < $55 -- not profitable to exercise)
```

**For Puts** (reversed logic):

```
Stock price: $50.00

  ITM (In The Money):   Strike $55   ($50 < $55 -- would be profitable to exercise)
  ATM (At The Money):   Strike $50   ($50 = $50 -- right at the boundary)
  OTM (Out of The Money): Strike $45 ($50 > $45 -- not profitable to exercise)
```

An OTM option has **no intrinsic value** -- its entire price is "time value"
(the chance it could become profitable before expiration). An ITM option has
**intrinsic value** equal to the difference between stock price and strike.

---

## The Greeks

"The Greeks" are measurements of how an option's price changes in response to
different market factors. They are named after Greek letters. The bot uses them
extensively for contract selection and risk management.

### Delta

**What it measures**: How much the option price changes for every $1 move in the
underlying stock price.

**Range**:
- Calls: 0.0 to +1.0
- Puts: -1.0 to 0.0

**Example**: A call with delta 0.50 will gain approximately $0.50 per share (or
$50 per contract) if the stock goes up $1.

**The probability shortcut**: Delta roughly approximates the market's implied
probability of the option expiring ITM. A 0.30 delta call has roughly a 30% chance
of being profitable at expiration.

**How the bot uses delta**: Delta is the primary contract selection criterion.
Each strategy has a `target_delta` in `settings.yaml`:

| Strategy | Target Delta | Meaning |
|----------|-------------|---------|
| `long_call` | 0.60 | ~60% chance ITM (moderately aggressive) |
| `long_put` | 0.55 | ~55% chance ITM |
| `credit_put_spread` | 0.30 | ~30% chance ITM (conservative, selling premium) |
| `cash_secured_put` | 0.20 | ~20% chance ITM (want to collect premium, not get assigned) |
| `covered_call` | 0.30 | ~30% chance ITM (sell upside above this) |
| `momentum_options` | 0.25-0.45 | Variable, OTM contracts for momentum breakouts |

The selection pipeline (covered below) calls `filter_by_delta()` to narrow the
chain and `select_by_delta()` to pick the contract closest to the target.

### Theta

**What it measures**: How much value the option loses each day due to the passage
of time. This is called "time decay."

**Range**: Always negative for option buyers (you lose money each day you hold).

**Example**: A call with theta of -0.05 will lose $0.05 per share ($5 per contract)
every day, all else being equal.

**The key insight**:

```
  You BUY options  -->  Theta works AGAINST you (you bleed money daily)
  You SELL options  -->  Theta works FOR you (you earn money daily)
```

This is why credit strategies (selling options to collect premium) are popular:
time is on your side. The bot's credit strategies (`credit_put_spread`,
`cash_secured_put`, `covered_call`) benefit from theta decay.

**Time decay is not linear** -- it accelerates as expiration approaches:

```
Theta decay over 60 days (conceptual):

Value
  |****
  |    ****
  |        ***
  |           ***
  |              **
  |                **
  |                  *
  |                   *
  |                    *
  +---------------------> Days to Expiration
 60                     0
```

This is why `momentum_options` (5-20 DTE) are risky for buyers: theta is at its
maximum, eating into your position daily.

### Implied Volatility (IV)

**What it measures**: The market's expectation of how much the stock will move in
the future. Higher IV means the market expects bigger price swings.

**Why it matters**: IV directly affects the option's price. High IV = expensive
options. Low IV = cheap options.

**Example**:
```
Same stock, same strike, same expiration:
  IV = 20%  -->  Call costs $1.50
  IV = 50%  -->  Call costs $3.80
```

**How the bot uses IV**: The `long_call` strategy has a `max_iv_percentile`
setting (default 1.00, effectively disabled). Rather than filtering on IV
percentile alone, the bot relies on the `max_contract_cost` cap ($75) to
limit overpaying for expensive options. The multi-indicator confluence
filters also help avoid entering during volatile, uncertain conditions.

Think of it like shopping: you do not want to buy options when they are "on sale"
at full price. High IV means you are paying a markup.

### Other Greeks (for reference)

The bot's `options_pricing.py` also computes:

- **Gamma**: Rate of change of delta. How much delta changes per $1 stock move.
  High gamma means delta is unstable -- common for near-expiration ATM options.
- **Vega**: Sensitivity to IV changes. A vega of 0.10 means the option gains
  $0.10 per share for every 1% increase in IV.

---

## How Options Work in This Bot

### Contract Selection Pipeline

When the bot evaluates an options trade, it follows a five-step pipeline defined
in `src/ai_trade/strategy/options/base.py`:

```
Step 1: get_options_chain(symbol)
        Fetches all active option contracts for a stock from the Alpaca API.
        Returns a list of dicts, each containing: symbol, strike_price,
        expiration_date, type (call/put), root_symbol.

                            |
                            v

Step 2: filter_contracts(chain, type="call", min_dte=20, max_dte=60)
        Filters by option type (call or put) and DTE range.
        Computes DTE from expiration_date vs. current time.
        Example: from 500 contracts, narrows to 80 calls with 20-60 DTE.

                            |
                            v

Step 3: enrich_greeks(contracts, snapshots)
        Looks up each contract in the live snapshots and attaches:
          _delta, _theta, _bid, _ask, _mid, _strike, _iv
        After this step, every contract has its Greeks and pricing.

                            |
                            v

Step 4: filter_by_delta(contracts, min_delta=0.50, max_delta=0.70)
        Narrows to contracts within the target delta range.
        Has automatic fallback: if no contracts match, relaxes to any
        contract with abs(delta) > fallback_min.

                            |
                            v

Step 5: select_by_delta(contracts, target_delta=0.60)
        Picks the single contract whose delta is closest to the target.
        Returns one dict -- the chosen contract.
```

**Concrete example**: The `long_call` strategy for AAPL at $150:

1. Fetch 600 AAPL option contracts
2. Filter to calls with 20-60 DTE: 120 contracts remain
3. Enrich with Greeks from snapshots (delta, theta, bid/ask, IV)
4. Filter to delta range 0.50-0.70: 25 contracts remain
5. Select the one closest to target delta 0.60: AAPL $145 Call, 35 DTE,
   delta 0.58, premium $7.20

### Debit vs. Credit Strategies

This is the most fundamental distinction in options strategies:

**Debit strategies** -- you PAY money to enter:
```
Your account:  $500.00
Buy 1 AAPL call for $2.00/share:  -$200.00
Your account after:  $300.00

Max loss = $200.00 (what you paid)
Max profit = theoretically unlimited (stock can keep going up)
```

**Credit strategies** -- you RECEIVE money to enter:
```
Your account:  $500.00
Sell 1 AAPL put for $1.50/share:  +$150.00
Your account after:  $650.00 (but $500 may be held as collateral)

Max profit = $150.00 (what you received)
Max loss = varies by strategy (can be very large for naked puts)
```

The bot's strategies:

| Strategy | Type | You... |
|----------|------|--------|
| Long Call | Debit | Pay premium, hope stock rises |
| Long Put | Debit | Pay premium, hope stock falls |
| Debit Call Spread | Debit | Pay net premium, capped upside |
| Credit Put Spread | Credit | Receive premium, defined risk |
| Cash Secured Put | Credit | Receive premium, may buy stock |
| Covered Call | Credit | Receive premium against shares |
| Covered Straddle | Credit | Receive premium on both sides |
| Momentum Options | Debit | Pay for cheap OTM bets |

---

## The 8 Options Strategies

### 1. Long Call (Single-leg, Debit)

**What it does**: Buy a call option. Profit if the stock goes up.

**When the bot uses it**: Bullish signal confirmed by multi-indicator confluence --
stacked EMAs (close > EMA-20 > EMA-50), MACD > 0 and rising, RSI 50-70, elevated
volume, and pre-breakout consolidation. IV not too high.

**Example**:
```
Stock: AAPL at $25.00
Buy: AAPL $27 Call, 35 DTE, for $0.75/share
Cost: $0.75 x 100 = $75.00

If AAPL goes to $30:
  Option worth: ($30 - $27) = $3.00/share = $300/contract
  Profit: $300 - $75 = $225 (300% return)

If AAPL stays at $25:
  Option expires worthless
  Loss: $75 (100% of premium -- this is your max loss)
```

**Bot settings** (`settings.yaml`):
```yaml
long_call:
  target_delta: 0.60        # Moderately ITM
  min_dte: 20               # At least 20 days
  max_dte: 60               # No more than 60 days
  max_contract_cost: 75.0   # $75 max per contract (for a $500 account)
  max_iv_percentile: 1.00   # Cost cap handles risk; IV check relaxed
```

### 2. Long Put (Single-leg, Debit)

**What it does**: Buy a put option. Profit if the stock goes down.

**When the bot uses it**: Bearish signal confirmed by multi-indicator confluence --
bearish EMA structure (EMA-20 < EMA-50), MACD < 0, RSI < 45, bearish candle
(close < open), and elevated volume.

**Example**:
```
Stock: XYZ at $30.00
Buy: XYZ $28 Put, 30 DTE, for $0.60/share
Cost: $0.60 x 100 = $60.00

If XYZ drops to $24:
  Option worth: ($28 - $24) = $4.00/share = $400/contract
  Profit: $400 - $60 = $340

If XYZ stays above $28:
  Option expires worthless
  Loss: $60
```

### 3. Cash Secured Put (Single-leg, Credit)

**What it does**: Sell a put option and set aside enough cash to buy 100 shares
if assigned. You collect premium and either keep it (if stock stays up) or buy
the stock at a discount.

**When the bot uses it**: Mildly bullish, wants to enter a stock position at a
lower price, or simply collect income.

**Example**:
```
Stock: XYZ at $4.50
Sell: XYZ $4.00 Put, 30 DTE, for $0.20/share
Credit received: $0.20 x 100 = $20.00
Cash required (collateral): $4.00 x 100 = $400.00

Scenario A -- Stock stays above $4.00:
  Put expires worthless, you keep the $20.
  Annualized return: ($20 / $400) x (365/30) = ~60%

Scenario B -- Stock drops to $3.00:
  You must buy 100 shares at $4.00 = $400.
  Your effective cost: $4.00 - $0.20 = $3.80/share.
  Current value: $3.00/share. Paper loss: $80.
  (But you own the shares and can hold for recovery.)
```

**Bot settings**:
```yaml
cash_secured_put:
  target_delta: 0.20             # Low probability of assignment (~80% PoP)
  max_stock_price: 3.00          # 100 shares x $3 = $300 max (fits $500 account)
  min_annualized_return: 0.15    # At least 15% annualized
```

### 4. Covered Call (Single-leg, Credit)

**What it does**: Sell a call option against 100 shares you already own. You
collect premium but cap your upside.

**When the bot uses it**: Owns 100 shares, neutral to mildly bullish, wants
income.

**Example**:
```
You own: 100 shares of XYZ at $4.00
Sell: XYZ $4.50 Call, 30 DTE, for $0.15/share
Credit received: $0.15 x 100 = $15.00

Scenario A -- Stock stays below $4.50:
  Call expires worthless, you keep the $15 + your shares.

Scenario B -- Stock rises to $5.00:
  Call is exercised, you sell shares at $4.50 (miss the move to $5.00).
  Total gain: ($4.50 - $4.00 + $0.15) x 100 = $65
  You gave up the extra $0.50/share ($50) of upside.
```

### 5. Credit Put Spread (Multi-leg, Credit)

**What it does**: Sell a put at a higher strike AND buy a put at a lower strike
(same expiration). You receive a net credit. Both profit and loss are capped.

**When the bot uses it**: Bullish, wants defined risk, wants to benefit from
theta decay.

**Example**:
```
Stock: XYZ at $30.00
Sell: XYZ $28 Put for $1.00   (short leg -- higher strike)
Buy:  XYZ $26 Put for $0.40   (long leg -- lower strike, protection)
Net credit: ($1.00 - $0.40) = $0.60/share = $60/contract

Spread width: $28 - $26 = $2.00
Max loss: ($2.00 - $0.60) x 100 = $140/contract
Max profit: $60/contract (the credit received)

Breakeven: $28.00 - $0.60 = $27.40
```

**Bot settings**:
```yaml
credit_put_spread:
  target_delta: 0.25           # Short put at ~25 delta (~75% PoP)
  max_spread_width: 1.50       # Max $1.50 between strikes (max loss $150)
  min_credit_pct: 0.30         # Credit must be >= 30% of spread width
  max_risk: 100.0              # Hard cap: max $100 loss per trade
```

### 6. Debit Call Spread (Multi-leg, Debit)

**What it does**: Buy a call at a lower strike AND sell a call at a higher
strike (same expiration). You pay a net debit. Both profit and loss are capped.

**When the bot uses it**: Bullish but wants to reduce cost by selling upside.

**Example**:
```
Stock: XYZ at $30.00
Buy:  XYZ $29 Call for $2.00   (long leg -- lower strike)
Sell: XYZ $31 Call for $0.80   (short leg -- higher strike)
Net debit: ($2.00 - $0.80) = $1.20/share = $120/contract

Spread width: $31 - $29 = $2.00
Max profit: ($2.00 - $1.20) x 100 = $80/contract
Max loss: $120/contract (the debit paid)

Breakeven: $29.00 + $1.20 = $30.20
```

### 7. Covered Straddle (Multi-leg, Credit)

**What it does**: While holding 100 shares, sell both a call AND a put (same
strike, same expiration). Collects premium from both sides but takes on
significant risk.

**When the bot uses it**: Low-volatility environment (measured by Bollinger Band
width), stock trading in a range.

**Risk**: If the stock drops hard, you lose on the shares AND are obligated to
buy more via the put. High premium, high risk.

### 8. Momentum Options (Single-leg, Debit)

**What it does**: Buy short-dated OTM options for quick percentage gains on
momentum breakouts.

**When the bot uses it**: Strong momentum signal, volume spike, breakout above
resistance.

**Example**:
```
Stock: XYZ at $10.00, breaking out with 2x normal volume
Buy: XYZ $11 Call, 7 DTE, for $0.15/share
Cost: $15.00

If XYZ surges to $12.00:
  Option worth: ($12 - $11) = $1.00/share = $100/contract
  Profit: $100 - $15 = $85 (567% return)

If XYZ stalls:
  Option expires worthless. Loss: $15.
```

This is the highest risk/reward strategy. Requires full directional confluence --
stacked EMAs aligned with direction, MACD confirming, volume > 2x average, and
pre-breakout consolidation. Delta range 0.25-0.45, short DTE (5-20 days).

---

## Payoff Diagrams

These ASCII diagrams show profit/loss at expiration for different stock prices.

### Long Call

```
Buy: XYZ $50 Call for $2.00 ($200 per contract)

Profit/Loss
     |
 +$300 |                                          /
 +$200 |                                        /
 +$100 |                                      /
     $0 |------------------------------+-----/------  <-- breakeven at $52
 -$100 |                              |   /
 -$200 |..............................+./...........  <-- max loss = $200
 -$300 |                              |
     +--+-----+-----+-----+-----+-----+-----+-----+
       $40   $42   $44   $46   $48   $50   $52   $54   $56
                                      strike
                                    ($50)

  - Below $50: Option expires worthless. Loss = $200 (premium paid).
  - At $52: Breakeven. The $2 gain exactly offsets the $2 premium.
  - Above $52: Profit grows $1 for every $1 the stock rises.
  - Max loss: $200 (fixed)
  - Max profit: Unlimited
```

### Credit Put Spread

```
Sell: XYZ $50 Put, Buy: XYZ $47 Put.  Net credit = $1.00 ($100/contract)

Profit/Loss
     |
 +$100 |..............................................  <-- max profit = $100
     $0 |--+-----+-----+-----+-----+--------+--------
 -$100 |              |                 |
 -$200 |............../.................|..........  <-- max loss = $200
 -$300 |             /
     +--+-----+-----+-----+-----+-----+-----+-----+
       $44   $45   $46   $47   $48   $49   $50   $51   $52
                    long       breakeven    short
                    strike     ($49)        strike
                    ($47)                   ($50)

  - Above $50: Both puts expire worthless. You keep the $100 credit. Max profit.
  - Between $49-$50: Partial loss on the short put, but still profitable.
  - At $49: Breakeven ($50 - $1.00 credit).
  - Between $47-$49: Increasing loss. Short put losing value, long put not yet helping.
  - Below $47: Both puts are ITM. Losses capped. Max loss = ($3 spread - $1 credit) x 100 = $200.
```

### Covered Call

```
Own 100 shares at $50.  Sell: XYZ $55 Call for $1.50 ($150/contract)

Profit/Loss (total position: shares + short call)
     |
 +$650 |                              ..................  <-- max profit = $650
 +$500 |                            / :
 +$300 |                          /   :
 +$150 |                        /     :    ($5 gain on shares + $1.50 premium)
     $0 |---------+------------/------:---
 -$150 |        /            :
 -$500 |      /              :
 -$850 |    /                :
     +--+-----+-----+-----+-----+-----+-----+-----+
       $40   $42   $44   $46   $48   $50   $52   $55   $58
                                    bought          strike
                                    shares          ($55)
                                    ($50)

  - Below $50: You lose on the shares but keep the $150 premium (cushion).
    At $48.50: breakeven ($1.50 premium offsets $1.50 share decline).
  - Between $50-$55: Shares gain value AND you keep the $150 premium. Best zone.
  - Above $55: Shares get called away at $55. You keep:
    ($55 - $50) x 100 + $150 = $650. This is your max profit.
  - You give up all gains above $55.
```

---

## Risk Sizing for Options

The bot enforces strict risk limits for options, configured in `settings.yaml`
under the `options:` section:

### Per-Trade Risk

```yaml
max_single_options_risk_pct: 0.12   # 12% of equity per trade
```

With a $500 account, this means max risk on any single options trade is $60.

### Portfolio-Level Limits

```yaml
max_options_capital_pct: 0.50   # Max 50% of portfolio in options
max_options_positions: 3        # Max 3 concurrent options positions
```

With a $500 account: at most $250 can be tied up in options, across at most 3
positions.

### Multi-Contract Sizing

The backtester (`src/ai_trade/backtest/engine.py`) sizes positions using this
formula:

```python
max_risk_budget = current_equity * max_risk_per_trade_pct
num_contracts = max(1, int(max_risk_budget / signal.max_loss))
```

Then capped at 10 contracts maximum, and further reduced if the total cash
required exceeds available cash or would breach the portfolio-level options
allocation.

**Example**:
```
Equity: $500
max_risk_per_trade_pct: 0.02 (2%)
max_risk_budget: $500 x 0.02 = $10

Signal max_loss: $75 (one long call contract)
num_contracts = max(1, int($10 / $75)) = max(1, 0) = 1

Signal max_loss: $3 (a cheap OTM call)
num_contracts = max(1, int($10 / $3)) = max(1, 3) = 3
```

---

## Close-Before-Expiration

The bot **never holds options through expiration**. This avoids assignment risk
(being forced to buy/sell 100 shares when you may not have the cash) and
after-hours exercise risk.

The `close_expiring_positions()` method in
`src/ai_trade/execution/options_order_manager.py` runs daily (typically around
3:00 PM) and does the following:

1. Fetch all open options positions via `get_options_positions()`
2. For each position, parse the expiration date from the OCC symbol
3. If the option expires within 1 day (the `days_until_expiration` parameter),
   close the position immediately
4. Log which positions were closed

The OCC symbol parsing extracts the date portion:

```
Symbol:    AAPL  250620C00200000
Stripped:  AAPL250620C00200000
                 ^^^^^^
                 250620 = June 20, 2025

The code grabs characters [-15:-9] from the stripped symbol to get "250620",
then parses it as YYMMDD.
```

---

## OCC Symbol Format

The OCC (Options Clearing Corporation) symbol is the standard identifier for
every options contract. Understanding this format is essential for debugging.

### Format

```
[Root][Date][Type][Strike]

Root:    1-6 characters, left-padded with spaces to 6 chars
Date:    YYMMDD (2-digit year, month, day)
Type:    C (call) or P (put)
Strike:  8 digits = strike price x 1000 (no decimal point)
```

### Examples

```
AAPL  250620C00200000
|   | |    |||      |
|   | |    |||      +-- Strike: 00200000 / 1000 = $200.00
|   | |    ||+--------- Type: C = Call
|   | |    |+---------- Date: 250620 = 2025-06-20
|   | +----+----------- Root: "AAPL  " (padded to 6 chars)
+---+

More examples:

TSLA  251219P00150000   = TSLA $150 Put  expiring Dec 19, 2025
SPY   250321C00450000   = SPY  $450 Call expiring Mar 21, 2025
F     250117C00012500   = F    $12.50 Call expiring Jan 17, 2025
NVDA  250502P00095000   = NVDA $95  Put  expiring May 2, 2025
```

### How the bot generates OCC symbols

In `src/ai_trade/backtest/options_pricing.py`:

```python
def _occ_symbol(underlying, expiration, opt_type, strike):
    sym = underlying.ljust(6)[:6]          # Pad/truncate to 6 chars
    date_str = expiration.strftime("%y%m%d")  # YYMMDD
    cp = "C" if opt_type == "call" else "P"
    strike_int = int(round(strike * 1000))
    strike_str = f"{strike_int:08d}"       # 8 digits, zero-padded
    return f"{sym}{date_str}{cp}{strike_str}"
```

---

## Black-Scholes in the Backtester

The Alpaca API provides live options data, but it does **not** provide historical
options data. So when backtesting, the bot needs to generate **synthetic** options
chains from historical stock prices. It does this using the Black-Scholes model.

### What is Black-Scholes?

Black-Scholes is a mathematical formula (from 1973) that calculates the
theoretical price of a European-style option. It takes five inputs and produces
a price.

### The Five Inputs

```
S     = Current stock price            (from historical bar data)
K     = Strike price                   (generated around the stock price)
T     = Time to expiration in years    (DTE / 365)
r     = Risk-free interest rate        (default 4.5% in the bot)
sigma = Volatility (annualized)        (computed from recent price history)
```

### The Formula (for a call)

```
Call Price = S * N(d1) - K * e^(-rT) * N(d2)

where:
  d1 = [ln(S/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))
  d2 = d1 - sigma * sqrt(T)
  N(x) = standard normal cumulative distribution function
```

You don't need to memorize this. The implementation lives in
`src/ai_trade/backtest/options_pricing.py` in the `bs_call_price()` and
`bs_put_price()` functions.

### How the Backtester Uses It

The `generate_synthetic_chain()` function creates a full options chain:

1. **Determine volatility**: Compute 30-day historical volatility from recent
   stock prices using log-returns (annualized by multiplying by sqrt(252)):
   ```python
   log_returns = log(price_today / price_yesterday)
   volatility = std(recent_30_log_returns) * sqrt(252)
   ```

2. **Generate expiration dates**: Short-dated (3, 5, 7, 10 days out) for
   momentum strategies, plus biweekly expirations out to ~60 days. All snapped
   to Fridays.

3. **Generate strike prices**: A range of strikes around the current stock price
   (+/- 15%). The step size depends on stock price:
   - Under $5: $0.50 steps
   - $5-$25: $1.00 steps
   - $25-$100: $2.50 steps
   - $100-$500: $5.00 steps
   - Over $500: $10.00 steps

4. **Price each contract**: For every combination of (expiration, strike,
   call/put), compute the Black-Scholes price and all Greeks.

5. **Simulate bid/ask spreads**: Real markets have a spread between the bid
   (what buyers will pay) and the ask (what sellers want). The backtester
   generates realistic spreads -- tighter for ATM options (2%), wider for OTM
   options (up to 7%):
   ```python
   moneyness = abs(stock_price - strike) / stock_price
   spread_pct = 0.02 + moneyness * 0.05
   bid = price - half_spread
   ask = price + half_spread
   ```

6. **Return in live API format**: The synthetic chain_data and snapshots use the
   exact same dict structure as the live Alpaca API, so all strategy code works
   without modification.

### Computed Greeks

For each synthetic contract, the backtester computes:

| Greek | Call Formula | Put Adjustment |
|-------|-------------|----------------|
| Delta | N(d1) | N(d1) - 1 |
| Gamma | N'(d1) / (S * sigma * sqrt(T)) | Same as call |
| Theta | -(S * N'(d1) * sigma) / (2*sqrt(T)) - r*K*e^(-rT)*N(d2), divided by 365 | Similar, with sign changes |
| Vega | S * N'(d1) * sqrt(T) / 100 | Same as call |

Where N'(x) is the standard normal probability density function.

### Example: Synthetic Chain Generation

```
Input:
  Stock: XYZ at $10.00
  Date: 2025-03-15
  Historical volatility: 35%
  Risk-free rate: 4.5%

Output (partial):

  XYZ   250321C00009000  $9 Call, 6 DTE
    Price: $1.08, Bid: $1.06, Ask: $1.10
    Delta: 0.82, Theta: -0.02, IV: 0.35

  XYZ   250321C00010000  $10 Call, 6 DTE (ATM)
    Price: $0.22, Bid: $0.21, Ask: $0.23
    Delta: 0.52, Theta: -0.03, IV: 0.35

  XYZ   250321C00011000  $11 Call, 6 DTE
    Price: $0.02, Bid: $0.01, Ask: $0.03
    Delta: 0.12, Theta: -0.01, IV: 0.35

  XYZ   250404C00010000  $10 Call, 20 DTE (ATM)
    Price: $0.45, Bid: $0.44, Ask: $0.46
    Delta: 0.54, Theta: -0.02, IV: 0.35

  ... plus puts, and all other strike/expiration combos
```

Notice how the 6 DTE ATM call ($0.22) is much cheaper than the 20 DTE ATM call
($0.45). More time = more value. Also notice how the deep ITM call ($9 strike,
delta 0.82) is much more expensive ($1.08) but moves nearly dollar-for-dollar
with the stock.

---

## Quick Reference

| Concept | One-line summary |
|---------|-----------------|
| Call | Right to buy at strike price |
| Put | Right to sell at strike price |
| Premium | Price of the option contract (x100 for real cost) |
| Strike | The locked-in buy/sell price |
| DTE | Days until the option expires |
| ITM | Option has intrinsic value right now |
| OTM | Option has no intrinsic value (only time value) |
| Delta | Price sensitivity to stock movement; ~probability of expiring ITM |
| Theta | Daily time decay (enemy of buyers, friend of sellers) |
| IV | How expensive options are relative to normal |
| Debit | You pay to enter |
| Credit | You get paid to enter |
| OCC Symbol | Standard 21-char option identifier |
| Black-Scholes | Formula to price options from 5 inputs |

---

## Key Source Files

| File | What it does |
|------|-------------|
| `src/ai_trade/strategy/options/base.py` | Shared pipeline: filter_contracts, enrich_greeks, filter_by_delta, select_by_delta |
| `src/ai_trade/strategy/options/long_call.py` | Long call strategy logic |
| `src/ai_trade/strategy/options/credit_put_spread.py` | Credit put spread strategy logic |
| `src/ai_trade/strategy/options/momentum_options.py` | Short-dated momentum options |
| `src/ai_trade/execution/options_order_manager.py` | Order submission, position closing, expiration management |
| `src/ai_trade/backtest/options_pricing.py` | Black-Scholes pricing, synthetic chain generation |
| `config/settings.yaml` | All options strategy parameters |
