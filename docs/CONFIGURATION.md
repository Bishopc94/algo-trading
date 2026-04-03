# Configuration Reference

This document covers every configurable parameter in the AI Trade bot. The bot reads its settings from two sources:

1. **`config/settings.yaml`** -- all trading parameters, strategy tuning, risk limits, and scheduling.
2. **`.env`** file -- API secrets (never committed to version control).

Both are loaded at startup by `src/ai_trade/config.py` and merged into a single configuration object.

---

## Table of Contents

- [How Configuration Loading Works](#how-configuration-loading-works)
- [The `.env` File (Secrets)](#the-env-file-secrets)
- [Overriding Settings](#overriding-settings)
- [Parameter Reference](#parameter-reference)
  - [alpaca](#alpaca)
  - [account](#account)
  - [pdt](#pdt)
  - [scanner](#scanner)
  - [strategies.mean_reversion](#strategiesmean_reversion)
  - [strategies.momentum](#strategiesmomentum)
  - [strategies.vwap](#strategiesvwap)
  - [strategies.credit_put_spread](#strategiescredit_put_spread)
  - [strategies.debit_call_spread](#strategiesdebit_call_spread)
  - [strategies.long_call](#strategieslong_call)
  - [strategies.long_put](#strategieslong_put)
  - [strategies.cash_secured_put](#strategiescash_secured_put)
  - [strategies.momentum_options](#strategiesmomentum_options)
  - [strategies.covered_call](#strategiescovered_call)
  - [strategies.covered_straddle](#strategiescovered_straddle)
  - [sentiment](#sentiment)
  - [options](#options)
  - [risk](#risk)
  - [schedule](#schedule)
- [Tuning Guide](#tuning-guide)

---

## How Configuration Loading Works

The config loader lives in `src/ai_trade/config.py`. Here is what happens at startup:

1. **Load `.env`**: The loader looks for `config/.env` first, then falls back to `.env` in the project root. It uses the `python-dotenv` library to read key-value pairs from that file and inject them into the process's environment variables.

2. **Parse YAML**: The loader reads `config/settings.yaml` (or whatever path you provide) using `yaml.safe_load()`, which turns the YAML text into a nested tree of Python dictionaries.

3. **Convert to SimpleNamespace**: The nested dictionary tree is recursively converted to Python `SimpleNamespace` objects. This is a convenience feature worth understanding:

   In Python, a dictionary requires bracket-and-quote syntax to access values:
   ```
   config["strategies"]["momentum"]["enabled"]    # dictionary style
   ```
   A `SimpleNamespace` lets you use **dot notation** instead, which reads more naturally:
   ```
   config.strategies.momentum.enabled              # SimpleNamespace style
   ```
   Under the hood, `SimpleNamespace` is a lightweight Python built-in (from the `types` module) that turns dictionary keys into object attributes. The config loader calls `SimpleNamespace(**some_dict)` which unpacks every key-value pair into an attribute. Because the YAML has nested sections (like `strategies` containing `momentum`), the conversion is done **recursively** -- inner dictionaries become their own `SimpleNamespace` objects, so the entire tree is dot-accessible.

4. **Inject secrets**: After the YAML is loaded, the two Alpaca API keys are read from environment variables and attached to `cfg.alpaca.api_key` and `cfg.alpaca.secret_key`. If either is missing, the bot raises an error immediately rather than failing later during a trade.

The returned object is a single `SimpleNamespace` tree. Every component in the bot receives only the sub-tree it needs -- for example, the scanner gets `cfg.scanner`, the PDT manager gets `cfg.pdt`, and each strategy gets its own section like `cfg.strategies.momentum`.

---

## The `.env` File (Secrets)

Create a file at `config/.env` (or `.env` in the project root). The format is one `KEY=VALUE` pair per line, no quotes required:

```
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

| Variable | Required | Description |
|---|---|---|
| `ALPACA_API_KEY` | Yes | Your Alpaca API key. Obtain from the [Alpaca dashboard](https://app.alpaca.markets/). Paper and live accounts have different keys. |
| `ALPACA_SECRET_KEY` | Yes | The corresponding secret key for the API key above. |

**Important**: Never put these values in `settings.yaml`. The loader enforces this separation -- YAML holds tuning parameters, `.env` holds secrets. If either key is missing at startup, the bot will exit with an `EnvironmentError` and a message telling you to fill in your keys.

---

## Overriding Settings

There are two ways to change configuration:

### 1. Edit `config/settings.yaml` directly

This is the primary method. Change values, save the file, and restart the bot. The file is plain YAML -- indentation matters (use spaces, not tabs).

### 2. Use the `--config` flag

Pass an alternate YAML file at startup:

```
python -m ai_trade.main --config path/to/my_settings.yaml
```

This completely replaces the default `config/settings.yaml`. The alternate file must contain all sections -- there is no partial-merge behavior. This is useful for keeping separate profiles (e.g., `settings_aggressive.yaml`, `settings_conservative.yaml`).

There is also a `--dry-run` flag that logs all signals and would-be trades without actually submitting orders to Alpaca. This is useful for validating configuration changes before going live.

---

## Parameter Reference

### `alpaca`

Broker connection settings.

| Parameter | Default | Type | Description |
|---|---|---|---|
| `paper` | `true` | boolean | When `true`, the bot connects to Alpaca's paper-trading environment (simulated money). Set to `false` for real-money trading. **Start with paper trading.** Switching to `false` requires that your `.env` keys correspond to a live-funded Alpaca account. |

**Trade-offs**: There is no reason to set `paper: false` until you have validated the bot's behavior over multiple weeks of paper trading. Paper mode uses the same API endpoints with the same latency characteristics, so it is a faithful simulation.

---

### `account`

Portfolio-level sizing and loss limits. These parameters define the "envelope" within which all strategies operate.

| Parameter | Default | Type | Valid Range | Description |
|---|---|---|---|---|
| `starting_capital` | `500.0` | float | > 0 | The dollar amount you funded the account with. Used as the baseline for percentage calculations. This does not need to match your actual account equity exactly -- it is a reference point for the position sizer. If your account grows to $800, the bot uses live equity from Alpaca for real-time sizing, but this value anchors initial calculations. |
| `max_position_pct` | `0.30` | float | 0.01 -- 1.0 | Maximum fraction of portfolio equity that can go into a single position. At 0.30 (30%), a $500 account can put at most $150 into one stock. **Lower = more diversified but smaller positions. Higher = concentrated bets.** For accounts under $1,000, going below 0.20 may produce positions too small to be practical (e.g., 1-2 shares of a $10 stock). |
| `max_risk_per_trade_pct` | `0.02` | float | 0.005 -- 0.10 | Maximum fraction of equity you are willing to lose on a single trade if the stop loss is hit. At 0.02 (2%), a $500 account risks at most $10 per trade. This is the classic "2% rule" from position sizing theory. The position sizer uses this along with the stop-loss distance to calculate share count. **Lower = smaller positions, slower drawdowns. Higher = larger positions, faster drawdowns.** |
| `daily_loss_limit_pct` | `0.05` | float | 0.01 -- 0.20 | If portfolio equity drops more than this percentage from the day's starting equity, the bot stops opening new trades for the rest of the day. At 0.05 (5%), a $500 account halts after losing $25 intraday. This is a circuit breaker. **Tighter limits protect capital but may cause you to miss recovery opportunities.** |
| `max_open_positions` | `4` | integer | 1 -- 20 | Hard cap on simultaneous open stock positions. With a $500 account and 30% max position size, 4 positions could theoretically use 120% of the account (impossible without margin), so in practice this acts as a diversification floor. **Fewer positions = simpler to monitor. More positions = better diversification but thinner per-position sizing.** |

---

### `pdt`

Pattern Day Trader rule management. US brokers flag accounts under $25,000 that make 4 or more day trades (buy and sell the same stock in the same day) within a rolling 5-business-day window. Getting flagged restricts your account for 90 days.

| Parameter | Default | Type | Valid Range | Description |
|---|---|---|---|---|
| `max_day_trades` | `3` | integer | 0 -- 3 | Maximum day trades the bot is allowed to make in a rolling 5-day window. The legal limit is 3 (the 4th triggers PDT). Set to 0 to disable day trading entirely, forcing all trades to be held overnight (swing/position). |
| `day_trade_reserve` | `1` | integer | 0 -- 3 | Number of day trades to keep "in reserve" and not use. With `max_day_trades: 3` and `day_trade_reserve: 1`, the bot will use at most 2 day trades, saving 1 for an emergency exit if a position moves sharply against you intraday. **Set to 0 if you want to use all 3 day trades, but you lose the safety net.** |
| `min_conviction_for_day_trade` | `0.80` | float | 0.0 -- 1.0 | Minimum signal conviction score (0 to 1) required for the bot to spend a day trade. Day trades are a scarce resource for sub-$25K accounts, so this threshold ensures they are only used for high-confidence setups. At 0.80, a signal must be very strong before the bot will enter knowing it needs to exit the same day. **Lower = uses day trades more freely. Higher = more selective, may miss some winners but preserves day trade budget.** |

---

### `scanner`

Pre-market stock screening filters. The scanner runs before market open to build a list of candidate symbols that strategies will evaluate.

| Parameter | Default | Type | Valid Range | Description |
|---|---|---|---|---|
| `min_price` | `2.00` | float | >= 0.01 | Minimum stock price in dollars. Stocks below $2 are often penny stocks with unreliable price action, wide spreads, and manipulation risk. **Lower = more candidates but riskier. Below $1 you enter OTC territory.** |
| `max_price` | `50.00` | float | > min_price | Maximum stock price. With a $500 account and 30% max position, the most you can spend on one position is ~$150, meaning you can buy at most 3 shares of a $50 stock. **Higher max_price finds better-quality stocks but yields tiny share counts. Lower max_price focuses on cheaper stocks where you can build more meaningful positions.** |
| `min_avg_volume` | `500000` | integer | >= 10000 | Minimum average daily volume (shares). Ensures adequate liquidity -- you want to enter and exit without moving the price. 500K shares/day is a moderate threshold. **Lower = more candidates but potential slippage issues. Higher = only highly liquid names.** |
| `min_relative_volume` | `1.5` | float | >= 1.0 | Minimum ratio of today's volume to average volume. A value of 1.5 means the stock is trading at 150% of its typical volume. This is a proxy for "something is happening." **Higher = fewer but more active candidates. A value of 1.0 disables this filter.** |
| `min_gap_pct` | `2.0` | float | >= 0.0 | Minimum pre-market gap percentage (how far the stock opened from yesterday's close). A 2% gap means the stock opened at least 2% above or below the prior close. Gap stocks tend to have momentum and catalysts. **Higher = fewer, more volatile candidates. 0.0 disables the gap filter.** |
| `max_candidates` | `20` | integer | 1 -- 100 | Maximum number of stocks the scanner returns. Candidates are ranked by relative volume and gap size, so the top 20 are the most active. **More candidates = more evaluation work for strategies (slightly slower). Fewer = may miss opportunities.** |

---

### `strategies.mean_reversion`

Buys stocks that have pulled back from recent highs, betting on a return to the mean. Uses the RSI (Relative Strength Index) indicator to identify oversold conditions and ATR (Average True Range) for stop/target placement.

| Parameter | Default | Type | Valid Range | Description |
|---|---|---|---|---|
| `enabled` | `true` | boolean | true/false | Whether this strategy participates in signal generation. Set to `false` to disable without removing the config. |
| `rsi_period` | `14` | integer | 2 -- 50 | Number of bars used to calculate RSI. The industry standard is 14. Shorter periods (e.g., 7) are more sensitive and trigger more signals. Longer periods are smoother and trigger fewer signals. |
| `rsi_oversold` | `40` | integer | 10 -- 50 | RSI value below which the stock is considered oversold (a buy signal). The textbook value is 30, but 40 is more lenient, catching pullbacks earlier. **Lower = stricter, fewer but higher-conviction entries. Higher = more entries, some may be premature.** |
| `rsi_exit` | `60` | integer | 40 -- 90 | RSI value at which to take profit (exit signal). When RSI climbs back to 60, the stock has reverted toward its mean. **Higher = lets winners run longer. Lower = exits sooner, locking in smaller gains.** |
| `atr_stop_multiplier` | `1.5` | float | 0.5 -- 5.0 | Stop loss is placed this many ATRs below the entry price. ATR measures typical daily price movement, so 1.5x ATR means the stop is 1.5 "normal days" of movement away. **Lower = tighter stops, more frequent stop-outs. Higher = wider stops, fewer stop-outs but larger losses when hit.** |
| `atr_tp_multiplier` | `3.0` | float | 1.0 -- 10.0 | Take-profit target is placed this many ATRs above entry. At 3.0 with a stop multiplier of 1.5, the reward-to-risk ratio is 2:1, which is a solid baseline. **Higher = larger potential wins but fewer trades reach the target. Lower = more frequent exits but smaller wins.** |
| `hold_type` | `"swing"` | string | `"day"`, `"swing"`, `"adaptive"` | How long the strategy intends to hold. `"swing"` means multi-day (avoids using a day trade). `"day"` means intraday only (uses a day trade). `"adaptive"` lets the bot decide based on conditions. Mean reversion is set to swing because pullback recovery typically takes 2-5 days. |
| `lookback_days` | `30` | integer | 10 -- 200 | Number of historical trading days used to assess the "mean" the stock should revert to. 30 days captures the recent trend. **Shorter = reacts to recent action, may miss larger context. Longer = incorporates more history but may be stale.** |

---

### `strategies.momentum`

Buys stocks breaking out above recent resistance on above-average volume. Rides the trend.

| Parameter | Default | Type | Valid Range | Description |
|---|---|---|---|---|
| `enabled` | `true` | boolean | true/false | Enable or disable this strategy. |
| `volume_spike_multiplier` | `1.5` | float | 1.0 -- 5.0 | Minimum ratio of current volume to average volume for a valid breakout. A breakout on low volume is often a false signal. 1.5 means volume must be at least 50% above average. **Higher = fewer but more reliable breakout signals. 1.0 disables volume confirmation.** |
| `breakout_lookback` | `20` | integer | 5 -- 100 | Number of bars to look back for the resistance level. The stock's highest high over this window becomes the breakout threshold. 20 days is roughly one trading month. **Shorter = catches minor breakouts. Longer = only catches breakouts above significant, long-standing resistance.** |
| `atr_stop_multiplier` | `1.5` | float | 0.5 -- 5.0 | Stop loss placement in ATR multiples below entry. Same mechanics as mean reversion. |
| `atr_tp_multiplier` | `3.0` | float | 1.0 -- 10.0 | Take-profit target in ATR multiples above entry. Same mechanics as mean reversion. |
| `hold_type` | `"adaptive"` | string | `"day"`, `"swing"`, `"adaptive"` | Momentum plays can resolve in hours or days. `"adaptive"` lets the bot decide based on time of day and signal strength. If a breakout happens at 3:30 PM, it may close same-day; if at 10:00 AM with a strong trend, it may hold overnight. |
| `min_adr_pct` | `2.0` | float | 0.5 -- 10.0 | Minimum Average Daily Range as a percentage of price. ADR measures how much the stock typically moves in a day. At 2.0%, a $20 stock must have an average daily range of at least $0.40. This filters out stocks that do not move enough to be worth trading. **Lower = more candidates but some may barely move. Higher = only volatile names.** |

---

### `strategies.vwap`

Intraday strategy that trades deviations from the Volume-Weighted Average Price. VWAP is a benchmark price that institutional traders use; stocks tend to gravitate toward it.

| Parameter | Default | Type | Valid Range | Description |
|---|---|---|---|---|
| `enabled` | `true` | boolean | true/false | Enable or disable this strategy. |
| `hold_type` | `"day"` | string | `"day"` | VWAP is inherently intraday -- the indicator resets at market open. This strategy always closes by end of day. Uses a day trade. |
| `entry_deviation_pct` | `0.5` | float | 0.1 -- 3.0 | How far the price must deviate from VWAP (as a percentage) to trigger an entry. At 0.5%, if VWAP is $10.00, the stock must drop to $9.95 or below for a long entry. **Larger deviation = fewer entries but the stock is further from "fair value," giving more room to revert. Smaller = more frequent entries, some may not revert enough to profit.** |
| `exit_deviation_pct` | `0.3` | float | 0.0 -- 2.0 | How close to VWAP (as a percentage) before exiting. At 0.3%, the position closes when price is within 0.3% of VWAP. Setting to 0.0 means exit exactly at VWAP. **Smaller = holds closer to VWAP, maximizes the reversion profit. Larger = exits early, may leave money on the table but reduces risk of reversal.** |

---

### `strategies.credit_put_spread`

Sells a put spread (sell a higher-strike put, buy a lower-strike put) to collect premium. Profits when the stock stays above the short strike. This is a defined-risk, bullish/neutral options strategy.

| Parameter | Default | Type | Valid Range | Description |
|---|---|---|---|---|
| `enabled` | `true` | boolean | true/false | Enable or disable this strategy. |
| `target_delta` | `0.30` | float | 0.05 -- 0.50 | Delta of the short (sold) put. Delta approximates the probability that the option expires in-the-money. At 0.30, there is roughly a 30% chance the short put finishes ITM, meaning a ~70% probability of keeping the full credit. **Lower delta = higher win rate but smaller premiums. Higher delta = larger premiums but more risk of assignment.** |
| `min_dte` | `20` | integer | 1 -- 90 | Minimum days to expiration. Avoids selling spreads too close to expiration where gamma risk (rapid price sensitivity changes) is highest. |
| `max_dte` | `45` | integer | > min_dte | Maximum days to expiration. The 20-45 DTE window is the "sweet spot" for theta decay (time value erosion) -- options lose value fastest in this range. Going further out ties up capital longer for less daily decay. |
| `max_spread_width` | `2.50` | float | 0.50 -- 20.0 | Maximum distance (in dollars) between the two strikes. This caps your maximum loss per spread at the width minus the credit received. At $2.50 width on a $500 account, max loss per spread is under $250. **Wider = more premium but more risk. Narrower = less premium, less risk.** |
| `min_credit_pct` | `0.30` | float | 0.10 -- 0.80 | Minimum credit received as a fraction of the spread width. At 0.30, a $2.50-wide spread must collect at least $0.75 in premium. This ensures the risk/reward is acceptable. **Higher = only enters high-premium trades (pickier). Lower = accepts thinner premiums.** |

---

### `strategies.debit_call_spread`

Buys a call spread (buy a lower-strike call, sell a higher-strike call). Profits when the stock rises. Costs less than a naked long call because the sold call offsets some of the purchase price.

| Parameter | Default | Type | Valid Range | Description |
|---|---|---|---|---|
| `enabled` | `true` | boolean | true/false | Enable or disable this strategy. |
| `long_delta` | `0.60` | float | 0.30 -- 0.90 | Delta of the long (purchased) call. At 0.60, the long call is slightly in-the-money, giving it a higher probability of profit but costing more. **Higher delta = more expensive, higher win rate. Lower delta = cheaper, more speculative.** |
| `short_delta` | `0.35` | float | 0.05 -- long_delta | Delta of the short (sold) call. Must be lower than `long_delta` (further out-of-the-money). The spread between the two deltas determines the spread's directional sensitivity. |
| `min_dte` | `30` | integer | 1 -- 120 | Minimum days to expiration. Debit spreads need time for the underlying to move. |
| `max_dte` | `60` | integer | > min_dte | Maximum days to expiration. Beyond 60 DTE, the cost of the spread increases and the theta decay you are paying for slows. |
| `max_debit_pct` | `0.60` | float | 0.10 -- 0.90 | Maximum debit (cost) as a fraction of the spread width. At 0.60, a $5-wide spread can cost at most $3.00 ($300 per contract). This ensures you are getting at least a 0.67:1 potential reward relative to the cost. **Lower = only cheap spreads (more speculative). Higher = accepts more expensive, higher-probability spreads.** |

---

### `strategies.long_call`

Buys a single call option outright. The simplest bullish options bet. Unlimited upside potential but the entire premium is at risk.

| Parameter | Default | Type | Valid Range | Description |
|---|---|---|---|---|
| `enabled` | `true` | boolean | true/false | Enable or disable this strategy. |
| `target_delta` | `0.60` | float | 0.20 -- 0.90 | Delta of the call to purchase. At 0.60, the call is slightly ITM, providing good directional exposure while retaining some extrinsic value. **Higher delta = more expensive, moves more with the stock. Lower delta = cheaper, more leveraged but lower probability.** |
| `min_dte` | `20` | integer | 1 -- 120 | Minimum days to expiration. Short-dated options lose value rapidly if the stock does not move quickly. |
| `max_dte` | `60` | integer | > min_dte | Maximum days to expiration. Balances time value cost against giving the trade time to work. |
| `max_contract_cost` | `75.0` | float | 10.0 -- 10000.0 | Maximum dollar cost per contract. One options contract controls 100 shares, so a $75 max means the premium cannot exceed $0.75 per share. This is calibrated for a $500 account. **Increase proportionally with account size.** |
| `max_iv_percentile` | `0.70` | float | 0.10 -- 1.0 | Maximum implied volatility percentile. IV percentile ranks current IV against historical IV. At 0.70, the strategy refuses to buy calls when IV is in the top 30% of its historical range, because options are expensive and you are overpaying for time value. **Lower = pickier about IV, buys only when options are cheap. Higher = allows buying in higher-IV environments.** |

---

### `strategies.long_put`

Buys a single put option for bearish directional bets or as a hedge. Profits when the stock falls.

| Parameter | Default | Type | Valid Range | Description |
|---|---|---|---|---|
| `enabled` | `true` | boolean | true/false | Enable or disable this strategy. |
| `target_delta` | `0.55` | float | 0.20 -- 0.90 | Delta of the put. At 0.55, the put is slightly ITM, providing solid downside exposure. Puts have negative delta, but this config stores the absolute value. |
| `min_dte` | `20` | integer | 1 -- 120 | Minimum days to expiration. |
| `max_dte` | `60` | integer | > min_dte | Maximum days to expiration. |
| `max_contract_cost` | `75.0` | float | 10.0 -- 10000.0 | Maximum dollar cost per contract. Same reasoning as long calls -- keeps individual trade risk proportional to a $500 account. |
| `breakdown_lookback` | `20` | integer | 5 -- 100 | Number of bars to look back for support levels. A put entry requires the stock to be breaking below a support level identified over this window. **Shorter = catches minor breakdowns. Longer = only enters on breaks of significant support.** |

---

### `strategies.cash_secured_put`

Sells a put option and sets aside enough cash to buy 100 shares if assigned. Collects premium while waiting to buy the stock at a lower price. Requires enough capital to purchase 100 shares.

| Parameter | Default | Type | Valid Range | Description |
|---|---|---|---|---|
| `enabled` | `true` | boolean | true/false | Enable or disable this strategy. |
| `target_delta` | `0.25` | float | 0.05 -- 0.50 | Delta of the put to sell. At 0.25, there is roughly a 25% chance of assignment. Conservative -- you collect modest premium with a high probability of keeping it. **Lower = safer but less income. Higher = more income but higher assignment risk.** |
| `min_dte` | `20` | integer | 1 -- 90 | Minimum days to expiration. |
| `max_dte` | `45` | integer | > min_dte | Maximum days to expiration. The 20-45 window optimizes theta decay. |
| `min_annualized_return` | `0.15` | float | 0.05 -- 1.0 | Minimum annualized return from the premium collected relative to the capital secured. At 0.15 (15%), if you secure $500 for a 30-day put, the premium must be at least ~$6.16. Filters out trades that are not worth the capital lock-up. **Higher = pickier, fewer trades. Lower = accepts thinner premiums.** |
| `max_stock_price` | `5.00` | float | 1.0 -- 10000.0 | Maximum stock price to sell puts on. Since you need cash to buy 100 shares, a $5 stock requires $500. This must align with your account size. **Scale this proportionally: $25K account could use $250.** |

---

### `strategies.momentum_options`

Buys cheap, short-dated options on stocks showing momentum breakouts. High risk, high reward. These are speculative plays on stocks that are already moving.

| Parameter | Default | Type | Valid Range | Description |
|---|---|---|---|---|
| `enabled` | `true` | boolean | true/false | Enable or disable this strategy. |
| `min_dte` | `2` | integer | 0 -- 30 | Minimum days to expiration. As low as 2 days -- these are very short-term bets. |
| `max_dte` | `20` | integer | > min_dte | Maximum days to expiration. Keeps the contracts cheap. |
| `min_delta` | `0.15` | float | 0.01 -- 0.50 | Minimum delta. Below 0.15, the option barely moves with the stock and is essentially a lottery ticket. |
| `max_delta` | `0.45` | float | min_delta -- 0.90 | Maximum delta. Caps cost -- higher delta options are more expensive. The 0.15-0.45 range targets OTM to slightly ITM contracts. |
| `max_contract_cost` | `100.0` | float | 5.0 -- 10000.0 | Maximum dollar cost per contract. At $100, this is 20% of a $500 account on a single speculative play. **Reduce for smaller accounts; increase proportionally for larger ones.** |
| `min_relative_volume` | `1.5` | float | 1.0 -- 10.0 | Minimum relative volume of the underlying stock. Ensures the stock has unusual activity, confirming the breakout has participation. |
| `breakout_lookback` | `20` | integer | 5 -- 100 | Number of bars used to identify the breakout level, same as the stock momentum strategy. |
| `min_roi_pct` | `0.25` | float | 0.05 -- 5.0 | Minimum estimated return-on-investment potential. At 0.25 (25%), the strategy only enters if the potential profit is at least 25% of the cost. Filters out contracts with poor leverage. |

---

### `strategies.covered_call`

Sells a call option against shares you already own. Generates income by collecting premium. You give up some upside beyond the strike price in exchange for guaranteed income.

| Parameter | Default | Type | Valid Range | Description |
|---|---|---|---|---|
| `enabled` | `true` | boolean | true/false | Enable or disable this strategy. |
| `target_delta` | `0.30` | float | 0.05 -- 0.50 | Delta of the call to sell. At 0.30, there is roughly a 30% chance the stock exceeds this strike, capping your upside. **Lower delta = less likely to be called away, less premium. Higher delta = more premium but stock is more likely to be called away.** |
| `min_dte` | `20` | integer | 1 -- 90 | Minimum days to expiration. |
| `max_dte` | `45` | integer | > min_dte | Maximum days to expiration. |
| `min_annualized_return` | `0.12` | float | 0.05 -- 1.0 | Minimum annualized return from the premium. At 0.12 (12%), the premium must annualize to at least a 12% return on the stock held. Ensures the income is worth the upside cap. |
| `max_stock_price` | `5.00` | float | 1.0 -- 10000.0 | Maximum price of the underlying stock. You must own 100 shares -- at $5/share that is $500, the entire default account. **Scale with account size.** |

---

### `strategies.covered_straddle`

Sells both a call and a put against shares you own. Collects double premium but adds the risk of put assignment (you may need to buy 100 more shares). Best in low-volatility, range-bound conditions.

| Parameter | Default | Type | Valid Range | Description |
|---|---|---|---|---|
| `enabled` | `true` | boolean | true/false | Enable or disable this strategy. |
| `min_dte` | `20` | integer | 1 -- 90 | Minimum days to expiration. |
| `max_dte` | `45` | integer | > min_dte | Maximum days to expiration. |
| `max_stock_price` | `5.00` | float | 1.0 -- 10000.0 | Maximum stock price. You need to hold 100 shares AND have cash to secure the put side. With a $5 stock, you need ~$1,000 total (100 shares at $5 plus $500 to secure the put). |
| `min_total_credit_pct` | `0.04` | float | 0.01 -- 0.20 | Minimum combined premium (call + put) as a percentage of the stock price. At 0.04 (4%), selling a straddle on a $5 stock must collect at least $0.20 total premium. **Higher = pickier, better risk/reward. Lower = accepts thinner premiums.** |
| `max_bb_width` | `0.10` | float | 0.02 -- 0.50 | Maximum Bollinger Band width (a volatility measure). A narrow BB width means the stock has been range-bound. At 0.10 (10%), the upper and lower Bollinger Bands must be within 10% of each other, ensuring the stock is in a low-volatility regime where a straddle is most profitable. **Lower = stricter, only in very quiet stocks. Higher = allows more volatile names.** |

---

### `sentiment`

News sentiment analysis that modifies conviction scores and can block trades on very negative news.

| Parameter | Default | Type | Valid Range | Description |
|---|---|---|---|---|
| `enabled` | `true` | boolean | true/false | Enable or disable sentiment analysis entirely. When disabled, all trades proceed without news checks. |
| `news_lookback_hours` | `36` | integer | 1 -- 168 | How many hours back to search for news articles. 36 hours covers the previous trading day plus overnight. **Longer = catches older catalysts. Shorter = focuses on very recent news.** |
| `news_max_articles` | `10` | integer | 1 -- 50 | Maximum articles to analyze per symbol. More articles give a better sentiment picture but increase processing time and API calls. |
| `min_conviction_after_mods` | `0.35` | float | 0.0 -- 1.0 | After all conviction modifiers are applied (market regime + news sentiment), a signal must still be above this threshold to proceed. This is the final gate. At 0.35, a signal that started at 0.80 conviction but was hammered by bearish regime and news can still trade if it stays above 0.35. **Higher = more conservative, blocks more trades. Lower = allows trades through even after negative modifiers.** |
| `block_on_bearish_news` | `-0.5` | float | -1.0 -- 0.0 | News sentiment score below which the trade is blocked entirely (regardless of conviction). The score ranges from -1.0 (extremely bearish) to +1.0 (extremely bullish). At -0.5, a stock with moderately bearish news is blocked. **Closer to 0.0 = blocks more easily. Closer to -1.0 = only blocks on very negative news.** |

---

### `options`

Global options trading limits that apply across all options strategies.

| Parameter | Default | Type | Valid Range | Description |
|---|---|---|---|---|
| `enabled` | `true` | boolean | true/false | Master switch for all options strategies. When `false`, the bot only trades stocks. |
| `max_options_positions` | `3` | integer | 1 -- 20 | Maximum number of concurrent options positions across all strategies. With a small account, 3 keeps risk manageable. Each position ties up capital (as margin, collateral, or premium). |
| `max_options_capital_pct` | `0.50` | float | 0.05 -- 1.0 | Maximum fraction of portfolio equity allocated to options. At 0.50 (50%), a $500 account can have up to $250 deployed in options. The other 50% stays available for stock trades or as cash reserves. **Higher = more options exposure. Lower = more conservative, keeps more cash free.** |
| `max_single_options_risk_pct` | `0.12` | float | 0.01 -- 0.50 | Maximum risk (max potential loss) on any single options trade as a fraction of equity. At 0.12 (12%), a $500 account limits single-trade max loss to $60. This is intentionally higher than the stock `max_risk_per_trade_pct` (2%) because options positions are sized differently. **Lower = smaller individual options bets. Higher = allows larger positions.** |

---

### `risk`

Portfolio-wide risk management parameters.

| Parameter | Default | Type | Valid Range | Description |
|---|---|---|---|---|
| `stop_loss_pct` | `0.03` | float | 0.01 -- 0.20 | Default stop loss as a percentage of entry price. At 0.03 (3%), a stock bought at $10 has a stop at $9.70. Strategies may override this with their own ATR-based stops, but this serves as a fallback. |
| `trailing_stop_pct` | `0.04` | float | 0.01 -- 0.20 | Trailing stop distance as a percentage. Once a position is profitable, the stop follows the price upward, always 4% below the highest price reached. Locks in gains while allowing the position to run. **Tighter = locks in profits sooner but may get stopped on normal pullbacks. Wider = gives the trade more room but risks giving back more profit.** |
| `take_profit_pct` | `0.06` | float | 0.02 -- 0.50 | Default take-profit level as a percentage of entry price. At 0.06 (6%), a stock bought at $10 has a target of $10.60. Combined with the 3% stop, this gives a 2:1 reward-to-risk ratio. Strategies may override this with ATR-based targets. |
| `position_sizing` | `"fractional"` | string | `"fractional"`, `"fixed"` | Position sizing method. `"fractional"` sizes positions based on the Kelly criterion and risk-per-trade percentage, meaning position size varies with conviction and volatility. `"fixed"` would use a fixed dollar amount per trade (not recommended for small accounts). |
| `kelly_fraction` | `0.25` | float | 0.05 -- 1.0 | Fraction of the full Kelly criterion to use. The Kelly criterion is a formula that determines optimal bet size given win rate and payoff ratio. Full Kelly (1.0) is mathematically optimal but has enormous volatility. Quarter-Kelly (0.25) sacrifices some expected growth for significantly smoother equity curves. **Higher = larger positions, faster growth in good times, deeper drawdowns in bad times. Lower = more conservative sizing.** |
| `max_portfolio_heat_pct` | `0.06` | float | 0.01 -- 0.30 | Maximum "heat" (total risk across all open positions) as a percentage of equity. If all open positions hit their stops simultaneously, the total loss must not exceed this percentage. At 0.06 (6%), total portfolio risk across all positions is capped at $30 on a $500 account. **Lower = very safe, but limits how many positions you can hold. Higher = allows more concurrent risk.** |

---

### `schedule`

All times are in US Eastern Time (ET). The bot uses a scheduler that fires jobs at these exact times on market days (Monday-Friday, excluding holidays).

| Parameter | Default | Type | Format | Description |
|---|---|---|---|---|
| `premarket_scan` | `"09:00"` | string | `"HH:MM"` | When the scanner runs to build the day's candidate list. 9:00 AM is 30 minutes before market open -- enough time for pre-market data to accumulate. |
| `market_open` | `"09:30"` | string | `"HH:MM"` | When the bot caches starting equity, syncs positions with Alpaca, and analyzes the market regime (SPY/QQQ/VIX analysis). This should match the NYSE market open. |
| `entry_window` | `"09:35"` | string | `"HH:MM"` | When the bot first evaluates strategies on the candidate list and submits trades. Set to 5 minutes after open to avoid the opening volatility spike where spreads are wide and prices are erratic. |
| `midday_check` | `"12:00"` | string | `"HH:MM"` | Mid-day re-evaluation. Syncs positions, re-runs strategies looking for new setups or managing existing ones. The lunch hour is often quieter, making it a good time for swing setups. |
| `power_hour_scan` | `"15:00"` | string | `"HH:MM"` | Final scan of the day. The last hour of trading (3:00-4:00 PM) often sees increased volume as institutional traders rebalance. Fresh candidates are scanned and evaluated. |
| `eod_close_day_trades` | `"15:50"` | string | `"HH:MM"` | 10 minutes before close, all positions with `hold_type: day` are force-closed. This ensures day trades do not accidentally become overnight positions (which could affect PDT status). |
| `eod_review` | `"16:05"` | string | `"HH:MM"` | 5 minutes after close, the bot saves a daily performance snapshot (equity, cash, P&L, positions). Waits until after close so final settlement prices are available. |

**Note on missed jobs**: If you start the bot during market hours (e.g., at 11:00 AM), it automatically catches up by running the premarket scan and market open jobs immediately. If there are no open positions, it also runs the entry window job to look for opportunities right away rather than waiting for the next scheduled window.

---

## Tuning Guide

### By Account Size

The default configuration is built for a **$500 paper-trading account**. Here is how to adjust for other sizes:

| Account Size | Key Changes |
|---|---|
| **$500 (default)** | Use as-is. Small positions, limited options capability. |
| **$1,000 - $2,500** | Increase `max_open_positions` to 5-6. Raise `max_contract_cost` to $150. Raise `max_stock_price` on cash_secured_put/covered_call to $15-$25. Consider raising `max_options_positions` to 4-5. |
| **$2,500 - $10,000** | Reduce `max_position_pct` to 0.15-0.20 (more diversification possible). Increase `max_open_positions` to 6-10. Raise `max_stock_price` to $50-100 on options strategies. Raise `max_contract_cost` to $300-500. |
| **$10,000 - $25,000** | Consider lowering `max_risk_per_trade_pct` to 0.01 (1%). Increase `max_open_positions` to 8-15. Scanner `max_price` can go to $200+. Options strategies can use wider spread widths. |
| **$25,000+** | PDT rules no longer apply -- set `max_day_trades: 3` or remove the constraint. More freedom with hold_type choices. Can run more aggressive momentum settings. |

### By Risk Tolerance

**Conservative (capital preservation)**:
- `max_risk_per_trade_pct`: 0.01 (1%)
- `daily_loss_limit_pct`: 0.03 (3%)
- `kelly_fraction`: 0.10
- `max_portfolio_heat_pct`: 0.04
- `rsi_oversold`: 30 (stricter entries)
- `min_conviction_for_day_trade`: 0.90
- Disable `momentum_options` (most speculative strategy)

**Moderate (the defaults)**:
- Use `settings.yaml` as-is. The defaults are moderately conservative with a slight growth bias.

**Aggressive (growth-oriented)**:
- `max_risk_per_trade_pct`: 0.03-0.04
- `daily_loss_limit_pct`: 0.08
- `kelly_fraction`: 0.40
- `max_portfolio_heat_pct`: 0.10
- `rsi_oversold`: 45 (catches earlier entries)
- `min_conviction_for_day_trade`: 0.65
- `volume_spike_multiplier`: 1.2 (catches more breakouts)
- `max_single_options_risk_pct`: 0.20

### By Market Conditions

**Bull Market (trending up)**:
- Favor momentum strategy: keep `breakout_lookback` at 20, lower `volume_spike_multiplier` to 1.2.
- Increase `max_options_capital_pct` for bullish options plays.
- Long calls and debit call spreads will be the primary options earners.

**Bear Market (trending down)**:
- Consider disabling momentum strategy or setting `hold_type: day` to avoid overnight gap risk.
- Long puts and credit put spreads (on strong names) become more relevant.
- Tighten `daily_loss_limit_pct` to 0.03.
- Lower `max_portfolio_heat_pct` to 0.04.

**Sideways / Low Volatility**:
- Covered calls, covered straddles, and credit put spreads thrive.
- Lower `max_bb_width` on covered straddle to 0.06 (very tight range required).
- Mean reversion strategy is most effective here.
- Consider raising `min_credit_pct` on credit put spreads since premiums may be thin.

**High Volatility (VIX > 25)**:
- The market regime analyzer handles some of this automatically, reducing position sizes and conviction.
- Avoid buying options (high IV inflates premiums) -- lower `max_iv_percentile` on long_call to 0.50.
- Selling premium strategies (credit spreads, covered calls) benefit from high IV.
- Widen stop losses: increase `atr_stop_multiplier` to 2.0-2.5 to avoid being stopped out on normal swings.
- Tighten `daily_loss_limit_pct` to 0.03.
