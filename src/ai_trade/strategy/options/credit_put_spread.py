"""
Bull Put Spread (Credit Put Spread) -- highest-probability options strategy.

Strategy Theory
---------------
A bull put spread is a two-leg credit strategy:
  1. **Sell** a put at a higher strike (collect premium)
  2. **Buy** a put at a lower strike (pay premium, acts as insurance)

You receive a **net credit** (the premium from the short put minus the cost of
the long put). This credit is yours to keep if both puts expire worthless.

Payoff Profile
~~~~~~~~~~~~~~
- **Max Profit**: The net credit received (if stock stays above short strike).
- **Max Loss**: (Spread width - credit received) x 100. This occurs if the stock
  falls below the long (lower) strike at expiration.
- **Breakeven**: Short strike - net credit received.

When It Works Best
~~~~~~~~~~~~~~~~~~
- Mildly to moderately **bullish** outlook -- you believe the stock will stay
  above the short put strike through expiration.
- Elevated IV (implied volatility) environments: you collect fatter premiums.
- Time is on your side: theta (time decay) benefits the seller.

Risk / Reward
~~~~~~~~~~~~~
- **Defined risk**: the worst case is the spread width minus the credit.
- **High probability**: choosing a short put delta of ~0.30 means roughly 70%
  chance of the stock staying above that strike.
- **Capped reward**: you cannot make more than the credit received.

Key Concepts in This File
--------------------------
- **Spread width**: The dollar distance between the two strikes (e.g. $2.50).
- **Credit as % of width**: ``credit / spread_width``. Higher is better -- a
  ratio of 0.33 means you collect 33% of max risk as premium.
- **Annualized return**: Not used directly here, but ``credit / width * (365/DTE)``
  lets you compare strategies of different durations.
"""

# ``from __future__ import annotations`` enables using modern type-hint syntax
# (e.g. ``list[dict]``, ``X | None``) on Python versions before 3.10.
from __future__ import annotations

import pandas as pd

from ai_trade.data.indicators import add_ema, add_rsi
from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.options.base import (
    BaseOptionsStrategy,
    OptionsSignal,
    OptionsStrategyType,
    enrich_greeks,
    filter_contracts,
    select_by_delta,
)

log = get_logger(__name__)


class CreditPutSpreadStrategy(BaseOptionsStrategy):
    """Enter a bull put spread when the underlying is in an uptrend.

    Inherits from ``BaseOptionsStrategy`` (an abstract base class) which
    requires implementing the ``evaluate()`` method.
    """

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        underlying: str,
        stock_bars: pd.DataFrame,
        chain_data: list[dict],
        snapshots: dict,
    ) -> OptionsSignal | None:
        # Early exit if this strategy is disabled in the config.
        if not self.enabled:
            return None

        # ---- Configuration parameters ----
        # ``getattr(self.config, "target_delta", 0.30)`` safely reads the
        # attribute "target_delta" from the config object. If it doesn't exist,
        # returns the default 0.30. This pattern is used throughout to make
        # config fields optional with sensible defaults.
        #
        # target_delta = 0.30 means we want to sell a put with ~30% probability
        # of expiring ITM (in-the-money). Lower delta = further OTM = safer
        # but less premium collected.
        target_delta: float = getattr(self.config, "target_delta", 0.30)
        # DTE range: 20-45 days is the "sweet spot" for theta decay.
        # Options lose time value fastest in the last 30-45 days.
        min_dte: int = getattr(self.config, "min_dte", 20)
        max_dte: int = getattr(self.config, "max_dte", 45)
        # max_spread_width caps how far apart the strikes can be (in dollars).
        # Wider spread = higher max loss but also higher credit.
        max_spread_width: float = getattr(self.config, "max_spread_width", 2.50)
        # Minimum credit as a fraction of spread width. 0.30 = we want at
        # least 30% of the spread width as credit (good risk/reward).
        min_credit_pct: float = getattr(self.config, "min_credit_pct", 0.30)
        # Hard cap on maximum dollar loss per trade.
        max_risk: float = getattr(self.config, "max_risk", 250.0)
        # Available capital for this strategy.
        available_capital: float = getattr(self.config, "available_capital", 500.0)

        # ------------------------------------------------------------------
        # 1. Stock filter -- require an uptrend before selling puts
        # ------------------------------------------------------------------
        # ``.copy()`` creates a shallow copy of the DataFrame to avoid
        # modifying the original (indicator functions add columns in-place).
        df = stock_bars.copy()
        # Need at least 21 bars to compute the 20-period RSI and EMA.
        if len(df) < 21:
            return None

        # Compute RSI (Relative Strength Index) and 20-period EMA
        # (Exponential Moving Average) on the DataFrame.
        # RSI ranges 0-100: above 50 = bullish momentum, below 50 = bearish.
        # EMA-20 is a trend-following indicator; price above EMA-20 = uptrend.
        add_rsi(df)
        add_ema(df, periods=[20])

        # ``df.iloc[-1]`` gets the last (most recent) row of the DataFrame.
        # ``.iloc`` is Pandas' integer-location indexer (0-based). -1 means
        # the last row, like Python list indexing.
        latest = df.iloc[-1]
        # ``.get("rsi_14", 0.0)`` reads the column value, defaulting to 0.0
        # if the column doesn't exist. This is dict-style access on a Pandas
        # Series (each DataFrame row is a Series).
        rsi: float = latest.get("rsi_14", 0.0)
        ema_20: float = latest.get("ema_20", 0.0)
        price: float = latest["close"]

        # Entry condition 1: RSI must be above 40. We don't want to sell puts
        # into a downtrend -- that's selling insurance when the house is on fire.
        if rsi <= 40:
            log.debug("credit_put_spread_skip", underlying=underlying, reason="RSI<=40", rsi=rsi)
            return None
        # Entry condition 2: Price must be above the 20-day EMA (uptrend).
        if price <= ema_20:
            log.debug("credit_put_spread_skip", underlying=underlying, reason="price<=EMA20")
            return None

        # ------------------------------------------------------------------
        # 2. Select expiration -- filter to puts with 20-45 DTE
        # ------------------------------------------------------------------
        # ``filter_contracts`` narrows the full options chain to only put
        # contracts within our desired DTE window.
        eligible_puts = filter_contracts(chain_data, "put", min_dte, max_dte)

        if not eligible_puts:
            log.debug("credit_put_spread_skip", underlying=underlying, reason="no eligible puts")
            return None

        # ------------------------------------------------------------------
        # 3. Select strikes for both legs
        # ------------------------------------------------------------------
        # Enrich each contract dict with greeks (_delta, _theta) and pricing
        # (_bid, _ask, _mid) from the real-time snapshots.
        enrich_greeks(eligible_puts, snapshots)

        # Short put: the one we SELL. We want its delta closest to our target
        # (0.30 by default). ``select_by_delta`` finds the contract whose
        # absolute delta is nearest to ``target_delta``.
        # For puts, delta is negative (e.g. -0.30), but ``select_by_delta``
        # uses absolute values by default so we pass 0.30 not -0.30.
        short_put = select_by_delta(eligible_puts, target_delta)
        if short_put is None:
            return None

        short_strike: float = short_put["_strike"]

        # Long put (the one we BUY for protection): must be at a lower strike
        # than the short put, within max_spread_width dollars, and in the
        # same expiration cycle.
        #
        # This is a list comprehension that filters eligible_puts to only those
        # contracts meeting all three conditions:
        #   1. Strike is below the short strike (further OTM)
        #   2. The distance is within max_spread_width
        #   3. Same expiration date as the short put
        long_candidates = [
            c
            for c in eligible_puts
            if c["_strike"] < short_strike
            and (short_strike - c["_strike"]) <= max_spread_width
            and c.get("expiration_date", c.get("expiration", ""))
            == short_put.get("expiration_date", short_put.get("expiration", ""))
        ]
        if not long_candidates:
            log.debug("credit_put_spread_skip", underlying=underlying, reason="no long put candidates")
            return None

        # Pick the long put with the *highest* strike that is still below the
        # short strike. This gives the narrowest spread (lowest risk) while
        # still providing downside protection.
        # ``max(..., key=lambda c: c["_strike"])`` returns the dict with the
        # largest "_strike" value from the list.
        long_put = max(long_candidates, key=lambda c: c["_strike"])
        long_strike: float = long_put["_strike"]

        # ------------------------------------------------------------------
        # 4. Calculate pricing -- credit, max loss, max profit
        # ------------------------------------------------------------------
        # Net credit = what we receive for the short put minus what we pay
        # for the long put. We use mid-prices as fair-value estimates.
        credit_received: float = short_put["_mid"] - long_put["_mid"]

        # Spread width = distance between strikes in dollars.
        spread_width: float = short_strike - long_strike
        if spread_width <= 0:
            return None  # Sanity check: short strike should be above long strike.

        # Max loss = (spread width * 100 shares per contract) minus credit collected.
        # Example: $2.50 wide spread, $0.80 credit -> max loss = $250 - $80 = $170.
        max_loss: float = (spread_width * 100) - (credit_received * 100)
        # Max profit = the full credit * 100 shares per contract.
        max_profit: float = credit_received * 100

        # Minimum credit threshold: don't enter trades with tiny credits
        # (commissions and slippage would eat the profit).
        if credit_received < 0.10:
            log.debug("credit_put_spread_skip", underlying=underlying, reason="credit<0.10")
            return None
        # Risk management: don't exceed per-trade risk limits.
        if max_loss > max_risk or max_loss > available_capital:
            log.debug("credit_put_spread_skip", underlying=underlying, reason="max_loss exceeds limits")
            return None
        # Credit-to-width ratio check: ensures adequate reward for the risk taken.
        # If we risk $2.50 of width, we want at least $0.75 in credit (30%).
        if credit_received / spread_width < min_credit_pct:
            log.debug("credit_put_spread_skip", underlying=underlying, reason="credit_pct too low")
            return None

        # ------------------------------------------------------------------
        # 5. Conviction scoring (0.0 to 1.0)
        # ------------------------------------------------------------------
        # Start with a moderate base conviction of 0.6 (this strategy is
        # inherently high-probability).
        conviction: float = 0.6
        # Boost if RSI confirms bullish momentum (above neutral 50).
        if rsi > 50:
            conviction += 0.1
        # Boost if credit/width ratio is strong (>33% of max risk as premium).
        if credit_received / spread_width > 0.33:
            conviction += 0.1
        # Net theta: the short put's theta minus the long put's theta.
        # Positive net theta means time decay is working in our favor.
        # Theta for puts is typically negative; the seller benefits because
        # they want the put to lose value (decay toward zero).
        net_theta: float = short_put["_theta"] - abs(long_put["_theta"])
        if net_theta > 0:
            conviction += 0.1
        # Clamp conviction to [0.5, 1.0] range.
        # ``max(0.5, min(1.0, conviction))`` is Python's way of clamping:
        # first cap at 1.0, then floor at 0.5.
        conviction = max(0.5, min(1.0, conviction))

        # ------------------------------------------------------------------
        # 6. Build and return the trade signal
        # ------------------------------------------------------------------
        short_put_symbol: str = short_put.get("symbol", "")
        long_put_symbol: str = long_put.get("symbol", "")
        expiration: str = short_put.get("expiration_date") or short_put.get("expiration", "")
        # Net delta of the spread = sum of both legs' deltas.
        # Short put has negative delta (e.g. -0.30) and long put also has
        # negative delta (e.g. -0.15), but since we sold the short put,
        # the combined position is slightly bullish (less negative).
        net_delta: float = short_put["_delta"] + long_put["_delta"]

        # Each leg is a dict describing one side of the spread.
        # "sell_to_open" = initiate a short position; "buy_to_open" = initiate
        # a long position. The order router uses these to construct API calls.
        legs = [
            {"symbol": short_put_symbol, "side": "sell", "qty": 1, "position_intent": "sell_to_open"},
            {"symbol": long_put_symbol, "side": "buy", "qty": 1, "position_intent": "buy_to_open"},
        ]

        log.info(
            "credit_put_spread_signal",
            underlying=underlying,
            short_strike=short_strike,
            long_strike=long_strike,
            credit=credit_received,
            max_loss=max_loss,
            conviction=conviction,
            expiration=expiration,
        )

        # Return the fully-populated OptionsSignal dataclass instance.
        return OptionsSignal(
            underlying=underlying,
            strategy_type=OptionsStrategyType.CREDIT_PUT_SPREAD,
            conviction=conviction,
            strategy_name="credit_put_spread",
            legs=legs,
            min_credit=credit_received,
            max_loss=max_loss,
            max_profit=max_profit,
            expiration=expiration,
            strikes=[short_strike, long_strike],
            net_delta=net_delta,
            net_theta=net_theta,
            metadata={
                "rsi": rsi,
                "price": price,
                "ema_20": ema_20,
                "spread_width": spread_width,
                "credit_pct": credit_received / spread_width,
                "dte": short_put.get("_dte", 0),
            },
        )
