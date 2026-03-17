"""
Bull Call Spread (Debit Call Spread) -- defined-risk directional strategy.

Strategy Theory
---------------
A bull call spread is a two-leg debit strategy:
  1. **Buy** a call near ATM (at-the-money) or slightly ITM (in-the-money)
  2. **Sell** a call further OTM (out-of-the-money)

You pay a **net debit** (the cost of the long call minus the credit from the
short call). This is cheaper than buying a naked call outright.

Payoff Profile
~~~~~~~~~~~~~~
- **Max Profit**: (Spread width - debit paid) x 100. Achieved when the stock
  is at or above the short strike at expiration.
- **Max Loss**: The debit paid x 100. Occurs if the stock is below the long
  (lower) strike at expiration -- both calls expire worthless.
- **Breakeven**: Long strike + net debit paid.

When It Works Best
~~~~~~~~~~~~~~~~~~
- **Moderately bullish** outlook with strong momentum confirmation.
- When you want directional exposure at a lower cost than a naked long call.
- In moderate-IV environments (not overpaying for premium).

Why Use a Spread Instead of a Naked Call?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Lower cost of entry (the short call offsets part of the long call cost).
- Defined max loss (the net debit is the most you can lose).
- Tradeoff: your upside is capped at the spread width minus debit.

Key Metrics in This File
-------------------------
- **Debit as % of spread width**: ``debit / spread_width``. Lower is better.
  Paying 40% of the width means 60% potential profit if the stock runs.
- **Relative volume**: Today's volume divided by the 20-day average. Values
  above 1.5 indicate unusual institutional activity.
"""

from __future__ import annotations

import pandas as pd

from ai_trade.data.indicators import add_ema, add_rsi, add_volume_profile
from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.options.base import (
    BaseOptionsStrategy,
    OptionsSignal,
    OptionsStrategyType,
    enrich_greeks,
    filter_by_delta,
    filter_contracts,
    select_by_delta,
)

log = get_logger(__name__)


class DebitCallSpreadStrategy(BaseOptionsStrategy):
    """Enter a bull call spread on strong-momentum setups.

    Requires price above EMA-20, RSI between 50-70 (strong but not
    overbought), and relative volume above 1.2x normal.
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
        if not self.enabled:
            return None

        # ---- Configuration ----
        # long_delta = 0.60: Buy the call with ~60 delta (slightly ITM or ATM).
        # Higher delta = more expensive but more responsive to stock movement.
        long_delta: float = getattr(self.config, "long_delta", 0.60)
        # short_delta = 0.35: Sell the call with ~35 delta (OTM).
        # This is the cap on our profit -- we give up gains above this strike.
        short_delta: float = getattr(self.config, "short_delta", 0.35)
        # 30-60 DTE: slightly longer than credit spreads because we need time
        # for the stock to move in our direction.
        min_dte: int = getattr(self.config, "min_dte", 30)
        max_dte: int = getattr(self.config, "max_dte", 60)
        # max_debit_pct = 0.60: don't pay more than 60% of the spread width.
        # Paying 60% to potentially make 40% profit is the minimum acceptable ratio.
        max_debit_pct: float = getattr(self.config, "max_debit_pct", 0.60)
        max_risk: float = getattr(self.config, "max_risk", 250.0)
        available_capital: float = getattr(self.config, "available_capital", 500.0)

        # ------------------------------------------------------------------
        # 1. Stock filter -- require strong momentum confirmation
        # ------------------------------------------------------------------
        df = stock_bars.copy()
        if len(df) < 21:
            return None

        # Add technical indicators to the DataFrame:
        # - RSI (Relative Strength Index): momentum oscillator, 0-100
        # - EMA-20 (Exponential Moving Average, 20-period): trend direction
        # - Volume profile: computes relative_volume (today vs 20-day average)
        add_rsi(df)
        add_ema(df, periods=[20])
        add_volume_profile(df)

        latest = df.iloc[-1]
        rsi: float = latest.get("rsi_14", 0.0)
        ema_20: float = latest.get("ema_20", 0.0)
        price: float = latest["close"]
        rel_vol: float = latest.get("relative_volume", 0.0)

        # Entry condition 1: RSI in the "strong momentum" zone (50-70).
        # Below 50 = no bullish momentum. Above 70 = overbought, risky to enter.
        # ``not (50 < rsi < 70)`` is Python's chained comparison -- it checks
        # that rsi is strictly between 50 and 70 in a single expression.
        if not (50 < rsi < 70):
            log.debug("debit_call_spread_skip", underlying=underlying, reason="RSI outside 50-70", rsi=rsi)
            return None
        # Entry condition 2: Price above EMA-20 confirms uptrend.
        if price <= ema_20:
            log.debug("debit_call_spread_skip", underlying=underlying, reason="price<=EMA20")
            return None
        # Entry condition 3: Volume at least 1.2x average. High volume
        # validates the price move (institutions are participating).
        if rel_vol <= 1.2:
            log.debug("debit_call_spread_skip", underlying=underlying, reason="relative_volume<=1.2")
            return None

        # ------------------------------------------------------------------
        # 2. Select expiration -- filter to calls with 30-60 DTE
        # ------------------------------------------------------------------
        eligible_calls = filter_contracts(chain_data, "call", min_dte, max_dte)

        if not eligible_calls:
            log.debug("debit_call_spread_skip", underlying=underlying, reason="no eligible calls")
            return None

        # ------------------------------------------------------------------
        # 3. Enrich with greeks and select both strikes
        # ------------------------------------------------------------------
        enrich_greeks(eligible_calls, snapshots)

        # --- Long call selection (the one we BUY) ---
        # Filter to calls with delta 0.55-0.65 (near ATM, responsive to moves).
        # ``use_absolute=False`` because call deltas are already positive.
        long_candidates = filter_by_delta(eligible_calls, 0.55, 0.65, use_absolute=False)
        if not long_candidates:
            # Fallback: any call with positive delta (i.e., any valid call).
            long_candidates = [c for c in eligible_calls if c["_delta"] > 0]
        if not long_candidates:
            return None

        # Pick the call whose delta is closest to our target (0.60).
        long_call = select_by_delta(long_candidates, long_delta, use_absolute=False)

        # --- Short call selection (the one we SELL) ---
        long_strike: float = long_call["_strike"]
        # Get the expiration of the long call so we match it exactly.
        long_exp = long_call.get("expiration_date") or long_call.get("expiration", "")

        # Filter to calls with a HIGHER strike in the SAME expiration.
        # Both legs must share an expiration for a proper vertical spread.
        same_exp_above = [
            c
            for c in eligible_calls
            if c["_strike"] > long_strike
            and (c.get("expiration_date") or c.get("expiration", "")) == long_exp
        ]
        # Filter these to delta 0.30-0.40 (OTM -- this defines our profit cap).
        short_candidates = filter_by_delta(same_exp_above, 0.30, 0.40, use_absolute=False)
        if not short_candidates:
            # Fallback: any call above long strike with positive delta.
            short_candidates = [c for c in same_exp_above if c["_delta"] > 0]
        if not short_candidates:
            return None

        short_call = select_by_delta(short_candidates, short_delta, use_absolute=False)
        short_strike: float = short_call["_strike"]

        # Spread width = distance between strikes (in dollars).
        spread_width: float = short_strike - long_strike
        if spread_width <= 0:
            return None  # Sanity check.

        # Cap spread width based on stock price. For a $50 stock, max width
        # is $1.00; for a $300 stock, max width is $5.00.
        # ``round(price * 0.02, 0)`` computes 2% of price, rounded to
        # the nearest dollar. ``min/max`` clamp it to $1-$5.
        max_width = min(5.0, max(1.0, round(price * 0.02, 0)))
        if spread_width > max_width:
            return None

        # ------------------------------------------------------------------
        # 4. Calculate pricing
        # ------------------------------------------------------------------
        # Net debit = cost of long call minus credit from short call.
        # We use mid-prices as fair-value estimates for the fill.
        debit: float = long_call["_mid"] - short_call["_mid"]
        if debit <= 0:
            return None  # Should always be positive for a bull call spread.

        # Max loss = debit * 100 shares per contract (what we paid to enter).
        max_loss: float = debit * 100
        # Max profit = (spread width - debit) * 100.
        # Example: $2.50 spread, $1.20 debit -> max profit = $1.30 * 100 = $130.
        max_profit: float = (spread_width - debit) * 100

        # Debit as a percentage of spread width. If > 60%, the risk/reward
        # is too unfavorable (paying too much for limited upside).
        if debit / spread_width > max_debit_pct:
            log.debug("debit_call_spread_skip", underlying=underlying, reason="debit_pct too high")
            return None
        if max_loss > max_risk or max_loss > available_capital:
            log.debug("debit_call_spread_skip", underlying=underlying, reason="max_loss exceeds limits")
            return None

        # ------------------------------------------------------------------
        # 5. Conviction scoring
        # ------------------------------------------------------------------
        conviction: float = 0.5
        # Higher relative volume signals stronger institutional participation.
        if rel_vol > 1.5:
            conviction += 0.15
        # Favorable debit/width ratio (paying less than 50% of the width).
        if debit / spread_width < 0.50:
            conviction += 0.1
        # RSI in the sweet spot (55-65): strong but not overbought.
        if 55 <= rsi <= 65:
            conviction += 0.1
        conviction = max(0.5, min(1.0, conviction))

        # ------------------------------------------------------------------
        # 6. Build and return the trade signal
        # ------------------------------------------------------------------
        long_call_symbol: str = long_call.get("symbol", "")
        short_call_symbol: str = short_call.get("symbol", "")
        expiration: str = long_exp
        # Net delta: sum of both legs. The long call has higher delta, so
        # net delta is positive (bullish).
        net_delta: float = long_call["_delta"] + short_call["_delta"]
        # Net theta: typically negative for debit spreads (time decay hurts
        # the buyer more than the short call theta helps).
        net_theta: float = long_call["_theta"] + short_call["_theta"]

        legs = [
            {"symbol": long_call_symbol, "side": "buy", "qty": 1, "position_intent": "buy_to_open"},
            {"symbol": short_call_symbol, "side": "sell", "qty": 1, "position_intent": "sell_to_open"},
        ]

        log.info(
            "debit_call_spread_signal",
            underlying=underlying,
            long_strike=long_strike,
            short_strike=short_strike,
            debit=debit,
            max_loss=max_loss,
            max_profit=max_profit,
            conviction=conviction,
            expiration=expiration,
        )

        return OptionsSignal(
            underlying=underlying,
            strategy_type=OptionsStrategyType.DEBIT_CALL_SPREAD,
            conviction=conviction,
            strategy_name="debit_call_spread",
            legs=legs,
            max_cost=debit,
            max_loss=max_loss,
            max_profit=max_profit,
            expiration=expiration,
            strikes=[long_strike, short_strike],
            net_delta=net_delta,
            net_theta=net_theta,
            metadata={
                "rsi": rsi,
                "price": price,
                "ema_20": ema_20,
                "relative_volume": rel_vol,
                "spread_width": spread_width,
                "debit_pct": debit / spread_width,
                "dte": long_call.get("_dte", 0),
            },
        )
