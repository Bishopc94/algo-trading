"""
Long Put -- directional bearish bet on breakdowns.

Strategy Theory
---------------
Buy a single put option when strong bearish signals confirm a breakdown:
  - Price breaks below its 20-day low (technical breakdown)
  - Price is below both EMA-20 and EMA-50 (confirmed downtrend)
  - RSI is below 40 (bearish momentum)
  - Volume is elevated (1.5x+ normal)

A put option gains value as the underlying stock falls. This is the bearish
counterpart to the Long Call strategy.

Payoff Profile
~~~~~~~~~~~~~~
- **Max Profit**: (Strike price - 0) x 100 - premium paid. Theoretically,
  the stock could go to zero, making the put worth its full strike value.
- **Max Loss**: The premium paid x 100 shares per contract. If the stock
  stays above the strike, the put expires worthless.
- **Breakeven**: Strike price - premium paid.

When It Works Best
~~~~~~~~~~~~~~~~~~
- Confirmed downtrends with volume confirmation (not just a dip).
- Moderate IV: like long calls, buying puts in high-IV is expensive.
- Crisis / panic selling environments where fear drives sharp declines.

Risk / Reward
~~~~~~~~~~~~~
- **Defined risk**: you can never lose more than the premium paid.
- **Large reward potential**: stocks can fall much faster than they rise
  (crashes vs. rallies), so puts can produce very large percentage gains.
- **Time decay works against you**: every day the stock doesn't drop, the
  put loses value (theta decay).

Key Concepts in This File
--------------------------
- **Breakdown**: Price falls below the lowest low of the last N days.
  The mirror image of a breakout.
- **Put delta**: Put deltas are negative (e.g. -0.55). A delta of -0.55
  means the put gains ~$0.55 for every $1 the stock drops. We use
  absolute values (0.55) when comparing to targets for simplicity.
- **EMA-50 as secondary trend filter**: The 50-day EMA is a slower trend
  indicator. Price below both EMA-20 and EMA-50 confirms a multi-timeframe
  downtrend, not just a short-term dip.
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


class LongPutStrategy(BaseOptionsStrategy):
    """Enter a long put on confirmed breakdowns with elevated volume.

    This is the bearish counterpart to LongCallStrategy. It requires
    price below both EMAs, RSI < 40, and a new 20-day low on volume.
    """

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
        # target_delta = 0.55: ATM to slightly ITM puts. Using absolute value
        # convention -- the actual put delta will be around -0.55.
        target_delta: float = getattr(self.config, "target_delta", 0.55)
        min_dte: int = getattr(self.config, "min_dte", 30)
        max_dte: int = getattr(self.config, "max_dte", 60)
        max_contract_cost: float = getattr(self.config, "max_contract_cost", 75.0)
        breakdown_lookback: int = getattr(self.config, "breakdown_lookback", 20)

        # ------------------------------------------------------------------
        # Stock filter -- require a bearish breakdown
        # ------------------------------------------------------------------
        df = stock_bars.copy()
        if len(df) < breakdown_lookback + 1:
            return None

        # Compute indicators: RSI, two EMAs (20 and 50), and volume profile.
        # Using two EMAs (short-term and medium-term) provides stronger trend
        # confirmation than a single EMA.
        add_rsi(df)
        add_ema(df, periods=[20, 50])
        add_volume_profile(df)

        # Compute the rolling 20-day low. ``.rolling(N).min()`` finds the
        # minimum value over a sliding window. ``.shift(1)`` ensures we
        # compare today's price to yesterday's 20-day low (no look-ahead).
        df["low_20"] = df["low"].rolling(breakdown_lookback).min().shift(1)

        latest = df.iloc[-1]
        rsi: float = latest.get("rsi_14", 50.0)
        price: float = latest["close"]
        ema_20: float = latest.get("ema_20", price)
        ema_50: float = latest.get("ema_50", price)
        rel_vol: float = latest.get("relative_volume", 1.0)
        low_20: float = latest.get("low_20", price)

        # --- Bearish entry conditions (ALL must be true) ---
        # 1. RSI below 40: confirms bearish momentum.
        if rsi > 40:
            return None
        # 2. Price below BOTH moving averages: multi-timeframe downtrend.
        if price >= ema_20 or price >= ema_50:
            return None
        # 3. Price below the prior 20-day low: technical breakdown.
        if price > low_20:
            return None
        # 4. Elevated volume: validates the breakdown as institutional selling
        #    (not just a random dip on low volume).
        if rel_vol < 1.5:
            return None

        # ------------------------------------------------------------------
        # Select contract -- ATM to slightly ITM put, 30-60 DTE
        # ------------------------------------------------------------------
        eligible_puts = filter_contracts(chain_data, "put", min_dte, max_dte)

        if not eligible_puts:
            return None

        enrich_greeks(eligible_puts, snapshots)

        # Filter to puts with absolute delta 0.45-0.65 (ATM to slightly ITM).
        # ``fallback_min=0.2`` means: if no puts match 0.45-0.65, fall back
        # to any put with abs(delta) > 0.20 (not too far OTM).
        delta_candidates = filter_by_delta(eligible_puts, 0.45, 0.65, fallback_min=0.2)
        if not delta_candidates:
            return None

        # Pick the put closest to our target delta (0.55 absolute).
        selected = select_by_delta(delta_candidates, target_delta)

        # Use the ask price since we're buying the put.
        ask_price: float = selected["_ask"]
        cost_per_contract: float = ask_price * 100

        if cost_per_contract > max_contract_cost or ask_price <= 0:
            return None

        # ------------------------------------------------------------------
        # Conviction scoring
        # ------------------------------------------------------------------
        conviction: float = 0.5
        # RSI below 30 = extremely oversold / panic selling = strong signal.
        if rsi < 30:
            conviction += 0.15
        # Volume 2x+ normal = heavy institutional selling.
        if rel_vol > 2.0:
            conviction += 0.15
        # Price more than 3% below EMA-50 = deep downtrend, strong momentum.
        if price < ema_50 * 0.97:
            conviction += 0.1
        conviction = max(0.5, min(1.0, conviction))

        # ------------------------------------------------------------------
        # Build and return the trade signal
        # ------------------------------------------------------------------
        put_symbol: str = selected.get("symbol", "")
        expiration: str = selected.get("expiration_date") or selected.get("expiration", "")
        strike: float = selected["_strike"]

        legs = [
            {"symbol": put_symbol, "side": "buy", "qty": 1, "position_intent": "buy_to_open"},
        ]

        log.info(
            "long_put_signal",
            underlying=underlying,
            strike=strike,
            ask=ask_price,
            cost=cost_per_contract,
            # ``abs()`` converts the negative put delta to positive for logging.
            delta=abs(selected["_delta"]),
            conviction=conviction,
            expiration=expiration,
        )

        return OptionsSignal(
            underlying=underlying,
            strategy_type=OptionsStrategyType.LONG_PUT,
            conviction=conviction,
            strategy_name="long_put",
            legs=legs,
            max_cost=ask_price,
            max_loss=cost_per_contract,
            # Max profit for a long put: if the stock goes to zero, the put
            # is worth (strike * 100). Subtract the cost to get net profit.
            max_profit=strike * 100 - cost_per_contract,
            expiration=expiration,
            strikes=[strike],
            # Put delta is negative (e.g. -0.55), indicating bearish exposure.
            net_delta=selected["_delta"],
            # Theta for a long put is negative (time decay hurts the buyer).
            net_theta=selected["_theta"],
            metadata={
                "rsi": rsi,
                "price": price,
                "relative_volume": rel_vol,
                "low_20": low_20,
                "cost_per_contract": cost_per_contract,
                "dte": selected.get("_dte", 0),
            },
        )
