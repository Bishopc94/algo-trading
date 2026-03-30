"""
Momentum Options -- short-dated OTM plays for 50-300%+ ROI.

Strategy Theory
---------------
Buy cheap OTM (out-of-the-money) options with very short DTE (2-10 days) on
high-conviction momentum moves. This strategy profits from **gamma**: the
acceleration of an option's delta as the stock moves toward the strike.

Two modes:
  - **Bullish breakout**: Buy OTM calls when the stock breaks above its 20-day
    high with elevated volume and RSI > 50.
  - **Bearish breakdown**: Buy OTM puts when the stock breaks below its 20-day
    low with elevated volume and RSI < 50.

Payoff Profile
~~~~~~~~~~~~~~
- **Max Profit**: Theoretically unlimited for calls; (strike x 100 - premium)
  for puts. Practical expectation is 50-300% ROI if the move hits.
- **Max Loss**: The entire premium paid. Short-dated OTM options are cheap
  ($10-$50 per contract), so the absolute dollar risk is small.
- **Breakeven**: Strike +/- premium paid (depending on call or put).

When It Works Best
~~~~~~~~~~~~~~~~~~
- **Large, fast directional moves**: earnings surprises, sector catalysts,
  technical breakouts confirmed by volume.
- **Low premium environment**: short DTE + OTM = very cheap options. A $0.15
  option that doubles to $0.30 is a 100% return on $15 of risk.
- **High gamma**: options near expiration with a strike near the stock price
  move very quickly as delta accelerates.

Risk / Reward
~~~~~~~~~~~~~
- **Highest risk**: Most of these trades will expire worthless. The strategy
  relies on the winners being large enough to offset the losers.
- **Highest potential return**: 2x to 5x returns are common on winners.
- **Position sizing is critical**: Use only 1-2% of account per trade.
  This is a lottery-ticket strategy.

Key Concepts in This File
--------------------------
- **Gamma**: The rate of change of delta. Near expiration, gamma is highest
  for ATM options. A small stock move causes a large change in delta, which
  causes a large change in the option price. This is why short-dated options
  near the money can produce explosive percentage gains.
- **ATR (Average True Range)**: Measures the stock's typical daily price range.
  Used here to estimate how far the stock might move, and thus what the
  option could be worth if the move hits.
- **Potential ROI calculation**: We estimate what the option would be worth
  if the stock moves 1.5x its ATR in our direction, then compare that to
  the premium paid. A minimum 50% ROI target filters out trades where the
  reward isn't worth the risk.
- **Short DTE (2-10 days)**: Very little time value = cheap options. But also
  very little time for the thesis to play out. This is a "binary" bet.
"""

from __future__ import annotations

import pandas as pd

from ai_trade.data.indicators import add_atr, add_ema, add_rsi, add_volume_profile
from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.options.base import (
    BaseOptionsStrategy,
    OptionsSignal,
    OptionsStrategyType,
    enrich_greeks,
    filter_by_delta,
    filter_contracts,
)

log = get_logger(__name__)


class MomentumOptionsStrategy(BaseOptionsStrategy):
    """Buy short-dated OTM options on high-conviction momentum moves.

    Calls on breakouts, puts on breakdowns. Targets 50-300% ROI with
    defined risk (premium paid). This is the most aggressive strategy
    in the system and should receive the smallest position allocation.
    """

    bias = "adaptive"

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
        # Very short DTE: 2-10 days. These options are cheap (little time value
        # remaining) but decay extremely fast -- you need the move to happen
        # within days.
        min_dte: int = getattr(self.config, "min_dte", 2)
        max_dte: int = getattr(self.config, "max_dte", 10)
        # Delta range for OTM but not too far OTM options.
        # 0.15-0.45 delta = options that are OTM but still have reasonable
        # probability of becoming profitable if the stock makes a big move.
        min_delta: float = getattr(self.config, "min_delta", 0.15)
        max_delta: float = getattr(self.config, "max_delta", 0.45)
        # Hard cap on per-contract cost: $50. Keeps individual trade risk tiny.
        max_contract_cost: float = getattr(self.config, "max_contract_cost", 50.0)
        # Minimum relative volume to confirm the directional signal.
        min_rel_vol: float = getattr(self.config, "min_relative_volume", 1.5)
        # Lookback period for determining breakout/breakdown levels.
        breakout_lookback: int = getattr(self.config, "breakout_lookback", 20)

        # ------------------------------------------------------------------
        # 1. Stock filter -- need a strong directional signal
        # ------------------------------------------------------------------
        df = stock_bars.copy()
        if len(df) < breakout_lookback + 1:
            return None

        add_rsi(df)
        add_ema(df, periods=[20])
        # ATR (Average True Range): measures daily volatility in dollar terms.
        # A stock with ATR of $2.00 typically moves about $2 per day.
        # Used later to estimate potential option profit.
        add_atr(df)
        add_volume_profile(df)

        # Compute 20-day high and low for breakout/breakdown detection.
        # ``.shift(1)`` ensures we compare today's price to the PRIOR
        # 20-day range (excludes today from its own reference).
        df["high_20"] = df["high"].rolling(breakout_lookback).max().shift(1)
        df["low_20"] = df["low"].rolling(breakout_lookback).min().shift(1)

        latest = df.iloc[-1]
        rsi: float = latest.get("rsi_14", 50.0)
        price: float = latest["close"]
        ema_20: float = latest.get("ema_20", price)
        atr: float = latest.get("atr_14", 0.0)
        rel_vol: float = latest.get("relative_volume", 0.0)
        high_20: float = latest.get("high_20", 0.0)
        low_20: float = latest.get("low_20", 0.0)

        # Entry condition: Volume must be elevated to validate the move.
        if rel_vol < min_rel_vol:
            return None

        # --- Determine direction: is this a breakout or a breakdown? ---
        # Breakout (bullish): price above 20-day high + RSI > 50 + above EMA.
        # Breakdown (bearish): price below 20-day low + RSI < 50 + below EMA.
        # If neither condition is met, no trade (direction stays None).
        direction = None
        if price > high_20 and rsi > 50 and price > ema_20:
            direction = "call"  # Bullish breakout -> buy calls
        elif price < low_20 and rsi < 50 and price < ema_20:
            direction = "put"  # Bearish breakdown -> buy puts

        if direction is None:
            return None

        # ------------------------------------------------------------------
        # 2. Select contract -- cheap OTM options with 2-10 DTE
        # ------------------------------------------------------------------
        # ``filter_contracts`` narrows the chain to just calls or puts
        # (depending on direction) within our very short DTE window.
        eligible = filter_contracts(chain_data, direction, min_dte, max_dte)

        if not eligible:
            log.debug("momentum_options_skip", underlying=underlying, reason="no eligible contracts", direction=direction)
            return None

        # ------------------------------------------------------------------
        # 3. Enrich with greeks and filter by delta
        # ------------------------------------------------------------------
        # ``include_iv=True`` because we record IV in metadata for analysis.
        enrich_greeks(eligible, snapshots, include_iv=True)

        # Filter by delta range: not too far OTM (too cheap, low probability)
        # and not too close to ATM (too expensive for the strategy's budget).
        # ``fallback_min=0.10`` widens the search if nothing matches the
        # primary range.
        delta_ok = filter_by_delta(eligible, min_delta, max_delta, fallback_min=0.10)
        if not delta_ok:
            return None

        # ------------------------------------------------------------------
        # 4. Select the best contract -- cheapest within budget
        # ------------------------------------------------------------------
        # Sort candidates by ask price (ascending) to find the cheapest first.
        # ``.sort(key=lambda c: c["_ask"])`` sorts the list IN-PLACE using the
        # ``_ask`` value as the sort key. ``lambda c: c["_ask"]`` is an
        # anonymous function that extracts the ask price from each contract dict.
        delta_ok.sort(key=lambda c: c["_ask"])

        # Walk through sorted candidates and pick the first one within budget.
        selected = None
        for c in delta_ok:
            cost_per_contract = c["_ask"] * 100
            # ``0 < cost_per_contract`` ensures the option isn't free/stale.
            if 0 < cost_per_contract <= max_contract_cost:
                selected = c
                break

        if selected is None:
            # Small tolerance: if the cheapest option is up to 10% over budget,
            # still consider it. Prevents missing good setups over a few dollars
            # without blowing the risk budget.
            cheapest = delta_ok[0]
            if cheapest["_ask"] * 100 <= max_contract_cost * 1.10:
                selected = cheapest
            else:
                return None

        # ------------------------------------------------------------------
        # 5. Calculate potential ROI using ATR-based target
        # ------------------------------------------------------------------
        ask_price: float = selected["_ask"]
        cost_per_contract: float = ask_price * 100
        strike: float = selected["_strike"]
        # Use absolute delta for comparisons (puts have negative delta).
        delta: float = abs(selected["_delta"])

        # Estimate profit if the stock moves 1.5x its Average True Range
        # in our direction. ATR measures a "typical" daily move; 1.5x ATR
        # is a moderately strong move over a few days.
        #
        # For CALLS: if stock moves up by 1.5*ATR, the call's intrinsic
        # value at that point would be max(0, target_price - strike).
        # For PUTS: if stock moves down by 1.5*ATR, the put's intrinsic
        # value would be max(0, strike - target_price).
        if direction == "call":
            target_price = price + 1.5 * atr
            intrinsic_at_target = max(0, target_price - strike)
        else:
            target_price = price - 1.5 * atr
            intrinsic_at_target = max(0, strike - target_price)

        # Potential profit = what the option would be worth minus what we paid.
        potential_profit = intrinsic_at_target - ask_price
        # ROI = profit / cost. A potential_roi of 1.0 = 100% return.
        potential_roi = potential_profit / ask_price if ask_price > 0 else 0

        # Require at least 50% ROI potential. If the expected move wouldn't
        # produce sufficient return, the trade isn't worth the risk.
        min_roi: float = getattr(self.config, "min_roi_pct", 0.50)
        if potential_roi < min_roi:
            log.debug(
                "momentum_options_skip",
                underlying=underlying,
                reason="roi_too_low",
                potential_roi=potential_roi,
                min_roi=min_roi,
            )
            return None

        # ------------------------------------------------------------------
        # 6. Conviction scoring
        # ------------------------------------------------------------------
        conviction: float = 0.45
        # Higher volume = more conviction that the move is real.
        if rel_vol > 2.5:
            conviction += 0.15
        elif rel_vol > 2.0:
            conviction += 0.10
        # Higher potential ROI = more asymmetric payoff.
        if potential_roi > 2.0:
            conviction += 0.15
        elif potential_roi > 1.0:
            conviction += 0.10
        # Higher delta (closer to ATM) = more responsive to stock movement.
        if delta > 0.35:
            conviction += 0.10
        # Theta decay penalty: very short DTE options lose value fast.
        # With <7 DTE, the stock must move immediately or it's a loss.
        dte = selected.get("_dte", 0)
        if dte < 7:
            conviction -= 0.10
        conviction = max(0.40, min(0.85, conviction))

        # ------------------------------------------------------------------
        # 7. Build and return the trade signal
        # ------------------------------------------------------------------
        option_symbol: str = selected.get("symbol", "")
        expiration: str = selected.get("expiration_date") or selected.get("expiration", "")

        # Single leg: buy the OTM option (call or put depending on direction).
        legs = [
            {"symbol": option_symbol, "side": "buy", "qty": 1, "position_intent": "buy_to_open"},
        ]

        # Estimate max profit at 3x the entry cost. For calls, profit is
        # theoretically unlimited; for puts, it's capped at (strike * 100).
        # 3x is a practical target used for position sizing and reporting.
        max_profit_est = cost_per_contract * 3.0

        log.info(
            "momentum_options_signal",
            underlying=underlying,
            direction=direction,
            strike=strike,
            ask=ask_price,
            cost=cost_per_contract,
            delta=selected["_delta"],
            dte=selected.get("_dte", 0),
            potential_roi=round(potential_roi, 2),
            conviction=conviction,
            rel_vol=rel_vol,
            atr=atr,
            expiration=expiration,
        )

        # Use the appropriate strategy type based on direction.
        # This is a ternary expression: ``X if condition else Y`` (Python's
        # equivalent of the C-style ``condition ? X : Y``).
        strategy_type = OptionsStrategyType.LONG_CALL if direction == "call" else OptionsStrategyType.LONG_PUT

        return OptionsSignal(
            underlying=underlying,
            strategy_type=strategy_type,
            conviction=conviction,
            strategy_name="momentum_options",
            legs=legs,
            max_cost=ask_price,
            max_loss=cost_per_contract,
            max_profit=max_profit_est,
            expiration=expiration,
            strikes=[strike],
            net_delta=selected["_delta"],
            # Theta is very negative for short-dated options (rapid time decay).
            # The buyer is fighting the clock -- the move needs to happen fast.
            net_theta=selected["_theta"],
            metadata={
                "direction": direction,
                "rsi": rsi,
                "price": price,
                "atr": atr,
                "relative_volume": rel_vol,
                "potential_roi": potential_roi,
                "cost_per_contract": cost_per_contract,
                "dte": selected.get("_dte", 0),
                # IV (Implied Volatility) recorded for post-trade analysis.
                # High IV at entry means we paid more for the option, which
                # could be justified if the move is large enough.
                "iv": selected["_iv"],
            },
        )
