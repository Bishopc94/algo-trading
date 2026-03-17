"""
Long Call -- high-leverage directional bet on momentum breakouts.

Strategy Theory
---------------
Buy a single call option when multiple bullish signals align:
  - Price breaks above its 20-day high (technical breakout)
  - Volume is at least 2x the 20-day average (institutional participation)
  - RSI is strong but not overbought (55-75)

This is the **most aggressive** directional strategy in the system. A single
call option provides leveraged exposure to the underlying stock's upside.

Payoff Profile
~~~~~~~~~~~~~~
- **Max Profit**: Theoretically unlimited. The call gains value as the stock
  rises above the strike price, with no cap.
- **Max Loss**: The premium (price) paid for the call x 100 shares per
  contract. If the stock doesn't rise above the strike by expiration, the
  call expires worthless and you lose the entire premium.
- **Breakeven**: Strike price + premium paid.

When It Works Best
~~~~~~~~~~~~~~~~~~
- High-conviction breakouts with volume confirmation.
- Low-to-moderate IV (implied volatility): buying options when IV is low
  means cheaper premiums. If IV is elevated, you're overpaying.
- Moderate DTE (30-60 days): enough time for the breakout to follow through,
  but not so much time that you overpay for time value.

Risk / Reward
~~~~~~~~~~~~~
- **Unlimited upside** with **defined risk** (the premium paid).
- However, options are a wasting asset -- theta decay works against you
  every day. If the stock doesn't move fast enough, you lose money even
  if the stock eventually goes in your direction.
- Position sizing should be small (the premium is 100% at risk).

Key Concepts in This File
--------------------------
- **Breakout**: Price exceeds the highest high of the last N days. Signals
  a potential trend continuation.
- **IV check**: If implied volatility is above a threshold, skip the trade.
  High IV means expensive options -- you'd need a bigger move to profit.
- **Delta selection (0.50-0.70)**: ATM to slightly ITM calls. Higher delta
  means the option moves more closely with the stock (more "stock-like").
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


class LongCallStrategy(BaseOptionsStrategy):
    """Enter a long call on a confirmed high-volume breakout.

    This strategy has the highest entry bar of all strategies -- it requires
    a price breakout above the 20-day high with 2x+ relative volume. These
    stringent filters keep the strategy from firing too frequently.
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
        # target_delta = 0.60: We want a call with ~60 delta (ATM to slightly
        # ITM). This gives strong directional exposure while keeping the
        # premium somewhat affordable.
        target_delta: float = getattr(self.config, "target_delta", 0.60)
        min_dte: int = getattr(self.config, "min_dte", 30)
        max_dte: int = getattr(self.config, "max_dte", 60)
        # max_contract_cost = $75: hard cap on per-contract cost (in dollars,
        # i.e. ask_price * 100). Keeps position sizing in check.
        max_contract_cost: float = getattr(self.config, "max_contract_cost", 75.0)
        # max_iv_percentile = 0.70: if the option's IV exceeds 70%, skip it.
        # Buying in high-IV environments means overpaying for the option.
        max_iv_percentile: float = getattr(self.config, "max_iv_percentile", 0.70)
        # Number of bars to look back for the breakout high.
        breakout_lookback: int = getattr(self.config, "breakout_lookback", 20)

        # ------------------------------------------------------------------
        # 1. Stock filter -- require very strong momentum
        # ------------------------------------------------------------------
        df = stock_bars.copy()
        if len(df) < breakout_lookback + 1:
            return None

        add_rsi(df)
        add_ema(df, periods=[20])
        add_volume_profile(df)

        # Compute the rolling 20-day high. ``.rolling(N).max()`` calculates
        # the maximum value over a sliding window of N bars.
        # ``.shift(1)`` shifts the result forward by one row, so we're comparing
        # today's price to *yesterday's* 20-day high (not including today).
        # This avoids a look-ahead bias where the breakout day is included
        # in its own reference range.
        df["high_20"] = df["high"].rolling(breakout_lookback).max().shift(1)

        latest = df.iloc[-1]
        rsi: float = latest.get("rsi_14", 0.0)
        price: float = latest["close"]
        rel_vol: float = latest.get("relative_volume", 0.0)
        high_20: float = latest.get("high_20", 0.0)

        # Entry condition 1: RSI in the 55-75 "strong momentum" range.
        # Below 55 = not enough momentum. Above 75 = overbought, the
        # breakout may be exhausted.
        if not (55 <= rsi <= 75):
            log.debug("long_call_skip", underlying=underlying, reason="RSI outside 55-75", rsi=rsi)
            return None
        # Entry condition 2: Price ABOVE the prior 20-day high = breakout.
        # This is the core signal: the stock is making new highs.
        if price <= high_20:
            log.debug("long_call_skip", underlying=underlying, reason="no breakout above 20d high")
            return None
        # Entry condition 3: Relative volume must be at least 2.0x average.
        # A breakout on low volume is unreliable ("fake breakout").
        if rel_vol <= 2.0:
            log.debug("long_call_skip", underlying=underlying, reason="relative_volume<=2.0")
            return None

        # ------------------------------------------------------------------
        # 2. Select contract -- ATM to slightly ITM call, 30-60 DTE
        # ------------------------------------------------------------------
        eligible_calls = filter_contracts(chain_data, "call", min_dte, max_dte)

        if not eligible_calls:
            log.debug("long_call_skip", underlying=underlying, reason="no eligible calls")
            return None

        # Enrich with snapshot data. ``include_iv=True`` because we need to
        # check implied volatility to avoid overpaying.
        enrich_greeks(eligible_calls, snapshots, include_iv=True)

        # Filter to calls with delta 0.50-0.70 (ATM to slightly ITM).
        # ``use_absolute=False`` because call deltas are already positive.
        delta_candidates = filter_by_delta(eligible_calls, 0.50, 0.70, use_absolute=False)
        if not delta_candidates:
            # Fallback: accept any call with positive delta.
            delta_candidates = [c for c in eligible_calls if c["_delta"] > 0]
        if not delta_candidates:
            return None

        # Pick the call closest to our target delta (0.60).
        selected = select_by_delta(delta_candidates, target_delta, use_absolute=False)

        # ------------------------------------------------------------------
        # 3. Pricing checks
        # ------------------------------------------------------------------
        # Use the ask price (what we'd actually pay to buy the option).
        ask_price: float = selected["_ask"]
        # Each option contract controls 100 shares, so the dollar cost is
        # ask_price * 100. For example, a $0.75 ask = $75 per contract.
        cost_per_contract: float = ask_price * 100
        # Max loss for a long call is the entire premium paid.
        max_loss: float = cost_per_contract

        # Reject if the contract is too expensive for our budget.
        if cost_per_contract > max_contract_cost:
            log.debug("long_call_skip", underlying=underlying, reason="cost too high", cost=cost_per_contract)
            return None

        # IV check: Implied Volatility is the market's forecast of future
        # stock movement. When IV is high, option premiums are inflated --
        # you're overpaying. Skip trades where IV exceeds our threshold.
        iv: float = selected["_iv"]
        if iv > max_iv_percentile and iv > 0:
            log.debug("long_call_skip", underlying=underlying, reason="IV too high", iv=iv)
            return None

        # ------------------------------------------------------------------
        # 4. Conviction scoring
        # ------------------------------------------------------------------
        conviction: float = 0.5
        # Exceptional volume (3x+ normal) is a very strong signal.
        if rel_vol > 3.0:
            conviction += 0.15
        # Higher delta = more stock-like behavior = more conviction.
        if selected["_delta"] > 0.60:
            conviction += 0.1
        # Cheaper contracts are less risky in absolute terms.
        if cost_per_contract < 50.0:
            conviction += 0.1
        conviction = max(0.5, min(1.0, conviction))

        # ------------------------------------------------------------------
        # 5. Build and return the trade signal
        # ------------------------------------------------------------------
        call_symbol: str = selected.get("symbol", "")
        expiration: str = selected.get("expiration_date") or selected.get("expiration", "")
        strike: float = selected["_strike"]

        # Single-leg trade: just one "buy" entry.
        legs = [
            {"symbol": call_symbol, "side": "buy", "qty": 1, "position_intent": "buy_to_open"},
        ]

        log.info(
            "long_call_signal",
            underlying=underlying,
            strike=strike,
            ask=ask_price,
            cost=cost_per_contract,
            delta=selected["_delta"],
            conviction=conviction,
            expiration=expiration,
            relative_volume=rel_vol,
        )

        return OptionsSignal(
            underlying=underlying,
            strategy_type=OptionsStrategyType.LONG_CALL,
            conviction=conviction,
            strategy_name="long_call",
            legs=legs,
            max_cost=ask_price,
            max_loss=max_loss,
            # max_profit=0.0 signals "theoretically unlimited" to downstream code.
            # A long call has no upper bound on profit.
            max_profit=0.0,  # Theoretically unlimited
            expiration=expiration,
            strikes=[strike],
            # For a single long call, net delta = the call's delta.
            net_delta=selected["_delta"],
            # Theta is negative for a long call (time decay hurts the buyer).
            net_theta=selected["_theta"],
            metadata={
                "rsi": rsi,
                "price": price,
                "relative_volume": rel_vol,
                "high_20": high_20,
                "iv": iv,
                "cost_per_contract": cost_per_contract,
                "dte": selected.get("_dte", 0),
            },
        )
