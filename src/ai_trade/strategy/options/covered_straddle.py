"""
Covered Straddle -- aggressive income strategy for range-bound stocks.

Strategy Theory
---------------
You already own 100 shares of the underlying. Simultaneously sell:
  1. An **ATM call** (at-the-money) -- collects call premium, caps upside.
  2. An **ATM put** (at-the-money) -- collects put premium, obligates you to
     buy 100 MORE shares if the stock drops below the strike.

This is essentially a covered call PLUS a cash-secured put on the same stock,
combined into a single strategy. You collect **double premium** compared to
either strategy alone.

Payoff Profile
~~~~~~~~~~~~~~
- **Max Profit**: Total premium (call + put) x 100. Achieved when the stock
  closes exactly at the strike at expiration (both options expire worthless).
- **Max Loss**: If the stock goes to zero, you lose the value of 200 shares
  (100 originally held + 100 from put assignment) minus total premium.
  Simplified: ``(strike - total_premium) x 100`` (for one contract lot).
- **Breakeven (upside)**: Strike + total premium (shares called away but you
  keep all premium).
- **Breakeven (downside)**: Strike - total premium (assigned more shares but
  premium offsets the loss).

When It Works Best
~~~~~~~~~~~~~~~~~~
- **Range-bound, low-volatility** stocks. The ideal outcome is the stock
  sitting still near the strike through expiration.
- RSI 40-60 (neutral zone).
- Tight Bollinger Bands (low recent volatility), measured by ``bb_width``.
- On stocks you're comfortable accumulating (you may end up owning 200 shares).

Risk / Reward
~~~~~~~~~~~~~
- **High income**: Double the premium of a simple covered call.
- **High risk**: Naked put exposure means you could be forced to buy 100 more
  shares at the strike during a downturn. You need the cash (or margin) to
  absorb this assignment.
- **Best for small-dollar stocks**: With a $500 account, you can only do this
  on stocks under $5 (need cash to secure the put side).

Key Concepts in This File
--------------------------
- **ATM (At-The-Money)**: Options where the strike is closest to the current
  stock price. These have the highest time value (and thus the most premium
  to collect) but are also the most likely to be exercised.
- **Bollinger Band Width (bb_width)**: Measures how wide the Bollinger Bands
  are, normalized by the middle band. A small bb_width (<0.10) means the stock
  has been trading in a tight range -- ideal for selling straddles.
- **credit_pct**: ``total_premium / stock_price``. Represents the premium
  collected as a percentage of the stock price. Higher = better income.

Note: Like covered_call.py, this file does its own inline contract filtering
and enrichment rather than using the shared ``filter_contracts``/``enrich_greeks``
utilities from base.py.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ai_trade.data.indicators import add_ema, add_rsi, add_bollinger
from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.options.base import (
    BaseOptionsStrategy,
    OptionsSignal,
    OptionsStrategyType,
)

log = get_logger(__name__)


class CoveredStraddleStrategy(BaseOptionsStrategy):
    """Sell an ATM call + ATM put against 100 shares held.

    Requires range-bound conditions: neutral RSI, price near EMA-20,
    and tight Bollinger Bands. This is the most income-heavy but also
    the riskiest of the income strategies.
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
        min_dte: int = getattr(self.config, "min_dte", 20)
        max_dte: int = getattr(self.config, "max_dte", 45)
        # Max stock price: we need cash for both the shares AND the put assignment.
        max_stock_price: float = getattr(self.config, "max_stock_price", 5.0)
        # Minimum combined credit as a fraction of stock price. 4% means:
        # on a $5 stock, we want at least $0.20 in total premium.
        min_total_credit_pct: float = getattr(self.config, "min_total_credit_pct", 0.04)
        # Maximum Bollinger Band width. A narrow band (< 10%) signals low
        # volatility = ideal for selling straddles.
        max_bb_width: float = getattr(self.config, "max_bb_width", 0.10)

        # ------------------------------------------------------------------
        # Stock filter -- must be low-volatility and range-bound
        # ------------------------------------------------------------------
        df = stock_bars.copy()
        if len(df) < 21:
            return None

        add_rsi(df)
        add_ema(df, periods=[20])
        # ``add_bollinger(df)`` computes Bollinger Bands: an upper band, lower
        # band, and width indicator. Bollinger Bands are a volatility measure --
        # they widen when the stock is volatile and narrow when it's calm.
        add_bollinger(df)

        latest = df.iloc[-1]
        rsi: float = latest.get("rsi_14", 50.0)
        price: float = latest["close"]
        ema_20: float = latest.get("ema_20", price)
        # bb_width: Bollinger Band width as a fraction of the middle band.
        # Small values (e.g. 0.05) mean the stock is in a tight range.
        bb_width: float = latest.get("bb_width", 0.0)

        # Entry condition 1: Neutral RSI (40-60). Not trending in either direction.
        if not (40 <= rsi <= 60):
            return None
        # Entry condition 2: Price within 3% of EMA-20. The stock is hugging
        # its moving average, not trending away from it.
        # ``abs(price - ema_20) / ema_20`` computes the percentage distance
        # from the EMA. abs() ensures we catch both above and below.
        if abs(price - ema_20) / ema_20 > 0.03:
            return None
        # Entry condition 3: Tight Bollinger Bands = low recent volatility.
        # This is crucial -- selling straddles in high-volatility environments
        # is very risky (the stock could make a big move).
        if bb_width > max_bb_width:
            return None
        # Entry condition 4: Stock must be affordable (need cash for put side).
        if price > max_stock_price:
            return None

        # ------------------------------------------------------------------
        # Select contracts -- ATM call + ATM put, same expiration
        # ------------------------------------------------------------------
        # Inline contract filtering (same approach as covered_call.py).
        now = datetime.now(tz=timezone.utc)
        eligible: list[dict] = []
        for contract in chain_data:
            exp_str = contract.get("expiration_date") or contract.get("expiration", "")
            try:
                exp_dt = datetime.fromisoformat(exp_str).replace(tzinfo=timezone.utc) if exp_str else None
            except (ValueError, TypeError):
                continue
            if exp_dt is None:
                continue
            dte = (exp_dt - now).days
            if min_dte <= dte <= max_dte:
                contract["_dte"] = dte
                eligible.append(contract)

        if not eligible:
            return None

        # Enrich each eligible contract with greeks and pricing.
        for c in eligible:
            sym = c.get("symbol", "")
            snap = snapshots.get(sym, {})
            c["_delta"] = snap.get("delta", 0.0)
            c["_theta"] = snap.get("theta", 0.0)
            c["_bid"] = snap.get("bid", 0.0) or 0.0
            c["_strike"] = float(c.get("strike_price") or c.get("strike", 0))

        # Separate calls and puts from the eligible contracts.
        # These list comprehensions filter by the "type" field.
        calls = [c for c in eligible if c.get("type", "").lower() == "call"]
        puts = [c for c in eligible if c.get("type", "").lower() == "put"]

        if not calls or not puts:
            return None

        # --- Find the ATM (At-The-Money) strike ---
        # The ATM strike is the one closest to the current stock price.
        # ``min(calls, key=lambda c: abs(c["_strike"] - price))`` returns the
        # call contract whose strike is nearest to ``price``. The ``lambda``
        # computes the absolute distance between each strike and the price;
        # ``min`` picks the smallest distance.
        atm_call = min(calls, key=lambda c: abs(c["_strike"] - price))
        atm_strike = atm_call["_strike"]

        # --- Find a matching put at the same strike and expiration ---
        # For a proper straddle, both options must share the same strike
        # and expiration date.
        call_exp = atm_call.get("expiration_date") or atm_call.get("expiration", "")
        matching_puts = [
            c for c in puts
            if c["_strike"] == atm_strike
            and (c.get("expiration_date") or c.get("expiration", "")) == call_exp
        ]
        if not matching_puts:
            # Fallback: try a put within $1 of the ATM strike (in case exact
            # strike matching fails due to rounding or different available strikes).
            matching_puts = [
                c for c in puts
                if abs(c["_strike"] - atm_strike) <= 1.0
                and (c.get("expiration_date") or c.get("expiration", "")) == call_exp
            ]
        if not matching_puts:
            return None

        # Take the first matching put (if multiple, they should be identical
        # or very close in strike).
        atm_put = matching_puts[0]

        # ------------------------------------------------------------------
        # Pricing -- combined premium from both legs
        # ------------------------------------------------------------------
        # Use bid prices for both legs since we're SELLING both options.
        call_premium: float = atm_call["_bid"]
        put_premium: float = atm_put["_bid"]
        # Total premium = income from selling both the call and the put.
        total_premium: float = call_premium + put_premium

        if total_premium <= 0:
            return None

        # Credit as a percentage of the stock price. Measures income relative
        # to the underlying's value.
        credit_pct = total_premium / price
        if credit_pct < min_total_credit_pct:
            return None

        # Max profit: if both options expire worthless (stock at the strike),
        # we keep the entire combined premium.
        max_profit: float = total_premium * 100
        # Max loss (simplified): stock goes to zero. We lose the stock value
        # PLUS we're assigned on the put (buy 100 more shares at strike).
        # Simplified formula: ``(strike - total_premium) x 100`` represents
        # the net cost of one lot of the put side.
        max_loss: float = (atm_strike - total_premium) * 100

        # Cash needed: we already hold 100 shares (for the covered call side).
        # We also need cash to secure the put (strike * 100) in case of assignment.
        cash_required = atm_strike * 100

        # ------------------------------------------------------------------
        # Conviction scoring
        # ------------------------------------------------------------------
        conviction: float = 0.55
        # Strong premium (>6% of stock price) is very attractive for income.
        if credit_pct > 0.06:
            conviction += 0.15
        # RSI right in the middle (45-55) = perfect neutrality.
        if 45 <= rsi <= 55:
            conviction += 0.1
        # Very tight Bollinger Bands (<6%) = the stock is barely moving.
        # This is ideal for a straddle seller who wants minimal stock movement.
        if bb_width < 0.06:
            conviction += 0.1
        conviction = max(0.5, min(1.0, conviction))

        call_symbol: str = atm_call.get("symbol", "")
        put_symbol: str = atm_put.get("symbol", "")
        expiration: str = call_exp

        # Two legs: sell the ATM call and sell the ATM put.
        legs = [
            {"symbol": call_symbol, "side": "sell", "qty": 1, "position_intent": "sell_to_open"},
            {"symbol": put_symbol, "side": "sell", "qty": 1, "position_intent": "sell_to_open"},
        ]

        # Net delta: call delta (positive) + put delta (negative).
        # For ATM options, these are roughly +0.50 and -0.50, so net delta
        # is approximately zero (market-neutral). However, the 100 shares
        # held add +1.00 delta, making the total position slightly bullish.
        # Only the option deltas are reported here (share delta handled elsewhere).
        net_delta = atm_call.get("_delta", 0.0) + atm_put.get("_delta", 0.0)
        # Net theta: both short options have positive theta for the seller
        # (time decay works in our favor on both legs). This is doubled
        # compared to a simple covered call.
        net_theta = atm_call.get("_theta", 0.0) + atm_put.get("_theta", 0.0)

        log.info(
            "covered_straddle_signal",
            underlying=underlying,
            strike=atm_strike,
            call_premium=call_premium,
            put_premium=put_premium,
            total_credit=total_premium,
            credit_pct=credit_pct,
            conviction=conviction,
            expiration=expiration,
        )

        return OptionsSignal(
            underlying=underlying,
            strategy_type=OptionsStrategyType.COVERED_STRADDLE,
            conviction=conviction,
            strategy_name="covered_straddle",
            legs=legs,
            min_credit=total_premium,
            max_loss=max_loss,
            max_profit=max_profit,
            expiration=expiration,
            strikes=[atm_strike],
            net_delta=net_delta,
            net_theta=net_theta,
            metadata={
                "rsi": rsi,
                "price": price,
                "bb_width": bb_width,
                "call_premium": call_premium,
                "put_premium": put_premium,
                "credit_pct": credit_pct,
                "cash_required": cash_required,
                "dte": atm_call.get("_dte", 0),
            },
        )
