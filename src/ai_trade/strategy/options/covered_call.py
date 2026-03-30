"""
Covered Call -- income strategy on existing stock positions.

Strategy Theory
---------------
You already own 100 shares of the underlying stock. Sell an OTM (out-of-the-
money) call against those shares to collect premium. This is one of the most
common options strategies and is considered conservative.

Two possible outcomes at expiration:
  1. **Stock stays below the strike**: The call expires worthless, you keep the
     premium AND your shares. You can repeat the process next month.
  2. **Stock rises above the strike**: You are assigned -- your shares are
     "called away" (sold) at the strike price. You keep the premium plus any
     gain from the stock price to the strike. The opportunity cost is that
     you miss out on gains above the strike.

Payoff Profile
~~~~~~~~~~~~~~
- **Max Profit**: (Strike - current price + premium) x 100. Achieved when
  the stock is exactly at or above the strike at expiration.
- **Max Loss**: (Current price - premium) x 100 -- if the stock goes to zero.
  The covered call provides a small downside buffer (the premium collected)
  but does NOT protect against large drops.
- **Breakeven**: Current stock price - premium received.

When It Works Best
~~~~~~~~~~~~~~~~~~
- **Mildly bullish to neutral** outlook: you think the stock will stay flat
  or drift slightly higher.
- On positions you already hold and wouldn't mind selling at the strike.
- Low-to-moderate volatility stocks (stable names, not pre-earnings).

Risk / Reward
~~~~~~~~~~~~~
- **Income generation**: Collecting premium month after month adds to returns.
- **Capped upside**: If the stock rallies hard, you miss the upside above the
  strike. This is the main cost of the strategy.
- **Annualized return**: Selling one call per month can add 10-20%+ annualized
  return on top of dividends and stock appreciation.

Key Concepts in This File
--------------------------
- **Annualized return**: ``(premium / stock_price) x (365 / DTE)``. Normalizes
  the return as if you could repeat this trade every DTE days for a year.
- **Delta ~0.30**: An OTM call with 30 delta has roughly a 30% chance of being
  called away. This is the sweet spot -- enough premium to be worthwhile, but
  a good probability of keeping your shares.
- **Don't sell calls when the stock is running**: If the stock is >5% above
  EMA-20, it's in a strong rally and selling calls would cap attractive upside.

Note: This file was refactored to use the shared ``filter_contracts`` /
``enrich_greeks`` / ``select_by_delta`` utilities from base.py, ensuring
consistent greeks extraction across all strategies.
"""

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


class CoveredCallStrategy(BaseOptionsStrategy):
    """Sell an OTM call against 100 shares already held.

    Precondition: the caller should only invoke this strategy for tickers
    where the account already holds >= 100 shares.
    """

    bias = "neutral"

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
        # target_delta = 0.30: Sell a call with ~30% probability of being
        # exercised. This balances premium income vs. probability of keeping shares.
        target_delta: float = getattr(self.config, "target_delta", 0.30)
        min_dte: int = getattr(self.config, "min_dte", 20)
        max_dte: int = getattr(self.config, "max_dte", 45)
        # Minimum annualized return from premium. 12% ensures the trade is
        # worth the effort and the capped upside risk.
        min_annualized_return: float = getattr(self.config, "min_annualized_return", 0.12)
        # Max stock price: relevant for position sizing (you need 100 shares).
        max_stock_price: float = getattr(self.config, "max_stock_price", 5.0)

        # ------------------------------------------------------------------
        # Stock filter -- mildly bullish to neutral outlook required
        # ------------------------------------------------------------------
        df = stock_bars.copy()
        if len(df) < 21:
            return None

        add_rsi(df)
        add_ema(df, periods=[20, 50])

        latest = df.iloc[-1]
        rsi: float = latest.get("rsi_14", 50.0)
        price: float = latest["close"]
        ema_20: float = latest.get("ema_20", price)

        # Entry condition 1: RSI 45-70 (neutral to mildly bullish).
        # Below 45 = bearish, selling calls doesn't help if stock is falling.
        # Above 70 = stock might rally hard, selling calls caps attractive upside.
        if not (45 <= rsi <= 70):
            return None
        # Entry condition 2: Don't sell calls when the stock is running.
        # If price is >5% above EMA-20, the stock is in a strong rally and we'd
        # be selling off potential upside too cheaply.
        if price > ema_20 * 1.05:  # Don't sell calls when stock is running hard
            return None
        # Entry condition 3: Stock must be affordable to hold 100 shares of.
        if price > max_stock_price:
            return None

        # ------------------------------------------------------------------
        # Select contract -- OTM call, 20-45 DTE
        # ------------------------------------------------------------------
        eligible_calls = filter_contracts(chain_data, "call", min_dte, max_dte)
        if not eligible_calls:
            return None

        enrich_greeks(eligible_calls, snapshots)

        # OTM calls only: strike above current price, delta 0.20-0.40.
        delta_candidates = [
            c for c in eligible_calls
            if c["_strike"] > price and 0.20 <= abs(c["_delta"]) <= 0.40
        ]
        if not delta_candidates:
            delta_candidates = [
                c for c in eligible_calls
                if c["_strike"] > price and abs(c["_delta"]) > 0
            ]
        if not delta_candidates:
            return None

        selected = select_by_delta(delta_candidates, target_delta)
        if selected is None:
            return None

        strike: float = selected["_strike"]
        dte: int = selected["_dte"]
        # Use bid since we're selling. Bid is the most a buyer will pay us.
        premium: float = selected["_bid"]

        if premium <= 0 or strike <= 0 or dte <= 0:
            return None

        # Annualized return: normalize the premium income as a percentage of
        # the stock price, annualized to compare across different DTE periods.
        # Formula: (premium / stock_price) * (365 / DTE)
        annualized_return: float = (premium / price) * (365.0 / dte)
        if annualized_return < min_annualized_return:
            return None

        # Max profit = premium + any stock appreciation up to the strike.
        # If assigned, we sell shares at strike and keep the premium.
        # Example: stock at $4.50, strike $5.00, premium $0.20
        #   max_profit = ($0.20 + $0.50) * 100 = $70
        max_profit: float = (premium + (strike - price)) * 100
        # Max loss = stock goes to zero. We lose the stock value but keep premium.
        max_loss: float = (price - premium) * 100

        # ------------------------------------------------------------------
        # Conviction scoring
        # ------------------------------------------------------------------
        # Base conviction lowered to 0.50 — covered calls tie up 100 shares,
        # so on a $500 account they use most of the capital.
        conviction: float = 0.50
        if annualized_return > 0.20:
            conviction += 0.10
        # RSI in the sweet spot (45-55): truly neutral, ideal for covered calls.
        if 45 <= rsi <= 55:
            conviction += 0.10
        # Theta < 0 means the call is decaying — good for us as the seller.
        if selected["_theta"] < 0:
            conviction += 0.05
        conviction = max(0.45, min(0.80, conviction))

        call_symbol: str = selected.get("symbol", "")
        expiration: str = selected.get("expiration_date") or selected.get("expiration", "")

        # Single leg: sell the call.
        legs = [
            {"symbol": call_symbol, "side": "sell", "qty": 1, "position_intent": "sell_to_open"},
        ]

        log.info(
            "covered_call_signal",
            underlying=underlying,
            strike=strike,
            premium=premium,
            annualized_return=annualized_return,
            conviction=conviction,
            expiration=expiration,
        )

        return OptionsSignal(
            underlying=underlying,
            strategy_type=OptionsStrategyType.COVERED_CALL,
            conviction=conviction,
            strategy_name="covered_call",
            legs=legs,
            min_credit=premium,
            max_loss=max_loss,
            max_profit=max_profit,
            expiration=expiration,
            strikes=[strike],
            # Net delta is negative because we sold a call. The short call's
            # positive delta (from the option) becomes negative exposure for us.
            # Combined with the +1.00 delta of owning 100 shares, the total
            # position delta is roughly ``1.00 - call_delta`` (e.g. 0.70 if
            # call delta was 0.30). Only the option delta is reported here.
            net_delta=-selected["_delta"],
            net_theta=selected["_theta"],
            metadata={
                "rsi": rsi,
                "price": price,
                "ema_20": ema_20,
                "annualized_return": annualized_return,
                "dte": dte,
            },
        )
