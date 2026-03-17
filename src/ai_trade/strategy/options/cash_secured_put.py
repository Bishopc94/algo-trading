"""
Cash-Secured Put (CSP) -- income / acquisition strategy.

Strategy Theory
---------------
Sell an OTM (out-of-the-money) put and collect the premium. You must hold
enough cash to buy 100 shares at the strike price if assigned ("cash-secured").

Two possible outcomes at expiration:
  1. **Stock stays above the strike**: The put expires worthless and you keep
     the entire premium as profit. This is the desired outcome.
  2. **Stock drops below the strike**: You are assigned (obligated to buy)
     100 shares at the strike price. Your effective cost basis is
     ``strike - premium``, which is below where the stock was when you
     opened the trade. If you wanted to own the stock anyway, this is
     buying at a discount.

Payoff Profile
~~~~~~~~~~~~~~
- **Max Profit**: Premium received x 100 shares per contract.
- **Max Loss**: (Strike - premium) x 100 -- if the stock goes to zero, you
  own shares that are worthless but you keep the premium.
- **Breakeven**: Strike price - premium received.

When It Works Best
~~~~~~~~~~~~~~~~~~
- On stocks you genuinely want to own at a lower price.
- RSI 35-55 (neutral territory, not in free-fall).
- Price near a support level (EMA-50 or below).
- Moderate to high IV (better premiums when selling options).

Risk / Reward
~~~~~~~~~~~~~
- **Large downside risk**: if the stock drops significantly, you're forced
  to buy at the strike (though you keep the premium).
- **Capped upside**: you can never make more than the premium collected.
- **Cash-heavy**: you need ``strike x 100`` in cash set aside (or margin).

Key Concepts in This File
--------------------------
- **Annualized return**: ``(premium / strike) x (365 / DTE)``. Normalizes
  returns across different expirations so you can compare a 30-DTE trade
  paying $0.50 on a $10 strike vs. a 45-DTE trade paying $0.70 on $15.
  A minimum annualized return of 15% ensures the premium is worth the
  capital lockup.
- **OTM put selection**: We target delta ~0.25 (roughly 25% probability of
  being assigned). Further OTM = less premium but less assignment risk.
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


class CashSecuredPutStrategy(BaseOptionsStrategy):
    """Sell a cash-secured put on stocks near support that you'd want to own.

    This strategy is conservative: it targets low-delta OTM puts on stocks
    in neutral RSI territory near their 50-day moving average.
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
        # target_delta = 0.25: Sell a put with ~25% probability of assignment.
        # This is fairly conservative -- the stock needs to drop ~5-10% before
        # you'd be assigned. The tradeoff is less premium collected.
        target_delta: float = getattr(self.config, "target_delta", 0.25)
        min_dte: int = getattr(self.config, "min_dte", 20)
        max_dte: int = getattr(self.config, "max_dte", 45)
        # Minimum annualized return we'll accept. 15% means: if we could
        # repeat this trade every DTE days for a year, we'd earn at least 15%.
        min_annualized_return: float = getattr(self.config, "min_annualized_return", 0.15)
        # Max stock price: CSPs require ``strike x 100`` in cash. With $500,
        # you can only secure puts on stocks under $5.
        max_stock_price: float = getattr(self.config, "max_stock_price", 5.0)
        available_capital: float = getattr(self.config, "available_capital", 500.0)

        # ------------------------------------------------------------------
        # 1. Stock filter -- neutral zone, near support
        # ------------------------------------------------------------------
        df = stock_bars.copy()
        # Need 51 bars to compute a 50-period EMA.
        if len(df) < 51:
            return None

        add_rsi(df)
        # EMA-50 is used as a support/trend reference. Stocks near or below
        # EMA-50 are in a "value" zone where selling puts is attractive.
        add_ema(df, periods=[50])

        latest = df.iloc[-1]
        rsi: float = latest.get("rsi_14", 0.0)
        ema_50: float = latest.get("ema_50", 0.0)
        price: float = latest["close"]

        # Entry condition 1: RSI between 35-55. This "neutral" zone means
        # the stock isn't in free-fall (don't sell puts into a crash) and
        # isn't overbought (less likely to pull back to our strike).
        if not (35 <= rsi <= 55):
            log.debug("cash_secured_put_skip", underlying=underlying, reason="RSI outside 35-55", rsi=rsi)
            return None

        # Entry condition 2: Price near or below EMA-50 (within 5% above).
        # This means the stock is at or below its medium-term average --
        # a "value" area where buying (via put assignment) would be acceptable.
        if price > ema_50 * 1.05:
            log.debug("cash_secured_put_skip", underlying=underlying, reason="price too far above EMA50")
            return None

        # Entry condition 3: Stock price must be low enough that we can
        # afford to buy 100 shares if assigned. This is the "cash-secured"
        # requirement. For a $5 stock, you need $500 set aside.
        if price > max_stock_price:
            log.debug("cash_secured_put_skip", underlying=underlying, reason="price too high for account")
            return None

        # ------------------------------------------------------------------
        # 2. Select contract -- OTM put with 20-45 DTE
        # ------------------------------------------------------------------
        eligible_puts = filter_contracts(chain_data, "put", min_dte, max_dte)

        if not eligible_puts:
            log.debug("cash_secured_put_skip", underlying=underlying, reason="no eligible puts")
            return None

        enrich_greeks(eligible_puts, snapshots)

        # Filter to OTM puts only (strike below current price).
        # An OTM put has a strike below the stock price -- the stock needs
        # to drop before the put has intrinsic value.
        otm_puts = [c for c in eligible_puts if c["_strike"] < price]
        # Further filter to delta range 0.20-0.35 (absolute value for puts).
        delta_candidates = [
            c for c in otm_puts if 0.20 <= abs(c["_delta"]) <= 0.35
        ]
        if not delta_candidates:
            # Fallback: any OTM put with non-zero delta.
            delta_candidates = [c for c in otm_puts if c["_delta"] != 0]
        if not delta_candidates:
            return None

        # Pick the put closest to our target delta (0.25).
        selected = select_by_delta(delta_candidates, target_delta)

        strike: float = selected["_strike"]
        dte: int = selected["_dte"]

        # ------------------------------------------------------------------
        # 3. Pricing checks
        # ------------------------------------------------------------------
        # Use the BID price since we're SELLING the put. The bid is what
        # someone will pay us for it (worst case for the seller).
        premium: float = selected["_bid"]
        # Cash required to secure the put = strike * 100 shares.
        # This is the capital that must be set aside in case of assignment.
        cash_required: float = strike * 100

        if cash_required > available_capital:
            log.debug(
                "cash_secured_put_skip",
                underlying=underlying,
                reason="insufficient capital",
                cash_required=cash_required,
            )
            return None

        if premium <= 0 or strike <= 0 or dte <= 0:
            return None

        # Annualized return calculation:
        # ``(premium / strike)`` = return per cycle as a fraction of capital at risk.
        # ``(365.0 / dte)`` = how many times you could repeat this trade per year.
        # Example: $0.30 premium on $10 strike, 30 DTE:
        #   (0.30 / 10) * (365 / 30) = 0.03 * 12.17 = 0.365 = 36.5% annualized.
        annualized_return: float = (premium / strike) * (365.0 / dte)
        if annualized_return < min_annualized_return:
            log.debug(
                "cash_secured_put_skip",
                underlying=underlying,
                reason="annualized return too low",
                annualized_return=annualized_return,
            )
            return None

        # Max loss: if the stock goes to zero, you own worthless shares.
        # Your loss is the strike price minus the premium you collected,
        # times 100 shares per contract.
        max_loss: float = (strike - premium) * 100  # Stock goes to zero
        # Max profit: the full premium collected * 100.
        max_profit: float = premium * 100

        # ------------------------------------------------------------------
        # 4. Conviction scoring
        # ------------------------------------------------------------------
        conviction: float = 0.6
        # Higher annualized return = more compelling trade.
        if annualized_return > 0.25:
            conviction += 0.1
        # RSI in the 40-50 sweet spot = ideal "neutral" zone.
        if 40 <= rsi <= 50:
            conviction += 0.1
        # Positive theta on the short put means time decay works in our
        # favor (the put loses value over time, which is what we want as
        # the seller).
        if selected["_theta"] > 0:
            conviction += 0.1
        conviction = max(0.5, min(1.0, conviction))

        # ------------------------------------------------------------------
        # 5. Build and return the trade signal
        # ------------------------------------------------------------------
        put_symbol: str = selected.get("symbol", "")
        expiration: str = selected.get("expiration_date") or selected.get("expiration", "")

        # Single leg: sell the put.
        legs = [
            {"symbol": put_symbol, "side": "sell", "qty": 1, "position_intent": "sell_to_open"},
        ]

        log.info(
            "cash_secured_put_signal",
            underlying=underlying,
            strike=strike,
            premium=premium,
            cash_required=cash_required,
            annualized_return=annualized_return,
            conviction=conviction,
            expiration=expiration,
        )

        return OptionsSignal(
            underlying=underlying,
            strategy_type=OptionsStrategyType.CASH_SECURED_PUT,
            conviction=conviction,
            strategy_name="cash_secured_put",
            legs=legs,
            min_credit=premium,
            max_loss=max_loss,
            max_profit=max_profit,
            expiration=expiration,
            strikes=[strike],
            # Short put delta is negative; the position profits when the stock
            # stays flat or rises (delta exposure is net short, i.e. slightly
            # bullish for the put seller).
            net_delta=selected["_delta"],
            net_theta=selected["_theta"],
            metadata={
                "rsi": rsi,
                "price": price,
                "ema_50": ema_50,
                "annualized_return": annualized_return,
                "cash_required": cash_required,
                "dte": dte,
            },
        )
