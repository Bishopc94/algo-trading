"""Covered Straddle -- aggressive income strategy for range-bound stocks.

Theory:
  Own 100 shares, then sell ATM call + ATM put for double premium.
  Requires extremely low-volatility, range-bound conditions.

  Multi-indicator confluence:
    - RSI 42-58 (tight neutral zone)
    - Price within 2% of EMA-20 (hugging the moving average)
    - EMA-20 within 2% of EMA-50 (flat trend, no directional bias)
    - Bollinger Band width < 8% (very low volatility)
    - MACD histogram near zero (flat momentum)
    - Volume normal or declining (no institutional activity)

  Max Profit: (Call premium + put premium) x 100.
  Max Loss: (Strike - total premium) x 100 (stock to zero + assignment).
"""

from __future__ import annotations

import pandas as pd

from ai_trade.data.indicators import add_bollinger, add_ema, add_macd, add_rsi, add_volume_profile
from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.options.base import (
    BaseOptionsStrategy,
    OptionsSignal,
    OptionsStrategyType,
    enrich_greeks,
    filter_contracts,
)

log = get_logger(__name__)


class CoveredStraddleStrategy(BaseOptionsStrategy):
    """Sell ATM call + ATM put against 100 shares with strict range-bound confirmation."""

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

        min_dte: int = getattr(self.config, "min_dte", 20)
        max_dte: int = getattr(self.config, "max_dte", 45)
        max_stock_price: float = getattr(self.config, "max_stock_price", 3.00)
        min_total_credit_pct: float = getattr(self.config, "min_total_credit_pct", 0.06)
        max_bb_width: float = getattr(self.config, "max_bb_width", 0.06)

        # ------------------------------------------------------------------
        # 1. Stock filter -- strict range-bound confirmation
        # ------------------------------------------------------------------
        df = stock_bars.copy()
        if len(df) < 52:
            return None

        add_rsi(df)
        add_ema(df, periods=[20, 50])
        add_bollinger(df)
        add_macd(df)
        add_volume_profile(df)

        latest = df.iloc[-1]
        rsi: float = latest.get("rsi_14", 50.0)
        price: float = latest["close"]
        ema_20: float = latest.get("ema_20", price)
        ema_50: float = latest.get("ema_50", price)
        bb_width: float = latest.get("bb_width", 0.0)
        macd_hist: float = latest.get("macd_hist", 0.0)
        rel_vol: float = latest.get("relative_volume", 1.0)

        # Tight neutral RSI (42-58)
        if not (42 <= rsi <= 58):
            return None

        # Price within 2% of EMA-20 (truly hugging the average)
        if abs(price - ema_20) / ema_20 > 0.02:
            return None

        # EMAs converged (flat trend, within 2%)
        if abs(ema_20 - ema_50) / ema_50 > 0.02:
            return None

        # Very tight Bollinger Bands (low volatility)
        if bb_width > max_bb_width:
            return None

        # MACD near zero (flat momentum)
        if abs(macd_hist) > 0.01 * price:
            return None

        # Stock affordable
        if price > max_stock_price:
            return None

        # No unusual volume (institutional activity would break the range)
        if rel_vol > 1.8:
            return None

        # ------------------------------------------------------------------
        # 2. Select ATM call + ATM put, same expiration
        # ------------------------------------------------------------------
        eligible_calls = filter_contracts(chain_data, "call", min_dte, max_dte)
        eligible_puts = filter_contracts(chain_data, "put", min_dte, max_dte)
        eligible = eligible_calls + eligible_puts

        if not eligible:
            return None

        enrich_greeks(eligible, snapshots)

        calls = [c for c in eligible if c.get("type", "").lower() == "call"]
        puts = [c for c in eligible if c.get("type", "").lower() == "put"]

        if not calls or not puts:
            return None

        # Find ATM strike
        atm_call = min(calls, key=lambda c: abs(c["_strike"] - price))
        atm_strike = atm_call["_strike"]
        call_exp = atm_call.get("expiration_date") or atm_call.get("expiration", "")

        # Matching put at same strike and expiration
        matching_puts = [
            c for c in puts
            if c["_strike"] == atm_strike
            and (c.get("expiration_date") or c.get("expiration", "")) == call_exp
        ]
        if not matching_puts:
            matching_puts = [
                c for c in puts
                if abs(c["_strike"] - atm_strike) <= 1.0
                and (c.get("expiration_date") or c.get("expiration", "")) == call_exp
            ]
        if not matching_puts:
            return None

        atm_put = matching_puts[0]

        # ------------------------------------------------------------------
        # 3. Pricing checks
        # ------------------------------------------------------------------
        call_premium: float = atm_call["_bid"]
        put_premium: float = atm_put["_bid"]
        total_premium: float = call_premium + put_premium

        if total_premium <= 0:
            return None

        credit_pct = total_premium / price
        if credit_pct < min_total_credit_pct:
            return None

        max_profit: float = total_premium * 100
        max_loss: float = (atm_strike - total_premium) * 100
        cash_required = atm_strike * 100

        # ------------------------------------------------------------------
        # 4. Conviction scoring (additive, 0.55-0.90)
        # ------------------------------------------------------------------
        conviction: float = 0.55

        # +0.08: Strong premium (>8% of stock price)
        if credit_pct > 0.10:
            conviction += 0.08
        elif credit_pct > 0.07:
            conviction += 0.05
        elif credit_pct > 0.06:
            conviction += 0.03

        # +0.07: Perfect neutral RSI (47-53)
        if 47 <= rsi <= 53:
            conviction += 0.07
        elif 45 <= rsi <= 55:
            conviction += 0.04

        # +0.07: Very tight Bollinger Bands
        if bb_width < 0.04:
            conviction += 0.07
        elif bb_width < 0.05:
            conviction += 0.04

        # +0.05: MACD dead flat
        if abs(macd_hist) < 0.003 * price:
            conviction += 0.05
        elif abs(macd_hist) < 0.005 * price:
            conviction += 0.03

        # +0.03: Low volume (quiet market)
        if rel_vol < 1.0:
            conviction += 0.03

        conviction = max(0.55, min(0.90, conviction))

        # ------------------------------------------------------------------
        # 5. Build and return the trade signal
        # ------------------------------------------------------------------
        call_symbol: str = atm_call.get("symbol", "")
        put_symbol: str = atm_put.get("symbol", "")
        expiration: str = call_exp

        legs = [
            {"symbol": call_symbol, "side": "sell", "qty": 1, "position_intent": "sell_to_open"},
            {"symbol": put_symbol, "side": "sell", "qty": 1, "position_intent": "sell_to_open"},
        ]

        net_delta = atm_call.get("_delta", 0.0) + atm_put.get("_delta", 0.0)
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
            bb_width=bb_width,
            macd_hist=macd_hist,
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
                "macd_hist": macd_hist,
                "relative_volume": rel_vol,
                "call_premium": call_premium,
                "put_premium": put_premium,
                "credit_pct": credit_pct,
                "cash_required": cash_required,
                "dte": atm_call.get("_dte", 0),
            },
        )
