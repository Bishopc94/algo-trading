"""Covered Call -- income strategy on existing stock positions.

Theory:
  Sell an OTM call against 100 shares already held. Collect premium in
  exchange for capping upside if the stock rallies past the strike.

  We add multi-indicator confluence:
    - RSI 40-65 (neutral to mildly bullish, not trending hard)
    - Price within 5% of EMA-20 (not running away from MA)
    - EMA-20 >= EMA-50 (trend structure intact, not in downtrend)
    - MACD histogram not strongly positive (stock not accelerating up)
    - Bollinger Band width < 15% (low volatility, range-bound)
    - Volume normal or declining (no institutional breakout in progress)

  Max Profit: (Strike - price + premium) x 100.
  Max Loss: (Price - premium) x 100 (stock goes to zero).
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
    select_by_delta,
)

log = get_logger(__name__)


class CoveredCallStrategy(BaseOptionsStrategy):
    """Sell an OTM call against 100 shares with multi-indicator range-bound confirmation."""

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

        target_delta: float = getattr(self.config, "target_delta", 0.30)
        min_dte: int = getattr(self.config, "min_dte", 20)
        max_dte: int = getattr(self.config, "max_dte", 45)
        min_annualized_return: float = getattr(self.config, "min_annualized_return", 0.12)
        max_stock_price: float = getattr(self.config, "max_stock_price", 3.00)

        # ------------------------------------------------------------------
        # 1. Stock filter -- neutral/range-bound confirmation
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

        # RSI 35-70 (widened: neutral to bullish)
        if rsi < 35:
            self._reject(underlying, "rsi_min", rsi, 35.0, "above")
            return None
        if rsi > 70:
            self._reject(underlying, "rsi_max", rsi, 70.0, "below")
            return None

        # Price within 8% of EMA-20 (not running away — relaxed from 5%)
        ema20_ceil = ema_20 * 1.08
        if price > ema20_ceil:
            self._reject(underlying, "price_near_ema20", price, ema20_ceil, "below")
            return None

        # Stock affordable for 100-share position ($5 = $500 collateral)
        if price > max_stock_price:
            self._reject(underlying, "max_stock_price", price, max_stock_price, "below")
            return None

        # EMA structure: not in deep downtrend (relaxed from 3% to 5%)
        ema_struct_floor = ema_50 * 0.95
        if ema_20 < ema_struct_floor:
            self._reject(underlying, "ema_structure", ema_20, ema_struct_floor, "above")
            return None

        # Don't sell calls during strong upward momentum
        macd_ceil = 0.02 * price
        if macd_hist > macd_ceil:
            self._reject(underlying, "macd_not_strong_up", macd_hist, macd_ceil, "below")
            return None

        # Bollinger Band width < 20% (relaxed from 15%)
        if bb_width > 0.20:
            self._reject(underlying, "bb_width_max", bb_width, 0.20, "below")
            return None

        # Don't sell calls on high-volume breakout days
        if rel_vol > 2.5 and price > ema_20:
            self._reject(underlying, "breakout_vol_spike", rel_vol, 2.5, "below")
            return None

        # ------------------------------------------------------------------
        # 2. Select contract -- OTM call
        # ------------------------------------------------------------------
        eligible_calls = filter_contracts(chain_data, "call", min_dte, max_dte)
        if not eligible_calls:
            return None

        enrich_greeks(eligible_calls, snapshots)

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
        premium: float = selected["_bid"]

        if premium <= 0 or strike <= 0 or dte <= 0:
            return None

        annualized_return: float = (premium / price) * (365.0 / dte)
        if annualized_return < min_annualized_return:
            return None

        max_profit: float = (premium + (strike - price)) * 100
        max_loss: float = (price - premium) * 100

        # ------------------------------------------------------------------
        # 3. Conviction scoring (additive, 0.50-0.85)
        # ------------------------------------------------------------------
        conviction: float = 0.50

        # +0.08: Strong annualized return
        if annualized_return > 0.25:
            conviction += 0.08
        elif annualized_return > 0.15:
            conviction += 0.05

        # +0.07: RSI ideal zone (45-55, truly neutral)
        if 45 <= rsi <= 55:
            conviction += 0.07
        elif 40 <= rsi <= 60:
            conviction += 0.04

        # +0.06: Tight Bollinger Bands (low volatility)
        if bb_width < 0.06:
            conviction += 0.06
        elif bb_width < 0.10:
            conviction += 0.03

        # +0.05: MACD near zero (flat momentum)
        if abs(macd_hist) < 0.005 * price:
            conviction += 0.05

        # +0.04: EMA structure intact
        if ema_20 >= ema_50:
            conviction += 0.04

        # +0.03: Theta decay benefits us
        if selected["_theta"] < 0:
            conviction += 0.03

        conviction = max(0.50, min(0.85, conviction))

        # ------------------------------------------------------------------
        # 4. Build and return the trade signal
        # ------------------------------------------------------------------
        call_symbol: str = selected.get("symbol", "")
        expiration: str = selected.get("expiration_date") or selected.get("expiration", "")

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
            bb_width=bb_width,
            macd_hist=macd_hist,
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
            net_delta=-selected["_delta"],
            net_theta=selected["_theta"],
            metadata={
                "rsi": rsi,
                "price": price,
                "ema_20": ema_20,
                "ema_50": ema_50,
                "bb_width": bb_width,
                "macd_hist": macd_hist,
                "relative_volume": rel_vol,
                "annualized_return": annualized_return,
                "dte": dte,
            },
        )
