"""Cash-Secured Put (CSP) -- income / acquisition strategy.

Theory:
  Sell an OTM put and collect premium. You must hold enough cash to buy
  100 shares at the strike price if assigned.

  We add multi-indicator confluence:
    - RSI 35-55 (neutral, not in freefall)
    - Price near EMA-50 support (within 5% above)
    - EMA-20 > EMA-50 or close to crossing (structure not broken)
    - MACD histogram not deeply negative (selling not accelerating)
    - Volume not spiking on bearish candle (no institutional distribution)

  Max Profit: Premium x 100.
  Max Loss: (Strike - premium) x 100 (stock goes to zero).
"""

from __future__ import annotations

import pandas as pd

from ai_trade.data.indicators import add_ema, add_macd, add_rsi, add_volume_profile
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
    """Sell a cash-secured put on stocks near support with multi-indicator confirmation."""

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

        target_delta: float = getattr(self.config, "target_delta", 0.20)
        min_dte: int = getattr(self.config, "min_dte", 20)
        max_dte: int = getattr(self.config, "max_dte", 45)
        min_annualized_return: float = getattr(self.config, "min_annualized_return", 0.15)
        max_stock_price: float = getattr(self.config, "max_stock_price", 3.00)
        available_capital: float = getattr(self.config, "available_capital", 500.0)

        # ------------------------------------------------------------------
        # 1. Stock filter -- neutral zone near support
        # ------------------------------------------------------------------
        df = stock_bars.copy()
        if len(df) < 52:
            return None

        add_rsi(df)
        add_ema(df, periods=[20, 50])
        add_volume_profile(df)
        add_macd(df)

        latest = df.iloc[-1]
        rsi: float = latest.get("rsi_14", 0.0)
        ema_20: float = latest.get("ema_20", 0.0)
        ema_50: float = latest.get("ema_50", 0.0)
        price: float = latest["close"]
        open_price: float = latest["open"]
        rel_vol: float = latest.get("relative_volume", 1.0)
        macd_hist: float = latest.get("macd_hist", 0.0)

        # RSI not in freefall (widened: 30-60)
        if rsi < 30:
            self._reject(underlying, "rsi_min", rsi, 30.0, "above")
            return None
        if rsi > 60:
            self._reject(underlying, "rsi_max", rsi, 60.0, "below")
            return None

        # Price not way above EMA-50 (within 8% above)
        ema50_ceil = ema_50 * 1.08
        if price > ema50_ceil:
            self._reject(underlying, "price_near_ema50", price, ema50_ceil, "below")
            return None

        # Stock must be affordable for cash-secured requirement
        if price > max_stock_price:
            self._reject(underlying, "max_stock_price", price, max_stock_price, "below")
            return None

        # MACD not deeply negative (selling not accelerating)
        macd_floor = -0.02 * price
        if macd_hist < macd_floor:
            self._reject(underlying, "macd_not_deep_neg", macd_hist, macd_floor, "above")
            return None

        # Reject if high volume on a bearish candle (distribution)
        if rel_vol > 2.0 and price < open_price:
            self._reject(underlying, "bearish_vol_spike", rel_vol, 2.0, "below")
            return None

        # EMA structure: allow EMA-20 slightly below EMA-50 (within 2%)
        ema_struct_floor = ema_50 * 0.98
        if ema_20 < ema_struct_floor:
            self._reject(underlying, "ema_structure", ema_20, ema_struct_floor, "above")
            return None

        # ------------------------------------------------------------------
        # 2. Select contract -- OTM put
        # ------------------------------------------------------------------
        eligible_puts = filter_contracts(chain_data, "put", min_dte, max_dte)
        if not eligible_puts:
            return None

        enrich_greeks(eligible_puts, snapshots)

        otm_puts = [c for c in eligible_puts if c["_strike"] < price]
        delta_candidates = [
            c for c in otm_puts if 0.15 <= abs(c["_delta"]) <= 0.30
        ]
        if not delta_candidates:
            delta_candidates = [c for c in otm_puts if c["_delta"] != 0]
        if not delta_candidates:
            return None

        selected = select_by_delta(delta_candidates, target_delta)

        strike: float = selected["_strike"]
        dte: int = selected["_dte"]

        # ------------------------------------------------------------------
        # 3. Pricing checks
        # ------------------------------------------------------------------
        premium: float = selected["_bid"]
        cash_required: float = strike * 100

        if cash_required > available_capital:
            return None
        if premium <= 0 or strike <= 0 or dte <= 0:
            return None

        annualized_return: float = (premium / strike) * (365.0 / dte)
        if annualized_return < min_annualized_return:
            return None

        max_loss: float = (strike - premium) * 100
        max_profit: float = premium * 100

        # ------------------------------------------------------------------
        # 4. Conviction scoring (additive, 0.50-0.85)
        # ------------------------------------------------------------------
        conviction: float = 0.50

        # +0.08: Strong annualized return
        if annualized_return > 0.30:
            conviction += 0.08
        elif annualized_return > 0.20:
            conviction += 0.05

        # +0.07: RSI in ideal zone (40-50)
        if 40 <= rsi <= 50:
            conviction += 0.07
        elif 35 <= rsi < 40:
            conviction += 0.04

        # +0.06: MACD positive or turning (not selling)
        if macd_hist > 0:
            conviction += 0.06
        elif macd_hist > -0.005 * price:
            conviction += 0.03

        # +0.05: EMA structure intact (EMA-20 > EMA-50)
        if ema_20 > ema_50:
            conviction += 0.05

        # +0.04: Positive theta (time decay benefits us)
        if selected["_theta"] > 0:
            conviction += 0.04

        conviction = max(0.50, min(0.85, conviction))

        # ------------------------------------------------------------------
        # 5. Build and return the trade signal
        # ------------------------------------------------------------------
        put_symbol: str = selected.get("symbol", "")
        expiration: str = selected.get("expiration_date") or selected.get("expiration", "")

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
            macd_hist=macd_hist,
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
            net_delta=selected["_delta"],
            net_theta=selected["_theta"],
            metadata={
                "rsi": rsi,
                "price": price,
                "ema_20": ema_20,
                "ema_50": ema_50,
                "macd_hist": macd_hist,
                "relative_volume": rel_vol,
                "annualized_return": annualized_return,
                "cash_required": cash_required,
                "dte": dte,
            },
        )
