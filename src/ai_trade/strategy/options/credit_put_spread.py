"""Bull Put Spread (Credit Put Spread) -- highest-probability options strategy.

Theory:
  A bull put spread is a two-leg credit strategy:
    1. Sell a put at a higher strike (collect premium)
    2. Buy a put at a lower strike (insurance)

  We add multi-indicator confluence:
    - Price above EMA-20 AND EMA-50 (confirmed uptrend -- safe to sell puts)
    - RSI > 45 (not bearish)
    - MACD histogram positive or rising (momentum not deteriorating)
    - Volume not spiking bearishly (rel_vol < 2.0 or bullish candle)
    - EMA-20 > EMA-50 (uptrend structure intact)

  Max Profit: Net credit received x 100.
  Max Loss: (Spread width - credit) x 100.
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


class CreditPutSpreadStrategy(BaseOptionsStrategy):
    """Enter a bull put spread with multi-indicator uptrend confirmation."""

    bias = "bullish"

    def evaluate(
        self,
        underlying: str,
        stock_bars: pd.DataFrame,
        chain_data: list[dict],
        snapshots: dict,
    ) -> OptionsSignal | None:
        if not self.enabled:
            return None

        target_delta: float = getattr(self.config, "target_delta", 0.25)
        min_dte: int = getattr(self.config, "min_dte", 20)
        max_dte: int = getattr(self.config, "max_dte", 45)
        max_spread_width: float = getattr(self.config, "max_spread_width", 1.50)
        min_credit_pct: float = getattr(self.config, "min_credit_pct", 0.30)
        max_risk: float = getattr(self.config, "max_risk", 100.0)
        available_capital: float = getattr(self.config, "available_capital", 500.0)

        # ------------------------------------------------------------------
        # 1. Stock filter -- require confirmed uptrend to sell puts
        # ------------------------------------------------------------------
        df = stock_bars.copy()
        if len(df) < 52:
            return None

        add_rsi(df)
        add_ema(df, periods=[20, 50])
        add_volume_profile(df)
        add_macd(df)

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        rsi: float = latest.get("rsi_14", 0.0)
        ema_20: float = latest.get("ema_20", 0.0)
        ema_50: float = latest.get("ema_50", 0.0)
        price: float = latest["close"]
        rel_vol: float = latest.get("relative_volume", 1.0)
        macd_hist: float = latest.get("macd_hist", 0.0)
        prev_macd_hist: float = prev.get("macd_hist", 0.0)

        # RSI not bearish (selling puts into downtrends is dangerous)
        if rsi <= 45:
            return None

        # Price above EMA-20 (short-term uptrend)
        if price <= ema_20:
            return None

        # Price above EMA-50 (medium-term uptrend)
        if price <= ema_50:
            return None

        # Uptrend EMA structure
        if ema_20 <= ema_50:
            return None

        # MACD not deeply negative (momentum not deteriorating)
        if macd_hist < -0.01 * price:
            return None

        # High volume on a down candle is a warning sign
        if rel_vol > 2.0 and price < latest["open"]:
            return None

        # ------------------------------------------------------------------
        # 2. Select strikes for both legs
        # ------------------------------------------------------------------
        eligible_puts = filter_contracts(chain_data, "put", min_dte, max_dte)
        if not eligible_puts:
            return None

        enrich_greeks(eligible_puts, snapshots)

        # Short put: sell at target delta
        short_put = select_by_delta(eligible_puts, target_delta)
        if short_put is None:
            return None

        short_strike: float = short_put["_strike"]

        # Long put: buy at lower strike, same expiration
        long_candidates = [
            c for c in eligible_puts
            if c["_strike"] < short_strike
            and (short_strike - c["_strike"]) <= max_spread_width
            and c.get("expiration_date", c.get("expiration", ""))
            == short_put.get("expiration_date", short_put.get("expiration", ""))
        ]
        if not long_candidates:
            return None

        long_put = max(long_candidates, key=lambda c: c["_strike"])
        long_strike: float = long_put["_strike"]

        # ------------------------------------------------------------------
        # 3. Pricing checks
        # ------------------------------------------------------------------
        credit_received: float = short_put["_mid"] - long_put["_mid"]
        spread_width: float = short_strike - long_strike

        if spread_width <= 0 or credit_received < 0.10:
            return None

        max_loss: float = (spread_width * 100) - (credit_received * 100)
        max_profit: float = credit_received * 100

        if max_loss > max_risk or max_loss > available_capital:
            return None
        if credit_received / spread_width < min_credit_pct:
            return None

        # ------------------------------------------------------------------
        # 4. Conviction scoring (additive, 0.55-0.90)
        # ------------------------------------------------------------------
        conviction: float = 0.55

        # +0.08: RSI confirms bullish momentum
        if rsi > 60:
            conviction += 0.08
        elif rsi > 50:
            conviction += 0.05

        # +0.07: Strong credit/width ratio
        credit_pct = credit_received / spread_width
        if credit_pct > 0.40:
            conviction += 0.07
        elif credit_pct > 0.33:
            conviction += 0.04

        # +0.06: MACD positive and rising
        if macd_hist > prev_macd_hist > 0:
            conviction += 0.06
        elif macd_hist > 0:
            conviction += 0.03

        # +0.05: Strong uptrend structure
        ema_gap_pct = (ema_20 - ema_50) / ema_50 if ema_50 > 0 else 0
        if ema_gap_pct > 0.02:
            conviction += 0.05
        elif ema_gap_pct > 0.01:
            conviction += 0.03

        # +0.04: Positive net theta (time decay benefits us)
        net_theta: float = short_put["_theta"] + long_put["_theta"]
        if net_theta > 0:
            conviction += 0.04

        conviction = max(0.55, min(0.90, conviction))

        # ------------------------------------------------------------------
        # 5. Build and return the trade signal
        # ------------------------------------------------------------------
        short_put_symbol: str = short_put.get("symbol", "")
        long_put_symbol: str = long_put.get("symbol", "")
        expiration: str = short_put.get("expiration_date") or short_put.get("expiration", "")
        net_delta: float = short_put["_delta"] + long_put["_delta"]

        legs = [
            {"symbol": short_put_symbol, "side": "sell", "qty": 1, "position_intent": "sell_to_open"},
            {"symbol": long_put_symbol, "side": "buy", "qty": 1, "position_intent": "buy_to_open"},
        ]

        log.info(
            "credit_put_spread_signal",
            underlying=underlying,
            short_strike=short_strike,
            long_strike=long_strike,
            credit=credit_received,
            max_loss=max_loss,
            conviction=conviction,
            expiration=expiration,
            macd_hist=macd_hist,
        )

        return OptionsSignal(
            underlying=underlying,
            strategy_type=OptionsStrategyType.CREDIT_PUT_SPREAD,
            conviction=conviction,
            strategy_name="credit_put_spread",
            legs=legs,
            min_credit=credit_received,
            max_loss=max_loss,
            max_profit=max_profit,
            expiration=expiration,
            strikes=[short_strike, long_strike],
            net_delta=net_delta,
            net_theta=net_theta,
            metadata={
                "rsi": rsi,
                "price": price,
                "ema_20": ema_20,
                "ema_50": ema_50,
                "relative_volume": rel_vol,
                "macd_hist": macd_hist,
                "spread_width": spread_width,
                "credit_pct": credit_pct,
                "dte": short_put.get("_dte", 0),
            },
        )
