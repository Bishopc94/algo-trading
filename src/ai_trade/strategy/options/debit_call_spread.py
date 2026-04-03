"""Bull Call Spread (Debit Call Spread) -- defined-risk directional strategy.

Theory:
  A bull call spread is a two-leg debit strategy:
    1. Buy a call near ATM (higher delta)
    2. Sell a call further OTM (lower delta)

  We add multi-indicator confluence:
    - Price above EMA-20 AND EMA-50 (confirmed uptrend structure)
    - RSI 50-70 (strong momentum, not overbought)
    - MACD histogram positive (momentum aligned)
    - Relative volume > 1.3x (institutional participation)
    - Bullish candle (close > open)

  Max Profit: (Spread width - debit) x 100.
  Max Loss: Debit paid x 100.
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
    filter_by_delta,
    filter_contracts,
    select_by_delta,
)

log = get_logger(__name__)


class DebitCallSpreadStrategy(BaseOptionsStrategy):
    """Enter a bull call spread on strong-momentum setups with multi-indicator confluence."""

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

        long_delta: float = getattr(self.config, "long_delta", 0.60)
        short_delta: float = getattr(self.config, "short_delta", 0.35)
        min_dte: int = getattr(self.config, "min_dte", 30)
        max_dte: int = getattr(self.config, "max_dte", 60)
        max_debit_pct: float = getattr(self.config, "max_debit_pct", 0.50)
        max_risk: float = getattr(self.config, "max_risk", 250.0)
        available_capital: float = getattr(self.config, "available_capital", 500.0)

        # ------------------------------------------------------------------
        # 1. Stock filter -- multi-indicator bullish confluence
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
        open_price: float = latest["open"]
        rel_vol: float = latest.get("relative_volume", 0.0)
        macd_hist: float = latest.get("macd_hist", 0.0)
        prev_macd_hist: float = prev.get("macd_hist", 0.0)

        # RSI in strong momentum zone (not overbought)
        if not (50 < rsi < 70):
            return None

        # Price above both EMAs (confirmed uptrend)
        if price <= ema_20 or price <= ema_50:
            return None

        # Uptrend EMA structure
        if ema_20 <= ema_50:
            return None

        # MACD histogram positive (momentum aligned)
        if macd_hist <= 0:
            return None

        # Volume confirmation
        if rel_vol <= 1.3:
            return None

        # Bullish candle
        if price <= open_price:
            return None

        # ------------------------------------------------------------------
        # 2. Select both strikes
        # ------------------------------------------------------------------
        eligible_calls = filter_contracts(chain_data, "call", min_dte, max_dte)
        if not eligible_calls:
            return None

        enrich_greeks(eligible_calls, snapshots)

        # Long call (buy): delta 0.55-0.65
        long_candidates = filter_by_delta(eligible_calls, 0.55, 0.65, use_absolute=False)
        if not long_candidates:
            long_candidates = [c for c in eligible_calls if c["_delta"] > 0]
        if not long_candidates:
            return None

        long_call = select_by_delta(long_candidates, long_delta, use_absolute=False)
        long_strike: float = long_call["_strike"]
        long_exp = long_call.get("expiration_date") or long_call.get("expiration", "")

        # Short call (sell): same expiration, higher strike, delta 0.30-0.40
        same_exp_above = [
            c for c in eligible_calls
            if c["_strike"] > long_strike
            and (c.get("expiration_date") or c.get("expiration", "")) == long_exp
        ]
        short_candidates = filter_by_delta(same_exp_above, 0.30, 0.40, use_absolute=False)
        if not short_candidates:
            short_candidates = [c for c in same_exp_above if c["_delta"] > 0]
        if not short_candidates:
            return None

        short_call = select_by_delta(short_candidates, short_delta, use_absolute=False)
        short_strike: float = short_call["_strike"]

        spread_width: float = short_strike - long_strike
        if spread_width <= 0:
            return None

        max_width = min(5.0, max(1.0, round(price * 0.02, 0)))
        if spread_width > max_width:
            return None

        # ------------------------------------------------------------------
        # 3. Pricing checks
        # ------------------------------------------------------------------
        debit: float = long_call["_mid"] - short_call["_mid"]
        if debit <= 0:
            return None

        max_loss: float = debit * 100
        max_profit: float = (spread_width - debit) * 100

        if debit / spread_width > max_debit_pct:
            return None
        if max_loss > max_risk or max_loss > available_capital:
            return None

        # Minimum 1.5:1 reward-to-risk
        if max_profit / max_loss < 1.5:
            return None

        # ------------------------------------------------------------------
        # 4. Conviction scoring (additive, 0.55-0.90)
        # ------------------------------------------------------------------
        conviction: float = 0.55

        # +0.08: Volume strength
        if rel_vol > 2.0:
            conviction += 0.08
        elif rel_vol > 1.5:
            conviction += 0.05
        elif rel_vol > 1.3:
            conviction += 0.03

        # +0.07: Favorable debit/width ratio
        debit_pct = debit / spread_width
        if debit_pct < 0.40:
            conviction += 0.07
        elif debit_pct < 0.50:
            conviction += 0.04

        # +0.07: RSI sweet spot (55-65)
        if 55 <= rsi <= 65:
            conviction += 0.07
        elif 50 < rsi < 55:
            conviction += 0.03

        # +0.06: MACD accelerating
        if macd_hist > prev_macd_hist > 0:
            conviction += 0.06
        elif macd_hist > 0:
            conviction += 0.03

        # +0.05: Strong uptrend structure (EMA gap)
        ema_gap_pct = (ema_20 - ema_50) / ema_50 if ema_50 > 0 else 0
        if ema_gap_pct > 0.02:
            conviction += 0.05
        elif ema_gap_pct > 0.01:
            conviction += 0.03

        conviction = max(0.55, min(0.90, conviction))

        # ------------------------------------------------------------------
        # 5. Build and return the trade signal
        # ------------------------------------------------------------------
        long_call_symbol: str = long_call.get("symbol", "")
        short_call_symbol: str = short_call.get("symbol", "")
        expiration: str = long_exp
        net_delta: float = long_call["_delta"] + short_call["_delta"]
        net_theta: float = long_call["_theta"] + short_call["_theta"]

        legs = [
            {"symbol": long_call_symbol, "side": "buy", "qty": 1, "position_intent": "buy_to_open"},
            {"symbol": short_call_symbol, "side": "sell", "qty": 1, "position_intent": "sell_to_open"},
        ]

        log.info(
            "debit_call_spread_signal",
            underlying=underlying,
            long_strike=long_strike,
            short_strike=short_strike,
            debit=debit,
            max_loss=max_loss,
            max_profit=max_profit,
            conviction=conviction,
            expiration=expiration,
            macd_hist=macd_hist,
        )

        return OptionsSignal(
            underlying=underlying,
            strategy_type=OptionsStrategyType.DEBIT_CALL_SPREAD,
            conviction=conviction,
            strategy_name="debit_call_spread",
            legs=legs,
            max_cost=debit,
            max_loss=max_loss,
            max_profit=max_profit,
            expiration=expiration,
            strikes=[long_strike, short_strike],
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
                "debit_pct": debit_pct,
                "dte": long_call.get("_dte", 0),
            },
        )
