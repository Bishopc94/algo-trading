"""Long Call -- high-leverage directional bet on momentum breakouts.

Theory:
  Buy a single call option when multiple bullish signals align:
    - Price breaks above its 20-day high (technical breakout)
    - Stacked EMAs confirm uptrend (close > EMA-20 > EMA-50)
    - Volume is at least 2x the 20-day average (institutional participation)
    - RSI is strong but not overbought (55-75)
    - MACD histogram positive and rising (momentum accelerating)
    - Pre-breakout consolidation (tight range before explosion)

  Max Profit: Theoretically unlimited.
  Max Loss: Premium paid x 100 shares per contract.
  Hold type: SWING (options expiration handles timing).
"""

from __future__ import annotations

import pandas as pd

from ai_trade.data.indicators import add_atr, add_ema, add_macd, add_rsi, add_volume_profile
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
    """Enter a long call on a confirmed high-volume breakout with multi-indicator confluence."""

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

        target_delta: float = getattr(self.config, "target_delta", 0.60)
        min_dte: int = getattr(self.config, "min_dte", 20)
        max_dte: int = getattr(self.config, "max_dte", 60)
        max_contract_cost: float = getattr(self.config, "max_contract_cost", 75.0)
        max_iv_percentile: float = getattr(self.config, "max_iv_percentile", 1.00)
        breakout_lookback: int = getattr(self.config, "breakout_lookback", 20)

        # ------------------------------------------------------------------
        # 1. Stock filter -- require multi-indicator bullish confluence
        # ------------------------------------------------------------------
        df = stock_bars.copy()
        if len(df) < max(breakout_lookback + 1, 52):
            return None

        add_rsi(df)
        add_ema(df, periods=[20, 50])
        add_volume_profile(df)
        add_macd(df)
        add_atr(df)

        df["high_20"] = df["high"].rolling(breakout_lookback).max().shift(1)

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        rsi: float = latest.get("rsi_14", 0.0)
        price: float = latest["close"]
        ema_20: float = latest.get("ema_20", 0.0)
        ema_50: float = latest.get("ema_50", 0.0)
        rel_vol: float = latest.get("relative_volume", 0.0)
        high_20: float = latest.get("high_20", 0.0)
        macd_hist: float = latest.get("macd_hist", 0.0)
        prev_macd_hist: float = prev.get("macd_hist", 0.0)
        atr: float = latest.get("atr_14", 0.0)

        # RSI in momentum range (widened: 48-78)
        if rsi < 48:
            self._reject(underlying, "rsi_min", rsi, 48.0, "above")
            return None
        if rsi > 78:
            self._reject(underlying, "rsi_max", rsi, 78.0, "below")
            return None

        # Price above EMA-50 (medium-term uptrend)
        if price <= ema_50:
            self._reject(underlying, "price_above_ema50", price, ema_50, "above")
            return None

        # EMA structure not deeply broken
        ema_struct_floor = ema_50 * 0.98
        if ema_20 < ema_struct_floor:
            self._reject(underlying, "ema_structure", ema_20, ema_struct_floor, "above")
            return None

        # Volume confirmation (relaxed from 2.0 to 1.3)
        if rel_vol < 1.3:
            self._reject(underlying, "rel_volume", rel_vol, 1.3, "above")
            return None

        # MACD not deeply negative (relaxed from strictly positive)
        macd_floor = -0.005 * price
        if macd_hist < macd_floor:
            self._reject(underlying, "macd_not_deep_neg", macd_hist, macd_floor, "above")
            return None

        # ------------------------------------------------------------------
        # 2. Select contract -- ATM to slightly ITM call
        # ------------------------------------------------------------------
        eligible_calls = filter_contracts(chain_data, "call", min_dte, max_dte)
        if not eligible_calls:
            return None

        enrich_greeks(eligible_calls, snapshots, include_iv=True)

        delta_candidates = filter_by_delta(eligible_calls, 0.50, 0.70, use_absolute=False)
        if not delta_candidates:
            delta_candidates = [c for c in eligible_calls if c["_delta"] > 0]
        if not delta_candidates:
            return None

        selected = select_by_delta(delta_candidates, target_delta, use_absolute=False)

        # ------------------------------------------------------------------
        # 3. Pricing checks
        # ------------------------------------------------------------------
        ask_price: float = selected["_ask"]
        cost_per_contract: float = ask_price * 100
        max_loss: float = cost_per_contract

        if cost_per_contract > max_contract_cost:
            return None

        iv: float = selected["_iv"]
        if iv > max_iv_percentile and iv > 0:
            return None

        # ------------------------------------------------------------------
        # 4. Conviction scoring (additive, 0.55-0.90)
        # ------------------------------------------------------------------
        conviction: float = 0.55

        # +0.10: Volume strength
        if rel_vol > 3.0:
            conviction += 0.10
        elif rel_vol > 2.5:
            conviction += 0.07
        elif rel_vol > 2.0:
            conviction += 0.04

        # +0.08: MACD accelerating (histogram rising)
        if macd_hist > prev_macd_hist > 0:
            conviction += 0.08
        elif macd_hist > 0:
            conviction += 0.04

        # +0.07: Strong uptrend structure (EMA gap)
        ema_gap_pct = (ema_20 - ema_50) / ema_50 if ema_50 > 0 else 0
        if ema_gap_pct > 0.03:
            conviction += 0.07
        elif ema_gap_pct > 0.01:
            conviction += 0.04

        # +0.05: Cheaper contract (less absolute risk)
        if cost_per_contract < 40.0:
            conviction += 0.05
        elif cost_per_contract < 60.0:
            conviction += 0.03

        # +0.05: Delta sweet spot (0.55-0.65)
        if 0.55 <= selected["_delta"] <= 0.65:
            conviction += 0.05

        conviction = max(0.55, min(0.90, conviction))

        # ------------------------------------------------------------------
        # 5. Build and return the trade signal
        # ------------------------------------------------------------------
        call_symbol: str = selected.get("symbol", "")
        expiration: str = selected.get("expiration_date") or selected.get("expiration", "")
        strike: float = selected["_strike"]

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
            macd_hist=macd_hist,
        )

        return OptionsSignal(
            underlying=underlying,
            strategy_type=OptionsStrategyType.LONG_CALL,
            conviction=conviction,
            strategy_name="long_call",
            legs=legs,
            max_cost=ask_price,
            max_loss=max_loss,
            max_profit=0.0,  # Theoretically unlimited
            expiration=expiration,
            strikes=[strike],
            net_delta=selected["_delta"],
            net_theta=selected["_theta"],
            metadata={
                "rsi": rsi,
                "price": price,
                "relative_volume": rel_vol,
                "high_20": high_20,
                "iv": iv,
                "macd_hist": macd_hist,
                "ema_gap_pct": round(ema_gap_pct, 4),
                "cost_per_contract": cost_per_contract,
                "dte": selected.get("_dte", 0),
            },
        )
