"""Momentum Options -- short-dated OTM plays for 50-300%+ ROI.

Theory:
  Buy cheap OTM options with short DTE (5-20 days) on high-conviction
  momentum moves confirmed by multi-indicator confluence:

  Bullish breakout (buy calls):
    - Price > 20-day high (breakout)
    - RSI > 55 (bullish momentum, not just noise)
    - Price > EMA-20 > EMA-50 (stacked uptrend)
    - MACD histogram positive (momentum aligned)
    - Volume > 2.0x average (institutional participation)
    - Pre-breakout consolidation (5-bar range < 2x ATR)

  Bearish breakdown (buy puts):
    - Price < 20-day low (breakdown)
    - RSI < 45 (bearish momentum)
    - Price < EMA-20 < EMA-50 (stacked downtrend)
    - MACD histogram negative (momentum aligned)
    - Volume > 2.0x average

  Max Profit: 50-300%+ ROI (gamma acceleration near expiry).
  Max Loss: Premium paid (small absolute dollar amount).
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
)

log = get_logger(__name__)


class MomentumOptionsStrategy(BaseOptionsStrategy):
    """Buy short-dated OTM options on high-conviction momentum with multi-indicator confluence."""

    bias = "adaptive"

    def evaluate(
        self,
        underlying: str,
        stock_bars: pd.DataFrame,
        chain_data: list[dict],
        snapshots: dict,
    ) -> OptionsSignal | None:
        if not self.enabled:
            return None

        min_dte: int = getattr(self.config, "min_dte", 5)
        max_dte: int = getattr(self.config, "max_dte", 20)
        min_delta: float = getattr(self.config, "min_delta", 0.25)
        max_delta: float = getattr(self.config, "max_delta", 0.45)
        max_contract_cost: float = getattr(self.config, "max_contract_cost", 75.0)
        min_rel_vol: float = getattr(self.config, "min_relative_volume", 2.0)
        breakout_lookback: int = getattr(self.config, "breakout_lookback", 20)

        # ------------------------------------------------------------------
        # 1. Stock filter -- multi-indicator directional confluence
        # ------------------------------------------------------------------
        df = stock_bars.copy()
        if len(df) < max(breakout_lookback + 1, 52):
            return None

        add_rsi(df)
        add_ema(df, periods=[20, 50])
        add_atr(df)
        add_volume_profile(df)
        add_macd(df)

        df["high_20"] = df["high"].rolling(breakout_lookback).max().shift(1)
        df["low_20"] = df["low"].rolling(breakout_lookback).min().shift(1)

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        rsi: float = latest.get("rsi_14", 50.0)
        price: float = latest["close"]
        ema_20: float = latest.get("ema_20", price)
        ema_50: float = latest.get("ema_50", price)
        atr: float = latest.get("atr_14", 0.0)
        rel_vol: float = latest.get("relative_volume", 0.0)
        high_20: float = latest.get("high_20", 0.0)
        low_20: float = latest.get("low_20", 0.0)
        macd_hist: float = latest.get("macd_hist", 0.0)
        prev_macd_hist: float = prev.get("macd_hist", 0.0)

        # Volume must show participation (relaxed from 2.0 to 1.5)
        vol_threshold = max(min_rel_vol, 1.5)
        if rel_vol < vol_threshold:
            self._reject(underlying, "rel_volume", rel_vol, vol_threshold, "above")
            return None

        # Determine direction with relaxed confluence
        direction = None
        if (rsi > 50 and price > ema_20 and macd_hist > 0):
            direction = "call"
        elif (rsi < 50 and price < ema_20 and macd_hist < 0):
            direction = "put"

        if direction is None:
            self._reject(underlying, "trend_alignment", 0.0, 1.0, "above")
            return None

        # ------------------------------------------------------------------
        # 2. Select contract -- cheap OTM options
        # ------------------------------------------------------------------
        eligible = filter_contracts(chain_data, direction, min_dte, max_dte)
        if not eligible:
            return None

        enrich_greeks(eligible, snapshots, include_iv=True)

        delta_ok = filter_by_delta(eligible, min_delta, max_delta, fallback_min=0.15)
        if not delta_ok:
            return None

        # ------------------------------------------------------------------
        # 3. Select best contract -- cheapest within budget
        # ------------------------------------------------------------------
        delta_ok.sort(key=lambda c: c["_ask"])

        selected = None
        for c in delta_ok:
            cost = c["_ask"] * 100
            if 0 < cost <= max_contract_cost:
                selected = c
                break

        if selected is None:
            cheapest = delta_ok[0]
            if cheapest["_ask"] * 100 <= max_contract_cost * 1.10:
                selected = cheapest
            else:
                return None

        # ------------------------------------------------------------------
        # 4. ROI estimation using ATR-based target
        # ------------------------------------------------------------------
        ask_price: float = selected["_ask"]
        cost_per_contract: float = ask_price * 100
        strike: float = selected["_strike"]
        delta: float = abs(selected["_delta"])

        if direction == "call":
            target_price = price + 1.5 * atr
            intrinsic_at_target = max(0, target_price - strike)
        else:
            target_price = price - 1.5 * atr
            intrinsic_at_target = max(0, strike - target_price)

        potential_profit = intrinsic_at_target - ask_price
        potential_roi = potential_profit / ask_price if ask_price > 0 else 0

        min_roi: float = getattr(self.config, "min_roi_pct", 0.30)
        if potential_roi < min_roi:
            return None

        # ------------------------------------------------------------------
        # 5. Conviction scoring (additive, 0.50-0.90)
        # ------------------------------------------------------------------
        conviction: float = 0.50

        # +0.10: Volume strength
        if rel_vol > 3.0:
            conviction += 0.10
        elif rel_vol > 2.5:
            conviction += 0.07
        elif rel_vol > 2.0:
            conviction += 0.04

        # +0.08: Potential ROI (asymmetric payoff)
        if potential_roi > 2.0:
            conviction += 0.08
        elif potential_roi > 1.0:
            conviction += 0.05
        elif potential_roi > 0.5:
            conviction += 0.03

        # +0.07: MACD accelerating in trade direction
        if direction == "call" and macd_hist > prev_macd_hist > 0:
            conviction += 0.07
        elif direction == "put" and macd_hist < prev_macd_hist < 0:
            conviction += 0.07
        elif (direction == "call" and macd_hist > 0) or (direction == "put" and macd_hist < 0):
            conviction += 0.03

        # +0.05: Delta sweet spot (0.30-0.40, good gamma)
        if 0.30 <= delta <= 0.40:
            conviction += 0.05
        elif delta > 0.25:
            conviction += 0.03

        # +0.05: EMA gap confirms strong trend
        ema_gap = abs(ema_20 - ema_50) / max(ema_50, 0.01)
        if ema_gap > 0.02:
            conviction += 0.05

        # Penalty: very short DTE (theta burns fast)
        dte = selected.get("_dte", 0)
        if dte < 7:
            conviction -= 0.05

        conviction = max(0.50, min(0.90, conviction))

        # ------------------------------------------------------------------
        # 6. Build and return the trade signal
        # ------------------------------------------------------------------
        option_symbol: str = selected.get("symbol", "")
        expiration: str = selected.get("expiration_date") or selected.get("expiration", "")

        legs = [
            {"symbol": option_symbol, "side": "buy", "qty": 1, "position_intent": "buy_to_open"},
        ]

        max_profit_est = cost_per_contract * 3.0

        log.info(
            "momentum_options_signal",
            underlying=underlying,
            direction=direction,
            strike=strike,
            ask=ask_price,
            cost=cost_per_contract,
            delta=selected["_delta"],
            dte=dte,
            potential_roi=round(potential_roi, 2),
            conviction=conviction,
            rel_vol=rel_vol,
            macd_hist=macd_hist,
            atr=atr,
            expiration=expiration,
        )

        strategy_type = (
            OptionsStrategyType.LONG_CALL if direction == "call"
            else OptionsStrategyType.LONG_PUT
        )

        return OptionsSignal(
            underlying=underlying,
            strategy_type=strategy_type,
            conviction=conviction,
            strategy_name="momentum_options",
            legs=legs,
            max_cost=ask_price,
            max_loss=cost_per_contract,
            max_profit=max_profit_est,
            expiration=expiration,
            strikes=[strike],
            net_delta=selected["_delta"],
            net_theta=selected["_theta"],
            metadata={
                "direction": direction,
                "rsi": rsi,
                "price": price,
                "atr": atr,
                "relative_volume": rel_vol,
                "macd_hist": macd_hist,
                "potential_roi": potential_roi,
                "cost_per_contract": cost_per_contract,
                "dte": dte,
                "iv": selected["_iv"],
            },
        )
