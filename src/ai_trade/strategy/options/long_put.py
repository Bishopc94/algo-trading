"""Long Put -- directional bearish bet on breakdowns.

Theory:
  Buy a single put option when strong bearish signals confirm a breakdown:
    - Price breaks below its 20-day low (technical breakdown)
    - Price below both EMA-20 and EMA-50 (confirmed downtrend)
    - EMA-20 < EMA-50 (bearish EMA structure)
    - RSI below 40 (bearish momentum)
    - MACD histogram negative (momentum aligned bearish)
    - Volume elevated (1.5x+ -- institutional selling)
    - Bearish candle (close < open -- sellers in control)

  Max Profit: (Strike - 0) x 100 - premium paid.
  Max Loss: Premium paid x 100 shares per contract.
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


class LongPutStrategy(BaseOptionsStrategy):
    """Enter a long put on confirmed breakdowns with multi-indicator bearish confluence."""

    bias = "bearish"

    def evaluate(
        self,
        underlying: str,
        stock_bars: pd.DataFrame,
        chain_data: list[dict],
        snapshots: dict,
    ) -> OptionsSignal | None:
        if not self.enabled:
            return None

        target_delta: float = getattr(self.config, "target_delta", 0.55)
        min_dte: int = getattr(self.config, "min_dte", 20)
        max_dte: int = getattr(self.config, "max_dte", 60)
        max_contract_cost: float = getattr(self.config, "max_contract_cost", 75.0)
        breakdown_lookback: int = getattr(self.config, "breakdown_lookback", 10)

        # ------------------------------------------------------------------
        # 1. Stock filter -- require multi-indicator bearish confluence
        # ------------------------------------------------------------------
        df = stock_bars.copy()
        if len(df) < max(breakdown_lookback + 1, 52):
            return None

        add_rsi(df)
        add_ema(df, periods=[20, 50])
        add_volume_profile(df)
        add_macd(df)

        df["low_20"] = df["low"].rolling(breakdown_lookback).min().shift(1)

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        rsi: float = latest.get("rsi_14", 50.0)
        price: float = latest["close"]
        open_price: float = latest["open"]
        ema_20: float = latest.get("ema_20", price)
        ema_50: float = latest.get("ema_50", price)
        rel_vol: float = latest.get("relative_volume", 1.0)
        low_20: float = latest.get("low_20", price)
        macd_hist: float = latest.get("macd_hist", 0.0)
        prev_macd_hist: float = prev.get("macd_hist", 0.0)

        # RSI confirms bearish momentum (relaxed from 40 to 48)
        if rsi > 48:
            self._reject(underlying, "rsi_max_bearish", rsi, 48.0, "below")
            return None

        # Price below EMA-20 (short-term weakness confirmed)
        if price >= ema_20:
            self._reject(underlying, "price_below_ema20", price, ema_20, "below")
            return None

        # EMA structure: at least EMA-20 weakening
        ema_struct_ceil = ema_50 * 1.02
        if ema_20 > ema_struct_ceil:
            self._reject(underlying, "ema_structure_bearish", ema_20, ema_struct_ceil, "below")
            return None

        # MACD not bullish (relaxed from strictly negative)
        macd_ceil = 0.005 * price
        if macd_hist > macd_ceil:
            self._reject(underlying, "macd_not_bullish", macd_hist, macd_ceil, "below")
            return None

        # Volume at least average (relaxed from 1.5x)
        if rel_vol < 1.0:
            self._reject(underlying, "rel_volume_min", rel_vol, 1.0, "above")
            return None

        # Bearish candle (sellers in control)
        if price >= open_price:
            self._reject(underlying, "bearish_candle", price, open_price, "below")
            return None

        # ------------------------------------------------------------------
        # 2. Select contract -- ATM to slightly ITM put
        # ------------------------------------------------------------------
        eligible_puts = filter_contracts(chain_data, "put", min_dte, max_dte)
        if not eligible_puts:
            return None

        enrich_greeks(eligible_puts, snapshots)

        delta_candidates = filter_by_delta(eligible_puts, 0.45, 0.65, fallback_min=0.2)
        if not delta_candidates:
            return None

        selected = select_by_delta(delta_candidates, target_delta)

        # ------------------------------------------------------------------
        # 3. Pricing checks
        # ------------------------------------------------------------------
        ask_price: float = selected["_ask"]
        cost_per_contract: float = ask_price * 100

        if cost_per_contract > max_contract_cost or ask_price <= 0:
            return None

        # ------------------------------------------------------------------
        # 4. Conviction scoring (additive, 0.55-0.90)
        # ------------------------------------------------------------------
        conviction: float = 0.55

        # +0.10: RSI deeply oversold (panic selling = strong signal)
        if rsi < 25:
            conviction += 0.10
        elif rsi < 30:
            conviction += 0.07
        elif rsi < 35:
            conviction += 0.04

        # +0.08: Volume strength
        if rel_vol > 2.5:
            conviction += 0.08
        elif rel_vol > 2.0:
            conviction += 0.05
        elif rel_vol > 1.5:
            conviction += 0.03

        # +0.07: MACD accelerating bearish (histogram getting more negative)
        if macd_hist < prev_macd_hist < 0:
            conviction += 0.07
        elif macd_hist < 0:
            conviction += 0.03

        # +0.05: Deep below EMA-50 (strong downtrend momentum)
        ema_distance = (ema_50 - price) / ema_50 if ema_50 > 0 else 0
        if ema_distance > 0.05:
            conviction += 0.05
        elif ema_distance > 0.03:
            conviction += 0.03

        # +0.05: Bearish EMA gap (EMA-50 well above EMA-20)
        ema_gap = (ema_50 - ema_20) / ema_50 if ema_50 > 0 else 0
        if ema_gap > 0.02:
            conviction += 0.05

        conviction = max(0.55, min(0.90, conviction))

        # ------------------------------------------------------------------
        # 5. Build and return the trade signal
        # ------------------------------------------------------------------
        put_symbol: str = selected.get("symbol", "")
        expiration: str = selected.get("expiration_date") or selected.get("expiration", "")
        strike: float = selected["_strike"]

        legs = [
            {"symbol": put_symbol, "side": "buy", "qty": 1, "position_intent": "buy_to_open"},
        ]

        log.info(
            "long_put_signal",
            underlying=underlying,
            strike=strike,
            ask=ask_price,
            cost=cost_per_contract,
            delta=abs(selected["_delta"]),
            conviction=conviction,
            expiration=expiration,
            macd_hist=macd_hist,
        )

        return OptionsSignal(
            underlying=underlying,
            strategy_type=OptionsStrategyType.LONG_PUT,
            conviction=conviction,
            strategy_name="long_put",
            legs=legs,
            max_cost=ask_price,
            max_loss=cost_per_contract,
            max_profit=strike * 100 - cost_per_contract,
            expiration=expiration,
            strikes=[strike],
            net_delta=selected["_delta"],
            net_theta=selected["_theta"],
            metadata={
                "rsi": rsi,
                "price": price,
                "relative_volume": rel_vol,
                "low_20": low_20,
                "macd_hist": macd_hist,
                "ema_20": ema_20,
                "ema_50": ema_50,
                "cost_per_contract": cost_per_contract,
                "dte": selected.get("_dte", 0),
            },
        )
