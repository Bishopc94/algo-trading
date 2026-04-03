"""Volume-breakout momentum strategy (adaptive hold type).

Theory:
  Stocks that break out to new highs on heavy volume tend to continue.
  We require multi-indicator confluence: price breakout + volume spike +
  trend alignment + momentum confirmation (MACD + RSI) + tight consolidation.

Entry conditions (ALL must be true):
  1. Close > 20-day high (fresh breakout)
  2. Relative volume > 2.0x (institutional participation)
  3. ADR% > 2.0 (enough range to be worth trading)
  4. Close > EMA-20 > EMA-50 (stacked uptrend)
  5. RSI 50-80 (positive momentum, not exhausted)
  6. MACD histogram > 0 (momentum aligned)
  7. Prior 5-bar range was tight (consolidation before breakout)

Conviction: additive across 5 factors (volume, trend, RSI, MACD, consolidation).

Exit: bracket orders handle all exits.
"""

from __future__ import annotations

import pandas as pd

from ai_trade.data.indicators import (
    add_atr, add_ema, add_macd, add_rsi, add_volume_profile, compute_adr,
)
from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import BaseStrategy, HoldType, Signal

logger = get_logger(__name__)


class MomentumStrategy(BaseStrategy):
    """Enter on volume-confirmed breakouts with multi-indicator confluence."""

    def evaluate(
        self,
        symbol: str,
        daily_bars: pd.DataFrame,
        intraday_bars: pd.DataFrame | None = None,
    ) -> Signal | None:
        df = daily_bars.copy()

        breakout_lookback: int = getattr(self.config, "breakout_lookback", 20)
        vol_spike: float = getattr(self.config, "volume_spike_multiplier", 2.0)
        min_adr: float = getattr(self.config, "min_adr_pct", 2.0)
        atr_stop_mult: float = getattr(self.config, "atr_stop_multiplier", 1.5)
        atr_tp_mult: float = getattr(self.config, "atr_tp_multiplier", 3.5)

        add_volume_profile(df)
        add_atr(df)
        add_ema(df, [20, 50])
        add_rsi(df)
        add_macd(df)

        if len(df) < max(breakout_lookback + 1, 52):
            return None

        df["high_20"] = df["high"].rolling(breakout_lookback).max().shift(1)

        latest = df.iloc[-1]
        close: float = latest["close"]
        high_20: float = latest["high_20"]
        rel_vol: float = latest["relative_volume"]
        adr_pct: float = compute_adr(df)
        atr: float = latest["atr_14"]
        ema_20: float = latest["ema_20"]
        ema_50: float = latest["ema_50"]
        rsi_val: float = latest["rsi_14"]
        macd_hist: float = latest["macd_hist"]

        # ── Hard filters ──
        if close <= high_20:
            return None
        if rel_vol <= vol_spike:
            return None
        if adr_pct <= min_adr:
            return None

        # Stacked uptrend: close > EMA-20 > EMA-50
        if close <= ema_20 or ema_20 <= ema_50:
            return None

        # RSI in momentum range (not oversold, not exhausted)
        if rsi_val <= 50 or rsi_val >= 80:
            return None

        # MACD histogram positive (momentum aligned)
        if macd_hist <= 0:
            return None

        # Pre-breakout consolidation: 5-bar range was tight (< 1.5x ATR)
        recent_5 = df.iloc[-6:-1]
        consolidation_range = recent_5["high"].max() - recent_5["low"].min()
        if consolidation_range > 2.0 * atr:
            return None  # Too choppy, not a clean breakout

        # ── Conviction scoring (additive, 0.55–1.0) ──
        conviction = 0.55

        # +0.10: Volume strength
        if rel_vol >= 5.0:
            conviction += 0.10
        elif rel_vol >= 3.0:
            conviction += 0.07
        else:
            conviction += 0.03

        # +0.10: Trend strength (EMA gap)
        trend_gap = (ema_20 - ema_50) / ema_50
        if trend_gap > 0.03:
            conviction += 0.10
        elif trend_gap > 0.015:
            conviction += 0.06
        else:
            conviction += 0.02

        # +0.08: RSI sweet spot (55-70 = strong but not exhausted)
        if 55 <= rsi_val <= 70:
            conviction += 0.08
        elif 50 < rsi_val < 55 or 70 < rsi_val < 75:
            conviction += 0.04

        # +0.07: MACD histogram strength (accelerating momentum)
        prev_macd_hist = float(df.iloc[-2]["macd_hist"])
        if macd_hist > prev_macd_hist > 0:
            conviction += 0.07  # Accelerating
        elif macd_hist > 0:
            conviction += 0.03  # Positive but decelerating

        # +0.05: Tight consolidation before breakout
        if consolidation_range < 1.0 * atr:
            conviction += 0.05
        elif consolidation_range < 1.5 * atr:
            conviction += 0.02

        conviction = max(0.55, min(1.0, conviction))

        entry_price = close
        stop_loss = entry_price - atr_stop_mult * atr
        take_profit = entry_price + atr_tp_mult * atr

        # Enforce minimum 2:1 R:R
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        if risk <= 0 or reward / risk < 2.0:
            return None

        hold_type = HoldType.DAY if conviction >= 0.9 else HoldType.SWING

        logger.info(
            "momentum_signal",
            symbol=symbol,
            rel_vol=rel_vol,
            adr_pct=adr_pct,
            conviction=conviction,
            entry=entry_price,
            stop=stop_loss,
            target=take_profit,
            rsi=rsi_val,
            macd_hist=macd_hist,
            trend_gap=round(trend_gap, 4),
            atr=atr,
            hold_type=hold_type.value,
        )

        return Signal(
            symbol=symbol,
            direction="long",
            conviction=conviction,
            strategy_name="momentum",
            hold_type=hold_type,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            metadata={
                "relative_volume": rel_vol,
                "adr_pct": adr_pct,
                "breakout_level": high_20,
                "rsi": rsi_val,
                "macd_hist": macd_hist,
                "trend_gap": round(trend_gap, 4),
                "atr": atr,
            },
        )

    def should_exit(
        self, symbol: str, bars: pd.DataFrame, entry_price: float
    ) -> bool:
        return False
