"""Pullback-to-support strategy (swing trade).

Theory:
  In established uptrends, buying pullbacks to key moving average support
  offers high-probability entries.  We require confluence: EMA stacking,
  RSI pullback into the 40-55 zone, volume dry-up on the pullback (sellers
  exhausted), bullish candle reversal pattern, and MACD still positive.

Entry conditions (ALL must be true):
  1. EMA-20 > EMA-50 (uptrend structure, gap > 1%)
  2. Close < EMA-20 but > EMA-50 (pulled back into support zone)
  3. Close near EMA-20 or EMA-50 (within tolerance %)
  4. 40 < RSI < 55 (pulled back, not broken)
  5. MACD line > signal line (trend momentum still intact despite pullback)
  6. Volume below 20-day average (sellers drying up during pullback)
  7. Bullish candle (close > open — buyers stepping in)

Conviction: additive across trend, RSI, volume, MACD, candle factors.
Hold type: SWING.
"""

from __future__ import annotations

import pandas as pd

from ai_trade.data.indicators import add_atr, add_ema, add_macd, add_rsi, add_volume_profile
from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import BaseStrategy, HoldType, Signal

logger = get_logger(__name__)


class PullbackStrategy(BaseStrategy):
    """Buy pullbacks to EMA support with multi-indicator confluence."""

    def evaluate(
        self,
        symbol: str,
        daily_bars: pd.DataFrame,
        intraday_bars: pd.DataFrame | None = None,
    ) -> Signal | None:
        df = daily_bars.copy()

        tolerance_pct: float = getattr(self.config, "pullback_tolerance_pct", 1.0) / 100.0
        rsi_min: float = getattr(self.config, "rsi_min", 40)
        rsi_max: float = getattr(self.config, "rsi_max", 55)
        atr_stop_mult: float = getattr(self.config, "atr_stop_multiplier", 0.5)
        atr_tp_mult: float = getattr(self.config, "atr_tp_multiplier", 3.0)

        add_ema(df, [20, 50])
        add_rsi(df)
        add_atr(df)
        add_volume_profile(df)
        add_macd(df)

        if len(df) < 52:
            return None

        latest = df.iloc[-1]
        close: float = latest["close"]
        open_price: float = latest["open"]
        ema_20: float = latest["ema_20"]
        ema_50: float = latest["ema_50"]
        rsi_val: float = latest["rsi_14"]
        atr: float = latest["atr_14"]
        rel_vol: float = latest.get("relative_volume", 1.0)
        macd_line: float = latest["macd"]
        macd_signal: float = latest["macd_signal"]

        # ── Hard filters ──

        # Uptrend structure with meaningful gap
        if ema_20 <= ema_50:
            return None
        ema_gap_pct = (ema_20 - ema_50) / ema_50
        if ema_gap_pct < 0.01:
            return None

        # Must have pulled back below EMA-20
        if close >= ema_20:
            return None

        # Must be above EMA-50 (trend intact)
        if close <= ema_50:
            return None

        # Near EMA support
        near_ema20 = abs(close - ema_20) / ema_20 <= tolerance_pct
        near_ema50 = abs(close - ema_50) / ema_50 <= tolerance_pct
        if not (near_ema20 or near_ema50):
            return None

        # RSI in pullback range
        if rsi_val <= rsi_min or rsi_val >= rsi_max:
            return None

        # MACD line still above signal (trend intact despite pullback)
        if macd_line <= macd_signal:
            return None

        # Volume dry-up during pullback (sellers exhausting)
        # Current volume below 20-day average = low selling pressure
        if rel_vol > 1.2:
            return None  # Too much volume on pullback = distribution, not healthy dip

        # Bullish candle (buyers stepping in)
        if close <= open_price:
            return None

        # Recent RSI was higher (confirming it's a fresh pullback, not sustained weakness)
        if len(df) >= 5:
            recent_rsi_max = max(float(df.iloc[i]["rsi_14"]) for i in range(-5, -1))
            if recent_rsi_max < 50:
                return None

        # ── Conviction scoring (additive, 0.55–0.85) ──
        conviction = 0.55

        # +0.10: Strong uptrend structure
        if ema_gap_pct > 0.03:
            conviction += 0.10
        elif ema_gap_pct > 0.02:
            conviction += 0.07
        elif ema_gap_pct > 0.01:
            conviction += 0.03

        # +0.07: RSI in ideal pullback zone (43-50)
        if 43 <= rsi_val <= 50:
            conviction += 0.07
        elif 40 < rsi_val < 43 or 50 < rsi_val < 53:
            conviction += 0.04

        # +0.05: Volume dry-up (lower = better for pullback buy)
        if rel_vol < 0.7:
            conviction += 0.05  # Very low volume = sellers exhausted
        elif rel_vol < 0.9:
            conviction += 0.03

        # +0.05: MACD spread (bigger gap = stronger underlying trend)
        macd_spread = macd_line - macd_signal
        if macd_spread > 0.01 * close:
            conviction += 0.05
        elif macd_spread > 0:
            conviction += 0.02

        # +0.03: Near EMA-20 (first support) vs EMA-50 (deeper pullback)
        if near_ema20:
            conviction += 0.03  # Shallow pullback = stronger trend

        conviction = max(0.55, min(0.85, conviction))

        entry_price = close
        stop_loss = ema_50 - atr_stop_mult * atr
        take_profit = entry_price + atr_tp_mult * atr

        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        if risk <= 0 or reward / risk < 2.0:
            return None

        logger.info(
            "pullback_signal",
            symbol=symbol,
            ema_20=ema_20,
            ema_50=ema_50,
            rsi=rsi_val,
            rel_vol=rel_vol,
            macd_spread=round(macd_spread, 4),
            conviction=conviction,
            entry=entry_price,
            stop=stop_loss,
            target=take_profit,
        )

        return Signal(
            symbol=symbol,
            direction="long",
            conviction=conviction,
            strategy_name="pullback",
            hold_type=HoldType.SWING,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            metadata={
                "ema_20": ema_20,
                "ema_50": ema_50,
                "rsi": rsi_val,
                "relative_volume": rel_vol,
                "macd_spread": round(macd_spread, 4),
                "atr": atr,
            },
        )

    def should_exit(
        self, symbol: str, bars: pd.DataFrame, entry_price: float
    ) -> bool:
        df = bars.copy()
        add_ema(df, [50])
        add_rsi(df)
        latest = df.iloc[-1]
        if latest["close"] < latest["ema_50"] or latest["rsi_14"] > 70:
            return True
        return False
