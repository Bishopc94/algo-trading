"""Pullback-to-support strategy (swing trade).

Theory:
  In established uptrends, buying pullbacks to key moving average support
  offers high-probability entries.  We require confluence: EMA stacking,
  RSI pullback into the 40-55 zone, volume dry-up on the pullback (sellers
  exhausted), bullish candle reversal pattern, and MACD still positive.

Entry conditions (ALL must be true):
  1. EMA-20 > EMA-50 (uptrend structure, gap > 0.5%)
  2. Close < EMA-20 but > EMA-50 (pulled back into support zone)
  3. Close near EMA-20 or EMA-50 (within tolerance 1.5%)
  4. 35 < RSI < 58 (pulled back, not broken — widened range)
  5. MACD line > signal line (trend momentum still intact despite pullback)
  6. Bullish candle (close > open — buyers stepping in)

Conviction: additive across trend, RSI, volume, MACD, proximity factors.
Volume dry-up is a conviction factor, not a hard gate.
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

        tolerance_pct: float = getattr(self.config, "pullback_tolerance_pct", 1.5) / 100.0
        rsi_min: float = getattr(self.config, "rsi_min", 35)
        rsi_max: float = getattr(self.config, "rsi_max", 58)
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

        # Uptrend structure with meaningful gap (relaxed from 1% to 0.5%)
        if ema_20 <= ema_50:
            self._reject(symbol, "ema20_above_ema50", ema_20, ema_50, "above")
            return None
        ema_gap_pct = (ema_20 - ema_50) / ema_50
        if ema_gap_pct < 0.005:
            self._reject(symbol, "ema_gap_pct", ema_gap_pct, 0.005, "above")
            return None

        # Must have pulled back below EMA-20
        if close >= ema_20:
            self._reject(symbol, "close_below_ema20", close, ema_20, "below")
            return None

        # Must be above EMA-50 (trend intact)
        if close <= ema_50:
            self._reject(symbol, "close_above_ema50", close, ema_50, "above")
            return None

        # Near EMA support (widened tolerance)
        near_ema20 = abs(close - ema_20) / ema_20 <= tolerance_pct
        near_ema50 = abs(close - ema_50) / ema_50 <= tolerance_pct
        if not (near_ema20 or near_ema50):
            dist_20 = abs(close - ema_20) / ema_20
            dist_50 = abs(close - ema_50) / ema_50
            self._reject(symbol, "near_ema_support", min(dist_20, dist_50), tolerance_pct, "below")
            return None

        # RSI in pullback range (widened: 35-58)
        if rsi_val <= rsi_min:
            self._reject(symbol, "rsi_min", rsi_val, rsi_min, "above")
            return None
        if rsi_val >= rsi_max:
            self._reject(symbol, "rsi_max", rsi_val, rsi_max, "below")
            return None

        # MACD line still above signal (trend intact despite pullback)
        if macd_line <= macd_signal:
            self._reject(symbol, "macd_above_signal", macd_line, macd_signal, "above")
            return None

        # Bullish candle (buyers stepping in)
        if close <= open_price:
            self._reject(symbol, "bullish_candle", close, open_price, "above")
            return None

        # Recent RSI was higher (confirming it's a fresh pullback)
        if len(df) >= 5:
            recent_rsi_max = max(float(df.iloc[i]["rsi_14"]) for i in range(-5, -1))
            if recent_rsi_max < 48:
                self._reject(symbol, "recent_rsi_max", recent_rsi_max, 48.0, "above")
                return None

        # ── Conviction scoring (additive, 0.50–0.90) ──
        conviction = 0.50

        # +0.12: Strong uptrend structure
        if ema_gap_pct > 0.03:
            conviction += 0.12
        elif ema_gap_pct > 0.02:
            conviction += 0.08
        elif ema_gap_pct > 0.01:
            conviction += 0.05
        elif ema_gap_pct > 0.005:
            conviction += 0.02

        # +0.08: RSI in ideal pullback zone (40-52)
        if 40 <= rsi_val <= 52:
            conviction += 0.08
        elif 35 < rsi_val < 40 or 52 < rsi_val < 56:
            conviction += 0.04

        # +0.07: Volume dry-up (lower = better for pullback buy)
        # Now a conviction factor, not a hard gate
        if rel_vol < 0.7:
            conviction += 0.07  # Very low volume = sellers exhausted
        elif rel_vol < 0.9:
            conviction += 0.05
        elif rel_vol < 1.2:
            conviction += 0.02
        # rel_vol > 1.2 = distribution risk, no bonus (but not rejected)

        # +0.06: MACD spread (bigger gap = stronger underlying trend)
        macd_spread = macd_line - macd_signal
        if macd_spread > 0.01 * close:
            conviction += 0.06
        elif macd_spread > 0:
            conviction += 0.03

        # +0.05: Proximity to support
        if near_ema20:
            conviction += 0.05  # Shallow pullback = stronger trend
        elif near_ema50:
            conviction += 0.03  # Deeper pullback but still at support

        conviction = max(0.50, min(0.90, conviction))

        entry_price = close
        # Pullback thesis: ema_50 is support. Keep the ema_50-anchored stop
        # but let the ExitPlanner find a smarter swing-high target.
        stop_loss = ema_50 - atr_stop_mult * atr
        levels = self._plan_long_exit(
            bars=df, entry_price=entry_price, atr=atr,
            base_stop_mult=atr_stop_mult, base_tp_mult=atr_tp_mult,
        )
        take_profit = levels.take_profit

        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        rr = reward / risk if risk > 0 else 0.0
        if risk <= 0 or rr < 2.0:
            self._reject(symbol, "risk_reward", rr, 2.0, "above")
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
