"""Opening Range Breakout (ORB) strategy (day trade — costs 1 PDT slot).

Theory:
  The first 30 minutes of trading establishes a range that reflects the
  initial battle between buyers and sellers.  A breakout above that range
  on strong volume predicts the intraday trend direction.

  We add multi-indicator confirmation: range must have meaningful width
  (not a flat open), breakout must be sustained (not a wick), volume must
  spike on the breakout bar, and the breakout bar itself must be bullish.

Entry conditions (ALL must be true):
  1. Intraday (minute) bars available with >= opening_range_minutes + 1
  2. Opening range width >= 0.3% of price (meaningful volatility)
  3. Current close > opening range high (breakout)
  4. Breakout bar volume > 1.5x average opening range volume
  5. Bullish candle on breakout bar (close > open)
  6. Close in upper half of breakout bar (sustained push, not wick)
  7. At least 2 of the last 3 bars closing above OR high (trend forming)

Conviction: 0.60-0.90 based on breakout magnitude, volume, bar strength.
Hold type: DAY (always — intraday pattern, costs 1 PDT slot).
"""

from __future__ import annotations

import pandas as pd

from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import BaseStrategy, HoldType, Signal

logger = get_logger(__name__)


class ORBStrategy(BaseStrategy):
    """Enter on opening range breakouts confirmed by volume and price action."""

    def evaluate(
        self,
        symbol: str,
        daily_bars: pd.DataFrame,
        intraday_bars: pd.DataFrame | None = None,
    ) -> Signal | None:
        if intraday_bars is None or intraday_bars.empty:
            return None

        opening_minutes: int = getattr(self.config, "opening_range_minutes", 30)
        min_vol_ratio: float = getattr(self.config, "min_volume_ratio", 1.5)
        min_range_pct: float = getattr(self.config, "min_range_pct", 0.3) / 100.0

        df = intraday_bars.copy()

        if len(df) < opening_minutes + 1:
            return None

        # Define opening range
        opening_range = df.iloc[:opening_minutes]
        or_high: float = float(opening_range["high"].max())
        or_low: float = float(opening_range["low"].min())
        or_range = or_high - or_low

        # Range must have meaningful width
        if or_range <= 0 or or_range / or_high < min_range_pct:
            return None

        avg_or_volume = opening_range["volume"].mean()
        if avg_or_volume <= 0:
            return None

        latest = df.iloc[-1]
        close: float = float(latest["close"])
        open_price: float = float(latest["open"])
        high: float = float(latest["high"])
        low: float = float(latest["low"])
        bar_volume: float = float(latest["volume"])

        # ── Hard filters ──

        # Breakout above opening range high
        if close <= or_high:
            return None

        # Volume confirms breakout
        vol_ratio = bar_volume / avg_or_volume if avg_or_volume > 0 else 0
        if vol_ratio < min_vol_ratio:
            return None

        # Bullish candle (buyers in control)
        if close <= open_price:
            return None

        # Close in upper half of bar (sustained push, not just a wick)
        bar_range = high - low
        if bar_range > 0 and (close - low) / bar_range < 0.5:
            return None

        # Trend forming: at least 2 of last 3 bars above OR high
        post_or = df.iloc[opening_minutes:]
        if len(post_or) >= 3:
            recent_3 = post_or.iloc[-3:]
            bars_above = (recent_3["close"] > or_high).sum()
            if bars_above < 2:
                return None
        # If fewer than 3 post-OR bars, just need the latest (already checked)

        # ── Conviction scoring (0.60-0.90) ──
        conviction = 0.60

        # +0.10: Breakout magnitude (how far above OR high as % of range)
        breakout_pct = (close - or_high) / or_range if or_range > 0 else 0
        conviction += min(0.10, breakout_pct * 0.10)

        # +0.08: Volume ratio strength
        vol_bonus = (vol_ratio - min_vol_ratio) / (3.0 - min_vol_ratio)
        conviction += 0.08 * min(1.0, max(0, vol_bonus))

        # +0.07: Bar strength (close position within bar)
        if bar_range > 0:
            bar_strength = (close - low) / bar_range
            conviction += 0.07 * min(1.0, bar_strength)

        # +0.05: Opening range tightness (tighter range = more explosive)
        range_pct = or_range / or_high
        if range_pct < 0.005:
            conviction += 0.05  # Very tight range
        elif range_pct < 0.01:
            conviction += 0.03

        conviction = max(0.60, min(0.90, conviction))

        entry_price = close
        stop_loss = (or_high + or_low) / 2.0
        take_profit = or_high + 2.0 * or_range

        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        if risk <= 0 or reward / risk < 1.5:
            return None

        logger.info(
            "orb_signal",
            symbol=symbol,
            or_high=or_high,
            or_low=or_low,
            or_range=round(or_range, 4),
            vol_ratio=round(vol_ratio, 2),
            conviction=conviction,
            entry=entry_price,
            stop=stop_loss,
            target=take_profit,
        )

        return Signal(
            symbol=symbol,
            direction="long",
            conviction=conviction,
            strategy_name="orb",
            hold_type=HoldType.DAY,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            metadata={
                "or_high": or_high,
                "or_low": or_low,
                "or_range": or_range,
                "volume_ratio": vol_ratio,
                "breakout_pct": round(breakout_pct, 4),
            },
        )

    def should_exit(
        self, symbol: str, bars: pd.DataFrame, entry_price: float
    ) -> bool:
        """Exit when price falls back into the opening range.

        In backtesting with daily bars, this always returns False since
        ORB is an intraday strategy -- bracket orders handle exits.
        """
        return False
