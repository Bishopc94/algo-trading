"""Opening Range Breakout (ORB) strategy (day trade).

Theory:
  The first 30 minutes of trading establishes a range that reflects the
  initial battle between buyers and sellers.  A breakout above that range
  on strong volume predicts the intraday trend direction.

  This is a classic day-trade strategy and the second intraday strategy
  alongside VWAP.  Unlike VWAP (which trades reclaims of a fair-value
  line), ORB trades directional breakouts from a defined price range.

Entry conditions (ALL must be true):
  1. Intraday (minute) bars available
  2. At least 30 minutes of trading data
  3. Current close > opening range high
  4. Breakout bar volume > 1.5x average opening range volume

Conviction: 0.60–0.90 based on breakout magnitude and volume.

Exit conditions:
  - Close < opening range high (fell back into range)

Hold type: DAY (always — intraday pattern, costs 1 PDT slot).
"""

from __future__ import annotations

import pandas as pd

from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import BaseStrategy, HoldType, Signal

logger = get_logger(__name__)


class ORBStrategy(BaseStrategy):
    """Enter on opening range breakouts confirmed by volume.

    Stop at the opening range midpoint.  Target is 2x the range height
    above the breakout level.  All exits handled by bracket orders.
    """

    def evaluate(
        self,
        symbol: str,
        daily_bars: pd.DataFrame,
        intraday_bars: pd.DataFrame | None = None,
    ) -> Signal | None:
        # ── Requires intraday bars ──
        if intraday_bars is None or intraday_bars.empty:
            return None

        # Config parameters
        opening_minutes: int = getattr(self.config, "opening_range_minutes", 30)
        min_vol_ratio: float = getattr(self.config, "min_volume_ratio", 1.5)

        df = intraday_bars.copy()

        # ── Entry condition 1: Enough data for opening range ──
        if len(df) < opening_minutes + 1:
            return None

        # Define the opening range from the first N bars
        opening_range = df.iloc[:opening_minutes]
        or_high: float = float(opening_range["high"].max())
        or_low: float = float(opening_range["low"].min())
        or_range = or_high - or_low

        # Sanity check: range must have meaningful width
        if or_range <= 0 or or_range / or_high < 0.001:
            return None

        # Average volume during the opening range
        avg_or_volume = opening_range["volume"].mean()
        if avg_or_volume <= 0:
            return None

        # Look at the most recent bar (post opening range)
        latest = df.iloc[-1]
        close: float = float(latest["close"])
        bar_volume: float = float(latest["volume"])

        # ── Entry condition 3: Breakout above opening range high ──
        if close <= or_high:
            return None

        # ── Entry condition 4: Volume confirms the breakout ──
        vol_ratio = bar_volume / avg_or_volume if avg_or_volume > 0 else 0
        if vol_ratio < min_vol_ratio:
            logger.debug("orb_reject", symbol=symbol, reason="low_volume",
                         vol_ratio=vol_ratio, threshold=min_vol_ratio)
            return None

        # ── Conviction scoring (0.60–0.90) ──
        conviction = 0.60

        # +0.15 max: Breakout magnitude (how far above or_high, as % of range)
        breakout_pct = (close - or_high) / or_range if or_range > 0 else 0
        conviction += min(0.15, breakout_pct * 0.15)

        # +0.15 max: Volume ratio strength
        vol_bonus = (vol_ratio - min_vol_ratio) / (3.0 - min_vol_ratio)
        conviction += 0.15 * min(1.0, max(0, vol_bonus))

        conviction = max(0.60, min(0.90, conviction))

        entry_price = close

        # Stop at range midpoint (breakout failed if back to middle)
        stop_loss = (or_high + or_low) / 2.0

        # Target: 2x the opening range height above the breakout level
        take_profit = or_high + 2.0 * or_range

        if stop_loss >= entry_price or take_profit <= entry_price:
            return None

        logger.info(
            "orb_signal",
            symbol=symbol,
            or_high=or_high,
            or_low=or_low,
            or_range=or_range,
            vol_ratio=vol_ratio,
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
            },
        )

    def should_exit(
        self, symbol: str, bars: pd.DataFrame, entry_price: float
    ) -> bool:
        """Exit when price falls back into the opening range.

        In backtesting with daily bars, this always returns False since
        ORB is an intraday strategy — bracket orders handle exits.
        """
        return False
