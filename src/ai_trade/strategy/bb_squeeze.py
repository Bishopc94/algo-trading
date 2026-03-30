"""Bollinger Band squeeze breakout strategy (adaptive hold type).

Theory:
  When Bollinger Band width contracts (volatility compression), it signals
  that a big move is imminent.  When the bands expand and price breaks out
  above the upper band on volume, it marks the start of a new trend.

  This is the opposite of mean reversion (which buys at the bands) — this
  strategy buys when price breaks through the bands after a squeeze.

Entry conditions (ALL must be true):
  1. BB width was below 0.08 within the last 5 bars (recent squeeze)
  2. Current BB width is expanding (current > previous bar)
  3. Close > BB upper (breakout above the upper band)
  4. Relative volume > 1.3 (volume confirms the breakout)

Conviction: 0.55–0.85 based on squeeze depth and volume.

Exit conditions:
  - Close < BB middle (retreated to mean — breakout failed)

Hold type: ADAPTIVE (high conviction → DAY, else → SWING).
"""

from __future__ import annotations

import pandas as pd

from ai_trade.data.indicators import add_atr, add_bollinger, add_volume_profile
from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import BaseStrategy, HoldType, Signal

logger = get_logger(__name__)


class BBSqueezeStrategy(BaseStrategy):
    """Enter on Bollinger Band squeeze breakouts confirmed by volume.

    Uses BB middle (20-SMA) as stop level — if price falls back to the
    mean, the breakout failed.  Target is ATR-based.
    """

    def evaluate(
        self,
        symbol: str,
        daily_bars: pd.DataFrame,
        intraday_bars: pd.DataFrame | None = None,
    ) -> Signal | None:
        df = daily_bars.copy()

        # Config parameters
        squeeze_threshold: float = getattr(self.config, "squeeze_threshold", 0.08)
        squeeze_lookback: int = getattr(self.config, "squeeze_lookback", 5)
        min_rel_vol: float = getattr(self.config, "min_relative_volume", 1.3)
        atr_tp_mult: float = getattr(self.config, "atr_tp_multiplier", 3.0)

        # Compute indicators
        add_bollinger(df)
        add_volume_profile(df)
        add_atr(df)

        if len(df) < 22:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        close: float = latest["close"]
        bb_upper: float = latest["bb_upper"]
        bb_middle: float = latest["bb_middle"]
        bb_width: float = latest["bb_width"]
        prev_bb_width: float = prev["bb_width"]
        rel_vol: float = latest.get("relative_volume", 1.0)
        atr: float = latest["atr_14"]

        # ── Entry condition 1: Recent squeeze (BB width below threshold in last N bars) ──
        lookback_start = max(0, len(df) - squeeze_lookback - 1)
        recent_widths = df["bb_width"].iloc[lookback_start:-1]  # Exclude current bar

        if recent_widths.empty or recent_widths.min() >= squeeze_threshold:
            return None

        # ── Entry condition 2: Bands expanding ──
        if bb_width <= prev_bb_width:
            logger.debug("bb_squeeze_reject", symbol=symbol, reason="not_expanding",
                         width=bb_width, prev_width=prev_bb_width)
            return None

        # ── Entry condition 3: Breakout above upper band ──
        if close <= bb_upper:
            logger.debug("bb_squeeze_reject", symbol=symbol, reason="not_above_upper_bb",
                         close=close, bb_upper=bb_upper)
            return None

        # ── Entry condition 4: Volume confirmation ──
        if rel_vol < min_rel_vol:
            logger.debug("bb_squeeze_reject", symbol=symbol, reason="low_volume",
                         rel_vol=rel_vol, threshold=min_rel_vol)
            return None

        # ── Conviction scoring (0.55–0.85) ──
        conviction = 0.55

        # +0.15 max: Squeeze depth (tighter squeeze = stronger breakout potential)
        min_width = recent_widths.min()
        squeeze_depth = max(0, squeeze_threshold - min_width) / squeeze_threshold
        conviction += 0.15 * min(1.0, squeeze_depth)

        # +0.15 max: Volume strength (1.3x→0, 3x+→max)
        if rel_vol > min_rel_vol:
            vol_bonus = (rel_vol - min_rel_vol) / (3.0 - min_rel_vol)
            conviction += 0.15 * min(1.0, max(0, vol_bonus))

        conviction = max(0.55, min(0.85, conviction))

        entry_price = close

        # Stop at BB middle (20-SMA) — breakout failed if price returns to mean
        stop_loss = bb_middle
        take_profit = entry_price + atr_tp_mult * atr

        if stop_loss >= entry_price or take_profit <= entry_price:
            return None

        # Adaptive hold type
        hold_type = HoldType.DAY if conviction >= 0.9 else HoldType.SWING

        logger.info(
            "bb_squeeze_signal",
            symbol=symbol,
            bb_width=bb_width,
            min_squeeze_width=min_width,
            rel_vol=rel_vol,
            conviction=conviction,
            entry=entry_price,
            stop=stop_loss,
            target=take_profit,
            hold_type=hold_type.value,
        )

        return Signal(
            symbol=symbol,
            direction="long",
            conviction=conviction,
            strategy_name="bb_squeeze",
            hold_type=hold_type,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            metadata={
                "bb_width": bb_width,
                "min_squeeze_width": min_width,
                "relative_volume": rel_vol,
                "bb_middle": bb_middle,
                "bb_upper": bb_upper,
                "atr": atr,
            },
        )

    def should_exit(
        self, symbol: str, bars: pd.DataFrame, entry_price: float
    ) -> bool:
        """Exit when price retreats below BB middle (breakout failed)."""
        df = bars.copy()
        add_bollinger(df)

        latest = df.iloc[-1]
        close: float = latest["close"]
        bb_middle: float = latest["bb_middle"]

        if close < bb_middle:
            logger.info(
                "bb_squeeze_exit",
                symbol=symbol,
                close=close,
                bb_middle=bb_middle,
            )
            return True

        return False
