"""MACD bullish divergence strategy (swing trade).

Theory:
  Bullish divergence occurs when price makes a lower low but the MACD
  histogram makes a higher low.  This signals that selling momentum is
  weakening even as price continues to fall — a powerful reversal signal.

  Complements the mean reversion strategy (which uses RSI + Bollinger Bands)
  by using a completely different indicator family (MACD).

Entry conditions (ALL must be true):
  1. Two swing lows found in last 20 bars (at least 3 bars apart, 1% price diff)
  2. Price low[recent] < price low[prior] (lower low in price)
  3. MACD hist at recent low > MACD hist at prior low (higher MACD low = divergence)
  4. Current MACD hist > 0 or just turned positive (divergence resolving)
  5. Close > EMA-20 * 0.97 (not in freefall)

Conviction: 0.55–0.80 based on divergence magnitude and EMA proximity.

Exit conditions:
  - MACD histogram turns negative (momentum reverting)
  - Close > upper Bollinger Band (target zone reached)

Hold type: SWING (divergences take 2-5 days to resolve).
"""

from __future__ import annotations

import pandas as pd

from ai_trade.data.indicators import add_atr, add_bollinger, add_ema, add_macd
from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import BaseStrategy, HoldType, Signal

logger = get_logger(__name__)


def _find_swing_lows(df: pd.DataFrame, lookback: int = 20, min_gap: int = 3) -> list[int]:
    """Find swing low indices in the last `lookback` bars.

    A swing low is a bar where low[i] < low[i-1] AND low[i] < low[i+1].
    Returns indices sorted by position (oldest first), at least `min_gap` bars apart.
    """
    start = max(1, len(df) - lookback)
    end = len(df) - 1  # Can't check i+1 for the last bar
    lows_data = df["low"].values

    swing_indices = []
    for i in range(start, end):
        if lows_data[i] < lows_data[i - 1] and lows_data[i] < lows_data[i + 1]:
            # Enforce minimum gap between swing lows
            if not swing_indices or (i - swing_indices[-1]) >= min_gap:
                swing_indices.append(i)

    return swing_indices


class MACDDivergenceStrategy(BaseStrategy):
    """Enter on bullish MACD divergence (price lower low, MACD higher low).

    Uses wider ATR-based stops (2.0x) because divergence entries can take
    time to resolve.  Exits when MACD histogram turns negative or price
    reaches the upper Bollinger Band.
    """

    def evaluate(
        self,
        symbol: str,
        daily_bars: pd.DataFrame,
        intraday_bars: pd.DataFrame | None = None,
    ) -> Signal | None:
        df = daily_bars.copy()

        # Config parameters
        lookback: int = getattr(self.config, "lookback_bars", 20)
        atr_stop_mult: float = getattr(self.config, "atr_stop_multiplier", 2.0)
        atr_tp_mult: float = getattr(self.config, "atr_tp_multiplier", 3.0)

        # Compute indicators
        add_macd(df)
        add_ema(df, [20])
        add_atr(df)
        add_bollinger(df)

        if len(df) < 30:
            return None

        latest = df.iloc[-1]
        close: float = latest["close"]
        ema_20: float = latest["ema_20"]
        macd_hist: float = latest["macd_hist"]
        atr: float = latest["atr_14"]

        # ── Entry condition 5 (check early): Not in freefall ──
        if close <= ema_20 * 0.97:
            return None

        # ── Entry condition 4: MACD hist positive or just turned positive ──
        prev_hist: float = df.iloc[-2]["macd_hist"]
        if macd_hist <= 0 and prev_hist <= 0:
            return None

        # ── Entry conditions 1-3: Find bullish divergence ──
        swing_indices = _find_swing_lows(df, lookback=lookback, min_gap=3)

        if len(swing_indices) < 2:
            return None

        # Use the two most recent swing lows
        prior_idx = swing_indices[-2]
        recent_idx = swing_indices[-1]

        prior_low: float = df.iloc[prior_idx]["low"]
        recent_low: float = df.iloc[recent_idx]["low"]
        prior_macd: float = df.iloc[prior_idx]["macd_hist"]
        recent_macd: float = df.iloc[recent_idx]["macd_hist"]

        # Require at least 1% price difference between lows (filter noise)
        if abs(recent_low - prior_low) / prior_low < 0.01:
            return None

        # ── Condition 2: Price makes lower low ──
        if recent_low >= prior_low:
            return None

        # ── Condition 3: MACD makes higher low (bullish divergence) ──
        if recent_macd <= prior_macd:
            return None

        # ── Conviction scoring (0.55–0.80) ──
        conviction = 0.55

        # +0.15 max: Divergence magnitude (bigger MACD difference = stronger signal)
        macd_diff = recent_macd - prior_macd
        macd_range = max(abs(prior_macd), 0.01)
        divergence_strength = min(1.0, abs(macd_diff) / macd_range)
        conviction += 0.15 * divergence_strength

        # +0.10 max: Proximity to EMA-20 (closer = better support)
        ema_distance_pct = abs(close - ema_20) / ema_20
        if ema_distance_pct < 0.01:
            conviction += 0.10
        elif ema_distance_pct < 0.02:
            conviction += 0.05

        conviction = max(0.55, min(0.80, conviction))

        entry_price = close
        stop_loss = entry_price - atr_stop_mult * atr
        take_profit = entry_price + atr_tp_mult * atr

        if stop_loss >= entry_price or take_profit <= entry_price:
            return None

        logger.info(
            "macd_divergence_signal",
            symbol=symbol,
            prior_low=prior_low,
            recent_low=recent_low,
            prior_macd=prior_macd,
            recent_macd=recent_macd,
            macd_hist=macd_hist,
            conviction=conviction,
            entry=entry_price,
            stop=stop_loss,
            target=take_profit,
        )

        return Signal(
            symbol=symbol,
            direction="long",
            conviction=conviction,
            strategy_name="macd_divergence",
            hold_type=HoldType.SWING,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            metadata={
                "prior_low": prior_low,
                "recent_low": recent_low,
                "prior_macd_hist": prior_macd,
                "recent_macd_hist": recent_macd,
                "divergence_strength": round(divergence_strength, 3),
                "atr": atr,
            },
        )

    def should_exit(
        self, symbol: str, bars: pd.DataFrame, entry_price: float
    ) -> bool:
        """Exit when MACD histogram turns negative or price hits upper BB."""
        df = bars.copy()
        add_macd(df)
        add_bollinger(df)

        latest = df.iloc[-1]
        macd_hist: float = latest["macd_hist"]
        close: float = latest["close"]
        bb_upper: float = latest["bb_upper"]

        if macd_hist < 0 or close > bb_upper:
            logger.info(
                "macd_divergence_exit",
                symbol=symbol,
                macd_hist=macd_hist,
                close=close,
                bb_upper=bb_upper,
            )
            return True

        return False
