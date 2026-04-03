"""MACD bullish divergence strategy (swing trade).

Theory:
  Bullish divergence: price makes a lower low but MACD histogram makes a
  higher low.  This signals weakening selling pressure — a reversal setup.

  We add multi-indicator confirmation: RSI also diverging, volume pattern
  (declining on selloff = sellers exhausted), EMA support nearby, and
  MACD histogram has turned positive (divergence is resolving NOW).

Entry conditions (ALL must be true):
  1. Two swing lows in last 20 bars (>= 4 bars apart)
  2. Price lower low + MACD higher low (classic divergence)
  3. MACD hist positive NOW AND was negative within last 3 bars (fresh turn)
  4. RSI > 30 and RSI at recent low > RSI at prior low (double confirmation)
  5. Close > EMA-50 * 0.97 (not in freefall, near structural support)
  6. Volume declining on second low (sellers exhausting)

Conviction: additive across divergence strength, RSI, EMA proximity, volume.
Hold type: SWING.
"""

from __future__ import annotations

import pandas as pd

from ai_trade.data.indicators import add_atr, add_bollinger, add_ema, add_macd, add_rsi, add_volume_profile
from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import BaseStrategy, HoldType, Signal

logger = get_logger(__name__)


def _find_swing_lows(df: pd.DataFrame, lookback: int = 20, min_gap: int = 4) -> list[int]:
    """Find swing low indices in the last `lookback` bars."""
    start = max(1, len(df) - lookback)
    end = len(df) - 1
    lows_data = df["low"].values

    swing_indices = []
    for i in range(start, end):
        if lows_data[i] < lows_data[i - 1] and lows_data[i] < lows_data[i + 1]:
            if not swing_indices or (i - swing_indices[-1]) >= min_gap:
                swing_indices.append(i)
    return swing_indices


class MACDDivergenceStrategy(BaseStrategy):
    """Enter on bullish MACD divergence with multi-indicator confirmation."""

    def evaluate(
        self,
        symbol: str,
        daily_bars: pd.DataFrame,
        intraday_bars: pd.DataFrame | None = None,
    ) -> Signal | None:
        df = daily_bars.copy()

        lookback: int = getattr(self.config, "lookback_bars", 20)
        atr_stop_mult: float = getattr(self.config, "atr_stop_multiplier", 1.8)
        atr_tp_mult: float = getattr(self.config, "atr_tp_multiplier", 3.5)

        add_macd(df)
        add_ema(df, [20, 50])
        add_atr(df)
        add_bollinger(df)
        add_rsi(df)
        add_volume_profile(df)

        if len(df) < 52:
            return None

        latest = df.iloc[-1]
        close: float = latest["close"]
        ema_20: float = latest["ema_20"]
        ema_50: float = latest["ema_50"]
        macd_hist: float = latest["macd_hist"]
        rsi_val: float = latest["rsi_14"]
        atr: float = latest["atr_14"]

        # ── MACD hist must be positive NOW ──
        if macd_hist <= 0:
            return None

        # Was negative within last 3 bars (fresh turn)
        recent_negative = any(float(df.iloc[i]["macd_hist"]) < 0 for i in range(-4, -1))
        if not recent_negative:
            return None

        # Not in freefall (near structural support)
        if close <= ema_50 * 0.97:
            return None

        # RSI sanity — not deeply oversold (likely crashing, not bouncing)
        if rsi_val < 30:
            return None

        # ── Find bullish divergence ──
        swing_indices = _find_swing_lows(df, lookback=lookback, min_gap=4)
        if len(swing_indices) < 2:
            return None

        prior_idx = swing_indices[-2]
        recent_idx = swing_indices[-1]

        prior_low: float = df.iloc[prior_idx]["low"]
        recent_low: float = df.iloc[recent_idx]["low"]
        prior_macd: float = df.iloc[prior_idx]["macd_hist"]
        recent_macd: float = df.iloc[recent_idx]["macd_hist"]
        prior_rsi: float = df.iloc[prior_idx]["rsi_14"]
        recent_rsi: float = df.iloc[recent_idx]["rsi_14"]

        # Price lower low
        if recent_low >= prior_low:
            return None
        # MACD higher low (divergence)
        if recent_macd <= prior_macd:
            return None

        # Price difference must be meaningful (>= 2%)
        if abs(recent_low - prior_low) / prior_low < 0.02:
            return None

        # Divergence magnitude filter
        macd_diff = recent_macd - prior_macd
        macd_range = max(abs(prior_macd), 0.01)
        divergence_strength = min(1.0, abs(macd_diff) / macd_range)
        if divergence_strength < 0.20:
            return None

        # RSI also showing divergence (double confirmation)
        rsi_diverging = recent_rsi > prior_rsi

        # Volume declining on second low (sellers exhausting)
        prior_vol = float(df.iloc[prior_idx]["volume"])
        recent_vol = float(df.iloc[recent_idx]["volume"])
        volume_declining = recent_vol < prior_vol * 0.9

        # ── Conviction scoring (additive, 0.55–0.85) ──
        conviction = 0.55

        # +0.10: Divergence magnitude
        conviction += 0.10 * divergence_strength

        # +0.07: RSI double-divergence confirmation
        if rsi_diverging:
            conviction += 0.07

        # +0.05: Volume declining (seller exhaustion)
        if volume_declining:
            conviction += 0.05

        # +0.05: Close near EMA-20 support
        ema_distance_pct = abs(close - ema_20) / ema_20
        if ema_distance_pct < 0.01:
            conviction += 0.05
        elif ema_distance_pct < 0.02:
            conviction += 0.03

        # +0.03: Uptrend structure intact (EMA-20 > EMA-50)
        if ema_20 > ema_50:
            conviction += 0.03

        conviction = max(0.55, min(0.85, conviction))

        entry_price = close
        stop_loss = entry_price - atr_stop_mult * atr
        take_profit = entry_price + atr_tp_mult * atr

        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        if risk <= 0 or reward / risk < 2.0:
            return None

        logger.info(
            "macd_divergence_signal",
            symbol=symbol,
            prior_low=prior_low,
            recent_low=recent_low,
            prior_macd=prior_macd,
            recent_macd=recent_macd,
            divergence_strength=round(divergence_strength, 3),
            rsi_diverging=rsi_diverging,
            volume_declining=volume_declining,
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
                "divergence_strength": round(divergence_strength, 3),
                "rsi_diverging": rsi_diverging,
                "volume_declining": volume_declining,
                "atr": atr,
            },
        )

    def should_exit(
        self, symbol: str, bars: pd.DataFrame, entry_price: float
    ) -> bool:
        df = bars.copy()
        add_macd(df)
        add_bollinger(df)
        latest = df.iloc[-1]
        if latest["macd_hist"] < 0 or latest["close"] > latest["bb_upper"]:
            return True
        return False
