"""MACD bullish divergence strategy (swing trade).

Theory:
  Bullish divergence: price makes a lower low but MACD histogram makes a
  higher low.  This signals weakening selling pressure — a reversal setup.

  We add multi-indicator confirmation: RSI also diverging, volume pattern
  (declining on selloff = sellers exhausted), EMA support nearby, and
  MACD histogram has turned positive (divergence is resolving NOW).

Entry conditions (ALL must be true):
  1. Two swing lows in last 25 bars (>= 3 bars apart — relaxed)
  2. Price lower low + MACD higher low (classic divergence)
  3. MACD hist positive NOW AND was negative within last 3 bars (fresh turn)
  4. RSI > 25 (not in total freefall)
  5. Close > EMA-50 * 0.95 (near structural support — widened from 97%)

Conviction: additive across divergence strength, RSI divergence, volume,
  EMA proximity. Volume/RSI divergence are conviction factors, not hard gates.
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
            self._reject(symbol, "macd_hist_positive", macd_hist, 0.0, "above")
            return None

        # Was negative within last 3 bars (fresh turn)
        recent_negative = any(float(df.iloc[i]["macd_hist"]) < 0 for i in range(-4, -1))
        if not recent_negative:
            self._reject(symbol, "recent_macd_negative", 0.0, 1.0, "above")
            return None

        # Not in freefall (near structural support — widened from 97% to 95%)
        ema50_floor = ema_50 * 0.95
        if close <= ema50_floor:
            self._reject(symbol, "close_above_ema50_95pct", close, ema50_floor, "above")
            return None

        # RSI sanity — relaxed from 30 to 25
        if rsi_val < 25:
            self._reject(symbol, "rsi_min", rsi_val, 25.0, "above")
            return None

        # ── Find bullish divergence (relaxed: 25-bar lookback, 3-bar min gap) ──
        swing_indices = _find_swing_lows(df, lookback=max(lookback, 25), min_gap=3)
        if len(swing_indices) < 2:
            self._reject(symbol, "swing_lows_found", float(len(swing_indices)), 2.0, "above")
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
            self._reject(symbol, "price_lower_low", recent_low, prior_low, "below")
            return None
        # MACD higher low (divergence)
        if recent_macd <= prior_macd:
            self._reject(symbol, "macd_higher_low", recent_macd, prior_macd, "above")
            return None

        # Price difference must be meaningful (relaxed from 2% to 1%)
        price_diff_pct = abs(recent_low - prior_low) / prior_low
        if price_diff_pct < 0.01:
            self._reject(symbol, "price_diff_pct", price_diff_pct, 0.01, "above")
            return None

        # Divergence magnitude filter (relaxed from 20% to 10%)
        macd_diff = recent_macd - prior_macd
        macd_range = max(abs(prior_macd), 0.01)
        divergence_strength = min(1.0, abs(macd_diff) / macd_range)
        if divergence_strength < 0.10:
            self._reject(symbol, "divergence_strength", divergence_strength, 0.10, "above")
            return None

        # RSI also showing divergence (conviction factor, not hard gate)
        rsi_diverging = recent_rsi > prior_rsi

        # Volume declining on second low (conviction factor, not hard gate)
        prior_vol = float(df.iloc[prior_idx]["volume"])
        recent_vol = float(df.iloc[recent_idx]["volume"])
        volume_declining = recent_vol < prior_vol * 0.9

        # ── Conviction scoring (additive, 0.50–0.90) ──
        conviction = 0.50

        # +0.12: Divergence magnitude
        conviction += 0.12 * divergence_strength

        # +0.08: RSI double-divergence confirmation
        if rsi_diverging:
            conviction += 0.08

        # +0.06: Volume declining (seller exhaustion)
        if volume_declining:
            conviction += 0.06
        else:
            conviction += 0.01  # Partial credit

        # +0.06: Close near EMA-20 support
        ema_distance_pct = abs(close - ema_20) / ema_20
        if ema_distance_pct < 0.01:
            conviction += 0.06
        elif ema_distance_pct < 0.02:
            conviction += 0.04
        elif ema_distance_pct < 0.03:
            conviction += 0.02

        # +0.05: Uptrend structure intact (EMA-20 > EMA-50)
        if ema_20 > ema_50:
            conviction += 0.05
        elif ema_20 > ema_50 * 0.99:
            conviction += 0.02  # Nearly flat — still okay

        conviction = max(0.50, min(0.90, conviction))

        entry_price = close
        levels = self._plan_long_exit(
            bars=df, entry_price=entry_price, atr=atr,
            base_stop_mult=atr_stop_mult, base_tp_mult=atr_tp_mult,
        )
        stop_loss = levels.stop_loss
        take_profit = levels.take_profit

        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        rr = reward / risk if risk > 0 else 0.0
        if risk <= 0 or rr < 2.0:
            self._reject(symbol, "risk_reward", rr, 2.0, "above")
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
