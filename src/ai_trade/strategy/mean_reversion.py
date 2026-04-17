"""RSI mean-reversion strategy (swing trade — no PDT cost).

Theory:
  Prices revert to the mean.  When a stock dips in an uptrend to the lower
  Bollinger Band with RSI oversold, it's likely to bounce.  We add
  multi-indicator confirmation: volume capitulation (declining volume on
  the selloff), MACD histogram turning less negative (selling decelerating),
  and close still above EMA-50 (trend structure intact).

Entry conditions (ALL must be true):
  1. RSI < 42 (oversold — widened from 38)
  2. Close > EMA-20 * 0.96 (still near short-term uptrend, 4% buffer)
  3. Close < BB_lower * 1.03 (near lower Bollinger Band, 3% buffer)
  4. Close > EMA-50 * 0.93 (medium-term structure not broken, 7% buffer)
  5. MACD histogram rising (less negative or turning positive)

Conviction: additive across RSI depth, BB proximity, MACD turn, volume,
  EMA structure. Volume capitulation is a conviction factor, not hard gate.
Hold type: SWING.
"""

from __future__ import annotations

import pandas as pd

from ai_trade.data.indicators import add_atr, add_bollinger, add_ema, add_macd, add_rsi, add_volume_profile
from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import BaseStrategy, HoldType, Signal

logger = get_logger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """Buy oversold dips with multi-indicator confluence."""

    def evaluate(
        self,
        symbol: str,
        daily_bars: pd.DataFrame,
        intraday_bars: pd.DataFrame | None = None,
    ) -> Signal | None:
        df = daily_bars.copy()

        rsi_period: int = getattr(self.config, "rsi_period", 14)
        rsi_oversold: float = getattr(self.config, "rsi_oversold", 42)
        atr_stop_mult: float = getattr(self.config, "atr_stop_multiplier", 1.5)
        atr_tp_mult: float = getattr(self.config, "atr_tp_multiplier", 3.5)

        add_rsi(df, rsi_period)
        add_ema(df, [20, 50])
        add_atr(df)
        add_bollinger(df)
        add_macd(df)
        add_volume_profile(df)

        if len(df) < 52:
            return None

        latest = df.iloc[-1]
        rsi_col = f"rsi_{rsi_period}"

        rsi_val: float = latest[rsi_col]
        close: float = latest["close"]
        ema_20: float = latest["ema_20"]
        ema_50: float = latest["ema_50"]
        bb_lower: float = latest["bb_lower"]
        atr: float = latest["atr_14"]
        macd_hist: float = latest["macd_hist"]

        # ── Hard filters (relaxed) ──

        # RSI oversold
        if rsi_val >= rsi_oversold:
            self._reject(symbol, "rsi_oversold", rsi_val, rsi_oversold, "below")
            return None

        # Still near short-term uptrend (4% buffer — widened from 3%)
        ema20_floor = ema_20 * 0.96
        if close <= ema20_floor:
            self._reject(symbol, "close_above_ema20_96pct", close, ema20_floor, "above")
            return None

        # Near lower Bollinger Band (3% buffer — widened from 2%)
        bb_ceil = bb_lower * 1.03
        if close > bb_ceil:
            self._reject(symbol, "close_near_bb_lower", close, bb_ceil, "below")
            return None

        # Medium-term structure not broken (7% buffer — widened from 5%)
        ema50_floor = ema_50 * 0.93
        if close <= ema50_floor:
            self._reject(symbol, "close_above_ema50_93pct", close, ema50_floor, "above")
            return None

        # MACD histogram rising (selling momentum decelerating)
        if len(df) >= 3:
            prev_macd = float(df.iloc[-2]["macd_hist"])
            prev2_macd = float(df.iloc[-3]["macd_hist"])
            macd_rising = macd_hist > prev_macd
            macd_accel = prev_macd > prev2_macd
        else:
            macd_rising = False
            macd_accel = False

        if not macd_rising:
            self._reject(symbol, "macd_rising", macd_hist, prev_macd if len(df) >= 3 else 0.0, "above")
            return None

        # Volume declining (conviction factor, not hard gate)
        if len(df) >= 3:
            vol_1 = float(df.iloc[-1]["volume"])
            vol_2 = float(df.iloc[-2]["volume"])
            vol_3 = float(df.iloc[-3]["volume"])
            volume_declining = vol_1 < vol_2 and vol_2 < vol_3
        else:
            volume_declining = False

        # ── Conviction scoring (additive, 0.50–0.90) ──
        conviction = 0.50

        # +0.12: RSI depth (more oversold = stronger reversion potential)
        rsi_depth = (rsi_oversold - rsi_val) / max(rsi_oversold - 20, 1)
        conviction += 0.12 * min(1.0, rsi_depth)

        # +0.08: Close near BB lower (statistical support)
        bb_distance = (bb_lower - close) / bb_lower if bb_lower > 0 else 0
        if bb_distance > 0:
            conviction += min(0.08, bb_distance * 4.0)

        # +0.07: MACD turning (less negative = selling decelerating)
        if macd_accel:
            conviction += 0.07  # Second derivative positive = strong turn
        elif macd_rising:
            conviction += 0.04

        # +0.06: Volume capitulation (declining volume = sellers exhausted)
        if volume_declining:
            conviction += 0.06
        else:
            conviction += 0.01  # Partial credit if volume not declining

        # +0.05: EMA-20 > EMA-50 (uptrend still intact)
        if ema_20 > ema_50:
            conviction += 0.05
        elif ema_20 > ema_50 * 0.99:
            conviction += 0.02  # EMAs nearly flat — still okay

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
            "mean_reversion_signal",
            symbol=symbol,
            rsi=rsi_val,
            conviction=conviction,
            entry=entry_price,
            stop=stop_loss,
            target=take_profit,
            macd_rising=macd_rising,
            volume_declining=volume_declining,
            atr=atr,
        )

        return Signal(
            symbol=symbol,
            direction="long",
            conviction=conviction,
            strategy_name="mean_reversion",
            hold_type=HoldType.SWING,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            metadata={
                "rsi": rsi_val,
                "ema_20": ema_20,
                "ema_50": ema_50,
                "bb_lower": bb_lower,
                "macd_hist": macd_hist,
                "macd_rising": macd_rising,
                "volume_declining": volume_declining,
                "atr": atr,
            },
        )

    def should_exit(
        self, symbol: str, bars: pd.DataFrame, entry_price: float
    ) -> bool:
        df = bars.copy()
        rsi_period: int = getattr(self.config, "rsi_period", 14)
        add_rsi(df, rsi_period)
        add_bollinger(df)

        latest = df.iloc[-1]
        rsi_val: float = latest[f"rsi_{rsi_period}"]
        close: float = latest["close"]
        bb_upper: float = latest["bb_upper"]

        if rsi_val > 60 or close > bb_upper:
            return True
        return False
