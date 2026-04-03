"""RSI mean-reversion strategy (swing trade — no PDT cost).

Theory:
  Prices revert to the mean.  When a stock dips in an uptrend to the lower
  Bollinger Band with RSI oversold, it's likely to bounce.  We add
  multi-indicator confirmation: volume capitulation (declining volume on
  the selloff), MACD histogram turning less negative (selling decelerating),
  and close still above EMA-50 (trend structure intact).

Entry conditions (ALL must be true):
  1. RSI < 38 (oversold)
  2. Close > EMA-20 * 0.97 (still near short-term uptrend)
  3. Close < BB_lower * 1.02 (at lower Bollinger Band)
  4. Close > EMA-50 * 0.95 (medium-term structure not broken)
  5. MACD histogram rising (less negative or turning positive)
  6. Volume declining over last 3 bars (capitulation pattern)

Conviction: additive across RSI depth, BB proximity, MACD turn, volume.
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
        rsi_oversold: float = getattr(self.config, "rsi_oversold", 38)
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

        # ── Hard filters ──

        # RSI oversold
        if rsi_val >= rsi_oversold:
            return None

        # Still near short-term uptrend (3% buffer)
        if close <= ema_20 * 0.97:
            return None

        # Near lower Bollinger Band (2% buffer)
        if close > bb_lower * 1.02:
            return None

        # Medium-term structure not broken (5% buffer for deep dips)
        if close <= ema_50 * 0.95:
            return None

        # MACD histogram rising (selling momentum decelerating)
        if len(df) >= 3:
            prev_macd = float(df.iloc[-2]["macd_hist"])
            prev2_macd = float(df.iloc[-3]["macd_hist"])
            macd_rising = macd_hist > prev_macd  # Less negative or turning positive
            macd_accel = prev_macd > prev2_macd  # Second derivative positive
        else:
            macd_rising = False
            macd_accel = False

        if not macd_rising:
            return None  # Selling still accelerating — don't catch the knife

        # Volume declining (capitulation pattern)
        if len(df) >= 3:
            vol_1 = float(df.iloc[-1]["volume"])
            vol_2 = float(df.iloc[-2]["volume"])
            vol_3 = float(df.iloc[-3]["volume"])
            volume_declining = vol_1 < vol_2 and vol_2 < vol_3
        else:
            volume_declining = False

        # ── Conviction scoring (additive, 0.55–0.85) ──
        conviction = 0.55

        # +0.10: RSI depth (more oversold = stronger reversion potential)
        rsi_depth = (rsi_oversold - rsi_val) / max(rsi_oversold - 20, 1)
        conviction += 0.10 * min(1.0, rsi_depth)

        # +0.07: Close near BB lower (statistical support)
        bb_distance = (bb_lower - close) / bb_lower if bb_lower > 0 else 0
        if bb_distance > 0:
            conviction += min(0.07, bb_distance * 3.5)

        # +0.05: MACD turning (less negative = selling decelerating)
        if macd_accel:
            conviction += 0.05  # Second derivative positive = strong turn
        elif macd_rising:
            conviction += 0.03

        # +0.05: Volume capitulation (declining volume = sellers exhausted)
        if volume_declining:
            conviction += 0.05

        # +0.03: EMA-20 > EMA-50 (uptrend still intact)
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
