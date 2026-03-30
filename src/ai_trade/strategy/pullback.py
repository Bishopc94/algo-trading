"""Pullback-to-support strategy (swing trade).

Theory:
  In established uptrends (EMA-20 > EMA-50), buying pullbacks to key
  moving averages offers high-probability entries.  Different from mean
  reversion (which buys extreme oversold dips at the Bollinger lower band)
  — this buys healthy dips in strong trends before they become extreme.

Entry conditions (ALL must be true):
  1. EMA-20 > EMA-50 (uptrend structure)
  2. Close within 1% of EMA-20 or EMA-50 (pulled back to support)
  3. 40 < RSI < 55 (pulled back but not extreme)
  4. Close > EMA-50 (hasn't broken medium-term trend)

Conviction: 0.50–0.85 based on trend strength, RSI position, volume.

Exit conditions (any one triggers):
  - Close < EMA-50 (uptrend broken)
  - RSI > 70 (overbought, time to take profit)

Hold type: SWING (pullbacks need 2-5 days to resolve).
"""

from __future__ import annotations

import pandas as pd

from ai_trade.data.indicators import add_atr, add_ema, add_rsi, add_volume_profile
from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import BaseStrategy, HoldType, Signal

logger = get_logger(__name__)


class PullbackStrategy(BaseStrategy):
    """Buy pullbacks to EMA support in established uptrends.

    Uses ATR-based stops below medium-term support.  Exits when the
    uptrend breaks (close < EMA-50) or becomes overbought (RSI > 70).
    """

    def evaluate(
        self,
        symbol: str,
        daily_bars: pd.DataFrame,
        intraday_bars: pd.DataFrame | None = None,
    ) -> Signal | None:
        df = daily_bars.copy()

        # Config parameters
        tolerance_pct: float = getattr(self.config, "pullback_tolerance_pct", 1.0) / 100.0
        rsi_min: float = getattr(self.config, "rsi_min", 40)
        rsi_max: float = getattr(self.config, "rsi_max", 55)
        atr_stop_mult: float = getattr(self.config, "atr_stop_multiplier", 0.5)
        atr_tp_mult: float = getattr(self.config, "atr_tp_multiplier", 2.5)

        # Compute indicators
        add_ema(df, [20, 50])
        add_rsi(df)
        add_atr(df)
        add_volume_profile(df)

        if len(df) < 52:
            return None

        latest = df.iloc[-1]
        close: float = latest["close"]
        ema_20: float = latest["ema_20"]
        ema_50: float = latest["ema_50"]
        rsi_val: float = latest["rsi_14"]
        atr: float = latest["atr_14"]
        rel_vol: float = latest.get("relative_volume", 1.0)

        # ── Entry condition 1: Uptrend structure ──
        if ema_20 <= ema_50:
            return None

        # ── Entry condition 2: Pulled back to support (within tolerance of EMA-20 or EMA-50) ──
        near_ema20 = abs(close - ema_20) / ema_20 <= tolerance_pct
        near_ema50 = abs(close - ema_50) / ema_50 <= tolerance_pct

        if not (near_ema20 or near_ema50):
            logger.debug("pullback_reject", symbol=symbol, reason="not_near_ema",
                         close=close, ema_20=ema_20, ema_50=ema_50)
            return None

        # ── Entry condition 3: RSI in pullback range ──
        if rsi_val <= rsi_min or rsi_val >= rsi_max:
            logger.debug("pullback_reject", symbol=symbol, reason="rsi_out_of_range",
                         rsi=rsi_val, min=rsi_min, max=rsi_max)
            return None

        # ── Entry condition 4: Close above medium-term trend ──
        if close <= ema_50:
            logger.debug("pullback_reject", symbol=symbol, reason="below_ema50",
                         close=close, ema_50=ema_50)
            return None

        # ── Conviction scoring (0.50–0.85) ──
        conviction = 0.50

        # +0.15 max: Strong uptrend (EMA-20/EMA-50 gap > 2%)
        ema_gap_pct = (ema_20 - ema_50) / ema_50
        if ema_gap_pct > 0.02:
            conviction += min(0.15, (ema_gap_pct - 0.02) * 7.5 + 0.05)
        elif ema_gap_pct > 0.01:
            conviction += 0.05

        # +0.10 max: RSI in ideal pullback zone (45-50)
        if 45 <= rsi_val <= 50:
            conviction += 0.10
        elif 40 < rsi_val < 45 or 50 < rsi_val < 55:
            conviction += 0.05

        # +0.10 max: Buying interest on pullback (relative volume > 1.0)
        if rel_vol > 1.0:
            conviction += min(0.10, (rel_vol - 1.0) * 0.10)

        conviction = max(0.50, min(0.85, conviction))

        entry_price = close

        # Stop below medium-term support (EMA-50 minus buffer)
        stop_loss = ema_50 - atr_stop_mult * atr
        take_profit = entry_price + atr_tp_mult * atr

        if stop_loss >= entry_price or take_profit <= entry_price:
            return None

        logger.info(
            "pullback_signal",
            symbol=symbol,
            ema_20=ema_20,
            ema_50=ema_50,
            rsi=rsi_val,
            rel_vol=rel_vol,
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
                "atr": atr,
                "near_ema20": near_ema20,
                "near_ema50": near_ema50,
            },
        )

    def should_exit(
        self, symbol: str, bars: pd.DataFrame, entry_price: float
    ) -> bool:
        """Exit when uptrend breaks or RSI becomes overbought."""
        df = bars.copy()
        add_ema(df, [50])
        add_rsi(df)

        latest = df.iloc[-1]
        close: float = latest["close"]
        ema_50: float = latest["ema_50"]
        rsi_val: float = latest["rsi_14"]

        if close < ema_50 or rsi_val > 70:
            logger.info(
                "pullback_exit",
                symbol=symbol,
                close=close,
                ema_50=ema_50,
                rsi=rsi_val,
            )
            return True

        return False
