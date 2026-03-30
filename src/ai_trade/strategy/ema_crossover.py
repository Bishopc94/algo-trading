"""EMA crossover trend-following strategy (swing trade).

Theory:
  When the fast EMA (9) crosses above the slow EMA (20) in an established
  uptrend (price above EMA-50), it signals the start of a sustained trend
  continuation.  Unlike the momentum strategy (which requires a 20-day high
  breakout + volume spike), this catches smooth trend continuations where
  the moving averages confirm directional momentum.

Entry conditions (ALL must be true):
  1. EMA-9 > EMA-20 (bullish crossover)
  2. Previous bar had EMA-9 <= EMA-20 (crossover is fresh)
  3. Close > EMA-50 (medium-term uptrend confirmation)
  4. 50 < RSI < 75 (positive momentum, not overbought)

Conviction: 0.50–0.85 based on EMA gap strength, RSI position, trend strength.

Exit conditions (any one triggers):
  - EMA-9 < EMA-20 (crossover reversal)
  - RSI > 75 (overbought)

Hold type: SWING (trend continuations need time to play out).
"""

from __future__ import annotations

import pandas as pd

from ai_trade.data.indicators import add_atr, add_ema, add_rsi
from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import BaseStrategy, HoldType, Signal

logger = get_logger(__name__)


class EMACrossoverStrategy(BaseStrategy):
    """Enter on EMA-9/EMA-20 bullish crossovers confirmed by EMA-50 uptrend.

    Uses ATR-based stops that adapt to each stock's volatility.
    Exits when the crossover reverses or RSI becomes overbought.
    """

    def evaluate(
        self,
        symbol: str,
        daily_bars: pd.DataFrame,
        intraday_bars: pd.DataFrame | None = None,
    ) -> Signal | None:
        df = daily_bars.copy()

        # Config parameters
        fast_period: int = getattr(self.config, "fast_period", 9)
        slow_period: int = getattr(self.config, "slow_period", 20)
        trend_period: int = getattr(self.config, "trend_period", 50)
        rsi_min: float = getattr(self.config, "rsi_min", 50)
        rsi_max: float = getattr(self.config, "rsi_max", 75)
        atr_stop_mult: float = getattr(self.config, "atr_stop_multiplier", 1.5)
        atr_tp_mult: float = getattr(self.config, "atr_tp_multiplier", 3.0)

        # Compute indicators
        add_ema(df, [fast_period, slow_period, trend_period])
        add_rsi(df)
        add_atr(df)

        if len(df) < trend_period + 2:
            return None

        fast_col = f"ema_{fast_period}"
        slow_col = f"ema_{slow_period}"
        trend_col = f"ema_{trend_period}"

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        close: float = latest["close"]
        fast_ema: float = latest[fast_col]
        slow_ema: float = latest[slow_col]
        trend_ema: float = latest[trend_col]
        rsi_val: float = latest["rsi_14"]
        atr: float = latest["atr_14"]

        prev_fast: float = prev[fast_col]
        prev_slow: float = prev[slow_col]

        # ── Entry condition 1: Bullish crossover (fast above slow) ──
        if fast_ema <= slow_ema:
            return None

        # ── Entry condition 2: Crossover is fresh (previous bar had fast <= slow) ──
        if prev_fast > prev_slow:
            # Check the bar before that for a 2-bar window
            if len(df) >= 3:
                prev2 = df.iloc[-3]
                if prev2[fast_col] > prev2[slow_col]:
                    logger.debug("ema_cross_reject", symbol=symbol, reason="stale_crossover")
                    return None
            else:
                return None

        # ── Entry condition 3: Medium-term uptrend ──
        if close <= trend_ema:
            logger.debug("ema_cross_reject", symbol=symbol, reason="below_trend_ema",
                         close=close, trend_ema=trend_ema)
            return None

        # ── Entry condition 4: RSI in positive-but-not-overbought range ──
        if rsi_val <= rsi_min or rsi_val >= rsi_max:
            logger.debug("ema_cross_reject", symbol=symbol, reason="rsi_out_of_range",
                         rsi=rsi_val, min=rsi_min, max=rsi_max)
            return None

        # ── Conviction scoring (0.50–0.85) ──
        conviction = 0.50

        # +0.15 max: EMA gap strength (how far fast pulled above slow)
        ema_gap_pct = (fast_ema - slow_ema) / close
        conviction += min(0.15, ema_gap_pct / 0.01 * 0.15)

        # +0.10 max: RSI in sweet spot (55-65 gets full bonus)
        if 55 <= rsi_val <= 65:
            conviction += 0.10
        elif 50 < rsi_val < 55 or 65 < rsi_val < 75:
            conviction += 0.05

        # +0.10 max: Strong trend (close well above trend EMA)
        trend_gap_pct = (close - trend_ema) / trend_ema
        if trend_gap_pct > 0.01:
            conviction += min(0.10, trend_gap_pct * 5.0)

        conviction = max(0.50, min(0.85, conviction))

        entry_price = close
        stop_loss = entry_price - atr_stop_mult * atr
        take_profit = entry_price + atr_tp_mult * atr

        if stop_loss >= entry_price or take_profit <= entry_price:
            return None

        logger.info(
            "ema_crossover_signal",
            symbol=symbol,
            fast_ema=fast_ema,
            slow_ema=slow_ema,
            trend_ema=trend_ema,
            rsi=rsi_val,
            conviction=conviction,
            entry=entry_price,
            stop=stop_loss,
            target=take_profit,
        )

        return Signal(
            symbol=symbol,
            direction="long",
            conviction=conviction,
            strategy_name="ema_crossover",
            hold_type=HoldType.SWING,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            metadata={
                "fast_ema": fast_ema,
                "slow_ema": slow_ema,
                "trend_ema": trend_ema,
                "rsi": rsi_val,
                "atr": atr,
                "ema_gap_pct": round(ema_gap_pct, 4),
            },
        )

    def should_exit(
        self, symbol: str, bars: pd.DataFrame, entry_price: float
    ) -> bool:
        """Exit when crossover reverses or RSI becomes overbought."""
        df = bars.copy()
        fast_period: int = getattr(self.config, "fast_period", 9)
        slow_period: int = getattr(self.config, "slow_period", 20)
        rsi_max: float = getattr(self.config, "rsi_max", 75)

        add_ema(df, [fast_period, slow_period])
        add_rsi(df)

        latest = df.iloc[-1]
        fast_ema: float = latest[f"ema_{fast_period}"]
        slow_ema: float = latest[f"ema_{slow_period}"]
        rsi_val: float = latest["rsi_14"]

        if fast_ema < slow_ema or rsi_val > rsi_max:
            logger.info(
                "ema_crossover_exit",
                symbol=symbol,
                fast_ema=fast_ema,
                slow_ema=slow_ema,
                rsi=rsi_val,
            )
            return True

        return False
