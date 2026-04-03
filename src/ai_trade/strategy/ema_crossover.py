"""EMA crossover trend-following strategy (swing trade).

Theory:
  When the fast EMA (9) crosses above the slow EMA (20) in an established
  uptrend (EMA-20 > EMA-50), it signals trend continuation.  We add
  multi-indicator confirmation: MACD alignment, RSI momentum, volume,
  and candle structure (bullish close above all EMAs).

Entry conditions (ALL must be true):
  1. EMA-9 > EMA-20 (bullish crossover)
  2. Previous bar: EMA-9 <= EMA-20 (crossover is fresh, single-bar)
  3. EMA-20 > EMA-50 (established uptrend structure)
  4. Close > EMA-50 (confirmed above medium-term trend)
  5. 50 < RSI < 70 (positive momentum, not overbought)
  6. MACD histogram > 0 (momentum aligned)
  7. Relative volume >= 1.2x (participation confirms move)
  8. Bullish candle close (close > open, buyers in control)

Conviction: additive across trend, RSI, MACD, volume, candle factors.

Exit: EMA-9 < EMA-20 (crossover reversed) OR RSI > 75 (overbought).
Hold type: SWING.
"""

from __future__ import annotations

import pandas as pd

from ai_trade.data.indicators import add_atr, add_ema, add_macd, add_rsi, add_volume_profile
from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import BaseStrategy, HoldType, Signal

logger = get_logger(__name__)


class EMACrossoverStrategy(BaseStrategy):
    """Enter on EMA-9/EMA-20 bullish crossovers with multi-indicator confluence."""

    def evaluate(
        self,
        symbol: str,
        daily_bars: pd.DataFrame,
        intraday_bars: pd.DataFrame | None = None,
    ) -> Signal | None:
        df = daily_bars.copy()

        fast_period: int = getattr(self.config, "fast_period", 9)
        slow_period: int = getattr(self.config, "slow_period", 20)
        trend_period: int = getattr(self.config, "trend_period", 50)
        rsi_min: float = getattr(self.config, "rsi_min", 50)
        rsi_max: float = getattr(self.config, "rsi_max", 70)
        atr_stop_mult: float = getattr(self.config, "atr_stop_multiplier", 1.5)
        atr_tp_mult: float = getattr(self.config, "atr_tp_multiplier", 3.0)

        add_ema(df, [fast_period, slow_period, trend_period])
        add_rsi(df)
        add_atr(df)
        add_volume_profile(df)
        add_macd(df)

        if len(df) < trend_period + 2:
            return None

        fast_col = f"ema_{fast_period}"
        slow_col = f"ema_{slow_period}"
        trend_col = f"ema_{trend_period}"

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        close: float = latest["close"]
        open_price: float = latest["open"]
        fast_ema: float = latest[fast_col]
        slow_ema: float = latest[slow_col]
        trend_ema: float = latest[trend_col]
        rsi_val: float = latest["rsi_14"]
        atr: float = latest["atr_14"]
        rel_vol: float = latest.get("relative_volume", 1.0)
        macd_hist: float = latest["macd_hist"]

        prev_fast: float = prev[fast_col]
        prev_slow: float = prev[slow_col]

        # ── Hard filters ──
        if fast_ema <= slow_ema:
            return None
        if prev_fast > prev_slow:
            return None  # Not a fresh crossover
        if slow_ema <= trend_ema:
            return None  # No established uptrend structure
        if close <= trend_ema:
            return None
        if rsi_val <= rsi_min or rsi_val >= rsi_max:
            return None
        if macd_hist <= 0:
            return None  # Momentum not aligned
        if rel_vol < 1.2:
            return None
        if close <= open_price:
            return None  # Bearish candle, buyers not in control

        # Meaningful EMA gap (not a noise crossover)
        ema_gap_pct = (fast_ema - slow_ema) / close
        if ema_gap_pct < 0.002:
            return None

        # ── Conviction scoring (additive, 0.55–0.90) ──
        conviction = 0.55

        # +0.10: Trend strength (EMA-20/EMA-50 gap)
        trend_gap = (slow_ema - trend_ema) / trend_ema
        if trend_gap > 0.03:
            conviction += 0.10
        elif trend_gap > 0.015:
            conviction += 0.06
        elif trend_gap > 0.005:
            conviction += 0.03

        # +0.08: RSI in sweet spot
        if 55 <= rsi_val <= 65:
            conviction += 0.08
        elif 50 < rsi_val < 55 or 65 < rsi_val < 68:
            conviction += 0.04

        # +0.07: MACD accelerating
        prev_macd_hist = float(prev["macd_hist"])
        if macd_hist > prev_macd_hist > 0:
            conviction += 0.07
        elif macd_hist > 0:
            conviction += 0.03

        # +0.07: Volume strength
        if rel_vol >= 2.0:
            conviction += 0.07
        elif rel_vol >= 1.5:
            conviction += 0.04

        # +0.05: EMA gap strength (wider = stronger signal)
        conviction += min(0.05, ema_gap_pct * 5.0)

        conviction = max(0.55, min(0.90, conviction))

        entry_price = close
        stop_loss = entry_price - atr_stop_mult * atr
        take_profit = entry_price + atr_tp_mult * atr

        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        if risk <= 0 or reward / risk < 2.0:
            return None

        logger.info(
            "ema_crossover_signal",
            symbol=symbol,
            fast_ema=fast_ema,
            slow_ema=slow_ema,
            trend_ema=trend_ema,
            rsi=rsi_val,
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
                "macd_hist": macd_hist,
                "atr": atr,
                "ema_gap_pct": round(ema_gap_pct, 4),
            },
        )

    def should_exit(
        self, symbol: str, bars: pd.DataFrame, entry_price: float
    ) -> bool:
        df = bars.copy()
        fast_period: int = getattr(self.config, "fast_period", 9)
        slow_period: int = getattr(self.config, "slow_period", 20)
        add_ema(df, [fast_period, slow_period])
        add_rsi(df)

        latest = df.iloc[-1]
        if latest[f"ema_{fast_period}"] < latest[f"ema_{slow_period}"]:
            return True
        if latest["rsi_14"] > 75:
            return True
        return False
