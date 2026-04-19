"""EMA crossover trend-following strategy (swing trade).

Theory:
  When the fast EMA (9) crosses above the slow EMA (20) in an established
  uptrend (EMA-20 > EMA-50), it signals trend continuation.  We add
  multi-indicator confirmation: MACD alignment, RSI momentum, volume,
  and candle structure (bullish close above all EMAs).

Entry conditions (ALL must be true):
  1. EMA-9 > EMA-20 (bullish crossover)
  2. Crossover happened within last 2 bars (fresh, not stale — relaxed)
  3. EMA-20 > EMA-50 (established uptrend structure)
  4. Close > EMA-50 (confirmed above medium-term trend)
  5. 45 < RSI < 75 (positive momentum — widened range)
  6. MACD histogram > 0 (momentum aligned)
  7. Bullish candle close (close > open, buyers in control)

Conviction: additive across trend, RSI, MACD, volume, candle factors.
Volume confirmation is a conviction factor, not a hard gate.

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
            self._reject(symbol, "fast_above_slow_ema", fast_ema, slow_ema, "above")
            return None

        # Allow crossover within last 2 bars (not just fresh single-bar)
        if len(df) >= 4:
            prev2 = df.iloc[-3]
            prev2_fast: float = prev2[fast_col]
            prev2_slow: float = prev2[slow_col]
            crossover_recent = (prev_fast <= prev_slow) or (prev2_fast <= prev2_slow)
        else:
            crossover_recent = prev_fast <= prev_slow

        if not crossover_recent:
            self._reject(symbol, "crossover_recent", 0.0, 1.0, "above")
            return None

        if slow_ema <= trend_ema:
            self._reject(symbol, "slow_above_trend_ema", slow_ema, trend_ema, "above")
            return None
        if close <= trend_ema:
            self._reject(symbol, "close_above_trend_ema", close, trend_ema, "above")
            return None
        if rsi_val <= rsi_min:
            self._reject(symbol, "rsi_min", rsi_val, rsi_min, "above")
            return None
        if rsi_val >= rsi_max:
            self._reject(symbol, "rsi_max", rsi_val, rsi_max, "below")
            return None
        if macd_hist <= 0:
            self._reject(symbol, "macd_hist_positive", macd_hist, 0.0, "above")
            return None
        if close <= open_price:
            self._reject(symbol, "bullish_candle", close, open_price, "above")
            return None

        # Meaningful EMA gap (not a noise crossover — relaxed from 0.2% to 0.1%)
        ema_gap_pct = (fast_ema - slow_ema) / close
        if ema_gap_pct < 0.001:
            self._reject(symbol, "ema_gap_pct", ema_gap_pct, 0.001, "above")
            return None

        # ── Conviction scoring (additive, 0.50–0.90) ──
        conviction = 0.50

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
        elif 50 <= rsi_val < 55 or 65 < rsi_val < 70:
            conviction += 0.05
        elif 45 < rsi_val < 50:
            conviction += 0.02

        # +0.07: MACD accelerating
        prev_macd_hist = float(prev["macd_hist"])
        if macd_hist > prev_macd_hist > 0:
            conviction += 0.07
        elif macd_hist > 0:
            conviction += 0.03

        # +0.08: Volume strength (conviction factor, not hard gate)
        if rel_vol >= 2.0:
            conviction += 0.08
        elif rel_vol >= 1.5:
            conviction += 0.05
        elif rel_vol >= 1.2:
            conviction += 0.03
        elif rel_vol >= 1.0:
            conviction += 0.01

        # +0.05: EMA gap strength (wider = stronger signal)
        conviction += min(0.05, ema_gap_pct * 5.0)

        # +0.03: Crossover freshness bonus
        if prev_fast <= prev_slow:
            conviction += 0.03  # Bar 1 of crossover (freshest)

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
