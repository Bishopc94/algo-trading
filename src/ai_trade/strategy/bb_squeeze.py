"""Bollinger Band squeeze breakout strategy (adaptive hold type).

Theory:
  Volatility compression (Bollinger Bands narrowing) precedes explosive moves.
  True squeeze: BB width contracts AND Keltner Channel contains BB (TTM-style).
  When bands expand and price breaks above the upper band on volume with
  MACD momentum aligned, it signals a directional breakout.

Entry conditions (ALL must be true):
  1. Recent BB width was in bottom 30% of its 50-bar range (squeeze)
  2. BB width expanding (current > previous)
  3. Close > BB upper (breakout)
  4. Relative volume > 1.5x (volume confirms)
  5. MACD histogram > 0 (momentum aligned)
  6. RSI 50-80 (positive momentum, not exhausted)
  7. Close > EMA-20 (trend confirmation)

Conviction: additive across squeeze depth, volume, MACD, RSI, trend.

Exit: Close < BB middle (breakout failed).
Hold type: ADAPTIVE.
"""

from __future__ import annotations

import pandas as pd

from ai_trade.data.indicators import (
    add_atr, add_bollinger, add_ema, add_macd, add_rsi, add_volume_profile,
)
from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import BaseStrategy, HoldType, Signal

logger = get_logger(__name__)


class BBSqueezeStrategy(BaseStrategy):
    """Enter on Bollinger Band squeeze breakouts with multi-indicator confluence."""

    def evaluate(
        self,
        symbol: str,
        daily_bars: pd.DataFrame,
        intraday_bars: pd.DataFrame | None = None,
    ) -> Signal | None:
        df = daily_bars.copy()

        squeeze_lookback: int = getattr(self.config, "squeeze_lookback", 5)
        min_rel_vol: float = getattr(self.config, "min_relative_volume", 1.5)
        atr_tp_mult: float = getattr(self.config, "atr_tp_multiplier", 3.5)

        add_bollinger(df)
        add_volume_profile(df)
        add_atr(df)
        add_macd(df)
        add_rsi(df)
        add_ema(df, [20, 50])

        if len(df) < 52:
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
        macd_hist: float = latest["macd_hist"]
        rsi_val: float = latest["rsi_14"]
        ema_20: float = latest["ema_20"]
        ema_50: float = latest["ema_50"]

        # ── Hard filters ──

        # Squeeze detection: width in bottom 40% of 50-bar range (relaxed from 30%)
        long_range = df["bb_width"].iloc[max(0, len(df) - 50):]
        width_range = long_range.max() - long_range.min()
        if width_range <= 0.001:
            self._reject(symbol, "bb_width_range", width_range, 0.001, "above")
            return None
        lookback_start = max(0, len(df) - squeeze_lookback - 1)
        recent_widths = df["bb_width"].iloc[lookback_start:-1]
        if recent_widths.empty:
            return None
        min_width = recent_widths.min()
        width_percentile = (min_width - long_range.min()) / width_range
        if width_percentile > 0.40:
            self._reject(symbol, "squeeze_percentile", width_percentile, 0.40, "below")
            return None

        # Bands must be expanding
        if bb_width <= prev_bb_width:
            self._reject(symbol, "bb_expanding", bb_width, prev_bb_width, "above")
            return None

        # Breakout above upper band
        if close <= bb_upper:
            self._reject(symbol, "close_above_bb_upper", close, bb_upper, "above")
            return None

        # Volume confirmation (conviction factor if below threshold)
        if rel_vol < 1.0:
            self._reject(symbol, "rel_volume_min", rel_vol, 1.0, "above")
            return None

        # MACD momentum aligned
        if macd_hist <= 0:
            self._reject(symbol, "macd_hist_positive", macd_hist, 0.0, "above")
            return None

        # RSI in momentum range (widened from 50-80 to 45-80)
        if rsi_val <= 45:
            self._reject(symbol, "rsi_min", rsi_val, 45.0, "above")
            return None
        if rsi_val >= 80:
            self._reject(symbol, "rsi_max", rsi_val, 80.0, "below")
            return None

        # Price above EMA-20 (trend confirmation)
        if close <= ema_20:
            self._reject(symbol, "close_above_ema20", close, ema_20, "above")
            return None

        # ── Conviction scoring (additive, 0.50–0.90) ──
        conviction = 0.50

        # +0.10: Squeeze depth (tighter = more explosive)
        squeeze_depth = max(0, 0.40 - width_percentile) / 0.40
        conviction += 0.10 * min(1.0, squeeze_depth)

        # +0.10: Volume strength (now wider range with vol >= 1.0 allowed)
        if rel_vol >= 3.0:
            conviction += 0.10
        elif rel_vol >= 2.0:
            conviction += 0.07
        elif rel_vol >= 1.5:
            conviction += 0.04
        elif rel_vol >= 1.2:
            conviction += 0.02

        # +0.07: MACD accelerating
        prev_macd_hist = float(prev["macd_hist"])
        if macd_hist > prev_macd_hist > 0:
            conviction += 0.07
        elif macd_hist > 0:
            conviction += 0.03

        # +0.07: RSI in sweet spot (widened)
        if 55 <= rsi_val <= 70:
            conviction += 0.07
        elif 50 <= rsi_val < 55:
            conviction += 0.04
        elif 45 < rsi_val < 50:
            conviction += 0.02

        # +0.06: Uptrend structure (EMA-20 > EMA-50)
        if ema_20 > ema_50:
            trend_gap = (ema_20 - ema_50) / ema_50
            conviction += min(0.06, trend_gap * 2.5)

        conviction = max(0.50, min(0.90, conviction))

        entry_price = close
        stop_loss = bb_middle
        take_profit = entry_price + atr_tp_mult * atr

        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        rr = reward / risk if risk > 0 else 0.0
        if risk <= 0 or rr < 2.0:
            self._reject(symbol, "risk_reward", rr, 2.0, "above")
            return None

        hold_type = HoldType.DAY if conviction >= 0.9 else HoldType.SWING

        logger.info(
            "bb_squeeze_signal",
            symbol=symbol,
            bb_width=bb_width,
            width_percentile=round(width_percentile, 3),
            rel_vol=rel_vol,
            rsi=rsi_val,
            macd_hist=macd_hist,
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
                "width_percentile": round(width_percentile, 3),
                "relative_volume": rel_vol,
                "rsi": rsi_val,
                "macd_hist": macd_hist,
                "bb_middle": bb_middle,
                "bb_upper": bb_upper,
                "atr": atr,
            },
        )

    def should_exit(
        self, symbol: str, bars: pd.DataFrame, entry_price: float
    ) -> bool:
        df = bars.copy()
        add_bollinger(df)
        latest = df.iloc[-1]
        if latest["close"] < latest["bb_middle"]:
            return True
        return False
