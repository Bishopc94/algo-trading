"""Volume-breakout momentum strategy (adaptive hold type).

Theory:
  Stocks that break out to new highs on heavy volume tend to continue.
  We require multi-indicator confluence: price breakout + volume spike +
  trend alignment + momentum confirmation (MACD + RSI).
  Consolidation tightness is a conviction factor, not a hard gate.

Entry conditions (ALL must be true):
  1. Close > 20-day high (fresh breakout)
  2. Relative volume > 1.5x (participation confirmation)
  3. Close > EMA-20 > EMA-50 (stacked uptrend)
  4. RSI 45-80 (positive momentum, not exhausted)
  5. MACD histogram > 0 (momentum aligned)

Conviction: additive across 6 factors (volume, trend, RSI, MACD,
  consolidation, ADR).

Exit: bracket orders handle all exits.
"""

from __future__ import annotations

import pandas as pd

from ai_trade.data.indicators import (
    add_atr, add_ema, add_macd, add_rsi, add_volume_profile, compute_adr,
)
from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import BaseStrategy, HoldType, Signal

logger = get_logger(__name__)


class MomentumStrategy(BaseStrategy):
    """Enter on volume-confirmed breakouts with multi-indicator confluence."""

    def evaluate(
        self,
        symbol: str,
        daily_bars: pd.DataFrame,
        intraday_bars: pd.DataFrame | None = None,
    ) -> Signal | None:
        df = daily_bars.copy()

        breakout_lookback: int = getattr(self.config, "breakout_lookback", 20)
        vol_spike: float = getattr(self.config, "volume_spike_multiplier", 1.5)
        atr_stop_mult: float = getattr(self.config, "atr_stop_multiplier", 1.5)
        atr_tp_mult: float = getattr(self.config, "atr_tp_multiplier", 3.5)
        max_gap_pct: float = getattr(self.config, "max_gap_pct", 0.15)

        add_volume_profile(df)
        add_atr(df)
        add_ema(df, [20, 50])
        add_rsi(df)
        add_macd(df)

        if len(df) < max(breakout_lookback + 1, 52):
            return None

        df["high_20"] = df["high"].rolling(breakout_lookback).max().shift(1)

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        close: float = latest["close"]
        high_20: float = latest["high_20"]
        rel_vol: float = latest["relative_volume"]
        adr_pct: float = compute_adr(df)
        atr: float = latest["atr_14"]
        ema_20: float = latest["ema_20"]
        ema_50: float = latest["ema_50"]
        rsi_val: float = latest["rsi_14"]
        macd_hist: float = latest["macd_hist"]

        # ── Hard filters ──

        # Minimum ADR: stocks with < 0.5% average daily range have sub-cent
        # ATR, which produces junk stops/targets that will be immediately
        # stopped out by the bid-ask spread.  (EM had ADR=0.15%, ATR=$0.008.)
        min_adr: float = getattr(self.config, "min_adr_pct", 1.0)
        hard_adr_floor = 0.5
        if adr_pct < hard_adr_floor:
            self._reject(symbol, "min_adr_floor", adr_pct, hard_adr_floor, "above")
            return None

        if close <= high_20:
            self._reject(symbol, "breakout", close, high_20, "above")
            return None

        # Gap filter: reject if the entire "breakout" happened overnight (e.g.
        # binary biotech news, earnings surprise in AH).  The move is already
        # done — chasing a 15%+ gap rarely has intraday follow-through.
        prev_close: float = float(prev["close"])
        if prev_close > 0:
            gap_pct = (close - prev_close) / prev_close
            if gap_pct > max_gap_pct:
                self._reject(symbol, "overnight_gap", round(gap_pct, 4), max_gap_pct, "below")
                return None
        if rel_vol < vol_spike:
            self._reject(symbol, "rel_volume", rel_vol, vol_spike, "above")
            return None

        # Stacked uptrend: close > EMA-20 > EMA-50
        if close <= ema_20:
            self._reject(symbol, "close_above_ema20", close, ema_20, "above")
            return None
        if ema_20 <= ema_50:
            self._reject(symbol, "ema20_above_ema50", ema_20, ema_50, "above")
            return None

        # RSI in momentum range (widened: allow earlier entries at 45+)
        if rsi_val < 45:
            self._reject(symbol, "rsi_min", rsi_val, 45.0, "above")
            return None
        if rsi_val >= 80:
            self._reject(symbol, "rsi_max", rsi_val, 80.0, "below")
            return None

        # MACD histogram positive (momentum aligned)
        if macd_hist <= 0:
            self._reject(symbol, "macd_hist_positive", macd_hist, 0.0, "above")
            return None

        # Pre-breakout consolidation: conviction factor, not hard gate
        recent_5 = df.iloc[-6:-1]
        consolidation_range = recent_5["high"].max() - recent_5["low"].min()

        # ── Conviction scoring (additive, 0.55–1.0) ──
        conviction = 0.55

        # +0.12: Volume strength (wider tiers)
        if rel_vol >= 5.0:
            conviction += 0.12
        elif rel_vol >= 3.0:
            conviction += 0.08
        elif rel_vol >= 2.0:
            conviction += 0.05
        else:
            conviction += 0.02

        # +0.10: Trend strength (EMA gap)
        trend_gap = (ema_20 - ema_50) / ema_50
        if trend_gap > 0.03:
            conviction += 0.10
        elif trend_gap > 0.015:
            conviction += 0.06
        else:
            conviction += 0.02

        # +0.08: RSI sweet spot (55-70 = strong but not exhausted)
        if 55 <= rsi_val <= 70:
            conviction += 0.08
        elif 50 <= rsi_val < 55 or 70 < rsi_val < 75:
            conviction += 0.04
        elif 45 <= rsi_val < 50:
            conviction += 0.01

        # +0.07: MACD histogram strength (accelerating momentum)
        prev_macd_hist = float(df.iloc[-2]["macd_hist"])
        if macd_hist > prev_macd_hist > 0:
            conviction += 0.07  # Accelerating
        elif macd_hist > 0:
            conviction += 0.03  # Positive but decelerating

        # +0.07: Consolidation tightness (conviction factor, not hard reject)
        if consolidation_range < 1.0 * atr:
            conviction += 0.07
        elif consolidation_range < 1.5 * atr:
            conviction += 0.04
        elif consolidation_range < 2.0 * atr:
            conviction += 0.02
        elif consolidation_range < 3.0 * atr:
            conviction += 0.0  # Wide but allowed
        else:
            conviction -= 0.05  # Penalize very choppy setups

        # +0.05: ADR bonus (conviction factor, not hard gate)
        if adr_pct >= 3.0:
            conviction += 0.05
        elif adr_pct >= 2.0:
            conviction += 0.03
        elif adr_pct >= 1.0:
            conviction += 0.01

        conviction = max(0.50, min(1.0, conviction))

        entry_price = close
        levels = self._plan_long_exit(
            bars=df, entry_price=entry_price, atr=atr,
            base_stop_mult=atr_stop_mult, base_tp_mult=atr_tp_mult,
        )
        stop_loss = levels.stop_loss
        take_profit = levels.take_profit

        # Enforce minimum 2:1 R:R
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        rr = reward / risk if risk > 0 else 0.0
        if risk <= 0 or rr < 2.0:
            self._reject(symbol, "risk_reward", rr, 2.0, "above")
            return None

        hold_type = HoldType.DAY if conviction >= 0.9 else HoldType.SWING

        # Pullback-entry limit price.  Breakout strategies fire AT the high
        # of the move, so a market entry pays the highest available price.
        # Place a LIMIT at the nearest meaningful support — EMA-20 if it's
        # within 2% below close, otherwise close × 0.99.  The 2% cap stops
        # us reaching for a far-away EMA-20 that won't get retested today.
        # The entry-management job will chase at market if price walks
        # away from this limit, so we don't strand setups.
        max_pullback_pct = 0.02
        ema_limit = ema_20 if 0 < (close - ema_20) / close <= max_pullback_pct else None
        limit_price = ema_limit if ema_limit is not None else close * (1 - 0.01)
        # Never set the limit above current close (would auto-fill on submit).
        limit_price = min(limit_price, close * 0.999)
        # Never set it below the stop (the trade would be pre-stopped).
        if limit_price <= stop_loss:
            limit_price = None  # support level is too deep — just market entry

        logger.info(
            "momentum_signal",
            symbol=symbol,
            rel_vol=rel_vol,
            adr_pct=adr_pct,
            conviction=round(conviction, 3),
            entry=entry_price,
            limit=limit_price,
            stop=stop_loss,
            target=take_profit,
            rsi=rsi_val,
            macd_hist=macd_hist,
            trend_gap=round(trend_gap, 4),
            consolidation_atr_ratio=round(consolidation_range / atr, 2),
            atr=atr,
            hold_type=hold_type.value,
        )

        return Signal(
            symbol=symbol,
            direction="long",
            conviction=conviction,
            strategy_name="momentum",
            hold_type=hold_type,
            entry_price=entry_price,
            limit_price=limit_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            metadata={
                "relative_volume": rel_vol,
                "adr_pct": adr_pct,
                "breakout_level": high_20,
                "rsi": rsi_val,
                "macd_hist": macd_hist,
                "trend_gap": round(trend_gap, 4),
                "atr": atr,
                "stop_method": levels.stop_method,
                "target_method": levels.target_method,
                "effective_stop_mult": round(levels.effective_stop_mult, 3),
                "effective_tp_mult": round(levels.effective_tp_mult, 3),
            },
        )

    def should_exit(
        self, symbol: str, bars: pd.DataFrame, entry_price: float
    ) -> bool:
        return False
