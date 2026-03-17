"""Volume-breakout momentum strategy (adaptive hold type).

Theory:
  Stocks that break out to new highs on heavy volume tend to keep going.
  The volume confirms the breakout is "real" — driven by institutional
  buying, not just a few retail traders.

Entry conditions (ALL must be true):
  1. Close > 20-day high (price is at a new short-term high)
  2. Relative volume > 1.5x (trading activity is well above average)
  3. ADR% > 2.0 (the stock moves enough daily to be worth trading)
  4. Close > EMA-20 (confirmed uptrend)

Conviction: mapped from relative volume (1.5x→0.5, 3x→0.75, 5x+→1.0).

Adaptive hold type:
  - Conviction >= 0.90 → DAY trade (costs 1 PDT slot)
  - Conviction < 0.90  → SWING trade (free, held overnight)

Exit: ``should_exit()`` always returns False — bracket orders handle it.
"""

from __future__ import annotations

import pandas as pd

from ai_trade.data.indicators import add_atr, add_ema, add_volume_profile, compute_adr
from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import BaseStrategy, HoldType, Signal

logger = get_logger(__name__)


class MomentumStrategy(BaseStrategy):
    """Enter on volume-confirmed breakouts above the 20-day high.

    Uses ATR-based stops that automatically adapt to each stock's volatility.
    Exits are handled entirely by bracket orders (stop loss + take profit +
    trailing stop).  The ``should_exit`` method always returns False so that
    the mechanical exits run the trade — no second-guessing.
    """

    def evaluate(
        self,
        symbol: str,
        daily_bars: pd.DataFrame,
        intraday_bars: pd.DataFrame | None = None,
    ) -> Signal | None:
        df = daily_bars.copy()

        # Read config parameters
        breakout_lookback: int = getattr(self.config, "breakout_lookback", 20)
        vol_spike: float = getattr(self.config, "volume_spike_multiplier", 1.5)
        min_adr: float = getattr(self.config, "min_adr_pct", 2.0)
        atr_stop_mult: float = getattr(self.config, "atr_stop_multiplier", 1.5)
        atr_tp_mult: float = getattr(self.config, "atr_tp_multiplier", 3.0)

        # Compute required indicators
        add_volume_profile(df)
        add_atr(df)
        add_ema(df, [20])

        if len(df) < breakout_lookback + 1:
            return None

        # 20-day high EXCLUDING the current bar (shift(1)) — so the
        # breakout must be fresh (today's close surpasses yesterday's
        # 20-day rolling max)
        df["high_20"] = df["high"].rolling(breakout_lookback).max().shift(1)

        latest = df.iloc[-1]
        close: float = latest["close"]
        high_20: float = latest["high_20"]
        rel_vol: float = latest["relative_volume"]
        adr_pct: float = compute_adr(df)
        atr: float = latest["atr_14"]
        ema_20: float = latest["ema_20"]

        # ----- Entry conditions (ALL must be true) -----

        # Condition 1: Price must break above the 20-day high
        if close <= high_20:
            logger.debug("momentum_reject", symbol=symbol, reason="no_breakout", close=close, high_20=high_20)
            return None

        # Condition 2: Volume must be elevated (confirms the breakout)
        if rel_vol <= vol_spike:
            logger.debug("momentum_reject", symbol=symbol, reason="volume_too_low", rel_vol=rel_vol, threshold=vol_spike)
            return None

        # Condition 3: Stock must have enough daily range to be profitable
        if adr_pct <= min_adr:
            logger.debug("momentum_reject", symbol=symbol, reason="adr_too_low", adr_pct=adr_pct, threshold=min_adr)
            return None

        # Condition 4: Trend confirmation — price above the 20-day EMA
        if close <= ema_20:
            logger.debug("momentum_reject", symbol=symbol, reason="below_ema20", close=close, ema_20=ema_20)
            return None

        # ----- Conviction scoring (linear interpolation) -----
        # Maps relative volume to conviction:
        #   1.5x → 0.50,  3.0x → 0.75,  5.0x → 1.00
        if rel_vol >= 5.0:
            conviction = 1.0
        elif rel_vol >= 3.0:
            conviction = 0.75 + 0.25 * (rel_vol - 3.0) / 2.0
        else:
            conviction = 0.5 + 0.25 * (rel_vol - 1.5) / 1.5
        conviction = max(0.5, min(1.0, conviction))

        entry_price = close

        # ATR-based stops adapt to each stock's volatility
        stop_loss = entry_price - atr_stop_mult * atr
        take_profit = entry_price + atr_tp_mult * atr

        # Sanity: stop must be below entry, target above entry
        if stop_loss >= entry_price or take_profit <= entry_price:
            return None

        # ----- Adaptive hold type -----
        # Very high conviction (0.9+) → use a day-trade slot for quick profit
        # Lower conviction → hold overnight as a swing trade (free PDT)
        hold_type = HoldType.DAY if conviction >= 0.9 else HoldType.SWING

        logger.info(
            "momentum_signal",
            symbol=symbol,
            rel_vol=rel_vol,
            adr_pct=adr_pct,
            conviction=conviction,
            entry=entry_price,
            stop=stop_loss,
            target=take_profit,
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
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            metadata={
                "relative_volume": rel_vol,
                "adr_pct": adr_pct,
                "breakout_level": high_20,
                "atr": atr,
            },
        )

    def should_exit(
        self, symbol: str, bars: pd.DataFrame, entry_price: float
    ) -> bool:
        """Always returns False — bracket orders handle all exits.

        The old dynamic should_exit was causing premature exits because
        the rolling high_20 recalculates after entry, creating a moving
        reference that triggers too easily.  Server-side bracket orders
        are more reliable and persist even if the bot crashes.
        """
        return False
