"""RSI mean-reversion strategy (swing trade — no PDT cost).

Theory:
  Prices tend to revert to a "mean" (average) over time.  When a stock
  dips temporarily but the broader trend is still up, it's likely to
  bounce back.  Think of a rubber band — the further it's stretched
  below the mean, the harder it snaps back.

Entry conditions (ALL must be true):
  1. RSI < 40 (oversold — selling has been overdone)
  2. Close > EMA-20 * 0.98 (still in a short-term uptrend — not a crash)
  3. Close < BB_lower * 1.03 (price is near the lower Bollinger Band)

The combination ensures we buy a *dip* in an uptrend, not a falling knife.

Exit conditions (any one triggers exit):
  - RSI > 60 (momentum has recovered — mean has reverted)
  - Close > upper Bollinger Band (price is now "expensive")

Risk management:
  - ATR-based stops: stop = entry - 1.5 * ATR, target = entry + 3.0 * ATR
  - This gives a 2:1 reward-to-risk ratio scaled to each stock's volatility.

Hold type: SWING (held overnight, 2-5 days typically).  Free under PDT rules.
"""

from __future__ import annotations

import pandas as pd

from ai_trade.data.indicators import add_atr, add_bollinger, add_ema, add_rsi
from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import BaseStrategy, HoldType, Signal

logger = get_logger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """Buy oversold dips in an uptrend, targeting a 2:1 reward-to-risk.

    Entry: RSI < 40 + close above 20-EMA (still in short-term uptrend) +
    close near or below the lower Bollinger Band (oversold at support).

    Uses ATR-based stops for volatility-adaptive risk management.
    Exit: RSI recovers above 60 or price touches upper BB (mean has reverted).
    """

    def evaluate(
        self,
        symbol: str,
        daily_bars: pd.DataFrame,
        intraday_bars: pd.DataFrame | None = None,
    ) -> Signal | None:
        df = daily_bars.copy()

        # Read config parameters with defaults
        rsi_period: int = getattr(self.config, "rsi_period", 14)
        rsi_oversold: float = getattr(self.config, "rsi_oversold", 40)
        atr_stop_mult: float = getattr(self.config, "atr_stop_multiplier", 1.5)
        atr_tp_mult: float = getattr(self.config, "atr_tp_multiplier", 3.0)

        # Compute required indicators (idempotent — skips if already present)
        add_rsi(df, rsi_period)
        add_ema(df, [20])
        add_atr(df)
        add_bollinger(df)

        # Need at least 21 bars for Bollinger Bands (20-period + 1)
        if len(df) < 21:
            return None

        # Get the most recent bar's indicator values
        latest = df.iloc[-1]
        rsi_col = f"rsi_{rsi_period}"

        rsi_val: float = latest[rsi_col]
        close: float = latest["close"]
        ema_20: float = latest["ema_20"]
        bb_lower: float = latest["bb_lower"]
        atr: float = latest["atr_14"]

        # ----- Entry condition #1: RSI is oversold (below threshold) -----
        if rsi_val >= rsi_oversold:
            logger.debug("mean_reversion_reject", symbol=symbol, reason="rsi_not_oversold", rsi=rsi_val, threshold=rsi_oversold)
            return None

        # ----- Entry condition #2: Still in short-term uptrend -----
        # Using EMA-20 (not EMA-50) because when RSI drops to 40,
        # price typically stays above the faster moving average.
        # The 2% buffer (0.98) allows slightly below EMA without rejecting.
        if close <= ema_20 * 0.98:
            logger.debug("mean_reversion_reject", symbol=symbol, reason="below_ema20", close=close, ema_20=ema_20)
            return None

        # ----- Entry condition #3: Price near lower Bollinger Band -----
        # Close must be within 3% of the lower band — at statistical support.
        if close > bb_lower * 1.03:
            logger.debug("mean_reversion_reject", symbol=symbol, reason="not_near_bb_lower", close=close, bb_lower=bb_lower)
            return None

        # ----- Conviction scoring -----
        # Linear scale: RSI at threshold (40) → 0.5, RSI at 20 → 1.0
        # More oversold = higher conviction (rubber band stretched further)
        conviction = 0.5 + 0.5 * (rsi_oversold - rsi_val) / max(rsi_oversold - 20, 1)
        conviction = max(0.5, min(1.0, conviction))

        entry_price = close

        # ----- ATR-based stop loss and take profit -----
        # ATR adapts to each stock's volatility:
        #   High-vol stock (ATR=$2): stop=$3 away, target=$6 away
        #   Low-vol stock (ATR=$0.50): stop=$0.75, target=$1.50
        stop_loss = entry_price - atr_stop_mult * atr
        take_profit = entry_price + atr_tp_mult * atr

        # Sanity check: stop must be below entry, target above entry
        if stop_loss >= entry_price or take_profit <= entry_price:
            return None

        logger.info(
            "mean_reversion_signal",
            symbol=symbol,
            rsi=rsi_val,
            conviction=conviction,
            entry=entry_price,
            stop=stop_loss,
            target=take_profit,
            atr=atr,
            ema_20=ema_20,
            bb_lower=bb_lower,
        )

        return Signal(
            symbol=symbol,
            direction="long",
            conviction=conviction,
            strategy_name="mean_reversion",
            hold_type=HoldType.SWING,  # Always swing — no PDT cost
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            metadata={
                "rsi": rsi_val,
                "ema_20": ema_20,
                "bb_lower": bb_lower,
                "atr": atr,
            },
        )

    def should_exit(
        self, symbol: str, bars: pd.DataFrame, entry_price: float
    ) -> bool:
        """Check if the mean reversion has completed — time to take profit.

        Exit when RSI recovers above 60 (momentum is back to normal) OR
        when price touches the upper Bollinger Band (statistically overbought).
        """
        df = bars.copy()
        rsi_period: int = getattr(self.config, "rsi_period", 14)
        rsi_exit: float = getattr(self.config, "rsi_exit", 60)

        add_rsi(df, rsi_period)
        add_bollinger(df)

        latest = df.iloc[-1]
        rsi_val: float = latest[f"rsi_{rsi_period}"]
        close: float = latest["close"]
        bb_upper: float = latest["bb_upper"]

        # Exit when the mean reversion has played out
        if rsi_val > rsi_exit or close > bb_upper:
            logger.info(
                "mean_reversion_exit",
                symbol=symbol,
                rsi=rsi_val,
                close=close,
                bb_upper=bb_upper,
            )
            return True

        return False
