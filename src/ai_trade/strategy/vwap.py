"""VWAP-reclaim intraday strategy (day trade — costs 1 PDT slot).

Theory:
  VWAP (Volume-Weighted Average Price) represents the "fair price" for
  the day.  Institutional traders use it as a benchmark.  When a stock
  dips below VWAP and then reclaims it on high volume, institutional
  buyers are stepping back in — a short-term bullish signal.

Entry conditions (ALL must be true):
  1. Current close > VWAP (price has reclaimed VWAP)
  2. Recent dip below VWAP in the last 10 bars (there was a dip to buy)
  3. Volume on the reclaim bar > 1.5x average (institutional participation)

This is the only strategy that uses intraday (minute) bars.

Conviction: Base 0.70, +0.15 max for dip depth, +0.15 for volume strength.

Hold type: DAY — force-closed by 3:50 PM ET.  Costs 1 PDT slot.
"""

from __future__ import annotations

import pandas as pd

from ai_trade.data.indicators import add_vwap
from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import BaseStrategy, HoldType, Signal

logger = get_logger(__name__)


class VWAPStrategy(BaseStrategy):
    """Enter when price reclaims VWAP from below on elevated volume."""

    def evaluate(
        self,
        symbol: str,
        daily_bars: pd.DataFrame,
        intraday_bars: pd.DataFrame | None = None,
    ) -> Signal | None:
        # This strategy REQUIRES intraday (minute) bars
        if intraday_bars is None or intraday_bars.empty:
            logger.debug("vwap_reject", symbol=symbol, reason="no_intraday_data")
            return None

        df = intraday_bars.copy()
        add_vwap(df)  # Adds the "vwap_calc" column

        lookback: int = 10  # Look back 10 bars for a recent dip
        if len(df) < lookback + 1:
            return None

        latest = df.iloc[-1]
        close: float = latest["close"]
        vwap: float = latest["vwap_calc"]
        bar_volume: float = latest["volume"]

        # Average bar volume over the lookback window
        window = df.iloc[-(lookback + 1) :]
        avg_bar_vol = window["volume"].mean()

        # ----- Condition 1: Recent dip below VWAP -----
        # Without a prior dip, there's no "reclaim" pattern.
        recent = df.iloc[-lookback:]
        dip_bars = recent[recent["close"] < recent["vwap_calc"]]
        if dip_bars.empty:
            return None

        # ----- Condition 2: Current bar above VWAP (the "reclaim") -----
        if close <= vwap:
            return None

        # ----- Condition 3: Volume confirmation -----
        if avg_bar_vol > 0 and bar_volume <= 1.5 * avg_bar_vol:
            return None

        # ----- Conviction scoring -----
        dip_low_price = dip_bars["close"].min()
        dip_vwap = dip_bars.loc[dip_bars["close"].idxmin(), "vwap_calc"]
        dip_pct = (dip_low_price - dip_vwap) / dip_vwap if dip_vwap else 0
        reclaim_pct = (close - vwap) / vwap if vwap else 0

        # Base conviction 0.7 — VWAP reclaims have a reasonable edge
        base_conviction = 0.7
        # Bonus for deeper dips (more potential upside)
        dip_adj = min(0.15, abs(dip_pct) * 10)
        # Bonus for stronger volume on the reclaim bar
        vol_ratio = bar_volume / avg_bar_vol if avg_bar_vol > 0 else 1.0
        vol_adj = min(0.15, (vol_ratio - 1.5) * 0.05)

        conviction = base_conviction + dip_adj + vol_adj
        conviction = max(0.5, min(1.0, conviction))

        entry_price = close

        # ----- Stop loss: below where the dip bottomed out -----
        dip_low = dip_bars["low"].min()
        stop_loss = max(dip_low, vwap * 0.99)  # At least VWAP - 1%

        # ----- Take profit: VWAP + deviation or the day's high -----
        exit_dev: float = getattr(self.config, "exit_deviation_pct", 0.3) / 100
        tp_from_vwap = vwap * (1 + exit_dev)
        prior_high = df["high"].max()
        take_profit = max(tp_from_vwap, prior_high)

        logger.info(
            "vwap_signal",
            symbol=symbol,
            conviction=conviction,
            entry=entry_price,
            stop=stop_loss,
            target=take_profit,
            vwap=vwap,
            reclaim_pct=reclaim_pct,
        )

        return Signal(
            symbol=symbol,
            direction="long",
            conviction=conviction,
            strategy_name="vwap",
            hold_type=HoldType.DAY,  # Always a day trade — costs 1 PDT slot
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            metadata={
                "vwap": vwap,
                "dip_pct": dip_pct,
                "reclaim_pct": reclaim_pct,
                "bar_volume": bar_volume,
                "avg_bar_volume": avg_bar_vol,
            },
        )

    def should_exit(
        self, symbol: str, bars: pd.DataFrame, entry_price: float
    ) -> bool:
        """Exit if price falls back below VWAP — the reclaim has failed."""
        df = bars.copy()
        add_vwap(df)

        latest = df.iloc[-1]
        close: float = latest["close"]
        vwap: float = latest["vwap_calc"]

        if close < vwap:
            logger.info(
                "vwap_exit",
                symbol=symbol,
                close=close,
                vwap=vwap,
            )
            return True

        return False
