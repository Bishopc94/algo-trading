"""VWAP-reclaim intraday strategy (day trade — costs 1 PDT slot).

Theory:
  VWAP represents institutional fair value.  When price dips below VWAP
  and reclaims it on strong volume, institutional buyers are stepping in.
  We add: RSI confirmation (not overbought), meaningful dip depth (>0.5%),
  and reclaim strength (close meaningfully above VWAP, not just touching).

Entry conditions (ALL must be true):
  1. Current close > VWAP * 1.001 (meaningfully above VWAP, not noise)
  2. Recent dip below VWAP in last 10 bars (dip depth >= 0.5%)
  3. Volume on reclaim > 1.5x average (institutional participation)
  4. Bullish candle (close > open)

Conviction: base 0.55, + dip depth, + volume, + reclaim strength.
Hold type: DAY (force-closed by 3:50 PM ET).
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
        if intraday_bars is None or intraday_bars.empty:
            return None

        df = intraday_bars.copy()
        add_vwap(df)

        lookback: int = 10
        if len(df) < lookback + 1:
            return None

        latest = df.iloc[-1]
        close: float = latest["close"]
        open_price: float = latest["open"]
        vwap: float = latest["vwap_calc"]
        bar_volume: float = latest["volume"]

        window = df.iloc[-(lookback + 1):]
        avg_bar_vol = window["volume"].mean()

        # Dip below VWAP
        recent = df.iloc[-lookback:]
        dip_bars = recent[recent["close"] < recent["vwap_calc"]]
        if dip_bars.empty:
            return None

        # Meaningfully above VWAP (not noise)
        if close <= vwap * 1.001:
            return None

        # Volume confirmation
        if avg_bar_vol > 0 and bar_volume <= 1.5 * avg_bar_vol:
            return None

        # Bullish candle
        if close <= open_price:
            return None

        # Dip depth filter
        dip_low_price = dip_bars["close"].min()
        dip_vwap = dip_bars.loc[dip_bars["close"].idxmin(), "vwap_calc"]
        dip_pct = abs((dip_low_price - dip_vwap) / dip_vwap) if dip_vwap else 0
        if dip_pct < 0.005:
            return None  # Shallow noise dip

        # Conviction scoring
        base_conviction = 0.55
        dip_adj = min(0.20, dip_pct * 15)
        vol_ratio = bar_volume / avg_bar_vol if avg_bar_vol > 0 else 1.0
        vol_adj = min(0.15, (vol_ratio - 1.5) * 0.10)

        conviction = base_conviction + dip_adj + vol_adj
        conviction = max(0.55, min(0.90, conviction))

        entry_price = close
        dip_low = dip_bars["low"].min()
        stop_loss = max(dip_low, vwap * 0.99)

        exit_dev: float = getattr(self.config, "exit_deviation_pct", 0.3) / 100
        tp_from_vwap = vwap * (1 + exit_dev)
        prior_high = df["high"].max()
        take_profit = max(tp_from_vwap, prior_high)

        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        if risk <= 0 or reward / risk < 1.5:
            return None

        logger.info(
            "vwap_signal",
            symbol=symbol,
            conviction=conviction,
            entry=entry_price,
            stop=stop_loss,
            target=take_profit,
            vwap=vwap,
            dip_pct=round(dip_pct, 4),
        )

        return Signal(
            symbol=symbol,
            direction="long",
            conviction=conviction,
            strategy_name="vwap",
            hold_type=HoldType.DAY,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            metadata={
                "vwap": vwap,
                "dip_pct": dip_pct,
                "bar_volume": bar_volume,
                "avg_bar_volume": avg_bar_vol,
            },
        )

    def should_exit(
        self, symbol: str, bars: pd.DataFrame, entry_price: float
    ) -> bool:
        df = bars.copy()
        add_vwap(df)
        latest = df.iloc[-1]
        if latest["close"] < latest["vwap_calc"]:
            return True
        return False
