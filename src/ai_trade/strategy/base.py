"""Strategy interface and shared data structures.

This module defines the core abstractions that every trading strategy must
implement.  It uses two important Python patterns:

  1. **Dataclasses** (``@dataclass``): Auto-generated classes with typed
     fields.  Think of them like structs — they hold data without needing
     boilerplate constructors, ``__repr__``, or equality methods.

  2. **Abstract Base Classes** (``ABC`` + ``@abstractmethod``): Defines a
     contract that subclasses MUST implement.  Any class inheriting from
     ``BaseStrategy`` that doesn't implement ``evaluate()`` and
     ``should_exit()`` will raise a ``TypeError`` at instantiation time.
     This is the same concept as interfaces in Java/C# or pure virtual
     functions in C++.

Key concepts:
  - **Signal**: A recommendation to enter a trade.  Contains the symbol,
    direction, entry/exit prices, conviction score, and hold type.
  - **HoldType**: Determines whether a trade will be a day trade (closed
    same day, costs 1 PDT slot) or a swing trade (held overnight, free).
  - **Conviction**: A 0.0–1.0 score representing the strategy's confidence
    in the signal.  Higher conviction = larger position size and priority
    in the execution queue.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

import pandas as pd


@dataclass
class Rejection:
    """Structured rejection from a strategy's evaluate() method.

    Captures which filter failed, the actual vs threshold values, and
    whether the rejection was a near-miss (close to passing).
    """
    symbol: str
    strategy: str
    filter_name: str
    actual: float
    threshold: float
    direction: str = "above"  # "above" = needed actual > threshold; "below" = needed actual < threshold
    miss_pct: float | None = None  # % distance from passing; set automatically

    NEAR_MISS_PCT: ClassVar[float] = 15.0  # anything within 15% of threshold is a near-miss

    def __post_init__(self) -> None:
        if self.threshold != 0:
            self.miss_pct = abs(self.actual - self.threshold) / abs(self.threshold) * 100
        else:
            self.miss_pct = None

    @property
    def is_near_miss(self) -> bool:
        return self.miss_pct is not None and self.miss_pct <= self.NEAR_MISS_PCT

    def to_reasoning(self) -> str:
        dir_word = "needed >=" if self.direction == "above" else "needed <="
        miss = f" (miss by {self.miss_pct:.1f}%)" if self.miss_pct is not None else ""
        return f"{self.filter_name}: {self.actual:.4g} {dir_word} {self.threshold:.4g}{miss}"


class HoldType(Enum):
    """How long a position will be held.

    - DAY:      Close by end of day.  Costs 1 PDT slot.
    - SWING:    Hold overnight (2+ days).  Free — no PDT cost.
    - ADAPTIVE: Strategy decides at signal time based on conviction.
                High conviction (>=0.9) → DAY; otherwise → SWING.
    """
    DAY = "day"
    SWING = "swing"
    ADAPTIVE = "adaptive"


@dataclass
class Signal:
    """A trade recommendation produced by a strategy.

    The signal aggregator collects Signals from all strategies, ranks them
    by priority, and passes the best ones to the order manager for execution.

    Attributes:
        symbol:           Stock ticker (e.g. "AAPL").
        direction:        Always "long" — cash accounts can't short.
        conviction:       0.0 to 1.0 confidence score.  Used for ranking
                          and position sizing.
        strategy_name:    Which strategy produced this signal (for logging).
        hold_type:        DAY, SWING, or ADAPTIVE.
        entry_price:      Expected entry price (latest close).
        stop_loss_price:  Server-side stop-loss price (bracket order).
        take_profit_price: Server-side take-profit price (bracket order).
        metadata:         Extra data for logging/debugging (RSI, ATR, etc.).
    """
    symbol: str
    direction: str  # always "long" for cash account (no shorting)
    conviction: float  # 0.0 to 1.0
    strategy_name: str
    hold_type: HoldType
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    metadata: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base class that all stock trading strategies must extend.

    Subclasses must implement:
      - ``evaluate()``: Analyse price data and return a Signal (or None).
      - ``should_exit()``: Check if an existing position should be closed.

    The ``config`` parameter is a SimpleNamespace loaded from
    ``settings.yaml`` containing strategy-specific parameters like RSI
    thresholds, ATR multipliers, and hold types.
    """

    def __init__(self, config):
        self.config = config
        self.enabled = getattr(config, "enabled", True)
        self._rejections: list[Rejection] = []
        # Optional market context (VIX, regime) set by the aggregator before
        # each evaluate() call. Strategies read it to scale stop/target widths.
        self.market_context = None

    def _reject(
        self, symbol: str, filter_name: str,
        actual: float, threshold: float, direction: str = "above",
    ) -> None:
        """Record a rejection with the filter that failed."""
        self._rejections.append(Rejection(
            symbol=symbol,
            strategy=type(self).__name__,
            filter_name=filter_name,
            actual=actual,
            threshold=threshold,
            direction=direction,
        ))

    def drain_rejections(self) -> list[Rejection]:
        """Return and clear all buffered rejections."""
        rej = self._rejections
        self._rejections = []
        return rej

    def _plan_long_exit(
        self,
        bars: pd.DataFrame,
        entry_price: float,
        atr: float,
        base_stop_mult: float,
        base_tp_mult: float,
    ):
        """Compute S/R-aware stop + target using the shared ExitPlanner.

        Pulls VIX/regime from self.market_context when available, so per-trade
        widths adapt to volatility and regime without threading ctx through
        every strategy signature.
        """
        from ai_trade.strategy.exit_planner import plan_long_exit
        ctx = self.market_context
        vix = getattr(ctx, "vix_level", None) if ctx else None
        regime_enum = getattr(ctx, "regime", None) if ctx else None
        regime = getattr(regime_enum, "value", None) if regime_enum else None
        return plan_long_exit(
            bars=bars, entry_price=entry_price, atr=atr,
            base_stop_mult=base_stop_mult, base_tp_mult=base_tp_mult,
            vix=vix, regime=regime,
        )

    @abstractmethod
    def evaluate(
        self,
        symbol: str,
        daily_bars: pd.DataFrame,
        intraday_bars: pd.DataFrame | None = None,
    ) -> Signal | None:
        """Analyse price data and return a Signal if entry conditions are met.

        Args:
            symbol:        Stock ticker (e.g. "AAPL").
            daily_bars:    DataFrame of daily OHLCV bars with indicators
                           already computed.
            intraday_bars: Optional DataFrame of minute bars (used only by
                           the VWAP strategy).

        Returns:
            A Signal object if the strategy wants to enter a trade, or
            None if conditions are not met.
        """
        pass

    @abstractmethod
    def should_exit(
        self, symbol: str, bars: pd.DataFrame, entry_price: float
    ) -> bool:
        """Check whether an existing position should be closed.

        This is called by the backtester to simulate exits.  In live
        trading, bracket orders handle exits server-side, but this method
        provides a software-level exit check as a safety net.

        Args:
            symbol:      Stock ticker.
            bars:        Recent price bars.
            entry_price: The price at which the position was entered.

        Returns:
            True if the position should be closed, False otherwise.
        """
        pass
