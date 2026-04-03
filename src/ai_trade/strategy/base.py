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

import pandas as pd


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
        # getattr() with a default is used because the config may not
        # always have an "enabled" key (defensive programming).
        self.enabled = getattr(config, "enabled", True)

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
