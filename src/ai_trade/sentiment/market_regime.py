"""Market regime analysis — determines overall market conditions.

WHAT THIS MODULE DOES:
    Analyzes broad market indicators (SPY, QQQ, VIX) to classify the
    current market environment into one of five regimes:
        STRONG_BULL → BULL → NEUTRAL → BEAR → STRONG_BEAR

    Each regime comes with modifiers that adjust how aggressively the
    bot trades:
    - conviction_modifier:     scales signal quality scores (0.3 to 1.3)
    - position_size_modifier:  scales position sizes (0.25 to 1.0)
    - allow_new_longs:         hard gate on new long positions
    - allow_options:           hard gate on options strategies

WHY IT EXISTS:
    Individual stock signals don't exist in a vacuum.  A great-looking
    breakout signal is far less reliable if the entire market is in a
    downtrend.  This module acts as a "top-down filter" — it adjusts
    trade aggressiveness based on the macro environment.

TRADING CONCEPTS EXPLAINED:

    SPY: An ETF (Exchange-Traded Fund) that tracks the S&P 500 index —
        the 500 largest US companies.  SPY's direction is the single best
        proxy for "is the overall stock market going up or down?"

    QQQ: An ETF tracking the Nasdaq-100 — heavily weighted toward tech
        stocks.  Comparing QQQ to SPY tells us if tech is leading or
        lagging the broader market.

    VIX: The CBOE Volatility Index, often called the "fear index."  It
        measures expected volatility (price swings) over the next 30 days,
        derived from S&P 500 option prices.
        - VIX < 15:  low fear, complacency (bullish for stocks)
        - VIX 15-22: normal conditions
        - VIX 22-30: elevated fear (caution warranted)
        - VIX > 30:  panic / crisis (avoid new long positions)

    EMA (Exponential Moving Average): A smoothed average of recent prices
        that gives more weight to recent data.  Common periods:
        - EMA-9/20:  short-term trend (days to weeks)
        - EMA-50:    medium-term trend (weeks to months)
        - EMA-200:   long-term trend (months to years)
        When price is ABOVE an EMA, the trend at that timeframe is "up."

    RSI (Relative Strength Index): A momentum oscillator ranging 0-100.
        - RSI > 70: "overbought" (may be due for a pullback)
        - RSI < 30: "oversold" (may be due for a bounce)
        - RSI 40-60: neutral zone

    MACD (Moving Average Convergence Divergence): A trend-following
        momentum indicator.  The "histogram" is the difference between the
        MACD line and its signal line.  Positive histogram = bullish
        momentum, negative = bearish momentum.

    Market Breadth: A composite score (-1.0 to +1.0) that aggregates
        multiple signals into a single number.  More components agreeing
        on direction → higher confidence in the regime classification.

Regime classifications:
    STRONG_BULL  — Aggressively take signals, boost conviction
    BULL         — Normal operation, slight conviction boost
    NEUTRAL      — Tighten entries, only high-conviction trades
    BEAR         — Reduce position sizes, avoid new longs except mean reversion
    STRONG_BEAR  — Halt new longs entirely, only defensive plays
"""

from __future__ import annotations

# PYTHON PATTERN — @dataclass:
# A dataclass auto-generates __init__, __repr__, and __eq__ methods from
# the class's type-annotated fields.  It's a concise way to define "data
# container" classes without writing boilerplate.
from dataclasses import dataclass
from datetime import datetime, timedelta

# PYTHON PATTERN — Enum:
# An Enum defines a fixed set of named constants.  This is safer than using
# plain strings because typos cause immediate errors instead of silent bugs.
from enum import Enum

# ZoneInfo is Python's built-in timezone database (added in Python 3.9).
# It replaces the older pytz library.
from zoneinfo import ZoneInfo

import pandas as pd

from ai_trade.data.indicators import add_ema, add_rsi, add_atr, add_macd
from ai_trade.monitoring.logger import get_logger

log = get_logger(__name__)

# Eastern Time — the timezone of US stock markets.
# Market hours are 9:30 AM - 4:00 PM ET.
ET = ZoneInfo("America/New_York")


class MarketRegime(Enum):
    """Enumeration of the five possible market regime classifications.

    PYTHON PATTERN — Enum:
        Each member has a name (e.g. STRONG_BULL) and a value (e.g.
        "strong_bull").  You compare enums with `==` or `is`, and access
        the string value with `.value`.  Enums prevent invalid states —
        you can't accidentally create a "SORTA_BULL" regime.
    """
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"


@dataclass
class MarketContext:
    """Snapshot of current market conditions.

    PYTHON PATTERN — @dataclass:
        The `@dataclass` decorator auto-generates an `__init__` method that
        accepts each annotated field as a parameter.  So declaring
        `regime: MarketRegime` below means the constructor accepts
        `MarketContext(regime=..., conviction_modifier=..., ...)`.

    This object is passed to strategy code so it can adjust its behavior
    based on market conditions.
    """

    regime: MarketRegime
    conviction_modifier: float   # Multiply strategy conviction by this (0.0 - 1.5)
    position_size_modifier: float  # Multiply position size by this (0.0 - 1.0)
    allow_new_longs: bool        # Whether new long positions are permitted
    allow_options: bool          # Whether options strategies are permitted

    # Underlying data used to derive the regime — stored for logging and debugging.
    spy_trend: str       # "up", "down", or "sideways"
    spy_rsi: float       # RSI reading for SPY (0-100 scale)
    spy_above_20ema: bool  # Is SPY above its 20-day EMA?
    spy_above_50ema: bool  # Is SPY above its 50-day EMA?
    spy_above_200ema: bool  # Is SPY above its 200-day EMA?
    qqq_trend: str       # "up", "down", or "sideways"
    vix_level: float     # Current VIX closing price
    vix_trend: str       # "rising", "falling", or "stable"
    breadth_score: float  # Composite score: -1.0 (all bearish) to +1.0 (all bullish)

    summary: str = ""    # Human-readable summary of notable conditions

    def __str__(self) -> str:
        """Human-readable one-line summary.

        PYTHON PATTERN — __str__:
            This "dunder" (double-underscore) method is called when you do
            str(obj) or print(obj).  It lets you control how the object
            displays as text.  The `:+.2f` format spec includes a + or -
            sign before the number.
        """
        return (
            f"Regime: {self.regime.value} | "
            f"SPY: {self.spy_trend} (RSI {self.spy_rsi:.0f}) | "
            f"VIX: {self.vix_level:.1f} ({self.vix_trend}) | "
            f"Breadth: {self.breadth_score:+.2f} | "
            f"Conv mod: {self.conviction_modifier:.2f}"
        )


# ── VIX thresholds ───────────────────────────────────────────
# These constants define the boundaries between VIX "zones."
# They are based on historical VIX behavior:
# - VIX averages around 15-20 in calm markets.
# - VIX spikes above 30 during corrections and crises.
# - VIX hit 80+ during the 2020 COVID crash.
_VIX_LOW = 15.0       # Below this: low fear, complacent market
_VIX_ELEVATED = 22.0  # Above this: caution warranted
_VIX_HIGH = 30.0      # Above this: panic / crisis conditions


class MarketRegimeAnalyzer:
    """Analyzes broad market indicators to classify the current regime.

    Call `analyze()` once pre-market (or at market open) to get a
    `MarketContext` that gates and modifies all strategy decisions for
    the day.

    The analysis pipeline:
    1. Compute technical indicators on SPY (EMA, RSI, MACD).
    2. Compute indicators on QQQ (EMA, RSI).
    3. Analyze VIX level and trend.
    4. Calculate a composite "breadth score" from all inputs.
    5. Map the breadth score to a regime classification.
    6. Apply overrides (e.g. VIX spike forces caution).
    7. Look up modifiers for the classified regime.
    """

    def __init__(self):
        # Cache the last computed context so other code can access it
        # without re-running the analysis.
        self._last_context: MarketContext | None = None

    @property
    def context(self) -> MarketContext | None:
        """Most recently computed market context.

        Returns None if analyze() has never been called.
        """
        return self._last_context

    def analyze(
        self,
        spy_bars: pd.DataFrame,
        qqq_bars: pd.DataFrame,
        vix_bars: pd.DataFrame | None = None,
    ) -> MarketContext:
        """Analyze market conditions from daily bars.

        This is the main entry point.  It takes historical daily price
        data (as pandas DataFrames) for SPY, QQQ, and optionally VIX,
        and returns a MarketContext with the regime classification and
        all modifiers.

        PYTHON PATTERN — pandas DataFrame:
            A DataFrame is a 2D table (like a spreadsheet or SQL table).
            Each column is a data series (e.g. "close", "volume"), and
            each row represents one time period (one trading day).
            `.iloc[-1]` gets the LAST row (most recent day).
            `.iloc[-6:-1]` gets the 5 rows before the last one.

        Args:
            spy_bars: SPY daily bars (at least 200 rows for EMA-200).
            qqq_bars: QQQ daily bars.
            vix_bars: VIX daily bars (optional; if None, VIX analysis is skipped).

        Returns:
            MarketContext with regime classification and modifiers.
        """
        # .copy() creates a shallow copy of the DataFrame so our indicator
        # calculations (which add columns) don't modify the caller's data.
        spy = spy_bars.copy()
        qqq = qqq_bars.copy()

        # ── SPY analysis ────────────────────────────────────
        # Add technical indicators as new columns to the DataFrame.
        # After this, spy will have columns like "ema_20", "rsi_14", "macd_hist".
        add_ema(spy, [9, 20, 50, 200])
        add_rsi(spy, 14)
        add_macd(spy)

        # Get the most recent day's values.
        # `.iloc[-1]` indexes from the end: -1 = last row, -2 = second-to-last, etc.
        latest_spy = spy.iloc[-1]
        spy_close = float(latest_spy["close"])
        # `.get("column", default)` returns default if the column doesn't exist.
        spy_rsi = float(latest_spy.get("rsi_14", 50))
        spy_ema20 = float(latest_spy.get("ema_20", spy_close))
        spy_ema50 = float(latest_spy.get("ema_50", spy_close))
        spy_ema200 = float(latest_spy.get("ema_200", spy_close))
        spy_macd_hist = float(latest_spy.get("macd_hist", 0))

        # Determine which EMAs price is above.
        spy_above_20 = spy_close > spy_ema20
        spy_above_50 = spy_close > spy_ema50
        spy_above_200 = spy_close > spy_ema200

        # SPY trend classification from EMA "stack" order.
        # TRADING CONCEPT — EMA Stack:
        # When price is above ALL major EMAs (20, 50, 200), the trend is
        # unambiguously "up" at all timeframes.  When below all, it's "down."
        # Mixed signals (above some, below others) indicate "sideways" or
        # transitional conditions.
        if spy_above_20 and spy_above_50 and spy_above_200:
            spy_trend = "up"
        elif not spy_above_20 and not spy_above_50 and not spy_above_200:
            spy_trend = "down"
        else:
            spy_trend = "sideways"

        # SPY momentum — 5-day return (how much SPY moved in the last week).
        # A large negative return signals a sharp selloff; a large positive
        # return signals a strong rally.
        if len(spy) >= 6:
            spy_5d_return = (spy_close - float(spy["close"].iloc[-6])) / float(spy["close"].iloc[-6])
        else:
            spy_5d_return = 0.0

        # ── QQQ analysis ────────────────────────────────────
        # Same approach as SPY but with fewer indicators (we only need
        # trend direction from QQQ, not detailed momentum).
        add_ema(qqq, [20, 50])
        add_rsi(qqq, 14)

        latest_qqq = qqq.iloc[-1]
        qqq_close = float(latest_qqq["close"])
        qqq_ema20 = float(latest_qqq.get("ema_20", qqq_close))
        qqq_ema50 = float(latest_qqq.get("ema_50", qqq_close))

        if qqq_close > qqq_ema20 and qqq_close > qqq_ema50:
            qqq_trend = "up"
        elif qqq_close < qqq_ema20 and qqq_close < qqq_ema50:
            qqq_trend = "down"
        else:
            qqq_trend = "sideways"

        # ── VIX analysis ────────────────────────────────────
        # VIX defaults to 18.0 (a moderate/"normal" level) if no data
        # is available, so the system degrades gracefully.
        vix_level = 18.0  # default moderate
        vix_trend = "stable"

        if vix_bars is not None and not vix_bars.empty:
            latest_vix = vix_bars.iloc[-1]
            vix_level = float(latest_vix["close"])

            # VIX trend: compare current level to its 5-day average.
            # A 10%+ increase = "rising" (fear increasing).
            # A 10%+ decrease = "falling" (fear decreasing).
            if len(vix_bars) >= 6:
                vix_5d_avg = float(vix_bars["close"].iloc[-6:-1].mean())
                vix_change = (vix_level - vix_5d_avg) / vix_5d_avg if vix_5d_avg > 0 else 0
                if vix_change > 0.10:
                    vix_trend = "rising"
                elif vix_change < -0.10:
                    vix_trend = "falling"
                else:
                    vix_trend = "stable"

        # ── Breadth score calculation ────────────────────────
        # The breadth score is a weighted average of multiple market signals,
        # normalized to the range [-1.0, +1.0].  Each component votes
        # bullish (+) or bearish (-), and the aggregate determines regime.
        #
        # Components and their weights:
        #   SPY trend:      ±1.0 (strongest single signal)
        #   QQQ trend:      ±1.0 (confirms or diverges from SPY)
        #   SPY RSI:        ±0.5 (momentum confirmation)
        #   VIX level:      ±0.5 to ±1.0 (fear gauge)
        #   VIX trend:      ±0.5 (direction of fear)
        #   MACD histogram: ±0.5 (trend momentum)
        #   5-day return:   ±0.5 (short-term momentum)
        breadth_points = 0.0
        breadth_count = 0

        # SPY trend component — the primary directional signal.
        if spy_trend == "up":
            breadth_points += 1.0
        elif spy_trend == "down":
            breadth_points -= 1.0
        breadth_count += 1

        # QQQ trend component — tech sector confirmation.
        if qqq_trend == "up":
            breadth_points += 1.0
        elif qqq_trend == "down":
            breadth_points -= 1.0
        breadth_count += 1

        # SPY RSI component — momentum above 60 is bullish, below 40 is bearish.
        if spy_rsi > 60:
            breadth_points += 0.5
        elif spy_rsi < 40:
            breadth_points -= 0.5
        breadth_count += 1

        # VIX level component (INVERTED: high VIX = bearish for stocks).
        # VIX measures fear, so high VIX is BAD for long positions.
        if vix_level < _VIX_LOW:
            breadth_points += 1.0    # Low fear → bullish
        elif vix_level > _VIX_HIGH:
            breadth_points -= 1.0    # Panic → strongly bearish
        elif vix_level > _VIX_ELEVATED:
            breadth_points -= 0.5    # Elevated fear → mildly bearish
        breadth_count += 1

        # VIX trend component — falling VIX = improving sentiment.
        if vix_trend == "falling":
            breadth_points += 0.5
        elif vix_trend == "rising":
            breadth_points -= 0.5
        breadth_count += 1

        # MACD histogram direction — positive = bullish momentum.
        if spy_macd_hist > 0:
            breadth_points += 0.5
        elif spy_macd_hist < 0:
            breadth_points -= 0.5
        breadth_count += 1

        # 5-day momentum — a >2% move in either direction is significant.
        if spy_5d_return > 0.02:
            breadth_points += 0.5
        elif spy_5d_return < -0.02:
            breadth_points -= 0.5
        breadth_count += 1

        # Normalize to [-1.0, +1.0] by dividing by the number of components.
        breadth_score = breadth_points / breadth_count if breadth_count > 0 else 0.0

        # ── Classify regime from breadth score ───────────────
        # The boundaries create 5 roughly-equal zones:
        #   >= +0.6  → STRONG_BULL  (most signals are bullish)
        #   >= +0.2  → BULL         (majority bullish)
        #   >= -0.2  → NEUTRAL      (mixed signals)
        #   >= -0.6  → BEAR         (majority bearish)
        #   < -0.6   → STRONG_BEAR  (most signals are bearish)
        if breadth_score >= 0.6:
            regime = MarketRegime.STRONG_BULL
        elif breadth_score >= 0.2:
            regime = MarketRegime.BULL
        elif breadth_score >= -0.2:
            regime = MarketRegime.NEUTRAL
        elif breadth_score >= -0.6:
            regime = MarketRegime.BEAR
        else:
            regime = MarketRegime.STRONG_BEAR

        # Override: VIX spike always forces caution regardless of other signals.
        # If VIX is above 30 (panic territory), we downgrade any bullish
        # regime to NEUTRAL.  This prevents the bot from going all-in
        # during a market crash that happens to have a few bullish signals.
        if vix_level > _VIX_HIGH and regime in (MarketRegime.STRONG_BULL, MarketRegime.BULL):
            regime = MarketRegime.NEUTRAL

        # ── Look up modifiers for the classified regime ──────
        # Each regime maps to a set of modifiers that control trade sizing
        # and gating.  These are the "tuning knobs" of the system.
        modifiers = {
            MarketRegime.STRONG_BULL: {
                "conviction_modifier": 1.3,        # +30% conviction boost
                "position_size_modifier": 1.0,     # Full position sizes
                "allow_new_longs": True,
                "allow_options": True,
            },
            MarketRegime.BULL: {
                "conviction_modifier": 1.1,        # +10% conviction boost
                "position_size_modifier": 1.0,     # Full position sizes
                "allow_new_longs": True,
                "allow_options": True,
            },
            MarketRegime.NEUTRAL: {
                "conviction_modifier": 0.9,        # -10% conviction penalty
                "position_size_modifier": 0.75,    # 75% position sizes
                "allow_new_longs": True,
                "allow_options": True,
            },
            MarketRegime.BEAR: {
                "conviction_modifier": 0.6,        # -40% conviction penalty
                "position_size_modifier": 0.5,     # Half position sizes
                "allow_new_longs": True,           # Only high conviction gets through
                "allow_options": False,            # No options — too risky
            },
            MarketRegime.STRONG_BEAR: {
                "conviction_modifier": 0.3,        # -70% conviction penalty (nearly kills all signals)
                "position_size_modifier": 0.25,    # Quarter position sizes
                "allow_new_longs": False,          # Hard stop on new longs
                "allow_options": False,
            },
        }

        mods = modifiers[regime]

        # Build a human-readable summary of notable conditions for logging.
        summary_parts = []
        if spy_trend == "up":
            summary_parts.append("SPY uptrend")
        elif spy_trend == "down":
            summary_parts.append("SPY downtrend")
        if vix_level > _VIX_ELEVATED:
            summary_parts.append(f"VIX elevated ({vix_level:.1f})")
        if vix_trend == "rising":
            summary_parts.append("VIX rising")
        if spy_5d_return < -0.03:
            summary_parts.append(f"SPY 5d selloff ({spy_5d_return:.1%})")
        elif spy_5d_return > 0.03:
            summary_parts.append(f"SPY 5d rally ({spy_5d_return:+.1%})")

        # Construct the final MarketContext with all computed values.
        ctx = MarketContext(
            regime=regime,
            conviction_modifier=mods["conviction_modifier"],
            position_size_modifier=mods["position_size_modifier"],
            allow_new_longs=mods["allow_new_longs"],
            allow_options=mods["allow_options"],
            spy_trend=spy_trend,
            spy_rsi=spy_rsi,
            spy_above_20ema=spy_above_20,
            spy_above_50ema=spy_above_50,
            spy_above_200ema=spy_above_200,
            qqq_trend=qqq_trend,
            vix_level=vix_level,
            vix_trend=vix_trend,
            breadth_score=breadth_score,
            summary="; ".join(summary_parts) if summary_parts else "normal conditions",
        )

        # Cache for later access via the .context property.
        self._last_context = ctx

        log.info(
            "market_regime_analyzed",
            regime=regime.value,
            breadth=round(breadth_score, 3),
            spy_trend=spy_trend,
            spy_rsi=round(spy_rsi, 1),
            vix=round(vix_level, 1),
            vix_trend=vix_trend,
            conv_mod=mods["conviction_modifier"],
            size_mod=mods["position_size_modifier"],
            allow_longs=mods["allow_new_longs"],
        )

        return ctx
