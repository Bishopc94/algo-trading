"""Market prediction module — multi-timeframe trend + momentum scoring.

Moves the bot from reactive (indicators say what happened) to predictive
(anticipate what's likely to happen in the next 1-3 bars).

Components:
    1. Momentum score — composite of RSI rate-of-change, MACD acceleration,
       EMA slopes, and volume trend. Predicts directional continuation.
    2. Multi-timeframe trend — weekly bars classify the higher-timeframe
       trend (up/down/flat). Daily signals aligned with the weekly trend
       get a conviction boost; counter-trend signals get a penalty.
    3. Sector strength ranking — uses ETF proxies to track which sectors
       are leading/lagging. Strategies can use this to bias toward
       sectors with positive momentum.

All functions are pure — they take DataFrames and return scores.
No API calls, no database access, no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai_trade.monitoring.logger import get_logger

log = get_logger(__name__)


@dataclass
class MomentumScore:
    """Composite momentum prediction for a single symbol."""
    symbol: str
    rsi_momentum: float       # RSI rate-of-change (positive = strengthening)
    macd_acceleration: float  # MACD histogram 2nd derivative
    ema_alignment: float      # EMA slope agreement (0-1)
    volume_support: float     # Volume trend (positive = growing participation)
    composite: float          # Weighted composite score (-1 to +1)
    predicted_direction: str  # "bullish", "bearish", "neutral"


@dataclass
class WeeklyTrend:
    """Higher-timeframe trend classification from weekly bars."""
    direction: str   # "up", "down", "flat"
    strength: float  # 0.0 to 1.0
    ema_20w_slope: float
    price_vs_ema_20w: float  # % above/below weekly EMA-20


@dataclass
class SectorStrength:
    """Relative strength ranking for a sector ETF."""
    sector: str
    etf: str
    roc_5d: float   # 5-day return %
    roc_20d: float  # 20-day return %
    rank: int        # 1 = strongest


# ── Sector ETF proxies ───────────────────────────────────────
SECTOR_ETFS = {
    "tech": "XLK",
    "healthcare": "XLV",
    "financials": "XLF",
    "energy": "XLE",
    "consumer_disc": "XLY",
    "consumer_staples": "XLP",
    "industrials": "XLI",
    "materials": "XLB",
    "utilities": "XLU",
    "real_estate": "XLRE",
    "communication": "XLC",
}


def compute_momentum_score(df: pd.DataFrame, symbol: str = "") -> MomentumScore:
    """Compute a composite momentum prediction from enriched daily bars.

    Requires columns from ``add_momentum_prediction()`` in indicators.py:
    rsi_roc_3, macd_accel, ema_9_slope, ema_20_slope, price_roc_5, volume_trend.

    Returns a MomentumScore with composite in [-1.0, +1.0].
    """
    if df.empty or len(df) < 5:
        return MomentumScore(
            symbol=symbol, rsi_momentum=0, macd_acceleration=0,
            ema_alignment=0, volume_support=0, composite=0,
            predicted_direction="neutral",
        )

    latest = df.iloc[-1]

    # 1. RSI momentum — normalized to [-1, +1] (RSI moves ~3-5 pts/bar normally)
    rsi_roc = float(latest.get("rsi_roc_3", 0) or 0)
    rsi_momentum = max(-1.0, min(1.0, rsi_roc / 10.0))

    # 2. MACD acceleration — positive = histogram growing (momentum building)
    macd_accel = float(latest.get("macd_accel", 0) or 0)
    macd_norm = max(-1.0, min(1.0, macd_accel / 0.5))

    # 3. EMA alignment — do EMA slopes agree on direction?
    ema_9_slope = float(latest.get("ema_9_slope", 0) or 0)
    ema_20_slope = float(latest.get("ema_20_slope", 0) or 0)

    if ema_9_slope > 0 and ema_20_slope > 0:
        ema_alignment = min(1.0, (ema_9_slope + ema_20_slope) / 2.0)
    elif ema_9_slope < 0 and ema_20_slope < 0:
        ema_alignment = max(-1.0, (ema_9_slope + ema_20_slope) / 2.0)
    else:
        ema_alignment = 0.0  # Disagreement → neutral

    # 4. Volume trend — growing volume confirms the move
    vol_trend = float(latest.get("volume_trend", 0) or 0)
    volume_support = max(-1.0, min(1.0, vol_trend))

    # Weighted composite: RSI 30%, MACD 30%, EMA 25%, Volume 15%
    composite = (
        0.30 * rsi_momentum
        + 0.30 * macd_norm
        + 0.25 * ema_alignment
        + 0.15 * volume_support
    )
    composite = max(-1.0, min(1.0, composite))

    if composite > 0.15:
        predicted_direction = "bullish"
    elif composite < -0.15:
        predicted_direction = "bearish"
    else:
        predicted_direction = "neutral"

    return MomentumScore(
        symbol=symbol,
        rsi_momentum=round(rsi_momentum, 3),
        macd_acceleration=round(macd_norm, 3),
        ema_alignment=round(ema_alignment, 3),
        volume_support=round(volume_support, 3),
        composite=round(composite, 3),
        predicted_direction=predicted_direction,
    )


def classify_weekly_trend(weekly_bars: pd.DataFrame) -> WeeklyTrend:
    """Classify higher-timeframe trend from weekly bars.

    Uses 20-week EMA slope and price position relative to it.
    Requires at least 25 weekly bars for meaningful analysis.
    """
    if weekly_bars.empty or len(weekly_bars) < 25:
        return WeeklyTrend(direction="flat", strength=0.0, ema_20w_slope=0.0, price_vs_ema_20w=0.0)

    close = weekly_bars["close"]
    ema_20 = close.ewm(span=20, adjust=False).mean()

    latest_price = float(close.iloc[-1])
    latest_ema = float(ema_20.iloc[-1])
    prev_ema = float(ema_20.iloc[-4]) if len(ema_20) > 4 else latest_ema

    # EMA slope (annualized rate of change as %)
    ema_slope = ((latest_ema - prev_ema) / max(prev_ema, 0.01)) * 100

    # Price vs EMA (%)
    price_vs_ema = ((latest_price - latest_ema) / max(latest_ema, 0.01)) * 100

    # Classification
    if ema_slope > 0.5 and price_vs_ema > 1.0:
        direction = "up"
        strength = min(1.0, (ema_slope + abs(price_vs_ema)) / 10.0)
    elif ema_slope < -0.5 and price_vs_ema < -1.0:
        direction = "down"
        strength = min(1.0, (abs(ema_slope) + abs(price_vs_ema)) / 10.0)
    else:
        direction = "flat"
        strength = 0.0

    return WeeklyTrend(
        direction=direction,
        strength=round(strength, 3),
        ema_20w_slope=round(ema_slope, 3),
        price_vs_ema_20w=round(price_vs_ema, 3),
    )


def weekly_trend_modifier(trend: WeeklyTrend, signal_direction: str = "long") -> float:
    """Compute a conviction modifier based on weekly trend alignment.

    - Signal aligned with weekly trend: boost up to +15%
    - Signal counter to weekly trend: penalty up to -20%
    - Flat/neutral trend: no change
    """
    if trend.direction == "flat" or trend.strength < 0.1:
        return 1.0

    aligned = (
        (trend.direction == "up" and signal_direction == "long")
        or (trend.direction == "down" and signal_direction == "short")
    )

    if aligned:
        return round(1.0 + 0.15 * trend.strength, 3)
    else:
        return round(1.0 - 0.20 * trend.strength, 3)


def rank_sector_strength(sector_bars: dict[str, pd.DataFrame]) -> list[SectorStrength]:
    """Rank sectors by recent momentum using ETF proxy bars.

    Args:
        sector_bars: Dict mapping sector name to ETF daily bars DataFrame.

    Returns:
        List of SectorStrength sorted by composite score (strongest first).
    """
    results: list[SectorStrength] = []

    for sector, etf in SECTOR_ETFS.items():
        df = sector_bars.get(sector)
        if df is None or len(df) < 25:
            continue

        close = df["close"]
        roc_5 = float((close.iloc[-1] / close.iloc[-5] - 1) * 100) if len(close) >= 5 else 0.0
        roc_20 = float((close.iloc[-1] / close.iloc[-20] - 1) * 100) if len(close) >= 20 else 0.0

        results.append(SectorStrength(
            sector=sector,
            etf=etf,
            roc_5d=round(roc_5, 2),
            roc_20d=round(roc_20, 2),
            rank=0,  # assigned below
        ))

    # Sort by composite: 60% 5-day + 40% 20-day
    results.sort(key=lambda s: 0.6 * s.roc_5d + 0.4 * s.roc_20d, reverse=True)
    for i, s in enumerate(results):
        # frozen=False not needed since SectorStrength is not frozen
        results[i] = SectorStrength(
            sector=s.sector, etf=s.etf,
            roc_5d=s.roc_5d, roc_20d=s.roc_20d, rank=i + 1,
        )

    return results


def momentum_conviction_modifier(score: MomentumScore) -> float:
    """Convert a momentum score to a conviction modifier.

    Strong positive momentum: up to +10% boost.
    Strong negative momentum: up to -15% penalty.
    Neutral: no change.
    """
    c = score.composite
    if c > 0.15:
        return round(1.0 + min(c * 0.15, 0.10), 3)
    elif c < -0.15:
        return round(1.0 + max(c * 0.20, -0.15), 3)
    return 1.0
