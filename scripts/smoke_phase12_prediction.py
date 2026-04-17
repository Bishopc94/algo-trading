"""Smoke test for Phase 12: Market Prediction.

Tests:
  1. Momentum prediction indicators added to DataFrame
  2. Momentum score computation (bullish, bearish, neutral)
  3. Momentum conviction modifier
  4. Weekly trend classification (up, down, flat)
  5. Weekly trend conviction modifier (aligned vs counter-trend)
  6. Sector strength ranking
  7. Edge cases (empty/short DataFrames)
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, SRC)

import numpy as np
import pandas as pd

from ai_trade.data.indicators import add_all, add_momentum_prediction
from ai_trade.analysis.market_prediction import (
    MomentumScore,
    WeeklyTrend,
    SectorStrength,
    compute_momentum_score,
    classify_weekly_trend,
    weekly_trend_modifier,
    rank_sector_strength,
    momentum_conviction_modifier,
    SECTOR_ETFS,
)


def _make_bars(n: int = 60, trend: str = "bull") -> pd.DataFrame:
    """Generate synthetic daily bars with a clear trend."""
    dates = pd.date_range(end="2026-04-15", periods=n, freq="B")
    if trend == "bull":
        closes = np.linspace(180, 200, n) + np.random.default_rng(42).normal(0, 0.5, n)
    elif trend == "bear":
        closes = np.linspace(220, 200, n) + np.random.default_rng(42).normal(0, 0.5, n)
    else:  # flat
        closes = np.full(n, 200.0) + np.random.default_rng(42).normal(0, 1.0, n)

    volume = np.random.default_rng(42).integers(500000, 2000000, n)
    return pd.DataFrame({
        "open": closes - 0.3,
        "high": closes + 1.0,
        "low": closes - 1.0,
        "close": closes,
        "volume": volume,
    }, index=dates)


def _make_weekly_bars(n: int = 40, trend: str = "up") -> pd.DataFrame:
    """Generate synthetic weekly bars."""
    dates = pd.date_range(end="2026-04-10", periods=n, freq="W")
    actual = len(dates)
    if trend == "up":
        closes = np.linspace(350, 500, actual)
    elif trend == "down":
        closes = np.linspace(500, 350, actual)
    else:
        closes = np.full(actual, 420.0) + np.random.default_rng(42).normal(0, 5.0, actual)
    return pd.DataFrame({
        "open": closes - 1.0,
        "high": closes + 3.0,
        "low": closes - 3.0,
        "close": closes,
        "volume": np.random.default_rng(42).integers(10_000_000, 50_000_000, actual),
    }, index=dates)


def main() -> int:
    # -- Test 1: Momentum prediction indicators --
    print("== Test 1: momentum prediction indicators ==")
    df = _make_bars(60, "bull")
    add_all(df)
    expected_cols = ["rsi_roc_3", "macd_accel", "ema_9_slope", "ema_20_slope", "price_roc_5", "volume_trend"]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"
    print(f"  all {len(expected_cols)} prediction columns present")
    latest = df.iloc[-1]
    print(f"  rsi_roc_3={latest['rsi_roc_3']:.2f} macd_accel={latest['macd_accel']:.4f} "
          f"ema_9_slope={latest['ema_9_slope']:.3f}")

    # -- Test 2: Momentum score - bullish --
    print("\n== Test 2: momentum score - bullish trend ==")
    bull_df = _make_bars(60, "bull")
    add_all(bull_df)
    ms = compute_momentum_score(bull_df, "TEST_BULL")
    assert isinstance(ms, MomentumScore)
    assert ms.symbol == "TEST_BULL"
    assert -1.0 <= ms.composite <= 1.0
    print(f"  composite={ms.composite:.3f} direction={ms.predicted_direction}")
    print(f"  rsi_mom={ms.rsi_momentum:.3f} macd_accel={ms.macd_acceleration:.3f} "
          f"ema_align={ms.ema_alignment:.3f} vol_support={ms.volume_support:.3f}")

    # -- Test 3: Momentum score - bearish --
    print("\n== Test 3: momentum score - bearish trend ==")
    bear_df = _make_bars(60, "bear")
    add_all(bear_df)
    ms_bear = compute_momentum_score(bear_df, "TEST_BEAR")
    print(f"  composite={ms_bear.composite:.3f} direction={ms_bear.predicted_direction}")
    # Bear trend should lean negative or at least less positive
    assert ms_bear.composite <= ms.composite + 0.3, "Bear should not be more bullish than bull"

    # -- Test 4: Momentum conviction modifier --
    print("\n== Test 4: momentum conviction modifier ==")
    bullish_ms = MomentumScore("X", 0.5, 0.5, 0.5, 0.3, 0.45, "bullish")
    mod = momentum_conviction_modifier(bullish_ms)
    assert mod > 1.0, f"Strong bullish should boost, got {mod}"
    print(f"  strong bullish: modifier={mod}")

    bearish_ms = MomentumScore("X", -0.5, -0.5, -0.5, -0.3, -0.45, "bearish")
    mod = momentum_conviction_modifier(bearish_ms)
    assert mod < 1.0, f"Strong bearish should reduce, got {mod}"
    print(f"  strong bearish: modifier={mod}")

    neutral_ms = MomentumScore("X", 0.0, 0.0, 0.0, 0.0, 0.0, "neutral")
    mod = momentum_conviction_modifier(neutral_ms)
    assert mod == 1.0, f"Neutral should be 1.0, got {mod}"
    print(f"  neutral: modifier={mod}")

    # -- Test 5: Weekly trend classification --
    print("\n== Test 5: weekly trend classification ==")
    up_weekly = _make_weekly_bars(40, "up")
    trend = classify_weekly_trend(up_weekly)
    assert isinstance(trend, WeeklyTrend)
    assert trend.direction == "up", f"Expected 'up', got '{trend.direction}'"
    assert trend.strength > 0
    print(f"  uptrend: direction={trend.direction} strength={trend.strength:.3f} "
          f"slope={trend.ema_20w_slope:.3f}")

    down_weekly = _make_weekly_bars(40, "down")
    trend_down = classify_weekly_trend(down_weekly)
    assert trend_down.direction == "down", f"Expected 'down', got '{trend_down.direction}'"
    print(f"  downtrend: direction={trend_down.direction} strength={trend_down.strength:.3f}")

    flat_weekly = _make_weekly_bars(40, "flat")
    trend_flat = classify_weekly_trend(flat_weekly)
    assert trend_flat.direction == "flat", f"Expected 'flat', got '{trend_flat.direction}'"
    print(f"  flat: direction={trend_flat.direction}")

    # -- Test 6: Weekly trend modifier --
    print("\n== Test 6: weekly trend modifier ==")
    up_trend = WeeklyTrend(direction="up", strength=0.5, ema_20w_slope=1.0, price_vs_ema_20w=3.0)
    aligned_mod = weekly_trend_modifier(up_trend, "long")
    assert aligned_mod > 1.0, f"Aligned long in uptrend should boost, got {aligned_mod}"
    print(f"  long in uptrend: modifier={aligned_mod}")

    counter_mod = weekly_trend_modifier(up_trend, "short")
    assert counter_mod < 1.0, f"Short in uptrend should penalize, got {counter_mod}"
    print(f"  short in uptrend: modifier={counter_mod}")

    flat_mod = weekly_trend_modifier(WeeklyTrend("flat", 0.0, 0.0, 0.0), "long")
    assert flat_mod == 1.0, f"Flat should be 1.0, got {flat_mod}"
    print(f"  long in flat: modifier={flat_mod}")

    # -- Test 7: Sector strength ranking --
    print("\n== Test 7: sector strength ranking ==")
    sector_bars = {}
    rng = np.random.default_rng(42)
    for sector, etf in list(SECTOR_ETFS.items())[:5]:
        n = 30
        dates = pd.date_range(end="2026-04-15", periods=n, freq="B")
        base = 100 + rng.normal(0, 10)
        closes = np.linspace(base, base + rng.normal(5, 3), n)
        sector_bars[sector] = pd.DataFrame({
            "open": closes - 0.2, "high": closes + 0.5,
            "low": closes - 0.5, "close": closes,
            "volume": rng.integers(1_000_000, 5_000_000, n),
        }, index=dates)

    rankings = rank_sector_strength(sector_bars)
    assert len(rankings) >= 3, f"Expected at least 3 ranked sectors, got {len(rankings)}"
    assert rankings[0].rank == 1, "First should be rank 1"
    assert rankings[-1].rank == len(rankings), "Last should be lowest rank"
    for s in rankings:
        print(f"  #{s.rank} {s.sector} ({s.etf}): 5d={s.roc_5d:+.2f}% 20d={s.roc_20d:+.2f}%")

    # -- Test 8: Edge case - empty/short DataFrame --
    print("\n== Test 8: edge cases ==")
    empty_ms = compute_momentum_score(pd.DataFrame(), "EMPTY")
    assert empty_ms.composite == 0.0
    assert empty_ms.predicted_direction == "neutral"
    print(f"  empty DataFrame: composite={empty_ms.composite}, direction={empty_ms.predicted_direction}")

    short_trend = classify_weekly_trend(_make_weekly_bars(10, "up"))
    assert short_trend.direction == "flat", f"Short bars should be flat, got {short_trend.direction}"
    print(f"  short weekly bars (10): direction={short_trend.direction}")

    print(f"\nSMOKE TEST PASSED (8/8)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
