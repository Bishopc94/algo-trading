"""Technical indicators built on the ``ta`` library.

All functions take a pandas DataFrame with OHLCV columns and add new
indicator columns in-place.  Functions are **idempotent** — if the target
column already exists, computation is skipped.  This means you can safely
call ``add_rsi()`` multiple times without wasted work.

Indicators provided:
  - **RSI** (Relative Strength Index): Momentum oscillator (0-100).
    Below 30 = oversold, above 70 = overbought.
  - **EMA** (Exponential Moving Average): Trend-following average that
    weights recent prices more heavily.
  - **VWAP** (Volume-Weighted Average Price): The "fair price" for the
    day.  Institutional traders use it as a benchmark.
  - **ATR** (Average True Range): Measures volatility.  Used for
    adaptive stop-loss and take-profit distances.
  - **Bollinger Bands**: Upper/lower bands at 2 standard deviations from
    a 20-period moving average.  Price near the lower band = oversold.
  - **MACD** (Moving Average Convergence Divergence): Trend and momentum
    indicator.  Histogram > 0 = bullish momentum.
  - **Relative Volume**: Today's volume divided by the 20-day average.
    A value of 2.0 means "2x normal volume" — a key breakout signal.
  - **ADR%** (Average Daily Range): Percentage range of daily price
    movement.  Higher ADR = more volatile (and more profitable) stocks.

Python-specific notes:
  - ``pd.DataFrame`` is modified **in-place** (new columns are added
    directly to the input DataFrame).  The return value is the same
    object, allowing chained calls: ``add_rsi(add_ema(df))``.
  - The ``ta`` library provides pre-built indicator classes (RSIIndicator,
    BollingerBands, etc.) so we don't need to implement the math ourselves.
  - ``ewm(span=N)`` computes an exponentially weighted moving average
    with a span of N periods — the pandas equivalent of an EMA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD as MACDIndicator
from ta.volatility import AverageTrueRange, BollingerBands


# ---------------------------------------------------------------------------
# Individual indicators
# ---------------------------------------------------------------------------

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Relative Strength Index column ``rsi_{period}``.

    RSI measures the speed and magnitude of recent price changes on a
    0-100 scale.  In this system:
      - RSI < 40 → triggers mean reversion (oversold)
      - RSI > 60 → triggers mean reversion exit (recovered)
      - RSI 55-75 → qualifies for long call options
    """
    col = f"rsi_{period}"
    if col not in df.columns:
        df[col] = RSIIndicator(close=df["close"], window=period).rsi()
    return df


def add_ema(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
    """Add Exponential Moving Average columns ``ema_{period}`` for each period.

    EMAs give more weight to recent prices than a simple moving average.
    Common uses in this system:
      - EMA-9:  Very short-term trend (used in market regime SPY analysis)
      - EMA-20: Short-term trend (strategies check price > EMA-20 for uptrend)
      - EMA-50: Medium-term trend (used by cash-secured puts, market regime)
      - EMA-200: Long-term trend (market regime SPY analysis only)
    """
    if periods is None:
        periods = [9, 20, 50]
    for period in periods:
        col = f"ema_{period}"
        if col not in df.columns:
            # ewm = Exponentially Weighted Moving average in pandas
            # span=N means the decay factor gives half-weight to data N periods old
            df[col] = df["close"].ewm(span=period, adjust=False).mean()
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Add Volume-Weighted Average Price column ``vwap_calc``.

    VWAP = cumulative(typical_price * volume) / cumulative(volume)
    where typical_price = (high + low + close) / 3.

    VWAP resets daily for intraday data (each new trading day starts
    fresh).  This is the standard behavior used by institutional traders.

    The VWAP strategy uses this to detect when price "reclaims" VWAP
    from below — a bullish signal that institutional buyers are stepping in.
    """
    col = "vwap_calc"
    if col in df.columns:
        return df

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical_price * df["volume"]

    # Reset VWAP calculation at the start of each trading day.
    # For intraday data with a DatetimeIndex, group by calendar date.
    if isinstance(df.index, pd.DatetimeIndex):
        date_groups = df.index.date
        cum_tp_vol = tp_vol.groupby(date_groups).cumsum()
        cum_vol = df["volume"].groupby(date_groups).cumsum()
    else:
        # Daily data: cumulative across the entire period
        cum_tp_vol = tp_vol.cumsum()
        cum_vol = df["volume"].cumsum()

    # Replace 0 volume with NaN to avoid division-by-zero
    df[col] = cum_tp_vol / cum_vol.replace(0, np.nan)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average True Range column ``atr_{period}``.

    ATR measures volatility by looking at the average of "true ranges"
    (the largest of: high-low, |high-prev_close|, |low-prev_close|).

    This system uses ATR for **adaptive stop-loss and take-profit**:
      - Stop loss = entry - 1.5 * ATR  (closer for less volatile stocks)
      - Take profit = entry + 3.0 * ATR (2:1 reward-to-risk ratio)

    A volatile stock with ATR=$2 gets wider stops than a stable stock
    with ATR=$0.50.  This prevents getting stopped out by normal noise.
    """
    col = f"atr_{period}"
    if col not in df.columns:
        df[col] = AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=period,
        ).average_true_range()
    return df


def add_bollinger(df: pd.DataFrame, period: int = 20, std: int = 2) -> pd.DataFrame:
    """Add Bollinger Band columns: bb_upper, bb_middle, bb_lower, bb_width.

    Bollinger Bands are a volatility envelope around a moving average:
      - bb_middle = 20-period simple moving average
      - bb_upper  = bb_middle + 2 * standard_deviation
      - bb_lower  = bb_middle - 2 * standard_deviation
      - bb_width  = (upper - lower) / middle  (normalized width)

    Approximately 95% of price action stays within the bands.  When price
    touches bb_lower, it's statistically oversold — used by mean reversion.
    When bb_width is small (< 0.10), volatility is compressed — used by
    covered straddle to find low-vol environments.
    """
    if "bb_upper" in df.columns:
        return df

    bb = BollingerBands(close=df["close"], window=period, window_dev=std)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Add MACD columns: macd, macd_signal, macd_hist.

    MACD (Moving Average Convergence Divergence) tracks the relationship
    between two EMAs (default: 12-period and 26-period):
      - macd = EMA(12) - EMA(26)
      - macd_signal = EMA(9) of the MACD line
      - macd_hist = macd - macd_signal

    When macd_hist > 0, short-term momentum is bullish.  This is used in
    the market regime analyzer as one of 7 breadth components.
    """
    if "macd" in df.columns:
        return df

    macd = MACDIndicator(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    return df


def add_volume_profile(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add ``relative_volume`` = today's volume / 20-day average volume.

    Relative volume (RVOL) tells you how active a stock is compared to
    normal.  Key thresholds in this system:
      - 1.5x: Minimum for momentum breakout entries
      - 2.0x: Minimum for long call options
      - 3.0x: Boosts momentum conviction to 0.75
      - 5.0x: Maximum momentum conviction (1.0)

    High relative volume confirms that a price move is "real" — driven by
    institutional participation, not just noise.
    """
    col = "relative_volume"
    if col not in df.columns:
        rolling_mean = df["volume"].rolling(window=period).mean()
        # Replace 0 with NaN to avoid division-by-zero
        df[col] = df["volume"] / rolling_mean.replace(0, np.nan)
    return df


# ---------------------------------------------------------------------------
# Standalone computations (return a value, don't modify the DataFrame)
# ---------------------------------------------------------------------------

def compute_adr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute Average Daily Range % over the last *period* days.

    ADR% = mean((high - low) / close) * 100

    A stock with ADR% of 5.0 moves an average of 5% per day — high
    volatility, good for momentum trading.  The momentum strategy
    requires ADR > 2.0% to filter out stocks that don't move enough.

    Returns:
        A single float (percentage).  Does NOT modify the DataFrame.
    """
    tail = df.tail(period)
    if tail.empty:
        return 0.0
    adr = ((tail["high"] - tail["low"]) / tail["close"]).mean() * 100
    return float(adr)


# ---------------------------------------------------------------------------
# Convenience — add all indicators at once
# ---------------------------------------------------------------------------

def add_all(df: pd.DataFrame, intraday: bool = False) -> pd.DataFrame:
    """Add all standard indicators to the DataFrame.

    This is called by the main orchestrator after fetching daily bars for
    all candidates.  Each strategy then reads the indicator columns it
    needs (RSI, EMA, etc.) from the already-enriched DataFrame.

    Args:
        df:       DataFrame with OHLCV columns.
        intraday: If True, also add VWAP (only meaningful for minute bars
                  since VWAP resets daily).
    """
    add_rsi(df)
    add_ema(df)
    add_atr(df)
    add_bollinger(df)
    add_macd(df)
    add_volume_profile(df)
    if intraday:
        add_vwap(df)
    return df
