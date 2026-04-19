"""Smart stop-loss and take-profit calculation for V2.

Replaces the naive `stop = entry - N*ATR` pattern with support/resistance-
aware stops, VIX-adjusted multipliers, and regime-based width tuning. One
shared module so strategies stay thin and Phase 4 ML can swap the whole
planner without touching strategy code.

Usage:
    from ai_trade.strategy.exit_planner import plan_long_exit

    levels = plan_long_exit(
        bars=df, entry_price=close, atr=atr_14,
        base_stop_mult=1.5, base_tp_mult=3.5,
        vix=ctx.vix_level if ctx else None,
        regime=ctx.regime.value if ctx else None,
    )
    stop_loss = levels.stop_loss
    take_profit = levels.take_profit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


@dataclass
class ExitLevels:
    stop_loss: float
    take_profit: float
    stop_method: str
    target_method: str
    effective_stop_mult: float
    effective_tp_mult: float
    details: dict = field(default_factory=dict)


# ── Volatility & regime adjustments ────────────────────────────────────────

def _vix_width_factor(vix: float | None) -> float:
    """Wider stops when VIX is high (market is swingier)."""
    if vix is None or vix <= 0:
        return 1.0
    if vix >= 30:
        return 1.25
    if vix >= 25:
        return 1.15
    if vix >= 20:
        return 1.05
    if vix >= 15:
        return 1.0
    return 0.95  # calm tape — tighter is fine


def _regime_width_factor(regime: str | None) -> float:
    """Trending markets deserve room; choppy/bearish cut fast."""
    if not regime:
        return 1.0
    r = regime.lower()
    if "strong_bull" in r or r == "strong_bull":
        return 1.10
    if "bull" in r:
        return 1.05
    if "strong_bear" in r:
        return 0.85
    if "bear" in r:
        return 0.90
    return 1.0  # neutral


# ── Swing point detection ─────────────────────────────────────────────────

def _swing_lows(bars: pd.DataFrame, lookback: int = 20, window: int = 2) -> list[float]:
    """Local minima in the last `lookback` bars.

    A bar is a swing low if its low is <= the lows of `window` bars on each
    side. Returns prices oldest→newest (caller picks most recent relevant one).
    """
    if len(bars) < 2 * window + 1:
        return []
    lows = bars["low"].values
    out: list[float] = []
    start = max(window, len(lows) - lookback - window)
    end = len(lows) - window
    for i in range(start, end):
        center = lows[i]
        left = lows[i - window:i]
        right = lows[i + 1:i + 1 + window]
        if center <= left.min() and center <= right.min():
            out.append(float(center))
    return out


def _swing_highs(bars: pd.DataFrame, lookback: int = 20, window: int = 2) -> list[float]:
    if len(bars) < 2 * window + 1:
        return []
    highs = bars["high"].values
    out: list[float] = []
    start = max(window, len(highs) - lookback - window)
    end = len(highs) - window
    for i in range(start, end):
        center = highs[i]
        left = highs[i - window:i]
        right = highs[i + 1:i + 1 + window]
        if center >= left.max() and center >= right.max():
            out.append(float(center))
    return out


# ── Long-side planner ─────────────────────────────────────────────────────

def plan_long_exit(
    bars: pd.DataFrame,
    entry_price: float,
    atr: float,
    base_stop_mult: float = 1.5,
    base_tp_mult: float = 3.5,
    vix: float | None = None,
    regime: str | None = None,
    lookback: int = 20,
    sr_tolerance_atr: float = 2.0,
    sr_buffer_pct: float = 0.0015,
) -> ExitLevels:
    """Compute stop and target for a long trade.

    Stop: nearest swing low below entry, buffered below by `sr_buffer_pct`.
    Falls back to `entry - mult * ATR` if no swing low within `sr_tolerance_atr`
    multiples of ATR, or if the swing-low stop would actually be wider than
    the ATR fallback (we never *loosen* a stop — tighter is always OK).

    Target: nearest swing high above entry. Falls back to ATR multiple if
    none within `sr_tolerance_atr * 2` of ATR.

    Multipliers are scaled by VIX and regime width factors.
    """
    vix_f = _vix_width_factor(vix)
    regime_f = _regime_width_factor(regime)
    eff_stop_mult = base_stop_mult * vix_f * regime_f
    eff_tp_mult = base_tp_mult * vix_f * regime_f

    atr_stop = entry_price - eff_stop_mult * atr
    atr_target = entry_price + eff_tp_mult * atr

    # --- Stop: swing-low support ---
    stop_loss = atr_stop
    stop_method = "atr_fallback"
    sr_range = sr_tolerance_atr * atr

    lows_below = [l for l in _swing_lows(bars, lookback=lookback) if l < entry_price]
    if lows_below:
        # Most recent (last in list) if within tolerance, else nearest-by-price.
        nearest_low = max(lows_below)  # highest swing low below entry
        if entry_price - nearest_low <= sr_range:
            candidate = nearest_low * (1 - sr_buffer_pct)
            # Only use if it's tighter than ATR fallback (never loosen a stop).
            if candidate >= atr_stop and candidate < entry_price:
                stop_loss = candidate
                stop_method = "swing_low_support"

    # --- Target: swing-high resistance ---
    take_profit = atr_target
    target_method = "atr_fallback"
    target_range = sr_tolerance_atr * 2.0 * atr

    highs_above = [h for h in _swing_highs(bars, lookback=lookback) if h > entry_price]
    if highs_above:
        nearest_high = min(highs_above)  # lowest swing high above entry
        if nearest_high - entry_price <= target_range:
            # Accept target only if it still clears a reasonable minimum (0.5*ATR)
            if nearest_high - entry_price >= 0.5 * atr:
                take_profit = nearest_high * (1 - sr_buffer_pct)
                target_method = "swing_high_resistance"

    return ExitLevels(
        stop_loss=stop_loss,
        take_profit=take_profit,
        stop_method=stop_method,
        target_method=target_method,
        effective_stop_mult=eff_stop_mult,
        effective_tp_mult=eff_tp_mult,
        details={
            "vix_factor": vix_f,
            "regime_factor": regime_f,
            "atr": atr,
            "atr_stop": atr_stop,
            "atr_target": atr_target,
        },
    )


def plan_short_exit(
    bars: pd.DataFrame,
    entry_price: float,
    atr: float,
    base_stop_mult: float = 1.5,
    base_tp_mult: float = 3.5,
    vix: float | None = None,
    regime: str | None = None,
    lookback: int = 20,
    sr_tolerance_atr: float = 2.0,
    sr_buffer_pct: float = 0.0015,
) -> ExitLevels:
    """Mirror of plan_long_exit for bearish setups."""
    vix_f = _vix_width_factor(vix)
    regime_f = _regime_width_factor(regime)
    eff_stop_mult = base_stop_mult * vix_f * regime_f
    eff_tp_mult = base_tp_mult * vix_f * regime_f

    atr_stop = entry_price + eff_stop_mult * atr
    atr_target = entry_price - eff_tp_mult * atr

    stop_loss = atr_stop
    stop_method = "atr_fallback"
    sr_range = sr_tolerance_atr * atr

    highs_above = [h for h in _swing_highs(bars, lookback=lookback) if h > entry_price]
    if highs_above:
        nearest_high = min(highs_above)
        if nearest_high - entry_price <= sr_range:
            candidate = nearest_high * (1 + sr_buffer_pct)
            if candidate <= atr_stop and candidate > entry_price:
                stop_loss = candidate
                stop_method = "swing_high_resistance"

    take_profit = atr_target
    target_method = "atr_fallback"
    target_range = sr_tolerance_atr * 2.0 * atr

    lows_below = [l for l in _swing_lows(bars, lookback=lookback) if l < entry_price]
    if lows_below:
        nearest_low = max(lows_below)
        if entry_price - nearest_low <= target_range:
            if entry_price - nearest_low >= 0.5 * atr:
                take_profit = nearest_low * (1 + sr_buffer_pct)
                target_method = "swing_low_support"

    return ExitLevels(
        stop_loss=stop_loss,
        take_profit=take_profit,
        stop_method=stop_method,
        target_method=target_method,
        effective_stop_mult=eff_stop_mult,
        effective_tp_mult=eff_tp_mult,
        details={
            "vix_factor": vix_f,
            "regime_factor": regime_f,
            "atr": atr,
            "atr_stop": atr_stop,
            "atr_target": atr_target,
        },
    )


# ── Trailing stop computation ─────────────────────────────────────────────

def _trail_params_for_conviction(conviction: float) -> tuple[float | None, float]:
    """Return (breakeven_trigger_atr, chandelier_atr_mult) for a conviction.

    Higher conviction signals earn more breathing room: a later breakeven
    trigger and a wider chandelier trail. At >=0.95 the breakeven ratchet
    is disabled entirely -- we only trail on new highs via chandelier.

    The rationale is the IONZ case: a 97% conviction trade ran +3%,
    triggered a hair-trigger breakeven at entry+0.1%, then got stopped
    out by a normal pullback before reaching its target. Scaling the
    trigger distance by conviction gives winners room to breathe.
    """
    if conviction >= 0.95:
        return None, 4.0
    if conviction >= 0.85:
        return 2.5, 3.5
    if conviction >= 0.75:
        return 1.5, 2.5
    return 1.0, 2.0


def compute_trailing_stop_long(
    entry_price: float,
    current_price: float,
    current_stop: float,
    atr: float,
    high_since_entry: float | None = None,
    conviction: float = 0.70,
    breakeven_buffer_pct: float = 0.001,
    breakeven_trigger_atr: float | None = None,
    chandelier_atr_mult: float | None = None,
) -> tuple[float | None, str]:
    """Propose a new stop for an open long trade.

    Conviction-aware: the breakeven trigger distance and chandelier
    multiplier scale with ``conviction``. A 0.97 conviction trade gets
    a chandelier-only trail at 4x ATR; a 0.70 conviction trade gets
    the classic 1x breakeven / 2x chandelier behavior.

    Returns (new_stop, mode) or (None, "") if no update is warranted.
    The proposal is only returned if it strictly *tightens* the stop
    (never loosens). Two modes:

    - **breakeven**: once price has moved ``breakeven_trigger_atr * ATR``
      above entry, move stop to ``entry * (1 + breakeven_buffer_pct)``.
      Disabled entirely for conviction >= 0.95.
    - **chandelier**: if ``high_since_entry`` is known, trail stop at
      ``high_since_entry - chandelier_atr_mult * ATR``. Only tightens
      when price has printed a new high recently; a pullback leaves
      the prior stop in place.

    Chandelier wins if it produces a tighter stop than breakeven.
    """
    if atr <= 0 or entry_price <= 0:
        return None, ""

    default_be, default_chand = _trail_params_for_conviction(conviction)
    if breakeven_trigger_atr is None:
        breakeven_trigger_atr = default_be
    if chandelier_atr_mult is None:
        chandelier_atr_mult = default_chand

    proposals: list[tuple[float, str]] = []

    # Breakeven trigger (skipped for very high conviction)
    if breakeven_trigger_atr is not None:
        if current_price >= entry_price + breakeven_trigger_atr * atr:
            be = entry_price * (1 + breakeven_buffer_pct)
            proposals.append((be, "breakeven"))

    # Chandelier trail. With a wide chandelier_atr_mult for high
    # conviction, the computed stop naturally stays below entry until
    # price runs well past the 1R mark -- that's exactly the breathing
    # room we want on pullbacks.
    if high_since_entry is not None and high_since_entry > entry_price:
        chand = high_since_entry - chandelier_atr_mult * atr
        if chand > entry_price:
            proposals.append((chand, "chandelier"))

    if not proposals:
        return None, ""

    proposals.sort(key=lambda p: p[0], reverse=True)
    new_stop, mode = proposals[0]

    if new_stop <= current_stop:
        return None, ""
    if new_stop >= current_price - 0.01:
        return None, ""
    return round(new_stop, 2), mode


# ── Stop-quality scoring (post-trade) ─────────────────────────────────────

StopQuality = Literal[
    "too_tight", "too_loose", "just_right", "trail_too_tight", "not_hit",
]

_TRAIL_MODES = {"breakeven", "chandelier", "time_breakeven"}


def score_stop_quality(
    exit_reason: str,
    entry_price: float,
    stop_price: float,
    max_favorable_price: float | None,
    max_adverse_price: float | None,
    direction: str = "long",
    stop_method: str | None = None,
    target_price: float | None = None,
) -> StopQuality:
    """Classify stop quality after a trade closes.

    - too_tight: stop hit, but price later reversed past 2x the initial risk.
    - too_loose: stop hit with heavy slippage past stop before fill.
    - trail_too_tight: stop was a *trailed* one (breakeven/chandelier) and
      the unrealized gain at peak was >=1/3 of the distance to target. We
      killed a winner that had earned meaningful breathing room.
    - just_right: stop hit near the actual reversal point.
    - not_hit: trade exited via take-profit or other reason.
    """
    if exit_reason != "stop_loss":
        return "not_hit"
    if max_favorable_price is None or max_adverse_price is None:
        return "just_right"

    if direction == "long":
        # Trail-too-tight check: ratcheted stop killed a profitable trade.
        if stop_method in _TRAIL_MODES and target_price is not None:
            target_gain = target_price - entry_price
            peak_gain = max_favorable_price - entry_price
            if target_gain > 0 and peak_gain >= target_gain / 3:
                return "trail_too_tight"

        initial_risk = entry_price - stop_price
        if initial_risk <= 0:
            return "just_right"
        reversal_gain = max_favorable_price - stop_price
        slippage = max(0.0, stop_price - max_adverse_price)
        if reversal_gain > 2 * initial_risk:
            return "too_tight"
        if slippage > 0.5 * initial_risk:
            return "too_loose"
        return "just_right"
    else:
        if stop_method in _TRAIL_MODES and target_price is not None:
            target_gain = entry_price - target_price
            peak_gain = entry_price - max_adverse_price
            if target_gain > 0 and peak_gain >= target_gain / 3:
                return "trail_too_tight"

        initial_risk = stop_price - entry_price
        if initial_risk <= 0:
            return "just_right"
        reversal_gain = stop_price - max_adverse_price
        slippage = max(0.0, max_favorable_price - stop_price)
        if reversal_gain > 2 * initial_risk:
            return "too_tight"
        if slippage > 0.5 * initial_risk:
            return "too_loose"
        return "just_right"
