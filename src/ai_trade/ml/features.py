"""Canonical feature extraction for the V2 signal-quality model.

The feature vector is intentionally small and fixed-schema.  A fixed
schema means (a) training and inference always agree on column
ordering, (b) new trades logged today can be joined with models
trained last week without breaking, and (c) the dataset stays
interpretable when we go back and inspect what the model learned.

What we extract:

    Numeric (from Signal + metadata)
        conviction             — strategy-level conviction before ML
        entry_price            — raw level (log-scaled in pipeline)
        atr                    — ATR at entry
        atr_pct                — atr / entry_price
        stop_distance_pct      — |entry - stop| / entry
        target_distance_pct    — |target - entry| / entry
        rr_ratio               — target_distance_pct / stop_distance_pct
        rsi                    — default 50.0 if missing
        relative_volume        — default 1.0 if missing

    Market context
        vix_level              — default 18.0 if missing
        regime_code            — ordinal: strong_bear=-2 .. strong_bull=+2

    Time
        hour_of_day            — 0-23
        day_of_week            — 0=Mon .. 4=Fri

    Strategy / hold type (label-encoded; unknown → -1)
        strategy_code
        hold_type_code

The label-encoded strategy and hold-type fields work with
gradient-boosted trees without requiring one-hot explosion.  If we
later swap to a linear model we can revisit.

Why not one-hot everything:
    Gradient boosting handles ordinal codes natively and it keeps
    the feature vector small (14 columns) so training stays fast
    on a $500 account's modest trade history.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

# The canonical feature order.  NEVER reorder — models are trained
# against this exact sequence.  Add new features at the END only.
FEATURE_ORDER: list[str] = [
    "conviction",
    "entry_price",
    "atr",
    "atr_pct",
    "stop_distance_pct",
    "target_distance_pct",
    "rr_ratio",
    "rsi",
    "relative_volume",
    "vix_level",
    "regime_code",
    "hour_of_day",
    "day_of_week",
    "strategy_code",
    "hold_type_code",
]

# Strategy name → integer code.  Matches the strategies enabled in
# config/settings.yaml.  Unknown strategies get -1 so the model can
# still see them during cold start without crashing.
STRATEGY_CODES: dict[str, int] = {
    "momentum": 0,
    "mean_reversion": 1,
    "vwap": 2,
    "ema_crossover": 3,
    "macd_divergence": 4,
    "bb_squeeze": 5,
    "orb": 6,
    "pullback": 7,
    # Options
    "credit_put_spread": 10,
    "debit_call_spread": 11,
    "long_call": 12,
    "long_put": 13,
    "cash_secured_put": 14,
    "covered_call": 15,
    "covered_straddle": 16,
    "momentum_options": 17,
}

HOLD_TYPE_CODES: dict[str, int] = {
    "swing": 0,
    "day": 1,
    "adaptive": 2,
}

REGIME_CODES: dict[str, int] = {
    "strong_bear": -2,
    "bear": -1,
    "neutral": 0,
    "bull": 1,
    "strong_bull": 2,
}


def _safe_float(x: Any, default: float = 0.0) -> float:
    """Coerce to float; return `default` on None / NaN / error."""
    if x is None:
        return default
    try:
        f = float(x)
    except (TypeError, ValueError):
        return default
    # Reject NaN — propagates into model and corrupts training
    if f != f:  # noqa: PLR0124 — NaN check
        return default
    return f


def extract_features(
    signal,
    market_context=None,
    now: datetime | None = None,
) -> dict[str, float]:
    """Build a feature dict from a Signal + MarketContext.

    This function is pure — given the same inputs it always produces
    the same output.  Missing fields use sensible defaults (RSI=50,
    VIX=18, regime=neutral) so the model never sees NaNs.

    Args:
        signal: Signal dataclass instance (from strategy/base.py)
        market_context: MarketContext or None.  None falls back to
                        neutral regime / VIX 18.
        now: Clock override for testing.  Defaults to datetime.now().

    Returns:
        Dict keyed by FEATURE_ORDER names, all float values.
    """
    meta: dict = getattr(signal, "metadata", {}) or {}
    clock = now or datetime.now()

    entry_price = _safe_float(getattr(signal, "entry_price", 0.0))
    stop_price = _safe_float(getattr(signal, "stop_loss_price", entry_price))
    target_price = _safe_float(
        getattr(signal, "take_profit_price", entry_price), default=entry_price
    )

    stop_dist_pct = (
        abs(entry_price - stop_price) / entry_price if entry_price > 0 else 0.0
    )
    target_dist_pct = (
        abs(target_price - entry_price) / entry_price if entry_price > 0 else 0.0
    )
    rr_ratio = target_dist_pct / stop_dist_pct if stop_dist_pct > 0 else 0.0

    atr = _safe_float(meta.get("atr"))
    atr_pct = atr / entry_price if entry_price > 0 else 0.0

    # Market context fallbacks for cold-start paths where the bot may
    # evaluate signals before the regime analyser has run.
    vix_level = 18.0
    regime_code = 0
    if market_context is not None:
        vix_level = _safe_float(getattr(market_context, "vix_level", 18.0), 18.0)
        regime_val = getattr(market_context, "regime", None)
        regime_str = getattr(regime_val, "value", regime_val)
        if isinstance(regime_str, str):
            regime_code = REGIME_CODES.get(regime_str.lower(), 0)

    strategy_name = getattr(signal, "strategy_name", "") or ""
    strategy_code = STRATEGY_CODES.get(strategy_name, -1)

    hold_type_val = getattr(signal, "hold_type", None)
    hold_type_str = getattr(hold_type_val, "value", hold_type_val)
    hold_type_code = HOLD_TYPE_CODES.get(
        hold_type_str if isinstance(hold_type_str, str) else "", -1
    )

    return {
        "conviction": _safe_float(getattr(signal, "conviction", 0.0)),
        "entry_price": entry_price,
        "atr": atr,
        "atr_pct": atr_pct,
        "stop_distance_pct": stop_dist_pct,
        "target_distance_pct": target_dist_pct,
        "rr_ratio": rr_ratio,
        "rsi": _safe_float(meta.get("rsi"), default=50.0),
        "relative_volume": _safe_float(meta.get("relative_volume"), default=1.0),
        "vix_level": vix_level,
        "regime_code": float(regime_code),
        "hour_of_day": float(clock.hour),
        "day_of_week": float(clock.weekday()),
        "strategy_code": float(strategy_code),
        "hold_type_code": float(hold_type_code),
    }


def features_to_vector(features: dict[str, float]) -> list[float]:
    """Deterministically order a feature dict into a numeric vector.

    Missing keys default to 0.0 so partial feature dicts (e.g. from
    older training records) don't crash the model.
    """
    return [float(features.get(name, 0.0)) for name in FEATURE_ORDER]
