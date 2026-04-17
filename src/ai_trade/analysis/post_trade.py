"""Per-trade post-mortem analysis.

For every closed trade we score the entry, the stop, and the exit —
then write a single row into ``trade_analysis`` plus a short
human-readable lesson.  The goal is not to second-guess individual
trades in isolation but to accumulate structured features that the
loss-pattern scanner and parameter optimizer can cluster on later.

Scoring philosophy:
    - Entry quality:  how much of the trade's favorable excursion
      happened AFTER entry vs how much adverse move came before the
      move we wanted.  0..1 scale; 0.5 = neutral, >0.7 = crisp entry.
    - Stop quality:   reuses ``score_stop_quality`` from exit_planner
      (too_tight / just_right / too_loose / not_hit).
    - Exit quality:   compares actual exit to the max favorable
      excursion.  If we captured <60% of the MFE, "left money on table";
      if >90%, "sold the top"; else "normal".
    - Lesson string:  one sentence the user can scan in a log.

All inputs come from the ``trades`` row, which already records
high_since_entry / low_since_entry from the trailing-stop loop.
No network calls, no candle re-fetch — this runs inside the hot path
at trade-close time.
"""

from __future__ import annotations

from typing import Any

from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.exit_planner import score_stop_quality

logger = get_logger(__name__)


EXIT_QUALITY_LEFT_MONEY = "left_money_on_table"
EXIT_QUALITY_SOLD_TOP = "sold_near_top"
EXIT_QUALITY_NORMAL = "normal"
EXIT_QUALITY_UNKNOWN = "unknown"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        f = float(x)
        if f != f:  # NaN
            return default
        return f
    except (TypeError, ValueError):
        return default


def classify_exit_reason(trade: dict) -> str:
    """Best-effort exit reason for a closed trade row."""
    exit_price = trade.get("exit_price")
    entry_price = trade.get("entry_price")
    stop = trade.get("stop_loss")
    target = trade.get("take_profit")
    pnl = trade.get("pnl")

    if exit_price is None or entry_price is None:
        return "unknown"
    exit_price = _safe_float(exit_price)
    entry_price = _safe_float(entry_price)
    if stop is not None:
        stop_f = _safe_float(stop)
        if stop_f > 0 and abs(exit_price - stop_f) / max(stop_f, 0.01) < 0.01:
            return "stop_loss"
    if target is not None:
        target_f = _safe_float(target)
        if target_f > 0 and abs(exit_price - target_f) / max(target_f, 0.01) < 0.01:
            return "take_profit"
    if pnl is not None:
        pnl_f = _safe_float(pnl)
        return "win" if pnl_f > 0 else "loss"
    return "unknown"


def score_entry_quality(trade: dict) -> float:
    """0..1 score on how favourable the post-entry tape was.

    Definition: share of the (MFE + |MAE|) envelope that was
    *favorable*.  A trade that immediately ran up to MFE with no
    drawdown scores 1.0; one that drew down to MAE before any move
    scores near 0.  Missing data → 0.5 (neutral).
    """
    entry = _safe_float(trade.get("entry_price"))
    hi = _safe_float(trade.get("high_since_entry"), default=entry)
    lo = _safe_float(trade.get("low_since_entry"), default=entry)
    if entry <= 0:
        return 0.5
    mfe = max(0.0, hi - entry)
    mae = max(0.0, entry - lo)
    envelope = mfe + mae
    if envelope <= 0:
        return 0.5
    return round(mfe / envelope, 3)


def score_exit_quality(trade: dict) -> str:
    """Compare actual exit to the max favourable excursion."""
    entry = _safe_float(trade.get("entry_price"))
    exit_price = _safe_float(trade.get("exit_price"))
    hi = _safe_float(trade.get("high_since_entry"))
    if entry <= 0 or exit_price <= 0 or hi <= 0 or hi <= entry:
        return EXIT_QUALITY_UNKNOWN
    mfe = hi - entry
    captured = max(0.0, exit_price - entry)
    if mfe <= 0:
        return EXIT_QUALITY_UNKNOWN
    ratio = captured / mfe
    if ratio >= 0.9:
        return EXIT_QUALITY_SOLD_TOP
    if ratio < 0.6:
        return EXIT_QUALITY_LEFT_MONEY
    return EXIT_QUALITY_NORMAL


def _make_lesson(
    trade: dict,
    exit_reason: str,
    entry_quality: float,
    exit_quality: str,
    stop_quality: str,
    regime_at_entry: str | None,
    regime_at_exit: str | None,
) -> str:
    symbol = trade.get("symbol", "?")
    strategy = trade.get("strategy", "?")
    pnl_pct = _safe_float(trade.get("pnl_pct"))
    sign = "+" if pnl_pct >= 0 else ""
    parts: list[str] = [f"{symbol}/{strategy} {sign}{pnl_pct:.2f}% via {exit_reason}"]

    if entry_quality >= 0.75:
        parts.append("entry was crisp")
    elif entry_quality <= 0.35:
        parts.append("entry chopped before working (consider later trigger)")

    if stop_quality == "too_tight":
        parts.append("stop too tight -- price reversed after stop fired")
    elif stop_quality == "too_loose":
        parts.append("stop too loose -- slippage past level")

    if exit_quality == EXIT_QUALITY_LEFT_MONEY:
        parts.append("exit left >40% of MFE on the table")
    elif exit_quality == EXIT_QUALITY_SOLD_TOP:
        parts.append("exit captured >90% of MFE — well-timed")

    if regime_at_entry and regime_at_exit and regime_at_entry != regime_at_exit:
        parts.append(f"regime shifted {regime_at_entry}->{regime_at_exit}")

    return "; ".join(parts)


def analyze_closed_trade(
    trade: dict,
    market_context: Any | None = None,
    regime_at_entry: str | None = None,
) -> dict:
    """Return a structured analysis dict for a single closed trade.

    Pure function — does not touch the database.  Caller is
    responsible for persistence via ``persist_analysis``.
    """
    exit_reason = classify_exit_reason(trade)
    entry_quality = score_entry_quality(trade)
    exit_quality = score_exit_quality(trade)

    entry_price = _safe_float(trade.get("entry_price"))
    stop_price = _safe_float(trade.get("stop_loss"))
    target_price = _safe_float(trade.get("take_profit"))
    hi = _safe_float(trade.get("high_since_entry"))
    lo = _safe_float(trade.get("low_since_entry"))

    try:
        stop_quality = score_stop_quality(
            exit_reason=exit_reason,
            entry_price=entry_price,
            stop_price=stop_price,
            max_favorable_price=hi if hi > 0 else None,
            max_adverse_price=lo if lo > 0 else None,
            direction=trade.get("side") or "long",
            stop_method=trade.get("stop_method"),
            target_price=target_price if target_price > 0 else None,
        )
    except Exception:
        stop_quality = "not_hit"

    regime_at_exit = None
    if market_context is not None:
        try:
            regime_at_exit = market_context.regime.value
        except Exception:
            regime_at_exit = None

    if regime_at_entry is None:
        regime_at_entry = regime_at_exit

    lesson = _make_lesson(
        trade=trade,
        exit_reason=exit_reason,
        entry_quality=entry_quality,
        exit_quality=exit_quality,
        stop_quality=stop_quality,
        regime_at_entry=regime_at_entry,
        regime_at_exit=regime_at_exit,
    )

    return {
        "exit_reason": exit_reason,
        "entry_quality": entry_quality,
        "stop_quality": stop_quality,
        "exit_quality": exit_quality,
        "market_regime": regime_at_entry,
        "regime_at_exit": regime_at_exit,
        "lesson": lesson,
    }


def persist_analysis(database, trade_id: int, analysis: dict) -> int | None:
    """Write one row to trade_analysis.  Best-effort, returns row id or None."""
    if trade_id is None or trade_id <= 0:
        return None
    try:
        row_id = database.insert_trade_analysis(
            trade_id=trade_id,
            entry_quality=analysis.get("entry_quality"),
            stop_quality=analysis.get("stop_quality"),
            exit_quality=analysis.get("exit_quality"),
            market_regime=analysis.get("market_regime"),
            regime_at_exit=analysis.get("regime_at_exit"),
            lessons=analysis.get("lesson"),
        )
        return row_id
    except Exception:
        logger.exception("trade_analysis_persist_failed", trade_id=trade_id)
        return None


def analyze_and_persist(
    database,
    trade: dict,
    market_context: Any | None = None,
    regime_at_entry: str | None = None,
) -> dict:
    """Convenience helper — analyze, persist, return the analysis dict."""
    analysis = analyze_closed_trade(
        trade=trade,
        market_context=market_context,
        regime_at_entry=regime_at_entry,
    )
    trade_id = trade.get("trade_id") or trade.get("id")
    persist_analysis(database, trade_id, analysis)
    return analysis
