"""Loss-pattern scanner.

Clusters losing trades across axes that the brief calls out —
hour-of-day, strategy, market regime, exit reason — and reports any
bucket that accounts for a disproportionate share of recent losses.

This is intentionally simple: no sklearn, no k-means.  We bin trades,
count losses per bin, and flag buckets whose loss rate is materially
worse than the population's baseline loss rate.  That's enough signal
to tell the parameter optimizer "something is off with strategy X in
BEAR regime" without overfitting to tiny samples.

The output is a list of ``LossCluster`` dicts which the EOD job logs
and the parameter optimizer consumes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from ai_trade.monitoring.logger import get_logger

logger = get_logger(__name__)


MIN_TRADES_FOR_CLUSTER = 3
MIN_LOSS_RATE_DELTA = 0.20  # cluster must be 20pp worse than baseline
MIN_BASELINE_TRADES = 10


def _safe_float(x, default: float = 0.0) -> float:
    try:
        f = float(x)
        if f != f:
            return default
        return f
    except (TypeError, ValueError):
        return default


def _hour_bucket(entry_time: str | None) -> str:
    """Map an ISO entry_time to an ET-hour bucket label.

    Stored entry_time is UTC ISO; we don't convert here — hour-of-day
    patterns are still visible in UTC and the bucket labels stay
    meaningful even if a reader expects ET.  The goal is bucketing,
    not wall-clock precision.
    """
    if not entry_time:
        return "unknown"
    try:
        hh = entry_time[11:13]
        if not hh.isdigit():
            return "unknown"
        return f"h{hh}"
    except Exception:
        return "unknown"


def _load_closed_trades(database, lookback: int) -> list[dict]:
    all_trades = database.get_all_trades()
    closed = [
        t for t in all_trades
        if t.get("status") == "closed" and t.get("pnl") is not None
    ]
    return closed[:lookback]


def _load_analysis_map(database) -> dict[int, dict]:
    """Map trade_id → trade_analysis row (most recent wins on dupes)."""
    try:
        with database._conn() as conn:  # noqa: SLF001
            rows = conn.execute(
                "SELECT trade_id, entry_quality, stop_quality, exit_quality, "
                "market_regime, regime_at_exit, lessons FROM trade_analysis "
                "ORDER BY id ASC"
            ).fetchall()
    except Exception:
        logger.exception("loss_pattern_analysis_load_failed")
        return {}
    return {int(r["trade_id"]): dict(r) for r in rows}


def _bucket_key(axis: str, trade: dict, analysis: dict | None) -> str:
    if axis == "strategy":
        return trade.get("strategy") or "unknown"
    if axis == "hold_type":
        return trade.get("hold_type") or "unknown"
    if axis == "hour":
        return _hour_bucket(trade.get("entry_time"))
    if axis == "regime":
        if analysis is None:
            return "unknown"
        return analysis.get("market_regime") or "unknown"
    if axis == "stop_quality":
        if analysis is None:
            return "unknown"
        return analysis.get("stop_quality") or "unknown"
    return "unknown"


AXES = ("strategy", "hour", "regime", "stop_quality", "hold_type")


def scan_loss_patterns(
    database,
    lookback: int = 50,
) -> list[dict]:
    """Return a list of significant loss clusters across multiple axes.

    Args:
        database: Database handle.
        lookback: Max number of most-recent closed trades to consider.

    Returns:
        List of cluster dicts, each:
            {
                "axis": str,          # which field was bucketed
                "bucket": str,        # the bucket label
                "trades": int,
                "losses": int,
                "loss_rate": float,
                "baseline_loss_rate": float,
                "avg_pnl": float,
                "severity": float,    # loss_rate - baseline_loss_rate
            }
        Empty list when the dataset is too small.
    """
    closed = _load_closed_trades(database, lookback)
    if len(closed) < MIN_BASELINE_TRADES:
        logger.info(
            "loss_pattern_insufficient_data",
            closed_trades=len(closed),
            required=MIN_BASELINE_TRADES,
        )
        return []

    baseline_losses = sum(1 for t in closed if _safe_float(t.get("pnl")) <= 0)
    baseline_rate = baseline_losses / len(closed)

    analysis_map = _load_analysis_map(database)

    clusters: list[dict] = []
    for axis in AXES:
        by_bucket: dict[str, list[dict]] = defaultdict(list)
        for t in closed:
            analysis = analysis_map.get(int(t.get("id") or 0))
            key = _bucket_key(axis, t, analysis)
            by_bucket[key].append(t)

        for bucket, trades in by_bucket.items():
            if bucket == "unknown" or len(trades) < MIN_TRADES_FOR_CLUSTER:
                continue
            losses = sum(1 for t in trades if _safe_float(t.get("pnl")) <= 0)
            loss_rate = losses / len(trades)
            severity = loss_rate - baseline_rate
            if severity < MIN_LOSS_RATE_DELTA:
                continue
            avg_pnl = sum(_safe_float(t.get("pnl")) for t in trades) / len(trades)
            clusters.append({
                "axis": axis,
                "bucket": bucket,
                "trades": len(trades),
                "losses": losses,
                "loss_rate": round(loss_rate, 3),
                "baseline_loss_rate": round(baseline_rate, 3),
                "avg_pnl": round(avg_pnl, 2),
                "severity": round(severity, 3),
            })

    clusters.sort(key=lambda c: c["severity"], reverse=True)

    if clusters:
        logger.info(
            "loss_patterns_found",
            count=len(clusters),
            baseline_loss_rate=round(baseline_rate, 3),
            worst_axis=clusters[0]["axis"],
            worst_bucket=clusters[0]["bucket"],
            worst_severity=clusters[0]["severity"],
        )
    else:
        logger.info(
            "loss_patterns_none",
            total_trades=len(closed),
            baseline_loss_rate=round(baseline_rate, 3),
        )

    return clusters
