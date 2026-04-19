"""Adaptive parameter optimizer (Phase 9).

Reads per-strategy quality distributions from ``trade_analysis`` and
proposes small, bounded adjustments to any parameter registered in
``parameter_specs.PARAM_SPECS``.

Phase 9 additions over the Phase 6 version:
    - Multi-parameter: iterates every ParamSpec, not just atr_stop_multiplier.
    - Per-regime: groups trades by ``trade_analysis.market_regime`` and
      writes regime-specific overrides so the bot tunes differently in
      bull vs bear markets.  A global (regime='') proposal is always
      generated as well.
    - Rolling window: only considers the most recent ``window`` trades
      per (strategy, regime) so the optimizer adapts to current market
      conditions rather than all-time averages.

Write model:
    - ``parameter_history`` — one row per proposal (audit trail).
    - ``parameter_overrides`` — the new active value, keyed by
      (strategy_name, param_name, regime).  Phase 4's
      ``apply_parameter_overrides`` patches these into the config on
      the NEXT bot restart.

Guardrails:
    - Requires at least ``min_trades`` quality-tagged trades before
      proposing anything.
    - Requires a decisive majority (>=60%) before nudging.
    - Hard bounds per ParamSpec.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ai_trade.analysis.parameter_specs import PARAM_SPECS, ParamSpec
from ai_trade.monitoring.logger import get_logger

logger = get_logger(__name__)

MIN_SCORED_TRADES = 8
DECISIVE_MAJORITY = 0.60
OPTIMIZER_SET_BY = "parameter_optimizer"
DEFAULT_WINDOW = 50


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        f = float(x)
        if f != f:
            return default
        return f
    except (TypeError, ValueError):
        return default


def _clamp(value: float, spec: ParamSpec) -> float:
    return max(spec.min_val, min(spec.max_val, round(value, 3)))


def _load_joined_rows(database, window: int = 0) -> list[dict]:
    """Join trades with trade_analysis.

    Returns the most recent ``window`` closed trades (by trade id) if
    window > 0, otherwise all closed trades.
    """
    suffix = f" LIMIT {int(window)}" if window > 0 else ""
    try:
        with database._conn() as conn:  # noqa: SLF001
            rows = conn.execute(
                f"""
                SELECT t.id, t.strategy, t.pnl, t.pnl_pct,
                       ta.stop_quality, ta.entry_quality, ta.exit_quality,
                       ta.market_regime
                FROM trades t
                JOIN trade_analysis ta ON ta.trade_id = t.id
                WHERE t.status = 'closed' AND t.pnl IS NOT NULL
                ORDER BY t.id DESC
                {suffix}
                """,
            ).fetchall()
    except Exception:
        logger.exception("optimizer_load_failed")
        return []
    return [dict(r) for r in rows]


def _current_param_value(cfg, strategy: str, param_name: str) -> float | None:
    try:
        node = getattr(cfg.strategies, strategy, None)
        if node is None:
            return None
        val = getattr(node, param_name, None)
        return _safe_float(val) if val is not None else None
    except Exception:
        return None


def _active_override(
    database, strategy: str, param_name: str, regime: str = "",
) -> float | None:
    try:
        rows = database.get_parameter_overrides(regime=regime)
    except Exception:
        return None
    for r in rows:
        if r.get("strategy_name") == strategy and r.get("param_name") == param_name:
            return _safe_float(r.get("value"))
    return None


def _analyze_param(
    trades: list[dict],
    spec: ParamSpec,
    current: float,
) -> dict | None:
    """Decide whether to widen, tighten, or skip for a single param+cohort."""
    quality_vals = [
        t.get(spec.quality_field)
        for t in trades
        if t.get(spec.quality_field) is not None
    ]
    actionable = [
        v for v in quality_vals
        if v in spec.widen_values or v in spec.tighten_values or v == "just_right"
    ]
    if len(actionable) < MIN_SCORED_TRADES:
        return None

    widen_count = sum(1 for v in actionable if v in spec.widen_values)
    tighten_count = sum(1 for v in actionable if v in spec.tighten_values)
    total = len(actionable)
    widen_rate = widen_count / total
    tighten_rate = tighten_count / total

    direction: str | None = None
    new_value = current
    reason = ""

    if widen_rate >= DECISIVE_MAJORITY:
        new_value = _clamp(current + spec.step, spec)
        direction = "widen"
        reason = (
            f"{widen_count}/{total} scored as widen-signal "
            f"({widen_rate:.0%} >= {DECISIVE_MAJORITY:.0%})"
        )
    elif tighten_rate >= DECISIVE_MAJORITY:
        new_value = _clamp(current - spec.step, spec)
        direction = "tighten"
        reason = (
            f"{tighten_count}/{total} scored as tighten-signal "
            f"({tighten_rate:.0%} >= {DECISIVE_MAJORITY:.0%})"
        )

    if direction is None or new_value == current:
        return None

    return {
        "old_value": round(current, 3),
        "new_value": round(new_value, 3),
        "direction": direction,
        "reason": reason,
    }


def review_and_adjust(
    database,
    cfg,
    min_trades: int = MIN_SCORED_TRADES,
    window: int = DEFAULT_WINDOW,
    apply_changes: bool = False,
) -> list[dict]:
    """Review quality distributions and propose/apply adjustments.

    Iterates every ParamSpec, groups trades by strategy then by regime,
    and produces per-regime proposals + a global (regime='') proposal.

    Returns a list of proposal dicts with keys: strategy, param, regime,
    old_value, new_value, direction, reason, applied.
    """
    rows = _load_joined_rows(database, window=window)
    if not rows:
        logger.info("optimizer_no_data")
        return []

    # Group: strategy -> regime -> [trades]
    by_strat_regime: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list),
    )
    for r in rows:
        strategy = r.get("strategy")
        regime = r.get("market_regime") or ""
        if strategy:
            by_strat_regime[strategy][regime].append(r)
            by_strat_regime[strategy][""].append(r)

    proposals: list[dict] = []

    for strategy, regime_groups in by_strat_regime.items():
        for regime, trades in regime_groups.items():
            for spec in PARAM_SPECS.values():
                # Resolve current value: override > config base
                current = _active_override(database, strategy, spec.name, regime)
                if current is None and regime:
                    current = _active_override(database, strategy, spec.name, "")
                if current is None:
                    current = _current_param_value(cfg, strategy, spec.name)
                if current is None:
                    continue

                result = _analyze_param(trades, spec, current)
                if result is None:
                    continue

                proposal = {
                    "strategy": strategy,
                    "param": spec.name,
                    "regime": regime,
                    **result,
                    "applied": False,
                }

                try:
                    database.log_parameter_change(
                        strategy=strategy,
                        param_name=spec.name,
                        old_value=result["old_value"],
                        new_value=result["new_value"],
                        reason=f"[{regime or 'global'}] {result['reason']}",
                    )
                except Exception:
                    logger.exception(
                        "optimizer_history_write_failed",
                        strategy=strategy, param=spec.name,
                    )

                if apply_changes:
                    try:
                        database.set_parameter_override(
                            strategy_name=strategy,
                            param_name=spec.name,
                            value=str(result["new_value"]),
                            regime=regime,
                            set_by=OPTIMIZER_SET_BY,
                            reason=result["reason"],
                        )
                        proposal["applied"] = True
                    except Exception:
                        logger.exception(
                            "optimizer_override_write_failed",
                            strategy=strategy, param=spec.name,
                        )

                logger.info(
                    "optimizer_proposal",
                    strategy=strategy,
                    param=spec.name,
                    regime=regime or "global",
                    old_value=result["old_value"],
                    new_value=result["new_value"],
                    direction=result["direction"],
                )
                proposals.append(proposal)

    if not proposals:
        logger.info(
            "optimizer_no_proposals",
            strategies_reviewed=len(by_strat_regime),
            window=window,
        )

    return proposals
