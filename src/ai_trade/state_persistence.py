"""Restart-safe state loading and reconciliation (V2 Phase 4).

This module centralises everything that needs to happen between
`load_config()` and `scheduler.start()` so the bot can resume a
previous session without losing learned state.

Three kinds of persistence are handled here:

1. **Parameter overrides** — Stored in the `parameter_overrides` table,
   written by the strategy optimiser (Phase 9) or manually.  On startup
   we patch them onto the `SimpleNamespace` config tree on top of the
   base `settings.yaml` values.  The base file remains the starting
   point; overrides are the learned current state.

2. **Bot state** — Key/value pairs in the `bot_state` table
   (streaks, circuit breaker state, last scan timestamps, recalc
   cursors, etc.).  This module exposes a small helper for getting
   and setting the "last shutdown" timestamp which drives the
   offline gap detection.

3. **Offline gap detection** — Compares the last shutdown timestamp
   to `now`.  If more than 5 business days have elapsed we flag for
   human review; smaller gaps just get logged so the operator can
   see how long the bot was down.

The strategy weighter handles its own persistence directly in
`strategy/weighter.py` because the weight rows live in a dedicated
table and are tightly coupled with the recalc cursor.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from ai_trade.monitoring.logger import get_logger

if TYPE_CHECKING:
    from ai_trade.monitoring.database import Database

log = get_logger(__name__)


STATE_LAST_SHUTDOWN = "bot.last_shutdown_utc"
STATE_LAST_STARTUP = "bot.last_startup_utc"


def _coerce(raw: str) -> Any:
    """Best-effort coerce a stored string value back to its Python type.

    Override values are stored as strings because `bot_state` and
    `parameter_overrides` use a TEXT column (mixed-type storage).
    On load we try int → float → bool → str so the patched config
    attribute matches the original type.
    """
    if raw is None:
        return None
    s = raw.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def apply_parameter_overrides(
    cfg: SimpleNamespace,
    database: Database,
    regime: str | None = None,
) -> list[dict]:
    """Overlay persisted parameter overrides onto the config tree.

    When ``regime`` is provided, loads regime-specific overrides with a
    global fallback (see ``Database.get_effective_overrides``).  When
    ``regime`` is None (first boot, unknown regime), only global
    overrides are applied.

    Returns the list of applied overrides so the boot log can show
    exactly what was patched.  Unknown strategies/params are skipped
    with a warning — we never want a stale override to crash startup.
    """
    try:
        if regime:
            rows = database.get_effective_overrides(regime=regime)
        else:
            rows = database.get_parameter_overrides(regime="")
    except Exception:
        log.exception("parameter_overrides_load_failed")
        return []

    applied: list[dict] = []
    strategies_ns = getattr(cfg, "strategies", None)
    if strategies_ns is None:
        return []

    for row in rows:
        strategy_name = row["strategy_name"]
        param_name = row["param_name"]
        raw_value = row["value"]
        target = getattr(strategies_ns, strategy_name, None)
        if target is None:
            log.warning(
                "parameter_override_skipped_unknown_strategy",
                strategy=strategy_name,
                param=param_name,
            )
            continue
        if not hasattr(target, param_name):
            log.warning(
                "parameter_override_skipped_unknown_param",
                strategy=strategy_name,
                param=param_name,
            )
            continue

        new_value = _coerce(raw_value)
        setattr(target, param_name, new_value)
        applied.append({
            "strategy": strategy_name,
            "param": param_name,
            "regime": row.get("regime", ""),
            "value": new_value,
            "reason": row.get("reason"),
        })

    return applied


STATE_CURRENT_REGIME = "market.current_regime"


def record_current_regime(database: Database, regime: str) -> None:
    """Persist the current market regime so it survives restarts."""
    try:
        database.set_state(STATE_CURRENT_REGIME, regime)
    except Exception:
        log.exception("record_regime_failed")


def get_current_regime(database: Database) -> str:
    """Read the last-persisted market regime (empty string if unset)."""
    try:
        return database.get_state(STATE_CURRENT_REGIME, "") or ""
    except Exception:
        log.exception("get_regime_failed")
        return ""


def record_startup(database: Database) -> None:
    """Write the startup timestamp to bot_state."""
    try:
        database.set_state(STATE_LAST_STARTUP, datetime.utcnow().isoformat())
    except Exception:
        log.exception("record_startup_failed")


def record_shutdown(database: Database) -> None:
    """Write the shutdown timestamp to bot_state (called from the handler)."""
    try:
        database.set_state(STATE_LAST_SHUTDOWN, datetime.utcnow().isoformat())
    except Exception:
        log.exception("record_shutdown_failed")


def detect_offline_gap(database: Database) -> dict | None:
    """Compare the last shutdown timestamp to `now`.

    Returns a dict describing the gap (for boot logging), or None if
    there is no prior shutdown timestamp (first run or fresh DB).
    """
    try:
        raw = database.get_state(STATE_LAST_SHUTDOWN)
    except Exception:
        log.exception("offline_gap_detect_failed")
        return None

    if not raw:
        return None

    try:
        last_shutdown = datetime.fromisoformat(raw)
    except ValueError:
        log.warning("offline_gap_bad_timestamp", raw=raw)
        return None

    now = datetime.utcnow()
    delta = now - last_shutdown
    # Rough business-day count: weekdays between last_shutdown and now.
    business_days = 0
    cursor = last_shutdown.date()
    while cursor < now.date():
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            business_days += 1

    return {
        "last_shutdown": raw,
        "now": now.isoformat(),
        "elapsed_hours": round(delta.total_seconds() / 3600, 2),
        "business_days": business_days,
        "flag_for_review": business_days > 5,
    }


def log_boot_summary(
    *,
    weighter_rows: int,
    overrides_applied: list[dict],
    gap: dict | None,
) -> None:
    """Emit the structured boot summary expected by the V2 brief."""
    log.info(
        "boot_state_restored",
        strategy_weights_loaded=weighter_rows,
        parameter_overrides_applied=len(overrides_applied),
        overrides=[
            f"{o['strategy']}.{o['param']}={o['value']}"
            for o in overrides_applied
        ],
        offline_gap=gap,
    )
    if gap and gap["flag_for_review"]:
        log.warning(
            "offline_gap_exceeds_review_threshold",
            business_days=gap["business_days"],
            last_shutdown=gap["last_shutdown"],
        )
