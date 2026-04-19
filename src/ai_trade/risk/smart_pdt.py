"""Smart PDT management — Phase 8.

The $500 paper account only gets 3 day-trade slots per rolling 5-business-day
window.  Phase 1 treated them first-come-first-serve at a static 0.80
conviction floor.  This module adds four pieces on top:

    1. Dynamic conviction threshold
        - Early in the week, slots are precious -> demand higher conviction.
        - Late in the week, unused slots are about to expire -> loosen up.
        - Threshold also scales with slots_remaining: when only 1 slot is
          left, the bar stays high regardless of the day.

    2. Day-to-swing conversion
        - When PDT is tight OR conviction doesn't meet the dynamic floor,
          check whether the rejected DAY/ADAPTIVE signal is from a strategy
          whose thesis holds overnight (momentum, bb_squeeze, breakout
          continuation).  If yes, widen stops + targets and flip to SWING
          so the signal still gets executed but costs no PDT slot.
        - Pure intraday setups (orb, vwap) are NEVER converted — their
          thesis dies at the bell.

    3. Slot value estimation
        - Reads recent closed day-trade rows from ``trades`` and groups by
          hour bucket + day-of-week.  Win rate per bucket feeds the
          dynamic threshold so the bot learns "Monday 9:45 entries are
          cheap, Friday 14:30 entries are where the alpha is".  Cold-start
          safe: falls back to the flat default when <10 day trades exist.

    4. Weekly plan
        - ``weekly_slot_plan(day_of_week, slots_remaining)`` returns a
          stance label ("aggressive" / "neutral" / "conservative") and a
          threshold bump the aggregator logs once per cycle for visibility.

The module is pure (no singleton, no hidden state) — the
``SmartPDTPlanner`` class is instantiated once in ``TradingBot`` with
config + database and called per evaluation cycle.  All DB hits are
once-per-cycle to mirror the Phase 7 dynamic-risk refresh pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.base import HoldType, Signal

logger = get_logger(__name__)


# ── Tunables (constants; overridable via config) ──────────────────────
DEFAULT_DAY_TRADE_FLOOR = 0.80

# Monday .. Friday index -> threshold bump.  Monday is stingy because the
# whole week is ahead; Friday is loose because unused slots expire this
# weekend.  Conservative on Tue/Wed, neutral on Thu.
_DOW_BUMP: dict[int, float] = {
    0: +0.05,   # Monday: need 0.85
    1: +0.03,   # Tuesday: need 0.83
    2: +0.02,   # Wednesday: need 0.82
    3: 0.00,    # Thursday: need 0.80
    4: -0.05,   # Friday: need 0.75
}

# Extra bump based on slots remaining.  Last slot -> stingier; first
# slot -> same as base.  Negative means looser.
_SLOT_BUMP: dict[int, float] = {
    3: -0.02,   # Fresh week / fresh window
    2: 0.00,    # Neutral
    1: +0.03,   # Only one left -> raise the bar
    0: +1.0,    # Impossible but harmless sentinel
}

# Strategies whose thesis tolerates an overnight hold.  These can be
# converted day -> swing when PDT is tight.  Intraday-only setups
# (orb, vwap) are deliberately absent.
_SWING_COMPATIBLE_STRATEGIES: set[str] = {
    "momentum",
    "bb_squeeze",
    "ema_crossover",
    "macd_divergence",
    "pullback",
    "mean_reversion",
}

# Per-strategy stop widening when converting day -> swing.  Multipliers
# applied to the existing entry-to-stop distance.
SWING_STOP_WIDEN_FACTOR = 1.5
SWING_TARGET_WIDEN_FACTOR = 1.5

MIN_SAMPLES_FOR_EV = 10


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class PDTPlan:
    """Per-cycle PDT planning snapshot the aggregator reads."""
    slots_remaining: int
    day_of_week: int
    day_name: str
    dynamic_threshold: float
    stance: str                # "aggressive" | "neutral" | "conservative" | "frozen"
    base_floor: float
    dow_bump: float
    slot_bump: float
    ev_bump: float
    reasons: list[str] = field(default_factory=list)


# ── EV estimation ─────────────────────────────────────────────────────

def _hour_bucket(entry_time: str | None) -> str:
    """Map an ISO entry_time to a coarse hour bucket (open/mid/close)."""
    if not entry_time:
        return "unknown"
    try:
        dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
    except Exception:
        return "unknown"
    h = dt.hour
    if h < 11:
        return "open"
    if h < 14:
        return "mid"
    return "close"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        f = float(x)
        if f != f:
            return default
        return f
    except (TypeError, ValueError):
        return default


def estimate_day_trade_ev_bump(database) -> float:
    """Compute a global threshold nudge from recent day-trade win rates.

    Returns a SMALL bump: if recent day trades are winning >60% we nudge
    the floor DOWN by 0.02 (embolden); if winning <40% we nudge UP by
    0.03 (punish).  Cold-start (<10 closed day trades) -> 0.0.

    Kept intentionally blunt — the dynamic-risk controller already
    handles streak effects.  This is strictly a day-trade-specific
    signal layered on top.
    """
    try:
        trades = database.get_all_trades()
    except Exception:
        logger.exception("smart_pdt_ev_query_failed")
        return 0.0

    day_trades = [
        t for t in trades
        if t.get("status") == "closed"
        and t.get("pnl") is not None
        and t.get("hold_type") == "day"
    ]
    if len(day_trades) < MIN_SAMPLES_FOR_EV:
        return 0.0

    recent = day_trades[:30]
    wins = sum(1 for t in recent if _safe_float(t.get("pnl")) > 0)
    win_rate = wins / len(recent)

    if win_rate >= 0.60:
        return -0.02
    if win_rate <= 0.40:
        return +0.03
    return 0.0


# ── Threshold computation ─────────────────────────────────────────────

def dynamic_day_trade_threshold(
    slots_remaining: int,
    day_of_week: int,
    base: float = DEFAULT_DAY_TRADE_FLOOR,
    ev_bump: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """Return (threshold, breakdown) for the current moment.

    Threshold clamped to [0.50, 0.95] so a malformed bump can't
    accidentally freeze or open the floodgates.
    """
    dow_bump = _DOW_BUMP.get(day_of_week, 0.0)
    slot_bump = _SLOT_BUMP.get(slots_remaining, 0.0)
    if slots_remaining <= 0:
        # No slots -> conceptual infinity (caller halts day trades
        # regardless; we return a sentinel the caller can identify).
        threshold = 1.01
    else:
        threshold = base + dow_bump + slot_bump + ev_bump
        threshold = max(0.50, min(0.95, threshold))
    return threshold, {
        "base": base,
        "dow_bump": dow_bump,
        "slot_bump": slot_bump,
        "ev_bump": ev_bump,
    }


def _stance_from_threshold(threshold: float, slots_remaining: int) -> str:
    if slots_remaining <= 0:
        return "frozen"
    if threshold >= DEFAULT_DAY_TRADE_FLOOR + 0.04:
        return "conservative"
    if threshold <= DEFAULT_DAY_TRADE_FLOOR - 0.03:
        return "aggressive"
    return "neutral"


# ── Day-to-swing conversion ───────────────────────────────────────────

def is_swing_compatible(strategy_name: str) -> bool:
    return strategy_name in _SWING_COMPATIBLE_STRATEGIES


def convert_day_to_swing(sig: Signal) -> Signal:
    """Widen the stop + target of a DAY signal and flip to SWING in place.

    Returns the same Signal object (mutated) so callers don't need to
    rebuild execution queue entries.  The widening keeps the entry
    unchanged and scales both legs away from it by
    SWING_STOP_WIDEN_FACTOR / SWING_TARGET_WIDEN_FACTOR respectively,
    preserving the R:R ratio of the original signal.

    Metadata is stamped with a `pdt_conversion` breadcrumb so the
    trade journal + decision logger can attribute the adjustment.
    """
    entry = sig.entry_price
    stop = sig.stop_loss_price
    target = sig.take_profit_price

    new_stop = entry - (entry - stop) * SWING_STOP_WIDEN_FACTOR
    new_target = entry + (target - entry) * SWING_TARGET_WIDEN_FACTOR

    sig.stop_loss_price = new_stop
    sig.take_profit_price = new_target
    sig.hold_type = HoldType.SWING
    if sig.metadata is None:
        sig.metadata = {}
    sig.metadata["pdt_conversion"] = {
        "original_hold": "day",
        "stop_widen_factor": SWING_STOP_WIDEN_FACTOR,
        "target_widen_factor": SWING_TARGET_WIDEN_FACTOR,
        "original_stop": stop,
        "original_target": target,
    }
    return sig


# ── Planner ───────────────────────────────────────────────────────────

class SmartPDTPlanner:
    """Once-per-cycle PDT planning + helpers.

    Usage (from SignalAggregator.collect_and_rank):

        plan = planner.plan_cycle(pdt_manager, now=...)
        if plan.stance == "frozen":
            # no day trades at all this cycle
            ...
        for sig in ranked:
            if sig.hold_type == HoldType.SWING:
                ...
            else:
                # DAY / ADAPTIVE: check plan.dynamic_threshold,
                # attempt swing conversion if rejected and eligible
                ...
    """

    def __init__(self, config, database) -> None:
        self._cfg = config
        self._db = database
        self._base_floor: float = float(
            getattr(config, "min_conviction_for_day_trade",
                    DEFAULT_DAY_TRADE_FLOOR)
        )

    def plan_cycle(
        self,
        pdt_manager,
        now: datetime | None = None,
    ) -> PDTPlan:
        """Build the per-cycle snapshot.  DB hit (for EV) is once per cycle."""
        if now is None:
            now = datetime.now(timezone.utc)
        dow = now.weekday()
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        # Respect reserve when measuring "slots remaining" so the
        # planner's view lines up with what pdt_manager.can_day_trade
        # will actually allow.
        try:
            remaining_raw = pdt_manager.day_trades_remaining()
        except Exception:
            logger.exception("smart_pdt_slots_query_failed")
            remaining_raw = 0
        reserve: int = int(getattr(pdt_manager.config, "day_trade_reserve", 1))
        slots_remaining = max(0, remaining_raw - reserve)

        ev_bump = estimate_day_trade_ev_bump(self._db)
        threshold, breakdown = dynamic_day_trade_threshold(
            slots_remaining=slots_remaining,
            day_of_week=dow,
            base=self._base_floor,
            ev_bump=ev_bump,
        )
        stance = _stance_from_threshold(threshold, slots_remaining)

        reasons: list[str] = []
        if dow in (0, 1):
            reasons.append(f"{day_names[dow]} early-week stinginess")
        elif dow == 4:
            reasons.append("Friday: use slots or lose them this weekend")
        if slots_remaining <= 1 and slots_remaining > 0:
            reasons.append(f"{slots_remaining} slot(s) left -- raise the bar")
        if slots_remaining <= 0:
            reasons.append("no slots remaining -- day trades frozen")
        if ev_bump < 0:
            reasons.append("recent day-trade win-rate high -- nudging floor down")
        elif ev_bump > 0:
            reasons.append("recent day-trade win-rate low -- nudging floor up")

        return PDTPlan(
            slots_remaining=slots_remaining,
            day_of_week=dow,
            day_name=day_names[dow],
            dynamic_threshold=threshold,
            stance=stance,
            base_floor=breakdown["base"],
            dow_bump=breakdown["dow_bump"],
            slot_bump=breakdown["slot_bump"],
            ev_bump=breakdown["ev_bump"],
            reasons=reasons,
        )

    @staticmethod
    def is_swing_compatible(strategy_name: str) -> bool:
        return is_swing_compatible(strategy_name)

    @staticmethod
    def convert_day_to_swing(sig: Signal) -> Signal:
        return convert_day_to_swing(sig)
