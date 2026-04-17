"""Dynamic risk tolerance — Phase 7.

The ``DynamicRiskController`` is the single place that decides how much
risk the bot is willing to take RIGHT NOW, blending four independent
factors into one set of sizing multipliers the position sizer and risk
manager consume:

    1. Conviction-scaled sizing
        0.50 .. 0.60  → 0.5x base
        0.60 .. 0.75  → 1.0x base
        0.75 .. 0.85  → 1.5x base
        0.85 .. 0.95  → 2.0x base
        >= 0.90       → also unlocks the high-conviction concentration
                        override (up to 50% of portfolio)

    2. Streak-based aggression
        Looks at the last ``streak_window`` closed trades:
            >60% win rate  → +0.5x risk scale
            <35% win rate  → -0.5x risk scale (floored at 0.5x)
            otherwise      → neutral

    3. Regime-aware sizing
        strong_bull → +0.1x, allow more concurrent positions
        bull        → neutral
        neutral     → neutral
        bear        → -0.1x, fewer positions, tighter concentration cap
        strong_bear → -0.3x, much fewer positions, block new longs entirely

    4. Tiered drawdown circuit breakers (intraday vs starting equity)
        tier 0:  0.0% ..  3.0% down → full risk
        tier 1:  3.0% ..  5.0% down → 0.5x risk, still allow new entries
        tier 2:  5.0% .. 15.0% down → halt new entries (original hard stop)
        tier 3: 15.0%+ down          → halt + require manual ack

Design rules:
    - Controller is stateful: streak + last-computed tier cached on
      ``bot_state`` so a mid-day restart resumes from the same footing.
    - All multipliers are multiplicative and clamped to sane bounds so a
      bug in one input can't blow up risk.
    - Controller is read-only from strategies' perspective; it never
      mutates signal fields directly.  Position sizer and risk manager
      pull multipliers via ``current_snapshot()``.

The brief describes these as independent knobs; this module is the one
place they compose, so the rest of the system stays thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_trade.monitoring.logger import get_logger

logger = get_logger(__name__)


# ── Tunables (kept as module constants; override via cfg.risk.dynamic) ──
STREAK_WINDOW = 10
STREAK_WIN_THRESHOLD = 0.60
STREAK_LOSS_THRESHOLD = 0.35
STREAK_WIN_DELTA = 0.50
STREAK_LOSS_DELTA = 0.50
STREAK_FLOOR = 0.50
STREAK_CEILING = 2.00

HIGH_CONVICTION_MIN = 0.90
HIGH_CONVICTION_POSITION_PCT = 0.50  # 50% of portfolio max

STATE_STREAK_SCALE = "risk.streak_scale"
STATE_STREAK_WINDOW_SAMPLE = "risk.streak_sample_size"
STATE_DD_TIER = "risk.drawdown_tier"
STATE_STARTING_EQUITY = "risk.starting_equity"


# ── Conviction scaling ────────────────────────────────────────────────

def conviction_size_multiplier(conviction: float) -> float:
    """Map conviction to base-size multiplier per the V2 brief ladder."""
    if conviction < 0.50:
        return 0.0
    if conviction < 0.60:
        return 0.5
    if conviction < 0.75:
        return 1.0
    if conviction < 0.85:
        return 1.5
    return 2.0


def is_high_conviction(conviction: float) -> bool:
    return conviction >= HIGH_CONVICTION_MIN


# ── Regime scaling ────────────────────────────────────────────────────

_REGIME_RISK_DELTA: dict[str, float] = {
    "strong_bull": 0.10,
    "bull": 0.0,
    "neutral": 0.0,
    "bear": -0.10,
    "strong_bear": -0.30,
}

_REGIME_POSITION_BONUS: dict[str, int] = {
    "strong_bull": 2,
    "bull": 1,
    "neutral": 0,
    "bear": -1,
    "strong_bear": -2,
}


def regime_risk_scale(regime: str | None) -> float:
    if not regime:
        return 1.0
    return 1.0 + _REGIME_RISK_DELTA.get(regime, 0.0)


def regime_position_bonus(regime: str | None) -> int:
    if not regime:
        return 0
    return _REGIME_POSITION_BONUS.get(regime, 0)


def regime_allows_new_longs(regime: str | None) -> bool:
    return regime != "strong_bear"


# ── Drawdown tiers ────────────────────────────────────────────────────

@dataclass
class DrawdownTier:
    tier: int
    loss_pct: float
    size_scale: float
    allow_new_entries: bool
    reason: str


def classify_drawdown(
    starting_equity: float | None,
    current_equity: float,
) -> DrawdownTier:
    """Classify the intraday drawdown against starting equity."""
    if starting_equity is None or starting_equity <= 0:
        return DrawdownTier(0, 0.0, 1.0, True, "no starting equity")
    loss_pct = max(0.0, (starting_equity - current_equity) / starting_equity)
    if loss_pct >= 0.15:
        return DrawdownTier(3, loss_pct, 0.0, False,
                            f"drawdown {loss_pct:.2%} >= 15% — halt + manual ack")
    if loss_pct >= 0.05:
        return DrawdownTier(2, loss_pct, 0.0, False,
                            f"drawdown {loss_pct:.2%} >= 5% — halt new entries")
    if loss_pct >= 0.03:
        return DrawdownTier(1, loss_pct, 0.5, True,
                            f"drawdown {loss_pct:.2%} >= 3% — half-size only")
    return DrawdownTier(0, loss_pct, 1.0, True, "within normal range")


# ── Streak tracking ───────────────────────────────────────────────────

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        f = float(x)
        if f != f:
            return default
        return f
    except (TypeError, ValueError):
        return default


def compute_streak_scale(database, window: int = STREAK_WINDOW) -> tuple[float, int, int]:
    """Look at the last ``window`` closed trades and return a risk scale.

    Returns:
        (scale, sample_size, wins)
    """
    try:
        trades = database.get_all_trades()
    except Exception:
        logger.exception("streak_load_failed")
        return 1.0, 0, 0

    closed = [
        t for t in trades
        if t.get("status") == "closed" and t.get("pnl") is not None
    ]
    recent = closed[:window]
    if len(recent) < 5:
        return 1.0, len(recent), sum(1 for t in recent if _safe_float(t.get("pnl")) > 0)

    wins = sum(1 for t in recent if _safe_float(t.get("pnl")) > 0)
    win_rate = wins / len(recent)

    if win_rate >= STREAK_WIN_THRESHOLD:
        scale = 1.0 + STREAK_WIN_DELTA
    elif win_rate <= STREAK_LOSS_THRESHOLD:
        scale = 1.0 - STREAK_LOSS_DELTA
    else:
        scale = 1.0

    scale = max(STREAK_FLOOR, min(STREAK_CEILING, scale))
    return scale, len(recent), wins


# ── Controller ────────────────────────────────────────────────────────

@dataclass
class RiskSnapshot:
    """Composite runtime snapshot the sizer + risk manager read from."""
    conviction_scale: float
    streak_scale: float
    regime_scale: float
    drawdown_scale: float
    total_risk_scale: float
    max_position_pct: float
    allow_new_entries: bool
    allow_new_longs: bool
    max_open_positions: int
    high_conviction_override: bool
    drawdown_tier: int
    regime: str | None
    reasons: list[str] = field(default_factory=list)


class DynamicRiskController:
    """Composes runtime risk factors into one snapshot per sizing call.

    Stateful via ``bot_state``:
        - ``risk.streak_scale``         last-computed streak multiplier
        - ``risk.streak_sample_size``   how many trades went into it
        - ``risk.drawdown_tier``        most recent tier classification
        - ``risk.starting_equity``      today's starting equity

    The controller caches its own view of streak_scale so strategies
    don't thrash the DB every signal; ``refresh_streak()`` is called
    once per collect_and_rank cycle.
    """

    def __init__(self, config, database) -> None:
        self._cfg = config
        self._db = database
        self._streak_scale: float = 1.0
        self._streak_sample: int = 0
        self._drawdown: DrawdownTier = DrawdownTier(0, 0.0, 1.0, True, "init")
        self._starting_equity: float | None = None
        self._base_max_position_pct: float = float(
            getattr(config, "max_position_pct", 0.25)
        )
        self._base_max_positions: int = int(
            getattr(config, "max_open_positions", 5)
        )
        self._load_persisted_state()

    # ── state persistence ─────────────────────────────────────
    def _load_persisted_state(self) -> None:
        try:
            raw = self._db.get_state(STATE_STREAK_SCALE)
            if raw is not None:
                self._streak_scale = _safe_float(raw, 1.0)
            sample_raw = self._db.get_state(STATE_STREAK_WINDOW_SAMPLE)
            if sample_raw is not None:
                self._streak_sample = int(_safe_float(sample_raw, 0))
            se_raw = self._db.get_state(STATE_STARTING_EQUITY)
            if se_raw is not None:
                self._starting_equity = _safe_float(se_raw, None)  # type: ignore[arg-type]
        except Exception:
            logger.exception("dynamic_risk_state_load_failed")

    def set_starting_equity(self, equity: float) -> None:
        """Cache today's opening equity and persist for restart-safety."""
        self._starting_equity = float(equity)
        try:
            self._db.set_state(STATE_STARTING_EQUITY, f"{equity:.2f}")
        except Exception:
            logger.exception("dynamic_risk_starting_equity_persist_failed")

    def refresh_streak(self) -> None:
        """Recompute the streak scale and persist it.

        Called once per evaluation cycle from TradingBot._evaluate_and_trade
        so every signal in the cycle reads the same streak view.
        """
        scale, sample, wins = compute_streak_scale(self._db)
        self._streak_scale = scale
        self._streak_sample = sample
        try:
            self._db.set_state(STATE_STREAK_SCALE, f"{scale:.3f}")
            self._db.set_state(STATE_STREAK_WINDOW_SAMPLE, str(sample))
        except Exception:
            logger.exception("dynamic_risk_streak_persist_failed")
        if sample >= 5:
            logger.info(
                "dynamic_risk_streak_refreshed",
                scale=scale,
                sample=sample,
                wins=wins,
                win_rate=round(wins / sample, 3),
            )

    def refresh_drawdown(self, current_equity: float) -> DrawdownTier:
        """Classify the current drawdown tier and persist."""
        dd = classify_drawdown(self._starting_equity, current_equity)
        self._drawdown = dd
        try:
            self._db.set_state(STATE_DD_TIER, str(dd.tier))
        except Exception:
            logger.exception("dynamic_risk_dd_persist_failed")
        if dd.tier >= 1:
            logger.warning(
                "dynamic_risk_drawdown_tier",
                tier=dd.tier,
                loss_pct=round(dd.loss_pct, 4),
                size_scale=dd.size_scale,
                allow_new=dd.allow_new_entries,
                reason=dd.reason,
            )
        return dd

    # ── snapshot composition ──────────────────────────────────
    def snapshot_for_signal(
        self,
        conviction: float,
        regime: str | None,
        current_equity: float,
    ) -> RiskSnapshot:
        """Compose a RiskSnapshot for one candidate signal.

        Does NOT touch the database — callers should have invoked
        ``refresh_streak`` and ``refresh_drawdown`` once per cycle.
        """
        conv_scale = conviction_size_multiplier(conviction)
        streak_scale = self._streak_scale
        regime_scale = regime_risk_scale(regime)
        drawdown_scale = self._drawdown.size_scale
        high_conv = is_high_conviction(conviction)

        total_scale = conv_scale * streak_scale * regime_scale * drawdown_scale
        total_scale = max(0.0, min(3.0, total_scale))

        max_pct = self._base_max_position_pct
        if high_conv:
            max_pct = max(max_pct, HIGH_CONVICTION_POSITION_PCT)

        max_positions = max(
            1,
            self._base_max_positions + regime_position_bonus(regime),
        )

        reasons: list[str] = []
        if conv_scale == 0.0:
            reasons.append(f"conviction {conviction:.2f} below 0.50 floor")
        if streak_scale > 1.0:
            reasons.append(f"winning streak x{streak_scale:.2f}")
        elif streak_scale < 1.0 and self._streak_sample >= 5:
            reasons.append(f"losing streak x{streak_scale:.2f}")
        if regime_scale != 1.0:
            reasons.append(f"regime {regime} x{regime_scale:.2f}")
        if self._drawdown.tier >= 1:
            reasons.append(self._drawdown.reason)
        if high_conv:
            reasons.append("high-conviction cap 50%")

        return RiskSnapshot(
            conviction_scale=conv_scale,
            streak_scale=streak_scale,
            regime_scale=regime_scale,
            drawdown_scale=drawdown_scale,
            total_risk_scale=total_scale,
            max_position_pct=max_pct,
            allow_new_entries=self._drawdown.allow_new_entries,
            allow_new_longs=regime_allows_new_longs(regime),
            max_open_positions=max_positions,
            high_conviction_override=high_conv,
            drawdown_tier=self._drawdown.tier,
            regime=regime,
            reasons=reasons,
        )

    # ── accessors ─────────────────────────────────────────────
    @property
    def starting_equity(self) -> float | None:
        return self._starting_equity

    @property
    def streak_scale(self) -> float:
        return self._streak_scale

    @property
    def drawdown(self) -> DrawdownTier:
        return self._drawdown

    @property
    def base_max_positions(self) -> int:
        return self._base_max_positions

    @property
    def base_max_position_pct(self) -> float:
        return self._base_max_position_pct
