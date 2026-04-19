"""Economic event calendar for macro-aware trading.

Hardcoded schedules for recurring high-impact economic events
(FOMC decisions, CPI releases, nonfarm payrolls). On event days,
the bot should widen stops, reduce conviction on new entries, or
skip trading entirely during the release window.

Why hardcode instead of an API:
    - FOMC dates are published a year in advance by the Fed.
    - CPI and jobs reports follow predictable monthly cadence.
    - No paid API dependency — zero cost, zero failure mode.
    - Updated annually (a 5-minute task when the Fed publishes
      the next year's schedule).

Event impact tiers:
    HIGH   — FOMC decision, CPI, nonfarm payrolls. Can move SPY 1-3%.
    MEDIUM — PPI, retail sales, FOMC minutes. Typically 0.5-1% SPY move.
    LOW    — housing starts, consumer confidence. Rarely market-moving.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from enum import Enum


class EventImpact(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class EconomicEvent:
    name: str
    event_date: date
    release_time: time  # Eastern Time
    impact: EventImpact
    description: str = ""


# ── 2026 FOMC Meeting Dates (2-day meetings end on these dates) ──
# Source: Federal Reserve Board — published annually.
# These are the DECISION dates (statement released at 2:00 PM ET).
_FOMC_2026 = [
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 5, 6),
    date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
    date(2026, 10, 28), date(2026, 12, 16),
]

# ── 2026 CPI Release Dates (Bureau of Labor Statistics) ──
# Typically released 2nd or 3rd Tuesday/Wednesday of the month at 8:30 AM ET.
_CPI_2026 = [
    date(2026, 1, 14), date(2026, 2, 12), date(2026, 3, 11),
    date(2026, 4, 10), date(2026, 5, 13), date(2026, 6, 10),
    date(2026, 7, 15), date(2026, 8, 12), date(2026, 9, 16),
    date(2026, 10, 14), date(2026, 11, 12), date(2026, 12, 9),
]

# ── 2026 Nonfarm Payrolls (Bureau of Labor Statistics) ──
# First Friday of each month at 8:30 AM ET.
_NFP_2026 = [
    date(2026, 1, 9), date(2026, 2, 6), date(2026, 3, 6),
    date(2026, 4, 3), date(2026, 5, 8), date(2026, 6, 5),
    date(2026, 7, 2), date(2026, 8, 7), date(2026, 9, 4),
    date(2026, 10, 2), date(2026, 11, 6), date(2026, 12, 4),
]

# ── 2026 FOMC Minutes Release Dates ──
# Released 3 weeks after the decision, at 2:00 PM ET.
_FOMC_MINUTES_2026 = [
    date(2026, 2, 18), date(2026, 4, 8), date(2026, 5, 27),
    date(2026, 7, 8), date(2026, 8, 19), date(2026, 10, 7),
    date(2026, 11, 18),
]


def _build_events() -> list[EconomicEvent]:
    events = []
    for d in _FOMC_2026:
        events.append(EconomicEvent(
            name="FOMC Decision",
            event_date=d,
            release_time=time(14, 0),
            impact=EventImpact.HIGH,
            description="Federal Reserve interest rate decision and policy statement",
        ))
    for d in _CPI_2026:
        events.append(EconomicEvent(
            name="CPI Release",
            event_date=d,
            release_time=time(8, 30),
            impact=EventImpact.HIGH,
            description="Consumer Price Index — key inflation measure",
        ))
    for d in _NFP_2026:
        events.append(EconomicEvent(
            name="Nonfarm Payrolls",
            event_date=d,
            release_time=time(8, 30),
            impact=EventImpact.HIGH,
            description="Monthly employment report — jobs added/lost",
        ))
    for d in _FOMC_MINUTES_2026:
        events.append(EconomicEvent(
            name="FOMC Minutes",
            event_date=d,
            release_time=time(14, 0),
            impact=EventImpact.MEDIUM,
            description="Detailed record of the most recent FOMC meeting",
        ))
    return sorted(events, key=lambda e: e.event_date)


_ALL_EVENTS = _build_events()


def get_events_for_date(d: date | None = None) -> list[EconomicEvent]:
    """Return all economic events scheduled for the given date."""
    if d is None:
        d = date.today()
    return [e for e in _ALL_EVENTS if e.event_date == d]


def get_high_impact_events(d: date | None = None) -> list[EconomicEvent]:
    """Return only HIGH impact events for the given date."""
    return [e for e in get_events_for_date(d) if e.impact == EventImpact.HIGH]


def is_high_impact_day(d: date | None = None) -> bool:
    """Check if the given date has any HIGH impact economic events."""
    return len(get_high_impact_events(d)) > 0


def get_upcoming_events(d: date | None = None, days_ahead: int = 5) -> list[EconomicEvent]:
    """Return events within the next N days (inclusive of today)."""
    if d is None:
        d = date.today()
    end = date.fromordinal(d.toordinal() + days_ahead)
    return [e for e in _ALL_EVENTS if d <= e.event_date <= end]


def conviction_modifier_for_events(events: list[EconomicEvent]) -> float:
    """Compute a conviction modifier based on today's economic events.

    HIGH events reduce conviction by 15% (0.85 modifier).
    MEDIUM events reduce by 5% (0.95).
    Multiple events stack multiplicatively.
    """
    modifier = 1.0
    for ev in events:
        if ev.impact == EventImpact.HIGH:
            modifier *= 0.85
        elif ev.impact == EventImpact.MEDIUM:
            modifier *= 0.95
    return round(modifier, 3)
