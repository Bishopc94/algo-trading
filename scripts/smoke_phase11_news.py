"""Smoke test for Phase 11: News & Event Intelligence.

Tests:
  1. Event classifier — structured categorisation + sector impacts
  2. Earnings guard — detection of upcoming/just-reported earnings
  3. Economic calendar — high-impact day detection + conviction modifier
  4. News sentiment caching (TTL-based)
  5. Blended conviction modifier (keyword + classifier)
  6. Config values loaded
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, SRC)

from datetime import date

from ai_trade.sentiment.event_classifier import (
    EventType,
    classify_article,
    compute_conviction_modifier,
    aggregate_sector_impacts,
)
from ai_trade.sentiment.earnings_guard import (
    EarningsGuard,
    check_earnings_from_text,
)
from ai_trade.sentiment.economic_calendar import (
    get_events_for_date,
    get_high_impact_events,
    is_high_impact_day,
    conviction_modifier_for_events,
    get_upcoming_events,
)


def main() -> int:
    # ── Test 1: Event classifier — earnings beat ────────────
    print("== Test 1: event classifier — earnings beat ==")
    events = classify_article(
        "AAPL beats Q1 earnings expectations, raises guidance",
        "Apple reported record revenue growth and raised full-year guidance."
    )
    assert len(events) >= 1
    earnings_events = [e for e in events if e.event_type == EventType.EARNINGS]
    assert len(earnings_events) >= 1, f"Expected earnings event, got {[e.event_type for e in events]}"
    assert earnings_events[0].sentiment > 0, "Earnings beat should be bullish"
    assert earnings_events[0].magnitude > 0.3, "Earnings beat should have meaningful magnitude"
    print(f"  earnings event: sentiment={earnings_events[0].sentiment:.3f} magnitude={earnings_events[0].magnitude:.3f}")

    # ── Test 2: Event classifier — macro with sector impacts ──
    print("\n== Test 2: event classifier — macro sector impacts ==")
    events = classify_article(
        "Fed signals rate cut in September as inflation cools",
        "Federal Reserve dovish turn suggests easing cycle ahead."
    )
    macro_events = [e for e in events if e.event_type == EventType.MACRO]
    assert len(macro_events) >= 1, "Should detect macro event"
    assert macro_events[0].sentiment > 0, "Rate cut + inflation cools should be bullish"
    sector_impacts = aggregate_sector_impacts(events)
    assert "real_estate" in sector_impacts, f"Rate cut should impact real_estate, got {sector_impacts}"
    assert sector_impacts["real_estate"] > 0, "Rate cut should be bullish for real estate"
    print(f"  macro event: sentiment={macro_events[0].sentiment:.3f}")
    print(f"  sector impacts: {sector_impacts}")

    # ── Test 3: Event classifier — bearish corporate ──────────
    print("\n== Test 3: event classifier — bearish corporate ==")
    events = classify_article(
        "Company CEO resigns amid SEC investigation, fraud allegations",
        "Insider selling preceded the announcement."
    )
    corp_events = [e for e in events if e.event_type == EventType.CORPORATE]
    assert len(corp_events) >= 1
    assert corp_events[0].sentiment < 0, "Fraud + SEC investigation should be bearish"
    assert corp_events[0].magnitude >= 0.5, "Fraud should be high magnitude"
    print(f"  corporate event: sentiment={corp_events[0].sentiment:.3f} magnitude={corp_events[0].magnitude:.3f}")

    # ── Test 4: Conviction modifier from events ────────────
    print("\n== Test 4: conviction modifier from classified events ==")
    bullish_events = classify_article("AAPL beats earnings, upgraded by Goldman Sachs", "")
    modifier = compute_conviction_modifier(bullish_events)
    assert modifier > 1.0, f"Bullish events should boost conviction, got {modifier}"
    print(f"  bullish modifier: {modifier}")

    bearish_events = classify_article("Company misses earnings, downgraded to sell, SEC investigation", "")
    modifier = compute_conviction_modifier(bearish_events)
    assert modifier < 1.0, f"Bearish events should reduce conviction, got {modifier}"
    print(f"  bearish modifier: {modifier}")

    neutral_events = classify_article("Stock trades flat on light volume", "")
    modifier = compute_conviction_modifier(neutral_events)
    assert modifier == 1.0, f"Neutral should be 1.0, got {modifier}"
    print(f"  neutral modifier: {modifier}")

    # ── Test 5: Earnings guard — upcoming detection ──────────
    print("\n== Test 5: earnings guard — upcoming detection ==")
    status = check_earnings_from_text([
        "AAPL to report Q2 earnings after hours Thursday",
        "Analysts expect strong iPhone revenue",
    ])
    assert status == "upcoming", f"Expected 'upcoming', got '{status}'"
    print(f"  'to report Q2 earnings' -> {status}")

    status = check_earnings_from_text([
        "AAPL beats Q2 earnings expectations with record revenue",
    ])
    assert status == "just_reported", f"Expected 'just_reported', got '{status}'"
    print(f"  'beats Q2 earnings' -> {status}")

    status = check_earnings_from_text([
        "AAPL launches new MacBook Pro with M5 chip",
        "Apple stock trades near all-time high",
    ])
    assert status == "clear", f"Expected 'clear', got '{status}'"
    print(f"  product launch (no earnings) -> {status}")

    # ── Test 6: Earnings guard — class with cache ────────────
    print("\n== Test 6: earnings guard class + cache ==")
    guard = EarningsGuard(block_days_before=1, block_days_after=1)

    class FakeArticle:
        def __init__(self, headline):
            self.headline = headline
            self.summary = ""

    blocked, reason = guard.should_block("AAPL", [FakeArticle("AAPL set to report quarterly earnings tomorrow")])
    assert blocked is True, "Should block near earnings"
    assert "upcoming" in reason.lower() or "earnings" in reason.lower()
    print(f"  blocked: {blocked}, reason: {reason}")

    blocked, reason = guard.should_block("MSFT", [FakeArticle("MSFT announces new Azure features")])
    assert blocked is False, "No earnings, should not block"
    print(f"  not blocked: {blocked}")

    # ── Test 7: Economic calendar — FOMC day ──────────────────
    print("\n== Test 7: economic calendar — FOMC day ==")
    fomc_date = date(2026, 1, 28)
    events = get_events_for_date(fomc_date)
    assert len(events) >= 1, f"Expected FOMC event on 2026-01-28, got {events}"
    assert events[0].name == "FOMC Decision"
    assert is_high_impact_day(fomc_date) is True
    modifier = conviction_modifier_for_events(events)
    assert modifier < 1.0, f"FOMC day should reduce conviction, got {modifier}"
    print(f"  FOMC 2026-01-28: {len(events)} event(s), modifier={modifier}")

    # ── Test 8: Economic calendar — CPI day ────────────────────
    print("\n== Test 8: economic calendar — CPI day ==")
    cpi_date = date(2026, 4, 10)
    events = get_events_for_date(cpi_date)
    assert len(events) >= 1
    assert any(e.name == "CPI Release" for e in events)
    assert is_high_impact_day(cpi_date) is True
    print(f"  CPI 2026-04-10: {len(events)} event(s)")

    # ── Test 9: Economic calendar — quiet day ──────────────────
    print("\n== Test 9: economic calendar — quiet day ==")
    # A random Saturday should have no events
    quiet_date = date(2026, 4, 11)
    events = get_events_for_date(quiet_date)
    assert len(events) == 0, f"Expected no events on 2026-04-11, got {len(events)}"
    assert is_high_impact_day(quiet_date) is False
    modifier = conviction_modifier_for_events(events)
    assert modifier == 1.0, f"Quiet day should be 1.0, got {modifier}"
    print(f"  quiet day: {len(events)} events, modifier={modifier}")

    # ── Test 10: Economic calendar — upcoming events ──────────
    print("\n== Test 10: economic calendar — upcoming events ==")
    upcoming = get_upcoming_events(date(2026, 1, 25), days_ahead=7)
    assert len(upcoming) >= 1, "Should have at least FOMC on Jan 28"
    print(f"  upcoming events from 2026-01-25 (+7d): {[f'{e.name} {e.event_date}' for e in upcoming]}")

    # ── Test 11: Geopolitical classification ──────────────────
    print("\n== Test 11: geopolitical classification ==")
    events = classify_article(
        "Tensions escalation in Middle East as conflict spreads",
        "Military operations and war fears drive markets lower."
    )
    geo_events = [e for e in events if e.event_type == EventType.GEOPOLITICAL]
    assert len(geo_events) >= 1, "Should detect geopolitical event"
    assert geo_events[0].sentiment < 0, "War/conflict should be bearish"
    print(f"  geopolitical: sentiment={geo_events[0].sentiment:.3f}")

    print(f"\nSMOKE TEST PASSED (11/11)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
