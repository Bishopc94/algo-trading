"""Earnings proximity detection and trade gating.

Detects whether a symbol has upcoming or just-released earnings
by scanning news article headlines for earnings-related phrases.
When earnings are detected, the guard recommends blocking or
reducing conviction on new entries.

Why block near earnings:
    - Binary event risk: stock can gap 5-20% either way overnight.
    - IV crush on options after the report makes pre-earnings
      options entries unprofitable even if direction is correct.
    - The bot's technical signals are noise before a binary event.

Detection strategy:
    Uses news heuristic (not a paid calendar API). Alpaca's news
    feed reliably publishes "X reports earnings", "X to report Q1
    results" articles 1-3 days before the event. The guard scans
    for these phrases and caches the result per symbol per day.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

from ai_trade.monitoring.logger import get_logger

log = get_logger(__name__)

_ET = ZoneInfo("America/New_York")

# Phrases indicating upcoming or just-released earnings.
# Checked against lowercased headline+summary.
_UPCOMING_EARNINGS_PATTERNS = [
    re.compile(r"\breports?\s+(?:q[1-4]|quarterly|annual|fourth.quarter|first.quarter|second.quarter|third.quarter)\s+(?:results?|earnings?)", re.IGNORECASE),
    re.compile(r"\b(?:q[1-4]|quarterly)\s+earnings?\s+(?:preview|ahead|expected|due|report)", re.IGNORECASE),
    re.compile(r"\bearnings?\s+(?:call|report|release|announcement|preview|ahead|tomorrow|tonight|after.?hours?|before.?(?:the.)?(?:bell|open|market))", re.IGNORECASE),
    re.compile(r"\b(?:to report|set to report|scheduled to report|will report|expected to report)", re.IGNORECASE),
    re.compile(r"\b(?:ahead of|before)\s+earnings?", re.IGNORECASE),
    re.compile(r"\bearnings?\s+(?:on|this)\s+(?:monday|tuesday|wednesday|thursday|friday)", re.IGNORECASE),
]

_JUST_REPORTED_PATTERNS = [
    re.compile(r"\b(?:beats?|misses?|tops?|falls? short)\s+(?:q[1-4]|quarterly|earnings?|estimates?|expectations?)", re.IGNORECASE),
    re.compile(r"\b(?:q[1-4]|quarterly)\s+(?:results?|earnings?)\s+(?:beat|miss|top|exceed|disappoint)", re.IGNORECASE),
    re.compile(r"\b(?:reports?|posted?|delivered?)\s+(?:strong|weak|mixed|record|solid|disappointing)\s+(?:q[1-4]|quarterly|earnings?)", re.IGNORECASE),
    re.compile(r"\bearnings?\s+(?:surprise|shock|beat|miss)", re.IGNORECASE),
    re.compile(r"\b(?:eps|revenue)\s+(?:beats?|misses?|tops?)", re.IGNORECASE),
]


class EarningsGuard:
    """Detects earnings proximity and recommends trade gating."""

    def __init__(self, block_days_before: int = 1, block_days_after: int = 1):
        self._block_before = block_days_before
        self._block_after = block_days_after
        self._cache: dict[str, tuple[date, str]] = {}  # symbol → (date_checked, status)

    def check_articles(
        self,
        symbol: str,
        articles: list,
    ) -> str:
        """Check if articles indicate earnings proximity.

        Returns:
            "upcoming" — earnings appear imminent (within block_days_before)
            "just_reported" — earnings just released (within block_days_after)
            "clear" — no earnings detected
        """
        today = datetime.now(_ET).date()
        cached = self._cache.get(symbol)
        if cached and cached[0] == today:
            return cached[1]

        status = "clear"

        for article in articles:
            headline = getattr(article, "headline", "") or ""
            summary = getattr(article, "summary", "") or ""
            text = f"{headline} {summary}"

            for pat in _UPCOMING_EARNINGS_PATTERNS:
                if pat.search(text):
                    status = "upcoming"
                    break

            if status == "clear":
                for pat in _JUST_REPORTED_PATTERNS:
                    if pat.search(text):
                        status = "just_reported"
                        break

            if status != "clear":
                break

        self._cache[symbol] = (today, status)

        if status != "clear":
            log.info(
                "earnings_detected",
                symbol=symbol,
                status=status,
            )

        return status

    def should_block(self, symbol: str, articles: list) -> tuple[bool, str]:
        """Check if a trade should be blocked due to earnings.

        Returns:
            (should_block, reason) tuple.
        """
        status = self.check_articles(symbol, articles)

        if status == "upcoming":
            return True, f"{symbol} has upcoming earnings — binary event risk"
        elif status == "just_reported":
            return True, f"{symbol} just reported earnings — post-event volatility"

        return False, ""

    def clear_cache(self) -> None:
        self._cache.clear()


def check_earnings_from_text(headlines: list[str]) -> str:
    """Standalone check for earnings proximity from headline strings.

    Useful for testing and for callers that already have extracted headlines.
    Returns "upcoming", "just_reported", or "clear".
    """
    for headline in headlines:
        for pat in _UPCOMING_EARNINGS_PATTERNS:
            if pat.search(headline):
                return "upcoming"
        for pat in _JUST_REPORTED_PATTERNS:
            if pat.search(headline):
                return "just_reported"
    return "clear"
