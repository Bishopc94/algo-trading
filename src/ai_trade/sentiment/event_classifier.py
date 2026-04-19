"""Structured event classification for news articles.

Replaces flat keyword matching with category-aware scoring.
Each article is classified into one or more event types, scored
by sentiment and magnitude, and mapped to affected sectors.

Event categories:
    EARNINGS    — quarterly results, guidance changes, revenue beats/misses
    ANALYST     — upgrades, downgrades, price target changes, coverage initiation
    MACRO       — Fed decisions, CPI, jobs, tariffs, trade policy
    CORPORATE   — M&A, layoffs, buybacks, insider activity, leadership changes
    PRODUCT     — launches, approvals, recalls, trials, contracts
    GEOPOLITICAL — war, sanctions, diplomacy, elections

Sector impact mapping:
    Macro events propagate to specific sectors (e.g., "rate cut" → REITs up,
    growth up, banks mixed).  The classifier returns sector-specific modifiers
    so downstream can apply targeted conviction adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EventType(str, Enum):
    EARNINGS = "earnings"
    ANALYST = "analyst"
    MACRO = "macro"
    CORPORATE = "corporate"
    PRODUCT = "product"
    GEOPOLITICAL = "geopolitical"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SectorImpact:
    sector: str
    direction: float  # -1.0 to +1.0


@dataclass
class ClassifiedEvent:
    event_type: EventType
    sentiment: float  # -1.0 (strongly bearish) to +1.0 (strongly bullish)
    magnitude: float  # 0.0 (noise) to 1.0 (extreme)
    headline: str = ""
    sector_impacts: list[SectorImpact] = field(default_factory=list)


# ── Keyword groups by event type ─────────────────────────────
# Each entry: keyword → (sentiment_direction, magnitude)
# sentiment_direction: +1 bullish, -1 bearish
# magnitude: 0.3 weak, 0.5 moderate, 0.7 strong, 1.0 extreme

_EARNINGS_KEYWORDS: dict[str, tuple[float, float]] = {
    "beats": (+1, 0.6), "beat expectations": (+1, 0.7),
    "earnings beat": (+1, 0.7), "revenue beat": (+1, 0.6),
    "record revenue": (+1, 0.7), "record profit": (+1, 0.7),
    "strong earnings": (+1, 0.6), "exceeded expectations": (+1, 0.7),
    "revenue growth": (+1, 0.5), "profit growth": (+1, 0.5),
    "raised guidance": (+1, 0.8), "raises guidance": (+1, 0.8),
    "raises outlook": (+1, 0.7), "margin expansion": (+1, 0.5),
    "misses": (-1, 0.6), "missed expectations": (-1, 0.7),
    "earnings miss": (-1, 0.7), "revenue miss": (-1, 0.6),
    "revenue decline": (-1, 0.6), "weak earnings": (-1, 0.6),
    "lowered guidance": (-1, 0.8), "lowers guidance": (-1, 0.8),
    "cuts guidance": (-1, 0.8), "cuts outlook": (-1, 0.7),
    "profit warning": (-1, 0.9), "disappointing": (-1, 0.5),
    "margin compression": (-1, 0.5),
}

_ANALYST_KEYWORDS: dict[str, tuple[float, float]] = {
    "upgraded": (+1, 0.6), "upgrade": (+1, 0.5),
    "price target raised": (+1, 0.6), "buy rating": (+1, 0.5),
    "outperform": (+1, 0.5), "overweight": (+1, 0.4),
    "initiated coverage": (+1, 0.3), "bullish": (+1, 0.4),
    "downgraded": (-1, 0.6), "downgrade": (-1, 0.5),
    "price target cut": (-1, 0.6), "price target lowered": (-1, 0.6),
    "sell rating": (-1, 0.5), "underperform": (-1, 0.5),
    "underweight": (-1, 0.4), "bearish": (-1, 0.4),
}

_MACRO_KEYWORDS: dict[str, tuple[float, float]] = {
    "rate cut": (+1, 0.7), "rate cuts": (+1, 0.7),
    "dovish": (+1, 0.5), "easing": (+1, 0.5),
    "stimulus": (+1, 0.6), "quantitative easing": (+1, 0.7),
    "rate hike": (-1, 0.7), "rate hikes": (-1, 0.7),
    "hawkish": (-1, 0.5), "tightening": (-1, 0.5),
    "inflation rises": (-1, 0.5), "inflation higher": (-1, 0.5),
    "inflation cools": (+1, 0.5), "inflation lower": (+1, 0.5),
    "strong jobs": (+1, 0.4), "jobs beat": (+1, 0.5),
    "weak jobs": (-1, 0.4), "jobs miss": (-1, 0.5),
    "unemployment rises": (-1, 0.5), "unemployment falls": (+1, 0.5),
    "tariff": (-1, 0.6), "tariffs": (-1, 0.6),
    "trade war": (-1, 0.7), "trade deal": (+1, 0.6),
    "sanctions": (-1, 0.5), "recession": (-1, 0.7),
    "soft landing": (+1, 0.5), "gdp growth": (+1, 0.4),
    "gdp contraction": (-1, 0.6),
}

_CORPORATE_KEYWORDS: dict[str, tuple[float, float]] = {
    "buyback": (+1, 0.5), "share repurchase": (+1, 0.5),
    "dividend increase": (+1, 0.5), "dividend hike": (+1, 0.5),
    "stock split": (+1, 0.3), "insider buying": (+1, 0.6),
    "acquisition": (+1, 0.4), "merger": (+1, 0.4),
    "partnership": (+1, 0.3),
    "layoffs": (-1, 0.5), "restructuring": (-1, 0.4),
    "dividend cut": (-1, 0.6), "insider selling": (-1, 0.6),
    "ceo departure": (-1, 0.5), "ceo resigns": (-1, 0.5),
    "cfo resigns": (-1, 0.5), "delisting": (-1, 0.9),
    "sec investigation": (-1, 0.8), "lawsuit": (-1, 0.4),
    "fraud": (-1, 1.0), "restatement": (-1, 0.8),
    "bankruptcy": (-1, 1.0),
}

_PRODUCT_KEYWORDS: dict[str, tuple[float, float]] = {
    "new product": (+1, 0.4), "launches": (+1, 0.3),
    "fda approval": (+1, 0.8), "approval": (+1, 0.4),
    "breakthrough": (+1, 0.5), "innovation": (+1, 0.3),
    "contract win": (+1, 0.5), "expansion": (+1, 0.3),
    "recall": (-1, 0.6), "fda rejection": (-1, 0.8),
    "failed trial": (-1, 0.8), "loses contract": (-1, 0.5),
    "supply chain": (-1, 0.3),
}

_GEOPOLITICAL_KEYWORDS: dict[str, tuple[float, float]] = {
    "war": (-1, 0.7), "conflict": (-1, 0.5),
    "invasion": (-1, 0.8), "military": (-1, 0.4),
    "peace deal": (+1, 0.6), "ceasefire": (+1, 0.5),
    "de-escalation": (+1, 0.5), "tensions ease": (+1, 0.4),
    "escalation": (-1, 0.6), "tensions rise": (-1, 0.5),
    "nuclear": (-1, 0.8), "missile": (-1, 0.6),
    "election": (0, 0.3), "coup": (-1, 0.7),
}

_ALL_CATEGORY_KEYWORDS: list[tuple[EventType, dict[str, tuple[float, float]]]] = [
    (EventType.EARNINGS, _EARNINGS_KEYWORDS),
    (EventType.ANALYST, _ANALYST_KEYWORDS),
    (EventType.MACRO, _MACRO_KEYWORDS),
    (EventType.CORPORATE, _CORPORATE_KEYWORDS),
    (EventType.PRODUCT, _PRODUCT_KEYWORDS),
    (EventType.GEOPOLITICAL, _GEOPOLITICAL_KEYWORDS),
]


# ── Macro event → sector impact mapping ──────────────────────
# When a macro keyword fires, these sectors are affected.
# direction: +1 = bullish for sector, -1 = bearish
_MACRO_SECTOR_MAP: dict[str, list[SectorImpact]] = {
    "rate cut": [
        SectorImpact("real_estate", +0.7), SectorImpact("growth", +0.6),
        SectorImpact("utilities", +0.5), SectorImpact("banks", -0.2),
    ],
    "rate cuts": [
        SectorImpact("real_estate", +0.7), SectorImpact("growth", +0.6),
        SectorImpact("utilities", +0.5), SectorImpact("banks", -0.2),
    ],
    "rate hike": [
        SectorImpact("banks", +0.4), SectorImpact("real_estate", -0.6),
        SectorImpact("growth", -0.5), SectorImpact("utilities", -0.4),
    ],
    "rate hikes": [
        SectorImpact("banks", +0.4), SectorImpact("real_estate", -0.6),
        SectorImpact("growth", -0.5), SectorImpact("utilities", -0.4),
    ],
    "tariff": [
        SectorImpact("industrials", -0.5), SectorImpact("consumer", -0.4),
        SectorImpact("tech", -0.3),
    ],
    "tariffs": [
        SectorImpact("industrials", -0.5), SectorImpact("consumer", -0.4),
        SectorImpact("tech", -0.3),
    ],
    "trade war": [
        SectorImpact("industrials", -0.6), SectorImpact("tech", -0.5),
        SectorImpact("consumer", -0.4),
    ],
    "trade deal": [
        SectorImpact("industrials", +0.5), SectorImpact("tech", +0.4),
        SectorImpact("consumer", +0.3),
    ],
    "war": [
        SectorImpact("defense", +0.6), SectorImpact("energy", +0.5),
        SectorImpact("travel", -0.6), SectorImpact("consumer", -0.3),
    ],
    "peace deal": [
        SectorImpact("defense", -0.4), SectorImpact("energy", -0.3),
        SectorImpact("travel", +0.5), SectorImpact("consumer", +0.3),
    ],
    "inflation rises": [
        SectorImpact("growth", -0.4), SectorImpact("real_estate", -0.3),
        SectorImpact("energy", +0.2),
    ],
    "inflation cools": [
        SectorImpact("growth", +0.4), SectorImpact("real_estate", +0.3),
    ],
    "recession": [
        SectorImpact("consumer", -0.6), SectorImpact("industrials", -0.5),
        SectorImpact("utilities", +0.3), SectorImpact("healthcare", +0.2),
    ],
    "stimulus": [
        SectorImpact("consumer", +0.5), SectorImpact("growth", +0.5),
        SectorImpact("industrials", +0.4),
    ],
}


def classify_article(headline: str, summary: str = "") -> list[ClassifiedEvent]:
    """Classify a news article into structured event categories.

    Returns one ClassifiedEvent per category that has keyword matches.
    If no categories match, returns a single UNKNOWN event with neutral sentiment.
    """
    text = f"{headline} {summary}".lower()
    events: list[ClassifiedEvent] = []

    for event_type, keywords in _ALL_CATEGORY_KEYWORDS:
        matches: list[tuple[float, float]] = []
        matched_keywords: list[str] = []

        for keyword, (direction, magnitude) in keywords.items():
            if keyword in text:
                matches.append((direction, magnitude))
                matched_keywords.append(keyword)

        if not matches:
            continue

        total_direction = sum(d * m for d, m in matches)
        total_magnitude = sum(m for _, m in matches)
        n = len(matches)

        sentiment = max(-1.0, min(1.0, total_direction / max(n, 1)))
        magnitude = min(1.0, total_magnitude / max(n, 1))

        sector_impacts: list[SectorImpact] = []
        if event_type == EventType.MACRO:
            for kw in matched_keywords:
                if kw in _MACRO_SECTOR_MAP:
                    sector_impacts.extend(_MACRO_SECTOR_MAP[kw])

        events.append(ClassifiedEvent(
            event_type=event_type,
            sentiment=round(sentiment, 3),
            magnitude=round(magnitude, 3),
            headline=headline,
            sector_impacts=sector_impacts,
        ))

    if not events:
        events.append(ClassifiedEvent(
            event_type=EventType.UNKNOWN,
            sentiment=0.0,
            magnitude=0.0,
            headline=headline,
        ))

    return events


def compute_conviction_modifier(
    events: list[ClassifiedEvent],
    recency_weight: float = 1.0,
) -> float:
    """Compute a conviction modifier from classified events.

    Weights event categories differently:
    - EARNINGS: highest weight (most direct impact on stock)
    - ANALYST: high weight
    - CORPORATE: moderate weight
    - MACRO/GEOPOLITICAL: lower per-article weight (broad market, less stock-specific)
    - PRODUCT: moderate weight
    """
    if not events:
        return 1.0

    category_weights = {
        EventType.EARNINGS: 1.5,
        EventType.ANALYST: 1.2,
        EventType.CORPORATE: 1.0,
        EventType.PRODUCT: 0.8,
        EventType.MACRO: 0.6,
        EventType.GEOPOLITICAL: 0.5,
        EventType.UNKNOWN: 0.0,
    }

    weighted_sentiment = 0.0
    total_weight = 0.0

    for ev in events:
        w = category_weights.get(ev.event_type, 0.5) * ev.magnitude * recency_weight
        weighted_sentiment += ev.sentiment * w
        total_weight += w

    if total_weight == 0:
        return 1.0

    net = weighted_sentiment / total_weight

    if net > 0.2:
        modifier = 1.0 + min(net * 0.4, 0.3)
    elif net < -0.2:
        modifier = 1.0 + max(net * 0.5, -0.5)
    else:
        modifier = 1.0

    return round(modifier, 3)


def aggregate_sector_impacts(events: list[ClassifiedEvent]) -> dict[str, float]:
    """Aggregate sector impacts across all classified events.

    Returns sector → net direction mapping. Used by the trading pipeline
    to apply sector-specific conviction adjustments.
    """
    sector_scores: dict[str, list[float]] = {}
    for ev in events:
        for si in ev.sector_impacts:
            sector_scores.setdefault(si.sector, []).append(si.direction)

    return {
        sector: round(sum(scores) / len(scores), 3)
        for sector, scores in sector_scores.items()
    }
