"""News sentiment analysis using Alpaca's News API.

WHAT THIS MODULE DOES:
    Fetches recent news articles for candidate stocks, scores each article's
    sentiment using keyword matching, and produces a conviction modifier
    that boosts or penalizes the signal strength for each symbol.

WHY IT EXISTS:
    Price patterns alone don't tell the whole story.  A stock gapping up
    due to a CEO departure is very different from one gapping up on a
    strong earnings beat.  This module provides a quick, deterministic
    "sanity check" on the news behind a price move.

HOW THE SCORING WORKS:
    1. For each article, scan the headline and summary text for keywords.
    2. Each keyword match adds its weight to the bullish or bearish score.
    3. Apply a recency weight: articles from 1 hour ago score higher than
       articles from 20 hours ago (news gets "stale").
    4. Normalize scores to a net score between -1.0 (strongly bearish)
       and +1.0 (strongly bullish).
    5. Map the net score to a conviction modifier (0.5 to 1.3) that
       scales the trading strategy's base conviction.
    6. If a "catalyst" is detected (very high aggregate score), apply
       an extra boost or penalty.

KEY DESIGN DECISIONS:
    - Uses simple keyword matching instead of ML/NLP.  This is intentional:
      keyword scoring is fast, deterministic, has no model dependencies,
      and is easy to debug and tune.
    - Keywords are weighted: "raised guidance" (2.5) matters more than
      "launches" (0.5).
    - Recency weighting uses a linear decay from 1.0 (just published) to
      0.3 (at the edge of the lookback window).
    - The fallback chain (data client → trading client) ensures we get
      news data even if the primary API is unavailable.

Alpaca provides free news data through their API — no external
news service required.
"""

from __future__ import annotations

import re

# PYTHON PATTERN — @dataclass with field():
# `dataclass` auto-generates __init__ from type annotations.
# `field(default_factory=list)` tells the dataclass to create a NEW empty
# list for each instance.  Without this, all instances would share the
# SAME list object (a common Python gotcha with mutable default arguments).
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from ai_trade.clients import get_news_client
from ai_trade.monitoring.logger import get_logger

log = get_logger(__name__)

# Eastern Time — US market timezone.
ET = ZoneInfo("America/New_York")

# ── Keyword dictionaries ───────────────────────────────────
# Each keyword maps to a numeric weight.  Higher weight = stronger signal.
# Keywords are checked against lowercased article text, so they should
# all be lowercase here.
#
# The weights are calibrated empirically:
#   0.5 = weak signal (common word, could be noise)
#   1.0 = moderate signal
#   1.5 = strong signal
#   2.0 = very strong signal (clear analyst/earnings event)
#   2.5-3.0 = extreme signal (fraud, FDA approval, guidance change)

_BULLISH_KEYWORDS: dict[str, float] = {
    # Earnings / financials — the most actionable catalysts
    "beats": 1.5, "beat expectations": 2.0, "record revenue": 2.0,
    "revenue growth": 1.5, "profit growth": 1.5, "earnings beat": 2.0,
    "raised guidance": 2.5, "raises guidance": 2.5, "raises outlook": 2.0,
    "strong earnings": 1.5, "exceeded expectations": 2.0,
    "record profit": 2.0, "margin expansion": 1.5,
    # Analyst / institutional activity
    "upgraded": 2.0, "upgrade": 1.5, "price target raised": 2.0,
    "buy rating": 1.5, "outperform": 1.5, "overweight": 1.0,
    "initiated coverage": 1.0, "bullish": 1.0,
    # Corporate actions
    "buyback": 1.5, "share repurchase": 1.5, "dividend increase": 1.5,
    "dividend hike": 1.5, "stock split": 1.0, "insider buying": 2.0,
    # Product / business developments
    "new product": 1.0, "partnership": 1.0, "contract win": 1.5,
    "fda approval": 2.5, "approval": 1.0, "expansion": 1.0,
    "breakthrough": 1.5, "launches": 0.5, "innovation": 0.5,
    # Price momentum words (weaker — these describe the move, not the cause)
    "all-time high": 1.5, "52-week high": 1.0, "rally": 0.5,
    "surge": 1.0, "soars": 1.5, "jumps": 1.0, "spikes": 0.5,
}

_BEARISH_KEYWORDS: dict[str, float] = {
    # Earnings / financials
    "misses": 1.5, "missed expectations": 2.0, "revenue decline": 1.5,
    "earnings miss": 2.0, "lowered guidance": 2.5, "lowers guidance": 2.5,
    "cuts guidance": 2.5, "cuts outlook": 2.0, "weak earnings": 1.5,
    "profit warning": 2.5, "disappointing": 1.5, "margin compression": 1.5,
    "revenue miss": 2.0,
    # Analyst / institutional activity
    "downgraded": 2.0, "downgrade": 1.5, "price target cut": 2.0,
    "price target lowered": 2.0, "sell rating": 1.5, "underperform": 1.5,
    "underweight": 1.0, "bearish": 1.0,
    # Corporate actions — negative events
    "layoffs": 1.5, "restructuring": 1.0, "dividend cut": 2.0,
    "insider selling": 2.0, "ceo departure": 1.5, "ceo resigns": 1.5,
    "delisting": 3.0, "sec investigation": 2.5, "lawsuit": 1.0,
    "fraud": 3.0, "restatement": 2.5,
    # Product / business problems
    "recall": 2.0, "fda rejection": 2.5, "failed trial": 2.5,
    "loses contract": 1.5, "supply chain": 0.5,
    # Price momentum words (bearish)
    "52-week low": 1.0, "crash": 2.0, "plunges": 2.0,
    "tumbles": 1.5, "drops": 0.5, "selloff": 1.0, "sell-off": 1.0,
    "sinks": 1.0, "tanks": 1.5,
}


@dataclass
class NewsSentiment:
    """Sentiment analysis result for a single symbol.

    Contains both the raw scores and the derived conviction modifier
    that strategies use to adjust their signal strength.
    """

    symbol: str
    article_count: int          # Number of articles analyzed
    bullish_score: float        # Sum of bullish keyword weights (after recency)
    bearish_score: float        # Sum of bearish keyword weights (after recency)
    net_score: float            # bullish - bearish, normalized to [-1.0, +1.0]
    conviction_modifier: float  # Multiplier for strategy conviction (e.g. 0.7 or 1.2)
    catalyst_detected: bool     # True if a strong news event was found
    top_headline: str = ""      # Most recent headline for display
    # PYTHON PATTERN — field(default_factory=list):
    # Mutable defaults (like lists or dicts) must use default_factory to
    # avoid the "shared mutable default" bug.  Each NewsSentiment instance
    # gets its own independent list.
    headlines: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Human-readable summary string."""
        direction = "bullish" if self.net_score > 0 else "bearish" if self.net_score < 0 else "neutral"
        return (
            f"{self.symbol}: {direction} ({self.net_score:+.2f}) | "
            f"{self.article_count} articles | "
            f"modifier: {self.conviction_modifier:.2f}"
        )


class NewsSentimentScanner:
    """Scans Alpaca news for sentiment signals on candidate stocks.

    Uses keyword-based sentiment scoring — no ML models needed.
    Fast, deterministic, and runs without external API keys.

    Typical usage:
        >>> scanner = NewsSentimentScanner(lookback_hours=24)
        >>> result = scanner.scan_symbol("AAPL")
        >>> print(result.conviction_modifier)  # e.g. 1.15
    """

    def __init__(self, lookback_hours: int = 24, max_articles: int = 10):
        # How far back to search for news articles.
        self._lookback_hours = lookback_hours
        # Maximum number of articles to fetch per symbol (API limit control).
        self._max_articles = max_articles

    def scan_symbol(self, symbol: str) -> NewsSentiment:
        """Analyze recent news sentiment for a single symbol.

        Scoring pipeline:
        1. Fetch recent articles from Alpaca's news API.
        2. For each article, score bullish and bearish keywords.
        3. Apply recency weighting (newer articles score higher).
        4. Normalize to net_score in [-1.0, +1.0].
        5. Detect catalysts (unusually strong signal).
        6. Calculate conviction modifier.

        Args:
            symbol: Stock ticker (e.g. "AAPL").

        Returns:
            NewsSentiment with scores and conviction modifier.
        """
        articles = self._fetch_news(symbol)

        # If no articles found, return neutral sentiment (modifier = 1.0,
        # meaning "no change to conviction").
        if not articles:
            return NewsSentiment(
                symbol=symbol,
                article_count=0,
                bullish_score=0.0,
                bearish_score=0.0,
                net_score=0.0,
                conviction_modifier=1.0,
                catalyst_detected=False,
            )

        total_bull = 0.0
        total_bear = 0.0
        headlines = []

        for article in articles:
            # getattr with fallback handles Alpaca API objects that might
            # have missing attributes depending on the SDK version.
            headline = getattr(article, "headline", "") or ""
            summary = getattr(article, "summary", "") or ""

            # Combine headline + summary and lowercase for keyword matching.
            text = f"{headline} {summary}".lower()
            headlines.append(headline)

            # ── Score this article's bullish keywords ────────
            # PYTHON PATTERN — generator expression with sum():
            # `sum(weight for keyword, weight in dict.items() if condition)`
            # is a concise way to iterate through a dictionary and sum the
            # values that match a condition.  The `for k, v in dict.items()`
            # unpacks each key-value pair.
            bull_score = sum(
                weight for keyword, weight in _BULLISH_KEYWORDS.items()
                if keyword in text
            )
            # ── Score this article's bearish keywords ────────
            bear_score = sum(
                weight for keyword, weight in _BEARISH_KEYWORDS.items()
                if keyword in text
            )

            # ── Recency weighting ────────────────────────────
            # More recent articles should influence the score more than
            # older ones.  A linear decay from 1.0 (just published) to
            # 0.3 (at 2× the lookback window) is applied.
            created = getattr(article, "created_at", None)
            if created:
                try:
                    # hasattr check: the created_at value might be a
                    # datetime object (with .timestamp) or a plain string.
                    if hasattr(created, "timestamp"):
                        age_hours = (datetime.now(ET) - created).total_seconds() / 3600
                    else:
                        age_hours = self._lookback_hours
                    # Linear decay: 1.0 at age_hours=0, 0.3 at 2×lookback.
                    recency_weight = max(0.3, 1.0 - (age_hours / (self._lookback_hours * 2)))
                except Exception:
                    recency_weight = 0.5  # Safe default on parsing errors
            else:
                recency_weight = 0.5  # No timestamp → use middle weight

            # Apply recency weighting to this article's scores.
            total_bull += bull_score * recency_weight
            total_bear += bear_score * recency_weight

        # ── Normalize scores to [-1.0, +1.0] ────────────────
        # Divide each side by the total magnitude so the net score is
        # bounded.  max(..., 1.0) prevents division by zero when there
        # are no keyword matches.
        max_possible = max(total_bull + total_bear, 1.0)
        norm_bull = total_bull / max_possible
        norm_bear = total_bear / max_possible
        # Clamp to [-1.0, +1.0] as a safety measure.
        net_score = max(-1.0, min(1.0, norm_bull - norm_bear))

        # ── Detect strong catalyst ──────────────────────────
        # If the raw (pre-normalization) score exceeds 4.0 on either side,
        # that indicates a significant news event (e.g. an earnings beat
        # with analyst upgrade = ~4.5 points).
        catalyst_detected = (total_bull > 4.0 or total_bear > 4.0)

        # ── Calculate conviction modifier ────────────────────
        # The modifier adjusts the strategy's base conviction score:
        # - Positive news (net > +0.3): boost conviction up to +30%.
        # - Negative news (net < -0.3): reduce conviction up to -50%.
        # - Neutral (between -0.3 and +0.3): no change (modifier = 1.0).
        #
        # The dead zone [-0.3, +0.3] prevents weak/noisy signals from
        # affecting trade decisions.
        if net_score > 0.3:
            conviction_modifier = 1.0 + min(net_score * 0.5, 0.3)  # Cap at +30%
        elif net_score < -0.3:
            conviction_modifier = 1.0 + max(net_score * 0.5, -0.5)  # Floor at -50%
        else:
            conviction_modifier = 1.0  # Neutral — no change

        # Strong catalysts get an extra multiplier on top.
        if catalyst_detected:
            if net_score > 0:
                conviction_modifier = min(conviction_modifier * 1.2, 1.5)  # Cap total at +50%
            else:
                conviction_modifier = max(conviction_modifier * 0.8, 0.3)  # Floor total at 0.3

        result = NewsSentiment(
            symbol=symbol,
            article_count=len(articles),
            bullish_score=round(total_bull, 2),
            bearish_score=round(total_bear, 2),
            net_score=round(net_score, 3),
            conviction_modifier=round(conviction_modifier, 3),
            catalyst_detected=catalyst_detected,
            top_headline=headlines[0] if headlines else "",
            headlines=headlines[:5],  # Keep only top 5 headlines
        )

        log.debug(
            "news_sentiment",
            symbol=symbol,
            articles=len(articles),
            net_score=result.net_score,
            modifier=result.conviction_modifier,
            catalyst=result.catalyst_detected,
        )

        return result

    def scan_symbols(self, symbols: list[str]) -> dict[str, NewsSentiment]:
        """Analyze news sentiment for multiple symbols.

        Iterates through the list sequentially.  If any single symbol's
        scan fails (e.g. API error), it returns neutral sentiment for
        that symbol and continues with the rest.

        Args:
            symbols: List of ticker symbols to scan.

        Returns:
            Dictionary mapping each symbol to its NewsSentiment result.
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.scan_symbol(symbol)
            except Exception as e:
                log.warning("news_scan_failed", symbol=symbol, error=str(e))
                # Return neutral sentiment on failure — don't let a news
                # API error block the entire trading pipeline.
                results[symbol] = NewsSentiment(
                    symbol=symbol, article_count=0,
                    bullish_score=0, bearish_score=0, net_score=0,
                    conviction_modifier=1.0, catalyst_detected=False,
                )
        return results

    def _fetch_news(self, symbol: str) -> list:
        """Fetch recent news articles from Alpaca's NewsClient.

        Uses the dedicated NewsClient (not StockHistoricalDataClient) —
        news lives on its own API endpoint in alpaca-py.
        """
        try:
            from alpaca.data import NewsRequest

            start = datetime.now(ET) - timedelta(hours=self._lookback_hours)
            request = NewsRequest(
                symbols=symbol,
                start=start,
                limit=self._max_articles,
                sort="desc",  # Most recent first
            )
            client = get_news_client()
            result = client.get_news(request)
            # NewsSet is dict-like: result["news"] contains the list of News objects.
            if hasattr(result, "__getitem__"):
                articles = result.get("news") if hasattr(result, "get") else result["news"]
                return list(articles) if articles else []
            return list(result) if result else []
        except Exception as e:
            log.debug("news_fetch_error", symbol=symbol, error=str(e))
            return []
