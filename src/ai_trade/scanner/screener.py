"""Pre-market stock screener — finds tradeable candidates each morning.

WHAT THIS MODULE DOES:
    Scans the entire universe of US equities (NYSE, NASDAQ, AMEX) each
    morning before market open to find stocks with the highest probability
    of making a large, tradeable move that day.

WHY IT EXISTS:
    There are ~8,000 stocks on US exchanges.  You can't watch them all.
    The screener narrows the field down to 10-20 candidates using objective
    criteria (price range, gap percentage, relative volume), which the
    strategy modules then evaluate for entry signals.

HOW THE SCORING WORKS:
    Each stock that passes the filters receives a composite score:
        score = |gap_pct| × 0.4 + relative_volume × 0.3 + ADR × 0.3

    Components:
    - Gap %:  How much the stock moved overnight (from yesterday's close
              to this morning's price).  Large gaps indicate a catalyst
              (earnings, news, etc.) that creates volatility and opportunity.
    - Relative Volume (RVOL): Today's volume divided by the average volume.
              RVOL > 1.5 means unusual activity — institutional interest,
              news-driven trading, or a catalyst.
    - ADR (Average Daily Range): A placeholder constant (10.0) for now;
              would measure how much the stock typically moves intraday
              as a percentage of its price.

KEY TRADING CONCEPTS:

    Gap:
        The difference between yesterday's closing price and this morning's
        opening (or pre-market) price.  Gaps happen when significant news
        arrives after market close or before market open:
        - Gap UP:   stock opens higher than yesterday's close (bullish catalyst)
        - Gap DOWN: stock opens lower than yesterday's close (bearish catalyst)

    Relative Volume (RVOL):
        Today's volume ÷ average daily volume.  RVOL = 2.0 means twice as
        many shares are trading as usual.  High RVOL confirms that the
        price move has real participation behind it (not just a thin-market
        fluke).

    Pre-Market Scanning:
        Running the scan before market open (e.g. 8:00 AM ET) gives the
        bot time to analyze candidates and prepare orders before the 9:30
        AM opening bell.  Pre-market data comes from Alpaca's snapshot API.

KEY DESIGN DECISIONS:
    - The stock universe is loaded once from Alpaca and cached (self._universe).
      Re-fetching ~8,000 assets every scan is unnecessary since the list
      changes very rarely.
    - Snapshots are fetched in batches of 500 to stay within Alpaca's API
      limits.  Each batch is a single HTTP request.
    - The filter funnel is tracked with counters (price_out_of_range,
      gap_too_small, etc.) for diagnostics — helps tune filter parameters.
    - Only stocks on major exchanges (NYSE, NASDAQ, AMEX) are included.
      OTC/pink sheet stocks are excluded because they have poor liquidity
      and unreliable data.
"""

from __future__ import annotations

from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

from ai_trade.clients import get_data_client, get_trading_client
from ai_trade.data.historical import fetch_bars, fetch_snapshots
from ai_trade.data.indicators import compute_adr
from ai_trade.monitoring.logger import get_logger

log = get_logger(__name__)

# Alpaca's snapshot API accepts up to ~1000 symbols per request;
# we use a conservative batch size to stay well within limits and
# avoid request timeouts.
_SNAPSHOT_BATCH_SIZE = 500

# Only include stocks from major US exchanges.  OTC, pink sheets,
# and other venues are excluded due to poor liquidity and data quality.
_VALID_EXCHANGES = {"NYSE", "NASDAQ", "AMEX"}


class StockScreener:
    """Scans the market for high-probability trading candidates.

    The scan pipeline:
    1. Load universe: fetch all active, tradeable US equity symbols.
    2. Fetch snapshots: get current price, previous close, and volume.
    3. Apply filters: price range, minimum gap, minimum RVOL.
    4. Score and rank: composite score from gap, RVOL, and ADR.
    5. Return top N candidates sorted by score.
    """

    def __init__(self, config) -> None:
        self._cfg = config
        # Cached symbol universe.  Set to None initially; populated on
        # first call to _load_universe() and reused thereafter.
        self._universe: list[str] | None = None

    # ── Universe loading ─────────────────────────────────────

    def _load_universe(self) -> list[str]:
        """Fetch all active, tradeable US equity symbols from Alpaca.

        This call typically returns ~8,000 symbols.  The result is cached
        in self._universe so subsequent scans don't repeat this API call.

        Filtering criteria:
        - asset_class = US_EQUITY (no crypto, no options)
        - status = ACTIVE (no delisted or halted securities)
        - tradable = True (the account can actually trade this asset)
        - exchange in {NYSE, NASDAQ, AMEX} (major US exchanges only)

        Returns:
            List of ticker symbol strings (e.g. ["AAPL", "MSFT", ...]).
        """
        if self._universe is not None:
            return self._universe

        try:
            request = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE,
            )
            assets = get_trading_client().get_all_assets(filter=request)

            # PYTHON PATTERN — list comprehension with multiple conditions:
            # Creates a list of symbols from the assets that are both
            # tradable AND on a valid exchange.  This is more concise than
            # a for-loop with if-statements.
            symbols = [
                asset.symbol
                for asset in assets
                if asset.tradable and asset.exchange in _VALID_EXCHANGES
            ]

            self._universe = symbols
            log.info("universe_loaded", count=len(symbols))
            return symbols

        except Exception:
            log.exception("universe_load_failed")
            return []

    # ── Main scan ────────────────────────────────────────────

    def scan(self) -> list[dict]:
        """Run the pre-market scan and return scored candidates.

        This is the main entry point.  It loads the universe, fetches
        market snapshots, applies filters, scores the survivors, and
        returns the top candidates.

        The filter cascade works like a funnel:
            ~8,000 symbols → price filter → gap filter → RVOL filter → ~10-20 candidates

        Returns a list of dicts sorted by score descending, each containing:
        symbol, price, prev_close, gap_pct, relative_volume, score.
        """
        universe = self._load_universe()
        if not universe:
            log.warning("scan_aborted_empty_universe")
            return []

        # Read filter parameters from config with sensible defaults.
        min_price: float = getattr(self._cfg, "min_price", 2.0)
        max_price: float = getattr(self._cfg, "max_price", 50.0)
        min_gap_pct: float = getattr(self._cfg, "min_gap_pct", 2.0)
        min_rvol: float = getattr(self._cfg, "min_relative_volume", 1.5)
        max_candidates: int = getattr(self._cfg, "max_candidates", 20)

        # ── Fetch snapshots in batches ───────────────────────
        # A "snapshot" is a single point-in-time view of a stock's current
        # price, volume, and previous day's data.  We fetch these in
        # batches because Alpaca limits how many symbols can be queried
        # in a single API call.
        all_snapshots: dict = {}
        for i in range(0, len(universe), _SNAPSHOT_BATCH_SIZE):
            # PYTHON PATTERN — list slicing:
            # `universe[i : i + 500]` extracts a sub-list of up to 500
            # elements starting at index i.  This is how we batch the API calls.
            batch = universe[i : i + _SNAPSHOT_BATCH_SIZE]
            try:
                snapshots = fetch_snapshots(batch)
                # .update() merges the batch results into the master dict.
                all_snapshots.update(snapshots)
            except Exception:
                log.exception(
                    "snapshot_batch_failed",
                    batch_start=i,
                    batch_size=len(batch),
                )

        log.info("snapshots_fetched", total=len(all_snapshots))

        # ── Score each symbol ────────────────────────────────
        # Track filter funnel counts for diagnostics — helps answer
        # "why did we only get 5 candidates today?" questions.
        candidates: list[dict] = []
        filter_counts = {
            "no_snapshot_data": 0,
            "price_out_of_range": 0,
            "gap_too_small": 0,
            "rvol_too_low": 0,
            "eval_error": 0,
            "passed": 0,
        }

        for symbol, snap in all_snapshots.items():
            try:
                # Extract price/volume data from the snapshot object.
                candidate = self._evaluate_snapshot(symbol, snap)
                if candidate is None:
                    filter_counts["no_snapshot_data"] += 1
                    continue

                price = candidate["price"]
                gap_pct = candidate["gap_pct"]
                rvol = candidate["relative_volume"]

                # ── Apply filter cascade ─────────────────────
                # Each filter removes stocks that don't meet our criteria.
                # The order matters for performance (cheapest checks first).

                # Price filter: avoid penny stocks (< $2) and expensive
                # stocks (> $50) that require too much capital per share.
                if not (min_price <= price <= max_price):
                    filter_counts["price_out_of_range"] += 1
                    continue

                # Gap filter: ignore stocks that didn't move overnight.
                # abs() handles both gap-up and gap-down.
                if abs(gap_pct) < min_gap_pct:
                    filter_counts["gap_too_small"] += 1
                    continue

                # Relative volume filter: ignore low-activity stocks.
                if rvol < min_rvol:
                    filter_counts["rvol_too_low"] += 1
                    continue

                # ── Composite score ──────────────────────────
                # Weighted sum of the three components.  The weights
                # (0.4, 0.3, 0.3) emphasize gap size slightly more than
                # volume and range.  The ADR component uses a placeholder
                # value of 10.0 for now.
                score = abs(gap_pct) * 0.4 + rvol * 0.3 + 10.0 * 0.3
                candidate["score"] = round(score, 4)
                candidates.append(candidate)
                filter_counts["passed"] += 1

            except Exception:
                filter_counts["eval_error"] += 1
                log.debug("symbol_evaluation_failed", symbol=symbol)

        # Sort by score descending (highest score = best candidate)
        # and take the top N results.
        # PYTHON PATTERN — key=lambda:
        # `lambda c: c["score"]` is an anonymous function that extracts
        # the "score" value from a dict.  It tells sorted() what to sort by.
        candidates.sort(key=lambda c: c["score"], reverse=True)
        results = candidates[:max_candidates]

        log.info(
            "scan_complete",
            evaluated=len(all_snapshots),
            passed_filters=len(candidates),
            returned=len(results),
            **filter_counts,
        )
        return results

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _evaluate_snapshot(symbol: str, snap) -> dict | None:
        """Extract pricing and volume data from a single snapshot.

        Alpaca snapshots contain several nested objects:
        - latest_trade: the most recent trade (price, timestamp)
        - daily_bar: today's OHLCV bar so far (open, high, low, close, volume)
        - previous_daily_bar: yesterday's OHLCV bar
        - minute_bar: the most recent 1-minute bar

        We prefer latest_trade for the current price (most accurate),
        falling back to daily_bar close if latest_trade isn't available.

        PYTHON PATTERN — @staticmethod:
            Like in PDTManager, this is a plain function that doesn't need
            `self`.  It's placed inside the class for organizational clarity.

        PYTHON PATTERN — hasattr():
            `hasattr(obj, "attr")` checks if an object has a particular
            attribute.  We use this because the snapshot object's structure
            varies depending on market state and data availability.

        Args:
            symbol: The ticker symbol.
            snap:   Alpaca snapshot object.

        Returns:
            A candidate dict or None if required data is missing.
        """
        # ── Get current price ────────────────────────────────
        # Prefer latest_trade (real-time) over daily_bar close (lagged).
        price: float | None = None
        if hasattr(snap, "latest_trade") and snap.latest_trade is not None:
            price = float(snap.latest_trade.price)
        elif hasattr(snap, "daily_bar") and snap.daily_bar is not None:
            price = float(snap.daily_bar.close)

        if price is None or price <= 0:
            return None

        # ── Get previous close ───────────────────────────────
        # Required to calculate the gap percentage.
        if not hasattr(snap, "previous_daily_bar") or snap.previous_daily_bar is None:
            return None
        prev_close = float(snap.previous_daily_bar.close)
        if prev_close <= 0:
            return None

        # Gap %: how much the stock moved from yesterday's close to now.
        # Positive = gap up, negative = gap down.
        gap_pct = (price - prev_close) / prev_close * 100.0

        # ── Get current volume ───────────────────────────────
        # Volume so far today.  Pre-market may only have minute-bar data.
        volume: float = 0.0
        if hasattr(snap, "daily_bar") and snap.daily_bar is not None:
            volume = float(snap.daily_bar.volume)
        elif hasattr(snap, "minute_bar") and snap.minute_bar is not None:
            volume = float(snap.minute_bar.volume)

        # ── Calculate relative volume ────────────────────────
        # RVOL = today's volume / yesterday's volume.
        # This is an approximation — ideally we'd use a multi-day average,
        # but the previous daily bar's volume is a reasonable proxy.
        avg_volume: float = 0.0
        if hasattr(snap.previous_daily_bar, "volume"):
            avg_volume = float(snap.previous_daily_bar.volume)

        # Avoid division by zero.
        relative_volume = (volume / avg_volume) if avg_volume > 0 else 0.0

        return {
            "symbol": symbol,
            "price": round(price, 2),
            "prev_close": round(prev_close, 2),
            "gap_pct": round(gap_pct, 2),
            "relative_volume": round(relative_volume, 2),
        }
