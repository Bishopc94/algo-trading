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
from ai_trade.data.historical import fetch_snapshots
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

# ── Scoring weight constants ──────────────────────────────
# Momentum scan: emphasize gap size, then volume and range
_MOMENTUM_GAP_WEIGHT = 0.4
_MOMENTUM_RVOL_WEIGHT = 0.3
_MOMENTUM_ADR_WEIGHT = 0.3
_MOMENTUM_ADR_PLACEHOLDER = 10.0

# Options scan: favor liquidity, moderate gap, affordability
_OPTIONS_VOL_WEIGHT = 0.5
_OPTIONS_GAP_WEIGHT = 0.3
_OPTIONS_AFFORD_WEIGHT = 0.2
_OPTIONS_VOL_CAP = 5.0          # Cap volume score to avoid mega-cap domination
_OPTIONS_AFFORD_MIDPOINT = 200   # Stocks below this price score higher on affordability

# Mean reversion scan: deeper pullback + higher volume
_MR_PULLBACK_WEIGHT = 0.6
_MR_VOL_WEIGHT = 0.4
_MR_PULLBACK_CAP = 10.0         # Cap pullback score at 10%
_MR_VOL_NORM = 5_000_000        # Volume normalization divisor
_MR_VOL_CAP = 3.0
_MR_MAX_CANDIDATES = 15

# VWAP scan: volume, gap catalyst, relative volume
_VWAP_VOL_WEIGHT = 0.4
_VWAP_GAP_WEIGHT = 0.3
_VWAP_RVOL_WEIGHT = 0.3
_VWAP_VOL_NORM = 10_000_000
_VWAP_SCORE_CAP = 3.0
_VWAP_MAX_CANDIDATES = 10


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
        # Snapshot cache — shared between scan() and secondary scans
        # to avoid redundant API calls.  Reset each scan cycle.
        self._all_snapshots: dict = {}
        # Track which symbols have options chains (True/False/unknown).
        # Persists across scans to avoid re-querying known-empty symbols.
        self._options_chain_cache: dict[str, bool] = {}

    def invalidate_snapshot_cache(self) -> None:
        """Clear cached snapshots so the next scan fetches fresh data.

        Call this before power hour or any re-scan that needs current prices.
        """
        self._all_snapshots = {}

    # ── Universe loading ─────────────────────────────────────

    def _load_universe(self) -> list[str]:
        """Fetch all active, tradeable US equity symbols from Alpaca.

        This call typically returns ~8,000 symbols.  The result is cached
        in self._universe so subsequent scans don't repeat this API call.
        Retries up to 3 times on transient failures (network, rate limits).

        Returns:
            List of ticker symbol strings (e.g. ["AAPL", "MSFT", ...]).
        """
        if self._universe is not None:
            return self._universe

        import time as _time
        for attempt in range(1, 4):
            try:
                request = GetAssetsRequest(
                    asset_class=AssetClass.US_EQUITY,
                    status=AssetStatus.ACTIVE,
                )
                assets = get_trading_client().get_all_assets(filter=request)

                symbols = [
                    asset.symbol
                    for asset in assets
                    if asset.tradable and asset.exchange in _VALID_EXCHANGES
                ]

                if not symbols:
                    log.warning("universe_loaded_but_empty",
                                hint="Alpaca returned 0 tradeable symbols — API issue?")
                    return []

                self._universe = symbols
                log.info("universe_loaded", count=len(symbols))
                return symbols

            except Exception as e:
                if attempt < 3:
                    wait = 2 ** attempt
                    log.warning("universe_load_retry", attempt=attempt, wait=wait, error=str(e))
                    _time.sleep(wait)
                else:
                    log.exception("universe_load_failed_after_retries")
                    return []

        return []  # Unreachable, but satisfies type checker

    # ── Snapshot fetching ─────────────────────────────────────

    def _ensure_snapshots(self) -> dict:
        """Fetch and cache snapshots if not already available.

        Reuses cached snapshots from a prior scan() call when available.
        If the cache is empty, fetches fresh snapshots for the full universe.

        Returns:
            The snapshot dict (also stored in self._all_snapshots).
        """
        if self._all_snapshots:
            return self._all_snapshots

        universe = self._load_universe()
        if not universe:
            return {}

        for i in range(0, len(universe), _SNAPSHOT_BATCH_SIZE):
            batch = universe[i : i + _SNAPSHOT_BATCH_SIZE]
            try:
                snapshots = fetch_snapshots(batch)
                self._all_snapshots.update(snapshots)
            except Exception:
                log.exception("snapshot_batch_failed", batch_start=i, batch_size=len(batch))

        return self._all_snapshots

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

        # Fetch snapshots in batches (also populates the cache for secondary scans)
        all_snapshots: dict = {}
        for i in range(0, len(universe), _SNAPSHOT_BATCH_SIZE):
            batch = universe[i : i + _SNAPSHOT_BATCH_SIZE]
            try:
                snapshots = fetch_snapshots(batch)
                all_snapshots.update(snapshots)
            except Exception:
                log.exception(
                    "snapshot_batch_failed",
                    batch_start=i,
                    batch_size=len(batch),
                )

        log.info("snapshots_fetched", total=len(all_snapshots))

        # Cache snapshots so secondary scans can reuse them
        self._all_snapshots = all_snapshots

        # Track filter funnel counts for diagnostics
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
                candidate = self._evaluate_snapshot(symbol, snap)
                if candidate is None:
                    filter_counts["no_snapshot_data"] += 1
                    continue

                price = candidate["price"]
                gap_pct = candidate["gap_pct"]
                rvol = candidate["relative_volume"]

                if not (min_price <= price <= max_price):
                    filter_counts["price_out_of_range"] += 1
                    continue

                if abs(gap_pct) < min_gap_pct:
                    filter_counts["gap_too_small"] += 1
                    continue

                if rvol < min_rvol:
                    filter_counts["rvol_too_low"] += 1
                    continue

                score = (abs(gap_pct) * _MOMENTUM_GAP_WEIGHT
                         + rvol * _MOMENTUM_RVOL_WEIGHT
                         + _MOMENTUM_ADR_PLACEHOLDER * _MOMENTUM_ADR_WEIGHT)
                candidate["score"] = round(score, 4)
                candidates.append(candidate)
                filter_counts["passed"] += 1

            except Exception:
                filter_counts["eval_error"] += 1
                log.debug("symbol_evaluation_failed", symbol=symbol)

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

    # ── Options universe scan ──────────────────────────────

    def scan_options_universe(self) -> list[dict]:
        """Scan for liquid, optionable stocks — separate from the micro-cap momentum scan.

        Options require underlying stocks that are:
        - Priced $10-500 (liquid enough to have listed options)
        - High average volume (>1M, ensures option chains exist and have liquidity)

        Unlike the momentum scanner, this does NOT require gaps or unusual volume.
        Options strategies have their own entry criteria.

        Reuses snapshot data from scan() if available, otherwise fetches fresh.
        """
        self._ensure_snapshots()
        if not self._all_snapshots:
            return []

        # Read options-specific filter params from config
        opts_cfg = getattr(self._cfg, "options_universe", None)
        min_price: float = getattr(opts_cfg, "min_price", 10.0) if opts_cfg else 10.0
        max_price: float = getattr(opts_cfg, "max_price", 500.0) if opts_cfg else 500.0
        min_avg_volume: float = getattr(opts_cfg, "min_avg_volume", 1_000_000) if opts_cfg else 1_000_000
        max_candidates: int = getattr(opts_cfg, "max_candidates", 30) if opts_cfg else 30

        candidates: list[dict] = []
        for symbol, snap in self._all_snapshots.items():
            if self._options_chain_cache.get(symbol) is False:
                continue

            try:
                candidate = self._evaluate_snapshot(symbol, snap)
                if candidate is None:
                    continue

                price = candidate["price"]
                if not (min_price <= price <= max_price):
                    continue

                avg_vol = _get_avg_volume(snap)
                if avg_vol < min_avg_volume:
                    continue

                gap_score = abs(candidate["gap_pct"])
                vol_score = min(avg_vol / _VWAP_VOL_NORM, _OPTIONS_VOL_CAP)
                afford_score = max(0, (_OPTIONS_AFFORD_MIDPOINT - price) / _OPTIONS_AFFORD_MIDPOINT)
                score = (vol_score * _OPTIONS_VOL_WEIGHT
                         + gap_score * _OPTIONS_GAP_WEIGHT
                         + afford_score * _OPTIONS_AFFORD_WEIGHT)
                candidate["score"] = round(score, 4)
                candidate["avg_volume"] = avg_vol
                candidates.append(candidate)

            except Exception:
                log.debug("options_universe_eval_failed", symbol=symbol)

        candidates.sort(key=lambda c: c["score"], reverse=True)
        results = candidates[:max_candidates]

        log.info(
            "scan_options_universe_complete",
            evaluated=len(self._all_snapshots),
            passed_filters=len(candidates),
            returned=len(results),
            symbols=[c["symbol"] for c in results[:10]],
        )
        return results

    # ── Mean reversion scan ─────────────────────────────────

    def scan_mean_reversion(self) -> list[dict]:
        """Scan for oversold pullback candidates suitable for mean reversion.

        Mean reversion needs the OPPOSITE of momentum: stocks that have
        pulled back (gapped down or dropped) but still have decent volume.
        These are dip-buying opportunities in otherwise healthy stocks.

        Criteria:
        - Price $5-200 (tradeable range, slightly broader than momentum)
        - Gap DOWN at least -1% (selling pressure / pullback)
        - Average volume > 300K (enough liquidity to trade)
        """
        self._ensure_snapshots()
        if not self._all_snapshots:
            return []

        candidates: list[dict] = []
        for symbol, snap in self._all_snapshots.items():
            try:
                candidate = self._evaluate_snapshot(symbol, snap)
                if candidate is None:
                    continue

                price = candidate["price"]
                gap_pct = candidate["gap_pct"]

                if not (5.0 <= price <= 200.0):
                    continue

                avg_vol = _get_avg_volume(snap)
                if avg_vol < 300_000:
                    continue

                if gap_pct > -1.0:
                    continue

                pullback_score = min(abs(gap_pct), _MR_PULLBACK_CAP)
                vol_score = min(avg_vol / _MR_VOL_NORM, _MR_VOL_CAP)
                score = pullback_score * _MR_PULLBACK_WEIGHT + vol_score * _MR_VOL_WEIGHT
                candidate["score"] = round(score, 4)
                candidate["scan_type"] = "mean_reversion"
                candidates.append(candidate)

            except Exception:
                log.debug("mean_reversion_eval_failed", symbol=symbol)

        candidates.sort(key=lambda c: c["score"], reverse=True)
        results = candidates[:_MR_MAX_CANDIDATES]

        log.info(
            "scan_mean_reversion_complete",
            passed_filters=len(candidates),
            returned=len(results),
            symbols=[c["symbol"] for c in results[:10]],
        )
        return results

    # ── VWAP / liquid intraday scan ───────────────────────────

    def scan_vwap_universe(self) -> list[dict]:
        """Scan for liquid stocks suitable for VWAP intraday trading.

        VWAP needs stocks with reliable minute-bar data, which means
        liquid mid/large-caps with consistent intraday volume.

        Criteria:
        - Price $10-300 (liquid enough for reliable VWAP)
        - Average volume > 1M (ensures minute bars are populated)
        - Any gap direction (VWAP works on both gap-up and gap-down stocks)
        """
        self._ensure_snapshots()
        if not self._all_snapshots:
            return []

        candidates: list[dict] = []
        for symbol, snap in self._all_snapshots.items():
            try:
                candidate = self._evaluate_snapshot(symbol, snap)
                if candidate is None:
                    continue

                price = candidate["price"]
                if not (10.0 <= price <= 300.0):
                    continue

                avg_vol = _get_avg_volume(snap)
                if avg_vol < 1_000_000:
                    continue

                vol_score = min(avg_vol / _VWAP_VOL_NORM, _VWAP_SCORE_CAP)
                gap_score = abs(candidate["gap_pct"])
                rvol = candidate["relative_volume"]
                rvol_score = min(rvol, _VWAP_SCORE_CAP)
                score = (vol_score * _VWAP_VOL_WEIGHT
                         + gap_score * _VWAP_GAP_WEIGHT
                         + rvol_score * _VWAP_RVOL_WEIGHT)
                candidate["score"] = round(score, 4)
                candidate["scan_type"] = "vwap"
                candidates.append(candidate)

            except Exception:
                log.debug("vwap_eval_failed", symbol=symbol)

        candidates.sort(key=lambda c: c["score"], reverse=True)
        results = candidates[:_VWAP_MAX_CANDIDATES]

        log.info(
            "scan_vwap_universe_complete",
            passed_filters=len(candidates),
            returned=len(results),
            symbols=[c["symbol"] for c in results[:10]],
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

        Args:
            symbol: The ticker symbol.
            snap:   Alpaca snapshot object.

        Returns:
            A candidate dict or None if required data is missing.
        """
        # Get current price — prefer latest_trade (real-time) over daily_bar close
        price: float | None = None
        if hasattr(snap, "latest_trade") and snap.latest_trade is not None:
            price = float(snap.latest_trade.price)
        elif hasattr(snap, "daily_bar") and snap.daily_bar is not None:
            price = float(snap.daily_bar.close)

        if price is None or price <= 0:
            return None

        # Get previous close — required for gap calculation
        if not hasattr(snap, "previous_daily_bar") or snap.previous_daily_bar is None:
            return None
        prev_close = float(snap.previous_daily_bar.close)
        if prev_close <= 0:
            return None

        gap_pct = (price - prev_close) / prev_close * 100.0

        # Get current volume — pre-market may only have minute-bar data
        volume: float = 0.0
        if hasattr(snap, "daily_bar") and snap.daily_bar is not None:
            volume = float(snap.daily_bar.volume)
        elif hasattr(snap, "minute_bar") and snap.minute_bar is not None:
            volume = float(snap.minute_bar.volume)

        # RVOL = today's volume / yesterday's volume
        avg_volume = _get_avg_volume(snap)
        relative_volume = (volume / avg_volume) if avg_volume > 0 else 0.0

        return {
            "symbol": symbol,
            "price": round(price, 2),
            "prev_close": round(prev_close, 2),
            "gap_pct": round(gap_pct, 2),
            "relative_volume": round(relative_volume, 2),
        }


def _get_avg_volume(snap) -> float:
    """Extract average volume from a snapshot's previous daily bar.

    Used across all scan profiles to filter by liquidity.
    Returns 0.0 if the data is unavailable.
    """
    if (hasattr(snap, "previous_daily_bar")
            and snap.previous_daily_bar is not None
            and hasattr(snap.previous_daily_bar, "volume")):
        return float(snap.previous_daily_bar.volume)
    return 0.0
