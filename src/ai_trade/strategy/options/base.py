"""
Options strategy interface, shared data structures, and utilities.

This module is the foundation for all options strategies in the system. It provides:

1. **OptionsStrategyType** -- An enumeration of every strategy the system supports.
2. **OptionsSignal** -- A data container (dataclass) that every strategy returns when
   it decides to enter a trade. Downstream code (order router, risk manager) reads
   this object to know *what* to trade.
3. **Shared utility functions** -- ``filter_contracts``, ``enrich_greeks``,
   ``filter_by_delta``, ``select_by_delta``. These encapsulate logic that would
   otherwise be copy-pasted in every strategy file.
4. **BaseOptionsStrategy** -- An abstract base class that every concrete strategy
   must subclass. It enforces a common ``evaluate()`` interface so the orchestrator
   can call any strategy the same way.

Options Terminology Quick Reference
------------------------------------
- **DTE** (Days To Expiration): How many calendar days remain until the option
  contract expires. More DTE = more time value = higher premium.
- **Delta**: Measures how much an option's price changes per $1 move in the
  underlying stock. A delta of 0.50 means the option gains ~$0.50 if the stock
  rises $1. Also roughly approximates the probability of expiring in-the-money.
- **Theta**: The daily rate of time-value decay. Negative for option buyers
  (you lose value each day), positive for sellers (you gain value each day).
- **IV** (Implied Volatility): The market's forecast of how much the stock will
  move. Higher IV = more expensive options premiums.
- **OTM / ATM / ITM**: Out-of-the-money, at-the-money, in-the-money. A call is
  OTM when strike > stock price; a put is OTM when strike < stock price.
- **Bid / Ask / Mid**: Bid is the best price a buyer will pay; ask is the lowest
  a seller will accept; mid is the average, used as a fair-value estimate.
"""

# "from __future__ import annotations" makes all type hints strings at parse time.
# This lets you use newer syntax like ``list[dict]`` and ``X | None`` even on
# older Python versions (3.9) that would otherwise raise a TypeError.
from __future__ import annotations

# abc = Abstract Base Class module.  ABC is the base you inherit from, and
# @abstractmethod marks methods that subclasses *must* override.
from abc import ABC, abstractmethod

# dataclass auto-generates __init__, __repr__, __eq__ from field declarations.
# field() lets you set defaults for mutable types (lists, dicts) safely.
from dataclasses import dataclass, field

from datetime import datetime, timezone

# Enum creates a fixed set of named constants (like a C/Java enum).
from enum import Enum


class OptionsStrategyType(Enum):
    """Enumeration of all supported options strategies.

    Each member's *value* is a snake_case string used in logs, signals, and
    serialized messages. Using an Enum (instead of bare strings) prevents typos
    and gives IDE auto-complete.

    Python Enum reminder: access the string with ``OptionsStrategyType.LONG_CALL.value``
    which returns ``"long_call"``.
    """

    CREDIT_PUT_SPREAD = "credit_put_spread"
    DEBIT_CALL_SPREAD = "debit_call_spread"
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    CASH_SECURED_PUT = "cash_secured_put"
    COVERED_CALL = "covered_call"
    COVERED_STRADDLE = "covered_straddle"


# ---------------------------------------------------------------------------
# OptionsSignal dataclass
# ---------------------------------------------------------------------------

# The ``@dataclass`` decorator (Python 3.7+) auto-generates an __init__ method
# from the annotated class fields below. Each field becomes a constructor
# argument. Fields *with* a default value must appear after those without one.
@dataclass
class OptionsSignal:
    """Represents a fully-formed options trade signal ready for execution.

    Every strategy's ``evaluate()`` method returns either an ``OptionsSignal``
    or ``None`` (no trade). The order router inspects this object to construct
    brokerage API calls.

    Key fields explained:
    - ``legs``: A list of dicts, one per option contract in the trade.
      Each dict has keys: ``symbol`` (OCC option symbol), ``side``
      ("buy" or "sell"), ``qty`` (integer), ``position_intent``
      ("buy_to_open", "sell_to_open", etc.).
    - ``max_cost`` / ``min_credit``: For debit strategies (you pay to enter)
      the max_cost is the most you'd pay per share. For credit strategies
      (you receive money to enter) min_credit is the least you'd accept.
    - ``max_loss`` / ``max_profit``: In dollar terms (per 1-lot, i.e. 100
      shares). These are theoretical worst/best case at expiration.
    - ``strikes``: List of strike prices involved. One element for single-leg
      strategies, two for spreads.
    - ``net_delta`` / ``net_theta``: The combined greeks across all legs.
      A spread's net_delta is the sum of its legs' deltas.
    """

    underlying: str                          # Ticker symbol, e.g. "AAPL"
    strategy_type: OptionsStrategyType       # Which strategy produced this signal
    conviction: float                        # 0.0 to 1.0 confidence score
    strategy_name: str                       # Human-readable name for logging

    # Legs -- each leg is a dict with: symbol, side (buy/sell), qty, position_intent.
    # ``field(default_factory=list)`` is required in dataclasses when the default
    # is a mutable object (list, dict). If you just wrote ``legs: list = []``,
    # every instance would *share* the same list -- a common Python gotcha.
    legs: list[dict] = field(default_factory=list)

    # Pricing fields
    max_cost: float = 0.0      # Max debit to pay (for debit strategies like long call, debit spread)
    min_credit: float = 0.0    # Min credit to receive (for credit strategies like credit spread, CSP)
    max_loss: float = 0.0      # Maximum possible dollar loss (always positive number)
    max_profit: float = 0.0    # Maximum possible dollar profit (0 = theoretically unlimited)

    # Contract details
    expiration: str = ""                           # ISO date string, e.g. "2024-07-19"
    strikes: list[float] = field(default_factory=list)  # Strike price(s) involved

    # Net greeks across all legs
    net_delta: float = 0.0     # Net directional exposure (positive = bullish, negative = bearish)
    net_theta: float = 0.0     # Net time decay (positive = benefits from time passing)

    # Catch-all for strategy-specific data (RSI, volume, etc.) used in logging/analysis.
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared utilities -- eliminate duplication across strategy files
# ---------------------------------------------------------------------------


def filter_contracts(
    chain_data: list[dict],
    contract_type: str,
    min_dte: int,
    max_dte: int,
) -> list[dict]:
    """Filter option contracts by type (call/put) and DTE range.

    This is the first step every strategy performs: narrow the full options chain
    down to only the contracts worth evaluating.

    Parameters
    ----------
    chain_data : list[dict]
        The full options chain from the broker API. Each dict represents one
        contract with keys like "type", "expiration_date", "strike_price", "symbol".
    contract_type : str
        Either "call" or "put" (case-insensitive comparison is used).
    min_dte / max_dte : int
        Desired DTE window. For example, min_dte=20, max_dte=45 selects
        contracts expiring in 20 to 45 calendar days. This balances:
        - Enough time for the thesis to play out (not too short)
        - Enough theta decay to be useful for sellers (not too long)

    Returns
    -------
    list[dict]
        A *new* list of contracts that match. Original ``chain_data`` is not
        mutated (except for caching ``_dte`` on individual dicts -- see below).

    Notes
    -----
    Uses pre-computed ``_dte`` if present (the backtest engine sets this to
    avoid live clock issues). Otherwise, computes DTE from ``expiration_date``
    relative to ``datetime.now(UTC)``.
    """
    now = datetime.now(tz=timezone.utc)
    eligible: list[dict] = []

    for contract in chain_data:
        # Skip contracts that aren't the right type (call vs put).
        # ``.get("type", "")`` safely returns "" if the key is missing,
        # preventing a KeyError. ``.lower()`` normalizes case.
        if contract.get("type", "").lower() != contract_type:
            continue

        # --- Compute DTE (Days To Expiration) ---
        # Check for pre-computed _dte first (set by backtest engine to use
        # simulated dates instead of wall-clock time).
        if "_dte" in contract:
            dte = contract["_dte"]
        else:
            # Try both possible key names the broker API might use.
            exp_str = contract.get("expiration_date") or contract.get("expiration", "")
            try:
                # ``datetime.fromisoformat()`` parses ISO 8601 strings like
                # "2024-07-19" or "2024-07-19T00:00:00".
                # ``.replace(tzinfo=...)`` adds timezone info so we can subtract
                # from ``now`` (which is timezone-aware).
                exp_dt = (
                    datetime.fromisoformat(exp_str).replace(tzinfo=timezone.utc)
                    if exp_str
                    else None
                )
            except (ValueError, TypeError):
                # Malformed date string -- skip this contract entirely.
                continue
            if exp_dt is None:
                continue
            # ``.days`` gives the integer number of days between two datetimes.
            dte = (exp_dt - now).days
            # Cache the computed DTE on the contract dict so we don't recompute
            # it later. The leading underscore ``_dte`` is a Python convention
            # meaning "internal / computed field, not from the original API data."
            contract["_dte"] = dte

        # Only keep contracts within the desired DTE window.
        if min_dte <= dte <= max_dte:
            eligible.append(contract)

    return eligible


def enrich_greeks(
    contracts: list[dict],
    snapshots: dict,
    *,
    include_iv: bool = False,
) -> None:
    """Enrich contracts in-place with greeks and pricing from market snapshots.

    After this function runs, every contract dict will have these computed keys:
      - ``_delta``: The option's delta (directional sensitivity).
      - ``_theta``: The option's theta (daily time decay in dollars).
      - ``_bid``: Current best bid price.
      - ``_ask``: Current best ask price.
      - ``_mid``: Midpoint of bid/ask -- used as a fair-value estimate for
        spread pricing calculations.
      - ``_strike``: The strike price, normalized to float.
      - ``_iv`` (optional): Implied volatility, included only when
        ``include_iv=True``.

    Parameters
    ----------
    contracts : list[dict]
        Contracts to enrich. Modified **in-place** (no return value).
    snapshots : dict
        Keyed by option symbol -> snapshot dict with nested "greeks" dict
        and top-level "bid"/"ask" fields. Comes from broker API.
    include_iv : bool
        Whether to also attach implied volatility. Some strategies (e.g.
        long_call) check IV to avoid overpaying in high-IV environments.

    Python Note
    -----------
    The ``*`` in the parameter list forces ``include_iv`` to be keyword-only.
    You must write ``enrich_greeks(contracts, snaps, include_iv=True)`` --
    you cannot pass it positionally. This prevents accidental argument mix-ups.
    """
    for c in contracts:
        sym = c.get("symbol", "")
        snap = snapshots.get(sym, {})

        # Greeks are nested inside ``snap["greeks"]`` in the broker API response.
        # If "greeks" key is missing or None, default to an empty dict.
        greeks = snap.get("greeks") or {}

        c["_delta"] = greeks.get("delta", 0.0)
        c["_theta"] = greeks.get("theta", 0.0)

        # Bid/ask: the ``or 0.0`` handles cases where the value is None
        # (which .get() would return if the key exists but the value is null).
        bid = snap.get("bid", 0.0) or 0.0
        ask = snap.get("ask", 0.0) or 0.0
        c["_bid"] = bid
        c["_ask"] = ask

        # Mid-price: the average of bid and ask. Used to estimate fair value
        # when computing spread credits/debits, since real fills happen
        # somewhere between bid and ask.
        c["_mid"] = (bid + ask) / 2.0

        # Normalize strike price to a float. Different API responses may use
        # "strike_price" or "strike" as the key name.
        c["_strike"] = float(c.get("strike_price") or c.get("strike", 0))

        if include_iv:
            # IV (Implied Volatility): the market's expectation of future
            # stock movement, expressed as an annualized percentage. An IV of
            # 0.40 means the market expects ~40% annualized movement.
            c["_iv"] = greeks.get("implied_volatility", 0.0)


def filter_by_delta(
    contracts: list[dict],
    min_delta: float,
    max_delta: float,
    *,
    use_absolute: bool = True,
    fallback_min: float = 0.0,
) -> list[dict]:
    """Filter contracts by delta range with automatic fallback.

    Delta filtering is how strategies select OTM/ATM/ITM contracts.
    For example, delta 0.25-0.35 typically selects OTM options with ~25-35%
    probability of expiring ITM. Delta ~0.50 is roughly ATM.

    Parameters
    ----------
    min_delta / max_delta : float
        Desired delta range. For puts, delta is negative (e.g. -0.30), so
        ``use_absolute=True`` compares ``abs(delta)`` to avoid sign confusion.
    use_absolute : bool
        If True, compare ``abs(delta)`` against the range. This is convenient
        for puts whose deltas are negative by convention.
    fallback_min : float
        If the primary filter returns no results, fall back to any contract
        with ``abs(delta) > fallback_min``. Prevents returning an empty list
        when the delta range is slightly too narrow. Set to 0.0 to disable.

    Returns
    -------
    list[dict]
        Contracts matching the delta range (or fallback).

    Python Note
    -----------
    The list comprehension ``[c for c in contracts if ...]`` is Python's concise
    syntax for building a new list by filtering an existing one. It's equivalent
    to a for-loop with an if-check and append, but in a single expression.
    """
    if use_absolute:
        # ``abs()`` returns the absolute value, converting e.g. -0.30 to 0.30.
        # This way we can specify delta ranges as positive numbers for both
        # calls (positive delta) and puts (negative delta).
        result = [c for c in contracts if min_delta <= abs(c["_delta"]) <= max_delta]
    else:
        result = [c for c in contracts if min_delta <= c["_delta"] <= max_delta]

    # Fallback: if no contracts matched the narrow range, widen the criteria
    # so the strategy still has something to work with.
    if not result and fallback_min > 0:
        result = [c for c in contracts if abs(c["_delta"]) > fallback_min]
    return result


def select_by_delta(
    contracts: list[dict],
    target_delta: float,
    *,
    use_absolute: bool = True,
) -> dict | None:
    """Return the single contract whose delta is closest to *target_delta*.

    This is the final selection step after filtering: from a shortlist of
    candidates, pick the one that best matches our desired delta exposure.

    Parameters
    ----------
    target_delta : float
        The ideal delta. For example, 0.30 for an OTM option, 0.50 for ATM.
    use_absolute : bool
        Compare using ``abs(delta)`` if True. Useful for puts.

    Returns
    -------
    dict or None
        The best-matching contract, or None if the input list is empty.

    Python Note
    -----------
    ``min(iterable, key=lambda c: ...)`` returns the element with the smallest
    value of the key function. ``lambda`` creates an anonymous (unnamed) function
    inline. Here, ``lambda c: abs(abs(c["_delta"]) - target_delta)`` computes
    the distance between each contract's delta and our target; ``min`` picks
    the contract with the smallest distance.

    The ``dict | None`` return type uses Python's union syntax (3.10+) meaning
    "this function returns either a dict or None." The ``from __future__ import
    annotations`` at the top of this file enables this syntax on older Pythons.
    """
    if not contracts:
        return None
    if use_absolute:
        return min(contracts, key=lambda c: abs(abs(c["_delta"]) - target_delta))
    return min(contracts, key=lambda c: abs(c["_delta"] - target_delta))


class BaseOptionsStrategy(ABC):
    """Abstract base class for all options strategies.

    Every concrete strategy (CreditPutSpreadStrategy, LongCallStrategy, etc.)
    inherits from this class and implements the ``evaluate()`` method.

    Python Concepts
    ---------------
    - **ABC** (Abstract Base Class): A class that *cannot* be instantiated
      directly. You must create a subclass that implements all ``@abstractmethod``
      methods. If you forget to implement one, Python raises ``TypeError`` at
      instantiation time.
    - **@abstractmethod**: Marks a method that subclasses *must* override.
      The base version here just has ``pass`` (does nothing) -- it exists solely
      to define the interface contract.
    - **getattr(obj, name, default)**: Safely reads an attribute from an object.
      ``getattr(config, "enabled", True)`` returns ``config.enabled`` if it
      exists, or ``True`` if it doesn't. This is more defensive than
      ``config.enabled`` which would raise ``AttributeError`` if the attribute
      is missing. We use it throughout because config objects may not have every
      field defined.
    """

    def __init__(self, config):
        self.config = config
        # ``getattr(config, "enabled", True)`` -- see docstring above.
        # Default to enabled if the config doesn't specify.
        self.enabled = getattr(config, "enabled", True)

    @abstractmethod
    def evaluate(
        self,
        underlying: str,
        stock_bars,
        chain_data: list[dict],
        snapshots: dict,
    ) -> OptionsSignal | None:
        """Evaluate whether to enter an options trade.

        This is the core method every strategy implements. The orchestrator
        calls this for each symbol on each evaluation cycle.

        Parameters
        ----------
        underlying : str
            Stock ticker symbol, e.g. "AAPL".
        stock_bars : pandas.DataFrame
            Historical daily OHLCV bars for the underlying stock, with any
            pre-computed technical indicators. Strategies add their own
            indicators (RSI, EMA, etc.) as needed.
        chain_data : list[dict]
            The full options chain from the broker API. Each dict is one
            contract with fields like "type", "strike_price", "expiration_date",
            "symbol".
        snapshots : dict
            Real-time option quotes keyed by option symbol. Each value contains
            "bid", "ask", and nested "greeks" with "delta", "theta", etc.

        Returns
        -------
        OptionsSignal or None
            An ``OptionsSignal`` if the strategy wants to trade, or ``None``
            if conditions are not met (no trade).
        """
        pass
