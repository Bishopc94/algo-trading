"""Black-Scholes options pricing for backtesting.

This module provides synthetic options pricing using the Black-Scholes model,
a foundational mathematical formula from quantitative finance that estimates
the theoretical (or "fair") price of European-style options contracts.

HOW IT FITS IN THE BACKTEST PIPELINE:
-------------------------------------
Real options market data is expensive and often unavailable for historical
backtesting. Instead, this module generates *synthetic* options chains --
fake but realistic-looking options data -- from historical stock prices.
The backtest engine (engine.py) calls into this module every simulated day
to produce options chains that options strategies can evaluate, just as if
they were looking at live market data.

KEY CONCEPTS FOR NON-PYTHON READERS:
------------------------------------
- ``from __future__ import annotations``: A Python compatibility directive
  that makes type hints (e.g., ``float | None``) work as strings rather than
  being evaluated immediately.  This avoids errors on older Python versions.

- ``@dataclass``: A Python decorator (think: annotation) that auto-generates
  boilerplate code (constructor, __repr__, equality) for simple data-holding
  classes.  ``Greeks`` below is equivalent to a struct with default values.

- ``math.erf``, ``math.exp``, ``math.log``: Standard math library functions.
  ``erf`` is the Gauss error function used to compute the normal CDF without
  pulling in the heavy ``scipy`` library.

TRADING CONCEPTS:
-----------------
- **Options**: Financial contracts giving the right (not obligation) to buy
  (call) or sell (put) a stock at a fixed price (strike) by a certain date
  (expiration).

- **Black-Scholes Model**: Assumes stock prices follow geometric Brownian
  motion with constant volatility. Inputs: stock price (S), strike (K),
  time to expiration (T, in years), risk-free rate (r), and volatility
  (sigma). Output: theoretical option price.

- **Greeks**: Sensitivity measures that describe how an option's price
  changes relative to various factors:
    - Delta: price change per $1 move in the underlying stock
    - Gamma: rate of change of delta (second derivative of price)
    - Theta: daily time decay (options lose value as expiration nears)
    - Vega: sensitivity to a 1% change in implied volatility

- **Implied Volatility (IV)**: The market's forecast of future price
  movement. In this module we use *historical* volatility as a proxy
  since we have no live options market to back-solve IV from.

- **OCC Symbol**: The Options Clearing Corporation's standardized format
  for uniquely identifying an options contract, e.g., "AAPL  250321C00150000"
  encodes the underlying, expiration date, call/put type, and strike price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np  # NumPy: fast array math library (like C arrays with SIMD)
import pandas as pd  # Pandas: tabular data library (like SQL tables in memory)

# ---------------------------------------------------------------------------
# Black-Scholes core
# ---------------------------------------------------------------------------

# Pre-compute sqrt(2*pi) as a module-level constant so we don't recalculate
# it on every call to _norm_pdf. The leading underscore is a Python convention
# meaning "private / internal to this module -- not part of the public API."
_SQRT_2PI = math.sqrt(2 * math.pi)


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function (CDF), pure Python.

    The CDF answers: "What is the probability that a standard normal random
    variable is less than or equal to x?"  This is central to Black-Scholes
    because option pricing depends on the probability that the stock ends up
    above or below the strike price at expiration.

    We use ``math.erf`` (the Gauss error function) to avoid depending on
    scipy.stats.norm, which would be a heavy dependency just for this one
    function.  The identity is:  CDF(x) = 0.5 * (1 + erf(x / sqrt(2))).
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function (PDF).

    The PDF gives the "height" of the bell curve at point x. Used in the
    Greeks calculations (gamma, theta, vega) because those are derivatives
    of the option price, and the derivative of the CDF is the PDF.

    Formula: (1 / sqrt(2*pi)) * exp(-x^2 / 2)
    """
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d1, one of the two key intermediate values in Black-Scholes.

    d1 represents a z-score measuring how far "in the money" an option is,
    adjusted for drift and volatility over the remaining time.

    Formula: [ln(S/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))
    Where:
        ln(S/K) = log-moneyness (positive if stock > strike)
        r + sigma^2/2 = expected drift under risk-neutral measure
        sigma * sqrt(T) = total expected volatility over remaining life

    Guard clause: returns 0.0 for degenerate inputs (expired, zero vol, etc.)
    to prevent division by zero or log-of-zero errors.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d2 = d1 - sigma * sqrt(T).

    d2 is the probability (in z-score terms) that the option expires in the
    money under the risk-neutral measure.  It differs from d1 by exactly
    one standard deviation of log-returns over the option's remaining life.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    return _d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes theoretical price for a European call option.

    The formula is:  C = S * N(d1) - K * e^(-rT) * N(d2)
    Where:
        S * N(d1)           = expected stock value received if exercised
        K * e^(-rT) * N(d2) = present value of strike paid, weighted by
                              the probability of exercise

    When T <= 0 (option has expired), the price collapses to intrinsic value:
    max(0, S - K).  A call's intrinsic value is how far the stock price
    exceeds the strike -- you'd exercise to buy at K and immediately sell at S.

    Args:
        S: Current stock price (spot price)
        K: Strike price (the agreed buy price if the option is exercised)
        T: Time to expiration in years (e.g., 30 days = 30/365 ~ 0.082)
        r: Risk-free interest rate, annualized (e.g., 0.045 for 4.5%)
        sigma: Annualized volatility (e.g., 0.30 for 30% annual volatility)
    """
    if T <= 0:
        # At expiration: intrinsic value only (no time value remains)
        return max(0.0, S - K)
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes theoretical price for a European put option.

    The formula is:  P = K * e^(-rT) * N(-d2) - S * N(-d1)
    This is derived from put-call parity: P = C - S + K*e^(-rT).

    A put gains value as the stock drops *below* the strike. At expiration,
    intrinsic value = max(0, K - S).
    """
    if T <= 0:
        # At expiration: intrinsic value only
        return max(0.0, K - S)
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    # Note: N(-x) = 1 - N(x), so N(-d2) and N(-d1) flip the probabilities
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

@dataclass
class Greeks:
    """Container for option sensitivity measures (the "Greeks").

    Python's @dataclass decorator auto-generates __init__, __repr__, and
    __eq__ methods.  Each field has a default value of 0.0, so you can
    create a Greeks() with no arguments and get all zeros.
    """
    delta: float = 0.0              # Price change per $1 move in underlying
    gamma: float = 0.0              # Rate of change of delta
    theta: float = 0.0              # Daily time decay (usually negative for long positions)
    vega: float = 0.0               # Sensitivity to 1% change in implied volatility
    implied_volatility: float = 0.0  # The volatility used in pricing (historical, in this module)


def call_greeks(S: float, K: float, T: float, r: float, sigma: float) -> Greeks:
    """Compute Greeks for a call option using Black-Scholes closed-form solutions.

    At expiration (T <= 0), delta is binary: 1.0 if in-the-money (S > K),
    0.0 otherwise.  All other Greeks go to zero because there is no remaining
    time or uncertainty.

    Formulas:
        delta = N(d1)                           -- always between 0 and 1 for calls
        gamma = phi(d1) / (S * sigma * sqrt(T)) -- same for calls and puts
        theta = [-S*phi(d1)*sigma / (2*sqrt(T)) - r*K*e^(-rT)*N(d2)] / 365
                 (divided by 365 to express as per-calendar-day decay)
        vega  = S * phi(d1) * sqrt(T) / 100
                 (divided by 100 to express per 1% move in volatility)

    Where phi(x) is the standard normal PDF.
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        # Expired or degenerate: return boundary values
        return Greeks(
            delta=1.0 if S > K else 0.0,
            implied_volatility=sigma,
        )
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    sqrt_T = math.sqrt(T)

    delta = _norm_cdf(d1)
    gamma = _norm_pdf(d1) / (S * sigma * sqrt_T)
    theta = (
        -(S * _norm_pdf(d1) * sigma) / (2 * sqrt_T)
        - r * K * math.exp(-r * T) * _norm_cdf(d2)
    ) / 365.0  # Convert annualized theta to per-calendar-day
    vega = S * _norm_pdf(d1) * sqrt_T / 100.0  # Per 1 percentage-point move in vol

    return Greeks(
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        implied_volatility=sigma,
    )


def put_greeks(S: float, K: float, T: float, r: float, sigma: float) -> Greeks:
    """Compute Greeks for a put option using Black-Scholes closed-form solutions.

    Put delta is always negative (put gains value when stock falls).
    The relationship is: put_delta = call_delta - 1.0 = N(d1) - 1.

    Gamma and vega are identical for calls and puts (same underlying sensitivity).
    Theta differs because the discounting term flips sign for puts.
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return Greeks(
            delta=-1.0 if S < K else 0.0,  # Fully in-the-money put has delta = -1
            implied_volatility=sigma,
        )
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    sqrt_T = math.sqrt(T)

    delta = _norm_cdf(d1) - 1.0  # Negative for puts (range: -1 to 0)
    gamma = _norm_pdf(d1) / (S * sigma * sqrt_T)  # Same formula as call gamma
    theta = (
        -(S * _norm_pdf(d1) * sigma) / (2 * sqrt_T)
        + r * K * math.exp(-r * T) * _norm_cdf(-d2)  # Note: + instead of - for puts
    ) / 365.0
    vega = S * _norm_pdf(d1) * sqrt_T / 100.0  # Same formula as call vega

    return Greeks(
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        implied_volatility=sigma,
    )


# ---------------------------------------------------------------------------
# Historical volatility
# ---------------------------------------------------------------------------

def historical_volatility(prices: pd.Series, window: int = 30) -> float:
    """Compute annualized historical volatility from a price series.

    METHODOLOGY:
    1. Compute daily log-returns: ln(price_today / price_yesterday).
       Log-returns are preferred over simple percentage returns because
       they are additive over time and approximately normal.
    2. Take the standard deviation of the most recent ``window`` log-returns.
    3. Annualize by multiplying by sqrt(252), where 252 is the approximate
       number of trading days per year.  This scaling comes from the
       statistical property that volatility scales with the square root
       of time.

    This value is used as the ``sigma`` input to Black-Scholes throughout
    this module, standing in for implied volatility (which would require
    real market option prices to compute).

    Args:
        prices: A pandas Series of daily closing prices, indexed by date.
                pandas Series is like a typed, labeled array -- similar to
                a single column from a spreadsheet.
        window: Number of trailing trading days to measure volatility over.
                Default 30 balances recency with statistical stability.

    Returns:
        Annualized volatility as a decimal (e.g., 0.30 means 30% annual vol).
        Falls back to 0.30 (a reasonable default) if insufficient data.
    """
    if len(prices) < window + 1:
        # Not enough data for the requested window; use what we have
        if len(prices) < 6:
            return 0.30  # Absolute minimum: return a sensible default (30% vol)
        window = len(prices) - 1

    # np.log computes element-wise natural logarithm on the entire Series.
    # prices.shift(1) offsets the series by one row (yesterday's prices).
    # Division gives today/yesterday ratios; log converts to log-returns.
    # .dropna() removes the first row which is NaN (no "yesterday" for day 1).
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # .iloc[-window:] is Python/pandas "negative indexing": take the last
    # ``window`` elements.  This is equivalent to slicing from the end.
    recent = log_returns.iloc[-window:]

    # .std() computes sample standard deviation. Multiply by sqrt(252) to
    # annualize (daily vol -> yearly vol).
    return float(recent.std() * math.sqrt(252))


# ---------------------------------------------------------------------------
# OCC symbol formatting
# ---------------------------------------------------------------------------

def _occ_symbol(underlying: str, expiration: datetime, opt_type: str, strike: float) -> str:
    """Generate an OCC-format option symbol string.

    The Options Clearing Corporation (OCC) defines a standardized symbology
    for identifying option contracts.  This format is used by most US brokers
    and exchanges.

    Format: SSSSSS YYMMDD C SSSSSIII
            |      |      | |
            |      |      | +-- Strike price * 1000, zero-padded to 8 digits
            |      |      +---- "C" for call, "P" for put
            |      +----------- Expiration date as YYMMDD
            +------------------ Underlying ticker, left-padded to 6 chars

    Example: "AAPL  250321C00150000" = AAPL call, expires 2025-03-21, $150 strike.

    Python notes:
        - ``str.ljust(6)[:6]``: Left-justify to 6 chars (pad with spaces),
          then truncate to exactly 6 chars if the ticker is longer.
        - ``f"{strike_int:08d}"``: f-string format specifier meaning
          "format as integer, zero-padded to 8 digits."
    """
    sym = underlying.ljust(6)[:6]  # Pad/truncate underlying to exactly 6 characters
    date_str = expiration.strftime("%y%m%d")  # strftime: format datetime as string (YY MM DD)
    cp = "C" if opt_type == "call" else "P"
    strike_int = int(round(strike * 1000))  # OCC encodes strike as integer * 1000
    strike_str = f"{strike_int:08d}"  # Zero-pad to 8 digits
    return f"{sym}{date_str}{cp}{strike_str}"


# ---------------------------------------------------------------------------
# Synthetic chain generation
# ---------------------------------------------------------------------------

def generate_synthetic_chain(
    underlying: str,
    stock_price: float,
    current_date: datetime,
    volatility: float,
    risk_free_rate: float = 0.045,
    num_expirations: int = 4,
    strike_range_pct: float = 0.15,
    strike_step: float | None = None,
) -> tuple[list[dict], dict]:
    """Generate a synthetic options chain with Black-Scholes pricing.

    This is the main entry point for the backtest engine. It creates a full
    options chain -- multiple expiration dates, multiple strike prices, both
    calls and puts -- complete with theoretical prices, bid/ask spreads,
    and Greeks.  The output format matches the live brokerage API so that
    strategy code doesn't need to know whether it's looking at real or
    synthetic data.

    WHAT AN OPTIONS CHAIN IS:
    An options chain is a grid of all available option contracts for a given
    underlying stock.  Each row is one contract defined by:
    - Expiration date (when the option expires)
    - Strike price (the agreed exercise price)
    - Type (call or put)
    Each contract has a market price (bid/ask) and Greeks.

    SYNTHETIC VS. REAL:
    Real chains come from exchanges and reflect actual market supply/demand.
    Synthetic chains use Black-Scholes math to estimate what those prices
    *should* be, given the stock price and volatility.  This is a simplification
    (real markets have skew, term structure, etc.) but adequate for backtesting.

    Args:
        underlying: Stock ticker symbol (e.g., "AAPL").
        stock_price: Current stock price on the simulated date.
        current_date: The backtest date, used to compute days-to-expiration (DTE).
        volatility: Annualized historical volatility from ``historical_volatility()``.
        risk_free_rate: Annualized risk-free rate (default 4.5%, roughly US Treasury yield).
        num_expirations: How many biweekly expiration dates to generate beyond
                        the short-dated ones.
        strike_range_pct: How far above/below the stock price to generate strikes
                         (0.15 = +/-15% of current price).
        strike_step: Dollar increment between strikes.  If None, auto-selected
                    based on stock price (cheaper stocks get finer granularity).

    Returns:
        A tuple of (chain_data, snapshots):
        - chain_data: list of dicts, each describing one contract (symbol,
          strike, expiration, type, root_symbol).
        - snapshots: dict mapping OCC symbol -> pricing/Greeks dict (bid, ask,
          mid_price, greeks, volume, open_interest).

    Python notes:
        - ``tuple[list[dict], dict]``: Type hint for a function returning two
          values.  Python functions can return multiple values as a tuple,
          which callers unpack with ``chain_data, snapshots = generate_...()``.
        - ``float | None``: Union type meaning "either a float or None."
          The pipe syntax (``|``) for unions requires Python 3.10+ or the
          ``from __future__ import annotations`` import at the top of this file.
    """
    # Guard: invalid inputs would produce nonsensical prices
    if stock_price <= 0 or volatility <= 0:
        return [], {}

    # ---------- Auto-select strike step based on stock price ----------
    # Real exchanges use different strike increments depending on the stock's
    # price level.  Cheap stocks ($5) have $0.50 strikes; expensive stocks
    # ($500+) use $10 increments.
    if strike_step is None:
        if stock_price < 5:
            strike_step = 0.50
        elif stock_price < 25:
            strike_step = 1.0
        elif stock_price < 100:
            strike_step = 2.50
        elif stock_price < 500:
            strike_step = 5.0
        else:
            strike_step = 10.0

    # ---------- Generate expiration dates ----------
    # Options expire on Fridays (weekly options) or the third Friday of the
    # month (monthly options).  We simulate weekly-style expirations by
    # picking the nearest Friday to each target date.
    expirations: list[datetime] = []

    # Short-dated expirations (3-10 days out) for aggressive/momentum strategies
    # that want quick, high-gamma trades.
    for days_out in (3, 5, 7, 10):
        exp_date = current_date + timedelta(days=days_out)
        # Find the next Friday: weekday() returns 0=Mon, 4=Fri.
        # (4 - weekday) % 7 gives days until Friday.
        days_to_friday = (4 - exp_date.weekday()) % 7
        if days_to_friday == 0 and exp_date.weekday() != 4:
            days_to_friday = 7  # If it's not actually Friday, skip to next week
        exp_date = exp_date + timedelta(days=days_to_friday)
        # Set time to market close (4:00 PM UTC, approximating US market close)
        exp_date = exp_date.replace(hour=16, minute=0, second=0, tzinfo=timezone.utc)
        expirations.append(exp_date)

    # Longer-dated expirations (biweekly, 2-8+ weeks out) for strategies
    # that hold positions for weeks (credit spreads, cash-secured puts).
    # range(start, stop, step) generates: 2, 4, 6, 8, ... weeks.
    for weeks_out in range(2, 2 + num_expirations * 2, 2):
        exp_date = current_date + timedelta(days=weeks_out * 7)
        days_to_friday = (4 - exp_date.weekday()) % 7
        exp_date = exp_date + timedelta(days=days_to_friday)
        exp_date = exp_date.replace(hour=16, minute=0, second=0, tzinfo=timezone.utc)
        expirations.append(exp_date)

    # Deduplicate (short-dated and long-dated may land on the same Friday)
    # using a set to track seen date strings.  Python sets have O(1) lookup.
    seen = set()
    unique_expirations = []
    for exp in expirations:
        key = exp.strftime("%Y-%m-%d")
        if key not in seen:
            seen.add(key)
            unique_expirations.append(exp)
    expirations = sorted(unique_expirations)  # Chronological order

    # ---------- Generate strike prices ----------
    # Create strikes in a range around the current stock price (at-the-money).
    # math.floor/ceil snap to the nearest strike_step boundary.
    low_strike = stock_price * (1.0 - strike_range_pct)
    high_strike = stock_price * (1.0 + strike_range_pct)
    low_strike = math.floor(low_strike / strike_step) * strike_step
    high_strike = math.ceil(high_strike / strike_step) * strike_step

    strikes: list[float] = []
    s = low_strike
    while s <= high_strike:
        if s > 0:  # Strikes must be positive
            strikes.append(round(s, 2))  # round() avoids floating-point drift (e.g., 2.50000001)
        s += strike_step

    # ---------- Build the chain: price every (expiration, strike, type) combo ----------
    chain_data: list[dict] = []       # Contract metadata (like a catalog listing)
    snapshots: dict[str, dict] = {}   # Pricing data keyed by OCC symbol

    for exp in expirations:
        # DTE = Days To Expiration. Computed as calendar days between
        # the current backtest date and the expiration date.
        dte = (exp - current_date.replace(tzinfo=timezone.utc)).days
        if dte <= 0:
            continue  # Skip expired contracts
        T = dte / 365.0  # Convert to years for Black-Scholes

        exp_str = exp.strftime("%Y-%m-%d")

        for strike in strikes:
            # Generate both a call and a put at each (expiration, strike) pair
            for opt_type in ("call", "put"):
                # Create the standardized OCC symbol for this contract
                occ = _occ_symbol(underlying, exp, opt_type, strike)

                # ---------- Compute theoretical price and Greeks ----------
                if opt_type == "call":
                    price = bs_call_price(stock_price, strike, T, risk_free_rate, volatility)
                    greeks = call_greeks(stock_price, strike, T, risk_free_rate, volatility)
                else:
                    price = bs_put_price(stock_price, strike, T, risk_free_rate, volatility)
                    greeks = put_greeks(stock_price, strike, T, risk_free_rate, volatility)

                # ---------- Simulate bid-ask spread ----------
                # In real markets, you can't buy at the theoretical price.
                # Market makers quote a "bid" (what they'll pay) and an "ask"
                # (what they'll sell for).  The spread is wider for:
                #   - Out-of-the-money (OTM) options (less liquid)
                #   - Cheap options (minimum tick size dominates)
                # "Moneyness" measures how far from ATM (at-the-money) we are.
                moneyness = abs(stock_price - strike) / stock_price
                spread_pct = 0.02 + moneyness * 0.05  # 2% base + 5% per unit of moneyness
                half_spread = max(0.01, price * spread_pct / 2)  # At least $0.01

                bid = max(0.01, round(price - half_spread, 2))  # Bid >= $0.01
                ask = max(0.02, round(price + half_spread, 2))  # Ask > bid
                mid = round((bid + ask) / 2, 2)  # Midpoint = fair estimate

                # Contract metadata -- matches the format returned by the
                # live brokerage API so strategies don't need separate code paths.
                contract = {
                    "symbol": occ,
                    "strike_price": strike,
                    "expiration_date": exp_str,
                    "type": opt_type,
                    "root_symbol": underlying,
                }
                chain_data.append(contract)

                # Pricing snapshot -- also matches the live API format.
                # Volume and open_interest are synthetic placeholders.
                snapshots[occ] = {
                    "bid": bid,
                    "ask": ask,
                    "mid_price": mid,
                    "last_trade": mid,  # Synthetic: assume last trade = midpoint
                    "volume": 100,      # Placeholder (real data would vary)
                    "open_interest": 500,  # Placeholder
                    "greeks": {
                        "delta": round(greeks.delta, 4),
                        "gamma": round(greeks.gamma, 6),  # Gamma is small, needs more decimals
                        "theta": round(greeks.theta, 4),
                        "vega": round(greeks.vega, 4),
                        "implied_volatility": round(greeks.implied_volatility, 4),
                    },
                }

    return chain_data, snapshots


def price_option_at_expiration(
    opt_type: str,
    strike: float,
    stock_price: float,
) -> float:
    """Calculate the intrinsic value of an option at expiration.

    At expiration, an option's time value is zero and only intrinsic value
    remains.  This is the simplest options pricing formula:

        Call intrinsic = max(0, stock_price - strike)
           "I can buy at strike and sell at market price"
        Put intrinsic  = max(0, strike - stock_price)
           "I can sell at strike and buy at market price"

    If the option is out-of-the-money (OTM), intrinsic is zero and the
    option expires worthless.  This function is used by the backtest engine
    to settle options positions at expiration.
    """
    if opt_type == "call":
        return max(0.0, stock_price - strike)
    else:
        return max(0.0, strike - stock_price)
