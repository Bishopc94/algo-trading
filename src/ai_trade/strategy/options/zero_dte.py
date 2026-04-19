"""0DTE (same-day expiration) options strategy.

Buys cheap OTM calls or puts expiring TODAY on highly liquid
underlyings (SPY, QQQ, and configurable individual stocks) when
multi-indicator directional confluence is strong.

Why 0DTE is different from longer-dated options:
    - Theta decay is massive -- the position must move fast.
    - Gamma is extreme -- delta changes rapidly, amplifying gains/losses.
    - Tight entry timing is critical (best windows: 9:45-10:30, 14:00-15:00).
    - Premium is cheap in absolute dollars, making it viable on a $500 account.

Risk profile:
    - Max loss = premium paid (small absolute dollar amount).
    - Max gain = theoretically unlimited but practically 100-500%+ ROI.
    - Hard time exit at 15:30 ET (the position monitor job handles this).
    - Loss cut at -50% of premium.
    - Trailing stop at 50% of max profit.

The position monitor job (``job_monitor_zero_dte`` in main.py) is
responsible for intraday management -- this module only generates
the entry signal.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from ai_trade.data.indicators import add_atr, add_ema, add_macd, add_rsi, add_volume_profile
from ai_trade.monitoring.logger import get_logger
from ai_trade.strategy.options.base import (
    BaseOptionsStrategy,
    OptionsSignal,
    OptionsStrategyType,
    enrich_greeks,
    filter_by_delta,
    filter_contracts,
)

log = get_logger(__name__)

_ET = ZoneInfo("America/New_York")

_DEFAULT_LIQUID_UNDERLYINGS = frozenset({
    "SPY", "QQQ", "AAPL", "TSLA", "AMZN", "NVDA", "META", "MSFT", "AMD", "GOOG",
})

_ENTRY_WINDOWS = [
    (9, 45, 10, 30),
    (14, 0, 15, 0),
]


def _in_entry_window(now: datetime | None = None) -> bool:
    now = now or datetime.now(_ET)
    for h1, m1, h2, m2 in _ENTRY_WINDOWS:
        start_min = h1 * 60 + m1
        end_min = h2 * 60 + m2
        cur_min = now.hour * 60 + now.minute
        if start_min <= cur_min <= end_min:
            return True
    return False


class ZeroDTEStrategy(BaseOptionsStrategy):
    """Buy 0DTE OTM options on high-conviction directional setups."""

    bias = "adaptive"

    def evaluate(
        self,
        underlying: str,
        stock_bars: pd.DataFrame,
        chain_data: list[dict],
        snapshots: dict,
    ) -> OptionsSignal | None:
        if not self.enabled:
            return None

        liquid = getattr(self.config, "liquid_underlyings", None)
        if liquid:
            allowed = set(liquid) if isinstance(liquid, (list, tuple)) else _DEFAULT_LIQUID_UNDERLYINGS
        else:
            allowed = _DEFAULT_LIQUID_UNDERLYINGS

        if underlying not in allowed:
            return None

        if not _in_entry_window():
            return None

        max_contract_cost: float = getattr(self.config, "max_contract_cost", 50.0)
        min_delta: float = getattr(self.config, "min_delta", 0.15)
        max_delta: float = getattr(self.config, "max_delta", 0.40)

        # ------------------------------------------------------------------
        # 1. Stock-level directional filter (daily bars)
        # ------------------------------------------------------------------
        df = stock_bars.copy()
        if len(df) < 30:
            return None

        add_rsi(df)
        add_ema(df, periods=[9, 20, 50])
        add_atr(df)
        add_volume_profile(df)
        add_macd(df)

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        rsi: float = latest.get("rsi_14", 50.0)
        price: float = latest["close"]
        ema_9: float = latest.get("ema_9", price)
        ema_20: float = latest.get("ema_20", price)
        ema_50: float = latest.get("ema_50", price)
        atr: float = latest.get("atr_14", 0.0)
        rel_vol: float = latest.get("relative_volume", 0.0)
        macd_hist: float = latest.get("macd_hist", 0.0)
        prev_macd_hist: float = prev.get("macd_hist", 0.0)

        min_rel_vol: float = getattr(self.config, "min_relative_volume", 1.5)
        if rel_vol < min_rel_vol:
            self._reject(underlying, "rel_volume", rel_vol, min_rel_vol, "above")
            return None

        # Multi-timeframe directional confluence
        direction = None
        if (rsi > 50 and price > ema_9 > ema_20 and macd_hist > 0):
            direction = "call"
        elif (rsi < 50 and price < ema_9 < ema_20 and macd_hist < 0):
            direction = "put"

        if direction is None:
            self._reject(underlying, "directional_confluence", 0.0, 1.0, "above")
            return None

        # ------------------------------------------------------------------
        # 2. Select 0DTE contract
        # ------------------------------------------------------------------
        eligible = filter_contracts(chain_data, direction, min_dte=0, max_dte=0)
        if not eligible:
            eligible = filter_contracts(chain_data, direction, min_dte=0, max_dte=1)
        if not eligible:
            return None

        enrich_greeks(eligible, snapshots, include_iv=True)

        delta_ok = filter_by_delta(eligible, min_delta, max_delta, fallback_min=0.10)
        if not delta_ok:
            return None

        # ------------------------------------------------------------------
        # 3. Select cheapest contract within budget
        # ------------------------------------------------------------------
        delta_ok.sort(key=lambda c: c["_ask"])

        selected = None
        for c in delta_ok:
            cost = c["_ask"] * 100
            if 0 < cost <= max_contract_cost:
                selected = c
                break

        if selected is None:
            return None

        # ------------------------------------------------------------------
        # 4. ROI estimation (gamma-driven)
        # ------------------------------------------------------------------
        ask_price: float = selected["_ask"]
        cost_per_contract: float = ask_price * 100
        strike: float = selected["_strike"]
        delta: float = abs(selected["_delta"])

        if direction == "call":
            target_price = price + 1.0 * atr
            intrinsic_at_target = max(0, target_price - strike)
        else:
            target_price = price - 1.0 * atr
            intrinsic_at_target = max(0, strike - target_price)

        potential_profit = intrinsic_at_target - ask_price
        potential_roi = potential_profit / ask_price if ask_price > 0 else 0

        min_roi: float = getattr(self.config, "min_roi_pct", 0.50)
        if potential_roi < min_roi:
            return None

        # ------------------------------------------------------------------
        # 5. Conviction scoring (0.50-0.92)
        # ------------------------------------------------------------------
        conviction: float = 0.50

        if rel_vol > 3.0:
            conviction += 0.10
        elif rel_vol > 2.0:
            conviction += 0.06
        elif rel_vol > 1.5:
            conviction += 0.03

        if potential_roi > 3.0:
            conviction += 0.10
        elif potential_roi > 1.5:
            conviction += 0.06
        elif potential_roi > 0.75:
            conviction += 0.03

        if direction == "call" and macd_hist > prev_macd_hist > 0:
            conviction += 0.07
        elif direction == "put" and macd_hist < prev_macd_hist < 0:
            conviction += 0.07
        elif (direction == "call" and macd_hist > 0) or (direction == "put" and macd_hist < 0):
            conviction += 0.03

        ema_stack = (
            (direction == "call" and ema_9 > ema_20 > ema_50)
            or (direction == "put" and ema_9 < ema_20 < ema_50)
        )
        if ema_stack:
            conviction += 0.08

        if 0.20 <= delta <= 0.35:
            conviction += 0.05
        elif delta > 0.15:
            conviction += 0.03

        now_et = datetime.now(_ET)
        cur_min = now_et.hour * 60 + now_et.minute
        if 585 <= cur_min <= 630:
            conviction += 0.02
        elif 840 <= cur_min <= 900:
            conviction += 0.02

        conviction = max(0.50, min(0.92, conviction))

        # ------------------------------------------------------------------
        # 6. Build signal
        # ------------------------------------------------------------------
        option_symbol: str = selected.get("symbol", "")
        expiration: str = selected.get("expiration_date") or selected.get("expiration", "")
        dte = selected.get("_dte", 0)

        legs = [
            {"symbol": option_symbol, "side": "buy", "qty": 1, "position_intent": "buy_to_open"},
        ]

        strategy_type = (
            OptionsStrategyType.ZERO_DTE_CALL if direction == "call"
            else OptionsStrategyType.ZERO_DTE_PUT
        )

        log.info(
            "zero_dte_signal",
            underlying=underlying,
            direction=direction,
            strike=strike,
            ask=ask_price,
            cost=cost_per_contract,
            delta=selected["_delta"],
            dte=dte,
            potential_roi=round(potential_roi, 2),
            conviction=conviction,
            rel_vol=rel_vol,
            atr=atr,
            expiration=expiration,
        )

        return OptionsSignal(
            underlying=underlying,
            strategy_type=strategy_type,
            conviction=conviction,
            strategy_name="zero_dte",
            legs=legs,
            max_cost=ask_price,
            max_loss=cost_per_contract,
            max_profit=cost_per_contract * 5.0,
            expiration=expiration,
            strikes=[strike],
            net_delta=selected["_delta"],
            net_theta=selected["_theta"],
            metadata={
                "direction": direction,
                "rsi": rsi,
                "price": price,
                "atr": atr,
                "relative_volume": rel_vol,
                "macd_hist": macd_hist,
                "potential_roi": potential_roi,
                "cost_per_contract": cost_per_contract,
                "dte": dte,
                "iv": selected["_iv"],
                "is_zero_dte": True,
            },
        )
