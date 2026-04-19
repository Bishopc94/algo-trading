"""Registry of tunable strategy parameters for the optimizer.

Each ParamSpec defines one knob the optimizer can turn: its bounds,
step size, and the trade_analysis field + values that signal whether
the knob should be nudged up or down.

Adding a new tunable parameter:
    1. Add a ParamSpec to PARAM_SPECS below.
    2. Ensure the analysis pipeline populates the quality_field with
       values that include at least one widen and one tighten value.
    3. The optimizer picks it up automatically on the next review.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParamSpec:
    name: str
    min_val: float
    max_val: float
    step: float
    quality_field: str
    widen_values: frozenset[str]
    tighten_values: frozenset[str]


PARAM_SPECS: dict[str, ParamSpec] = {
    "atr_stop_multiplier": ParamSpec(
        name="atr_stop_multiplier",
        min_val=0.75,
        max_val=3.00,
        step=0.10,
        quality_field="stop_quality",
        widen_values=frozenset({"too_tight", "trail_too_tight"}),
        tighten_values=frozenset({"too_loose"}),
    ),
    "atr_tp_multiplier": ParamSpec(
        name="atr_tp_multiplier",
        min_val=1.50,
        max_val=5.00,
        step=0.10,
        quality_field="exit_quality",
        widen_values=frozenset({"left_money_on_table"}),
        tighten_values=frozenset({"sold_near_top"}),
    ),
}
