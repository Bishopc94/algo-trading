"""Smoke test the conviction-aware trailing stop logic.

Replays the IONZ scenario: entry $8.55, stop $8.37, ATR ~0.18, price
runs to $8.78, then normal pullback. The old logic ratcheted the stop
to $8.53 (breakeven) on the peak and killed the trade on pullback.
The reworked logic should leave a 97% conviction trade alone until
price runs much further.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, SRC)

from ai_trade.strategy.exit_planner import (  # noqa: E402
    _trail_params_for_conviction,
    compute_trailing_stop_long,
)


def case(name, **kw):
    new_stop, mode = compute_trailing_stop_long(**kw)
    print(f"  {name}: new_stop={new_stop} mode={mode!r}")
    return new_stop, mode


def main() -> int:
    print("== Test 1: Conviction -> trail param table ==")
    for conv in [0.70, 0.78, 0.85, 0.90, 0.95, 0.97]:
        be, chand = _trail_params_for_conviction(conv)
        print(f"  conv={conv:.2f} breakeven_trigger={be} chandelier_mult={chand}")
    assert _trail_params_for_conviction(0.97)[0] is None, "0.97 should skip breakeven"
    assert _trail_params_for_conviction(0.70)[0] == 1.0, "0.70 should keep 1.0 trigger"
    assert _trail_params_for_conviction(0.85)[1] == 3.5

    print("\n== Test 2: IONZ replay at LOW conviction (baseline behaviour) ==")
    # Low conviction trade: classic 1x breakeven still fires as before
    new_stop, mode = case(
        "low_conv_peak",
        entry_price=8.55,
        current_price=8.78,
        current_stop=8.37,
        atr=0.18,
        high_since_entry=8.78,
        conviction=0.70,
    )
    assert mode == "breakeven", "low conviction should still ratchet at 1x ATR"
    assert new_stop is not None and abs(new_stop - 8.56) < 0.02

    print("\n== Test 3: IONZ replay at HIGH conviction (fix) ==")
    # 97% conviction: breakeven disabled, chandelier at 4x ATR = 0.72
    # chandelier proposal = 8.78 - 0.72 = 8.06 which is < entry, rejected.
    new_stop, mode = case(
        "high_conv_peak",
        entry_price=8.55,
        current_price=8.78,
        current_stop=8.37,
        atr=0.18,
        high_since_entry=8.78,
        conviction=0.97,
    )
    assert new_stop is None, (
        f"97% conviction should NOT tighten at peak ~1.3 ATR above entry; "
        f"got {new_stop} {mode}"
    )

    print("\n== Test 4: Pullback from peak -- high conviction ==")
    # Price pulled back to 8.65 from 8.78 peak. Should still NOT tighten.
    new_stop, mode = case(
        "high_conv_pullback",
        entry_price=8.55,
        current_price=8.65,
        current_stop=8.37,
        atr=0.18,
        high_since_entry=8.78,
        conviction=0.97,
    )
    assert new_stop is None, (
        f"pullback on high conviction should NOT tighten; got {new_stop} {mode}"
    )

    print("\n== Test 5: Big runner at high conviction -- chandelier eventually engages ==")
    # Price at 9.50, high 9.50, atr 0.18, chand_mult 4 -> 9.50 - 0.72 = 8.78
    # Above entry 8.55, valid, above current_stop 8.37 -> locks in +0.23.
    new_stop, mode = case(
        "high_conv_runner",
        entry_price=8.55,
        current_price=9.50,
        current_stop=8.37,
        atr=0.18,
        high_since_entry=9.50,
        conviction=0.97,
    )
    assert new_stop is not None and mode == "chandelier"
    assert new_stop >= 8.70 and new_stop < 9.50

    print("\n== Test 6: Mid conviction (0.85) gets an intermediate trail ==")
    # breakeven_trigger = 2.5, chandelier_mult = 3.5
    # peak 8.78 = entry + 0.23; 2.5 * 0.18 = 0.45; not enough to trigger breakeven
    new_stop, mode = case(
        "mid_conv_peak",
        entry_price=8.55,
        current_price=8.78,
        current_stop=8.37,
        atr=0.18,
        high_since_entry=8.78,
        conviction=0.85,
    )
    assert new_stop is None, (
        f"0.85 conviction at +1.3 ATR should NOT trigger 2.5x breakeven; "
        f"got {new_stop} {mode}"
    )

    print("\n== Test 7: Floor when ATR is missing on trade row ==")
    # Simulates job_update_trailing_stops path: ATR defaulted to entry*0.02 = 0.17
    # 0.17 is close to the real IONZ ATR, so high conviction still sits tight.
    new_stop, mode = case(
        "missing_atr_high_conv",
        entry_price=8.55,
        current_price=8.78,
        current_stop=8.37,
        atr=8.55 * 0.02,
        high_since_entry=8.78,
        conviction=0.97,
    )
    assert new_stop is None

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
