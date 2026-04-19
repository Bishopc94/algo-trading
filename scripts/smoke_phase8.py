"""Phase 8 smoke test -- SmartPDTPlanner end-to-end.

Exercises:
  1. Dynamic threshold across Mon..Fri with fixed slot count
  2. Slot-remaining bump (3 vs 1 slots)
  3. EV bump from recent day-trade win rate (win-heavy, neutral, loss-heavy)
  4. Slots=0 produces "frozen" stance
  5. is_swing_compatible gating (momentum yes, orb/vwap no)
  6. convert_day_to_swing preserves entry, widens stop + target, flips HoldType
  7. PDTPlan reasons list populated for key edge cases
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, SRC)

from ai_trade.monitoring.database import Database  # noqa: E402
from ai_trade.risk.smart_pdt import (  # noqa: E402
    SmartPDTPlanner,
    convert_day_to_swing,
    dynamic_day_trade_threshold,
    estimate_day_trade_ev_bump,
    is_swing_compatible,
)
from ai_trade.strategy.base import HoldType, Signal  # noqa: E402


class FakePDTManager:
    def __init__(self, remaining: int, reserve: int = 1):
        self._remaining = remaining
        self.config = SimpleNamespace(
            max_day_trades=3,
            day_trade_reserve=reserve,
            min_conviction_for_day_trade=0.80,
        )

    def day_trades_remaining(self) -> int:
        return self._remaining


def _mk_signal(strategy="momentum", conv=0.78, hold=HoldType.DAY):
    return Signal(
        symbol="TEST",
        strategy_name=strategy,
        direction="long",
        entry_price=100.0,
        stop_loss_price=98.0,
        take_profit_price=104.0,
        conviction=conv,
        hold_type=hold,
    )


def _seed_day_trades(db, wins: int, losses: int) -> None:
    for i in range(wins):
        db._insert(
            "trades", symbol=f"DW{i}", strategy="momentum", side="long",
            entry_price=100.0, stop_loss=98.0, take_profit=104.0,
            shares=10, status="closed", pnl=5.0, hold_type="day",
        )
    for i in range(losses):
        db._insert(
            "trades", symbol=f"DL{i}", strategy="momentum", side="long",
            entry_price=100.0, stop_loss=98.0, take_profit=104.0,
            shares=10, status="closed", pnl=-5.0, hold_type="day",
        )


def main() -> int:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = tmp.name
    try:
        db = Database(db_path)

        # == Test 1: Threshold across weekdays (3 slots) ==
        print("== Test 1: DoW threshold (3 slots, no EV) ==")
        for dow, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]):
            th, breakdown = dynamic_day_trade_threshold(
                slots_remaining=3, day_of_week=dow, base=0.80,
            )
            print(f"  {name} dow={dow} threshold={th:.3f} dow_bump={breakdown['dow_bump']}")
        mon_th, _ = dynamic_day_trade_threshold(3, 0, base=0.80)
        fri_th, _ = dynamic_day_trade_threshold(3, 4, base=0.80)
        assert mon_th > fri_th, "Monday should be stingier than Friday"

        # == Test 2: Slot-remaining bump ==
        print("\n== Test 2: Slot bump (Wed, 3 vs 1 slots) ==")
        t3, _ = dynamic_day_trade_threshold(3, 2, base=0.80)
        t1, _ = dynamic_day_trade_threshold(1, 2, base=0.80)
        print(f"  3 slots -> {t3:.3f}, 1 slot -> {t1:.3f}")
        assert t1 > t3, "1 slot should be stingier than 3"

        # == Test 3: EV bump (win-heavy, neutral, loss-heavy) ==
        print("\n== Test 3: EV bump from recent day trades ==")
        assert estimate_day_trade_ev_bump(db) == 0.0  # cold start
        _seed_day_trades(db, wins=12, losses=3)
        bump_win = estimate_day_trade_ev_bump(db)
        print(f"  win-heavy (12W/3L): ev_bump={bump_win}")
        assert bump_win == -0.02
        # Reset
        with db._conn() as conn:
            conn.execute("DELETE FROM trades")
        _seed_day_trades(db, wins=3, losses=12)
        bump_loss = estimate_day_trade_ev_bump(db)
        print(f"  loss-heavy (3W/12L): ev_bump={bump_loss}")
        assert bump_loss == +0.03
        with db._conn() as conn:
            conn.execute("DELETE FROM trades")

        # == Test 4: Zero slots -> frozen ==
        print("\n== Test 4: Frozen stance (0 slots) ==")
        cfg = SimpleNamespace(min_conviction_for_day_trade=0.80)
        planner = SmartPDTPlanner(cfg, db)
        plan_frozen = planner.plan_cycle(
            FakePDTManager(remaining=1),  # 1 - 1 reserve = 0
            now=datetime(2026, 4, 15, 14, 30, tzinfo=timezone.utc),  # Wed
        )
        print(
            f"  slots={plan_frozen.slots_remaining} "
            f"stance={plan_frozen.stance} th={plan_frozen.dynamic_threshold}"
        )
        assert plan_frozen.stance == "frozen"
        assert plan_frozen.slots_remaining == 0
        assert plan_frozen.dynamic_threshold > 1.0

        # == Test 5: Swing compatibility ==
        print("\n== Test 5: is_swing_compatible ==")
        for s in ["momentum", "bb_squeeze", "ema_crossover"]:
            assert is_swing_compatible(s), f"{s} should be swing-compatible"
        for s in ["orb", "vwap"]:
            assert not is_swing_compatible(s), f"{s} should NOT be swing-compatible"
        print("  ok")

        # == Test 6: convert_day_to_swing ==
        print("\n== Test 6: convert_day_to_swing ==")
        sig = _mk_signal(strategy="momentum", conv=0.70, hold=HoldType.DAY)
        orig_entry = sig.entry_price
        orig_stop = sig.stop_loss_price
        orig_target = sig.take_profit_price
        orig_rr = (orig_target - orig_entry) / (orig_entry - orig_stop)
        convert_day_to_swing(sig)
        new_rr = (sig.take_profit_price - sig.entry_price) / (
            sig.entry_price - sig.stop_loss_price
        )
        print(
            f"  entry={sig.entry_price} stop {orig_stop}->{sig.stop_loss_price} "
            f"target {orig_target}->{sig.take_profit_price}"
        )
        print(f"  original R:R={orig_rr:.2f} new R:R={new_rr:.2f}")
        assert sig.hold_type == HoldType.SWING
        assert sig.entry_price == orig_entry
        assert sig.stop_loss_price < orig_stop  # farther from entry (long)
        assert sig.take_profit_price > orig_target
        assert abs(new_rr - orig_rr) < 0.01  # R:R preserved
        assert sig.metadata["pdt_conversion"]["original_hold"] == "day"

        # == Test 7: Plan reasons populated ==
        print("\n== Test 7: Plan reasons ==")
        # Monday with 1 usable slot (2 remaining - 1 reserve) should say "early-week"
        mon_plan = planner.plan_cycle(
            FakePDTManager(remaining=2),  # -> 1 usable
            now=datetime(2026, 4, 13, 10, 0, tzinfo=timezone.utc),  # Mon
        )
        print(f"  Mon 1-slot: stance={mon_plan.stance} reasons={mon_plan.reasons}")
        assert any("early-week" in r or "Mon" in r for r in mon_plan.reasons)
        assert any("slot" in r.lower() for r in mon_plan.reasons)

        # Friday with 2 slots should say "use or lose"
        fri_plan = planner.plan_cycle(
            FakePDTManager(remaining=3),  # -> 2 usable
            now=datetime(2026, 4, 17, 14, 0, tzinfo=timezone.utc),  # Fri
        )
        print(f"  Fri 2-slot: stance={fri_plan.stance} reasons={fri_plan.reasons}")
        assert any("Friday" in r or "use" in r for r in fri_plan.reasons)
        assert fri_plan.dynamic_threshold < mon_plan.dynamic_threshold

        print("\nSMOKE TEST PASSED")
        return 0
    finally:
        try:
            os.remove(db_path)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
