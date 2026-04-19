"""Phase 7 smoke test — DynamicRiskController end-to-end.

Exercises:
  1. Conviction ladder sizing (0.55 / 0.80 / 0.92)
  2. Losing streak halves risk (10 losses)
  3. Winning streak increases risk
  4. Drawdown tier 1 (3-5% down) -> 0.5x
  5. Drawdown tier 2 (>= 5% down) -> halt new entries
  6. Regime strong_bear blocks new longs + cuts positions
  7. High-conviction (>=0.90) unlocks 50% concentration cap
  8. bot_state persistence roundtrip (simulate restart)
"""
from __future__ import annotations

import os
import sys
import tempfile
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, SRC)

from ai_trade.monitoring.database import Database  # noqa: E402
from ai_trade.risk.dynamic_risk import DynamicRiskController  # noqa: E402
from ai_trade.risk.position_sizer import PositionSizer  # noqa: E402
from ai_trade.risk.risk_manager import RiskManager  # noqa: E402
from ai_trade.strategy.base import HoldType, Signal  # noqa: E402


def _mk_signal(conviction=0.80, entry=50.0, stop=48.0, direction="long"):
    return Signal(
        symbol="TEST",
        strategy_name="smoke",
        direction=direction,
        entry_price=entry,
        stop_loss_price=stop,
        take_profit_price=entry * 1.05,
        conviction=conviction,
        hold_type=HoldType.SWING,
    )


def _mk_cfg():
    return SimpleNamespace(
        max_risk_per_trade_pct=0.02,
        max_position_pct=0.25,
        max_open_positions=5,
        daily_loss_limit_pct=0.05,
        max_portfolio_heat_pct=0.08,
    )


def _seed_trades(db, wins: int, losses: int) -> None:
    for i in range(wins):
        db._insert(
            "trades",
            symbol=f"W{i}",
            strategy="smoke",
            side="long",
            entry_price=10.0,
            stop_loss=9.0,
            take_profit=12.0,
            shares=10,
            status="closed",
            pnl=5.0,
            hold_type="swing",
        )
    for i in range(losses):
        db._insert(
            "trades",
            symbol=f"L{i}",
            strategy="smoke",
            side="long",
            entry_price=10.0,
            stop_loss=9.0,
            take_profit=12.0,
            shares=10,
            status="closed",
            pnl=-5.0,
            hold_type="swing",
        )


def main() -> int:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = tmp.name
    try:
        db = Database(db_path)
        cfg = _mk_cfg()
        equity = 1000.0

        # Controller with no streak history
        ctrl = DynamicRiskController(cfg, db)
        ctrl.set_starting_equity(equity)
        ctrl.refresh_streak()
        ctrl.refresh_drawdown(equity)

        sizer = PositionSizer(cfg)

        # == Test 1: Conviction ladder ==
        print("== Test 1: Conviction ladder ==")
        for conv in [0.55, 0.70, 0.80, 0.92]:
            snap = ctrl.snapshot_for_signal(conv, "neutral", equity)
            sig = _mk_signal(conviction=conv)
            shares = sizer.calculate_shares(
                sig, equity, equity,
                risk_scale=snap.total_risk_scale,
                max_position_pct_override=snap.max_position_pct,
            )
            print(
                f"  conv={conv}: conv_scale={snap.conviction_scale} "
                f"total={snap.total_risk_scale:.2f} "
                f"max_pct={snap.max_position_pct:.2f} "
                f"shares={shares} high_conv={snap.high_conviction_override}"
            )

        # == Test 2: Losing streak ==
        print("\n== Test 2: Losing streak (10 losses) ==")
        _seed_trades(db, wins=0, losses=10)
        ctrl.refresh_streak()
        snap = ctrl.snapshot_for_signal(0.80, "neutral", equity)
        print(f"  streak_scale={snap.streak_scale} sample={ctrl._streak_sample}")
        assert snap.streak_scale == 0.5, f"expected 0.5, got {snap.streak_scale}"

        # == Test 3: Winning streak ==
        print("\n== Test 3: Winning streak (7 wins, 3 losses) ==")
        with db._conn() as conn:
            conn.execute("DELETE FROM trades")
        _seed_trades(db, wins=7, losses=3)
        ctrl.refresh_streak()
        snap = ctrl.snapshot_for_signal(0.80, "neutral", equity)
        print(f"  streak_scale={snap.streak_scale}")
        assert snap.streak_scale == 1.5, f"expected 1.5, got {snap.streak_scale}"

        # == Test 4: Drawdown tier 1 (half-size) ==
        print("\n== Test 4: Drawdown tier 1 (-3.5%) ==")
        dd_eq = equity * 0.965
        ctrl.refresh_drawdown(dd_eq)
        snap = ctrl.snapshot_for_signal(0.80, "neutral", dd_eq)
        print(
            f"  tier={snap.drawdown_tier} dd_scale={snap.drawdown_scale} "
            f"allow_new={snap.allow_new_entries}"
        )
        assert snap.drawdown_tier == 1
        assert snap.drawdown_scale == 0.5
        assert snap.allow_new_entries is True

        # == Test 5: Drawdown tier 2 (halt) ==
        print("\n== Test 5: Drawdown tier 2 (-6%) ==")
        dd_eq = equity * 0.94
        ctrl.refresh_drawdown(dd_eq)
        snap = ctrl.snapshot_for_signal(0.80, "neutral", dd_eq)
        print(
            f"  tier={snap.drawdown_tier} allow_new={snap.allow_new_entries}"
        )
        assert snap.drawdown_tier == 2
        assert snap.allow_new_entries is False

        # Reset drawdown for subsequent tests
        ctrl.refresh_drawdown(equity)

        # == Test 6: strong_bear blocks new longs ==
        print("\n== Test 6: strong_bear regime ==")
        snap = ctrl.snapshot_for_signal(0.80, "strong_bear", equity)
        print(
            f"  regime_scale={snap.regime_scale} "
            f"max_positions={snap.max_open_positions} "
            f"allow_longs={snap.allow_new_longs}"
        )
        assert snap.allow_new_longs is False
        assert snap.max_open_positions == max(1, 5 - 2)

        # strong_bull bonus
        snap = ctrl.snapshot_for_signal(0.80, "strong_bull", equity)
        print(
            f"  strong_bull regime_scale={snap.regime_scale} "
            f"max_positions={snap.max_open_positions}"
        )
        assert snap.max_open_positions == 5 + 2

        # == Test 7: High-conviction concentration cap ==
        print("\n== Test 7: High-conviction (0.92) concentration ==")
        snap = ctrl.snapshot_for_signal(0.92, "neutral", equity)
        print(
            f"  max_pct={snap.max_position_pct} override={snap.high_conviction_override}"
        )
        assert snap.max_position_pct == 0.50
        assert snap.high_conviction_override is True

        # == Test 8: bot_state persistence roundtrip ==
        print("\n== Test 8: Persistence roundtrip ==")
        streak_before = ctrl.streak_scale
        se_before = ctrl.starting_equity
        ctrl2 = DynamicRiskController(cfg, db)
        print(
            f"  before: streak={streak_before} starting_eq={se_before}"
        )
        print(
            f"  after:  streak={ctrl2.streak_scale} starting_eq={ctrl2.starting_equity}"
        )
        assert ctrl2.streak_scale == streak_before
        assert ctrl2.starting_equity == se_before

        # == Test 9: RiskManager drawdown breaker integration ==
        # Raise daily_loss_limit so the drawdown breaker (tier2 at 5%)
        # fires alone without being preempted by the hard daily cap.
        print("\n== Test 9: RiskManager approve_trade + drawdown breaker ==")
        cfg9 = _mk_cfg()
        cfg9.daily_loss_limit_pct = 0.20
        risk_mgr = RiskManager(cfg9, db, dynamic_controller=ctrl2)
        risk_mgr.set_starting_equity(equity)
        # Tier 2 drawdown should reject
        ctrl2.refresh_drawdown(equity * 0.94)
        sig = _mk_signal(conviction=0.80)
        ok, reason = risk_mgr.approve_trade(
            signal=sig, shares=10, current_equity=equity * 0.94,
            available_cash=equity, open_positions_count=0,
            open_trades=[],
        )
        print(f"  tier2 -> approved={ok} reason={reason}")
        assert ok is False
        assert "drawdown" in reason.lower()

        # Normal equity should approve
        ctrl2.refresh_drawdown(equity)
        ok, reason = risk_mgr.approve_trade(
            signal=sig, shares=10, current_equity=equity,
            available_cash=equity, open_positions_count=0,
            open_trades=[],
        )
        print(f"  normal -> approved={ok} reason={reason}")
        assert ok is True

        print("\nSMOKE TEST PASSED")
        return 0
    finally:
        try:
            db.close()
        except Exception:
            pass
        try:
            os.remove(db_path)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
