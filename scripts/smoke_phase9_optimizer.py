"""Smoke test for Phase 9 parameter optimizer.

Tests:
  1. Schema migration: parameter_overrides has regime column
  2. Multi-param: optimizer processes atr_stop_multiplier + atr_tp_multiplier
  3. Per-regime: proposals generated per regime
  4. Rolling window: only recent trades considered
  5. Effective overrides: regime-specific with global fallback
  6. State persistence: regime round-trips through bot_state
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
from ai_trade.analysis.parameter_optimizer import review_and_adjust  # noqa: E402
from ai_trade.state_persistence import (  # noqa: E402
    apply_parameter_overrides,
    record_current_regime,
    get_current_regime,
)


def _make_cfg(**strategy_params) -> SimpleNamespace:
    """Build a minimal config namespace matching what the optimizer reads."""
    strategies = SimpleNamespace()
    for name, params in strategy_params.items():
        setattr(strategies, name, SimpleNamespace(**params))
    return SimpleNamespace(strategies=strategies)


def _seed_trades(db: Database, strategy: str, regime: str, n: int,
                 stop_quality: str = "too_tight",
                 exit_quality: str = "normal") -> None:
    """Insert n closed trades + trade_analysis rows for testing."""
    for i in range(n):
        tid = db.insert_trade(
            symbol=f"TEST{i}", strategy=strategy, side="buy",
            shares=10, entry_price=100.0, exit_price=101.0,
            pnl=10.0, pnl_pct=0.01, status="closed",
            hold_type="day",
        )
        db.insert_trade_analysis(
            trade_id=tid,
            entry_quality=0.7,
            stop_quality=stop_quality,
            exit_quality=exit_quality,
            market_regime=regime,
        )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        db = Database(db_path)

        # ── Test 1: Schema has regime column ─────────────────
        print("== Test 1: schema has regime column ==")
        with db._conn() as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(parameter_overrides)").fetchall()]
        assert "regime" in cols, f"Missing 'regime' column. Columns: {cols}"
        print(f"  columns: {cols}")

        # ── Test 2: set/get with regime ─────────────────────
        print("\n== Test 2: set/get parameter overrides with regime ==")
        db.set_parameter_override("momentum", "atr_stop_multiplier", "2.0", regime="", set_by="test")
        db.set_parameter_override("momentum", "atr_stop_multiplier", "2.5", regime="strong_bull", set_by="test")
        db.set_parameter_override("momentum", "atr_stop_multiplier", "1.5", regime="strong_bear", set_by="test")

        all_rows = db.get_parameter_overrides()
        assert len(all_rows) == 3, f"Expected 3 rows, got {len(all_rows)}"

        global_only = db.get_parameter_overrides(regime="")
        assert len(global_only) == 1 and global_only[0]["value"] == "2.0"

        bull_only = db.get_parameter_overrides(regime="strong_bull")
        assert len(bull_only) == 1 and bull_only[0]["value"] == "2.5"
        print("  set/get with regime: OK")

        # ── Test 3: effective overrides (regime + global fallback) ──
        print("\n== Test 3: effective overrides with fallback ==")
        db.set_parameter_override("vwap", "atr_stop_multiplier", "1.8", regime="", set_by="test")
        # No vwap override for strong_bull, so fallback to global
        eff = db.get_effective_overrides(regime="strong_bull")
        vwap_eff = [r for r in eff if r["strategy_name"] == "vwap"]
        assert len(vwap_eff) == 1 and vwap_eff[0]["value"] == "1.8"
        momentum_eff = [r for r in eff if r["strategy_name"] == "momentum"]
        assert len(momentum_eff) == 1 and momentum_eff[0]["value"] == "2.5"
        print("  fallback: vwap gets global 1.8, momentum gets bull-specific 2.5")

        # ── Test 4: Multi-param optimizer with per-regime grouping ──
        print("\n== Test 4: optimizer proposals by regime ==")
        # Clean slate DB for optimizer test
        db2 = Database(os.path.join(tmp, "test2.db"))
        cfg = _make_cfg(
            momentum={"atr_stop_multiplier": 2.0, "atr_tp_multiplier": 3.0},
            vwap={"atr_stop_multiplier": 1.5},
        )

        # Seed: 10 too_tight stops in strong_bull for momentum
        _seed_trades(db2, "momentum", "strong_bull", 10, stop_quality="too_tight")
        # Seed: 10 too_loose stops in strong_bear for momentum
        _seed_trades(db2, "momentum", "strong_bear", 10, stop_quality="too_loose")

        proposals = review_and_adjust(db2, cfg, min_trades=8, window=0, apply_changes=True)

        # Should have proposals for momentum across regimes
        momentum_props = [p for p in proposals if p["strategy"] == "momentum" and p["param"] == "atr_stop_multiplier"]
        regimes_proposed = {p["regime"] for p in momentum_props}
        print(f"  momentum atr_stop proposals for regimes: {regimes_proposed}")
        for p in momentum_props:
            print(f"    regime={p['regime'] or 'global'}: {p['old_value']} -> {p['new_value']} ({p['direction']})")

        # The global cohort has mixed signals (10 tight + 10 loose = 50/50) so
        # no global proposal. But regime-specific should fire.
        assert "strong_bull" in regimes_proposed, "Expected strong_bull proposal"
        assert "strong_bear" in regimes_proposed, "Expected strong_bear proposal"

        bull_prop = [p for p in momentum_props if p["regime"] == "strong_bull"][0]
        assert bull_prop["direction"] == "widen"
        bear_prop = [p for p in momentum_props if p["regime"] == "strong_bear"][0]
        assert bear_prop["direction"] == "tighten"

        # Verify override was written with regime
        overrides = db2.get_parameter_overrides(regime="strong_bull")
        mom_override = [r for r in overrides if r["strategy_name"] == "momentum"]
        assert len(mom_override) == 1 and float(mom_override[0]["value"]) == 2.1
        print("  bull widen -> 2.1, bear tighten -> confirmed")

        # ── Test 5: Rolling window ──────────────────────────
        print("\n== Test 5: rolling window limits trades ==")
        db3 = Database(os.path.join(tmp, "test3.db"))
        cfg3 = _make_cfg(vwap={"atr_stop_multiplier": 1.5})

        # Seed 20 old trades (just_right) then 10 recent too_tight
        _seed_trades(db3, "vwap", "neutral", 20, stop_quality="just_right")
        _seed_trades(db3, "vwap", "neutral", 10, stop_quality="too_tight")

        # Window=15 should only see the 10 recent too_tight + 5 just_right
        props_windowed = review_and_adjust(db3, cfg3, min_trades=8, window=15, apply_changes=False)
        vwap_props = [p for p in props_windowed if p["strategy"] == "vwap" and p["param"] == "atr_stop_multiplier"]
        # 10/15 too_tight = 67% > 60% -> should propose widen
        has_widen = any(p["direction"] == "widen" for p in vwap_props)
        assert has_widen, f"Expected widen proposal with window=15. Got: {vwap_props}"
        print("  window=15 -> widen proposal: OK")

        # Without window, 10/30 too_tight = 33% < 60% -> no proposal
        props_all = review_and_adjust(db3, cfg3, min_trades=8, window=0, apply_changes=False)
        vwap_all = [p for p in props_all if p["strategy"] == "vwap"
                    and p["param"] == "atr_stop_multiplier" and p["regime"] == "neutral"]
        has_widen_all = any(p["direction"] == "widen" for p in vwap_all)
        assert not has_widen_all, "Should NOT widen without window (10/30 = 33%)"
        print("  window=0 (all trades) -> no proposal: OK")

        # ── Test 6: Regime persistence round-trip ────────────
        print("\n== Test 6: regime persistence ==")
        record_current_regime(db, "weak_bull")
        got = get_current_regime(db)
        assert got == "weak_bull", f"Expected 'weak_bull', got '{got}'"
        print("  round-trip: OK")

        # ── Test 7: apply_parameter_overrides with regime ────
        print("\n== Test 7: apply_parameter_overrides regime-aware ==")
        cfg7 = _make_cfg(
            momentum={"atr_stop_multiplier": 1.0},
            vwap={"atr_stop_multiplier": 1.0},
        )
        applied = apply_parameter_overrides(cfg7, db, regime="strong_bull")
        momentum_applied = [a for a in applied if a["strategy"] == "momentum"]
        assert len(momentum_applied) == 1
        assert momentum_applied[0]["value"] == 2.5  # bull-specific
        vwap_applied = [a for a in applied if a["strategy"] == "vwap"]
        assert len(vwap_applied) == 1
        assert vwap_applied[0]["value"] == 1.8  # global fallback
        print(f"  applied {len(applied)} overrides with regime=strong_bull")
        print(f"  momentum -> {cfg7.strategies.momentum.atr_stop_multiplier}")
        print(f"  vwap -> {cfg7.strategies.vwap.atr_stop_multiplier}")

        print(f"\nSMOKE TEST PASSED (7/7)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
