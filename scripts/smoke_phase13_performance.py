"""Smoke test for Phase 13: Performance Optimization.

Tests:
  1. CycleTimer phase tracking
  2. CycleTimer budget detection (within budget)
  3. CycleTimer budget detection (over budget)
  4. CycleTimer summary_line format
  5. CycleTimer reset clears phases
  6. WAL mode enabled on database connection
  7. Database cleanup_old_data works
  8. ThreadPoolExecutor parallel pattern (functional test)
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, SRC)

from ai_trade.monitoring.cycle_timer import CycleTimer, PhaseResult
from ai_trade.monitoring.database import Database


def main() -> int:
    # -- Test 1: CycleTimer phase tracking --
    print("== Test 1: CycleTimer phase tracking ==")
    timer = CycleTimer()
    with timer.phase("fetch"):
        time.sleep(0.01)
    with timer.phase("evaluate"):
        time.sleep(0.01)
    with timer.phase("execute"):
        time.sleep(0.005)

    assert len(timer._phases) == 3, f"Expected 3 phases, got {len(timer._phases)}"
    names = [p.name for p in timer._phases]
    assert names == ["fetch", "evaluate", "execute"], f"Got {names}"
    for p in timer._phases:
        assert p.elapsed_ms > 0, f"Phase {p.name} has zero elapsed time"
    print(f"  3 phases tracked: {', '.join(f'{p.name}={p.elapsed_ms:.1f}ms' for p in timer._phases)}")

    # -- Test 2: Within budget --
    print("\n== Test 2: CycleTimer within budget ==")
    timer2 = CycleTimer(budget_ms=5000)
    with timer2.phase("fast"):
        time.sleep(0.005)
    assert timer2.within_budget, f"Should be within 5s budget, total={timer2.total_ms}ms"
    print(f"  total={timer2.total_ms:.1f}ms budget=5000ms within_budget=True")

    # -- Test 3: Over budget --
    print("\n== Test 3: CycleTimer over budget ==")
    timer3 = CycleTimer(budget_ms=1)  # 1ms budget -> guaranteed to exceed
    with timer3.phase("slow"):
        time.sleep(0.01)
    assert not timer3.within_budget, f"Should exceed 1ms budget, total={timer3.total_ms}ms"
    print(f"  total={timer3.total_ms:.1f}ms budget=1ms within_budget=False")

    # -- Test 4: summary_line format --
    print("\n== Test 4: summary_line format ==")
    timer4 = CycleTimer(budget_ms=50000)
    with timer4.phase("fetch"):
        time.sleep(0.005)
    with timer4.phase("evaluate"):
        time.sleep(0.005)
    line = timer4.summary_line()
    assert "fetch=" in line, f"Missing 'fetch=' in: {line}"
    assert "evaluate=" in line, f"Missing 'evaluate=' in: {line}"
    assert "total=" in line, f"Missing 'total=' in: {line}"
    assert "OVER BUDGET" not in line, f"Should not be over budget: {line}"
    print(f"  {line}")

    # Over budget line
    timer4b = CycleTimer(budget_ms=1)
    with timer4b.phase("slow"):
        time.sleep(0.01)
    line_b = timer4b.summary_line()
    assert "OVER BUDGET" in line_b, f"Should show OVER BUDGET: {line_b}"
    print(f"  {line_b}")

    # -- Test 5: reset clears phases --
    print("\n== Test 5: CycleTimer reset ==")
    timer5 = CycleTimer()
    with timer5.phase("a"):
        pass
    with timer5.phase("b"):
        pass
    assert len(timer5._phases) == 2
    timer5.reset()
    assert len(timer5._phases) == 0, f"Reset should clear phases, got {len(timer5._phases)}"
    assert timer5.total_ms == 0.0
    print("  reset clears all phases")

    # -- Test 6: WAL mode enabled --
    print("\n== Test 6: WAL mode enabled ==")
    tmpdir6 = tempfile.mkdtemp()
    db_path6 = os.path.join(tmpdir6, "test_wal.db")
    db6 = Database(db_path=db_path6)
    # Check WAL mode by reading the PRAGMA
    conn6 = sqlite3.connect(db_path6)
    mode = conn6.execute("PRAGMA journal_mode").fetchone()[0]
    conn6.close()
    del db6  # release any internal refs
    assert mode == "wal", f"Expected WAL mode, got '{mode}'"
    print(f"  journal_mode={mode}")

    # -- Test 7: cleanup_old_data --
    print("\n== Test 7: cleanup_old_data ==")
    tmpdir7 = tempfile.mkdtemp()
    db_path7 = os.path.join(tmpdir7, "test_cleanup.db")
    db7 = Database(db_path=db_path7)
    # Insert some old decisions manually
    conn7 = sqlite3.connect(db_path7)
    conn7.execute("PRAGMA journal_mode=WAL")
    # Insert a decision with a very old timestamp
    conn7.execute(
        "INSERT INTO decisions (symbol, strategy, action, decision_type, timestamp) "
        "VALUES (?, ?, ?, ?, datetime('now', '-120 days'))",
        ("TEST", "momentum", "reject", "evaluate"),
    )
    # Insert a recent decision
    conn7.execute(
        "INSERT INTO decisions (symbol, strategy, action, decision_type, timestamp) "
        "VALUES (?, ?, ?, ?, datetime('now'))",
        ("TEST2", "momentum", "execute", "evaluate"),
    )
    conn7.commit()
    # Verify both exist
    count = conn7.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    assert count == 2, f"Expected 2 decisions, got {count}"
    conn7.close()

    # Run cleanup with 90-day retention
    deleted = db7.cleanup_old_data(retention_days=90)
    assert deleted.get("decisions", 0) == 1, f"Expected 1 deleted, got {deleted}"
    print(f"  deleted={deleted}")

    # Verify the recent one survived
    conn7b = sqlite3.connect(db_path7)
    remaining = conn7b.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    conn7b.close()
    del db7
    assert remaining == 1, f"Expected 1 remaining, got {remaining}"
    print(f"  1 old row deleted, 1 recent row kept")

    # -- Test 8: ThreadPoolExecutor parallel pattern --
    print("\n== Test 8: parallel fetch pattern ==")
    results = {}

    def task_a():
        time.sleep(0.05)
        return "daily_bars"

    def task_b():
        time.sleep(0.05)
        return "intraday_bars"

    def task_c():
        time.sleep(0.05)
        return "news"

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as pool:
        fa = pool.submit(task_a)
        fb = pool.submit(task_b)
        fc = pool.submit(task_c)
        results["a"] = fa.result()
        results["b"] = fb.result()
        results["c"] = fc.result()
    elapsed = (time.perf_counter() - t0) * 1000

    assert results == {"a": "daily_bars", "b": "intraday_bars", "c": "news"}
    # Sequential would take ~150ms, parallel should be ~50-80ms
    assert elapsed < 130, f"Parallel should be <130ms, took {elapsed:.0f}ms (sequential would be ~150ms)"
    print(f"  3 tasks in {elapsed:.0f}ms (parallel speedup confirmed)")

    print(f"\nSMOKE TEST PASSED (8/8)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
