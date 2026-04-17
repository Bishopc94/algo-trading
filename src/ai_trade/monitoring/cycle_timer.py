"""Cycle-phase timing — lightweight decorator + per-cycle budget log.

Tracks wall-clock time for each named phase of a trading cycle (scan,
fetch, evaluate, rank, execute) and logs a summary line at the end:

    cycle_timing | scan=120ms fetch=2100ms evaluate=340ms rank=5ms execute=890ms total=3455ms budget_ok=True

No external dependencies beyond the standard library and our logger.
"""
from __future__ import annotations

import functools
import threading
import time
from dataclasses import dataclass, field

from ai_trade.monitoring.logger import get_logger

log = get_logger(__name__)

# Default budget: entire cycle should finish in <10 seconds
DEFAULT_BUDGET_MS = 10_000


@dataclass
class PhaseResult:
    """Timing result for a single phase."""
    name: str
    elapsed_ms: float


@dataclass
class CycleTimer:
    """Accumulates phase timings for a single evaluation cycle.

    Usage::

        timer = CycleTimer()
        with timer.phase("fetch"):
            data = fetch_all()
        with timer.phase("evaluate"):
            signals = evaluate(data)
        timer.log_summary()
    """
    budget_ms: float = DEFAULT_BUDGET_MS
    _phases: list[PhaseResult] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    class _PhaseContext:
        """Context manager that times a named phase."""
        def __init__(self, timer: CycleTimer, name: str):
            self._timer = timer
            self._name = name
            self._start: float = 0

        def __enter__(self):
            self._start = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = (time.perf_counter() - self._start) * 1000
            with self._timer._lock:
                self._timer._phases.append(PhaseResult(self._name, round(elapsed, 1)))
            return False  # don't suppress exceptions

    def phase(self, name: str) -> _PhaseContext:
        """Return a context manager that times *name*."""
        return self._PhaseContext(self, name)

    @property
    def total_ms(self) -> float:
        return round(sum(p.elapsed_ms for p in self._phases), 1)

    @property
    def within_budget(self) -> bool:
        return self.total_ms <= self.budget_ms

    def summary_dict(self) -> dict:
        """Return phase timings as a flat dict for structured logging."""
        d: dict = {}
        for p in self._phases:
            d[p.name] = f"{p.elapsed_ms:.0f}ms"
        d["total"] = f"{self.total_ms:.0f}ms"
        d["budget_ok"] = self.within_budget
        return d

    def summary_line(self) -> str:
        """One-line human-readable summary for console output."""
        parts = [f"{p.name}={p.elapsed_ms:.0f}ms" for p in self._phases]
        parts.append(f"total={self.total_ms:.0f}ms")
        if not self.within_budget:
            parts.append(f"OVER BUDGET ({self.budget_ms:.0f}ms)")
        return " | ".join(parts)

    def log_summary(self) -> None:
        """Emit a structured log line with all phase timings."""
        if not self._phases:
            return
        log.info("cycle_timing", **self.summary_dict())
        if not self.within_budget:
            log.warning(
                "cycle_over_budget",
                total_ms=self.total_ms,
                budget_ms=self.budget_ms,
                slowest_phase=max(self._phases, key=lambda p: p.elapsed_ms).name,
            )

    def reset(self) -> None:
        """Clear all recorded phases for a fresh cycle."""
        with self._lock:
            self._phases.clear()


def timed_phase(phase_name: str):
    """Decorator that times a method and appends the result to ``self._cycle_timer``.

    The decorated class must have a ``_cycle_timer: CycleTimer`` attribute.

    Usage::

        class TradingBot:
            def __init__(self):
                self._cycle_timer = CycleTimer()

            @timed_phase("fetch")
            def _fetch_data(self):
                ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            timer: CycleTimer | None = getattr(self, "_cycle_timer", None)
            if timer is None:
                return fn(self, *args, **kwargs)
            with timer.phase(phase_name):
                return fn(self, *args, **kwargs)
        return wrapper
    return decorator
