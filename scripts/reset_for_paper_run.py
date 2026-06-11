"""Reset local state for a clean paper-trading run.

Use this BEFORE starting a fresh paper run (e.g. after resetting the
paper account on Alpaca's dashboard).  It:

  * Backs up the current SQLite database to data/backups/
  * Clears tables that should start empty: trades, day_trades, signals,
    decisions, ml_predictions, ml_features, strategy_weights, options_trades
  * Resets bot_state cursors (weighter cursor, drawdown tier, streak,
    starting equity) so the bot recomputes them from scratch
  * Preserves the trained ml_models registry — no need to retrain

It does NOT touch the Alpaca paper account itself.  Reset that manually
from the Alpaca dashboard before running this script, otherwise positions
on Alpaca's side won't match the empty local DB and sync_positions will
re-import them.

Usage:
    python scripts/reset_for_paper_run.py
    python scripts/reset_for_paper_run.py --confirm   # skip prompt
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "ai_trade.db"
BACKUP_DIR = DB_PATH.parent / "backups"

# Tables wiped to start fresh.  Anything not listed is preserved.
TABLES_TO_CLEAR = [
    "trades",
    "options_trades",
    "day_trades",
    "signals",
    "decisions",
    "ml_predictions",
    "ml_features",
    "strategy_weights",
]

# bot_state keys reset to their startup defaults.
STATE_KEYS_TO_RESET = [
    "weighter.trade_count_at_last_recalc",
    "risk.starting_equity",
    "risk.streak_scale",
    "risk.streak_sample_size",
    "risk.drawdown_tier",
    "analysis.last_pattern_scan",
    "analysis.last_pattern_clusters",
    "analysis.last_optimizer_run",
    "analysis.last_optimizer_proposals",
]


def backup_db() -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"ai_trade_{stamp}.db"
    shutil.copy2(DB_PATH, dest)
    return dest


def reset(conn: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tbl in TABLES_TO_CLEAR:
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        except sqlite3.OperationalError:
            counts[tbl] = -1
            continue
        conn.execute(f"DELETE FROM {tbl}")
        counts[tbl] = n
    for key in STATE_KEYS_TO_RESET:
        conn.execute("DELETE FROM bot_state WHERE key = ?", (key,))
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="skip the prompt")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}", file=sys.stderr)
        return 1

    if not args.confirm:
        print(f"This will WIPE the following tables in {DB_PATH}:")
        for t in TABLES_TO_CLEAR:
            print(f"   - {t}")
        print(f"And reset {len(STATE_KEYS_TO_RESET)} bot_state keys.")
        print()
        ans = input("Type 'yes' to proceed: ").strip().lower()
        if ans != "yes":
            print("Aborted.")
            return 1

    backup = backup_db()
    print(f"Backup written: {backup}")

    with sqlite3.connect(DB_PATH) as conn:
        counts = reset(conn)
        conn.commit()

    print("\nCleared rows:")
    for tbl, n in counts.items():
        marker = "(missing)" if n < 0 else f"{n} rows"
        print(f"   {tbl:<22} {marker}")
    print("\nDone.  Restart the bot to begin a fresh run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
