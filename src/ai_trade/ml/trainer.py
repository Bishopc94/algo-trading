"""Offline training of the signal-quality classifier.

The goal: given a pool of closed trades and the feature snapshots we
captured at entry, learn a probability that a future signal with
similar features will be profitable.  The output probability becomes
an ensemble modifier on the rule-based conviction at inference time
(see predictor.py and strategy/signal.py).

Why sklearn GradientBoosting instead of LightGBM:
    LightGBM isn't installed in this environment and adding a new
    system dependency for Phase 5 would stall progress.  Sklearn's
    GradientBoostingClassifier is shipped in the existing sklearn
    install, trains in under a second on <1000 rows, and produces
    calibrated probabilities out of the box.  When we accumulate
    enough data (>5k closed trades) we can swap to LightGBM for
    speed — the feature pipeline and predictor API won't change.

Training pipeline:
    1. Query `ml_features` joined with `trades` on trade_id.
    2. Label each row: y = 1 if pnl > 0 else 0.
    3. Time-order the data and split 80/20 (most recent 20% held out).
    4. Fit GradientBoostingClassifier.
    5. Measure train + validation accuracy.
    6. Save model to models/signal_quality_v<N>.joblib via joblib.
    7. Register in `ml_models` table with is_active=1.
    8. Deactivate prior versions (only one active at a time).
    9. Return a summary dict (used by the caller to log training runs).

Cold start:
    If `ml_features` has fewer than `min_trades` labelled rows, the
    trainer returns a structured "insufficient_data" result and does
    NOT write a model row.  Live inference falls back to rule-based
    conviction in that case (see predictor.py).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_trade.monitoring.logger import get_logger

if TYPE_CHECKING:
    from ai_trade.monitoring.database import Database

log = get_logger(__name__)

# Models live in <project_root>/models/ — the path is resolved relative
# to this file so it works regardless of cwd.
_MODELS_DIR = Path(__file__).resolve().parents[3] / "models"

MODEL_NAME = "signal_quality"


def _load_labelled_dataset(database: Database) -> list[dict]:
    """Join ml_features with trades to build (features, outcome) rows.

    Returns list of dicts: {trade_id, features: dict, pnl, entry_time}
    ordered by entry_time ascending.  Rows without a resolved pnl are
    skipped (still open, or the outcome hasn't been logged yet).
    """
    with database._conn() as conn:  # noqa: SLF001 — direct read for speed
        rows = conn.execute(
            """
            SELECT f.trade_id, f.features, t.pnl, t.entry_time, t.status
            FROM ml_features f
            JOIN trades t ON t.id = f.trade_id
            WHERE t.status = 'closed'
              AND t.pnl IS NOT NULL
            ORDER BY t.entry_time ASC
            """
        ).fetchall()

    dataset: list[dict] = []
    for r in rows:
        try:
            feat = json.loads(r["features"])
        except (TypeError, ValueError):
            continue
        dataset.append({
            "trade_id": r["trade_id"],
            "features": feat,
            "pnl": float(r["pnl"]),
            "entry_time": r["entry_time"],
        })
    return dataset


def _next_version(database: Database, model_name: str) -> int:
    with database._conn() as conn:
        row = conn.execute(
            "SELECT MAX(version) FROM ml_models WHERE model_name = ?",
            (model_name,),
        ).fetchone()
    cur = row[0] if row and row[0] is not None else 0
    return int(cur) + 1


def _deactivate_prior_versions(database: Database, model_name: str) -> None:
    with database._conn() as conn:
        conn.execute(
            "UPDATE ml_models SET is_active = 0 WHERE model_name = ?",
            (model_name,),
        )


def train_signal_quality_model(
    database: Database,
    min_trades: int = 30,
    val_split: float = 0.2,
) -> dict[str, Any]:
    """Train a new signal-quality model on all closed labelled trades.

    Args:
        database: Database instance (reads ml_features + trades, writes ml_models).
        min_trades: Skip training if fewer than this many labelled trades exist.
        val_split: Fraction of the most-recent data to hold out as validation.

    Returns:
        Dict with: status, trades_used, train_accuracy, val_accuracy,
        model_path, version (when training runs), or status=insufficient_data
        / training_failed / single_class in error paths.
    """
    # Import sklearn lazily so that importing ai_trade.ml.trainer at
    # module init doesn't crash environments without sklearn (e.g.
    # future lean deployments).  joblib is the standard serializer for
    # sklearn estimators.
    try:
        import joblib
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score
        import numpy as np
    except ImportError as e:
        log.error("ml_trainer_missing_dependency", error=str(e))
        return {"status": "missing_dependency", "error": str(e)}

    from ai_trade.ml.features import FEATURE_ORDER, features_to_vector

    dataset = _load_labelled_dataset(database)
    if len(dataset) < min_trades:
        log.info(
            "ml_trainer_insufficient_data",
            have=len(dataset),
            need=min_trades,
        )
        return {
            "status": "insufficient_data",
            "trades_available": len(dataset),
            "trades_required": min_trades,
        }

    X = np.array([features_to_vector(row["features"]) for row in dataset])
    y = np.array([1 if row["pnl"] > 0 else 0 for row in dataset])

    # Guard against single-class datasets (e.g. 30 consecutive losses).
    # GradientBoostingClassifier would fit fine but the resulting model
    # only ever predicts that one class — useless for ranking and
    # misleading in metrics.
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        log.warning(
            "ml_trainer_single_class",
            class_=int(unique_classes[0]),
            rows=len(y),
        )
        return {
            "status": "single_class",
            "class": int(unique_classes[0]),
            "trades_used": len(y),
        }

    # Time-ordered split — most recent 20% is the validation set.
    # We rely on _load_labelled_dataset ordering the rows by entry_time
    # ASC so taking a tail slice is equivalent to a walk-forward split.
    split_idx = int(len(X) * (1 - val_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    if len(X_val) == 0 or len(np.unique(y_train)) < 2:
        # Tiny dataset edge case — skip the val split and train on all.
        X_train, y_train = X, y
        X_val, y_val = X, y

    try:
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        )
        model.fit(X_train, y_train)
    except Exception as e:
        log.exception("ml_trainer_fit_failed")
        return {"status": "training_failed", "error": str(e)}

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_acc = float(accuracy_score(y_train, train_pred))
    val_acc = float(accuracy_score(y_val, val_pred))

    # Persist model to disk.  joblib is standard for sklearn — it
    # compresses numpy arrays and is faster than pickle for this.
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    version = _next_version(database, MODEL_NAME)
    filename = f"{MODEL_NAME}_v{version}.joblib"
    model_path = _MODELS_DIR / filename

    payload = {
        "model": model,
        "feature_order": FEATURE_ORDER,
        "trained_at": datetime.utcnow().isoformat(),
        "version": version,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "training_trades": len(X),
    }
    joblib.dump(payload, model_path)

    _deactivate_prior_versions(database, MODEL_NAME)
    try:
        database.insert_ml_model(
            model_name=MODEL_NAME,
            version=version,
            trained_at=payload["trained_at"],
            training_trades=len(X),
            backtest_accuracy=val_acc,
            is_active=1,
            model_path=str(model_path),
        )
    except Exception:
        log.exception("ml_model_registry_insert_failed")

    log.info(
        "ml_trainer_success",
        version=version,
        trades_used=len(X),
        train_accuracy=round(train_acc, 4),
        val_accuracy=round(val_acc, 4),
        model_path=str(model_path),
    )

    return {
        "status": "ok",
        "version": version,
        "trades_used": int(len(X)),
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "model_path": str(model_path),
    }
