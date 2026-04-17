"""Runtime inference for the signal-quality classifier.

The predictor is designed to be cold-start safe: if no model has
been trained yet (brand new bot, freshly wiped data folder), it
simply returns None from `predict()` and the caller falls back to
the rule-based conviction.  This is the "bootstrap" phase described
in the V2 brief — rule logic is the decision maker until enough
trade outcomes exist to train a useful model.

Reload semantics:
    A model is loaded once at bot startup.  The nightly trainer
    writes new model versions to `models/` and marks them active in
    `ml_models`, but the live predictor does NOT automatically swap
    them in — that happens on the next bot restart.  This is
    intentional: mid-session model swaps would make it harder to
    reason about a trading session's decisions post-hoc.

Ensemble with rule conviction:
    The predictor returns a probability in [0, 1] representing
    "likelihood this signal will be profitable".  The SignalAggregator
    blends this with the existing rule-based conviction — see
    `_apply_ml_prediction` in signal.py.  The blend weight ramps from
    0 (no model) to 0.5 (model with >50 training trades) so the
    system doesn't flip to ML overnight.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_trade.monitoring.logger import get_logger

if TYPE_CHECKING:
    from ai_trade.monitoring.database import Database

log = get_logger(__name__)

MODEL_NAME = "signal_quality"


class SignalQualityPredictor:
    """Loads the active signal-quality model and exposes `predict()`.

    Thread-safety: single-instance, read-only after init.  The bot
    creates one at startup and shares it across strategies.
    """

    def __init__(self, database: Database) -> None:
        self._db = database
        self._model = None
        self._feature_order: list[str] | None = None
        self._version: int | None = None
        self._training_trades: int = 0
        self._val_accuracy: float | None = None
        self._load_active_model()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _load_active_model(self) -> None:
        """Read the `ml_models` registry and load the active model file.

        Silently stays in cold-start mode if no active row exists or
        the file is missing — the bot should not crash on a fresh DB.
        """
        try:
            with self._db._conn() as conn:  # noqa: SLF001
                row = conn.execute(
                    """
                    SELECT version, model_path, training_trades, backtest_accuracy
                    FROM ml_models
                    WHERE model_name = ? AND is_active = 1
                    ORDER BY version DESC
                    LIMIT 1
                    """,
                    (MODEL_NAME,),
                ).fetchone()
        except Exception:
            log.exception("predictor_registry_read_failed")
            return

        if not row:
            log.info("predictor_cold_start", reason="no_active_model")
            return

        path = Path(row["model_path"]) if row["model_path"] else None
        if not path or not path.exists():
            log.warning(
                "predictor_model_file_missing",
                version=row["version"],
                path=str(path),
            )
            return

        try:
            import joblib
        except ImportError:
            log.error("predictor_joblib_missing")
            return

        try:
            payload = joblib.load(path)
        except Exception:
            log.exception("predictor_model_load_failed", path=str(path))
            return

        self._model = payload.get("model")
        self._feature_order = payload.get("feature_order")
        self._version = int(row["version"])
        self._training_trades = int(row["training_trades"] or 0)
        self._val_accuracy = (
            float(row["backtest_accuracy"])
            if row["backtest_accuracy"] is not None
            else None
        )
        log.info(
            "predictor_model_loaded",
            version=self._version,
            training_trades=self._training_trades,
            val_accuracy=self._val_accuracy,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """True if a trained model is loaded (not cold-start)."""
        return self._model is not None

    @property
    def version(self) -> int | None:
        return self._version

    @property
    def training_trades(self) -> int:
        return self._training_trades

    def blend_weight(self) -> float:
        """How much weight the ML prediction gets vs the rule conviction.

        Ramps linearly from 0 (no model) to a cap of 0.5 once the
        model has trained on 100+ trades.  The V2 brief explicitly
        requires this ramp so the bot doesn't flip to pure ML
        overnight — rules bootstrap the model, and the model grows
        more influential as its training set grows.
        """
        if not self.is_ready():
            return 0.0
        return min(0.5, self._training_trades / 200.0)

    def predict(
        self,
        signal,
        market_context=None,
    ) -> float | None:
        """Return P(profitable) for a signal, or None if cold-start.

        None means "no model available" — the caller should keep the
        rule-based conviction untouched.  A float in [0, 1] means
        "use this as the ML probability" — the caller blends it.
        """
        if not self.is_ready():
            return None

        try:
            from ai_trade.ml.features import extract_features, features_to_vector
            import numpy as np
        except ImportError:
            log.exception("predictor_feature_import_failed")
            return None

        try:
            feats = extract_features(signal, market_context)
            vec = np.array([features_to_vector(feats)])
            proba = self._model.predict_proba(vec)[0]
            # Class 1 = profitable (label scheme matches trainer.py)
            if len(proba) >= 2:
                return float(proba[1])
            return float(proba[0])
        except Exception:
            log.exception(
                "predictor_inference_failed",
                symbol=getattr(signal, "symbol", "?"),
                strategy=getattr(signal, "strategy_name", "?"),
            )
            return None

    def apply_to_conviction(
        self,
        rule_conviction: float,
        ml_probability: float | None,
    ) -> tuple[float, dict[str, Any]]:
        """Blend the rule conviction with the ML probability.

        Returns (blended_conviction, trace_dict).  Trace dict is
        attached to the signal metadata and logged to the decisions
        table for audit.  Blend is linear:

            blended = (1 - w) * rule + w * ml

        where w = blend_weight() (ramps 0 → 0.5).  If ML probability
        is None, the rule conviction is returned unchanged.
        """
        if ml_probability is None:
            return rule_conviction, {"ml_applied": False, "reason": "cold_start"}

        w = self.blend_weight()
        blended = (1.0 - w) * rule_conviction + w * ml_probability
        blended = max(0.0, min(1.0, blended))
        return blended, {
            "ml_applied": True,
            "ml_probability": round(ml_probability, 4),
            "blend_weight": round(w, 3),
            "rule_conviction": round(rule_conviction, 4),
            "blended_conviction": round(blended, 4),
            "model_version": self._version,
        }
