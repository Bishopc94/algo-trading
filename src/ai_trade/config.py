"""Configuration loader — merges YAML settings with environment variables.

This module handles two kinds of configuration:
  1. **Strategy/risk/schedule parameters** — stored in ``config/settings.yaml``
     (version-controlled, safe to share).
  2. **API secrets** — stored in ``config/.env`` (never committed to Git).

The loader reads the YAML file, converts the nested dictionary into a
``SimpleNamespace`` tree so that values can be accessed with dot-notation
(e.g. ``cfg.strategies.momentum.enabled`` instead of
``cfg["strategies"]["momentum"]["enabled"]``), then injects the API keys
from environment variables.

Python-specific notes for non-Python readers:
  - ``SimpleNamespace`` is a lightweight object from the standard library
    whose attributes can be set freely.  Think of it like a JavaScript
    plain object — ``ns.foo`` works the same as ``obj.foo`` in JS.
  - ``from __future__ import annotations`` makes all type hints strings
    at runtime, which lets you write ``str | None`` even on older Python
    versions.  It has zero effect on actual behaviour.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import yaml
from dotenv import load_dotenv

# Resolve project root: this file lives at src/ai_trade/config.py,
# so .parents[2] goes up two directories → the project root.
_ROOT = Path(__file__).resolve().parents[2]  # project root
_DEFAULT_CONFIG = _ROOT / "config" / "settings.yaml"


def _to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a nested dict into a SimpleNamespace tree.

    After conversion every key becomes an attribute, allowing dot-access:
        ``cfg.strategies.momentum.enabled``
    instead of:
        ``cfg["strategies"]["momentum"]["enabled"]``

    This is applied recursively so that nested dicts (like strategy configs)
    also become SimpleNamespace objects.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _to_namespace(v)
    return SimpleNamespace(**d)


def load_config(config_path: str | Path | None = None) -> SimpleNamespace:
    """Load YAML config and overlay environment variables for API secrets.

    Steps:
      1. Load ``.env`` file (first from ``config/`` then project root).
      2. Parse the YAML settings file.
      3. Convert the raw dict to a SimpleNamespace tree.
      4. Inject ``ALPACA_API_KEY`` and ``ALPACA_SECRET_KEY`` from the
         environment so that secrets never appear in the YAML file.

    Args:
        config_path: Optional path to a custom YAML file.  Defaults to
                     ``config/settings.yaml`` in the project root.

    Returns:
        A SimpleNamespace tree with all configuration parameters.

    Raises:
        EnvironmentError: If the required Alpaca API keys are missing.
    """
    # --- Step 1: Load .env file ---
    # The .env file holds ALPACA_API_KEY and ALPACA_SECRET_KEY.
    # load_dotenv() reads the file and sets the values as environment
    # variables so they can be retrieved with os.environ.get().
    for env_path in [_ROOT / "config" / ".env", _ROOT / ".env"]:
        if env_path.exists():
            load_dotenv(env_path)
            break

    # --- Step 2: Parse YAML ---
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    with open(path) as f:
        raw = yaml.safe_load(f)  # safe_load avoids arbitrary code execution

    # --- Step 3: Convert to dot-accessible namespace ---
    cfg = _to_namespace(raw)

    # --- Step 4: Inject API secrets from environment ---
    # Secrets are NEVER stored in the YAML file — only in .env.
    cfg.alpaca.api_key = os.environ.get("ALPACA_API_KEY", "")
    cfg.alpaca.secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

    if not cfg.alpaca.api_key or not cfg.alpaca.secret_key:
        raise EnvironmentError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set. "
            "Copy config/.env.example to config/.env and fill in your keys."
        )

    return cfg
