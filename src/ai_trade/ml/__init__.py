"""Machine learning core for V2 self-learning behaviour.

Modules:
    features  — canonical feature extraction from signals + market context
    trainer   — offline training of signal-quality classifier on closed trades
    predictor — runtime inference (cold-start safe, pass-through if no model)
"""
