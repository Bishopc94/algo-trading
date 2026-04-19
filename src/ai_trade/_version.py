"""Single source of truth for the application version.

Bump this value whenever a meaningful change is deployed. The version
is injected into structured logs so every trade and event can be
traced back to the exact code revision that produced it.

Format: MAJOR.MINOR.PATCH
  - MAJOR: breaking changes to strategy logic or risk rules
  - MINOR: new features (strategies, indicators, integrations)
  - PATCH: bug fixes, hardening, config tweaks
"""

__version__ = "2.1.0"
