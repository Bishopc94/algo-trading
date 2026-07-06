# Claude Instructions — Ai Trade

This file is loaded into every Claude Code session for this project. It is the operating brief; the detailed objective + work history lives in [docs/OBJECTIVE.md](docs/OBJECTIVE.md), and the active work queue in [docs/HANDOFF_2026-06-20.md](docs/HANDOFF_2026-06-20.md). Read those when you need depth; this file is the always-on context.

---

## What this project is

A multi-strategy automated trading bot running on Alpaca paper, intended to go live. Trades equities + equity options. Rule-based with an ML conviction blend (capped at 25% weight). One account, one strategy roster, one ML model active at a time. Python 3.11+, SQLite, APScheduler. Current version: see [src/ai_trade/_version.py](src/ai_trade/_version.py).

## Objective (the short version)

Maximize risk-adjusted returns on a $5,000 account while making every loss strictly less expensive than the last by absorbing its lesson into the system. Capital preservation > sustainable compounding > strategy breadth > self-correction, in that order. See [docs/OBJECTIVE.md](docs/OBJECTIVE.md) for the full statement.

## Operating principles (apply on every change)

1. **Defense in depth on risk.** Risk gates compose: position → symbol → portfolio → regime → account. When something slips through, add a new layer; don't reinforce the failed one.
2. **Every loss earns a structural fix.** After a loss, ask "what mechanism would have prevented this without human intervention?" and build *that*. Never fix the single symbol; fix the class.
3. **Alpaca is the source of truth.** The DB is a cache. `sync_positions` reconciles every cycle. Bracket orders are server-side so stops fire even when the bot is offline.
4. **ML is a hypothesis, not an oracle.** Quality gates on training, auto-rollback on load, capped blend weight. The bot must operate correctly with no model loaded (cold-start).
5. **Decision provenance is non-negotiable.** Every signal, rejection, ML prediction, parameter change, and adopted position writes a row. Don't add a code path that silently drops a decision.
6. **Don't over-build the future.** A working bot today compounds. An elegant one in a month doesn't.

## Hard constraints

- **Paper account, but ask before destructive actions** (`cancel_all`, `close_all`, force-pushes, drop tables, `git reset --hard`, `git push --force`). Local file edits + single-row DB updates: fine without asking.
- **`Read(./.env)` is denied** by `.claude/settings.json`. Don't attempt.
- **Never commit unless the user asks explicitly.** When asked, use HEREDOC for the message. Co-author tag per the global instructions.
- **Never amend or force-push.** Always create new commits.
- **Account is margin-type with multiplier 1.** No leverage. No shorting on stock side (some options strategies are short premium).
- **The local branch `2.0-refactor` tracks `origin/master`** (misconfigured). When a push comes up, ask which side.

## Coding conventions

- **Bump `_version.py` and add a README Changelog entry on every meaningful change.** Match the format of recent entries: title, motivating incident, then numbered behavior changes. Patch for bug fixes/hardening/config, minor for new features, major for breaking strategy/risk changes.
- **Comments are for non-obvious *why*.** Don't narrate what code does. Don't reference issue numbers, callers, or the current PR — those belong in the commit message.
- **No half-finished implementations.** Don't add feature-flag wrappers or "TODO" stubs. Either build it or don't.
- **Don't add defensive try/except around things that can't fail.** Trust internal code and framework guarantees. Validate at system boundaries only.
- **Prefer Edit over Write.** Prefer dedicated tools (Read, Grep, Glob) over Bash for file ops.

## Where things live

| Need | Look at |
|---|---|
| Full objective + work history | [docs/OBJECTIVE.md](docs/OBJECTIVE.md) |
| Current open issues + priority queue | [docs/HANDOFF_2026-06-20.md](docs/HANDOFF_2026-06-20.md) |
| Module-level design | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Risk gates + sizing | [docs/RISK_MANAGEMENT.md](docs/RISK_MANAGEMENT.md) |
| Per-strategy specs | [docs/STRATEGIES.md](docs/STRATEGIES.md) |
| Config knobs | [docs/CONFIGURATION.md](docs/CONFIGURATION.md), [config/settings.yaml](config/settings.yaml) |
| Per-release detail | [README.md](README.md) Changelog |
| The brain | [src/ai_trade/strategy/signal.py](src/ai_trade/strategy/signal.py) `SignalAggregator` |
| The orchestrator | [src/ai_trade/main.py](src/ai_trade/main.py) `TradingBot` |
| Risk gates | [src/ai_trade/risk/risk_manager.py](src/ai_trade/risk/risk_manager.py), [src/ai_trade/risk/dynamic_risk.py](src/ai_trade/risk/dynamic_risk.py) |
| Order submission + reconciliation | [src/ai_trade/execution/order_manager.py](src/ai_trade/execution/order_manager.py) |
| ML training + inference | [src/ai_trade/ml/trainer.py](src/ai_trade/ml/trainer.py), [src/ai_trade/ml/predictor.py](src/ai_trade/ml/predictor.py) |

## What this bot is *not*

Out of scope (ask before reaching for any of these):
- Crypto
- HFT / sub-second strategies
- Margin / leverage / stock shorting
- News-trading as a signal (sentiment is a filter, not a signal)
- Manual override UI
- Multiple accounts or simultaneous active models

## Session expectations

- State what you're about to do in one sentence before the first tool call.
- Give short updates at decision points (found something, changed direction, hit a blocker). Don't narrate deliberation.
- End-of-turn summary: one or two sentences. What changed and what's next.
- When you find new issues during execution, add them to the active handoff doc rather than going off-task.
- Test changes before declaring them done. For risk/ML changes, write a smoke test that exercises the failure case.

## Recent release line (context for what's in flight)

- **v2.4.2** — ML rollback (manual + startup auto-heal in predictor)
- **v2.4.1** — ML promotion quality gate, `ema_crossover.min_entry_price`, per-symbol cooldown
- **v2.4.0** — Entry-price reconciliation, bot heartbeat, per-strategy price floor, post-PDT displays
- **v2.3.5** — Heat-lockup root-cause fix, tracked-position re-protect, PDT framework flip to `intraday_margin`

The v2.4.x line is all loss-driven hardening — each release responds to a specific incident. If the user mentions an incident date (RGNT 2026-06-10, LNKS 2026-06-17, ML degradation 2026-06-19), the corresponding fix is in the changelog.
