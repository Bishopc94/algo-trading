# Project Objective & Work Summary

Last updated: 2026-06-20 (v2.4.2)

This document is the single-source statement of what this bot is for, the principles it operates under, and the work that has shaped it. Other docs cover *how* (`ARCHITECTURE.md`, `RISK_MANAGEMENT.md`, `STRATEGIES.md`); this one covers *why*.

---

## 1. Objective

**Maximize risk-adjusted returns** on a small ($5,000) Alpaca account through fully-automated, multi-strategy trading — while making each loss strictly less expensive than the last by structurally absorbing its lesson into the system.

The bot is rule-based today, ML-blended now, and intended to become ML-led as it accumulates a trustworthy track record. Trust is earned, not assumed: the ML model only gets a larger share of the conviction signal after it has proven calibrated on live trades.

Concretely, the objective decomposes into four sub-goals, in priority order:

1. **Capital preservation.** No single trade, no single day, no single offline window can deal a wound the account can't recover from in a normal week of trading. Every risk gate exists to enforce this.
2. **Sustainable compounding.** Bias toward a high Sharpe / high profit factor over peak returns. A 1.5x annual return with a 3.0 Sharpe is better than 3x at 1.2 Sharpe; the bot survives a bad month at the former and may not at the latter.
3. **Strategy breadth.** Run all enabled strategies (stocks + options) so that any given regime has at least one strategy that fits. The bot should not depend on one or two strategies firing — that was the v1.x failure mode.
4. **Self-correction.** When something breaks (a strategy bleeds, an ML model degrades, an entry pattern keeps stopping out at the same level), the bot either auto-corrects (cooldown, ML rollback, dynamic risk halt) or surfaces the pattern in the EOD filter report for a human to encode as a structural fix.

### Performance baseline to beat

From the V2 brief (the rule-based-only era):

| Metric | Baseline | Target |
|---|---|---|
| Sharpe | 3.12 | ≥ 3.5 |
| Win rate | 43.75% | ≥ 50% |
| Profit factor | 1.55 | ≥ 2.0 |
| Strategies firing | 2 of 16 | ≥ 8 of 19 |

Current state is below baseline on win rate and P&L (the 2026-06 micro-cap incidents) — the v2.4.x release line is the response to that drift.

---

## 2. Operating principles

These principles are *load-bearing*. Every architectural decision and every fix in the changelog can be traced back to one of them.

### 2.1 Defense in depth on risk

Risk gates compose rather than override each other. A trade must pass:
- **Position-level**: stop loss + position sizer (Kelly fraction + max_position_pct)
- **Symbol-level**: per-symbol cooldown + per-strategy `min_entry_price`
- **Portfolio-level**: max_open_positions + portfolio heat + daily loss limit
- **Regime-level**: DynamicRiskController tiers (drawdown halts, conviction-streak scaling, regime-based long/short bias)
- **Account-level**: PDT framework rules (legacy or intraday_margin), buying-power, min_deployable_cash

The redundancy is intentional. The RGNT and LNKS incidents both happened because exactly one of these layers had a gap; the response in each case was to add a new layer rather than reinforce the failed one.

### 2.2 Every loss earns a structural fix

The 2026-06-10 RGNT loss (−$648) produced: entry-price reconciliation, bot heartbeat alert, `macd_divergence.min_entry_price`. The 2026-06-17 LNKS spam (−$176) produced: `ema_crossover.min_entry_price`, per-symbol cooldown. The 2026-06-19 ML-degradation incident produced: trainer quality gate, predictor startup auto-rollback.

The rule: **after a loss, ask "what mechanism would have prevented this without human intervention?" and build that.** Never fix the single symbol; fix the class of pattern.

### 2.3 Alpaca is the source of truth

The bot's database is a cache of trading intent + an analysis store. The broker is reality. Therefore:
- `sync_positions` reconciles every cycle: adopts untracked positions, re-protects tracked-but-naked positions, repairs stop_loss=0 rows, and updates DB `entry_price` to Alpaca's `avg_entry_price` when they diverge.
- Bracket orders are server-side. Even when the bot is offline, stops still fire.
- The bot heartbeat catches offline windows so server-side fills during downtime aren't a surprise on restart.

### 2.4 ML is a hypothesis, not an oracle

The signal-quality predictor blends at most 25% weight into conviction (capped from the original 50% after observed under-prediction on micro-cap signals). A new model must pass a quality gate before it can go active; an active model that doesn't pass the gate is auto-rolled back at startup. Cold-start (no model) leaves rule conviction untouched. The system is designed to operate forever without ML — ML makes it better, not functional.

### 2.5 Decision provenance is non-negotiable

Every signal, every rejection, every ML prediction, every parameter change, every adopted position writes a row. The `decisions` table receives ~25k rows/day during paper trading. This is intentional: post-hoc analysis ("why didn't we trade X?", "what filter killed Y?") is more valuable than the disk it consumes, and the EOD filter-effectiveness report needs the data to surface chronic near-misses.

### 2.6 Don't over-build the future

When in doubt, prefer the conservative path that ships today over the elegant one that ships in two weeks. The bb_squeeze strategy was *disabled* rather than rewritten when its WR dropped. v3 was kept as the production ML model rather than retraining a v7. A working bot today compounds; an elegant one in a month doesn't.

---

## 3. Architecture at a glance

```
candidates (scanner) → strategies (per-symbol evaluate) → SignalAggregator
                                                              ↓
                                              [filter & rank: cooldown,
                                               min_entry_price, weighter,
                                               ML blend, conviction sort]
                                                              ↓
                                              [risk gates: heat, daily
                                               loss, drawdown tier,
                                               position count]
                                                              ↓
                                              execution_queue → OrderManager
                                                              ↓
                                              bracket orders → Alpaca
                                                              ↓
                                              sync_positions ←┘
                                                  (reconcile every cycle)
```

See `ARCHITECTURE.md` for module-level detail. The handoff doc (`HANDOFF_2026-06-20.md`) covers known open issues and a prioritized work queue.

---

## 4. Work summary (release-by-release)

This is the abbreviated changelog with *why*, not *what*. Full per-release detail lives in `README.md`.

### Foundation (v0.x — v1.x)

The original system: scheduled scans, a handful of strategies (momentum, mean_reversion, VWAP), Alpaca brackets, email alerts, basic position sizing. v1.2.0 added multi-indicator confluence to all strategies and a failed-symbol blacklist.

### V2.0 — Self-learning overhaul

13 phases shipped together. Introduced: market-regime analyzer (trend/chop/volatility classification), news sentiment + earnings guard, SignalQualityPredictor (ML), strategy weighter (adaptive conviction multipliers from historical win rate / PF), SmartPDTPlanner (day-of-week × slots × EV-based dynamic threshold), DynamicRiskController (drawdown tiers, conviction-streak scaling, regime-based bias), 0DTE options, and a full options stack (9 strategies).

This release is the one that took the bot from "rule-based intraday trader" to "self-learning multi-strategy system." Everything since has been refining the seams.

### V2.1 — Backtest ML training pipeline

Closed the loop on ML: a backtest produces synthetic trades, the trainer learns from them, and a production model can be bootstrapped without waiting months of live data. The model registry (`ml_models` table) and the joblib-on-disk pattern date from here.

### V2.2 — Pre-live hardening

Conviction integrity audit (no more silent zero-clamps), PDT visibility, options-side risk approval, backtest fidelity fixes. The "are we actually ready to flip to live?" pass.

### V2.3 — Operational reality

Lots of small fixes that came out of running the paper account day-in-day-out:
- **v2.3.0** — Decoupled the full-universe scan (15 min) from per-candidate strategy evaluation (5 min). Catches signals as they print rather than as they're scanned. Pullback-entry limit orders for better fills.
- **v2.3.1** — Holiday gate, cash-account settlement handling, stale-order cleanup, `min_deployable_cash` to stop spamming orders the account can't fund.
- **v2.3.2** — Adopted untracked positions (Alpaca is source of truth — see §2.3).
- **v2.3.3** — Backtest trailing-stop fidelity to match the live ratchet exactly.
- **v2.3.4** — Decision-table retention to cap DB growth.
- **v2.3.5** — Heat-lockup root cause fix (ONDS at `stop_loss=0` made heat math read 21% of equity at risk, freezing all entries for 4 days). PDT framework flipped to `intraday_margin` after the 2026-06-04 FINRA Rule 4210 amendment. Filter-effectiveness report added.

### V2.4 — Loss-driven hardening (the current release line)

Three releases in three days, each a direct response to an observed loss pattern.

- **v2.4.0** — Triggered by the RGNT 2026-06-10 −$648 gap-down. Added entry-price reconciliation (DB stored signal price, not Alpaca fill — 2% off on RGNT), bot heartbeat alert (catches the silent-crash class), per-strategy `min_entry_price` (default applied to `macd_divergence`: $5.00), post-PDT console/email displays, richer exit emails with MFE/MAE %.
- **v2.4.1** — Triggered by the LNKS 2026-06-17 spam (6 entries on a $1.85 ticker in 51 min, −$176) and the v5 ML promotion at 12.5% accuracy. Added: ML promotion quality gate (val_acc ≥ 0.55, ≥ 50 trades, ≤ 10% regression), `ema_crossover.min_entry_price: 5.00`, per-symbol re-entry cooldown (30 min default).
- **v2.4.2** — Manual ML rollback (v6 → v3 active) plus startup auto-heal in the predictor (re-checks the active model against the gate constants on every restart; rolls back to the most recent passing version if it fails). Makes the gate symmetric: trainer blocks bad new models, predictor heals bad active models.

---

## 5. Known open work

See `docs/HANDOFF_2026-06-20.md` for the prioritized queue. Top items:

1. **Heat-math direction bug** (JRSH-style: stop above entry on a long → negative "risk" → bypasses heat cap). Fix `abs()` + sign assertion in `risk_manager.py`.
2. **Min entry-price floor on other swing strategies** (`pullback`, `mean_reversion`). Or a hold-type-aware global gate.
3. **Decisions table at 240k rows despite 14-day retention** — prune job isn't pruning. Add index + investigate.
4. **Per-symbol cooldown is symmetric** — skip on winning exits, scale by `pnl_pct`.
5. **Cleanup backlog** in `HANDOFF_2026-06-20.md` §7: log rotation, stale model files, `pdt_manager.py` (dead under intraday_margin), doc drift.

---

## 6. Non-goals

These are deliberately out of scope. If you find yourself reaching for them, stop and ask first.

- **Crypto.** Alpaca supports it, but the bot is equities + equity options.
- **HFT / sub-second strategies.** The bot operates on 5-min intraday + daily bars. Tick-level data and order books are not in the design.
- **Margin / leverage.** Account is margin-type but multiplier is 1. No leverage, no shorting (current strategies are long-only; some options strategies are short premium).
- **News-trading / event-driven.** Sentiment is a *filter* (block near earnings, block on bearish news) — not a *signal*.
- **Manual override UI.** The bot is autonomous. Manual intervention is via DB edits, config changes, or restarting with new settings — not a control panel.
- **Multiple accounts / portfolios.** One account, one strategy roster, one ML model active at a time.
