# Roadmap to Live Trading

**Status:** planning, not yet started
**Target capital:** $2,000 live (paper account currently $504)
**Target go-live:** after Phase 2 ships (est. 4-6 weeks of focused work)
**Current version:** 2.1.1 — see [src/ai_trade/_version.py](../src/ai_trade/_version.py)

---

## Mission

Prepare the bot for live trading on a small ($2k) account while **maximizing real-world profit** and **minimizing the paper-to-live performance gap**. The system must remain profitable under honest frictions (slippage, commissions, partial fills, PDT constraints) and be resilient to edge degradation, model drift, and upcoming regulatory changes.

**Non-goals:**
- Making the backtest look better on paper
- Preserving a particular headline metric at the expense of real-world validity
- Adding features the $2k account can't meaningfully use (e.g., portfolio-level correlation hedging)

---

## Context for the incoming agent

### Why this document exists

The owner is preparing to transition from paper trading (Alpaca paper account, ~$500 equity) to a live account funded at ~$2,000. Before that transition, several architectural changes are needed to ensure the live bot is both safe and honestly profitable. This doc is the ordered plan of work.

### What the bot does today (v2.1.1)

- Scans ~8,600 symbols every 15 minutes for momentum, mean-reversion, and VWAP setups
- Runs 6+ stock strategies (ORB, VWAP, momentum, mean-reversion, BB-squeeze, MACD-div) and 7 options strategies
- Uses a GradientBoostingClassifier (v3, trained on 433 backtested trades) to score signal quality; predictions blended with rule-based conviction
- Enforces PDT via [src/ai_trade/risk/pdt_manager.py](../src/ai_trade/risk/pdt_manager.py) — currently hardcoded to 3 day trades per 5-business-day window, assumes <$25k equity
- Submits Alpaca bracket orders (entry + stop + target) and reconciles positions every 60s via `job_sync_positions`
- Logs to both `ai_trade.log` (detailed) and `ai_trade_run.log` (structured events)

### Paper performance as of 2026-04-21

- 19-trade rolling sample: 53% W, 2.11 profit factor, 5.22 Sharpe, 6.1% max DD
- **Do not anchor on these numbers.** See the reframing section below.

### The Sharpe 5.22 reframing (critical)

The headline Sharpe is misleading for three reasons the incoming agent must internalize:

1. **Sample size:** 19 trades produces a Sharpe point estimate with a 95% confidence interval of roughly [2.8, 7.6]. The true Sharpe is unknown.
2. **Paper-to-live gap:** Alpaca's paper engine fills at mid-quote and assumes infinite liquidity. Live fills cross the spread and experience market impact. Expect a 30-50% Sharpe drop on transition for *structural* reasons, independent of strategy quality.
3. **Cost-free backtest:** commissions, regulatory fees, SEC/TAF fees, and realistic slippage are not modeled. Adding them will *lower the headline Sharpe* but produce numbers that predict live behavior.

The goal of this roadmap is **NOT to preserve 5.22**. The goal is to build a system where:
- The backtest Sharpe closely tracks live Sharpe (small, known gap)
- Live returns are genuinely positive after all real costs
- The system self-corrects when edge degrades

If a change in this roadmap *lowers* the paper Sharpe but *raises* real-money expected return, take it.

---

## Cross-cutting requirement: configurable day-trading (PDT)

**This is a hard requirement that threads through every phase.** The US SEC/FINRA is in the process of changing the PDT rule — the proposed change lowers the minimum equity threshold from $25,000 to $2,000 and may eliminate the 5-business-day day-trade cap for accounts above that threshold. The effective date and Alpaca's implementation timeline are unknown as of writing.

### What this means for the codebase

Every day-trading-related decision must be driven by config, not hardcoded values:

```yaml
# settings.yaml (proposed)
pdt:
  enabled: true                    # Master switch. If false, treat as unrestricted.
  min_equity_threshold: 25000      # Account equity above which unlimited day trades allowed.
  max_day_trades_per_window: 3     # Applied only when equity < threshold.
  window_business_days: 5
  reserve: 1                       # Keep N slots untouched for manual/emergency use.
  enforce_unsettled_funds: true    # Block trades using unsettled cash (T+1/T+0 dependent).
  settlement_days: 1               # T+1 today; will become T+0 for some asset classes.
```

### Design principles

1. **Single source of truth:** one module (`src/ai_trade/risk/pdt_manager.py`) owns all PDT/settlement logic. No other file should compute day-trade eligibility.
2. **Runtime-togglable:** flipping `enabled: false` or raising `min_equity_threshold` should immediately unlock day-trading-gated strategies with no code changes.
3. **Strategy opt-in:** strategies declare themselves `requires_day_trading: bool`. When PDT is active and slots are exhausted, only non-day-trading strategies remain eligible.
4. **Settlement tracking:** each fill records a `settlement_date` so the bot knows which cash is tradable *now* vs pending.
5. **Forward compatibility:** treat the rule change as a config flag, not a code migration. When Alpaca enables the new rule, flip the config.
6. **Broker override awareness:** even if FINRA relaxes the rule, Alpaca or clearing firms may enforce stricter policies. Config must be able to tighten beyond FINRA minimums.

### Audit targets

The incoming agent should grep for these and move them to config:

- `"25000"` / `25_000` / `"$25k"` — hardcoded PDT equity threshold
- `max_day_trades` / `day_trade_reserve` in [src/ai_trade/risk/pdt_manager.py](../src/ai_trade/risk/pdt_manager.py)
- `HoldType.DAY` checks that short-circuit strategies
- Any string like `"pdt"`, `"day trade"`, `"pattern day"` in logs/config — audit for assumptions

---

## Core principle: config-driven limits with hard code floors

The user is explicit that they want **high-risk trading that is profitable** — not maximum conservatism for its own sake. Every risk limit in this roadmap follows the same pattern:

1. **Config value sets the operating limit** (tunable as the system accumulates evidence)
2. **Code enforces a hard floor beyond which config cannot go** (tamper guard against accidentally disabling safety)
3. **Default config starts conservative; relax as data validates the edge**

This means early-stage limits can be tight while the system proves itself, then loosened after 100+ live trades without any code changes. Every `risk:` or `portfolio:` value in `settings.yaml` should have a corresponding `assert value <= CODE_MAX_*` check in [src/ai_trade/config.py](../src/ai_trade/config.py).

### Proposed config schema

```yaml
risk:
  per_trade_pct: 0.025             # start 2.5%, tunable 0.01-0.03, code floor = 0.03
  daily_drawdown_halt: 0.05        # hard stop — no new entries
  daily_drawdown_soft: 0.03        # soft — shrink sizing to 50%
  consecutive_loss_pause_min: 30
  consecutive_loss_threshold: 3
  sizer_weight_range: [0.5, 1.5]   # strategy-Sharpe clamp; 1.5 lets hot strategies amplify
  ev_gate_min_r: 0.0               # require positive EV, no margin by default
  ev_high_conf_r: 0.3              # flag as "high confidence" — gets priority, not used as reject
  max_correlated_per_sector: 2
  min_equity_live: 1000

portfolio:
  sleeve_weights: {safe: 0.85, convex: 0.15}
  rebalance_mode: threshold        # "threshold" | "weekly" | "monthly" | "off"
  rebalance_drift_pct: 0.05        # rebalance only when sleeve drifts >5pp from target
  convex_max_concurrent_base: 2    # scales with equity: max(base, floor(equity/1000))
```

### What CAN vs CANNOT be loosened via config

**Configurable (safe to tune):** per-trade risk %, drawdown halt thresholds, sizer weights, EV gate, sleeve weights, rebalance cadence, max concurrent positions.

**Hardcoded floors (cannot exceed via config):** 3% absolute per-trade risk cap, 10% absolute daily drawdown halt, kill switch always active, pre-trade validator always runs, barbell firewall always enforced.

**Not configurable at all:** correctness mechanisms — calibration, drift detection, EV computation, settlement tracking. Turning these off is a bug, not a tuning decision.

---

## Known ML pipeline bugs (discovered 2026-04-22 — fix before or during Phase 0)

These were found while auditing whether the GradientBoostingClassifier is actually working. They are pre-existing bugs, not new work — list them here so the incoming agent tackles them as table-stakes before anything else.

### Bug A: predictor startup logs are silently dropped

**Location:** [src/ai_trade/main.py:230](../src/ai_trade/main.py#L230) vs [main.py:306](../src/ai_trade/main.py#L306)
**Symptom:** `grep -c "predictor_" logs/ai_trade.log` returns **0** across every bot start.
**Root cause:** `TradingBot.__init__()` instantiates `SignalQualityPredictor` at line 230, but `setup_logging()` doesn't run until `bot.start()` at line 306. Every log call the predictor makes during `_load_active_model()` goes to an unconfigured root logger and disappears.
**Impact:** you cannot tell which model version is loaded, cannot see load failures, cannot diagnose a silent downgrade.
**Fix:** move `setup_logging()` to the top of `TradingBot.__init__()` — one line change. Verify by checking for `predictor_model_loaded` in the log after restart.

### Bug B: `insert_ml_prediction()` is orphaned — no callers

**Location:** [src/ai_trade/monitoring/database.py:800](../src/ai_trade/monitoring/database.py#L800)
**Symptom:** `ml_predictions` table has 0 rows despite the predictor being active and running in the signal pipeline.
**Root cause:** the function exists but is never called. The `_apply_ml_prediction` path in [signal.py:451](../src/ai_trade/strategy/signal.py#L451) stores the ML trace inside `sig.metadata["ml_trace"]` and logs via `DecisionLogger`, but never persists to the dedicated table.
**Impact:** no audit trail of what probability the model assigned to each signal, so Phase 2 calibration and Phase 5 drift detection have nothing to work against.
**Fix:** in `_apply_ml_prediction`, after the blend, call `self._db.insert_ml_prediction(decision_id=…, model_version=…, probability=ml_prob, features_hash=…, outcome=None)`. Outcome gets filled later by the trade-close sync job.

### Bug C: model swap has no quality gate (see Phase 5 deliverable below)

This is more than a bug, so it's captured as a formal Phase 5 deliverable. Summary: the trainer blindly marks every new model `is_active=1`, replacing a good model with a bad one on any training run that happens to fit. See *Phase 5 → Model promotion pipeline* below.

---

## The roadmap (8 phases)

Each phase lists: **goal**, **deliverables**, **reasoning**, **exit criteria**, **dependencies**. Phases are ordered for dependency correctness. Do not skip ahead without reading the reasoning.

---

### Phase 0 — Baseline & validation (week 1, BLOCKING)

**Goal:** Know what the current edge actually is before changing anything.

**Deliverables:**
- Run v2.1.0 ML model on walk-forward holdout from [data/backtest_ml.db](../data/backtest_ml.db); confirm val accuracy (target: ≥58%) holds out-of-sample
- Generate 100+ additional paper trades (loosen PDT in a paper-only config to hit this faster) — 19 trades is insufficient for statistical inference
- Add synthetic friction to the backtester: half-spread slippage + per-share commission + SEC/TAF fees
- Rerun backtest with friction enabled; record the degradation (expect Sharpe to drop 1-2 full points)
- Compute bootstrapped 95% confidence intervals on Sharpe, win rate, profit factor
- Document results in a `docs/BASELINE.md`

**Reasoning:**
You cannot know if a later change helps or hurts if you don't have a trustworthy baseline. The current 19-trade Sharpe is a point estimate in a wide distribution. Everything downstream of this phase depends on knowing the *real* edge after costs.

**Exit criteria:**
- ≥100 paper trades logged in [data/ai_trade.db](../data/ai_trade.db)
- Bootstrapped Sharpe CI with cost-aware backtest lower bound ≥ 1.0
- Walk-forward out-of-sample accuracy within 5% of training accuracy (not a regime overfit)

**Dependencies:** none.

---

### Phase 1 — Guardrails (week 2, BLOCKING BEFORE LIVE)

**Goal:** Make blowup impossible, not just unlikely. These are cheap to build and non-negotiable.

**Deliverables:**
- **Per-trade risk cap** (configurable): default 2.5% equity (≈$50 on $2k), config range 1-3%, hardcoded code ceiling at 3%. Config cannot set higher; can always set lower.
- **Two-tier daily drawdown breaker**:
  - **Soft trigger** at -3% intraday: shrink all new-entry sizing to 50% of normal. Preserves participation in recovery trades.
  - **Hard halt** at -5% intraday: refuse new entries until EOD reset. Existing positions continue unmanaged by new entries (trailing stops still active).
  - Hardcoded code ceiling at -10% regardless of config.
- **Consecutive loss breaker**: pause new entries for 30 minutes after 3 straight losses. Count and pause duration are configurable.
- **Correlated exposure cap**: max 2 open positions in the same sector, configurable. Use Alpaca asset metadata or a static sector map.
- **Emergency kill switch**: file-based flag (`KILL_SWITCH` at repo root) or remote endpoint; when detected, force-flat all positions and refuse new entries until removed. Not configurable — always active.
- **Pre-trade validation**: every order reviewed by a `validate_order(order, account)` function that refuses if risk exceeds cap, if account is below `min_equity_live`, or if kill switch is active.
- **Config validation on boot**: sanity-check all risk params against code ceilings; refuse to start if any exceed the hardcoded floors (e.g., `risk.per_trade_pct > 0.03`, `daily_drawdown_halt > 0.10`).

**Reasoning:**
A $2k account cannot tolerate a 50% drawdown — recovery requires a 100% gain, and the bot may never see one. But maximum conservatism also fails: 1.5% of $2k is $30 risk, which barely clears commission + slippage on a typical position. The 2.5% default is Van Tharp's aggressive-but-survivable zone; the code ceiling at 3% prevents a config typo from turning a 2.5% limit into a 25% limit. The two-tier drawdown breaker matters because the historical max DD is 6.1% — a single-tier -3% halt would trip on normal variance and miss recovery days.

**Exit criteria:**
- All five circuit breakers have unit tests
- Integration test demonstrates force-flat under each trigger
- Can be tested in paper without disabling any safety

**Dependencies:** Phase 0 (need baseline drawdown distribution to calibrate thresholds).

---

### Phase 2 — EV gate + honest sizing (weeks 3-4, HIGHEST ROI)

**Goal:** Stop taking trades that lose money after real-world costs.

**Deliverables:**
- **Slippage model** — module `src/ai_trade/execution/slippage.py`:
  - Formula: `slippage_bps = half_spread_bps + impact_coef × (order_size / avg_volume)`
  - Coefficients calibrated from actual fill data in the `trades` table (compare `entry_price` to pre-order snapshot mid)
  - Applied to both backtest and live pre-trade EV calculation
- **Commission/fee model** — module `src/ai_trade/execution/costs.py`:
  - Per-share stock commission (Alpaca = $0 but model it anyway for future-proofing)
  - Per-contract options commission
  - SEC fee (sells only), TAF, ORF — current 2026 rates from config
- **Isotonic probability calibration** — extend [src/ai_trade/ml/predictor.py](../src/ai_trade/ml/predictor.py):
  - Fit `sklearn.isotonic.IsotonicRegression` on holdout predictions during training
  - Wrap the model's `predict_proba` output through the calibrator at inference
  - Validate: reliability diagram shows predicted vs actual in 10 bins, all within ±5%
- **EV gate at entry** — new check in `SignalAggregator.collect_and_rank()`:
  - Compute `EV = P × avg_win − (1−P) × avg_loss − slippage − commission`
  - **Reject only when `EV ≤ risk.ev_gate_min_r` (default 0 — just require positive EV)**
  - Separately flag signals where `EV ≥ risk.ev_high_conf_r` (default 0.3R) as "high confidence" — these get priority in the execution queue when multiple signals compete, but the threshold is not a reject gate
  - Log rejected-for-EV signals separately so they can be analyzed
- **Three-layer position sizer** — replaces existing conviction×streak logic:
  1. **Layer 1 (ATR-normalized):** `shares = (equity × risk_pct) / (entry − stop)`
  2. **Layer 2 (hard cap):** clamp risk to `min(risk.per_trade_pct, CODE_MAX_RISK=0.03)` of equity, absolute
  3. **Layer 3 (strategy-Sharpe weight):** multiply by rolling 50-trade Sharpe of that strategy, normalized to `risk.sizer_weight_range` (default `[0.5, 1.5]`). Poor strategies shrink automatically; hot strategies can amplify up to 1.5× *subject to the Layer 2 cap*. The cap prevents blowup; the ceiling lift captures real edge when a strategy is genuinely working.

**Reasoning:**
This is the single highest-ROI phase. Small-account trading dies by a thousand cuts — 5¢ slippage on a $2 stock is 2.5% of the move, which eats a third of the expected R. The EV gate catches the marginal trades that paper-trading makes look OK. The three-layer sizer replaces Kelly (which requires trustworthy edge estimates) with a system that's self-regulating and robust to overestimation.

**Why EV gate at 0R (positive-only) rather than 0.25R margin:** On a PDT-constrained small account, signal throughput is already scarce — last month's logs showed 85% of signals killed by PDT alone. Requiring an extra 0.25R margin *in addition to* PDT filtering kills too many legitimately-positive-EV setups. A 50% win rate at 1.5R has EV = 0.25R, which is real edge but would fail a 0.25R margin gate. Instead, we use the high-confidence flag to *prioritize* (not gate) the 0.3R+ trades when the execution queue is full.

**Why not Kelly:** Kelly is mathematically optimal *assuming accurate W and R*. Small-account traders overestimate both; full Kelly amplifies that error and blows accounts up in 10-trade streaks. Fractional Kelly helps but still depends on honest probability estimates. The three-layer sizer sidesteps this entirely — it degrades gracefully.

**Why the sizer range goes to 1.5× (not capped at 1.0):** "Never amplify" forgoes real edge. When a strategy has shown positive Sharpe over 50 live trades, refusing to size above baseline is leaving money on the table. The 1.5× ceiling still sits under the Layer 2 hard cap (2.5-3% of equity), so it can't blow up the account; it just captures compounding when the edge is confirmed.

**Exit criteria:**
- Rerun Phase 0 baseline with these changes enabled; EV-gated backtest Sharpe is >0 after all costs
- Rejected-signal log shows 5-15% of previously-entered signals now correctly filtered (lower than a 0.25R gate; compensates via prioritization)
- Isotonic calibrator's reliability diagram within ±5% across all bins
- Strategy-Sharpe weighting demonstrably shrinks the weakest strategy (`momentum` currently negative net) and scales up the best (`vwap` currently highest) over 50 trades

**Dependencies:** Phase 0 (needs baseline with and without gate to measure impact), Phase 1 (guardrails must exist before loosening anything).

---

### Phase 3 — PDT forward-compatibility (week 5)

**Goal:** Be ready for either PDT regime Alpaca rolls out.

**Deliverables:**
- Refactor [src/ai_trade/risk/pdt_manager.py](../src/ai_trade/risk/pdt_manager.py) per the cross-cutting requirement above
- Move all hardcoded PDT values to `settings.yaml` under a `pdt:` section
- Add `requires_day_trading: bool` class attribute on every strategy; propagate into eligibility logic
- Add settlement-date tracking: each row in `trades` records `settlement_date`; cash-availability query accounts for pending settlements
- Add `--pdt-regime` CLI override for testing both rule sets without config changes
- Integration tests: full trading cycle under (a) current PDT rules, (b) proposed relaxed rules, (c) PDT disabled entirely

**Reasoning:**
Regulatory changes happen on schedules the bot author doesn't control. Baking assumptions into code means a rushed migration when the change lands. Config-driven PDT means flipping a single file when Alpaca turns it on.

**Exit criteria:**
- `grep -r "25000\|25_000" src/` returns no business-logic matches
- Toggling `pdt.enabled` between true/false changes runtime behavior with no restart required (or documented restart procedure)
- Integration tests pass under all three regime configs

**Dependencies:** Phase 1 (pre-trade validation hook is where settlement/PDT checks attach).

---

### Phase 4 — Barbell sleeves (weeks 6-7)

**Goal:** Asymmetric upside with bounded downside. Implements Taleb's barbell explicitly in code.

**Deliverables:**
- Introduce `Sleeve` concept to portfolio tracking, weights driven by `portfolio.sleeve_weights` config:
  - **Safe sleeve** — default 85% of equity ($1,700 on $2k) — stocks only, Phase 2 sizer applies
  - **Convex sleeve** — default 15% of equity ($300 on $2k) — long options only, sized by premium paid (max loss = cost of contract)
- Convex sleeve constraints (all except "long-only" are configurable):
  - Long calls, long puts, or defined-risk debit spreads only (**hardcoded — this is the barbell definition**)
  - No naked shorts, no credit spreads (undefined or large-capped downside) — **hardcoded**
  - Max concurrent convex positions: `max(portfolio.convex_max_concurrent_base, floor(equity / 1000))` — scales with account size automatically
  - 1-7 DTE typical (configurable), 0-DTE allowed with `ev_high_conf_r` threshold raised
- **Rebalance job** — driven by `portfolio.rebalance_mode`:
  - `threshold` (default): rebalance only when a sleeve drifts more than `rebalance_drift_pct` (default 5pp) from target — avoids churn, lets hot sleeves run
  - `weekly` / `monthly`: time-based schedule
  - `off`: manual only
- Firewall: safe-sleeve signals cannot open convex positions and vice versa. Enforced at `OrderManager` level. **Hardcoded — cannot be disabled via config.**
- Separate dashboards/reporting per sleeve so hot convex streaks don't mask safe-sleeve rot

**Reasoning:**
The theoretical point of a barbell is *bounded downside + unbounded upside* on the convex side — NOT "higher risk for higher return." Most small accounts die from slow erosion in the "middle" (mid-risk swing trades with poor R:R). Concentrating risk in either extreme (very safe or capped convex) is mathematically superior. Threshold-based rebalancing (default) prevents drift from getting extreme while still letting a hot sleeve compound — a weekly schedule forces profit-taking even during a strong run, which fights the strategy.

**Why NOT scale up convex during hot streaks:**
That defeats the entire barbell premise. The point is that the convex sleeve can go to zero and the safe sleeve survives. If the convex sleeve is sized by recent wins, one bad day wipes both. The sleeve *weights* are configurable (you can move from 85/15 to 75/25 as evidence accumulates) but the *amplification based on recent P&L* is not — that's a Kelly-style trap the barbell is specifically designed to avoid.

**Exit criteria:**
- Portfolio reconciliation correctly categorizes every position by sleeve
- Weekly rebalance runs without manual intervention
- Firewall prevents a convex signal from being sized using safe-sleeve equity
- At least 10 paper trades executed in each sleeve

**Dependencies:** Phase 2 (convex sleeve needs EV gate — options slippage/spread is brutal).

---

### Phase 5 — Prediction quality upgrades (weeks 8-10)

**Goal:** Tighter probability estimates → better EV gate precision → more trades clearing the gate with real edge.

**Deliverables in ROI order:**

1. **Model promotion pipeline (FOUNDATIONAL — blocks every other item in this phase)** — fixes Bug C. Today [src/ai_trade/ml/trainer.py:221](../src/ai_trade/ml/trainer.py#L221) runs `_deactivate_prior_versions()` and inserts the new model with `is_active=1` on every training run, with no quality check. A regression-prone retrain silently replaces a working model. Without this fix, everything else below is building on sand — a worse regime head or a worse ensemble member gets promoted just as easily as a better one.
   - **Schema change:** add `promotion_stage TEXT` to `ml_models` with allowed values `candidate` | `shadow` | `active` | `retired` | `rejected`. Migrate existing rows (`is_active=1` → `active`, `is_active=0` → `retired`). Keep `is_active` as a computed view (`is_active = (promotion_stage = 'active')`) for backwards compatibility.
   - **Gate 1 — quality gate at training time** (cheap, ship first):
     - `val_accuracy ≥ current_active.val_accuracy + 0.02` (require measurable lift, not noise)
     - `training_trades ≥ max(30, current_active.training_trades × 0.75)` (no promoting on a tiny sample)
     - Both classes present with `≥20%` minority class (no trivial models)
     - Calibration: max per-bin reliability error `≤ 0.05` (matches Phase 2 calibration bar)
     - On failure: set `promotion_stage='rejected'`, log `model_promotion_rejected` with the specific gate that failed. Active model stays active.
   - **Gate 2 — shadow mode** (optional, ship after Gate 1 proves stable):
     - New candidates that clear Gate 1 enter `promotion_stage='shadow'` instead of going straight to active.
     - Predictor loads both `active` and `shadow` models; shadow predictions are logged to `ml_predictions` (uses Bug B fix) but do *not* affect signals.
     - After `N` shadow trades (default 50, configurable), compare shadow's realized accuracy to active's over the same window. If shadow is ≥1% better, promote: demote active → `retired`, promote shadow → `active`.
     - Shadow-mode config flag: `ml.promotion_mode: gate_only | shadow` so the cheap path can ship before shadow mode is wired.
   - **Rollback mechanism:** keep the last 3 `retired` models' joblib payloads on disk. `--rollback-model <version>` CLI flips `promotion_stage` back to `active` on the chosen version and demotes the current active to `retired`. No retraining required.
   - **Cheapest pre-Phase-5 fix:** Gate 1 alone is a ~20-line change to `trainer._deactivate_prior_versions()` — read current active's metrics, compare, abort on failure. Worth shipping during Phase 0 as part of the bug fixes; it removes the single biggest silent-regression risk with almost no engineering cost. Shadow mode is the full Phase 5 deliverable.
2. **Per-regime ML heads** — train three GradientBoostingClassifier instances (bull / bear / chop), dispatched by the existing regime detector at predict time. Stored alongside current `signal_quality_v3.joblib`. Each head runs through the promotion pipeline (#1) independently — the bull head can be in shadow while bear stays active.
3. **Meta-labeling (López de Prado):** two-stage model — first predicts "take the trade?" (binary), second predicts "what conviction?" (score). Dramatically reduces false positives.
4. **Microstructure features:** add bid-ask spread, trade imbalance, sector relative strength, IV rank for options. These are currently missing from the feature set in [src/ai_trade/ml/features.py](../src/ai_trade/ml/features.py).
5. **Ensemble voter:** GradientBoost + LogisticRegression + a rule-based baseline, weighted by rolling 30-day live accuracy. Robust to any individual model breaking.
6. **Drift detector:** compare live win rate to backtest win rate over trailing 30 trades. If live is >2σ below backtest, auto-shrink all position sizes to 50% of normal until recovery. Emit a `prediction_drift_detected` log event. Coupled to the promotion pipeline: a sustained drift signal can auto-demote the active model to `retired` and roll back to the previous `retired` model.

**Reasoning:**
Phase 2 built an EV gate. Phase 5 makes that gate more discriminating. Better calibrated probabilities → gate rejects fewer good trades and catches more bad ones.

Item #1 (promotion pipeline) is ordered first because it is **load-bearing for the entire phase**: without it, a regime-head retrain, an ensemble update, or a meta-labeling run can silently replace a working model with a worse one. Drift detection (item #6) only helps *after* a bad model is already live — Gate 1 stops it from getting there in the first place. The cheapest pre-Phase-5 slice (Gate 1 only) can and should ship during Phase 0 alongside the other ML bug fixes.

Within the remaining items: regime heads are the biggest per-hour-of-work gain because regime detection already exists; meta-labeling is the biggest theoretical lift but requires labeling pipeline changes; drift detection is the cheapest but most valuable long-term safety net.

**Exit criteria:**
- Promotion pipeline: a deliberately-degraded retrain (e.g., trained on 10% of data) is correctly routed to `rejected` without touching the active model. Verified via test.
- Rollback: `--rollback-model` command restores a prior version within 30 seconds, no retraining.
- Holdout accuracy improves ≥3% over v3 baseline (measured on the newly-active model after at least one successful promotion)
- Calibration reliability diagram still within ±5% after regime/meta-labeling changes
- Drift detector fires correctly when backtest data is artificially degraded, and (if configured) triggers an auto-rollback

**Dependencies:** Phase 0 (holdout methodology *and* Bug A/B fixes — promotion logging is useless if predictor logs are dropped), Phase 2 (calibration infra — Gate 1's calibration check reuses Phase 2's reliability diagram).

---

### Phase 6 — Observability (week 11, can run parallel to 4-5)

**Goal:** Know *immediately* when something breaks, not at EOD review.

**Deliverables:**
- **Daily paper-vs-backtest divergence report**: for each strategy, compare today's (or this week's) metrics to the backtest baseline. Flag any >1σ deviation.
- **Rolling 30-trade metrics**: Sharpe, profit factor, win rate, expectancy — published to a file or dashboard so trend is visible day-over-day.
- **Alerts**: email/Discord on edge degradation, slippage outliers (>2× model prediction), strategy-specific drawdown >2%, circuit breaker trips.
- **Full trade attribution**: every trade tagged with `strategy`, `regime`, `ml_score_bucket`, `sleeve` — enables breakdown of where P&L comes from.
- **Structured log shipping (optional)**: if the owner wants to query history beyond local SQLite, integrate a remote sink.

**Reasoning:**
Most algo blowups are preceded by a week of subtle degradation that's invisible until EOD review catches it. Real-time visibility lets the owner (or future automated circuit breakers) intervene before a bad day becomes a blown account.

**Exit criteria:**
- Owner can answer "how is the bot doing right now?" in <10 seconds without touching code
- At least one simulated degradation scenario triggers an alert correctly

**Dependencies:** none architectural, but most valuable after Phase 2-4 are producing rich trade attribution data.

---

### Phase 7 — Go-live pilot (week 12+)

**Goal:** Catch any remaining live-vs-paper gaps before full capital is exposed.

**Deliverables:**
- **Funded live account with $1,000 initially** (50% of target capital) — smaller blast radius while validating, faster ramp than starting at $500
- Run live and paper in parallel for 1 week (not 2 — the slippage model from Phase 2 has already quantified most of the gap), executing the same strategy configuration
- Daily comparison report: live vs paper Sharpe, win rate, avg R, realized slippage per trade
- Acceptance criteria:
  - If live Sharpe is within 0.5 of paper Sharpe AND realized slippage is within 50% of modeled slippage → scale to full $2k after 1 week
  - If Sharpe gap is 0.5-1.0 OR slippage 50-100% higher than modeled → extend pilot another week, investigate root cause before scaling
  - If Sharpe gap >1.0 OR slippage 2×+ modeled → HALT live, do not scale, return to roadmap (probably Phase 2 slippage recalibration)
- Kill switch from Phase 1 must be tested in live within the first day — small real trade, manually triggered flat, confirm execution

**Reasoning:**
Paper and live are different environments regardless of how good the slippage model is. Network latency, order routing, partial fills, and margin behavior all differ. Funding $1k first (instead of $2k) means a catastrophic slippage discovery costs half as much while preserving the ability to actually trade meaningfully — $500 is too small for strategies that risk 2.5% per trade ($12.50 risk is below commission+slippage on many names). One week is enough to expose systematic gaps; two weeks was overly conservative once Phase 2 slippage modeling exists.

**Exit criteria:**
- 2-week live pilot complete with quantified paper-to-live gap
- Gap is understood and accepted before scaling, not discovered mid-scale

**Dependencies:** all prior phases complete.

---

## Priority ordering

If working with limited time, ship in this order:

1. **Phase 2 (EV gate + slippage + sizing)** — biggest real-edge uplift. The single change that most affects live profitability.
2. **Phase 1 (guardrails)** — prevents account blowup regardless of edge quality. Non-negotiable before live.
3. **Phase 0 (baseline validation)** — prevents chasing a mirage; informs every subsequent phase.
4. **Phase 3 (PDT config)** — required before going live *and* required for the regulatory change, whenever it lands.
5. **Phase 7 (live pilot)** — must happen last, but plan it early.
6. **Phase 4 (barbell)** — real edge but diminishing returns vs Phase 2.
7. **Phase 5 (prediction upgrades)** — refinement, not foundation.
8. **Phase 6 (observability)** — run in parallel with Phase 4-5.

**Minimum viable pre-live set**: Phases 0, 1, 2, 3, 7.

Phases 4, 5, 6 are post-live enhancements if time-constrained.

---

## What to explicitly NOT do

- **Kelly sizing (full or fractional)** — requires calibrated probabilities you won't have until Phase 5. Even then, the three-layer sizer from Phase 2 is more robust to estimation error. Revisit after 200+ live trades.
- **News sentiment as a primary signal** — current scan is best-effort; improving it has low ROI until calibration lands. Keep as contextual modifier only.
- **Regime detection rewrite** — current bull/bear/chop classifier works. Do not touch until Phase 5 needs per-regime models.
- **Real-time streaming bars** — batch scans at 15-min intervals match the timeframe of the strategies. Streaming adds complexity with no edge improvement at this horizon.
- **More strategies** — 6 stock + 7 options strategies is plenty. Depth (better sizing, better EV gate) beats breadth (more strategies with marginal edge).
- **Portfolio optimization (Markowitz/risk parity)** — not useful for 1-3 concurrent positions on a $2k account. Wait for $25k+.
- **Preserving the 5.22 Sharpe as a metric goal** — it's a small-sample artifact; see the reframing section.

---

## Handoff notes for the incoming agent

### Start here
1. Read this entire document before writing any code.
2. Read [docs/V2_AGENT_BRIEF.md](V2_AGENT_BRIEF.md) for original V2 context.
3. Read [docs/ARCHITECTURE.md](ARCHITECTURE.md) and [docs/RISK_MANAGEMENT.md](RISK_MANAGEMENT.md) for current system design.
4. Run the current backtest to establish the cost-free baseline: `python -m ai_trade.backtest.runner --full-universe --capital 100000`.
5. Open a Phase 0 branch; do not touch other phases until 0 is complete and documented.

### Conventions
- Branch per phase: `phase-0-baseline`, `phase-1-guardrails`, etc.
- Config changes live in `settings.yaml` with schema validation in [src/ai_trade/config.py](../src/ai_trade/config.py)
- Every new risk rule has a unit test that would fail if the rule weren't applied
- Update [docs/V2_PROGRESS.md](V2_PROGRESS.md) at the end of each phase with what shipped
- Bump version (`src/ai_trade/_version.py`) per semver: PATCH for fixes, MINOR for phase completion, MAJOR only for breaking strategy-logic changes

### Things to ask the owner before acting
- **Live account funding:** confirm actual live deposit amount before Phase 7 pilot
- **PDT rule effective date:** if the owner has updated info on when Alpaca turns on the new rule, adjust Phase 3 priority accordingly
- **Broker-specific constraints:** verify Alpaca's current live-account requirements (may differ from FINRA minimums)
- **Risk tolerance:** the 1.5% per-trade / 3% daily DD caps in Phase 1 are starting proposals; confirm with the owner before implementation

### Escalate to the owner (do not silently decide)
- Any change that touches risk caps
- Any Phase 3 PDT config defaults — regulatory assumptions should be owner-approved
- Any change to the barbell sleeve percentages
- Going to live trading — Phase 7 funding decision is owner-only
- Any modification to kill switch behavior

### Do not delete or significantly modify
- [data/ai_trade.db](../data/ai_trade.db) — live trade history
- [data/ml_training_universe.txt](../data/ml_training_universe.txt) — curated 283-symbol universe, regeneration is expensive
- [src/ai_trade/models/signal_quality_v3.joblib](../src/ai_trade/models/signal_quality_v3.joblib) — v3 model; keep as fallback even when v4+ trains
- Anything under [logs/](../logs/) without confirming — evidence for debugging live issues

---

## Success definition

The roadmap is complete when:

1. **The bot is live on $2,000** with no open questions about its safety
2. **Live Sharpe is within 0.5 of the cost-aware backtest Sharpe**, proving the simulation is honest
3. **Guardrails have tripped at least once in paper or live** without intervention — proof they work
4. **The PDT rule change can be accommodated by flipping a config value**, not by shipping code
5. **30-day rolling metrics are visible at a glance** without querying the database

If all five are true, the system is ready for scaled capital and ongoing optimization.
