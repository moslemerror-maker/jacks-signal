# Project Rules

## Mission

Build a serious, research-first quantitative trading platform for Indian markets.

The system is intended to research and validate statistical edges across equities, indices, futures and options, with portfolio/risk management and controlled execution only after sufficient evidence.

The objective is NOT 90% prediction accuracy.

The objective is:

> durable, statistically defensible expected value after realistic costs, slippage, changing regimes and out-of-sample validation.

## ABSOLUTE FIRST RULE — DO NOT START CODING IMMEDIATELY

On a fresh repository or major new stage, do not immediately implement the project.

The first job is to:

**UNDERSTAND → EVALUATE → PLAN → CHALLENGE → SUGGEST → APPROVAL → IMPLEMENT → TEST → REPORT**

For this project, **Phase 0 — Existing-System & Technology Evaluation is mandatory before Phase 1 implementation.**

An honest assessment of whether parts of this system already exist and should be reused comes first.

## Phase 0 requirements

Before substantial implementation:

1. Read `PROJECT-RULES.md` and `README.md`.
2. Inspect the repository.
3. Evaluate mature existing systems and libraries.
4. Evaluate build-vs-buy for each major subsystem.
5. Verify current capabilities rather than relying on assumptions.
6. Identify India-specific gaps.
7. Identify licensing and lock-in risks.
8. Propose a revised architecture.
9. Estimate engineering effort and potential savings from reuse.
10. Give honest criticism and suggestions.
11. STOP.

Do not create the production data pipeline, strategy engine, broker execution or dashboard during Phase 0.

## Mandatory approval workflow

Before any non-trivial edit:

### 1. UNDERSTAND
Read the relevant code and inspect the existing implementation.

Do not edit during understanding.

Identify:

- current state,
- intended state,
- assumptions,
- dependencies,
- risks,
- unknowns,
- missing requirements.

### 2. EVALUATE
Where an existing framework/product/library may solve the problem:

- investigate it,
- compare it,
- test it where practical,
- document what is verified.

Do not reinvent solved infrastructure without justification.

### 3. PLAN
State:

- exact files/modules,
- architecture,
- data/database implications,
- tests,
- acceptance criteria,
- risks,
- rollback/recovery considerations.

### 4. SUGGEST
Separate:

- required changes,
- recommended improvements,
- optional enhancements,
- trade-offs.

### 5. APPROVAL
STOP.

Wait for explicit approval.

Never interpret:

- silence,
- previous approval,
- a vague "okay",
- or the existence of a plan

as authorization for substantial implementation.

### 6. IMPLEMENT
After approval:

- make the smallest coherent change,
- preserve architecture decisions,
- do not silently expand scope,
- do not introduce unnecessary dependencies.

### 7. TEST
Run relevant:

- unit tests,
- integration tests,
- data-quality checks,
- type checks,
- linting,
- reproducibility checks,
- research-bias checks.

### 8. REPORT
Explain:

- what changed,
- what was tested,
- what passed,
- what failed,
- what was verified,
- what remains uncertain,
- what should happen next.

## Research integrity rules

Absolute:

- No look-ahead bias.
- No survivorship bias.
- No future information in historical features.
- Point-in-time data wherever required.
- Chronological train/test splits.
- Walk-forward validation.
- Realistic brokerage, taxes, fees, spread and slippage.
- Model liquidity and execution assumptions.
- Probability calibration matters.
- Expected value matters more than raw accuracy.
- NO TRADE is a valid output.
- Portfolio/risk can veto any signal.
- Do not optimize repeatedly against the same test set.
- Complexity must be justified by robust out-of-sample improvement.
- Stress-test promising strategies.
- Strategy health and drift must be monitored.
- Do not automatically retrain/deploy because performance falls.

## Data integrity rules

- Raw source files are immutable.
- Preserve source provenance.
- Preserve checksums where possible.
- Never silently overwrite corrected/re-published source data.
- Legacy and current NSE formats must have separate versioned normalizers.
- Canonical schema must remain source-independent.
- Store raw/unadjusted prices as source truth.
- Corporate-action adjustments are derived.
- Instrument identity must support symbol changes.
- Universe membership must support effective dates.
- Trading calendar must be explicit.
- Ingestion must be idempotent.
- Reprocessing must be deterministic.

## Architecture rules

Signals must be isolated from broker execution.

The conceptual boundary is:

```text
DATA
  ↓
VALIDATION
  ↓
RESEARCH DATA
  ↓
FEATURES
  ↓
REGIME
  ↓
STRATEGY
  ↓
MODEL / PROBABILITY / EV
  ↓
PORTFOLIO
  ↓
RISK / HEDGE
  ↓
TRADE / WAIT / NO TRADE
  ↓
PAPER
  ↓
CONTROLLED EXECUTION
```

Execution must have its own safety controls.

A feed failure must never cause an unsafe trade.

Never hard-code:

- API keys,
- passwords,
- broker secrets,
- tokens.

## Real-money rule

Never begin with automated real-money execution.

Progress through:

```text
Research
→ Backtest
→ OOS
→ Walk-forward
→ Stress
→ Paper
→ Shadow/live observation
→ Tiny controlled capital
→ Gradual scale
```

Engineering success does not imply trading profitability.

## Scope discipline

Do not implement all phases at once.

Do not add:

- ML,
- options,
- intraday,
- live trading,
- frontend,
- cloud infrastructure

to an earlier phase merely because they are interesting.
