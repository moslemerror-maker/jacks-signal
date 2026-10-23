# Jack's Signal

An alerts-only toolkit for Indian markets (NSE), built on the Dhan data API: a live Nifty options alert desk,
a set of honest research labs that test trading ideas on real data, and the research platform underneath.

> Not financial advice. Nothing here places orders. Every rule was tested with realistic costs, and most ideas
> failed - the results are published below exactly as they came out.

## What's inside

| Folder | What it is |
|---|---|
| `signal_desk/` | Live Nifty option alerts: 15-min breakout rules, trend filter, OI walls, risk guard, Telegram + local web page, auto token renewal, Windows autostart |
| `range_lab/` | Previous-day range (narrow / wide) as a grade adjustment - **passed**, used in Signal Desk |
| `momentum_lab/` | First half-hour predicts last half-hour (US research) on Nifty - **failed** |
| `expiry_lab/` | Do breakouts behave differently on weekly expiry day? - **failed** |
| `expiry_0dte_lab/` | Real expired-option premiums: 0DTE straddle (failed) and 0DTE vs next-week contract (next week wins) |
| `premium_lab/` | The Signal Desk rules re-tested with real option premiums - **mixed** (thin edge after costs) |
| `exit_lab/` | Premium stop/target exits vs Nifty-level exits - **failed** (Nifty levels are better) |
| `stock_lab/`, `intraday_lab/` | Intraday F&O stock ideas (9:17 movers, news vs no-news, stocks-in-play ORB, relative strength) - **all failed after costs** |
| `backend/`, `frontend/`, `scripts/` | The underlying research platform (data ingestion, research pipeline, dashboard) |

Every lab fixes its rules and pass mark before the first run, splits the data into older and newer periods,
compares against a random benchmark and reports a luck check. `intraday_lab` also keeps a locked "vault" period.

## Quick start (Signal Desk)

1. Python 3.12 + [uv](https://docs.astral.sh/uv/) (or a plain venv with `requests`).
2. Copy `.env.example` to `.env` and set `DHAN_CLIENT_ID`, `DHAN_ACCESS_TOKEN` (Dhan Data API plan needed);
   Telegram bot token / chat id can be added later from the page.
3. `python -m signal_desk simulate` - replays a synthetic day offline.
4. `signal_desk\check_signal_desk.bat` - checks every Dhan call; `signal_desk\start_signal_desk.bat` - starts it
   (page at http://127.0.0.1:8765). `install_autostart.bat` runs it every weekday 9:00-15:40.
5. Tests: `python -m signal_desk.tests`, and `python -m <lab>.tests` for each lab.

Details: `signal_desk/README.md` and the README in each lab folder.

## What the research found (short version)

- The breakout core with a trend filter and a 13:00 cutoff was profitable on index-point estimates, but with real
  option premiums it is close to break-even (PF 1.01 and 1.13 in two years). Paper-trade it, don't size it up.
- Exit on Nifty levels, not on option-premium stops. On expiry day, buy next week's contract, not the expiring one.
- Short-term intraday moves in F&O stocks average about +/-0.1-0.15% before costs, while a cash round trip costs
  about 0.18%. None of the simple intraday stock ideas survived that.

## Research platform (backend/, frontend/)

Research-first quantitative platform for Indian stocks, indices, futures, options, portfolio management and hedging.

## Important: This is not a "build everything from scratch" project

Before implementation, we evaluate mature systems and decide what should be:

- reused,
- integrated,
- extended,
- or built ourselves.

The project's value should come from the parts that genuinely need custom development, especially India-specific data/research infrastructure and the research/intelligence layer.

## Core philosophy

The goal is not 90% prediction accuracy.

The goal is:

> durable statistical edge after realistic costs, slippage, risk, changing regimes and out-of-sample validation.

The system must be comfortable producing:

- TRADE
- WAIT
- NO TRADE

## Mandatory first stage

### Phase 0 — Technology & Existing-System Evaluation

Before writing substantial project code, evaluate:

- QuantConnect LEAN
- Backtrader
- VectorBT
- maintained alternatives
- AlgoTest
- Tradetron
- relevant Indian broker APIs
- relevant market-data providers

Then produce a build-vs-buy decision.

## Revised architecture

Conceptually:

```text
Existing Infrastructure / APIs
             ↓
      Adapter Interfaces
             ↓
     Canonical Data Layer
             ↓
      Research Platform
             ↓
      Features / Regimes
             ↓
     Strategy / Models
             ↓
    Probability + Expected Value
             ↓
       Portfolio / Hedge
             ↓
          Risk Engine
             ↓
      TRADE / WAIT / NO TRADE
             ↓
       Paper / Execution
             ↓
     Health / Drift / Research
```

The exact backtesting/execution engine is intentionally not fixed until Phase 0.

## Initial data decisions

Phase 1 target:

- NIFTY 50 index
- BANKNIFTY index
- configurable ~25-stock liquid equity basket
- daily EOD data
- historical start target: 2015-01-01
- official NSE EOD data as initial authoritative source
- manual file-drop initially
- raw immutable Parquet + curated PostgreSQL
- Docker Compose PostgreSQL
- Python 3.12 + uv

## Research integrity

- No look-ahead bias.
- No survivorship bias.
- Point-in-time data.
- Chronological/walk-forward validation.
- Realistic costs/slippage.
- Probability calibration.
- Expected value.
- Risk veto.
- Strategy health/drift.
- Controlled execution only after evidence.

## Development workflow

Always:

**UNDERSTAND → EVALUATE → PLAN → SUGGEST → APPROVAL → IMPLEMENT → TEST → REPORT**

Read `PROJECT-RULES.md` first.

Nothing gets built before the repository is understood, existing solutions are evaluated, assumptions are challenged and a plan is agreed.

