# Expiry Lab (research only)

Separate from Signal Desk: imports its rule code read-only, shares the 15-min index cache in `range_lab/cache`.
Never writes into `signal_desk/`, never alerts, never places orders.

## Question
Do Signal Desk's live Nifty breakout TAKE signals behave differently on weekly expiry day (heavy option-writer
hedging: pinning or acceleration)?

## Rules (fixed 22 Oct 2026, before any run)
- Expiry: Thursday till 28 Aug 2025, Tuesday from 2 Sep 2025; previous trading day if a holiday.
- Groups: expiry day / day before / other days. Live rules replay (13:00 cutoff + trend), ₹60 costs, older vs newer year.
- PASS: expiry-minus-other avg R has the same sign in both years, |gap| >= 0.15 R in both, >= 15 expiry trades each
  year, two-sided day-shuffle p <= 0.10. If PASS -> a grade adjustment (A<->B) on expiry days, never a block.

## Run
`python -m expiry_lab report` (cache only) or `run_expiry_lab.bat` after 3:35 PM. Results: `expiry_lab/results/report.txt`.
