# Expiry 0DTE Lab (research only)

Separate from Signal Desk: imports its rule code read-only, reads `.env` for the token, shares the 15-min index cache
in `range_lab/cache`, keeps its own option cache in `expiry_0dte_lab/cache`. Never writes into `signal_desk/`, never
alerts, never places orders.

Data: Dhan `/v2/charts/rollingoption` (expired options data) - real past Nifty weekly premiums, 5-min candles,
strikes relative to ATM. The lab works out which `expiryCode` is the expiring week (cheapest ATM straddle at 15:15).

## Questions and pass marks (fixed 22 Oct 2026, before any run)
Q1 (primary): on expiry day buy the ATM call + put of the EXPIRING week at 9:20, sell 15:15 (same strike).
  PASS (buyers' edge): both years >= 40 days, PF >= 1.2, sign-flip p <= 0.10.
  PASS (sellers' edge): the same with PF <= 0.8 in both years. Otherwise FAIL.
Q2: Signal Desk's expiry-day TAKE signals - expiring (0DTE) ATM option vs next week's ATM option, same entry/exit
  times (half at T1). Verdict only if the paired difference has the same sign in both years and p <= 0.10.

## Run
After 3:35 PM double-click `expiry_0dte_lab\run_expiry_0dte_lab.bat`. It asks Dhan for one expiry day at a time
(expiryCode 1 = the expiring week, 2 = next week; Dhan rejects 0), only for the strikes it needs - roughly 8-10
requests per expiry day. Everything is cached, so you can close the window and re-run: it resumes where it stopped. `python -m expiry_0dte_lab report` re-runs from the cache.
Results: `expiry_0dte_lab\results\report.txt`. Tests: `python -m expiry_0dte_lab.tests`.
