# Premium Lab (research only)

Re-tests Signal Desk's live Nifty rules with REAL past option premiums (Dhan expired-options data) instead of the
"index points x 0.5 delta" estimate used by every earlier lab. Imports Signal Desk's rule code read-only and reuses
Expiry 0DTE Lab's downloader and option cache (`expiry_0dte_lab/cache`). Never writes into `signal_desk/`, never
alerts, never places orders.

## Rules (fixed 23 Oct 2026, before any run)
- Signals: live rules replay (13:00 cutoff + trend filter), TAKE only, 2 years.
- Contract as alerted: nearest weekly expiry; next week's on expiry day. Strike: CE at/below entry, PE at/above.
- Exits from the index rules: half at T1, rest at stop / T2 / back to cost / 15:15. Lot 65, ₹60 per trade.
- SURVIVES if REAL avg ₹ > 0 and PF >= 1.10 in both years (MIXED = one year, FAILS = neither); >= 80% of signals
  must be priced, otherwise INCOMPLETE.

## Run
After 3:35 PM double-click `premium_lab\run_premium_lab.bat` (about 1,000 option requests, 4 at a time; cached, so
re-running resumes). `python -m premium_lab report` re-runs from the cache. Results: `premium_lab\results\report.txt`.
