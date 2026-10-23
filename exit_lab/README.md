# Exit Lab (research only)

Same Signal Desk signals, same option (as alerted), REAL premiums: does exiting on the PREMIUM stop/target that every
alert shows beat exiting on the Nifty index levels? Reuses Premium Lab's trade rules and Expiry 0DTE Lab's downloader
and option cache. Never writes into `signal_desk/`, never alerts, never places orders.

## Rules (fixed 23 Oct 2026, before any run)
- Premium exit (delta 0.5, r = the signal's Nifty risk): stop = entry - 0.5r, T1 = entry + 0.75r (half off, stop to
  entry), T2 = entry + 1.25r, else 15:15. Checked on every 5-min close, filled at that close.
- Index exit: Premium Lab's (half at Nifty T1, rest at stop / T2 / cost / 15:15).
- PASS: premium beats index (avg ₹/trade) in both years, premium avg ₹ > 0 and PF >= 1.10 in both years, paired
  sign-flip p <= 0.10. Otherwise FAIL. Needs >= 80% of trades with a full price path.

## Run
After 3:35 PM double-click `exit_lab\run_exit_lab.bat` (downloads the rest of each trade's price path, 4 at a time;
cached and resumable). Results: `exit_lab\results\report.txt`.
