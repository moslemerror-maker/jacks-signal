# Momentum Lab (research only)

Separate from Signal Desk: imports its candle code read-only, reads `.env` for the token, and shares the 15-min index
cache in `range_lab/cache`. Never writes into `signal_desk/`, never alerts, never places orders.

## The question
Published research (Gao, Han, Li, Zhou, JFE 2018, S&P 500) found the first half-hour predicts the last half-hour.
It held in most developed markets, was mixed in Asia (China/Japan yes, Hong Kong/Singapore no), absent in Australia,
and has **no published test on Nifty** that we could find. So this lab is that test.

## Rule (fixed 22 Oct 2026, before any run)
- r1 = prev close -> 9:45 close; rpen = 14:30 -> 15:00. Same sign -> buy CE (up) / PE (down) at 15:00, exit 15:30.
- Rupees = Nifty points x 0.5 delta x 65 - ₹60. No stop (30-minute hold).
- Info only: first half-hour alone, 15:15 exit, high/low VIX days, approximate expiry days.

## Pass mark (fixed before any run)
Both years: >= 40 trades, avg ₹ > 0, PF >= 1.15, hit rate > 50%; random-direction test p <= 0.10.

## Run
`python -m momentum_lab report` (uses the cache, no Dhan calls), or double-click `run_momentum_lab.bat`
after 3:35 PM to refresh the cache first. Results: `momentum_lab/results/report.txt`.
