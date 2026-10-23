# Range Lab (research only)

Separate from Signal Desk: it only *imports* Signal Desk's rule code (read-only) and reads `.env` for the Dhan token.
It never writes into `signal_desk/`, never sends alerts, never places orders.

## The one question
Are Signal Desk's live Nifty breakout signals better on days that follow an unusually **narrow** day?

- Measure (known before 9:15): `yesterday's range / average range of the 10 days before yesterday`
- Buckets: narrow / normal / wide = terciles, with cut-offs taken from the **older half only**
- Gate 0: raw days - do days after a narrow day expand or trend more?
- Gate 1: the exact live rules (13:00 cutoff + trend filter), TAKE signals, ₹60 costs, older vs newer half

## Pass mark (fixed before the first run - do not change after seeing results)
In BOTH halves: narrow-bucket trades >= 15, narrow avg R beats wide avg R by >= 0.10, narrow PF >= 1.2;
and a day-shuffle test (2,000 random re-labellings of days) gives p <= 0.10.

- PASS -> worth adding to Signal Desk as a grade booster (not a hard filter), after you approve
- PROMISING -> right direction in both halves but below the bar; keep as a paper idea
- FAIL -> drop it

## Run
After 3:35 PM, double-click `range_lab\run_range_lab.bat` (first run fetches ~2 years of 15-min index candles, a few minutes).
`python -m range_lab report` re-runs from the cache. Options: `--years 2` (1-4), `--force`.
Results: `range_lab\results\report.txt`. Tests: `python -m range_lab.tests`.
