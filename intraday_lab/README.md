# Intraday Lab (research only)

Intraday ideas on NSE F&O stocks with 1-minute data and real cash-intraday costs. Imports Signal Desk / Stock Lab code
read-only (scrip-master parser, Dhan client, cost model); writes only into `intraday_lab/`.

## The vault
Everything from **23 Apr 2026** onwards is locked. Ideas are explored only on 23 Oct 2024 - 22 Apr 2026 (split into an
older and a newer half). Only versions that pass exploration are run on the vault, ONCE (`python -m intraday_lab vault b`).

## Idea B - the 9:17 movers, short holds (rules fixed 23 Oct 2026, before any run)
Eligible: 20-day median turnover >= ₹50 cr, previous close >= ₹100 (previous days only). Previous close = VWAP of the
15:00-15:30 bars. At 9:17 rank by change from previous close; top 3 gainers + top 3 losers. FOLLOW and FADE.
Entry 9:17 open. Exits: target/stop 0.3%, 0.5%, 1.0% (else 10:15), or time exits 9:30, 9:45, 10:15.
₹1 lakh per trade, Dhan intraday charges + 0.05% slippage per side. Benchmark: 6 random eligible stocks, random side.
PASS: both halves net > 0, PF >= 1.2, better than random; random-direction p <= 0.01 (strict: 12 versions).
Vault PASS: net > 0 and PF >= 1.1.

## Run
After 3:35 PM double-click `intraday_lab\run_intraday_lab.bat` (first run downloads 2 years of 1-min candles for the
F&O stocks - roughly 30-60 minutes; later runs only add new days). Results: `intraday_lab\results\idea_b.txt`.
Ideas C (stocks-in-play ORB, short exits) and D (relative strength vs Nifty) come next, on the same data.
