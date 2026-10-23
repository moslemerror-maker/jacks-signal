# Stock Lab (research only)

Separate from `signal_desk/`: it never writes there and never changes live Nifty alerts. It reads Dhan data with
the same `.env` credentials, stores 5-minute candles in `stock_lab/cache/`, and writes results to `stock_lab/results/`.

## Question
Can we pick 1-3 F&O stocks early in the day (on 5-minute candles) whose moves are big enough to beat cash-intraday
costs? Ideas compared: % change (your idea), relative volume ("stocks in play"), both, and a random benchmark.

## Gate 0 - is the raw move big enough? (no costs, no stops)
Each day, rank eligible stocks at 9:20 / 9:30, take the top 3, and measure the move to 15:00 in the direction of the
opening move ("follow") and the opposite ("fade"). PASS only if the best rule averages >= 0.25% per trade, with the
same sign, in both the older and the recent 120 sessions, t >= 2.

## Gate 1 - full rules with costs
Top-3 by 5-min relative volume (RV >= 2) at 9:20; stop order beyond the first 5-min candle (9:15-9:20) until 10:30;
stop at its other side (widened to at least 0.5%; skipped if over 2% of price); exit at 15:00 (or 2R). Rs 1 lakh per trade (about Rs 20k margin at 5x),
Dhan charges + 0.05% slippage per side. PASS: >= 200 older trades, PF >= 1.2, t >= 3, and PF >= 1 on the recent period.

## Run (after 3:35 PM)
Double-click `stock_lab\run_stock_lab.bat` (first run downloads ~2 years of 5-min candles: 15-30 min).
Options: `--years 3`.

## Not yet included (planned)
- News flag: free sources only (NSE corporate announcements, exchange filings). Added as a live filter / Gate 1b
  once Gate 0/1 show a raw edge worth refining - historical free news is too patchy to backtest reliably.
- Support/resistance: only as an exact, testable rule (e.g. distance to previous-day high/low), never by eye.
