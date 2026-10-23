# Signal Desk

Live Nifty option-chain + F&O stock **alert** assistant on Dhan data, with Telegram alerts and a local web page.
It never places orders - you decide and place trades yourself.

## Start
1. `.env` (project root) needs `DHAN_CLIENT_ID` and `DHAN_ACCESS_TOKEN` (Dhan tokens expire every 24 h -
   Signal Desk renews it automatically at 15:30 each day it runs (Dhan /RenewToken), and at start-up if it would run
   out before the close. An already-expired token cannot be renewed: after a weekend/holiday paste one manually in
   Settings and the loop continues. If renewal fails you get a Telegram alert. Turn off with SD_TOKEN_AUTORENEW=0).
2. Double-click `signal_desk\check_signal_desk.bat` once - it tests every Dhan call and prints PASS/FAIL.
3. Double-click `signal_desk\start_signal_desk.bat` before 9:15. The page opens at http://127.0.0.1:8765.
4. Settings -> Telegram: paste bot token, Save, send "hi" to your bot, "Find my chat ID", "Send test".

Try it offline any time: `python -m signal_desk simulate` (replays a synthetic day and prints every alert).

## What it does
| Time | What happens |
|---|---|
| from 08:00 | Builds a profile of every F&O stock from daily candles; ~9:00 sends a pre-market watchlist |
| every 30 s | Nifty, Bank Nifty, VIX quotes; your logged trades' prices; stop/target/breakeven alerts |
| every 60 s | Nifty option chain: call/put OI walls, PCR, max pain, OI-change bias |
| each 15-min close (from 9:30) | Nifty breakout/breakdown check -> graded trade plan (strike, expiry, premium SL/T1, risk per lot); F&O stock scan (shortlist from one batch quote, opening-range/day-high break with relative volume, VWAP, prev-day level, Nifty direction) |
| near a trigger | "Get ready" heads-up (max one per 30 min) |
| 15:05 / 15:15 | Exit reminder / square-off alert for open trades |
| 15:31 | End-of-day summary; every signal's outcome (T1 / stop / timeout, in R) appended to `state/journal.csv` |

## Nifty rules (nifty.py)
- Trigger: a completed 15-min candle **closes** beyond every earlier high (CE) / low (PE) of the day, after 9:30.
- Stop: breakout candle's low/high (+3 pts), kept between 20 and 60 points. T1 = 1.5R, T2 = 2.5R (T1 pulled below a near OI wall).
- Checks: time before 13:00 (Nifty; SD_NIFTY_CUTOFF - rule testing showed later entries lost money), close near candle extreme (no big wick), room to the call/put OI wall, Bank Nifty agreeing,
  VIX not working against the option, OI-change flow, rupee risk of 1 lot within budget, daily risk guard.
- Grade A = all pass, B = one warning, C = any fail or 2+ warnings -> SKIP (shown, not pushed).
- Strike: ATM / slightly ITM on the nearest weekly expiry, but never the expiring contract on expiry day.
- A second same-side signal needs a fresh base (2+ candles without a new extreme) first.

## Trend filter and 1-hour paper alerts (since 10 Oct 2026)
- Trend filter (on by default, Settings > Alert rules): CE only when the 15-min close is above today's open AND
  yesterday's close; PE only when it is below both. Breakouts against the day show as SKIP with reason "Trend".
- 1-hour PAPER alerts: the same breakout rules on 1-hour candles (9:15-10:15 range; signals at 11:15 / 12:15).
  Labelled "PAPER ONLY" everywhere and tracked in the journal - for evidence, not for real money yet.

## Calls on paper (since 10 Oct 2026)
- Rule testing split by side (Oct 2025 - Oct 2026): puts (PE) made all the profit in both periods (PF 2.03 / 1.75);
  calls (CE) broke even or lost (PF 1.00 / 0.66) in every variant. So Nifty CE signals are sent as "PAPER ONLY" by
  default (Settings > Alert rules > Nifty calls). Caveat: that year Nifty fell, so re-check calls when the trend turns.

## Risk rules (.env, optional)
`SD_DAILY_LOSS_WARN=1000`, `SD_DAILY_LOSS_LIMIT=1500`, `SD_MAX_TRADES=2`, `SD_OPTION_RISK=1500`, `SD_STOCK_RISK=500`,
`SD_NIFTY_CUTOFF=13:00`, `SD_ENTRY_CUTOFF=14:30` (stocks), `SD_EXIT_REMINDER=15:05`, `SD_SQUAREOFF=15:15`, `SD_MIN_GRADE=B`, `SD_PORT=8765`.
Log the trades you actually take in the page ("I took this" pre-fills it) - that is what drives P&L, limits and stop alerts.

## Backtest and rule testing
- **Backtest** card: replays the live Nifty rules on the last 20-60 sessions of real 15-min candles.
- **Rule testing** card (or `python -m signal_desk research --days 250 --seen 60`): runs the live rules and 12 variants
  (first breakout only, trend filter, ATR stop, 30-pt min stop, 13:00 cutoff, T1 at 1R, no BNF/VIX filters, ...) on
  two periods - the recent 60 sessions ("seen") and the older sessions before them ("unseen"). A variant counts only if
  it is profitable and better than live on BOTH. Results are saved to `state/research.json`. It never changes live alerts.

## Honest limits
- These are rule-based alerts, not a proven edge. The journal tracks every signal's outcome so you can judge after a few weeks.
- Polling (30-60 s, candles at the 15-min close), not tick-by-tick. Keep broker price alerts as a backup.
- Lot sizes and index ids come from Dhan's scrip master daily; check the page's Errors panel if something looks off.
