"""Phase P: the stock/intraday live-data collector entry point.

Independent of, and never importing from, backend.app.options.* - this is the STOCK collector, not the
options live tracker. It shares only the Dhan credentials env vars and the same Postgres database with the
(currently disabled) AEVORYN-Daily-Live-Session task; it has its own code path, its own process, and its own
audit log (StockCollectionEvent).

STAGES, matching Phase P's approved design exactly:
  1. Pre-market: is today a trading day? (equity_trading_calendar.is_trading_day)
  2. Catch-up: per-symbol watermark -> missing trading dates -> backfill via the existing chunk_planner/executor
     -> normalize -> upsert (mirrors scripts/run_phase_f_full_persistence.py's per-symbol resumable pattern).
  3. Live: repeated universe sweeps via backend.app.data.live.live_poller, persisting completed-minute bars only.
  4. Post-market: one finalization pass (backend.app.data.live.finalize) once the session window has closed.

Every stage logs to StockCollectionEvent. No direction, prediction, or trade logic exists anywhere in this
script or the package it orchestrates.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

from backend.app.config import load_dotenv  # noqa: E402
from backend.app.data.backfill.chunk_planner import plan_chunks
from backend.app.data.backfill.executor import ChunkOutcome, ExecutorConfig, execute_plan
from backend.app.data.calendar.equity_trading_calendar import (
    HolidayCalendarUnavailable,
    IST,
    LAST_BAR_IST,
    is_trading_day,
    load_equity_holiday_dates,
)
from backend.app.data.canonical.intraday_loader import upsert_intraday_bars_batched
from backend.app.data.live import events
from backend.app.data.live.finalize import finalize_trading_day
from backend.app.data.live.live_poller import poll_universe_once
from backend.app.data.live.watermark import missing_trading_dates, per_symbol_watermark
from backend.app.data.normalizers import dhan_intraday_equity as norm
from backend.app.data.raw.store import RawStore
from backend.app.data.sources.credential_preflight import dhan_credential_preflight
from backend.app.data.sources.provider_credentials import DhanCredentials
from backend.app.data.sources.dhan_equity_historical import DhanEquityHistoricalAdapter
from backend.app.db.session import get_session

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_FILE = ROOT / "data" / "derived" / "phase_d_research_dataset" / "universe.json"
HOLIDAY_MASTER_FILE = ROOT / "data" / "raw" / "nse_calendar" / "holiday_master.json"


def load_universe() -> list[tuple[int, str]]:
    universe = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))["universe"]
    return [(entry["security_id"], entry["symbol"]) for entry in universe]


def now_ist() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).astimezone(IST)


def run_catchup(session, store, adapter, securities: list[tuple[int, str]], *, today: dt.date, holiday_dates) -> dict:
    security_ids = [sid for sid, _ in securities]
    watermark = per_symbol_watermark(session, security_ids=security_ids)

    groups: dict[tuple[dt.date, dt.date], list[tuple[int, str]]] = {}
    bootstrap_stragglers: list[tuple[int, str]] = []
    for sid, sym in securities:
        wm = watermark.get(sid)
        try:
            missing = missing_trading_dates(wm, today=today, holiday_dates=holiday_dates)
        except ValueError:
            bootstrap_stragglers.append((sid, sym))
            continue
        if not missing:
            continue
        key = (missing[0], missing[-1])
        groups.setdefault(key, []).append((sid, sym))

    if bootstrap_stragglers:
        events.log_event(
            session, trading_date=today, severity="WARNING", category=events.BACKFILL_FAILED,
            detail=f"{len(bootstrap_stragglers)} symbol(s) have no complete session on record at all and were "
                   f"skipped (no watermark to bootstrap from - requires an explicit decision, not made here): "
                   f"{[s for _, s in bootstrap_stragglers][:10]}",
            symbols_failed=len(bootstrap_stragglers),
        )

    if not groups:
        return {"groups": 0, "symbols_backfilled": 0, "bars_persisted": 0, "failed_symbols": [], "bootstrap_skipped": len(bootstrap_stragglers)}

    events.log_event(
        session, trading_date=today, severity="INFO", category=events.BACKFILL_STARTED,
        detail=f"{sum(len(g) for g in groups.values())} symbols across {len(groups)} missing-date group(s) need backfill",
        symbols_expected=sum(len(g) for g in groups.values()),
    )
    session.commit()

    total_bars = 0
    total_symbols_ok = 0
    failed_symbols: list[str] = []

    for (from_date, to_date), group_securities in groups.items():
        chunks = plan_chunks(group_securities, overall_from=from_date, overall_to=to_date)
        plan_result = execute_plan(store, adapter, chunks, config=ExecutorConfig())

        if plan_result.stopped_early and plan_result.stop_reason == "HTTP_401_AUTHENTICATION":
            events.log_event(
                session, trading_date=today, severity="ERROR", category=events.TOKEN_EXPIRED,
                detail="backfill stopped: Dhan returned HTTP 401 - token expired or invalid, no retry attempted",
            )
            session.commit()
            break

        by_symbol: dict[str, list] = {}
        by_symbol_sid: dict[str, int] = {}
        for r in plan_result.results:
            sym = r.chunk.trading_symbol
            by_symbol.setdefault(sym, []).append(r)
            by_symbol_sid[sym] = r.chunk.security_id

        for sym, chunk_results in by_symbol.items():
            ok = all(r.outcome in (ChunkOutcome.SUCCESS, ChunkOutcome.SKIPPED_EXISTING) for r in chunk_results)
            if not ok:
                bad = [r for r in chunk_results if r.outcome not in (ChunkOutcome.SUCCESS, ChunkOutcome.SKIPPED_EXISTING)]
                failed_symbols.append(sym)
                events.log_event(
                    session, trading_date=today, severity="WARNING", category=events.SYMBOL_FAILED,
                    detail=f"{sym}: backfill chunk failure(s): {[r.outcome.value for r in bad]}",
                    symbols_failed=1,
                )
                continue
            try:
                all_bars = []
                all_rejected = 0
                for r in chunk_results:
                    bars, rejected = norm.normalize_dhan_intraday_artifact(store, source_key=r.source_key, trading_symbol=sym)
                    all_bars.extend(bars)
                    all_rejected += len(rejected)
                n = upsert_intraday_bars_batched(session, all_bars) if all_bars else 0
                session.commit()
                total_bars += n
                total_symbols_ok += 1
            except Exception as exc:  # noqa: BLE001 - one symbol's failure must never abort the run
                session.rollback()
                failed_symbols.append(sym)
                events.log_event(
                    session, trading_date=today, severity="ERROR", category=events.SYMBOL_FAILED,
                    detail=f"{sym}: normalize/persist failed: {exc}", symbols_failed=1,
                )
                session.commit()

    events.log_event(
        session, trading_date=today, severity="INFO" if not failed_symbols else "WARNING",
        category=events.BACKFILL_COMPLETE,
        detail=f"backfill pass complete: {total_symbols_ok} symbols persisted, {len(failed_symbols)} failed, {total_bars} bars persisted",
        symbols_succeeded=total_symbols_ok, symbols_failed=len(failed_symbols), bars_persisted=total_bars,
    )
    session.commit()

    return {
        "groups": len(groups), "symbols_backfilled": total_symbols_ok, "bars_persisted": total_bars,
        "failed_symbols": failed_symbols, "bootstrap_skipped": len(bootstrap_stragglers),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["catchup-only", "live-sweep", "finalize-only", "full"], default="catchup-only")
    ap.add_argument("--max-sweeps", type=int, default=1, help="live-sweep/full mode: number of universe sweeps to run, then stop (bounded run, not continuous)")
    ap.add_argument("--run-until-session-close", action="store_true", help="live-sweep/full mode: ignore --max-sweeps and keep sweeping until wall-clock IST passes LAST_BAR_IST (15:14) - what a scheduled unattended daily run should pass")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "derived" / "phase_p_stock_collector" / "run_report.json")
    args = ap.parse_args(argv)

    load_dotenv()
    pre = dhan_credential_preflight()
    print(f"credential preflight: {pre.status} | {pre.detail}")
    if pre.status.upper() != "OK":
        print("ABORT: credential preflight failed")
        return 2

    today = now_ist().date()
    trading = is_trading_day(today, _load_holidays())
    print(f"today (IST): {today}, is_trading_day: {trading}")

    session = get_session()
    store = RawStore(ROOT / "data" / "raw" / "providers")
    creds = DhanCredentials.from_env()
    adapter = DhanEquityHistoricalAdapter(creds)
    securities = load_universe()
    print(f"universe: {len(securities)} symbols")

    report: dict = {"today": today.isoformat(), "is_trading_day": trading, "mode": args.mode}

    if not trading:
        events.log_event(session, trading_date=today, severity="INFO", category=events.NON_TRADING_DAY, detail="not a trading day - collector exiting without action")
        session.commit()
        report["result"] = "NON_TRADING_DAY"
        _write_report(args.out, report)
        print("not a trading day - nothing to do")
        return 0

    holiday_dates = _load_holidays()
    catchup = run_catchup(session, store, adapter, securities, today=today, holiday_dates=holiday_dates)
    report["catchup"] = catchup
    print(f"catchup: {catchup}")

    if args.mode == "catchup-only":
        _write_report(args.out, report)
        session.close()
        return 0

    if args.mode in ("live-sweep", "full"):
        sweeps = []
        i = 0
        while True:
            if args.run_until_session_close:
                if now_ist().time() > LAST_BAR_IST:
                    print(f"session window closed (past {LAST_BAR_IST} IST) - stopping live sweeps")
                    break
            elif i >= args.max_sweeps:
                break
            i += 1
            t0 = time.monotonic()
            results = poll_universe_once(session, store, adapter, securities=securities, today=today, now_utc_fn=lambda: dt.datetime.now(dt.timezone.utc))
            session.commit()
            n_ok = sum(1 for r in results if r.outcome.value in ("SUCCESS", "SKIPPED_EXISTING"))
            n_failed = len(results) - n_ok
            n_bars = sum(r.bars_persisted for r in results)
            latest = max((r.latest_completed_bar_utc for r in results if r.latest_completed_bar_utc), default=None)
            failed_results = [r for r in results if r.outcome.value not in ("SUCCESS", "SKIPPED_EXISTING")]
            for r in failed_results:
                events.log_event(
                    session, trading_date=today, severity="WARNING", category=events.SYMBOL_FAILED,
                    detail=f"{r.trading_symbol} (security_id={r.security_id}): {r.outcome.value} - {r.detail}",
                    symbols_failed=1,
                )
            sweep_label = f"sweep {i}" if args.run_until_session_close else f"sweep {i}/{args.max_sweeps}"
            events.log_event(
                session, trading_date=today, severity="INFO" if n_failed == 0 else "WARNING", category=events.CYCLE_COMPLETE,
                detail=f"{sweep_label}: {n_ok} ok, {n_failed} failed, {n_bars} bars persisted"
                       + (f"; failed: {[r.trading_symbol for r in failed_results]}" if failed_results else ""),
                symbols_expected=len(securities), symbols_succeeded=n_ok, symbols_failed=n_failed,
                latest_completed_minute=latest, bars_persisted=n_bars,
            )
            session.commit()
            elapsed = time.monotonic() - t0
            sweeps.append({"sweep": i, "ok": n_ok, "failed": n_failed, "bars_persisted": n_bars, "elapsed_seconds": round(elapsed, 1), "latest_bar_utc": latest.isoformat() if latest else None})
            print(f"{sweep_label}: {n_ok} ok, {n_failed} failed, {n_bars} bars persisted, {elapsed:.1f}s, latest={latest}")
            if n_failed > 0 and any(r.outcome is ChunkOutcome.HTTP_401_AUTHENTICATION for r in results):
                print("ABORT: token expired mid-sweep")
                break
        report["sweeps"] = sweeps

    if args.mode == "full":
        fin = finalize_trading_day(session, store, adapter, securities=securities, trading_date=today, now_utc_fn=lambda: dt.datetime.now(dt.timezone.utc))
        report["finalization"] = {
            "symbols_complete": fin.symbols_complete, "symbols_partial": fin.symbols_partial,
            "symbols_failed": fin.symbols_failed, "bars_persisted": fin.total_bars_persisted,
            "is_session_complete": fin.is_session_complete,
        }
        session.commit()
        print(f"finalization: {report['finalization']}")

    if args.mode == "finalize-only":
        fin = finalize_trading_day(session, store, adapter, securities=securities, trading_date=today, now_utc_fn=lambda: dt.datetime.now(dt.timezone.utc))
        report["finalization"] = {
            "symbols_complete": fin.symbols_complete, "symbols_partial": fin.symbols_partial,
            "symbols_failed": fin.symbols_failed, "bars_persisted": fin.total_bars_persisted,
            "is_session_complete": fin.is_session_complete,
        }
        session.commit()
        print(f"finalization: {report['finalization']}")

    _write_report(args.out, report)
    session.close()
    return 0


def _load_holidays():
    try:
        return load_equity_holiday_dates(HOLIDAY_MASTER_FILE)
    except HolidayCalendarUnavailable as exc:
        print(f"FATAL: holiday calendar unavailable: {exc}")
        raise


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
