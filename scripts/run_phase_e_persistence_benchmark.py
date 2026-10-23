from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

from sqlalchemy import text

from backend.app.data.canonical.intraday_equity_bar import CanonicalIntradayEquityBar, IntradayProvider
from backend.app.data.canonical.intraday_loader import upsert_intraday_bars, upsert_intraday_bars_batched
from backend.app.data.normalizers import dhan_intraday_equity as norm
from backend.app.data.raw.store import RawStore
from backend.app.db.session import get_session

ROOT = Path(__file__).resolve().parents[1]
STORE = RawStore(ROOT / "data" / "raw" / "providers")

BENCHMARK_SECURITY_ID = 900001
UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

REAL_SYMBOLS = {
    "RELIANCE": (2885, ["NSE_EQ-2885-1m-20250925-20251223", "NSE_EQ-2885-1m-20251224-20260323", "NSE_EQ-2885-1m-20260324-20260621",
                         "NSE_EQ-2885-1m-20260622-20260919", "NSE_EQ-2885-1m-20260920-20260925"]),
    "ABBOTINDIA": (17903, ["NSE_EQ-17903-1m-20250925-20251223", "NSE_EQ-17903-1m-20251224-20260323", "NSE_EQ-17903-1m-20260324-20260621",
                           "NSE_EQ-17903-1m-20260622-20260919", "NSE_EQ-17903-1m-20260920-20260925"]),
}


def _synthetic_bars(n: int) -> list[CanonicalIntradayEquityBar]:
    base = dt.datetime(2020, 1, 1, 3, 45, tzinfo=UTC)
    out = []
    for i in range(n):
        bs_utc = base + dt.timedelta(minutes=i)
        bs_ist = bs_utc.astimezone(IST)
        out.append(CanonicalIntradayEquityBar(
            provider=IntradayProvider.DHAN, exchange_segment="NSE_EQ", security_id=BENCHMARK_SECURITY_ID, trading_symbol="BENCHMARKONLY",
            raw_timestamp_epoch=int(bs_utc.timestamp()), bar_start_utc=bs_utc, bar_start_ist=bs_ist, trading_date=bs_ist.date(),
            interval_minutes=1, open=100.0, high=101.0, low=99.0, close=100.5, volume=10.0,
            source_provider="dhan", source_dataset="nse_eq_intraday_1m", source_key="phase-e-benchmark-synthetic",
            source_sha256="a" * 64, normalizer_version="dhan_intraday_equity/1",
        ))
    return out


def _load_real_bars(symbol: str, keys: list[str]) -> list[CanonicalIntradayEquityBar]:
    bars = []
    for key in keys:
        b, rejected = norm.normalize_dhan_intraday_artifact(STORE, source_key=key, trading_symbol=symbol)
        bars.extend(b)
    return bars


def _cleanup_benchmark_rows(session) -> None:
    session.execute(text("DELETE FROM intraday_equity_bars WHERE security_id = :sid"), {"sid": BENCHMARK_SECURITY_ID})
    session.commit()


def main() -> int:
    session = get_session()
    results: dict = {}

    pre_existing = session.execute(text("SELECT COUNT(*) FROM intraday_equity_bars WHERE security_id = :sid"), {"sid": BENCHMARK_SECURITY_ID}).scalar_one()
    if pre_existing:
        print(f"REFUSING: security_id {BENCHMARK_SECURITY_ID} already has {pre_existing} real rows - not a safe benchmark ID")
        return 2

    synth = _synthetic_bars(20000)

    t0 = time.monotonic()
    upsert_intraday_bars(session, synth)
    session.commit()
    elapsed = time.monotonic() - t0
    results["per_row"] = {"bars": len(synth), "elapsed_s": elapsed, "bars_per_sec": len(synth) / elapsed}
    print(f"per-row loader: {len(synth)} bars in {elapsed:.2f}s -> {len(synth)/elapsed:.1f} bars/sec")
    _cleanup_benchmark_rows(session)

    for batch_size in (500, 2000, 3000):
        t0 = time.monotonic()
        n = upsert_intraday_bars_batched(session, synth, batch_size=batch_size)
        session.commit()
        elapsed = time.monotonic() - t0
        results[f"batched_{batch_size}"] = {"bars": n, "elapsed_s": elapsed, "bars_per_sec": n / elapsed}
        print(f"batched (size={batch_size}): {n} bars in {elapsed:.2f}s -> {n/elapsed:.1f} bars/sec")
        _cleanup_benchmark_rows(session)

    best_batch = 3000
    t0 = time.monotonic()
    n1 = upsert_intraday_bars_batched(session, synth, batch_size=best_batch)
    session.commit()
    e1 = time.monotonic() - t0
    count_after_1 = session.execute(text("SELECT COUNT(*) FROM intraday_equity_bars WHERE security_id = :sid"), {"sid": BENCHMARK_SECURITY_ID}).scalar_one()

    t0 = time.monotonic()
    n2 = upsert_intraday_bars_batched(session, synth, batch_size=best_batch)
    session.commit()
    e2 = time.monotonic() - t0
    count_after_2 = session.execute(text("SELECT COUNT(*) FROM intraday_equity_bars WHERE security_id = :sid"), {"sid": BENCHMARK_SECURITY_ID}).scalar_one()

    dup_groups = session.execute(text(
        "SELECT provider, exchange_segment, security_id, timeframe, bar_start_utc, COUNT(*) c FROM intraday_equity_bars "
        "WHERE security_id = :sid GROUP BY 1,2,3,4,5 HAVING COUNT(*) > 1"
    ), {"sid": BENCHMARK_SECURITY_ID}).fetchall()

    _cleanup_benchmark_rows(session)
    half = synth[: len(synth) // 2]
    upsert_intraday_bars_batched(session, half, batch_size=best_batch)
    session.commit()
    count_after_half = session.execute(text("SELECT COUNT(*) FROM intraday_equity_bars WHERE security_id = :sid"), {"sid": BENCHMARK_SECURITY_ID}).scalar_one()
    upsert_intraday_bars_batched(session, synth, batch_size=best_batch)
    session.commit()
    count_after_resume = session.execute(text("SELECT COUNT(*) FROM intraday_equity_bars WHERE security_id = :sid"), {"sid": BENCHMARK_SECURITY_ID}).scalar_one()

    results["idempotency"] = {
        "run1_bars": n1, "run1_elapsed_s": e1, "run1_row_count": count_after_1,
        "run2_bars": n2, "run2_elapsed_s": e2, "run2_row_count": count_after_2,
        "duplicate_groups": len(dup_groups),
    }
    results["restart_resume_simulation"] = {
        "after_partial_half": count_after_half, "expected_half": len(half),
        "after_full_resume": count_after_resume, "expected_full": len(synth),
    }
    print(f"idempotency: run1={count_after_1} rows in {e1:.2f}s, run2={count_after_2} rows in {e2:.2f}s, duplicate_groups={len(dup_groups)}")
    print(f"restart/resume: after partial={count_after_half} (expected {len(half)}), after resume={count_after_resume} (expected {len(synth)})")

    _cleanup_benchmark_rows(session)

    real_persist_summary = {}
    for symbol, (sec_id, keys) in REAL_SYMBOLS.items():
        pre = session.execute(text("SELECT COUNT(*) FROM intraday_equity_bars WHERE security_id = :sid"), {"sid": sec_id}).scalar_one()
        bars = _load_real_bars(symbol, keys)
        t0 = time.monotonic()
        n = upsert_intraday_bars_batched(session, bars, batch_size=best_batch)
        session.commit()
        elapsed = time.monotonic() - t0
        post = session.execute(text("SELECT COUNT(*) FROM intraday_equity_bars WHERE security_id = :sid"), {"sid": sec_id}).scalar_one()
        real_persist_summary[symbol] = {"security_id": sec_id, "canonical_bars": len(bars), "upserted": n, "elapsed_s": elapsed,
                                        "rows_before": pre, "rows_after": post}
        print(f"REAL persist {symbol}: {len(bars)} canonical bars, {n} upserted in {elapsed:.2f}s, rows {pre} -> {post}")

    results["real_data_persisted"] = real_persist_summary
    session.close()

    out = ROOT / "data" / "derived" / "phase_e_persistence_benchmark" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print("results:", out)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
