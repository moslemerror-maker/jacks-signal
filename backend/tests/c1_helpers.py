from __future__ import annotations

import datetime
import hashlib
import json

from backend.app.research.corporate_action_diagnostics import (
    isin_transition_corroboration,
    scan_close_discontinuities,
)
from backend.app.research.corporate_action_evidence import (
    BoundarySemantics,
    SnapshotRecord,
    cap_observations,
    cap_signal,
    partition_consistency,
)
from backend.app.research.corporate_action_reconciliation import (
    EconomicComponent,
    bse_components,
    classify_revisions,
    nse_components,
    reconcile_economic_events,
)
from backend.app.research.corporate_actions_coverage import ProviderEvidence, derive_coverage_evidence

D = datetime.date
INCLUSIVE = BoundarySemantics(True, True, "test: both inclusive")
WINDOW = (D(2015, 1, 1), D(2026, 9, 11))
PARTS = [(D(2015, 1, 1), D(2018, 12, 31)), (D(2019, 1, 1), D(2022, 12, 31)), (D(2023, 1, 1), D(2026, 9, 20))]
FETCHED = "2026-09-20T10:00:00+00:00"


def comp(ex, cls, terms, source="NSE", fragment=None, ref="t#0", revision=None):
    return EconomicComponent(ex, cls, terms, source, fragment or f"{cls} {terms}", ref, revision)


def nse_row(ex: datetime.date, subject: str) -> dict:
    return {"symbol": "TESTCO", "exDate": ex.strftime("%d-%b-%Y"), "subject": subject, "isin": "X", "series": "EQ"}


def bse_row(ex: datetime.date, purpose: str) -> dict:
    return {"scrip_code": 1, "short_name": "TESTCO", "Ex_date": ex.strftime("%d %b %Y"), "Purpose": purpose}


def snap(provider, rows, q_from, q_to, purpose="c1-whole", *, symbol="TESTCO", mode="windowed",
         fetched=FETCHED, integrity=True, count=None) -> SnapshotRecord:
    body = json.dumps(rows, sort_keys=True)
    sha = hashlib.sha256(f"{provider}{symbol}{q_from}{q_to}{purpose}{mode}{body}".encode()).hexdigest()
    return SnapshotRecord(
        provider=provider, symbol=symbol, path=f"raw/{provider}/{symbol}/{sha[:16]}.json", sha256=sha,
        fetched_at=fetched, record_count=len(rows) if count is None else count, fetch_mode=mode,
        query_from=q_from, query_to=q_to, purpose=purpose, query_params={}, rows=tuple(rows),
        integrity_ok=integrity,
    )


def _slice(rows_with_dates, lo, hi):
    return [r for d, r in rows_with_dates if lo <= d <= hi]


def clean_provider(provider, dated_rows, *, semantics=INCLUSIVE, extra_snapshots=()):
    whole = snap(provider, [r for _, r in dated_rows], WINDOW[0], PARTS[-1][1], "c1-whole")
    parts = [snap(provider, _slice(dated_rows, lo, hi), lo, hi, f"c1-part-{i}of3") for i, (lo, hi) in enumerate(PARTS, 1)]
    wide = snap(provider, [r for _, r in dated_rows] * 3, D(1990, 1, 1), PARTS[-1][1], "c1-cap-probe-wide")
    obs = cap_observations([whole, *parts, wide, *extra_snapshots])
    return ProviderEvidence(
        provider=provider, whole=whole, parts=parts, boundary=semantics,
        partition=partition_consistency(whole, parts, semantics), cap_observations=obs,
        cap_signals=[cap_signal(s, obs) for s in [whole, *parts]],
    )


BASE_NSE = [
    (D(2016, 6, 10), nse_row(D(2016, 6, 10), "Dividend - Rs 6 Per Share")),
    (D(2017, 9, 7), nse_row(D(2017, 9, 7), "Bonus 1:1")),
    (D(2019, 9, 19), nse_row(D(2019, 9, 19), "Face Value Split (Sub-Division) - From Rs 2 Per Share To Rs 1 Per Share")),
    (D(2024, 8, 5), nse_row(D(2024, 8, 5), "Dividend - Rs 8 Per Share")),
]
BASE_BSE = [
    (D(2016, 6, 10), bse_row(D(2016, 6, 10), "Final Dividend - Rs. - 6.0000")),
    (D(2017, 9, 7), bse_row(D(2017, 9, 7), "Bonus issue 1:1")),
    (D(2019, 9, 19), bse_row(D(2019, 9, 19), "Stock  Split From Rs.2/- to Rs.1/-")),
    (D(2024, 8, 5), bse_row(D(2024, 8, 5), "Final Dividend - Rs. - 8.0000")),
]

CLEAN_JUMP = scan_close_discontinuities([(D(2015, 1, 1), 100.0), (D(2015, 1, 2), 101.0)], [])
CLEAN_ISIN = isin_transition_corroboration([], [D(2015, 1, 1)], [])


def derive(nse_dated=BASE_NSE, bse_dated=BASE_BSE, *, providers=None, recon_window=WINDOW, jump=CLEAN_JUMP,
           isin=CLEAN_ISIN, diffs=None, revisions=None, nse_extra=(), bse_extra=(), semantics=INCLUSIVE):
    providers = providers if providers is not None else {
        "NSE": clean_provider("NSE", nse_dated, semantics=semantics, extra_snapshots=nse_extra),
        "BSE": clean_provider("BSE", bse_dated, semantics=semantics, extra_snapshots=bse_extra),
    }
    nse_c, nse_u = nse_components(providers["NSE"].whole.rows, "n") if providers.get("NSE") and providers["NSE"].whole else ([], [])
    bse_c, bse_u = bse_components(providers["BSE"].whole.rows, "b") if providers.get("BSE") and providers["BSE"].whole else ([], [])
    recon = reconcile_economic_events(nse_c, bse_c, recon_window, nse_u + bse_u)
    revs = revisions if revisions is not None else classify_revisions(nse_c, recon)
    return derive_coverage_evidence(
        symbol="TESTCO", instrument_id="00000000-0000-0000-0000-000000000001", window=WINDOW,
        providers=providers, reconciliation=recon, revisions=revs,
        snapshot_diffs=diffs or {"NSE": [], "BSE": []}, jump_scan=jump, isin_diagnostic=isin,
    )
