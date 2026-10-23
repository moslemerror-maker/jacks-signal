from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

from sqlalchemy import select

from backend.app.data.instruments.corporate_actions import (
    nse_event_verification_lookup,
    normalize_snapshot,
    record_derived_coverage_assessment,
)
from backend.app.db.models import (
    AssetType,
    CorporateActionEvent,
    Instrument,
    InstrumentIsin,
    InstrumentSymbol,
    MarketBar,
)
from backend.app.db.session import get_session
from backend.app.research.corporate_action_diagnostics import (
    isin_transition_corroboration,
    scan_close_discontinuities,
)
from backend.app.research.corporate_action_evidence import (
    SnapshotRecord,
    cap_observations,
    cap_signal,
    comparable_window,
    discover_snapshots,
    infer_boundary_semantics,
    partition_consistency,
)
from backend.app.research.corporate_action_reconciliation import (
    PRICE_AFFECTING,
    bse_components,
    classify_revisions,
    diff_snapshots,
    nse_components,
    reconcile_economic_events,
)
from backend.app.research.corporate_action_register import build_unresolved_register
from backend.app.research.corporate_actions_coverage import ProviderEvidence, derive_coverage_evidence

RAW = {"NSE": Path("data/raw/corporate_actions"), "BSE": Path("data/raw/corporate_actions_bse")}
SYMBOLS = ["HDFCBANK", "KOTAKBANK", "BAJFINANCE", "JSWSTEEL", "RELIANCE"]
BOUNDARY_DAY = datetime.date(2017, 9, 7)


def _latest(snapshots, purpose):
    matches = sorted((s for s in snapshots if s.purpose == purpose), key=lambda s: s.fetched_at)
    return matches[-1] if matches else None


def _dates_of(snapshot: SnapshotRecord | None):
    return [d for d in snapshot.row_dates() if d] if snapshot else []


def boundary_semantics(provider: str, reliance: list[SnapshotRecord]):
    single = _latest(reliance, "c1-boundary-single-day")
    frm = _latest(reliance, "c1-boundary-from")
    to = _latest(reliance, "c1-boundary-to")
    control = _latest(reliance, "c1-whole")
    if not all([single, frm, to, control]):
        from backend.app.research.corporate_action_evidence import BoundarySemantics
        return BoundarySemantics(None, None, "boundary probes not all present")
    return infer_boundary_semantics(
        BOUNDARY_DAY, _dates_of(single), _dates_of(frm), _dates_of(to), set(_dates_of(control))
    )


def instrument_id(session, symbol: str):
    return session.scalar(
        select(InstrumentSymbol.instrument_id)
        .join(Instrument, Instrument.id == InstrumentSymbol.instrument_id)
        .where(InstrumentSymbol.symbol == symbol, InstrumentSymbol.effective_to.is_(None),
               Instrument.asset_type == AssetType.EQUITY)
    )


def eq_bars(session, iid):
    rows = session.execute(
        select(MarketBar.trade_date, MarketBar.close)
        .where(MarketBar.instrument_id == iid, MarketBar.series == "EQ", MarketBar.timeframe == "1D")
        .order_by(MarketBar.trade_date)
    ).all()
    return [(d, c) for d, c in rows]


def isin_transitions(session, iid, window):
    rows = session.scalars(
        select(InstrumentIsin).where(InstrumentIsin.instrument_id == iid).order_by(InstrumentIsin.effective_from)
    ).all()
    return [r.effective_from for r in rows[1:] if window[0] <= r.effective_from <= window[1]]


def assess(session, symbol, all_snapshots, dry_run):
    iid = instrument_id(session, symbol)
    bars = eq_bars(session, iid)
    window = (bars[0][0], bars[-1][0])

    per_provider, providers, comps, unparse = {}, {}, {}, {}
    for provider in ("NSE", "BSE"):
        snaps = discover_snapshots(RAW[provider], symbol)
        per_provider[provider] = snaps
        whole = _latest(snaps, "c1-whole")
        parts = [s for p in ("c1-part-1of3", "c1-part-2of3", "c1-part-3of3") if (s := _latest(snaps, p))]
        sem = boundary_semantics(provider, discover_snapshots(RAW[provider], "RELIANCE"))
        obs = cap_observations(all_snapshots[provider])
        coverage = ([whole] if whole else []) + parts
        providers[provider] = ProviderEvidence(
            provider=provider, whole=whole, parts=parts, boundary=sem,
            partition=partition_consistency(whole, parts, sem) if whole and parts else None,
            cap_observations=obs, cap_signals=[cap_signal(s, obs) for s in coverage],
        )
        if whole is not None:
            build = nse_components if provider == "NSE" else bse_components
            comps[provider], unparse[provider] = build(whole.rows, whole.ref)
        else:
            comps[provider], unparse[provider] = [], []

    new_events = 0
    if providers["NSE"].whole is not None and not dry_run:
        new_events = normalize_snapshot(session, Path(providers["NSE"].whole.path), iid, created_by="stage3.9-c1")

    recon = reconcile_economic_events(comps["NSE"], comps["BSE"], window, unparse["NSE"] + unparse["BSE"])
    in_window_nse = [c for c in comps["NSE"] if window[0] <= c.ex_date <= window[1]]
    revisions = classify_revisions(in_window_nse, recon)

    diffs, comparisons = {}, []
    for provider in ("NSE", "BSE"):
        whole = providers[provider].whole
        diffs[provider] = []
        if whole is None:
            continue
        build = nse_components if provider == "NSE" else bse_components
        for older in per_provider[provider]:
            if older.purpose and older.purpose.startswith("c1-"):
                continue
            if older.fetched_at >= whole.fetched_at:
                continue
            cw = comparable_window(older, whole, window)
            if cw is None:
                continue
            o_comps, _ = build(older.rows, older.ref)
            n_comps, _ = build(whole.rows, whole.ref)
            changes = diff_snapshots(o_comps, n_comps, cw)
            diffs[provider] += changes
            comparisons.append({
                "provider": provider, "older_ref": older.ref, "older_fetched_at": older.fetched_at,
                "older_mode": older.fetch_mode, "newer_ref": whole.ref, "newer_fetched_at": whole.fetched_at,
                "window": [cw[0].isoformat(), cw[1].isoformat()], "change_count": len(changes),
            })

    price_dates = {m.ex_date for m in recon.matches if m.action_class in PRICE_AFFECTING}
    jump = scan_close_discontinuities(bars, price_dates)
    isin = isin_transition_corroboration(
        isin_transitions(session, iid, window), [d for d, _ in bars], price_dates
    )

    evidence = derive_coverage_evidence(
        symbol=symbol, instrument_id=str(iid), window=window, providers=providers, reconciliation=recon,
        revisions=revisions, snapshot_diffs=diffs, jump_scan=jump, isin_diagnostic=isin,
        diff_comparisons=comparisons,
    )
    register = build_unresolved_register(recon, revisions, diffs, jump, isin, nse_event_verification_lookup(session, iid))
    assessment_id = None
    if not dry_run:
        assessment_id = record_derived_coverage_assessment(session, iid, evidence, register)
        session.commit()
    return iid, evidence, register, assessment_id, new_events


def basket_scan(session):
    print("\n== basket-wide discontinuity scan (DIAGNOSTIC ONLY - blocker/annotation, never proof) ==")
    from backend.app.data.instruments.corporate_actions import current_components

    equity = session.execute(
        select(InstrumentSymbol.symbol, InstrumentSymbol.instrument_id)
        .join(Instrument, Instrument.id == InstrumentSymbol.instrument_id)
        .where(InstrumentSymbol.effective_to.is_(None), Instrument.asset_type == AssetType.EQUITY)
    ).all()
    for symbol, iid in sorted(equity):
        bars = eq_bars(session, iid)
        if len(bars) < 100:
            continue
        ex = set()
        n_events = 0
        for e in session.scalars(select(CorporateActionEvent).where(CorporateActionEvent.instrument_id == iid)):
            n_events += 1
            for c in current_components(session, e.id):
                if (c.component_type.value if hasattr(c.component_type, "value") else c.component_type) in PRICE_AFFECTING:
                    ex.add(e.ex_date)
        scan = scan_close_discontinuities(bars, ex)
        if scan["blocking_count"] or scan["explained_jumps"] or scan["watch_unexplained"]:
            print(f"  {symbol:11} corporate-action events in DB: {n_events:2} | explained jumps: {len(scan['explained_jumps'])} | "
                  f"UNEXPLAINED >=28%: {scan['blocking_count']} | unexplained watch 15-28%: {len(scan['watch_unexplained'])}")
            for j in scan["unexplained_blockers"]:
                print(f"      BLOCKER {j['bar_date']} {j['change']:+.1%}")


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    session = get_session()
    try:
        all_snapshots = {p: [s for sym in SYMBOLS for s in discover_snapshots(RAW[p], sym)] for p in RAW}
        results = []
        for symbol in SYMBOLS:
            iid, evidence, register, aid, new_events = assess(session, symbol, all_snapshots, dry_run)
            results.append((symbol, evidence, register, aid, new_events))
            checks = evidence["checks"]
            print(f"\n=== {symbol}: {evidence['verdict']}  window {evidence['window']['from']}..{evidence['window']['to']}"
                  f"  assessment={aid}")
            print("   reasoning:", evidence["reasoning"])
            print("   failed checks:", [k for k, v in checks.items() if not v] or "none")
            r = evidence["reconciliation"]
            print(f"   reconciliation: term_matches={r['term_matches']} class_only={r['class_only']} conflicts={r['conflict_count']} "
                  f"one_sided={r['one_sided_count']} unclassified={r['unclassified_count']} dup_price_affecting={r['duplicate_price_affecting_count']} "
                  f"ambiguous={r['ambiguous_same_date_count']}  | NSE events newly normalized: {new_events}")
            print(f"   revisions: {[(f['code'], f['ex_date']) for f in evidence['revisions']['findings']]}")
            print(f"   snapshot diffs: { {k: len(v) for k, v in evidence['snapshot_diffs'].items()} }  "
                  f"comparisons={len(evidence['snapshot_diff_comparisons'])}")
            j = evidence["diagnostics"]["jump_scan"]
            print(f"   jump scan: blockers={j['blocking_count']} explained={len(j['explained_jumps'])} watch_unexplained={len(j['watch_unexplained'])}")
            i = evidence["diagnostics"]["isin_transitions"]
            print(f"   ISIN transitions: {[(t['transition_date'], t['corroborated'], t['lag_sessions']) for t in i['transitions']]}")
            by = {}
            for it in register:
                by[(it["severity"], it["reason_code"])] = by.get((it["severity"], it["reason_code"]), 0) + 1
            print("   register:", dict(sorted(by.items())))
        json.dump(
            [{"symbol": s, "verdict": e["verdict"], "assessment_id": str(a) if a else None,
              "checks": e["checks"], "register": [{**{k: v for k, v in it.items() if k not in ("detail", "event_id")},
                                                    "ex_date": str(it["ex_date"]) if it["ex_date"] else None}
                                                   for it in reg]}
             for s, e, reg, a, _ in results],
            open("data/raw/corporate_actions_c1_assessment_summary.json", "w"), indent=1, default=str,
        )
        if "--basket-scan" in sys.argv:
            basket_scan(session)
    finally:
        session.close()


if __name__ == "__main__":
    main()
