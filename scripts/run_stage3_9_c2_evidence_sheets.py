from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from sqlalchemy import func, select, text

from backend.app.data.instruments.corporate_actions import _enum_value
from backend.app.data.instruments.term_verification import _current_rows, _ctype, build_component_sheet, legacy_verification_inventory
from backend.app.db.models import CorporateActionEvent as E, InstrumentSymbol, MarketBar
from backend.app.db.session import get_session
from backend.app.research.corporate_action_reconciliation import PRICE_AFFECTING, is_benign_meeting_fragment

OUT_DIR = Path("data/raw/corporate_actions_c2_evidence_sheets")
SUPPORTED = ("SPLIT", "BONUS")


def _is_candidate_row(row) -> bool:
    t = _ctype(row)
    return t in PRICE_AFFECTING or (t == "OTHER" and not is_benign_meeting_fragment(row.raw_text_fragment))


def main() -> None:
    session = get_session()
    session.execute(text("SET TRANSACTION READ ONLY"))
    symbols = dict(session.execute(
        select(InstrumentSymbol.instrument_id, InstrumentSymbol.symbol).where(InstrumentSymbol.effective_to.is_(None))).all())
    window_start = session.scalar(select(func.min(MarketBar.trade_date)))
    print(f"stored-bar window starts {window_start}; candidate set = every price-affecting or non-benign-OTHER component "
          f"in the CURRENT reading of every event with ex_date >= {window_start}\n")

    events = session.execute(
        select(E.id, E.instrument_id, E.ex_date, E.raw_subject_text).order_by(E.instrument_id, E.ex_date, E.raw_subject_text)).all()
    current = _current_rows(session, [e.id for e in events])

    in_window, pre_window = [], []
    for e in events:
        for row in current.get(e.id, []):
            if _is_candidate_row(row):
                (in_window if e.ex_date >= window_start else pre_window).append((e, row))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for e, row in in_window:
        sheet = build_component_sheet(session, row.id)
        c = sheet.content
        t = _ctype(row)
        group = ("NEGATIVE_CONTROL_UNSUPPORTED" if t not in SUPPORTED
                 else "LEGACY_PHASE_B_VERIFIED" if c["verification_state"].startswith("LEGACY")
                 else "C2_VERIFIED" if c["verification_state"] == "C2_VERIFIED"
                 else "UNRESOLVED_CANDIDATE")
        nse = c["evidence"].get("NSE", {}).get("primary_snapshot", {})
        bse = c["evidence"].get("BSE", {}).get("primary_snapshot", {})
        rec = {
            "group": group, "instrument": symbols.get(e.instrument_id), "ex_date": e.ex_date.isoformat(), "class": t,
            "component_id": str(row.id), "event_id": str(e.id), "component_status": _enum_value(row.verification_status),
            "verification_state": c["verification_state"], "terms": c["terms_statement"],
            "readiness": sheet.readiness, "tier_required": sheet.tier_required,
            "tier_reasons": [r["code"] for r in c["readiness"]["tier_reasons"]],
            "blockers": [b["code"] for b in sheet.blockers],
            "nse_snapshot": nse.get("sha256"), "bse_snapshot": bse.get("sha256"),
            "nse_fetched_at": nse.get("fetched_at"), "bse_fetched_at": bse.get("fetched_at"),
            "raw_nse": [r.get("subject") for r in c["evidence"].get("NSE", {}).get("rows_at_ex_date", [])],
            "raw_bse": [r.get("Purpose") for r in c["evidence"].get("BSE", {}).get("rows_at_ex_date", [])],
            "sheet_sha256": sheet.sha256,
        }
        rows_out.append(rec)
        fname = f"{rec['instrument']}_{rec['ex_date']}_{t}_{str(row.id)[:8]}.json"
        (OUT_DIR / fname).write_text(json.dumps(c, indent=1, sort_keys=True, default=str), encoding="utf-8")
        rec["file"] = str(OUT_DIR / fname)

    (OUT_DIR / "index.json").write_text(json.dumps(rows_out, indent=1, sort_keys=True), encoding="utf-8")

    order = ["UNRESOLVED_CANDIDATE", "LEGACY_PHASE_B_VERIFIED", "C2_VERIFIED", "NEGATIVE_CONTROL_UNSUPPORTED"]
    for g in order:
        subset = [r for r in rows_out if r["group"] == g]
        print(f"=== {g} ({len(subset)})")
        for r in subset:
            print(f"  {r['instrument']:10} {r['ex_date']} {r['class']:8} {r['terms'] or '(no terms)':52} state={r['verification_state']}")
            print(f"      readiness={r['readiness']} tier={r['tier_required']} tier_reasons={r['tier_reasons']} blockers={r['blockers']}")
            print(f"      NSE {r['raw_nse']} sha={str(r['nse_snapshot'])[:16]} fetched={str(r['nse_fetched_at'])[:19]}")
            print(f"      BSE {r['raw_bse']} sha={str(r['bse_snapshot'])[:16]} fetched={str(r['bse_fetched_at'])[:19]}")
            print(f"      sheet_sha256={r['sheet_sha256']}  component_id={r['component_id']}")
        print()

    ready = collections.Counter(r["readiness"] for r in rows_out)
    print("readiness summary:", dict(ready))
    inv = legacy_verification_inventory(session)
    legacy_in_window = {r["component_id"] for r in rows_out if r["group"] == "LEGACY_PHASE_B_VERIFIED"}
    inv_ids = {i["component_id"] for i in inv if i["ex_date"] >= window_start.isoformat()}
    print(f"legacy inventory (in window) == sheets marked legacy: {inv_ids == legacy_in_window} ({len(inv_ids)})")
    print(f"pre-window components (outside the stored bars; no sheet built): "
          f"{[(symbols.get(e.instrument_id), e.ex_date.isoformat(), _ctype(row), _enum_value(row.verification_status)) for e, row in pre_window]}")
    print(f"\nsheets written to {OUT_DIR}/ (index.json + one file per component). NOTHING was verified, promoted or written to the database.")
    session.rollback()
    session.close()


if __name__ == "__main__":
    main()
    sys.exit(0)
