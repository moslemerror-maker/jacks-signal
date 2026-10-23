from __future__ import annotations

import dataclasses
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from backend.app.data.sources.bse_corporate_actions import (
    _new_session as new_bse_session,
    fetch_bse_corporate_actions_snapshot,
)
from backend.app.data.sources.nse_corporate_actions import (
    fetch_corporate_actions_snapshot,
    open_session as open_nse_session,
)

RAW_NSE = Path("data/raw/corporate_actions")
RAW_BSE = Path("data/raw/corporate_actions_bse")
LEDGER = Path("data/raw/corporate_actions_c1_fetch_ledger.json")

BSE_CODES = {
    "HDFCBANK": "500180", "KOTAKBANK": "500247", "BAJFINANCE": "500034",
    "JSWSTEEL": "500228", "RELIANCE": "500325",
}
WINDOW_START = date(2015, 1, 1)
BOUNDARY_DAY = date(2017, 9, 7)
SPACING_SECONDS = 1.5


@dataclasses.dataclass(frozen=True)
class Request:
    provider: str
    symbol: str
    from_date: date
    to_date: date
    purpose: str


def parts_for(to_date: date) -> list[tuple[date, date]]:
    return [
        (WINDOW_START, date(2018, 12, 31)),
        (date(2019, 1, 1), date(2022, 12, 31)),
        (date(2023, 1, 1), to_date),
    ]


def build_plan(to_date: date) -> list[Request]:
    plan: list[Request] = []
    d = BOUNDARY_DAY
    for provider in ("NSE", "BSE"):
        plan += [
            Request(provider, "RELIANCE", d, d, "c1-boundary-single-day"),
            Request(provider, "RELIANCE", d, d + timedelta(days=30), "c1-boundary-from"),
            Request(provider, "RELIANCE", d - timedelta(days=30), d, "c1-boundary-to"),
        ]
    for symbol in BSE_CODES:
        for provider in ("NSE", "BSE"):
            plan.append(Request(provider, symbol, WINDOW_START, to_date, "c1-whole"))
            for i, (lo, hi) in enumerate(parts_for(to_date), start=1):
                plan.append(Request(provider, symbol, lo, hi, f"c1-part-{i}of3"))
    plan += [
        Request("NSE", "RELIANCE", date(1990, 1, 1), to_date, "c1-cap-probe-wide"),
        Request("BSE", "RELIANCE", date(1990, 1, 1), to_date, "c1-cap-probe-wide"),
    ]
    return plan


def _existing(req: Request) -> dict | None:
    root = RAW_NSE if req.provider == "NSE" else RAW_BSE
    for meta_path in sorted((root / req.symbol).glob("*.meta.json")) if (root / req.symbol).is_dir() else []:
        meta = json.loads(meta_path.read_text())
        w = meta.get("query_window") or {}
        if (
            w.get("from_date") == req.from_date.isoformat()
            and w.get("to_date") == req.to_date.isoformat()
            and meta.get("purpose") == req.purpose
        ):
            return {**meta, "path": meta_path.as_posix()[: -len(".meta.json")]}
    return None


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    to_date = date.today()
    if "--to" in sys.argv:
        to_date = date.fromisoformat(sys.argv[sys.argv.index("--to") + 1])
    plan = build_plan(to_date)
    todo = [r for r in plan if _existing(r) is None]
    print(f"plan: {len(plan)} requests ({len(plan) - len(todo)} already on disk, {len(todo)} to send); to_date={to_date}")
    if dry_run:
        for r in plan:
            print(f"  {'SKIP' if _existing(r) else 'SEND'} {r.provider} {r.symbol:10} {r.from_date}..{r.to_date}  {r.purpose}")
        return

    nse = open_nse_session()
    bse = new_bse_session()
    ledger = []
    for i, r in enumerate(plan, start=1):
        have = _existing(r)
        if have is not None:
            ledger.append({"provider": r.provider, "symbol": r.symbol, "purpose": r.purpose,
                           "from": r.from_date.isoformat(), "to": r.to_date.isoformat(),
                           "count": have["record_count"], "sha256": have["sha256"], "fetched_at": have["fetched_at"],
                           "path": have["path"], "created": False})
            continue
        time.sleep(SPACING_SECONDS)
        if r.provider == "NSE":
            res = fetch_corporate_actions_snapshot(
                r.symbol, RAW_NSE, from_date=r.from_date, to_date=r.to_date, purpose=r.purpose, session=nse
            )
        else:
            res = fetch_bse_corporate_actions_snapshot(
                BSE_CODES[r.symbol], r.symbol, RAW_BSE, r.from_date, r.to_date, purpose=r.purpose, session=bse
            )
        print(f"[{i:2}/{len(plan)}] {r.provider} {r.symbol:10} {r.from_date}..{r.to_date} {r.purpose:24} "
              f"n={res.record_count:3} sha={res.sha256[:16]} {'NEW' if res.created else 'same-bytes'}")
        ledger.append({"provider": r.provider, "symbol": r.symbol, "purpose": r.purpose,
                       "from": r.from_date.isoformat(), "to": r.to_date.isoformat(), "count": res.record_count,
                       "sha256": res.sha256, "fetched_at": res.fetched_at, "path": res.path.as_posix(),
                       "created": res.created})
    LEDGER.write_text(json.dumps(ledger, indent=1))
    print(f"ledger written: {LEDGER} ({len(ledger)} entries)")


if __name__ == "__main__":
    main()
