from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence


def _row_date(provider: str, row: dict) -> datetime.date | None:
    try:
        if provider == "NSE":
            return datetime.datetime.strptime(row["exDate"], "%d-%b-%Y").date()
        return datetime.datetime.strptime(row["Ex_date"], "%d %b %Y").date()
    except (KeyError, ValueError, TypeError):
        return None


def _canonical(row: dict) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"))


@dataclasses.dataclass(frozen=True)
class SnapshotRecord:
    provider: str
    symbol: str
    path: str
    sha256: str
    fetched_at: str
    record_count: int
    fetch_mode: str
    query_from: datetime.date | None
    query_to: datetime.date | None
    purpose: str | None
    query_params: dict
    rows: tuple
    integrity_ok: bool

    @property
    def ref(self) -> str:
        return self.sha256[:16]

    def row_dates(self) -> list[datetime.date | None]:
        return [_row_date(self.provider, r) for r in self.rows]

    def as_ref_dict(self) -> dict:
        return {
            "provider": self.provider, "symbol": self.symbol, "path": self.path,
            "sha256": self.sha256, "fetched_at": self.fetched_at, "record_count": self.record_count,
            "fetch_mode": self.fetch_mode, "purpose": self.purpose,
            "query_from": self.query_from.isoformat() if self.query_from else None,
            "query_to": self.query_to.isoformat() if self.query_to else None,
            "query_params": self.query_params, "integrity_ok": self.integrity_ok,
        }


def load_snapshot(path: Path) -> SnapshotRecord:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta = json.loads(meta_path.read_text())
    content = path.read_bytes()
    provider = "NSE" if str(meta.get("provider", "")).startswith("NSE") else "BSE"
    params = meta.get("query_params", {})
    window = meta.get("query_window")
    q_from = q_to = None
    if window:
        q_from = datetime.date.fromisoformat(window["from_date"])
        q_to = datetime.date.fromisoformat(window["to_date"])
    elif provider == "BSE" and "Fdate" in params:
        q_from = datetime.datetime.strptime(params["Fdate"], "%Y%m%d").date()
        q_to = datetime.datetime.strptime(params["TDate"], "%Y%m%d").date()
    mode = meta.get("fetch_mode") or ("windowed" if q_from else "default")
    return SnapshotRecord(
        provider=provider, symbol=meta["symbol"], path=path.as_posix(), sha256=meta["sha256"],
        fetched_at=meta["fetched_at"], record_count=meta["record_count"], fetch_mode=mode,
        query_from=q_from, query_to=q_to, purpose=meta.get("purpose"), query_params=params,
        rows=tuple(json.loads(content)), integrity_ok=hashlib.sha256(content).hexdigest() == meta["sha256"],
    )


def discover_snapshots(root: Path, symbol: str) -> list[SnapshotRecord]:
    directory = root / symbol
    if not directory.is_dir():
        return []
    files = sorted(p for p in directory.glob("*.json") if not p.name.endswith(".meta.json"))
    return [load_snapshot(p) for p in files if p.with_suffix(p.suffix + ".meta.json").exists()]


def cap_observations(snapshots: Sequence[SnapshotRecord]) -> dict[str, dict]:
    groups: dict[str, list[SnapshotRecord]] = defaultdict(list)
    for s in snapshots:
        groups[f"{s.provider}:{s.fetch_mode}"].append(s)
    out: dict[str, dict] = {}
    for key, group in sorted(groups.items()):
        top = max(s.record_count for s in group)
        at_top = [s for s in group if s.record_count == top]
        distinct = {(s.symbol, s.query_from, s.query_to) for s in at_top}
        out[key] = {
            "responses_observed": len(group),
            "max_observed_count": top,
            "responses_at_max": len(at_top),
            "distinct_queries_at_max": len(distinct),
            "max_observed_refs": sorted(s.ref for s in at_top),
            "statement": (
                f"largest response observed for {key} is {top} rows across {len(group)} responses. "
                "This is EVIDENCE about where a row limit is not; it is not proof of a fixed cap."
            ),
        }
    return out


def cap_signal(snapshot: SnapshotRecord, observations: dict[str, dict]) -> dict:
    obs = observations[f"{snapshot.provider}:{snapshot.fetch_mode}"]
    at_max = snapshot.record_count == obs["max_observed_count"]
    recurring = obs["distinct_queries_at_max"] >= 2
    signal = at_max and recurring
    below = snapshot.record_count < obs["max_observed_count"]
    if signal:
        why = (f"{snapshot.record_count} rows equals the maximum observed for {snapshot.provider}:"
               f"{snapshot.fetch_mode}, and that same maximum recurred across "
               f"{obs['distinct_queries_at_max']} distinct queries - the signature of a row limit")
    elif below:
        why = (f"{snapshot.record_count} rows is below the {obs['max_observed_count']}-row maximum this "
               "mode has returned (evidence a cap, if any, is higher; not proof of no cap)")
    else:
        why = (f"{snapshot.record_count} rows is the largest response observed for this mode but has not "
               "recurred - no cap signal, but also no evidence ruling one out")
    return {"cap_signal": signal, "below_observed_max": below, "explanation": why}


@dataclasses.dataclass(frozen=True)
class BoundarySemantics:
    from_inclusive: bool | None
    to_inclusive: bool | None
    detail: str

    @property
    def verified(self) -> bool:
        return self.from_inclusive is not None and self.to_inclusive is not None

    def as_dict(self) -> dict:
        return {"from_inclusive": self.from_inclusive, "to_inclusive": self.to_inclusive,
                "verified": self.verified, "detail": self.detail}


def infer_boundary_semantics(
    day: datetime.date,
    single_day_dates: Sequence[datetime.date],
    from_at_day_dates: Sequence[datetime.date],
    to_at_day_dates: Sequence[datetime.date],
    control_dates: set[datetime.date],
) -> BoundarySemantics:
    if day not in control_dates:
        return BoundarySemantics(None, None, f"{day} not confirmed present by an independent query; cannot test")
    from_inc = day in set(from_at_day_dates)
    to_inc = day in set(to_at_day_dates)
    single = day in set(single_day_dates)
    if single != (from_inc and to_inc):
        return BoundarySemantics(
            None, None,
            f"inconsistent probes: single-day window returned the event={single}, but from-inclusive={from_inc}, "
            f"to-inclusive={to_inc}",
        )
    return BoundarySemantics(from_inc, to_inc, f"probed on {day}: from inclusive={from_inc}, to inclusive={to_inc}")


def _covered(day: datetime.date, lo: datetime.date, hi: datetime.date, sem: BoundarySemantics) -> bool:
    if lo < day < hi:
        return True
    return (day == lo and bool(sem.from_inclusive)) or (day == hi and bool(sem.to_inclusive))


def partition_consistency(
    whole: SnapshotRecord, parts: Sequence[SnapshotRecord], semantics: BoundarySemantics
) -> dict:
    result: dict = {
        "whole_ref": whole.ref, "part_refs": [p.ref for p in parts],
        "boundary_semantics_verified": semantics.verified,
    }
    if not semantics.verified:
        result.update(consistent=False, reason="boundary semantics not verified: gaps at partition boundaries cannot be ruled out",
                      gap_days=None, only_in_whole=[], only_in_parts=[], out_of_window_rows=[])
        return result

    lo, hi = whole.query_from, whole.query_to
    gaps, d = [], lo
    while d <= hi:
        if not any(_covered(d, p.query_from, p.query_to, semantics) for p in parts):
            gaps.append(d.isoformat())
        d += datetime.timedelta(days=1)

    out_of_window = []
    for s in (whole, *parts):
        for row, rd in zip(s.rows, s.row_dates()):
            if rd is None or not _covered(rd, s.query_from, s.query_to, semantics):
                out_of_window.append({"snapshot": s.ref, "ex_date": rd.isoformat() if rd else None})

    whole_ids = {_canonical(r): rd for r, rd in zip(whole.rows, whole.row_dates())}
    part_ids = {}
    for p in parts:
        for r, rd in zip(p.rows, p.row_dates()):
            part_ids[_canonical(r)] = rd
    only_whole = sorted((rd.isoformat() if rd else "?") for k, rd in whole_ids.items() if k not in part_ids)
    only_parts = sorted((rd.isoformat() if rd else "?") for k, rd in part_ids.items() if k not in whole_ids)

    consistent = not (gaps or out_of_window or only_whole or only_parts)
    result.update(
        consistent=consistent,
        reason="whole window equals the union of its parts" if consistent else
        "gaps, out-of-window rows, or rows present in only one of whole/union",
        gap_days=gaps[:20], gap_day_count=len(gaps), only_in_whole=only_whole, only_in_parts=only_parts,
        out_of_window_rows=out_of_window[:20],
        whole_rows=len(whole.rows), union_rows=len(part_ids),
    )
    return result


def snapshot_coverage_window(s: SnapshotRecord) -> tuple[datetime.date, datetime.date] | None:
    if s.fetch_mode == "windowed":
        return (s.query_from, s.query_to)
    dates = [d for d in s.row_dates() if d]
    if not dates:
        return None
    return (min(dates), datetime.date.fromisoformat(s.fetched_at[:10]))


def comparable_window(
    older: SnapshotRecord, newer: SnapshotRecord, evidence_window: tuple[datetime.date, datetime.date]
) -> tuple[datetime.date, datetime.date] | None:
    a, b = snapshot_coverage_window(older), snapshot_coverage_window(newer)
    if a is None or b is None:
        return None
    lo = max(a[0], b[0], evidence_window[0])
    hi = min(a[1], b[1], evidence_window[1])
    return (lo, hi) if lo <= hi else None
