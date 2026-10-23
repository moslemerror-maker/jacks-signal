from __future__ import annotations

import dataclasses
import datetime
import hashlib
from decimal import Decimal
from fractions import Fraction
from typing import Sequence

from backend.app.research.adjustment_eligibility import (
    SUPPORTED_ADJUSTMENT_TYPES,
    AdjustmentEligibility,
    EventView,
    in_interval,
)
from backend.app.research.corporate_actions_coverage import canonical_json
from backend.app.research.exact_terms import canonical_decimal, normalize_terms, terms_equal, try_normalize

ADJUSTMENT_BASIS_VERSION = "c3-adjustment-basis/1"
PRICE_BASIS_RAW = "RAW_AS_TRADED_UNADJUSTED"
PRICE_BASIS_ADJUSTED = "SPLIT_BONUS_ADJUSTED_TO_ASOF_NOT_DIVIDEND_ADJUSTED"
CALENDAR_BASIS = "OBSERVED_SESSIONS_ACROSS_ALL_INSTRUMENTS_NOT_AN_EXCHANGE_CALENDAR"
PIT_NOTE = (
    "ex-date-effective: an event adjusts a bar iff bar_date < ex_date <= as_of. The announcement-time "
    "point-in-time status of the attested terms is NOT_ESTABLISHED (Phase C2)."
)

ADJUSTED = "ADJUSTED"
ADJUSTED_WITH_GAPS = "ADJUSTED_WITH_GAPS"
REFUSED = "REFUSED"
NO_BARS = "NO_BARS"
ELIGIBILITY_BLOCKED = "ELIGIBILITY_BLOCKED"
INVALID_RAW_BARS = "INVALID_RAW_BARS"
MISSING_BARS = "MISSING_BARS"


class AdjustmentError(ValueError):
    pass


def fraction_dict(f: Fraction) -> dict:
    return {"numerator": str(f.numerator), "denominator": str(f.denominator)}


def fraction_text(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}"


def component_factor(kind: str, term_a, term_b) -> Fraction:
    t = normalize_terms(kind, term_a, term_b)
    a, b = Fraction(t.term_a), Fraction(t.term_b)
    if kind == "SPLIT":
        return b / a
    return b / (a + b)


@dataclasses.dataclass(frozen=True)
class ComponentTerms:
    kind: str
    term_a: Decimal
    term_b: Decimal
    c2_record_id: str
    tier: int = 1

    def factor(self) -> Fraction:
        return component_factor(self.kind, self.term_a, self.term_b)


@dataclasses.dataclass(frozen=True)
class FactorEvent:
    event_id: str
    ex_date: datetime.date
    components: tuple[ComponentTerms, ...]

    def factor(self) -> Fraction:
        f = Fraction(1)
        for c in self.components:
            f *= c.factor()
        return f


def factor_events_from_views(events: Sequence[EventView]) -> tuple[FactorEvent, ...]:
    out = []
    for ev in events:
        comps = []
        for c in ev.components:
            if c.component_type not in SUPPORTED_ADJUSTMENT_TYPES:
                continue
            rec = c.verification_record
            if rec is None:
                raise AdjustmentError(
                    f"event {ev.event_id} ({ev.ex_date}): {c.component_type} component has no active C2 record; "
                    "it cannot be adjusted")
            have, why = try_normalize(c.component_type, *(c.terms or (None, None)))
            attested, _ = try_normalize(rec.kind, rec.term_a, rec.term_b)
            if have is None or attested is None or rec.kind != c.component_type or not terms_equal(have, attested):
                raise AdjustmentError(
                    f"event {ev.event_id} ({ev.ex_date}): component terms {c.terms} ({why or 'well-formed'}) do not equal "
                    f"the active C2 record's {rec.term_a}:{rec.term_b}")
            comps.append(ComponentTerms(rec.kind, attested.term_a, attested.term_b, rec.record_id, rec.tier))
        if comps:
            out.append(FactorEvent(ev.event_id, ev.ex_date, tuple(comps)))
    return tuple(sorted(out, key=lambda e: (e.ex_date, e.event_id)))


def cumulative_factor(events: Sequence[FactorEvent], *, start: datetime.date, as_of: datetime.date) -> Fraction:
    if start > as_of:
        raise AdjustmentError(f"start {start} is after as_of {as_of}")
    seen = set()
    f = Fraction(1)
    for ev in events:
        if ev.event_id in seen:
            raise AdjustmentError(f"event {ev.event_id} appears more than once")
        seen.add(ev.event_id)
        if in_interval(ev.ex_date, start, as_of):
            f *= ev.factor()
    return f


def price_exact(x) -> Fraction:
    if isinstance(x, bool) or x is None:
        raise AdjustmentError(f"not a price: {x!r}")
    if isinstance(x, Decimal):
        d = x
    elif isinstance(x, int):
        d = Decimal(x)
    elif isinstance(x, float):
        d = Decimal(repr(x))
    else:
        raise AdjustmentError(f"unsupported price type {type(x).__name__}")
    if not d.is_finite():
        raise AdjustmentError(f"price is not finite: {x!r}")
    return Fraction(d)


@dataclasses.dataclass(frozen=True)
class RawBar:
    bar_id: str
    trade_date: datetime.date
    series: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float


def raw_bar_problem(bar: RawBar) -> str | None:
    try:
        o, h, l, c = (price_exact(v) for v in (bar.open, bar.high, bar.low, bar.close))
    except AdjustmentError as exc:
        return str(exc)
    if min(o, h, l, c) <= 0:
        return "non-positive price"
    if not (l <= o <= h and l <= c <= h and l <= h):
        return "OHLC ordering violated"
    return None


@dataclasses.dataclass(frozen=True)
class AdjustedBar:
    raw: RawBar
    factor: Fraction
    open_exact: Fraction
    high_exact: Fraction
    low_exact: Fraction
    close_exact: Fraction
    open: float
    high: float
    low: float
    close: float

    def as_dict(self) -> dict:
        r = self.raw
        return {
            "trade_date": r.trade_date.isoformat(), "source_bar_id": r.bar_id, "series": r.series, "timeframe": r.timeframe,
            "raw": {"open": r.open, "high": r.high, "low": r.low, "close": r.close},
            "factor": fraction_dict(self.factor),
            "adjusted_exact": {"open": fraction_text(self.open_exact), "high": fraction_text(self.high_exact),
                               "low": fraction_text(self.low_exact), "close": fraction_text(self.close_exact)},
            "adjusted": {"open": self.open, "high": self.high, "low": self.low, "close": self.close},
        }


def adjust_bar(bar: RawBar, factor: Fraction) -> AdjustedBar:
    if factor <= 0:
        raise AdjustmentError(f"a factor must be positive, got {factor}")
    problem = raw_bar_problem(bar)
    if problem:
        raise AdjustmentError(f"raw bar {bar.trade_date}: {problem}")
    o, h, l, c = (price_exact(v) * factor for v in (bar.open, bar.high, bar.low, bar.close))
    return AdjustedBar(bar, factor, o, h, l, c, float(o), float(h), float(l), float(c))


def adjust_bars(raw_bars: Sequence[RawBar], events: Sequence[FactorEvent], *, as_of: datetime.date) -> tuple[AdjustedBar, ...]:
    ordered = sorted(raw_bars, key=lambda b: b.trade_date)
    if len({(b.series, b.timeframe) for b in ordered}) > 1:
        raise AdjustmentError("bars of more than one series/timeframe cannot be adjusted together")
    if len({b.trade_date for b in ordered}) != len(ordered):
        raise AdjustmentError("more than one bar on one trade date")
    for b in ordered:
        if b.trade_date > as_of:
            raise AdjustmentError(f"bar {b.trade_date} is after as_of {as_of}: the view must not contain the future")
    return tuple(adjust_bar(b, cumulative_factor(events, start=b.trade_date, as_of=as_of)) for b in ordered)


def missing_sessions(bar_dates: Sequence[datetime.date], session_dates: Sequence[datetime.date],
                     *, as_of: datetime.date) -> tuple[datetime.date, ...]:
    if not bar_dates:
        return ()
    first, have = min(bar_dates), set(bar_dates)
    return tuple(sorted(s for s in set(session_dates) if first <= s <= as_of and s not in have))


def sha256_of(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class AdjustedView:
    status: str
    refusal_reasons: tuple[str, ...]
    instrument_id: str
    series: str
    timeframe: str
    requested_start: datetime.date
    as_of: datetime.date
    eligibility: AdjustmentEligibility
    basis: dict | None
    bars: tuple[AdjustedBar, ...]
    data_quality: dict
    basis_sha256: str | None
    view_sha256: str
    safe_start_hint: datetime.date | None = None

    def content(self) -> dict:
        return {
            "status": self.status, "refusal_reasons": list(self.refusal_reasons), "instrument_id": self.instrument_id,
            "series": self.series, "timeframe": self.timeframe, "requested_start": self.requested_start.isoformat(),
            "as_of": self.as_of.isoformat(), "eligibility": self.eligibility.as_dict(), "basis": self.basis,
            "data_quality": self.data_quality, "bars": [b.as_dict() for b in self.bars],
            "safe_start_hint": self.safe_start_hint.isoformat() if self.safe_start_hint else None,
        }

    def as_dict(self) -> dict:
        return {**self.content(), "basis_sha256": self.basis_sha256, "view_sha256": self.view_sha256}


def _basis(*, instrument_id, series, timeframe, requested_start, as_of, eligibility, events, bars_first_factor) -> dict:
    used = [e for e in events if in_interval(e.ex_date, requested_start, as_of)]
    return {
        "version": ADJUSTMENT_BASIS_VERSION, "instrument_id": instrument_id, "series": series, "timeframe": timeframe,
        "requested_start": requested_start.isoformat(), "as_of": as_of.isoformat(),
        "interval_semantics": "per bar: bar_date < ex_date <= as_of (the gate's (start, end] anchored at end)",
        "price_basis_raw": PRICE_BASIS_RAW, "price_basis_adjusted": PRICE_BASIS_ADJUSTED,
        "dividends_adjusted": False, "volume_included": False, "point_in_time": PIT_NOTE,
        "eligibility": {
            "eligible": eligibility.eligible, "coverage_status": eligibility.coverage_status,
            "governing_assessment_id": eligibility.governing_assessment_id, "events_considered": eligibility.events_considered,
            "start": eligibility.start.isoformat(), "end": eligibility.end.isoformat(),
        },
        "events": [
            {"event_id": e.event_id, "ex_date": e.ex_date.isoformat(), "event_factor": fraction_dict(e.factor()),
             "components": [
                 {"kind": c.kind, "term_a": canonical_decimal(c.term_a), "term_b": canonical_decimal(c.term_b),
                  "c2_record_id": c.c2_record_id, "tier": c.tier, "factor": fraction_dict(c.factor())}
                 for c in e.components]}
            for e in used],
        "cumulative_factor_at_requested_start": fraction_dict(bars_first_factor),
    }


def build_adjusted_view(
    *,
    instrument_id: str,
    series: str,
    timeframe: str,
    requested_start: datetime.date,
    as_of: datetime.date,
    raw_bars: Sequence[RawBar],
    session_dates: Sequence[datetime.date],
    eligibility: AdjustmentEligibility,
    gate_events: Sequence[EventView],
    allow_missing_bars: bool = False,
) -> AdjustedView:
    if requested_start > as_of:
        raise AdjustmentError(f"requested_start {requested_start} is after as_of {as_of}")
    if eligibility.start != requested_start or eligibility.end != as_of:
        raise AdjustmentError(
            f"the eligibility was evaluated for ({eligibility.start}, {eligibility.end}], not for the requested "
            f"({requested_start}, {as_of}]")
    for b in raw_bars:
        if not (requested_start <= b.trade_date <= as_of) or (b.series, b.timeframe) != (series, timeframe):
            raise AdjustmentError(f"raw bar {b.trade_date} {b.series}/{b.timeframe} is outside the request")

    bar_dates = [b.trade_date for b in raw_bars]
    invalid = [(b.trade_date.isoformat(), p) for b in sorted(raw_bars, key=lambda x: x.trade_date) if (p := raw_bar_problem(b))]
    missing = missing_sessions(bar_dates, session_dates, as_of=as_of)
    expected = 0 if not bar_dates else len({s for s in session_dates if min(bar_dates) <= s <= as_of})
    quality = {
        "raw_bar_count": len(raw_bars), "first_bar_date": min(bar_dates).isoformat() if bar_dates else None,
        "last_bar_date": max(bar_dates).isoformat() if bar_dates else None, "expected_session_count": expected,
        "missing_sessions": [d.isoformat() for d in missing], "invalid_raw_bars": invalid,
        "complete": bool(bar_dates) and not missing and not invalid, "calendar_basis": CALENDAR_BASIS,
    }

    reasons = []
    if not raw_bars:
        reasons.append(NO_BARS)
    if not eligibility.eligible:
        reasons.append(ELIGIBILITY_BLOCKED)
    if invalid:
        reasons.append(INVALID_RAW_BARS)
    if missing and not allow_missing_bars:
        reasons.append(MISSING_BARS)

    if reasons:
        hint = None
        if ELIGIBILITY_BLOCKED in reasons and eligibility.barrier_dates and all(b.ex_date is not None for b in eligibility.blockers):
            hint = max(eligibility.barrier_dates)
        view = AdjustedView(REFUSED, tuple(reasons), instrument_id, series, timeframe, requested_start, as_of, eligibility,
                            None, (), quality, None, "", hint)
        return dataclasses.replace(view, view_sha256=sha256_of(view.content()))

    events = factor_events_from_views([e for e in gate_events if in_interval(e.ex_date, requested_start, as_of)])
    bars = adjust_bars(raw_bars, events, as_of=as_of)
    basis = _basis(instrument_id=instrument_id, series=series, timeframe=timeframe, requested_start=requested_start, as_of=as_of,
                   eligibility=eligibility, events=events, bars_first_factor=cumulative_factor(events, start=requested_start, as_of=as_of))
    status = ADJUSTED if quality["complete"] else ADJUSTED_WITH_GAPS
    view = AdjustedView(status, (), instrument_id, series, timeframe, requested_start, as_of, eligibility, basis, bars, quality,
                        sha256_of(basis), "", None)
    return dataclasses.replace(view, view_sha256=sha256_of(view.content()))
