from __future__ import annotations

import copy
import dataclasses
import itertools
import json
import math
import random
from decimal import Decimal
from fractions import Fraction as F

import pytest

from backend.app.research import adjustment_factors as af
from backend.app.research.adjustment_eligibility import (
    ComponentView,
    EventView,
    RecordView,
    RegisterItemView,
    evaluate_adjustment_eligibility,
)
from backend.app.research.exact_terms import InvalidTerms
from datetime import date as D, timedelta

SEED = 20260921


def ce(kind, a, b, rid="rec"):
    return af.ComponentTerms(kind, Decimal(a), Decimal(b), rid)


def fe(eid, ex, *comps):
    return af.FactorEvent(eid, ex, tuple(comps))


def split(eid, ex, old, new):
    return fe(eid, ex, ce("SPLIT", old, new, f"rec-{eid}"))


def bonus(eid, ex, new, existing):
    return fe(eid, ex, ce("BONUS", new, existing, f"rec-{eid}"))


def bar(d, o, h, l, c, series="EQ", tf="1D", bid=None):
    return af.RawBar(bid or f"bar-{d.isoformat()}", d, series, tf, o, h, l, c)


def view_event(eid, ex, *comps, status="MANUALLY_VERIFIED"):
    return EventView(eid, ex, status, tuple(comps))


def vsplit(old=10.0, new=1.0, status="MANUALLY_VERIFIED", record=True, rec_terms=None, tier=1):
    rt = rec_terms or (Decimal(str(old)), Decimal(str(new)))
    return ComponentView("SPLIT", status, "Face Value Split", (old, new),
                         RecordView("rec-split", "SPLIT", rt[0], rt[1], tier) if record else None)


def vbonus(new=1, existing=1, status="MANUALLY_VERIFIED", record=True):
    return ComponentView("BONUS", status, "Bonus", (new, existing),
                         RecordView("rec-bonus", "BONUS", Decimal(new), Decimal(existing), 1) if record else None)


def gate(start, end, events, register=(), coverage="COMPLETE"):
    return evaluate_adjustment_eligibility(start=start, end=end, coverage_status=coverage, governing_assessment_id="A1",
                                           events=events, register_items=register)


JSW = [
    bar(D(2016, 12, 30), 1599.0, 1632.45, 1595.65, 1627.5),
    bar(D(2017, 1, 2), 1628.0, 1641.85, 1610.1, 1634.3),
    bar(D(2017, 1, 3), 1641.0, 1653.6, 1631.0, 1644.55),
    bar(D(2017, 1, 4), 166.75, 166.75, 160.6, 163.05),
    bar(D(2017, 1, 5), 164.05, 176.65, 163.5, 175.65),
]
JSW_SESSIONS = [b.trade_date for b in JSW]
EX = D(2017, 1, 4)


def view(bars=JSW, start=D(2016, 12, 30), as_of=D(2017, 1, 5), events=None, register=(), coverage="COMPLETE", sessions=None,
         allow=False, ev_views=None):
    ev_views = ev_views if ev_views is not None else [view_event("evt-jsw", EX, vsplit())]
    return af.build_adjusted_view(
        instrument_id="inst-1", series="EQ", timeframe="1D", requested_start=start, as_of=as_of, raw_bars=bars,
        session_dates=JSW_SESSIONS if sessions is None else sessions,
        eligibility=gate(start, as_of, ev_views, register, coverage), gate_events=ev_views, allow_missing_bars=allow)


class TestExactFactors:
    @pytest.mark.parametrize("kind,a,b,expected", [
        ("SPLIT", 10, 1, F(1, 10)),
        ("SPLIT", 10, 2, F(1, 5)),
        ("SPLIT", 2, 1, F(1, 2)),
        ("SPLIT", 5, 1, F(1, 5)),
        ("BONUS", 1, 1, F(1, 2)),
        ("BONUS", 4, 1, F(1, 5)),
        ("BONUS", 3, 2, F(2, 5)),
        ("BONUS", 1, 2, F(2, 3)),
        ("SPLIT", Decimal("2.5"), 1, F(2, 5)),
        ("SPLIT", 1, 10, F(10, 1)),
    ])
    def test_the_documented_values(self, kind, a, b, expected):
        f = af.component_factor(kind, a, b)
        assert f == expected and isinstance(f, F)

    def test_the_source_of_truth_is_never_a_float(self):
        f = af.component_factor("SPLIT", 3, 1)
        assert isinstance(f, F) and not isinstance(f, float)
        assert f * 3 == 1
        assert af.fraction_dict(f) == {"numerator": "1", "denominator": "3"}
        assert af.fraction_text(f) == "1/3"

    @pytest.mark.parametrize("a,b", [(10.0, 1.0), (Decimal("10.000000"), Decimal("1.000000")), (10, Decimal("1")), (Decimal(10), 1.0)])
    def test_every_numeric_form_of_the_same_terms_gives_the_same_factor(self, a, b):
        assert af.component_factor("SPLIT", a, b) == F(1, 10)

    @pytest.mark.parametrize("kind,a,b", [
        ("SPLIT", 0, 1), ("SPLIT", 1, 0), ("SPLIT", -10, 1), ("SPLIT", 5, 5),
        ("BONUS", 0, 1), ("BONUS", 1, 0), ("BONUS", Decimal("1.5"), 1),
        ("RIGHTS", 1, 1), ("DEMERGER", 1, 1), ("OTHER", 1, 1),
        ("SPLIT", None, 1), ("SPLIT", float("nan"), 1), ("SPLIT", float("inf"), 1), ("SPLIT", True, 1),
    ])
    def test_invalid_terms_are_refused_by_the_same_validation_c2_uses(self, kind, a, b):
        with pytest.raises(InvalidTerms):
            af.component_factor(kind, a, b)

    def test_exhaustive_small_domain_formulas(self):
        for a in range(1, 31):
            for b in range(1, 31):
                if a != b:
                    s = af.component_factor("SPLIT", a, b)
                    assert s == F(b, a) and s.denominator > 0
                bo = af.component_factor("BONUS", a, b)
                assert bo == F(b, a + b) and 0 < bo < 1
                assert bo * F(a + b, b) == 1

    def test_the_numerator_and_denominator_are_in_lowest_terms(self):
        f = af.component_factor("SPLIT", 10, 2)
        assert (f.numerator, f.denominator) == (1, 5)

    def test_an_event_and_a_compound_event_multiply_exactly(self):
        e = fe("baj16", D(2016, 9, 8), ce("BONUS", 1, 1), ce("SPLIT", 10, 2))
        assert e.factor() == F(1, 2) * F(1, 5) == F(1, 10)

    def test_a_compound_factor_equals_the_exact_product_of_random_components(self):
        rng = random.Random(SEED)
        for _ in range(200):
            comps = []
            for _ in range(rng.randint(1, 5)):
                if rng.random() < 0.5:
                    a = rng.randint(2, 50); b = rng.randint(1, a - 1)
                    comps.append(ce("SPLIT", a, b))
                else:
                    comps.append(ce("BONUS", rng.randint(1, 9), rng.randint(1, 9)))
            expected = F(1)
            for c in comps:
                expected *= af.component_factor(c.kind, c.term_a, c.term_b)
            assert fe("e", D(2020, 1, 1), *comps).factor() == expected


class TestBajfinanceCompoundRegressions:
    S16 = split("baj16-split", D(2016, 9, 8), 10, 2)
    B16 = bonus("baj16-bonus", D(2016, 9, 8), 1, 1)
    S25 = split("baj25-split", D(2025, 6, 16), 2, 1)
    B25 = bonus("baj25-bonus", D(2025, 6, 16), 4, 1)

    def test_the_2016_pair(self):
        assert self.S16.factor() == F(1, 5) and self.B16.factor() == F(1, 2)
        assert af.cumulative_factor([self.S16, self.B16], start=D(2016, 9, 7), as_of=D(2016, 9, 8)) == F(1, 10)

    def test_the_2025_pair(self):
        assert self.S25.factor() == F(1, 2) and self.B25.factor() == F(1, 5)
        assert af.cumulative_factor([self.S25, self.B25], start=D(2025, 6, 13), as_of=D(2025, 6, 16)) == F(1, 10)

    def test_both_pairs_together_are_exactly_one_hundredth(self):
        evs = [self.S16, self.B16, self.S25, self.B25]
        assert af.cumulative_factor(evs, start=D(2015, 1, 1), as_of=D(2026, 9, 11)) == F(1, 100)

    def test_a_bar_between_the_pairs_sees_only_the_later_pair_and_a_bar_after_both_sees_none(self):
        evs = [self.S16, self.B16, self.S25, self.B25]
        assert af.cumulative_factor(evs, start=D(2016, 9, 8), as_of=D(2026, 9, 11)) == F(1, 10)
        assert af.cumulative_factor(evs, start=D(2025, 6, 16), as_of=D(2026, 9, 11)) == F(1)

    def test_the_same_date_events_of_2025_are_separate_events_and_both_apply(self):
        assert self.S25.ex_date == self.B25.ex_date and self.S25.event_id != self.B25.event_id
        assert af.cumulative_factor([self.S25], start=D(2025, 6, 13), as_of=D(2025, 6, 16)) == F(1, 2)
        assert af.cumulative_factor([self.S25, self.B25], start=D(2025, 6, 13), as_of=D(2025, 6, 16)) == F(1, 10)

    def test_the_real_raw_prices_fall_by_the_exact_factor_to_within_the_market_move(self):
        adj = af.price_exact(11393.3) * F(1, 10)
        assert adj == F(113933, 100)
        assert abs(float(adj) / 1162.8 - 1) < 0.03


class TestBoundarySemantics:
    E = [split("e1", D(2020, 6, 1), 10, 1)]

    def test_no_event_means_factor_one_exactly(self):
        f = af.cumulative_factor([], start=D(2020, 1, 1), as_of=D(2020, 12, 31))
        assert f == 1 and f.numerator == 1 and f.denominator == 1

    def test_an_event_exactly_on_the_start_is_excluded(self):
        assert af.cumulative_factor(self.E, start=D(2020, 6, 1), as_of=D(2020, 12, 31)) == 1

    def test_an_event_the_day_after_the_start_is_included(self):
        assert af.cumulative_factor(self.E, start=D(2020, 5, 31), as_of=D(2020, 12, 31)) == F(1, 10)

    def test_an_event_exactly_on_the_as_of_is_included(self):
        assert af.cumulative_factor(self.E, start=D(2020, 1, 1), as_of=D(2020, 6, 1)) == F(1, 10)

    def test_an_event_the_day_after_the_as_of_is_excluded(self):
        assert af.cumulative_factor(self.E, start=D(2020, 1, 1), as_of=D(2020, 5, 31)) == 1

    def test_start_equal_to_as_of_is_an_empty_interval(self):
        assert af.cumulative_factor(self.E, start=D(2020, 6, 1), as_of=D(2020, 6, 1)) == 1
        assert af.cumulative_factor(self.E, start=D(2020, 5, 31), as_of=D(2020, 5, 31)) == 1

    def test_start_after_as_of_is_refused(self):
        with pytest.raises(af.AdjustmentError):
            af.cumulative_factor(self.E, start=D(2020, 6, 2), as_of=D(2020, 6, 1))

    def test_the_boundary_is_exactly_the_gates_own_definition(self):
        from backend.app.research.adjustment_eligibility import in_interval
        base = D(2020, 1, 1)
        for start_off, end_off, ex_off in itertools.product(range(0, 5), range(0, 5), range(-1, 7)):
            start, end, ex = base + timedelta(start_off), base + timedelta(start_off + end_off), base + timedelta(ex_off)
            f = af.cumulative_factor([split("x", ex, 10, 1)], start=start, as_of=end)
            assert (f == F(1, 10)) == in_interval(ex, start, end) == (start < ex <= end)

    def test_a_non_trading_ex_date_needs_no_calendar(self):
        sat = D(2020, 6, 6)
        assert sat.weekday() == 5
        assert af.cumulative_factor([split("x", sat, 10, 1)], start=D(2020, 6, 5), as_of=D(2020, 6, 8)) == F(1, 10)
        assert af.cumulative_factor([split("x", sat, 10, 1)], start=D(2020, 6, 6), as_of=D(2020, 6, 8)) == 1


class TestPointInTime:
    def _random_events(self, rng, n):
        base = D(2015, 1, 1)
        out = []
        for i in range(n):
            ex = base + timedelta(rng.randint(0, 4000))
            if rng.random() < 0.5:
                a = rng.randint(2, 20); out.append(split(f"s{i}", ex, a, rng.randint(1, a - 1)))
            else:
                out.append(bonus(f"b{i}", ex, rng.randint(1, 6), rng.randint(1, 6)))
        return out

    def test_a_future_event_never_changes_an_earlier_as_of_value(self):
        rng = random.Random(SEED)
        for _ in range(300):
            events = self._random_events(rng, rng.randint(0, 6))
            as_of = D(2015, 1, 1) + timedelta(rng.randint(0, 4000))
            start = as_of - timedelta(rng.randint(0, 1500))
            future = [split(f"f{i}", as_of + timedelta(rng.randint(1, 3000)), 10, 1) for i in range(rng.randint(1, 3))]
            known = [e for e in events if e.ex_date <= as_of]
            assert (af.cumulative_factor(events + future, start=start, as_of=as_of)
                    == af.cumulative_factor(events, start=start, as_of=as_of)
                    == af.cumulative_factor(known, start=start, as_of=as_of))

    def test_adjusted_bars_as_of_an_earlier_date_are_identical_with_or_without_a_later_event(self):
        early = split("early", D(2017, 1, 4), 10, 1)
        later = split("later", D(2018, 3, 1), 2, 1)
        a = af.adjust_bars(JSW, [early], as_of=D(2017, 1, 5))
        b = af.adjust_bars(JSW, [early, later], as_of=D(2017, 1, 5))
        assert a == b

    def test_as_of_before_at_and_after_the_event_for_each_bar(self):
        ev = [split("jsw", EX, 10, 1)]
        for as_of, expected in (
            (D(2017, 1, 3), {D(2016, 12, 30): 1, D(2017, 1, 2): 1, D(2017, 1, 3): 1}),
            (D(2017, 1, 4), {D(2016, 12, 30): F(1, 10), D(2017, 1, 2): F(1, 10), D(2017, 1, 3): F(1, 10), D(2017, 1, 4): 1}),
            (D(2017, 1, 5), {D(2016, 12, 30): F(1, 10), D(2017, 1, 2): F(1, 10), D(2017, 1, 3): F(1, 10), D(2017, 1, 4): 1, D(2017, 1, 5): 1}),
        ):
            got = {b.raw.trade_date: b.factor for b in af.adjust_bars([b for b in JSW if b.trade_date <= as_of], ev, as_of=as_of)}
            assert got == {k: F(v) for k, v in expected.items()}

    def test_a_bar_after_the_as_of_is_refused_not_dropped(self):
        with pytest.raises(af.AdjustmentError, match="future"):
            af.adjust_bars(JSW, [split("jsw", EX, 10, 1)], as_of=D(2017, 1, 3))

    def test_reordering_the_events_never_changes_the_cumulative_factor(self):
        evs = [split("a", D(2016, 9, 8), 10, 2), bonus("b", D(2016, 9, 8), 1, 1), split("c", D(2025, 6, 16), 2, 1),
               bonus("d", D(2025, 6, 16), 4, 1), split("e", D(2017, 1, 4), 10, 1)]
        want = af.cumulative_factor(evs, start=D(2015, 1, 1), as_of=D(2026, 9, 11))
        assert want == F(1, 1000)
        for perm in itertools.permutations(evs):
            assert af.cumulative_factor(list(perm), start=D(2015, 1, 1), as_of=D(2026, 9, 11)) == want

    def test_reordering_the_components_of_one_event_never_changes_its_factor(self):
        comps = [ce("BONUS", 1, 1), ce("SPLIT", 10, 2), ce("BONUS", 4, 1)]
        for perm in itertools.permutations(comps):
            assert fe("e", D(2020, 1, 1), *perm).factor() == F(1, 50)

    def test_the_same_event_twice_is_refused_it_would_silently_apply_twice(self):
        e = split("dup", D(2020, 6, 1), 10, 1)
        with pytest.raises(af.AdjustmentError, match="more than once"):
            af.cumulative_factor([e, e], start=D(2020, 1, 1), as_of=D(2020, 12, 31))

    def test_rerunning_is_idempotent_and_deterministic(self):
        ev = [split("jsw", EX, 10, 1)]
        first = af.adjust_bars(JSW, ev, as_of=D(2017, 1, 5))
        for _ in range(3):
            assert af.adjust_bars(JSW, ev, as_of=D(2017, 1, 5)) == first
        assert view().view_sha256 == view().view_sha256 == view().as_dict()["view_sha256"]


class TestExactPrices:
    def test_a_price_is_its_published_decimal_not_its_binary_expansion(self):
        assert af.price_exact(2.5) == F(5, 2)
        assert af.price_exact(1644.55) == F(32891, 20)
        assert af.price_exact(0.1) == F(1, 10)
        assert af.price_exact(Decimal("0.10")) == F(1, 10) and af.price_exact(7) == 7

    @pytest.mark.parametrize("bad", [None, True, "1", float("nan"), float("inf"), float("-inf"), Decimal("NaN")])
    def test_non_prices_are_refused(self, bad):
        with pytest.raises(af.AdjustmentError):
            af.price_exact(bad)

    def test_the_jswsteel_bar_before_the_split_is_one_tenth_exactly(self):
        (b,) = af.adjust_bars([JSW[2]], [split("jsw", EX, 10, 1)], as_of=D(2017, 1, 5))
        assert b.factor == F(1, 10)
        assert (b.open_exact, b.high_exact, b.low_exact, b.close_exact) == (F(1641, 10), F(16536, 100), F(1631, 10), F(32891, 200))
        assert (b.open, b.high, b.low, b.close) == (164.1, 165.36, 163.1, 164.455)
        assert b.raw is JSW[2] and (b.raw.open, b.raw.high, b.raw.low, b.raw.close) == (1641.0, 1653.6, 1631.0, 1644.55)

    def test_a_bar_on_and_after_the_ex_date_is_unchanged(self):
        out = af.adjust_bars(JSW[3:], [split("jsw", EX, 10, 1)], as_of=D(2017, 1, 5))
        for b in out:
            assert b.factor == 1 and (b.open, b.high, b.low, b.close) == (b.raw.open, b.raw.high, b.raw.low, b.raw.close)

    def test_the_adjusted_float_is_the_correctly_rounded_exact_value(self):
        for b in af.adjust_bars(JSW, [split("jsw", EX, 10, 1)], as_of=D(2017, 1, 5)):
            assert b.open == float(b.open_exact) and b.high == float(b.high_exact)
            assert b.low == float(b.low_exact) and b.close == float(b.close_exact)

    def test_ohlc_ordering_is_preserved_for_random_bars_and_factors(self):
        rng = random.Random(SEED)
        for _ in range(3000):
            lo = round(rng.uniform(1, 5000), rng.choice([0, 1, 2, 3]))
            hi = round(lo + rng.uniform(0, 500), 2)
            o = round(rng.uniform(lo, hi), 2); c = round(rng.uniform(lo, hi), 2)
            lo, hi = min(lo, o, c), max(hi, o, c)
            b = bar(D(2020, 1, 1), o, hi, lo, c)
            if af.raw_bar_problem(b):
                continue
            a = rng.randint(1, 60); bb = rng.randint(1, 60)
            f = F(a, bb)
            r = af.adjust_bar(b, f)
            assert r.low_exact <= min(r.open_exact, r.close_exact) and max(r.open_exact, r.close_exact) <= r.high_exact
            assert r.low <= min(r.open, r.close) and max(r.open, r.close) <= r.high
            assert r.high_exact / f == af.price_exact(hi) and r.low_exact / f == af.price_exact(lo)

    @pytest.mark.parametrize("factor", [F(0), F(-1, 10)])
    def test_a_non_positive_factor_is_refused(self, factor):
        with pytest.raises(af.AdjustmentError):
            af.adjust_bar(JSW[0], factor)

    @pytest.mark.parametrize("bad,why", [
        (bar(D(2020, 1, 1), 0.0, 1.0, 0.0, 1.0), "non-positive"),
        (bar(D(2020, 1, 1), -1.0, 1.0, -2.0, 1.0), "non-positive"),
        (bar(D(2020, 1, 1), 10.0, 9.0, 8.0, 9.0), "ordering"),
        (bar(D(2020, 1, 1), 5.0, 6.0, 7.0, 6.0), "ordering"),
        (bar(D(2020, 1, 1), float("nan"), 2.0, 1.0, 1.5), "finite"),
    ])
    def test_invalid_raw_bars_are_identified(self, bad, why):
        assert why in af.raw_bar_problem(bad)
        with pytest.raises(af.AdjustmentError):
            af.adjust_bar(bad, F(1, 2))


class TestRawInputsAreNeverModified:
    def test_raw_bars_and_events_are_immutable_and_unchanged_by_adjustment(self):
        bars = list(JSW)
        before = copy.deepcopy(bars)
        events = [split("jsw", EX, 10, 1)]
        ev_before = copy.deepcopy(events)
        af.adjust_bars(bars, events, as_of=D(2017, 1, 5))
        view()
        assert bars == before and events == ev_before
        with pytest.raises(dataclasses.FrozenInstanceError):
            bars[0].close = 1.0
        with pytest.raises(dataclasses.FrozenInstanceError):
            events[0].ex_date = D(2000, 1, 1)
        assert [(b.open, b.high, b.low, b.close) for b in bars] == [(b.open, b.high, b.low, b.close) for b in before]

    def test_the_adjusted_bar_keeps_the_raw_bar_by_identity(self):
        out = af.adjust_bars(JSW, [split("jsw", EX, 10, 1)], as_of=D(2017, 1, 5))
        assert [o.raw for o in out] == JSW and all(o.raw is r for o, r in zip(out, JSW))

    def test_duplicate_dates_and_mixed_series_are_refused(self):
        with pytest.raises(af.AdjustmentError, match="one trade date"):
            af.adjust_bars([JSW[0], JSW[0]], [], as_of=D(2017, 1, 5))
        with pytest.raises(af.AdjustmentError, match="series"):
            af.adjust_bars([JSW[0], bar(D(2017, 1, 2), 1.0, 2.0, 1.0, 2.0, series="BL")], [], as_of=D(2017, 1, 5))

    def test_output_is_sorted_by_date_whatever_the_input_order(self):
        out = af.adjust_bars(list(reversed(JSW)), [], as_of=D(2017, 1, 5))
        assert [o.raw.trade_date for o in out] == sorted(b.trade_date for b in JSW)


class TestFactorEventsFromGateViews:
    def test_the_factor_comes_from_the_attested_c2_record(self):
        (e,) = af.factor_events_from_views([view_event("evt", EX, vsplit())])
        assert e.factor() == F(1, 10) and e.components[0].c2_record_id == "rec-split"

    def test_dividends_and_benign_meeting_fragments_have_no_factor(self):
        div = ComponentView("DIVIDEND", "MACHINE_PARSED", "Dividend", None, None)
        agm = ComponentView("OTHER", "UNVERIFIED", "Annual General Meeting", None, None)
        assert af.factor_events_from_views([view_event("e", EX, div, agm, status="UNVERIFIED")]) == ()

    def test_a_split_and_a_bonus_in_one_event_compound(self):
        (e,) = af.factor_events_from_views([view_event("e", D(2016, 9, 8), vbonus(1, 1), vsplit(10.0, 2.0))])
        assert e.factor() == F(1, 10) and len(e.components) == 2

    def test_a_supported_component_without_an_active_record_raises(self):
        with pytest.raises(af.AdjustmentError, match="no active C2 record"):
            af.factor_events_from_views([view_event("e", EX, vsplit(record=False))])

    def test_component_terms_that_differ_from_the_record_raise(self):
        with pytest.raises(af.AdjustmentError, match="do not equal"):
            af.factor_events_from_views([view_event("e", EX, vsplit(10.0, 1.0, rec_terms=(Decimal(10), Decimal(2))))])

    def test_a_record_of_another_kind_raises(self):
        c = ComponentView("SPLIT", "MANUALLY_VERIFIED", "x", (10.0, 1.0), RecordView("r", "BONUS", Decimal(10), Decimal(1), 1))
        with pytest.raises(af.AdjustmentError):
            af.factor_events_from_views([view_event("e", EX, c)])

    def test_malformed_terms_raise(self):
        c = ComponentView("SPLIT", "MANUALLY_VERIFIED", "x", None, RecordView("r", "SPLIT", Decimal(10), Decimal(1), 1))
        with pytest.raises(af.AdjustmentError):
            af.factor_events_from_views([view_event("e", EX, c)])


class TestTheViewObeysTheGate:
    def test_the_pilot_positive_case(self):
        v = view()
        assert v.status == af.ADJUSTED and v.refusal_reasons == () and v.eligibility.eligible
        assert [b.factor for b in v.bars] == [F(1, 10)] * 3 + [F(1)] * 2
        assert v.basis["cumulative_factor_at_requested_start"] == {"numerator": "1", "denominator": "10"}
        assert v.basis["events"][0]["components"][0]["c2_record_id"] == "rec-split"
        assert v.data_quality["complete"] and v.data_quality["missing_sessions"] == []

    @pytest.mark.parametrize("ev_views,register,coverage,code", [
        ([view_event("e", EX, vsplit(status="MACHINE_PARSED"), status="UNVERIFIED")], (), "COMPLETE", "EVENT_NOT_MANUALLY_VERIFIED"),
        ([view_event("e", EX, vsplit(record=False))], (), "COMPLETE", "NO_ACTIVE_C2_VERIFICATION"),
        ([view_event("e", EX, ComponentView("RIGHTS", "UNVERIFIED", "Rights 1:1", None, None), status="UNVERIFIED")], (), "COMPLETE", "UNSUPPORTED_ACTION_TYPE"),
        ([view_event("e", EX, ComponentView("DEMERGER", "UNVERIFIED", "Demerger", None, None), status="UNVERIFIED")], (), "COMPLETE", "UNSUPPORTED_ACTION_TYPE"),
        ([view_event("e", EX, vsplit(status="DISPUTED"))], (), "COMPLETE", "COMPONENT_NOT_MANUALLY_VERIFIED"),
        ([view_event("e", EX, ComponentView("SPLIT", "MANUALLY_VERIFIED", "x", (10.0, 0.0), RecordView("r", "SPLIT", Decimal(10), Decimal(1), 1)))], (), "COMPLETE", "TERMS_MISSING_OR_INVALID"),
        ([view_event("e", EX, vsplit(10.0, 1.0, rec_terms=(Decimal(10), Decimal(2))))], (), "COMPLETE", "C2_VERIFICATION_TERMS_MISMATCH"),
        ([view_event("e", EX, vsplit())], (RegisterItemView(EX, "SPLIT", "TERMS_NOT_MANUALLY_VERIFIED", "BLOCKS_ADJUSTMENT"),), "COMPLETE", "UNRESOLVED_REGISTER_ITEM"),
        ([view_event("e", EX, vsplit())], (), "UNKNOWN", "COVERAGE_NOT_COMPLETE"),
        ([view_event("e", EX, vsplit())], (), "CONFLICTED", "COVERAGE_NOT_COMPLETE"),
    ])
    def test_every_blocker_refuses_and_emits_no_adjusted_value(self, ev_views, register, coverage, code):
        v = view(ev_views=ev_views, register=register, coverage=coverage)
        assert v.status == af.REFUSED and af.ELIGIBILITY_BLOCKED in v.refusal_reasons
        assert v.bars == () and v.basis is None and v.basis_sha256 is None
        assert code in {b.code for b in v.eligibility.blockers} and not v.eligibility.eligible

    def test_a_blocker_outside_the_interval_does_not_refuse(self):
        v = view(as_of=D(2017, 1, 3), start=D(2016, 12, 30), ev_views=[view_event("e", EX, vsplit(record=False))],
                 bars=JSW[:3], sessions=JSW_SESSIONS[:3])
        assert v.status == af.ADJUSTED and all(b.factor == 1 for b in v.bars)

    def test_an_event_on_the_start_does_not_straddle_and_does_not_refuse(self):
        v = view(start=EX, ev_views=[view_event("e", EX, vsplit(record=False))], bars=JSW[3:], sessions=JSW_SESSIONS[3:])
        assert v.status == af.ADJUSTED and all(b.factor == 1 for b in v.bars)

    def test_the_gate_cannot_be_bypassed_by_passing_verified_events_alongside_a_blocked_eligibility(self):
        blocked = gate(D(2016, 12, 30), D(2017, 1, 5), [view_event("e", EX, vsplit(record=False))])
        v = af.build_adjusted_view(
            instrument_id="i", series="EQ", timeframe="1D", requested_start=D(2016, 12, 30), as_of=D(2017, 1, 5), raw_bars=JSW,
            session_dates=JSW_SESSIONS, eligibility=blocked, gate_events=[view_event("e", EX, vsplit())])
        assert v.status == af.REFUSED and v.bars == ()

    def test_an_eligibility_evaluated_for_another_interval_is_refused(self):
        other = gate(D(2017, 1, 1), D(2017, 1, 5), [])
        with pytest.raises(af.AdjustmentError, match="evaluated for"):
            af.build_adjusted_view(instrument_id="i", series="EQ", timeframe="1D", requested_start=D(2016, 12, 30), as_of=D(2017, 1, 5),
                                   raw_bars=JSW, session_dates=JSW_SESSIONS, eligibility=other, gate_events=[])

    def test_raw_bars_outside_the_request_are_refused_not_silently_dropped(self):
        with pytest.raises(af.AdjustmentError, match="outside the request"):
            view(bars=JSW + [bar(D(2017, 1, 6), 1.0, 2.0, 1.0, 2.0)])
        with pytest.raises(af.AdjustmentError, match="outside the request"):
            view(bars=JSW + [bar(D(2017, 1, 5), 1.0, 2.0, 1.0, 2.0, series="BL")])

    def test_start_after_as_of_is_refused_by_the_builder_itself(self):
        with pytest.raises(af.AdjustmentError, match="after as_of"):
            af.build_adjusted_view(instrument_id="i", series="EQ", timeframe="1D", requested_start=D(2017, 1, 6), as_of=D(2017, 1, 5),
                                   raw_bars=[], session_dates=[], eligibility=gate(D(2017, 1, 5), D(2017, 1, 5), []), gate_events=[])

    def test_a_safe_start_hint_only_when_every_blocker_is_dated(self):
        v = view(ev_views=[view_event("e", EX, vsplit(record=False))])
        assert v.safe_start_hint == EX
        assert view(coverage="UNKNOWN").safe_start_hint is None
        assert view().safe_start_hint is None

    def test_no_hint_when_an_undated_blocker_sits_beside_a_dated_barrier(self):
        v = view(ev_views=[view_event("e", EX, vsplit(record=False))], coverage="UNKNOWN")
        assert v.eligibility.barrier_dates == (EX,) and {b.code for b in v.eligibility.blockers} >= {"COVERAGE_NOT_COMPLETE", "NO_ACTIVE_C2_VERIFICATION"}
        assert v.safe_start_hint is None
        both = view(ev_views=[view_event("e", EX, vsplit(record=False))],
                    register=(RegisterItemView(None, "OTHER", "X", "BLOCKS_ADJUSTMENT"),))
        assert both.eligibility.barrier_dates == (EX,) and both.safe_start_hint is None


class TestDataQualityNeverLooksComplete:
    def test_no_bars_is_a_refusal(self):
        v = view(bars=[], sessions=[])
        assert v.status == af.REFUSED and v.refusal_reasons == (af.NO_BARS,) and not v.data_quality["complete"]

    def test_a_missing_session_is_refused_by_default_and_listed(self):
        bars = [b for b in JSW if b.trade_date != D(2017, 1, 3)]
        v = view(bars=bars)
        assert v.status == af.REFUSED and af.MISSING_BARS in v.refusal_reasons and v.bars == ()
        assert v.data_quality["missing_sessions"] == ["2017-01-03"] and v.data_quality["complete"] is False

    def test_allowing_gaps_returns_bars_but_never_as_complete(self):
        bars = [b for b in JSW if b.trade_date != D(2017, 1, 3)]
        v = view(bars=bars, allow=True)
        assert v.status == af.ADJUSTED_WITH_GAPS and len(v.bars) == 4 and v.data_quality["complete"] is False
        assert v.data_quality["missing_sessions"] == ["2017-01-03"]

    def test_a_missing_last_session_is_detected(self):
        v = view(bars=JSW[:-1])
        assert v.data_quality["missing_sessions"] == ["2017-01-05"] and v.status == af.REFUSED

    def test_sessions_before_the_first_bar_or_after_as_of_are_not_gaps(self):
        v = view(bars=JSW[1:], start=D(2017, 1, 2), sessions=[D(2016, 12, 30)] + JSW_SESSIONS[1:] + [D(2017, 1, 6)])
        assert v.status == af.ADJUSTED and v.data_quality["complete"]

    def test_gaps_do_not_change_any_factor(self):
        bars = [b for b in JSW if b.trade_date != D(2017, 1, 3)]
        full = {b.raw.trade_date: b.factor for b in view().bars}
        assert {b.raw.trade_date: b.factor for b in view(bars=bars, allow=True).bars} == {d: f for d, f in full.items() if d != D(2017, 1, 3)}

    def test_an_invalid_raw_bar_is_a_refusal(self):
        bad = JSW[:2] + [dataclasses.replace(JSW[2], high=1.0)] + JSW[3:]
        v = view(bars=bad)
        assert v.status == af.REFUSED and af.INVALID_RAW_BARS in v.refusal_reasons
        assert v.data_quality["invalid_raw_bars"][0][0] == "2017-01-03"

    def test_the_calendar_basis_is_stated_as_an_observed_proxy(self):
        assert "NOT_AN_EXCHANGE_CALENDAR" in view().data_quality["calendar_basis"]


class TestReproducibility:
    def test_the_basis_states_everything_needed_to_reproduce_the_view(self):
        b = view().basis
        for key in ("version", "instrument_id", "series", "timeframe", "requested_start", "as_of", "interval_semantics", "price_basis_raw",
                    "price_basis_adjusted", "dividends_adjusted", "point_in_time", "eligibility", "events", "cumulative_factor_at_requested_start"):
            assert key in b
        assert b["version"] == af.ADJUSTMENT_BASIS_VERSION and b["dividends_adjusted"] is False and b["volume_included"] is False
        assert (b["instrument_id"], b["series"], b["timeframe"]) == ("inst-1", "EQ", "1D")
        assert (b["requested_start"], b["as_of"]) == ("2016-12-30", "2017-01-05")
        assert (b["eligibility"]["start"], b["eligibility"]["end"]) == ("2016-12-30", "2017-01-05")
        assert b["price_basis_raw"] == af.PRICE_BASIS_RAW and b["price_basis_adjusted"] == af.PRICE_BASIS_ADJUSTED
        assert "NOT_ESTABLISHED" in b["point_in_time"]
        assert b["eligibility"]["governing_assessment_id"] == "A1" and b["eligibility"]["coverage_status"] == "COMPLETE"
        assert b["events"][0]["components"][0]["term_a"] == "10" and b["events"][0]["components"][0]["term_b"] == "1"

    def test_each_bar_carries_source_raw_factor_and_adjusted_values(self):
        d = view().as_dict()["bars"][2]
        assert d["trade_date"] == "2017-01-03" and d["source_bar_id"] == "bar-2017-01-03" and d["series"] == "EQ" and d["timeframe"] == "1D"
        assert d["raw"]["close"] == 1644.55 and d["factor"] == {"numerator": "1", "denominator": "10"}
        assert d["adjusted_exact"]["close"] == "32891/200" and d["adjusted"]["close"] == 164.455

    def test_the_view_is_json_serialisable_and_its_hash_is_recomputable(self):
        v = view()
        d = v.as_dict()
        assert json.loads(json.dumps(d))["view_sha256"] == v.view_sha256
        assert af.sha256_of(v.content()) == v.view_sha256 and af.sha256_of(v.basis) == v.basis_sha256

    def test_the_view_hash_moves_with_any_input_that_could_change_a_value(self):
        base = view()
        assert view(bars=[dataclasses.replace(JSW[0], high=1633.0)] + JSW[1:]).view_sha256 != base.view_sha256
        assert view(ev_views=[view_event("evt-jsw", EX, vsplit(10.0, 2.0))]).view_sha256 != base.view_sha256
        assert view(as_of=D(2017, 1, 4), bars=JSW[:4], sessions=JSW_SESSIONS[:4]).view_sha256 != base.view_sha256
        assert view(ev_views=[view_event("evt-other", EX, vsplit())]).view_sha256 != base.view_sha256

    def test_the_basis_hash_ignores_raw_prices_but_not_terms_or_as_of(self):
        base = view()
        assert view(bars=[dataclasses.replace(JSW[0], high=1633.0)] + JSW[1:]).basis_sha256 == base.basis_sha256
        assert view(ev_views=[view_event("evt-jsw", EX, vsplit(10.0, 2.0))]).basis_sha256 != base.basis_sha256
        assert view(as_of=D(2017, 1, 4), bars=JSW[:4], sessions=JSW_SESSIONS[:4]).basis_sha256 != base.basis_sha256

    def test_a_refused_view_is_hashed_too(self):
        v = view(coverage="UNKNOWN")
        assert v.status == af.REFUSED and len(v.view_sha256) == 64 and v.basis_sha256 is None

    def test_events_outside_the_interval_are_not_listed_in_the_basis(self):
        extra = view_event("evt-late", D(2020, 1, 1), vsplit(2.0, 1.0))
        v = view(ev_views=[view_event("evt-jsw", EX, vsplit()), extra])
        assert [e["event_id"] for e in v.basis["events"]] == ["evt-jsw"]

    def test_a_reverse_split_is_representable_exactly(self):
        v = view(ev_views=[view_event("evt-jsw", EX, vsplit(1.0, 10.0))])
        assert v.status == af.ADJUSTED and v.bars[0].factor == F(10)


class TestBonusFormulaEquivalence:

    @staticmethod
    def mutant(a, b):
        return F(1) / (1 + F(a, b))

    def test_identical_over_the_whole_small_domain(self):
        for a in range(1, 81):
            for b in range(1, 81):
                got = af.component_factor("BONUS", a, b)
                assert got == self.mutant(a, b) == F(b, a + b)
                assert (got.numerator, got.denominator) == (self.mutant(a, b).numerator, self.mutant(a, b).denominator)

    def test_identical_at_the_edges_of_the_accepted_domain(self):
        top = 10 ** 14
        for a, b in [(1, 1), (1, top), (top, 1), (top, top), (top - 1, top), (1, top - 1), (2, 3), (top, top - 1)]:
            assert af.component_factor("BONUS", a, b) == self.mutant(a, b)

    def test_identical_for_log_uniform_random_terms_up_to_the_bound(self):
        rng = random.Random(SEED)
        for _ in range(5000):
            a, b = (max(1, int(10 ** rng.uniform(0, 14))) for _ in range(2))
            assert af.component_factor("BONUS", a, b) == self.mutant(a, b)

    @pytest.mark.parametrize("a,b", [(4, 1), (1, 1), (3, 2)])
    def test_identical_for_every_numeric_form_the_contract_accepts(self, a, b):
        forms = [lambda v: v, lambda v: Decimal(v), lambda v: Decimal(f"{v}.000000"), lambda v: Decimal(f"{v}E+0"), lambda v: float(v)]
        for fa in forms:
            for fb in forms:
                assert af.component_factor("BONUS", fa(a), fb(b)) == self.mutant(a, b)

    def test_the_share_count_meaning_holds_for_both_forms(self):
        for a, b in [(1, 1), (4, 1), (3, 2), (1, 7)]:
            assert af.component_factor("BONUS", a, b) * F(a + b, b) == 1 == self.mutant(a, b) * F(a + b, b)

    def test_the_only_inputs_where_the_forms_could_differ_are_refused_before_the_formula_runs(self):
        for a, b in [(1, 0), (0, 0), (0, 1), (10 ** 14 + 1, 1), (1, 10 ** 14 + 1), (Decimal("1.5"), 1), (1, Decimal("0.5"))]:
            with pytest.raises(InvalidTerms):
                af.component_factor("BONUS", a, b)


class TestOhlcOrderingInvariant:

    @staticmethod
    def _ordered(r):
        return (r.low_exact <= min(r.open_exact, r.close_exact) and max(r.open_exact, r.close_exact) <= r.high_exact
                and r.low <= min(r.open, r.close) and max(r.open, r.close) <= r.high)

    def test_link_1_exact_scaling_by_a_positive_rational_is_monotone(self):
        rng = random.Random(SEED)
        for _ in range(3000):
            x, y = sorted(F(rng.randint(1, 10 ** 9), rng.randint(1, 10 ** 4)) for _ in range(2))
            f = F(rng.randint(1, 10 ** 14), rng.randint(1, 10 ** 14))
            assert x <= y and x * f <= y * f

    def test_link_2_float_of_a_fraction_is_monotone_including_at_rounding_ties(self):
        rng = random.Random(SEED)
        for _ in range(4000):
            x, y = sorted(F(rng.randint(1, 10 ** 15), rng.randint(1, 10 ** 15)) for _ in range(2))
            assert float(x) <= float(y)
        eps = F(1, 10 ** 30)
        for base in (1.0, 0.1, 163.05, 1644.55, 9331.0, 1e-9, 1e12):
            lo, hi = F(base), F(math.nextafter(base, math.inf))
            mid = (lo + hi) / 2
            assert float(lo) <= float(mid - eps) <= float(mid) <= float(mid + eps) <= float(hi)

    def test_link_3_the_two_guards_are_the_only_way_past_and_they_refuse_every_violating_input(self):
        rng = random.Random(SEED)
        d = D(2020, 1, 1)
        for _ in range(3000):
            vals = [round(rng.uniform(-5, 500), rng.choice([0, 2])) for _ in range(4)]
            o, h, l, c = vals
            b = af.RawBar("x", d, "EQ", "1D", o, h, l, c)
            valid = min(vals) > 0 and l <= o <= h and l <= c <= h
            assert (af.raw_bar_problem(b) is None) == valid
            if valid:
                assert self._ordered(af.adjust_bar(b, F(rng.randint(1, 60), rng.randint(1, 60))))
            else:
                with pytest.raises(af.AdjustmentError):
                    af.adjust_bar(b, F(1, 2))
        for f in (F(0), F(-1), F(-1, 10 ** 14)):
            with pytest.raises(af.AdjustmentError):
                af.adjust_bar(af.RawBar("x", d, "EQ", "1D", 1.0, 2.0, 1.0, 2.0), f)

    def test_the_invariant_holds_at_extreme_factors_flat_bars_and_float_resolution(self):
        d = D(2020, 1, 1)
        adjacent = math.nextafter(163.05, math.inf)
        bars = [
            af.RawBar("a", d, "EQ", "1D", 100.0, 100.0, 100.0, 100.0),
            af.RawBar("b", d, "EQ", "1D", 100.0, 105.0, 100.0, 105.0),
            af.RawBar("c", d, "EQ", "1D", 105.0, 105.0, 100.0, 100.0),
            af.RawBar("d", d, "EQ", "1D", 163.05, adjacent, 163.05, adjacent),
            af.RawBar("e", d, "EQ", "1D", 1e-9, 2e-9, 1e-9, 2e-9),
            af.RawBar("f", d, "EQ", "1D", 1e12, 2e12, 1e12, 2e12),
            af.RawBar("g", d, "EQ", "1D", 0.01, 0.02, 0.01, 0.02),
        ]
        factors_ = [F(1, 10 ** 14), F(1, 3), F(1), F(3, 2), F(10), F(10 ** 14), F(1, 10 ** 8), F(7, 1000)]
        for b in bars:
            for f in factors_:
                assert self._ordered(af.adjust_bar(b, f))

    def test_the_invariant_holds_end_to_end_for_random_bars_under_a_compound_of_events(self):
        rng = random.Random(SEED)
        events = [split("s", D(2016, 1, 1), 10, 2), bonus("b", D(2016, 1, 1), 1, 1), split("t", D(2020, 6, 1), 5, 1), bonus("u", D(2022, 3, 3), 3, 2)]
        bars = []
        for i in range(600):
            lo = round(rng.uniform(1, 3000), 2)
            hi = round(lo + rng.uniform(0, 300), 2)
            bars.append(bar(D(2015, 1, 1) + timedelta(i * 7), round(rng.uniform(lo, hi), 2), hi, lo, round(rng.uniform(lo, hi), 2)))
        assert bars[-1].trade_date < D(2026, 9, 11)
        out = af.adjust_bars(bars, events, as_of=D(2026, 9, 11))
        assert len(out) == 600 and {r.factor for r in out} == {F(1, 125), F(2, 25), F(2, 5), F(1)}
        assert all(self._ordered(r) for r in out)

    def test_the_ordering_check_used_here_can_fail(self):
        r = af.adjust_bar(af.RawBar("x", D(2020, 1, 1), "EQ", "1D", 10.0, 12.0, 9.0, 11.0), F(1, 2))
        assert self._ordered(r)
        assert not self._ordered(dataclasses.replace(r, high_exact=r.low_exact, high=r.low))


class TestFirstSessionBoundEquivalence:

    @staticmethod
    def mutant(bar_dates, session_dates, as_of):
        if not bar_dates:
            return ()
        first, have = min(bar_dates), set(bar_dates)
        return tuple(sorted(s for s in set(session_dates) if first < s <= as_of and s not in have))

    def test_exhaustive_over_every_bar_set_session_set_and_as_of_on_a_seven_day_universe(self):
        days = [D(2020, 1, 1) + timedelta(i) for i in range(7)]
        n = 0
        for bmask in range(1, 128):
            bars = [d for i, d in enumerate(days) if bmask >> i & 1]
            for smask in range(128):
                sess = [d for i, d in enumerate(days) if smask >> i & 1]
                for as_of in days:
                    if any(b > as_of for b in bars):
                        continue
                    got = af.missing_sessions(bars, sess, as_of=as_of)
                    assert got == self.mutant(bars, sess, as_of)
                    assert min(bars) not in got
                    assert all(min(bars) < s <= as_of and s not in bars for s in got)
                    n += 1
        assert n == 247 * 128 == 31_616

    def test_random_larger_universes_with_duplicates_and_unsorted_input(self):
        rng = random.Random(SEED)
        for _ in range(500):
            pool = [D(2015, 1, 1) + timedelta(i) for i in range(60)]
            bars = [rng.choice(pool) for _ in range(rng.randint(1, 30))]
            sess = [rng.choice(pool) for _ in range(rng.randint(0, 60))]
            as_of = max(max(bars), rng.choice(pool))
            got = af.missing_sessions(bars, sess, as_of=as_of)
            assert got == self.mutant(bars, sess, as_of) and min(bars) not in got

    def test_through_the_view_the_first_bar_date_is_never_reported_missing_whatever_sessions_precede_it(self):
        sessions = [D(2016, 12, 29)] + JSW_SESSIONS
        v = view(sessions=sessions, bars=JSW[1:], start=D(2016, 12, 30))
        assert v.data_quality["first_bar_date"] == "2017-01-02"
        assert v.data_quality["missing_sessions"] == [] and v.status == af.ADJUSTED

    def test_the_expected_session_count_does_include_the_first_session(self):
        v = view()
        assert v.data_quality["expected_session_count"] == 5 == v.data_quality["raw_bar_count"]
