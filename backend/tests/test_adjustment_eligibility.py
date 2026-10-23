from __future__ import annotations

import inspect
import itertools
import json
import math
import random

import pytest

from backend.app.research import adjustment_eligibility as ae
from backend.app.research.adjustment_eligibility import (
    ComponentView,
    EventView,
    RecordView,
    RegisterItemView,
    evaluate_adjustment_eligibility,
    in_interval,
)
from backend.app.research.exact_terms import try_normalize
from datetime import date as D

V, MP, UV, DISP = "MANUALLY_VERIFIED", "MACHINE_PARSED", "UNVERIFIED", "DISPUTED"
FULL = (D(2015, 1, 1), D(2026, 9, 11))


_n = itertools.count(1)


def C(ctype, status=V, terms=None, frag=None, record=True):
    rec = None
    if record and status == V and ctype in ("SPLIT", "BONUS"):
        t = try_normalize(ctype, *terms)[0] if terms is not None and len(terms) == 2 else None
        rec = RecordView(f"rec-{next(_n)}", ctype, *(t.term_a, t.term_b) if t else ((1, 1) if ctype == "BONUS" else (2, 1)))
    return ComponentView(ctype, status, frag or f"{ctype} text", terms, rec)


def E(ex, *comps, status=V, eid=None):
    return EventView(eid or f"ev{next(_n)}", ex, status, tuple(comps))


def split(ex, terms=(10.0, 2.0), status=V, comp_status=V):
    return E(ex, C("SPLIT", comp_status, terms), status=status)


def bonus(ex, terms=(1, 1), status=V, comp_status=V):
    return E(ex, C("BONUS", comp_status, terms), status=status)


def agm_div(ex):
    return E(ex, C("OTHER", UV, None, "Annual General Meeting"), C("DIVIDEND", MP, None), status=UV)


def R(ex, cls, code="TERMS_NOT_MANUALLY_VERIFIED", sev="BLOCKS_ADJUSTMENT"):
    return RegisterItemView(ex, cls, code, sev, "assess-1")


def ev(start, end, events=(), register=(), coverage="COMPLETE"):
    return evaluate_adjustment_eligibility(
        start=start, end=end, coverage_status=coverage,
        governing_assessment_id="assess-1" if coverage != "UNKNOWN" else None,
        events=list(events), register_items=list(register),
    )


def codes(r):
    return sorted({b.code for b in r.blockers})


def kotakbank():
    events = [
        agm_div(D(2015, 6, 18)), bonus(D(2015, 7, 8), (1, 1), status=UV, comp_status=MP),
        agm_div(D(2017, 7, 12)), agm_div(D(2018, 7, 11)), agm_div(D(2019, 7, 12)),
        split(D(2026, 1, 14), (5.0, 1.0)),
    ]
    return events, [R(D(2015, 7, 8), "BONUS")]


def hdfcbank():
    return [split(D(2019, 9, 19), (2.0, 1.0)), bonus(D(2025, 8, 26), (1, 1), status=UV, comp_status=MP)], \
           [R(D(2025, 8, 26), "BONUS")]


def bajfinance():
    e2016 = E(D(2016, 9, 8), C("BONUS", V, (1, 1)), C("SPLIT", V, (10.0, 2.0)))
    events = [
        e2016, agm_div(D(2017, 7, 6)), agm_div(D(2021, 7, 8)),
        bonus(D(2025, 6, 16), (4, 1), status=UV, comp_status=MP),
        split(D(2025, 6, 16), (2.0, 1.0), status=UV, comp_status=MP),
    ]
    return events, [R(D(2025, 6, 16), "BONUS"), R(D(2025, 6, 16), "SPLIT"),
                    R(D(2016, 3, 16), "REVISION", "REVISED_NON_PRICE_AFFECTING", "INFO"),
                    R(D(2020, 3, 23), "DIAGNOSTIC", "WATCH_JUMP", "INFO")]


def jswsteel():
    return [split(D(2017, 1, 4), (10.0, 1.0)), agm_div(D(2017, 6, 12)), agm_div(D(2019, 7, 8)),
            agm_div(D(2021, 7, 5)), agm_div(D(2022, 7, 4)), agm_div(D(2023, 7, 11)), agm_div(D(2024, 7, 9))], \
           [R(D(2020, 3, 23), "DIAGNOSTIC", "WATCH_JUMP", "INFO")]


def reliance():
    events = [
        bonus(D(2017, 9, 7), (1, 1), status=UV, comp_status=MP), agm_div(D(2019, 8, 2)),
        E(D(2020, 5, 13), C("RIGHTS", UV, None), status=UV),
        E(D(2023, 7, 20), C("DEMERGER", UV, None), status=UV),
        bonus(D(2024, 10, 28), (1, 1), status=UV, comp_status=MP),
    ]
    return events, [R(D(2017, 9, 7), "BONUS"), R(D(2020, 5, 13), "RIGHTS", "UNSUPPORTED_ACTION_TYPE"),
                    R(D(2023, 7, 20), "DEMERGER", "UNSUPPORTED_ACTION_TYPE"), R(D(2024, 10, 28), "BONUS"),
                    R(D(2016, 3, 17), "REVISION", "REVISED_NON_PRICE_AFFECTING", "INFO")]


class TestIntervalSemanticsStartExcludedEndIncluded:
    def test_the_one_definition_is_start_lt_ex_le_end(self):
        s, e = D(2020, 1, 1), D(2020, 1, 10)
        assert not in_interval(D(2020, 1, 1), s, e)
        assert in_interval(D(2020, 1, 2), s, e)
        assert in_interval(D(2020, 1, 10), s, e)
        assert not in_interval(D(2020, 1, 11), s, e)
        assert not in_interval(D(2019, 12, 31), s, e)

    BLOCKER = D(2020, 3, 10)

    @pytest.mark.parametrize("start,end,eligible", [
        (D(2020, 1, 1), D(2020, 3, 9), True),
        (D(2020, 1, 1), D(2020, 3, 10), False),
        (D(2020, 1, 1), D(2020, 3, 11), False),
        (D(2020, 3, 9), D(2020, 3, 10), False),
        (D(2020, 3, 10), D(2020, 6, 1), True),
        (D(2020, 3, 11), D(2020, 6, 1), True),
        (D(2020, 3, 10), D(2020, 3, 10), True),
    ])
    def test_a_blocking_event_only_affects_intervals_containing_it_in_the_half_open_sense(self, start, end, eligible):
        r = ev(start, end, [bonus(self.BLOCKER, status=UV, comp_status=MP)], [R(self.BLOCKER, "BONUS")])
        assert r.eligible is eligible
        assert (self.BLOCKER in r.barrier_dates) is (not eligible)

    def test_a_saturday_ex_date_lands_in_the_interval_spanning_the_weekend(self):
        sat = D(2020, 1, 4)
        blocker = [bonus(sat, status=UV, comp_status=MP)]
        assert not ev(D(2020, 1, 3), D(2020, 1, 6), blocker).eligible
        assert ev(D(2020, 1, 6), D(2020, 2, 1), blocker).eligible
        assert ev(D(2019, 12, 1), D(2020, 1, 3), blocker).eligible
        assert ev(D(2020, 1, 4), D(2020, 1, 6), blocker).eligible

    def test_a_single_day_interval_has_no_affected_events(self):
        d = D(2020, 3, 10)
        r = ev(d, d, [bonus(d, status=UV, comp_status=MP)], [R(d, "BONUS")])
        assert r.eligible and r.events_considered == 0 and r.barrier_dates == ()

    def test_a_single_day_interval_is_still_blocked_by_coverage_and_by_a_global_register_row(self):
        d = D(2020, 3, 10)
        assert not ev(d, d, coverage="UNKNOWN").eligible
        assert not ev(d, d, register=[R(None, "UNCLASSIFIED", "UNPARSEABLE_ROW", "BLOCKS_COMPLETE")]).eligible

    def test_start_after_end_is_rejected_not_silently_empty(self):
        with pytest.raises(ValueError):
            ev(D(2020, 3, 11), D(2020, 3, 10))

    def test_an_interval_with_no_events_at_all_is_eligible_when_coverage_is_complete(self):
        r = ev(*FULL)
        assert r.eligible and r.blockers == () and r.barrier_dates == () and r.events_considered == 0

    def test_register_rows_use_the_same_half_open_rule(self):
        item = [R(D(2020, 3, 10), "BONUS")]
        assert ev(D(2020, 3, 10), D(2020, 4, 1), register=item).eligible
        assert not ev(D(2020, 1, 1), D(2020, 3, 10), register=item).eligible


class TestKotakbank2015_07_08_BoundaryCannotBeAmbiguous:

    @pytest.mark.parametrize("start,end,eligible,why", [
        (D(2015, 1, 1), D(2015, 7, 7), True, "wholly before the blocker: ends the day before"),
        (D(2015, 1, 1), D(2015, 7, 8), False, "ends ON the ex-date: the ex-date bar is included"),
        (D(2015, 7, 7), D(2015, 7, 8), False, "the day before -> the ex-date"),
        (D(2015, 7, 7), D(2015, 7, 7), True, "single day before"),
        (D(2015, 7, 8), D(2015, 7, 8), True, "single day ON the ex-date: (d, d] is empty"),
        (D(2015, 7, 8), D(2026, 9, 11), True, "starts ON the ex-date: excluded; includes the VERIFIED 2026 split"),
        (D(2015, 7, 9), D(2026, 9, 11), True, "starts after the blocker"),
        (D(2015, 1, 1), D(2026, 9, 11), False, "the whole window straddles it"),
        (D(2015, 7, 7), D(2026, 9, 11), False, "starts one day before it"),
    ])
    def test_boundary_matrix(self, start, end, eligible, why):
        events, reg = kotakbank()
        r = ev(start, end, events, reg)
        assert r.eligible is eligible, why
        if not eligible:
            assert r.barrier_dates == (D(2015, 7, 8),)

    def test_the_blocker_is_reported_from_both_the_live_event_and_the_register(self):
        events, reg = kotakbank()
        r = ev(*FULL, events, reg)
        assert {(b.source, b.code) for b in r.blockers} == {
            ("EVENT", ae.EVENT_NOT_MANUALLY_VERIFIED), ("EVENT", ae.COMPONENT_NOT_MANUALLY_VERIFIED),
            ("EVENT", ae.NO_ACTIVE_C2_VERIFICATION), ("REGISTER", ae.UNRESOLVED_REGISTER_ITEM),
        }
        assert all(b.ex_date == D(2015, 7, 8) for b in r.blockers)

    def test_the_agm_only_other_components_on_kotakbank_do_not_block(self):
        events, reg = kotakbank()
        assert ev(D(2015, 7, 8), D(2026, 9, 11), events, reg).eligible

    def test_the_verified_2026_split_is_understood_and_adjustable(self):
        events, _ = kotakbank()
        assert ev(D(2025, 1, 1), D(2026, 9, 11), events, []).eligible


class TestRequiredAdversarialCases:
    def test_1_complete_plus_unresolved_event_is_ineligible(self):
        r = ev(*FULL, [], [R(D(2020, 5, 13), "RIGHTS", "UNSUPPORTED_ACTION_TYPE")])
        assert not r.eligible and codes(r) == [ae.UNRESOLVED_REGISTER_ITEM]

    def test_1b_a_blocks_complete_register_row_also_blocks(self):
        assert not ev(*FULL, [], [R(D(2020, 5, 13), "BONUS", "TERMS_CONFLICT", "BLOCKS_COMPLETE")]).eligible

    def test_1c_info_rows_never_block(self):
        assert ev(*FULL, [], [R(D(2020, 5, 13), "REVISION", "REVISED_NON_PRICE_AFFECTING", "INFO"),
                              R(None, "DIAGNOSTIC", "WATCH_JUMP", "INFO")]).eligible

    def test_2_complete_plus_unverified_split_is_ineligible(self):
        for status, comp_status in [(UV, UV), (UV, MP), (MP, MP), (V, MP), (MP, V), (DISP, V), (V, DISP)]:
            r = ev(*FULL, [split(D(2020, 1, 1), status=status, comp_status=comp_status)])
            assert not r.eligible, (status, comp_status)

    def test_3_complete_plus_verified_split_plus_verified_bonus_is_eligible(self):
        r = ev(*FULL, [split(D(2016, 3, 1)), bonus(D(2018, 3, 1))])
        assert r.eligible and r.events_considered == 2 and r.blockers == ()

    @pytest.mark.parametrize("ctype", ["RIGHTS", "DEMERGER", "MERGER"])
    def test_4_5_unsupported_types_block_the_affected_interval_even_when_fully_verified(self, ctype):
        d = D(2020, 5, 13)
        event = E(d, C(ctype, V, None), status=V)
        assert not ev(*FULL, [event]).eligible
        assert not ev(D(2020, 1, 1), D(2020, 5, 13), [event]).eligible
        assert not ev(D(2020, 5, 12), D(2021, 1, 1), [event]).eligible
        assert ev(D(2020, 5, 13), D(2021, 1, 1), [event]).eligible
        assert ev(D(2015, 1, 1), D(2020, 5, 12), [event]).eligible
        (b,) = ev(*FULL, [event]).blockers
        assert (b.code, b.action_class, b.ex_date) == (ae.UNSUPPORTED_ACTION_TYPE, ctype, d)

    def test_a_future_unrecognised_price_affecting_type_is_unsupported_by_default(self):
        r = ev(*FULL, [E(D(2020, 1, 1), C("SPINOFF_V2", V, (1, 1)))])
        assert not r.eligible and codes(r) == [ae.UNSUPPORTED_ACTION_TYPE]

    def test_the_supported_set_is_exactly_split_and_bonus(self):
        assert ae.SUPPORTED_ADJUSTMENT_TYPES == {"SPLIT", "BONUS"}

    def test_the_register_modules_unsupported_set_is_derived_from_it(self):
        from backend.app.research.corporate_action_register import UNSUPPORTED_FOR_ADJUSTMENT
        from backend.app.research.corporate_action_reconciliation import PRICE_AFFECTING

        assert UNSUPPORTED_FOR_ADJUSTMENT == PRICE_AFFECTING - ae.SUPPORTED_ADJUSTMENT_TYPES == {"RIGHTS", "DEMERGER", "MERGER"}


class TestBajfinanceSameDateSplitAndBonus:
    def test_the_2016_compound_event_with_both_components_verified_is_eligible(self):
        events, _ = bajfinance()
        assert ev(D(2015, 1, 1), D(2025, 6, 13), events, []).eligible

    def test_a_compound_event_with_one_unverified_component_is_ineligible(self):
        e = E(D(2016, 9, 8), C("BONUS", V, (1, 1)), C("SPLIT", MP, (10.0, 2.0)))
        r = ev(*FULL, [e])
        assert not r.eligible and codes(r) == [ae.COMPONENT_NOT_MANUALLY_VERIFIED, ae.NO_ACTIVE_C2_VERIFICATION]

    def test_2025_pair_both_unverified_blocks_and_reports_both_classes_on_the_date(self):
        events, reg = bajfinance()
        r = ev(*FULL, events, reg)
        assert not r.eligible and r.barrier_dates == (D(2025, 6, 16),)
        assert {b.action_class for b in r.blockers} == {"BONUS", "SPLIT"}

    def test_only_the_split_verified_is_still_ineligible_because_of_the_same_date_bonus(self):
        events = [split(D(2025, 6, 16), (2.0, 1.0)), bonus(D(2025, 6, 16), (4, 1), status=UV, comp_status=MP)]
        r = ev(*FULL, events)
        assert not r.eligible and {b.action_class for b in r.blockers} == {"BONUS"}

    def test_only_the_bonus_verified_is_still_ineligible_because_of_the_same_date_split(self):
        events = [bonus(D(2025, 6, 16), (4, 1)), split(D(2025, 6, 16), (2.0, 1.0), status=UV, comp_status=MP)]
        r = ev(*FULL, events)
        assert not r.eligible and {b.action_class for b in r.blockers} == {"SPLIT"}

    def test_both_verified_but_the_historical_register_still_blocks_until_a_new_assessment(self):
        events = [bonus(D(2025, 6, 16), (4, 1)), split(D(2025, 6, 16), (2.0, 1.0))]
        stale = [R(D(2025, 6, 16), "BONUS"), R(D(2025, 6, 16), "SPLIT")]
        r = ev(*FULL, events, stale)
        assert not r.eligible and {(b.source, b.code) for b in r.blockers} == {("REGISTER", ae.UNRESOLVED_REGISTER_ITEM)}
        assert ev(*FULL, events, []).eligible

    def test_intervals_either_side_of_the_2025_pair(self):
        events, reg = bajfinance()
        assert ev(D(2015, 1, 1), D(2025, 6, 13), events, reg).eligible
        assert ev(D(2025, 6, 16), D(2026, 9, 11), events, reg).eligible
        assert not ev(D(2025, 6, 13), D(2025, 6, 16), events, reg).eligible


class TestRelianceHistoricalBarriers:
    EPOCHS = [
        (D(2015, 1, 1), D(2017, 9, 6)), (D(2017, 9, 7), D(2020, 5, 12)), (D(2020, 5, 13), D(2023, 7, 19)),
        (D(2023, 7, 20), D(2024, 10, 27)), (D(2024, 10, 28), D(2026, 9, 11)),
    ]

    def test_the_whole_window_is_ineligible_with_exactly_four_barriers(self):
        events, reg = reliance()
        r = ev(*FULL, events, reg)
        assert not r.eligible
        assert r.barrier_dates == (D(2017, 9, 7), D(2020, 5, 13), D(2023, 7, 20), D(2024, 10, 28))

    @pytest.mark.parametrize("start,end", EPOCHS)
    def test_each_epoch_between_barriers_is_eligible_on_its_own(self, start, end):
        events, reg = reliance()
        r = ev(start, end, events, reg)
        assert r.eligible, r.blockers

    @pytest.mark.parametrize("a,b", list(itertools.combinations(range(5), 2)))
    def test_any_interval_spanning_two_epochs_is_ineligible(self, a, b):
        events, reg = reliance()
        assert not ev(self.EPOCHS[a][0], self.EPOCHS[b][1], events, reg).eligible

    def test_each_barrier_is_one_day_wide_at_the_seam(self):
        events, reg = reliance()
        for barrier in (D(2017, 9, 7), D(2020, 5, 13), D(2023, 7, 20), D(2024, 10, 28)):
            before = barrier.fromordinal(barrier.toordinal() - 1)
            assert not ev(before, barrier, events, reg).eligible
            assert ev(barrier, D(2026, 9, 11), [e for e in events if e.ex_date <= barrier], [r for r in reg if r.ex_date <= barrier]).eligible

    def test_rights_and_demerger_stay_barriers_even_if_both_bonuses_are_verified_and_re_registered(self):
        events, _ = reliance()
        fixed = [bonus(D(2017, 9, 7)) if e.ex_date == D(2017, 9, 7) else bonus(D(2024, 10, 28)) if e.ex_date == D(2024, 10, 28) else e
                 for e in events]
        new_reg = [R(D(2020, 5, 13), "RIGHTS", "UNSUPPORTED_ACTION_TYPE"), R(D(2023, 7, 20), "DEMERGER", "UNSUPPORTED_ACTION_TYPE")]
        r = ev(*FULL, fixed, new_reg)
        assert not r.eligible and r.barrier_dates == (D(2020, 5, 13), D(2023, 7, 20))
        assert ev(D(2015, 1, 1), D(2020, 5, 12), fixed, new_reg).eligible
        assert ev(D(2020, 5, 13), D(2023, 7, 19), fixed, new_reg).eligible
        assert ev(D(2023, 7, 20), D(2026, 9, 11), fixed, new_reg).eligible
        assert not ev(D(2015, 1, 1), D(2023, 7, 19), fixed, new_reg).eligible

    def test_reliance_has_no_verified_price_affecting_event_so_no_interval_containing_one_is_eligible(self):
        events, reg = reliance()
        for e in events:
            if any(c.component_type != "OTHER" and c.component_type != "DIVIDEND" for c in e.components):
                before = D.fromordinal(e.ex_date.toordinal() - 1)
                assert not ev(before, e.ex_date, events, reg).eligible


class TestOtherInstrumentsRealShapes:
    def test_jswsteel_whole_window_is_eligible_because_agm_other_is_exempt_and_the_split_is_verified(self):
        events, reg = jswsteel()
        r = ev(*FULL, events, reg)
        assert r.eligible and r.events_considered == 7

    def test_hdfcbank_only_the_2025_bonus_is_a_barrier(self):
        events, reg = hdfcbank()
        r = ev(*FULL, events, reg)
        assert not r.eligible and r.barrier_dates == (D(2025, 8, 26),)
        assert ev(D(2015, 1, 1), D(2025, 8, 25), events, reg).eligible
        assert ev(D(2025, 8, 26), D(2026, 9, 11), events, reg).eligible

    def test_kotakbank_bajfinance_hdfcbank_reliance_full_windows_are_all_ineligible_only_jswsteel_is_eligible(self):
        verdicts = {n: ev(*FULL, *f()).eligible for n, f in
                    [("HDFCBANK", hdfcbank), ("KOTAKBANK", kotakbank), ("BAJFINANCE", bajfinance),
                     ("JSWSTEEL", jswsteel), ("RELIANCE", reliance)]}
        assert verdicts == {"HDFCBANK": False, "KOTAKBANK": False, "BAJFINANCE": False, "JSWSTEEL": True, "RELIANCE": False}


class TestMultipleHistoricalEpochs:
    def _world(self):
        events = [split(D(2016, 3, 1)), bonus(D(2018, 3, 1)), bonus(D(2020, 3, 1), status=UV, comp_status=MP),
                  split(D(2022, 3, 1), (5.0, 1.0)), E(D(2024, 3, 1), C("RIGHTS", V)), bonus(D(2025, 3, 1), (2, 1))]
        reg = [R(D(2020, 3, 1), "BONUS"), R(D(2024, 3, 1), "RIGHTS", "UNSUPPORTED_ACTION_TYPE")]
        return events, reg

    def test_barriers_split_the_timeline_into_three_eligible_epochs(self):
        events, reg = self._world()
        assert ev(*FULL, events, reg).barrier_dates == (D(2020, 3, 1), D(2024, 3, 1))
        assert ev(D(2015, 1, 1), D(2020, 2, 29), events, reg).eligible
        assert ev(D(2020, 3, 1), D(2024, 2, 29), events, reg).eligible
        assert ev(D(2024, 3, 1), D(2026, 9, 11), events, reg).eligible

    def test_intervals_crossing_a_barrier_are_ineligible_wherever_they_start_and_end(self):
        events, reg = self._world()
        for s, e in [(D(2019, 1, 1), D(2021, 1, 1)), (D(2023, 1, 1), D(2025, 6, 1)), (D(2015, 1, 1), D(2026, 9, 11))]:
            assert not ev(s, e, events, reg).eligible

    def test_verified_events_alone_never_create_a_barrier(self):
        events, _ = self._world()
        verified_only = [e for e in events if e.verification_status == V and e.components[0].component_type in ("SPLIT", "BONUS")]
        r = ev(*FULL, verified_only, [])
        assert r.eligible and r.events_considered == len(verified_only)


class TestNoEventIntervalsAroundABlocker:
    def test_quiet_intervals_before_and_after_a_blocker_are_eligible(self):
        events = [bonus(D(2020, 6, 1), status=UV, comp_status=MP)]
        reg = [R(D(2020, 6, 1), "BONUS")]
        assert ev(D(2018, 1, 1), D(2020, 5, 31), events, reg).eligible
        assert ev(D(2020, 6, 1), D(2023, 1, 1), events, reg).eligible
        assert ev(D(2020, 6, 2), D(2023, 1, 1), events, reg).eligible

    def test_dividend_only_intervals_are_eligible_and_dividends_never_create_barriers(self):
        div = E(D(2020, 6, 1), C("DIVIDEND", UV, None), status=UV)
        assert ev(*FULL, [div]).eligible


class TestOtherComponents:
    @pytest.mark.parametrize("phrase", ["Annual General Meeting", "Agm", "AGM", "EGM", "Extra Ordinary General Meeting",
                                        "Extra-Ordinary General Meeting", "  Agm  ", "annual general meeting"])
    def test_only_the_exact_benign_meeting_phrases_are_exempt(self, phrase):
        assert ev(*FULL, [E(D(2020, 1, 1), C("OTHER", UV, None, phrase))]).eligible

    @pytest.mark.parametrize("phrase", ["Annual General Meeting and Bonus 1:1", "Buyback", "Record Date", "", "AGM 2020",
                                        "Fv Splt Frm Rs 10 To Re 1", "Scheme Of Arrangement", "Agm/Egm"])
    def test_every_other_other_blocks(self, phrase):
        r = ev(*FULL, [E(D(2020, 1, 1), C("OTHER", UV, None, phrase))])
        assert not r.eligible and codes(r) == [ae.UNCLASSIFIED_COMPONENT]

    def test_the_jswsteel_style_hidden_split_as_other_can_never_slip_through_even_when_verified(self):
        r = ev(*FULL, [E(D(2017, 1, 4), C("OTHER", V, None, "Fv Splt Frm Rs 10 To Re 1"), status=V)])
        assert not r.eligible and codes(r) == [ae.UNCLASSIFIED_COMPONENT]

    def test_an_event_with_no_current_components_blocks(self):
        r = ev(*FULL, [E(D(2020, 1, 1))])
        assert not r.eligible and codes(r) == [ae.EVENT_HAS_NO_COMPONENTS]

    def test_a_benign_other_alongside_a_real_split_does_not_hide_the_split(self):
        e = E(D(2020, 1, 1), C("OTHER", UV, None, "Agm"), C("SPLIT", MP, (10.0, 2.0)), status=UV)
        assert not ev(*FULL, [e]).eligible


class TestTermsMustBeCompleteAndPositive:
    @pytest.mark.parametrize("terms", [None, (10.0,), (10.0, 2.0, 1.0), (10.0, None), (None, 2.0), (0.0, 2.0), (10.0, 0.0),
                                       (-10.0, 2.0), (10.0, -2.0), (math.nan, 2.0), (10.0, math.inf), (10.0, 10.0),
                                       ("10", "2"), (True, 2.0)])
    def test_bad_split_terms_block_even_when_fully_verified(self, terms):
        r = ev(*FULL, [split(D(2020, 1, 1), terms)])
        assert not r.eligible and codes(r) == [ae.TERMS_MISSING_OR_INVALID]

    @pytest.mark.parametrize("terms", [None, (1,), (0, 1), (1, 0), (-1, 1), (1, -1), (1, None), (None, 1), (1.5, "x"), (True, 1)])
    def test_bad_bonus_terms_block_even_when_fully_verified(self, terms):
        r = ev(*FULL, [bonus(D(2020, 1, 1), terms)])
        assert not r.eligible and codes(r) == [ae.TERMS_MISSING_OR_INVALID]

    @pytest.mark.parametrize("terms", [(10.0, 2.0), (2.0, 1.0), (1.0, 10.0), (5, 1)])
    def test_good_split_terms_pass_including_a_reverse_split(self, terms):
        assert ev(*FULL, [split(D(2020, 1, 1), terms)]).eligible

    @pytest.mark.parametrize("terms", [(1, 1), (4, 1), (1, 2), (3, 2)])
    def test_good_bonus_terms_pass(self, terms):
        assert ev(*FULL, [bonus(D(2020, 1, 1), terms)]).eligible


class TestVerificationRequiredAtBothLevels:
    def test_event_verified_component_not(self):
        r = ev(*FULL, [bonus(D(2020, 1, 1), status=V, comp_status=MP)])
        assert codes(r) == [ae.COMPONENT_NOT_MANUALLY_VERIFIED, ae.NO_ACTIVE_C2_VERIFICATION]

    def test_component_verified_event_not(self):
        r = ev(*FULL, [bonus(D(2020, 1, 1), status=UV, comp_status=V)])
        assert codes(r) == [ae.EVENT_NOT_MANUALLY_VERIFIED]

    @pytest.mark.parametrize("bad", [UV, MP, DISP, "", "verified", "manually_verified"])
    def test_only_the_exact_manually_verified_status_passes(self, bad):
        assert not ev(*FULL, [bonus(D(2020, 1, 1), status=bad)]).eligible
        assert not ev(*FULL, [bonus(D(2020, 1, 1), comp_status=bad)]).eligible

    def test_a_disputed_event_is_a_blocker(self):
        assert not ev(*FULL, [bonus(D(2020, 1, 1), status=DISP)]).eligible


class TestRegisterIsStrictAndHistorical:
    def test_a_dated_row_in_the_interval_blocks_even_with_a_fully_verified_event_present(self):
        r = ev(*FULL, [bonus(D(2020, 1, 1))], [R(D(2020, 1, 1), "BONUS")])
        assert not r.eligible and {b.source for b in r.blockers} == {"REGISTER"}

    def test_a_row_outside_the_interval_does_not_block_it(self):
        assert ev(D(2021, 1, 1), D(2022, 1, 1), [], [R(D(2020, 1, 1), "BONUS")]).eligible

    @pytest.mark.parametrize("start,end", [FULL, (D(2020, 1, 1), D(2020, 1, 1)), (D(1990, 1, 1), D(1990, 1, 2)), (D(2030, 1, 1), D(2030, 6, 1))])
    def test_a_null_dated_blocking_row_blocks_every_interval(self, start, end):
        r = ev(start, end, [], [R(None, "UNCLASSIFIED", "UNPARSEABLE_ROW", "BLOCKS_COMPLETE")])
        assert not r.eligible and r.blockers[0].detail["global"] is True and r.barrier_dates == ()

    def test_a_null_dated_info_row_does_not_block(self):
        assert ev(*FULL, [], [R(None, "DIAGNOSTIC", "WATCH_JUMP", "INFO")]).eligible

    def test_the_evaluator_never_reads_verification_to_clear_a_register_row(self):
        events = [bonus(D(2020, 1, 1))]
        assert not ev(*FULL, events, [R(D(2020, 1, 1), "BONUS", "TERMS_NOT_MANUALLY_VERIFIED")]).eligible
        assert not ev(*FULL, events, [R(D(2020, 1, 1), "BONUS", "NOT_IN_EVENTS_TABLE")]).eligible
        assert not ev(*FULL, events, [R(D(2020, 1, 1), "BONUS", "CLASS_ONLY_TERMS_UNVERIFIED")]).eligible

    def test_a_register_only_event_absent_from_the_events_table_still_blocks(self):
        r = ev(*FULL, [], [R(D(2020, 1, 1), "BONUS", "NOT_IN_EVENTS_TABLE")])
        assert not r.eligible and r.barrier_dates == (D(2020, 1, 1),)


class TestCompleteAloneNeverEstablishesEligibility:
    @pytest.mark.parametrize("coverage", ["UNKNOWN", "PARTIAL", "CONFLICTED", "", "complete", None])
    def test_anything_but_exact_complete_is_ineligible_even_with_perfect_events_and_register(self, coverage):
        r = ev(*FULL, [split(D(2016, 3, 1)), bonus(D(2018, 3, 1))], [], coverage=coverage)
        assert not r.eligible and codes(r) == [ae.COVERAGE_NOT_COMPLETE]

    def test_other_blockers_are_still_reported_alongside_a_coverage_failure(self):
        r = ev(*FULL, [bonus(D(2020, 1, 1), status=UV, comp_status=MP)], [R(D(2020, 1, 1), "BONUS")], coverage="PARTIAL")
        assert {ae.COVERAGE_NOT_COMPLETE, ae.EVENT_NOT_MANUALLY_VERIFIED, ae.UNRESOLVED_REGISTER_ITEM} <= set(codes(r))

    DEFECTS = {
        "unverified_bonus": dict(events=[bonus(D(2020, 1, 1), status=UV, comp_status=MP)]),
        "unverified_split": dict(events=[split(D(2020, 1, 1), status=UV, comp_status=UV)]),
        "rights": dict(events=[E(D(2020, 1, 1), C("RIGHTS", V))]),
        "demerger": dict(events=[E(D(2020, 1, 1), C("DEMERGER", V))]),
        "merger": dict(events=[E(D(2020, 1, 1), C("MERGER", V))]),
        "future_type": dict(events=[E(D(2020, 1, 1), C("XYZ", V, (1, 1)))]),
        "other_text": dict(events=[E(D(2020, 1, 1), C("OTHER", V, None, "Buyback"))]),
        "no_components": dict(events=[E(D(2020, 1, 1))]),
        "bad_terms": dict(events=[bonus(D(2020, 1, 1), (0, 1))]),
        "register_dated": dict(register=[R(D(2020, 1, 1), "BONUS")]),
        "register_global": dict(register=[R(None, "UNCLASSIFIED", "UNPARSEABLE_ROW", "BLOCKS_COMPLETE")]),
    }

    @pytest.mark.parametrize("defect", sorted(DEFECTS))
    def test_complete_plus_any_single_defect_is_ineligible(self, defect):
        assert not ev(*FULL, **self.DEFECTS[defect]).eligible

    @pytest.mark.parametrize("a,b", list(itertools.combinations(sorted(DEFECTS), 2)))
    def test_complete_plus_any_pair_of_defects_is_ineligible(self, a, b):
        events = self.DEFECTS[a].get("events", []) + self.DEFECTS[b].get("events", [])
        register = self.DEFECTS[a].get("register", []) + self.DEFECTS[b].get("register", [])
        assert not ev(*FULL, events, register).eligible

    def test_there_is_no_way_to_ask_from_a_status_alone(self):
        params = inspect.signature(evaluate_adjustment_eligibility).parameters
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty for p in params.values())
        with pytest.raises(TypeError):
            evaluate_adjustment_eligibility(start=FULL[0], end=FULL[1], coverage_status="COMPLETE")  # type: ignore[call-arg]

    def test_the_old_status_only_gate_is_gone_everywhere(self):
        import backend.app.research.corporate_actions_coverage as cov
        import backend.app.data.instruments.corporate_actions as db

        for mod in (cov, db, ae):
            assert not hasattr(mod, "is_eligible_for_adjustment")


class TestReporting:
    def test_result_is_json_serialisable_and_lists_every_blocker_with_provenance(self):
        events, reg = bajfinance()
        d = ev(*FULL, events, reg).as_dict()
        json.dumps(d)
        assert d["eligible"] is False and d["barrier_dates"] == ["2025-06-16"] and d["blockers"]
        assert {"source", "code", "ex_date", "action_class", "detail"} <= d["blockers"][0].keys()

    def test_barrier_dates_are_sorted_distinct_and_only_inside_the_interval(self):
        events, reg = reliance()
        r = ev(D(2018, 1, 1), D(2024, 1, 1), events, reg)
        assert r.barrier_dates == (D(2020, 5, 13), D(2023, 7, 20))

    def test_the_result_states_its_interval_semantics(self):
        assert "(start, end] anchored at end" in ev(*FULL).note

    def test_the_result_is_deterministic_and_independent_of_input_order(self):
        events, reg = reliance()
        base = ev(*FULL, events, reg)
        rng = random.Random(7)
        for _ in range(5):
            e2, r2 = events[:], reg[:]
            rng.shuffle(e2); rng.shuffle(r2)
            other = ev(*FULL, e2, r2)
            assert other.eligible == base.eligible and other.barrier_dates == base.barrier_dates
            key = lambda r: sorted(json.dumps(b.as_dict(), sort_keys=True) for b in r.blockers)
            assert key(other) == key(base)

    def test_the_gate_computes_no_adjustment_factor(self):
        r = ev(*FULL, [split(D(2016, 3, 1)), bonus(D(2018, 3, 1))])
        assert r.eligible
        assert not [f for f in dir(r) if "factor" in f.lower()]
        assert not [n for n in dir(ae) if "factor" in n.lower()]


class TestMarkedVerifiedIsNotSufficientWithoutAnActiveC2Record:

    @staticmethod
    def _legacy(ex, ctype="BONUS", terms=(1, 1)):
        return E(ex, C(ctype, V, terms, record=False), status=V)

    def test_a_legacy_status_with_no_record_blocks_and_is_identified_as_legacy(self):
        r = ev(*FULL, [self._legacy(D(2020, 1, 1))])
        assert not r.eligible and codes(r) == [ae.NO_ACTIVE_C2_VERIFICATION]
        assert r.blockers[0].detail["legacy_manually_verified_without_c2_evidence"] is True

    def test_the_same_component_with_a_matching_active_record_passes(self):
        assert ev(*FULL, [bonus(D(2020, 1, 1))]).eligible

    def test_a_missing_record_is_not_flagged_as_legacy_when_the_status_never_claimed_verification(self):
        r = ev(*FULL, [bonus(D(2020, 1, 1), status=UV, comp_status=MP)])
        assert ae.NO_ACTIVE_C2_VERIFICATION in codes(r)
        assert [b for b in r.blockers if b.code == ae.NO_ACTIVE_C2_VERIFICATION][0].detail["legacy_manually_verified_without_c2_evidence"] is False

    @pytest.mark.parametrize("rec_kind,a,b,comp_kind,terms", [
        ("BONUS", 2, 1, "BONUS", (1, 1)),
        ("BONUS", 1, 2, "BONUS", (2, 1)),
        ("SPLIT", 10, 2, "SPLIT", (10.0, 5.0)),
        ("SPLIT", 2, 1, "BONUS", (2, 1)),
    ])
    def test_a_record_for_different_exact_terms_or_kind_blocks(self, rec_kind, a, b, comp_kind, terms):
        comp = ComponentView(comp_kind, V, "x", terms, RecordView("r1", rec_kind, a, b))
        r = ev(*FULL, [E(D(2020, 1, 1), comp, status=V)])
        assert not r.eligible and codes(r) == [ae.C2_VERIFICATION_TERMS_MISMATCH]

    def test_terms_are_compared_exactly_a_decimal_record_equals_the_stored_float(self):
        from decimal import Decimal

        comp = ComponentView("SPLIT", V, "x", (2.5, 0.5), RecordView("r1", "SPLIT", Decimal("2.500000"), Decimal("0.500000")))
        assert ev(*FULL, [E(D(2020, 1, 1), comp, status=V)]).eligible

    def test_a_record_does_not_rescue_a_component_whose_status_is_not_verified(self):
        comp = ComponentView("BONUS", MP, "x", (1, 1), RecordView("r1", "BONUS", 1, 1))
        r = ev(*FULL, [E(D(2020, 1, 1), comp, status=V)])
        assert not r.eligible and codes(r) == [ae.COMPONENT_NOT_MANUALLY_VERIFIED]

    def test_a_record_does_not_rescue_a_disputed_event(self):
        r = ev(*FULL, [bonus(D(2020, 1, 1), status=DISP)])
        assert not r.eligible and codes(r) == [ae.EVENT_NOT_MANUALLY_VERIFIED]

    def test_a_record_never_makes_an_unsupported_type_adjustable(self):
        comp = ComponentView("RIGHTS", V, "x", None, RecordView("r1", "RIGHTS", 1, 15))
        r = ev(*FULL, [E(D(2020, 5, 13), comp, status=V)])
        assert not r.eligible and codes(r) == [ae.UNSUPPORTED_ACTION_TYPE]

    def test_a_record_does_not_override_register_or_coverage(self):
        assert not ev(*FULL, [bonus(D(2020, 1, 1))], [R(D(2020, 1, 1), "BONUS")]).eligible
        assert not ev(*FULL, [bonus(D(2020, 1, 1))], coverage="PARTIAL").eligible

    def test_every_compound_component_needs_its_own_record(self):
        both = E(D(2016, 9, 8), C("BONUS", V, (1, 1)), C("SPLIT", V, (10.0, 2.0)), status=V)
        assert ev(*FULL, [both]).eligible
        half = E(D(2016, 9, 8), C("BONUS", V, (1, 1)), C("SPLIT", V, (10.0, 2.0), record=False), status=V)
        r = ev(*FULL, [half])
        assert not r.eligible and [b.action_class for b in r.blockers] == ["SPLIT"]

    def test_the_five_phase_b_components_block_until_they_go_through_c2(self):
        legacy = {
            "BAJFINANCE 2016": E(D(2016, 9, 8), C("BONUS", V, (1, 1), record=False), C("SPLIT", V, (10.0, 2.0), record=False), status=V),
            "HDFCBANK 2019": E(D(2019, 9, 19), C("SPLIT", V, (2.0, 1.0), record=False), status=V),
            "KOTAKBANK 2026": E(D(2026, 1, 14), C("SPLIT", V, (5.0, 1.0), record=False), status=V),
            "JSWSTEEL 2017": E(D(2017, 1, 4), C("SPLIT", V, (10.0, 1.0), record=False), status=V),
        }
        for name, e in legacy.items():
            r = ev(*FULL, [e])
            assert not r.eligible and codes(r) == [ae.NO_ACTIVE_C2_VERIFICATION], name

    def test_jswsteel_is_no_longer_eligible_until_its_split_is_c2_verified(self):
        events, reg = jswsteel()
        assert ev(*FULL, events, reg).eligible
        no_records = [E(e.ex_date, *(ComponentView(c.component_type, c.verification_status, c.raw_text_fragment, c.terms)
                                    for c in e.components), status=e.verification_status, eid=e.event_id) for e in events]
        assert not ev(*FULL, no_records, reg).eligible


class TestExactTermsSharedWithTheGate:
    @pytest.mark.parametrize("terms", [(1.5, 1), (0.5, 1), (4.0000001, 1)])
    def test_a_bonus_must_be_whole_numbers(self, terms):
        r = ev(*FULL, [bonus(D(2020, 1, 1), terms)])
        assert ae.TERMS_MISSING_OR_INVALID in codes(r)
