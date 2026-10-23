from __future__ import annotations

import pytest

from backend.app.research.corporate_action_reconciliation import (
    bse_components,
    classify_revisions,
    detect_revision,
    diff_snapshots,
    nse_components,
    reconcile_economic_events,
)
from backend.tests.c1_helpers import D, WINDOW, bse_row, comp, nse_row

_BAJ_2016_NSE = "Bonus 1:1/Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share"


def _recon(nse_rows, bse_rows, window=WINDOW):
    n, nu = nse_components(nse_rows, "n")
    b, bu = bse_components(bse_rows, "b")
    return reconcile_economic_events(n, b, window, nu + bu)


class TestCompoundVersusSplitRowRepresentation:
    def test_nse_compound_row_reconciles_with_two_bse_rows_in_either_order(self):
        d = D(2016, 9, 8)
        for order in (
            [bse_row(d, "Bonus issue 1:1"), bse_row(d, "Stock  Split From Rs.10/- to Rs.2/-")],
            [bse_row(d, "Stock  Split From Rs.10/- to Rs.2/-"), bse_row(d, "Bonus issue 1:1")],
        ):
            r = _recon([nse_row(d, _BAJ_2016_NSE)], order)
            assert r.clean
            assert sorted(m.action_class for m in r.matches) == ["BONUS", "SPLIT"]
            assert all(m.status == "TERMS_MATCH" for m in r.matches)

    def test_two_separate_nse_rows_reconcile_with_one_bse_compound_representation(self):
        d = D(2025, 6, 16)
        r = _recon(
            [nse_row(d, "Bonus 4:1"), nse_row(d, "Face Value Split (Sub-Division) - From Rs 2/- Per Share To Re 1/- Per Share")],
            [bse_row(d, "Stock  Split From Rs.2/- to Rs.1/-"), bse_row(d, "Bonus issue 4:1")],
        )
        assert r.clean and r.count("TERMS_MATCH") == 2

    def test_result_is_independent_of_row_order(self):
        d = D(2016, 9, 8)
        a = _recon([nse_row(d, _BAJ_2016_NSE)], [bse_row(d, "Bonus issue 1:1"), bse_row(d, "Stock  Split From Rs.10/- to Rs.2/-")])
        b = _recon([nse_row(d, _BAJ_2016_NSE)], [bse_row(d, "Stock  Split From Rs.10/- to Rs.2/-"), bse_row(d, "Bonus issue 1:1")])
        def econ(r):
            return [{k: v for k, v in m.as_dict().items() if not k.endswith("_ref")} for m in r.matches]
        assert econ(a) == econ(b)


class TestClassOnlyAgreementNeverVerifiesTerms:
    def test_rights_with_terms_on_nse_and_none_on_bse_is_class_only(self):
        d = D(2020, 5, 13)
        r = _recon([nse_row(d, "Rights 1:15 @ Premium Rs 1247")], [bse_row(d, "Right Issue of Equity Shares ")])
        (m,) = r.matches
        assert m.status == "CLASS_ONLY" and m.action_class == "RIGHTS"
        assert r.count("TERMS_MATCH") == 0 and r.clean

    def test_demerger_and_spin_off_are_the_same_class(self):
        d = D(2023, 7, 20)
        r = _recon([nse_row(d, "Demerger")], [bse_row(d, "Spin Off ")])
        (m,) = r.matches
        assert m.status == "CLASS_ONLY" and m.action_class == "DEMERGER"

    def test_a_split_with_no_terms_on_one_side_is_class_only_never_terms_match(self):
        d = D(2019, 9, 19)
        r = _recon([nse_row(d, "Face Value Split - details awaited")],
                   [bse_row(d, "Stock  Split From Rs.2/- to Rs.1/-")])
        assert r.matches[0].status == "CLASS_ONLY"

    def test_terms_match_requires_both_sides_to_state_equal_terms(self):
        d = D(2017, 9, 7)
        assert _recon([nse_row(d, "Bonus 1:1")], [bse_row(d, "Bonus issue 1:1")]).matches[0].status == "TERMS_MATCH"


class TestConflictsAndOneSidedEventsPreventClean:
    def test_terms_disagreement_is_a_conflict(self):
        d = D(2017, 9, 7)
        r = _recon([nse_row(d, "Bonus 1:1")], [bse_row(d, "Bonus issue 2:1")])
        assert r.matches[0].status == "TERMS_CONFLICT" and r.conflict_count == 1 and not r.clean

    def test_split_ratio_disagreement_is_a_conflict_not_a_silent_pick(self):
        d = D(2026, 1, 14)
        r = _recon([nse_row(d, "Face Value Split (Sub-Division) - From Rs 5/- Per Share To Re 1/- Per Share")],
                   [bse_row(d, "Stock  Split From Rs.5/- to Rs.2/-")])
        assert r.conflict_count == 1

    def test_dividend_amount_disagreement_is_a_conflict(self):
        d = D(2016, 6, 10)
        r = _recon([nse_row(d, "Dividend - Rs 6 Per Share")], [bse_row(d, "Final Dividend - Rs. - 7.0000")])
        assert r.conflict_count == 1 and not r.clean

    def test_event_only_on_nse_is_one_sided(self):
        r = _recon([nse_row(D(2017, 9, 7), "Bonus 1:1")], [])
        assert [m.status for m in r.matches] == ["ONE_SIDED_NSE"] and not r.clean

    def test_event_only_on_bse_is_one_sided(self):
        r = _recon([], [bse_row(D(2017, 9, 7), "Bonus issue 1:1")])
        assert [m.status for m in r.matches] == ["ONE_SIDED_BSE"] and not r.clean

    def test_same_class_on_different_dates_is_two_one_sided_events_not_a_match(self):
        r = _recon([nse_row(D(2017, 9, 7), "Bonus 1:1")], [bse_row(D(2017, 9, 8), "Bonus issue 1:1")])
        assert sorted(m.status for m in r.matches) == ["ONE_SIDED_BSE", "ONE_SIDED_NSE"]

    def test_unrecognised_text_on_either_side_is_unclassified_and_blocks(self):
        d = D(2018, 1, 1)
        assert not _recon([nse_row(d, "Some Entirely Novel Event")], []).clean
        assert not _recon([], [bse_row(d, "Buy Back of Shares")]).clean

    def test_row_with_an_unparseable_date_is_reported_never_dropped(self):
        r = _recon([{"exDate": "-", "subject": "Bonus 1:1"}], [])
        assert len(r.unparseable_rows) == 1 and not r.clean


class TestDuplicateEconomicEvents:
    def test_duplicate_price_affecting_row_within_one_source_blocks(self):
        d = D(2017, 9, 7)
        r = _recon([nse_row(d, "Bonus 1:1"), nse_row(d, "Bonus 1:1")], [bse_row(d, "Bonus issue 1:1")])
        assert r.duplicate_price_affecting_count == 1 and not r.clean
        assert r.count("TERMS_MATCH") == 1

    def test_duplicate_dividend_is_collapsed_and_only_noted(self):
        d = D(2016, 6, 10)
        r = _recon([nse_row(d, "Dividend - Rs 6 Per Share"), nse_row(d, "Dividend - Rs 6 Per Share")],
                   [bse_row(d, "Final Dividend - Rs. - 6.0000")])
        assert r.duplicate_price_affecting_count == 0 and r.clean
        assert r.summary()["duplicate_non_price_affecting_count"] == 1

    def test_same_price_affecting_class_twice_with_different_terms_is_ambiguous(self):
        d = D(2019, 9, 19)
        r = _recon([nse_row(d, "Face Value Split From Rs.10/- To Rs.5/-"), nse_row(d, "Face Value Split From Rs.5/- To Rs.1/-")],
                   [bse_row(d, "Stock  Split From Rs.10/- to Rs.5/-"), bse_row(d, "Stock  Split From Rs.5/- to Rs.1/-")])
        assert len(r.ambiguous_same_date) == 2 and not r.clean

    def test_two_different_dividends_the_same_day_are_not_ambiguous(self):
        d = D(2016, 6, 10)
        r = _recon([nse_row(d, "Dividend - Rs 6 Per Share"), nse_row(d, "Special Dividend - Rs 12 Per Share")],
                   [bse_row(d, "Final Dividend - Rs. - 6.0000"), bse_row(d, "Special Dividend - Rs. - 12.0000")])
        assert r.clean and r.count("TERMS_MATCH") == 2


class TestWindowAndDeterminism:
    def test_events_outside_the_window_are_ignored(self):
        r = _recon([nse_row(D(2011, 7, 14), "Face Value Split From Rs.10/- To Rs.2/-")], [])
        assert r.matches == [] and r.clean

    def test_events_on_the_window_boundaries_are_included(self):
        r = _recon([nse_row(WINDOW[0], "Bonus 1:1"), nse_row(WINDOW[1], "Bonus 1:1")], [])
        assert len(r.matches) == 2

    def test_agm_only_rows_are_non_economic_and_dropped_but_other_text_is_not(self):
        c, _ = nse_components([nse_row(D(2016, 7, 4), "Annual General Meeting")], "n")
        assert c == []
        c, _ = nse_components([nse_row(D(2016, 7, 4), "Some Novel Corporate Event")], "n")
        assert [x.action_class for x in c] == ["UNCLASSIFIED"]


class TestRevisionDetectionAndClassification:
    @pytest.mark.parametrize("text", [
        "Interim Dividend - Rs 18/- Per Share (Purpose Revised)",
        " Interim Dividend - Rs 10.50/- Per Share (Purpose Revised)",
    ])
    def test_the_two_real_nse_revision_strings_are_recognised(self, text):
        assert detect_revision(text).kind == "PURPOSE_REVISED_MARKER"

    @pytest.mark.parametrize("text", [
        "Bonus 1:1 - Purpose Revised", "Dividend - Rs 6 Per Share (Corrected)",
        "Bonus 1:1 (Cancelled)", "Split postponed", "Bonus 1:1 - rescheduled", "Amended Dividend Rs 5 Per Share",
    ])
    def test_any_other_revision_language_is_reported_as_unrecognised_never_ignored(self, text):
        assert detect_revision(text).kind == "UNRECOGNIZED_REVISION_LANGUAGE"

    def test_the_recognised_marker_is_recognised_on_price_affecting_rows_too(self):
        assert detect_revision("Bonus 1:1 (Purpose Revised)").kind == "PURPOSE_REVISED_MARKER"

    def test_ordinary_text_has_no_revision_signal(self):
        assert detect_revision("Bonus 1:1") is None and detect_revision("Face Value Split From Rs.10/- To Rs.2/-") is None

    def test_revised_dividend_is_recorded_but_non_blocking(self):
        d = D(2016, 3, 16)
        n, nu = nse_components([nse_row(d, "Interim Dividend - Rs 18/- Per Share (Purpose Revised)")], "n")
        b, bu = bse_components([bse_row(d, "Interim Dividend - Rs. - 18.0000")], "b")
        f = classify_revisions(n, reconcile_economic_events(n, b, WINDOW))
        assert [(x.code, x.blocking) for x in f] == [("REVISED_NON_PRICE_AFFECTING", False)]

    def test_revised_price_affecting_row_is_corroborated_only_by_a_terms_match(self):
        d = D(2017, 9, 7)
        n, _ = nse_components([nse_row(d, "Bonus 1:1 (Purpose Revised)")], "n")
        b, _ = bse_components([bse_row(d, "Bonus issue 1:1")], "b")
        f = classify_revisions(n, reconcile_economic_events(n, b, WINDOW))
        assert [(x.code, x.blocking) for x in f] == [("REVISED_PRICE_AFFECTING_CORROBORATED", False)]

    def test_revised_price_affecting_row_with_class_only_agreement_is_uncorroborated_and_blocks(self):
        d = D(2020, 5, 13)
        n, _ = nse_components([nse_row(d, "Rights 1:15 @ Premium Rs 1247 (Purpose Revised)")], "n")
        b, _ = bse_components([bse_row(d, "Right Issue of Equity Shares")], "b")
        f = classify_revisions(n, reconcile_economic_events(n, b, WINDOW))
        assert [(x.code, x.blocking) for x in f] == [("REVISED_PRICE_AFFECTING_UNCORROBORATED", True)]

    def test_revised_price_affecting_row_that_bse_disagrees_with_blocks(self):
        d = D(2017, 9, 7)
        n, _ = nse_components([nse_row(d, "Bonus 1:1 (Purpose Revised)")], "n")
        b, _ = bse_components([bse_row(d, "Bonus issue 3:1")], "b")
        assert classify_revisions(n, reconcile_economic_events(n, b, WINDOW))[0].blocking

    def test_unrecognised_revision_language_on_a_price_affecting_row_blocks(self):
        d = D(2017, 9, 7)
        n, _ = nse_components([nse_row(d, "Bonus 1:1 (Cancelled)")], "n")
        b, _ = bse_components([bse_row(d, "Bonus issue 1:1")], "b")
        f = classify_revisions(n, reconcile_economic_events(n, b, WINDOW))
        assert (f[0].code, f[0].blocking) == ("REVISION_UNRECOGNIZED_PRICE_AFFECTING", True)

    def test_every_revision_finding_disclaims_any_point_in_time_knowledge(self):
        d = D(2016, 3, 16)
        n, _ = nse_components([nse_row(d, "Interim Dividend - Rs 18/- Per Share (Purpose Revised)")], "n")
        f = classify_revisions(n, reconcile_economic_events(n, [], WINDOW))
        assert "NO claim about when the revision occurred" in f[0].note


class TestSnapshotDiffs:
    def _c(self, *specs):
        return [comp(D(2017, 9, 7), cls, terms, fragment=frag) for cls, terms, frag in specs]

    def test_identical_snapshots_have_no_changes(self):
        s = self._c(("BONUS", (1, 1), "Bonus 1:1"))
        assert diff_snapshots(s, s, WINDOW) == []

    def test_a_price_affecting_action_that_disappears_is_a_blocking_change(self):
        (c,) = diff_snapshots(self._c(("BONUS", (1, 1), "Bonus 1:1")), [], WINDOW)
        assert c.kind == "DISAPPEARED" and c.blocking

    def test_terms_changed_on_a_price_affecting_action_is_blocking(self):
        (c,) = diff_snapshots(self._c(("BONUS", (1, 1), "Bonus 1:1")), self._c(("BONUS", (2, 1), "Bonus 2:1")), WINDOW)
        assert c.kind == "TERMS_CHANGED" and c.blocking

    def test_a_dividend_disappearing_is_recorded_but_not_blocking(self):
        (c,) = diff_snapshots(self._c(("DIVIDEND", (6.0,), "Dividend - Rs 6")), [], WINDOW)
        assert c.kind == "DISAPPEARED" and not c.blocking

    def test_a_late_appearing_action_is_recorded_and_not_itself_contradictory(self):
        (c,) = diff_snapshots([], self._c(("BONUS", (1, 1), "Bonus 1:1")), WINDOW)
        assert c.kind == "APPEARED" and not c.blocking

    def test_text_only_revision_of_an_unchanged_economic_event_is_text_revised_non_blocking(self):
        (c,) = diff_snapshots(self._c(("BONUS", (1, 1), "Bonus 1:1")),
                              self._c(("BONUS", (1, 1), "Bonus 1:1 (Purpose Revised)")), WINDOW)
        assert c.kind == "TEXT_REVISED" and not c.blocking

    def test_comparison_is_restricted_to_the_supplied_window(self):
        old = [comp(D(2011, 7, 14), "SPLIT", (10.0, 2.0))]
        assert diff_snapshots(old, [], WINDOW) == []
