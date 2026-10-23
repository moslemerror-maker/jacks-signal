from __future__ import annotations

import uuid

import pytest

from backend.app.research.corporate_action_diagnostics import isin_transition_corroboration, scan_close_discontinuities
from backend.app.research.corporate_action_reconciliation import (
    bse_components,
    classify_revisions,
    diff_snapshots,
    nse_components,
    reconcile_economic_events,
)
from backend.app.research.corporate_action_register import build_unresolved_register
from backend.tests.c1_helpers import CLEAN_ISIN, CLEAN_JUMP, WINDOW, D, bse_row, comp, nse_row

EID = uuid.uuid4()


def _register(nse, bse, *, lookup=None, diffs=None, jump=CLEAN_JUMP, isin=CLEAN_ISIN):
    n, nu = nse_components([nse_row(d, t) for d, t in nse], "n")
    b, bu = bse_components([bse_row(d, t) for d, t in bse], "b")
    recon = reconcile_economic_events(n, b, WINDOW, nu + bu)
    revs = classify_revisions(n, recon)
    return build_unresolved_register(recon, revs, diffs or {"NSE": [], "BSE": []}, jump, isin, lookup or {})


def _codes(items, severity=None):
    return sorted(i["reason_code"] for i in items if severity is None or i["severity"] == severity)


D1 = D(2017, 9, 7)


class TestCoverageBlockers:
    def test_a_terms_conflict_blocks_complete(self):
        items = _register([(D1, "Bonus 1:1")], [(D1, "Bonus issue 3:1")])
        assert _codes(items, "BLOCKS_COMPLETE") == ["TERMS_CONFLICT"]

    @pytest.mark.parametrize("nse,bse,code", [
        ([(D1, "Bonus 1:1")], [], "ONE_SIDED_NSE"),
        ([], [(D1, "Bonus issue 1:1")], "ONE_SIDED_BSE"),
    ])
    def test_one_sided_events_block_complete(self, nse, bse, code):
        assert _codes(_register(nse, bse), "BLOCKS_COMPLETE") == [code]

    def test_unclassified_text_blocks_complete_and_keeps_the_text(self):
        (item,) = _register([(D1, "A Wholly Novel Event")], [])
        assert (item["reason_code"], item["severity"], item["action_class"]) == ("UNCLASSIFIED_TEXT", "BLOCKS_COMPLETE", "UNCLASSIFIED")

    def test_a_price_affecting_duplicate_blocks_complete_but_a_dividend_duplicate_is_info(self):
        items = _register([(D1, "Bonus 1:1"), (D1, "Bonus 1:1")], [(D1, "Bonus issue 1:1")], lookup={})
        assert "DUPLICATE_ECONOMIC_EVENT" in _codes(items, "BLOCKS_COMPLETE")
        d = D(2016, 6, 10)
        items = _register([(d, "Dividend - Rs 6 Per Share")] * 2, [(d, "Final Dividend - Rs. - 6.0000")])
        assert _codes(items) == ["DUPLICATE_ECONOMIC_EVENT"] and items[0]["severity"] == "INFO"

    def test_ambiguous_same_date_actions_block_complete(self):
        d = D(2019, 9, 19)
        items = _register(
            [(d, "Face Value Split From Rs.10/- To Rs.5/-"), (d, "Face Value Split From Rs.5/- To Rs.1/-")],
            [(d, "Stock  Split From Rs.10/- to Rs.5/-"), (d, "Stock  Split From Rs.5/- to Rs.1/-")])
        assert "AMBIGUOUS_SAME_DATE" in _codes(items, "BLOCKS_COMPLETE")

    def test_an_unparseable_row_is_registered_with_no_date(self):
        items = _register([], [])
        n, nu = nse_components([{"exDate": "-", "subject": "Bonus 1:1"}], "n")
        recon = reconcile_economic_events(n, [], WINDOW, nu)
        (item,) = [i for i in build_unresolved_register(recon, [], {"NSE": [], "BSE": []}, CLEAN_JUMP, CLEAN_ISIN, {})
                   if i["reason_code"] == "UNPARSEABLE_ROW"]
        assert item["ex_date"] is None and item["severity"] == "BLOCKS_COMPLETE" and items == []

    def test_blocking_revisions_block_and_non_blocking_are_info(self):
        d = D(2016, 3, 16)
        info = _register([(d, "Interim Dividend - Rs 18/- Per Share (Purpose Revised)")], [(d, "Interim Dividend - Rs. - 18.0000")])
        assert _codes(info) == ["REVISED_NON_PRICE_AFFECTING"] and info[0]["severity"] == "INFO"
        d2 = D(2020, 5, 13)
        blocking = _register([(d2, "Rights 1:15 @ Premium Rs 1247 (Purpose Revised)")], [(d2, "Right Issue of Equity Shares")])
        assert "REVISED_PRICE_AFFECTING_UNCORROBORATED" in _codes(blocking, "BLOCKS_COMPLETE")

    def test_price_affecting_snapshot_changes_block_and_others_are_info(self):
        old = [comp(D1, "BONUS", (1, 1)), comp(D(2016, 6, 10), "DIVIDEND", (6.0,))]
        items = _register([], [], diffs={"NSE": diff_snapshots(old, [], WINDOW), "BSE": []})
        by = {i["action_class"]: i for i in items}
        assert by["BONUS"]["reason_code"] == "SNAPSHOT_DISAPPEARED" and by["BONUS"]["severity"] == "BLOCKS_COMPLETE"
        assert by["DIVIDEND"]["severity"] == "INFO" and by["BONUS"]["detail"]["provider"] == "NSE"

    def test_diagnostic_blockers_are_registered_and_watch_jumps_are_info(self):
        bars = [(D(2020, 1, 1), 100.0), (D(2020, 1, 2), 60.0), (D(2020, 1, 3), 45.0)]
        jump = scan_close_discontinuities(bars, [])
        items = _register([], [], jump=jump)
        assert _codes(items, "BLOCKS_COMPLETE") == ["UNEXPLAINED_JUMP"] and _codes(items, "INFO") == ["WATCH_JUMP"]
        isin = isin_transition_corroboration([D(2020, 1, 3)], [D(2020, 1, 1), D(2020, 1, 2), D(2020, 1, 3)], [])
        assert _codes(_register([], [], isin=isin), "BLOCKS_COMPLETE") == ["ISIN_TRANSITION_UNCORROBORATED"]


class TestAdjustmentBlockersThatDoNotBlockCoverage:
    def test_a_rights_issue_is_unsupported_for_adjustment_even_when_both_sources_agree(self):
        d = D(2020, 5, 13)
        (item,) = _register([(d, "Rights 1:15 @ Premium Rs 1247")], [(d, "Right Issue of Equity Shares")])
        assert item["reason_code"] == "UNSUPPORTED_ACTION_TYPE" and item["severity"] == "BLOCKS_ADJUSTMENT"
        assert "class-level only" in item["detail"]["note"]

    def test_a_demerger_is_unsupported_for_adjustment(self):
        d = D(2023, 7, 20)
        (item,) = _register([(d, "Demerger")], [(d, "Spin Off")])
        assert (item["action_class"], item["reason_code"], item["severity"]) == ("DEMERGER", "UNSUPPORTED_ACTION_TYPE", "BLOCKS_ADJUSTMENT")

    def test_class_only_agreement_on_a_terms_bearing_action_never_verifies_terms(self):
        d = D(2019, 9, 19)
        (item,) = _register([(d, "Face Value Split - details awaited")], [(d, "Stock  Split From Rs.2/- to Rs.1/-")])
        assert item["reason_code"] == "CLASS_ONLY_TERMS_UNVERIFIED" and item["severity"] == "BLOCKS_ADJUSTMENT"

    def test_a_term_matched_bonus_absent_from_the_events_table_is_registered(self):
        (item,) = _register([(D1, "Bonus 1:1")], [(D1, "Bonus issue 1:1")], lookup={})
        assert item["reason_code"] == "NOT_IN_EVENTS_TABLE" and item["severity"] == "BLOCKS_ADJUSTMENT"

    def test_a_term_matched_bonus_that_is_not_manually_verified_is_registered_with_its_event_id(self):
        lookup = {(D1, "BONUS"): [{"event_id": EID, "status": "MACHINE_PARSED", "terms": (1, 1)}]}
        (item,) = _register([(D1, "Bonus 1:1")], [(D1, "Bonus issue 1:1")], lookup=lookup)
        assert item["reason_code"] == "TERMS_NOT_MANUALLY_VERIFIED" and item["event_id"] == EID

    def test_a_manually_verified_event_with_matching_terms_is_resolved_and_not_registered(self):
        lookup = {(D1, "BONUS"): [{"event_id": EID, "status": "MANUALLY_VERIFIED", "terms": (1, 1), "c2_active": True}]}
        assert _register([(D1, "Bonus 1:1")], [(D1, "Bonus issue 1:1")], lookup=lookup) == []

    def test_manual_verification_of_DIFFERENT_terms_does_not_resolve_the_event(self):
        lookup = {(D1, "BONUS"): [{"event_id": EID, "status": "MANUALLY_VERIFIED", "terms": (2, 1)}]}
        (item,) = _register([(D1, "Bonus 1:1")], [(D1, "Bonus issue 1:1")], lookup=lookup)
        assert item["reason_code"] == "TERMS_NOT_MANUALLY_VERIFIED"

    @pytest.mark.parametrize("entry", [
        {"event_id": EID, "status": "MANUALLY_VERIFIED", "terms": (1, 1), "c2_active": False},
        {"event_id": EID, "status": "MANUALLY_VERIFIED", "terms": (1, 1)},
    ])
    def test_a_legacy_verified_event_without_an_active_c2_record_is_identified_and_not_resolved(self, entry):
        (item,) = _register([(D1, "Bonus 1:1")], [(D1, "Bonus issue 1:1")], lookup={(D1, "BONUS"): [entry]})
        assert item["reason_code"] == "LEGACY_VERIFICATION_WITHOUT_C2_EVIDENCE" and item["severity"] == "BLOCKS_ADJUSTMENT"
        assert item["detail"]["events_table"][0]["c2_active"] is False

    def test_an_active_c2_record_for_different_terms_never_resolves_the_event(self):
        lookup = {(D1, "BONUS"): [{"event_id": EID, "status": "MANUALLY_VERIFIED", "terms": (2, 1), "c2_active": True}]}
        (item,) = _register([(D1, "Bonus 1:1")], [(D1, "Bonus issue 1:1")], lookup=lookup)
        assert item["reason_code"] == "TERMS_NOT_MANUALLY_VERIFIED"

    def test_an_active_record_without_a_manually_verified_status_does_not_resolve_it(self):
        lookup = {(D1, "BONUS"): [{"event_id": EID, "status": "MACHINE_PARSED", "terms": (1, 1), "c2_active": True}]}
        (item,) = _register([(D1, "Bonus 1:1")], [(D1, "Bonus issue 1:1")], lookup=lookup)
        assert item["reason_code"] == "TERMS_NOT_MANUALLY_VERIFIED"

    def test_a_dividend_that_matches_needs_no_register_entry(self):
        d = D(2016, 6, 10)
        assert _register([(d, "Dividend - Rs 6 Per Share")], [(d, "Final Dividend - Rs. - 6.0000")]) == []

    def test_a_clean_fully_verified_window_has_an_empty_register(self):
        lookup = {(D1, "BONUS"): [{"event_id": EID, "status": "MANUALLY_VERIFIED", "terms": (1, 1), "c2_active": True}]}
        assert _register([(D1, "Bonus 1:1"), (D(2016, 6, 10), "Dividend - Rs 6 Per Share")],
                         [(D1, "Bonus issue 1:1"), (D(2016, 6, 10), "Final Dividend - Rs. - 6.0000")], lookup=lookup) == []


class TestRegisterShape:
    def test_every_item_carries_the_persistable_fields(self):
        items = _register([(D1, "Bonus 1:1")], [])
        for i in items:
            assert {"ex_date", "action_class", "reason_code", "severity", "detail", "event_id"} <= i.keys()
            assert i["severity"] in {"BLOCKS_COMPLETE", "BLOCKS_ADJUSTMENT", "INFO"}

    def test_a_reliance_shaped_window_is_complete_with_unresolved_events_listed(self):
        d20, d23 = D(2020, 5, 13), D(2023, 7, 20)
        items = _register(
            [(D1, "Bonus 1:1"), (d20, "Rights 1:15 @ Premium Rs 1247"), (d23, "Demerger")],
            [(D1, "Bonus issue 1:1"), (d20, "Right Issue of Equity Shares"), (d23, "Spin Off")])
        assert _codes(items, "BLOCKS_COMPLETE") == []
        assert _codes(items, "BLOCKS_ADJUSTMENT") == ["NOT_IN_EVENTS_TABLE", "UNSUPPORTED_ACTION_TYPE", "UNSUPPORTED_ACTION_TYPE"]
