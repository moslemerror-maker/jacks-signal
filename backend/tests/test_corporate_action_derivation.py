from __future__ import annotations

import dataclasses

import pytest

from backend.app.research.corporate_action_evidence import (
    BoundarySemantics,
    cap_observations,
    cap_signal,
    partition_consistency,
)
from backend.app.research.corporate_action_diagnostics import isin_transition_corroboration, scan_close_discontinuities
from backend.app.research.corporate_action_reconciliation import diff_snapshots
from backend.app.research.corporate_actions_coverage import REQUIRED_CHECKS, canonical_json, evidence_sha256
from backend.tests.c1_helpers import (
    BASE_BSE, BASE_NSE, D, INCLUSIVE, WINDOW, bse_row, clean_provider, comp, derive, nse_row, snap,
)


def _all_checks_true(ev):
    return all(ev["checks"][k] for k in REQUIRED_CHECKS)


def _with(provider, **changes):
    return dataclasses.replace(provider, **changes)


def _whole_replaced(provider, **changes):
    return _with(provider, whole=dataclasses.replace(provider.whole, **changes))


def _providers(nse=None, bse=None):
    return {"NSE": nse or clean_provider("NSE", BASE_NSE), "BSE": bse or clean_provider("BSE", BASE_BSE)}


class TestTheCleanCase:
    def test_two_agreeing_verified_consistent_sources_derive_complete(self):
        ev = derive()
        assert ev["verdict"] == "COMPLETE" and _all_checks_true(ev)

    def test_the_verdict_is_derived_and_named_and_carries_the_schema(self):
        ev = derive()
        assert ev["schema"] == "coverage-evidence/1" and ev["derivation_version"].startswith("c1-derive")

    def test_evidence_records_every_source_snapshot_with_provenance(self):
        ev = derive()
        assert [s["provider"] for s in ev["sources"]] == ["NSE", "BSE"]
        for s in ev["sources"]:
            assert s["whole"]["sha256"] and s["whole"]["fetched_at"] and len(s["parts"]) == 3
            assert s["boundary_semantics"]["verified"] and s["partition"]["consistent"]

    def test_the_bundle_is_json_serialisable_and_its_hash_is_deterministic(self):
        a, b = derive(), derive()
        assert canonical_json(a) == canonical_json(b) and evidence_sha256(a) == evidence_sha256(b)

    def test_any_change_to_the_evidence_changes_its_hash(self):
        assert evidence_sha256(derive()) != evidence_sha256(derive(nse_dated=BASE_NSE[:3], bse_dated=BASE_BSE[:3]))


class TestChronologyIsSeparatedAndNoPointInTimeClaimIsMade:
    def test_information_chronology_is_explicitly_not_established(self):
        c = derive()["chronology"]
        assert "NOT ESTABLISHED" in c["information_chronology"]
        assert "ex_date" in c["economic_adjustment_chronology"]

    def test_fetched_at_is_declared_our_clock_not_publication_time(self):
        assert "OUR retrieval clock only" in derive()["chronology"]["information_chronology"]

    def test_complete_reasoning_does_not_claim_pit_knowledge(self):
        r = derive()["reasoning"].lower()
        assert "point-in-time" not in r and "publicly known" not in r


class TestCompleteMeansEventSetKnownNotTermsVerified:
    def test_class_only_agreement_still_derives_complete_but_is_counted_separately(self):
        nse = BASE_NSE + [(D(2020, 5, 13), nse_row(D(2020, 5, 13), "Rights 1:15 @ Premium Rs 1247"))]
        bse = BASE_BSE + [(D(2020, 5, 13), bse_row(D(2020, 5, 13), "Right Issue of Equity Shares "))]
        ev = derive(nse, bse)
        assert ev["verdict"] == "COMPLETE"
        assert ev["reconciliation"]["class_only"] == 1 and ev["reconciliation"]["term_matches"] == 4

    def test_the_bundle_never_states_that_terms_are_verified(self):
        assert "does not mean every event's terms are verified" in derive()["reasoning"]


class TestConflictingSources:
    def test_terms_disagreement_is_conflicted(self):
        bse = [(d, bse_row(d, "Bonus issue 3:1")) if d == D(2017, 9, 7) else (d, r) for d, r in BASE_BSE]
        ev = derive(bse_dated=bse)
        assert ev["verdict"] == "CONFLICTED" and not ev["checks"]["reconciliation_clean"]

    def test_event_only_one_source_reports_is_conflicted_either_way(self):
        assert derive(nse_dated=BASE_NSE, bse_dated=BASE_BSE[:2] + BASE_BSE[3:])["verdict"] == "CONFLICTED"
        assert derive(nse_dated=BASE_NSE[:2] + BASE_NSE[3:], bse_dated=BASE_BSE)["verdict"] == "CONFLICTED"

    def test_conflict_outranks_every_other_failure(self):
        bse = [(d, bse_row(d, "Bonus issue 3:1")) if d == D(2017, 9, 7) else (d, r) for d, r in BASE_BSE]
        prov = _providers(bse=_whole_replaced(clean_provider("BSE", bse), integrity_ok=False, fetched_at="2020-01-01T00:00:00+00:00"))
        ev = derive(nse_dated=BASE_NSE, bse_dated=bse, providers=prov)
        assert ev["verdict"] == "CONFLICTED"
        assert not ev["checks"]["source_integrity_ok"] and not ev["checks"]["freshness_ok"]

    def test_a_price_affecting_action_that_disappears_between_snapshots_is_conflicted(self):
        old = [comp(D(2017, 9, 7), "BONUS", (1, 1), fragment="Bonus 1:1")]
        ev = derive(diffs={"NSE": diff_snapshots(old, [], WINDOW), "BSE": []})
        assert ev["verdict"] == "CONFLICTED" and not ev["checks"]["snapshot_diffs_clear"]

    def test_terms_changing_between_snapshots_is_conflicted(self):
        old = [comp(D(2017, 9, 7), "BONUS", (1, 1), fragment="Bonus 1:1")]
        new = [comp(D(2017, 9, 7), "BONUS", (2, 1), fragment="Bonus 2:1")]
        assert derive(diffs={"NSE": diff_snapshots(old, new, WINDOW), "BSE": []})["verdict"] == "CONFLICTED"

    def test_bse_disappearance_is_treated_the_same_as_nse_disappearance(self):
        old = [comp(D(2017, 9, 7), "SPLIT", (10.0, 2.0), source="BSE")]
        assert derive(diffs={"NSE": [], "BSE": diff_snapshots(old, [], WINDOW)})["verdict"] == "CONFLICTED"


class TestLaterEvidencePreservesHistory:
    def test_a_late_appearing_action_is_recorded_and_changes_the_evidence_hash(self):
        new = [comp(D(2017, 9, 7), "BONUS", (1, 1), fragment="Bonus 1:1")]
        diffs = {"NSE": diff_snapshots([], new, WINDOW), "BSE": []}
        with_later, without = derive(diffs=diffs), derive()
        assert with_later["snapshot_diffs"]["NSE"][0]["kind"] == "APPEARED"
        assert evidence_sha256(with_later) != evidence_sha256(without)

    def test_a_late_appearance_is_not_itself_a_contradiction_of_the_new_snapshot(self):
        new = [comp(D(2017, 9, 7), "BONUS", (1, 1), fragment="Bonus 1:1")]
        assert derive(diffs={"NSE": diff_snapshots([], new, WINDOW), "BSE": []})["verdict"] == "COMPLETE"

    def test_a_new_event_appearing_in_the_later_snapshot_is_evaluated_by_the_full_pipeline(self):
        nse = BASE_NSE + [(D(2021, 3, 3), nse_row(D(2021, 3, 3), "Bonus 1:1"))]
        assert derive(nse_dated=nse)["verdict"] == "CONFLICTED"


class TestDuplicateAndAmbiguousEconomicEvents:
    def test_a_duplicated_price_affecting_row_prevents_complete(self):
        nse = BASE_NSE + [(D(2017, 9, 7), nse_row(D(2017, 9, 7), "Bonus 1:1"))]
        ev = derive(nse_dated=nse)
        assert ev["verdict"] != "COMPLETE" and not ev["checks"]["reconciliation_clean"]
        assert ev["reconciliation"]["duplicate_price_affecting_count"] == 1

    def test_two_different_splits_on_one_day_prevent_complete(self):
        d = D(2019, 9, 19)
        nse = BASE_NSE + [(d, nse_row(d, "Face Value Split (Sub-Division) - From Rs 5 Per Share To Rs 1 Per Share"))]
        bse = BASE_BSE + [(d, bse_row(d, "Stock  Split From Rs.5/- to Rs.1/-"))]
        ev = derive(nse_dated=nse, bse_dated=bse)
        assert ev["verdict"] != "COMPLETE" and ev["reconciliation"]["ambiguous_same_date_count"] >= 1

    def test_unclassifiable_text_prevents_complete(self):
        d = D(2021, 1, 4)
        ev = derive(BASE_NSE + [(d, nse_row(d, "Some Entirely Novel Event"))], BASE_BSE)
        assert ev["verdict"] != "COMPLETE"

    def test_a_row_whose_date_cannot_be_read_prevents_complete(self):
        bad = (D(2021, 1, 4), {"symbol": "TESTCO", "exDate": "-", "subject": "Bonus 1:1"})
        assert derive(BASE_NSE + [bad], BASE_BSE)["verdict"] != "COMPLETE"


class TestMissingSourcesAreAbsentEvidenceNotContradiction:
    def test_no_bse_at_all_is_unknown(self):
        ev = derive(providers={"NSE": clean_provider("NSE", BASE_NSE)})
        assert ev["verdict"] == "UNKNOWN" and not ev["checks"]["two_independent_providers"]

    def test_no_nse_at_all_is_unknown(self):
        assert derive(providers={"BSE": clean_provider("BSE", BASE_BSE)})["verdict"] == "UNKNOWN"

    def test_no_providers_at_all_is_unknown(self):
        ev = derive(providers={})
        assert ev["verdict"] == "UNKNOWN" and not any(ev["checks"][k] for k in ("two_independent_providers", "source_integrity_ok"))

    def test_a_provider_with_no_whole_window_snapshot_is_unknown_never_complete(self):
        ev = derive(providers=_providers(bse=_with(clean_provider("BSE", BASE_BSE), whole=None)))
        assert ev["verdict"] == "UNKNOWN" and not ev["checks"]["two_independent_providers"]

    def test_a_missing_source_is_never_reported_as_a_source_that_saw_nothing(self):
        ev = derive(providers={"NSE": clean_provider("NSE", BASE_NSE)})
        assert ev["verdict"] == "UNKNOWN"


class TestSourceIntegrityAndFreshness:
    def test_a_tampered_whole_snapshot_prevents_complete(self):
        ev = derive(providers=_providers(nse=_whole_replaced(clean_provider("NSE", BASE_NSE), integrity_ok=False)))
        assert ev["verdict"] == "UNKNOWN" and not ev["checks"]["source_integrity_ok"]

    def test_a_tampered_partition_snapshot_also_prevents_complete(self):
        p = clean_provider("NSE", BASE_NSE)
        parts = [dataclasses.replace(p.parts[0], integrity_ok=False), *p.parts[1:]]
        assert derive(providers=_providers(nse=_with(p, parts=parts)))["checks"]["source_integrity_ok"] is False

    def test_a_snapshot_fetched_before_the_window_end_is_stale(self):
        ev = derive(providers=_providers(bse=_whole_replaced(clean_provider("BSE", BASE_BSE), fetched_at="2026-09-01T00:00:00+00:00")))
        assert ev["verdict"] == "UNKNOWN" and not ev["checks"]["freshness_ok"]

    def test_a_snapshot_fetched_on_the_window_end_day_is_fresh(self):
        ev = derive(providers=_providers(bse=_whole_replaced(clean_provider("BSE", BASE_BSE), fetched_at="2026-09-11T00:00:00+00:00")))
        assert ev["checks"]["freshness_ok"]


class TestWindowCoverage:
    def test_a_whole_query_starting_after_the_window_start_is_partial(self):
        ev = derive(providers=_providers(nse=_whole_replaced(clean_provider("NSE", BASE_NSE), query_from=D(2016, 1, 1))))
        assert ev["verdict"] == "PARTIAL" and not ev["checks"]["window_covered"]

    def test_a_whole_query_ending_before_the_window_end_is_partial(self):
        ev = derive(providers=_providers(bse=_whole_replaced(clean_provider("BSE", BASE_BSE), query_to=D(2026, 9, 1))))
        assert ev["verdict"] == "PARTIAL"

    def test_a_default_mode_snapshot_has_no_query_window_and_cannot_cover_the_window(self):
        ev = derive(providers=_providers(nse=_whole_replaced(clean_provider("NSE", BASE_NSE), query_from=None, query_to=None, fetch_mode="default")))
        assert ev["verdict"] == "PARTIAL"

    def test_a_query_wider_than_the_window_covers_it(self):
        ev = derive(providers=_providers(nse=_whole_replaced(clean_provider("NSE", BASE_NSE), query_from=D(2010, 1, 1))))
        assert ev["checks"]["window_covered"]


def _with_cap_evidence(provider, snapshots):
    obs = cap_observations(snapshots)
    return _with(provider, cap_observations=obs, cap_signals=[cap_signal(s, obs) for s in [provider.whole, *provider.parts]])


class TestTruncation:
    def test_a_recurring_maximum_at_the_whole_response_size_is_a_cap_signal_and_partial(self):
        p = clean_provider("NSE", BASE_NSE)
        twins = [snap("NSE", [{}] * p.whole.record_count, D(2015, 1, 1), D(2018, 12, 31), symbol=f"X{i}") for i in range(2)]
        p = _with_cap_evidence(p, [p.whole, *p.parts, *twins])
        ev = derive(providers=_providers(nse=p))
        assert ev["verdict"] == "PARTIAL" and not ev["checks"]["truncation_evidence_clear"]

    def test_no_wider_probe_means_no_affirmative_evidence_and_therefore_not_complete(self):
        p = clean_provider("NSE", BASE_NSE)
        p = _with_cap_evidence(p, [p.whole, *p.parts])
        ev = derive(providers=_providers(nse=p))
        assert not ev["checks"]["truncation_evidence_clear"] and ev["verdict"] == "PARTIAL"

    def test_missing_cap_signals_are_never_read_as_clear(self):
        ev = derive(providers=_providers(bse=_with(clean_provider("BSE", BASE_BSE), cap_signals=[])))
        assert not ev["checks"]["truncation_evidence_clear"] and ev["verdict"] != "COMPLETE"

    def test_a_signal_on_only_one_partition_is_enough_to_block(self):
        p = clean_provider("NSE", BASE_NSE)
        obs = p.cap_observations
        sigs = [dict(s) for s in p.cap_signals]
        sigs[1] = {"cap_signal": True, "below_observed_max": False, "explanation": "x"}
        assert derive(providers=_providers(nse=_with(p, cap_signals=sigs, cap_observations=obs)))["verdict"] == "PARTIAL"

    def test_the_bundle_states_caps_as_evidence_not_proof(self):
        for s in derive()["sources"]:
            for o in s["cap_observations"].values():
                assert "not proof of a fixed cap" in o["statement"]


class TestPartitionBoundaries:
    def test_unverified_boundary_semantics_prevent_complete_as_unknown(self):
        unverified = BoundarySemantics(None, None, "unverified")
        ev = derive(semantics=unverified)
        assert ev["verdict"] == "UNKNOWN" and not ev["checks"]["boundary_semantics_verified"]
        assert not ev["checks"]["partition_consistent"]

    def test_a_boundary_event_lost_between_partitions_is_partial(self):
        seam = D(2018, 12, 31)
        nse = BASE_NSE + [(seam, nse_row(seam, "Bonus 1:1"))]
        bse = BASE_BSE + [(seam, bse_row(seam, "Bonus issue 1:1"))]
        p = clean_provider("NSE", nse)
        lost = [dataclasses.replace(p.parts[0], rows=tuple(r for r in p.parts[0].rows if r["exDate"] != "31-Dec-2018")), *p.parts[1:]]
        p = _with(p, parts=lost, partition=partition_consistency(p.whole, lost, INCLUSIVE))
        ev = derive(nse, bse, providers=_providers(nse=p, bse=clean_provider("BSE", bse)))
        assert ev["verdict"] == "PARTIAL" and not ev["checks"]["partition_consistent"]
        assert ev["sources"][0]["partition"]["only_in_whole"] == ["2018-12-31"]

    def test_exclusive_semantics_with_touching_partitions_open_a_gap_and_prevent_complete(self):
        excl = BoundarySemantics(True, False, "to exclusive")
        ev = derive(semantics=excl)
        assert ev["verdict"] == "PARTIAL" and ev["sources"][0]["partition"]["gap_days"]

    def test_a_truncated_whole_response_is_partial_via_rows_only_in_the_parts(self):
        p = clean_provider("NSE", BASE_NSE)
        cut = dataclasses.replace(p.whole, rows=p.whole.rows[1:])
        p = _with(p, whole=cut, partition=partition_consistency(cut, p.parts, INCLUSIVE))
        ev = derive(providers=_providers(nse=p))
        assert ev["verdict"] in ("PARTIAL", "CONFLICTED") and not ev["checks"]["partition_consistent"]


class TestRevisions:
    def test_a_revised_dividend_does_not_block(self):
        nse = [(d, nse_row(d, "Dividend - Rs 6 Per Share (Purpose Revised)")) if d == D(2016, 6, 10) else (d, r) for d, r in BASE_NSE]
        ev = derive(nse, BASE_BSE)
        assert ev["verdict"] == "COMPLETE" and ev["revisions"]["blocking_count"] == 0
        assert ev["revisions"]["findings"][0]["code"] == "REVISED_NON_PRICE_AFFECTING"

    def test_a_revised_price_affecting_action_the_other_source_confirms_does_not_block(self):
        nse = [(d, nse_row(d, "Bonus 1:1 (Purpose Revised)")) if d == D(2017, 9, 7) else (d, r) for d, r in BASE_NSE]
        assert derive(nse, BASE_BSE)["verdict"] == "COMPLETE"

    def test_a_revised_price_affecting_action_with_only_class_agreement_prevents_complete(self):
        d = D(2020, 5, 13)
        nse = BASE_NSE + [(d, nse_row(d, "Rights 1:15 @ Premium Rs 1247 (Purpose Revised)"))]
        bse = BASE_BSE + [(d, bse_row(d, "Right Issue of Equity Shares"))]
        ev = derive(nse, bse)
        assert ev["verdict"] == "UNKNOWN" and not ev["checks"]["revisions_resolved"]

    def test_unrecognised_revision_language_on_a_price_affecting_action_prevents_complete(self):
        nse = [(d, nse_row(d, "Bonus 1:1 (Cancelled)")) if d == D(2017, 9, 7) else (d, r) for d, r in BASE_NSE]
        ev = derive(nse, BASE_BSE)
        assert ev["verdict"] == "UNKNOWN" and ev["revisions"]["blocking_count"] == 1

    def test_revision_findings_in_the_bundle_disclaim_point_in_time_knowledge(self):
        nse = [(d, nse_row(d, "Dividend - Rs 6 Per Share (Purpose Revised)")) if d == D(2016, 6, 10) else (d, r) for d, r in BASE_NSE]
        assert "NO claim about when the revision occurred" in derive(nse, BASE_BSE)["revisions"]["findings"][0]["note"]


def _bars(*closes):
    return [(D(2020, 1, 1 + i), float(c)) for i, c in enumerate(closes)]


class TestDiagnosticsCanOnlyBlockNeverEstablish:
    def test_an_unexplained_28_percent_jump_alone_prevents_complete(self):
        ev = derive(jump=scan_close_discontinuities(_bars(100, 60), []))
        assert ev["verdict"] == "UNKNOWN" and not ev["checks"]["blockers_clear"]

    def test_watch_level_jumps_do_not_block(self):
        assert derive(jump=scan_close_discontinuities(_bars(100, 80), []))["verdict"] == "COMPLETE"

    def test_an_uncorroborated_isin_transition_alone_prevents_complete(self):
        isin = isin_transition_corroboration([D(2020, 1, 3)], [D(2020, 1, 1), D(2020, 1, 2), D(2020, 1, 3)], [])
        ev = derive(isin=isin)
        assert ev["verdict"] == "UNKNOWN" and not ev["checks"]["blockers_clear"]

    def test_a_corroborated_isin_transition_does_not_block(self):
        bars = [D(2020, 1, 1), D(2020, 1, 2), D(2020, 1, 3)]
        assert derive(isin=isin_transition_corroboration([D(2020, 1, 3)], bars, [D(2020, 1, 2)]))["verdict"] == "COMPLETE"

    def test_a_perfectly_clean_jump_scan_and_isin_history_cannot_rescue_a_missing_source(self):
        ev = derive(providers={"NSE": clean_provider("NSE", BASE_NSE)})
        assert ev["diagnostics"]["jump_scan"]["blocking_count"] == 0 and ev["diagnostics"]["isin_transitions"]["uncorroborated_count"] == 0
        assert ev["verdict"] != "COMPLETE"

    def test_a_clean_jump_scan_cannot_rescue_a_conflict(self):
        bse = [(d, bse_row(d, "Bonus issue 3:1")) if d == D(2017, 9, 7) else (d, r) for d, r in BASE_BSE]
        assert derive(bse_dated=bse)["verdict"] == "CONFLICTED"

    def test_a_clean_jump_scan_cannot_rescue_an_unverified_boundary(self):
        assert derive(semantics=BoundarySemantics(None, None, "u"))["verdict"] != "COMPLETE"

    def test_diagnostics_are_embedded_as_diagnostic_only(self):
        d = derive()["diagnostics"]
        assert d["jump_scan"]["diagnostic_only"] is True and d["isin_transitions"]["diagnostic_only"] is True


class TestVerdictInvariants:

    BROKEN = {
        "integrity": lambda: dict(providers=_providers(nse=_whole_replaced(clean_provider("NSE", BASE_NSE), integrity_ok=False))),
        "stale": lambda: dict(providers=_providers(bse=_whole_replaced(clean_provider("BSE", BASE_BSE), fetched_at="2026-01-01T00:00:00+00:00"))),
        "window": lambda: dict(providers=_providers(nse=_whole_replaced(clean_provider("NSE", BASE_NSE), query_from=D(2016, 1, 1)))),
        "boundary": lambda: dict(semantics=BoundarySemantics(None, None, "u")),
        "no_bse": lambda: dict(providers={"NSE": clean_provider("NSE", BASE_NSE)}),
        "jump": lambda: dict(jump=scan_close_discontinuities(_bars(100, 50), [])),
        "one_sided": lambda: dict(bse_dated=BASE_BSE[:2]),
        "dup": lambda: dict(nse_dated=BASE_NSE + [BASE_NSE[1]]),
    }

    @pytest.mark.parametrize("name", sorted(BROKEN))
    def test_any_single_defect_prevents_complete(self, name):
        ev = derive(**self.BROKEN[name]())
        assert ev["verdict"] != "COMPLETE"

    @pytest.mark.parametrize("a", sorted(BROKEN))
    @pytest.mark.parametrize("b", sorted(BROKEN))
    def test_no_combination_of_defects_can_produce_complete(self, a, b):
        kwargs = {**self.BROKEN[a](), **self.BROKEN[b]()}
        assert derive(**kwargs)["verdict"] != "COMPLETE"

    def test_complete_is_exactly_all_checks_true_and_no_contradiction(self):
        for name, mk in {"clean": lambda: {}, **self.BROKEN}.items():
            ev = derive(**mk())
            contradiction = (ev["reconciliation"]["conflict_count"] or ev["reconciliation"]["one_sided_count"]
                             or not ev["checks"]["snapshot_diffs_clear"])
            assert (ev["verdict"] == "COMPLETE") == (_all_checks_true(ev) and not contradiction), name

    def test_every_required_check_is_present_in_every_bundle(self):
        for mk in (lambda: {}, self.BROKEN["no_bse"], self.BROKEN["boundary"]):
            assert set(REQUIRED_CHECKS) <= set(derive(**mk())["checks"])

    def test_a_non_complete_verdict_names_its_failed_checks(self):
        ev = derive(**self.BROKEN["integrity"]())
        assert "source_integrity_ok" in ev["reasoning"]
