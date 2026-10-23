"""Phase 3 Stage 3.9 Phase C2: adversarial tests of readiness and the evidence
sheet (pure - no database).

Readiness decides whether a HUMAN may be asked to attest a component's exact
terms. Every case below that must stay blocked, must stay blocked no matter how
much else looks fine. Real row shapes (BAJFINANCE compound, JSWSTEEL's
abbreviated split, KOTAKBANK's 'Bonus 1: 1', RELIANCE rights/demerger) are used
verbatim.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from datetime import date as D

import pytest

from backend.app.research import term_verification as tv
from backend.app.research.corporate_action_evidence import SnapshotRecord
from backend.app.research.exact_terms import InvalidTerms, normalize_terms, terms_equal, try_normalize
from backend.app.research.term_verification import (
    ComponentFacts, HistoryFact, PrimarySourceDoc, ProviderEvidence, RecordFact, SheetInputs, SiblingFact,
    build_evidence_sheet, validate_attester,
)

EX = D(2020, 1, 1)
REC = D(2020, 1, 2)
FETCHED = "2026-09-20T10:00:00+00:00"


def nse_r(subject, ex=EX, rec=REC, symbol="TESTCO"):
    return {"symbol": symbol, "series": "EQ", "comp": "Testco Limited", "isin": "INE000A01018", "subject": subject,
            "exDate": ex.strftime("%d-%b-%Y"), "recDate": rec.strftime("%d-%b-%Y") if rec else "-",
            "faceVal": "1", "caBroadcastDate": None}


def bse_r(purpose, ex=EX, rec=REC, short="TESTCO", scrip=500001):
    return {"scrip_code": scrip, "short_name": short, "long_name": "Testco Limited", "Purpose": purpose,
            "Ex_date": ex.strftime("%d %b %Y"), "RD_Date": rec.strftime("%d %b %Y") if rec else ""}


def snap(provider, rows, *, fetched=FETCHED, symbol="TESTCO", integrity=True, mode="windowed", scrip=500001, tag=""):
    body = json.dumps(rows, sort_keys=True) + provider + fetched + tag
    return SnapshotRecord(
        provider=provider, symbol=symbol, path=f"raw/{provider}/{symbol}.json",
        sha256=hashlib.sha256(body.encode()).hexdigest(), fetched_at=fetched, record_count=len(rows), fetch_mode=mode,
        query_from=D(2015, 1, 1) if mode == "windowed" else None, query_to=D(2026, 9, 20) if mode == "windowed" else None,
        purpose="c1-whole", query_params={"scripcode": str(scrip)} if provider == "BSE" else {}, rows=tuple(rows),
        integrity_ok=integrity,
    )


def facts(**kw):
    base = dict(component_id="comp-1", event_id="ev-1", instrument_id="inst-1", symbol="TESTCO", ex_date=EX, record_date=REC,
                component_type="BONUS", component_status="MACHINE_PARSED", event_status="UNVERIFIED", parse_revision=1,
                current_parse_revision=1, parser_version="stage3.9-c0-v2", raw_fragment="Bonus 1:1", event_raw_subject="Bonus 1:1",
                raw_terms=(1, 1))
    base.update(kw)
    return ComponentFacts(**base)


def pe(provider, *snaps):
    return ProviderEvidence(provider, snaps[0], tuple(snaps))


def inputs(nse_rows=None, bse_rows=None, *, f=None, siblings=(), history=(), records=(), primary=None, nse=None, bse=None):
    nse_rows = [nse_r("Bonus 1:1")] if nse_rows is None else nse_rows
    bse_rows = [bse_r("Bonus issue 1:1")] if bse_rows is None else bse_rows
    return SheetInputs(
        facts=f or facts(), siblings=tuple(siblings), history=tuple(history), records=tuple(records),
        nse=nse if nse is not None else pe("NSE", snap("NSE", nse_rows)),
        bse=bse if bse is not None else pe("BSE", snap("BSE", bse_rows)),
        primary_source=primary,
    )


def sheet(*a, **k):
    return build_evidence_sheet(inputs(*a, **k))


def codes(s):
    return sorted({b["code"] for b in s.blockers})


def tier_codes(s):
    return sorted({r["code"] for r in s.content["readiness"]["tier_reasons"]})


def split_facts(a=10.0, b=2.0, frag=None, **kw):
    kw.setdefault("event_raw_subject", frag or "")
    return facts(component_type="SPLIT", raw_terms=(a, b),
                 raw_fragment=frag or f"Face Value Split (Sub-Division) - From Rs {int(a)}/- Per Share To Rs {int(b)}/- Per Share", **kw)


SPLIT_NSE = "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share"
SPLIT_BSE = "Stock  Split From Rs.10/- to Rs.2/-"


def good_primary(quote="the Board approved a bonus issue of 4 equity shares for every 1 equity share held", terms_ok=True):
    return PrimarySourceDoc("Company outcome of board meeting", "data/raw/corporate_actions_primary/x.pdf", "a" * 64, terms_ok, quote)


class TestNormalCandidates:
    def test_a_plain_1_to_1_bonus_with_two_agreeing_sources_is_ready_tier1(self):
        s = sheet()
        assert s.readiness == tv.READY_TIER1 and s.tier_required == 1 and s.blockers == [] and s.content["readiness"]["verifiable"]

    def test_a_standard_split_is_ready_tier1(self):
        s = sheet([nse_r(SPLIT_NSE)], [bse_r(SPLIT_BSE)], f=split_facts())
        assert s.readiness == tv.READY_TIER1 and s.terms.as_dict() == {"kind": "SPLIT", "term_a": "10", "term_b": "2"}

    def test_the_kotakbank_style_bonus_1_colon_space_1_is_harmless_formatting_not_a_variance(self):
        s = sheet([nse_r("Bonus 1: 1")], [bse_r("Bonus issue 1:1")], f=facts(raw_fragment="Bonus 1: 1", event_raw_subject="Bonus 1: 1"))
        assert s.readiness == tv.READY_TIER1 and tier_codes(s) == []

    @pytest.mark.parametrize("nse_text", [" Bonus 1:1", "BONUS 1:1", "bonus  1:1", "Bonus 1 :1"])
    def test_case_and_whitespace_variants_are_harmless(self, nse_text):
        assert sheet([nse_r(nse_text)]).readiness == tv.READY_TIER1

    def test_real_split_wordings_are_standard(self):
        for nse_text in ("Face Value Split (Sub-Division) - From Rs 5/- Per Share To Re 1/- Per Share",
                         "Face Value Split (Sub-Division) - From Rs 2 Per Share To Rs 1 Per Share",
                         "Face Value Split From Rs.10/- To Rs.2/-"):
            assert tv.wording_is_standard("NSE", "SPLIT", nse_text), nse_text
        assert tv.wording_is_standard("BSE", "SPLIT", "Stock  Split From Rs.2/- to Rs.1/-")

    def test_the_sheet_states_that_two_agreeing_sources_are_transcriptions_not_independent_establishment(self):
        assert "repeated reads of the same endpoint" in sheet().content["evidence"]["NSE"]["independence_note"]
        assert "Tier 1 therefore reduces" in tv.__doc__


class TestDirectionalBonusNeedsTier2:
    def _s(self, **kw):
        return sheet([nse_r("Bonus 4:1")], [bse_r("Bonus issue 4:1")],
                     f=facts(raw_terms=(4, 1), raw_fragment="Bonus 4:1", event_raw_subject="Bonus 4:1"), **kw)

    def test_a_4_to_1_bonus_without_a_primary_source_needs_one(self):
        s = self._s()
        assert s.readiness == tv.READY_TIER2_NEEDS_PRIMARY and tier_codes(s) == ["DIRECTIONAL_BONUS"] and not s.content["readiness"]["verifiable"]

    def test_with_a_valid_hashed_primary_source_it_becomes_ready_tier2(self):
        s = self._s(primary=good_primary())
        assert s.readiness == tv.READY_TIER2 and s.tier_required == 2 and s.content["primary_source"]["valid"]

    @pytest.mark.parametrize("doc,why", [
        (good_primary(terms_ok=False), "recomputed SHA-256"),
        (good_primary(quote="too short"), "too short"),
        (good_primary(quote="the Board approved a bonus issue of 7 equity shares for every 9 equity share held"), "does not contain the attested terms"),
        (dataclasses.replace(good_primary(), sha256="XYZ"), "64-character"),
        (dataclasses.replace(good_primary(), description=" "), "description is empty"),
    ])
    def test_a_primary_source_that_is_not_actually_available_hashed_and_on_point_blocks(self, doc, why):
        s = self._s(primary=doc)
        assert s.readiness == tv.BLOCKED and "PRIMARY_SOURCE_INVALID" in codes(s)
        assert why in json.dumps(s.blockers)

    def test_a_1_to_1_bonus_is_symmetric_so_it_does_not_need_tier2(self):
        assert tier_codes(sheet()) == []

    @pytest.mark.parametrize("a,b", [(1, 2), (2, 1), (3, 2), (10, 1)])
    def test_every_asymmetric_bonus_needs_tier2(self, a, b):
        s = sheet([nse_r(f"Bonus {a}:{b}")], [bse_r(f"Bonus issue {a}:{b}")], f=facts(raw_terms=(a, b)))
        assert "DIRECTIONAL_BONUS" in tier_codes(s)

    def test_a_reverse_split_needs_tier2(self):
        s = sheet([nse_r("Face Value Split (Sub-Division) - From Rs 1/- Per Share To Rs 10/- Per Share")],
                  [bse_r("Stock  Split From Rs.1/- to Rs.10/-")], f=split_facts(1.0, 10.0))
        assert "REVERSE_SPLIT" in tier_codes(s)

    def test_a_tier1_case_with_a_primary_supplied_anyway_stays_tier1(self):
        assert sheet(primary=good_primary()).readiness == tv.READY_TIER1


class TestCompoundBonusAndSplit:
    NSE_COMPOUND = " Bonus 1:1/Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share"
    BSE_ROWS = [bse_r("Bonus issue 1:1"), bse_r(SPLIT_BSE)]

    def _pair(self):
        bonus = facts(component_id="c-bonus", event_id="e1", raw_fragment="Bonus 1:1", event_raw_subject=self.NSE_COMPOUND)
        split = split_facts(component_id="c-split", event_id="e1", event_raw_subject=self.NSE_COMPOUND)
        sib = lambda f: SiblingFact(f.event_id, f.component_id, f.component_type, f.component_status, f.raw_fragment, f.raw_terms)
        return (bonus, [sib(split)]), (split, [sib(bonus)])

    def test_nse_joined_and_bse_separate_representations_reconcile_component_by_component(self):
        for f, sibs in self._pair():
            s = sheet([nse_r(self.NSE_COMPOUND)], self.BSE_ROWS, f=f, siblings=sibs)
            assert s.readiness == tv.READY_TIER1, (f.component_type, s.blockers)
            assert [x["class"] for x in s.content["same_date_siblings"]] and s.content["reconciliation"]["summary"]["term_matches"] == 2

    def test_a_different_class_sibling_is_listed_but_is_not_ambiguity(self):
        (f, sibs), _ = self._pair()
        assert sheet([nse_r(self.NSE_COMPOUND)], self.BSE_ROWS, f=f, siblings=sibs).content["same_date_siblings"][0]["class"] == "SPLIT"

    def test_two_separate_nse_events_on_one_date_as_in_2025_also_reconcile(self):
        s = sheet([nse_r("Bonus 4:1"), nse_r("Face Value Split (Sub-Division) - From Rs 2/- Per Share To Re 1/- Per Share")],
                  [bse_r("Stock  Split From Rs.2/- to Rs.1/-"), bse_r("Bonus issue 4:1")], f=split_facts(2.0, 1.0))
        assert s.readiness == tv.READY_TIER1


class TestConflictsAndPartialAgreementStayBlocked:
    def test_terms_conflict_between_sources(self):
        s = sheet([nse_r("Bonus 1:1")], [bse_r("Bonus issue 2:1")])
        assert s.readiness == tv.BLOCKED and "TERMS_CONFLICT" in codes(s)

    def test_split_ratio_conflict(self):
        s = sheet([nse_r(SPLIT_NSE)], [bse_r("Stock  Split From Rs.10/- to Rs.5/-")], f=split_facts())
        assert "TERMS_CONFLICT" in codes(s)

    def test_class_and_date_agree_but_terms_disagree_can_never_be_verified(self):
        s = sheet([nse_r("Bonus 3:2")], [bse_r("Bonus issue 2:3")], f=facts(raw_terms=(3, 2)))
        assert not s.content["readiness"]["verifiable"] and "TERMS_CONFLICT" in codes(s)

    def test_class_only_agreement_is_not_terms_verification(self):
        s = sheet([nse_r("Face Value Split - details awaited")], [bse_r(SPLIT_BSE)], f=split_facts())
        assert s.readiness == tv.BLOCKED and codes(s) == ["CLASS_ONLY_AGREEMENT"]

    @pytest.mark.parametrize("nse_rows,bse_rows", [([nse_r("Bonus 1:1")], []), ([], [bse_r("Bonus issue 1:1")])])
    def test_one_sided_event(self, nse_rows, bse_rows):
        nse = pe("NSE", snap("NSE", nse_rows))
        bse = pe("BSE", snap("BSE", bse_rows))
        s = build_evidence_sheet(inputs(nse=nse, bse=bse))
        assert s.readiness == tv.BLOCKED and codes(s) == ["ONE_SIDED_EVENT"]

    def test_neither_provider_lists_the_class_on_that_date(self):
        s = sheet([nse_r("Bonus 1:1", ex=D(2021, 1, 1))], [bse_r("Bonus issue 1:1", ex=D(2021, 1, 1))])
        assert "NOT_IN_EVIDENCE" in codes(s)

    def test_component_terms_that_disagree_with_agreeing_evidence(self):
        s = sheet(f=facts(raw_terms=(2, 1)))
        assert "COMPONENT_TERMS_DISAGREE_WITH_EVIDENCE" in codes(s)

    def test_unrecognised_same_date_text_could_hide_an_action(self):
        s = sheet([nse_r("Bonus 1:1"), nse_r("Some Entirely Novel Event")], [bse_r("Bonus issue 1:1")])
        assert s.readiness == tv.BLOCKED and "SAME_DATE_UNCLASSIFIED_EVIDENCE" in codes(s)

    def test_a_same_date_unsupported_sibling_is_shown_but_does_not_change_this_components_own_terms(self):
        s = sheet([nse_r("Bonus 1:1"), nse_r("Scheme Of Arrangement")], [bse_r("Bonus issue 1:1"), bse_r("Spin Off ")])
        assert s.readiness == tv.READY_TIER1
        assert {m["class"] for m in s.content["reconciliation"]["matches_on_date"]} == {"BONUS", "DEMERGER"}

    def test_a_row_with_no_subject_is_unparseable_evidence(self):
        s = sheet([nse_r("Bonus 1:1"), nse_r("")], [bse_r("Bonus issue 1:1")])
        assert "UNPARSEABLE_EVIDENCE_ROW" in codes(s)


class TestUnsupportedAndUnknownStayBlockedEvenWithPerfectEvidence:
    @pytest.mark.parametrize("ctype,nse_text,bse_text", [
        ("RIGHTS", "Rights 1:15 @ Premium Rs 1247", "Right Issue of Equity Shares "),
        ("DEMERGER", "Demerger", "Spin Off "),
        ("MERGER", "Amalgamation", "Amalgamation"),
    ])
    def test_unsupported_classes(self, ctype, nse_text, bse_text):
        s = sheet([nse_r(nse_text)], [bse_r(bse_text)], f=facts(component_type=ctype, raw_fragment=nse_text, raw_terms=None))
        assert s.readiness == tv.BLOCKED and codes(s) == ["UNSUPPORTED_ACTION_TYPE"]
        assert not s.content["readiness"]["verifiable"]

    @pytest.mark.parametrize("frag", ["Buyback", "Scheme Of Arrangement", "Fv Splt Frm Rs 10 To Re 1", "Annual General Meeting and Bonus 1:1", ""])
    def test_unknown_other_text_blocks_even_when_the_fragment_looks_like_a_split(self, frag):
        s = sheet(f=facts(component_type="OTHER", raw_fragment=frag, raw_terms=None))
        assert s.readiness == tv.BLOCKED and "UNCLASSIFIED_COMPONENT" in codes(s)

    @pytest.mark.parametrize("ctype,frag,reason", [("DIVIDEND", "Dividend - Rs 5 Per Share", "DIVIDEND_OUT_OF_SCOPE"),
                                                   ("OTHER", "Annual General Meeting", "BENIGN_MEETING_TEXT"),
                                                   ("OTHER", "Agm", "BENIGN_MEETING_TEXT")])
    def test_dividends_and_benign_meeting_text_are_not_candidates(self, ctype, frag, reason):
        s = sheet(f=facts(component_type=ctype, raw_fragment=frag, raw_terms=None))
        assert s.readiness == tv.NOT_A_CANDIDATE and s.content["readiness"]["not_a_candidate_reason"] == reason


class TestTermsMustBeExactAndWellFormed:
    @pytest.mark.parametrize("ctype,terms", [
        ("BONUS", (0, 1)), ("BONUS", (1, 0)), ("BONUS", (-1, 1)), ("BONUS", (1, -1)), ("BONUS", (1.5, 1)), ("BONUS", (1, None)),
        ("BONUS", (None, None)), ("BONUS", (True, 1)), ("BONUS", ("1", "1")), ("BONUS", (1,)), ("BONUS", (1, 2, 3)),
        ("SPLIT", (10.0, 10.0)), ("SPLIT", (0.0, 2.0)), ("SPLIT", (10.0, -2.0)), ("SPLIT", (float("nan"), 2.0)),
        ("SPLIT", (float("inf"), 2.0)), ("SPLIT", (10.0, 0.0000001)), ("SPLIT", (1e20, 2.0)), ("SPLIT", None),
    ])
    def test_invalid_or_missing_terms_block(self, ctype, terms):
        s = sheet(f=facts(component_type=ctype, raw_terms=terms))
        assert s.readiness == tv.BLOCKED and "TERMS_INVALID" in codes(s) and s.terms is None

    def test_floats_are_read_through_their_decimal_text(self):
        assert normalize_terms("SPLIT", 2.5, 0.5).as_dict() == {"kind": "SPLIT", "term_a": "2.5", "term_b": "0.5"}
        assert normalize_terms("SPLIT", 0.1, 0.3).term_a * 3 == normalize_terms("SPLIT", 0.1, 0.3).term_b
        assert normalize_terms("SPLIT", 10.0, 2.0) == normalize_terms("SPLIT", 10, 2) == normalize_terms("SPLIT", 10.000, 2.0)

    def test_exact_ratio_examples_are_representable_without_computing_a_factor(self):
        assert str(normalize_terms("SPLIT", 10, 2).ratio) == "5"
        assert str(normalize_terms("BONUS", 4, 1).ratio) == "4"
        assert not [n for n in dir(normalize_terms("BONUS", 1, 1)) if "factor" in n.lower()]

    def test_terms_equality_is_exact_and_kind_aware(self):
        a, b = normalize_terms("BONUS", 1, 1), normalize_terms("BONUS", 1, 1)
        assert terms_equal(a, b) and not terms_equal(a, normalize_terms("BONUS", 2, 1)) and not terms_equal(a, None)
        assert not terms_equal(normalize_terms("SPLIT", 2, 1), normalize_terms("BONUS", 2, 1))

    def test_try_normalize_reports_instead_of_raising(self):
        assert try_normalize("BONUS", 0, 1)[0] is None and "positive" in try_normalize("BONUS", 0, 1)[1]
        with pytest.raises(InvalidTerms):
            normalize_terms("RIGHTS", 1, 15)

    def test_a_missing_ex_date_blocks(self):
        s = sheet(f=facts(ex_date=None))
        assert s.readiness == tv.BLOCKED and "MISSING_EX_DATE" in codes(s)


class TestRevisionsAndWording:
    def test_a_revision_marker_forces_tier2(self):
        s = sheet([nse_r("Bonus 1:1 (Purpose Revised)")], f=facts(event_raw_subject="Bonus 1:1 (Purpose Revised)"))
        assert s.readiness == tv.READY_TIER2_NEEDS_PRIMARY and "REVISION_MARKER" in tier_codes(s) and s.content["revision_markers"]

    def test_a_marker_that_appears_only_in_the_evidence_not_in_the_stored_event_text_still_forces_tier2(self):
        s = sheet([nse_r("Bonus 1:1 (Purpose Revised)")], f=facts(event_raw_subject="Bonus 1:1"))
        assert s.readiness == tv.READY_TIER2_NEEDS_PRIMARY and s.content["revision_markers"][0]["provider"] == "NSE"

    def test_a_marker_only_in_the_stored_event_text_forces_tier2_even_if_the_current_evidence_row_is_clean(self):
        s = sheet([nse_r("Bonus 1:1")], f=facts(event_raw_subject="Bonus 1:1 (Purpose Revised)"))
        assert s.readiness == tv.READY_TIER2_NEEDS_PRIMARY and "REVISION_MARKER" in tier_codes(s)

    def test_a_revision_marker_in_the_bse_purpose_forces_tier2(self):
        s = sheet(bse_rows=[bse_r("Bonus issue 1:1 (Purpose Revised)")])
        assert {"REVISION_MARKER", "WORDING_VARIANCE"} <= set(tier_codes(s)) and s.content["revision_markers"][0]["provider"] == "BSE"

    @pytest.mark.parametrize("word", ["Cancelled", "Postponed", "Amended", "Corrected"])
    def test_unrecognised_revision_language_also_forces_tier2(self, word):
        s = sheet([nse_r(f"Bonus 1:1 ({word})")], f=facts(event_raw_subject=f"Bonus 1:1 ({word})"))
        assert "REVISION_MARKER" in tier_codes(s)

    def test_the_jswsteel_abbreviated_split_wording_is_still_a_non_template_wording_but_is_explicit_so_it_stays_tier1(self):
        frag = "Fv Splt Frm Rs 10 To Re 1"
        s = sheet([nse_r(" " + frag)], [bse_r("Stock  Split From Rs.10/- to Rs.1/-")], f=split_facts(10.0, 1.0, frag=frag))
        assert s.readiness == tv.READY_TIER1 and tier_codes(s) == [] and s.blockers == []
        assert s.content["wording"]["NSE"]["standard_template"] is False and s.content["wording"]["BSE"]["standard_template"] is True
        assert s.content["wording"]["NSE"]["wording_class"] == "EXPLICIT_NON_TEMPLATE"

    @pytest.mark.parametrize("nse_text", ["Special Bonus 1:1", "Bonus Issue 1:1", "Bonus 1:1 Per Share", "Bonus shares 1:1"])
    def test_any_word_not_in_the_standard_template_is_a_variance(self, nse_text):
        s = sheet([nse_r(nse_text)])
        assert "WORDING_VARIANCE" in tier_codes(s)

    def test_a_bse_wording_variance_is_also_caught(self):
        s = sheet(bse_rows=[bse_r("Bonus 1:1")])
        assert "WORDING_VARIANCE" in tier_codes(s)

    def test_differing_record_dates_are_a_provider_disagreement(self):
        s = sheet([nse_r("Bonus 1:1", rec=D(2020, 1, 2))], [bse_r("Bonus issue 1:1", rec=D(2020, 1, 3))])
        assert s.readiness == tv.READY_TIER2_NEEDS_PRIMARY and "PROVIDER_DISAGREEMENT" in tier_codes(s)

    def test_a_missing_record_date_on_one_side_is_not_a_disagreement(self):
        assert tier_codes(sheet([nse_r("Bonus 1:1", rec=None)])) == []


class TestAmbiguityTwinsDisputesAndReparse:
    def _twin(self, **kw):
        return SiblingFact("ev-2", "comp-2", "BONUS", "MACHINE_PARSED", "Bonus 1:1", (1, 1), **kw) if False else SiblingFact("ev-2", "comp-2", "BONUS", "MACHINE_PARSED", "Bonus 1:1", (1, 1))

    def test_a_same_class_twin_in_the_events_table_blocks_even_when_identical(self):
        s = sheet(siblings=[self._twin()])
        assert s.readiness == tv.BLOCKED and "AMBIGUOUS_SAME_DATE" in codes(s)

    def test_two_same_class_rows_with_different_terms_in_the_evidence_block(self):
        s = sheet([nse_r("Bonus 1:1"), nse_r("Bonus 2:1")], [bse_r("Bonus issue 1:1"), bse_r("Bonus issue 2:1")])
        assert s.readiness == tv.BLOCKED and "AMBIGUOUS_SAME_DATE" in codes(s)

    def test_a_duplicated_price_affecting_row_blocks(self):
        s = sheet([nse_r("Bonus 1:1"), nse_r("Bonus 1:1")])
        assert s.readiness == tv.BLOCKED and "DUPLICATE_ECONOMIC_EVENT" in codes(s)

    @pytest.mark.parametrize("cs,es", [("DISPUTED", "UNVERIFIED"), ("MACHINE_PARSED", "DISPUTED"), ("DISPUTED", "DISPUTED")])
    def test_a_disputed_component_or_event_blocks(self, cs, es):
        s = sheet(f=facts(component_status=cs, event_status=es))
        assert s.readiness == tv.BLOCKED and "DISPUTED" in codes(s)

    def test_a_reparsed_component_that_is_no_longer_current_blocks(self):
        s = sheet(f=facts(parse_revision=1, current_parse_revision=2))
        assert s.readiness == tv.BLOCKED and "NOT_CURRENT_PARSE_REVISION" in codes(s)


class TestVerificationStateAndLegacyPhaseB:
    def _rec(self, action="VERIFIED", rid="r1", seq=1, revokes=None, a=1, b=1, kind="BONUS"):
        return RecordFact(rid, seq, action, kind, a, b, 1, "Dr Ada Lovelace", "2026-09-21T00:00:00+00:00", revokes)

    def test_a_manually_verified_status_with_no_c2_record_is_identified_as_legacy(self):
        s = sheet(f=facts(component_status="MANUALLY_VERIFIED", event_status="MANUALLY_VERIFIED"))
        assert s.content["verification_state"] == tv.STATE_LEGACY
        assert s.content["active_verification"] is None
        assert s.readiness == tv.READY_TIER1 and s.content["readiness"]["verifiable"]

    def test_legacy_status_on_only_the_event_is_also_identified(self):
        assert sheet(f=facts(component_status="MACHINE_PARSED", event_status="MANUALLY_VERIFIED")).content["verification_state"] == tv.STATE_LEGACY

    def test_a_new_candidate_is_not_verified(self):
        assert sheet().content["verification_state"] == tv.STATE_NOT_VERIFIED

    def test_an_active_matching_c2_record_makes_it_already_verified_and_no_longer_a_candidate(self):
        s = sheet(f=facts(component_status="MANUALLY_VERIFIED", event_status="MANUALLY_VERIFIED"), records=[self._rec()])
        assert s.content["verification_state"] == tv.STATE_C2_VERIFIED
        assert s.readiness == tv.NOT_A_CANDIDATE and s.content["readiness"]["not_a_candidate_reason"] == "ALREADY_VERIFIED_C2"

    def test_a_revoked_record_returns_the_component_to_a_candidate_and_keeps_the_history(self):
        recs = [self._rec("VERIFIED", "r1", 1), self._rec("REVOKED", "r2", 2, revokes="r1")]
        s = sheet(records=recs)
        assert s.content["verification_state"] == tv.STATE_C2_REVOKED and s.readiness == tv.READY_TIER1
        assert [r["action"] for r in s.content["verification_records"]] == ["VERIFIED", "REVOKED"]

    def test_the_active_record_is_derived_from_the_immutable_history(self):
        recs = [self._rec("VERIFIED", "r1", 1), self._rec("REVOKED", "r2", 2, revokes="r1"), self._rec("VERIFIED", "r3", 3)]
        assert tv.active_record(recs).record_id == "r3"
        assert tv.active_record(recs[:2]) is None and tv.active_record([]) is None

    def test_an_active_record_for_different_terms_blocks_rather_than_passing(self):
        s = sheet(records=[self._rec(a=2, b=1)])
        assert s.readiness == tv.BLOCKED and "ACTIVE_RECORD_TERMS_MISMATCH" in codes(s)

    def test_prior_history_is_carried_verbatim_and_ordered(self):
        h = [HistoryFact(9, "COMPONENT", "STATUS_CHANGE", "MACHINE_PARSED", "MANUALLY_VERIFIED", "someone", "t2", "n2"),
             HistoryFact(3, "EVENT", "BASELINE", None, "MANUALLY_VERIFIED", "stage3.9-c0-remediation", "t1", "n1")]
        assert [x["seq"] for x in sheet(history=h).content["prior_verification_history"]] == [3, 9]


class TestEvidenceQualityIntegrityIdentityStalenessStability:
    def test_a_tampered_snapshot_blocks(self):
        s = sheet(nse=pe("NSE", snap("NSE", [nse_r("Bonus 1:1")], integrity=False)))
        assert s.readiness == tv.BLOCKED and "EVIDENCE_INTEGRITY_FAILED" in codes(s)

    def test_evidence_fetched_before_the_ex_date_is_stale(self):
        s = sheet(nse=pe("NSE", snap("NSE", [nse_r("Bonus 1:1")], fetched="2019-12-01T00:00:00+00:00")))
        assert "EVIDENCE_PREDATES_EX_DATE" in codes(s)

    def test_a_missing_provider_blocks(self):
        s = build_evidence_sheet(dataclasses.replace(inputs(), bse=None))
        assert s.readiness == tv.BLOCKED and "EVIDENCE_UNAVAILABLE" in codes(s)

    def test_a_snapshot_whose_symbol_is_another_instrument_blocks(self):
        s = sheet(nse=pe("NSE", snap("NSE", [nse_r("Bonus 1:1")], symbol="OTHERCO")))
        assert "EVIDENCE_IDENTITY_MISMATCH" in codes(s)

    def test_nse_rows_belonging_to_another_symbol_block(self):
        s = sheet([nse_r("Bonus 1:1", symbol="OTHERCO")])
        assert "EVIDENCE_IDENTITY_MISMATCH" in codes(s)

    def test_a_bse_row_for_another_company_blocks(self):
        s = sheet(bse_rows=[bse_r("Bonus issue 1:1", short="OTHERCO")])
        assert "EVIDENCE_IDENTITY_MISMATCH" in codes(s)

    def test_bse_rows_with_inconsistent_scrip_codes_block(self):
        s = sheet(bse_rows=[bse_r("Bonus issue 1:1", scrip=999999)])
        assert "EVIDENCE_IDENTITY_MISMATCH" in codes(s)

    def test_nse_isin_is_informational_only_and_never_identity(self):
        sec = sheet().content["evidence"]["NSE"]["identity"]
        assert "row_isin_informational_only" in sec and sheet([nse_r("Bonus 1:1")]).readiness == tv.READY_TIER1

    def test_bse_identity_is_never_isin_based(self):
        assert "never used for ISIN identity" in sheet().content["evidence"]["BSE"]["identity"]["note"]

    def test_an_earlier_snapshot_that_lacks_the_event_makes_the_evidence_unstable(self):
        latest = snap("NSE", [nse_r("Bonus 1:1")], fetched="2026-09-20T10:00:00+00:00")
        earlier = snap("NSE", [], fetched="2026-09-18T10:00:00+00:00", mode="default")
        s = sheet(nse=ProviderEvidence("NSE", latest, (earlier, latest)))
        assert "EVIDENCE_UNSTABLE" in codes(s)

    def test_snapshots_that_agree_at_the_ex_date_are_stable_and_still_one_source(self):
        a = snap("NSE", [nse_r("Bonus 1:1")], fetched="2026-09-18T10:00:00+00:00", tag="a")
        b = snap("NSE", [nse_r("Bonus 1:1")], fetched="2026-09-20T10:00:00+00:00", tag="b")
        s = sheet(nse=ProviderEvidence("NSE", b, (a, b)))
        assert s.readiness == tv.READY_TIER1 and s.content["evidence"]["NSE"]["observations_identical_at_ex_date"]
        assert len(s.content["evidence"]["NSE"]["observations"]) == 2

    def test_a_text_change_between_snapshots_is_unstable_not_silently_taken(self):
        a = snap("NSE", [nse_r("Bonus 1:1")], fetched="2026-09-18T10:00:00+00:00", tag="a")
        b = snap("NSE", [nse_r("Bonus 2:1")], fetched="2026-09-20T10:00:00+00:00", tag="b")
        assert "EVIDENCE_UNSTABLE" in codes(sheet(nse=ProviderEvidence("NSE", b, (a, b)), f=facts(raw_terms=(2, 1))))


class TestSheetIsDeterministicAndCarriesEverythingTheReviewerNeeds:
    REQUIRED = ["schema", "pit_status", "instrument", "event", "component", "terms", "terms_statement", "verification_state",
                "prior_verification_history", "same_date_siblings", "evidence", "reconciliation", "wording", "revision_markers",
                "primary_source", "readiness"]

    def test_the_sheet_holds_every_required_element(self):
        c = sheet().content
        assert all(k in c for k in self.REQUIRED)
        assert {"instrument_id", "symbol"} <= c["instrument"].keys()
        assert {"event_id", "ex_date", "raw_subject"} <= c["event"].keys() and {"component_id", "class", "status"} <= c["component"].keys()
        for prov in ("NSE", "BSE"):
            e = c["evidence"][prov]
            assert e["rows_at_ex_date"] and e["primary_snapshot"]["sha256"] and e["primary_snapshot"]["fetched_at"] and "identity" in e
        assert {"result", "blockers", "tier_required", "tier_reasons"} <= c["readiness"].keys()
        assert c["reconciliation"]["summary"]["term_matches"] == 1

    def test_raw_text_is_carried_verbatim_not_normalised(self):
        c = sheet([nse_r(" Bonus 1: 1")]).content
        assert c["evidence"]["NSE"]["rows_at_ex_date"][0]["subject"] == " Bonus 1: 1"

    def test_pit_status_is_not_established_on_every_sheet_including_blocked_and_not_candidates(self):
        for s in (sheet(), sheet(f=facts(component_type="RIGHTS", raw_terms=None)), sheet(f=facts(component_type="DIVIDEND", raw_terms=None)),
                  sheet([nse_r("Bonus 1:1")], [bse_r("Bonus issue 2:1")])):
            assert s.content["pit_status"] == "NOT_ESTABLISHED" and "NOT ESTABLISHED" in s.content["pit_note"]

    def test_the_hash_is_deterministic_and_is_the_sha256_of_the_canonical_form(self):
        a, b = sheet(), sheet()
        assert a.sha256 == b.sha256 and a.canonical == b.canonical
        assert a.sha256 == hashlib.sha256(a.canonical.encode()).hexdigest() and len(a.sha256) == 64
        json.loads(a.canonical)

    def test_the_hash_is_independent_of_the_order_of_siblings_history_and_records(self):
        sibs = [SiblingFact("e2", "c2", "SPLIT", "MACHINE_PARSED", "x", (10.0, 2.0)), SiblingFact("e3", "c3", "RIGHTS", "UNVERIFIED", "y", None)]
        hist = [HistoryFact(i, "EVENT", "BASELINE", None, "X", "a", "t", "n") for i in (5, 2, 9)]
        recs = [RecordFact(f"r{i}", i, "VERIFIED" if i == 1 else "REVOKED", "BONUS", 1, 1, 1, "Dr Ada Lovelace", "t", "r1" if i != 1 else None) for i in (1, 2)]
        base = sheet(siblings=sibs, history=hist, records=recs).sha256
        rng = random.Random(3)
        for _ in range(6):
            s2, h2, r2 = sibs[:], hist[:], recs[:]
            rng.shuffle(s2); rng.shuffle(h2); rng.shuffle(r2)
            assert sheet(siblings=s2, history=h2, records=r2).sha256 == base

    @pytest.mark.parametrize("mutate", [
        lambda k: dict(nse_rows=[nse_r("Bonus 1:1", rec=D(2020, 1, 5))]),
        lambda k: dict(bse_rows=[bse_r("Bonus issue 1:1", rec=D(2020, 1, 5))]),
        lambda k: dict(f=facts(component_status="UNVERIFIED")),
        lambda k: dict(f=facts(component_id="comp-2")),
        lambda k: dict(f=facts(instrument_id="inst-2")),
        lambda k: dict(history=[HistoryFact(1, "EVENT", "BASELINE", None, "X", "a", "t", "n")]),
        lambda k: dict(nse=pe("NSE", snap("NSE", [nse_r("Bonus 1:1")], fetched="2026-09-21T10:00:00+00:00"))),
    ])
    def test_any_change_to_what_the_reviewer_saw_changes_the_hash(self, mutate):
        assert sheet(**mutate(None)).sha256 != sheet().sha256

    def test_a_sheet_for_a_different_component_has_a_different_hash_so_it_cannot_be_reused(self):
        assert sheet(f=facts(component_id="comp-A")).sha256 != sheet(f=facts(component_id="comp-B")).sha256

    def test_the_sheet_never_contains_an_adjustment_factor(self):
        text = sheet().canonical.lower()
        assert "adjustment_factor" not in text and "factor" not in text


class TestAttesterMustBeAHumanNotASystem:
    @pytest.mark.parametrize("name", ["Dr Ada Lovelace", "R. Sharma", "Priya Nair", "Botha Nel", "Testa Rossi", "Autumn Rao", "ithead@example.org"])
    def test_named_humans_are_accepted(self, name):
        assert validate_attester(name) == name.strip()

    @pytest.mark.parametrize("name", [
        "", "  ", "ab", "123", "unspecified", "system", "System User", "auto", "automation", "automated-runner", "machine", "llm",
        "Assistant Code", "gpt-5-agent", "automated", "assistant", "Model 5", "stage3.9-c1", "stage3.9-c0-remediation", "script", "bot", "cron",
        "batch-job", "tool", "agent", "llm", "gpt-5", "pytest", "test", "tests", "root", "admin", "none", "null", "unknown", "human", "  SYSTEM  ",
    ])
    def test_reserved_system_and_model_identities_are_rejected(self, name):
        with pytest.raises(tv.AttesterRejected):
            validate_attester(name)

    @pytest.mark.parametrize("bad", [None, 7, ["Dr Ada"], b"Dr Ada"])
    def test_non_strings_are_rejected(self, bad):
        with pytest.raises(tv.AttesterRejected):
            validate_attester(bad)


class TestRealCandidateShapes:

    def test_reliance_rights_is_class_only_and_unsupported(self):
        s = sheet([nse_r(" Rights 1:15 @ Premium Rs 1247")], [bse_r("Right Issue of Equity Shares ")],
                  f=facts(component_type="RIGHTS", raw_fragment="Rights 1:15 @ Premium Rs 1247", raw_terms=(1, 15)))
        assert s.readiness == tv.BLOCKED and codes(s) == ["UNSUPPORTED_ACTION_TYPE"]

    def test_reliance_demerger_has_no_terms_and_is_unsupported(self):
        s = sheet([nse_r("Demerger")], [bse_r("Spin Off ")], f=facts(component_type="DEMERGER", raw_fragment="Demerger", raw_terms=None))
        assert s.readiness == tv.BLOCKED

    def test_bajfinance_2025_split_2_to_1_is_tier1_while_its_same_date_4_to_1_bonus_needs_tier2(self):
        rows_n = [nse_r("Bonus 4:1"), nse_r("Face Value Split (Sub-Division) - From Rs 2/- Per Share To Re 1/- Per Share")]
        rows_b = [bse_r("Bonus issue 4:1"), bse_r("Stock  Split From Rs.2/- to Rs.1/-")]
        assert sheet(rows_n, rows_b, f=split_facts(2.0, 1.0)).readiness == tv.READY_TIER1
        assert sheet(rows_n, rows_b, f=facts(raw_terms=(4, 1), raw_fragment="Bonus 4:1")).readiness == tv.READY_TIER2_NEEDS_PRIMARY


JSW_NSE = "Fv Splt Frm Rs 10 To Re 1"
JSW_BSE = "Stock  Split From Rs.10/- to Rs.1/-"


def jsw(nse_text=JSW_NSE, bse_text=JSW_BSE, *, terms=(10.0, 1.0), **kw):
    return sheet([nse_r(" " + nse_text)], [bse_r(bse_text)], f=split_facts(terms[0], terms[1], frag=nse_text), **kw)


class TestExplicitSplitWording:

    def test_jswsteel_2017_is_ready_tier1_and_needs_no_primary_document(self):
        s = jsw()
        assert s.readiness == tv.READY_TIER1 and s.tier_required == 1 and s.blockers == []
        assert s.content["readiness"]["verifiable"] and s.terms.as_dict() == {"kind": "SPLIT", "term_a": "10", "term_b": "1"}
        assert s.content["primary_source"] is None

    def test_the_variance_is_still_recorded_and_visible_to_the_reviewer(self):
        w = jsw().content["wording"]["NSE"]
        assert w["standard_template"] is False and w["wording_class"] == "EXPLICIT_NON_TEMPLATE"
        assert w["explicit_reading"] == {"old_face_value": "10", "new_face_value": "1"}
        (note,) = jsw().content["readiness"]["notes"]
        assert note["code"] == "NON_TEMPLATE_WORDING_READ_EXPLICITLY" and note["provider"] == "NSE" and note["old_face_value"] == "10"

    def test_it_can_be_verified_without_tier2_evidence(self):
        assert jsw().tier_required == 1 and not jsw().content["readiness"]["tier_reasons"]

    def test_standard_split_wording_is_unchanged_and_carries_no_note(self):
        s = sheet([nse_r(SPLIT_NSE)], [bse_r(SPLIT_BSE)], f=split_facts())
        assert s.readiness == tv.READY_TIER1 and "notes" not in s.content["readiness"]
        assert "wording_class" not in s.content["wording"]["NSE"]

    @pytest.mark.parametrize("nse_text,terms", [
        ("Fv Splt Frm Rs 10 To Re 1", (10.0, 1.0)), ("Fv Split Frm Rs 5 To Re 1", (5.0, 1.0)), ("FV SPLT FRM RS 10 TO RE 2", (10.0, 2.0)),
        ("Face Value Splt From Rs 2 To Re 1", (2.0, 1.0)), ("Fv Splt Frm Rs.10/- To Re.1/-", (10.0, 1.0)),
        ("Fv Splt Frm Rs 10 Per Share To Re 1 Per Share", (10.0, 1.0)), ("Sub-Division Frm Rs 10 To Re 1", (10.0, 1.0)),
        ("Fv Subdivision From Rs 5 To Rs 1", (5.0, 1.0)), ("Fv Splt Frm Rs 2.5 To Re 0.5", (2.5, 0.5)),
    ])
    def test_the_explicit_grammar_accepts_abbreviations_and_formatting_that_state_both_values_and_direction(self, nse_text, terms):
        assert tv.explicit_split_terms(nse_text) == (tv.Decimal(str(terms[0])), tv.Decimal(str(terms[1])))
        assert jsw(nse_text, terms=terms, bse_text=f"Stock  Split From Rs.{terms[0]:g}/- to Rs.{terms[1]:g}/-").readiness == tv.READY_TIER1

    def test_both_providers_may_use_explicit_non_template_wording(self):
        s = jsw(bse_text="Face Value Split From Rs.10/- To Rs.1/-")
        assert s.readiness == tv.READY_TIER1 and len(s.content["readiness"]["notes"]) == 2
        assert {w["wording_class"] for w in s.content["wording"].values()} == {"EXPLICIT_NON_TEMPLATE"}

    def test_a_provider_wording_the_parser_cannot_read_never_reconciles_so_it_blocks(self):
        s = jsw(bse_text="Split Frm Rs 10 To Re 1")
        assert s.readiness == tv.BLOCKED and "CLASS_ONLY_AGREEMENT" in codes(s)

    def test_a_readable_but_wordy_bse_text_is_not_explicit_and_stays_tier2(self):
        s = jsw(bse_text="Sub-Division of Equity Shares From Rs.10/- to Re.1/-")
        assert tv.explicit_split_terms("Sub-Division of Equity Shares From Rs.10/- to Re.1/-") is None
        assert s.readiness == tv.READY_TIER2_NEEDS_PRIMARY and tier_codes(s) == ["WORDING_VARIANCE"]
        assert s.content["wording"]["BSE"]["wording_class"] == "VARIANT"

    @pytest.mark.parametrize("text", [
        "Fv Splt 10 1",
        "Fv Splt Rs 10 To Re 1",
        "Fv Splt Frm Rs 10 Re 1",
        "Split 10:1", "Fv Splt 10:1",
        "Fv Splt Frm 10 To 5 To 1",
        "Fv Splt Frm Rs 10",
        "Fv Splt Frm Rs 10 To Re",
        "Fv Splt To Re 1 Frm Rs 10",
        "Frm Rs 10 To Re 1",
        "Fv Splt Frm Rs 10 To Re 1 Cum",
        "Special Fv Splt Frm Rs 10 To Re 1",
        "Fv Splt Frm Rs 10 To Re 1 (Purpose Revised)",
        "Fv Splt Frm Rs 10 To Re 1 @ Premium",
        "Bonus 1:1 Fv Splt Frm Rs 10 To Re 1",
    ])
    def test_ambiguous_or_incomplete_wording_is_not_explicit_and_stays_tier2_or_blocked(self, text):
        assert tv.explicit_split_terms(text) is None
        s = jsw(text)
        assert s.readiness in (tv.READY_TIER2_NEEDS_PRIMARY, tv.BLOCKED)
        assert "WORDING_VARIANCE" in tier_codes(s) or s.readiness == tv.BLOCKED

    def test_an_extra_action_after_a_comma_is_a_separate_component_and_is_never_hidden(self):
        s = jsw("Fv Splt Frm Rs 10 To Re 1, Bonus")
        assert tv.explicit_split_terms("Fv Splt Frm Rs 10 To Re 1, Bonus") is None
        classes = {m["class"]: m["status"] for m in s.content["reconciliation"]["matches_on_date"]}
        assert classes.get("BONUS") == "ONE_SIDED_NSE"

    def test_an_explicit_reading_that_differs_from_the_components_terms_does_not_qualify(self):
        s = sheet([nse_r(" Fv Splt Frm Rs 10 To Re 5")], [bse_r(JSW_BSE)], f=split_facts(10.0, 1.0, frag="Fv Splt Frm Rs 10 To Re 5"))
        assert s.readiness == tv.BLOCKED and "TERMS_CONFLICT" in codes(s)

    def test_an_explicit_reading_the_components_own_terms_do_not_match_is_blocked(self):
        s = jsw(terms=(10.0, 2.0), bse_text="Stock  Split From Rs.10/- to Rs.2/-")
        assert s.readiness == tv.BLOCKED and codes(s)

    def test_a_reverse_reading_is_explicit_but_a_reverse_split_still_needs_tier2(self):
        s = sheet([nse_r(" Fv Splt Frm Rs 1 To Re 10")], [bse_r("Stock  Split From Rs.1/- to Rs.10/-")],
                  f=split_facts(1.0, 10.0, frag="Fv Splt Frm Rs 1 To Re 10"))
        assert s.readiness == tv.READY_TIER2_NEEDS_PRIMARY and tier_codes(s) == ["REVERSE_SPLIT"]

    def test_conflicting_terms_between_the_sources_still_block(self):
        s = jsw(bse_text="Stock  Split From Rs.10/- to Rs.2/-")
        assert s.readiness == tv.BLOCKED and "TERMS_CONFLICT" in codes(s)

    def test_a_revision_marker_still_forces_tier2(self):
        s = sheet([nse_r(" Fv Splt Frm Rs 10 To Re 1 (Purpose Revised)")], [bse_r(JSW_BSE)],
                  f=split_facts(10.0, 1.0, frag=JSW_NSE, event_raw_subject="Fv Splt Frm Rs 10 To Re 1 (Purpose Revised)"))
        assert s.readiness == tv.READY_TIER2_NEEDS_PRIMARY and "REVISION_MARKER" in tier_codes(s)

    def test_a_revision_marker_only_in_the_stored_event_text_still_forces_tier2(self):
        s = sheet([nse_r(" " + JSW_NSE)], [bse_r(JSW_BSE)],
                  f=split_facts(10.0, 1.0, frag=JSW_NSE, event_raw_subject=JSW_NSE + " (Purpose Revised)"))
        assert "REVISION_MARKER" in tier_codes(s) and s.readiness == tv.READY_TIER2_NEEDS_PRIMARY

    def test_a_record_date_disagreement_still_forces_tier2(self):
        s = sheet([nse_r(" " + JSW_NSE, rec=D(2020, 1, 2))], [bse_r(JSW_BSE, rec=D(2020, 1, 3))], f=split_facts(10.0, 1.0, frag=JSW_NSE))
        assert s.readiness == tv.READY_TIER2_NEEDS_PRIMARY and tier_codes(s) == ["PROVIDER_DISAGREEMENT"]

    def test_a_same_date_ambiguous_twin_still_blocks(self):
        twin = SiblingFact("ev-2", "comp-2", "SPLIT", "MACHINE_PARSED", "Fv Splt Frm Rs 5 To Re 1", (5.0, 1.0))
        s = sheet([nse_r(" " + JSW_NSE)], [bse_r(JSW_BSE)], f=split_facts(10.0, 1.0, frag=JSW_NSE), siblings=[twin])
        assert s.readiness == tv.BLOCKED and "AMBIGUOUS_SAME_DATE" in codes(s)

    def test_two_split_rows_on_the_date_in_the_evidence_still_block(self):
        s = sheet([nse_r(" " + JSW_NSE), nse_r(" Fv Splt Frm Rs 5 To Re 1")], [bse_r(JSW_BSE), bse_r("Stock  Split From Rs.5/- to Rs.1/-")],
                  f=split_facts(10.0, 1.0, frag=JSW_NSE))
        assert s.readiness == tv.BLOCKED and "AMBIGUOUS_SAME_DATE" in codes(s)

    def test_missing_or_incomplete_terms_still_block(self):
        for raw in (None, (10.0, None), (None, 1.0), (10.0,), (10.0, 0.0), (10.0, 10.0)):
            s = sheet([nse_r(" " + JSW_NSE)], [bse_r(JSW_BSE)], f=dataclasses.replace(split_facts(10.0, 1.0, frag=JSW_NSE), raw_terms=raw))
            assert s.readiness == tv.BLOCKED and "TERMS_INVALID" in codes(s), raw

    def test_class_only_agreement_with_unreadable_text_still_blocks(self):
        s = sheet([nse_r(" Fv Splt")], [bse_r(JSW_BSE)], f=split_facts(10.0, 1.0, frag="Fv Splt"))
        assert s.readiness == tv.BLOCKED

    def test_one_sided_evidence_still_blocks(self):
        s = build_evidence_sheet(inputs(nse=pe("NSE", snap("NSE", [nse_r(" " + JSW_NSE)])), bse=pe("BSE", snap("BSE", [])),
                                        f=split_facts(10.0, 1.0, frag=JSW_NSE)))
        assert s.readiness == tv.BLOCKED and "ONE_SIDED_EVENT" in codes(s)

    @pytest.mark.parametrize("cs,es", [("DISPUTED", "UNVERIFIED"), ("MACHINE_PARSED", "DISPUTED")])
    def test_a_disputed_component_or_event_still_blocks(self, cs, es):
        s = sheet([nse_r(" " + JSW_NSE)], [bse_r(JSW_BSE)], f=split_facts(10.0, 1.0, frag=JSW_NSE, component_status=cs, event_status=es))
        assert s.readiness == tv.BLOCKED and "DISPUTED" in codes(s)

    def test_a_reparsed_component_still_blocks(self):
        s = sheet([nse_r(" " + JSW_NSE)], [bse_r(JSW_BSE)], f=split_facts(10.0, 1.0, frag=JSW_NSE, parse_revision=1, current_parse_revision=2))
        assert s.readiness == tv.BLOCKED and "NOT_CURRENT_PARSE_REVISION" in codes(s)

    def test_evidence_integrity_and_identity_failures_still_block(self):
        assert "EVIDENCE_INTEGRITY_FAILED" in codes(sheet(nse=pe("NSE", snap("NSE", [nse_r(" " + JSW_NSE)], integrity=False)),
                                                          f=split_facts(10.0, 1.0, frag=JSW_NSE), bse_rows=[bse_r(JSW_BSE)]))
        assert "EVIDENCE_IDENTITY_MISMATCH" in codes(sheet([nse_r(" " + JSW_NSE)], [bse_r(JSW_BSE, short="OTHERCO")], f=split_facts(10.0, 1.0, frag=JSW_NSE)))

    @pytest.mark.parametrize("ctype,frag", [("RIGHTS", "Rights 1:15 @ Premium Rs 1247"), ("DEMERGER", "Demerger"), ("MERGER", "Amalgamation")])
    def test_unsupported_actions_stay_blocked(self, ctype, frag):
        s = sheet([nse_r(frag)], [bse_r("Spin Off ")], f=facts(component_type=ctype, raw_fragment=frag, raw_terms=None))
        assert s.readiness == tv.BLOCKED and codes(s) == ["UNSUPPORTED_ACTION_TYPE"]

    @pytest.mark.parametrize("frag", ["Fv Splt Frm Rs 10 To Re 1", "Some Unknown Event", "Buyback"])
    def test_unknown_or_split_looking_OTHER_components_stay_blocked(self, frag):
        s = sheet(f=facts(component_type="OTHER", raw_fragment=frag, raw_terms=None))
        assert s.readiness == tv.BLOCKED and "UNCLASSIFIED_COMPONENT" in codes(s)

    def test_bonus_wording_the_parser_cannot_read_blocks(self):
        s = sheet([nse_r("Bonus 1 for 1")])
        assert s.readiness == tv.BLOCKED and "WORDING_VARIANCE" in tier_codes(s)

    @pytest.mark.parametrize("nse_text", ["Special Bonus 1:1", "Bonus Issue 1:1", "Bonus 1:1 Per Share", "Bonus shares 1:1"])
    def test_bonus_wording_variance_is_untouched_and_still_needs_tier2(self, nse_text):
        s = sheet([nse_r(nse_text)])
        assert "WORDING_VARIANCE" in tier_codes(s) and s.readiness == tv.READY_TIER2_NEEDS_PRIMARY
        assert "wording_class" in s.content["wording"]["NSE"] and s.content["wording"]["NSE"]["wording_class"] == "VARIANT"

    def test_a_directional_bonus_still_needs_tier2_even_with_perfectly_standard_wording(self):
        s = sheet([nse_r("Bonus 4:1")], [bse_r("Bonus issue 4:1")], f=facts(raw_terms=(4, 1), raw_fragment="Bonus 4:1", event_raw_subject="Bonus 4:1"))
        assert s.readiness == tv.READY_TIER2_NEEDS_PRIMARY and tier_codes(s) == ["DIRECTIONAL_BONUS"]

    @pytest.mark.parametrize("bad", ["", "   ", None, "10", "to", "from", "Fv Splt Frm To"])
    def test_degenerate_inputs_are_not_explicit(self, bad):
        assert tv.explicit_split_terms(bad) is None

    def test_only_the_jswsteel_sheet_changes_every_other_real_shaped_sheet_is_byte_identical(self):
        standard = [sheet(), sheet([nse_r("Bonus 1: 1")]), sheet([nse_r(SPLIT_NSE)], [bse_r(SPLIT_BSE)], f=split_facts()),
                    sheet([nse_r("Bonus 4:1")], [bse_r("Bonus issue 4:1")], f=facts(raw_terms=(4, 1), raw_fragment="Bonus 4:1"))]
        for sh in standard:
            assert "notes" not in sh.content["readiness"] and all("wording_class" not in w for w in sh.content["wording"].values())


class TestExplicitReaderIsStrictSlotBySlot:

    @pytest.mark.parametrize("text", [
        "Fv Splt Frm Rs 10 To Re 1 ?", "Fv Splt Frm Rs 10 To Re 1 %", "Fv Splt Frm Rs 10 To Re 1 @", "Fv Splt Frm Rs 10 To Re 1 :",
        "Fv Splt Frm Rs 10 To Re 1 *", "Fv Splt Frm Rs 10 To Re 1 &", "Fv Splt; Frm Rs 10 To Re 1", "Fv Splt Frm Rs 10, To Re 1",
    ])
    def test_stray_punctuation_is_never_explicit(self, text):
        assert tv.explicit_split_terms(text) is None

    @pytest.mark.parametrize("text", [
        "Fv Splt Frm Per Rs 10 To Re 1",
        "Fv Splt Frm Rs 10 Rs To Re 1",
        "Fv Splt Frm Rs 10 To Per Re 1",
        "Fv Splt Frm Rs 10 To Re 1 Rs",
        "Fv Splt Frm Rs 10 To Re 1 To",
        "Fv Splt Frm Rs 10 To To Re 1",
        "Frm Fv Splt Frm Rs 10 To Re 1",
        "Fv Splt Frm Frm Rs 10 To Re 1",
        "Fv Splt To Re 1 Frm Rs 10",
        "10 Fv Splt Frm Rs 10 To Re 1",
    ])
    def test_every_slot_rejects_words_that_do_not_belong_in_it(self, text):
        assert tv.explicit_split_terms(text) is None

    @pytest.mark.parametrize("text", ["Fv Splt Frm Rs 10 To Re 1 5", "Fv Splt Frm Rs 10 To Re 1 5 6", "Fv Splt Frm Rs 10 5 To Re 1"])
    def test_a_third_number_is_never_explicit_even_with_one_from_and_one_to(self, text):
        assert tv.explicit_split_terms(text) is None

    def test_the_guarantee_that_makes_the_from_to_and_order_checks_redundant_holds_exhaustively(self):
        import itertools

        pool = ["fv", "splt", "frm", "rs", "10", "to", "re", "1", "per", "share"]
        accepted = 0
        for n in range(3, 8):
            for combo in itertools.permutations(pool, n):
                text = " ".join(combo)
                out = tv.explicit_split_terms(text)
                if out is None:
                    continue
                accepted += 1
                toks = text.split()
                numbers = [t for t in toks if t.isdigit()]
                assert toks.count("frm") == 1 and toks.count("to") == 1 and len(numbers) == 2, text
                first, second = (i for i, t in enumerate(toks) if t.isdigit())
                assert toks.index("frm") < first < toks.index("to") < second, text
                assert out == (tv.Decimal(numbers[0]), tv.Decimal(numbers[1])), text
        assert accepted > 0

    def test_the_readers_numbers_must_equal_the_components_terms_a_forced_disagreement_is_not_explicit(self, monkeypatch):
        monkeypatch.setattr(tv, "explicit_split_terms", lambda fragment: (tv.Decimal(10), tv.Decimal(5)))
        s = jsw()
        assert s.readiness == tv.READY_TIER2_NEEDS_PRIMARY and tier_codes(s) == ["WORDING_VARIANCE"]
        assert s.content["wording"]["NSE"]["wording_class"] == "VARIANT"

    def test_the_exception_is_split_only_even_if_the_reader_were_fooled_into_accepting_a_bonus(self, monkeypatch):
        monkeypatch.setattr(tv, "explicit_split_terms", lambda fragment: (tv.Decimal(1), tv.Decimal(1)))
        s = sheet([nse_r("Special Bonus 1:1")])
        assert "WORDING_VARIANCE" in tier_codes(s) and s.readiness == tv.READY_TIER2_NEEDS_PRIMARY
