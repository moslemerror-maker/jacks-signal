from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.data.normalizers.bse_corporate_actions import parse_bse_purpose

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "real_bse_purposes.json").read_text()
)


def _as_dict(components):
    return [{"class": c.action_class, "terms": list(c.terms) if c.terms else None} for c in components]


class TestRealBsePurposeFixtures:
    @pytest.mark.parametrize("entry", _FIXTURE, ids=[e["purpose"].strip()[:50] for e in _FIXTURE])
    def test_parse_matches_the_reviewed_expectation(self, entry):
        assert _as_dict(parse_bse_purpose(entry["purpose"])) == entry["expected"]

    def test_no_real_bse_purpose_is_unclassified(self):
        for entry in _FIXTURE:
            assert all(c["class"] != "UNCLASSIFIED" for c in entry["expected"]), entry["purpose"]

    def test_every_economic_class_bse_uses_is_represented(self):
        seen = {c["class"] for e in _FIXTURE for c in e["expected"]}
        assert {"SPLIT", "BONUS", "RIGHTS", "DIVIDEND", "DEMERGER"} <= seen


class TestBseVocabulary:
    def test_spin_off_is_bses_word_for_the_demerger_class(self):
        (c,) = parse_bse_purpose("Spin Off ")
        assert c.action_class == "DEMERGER" and c.terms is None

    def test_rights_carries_no_terms_and_that_is_a_property_of_the_source_not_a_parse_gap(self):
        (c,) = parse_bse_purpose("Right Issue of Equity Shares ")
        assert c.action_class == "RIGHTS" and c.terms is None

    def test_double_space_split_text_parses(self):
        (c,) = parse_bse_purpose("Stock  Split From Rs.10/- to Rs.2/-")
        assert c.action_class == "SPLIT" and c.terms == (10.0, 2.0)

    def test_bonus_is_new_over_existing(self):
        (c,) = parse_bse_purpose("Bonus issue 4:1")
        assert c.terms == (4, 1)


class TestUnrecognisedBseTextIsNeverDroppedOrGuessed:
    def test_unknown_purpose_is_unclassified_with_its_text_preserved(self):
        (c,) = parse_bse_purpose("Buy Back of Shares")
        assert c.action_class == "UNCLASSIFIED" and c.fragment == "Buy Back of Shares"

    def test_empty_purpose_is_unclassified_not_silently_ignored(self):
        assert parse_bse_purpose("   ")[0].action_class == "UNCLASSIFIED"

    def test_administrative_meeting_rows_are_the_only_thing_dropped(self):
        assert parse_bse_purpose("Annual General Meeting") == []
        assert parse_bse_purpose("AGM") == []

    def test_a_split_keyword_without_parseable_ratio_keeps_the_class_but_no_terms(self):
        (c,) = parse_bse_purpose("Stock Split - details awaited")
        assert c.action_class == "SPLIT" and c.terms is None

    def test_multi_class_purpose_is_separated_or_left_unclassified_never_collapsed(self):
        comps = parse_bse_purpose("Bonus issue 1:1 and Stock Split From Rs.10/- to Rs.2/-")
        assert [c.action_class for c in comps] == ["BONUS", "SPLIT"]
        (c,) = parse_bse_purpose("Bonus cum Split")
        assert c.action_class == "UNCLASSIFIED"

    def test_bse_parser_produces_no_adjustment_factor_of_any_kind(self):
        for c in parse_bse_purpose("Stock  Split From Rs.10/- to Rs.2/-"):
            assert set(vars(c)) == {"action_class", "terms", "fragment"}
