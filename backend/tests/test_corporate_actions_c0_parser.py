from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from backend.app.data.normalizers import corporate_actions as ca
from backend.app.data.normalizers.corporate_actions import (
    PARSER_VERSION,
    parse_subject_components,
)

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "real_corporate_action_subjects.json").read_text()
)


def _as_expected_dict(c) -> dict:
    d = {"type": c.component_type, "confident": c.confidently_parsed}
    for k in (
        "face_value_from", "face_value_to", "ratio_new", "ratio_existing",
        "cash_amount_per_share", "currency",
    ):
        v = getattr(c, k)
        if v is not None:
            d[k] = v
    d["fragment"] = c.raw_text_fragment
    return d


class TestRealStringFixtures:
    @pytest.mark.parametrize("entry", _FIXTURE, ids=[e["subject"][:60] for e in _FIXTURE])
    def test_parse_matches_the_reviewed_expectation(self, entry):
        actual = [_as_expected_dict(c) for c in parse_subject_components(entry["subject"])]
        assert actual == entry["expected"]

    def test_fixture_covers_every_action_class_seen_in_real_data(self):
        seen = {c["type"] for e in _FIXTURE for c in e["expected"]}
        assert {"SPLIT", "BONUS", "RIGHTS", "DIVIDEND", "DEMERGER", "OTHER"} <= seen

    def test_no_price_affecting_keyword_is_ever_left_as_other(self):
        for entry in _FIXTURE:
            for c in entry["expected"]:
                if c["type"] == "OTHER":
                    for name, pattern in ca._CLASS_PATTERNS:
                        assert not pattern.search(c["fragment"]), (
                            f"{name} keyword left as OTHER: {c['fragment']!r}"
                        )


class TestDiscoveredRealFailures:
    def test_jswsteel_abbreviated_split_is_a_split_not_other(self):
        (c,) = parse_subject_components("Fv Splt Frm Rs 10 To Re 1")
        assert c.component_type == "SPLIT"
        assert (c.face_value_from, c.face_value_to) == (10.0, 1.0)
        assert c.confidently_parsed is True

    @pytest.mark.parametrize(
        "subject",
        ["Splt From Rs 10 To Rs 5", "Sub-Divn From Rs 10 To Rs 1", "Stock Split From Rs.10/- to Rs.2/-"],
    )
    def test_split_phrasing_variants_are_recognized(self, subject):
        assert parse_subject_components(subject)[0].component_type == "SPLIT"

    def test_the_word_division_inside_sub_division_is_not_a_dividend(self):
        (c,) = parse_subject_components("Sub-Division - From Rs 10 To Rs 2")
        assert c.component_type == "SPLIT"

    @pytest.mark.parametrize(
        "subject, amount",
        [
            ("Dividend - Rs 11/- Per Share", 11.0),
            ("Annual General Meeting/Dividend- Re 0.70 Per Share", 0.7),
            ("Agm/Div-Re.1/- Per Share", 1.0),
            ("Annual General Meeting/ Divdend - Re 0.50/- Per Share", 0.5),
            ("Dividend - Rs 1.50 Per Sh", 1.5),
            ("Interim Dividend - Rs 10.50/- Per Share (Purpose Revised)", 10.5),
        ],
    )
    def test_real_dividend_abbreviations_now_parse_confidently(self, subject, amount):
        dividends = [c for c in parse_subject_components(subject) if c.component_type == "DIVIDEND"]
        assert len(dividends) == 1
        assert dividends[0].cash_amount_per_share == amount
        assert dividends[0].confidently_parsed is True

    def test_percentage_dividend_is_classified_but_never_given_an_invented_amount(self):
        dividends = [
            c for c in parse_subject_components("Agm/Dividend - 140%") if c.component_type == "DIVIDEND"
        ]
        assert len(dividends) == 1
        assert dividends[0].cash_amount_per_share is None
        assert dividends[0].confidently_parsed is False

    def test_rights_is_never_confident_even_when_the_ratio_parses(self):
        (c,) = parse_subject_components("Rights 1:15 @ Premium Rs 1247")
        assert (c.ratio_new, c.ratio_existing) == (1, 15)
        assert c.confidently_parsed is False
        assert "issue price" in c.note


class TestMultiClassFragmentsAreNeverCollapsedToOneKeyword:
    def test_bonus_and_split_joined_by_and_yield_two_components(self):
        comps = parse_subject_components(
            "Bonus 1:1 And Face Value Split - From Rs 10/- Per Share To Rs 2/- Per Share"
        )
        assert [c.component_type for c in comps] == ["BONUS", "SPLIT"]
        assert comps[0].ratio_new == 1 and comps[1].face_value_to == 2.0

    def test_inseparable_multi_class_fragment_is_preserved_whole_as_other(self):
        (c,) = parse_subject_components("Bonus cum Split 1:1")
        assert c.component_type == "OTHER"
        assert c.confidently_parsed is False
        assert c.raw_text_fragment == "Bonus cum Split 1:1"
        assert "multiple distinct action classes" in c.note

    def test_slash_currency_notation_is_not_a_component_boundary(self):
        comps = parse_subject_components(
            "Bonus 1:1/Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share"
        )
        assert len(comps) == 2


def _price_factor(component) -> Fraction:
    if component.component_type == "SPLIT":
        return Fraction(int(component.face_value_to), int(component.face_value_from))
    if component.component_type == "BONUS":
        return Fraction(component.ratio_existing, component.ratio_new + component.ratio_existing)
    raise AssertionError(f"no price factor for {component.component_type}")


def _event_factor(subject: str) -> Fraction:
    factor = Fraction(1)
    for c in parse_subject_components(subject):
        factor *= _price_factor(c)
    return factor


_BAJ_2016 = (
    "Bonus 1:1/Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share"
)
_BAJ_2025_SPLIT = "Face Value Split (Sub-Division) - From Rs 2/- Per Share To Re 1/- Per Share"
_BAJ_2025_BONUS = "Bonus 4:1"


class TestBajfinanceCompoundExactRational:

    def test_2016_event_split_10_to_2_is_one_fifth(self):
        split = next(c for c in parse_subject_components(_BAJ_2016) if c.component_type == "SPLIT")
        assert _price_factor(split) == Fraction(1, 5)

    def test_2016_event_bonus_1_to_1_is_one_half(self):
        bonus = next(c for c in parse_subject_components(_BAJ_2016) if c.component_type == "BONUS")
        assert _price_factor(bonus) == Fraction(1, 2)

    def test_2016_event_combined_is_one_tenth_not_one_hundredth(self):
        assert _event_factor(_BAJ_2016) == Fraction(1, 10)
        assert _event_factor(_BAJ_2016) != Fraction(1, 100)

    def test_2025_same_date_events_split_2_to_1_and_bonus_4_to_1_combine_to_one_tenth(self):
        (split,) = parse_subject_components(_BAJ_2025_SPLIT)
        (bonus,) = parse_subject_components(_BAJ_2025_BONUS)
        assert _price_factor(split) == Fraction(1, 2)
        assert _price_factor(bonus) == Fraction(1, 5)
        assert _price_factor(split) * _price_factor(bonus) == Fraction(1, 10)

    def test_cumulative_factor_for_a_bar_before_both_events_is_one_hundredth(self):
        e2016 = _event_factor(_BAJ_2016)
        e2025 = _event_factor(_BAJ_2025_SPLIT) * _event_factor(_BAJ_2025_BONUS)
        assert e2016 * e2025 == Fraction(1, 100)
        assert e2025 * e2016 == Fraction(1, 100)
        assert e2016 * e2025 != e2016

    def test_a_bar_between_the_two_events_carries_only_the_2025_factor(self):
        assert _event_factor(_BAJ_2025_SPLIT) * _event_factor(_BAJ_2025_BONUS) == Fraction(1, 10)

    def test_exact_rational_arithmetic_has_no_drift_where_floats_would(self):
        exact = Fraction(1, 5) * Fraction(1, 2) * Fraction(1, 2) * Fraction(1, 5)
        assert exact == Fraction(1, 100)
        assert float(exact) == 0.01
        assert Fraction(0.2) != Fraction(1, 5)

    def test_other_verified_events_have_the_expected_exact_factors(self):
        assert _event_factor("Face Value Split (Sub-Division) - From Rs 5/- Per Share To Re 1/- Per Share") == Fraction(1, 5)
        assert _event_factor("Face Value Split (Sub-Division) - From Rs 2 Per Share To Rs 1 Per Share") == Fraction(1, 2)
        assert _event_factor("Fv Splt Frm Rs 10 To Re 1") == Fraction(1, 10)


def test_parser_version_is_distinct_from_the_phase_b_parser():
    assert PARSER_VERSION == "stage3.9-c0-v2"
    assert PARSER_VERSION != "stage3.9-phase-b-v1"
