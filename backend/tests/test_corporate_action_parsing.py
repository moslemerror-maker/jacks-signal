from __future__ import annotations

from backend.app.data.normalizers.corporate_actions import parse_subject_components


class TestSimpleDividend:
    def test_dividend_amount_extracted(self):
        components = parse_subject_components("Dividend - Rs 6 Per Share")
        assert len(components) == 1
        c = components[0]
        assert c.component_type == "DIVIDEND"
        assert c.cash_amount_per_share == 6.0
        assert c.currency == "INR"
        assert c.confidently_parsed is True

    def test_decimal_dividend_amount(self):
        components = parse_subject_components("Dividend - Rs 5.5 Per Share")
        assert components[0].cash_amount_per_share == 5.5


class TestSimpleBonus:
    def test_bonus_ratio_extracted(self):
        components = parse_subject_components("Bonus 1:1")
        assert len(components) == 1
        c = components[0]
        assert c.component_type == "BONUS"
        assert c.ratio_new == 1
        assert c.ratio_existing == 1
        assert c.confidently_parsed is True


class TestSimpleSplit:
    def test_modern_phrasing_with_slash_dash(self):
        subject = "Face Value Split (Sub-Division) - From Rs 5/- Per Share To Re 1/- Per Share"
        components = parse_subject_components(subject)
        assert len(components) == 1
        c = components[0]
        assert c.component_type == "SPLIT"
        assert c.face_value_from == 5.0
        assert c.face_value_to == 1.0
        assert c.confidently_parsed is True

    def test_older_phrasing_without_per_share_suffix(self):
        subject = "Face Value Split From Rs.10/- To Rs.5/-"
        components = parse_subject_components(subject)
        assert len(components) == 1
        c = components[0]
        assert c.component_type == "SPLIT"
        assert c.face_value_from == 10.0
        assert c.face_value_to == 5.0
        assert c.confidently_parsed is True

    def test_phrasing_without_slash_dash_suffix(self):
        subject = "Face Value Split (Sub-Division) - From Rs 2 Per Share To Rs 1 Per Share"
        components = parse_subject_components(subject)
        c = components[0]
        assert c.component_type == "SPLIT"
        assert c.face_value_from == 2.0
        assert c.face_value_to == 1.0
        assert c.confidently_parsed is True

    def test_bajfinance_2025_split(self):
        subject = "Face Value Split (Sub-Division) - From Rs 2/- Per Share To Re 1/- Per Share"
        components = parse_subject_components(subject)
        c = components[0]
        assert c.face_value_from == 2.0
        assert c.face_value_to == 1.0


class TestCompoundSubjects:
    def test_bajfinance_2016_bonus_plus_split_is_two_distinct_components(self):
        subject = (
            "Bonus 1:1/Face Value Split (Sub-Division) - From Rs 10/- "
            "Per Share To Rs 2/- Per Share"
        )
        components = parse_subject_components(subject)
        assert len(components) == 2

        bonus, split = components
        assert bonus.component_type == "BONUS"
        assert bonus.ratio_new == 1
        assert bonus.ratio_existing == 1
        assert bonus.confidently_parsed is True

        assert split.component_type == "SPLIT"
        assert split.face_value_from == 10.0
        assert split.face_value_to == 2.0
        assert split.confidently_parsed is True

    def test_currency_slash_dash_is_never_mistaken_for_a_component_separator(self):
        subject = (
            "Bonus 1:1/Face Value Split (Sub-Division) - From Rs 10/- "
            "Per Share To Rs 2/- Per Share"
        )
        components = parse_subject_components(subject)
        assert len(components) == 2
        for c in components:
            assert c.raw_text_fragment.strip() not in ("-", "/-")

    def test_agm_plus_dividend_the_administrative_fragment_is_preserved_as_other(self):
        subject = "Annual General Meeting/Dividend - Rs 7.30 Per Share"
        components = parse_subject_components(subject)
        assert len(components) == 2

        agm, dividend = components
        assert agm.component_type == "OTHER"
        assert agm.confidently_parsed is False
        assert agm.raw_text_fragment == "Annual General Meeting"

        assert dividend.component_type == "DIVIDEND"
        assert dividend.cash_amount_per_share == 7.30
        assert dividend.confidently_parsed is True


class TestUnrecognizedTextNeverGuessed:
    def test_completely_unfamiliar_text_becomes_other_with_no_structured_fields(self):
        components = parse_subject_components("Some Entirely Novel Corporate Event Type")
        assert len(components) == 1
        c = components[0]
        assert c.component_type == "OTHER"
        assert c.confidently_parsed is False
        assert c.face_value_from is None
        assert c.face_value_to is None
        assert c.ratio_new is None
        assert c.ratio_existing is None
        assert c.cash_amount_per_share is None

    def test_split_keyword_present_but_ratio_text_unparseable_stays_unverified(self):
        components = parse_subject_components("Face Value Split - details awaited")
        c = components[0]
        assert c.component_type == "SPLIT"
        assert c.confidently_parsed is False
        assert c.face_value_from is None
        assert c.face_value_to is None

    def test_merger_and_demerger_keywords_are_classified_but_never_given_a_ratio(self):
        merger = parse_subject_components("Scheme of Amalgamation - Merger with XYZ Ltd")[0]
        assert merger.component_type == "MERGER"
        assert merger.confidently_parsed is False
        assert merger.ratio_new is None

        demerger = parse_subject_components("Scheme of Arrangement - Demerger")[0]
        assert demerger.component_type == "DEMERGER"
        assert demerger.confidently_parsed is False


class TestInferredPriceRatiosNeverPopulateAnything:
    def test_parser_never_looks_at_price_data_at_all(self):
        import inspect

        sig = inspect.signature(parse_subject_components)
        assert list(sig.parameters.keys()) == ["subject"]
