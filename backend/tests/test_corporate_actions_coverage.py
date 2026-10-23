from __future__ import annotations

from datetime import date

from backend.app.db.models import CorporateActionCoverageStatus as Status
from backend.app.research.corporate_actions_coverage import (
    SourceCoverageObservation,
    assess_single_source_coverage,
    reconcile_coverage,
)


def _obs(**overrides) -> SourceCoverageObservation:
    defaults = dict(
        provider="NSE",
        record_count=5,
        window_start=date(2015, 1, 1),
        window_end=date(2026, 9, 11),
        known_page_size_cap=20,
        earliest_ex_date=date(2015, 1, 1),
        latest_ex_date=date(2026, 6, 1),
        instrument_listing_date=None,
    )
    defaults.update(overrides)
    return SourceCoverageObservation(**defaults)


class TestATruncatedSourceCannotBeTreatedAsComplete:
    def test_record_count_at_the_known_cap_is_partial(self):
        obs = _obs(record_count=20, known_page_size_cap=20)
        status, reason = assess_single_source_coverage(obs)
        assert status == Status.PARTIAL
        assert "truncat" in reason.lower()

    def test_record_count_above_the_cap_is_still_partial(self):
        obs = _obs(record_count=25, known_page_size_cap=20)
        status, _ = assess_single_source_coverage(obs)
        assert status == Status.PARTIAL

    def test_a_clean_looking_single_source_is_never_complete(self):
        obs = _obs(record_count=5, known_page_size_cap=20)
        status, reason = assess_single_source_coverage(obs)
        assert status != Status.COMPLETE
        assert status == Status.UNKNOWN


class TestJswsteelMissingActionIsPartialOrUnknown:
    def test_jswsteel_style_nse_window_is_partial(self):
        obs = _obs(
            provider="NSE", record_count=20, known_page_size_cap=20,
            earliest_ex_date=date(2024, 7, 9), latest_ex_date=date(2026, 7, 7),
        )
        status, reason = assess_single_source_coverage(obs)
        assert status == Status.PARTIAL
        assert "NSE" in reason


class TestConflictingSourcesProduceConflicted:
    def test_disagreement_between_sources_is_conflicted_regardless_of_individual_statuses(self):
        obs_a = _obs(provider="NSE", record_count=5, known_page_size_cap=20)
        obs_b = _obs(provider="BSE", record_count=8, known_page_size_cap=None)
        status_a, reason_a = assess_single_source_coverage(obs_a)
        status_b, reason_b = assess_single_source_coverage(obs_b)

        overall, reason = reconcile_coverage(
            [(obs_a, status_a, reason_a), (obs_b, status_b, reason_b)],
            cross_source_agreement=False,
        )
        assert overall == Status.CONFLICTED
        assert "disagree" in reason.lower()

    def test_conflicted_overrides_even_when_neither_source_shows_truncation(self):
        obs_a = _obs(provider="NSE", record_count=3, known_page_size_cap=20)
        obs_b = _obs(provider="BSE", record_count=3, known_page_size_cap=None)
        status_a, reason_a = assess_single_source_coverage(obs_a)
        status_b, reason_b = assess_single_source_coverage(obs_b)
        overall, _ = reconcile_coverage(
            [(obs_a, status_a, reason_a), (obs_b, status_b, reason_b)],
            cross_source_agreement=False,
        )
        assert overall == Status.CONFLICTED


class TestCoverageStatusIsNeverAdjustmentEligibilityByItself:
    def test_no_status_to_eligible_function_exists(self):
        import backend.app.research.corporate_actions_coverage as mod

        assert not hasattr(mod, "is_eligible_for_adjustment")
        assert not [n for n in dir(mod) if "eligible" in n.lower()]

    def test_reconcile_requires_real_cross_validation_for_complete(self):
        obs_a = _obs(provider="NSE", record_count=3, known_page_size_cap=20)
        obs_b = _obs(provider="BSE", record_count=3, known_page_size_cap=None)
        status_a, reason_a = assess_single_source_coverage(obs_a)
        status_b, reason_b = assess_single_source_coverage(obs_b)

        overall, reason = reconcile_coverage(
            [(obs_a, status_a, reason_a), (obs_b, status_b, reason_b)],
            cross_source_agreement=True,
        )
        assert overall == Status.COMPLETE

    def test_a_single_source_can_never_reconcile_to_complete_even_with_agreement_flag_true(self):
        obs_a = _obs(provider="NSE", record_count=3, known_page_size_cap=20)
        status_a, reason_a = assess_single_source_coverage(obs_a)
        overall, reason = reconcile_coverage(
            [(obs_a, status_a, reason_a)], cross_source_agreement=True
        )
        assert overall != Status.COMPLETE
        assert overall == Status.UNKNOWN

    def test_partial_signal_blocks_complete_even_with_agreement(self):
        obs_a = _obs(provider="NSE", record_count=20, known_page_size_cap=20)
        obs_b = _obs(provider="BSE", record_count=22, known_page_size_cap=None)
        status_a, reason_a = assess_single_source_coverage(obs_a)
        status_b, reason_b = assess_single_source_coverage(obs_b)
        overall, reason = reconcile_coverage(
            [(obs_a, status_a, reason_a), (obs_b, status_b, reason_b)],
            cross_source_agreement=True,
        )
        assert overall == Status.PARTIAL


class TestZeroRecordsNeverSilentlyMeansNoEventsExisted:
    def test_zero_records_is_unknown_not_complete(self):
        obs = _obs(record_count=0, earliest_ex_date=None, latest_ex_date=None)
        status, reason = assess_single_source_coverage(obs)
        assert status == Status.UNKNOWN
        assert "cannot distinguish" in reason
