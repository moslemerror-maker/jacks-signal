from __future__ import annotations

import datetime as dt

import pytest

from backend.app.data.backfill.chunk_planner import BackfillChunk, describe_plan, plan_chunks

D = dt.date


class TestSingleSecurityWindows:
    def test_exactly_90_days_is_one_chunk(self):
        chunks = plan_chunks([(2885, "RELIANCE")], overall_from=D(2026, 1, 1), overall_to=D(2026, 3, 31), chunk_days=90)
        assert (D(2026, 1, 1) - D(2026, 1, 1)).days == 0
        assert (D(2026, 3, 31) - D(2026, 1, 1)).days == 89
        assert len(chunks) == 1
        assert chunks[0] == BackfillChunk(2885, "RELIANCE", D(2026, 1, 1), D(2026, 3, 31))

    def test_91_days_is_one_full_chunk_plus_one_day_remainder(self):
        chunks = plan_chunks([(2885, "RELIANCE")], overall_from=D(2026, 1, 1), overall_to=D(2026, 4, 1), chunk_days=90)
        assert len(chunks) == 2
        assert chunks[0] == BackfillChunk(2885, "RELIANCE", D(2026, 1, 1), D(2026, 3, 31))
        assert chunks[1] == BackfillChunk(2885, "RELIANCE", D(2026, 4, 1), D(2026, 4, 1))

    def test_180_days_is_exactly_two_full_chunks_no_remainder(self):
        chunks = plan_chunks([(2885, "RELIANCE")], overall_from=D(2026, 1, 1), overall_to=D(2026, 6, 29), chunk_days=90)
        assert (D(2026, 6, 29) - D(2026, 1, 1)).days == 179
        assert len(chunks) == 2
        assert chunks[0].to_date == D(2026, 3, 31) and chunks[1].from_date == D(2026, 4, 1)

    def test_one_year_is_four_full_chunks_plus_remainder(self):
        overall_from, overall_to = D(2025, 9, 26), D(2026, 9, 25)
        chunks = plan_chunks([(2885, "RELIANCE")], overall_from=overall_from, overall_to=overall_to, chunk_days=90)
        assert len(chunks) == 5
        full = [c for c in chunks if (c.to_date - c.from_date).days == 89]
        assert len(full) == 4
        remainder = chunks[-1]
        assert remainder.to_date == overall_to
        assert (remainder.to_date - remainder.from_date).days < 89

    def test_short_range_shorter_than_chunk_size_is_one_chunk(self):
        chunks = plan_chunks([(2885, "RELIANCE")], overall_from=D(2026, 9, 1), overall_to=D(2026, 9, 5), chunk_days=90)
        assert len(chunks) == 1
        assert chunks[0] == BackfillChunk(2885, "RELIANCE", D(2026, 9, 1), D(2026, 9, 5))

    def test_single_day_range(self):
        chunks = plan_chunks([(2885, "RELIANCE")], overall_from=D(2026, 9, 25), overall_to=D(2026, 9, 25), chunk_days=90)
        assert chunks == [BackfillChunk(2885, "RELIANCE", D(2026, 9, 25), D(2026, 9, 25))]


class TestBoundariesAndOrdering:
    def test_no_overlap_and_no_gap_across_chunks(self):
        chunks = plan_chunks([(2885, "RELIANCE")], overall_from=D(2025, 1, 1), overall_to=D(2026, 12, 31), chunk_days=90)
        for a, b in zip(chunks, chunks[1:]):
            assert b.from_date == a.to_date + dt.timedelta(days=1)
            assert b.from_date > a.to_date

    def test_chunks_are_chronologically_ordered(self):
        chunks = plan_chunks([(2885, "RELIANCE")], overall_from=D(2025, 1, 1), overall_to=D(2026, 12, 31), chunk_days=90)
        assert [c.from_date for c in chunks] == sorted(c.from_date for c in chunks)

    def test_final_chunk_covers_exactly_overall_to(self):
        overall_to = D(2026, 9, 25)
        chunks = plan_chunks([(2885, "RELIANCE")], overall_from=D(2025, 1, 1), overall_to=overall_to, chunk_days=90)
        assert chunks[-1].to_date == overall_to

    def test_deterministic_reproducible_output(self):
        a = plan_chunks([(2885, "RELIANCE"), (11536, "TCS")], overall_from=D(2025, 9, 26), overall_to=D(2026, 9, 25), chunk_days=90)
        b = plan_chunks([(2885, "RELIANCE"), (11536, "TCS")], overall_from=D(2025, 9, 26), overall_to=D(2026, 9, 25), chunk_days=90)
        assert a == b


class TestMultipleSecurities:
    def test_all_of_one_securitys_chunks_come_before_the_next_security(self):
        chunks = plan_chunks([(2885, "RELIANCE"), (11536, "TCS")], overall_from=D(2025, 9, 26), overall_to=D(2026, 9, 25), chunk_days=90)
        security_sequence = [c.security_id for c in chunks]
        first_tcs_index = security_sequence.index(11536)
        assert all(s == 2885 for s in security_sequence[:first_tcs_index])
        assert all(s == 11536 for s in security_sequence[first_tcs_index:])

    def test_input_security_order_is_preserved(self):
        chunks = plan_chunks([(11536, "TCS"), (2885, "RELIANCE")], overall_from=D(2026, 1, 1), overall_to=D(2026, 3, 31), chunk_days=90)
        assert [c.security_id for c in chunks] == [11536, 2885]

    def test_total_chunk_count_is_securities_times_windows(self):
        chunks = plan_chunks([(1, "A"), (2, "B"), (3, "C")], overall_from=D(2025, 9, 26), overall_to=D(2026, 9, 25), chunk_days=90)
        assert len(chunks) == 3 * 5


class TestValidation:
    def test_overall_to_before_overall_from_raises(self):
        with pytest.raises(ValueError, match="overall_to"):
            plan_chunks([(2885, "RELIANCE")], overall_from=D(2026, 9, 25), overall_to=D(2026, 1, 1))

    def test_chunk_days_less_than_one_raises(self):
        with pytest.raises(ValueError, match="chunk_days"):
            plan_chunks([(2885, "RELIANCE")], overall_from=D(2026, 1, 1), overall_to=D(2026, 3, 31), chunk_days=0)

    def test_empty_universe_produces_empty_plan(self):
        assert plan_chunks([], overall_from=D(2026, 1, 1), overall_to=D(2026, 3, 31)) == []


class TestNoNetworkOrDbCalls:
    def test_module_imports_neither_requests_nor_a_db_session(self):
        import backend.app.data.backfill.chunk_planner as mod
        source = mod.__file__
        text = open(source).read()
        assert "import requests" not in text and "get_session" not in text and "Session(" not in text


class TestDryRunDescribe:
    def test_describe_plan_makes_no_io_and_reports_correct_counts(self):
        chunks = plan_chunks([(2885, "RELIANCE"), (11536, "TCS")], overall_from=D(2025, 9, 26), overall_to=D(2026, 9, 25), chunk_days=90)
        desc = describe_plan(chunks, preview=2)
        assert desc["total_chunks"] == 10
        assert desc["distinct_securities"] == 2
        assert desc["securities"] == [(2885, "RELIANCE"), (11536, "TCS")]
        assert len(desc["first_chunks"]) == 2 and len(desc["last_chunks"]) == 2
        assert desc["first_chunks"][0]["from_date"] == D(2025, 9, 26)
        assert desc["last_chunks"][-1]["to_date"] == D(2026, 9, 25)

    def test_describe_plan_on_empty_list(self):
        desc = describe_plan([])
        assert desc == {"total_chunks": 0, "distinct_securities": 0, "securities": [], "first_chunks": [], "last_chunks": []}
