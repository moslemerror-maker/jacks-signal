from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from backend.app.research.corporate_action_diagnostics import (
    BLOCKER_THRESHOLD,
    ISIN_MAX_LAG_SESSIONS,
    WATCH_THRESHOLD,
    isin_transition_corroboration,
    scan_close_discontinuities,
)
from backend.app.research.corporate_action_evidence import (
    BoundarySemantics,
    SnapshotRecord,
    _row_date,
    cap_observations,
    cap_signal,
    comparable_window,
    discover_snapshots,
    infer_boundary_semantics,
    load_snapshot,
    partition_consistency,
    snapshot_coverage_window,
)
from backend.app.research.corporate_action_reconciliation import PRICE_AFFECTING
from backend.tests.c1_helpers import INCLUSIVE, D, nse_row, snap

FIXTURES = Path(__file__).parent / "fixtures"
EXCLUSIVE_TO = BoundarySemantics(True, False, "test: to exclusive")
EXCLUSIVE_FROM = BoundarySemantics(False, True, "test: from exclusive")
UNVERIFIED = BoundarySemantics(None, None, "test: unverified")


def _write_snapshot(directory: Path, rows: list, *, provider="NSE", symbol="TESTCO", meta_extra=None, name="s.json"):
    directory.mkdir(parents=True, exist_ok=True)
    body = json.dumps(rows).encode()
    path = directory / name
    path.write_bytes(body)
    meta = {"provider": provider, "symbol": symbol, "sha256": hashlib.sha256(body).hexdigest(),
            "fetched_at": "2026-09-20T10:00:00+00:00", "record_count": len(rows), "query_params": {}}
    meta.update(meta_extra or {})
    path.with_suffix(".json.meta.json").write_text(json.dumps(meta))
    return path


class TestSnapshotIntegrity:
    def test_untouched_snapshot_verifies(self, tmp_path):
        p = _write_snapshot(tmp_path, [nse_row(D(2017, 9, 7), "Bonus 1:1")])
        s = load_snapshot(p)
        assert s.integrity_ok and s.record_count == 1 and s.fetch_mode == "default"

    def test_a_snapshot_edited_after_capture_is_flagged_not_trusted(self, tmp_path):
        p = _write_snapshot(tmp_path, [nse_row(D(2017, 9, 7), "Bonus 1:1")])
        p.write_bytes(json.dumps([nse_row(D(2017, 9, 7), "Bonus 9:1")]).encode())
        assert load_snapshot(p).integrity_ok is False

    def test_windowed_metadata_is_read_from_query_window(self, tmp_path):
        p = _write_snapshot(tmp_path, [], meta_extra={"fetch_mode": "windowed", "purpose": "c1-whole",
                            "query_window": {"from_date": "2015-01-01", "to_date": "2026-09-20"}})
        s = load_snapshot(p)
        assert (s.query_from, s.query_to, s.fetch_mode, s.purpose) == (D(2015, 1, 1), D(2026, 9, 20), "windowed", "c1-whole")

    def test_legacy_bse_snapshot_window_is_recovered_from_its_request_params(self, tmp_path):
        p = _write_snapshot(tmp_path, [], provider="BSE", meta_extra={"query_params": {"Fdate": "20150101", "TDate": "20260920"}})
        s = load_snapshot(p)
        assert (s.provider, s.query_from, s.query_to, s.fetch_mode) == ("BSE", D(2015, 1, 1), D(2026, 9, 20), "windowed")

    def test_discovery_ignores_sidecars_and_files_without_a_sidecar(self, tmp_path):
        d = tmp_path / "TESTCO"
        _write_snapshot(d, [], name="a.json")
        (d / "orphan.json").write_text("[]")
        assert [Path(s.path).name for s in discover_snapshots(tmp_path, "TESTCO")] == ["a.json"]
        assert discover_snapshots(tmp_path, "MISSING") == []

    def test_ref_dict_carries_full_provenance_and_no_rows(self):
        s = snap("NSE", [nse_row(D(2017, 9, 7), "Bonus 1:1")], D(2015, 1, 1), D(2026, 9, 20))
        ref = s.as_ref_dict()
        assert {"provider", "symbol", "path", "sha256", "fetched_at", "record_count", "query_from", "query_to",
                "query_params", "purpose", "fetch_mode", "integrity_ok"} <= ref.keys()
        assert "rows" not in ref


def _counts_fixture():
    return json.loads((FIXTURES / "real_c1_response_counts.json").read_text())


def _snap_from_fixture(r):
    default = r["from"] is None
    return SnapshotRecord(
        provider=r["provider"], symbol=r["symbol"], path="fixture", sha256=r["sha256"],
        fetched_at="2026-09-20T00:00:00+00:00", record_count=r["count"],
        fetch_mode="default" if default else "windowed",
        query_from=None if default else D.fromisoformat(r["from"]),
        query_to=None if default else D.fromisoformat(r["to"]),
        purpose=r["purpose"], query_params={}, rows=(), integrity_ok=True,
    )


class TestObservedResponseCapsUsingRealCounts:
    @pytest.fixture
    def real(self):
        snaps = [_snap_from_fixture(r) for r in _counts_fixture()]
        return snaps, cap_observations(snaps)

    def test_nse_default_mode_shows_the_row_limit_signature_of_20_across_five_symbols(self, real):
        _, obs = real
        o = obs["NSE:default"]
        assert o["max_observed_count"] == 20 and o["distinct_queries_at_max"] == 5

    def test_nse_windowed_maximum_is_28_from_one_query_only(self, real):
        _, obs = real
        o = obs["NSE:windowed"]
        assert o["max_observed_count"] == 28 and o["distinct_queries_at_max"] == 1

    def test_bse_windowed_maximum_is_26_from_one_query_only(self, real):
        _, obs = real
        assert obs["BSE:windowed"]["max_observed_count"] == 26 and obs["BSE:windowed"]["distinct_queries_at_max"] == 1

    def test_default_mode_responses_carry_a_cap_signal(self, real):
        snaps, obs = real
        default = [s for s in snaps if s.fetch_mode == "default"]
        assert default and all(cap_signal(s, obs)["cap_signal"] for s in default)

    def test_windowed_28_is_not_a_cap_signal_because_it_did_not_recur(self, real):
        snaps, obs = real
        top = next(s for s in snaps if s.provider == "NSE" and s.record_count == 28)
        sig = cap_signal(top, obs)
        assert not sig["cap_signal"] and not sig["below_observed_max"]
        assert "no evidence ruling one out" in sig["explanation"]

    def test_smaller_windowed_responses_are_affirmatively_below_the_observed_maximum(self, real):
        snaps, obs = real
        small = [s for s in snaps if s.fetch_mode == "windowed" and s.record_count < obs[f"{s.provider}:windowed"]["max_observed_count"]]
        assert small and all(cap_signal(s, obs)["below_observed_max"] and not cap_signal(s, obs)["cap_signal"] for s in small)

    def test_the_statement_describes_evidence_never_proof(self, real):
        _, obs = real
        for o in obs.values():
            assert "not proof of a fixed cap" in o["statement"]

    def test_modes_are_never_pooled(self):
        a = snap("NSE", [{}] * 20, None, None, symbol="A", mode="default")
        b = snap("NSE", [{}] * 5, D(2015, 1, 1), D(2016, 1, 1), symbol="B")
        obs = cap_observations([a, b])
        assert obs["NSE:default"]["max_observed_count"] == 20 and obs["NSE:windowed"]["max_observed_count"] == 5

    def test_the_same_maximum_from_two_windows_of_one_symbol_is_two_distinct_queries(self):
        a = snap("NSE", [{}] * 9, D(2015, 1, 1), D(2018, 12, 31), symbol="A")
        b = snap("NSE", [{}] * 9, D(2019, 1, 1), D(2022, 12, 31), symbol="A")
        assert cap_observations([a, b])["NSE:windowed"]["distinct_queries_at_max"] == 2

    def test_an_identical_query_repeated_is_one_distinct_query_not_two(self):
        a = snap("NSE", [{}] * 9, D(2015, 1, 1), D(2018, 12, 31), symbol="A", fetched="2026-09-20T10:00:00+00:00")
        b = snap("NSE", [{}] * 9, D(2015, 1, 1), D(2018, 12, 31), symbol="A", fetched="2026-09-21T10:00:00+00:00", purpose="again")
        o = cap_observations([a, b])["NSE:windowed"]
        assert o["responses_at_max"] == 2 and o["distinct_queries_at_max"] == 1
        assert not cap_signal(a, cap_observations([a, b]))["cap_signal"]


def _probe_dates(provider, key):
    rows = json.loads((FIXTURES / "real_c1_boundary_probes.json").read_text())[provider][key]
    return [_row_date(provider, r) for r in rows]


class TestBoundarySemanticsFromRealProbes:
    DAY = D(2017, 9, 7)

    @pytest.mark.parametrize("provider", ["NSE", "BSE"])
    def test_both_providers_are_inclusive_at_both_ends(self, provider):
        sem = infer_boundary_semantics(
            self.DAY, _probe_dates(provider, "single-day"), _probe_dates(provider, "from"),
            _probe_dates(provider, "to"), {self.DAY},
        )
        assert sem.verified and sem.from_inclusive is True and sem.to_inclusive is True

    def test_exclusive_to_is_detected(self):
        sem = infer_boundary_semantics(self.DAY, [], [self.DAY], [], {self.DAY})
        assert sem.verified and sem.from_inclusive and sem.to_inclusive is False

    def test_exclusive_from_is_detected(self):
        sem = infer_boundary_semantics(self.DAY, [], [], [self.DAY], {self.DAY})
        assert sem.verified and sem.from_inclusive is False and sem.to_inclusive

    def test_without_an_independent_control_nothing_is_concluded(self):
        sem = infer_boundary_semantics(self.DAY, [self.DAY], [self.DAY], [self.DAY], set())
        assert not sem.verified and sem.from_inclusive is None

    def test_a_single_day_probe_contradicting_the_others_is_unverified_not_guessed(self):
        sem = infer_boundary_semantics(self.DAY, [], [self.DAY], [self.DAY], {self.DAY})
        assert not sem.verified and "inconsistent" in sem.detail

    def test_the_two_real_probe_dates_are_the_documented_reliance_bonus(self):
        assert _probe_dates("NSE", "single-day") == [self.DAY] and _probe_dates("BSE", "single-day") == [self.DAY]


def _dated(*days):
    return [(d, nse_row(d, f"Bonus {i + 1}:1")) for i, d in enumerate(days)]


def _snap_for(rows_dated, lo, hi, purpose):
    return snap("NSE", [r for d, r in rows_dated if lo <= d <= hi], lo, hi, purpose)


class TestPartitionConsistency:
    LO, MID_A, MID_B, HI = D(2015, 1, 1), D(2018, 12, 31), D(2019, 1, 1), D(2026, 9, 20)

    def _parts(self, dated):
        return [_snap_for(dated, self.LO, self.MID_A, "p1"), _snap_for(dated, self.MID_B, self.HI, "p2")]

    def test_consistent_partitions_pass(self):
        dated = _dated(D(2016, 6, 1), D(2018, 12, 31), D(2019, 1, 1), D(2024, 3, 1))
        whole = _snap_for(dated, self.LO, self.HI, "whole")
        r = partition_consistency(whole, self._parts(dated), INCLUSIVE)
        assert r["consistent"] and r["gap_days"] == [] and r["union_rows"] == 4

    def test_events_on_both_sides_of_the_seam_are_kept_under_inclusive_semantics(self):
        dated = _dated(D(2018, 12, 31), D(2019, 1, 1))
        whole = _snap_for(dated, self.LO, self.HI, "whole")
        assert partition_consistency(whole, self._parts(dated), INCLUSIVE)["consistent"]

    def test_unverified_boundary_semantics_can_never_pass(self):
        dated = _dated(D(2016, 6, 1))
        whole = _snap_for(dated, self.LO, self.HI, "whole")
        r = partition_consistency(whole, self._parts(dated), UNVERIFIED)
        assert r["consistent"] is False and "not verified" in r["reason"]

    def test_exclusive_to_leaves_a_gap_day_at_the_seam_and_the_event_on_it_is_lost(self):
        dated = _dated(D(2018, 12, 31))
        whole = _snap_for(dated, self.LO, self.HI, "whole")
        parts = [snap("NSE", [], self.LO, self.MID_A, "p1"), _snap_for(dated, self.MID_B, self.HI, "p2")]
        r = partition_consistency(whole, parts, EXCLUSIVE_TO)
        assert not r["consistent"] and "2018-12-31" in r["gap_days"] and r["only_in_whole"] == ["2018-12-31"]

    def test_exclusive_from_leaves_a_gap_day_at_the_seam(self):
        dated = _dated(D(2019, 1, 1))
        whole = _snap_for(dated, self.LO, self.HI, "whole")
        r = partition_consistency(whole, self._parts(dated), EXCLUSIVE_FROM)
        assert not r["consistent"] and "2019-01-01" in r["gap_days"]

    def test_overlapping_partitions_cover_the_seam_even_under_exclusive_semantics(self):
        dated = _dated(D(2018, 12, 31))
        whole = _snap_for(dated, self.LO, self.HI, "whole")
        parts = [_snap_for(dated, self.LO, D(2019, 1, 1), "p1"), _snap_for(dated, D(2018, 12, 30), self.HI, "p2")]
        assert partition_consistency(whole, parts, BoundarySemantics(True, True, "x"))["consistent"]

    def test_a_row_in_the_whole_response_absent_from_every_part_is_reported(self):
        dated = _dated(D(2016, 6, 1), D(2024, 3, 1))
        whole = _snap_for(dated, self.LO, self.HI, "whole")
        parts = [_snap_for(dated[:1], self.LO, self.MID_A, "p1"), snap("NSE", [], self.MID_B, self.HI, "p2")]
        r = partition_consistency(whole, parts, INCLUSIVE)
        assert not r["consistent"] and r["only_in_whole"] == ["2024-03-01"]

    def test_a_truncated_whole_response_is_caught_by_rows_only_in_the_parts(self):
        dated = _dated(D(2016, 6, 1), D(2024, 3, 1))
        whole = snap("NSE", [dated[1][1]], self.LO, self.HI, "whole")
        r = partition_consistency(whole, self._parts(dated), INCLUSIVE)
        assert not r["consistent"] and r["only_in_parts"] == ["2016-06-01"]

    def test_a_row_outside_the_window_it_was_returned_for_is_reported(self):
        dated = _dated(D(2016, 6, 1))
        whole = _snap_for(dated, self.LO, self.HI, "whole")
        stray = snap("NSE", [nse_row(D(2011, 7, 14), "Bonus 1:1")], self.LO, self.MID_A, "p1")
        r = partition_consistency(whole, [stray, snap("NSE", [], self.MID_B, self.HI, "p2")], INCLUSIVE)
        assert not r["consistent"] and r["out_of_window_rows"][0]["ex_date"] == "2011-07-14"

    def test_a_row_with_an_unreadable_ex_date_is_reported_never_ignored(self):
        whole = snap("NSE", [{"exDate": "-", "subject": "Bonus 1:1"}], self.LO, self.HI, "whole")
        r = partition_consistency(whole, [snap("NSE", [], self.LO, self.MID_A, "p1"), snap("NSE", [], self.MID_B, self.HI, "p2")], INCLUSIVE)
        assert not r["consistent"] and r["out_of_window_rows"][0]["ex_date"] is None

    def test_partitions_that_leave_the_end_of_the_window_uncovered_have_gap_days(self):
        dated = _dated(D(2016, 6, 1))
        whole = _snap_for(dated, self.LO, self.HI, "whole")
        parts = [_snap_for(dated, self.LO, D(2025, 12, 31), "p1")]
        r = partition_consistency(whole, parts, INCLUSIVE)
        assert not r["consistent"] and r["gap_day_count"] > 0


class TestSnapshotCoverageWindows:
    def test_windowed_snapshot_speaks_for_its_query_window(self):
        s = snap("NSE", [], D(2015, 1, 1), D(2018, 12, 31))
        assert snapshot_coverage_window(s) == (D(2015, 1, 1), D(2018, 12, 31))

    def test_default_snapshot_speaks_only_from_its_earliest_row_to_its_fetch_date(self):
        s = snap("NSE", [nse_row(D(2019, 9, 19), "Bonus 1:1"), nse_row(D(2024, 1, 1), "Bonus 1:1")], None, None,
                 mode="default", fetched="2026-05-01T00:00:00+00:00")
        assert snapshot_coverage_window(s) == (D(2019, 9, 19), D(2026, 5, 1))

    def test_empty_default_snapshot_speaks_for_nothing(self):
        assert snapshot_coverage_window(snap("NSE", [], None, None, mode="default")) is None

    def test_comparable_window_is_the_intersection_inside_the_evidence_window(self):
        old = snap("NSE", [nse_row(D(2019, 9, 19), "Bonus 1:1")], None, None, mode="default", fetched="2026-05-01T00:00:00+00:00")
        new = snap("NSE", [], D(2015, 1, 1), D(2026, 9, 20))
        assert comparable_window(old, new, (D(2015, 1, 1), D(2026, 9, 11))) == (D(2019, 9, 19), D(2026, 5, 1))

    def test_no_shared_coverage_means_no_comparison(self):
        a = snap("NSE", [], D(2015, 1, 1), D(2016, 1, 1))
        b = snap("NSE", [], D(2020, 1, 1), D(2021, 1, 1))
        assert comparable_window(a, b, (D(2015, 1, 1), D(2026, 9, 11))) is None


def _bars(*closes, start=D(2020, 1, 1)):
    out, d = [], start
    for c in closes:
        out.append((d, float(c)))
        d += __import__("datetime").timedelta(days=1)
    return out


class TestJumpScanIsADiagnosticBlockerOnly:
    def test_thresholds_are_the_approved_heuristics(self):
        assert (BLOCKER_THRESHOLD, WATCH_THRESHOLD, ISIN_MAX_LAG_SESSIONS) == (0.28, 0.15, 2)

    def test_an_unexplained_28_percent_move_blocks(self):
        r = scan_close_discontinuities(_bars(100, 72), [])
        assert r["blocking_count"] == 1 and r["unexplained_blockers"][0]["tier"] == "BLOCKER"

    def test_the_same_move_in_the_up_direction_also_blocks(self):
        assert scan_close_discontinuities(_bars(100, 128), [])["blocking_count"] == 1

    def test_just_under_28_percent_is_watch_not_blocker(self):
        r = scan_close_discontinuities(_bars(100, 72.5), [])
        assert r["blocking_count"] == 0 and len(r["watch_unexplained"]) == 1

    def test_just_under_15_percent_is_ignored(self):
        r = scan_close_discontinuities(_bars(100, 85.5), [])
        assert r["blocking_count"] == 0 and r["watch_unexplained"] == [] and r["explained_jumps"] == []

    def test_a_price_affecting_ex_date_on_the_bar_explains_the_jump(self):
        r = scan_close_discontinuities(_bars(100, 50), [D(2020, 1, 2)])
        assert r["blocking_count"] == 0 and r["explained_jumps"][0]["explained_by_ex_date"] == "2020-01-02"

    def test_a_non_trading_ex_date_explains_the_first_bar_after_it(self):
        bars = [(D(2020, 1, 3), 100.0), (D(2020, 1, 6), 50.0)]
        assert scan_close_discontinuities(bars, [D(2020, 1, 4)])["blocking_count"] == 0

    def test_an_ex_date_on_a_different_bar_does_not_explain_the_jump(self):
        r = scan_close_discontinuities(_bars(100, 100, 50), [D(2020, 1, 2)])
        assert r["blocking_count"] == 1

    def test_an_explained_jump_does_not_hide_a_second_unexplained_one(self):
        r = scan_close_discontinuities(_bars(100, 50, 50, 20), [D(2020, 1, 2)])
        assert r["blocking_count"] == 1 and r["unexplained_blockers"][0]["bar_date"] == "2020-01-04"

    def test_a_zero_previous_close_is_skipped_not_divided(self):
        assert scan_close_discontinuities(_bars(0, 50), [])["blocking_count"] == 0

    def test_unsorted_bars_are_ordered_first(self):
        assert scan_close_discontinuities(list(reversed(_bars(100, 50))), [])["blocking_count"] == 1

    def test_dividends_are_not_price_affecting_so_they_can_never_be_passed_as_explanations(self):
        assert "DIVIDEND" not in PRICE_AFFECTING
        assert PRICE_AFFECTING == {"SPLIT", "BONUS", "RIGHTS", "DEMERGER", "MERGER"}

    def test_a_clean_scan_returns_no_status_and_says_it_proves_nothing(self):
        r = scan_close_discontinuities(_bars(100, 101), [])
        assert r["diagnostic_only"] is True and not {"status", "verdict", "complete"} & r.keys()
        assert "never establish completeness" in r["statement"]

    def test_a_small_effect_action_is_invisible_to_the_scan(self):
        r = scan_close_discontinuities(_bars(100, 96), [])
        assert r["blocking_count"] == 0 and r["watch_unexplained"] == [] and r["explained_jumps"] == []


class TestIsinTransitionCorroborationIsADiagnosticOnly:
    BARS = [D(2020, 1, 1) + __import__("datetime").timedelta(days=i) for i in range(10)]

    @pytest.mark.parametrize("lag", [0, 1, 2])
    def test_a_transition_0_to_2_sessions_after_the_ex_date_is_corroborated(self, lag):
        ex = self.BARS[3]
        r = isin_transition_corroboration([self.BARS[3 + lag]], self.BARS, [ex])
        assert r["uncorroborated_count"] == 0 and r["transitions"][0]["lag_sessions"] == lag

    def test_a_transition_3_sessions_after_is_not_corroborated(self):
        assert isin_transition_corroboration([self.BARS[6]], self.BARS, [self.BARS[3]])["uncorroborated_count"] == 1

    def test_a_transition_before_the_ex_date_is_not_corroborated(self):
        assert isin_transition_corroboration([self.BARS[2]], self.BARS, [self.BARS[3]])["uncorroborated_count"] == 1

    def test_a_transition_with_no_price_affecting_action_is_not_corroborated(self):
        assert isin_transition_corroboration([self.BARS[3]], self.BARS, [])["uncorroborated_count"] == 1

    def test_no_transitions_means_no_uncorroborated_transitions(self):
        assert isin_transition_corroboration([], self.BARS, [self.BARS[3]])["uncorroborated_count"] == 0

    def test_lag_is_counted_in_sessions_not_calendar_days(self):
        bars = [D(2020, 1, 3), D(2020, 1, 6), D(2020, 1, 7), D(2020, 1, 8), D(2020, 1, 9)]
        r = isin_transition_corroboration([D(2020, 1, 8)], bars, [D(2020, 1, 3)])
        assert r["uncorroborated_count"] == 1
        r = isin_transition_corroboration([D(2020, 1, 7)], bars, [D(2020, 1, 3)])
        assert r["uncorroborated_count"] == 0

    def test_it_returns_no_status_and_says_it_never_anchors_or_establishes_completeness(self):
        r = isin_transition_corroboration([], self.BARS, [])
        assert r["diagnostic_only"] is True and not {"status", "verdict", "complete"} & r.keys()
        assert "never anchors an adjustment and never establishes completeness" in r["statement"]
