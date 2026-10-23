from __future__ import annotations

import datetime as dt

import pytest

from backend.app.equity_live.candidate_scanner import (
    ABNORMAL_RELATIVE_VOLUME,
    OPENING_DRIVE_MAX_MINUTES,
    OPENING_DRIVE_MIN_MOVE,
    VOLATILITY_COMPRESSION_THRESHOLD,
    VOLATILITY_EXPANSION_THRESHOLD,
    build_candidate,
    classify_breakout_state,
    classify_opening_drive_state,
    classify_volatility_state,
    scan_universe,
)
from backend.app.equity_live.live_feature_service import FeatureSnapshot

NOW = dt.datetime(2026, 8, 1, 10, 0, tzinfo=dt.timezone.utc)


def _snap(**overrides) -> FeatureSnapshot:
    base = dict(
        security_id=1, symbol="TEST", as_of_utc=NOW, trading_date=dt.date(2026, 8, 1), price=100.0,
        return_1m=0.0, return_5m=0.0, return_15m=0.0, relative_volume_20=1.0, realized_volatility_20=0.01,
        volatility_expansion_ratio=1.0, distance_from_vwap=0.0, session_vwap=100.0, opening_range_high=101.0,
        opening_range_low=99.0, overnight_gap=0.0, recent_breakout_up=False, recent_breakout_down=False,
        distance_from_session_open=0.0, minutes_since_open=60, relative_strength=0.0,
        n_bars_today=45, n_bars_total=400, latest_bar_utc=NOW, data_quality="OK",
    )
    base.update(overrides)
    return FeatureSnapshot(**base)


class TestVolatilityState:
    def test_expanding_at_or_above_threshold(self):
        assert classify_volatility_state(_snap(volatility_expansion_ratio=VOLATILITY_EXPANSION_THRESHOLD)) == "EXPANDING"

    def test_compressed_at_or_below_threshold(self):
        assert classify_volatility_state(_snap(volatility_expansion_ratio=VOLATILITY_COMPRESSION_THRESHOLD)) == "COMPRESSED"

    def test_normal_between_thresholds(self):
        assert classify_volatility_state(_snap(volatility_expansion_ratio=1.0)) == "NORMAL"

    def test_unknown_when_undefined(self):
        assert classify_volatility_state(_snap(volatility_expansion_ratio=None)) == "UNKNOWN"


class TestBreakoutState:
    def test_breakout_up(self):
        assert classify_breakout_state(_snap(recent_breakout_up=True, recent_breakout_down=False)) == "BREAKOUT_UP"

    def test_breakout_down(self):
        assert classify_breakout_state(_snap(recent_breakout_up=False, recent_breakout_down=True)) == "BREAKOUT_DOWN"

    def test_none_when_neither(self):
        assert classify_breakout_state(_snap(recent_breakout_up=False, recent_breakout_down=False)) == "NONE"

    def test_unknown_when_both_undefined(self):
        assert classify_breakout_state(_snap(recent_breakout_up=None, recent_breakout_down=None)) == "UNKNOWN"


class TestOpeningDriveState:
    def test_drive_up_within_window_and_above_threshold(self):
        s = _snap(minutes_since_open=10, distance_from_session_open=OPENING_DRIVE_MIN_MOVE)
        assert classify_opening_drive_state(s) == "DRIVE_UP"

    def test_drive_down_within_window_and_below_negative_threshold(self):
        s = _snap(minutes_since_open=10, distance_from_session_open=-OPENING_DRIVE_MIN_MOVE)
        assert classify_opening_drive_state(s) == "DRIVE_DOWN"

    def test_none_when_move_too_small(self):
        s = _snap(minutes_since_open=10, distance_from_session_open=OPENING_DRIVE_MIN_MOVE / 2)
        assert classify_opening_drive_state(s) == "NONE"

    def test_none_when_outside_the_opening_window(self):
        s = _snap(minutes_since_open=OPENING_DRIVE_MAX_MINUTES + 1, distance_from_session_open=0.05)
        assert classify_opening_drive_state(s) == "NONE"

    def test_unknown_when_undefined(self):
        assert classify_opening_drive_state(_snap(minutes_since_open=None, distance_from_session_open=None)) == "UNKNOWN"


class TestCandidateReasonsAndDecision:
    def test_no_reasons_means_no_candidate(self):
        assert build_candidate(_snap(), rank=0.5, liquidity_tier="high_liquidity") is None

    def test_abnormal_volume_alone_produces_a_candidate(self):
        c = build_candidate(_snap(relative_volume_20=ABNORMAL_RELATIVE_VOLUME), rank=0.5, liquidity_tier="high_liquidity")
        assert c is not None
        assert "ABNORMAL_VOLUME" in c.candidate_reasons

    def test_extreme_cross_sectional_rank_produces_a_candidate(self):
        c = build_candidate(_snap(), rank=0.97, liquidity_tier="high_liquidity")
        assert c is not None
        assert "EXTREME_CROSS_SECTIONAL_MOVE" in c.candidate_reasons

    def test_decision_wait_for_tradable_liquidity_and_ok_data(self):
        c = build_candidate(_snap(relative_volume_20=ABNORMAL_RELATIVE_VOLUME), rank=0.5, liquidity_tier="mid_liquidity")
        assert c.decision == "WAIT"

    def test_decision_observe_for_low_liquidity_even_with_ok_data(self):
        c = build_candidate(_snap(relative_volume_20=ABNORMAL_RELATIVE_VOLUME), rank=0.5, liquidity_tier="low_liquidity")
        assert c.decision == "OBSERVE"

    def test_decision_observe_for_undefined_liquidity_tier(self):
        c = build_candidate(_snap(relative_volume_20=ABNORMAL_RELATIVE_VOLUME), rank=0.5, liquidity_tier=None)
        assert c.decision == "OBSERVE"

    def test_decision_observe_when_data_quality_is_not_ok_even_with_high_liquidity(self):
        c = build_candidate(_snap(relative_volume_20=ABNORMAL_RELATIVE_VOLUME, data_quality="STALE"), rank=0.5, liquidity_tier="high_liquidity")
        assert c.decision == "OBSERVE"


class TestDirectionIsAlwaysUnknown:
    @pytest.mark.parametrize("liquidity_tier", ["low_liquidity", "mid_liquidity", "high_liquidity", None])
    @pytest.mark.parametrize("data_quality", ["OK", "STALE", "INSUFFICIENT_HISTORY"])
    def test_direction_is_unknown_across_every_combination(self, liquidity_tier, data_quality):
        c = build_candidate(_snap(relative_volume_20=ABNORMAL_RELATIVE_VOLUME, data_quality=data_quality), rank=0.5, liquidity_tier=liquidity_tier)
        assert c is not None
        assert c.direction == "UNKNOWN"


class TestScanUniverse:
    def test_only_flagged_symbols_are_returned(self):
        filler = [_snap(symbol=f"F{i}", security_id=10 + i, return_15m=(i - 10) * 0.01) for i in range(20)]
        quiet = _snap(symbol="QUIET", security_id=1, return_15m=0.0)
        loud = _snap(symbol="LOUD", security_id=2, return_15m=0.0, relative_volume_20=ABNORMAL_RELATIVE_VOLUME)
        snapshots = filler + [quiet, loud]
        liquidity_tiers = {s.security_id: "high_liquidity" for s in snapshots}
        candidates = scan_universe(snapshots, liquidity_tiers=liquidity_tiers)
        symbols = {c.symbol for c in candidates}
        assert "QUIET" not in symbols
        assert "LOUD" in symbols

    def test_no_ranking_field_forces_best_stock_language(self):
        from backend.app.equity_live.candidate_scanner import CandidateRecord
        field_names = {f.name for f in __import__("dataclasses").fields(CandidateRecord)}
        assert "score" not in field_names and "rank" not in field_names and "best" not in field_names
