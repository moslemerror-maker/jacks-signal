from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_phase_i_analysis import (
    BASELINE_FEATURES_FOR_ROBUSTNESS,
    HORIZONS,
    ROBUSTNESS_HORIZONS,
    SIMPLE_BASELINES,
    final_verdict,
    run_i2_cost_scenarios,
    run_i5_magnitude_thresholds,
    session_phase,
)
from scripts.run_phase_i_analysis import test_budget as compute_test_budget


class TestSessionPhase:
    def test_boundaries_are_exact_and_half_open(self):
        assert session_phase(0) == "OPENING"
        assert session_phase(29) == "OPENING"
        assert session_phase(30) == "MID_SESSION"
        assert session_phase(179) == "MID_SESSION"
        assert session_phase(180) == "AFTERNOON"
        assert session_phase(299) == "AFTERNOON"
        assert session_phase(300) == "FINAL_HOUR"
        assert session_phase(400) == "FINAL_HOUR"

    def test_none_and_nan_pass_through_as_none(self):
        assert session_phase(None) is None
        assert session_phase(float("nan")) is None


class TestBudget:
    def test_matches_exact_formula(self):
        b = compute_test_budget()
        assert b["I3_entry_timing"] == len(HORIZONS)
        assert b["I4_stability"] == (4 + 5 + 2 + 13) * len(HORIZONS)
        assert b["I7_robustness"] == len(BASELINE_FEATURES_FOR_ROBUSTNESS) * len(ROBUSTNESS_HORIZONS) * 2
        assert b["I8_baselines"] == (len(SIMPLE_BASELINES) - 1) * len(HORIZONS)
        assert b["I9_cross_stock"] == len(HORIZONS)

    def test_is_deterministic(self):
        assert compute_test_budget() == compute_test_budget()


class TestCostScenarios:
    def test_zero_cost_never_flags_cost_exceeds_move_for_a_positive_gross_move(self):
        n = 100
        df = pd.DataFrame({
            "relative_strength_vs_universe": [i / n for i in range(n)],
            "mover_future_abs_return_5m": [0.01] * n,
        })
        result = run_i2_cost_scenarios(df)
        assert result["5m"]["scenarios"]["zero_cost"]["cost_exceeds_gross_move"] is False

    def test_high_cost_can_exceed_a_tiny_gross_move(self):
        n = 100
        df = pd.DataFrame({
            "relative_strength_vs_universe": [i / n for i in range(n)],
            "mover_future_abs_return_5m": [0.0001] * n,
        })
        result = run_i2_cost_scenarios(df)
        assert result["5m"]["scenarios"]["low_cost"]["cost_exceeds_gross_move"] is True
        assert result["5m"]["scenarios"]["conservative_high_cost"]["cost_exceeds_gross_move"] is True

    def test_reports_a_scenario_for_every_configured_cost_level(self):
        n = 100
        df = pd.DataFrame({"relative_strength_vs_universe": [i / n for i in range(n)], "mover_future_abs_return_5m": [0.005] * n})
        result = run_i2_cost_scenarios(df)
        assert set(result["5m"]["scenarios"].keys()) == {"zero_cost", "low_cost", "medium_cost", "conservative_high_cost"}


class TestMagnitudeThresholds:
    def test_lift_is_greater_than_one_when_extreme_buckets_move_more(self):
        n = 200
        rs = [i / n for i in range(n)]
        move = [(r - 0.5) * 4.0 for r in rs]
        df = pd.DataFrame({"relative_strength_vs_universe": rs, "label_normalized_move_5m": move})
        result = run_i5_magnitude_thresholds(df)
        assert result["5m"]["by_threshold"]["1.0"]["lift"] > 1.0

    def test_skips_gracefully_with_too_few_rows(self):
        df = pd.DataFrame({"relative_strength_vs_universe": [0.1, 0.2], "label_normalized_move_5m": [0.1, 0.2]})
        result = run_i5_magnitude_thresholds(df)
        assert "skipped_reason" in result["5m"]


class TestFinalVerdict:
    def _i2(self, *, medium_ok: bool, conservative_ok: bool):
        return {f"{h}m": {"scenarios": {"medium_cost": {"cost_exceeds_gross_move": not medium_ok}, "conservative_high_cost": {"cost_exceeds_gross_move": not conservative_ok}}} for h in HORIZONS}

    def _i2_all_pass(self):
        return self._i2(medium_ok=True, conservative_ok=True)

    def _i2_fails_medium(self):
        return self._i2(medium_ok=False, conservative_ok=False)

    def _i2_survives_medium_only(self):
        return self._i2(medium_ok=True, conservative_ok=False)

    def _i3_all_sig(self):
        return {f"{h}m": {"next_bar_open_entry": {"significant_after_correction": True}} for h in HORIZONS}

    def _i3_not_sig(self):
        return {f"{h}m": {"next_bar_open_entry": {"significant_after_correction": False}} for h in HORIZONS}

    def _i5_strong_lift(self):
        return {f"{h}m": {"by_threshold": {"1.0": {"lift": 1.3}}} for h in HORIZONS}

    def _i5_no_lift(self):
        return {f"{h}m": {"by_threshold": {"1.0": {"lift": 1.0}}} for h in HORIZONS}

    def _i9_survives(self):
        return {
            "per_symbol_effect_distribution": {"fraction_negative": 0.85},
            "excluding_top_5_strongest_symbols": {f"{h}m": {"significant_after_correction": True} for h in HORIZONS},
        }

    def _i7_all_sig(self):
        return {f: {"15m": {"below_median": {"significant_after_correction": True}, "at_or_above_median": {"significant_after_correction": True}}} for f in BASELINE_FEATURES_FOR_ROBUSTNESS}

    def _i4_empty(self):
        return {}

    def _i8_empty(self):
        return {}

    def test_verdict_c_when_even_medium_cost_exceeds_gross_move(self):
        v = final_verdict(self._i2_fails_medium(), self._i3_all_sig(), self._i4_empty(), self._i5_strong_lift(), self._i7_all_sig(), self._i8_empty(), self._i9_survives())
        assert v["verdict"] == "C"

    def test_verdict_b_when_medium_cost_survives_but_conservative_cost_does_not(self):
        v = final_verdict(self._i2_survives_medium_only(), self._i3_all_sig(), self._i4_empty(), self._i5_strong_lift(), self._i7_all_sig(), self._i8_empty(), self._i9_survives())
        assert v["verdict"] == "B"
        assert v["survives_medium_cost_all_horizons"] is True
        assert v["survives_conservative_high_cost_all_horizons"] is False

    def test_verdict_b_when_costs_pass_but_entry_timing_fails(self):
        v = final_verdict(self._i2_all_pass(), self._i3_not_sig(), self._i4_empty(), self._i5_strong_lift(), self._i7_all_sig(), self._i8_empty(), self._i9_survives())
        assert v["verdict"] == "B"

    def test_verdict_b_when_no_meaningful_movement_lift(self):
        v = final_verdict(self._i2_all_pass(), self._i3_all_sig(), self._i4_empty(), self._i5_no_lift(), self._i7_all_sig(), self._i8_empty(), self._i9_survives())
        assert v["verdict"] == "B"
        assert v["criteria"]["5_economically_meaningful_movement_potential"] is False

    def test_verdict_a_only_when_every_criterion_passes(self):
        v = final_verdict(self._i2_all_pass(), self._i3_all_sig(), self._i4_empty(), self._i5_strong_lift(), self._i7_all_sig(), self._i8_empty(), self._i9_survives())
        assert v["verdict"] == "A"
        assert v["n_criteria_met"] == v["n_criteria_total"]

    def test_never_hardcodes_a_without_checking_criteria(self):
        import inspect
        from scripts import run_phase_i_analysis as mod
        source = inspect.getsource(mod.final_verdict)
        assert source.index("n_criteria_met == len(criteria)") < source.index('verdict = "A"')
