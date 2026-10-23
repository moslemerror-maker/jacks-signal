from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_phase_k_economic_analysis import (
    COST_SCENARIOS_BPS,
    DIRECTIONAL_MECHANISMS,
    MAGNITUDE_ONLY_MECHANISMS,
    MECHANISMS,
    _cost_adjusted_stats,
    _mechanism_mask_and_sign,
    _oriented_returns,
    build_ledger,
    final_verdict,
    pick_top_mechanisms_by_family_a,
)
from scripts.run_phase_k_economic_analysis import test_budget as compute_test_budget


def _df():
    n = 100
    return pd.DataFrame({
        "symbol": [f"S{i % 5}" for i in range(n)],
        "bar_start_utc": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC"),
        "relative_strength_vs_universe": [((-1) ** i) * (i / n) for i in range(n)],
        "recent_breakout_up_opening_range": [i % 4 == 0 for i in range(n)],
        "relative_volume_20": [i / n for i in range(n)],
        "compression_then_expansion": [i % 5 == 0 for i in range(n)],
        "level_touch_count_resistance": [i % 10 for i in range(n)],
        "gross_return_5m": [0.001 * ((-1) ** i) * (i % 7) for i in range(n)],
    })


class TestMechanismMaskAndSign:
    def test_m1_assigns_long_to_low_quantile_and_short_to_high_quantile(self):
        df = _df()
        mask, sign = _mechanism_mask_and_sign(df, "M1_relative_strength_vs_universe", 5)
        lowest_row = df["relative_strength_vs_universe"].idxmin()
        highest_row = df["relative_strength_vs_universe"].idxmax()
        assert sign.loc[lowest_row] == 1.0
        assert sign.loc[highest_row] == -1.0

    def test_m2_is_always_long_when_the_flag_is_true(self):
        df = _df()
        mask, sign = _mechanism_mask_and_sign(df, "M2_recent_breakout_up", 5)
        assert (mask == df["recent_breakout_up_opening_range"]).all()
        assert (sign == 1.0).all()

    def test_m5_requires_both_breakout_and_above_median_volume(self):
        df = _df()
        mask, _ = _mechanism_mask_and_sign(df, "M5_breakout_x_relative_volume", 5)
        med = df["relative_volume_20"].median()
        expected = df["recent_breakout_up_opening_range"] & (df["relative_volume_20"] >= med)
        assert (mask == expected).all()

    def test_magnitude_only_mechanisms_have_no_sign(self):
        df = _df()
        for mechanism in MAGNITUDE_ONLY_MECHANISMS:
            _, sign = _mechanism_mask_and_sign(df, mechanism, 5)
            assert sign.isna().all()

    def test_every_declared_mechanism_is_handled(self):
        df = _df()
        for mechanism in MECHANISMS:
            _mechanism_mask_and_sign(df, mechanism, 5)


class TestOrientedReturns:
    def test_directional_mechanism_orients_return_by_sign(self):
        df = pd.DataFrame({
            "symbol": ["A", "A"], "bar_start_utc": pd.date_range("2026-01-01", periods=2, freq="min", tz="UTC"),
            "recent_breakout_up_opening_range": [True, True], "gross_return_5m": [0.01, -0.01],
        })
        result = _oriented_returns(df, "M2_recent_breakout_up", 5)
        assert result["_oriented"].tolist() == [0.01, -0.01]

    def test_magnitude_only_mechanism_uses_absolute_value(self):
        df = pd.DataFrame({
            "symbol": ["A", "A"], "bar_start_utc": pd.date_range("2026-01-01", periods=2, freq="min", tz="UTC"),
            "compression_then_expansion": [True, True], "gross_return_5m": [0.01, -0.02],
        })
        result = _oriented_returns(df, "M3_compression_then_expansion", 5)
        assert result["_oriented"].tolist() == [0.01, 0.02]

    def test_m1_short_side_flips_the_sign(self):
        n = 20
        df = pd.DataFrame({
            "symbol": ["A"] * n, "bar_start_utc": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC"),
            "relative_strength_vs_universe": list(range(n)),
            "gross_return_5m": [0.01] * n,
        })
        result = _oriented_returns(df, "M1_relative_strength_vs_universe", 5)
        highest_row = df["relative_strength_vs_universe"].idxmax()
        assert result.loc[highest_row, "_oriented"] == pytest.approx(-0.01)


class TestCostAdjustedStats:
    def _oriented_df(self, values: list[float]) -> pd.DataFrame:
        n = len(values)
        return pd.DataFrame({"_oriented": values, "symbol": ["A"] * n, "bar_start_utc": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")})

    def test_reports_every_cost_scenario(self):
        oriented = self._oriented_df([0.01] * 40)
        result = _cost_adjusted_stats(oriented, alpha=0.5, horizon=5)
        assert set(result["cost_scenarios"].keys()) == set(COST_SCENARIOS_BPS.keys())

    def test_net_is_gross_minus_cost_in_correct_units(self):
        oriented = self._oriented_df([0.01] * 40)
        result = _cost_adjusted_stats(oriented, alpha=0.5, horizon=5)
        assert result["cost_scenarios"]["low_20bps"]["mean_net"] == pytest.approx(0.008, abs=1e-9)

    def test_skips_gracefully_with_too_few_rows(self):
        oriented = self._oriented_df([0.01] * 5)
        result = _cost_adjusted_stats(oriented, alpha=0.5, horizon=5)
        assert "skipped_reason" in result

    def test_significance_requires_positive_mean_not_just_a_small_p_value(self):
        oriented = self._oriented_df([-0.05] * 40)
        result = _cost_adjusted_stats(oriented, alpha=0.99, horizon=5)
        assert result["significant_after_correction"] is False


class TestTestBudget:
    def test_matches_exact_formula(self):
        b = compute_test_budget()
        assert b["A_headline_economic"] == len(MECHANISMS) * 4
        assert b["B_incremental_value"] == (len(MECHANISMS) - 1) * 2
        assert b["C_cross_stock"] == len(MECHANISMS) * 4


class TestPickTopMechanisms:
    def test_picks_by_largest_absolute_gross_mean_among_significant(self):
        rows = [
            {"mechanism": "A", "significant_after_correction": True, "gross_mean": 0.001},
            {"mechanism": "B", "significant_after_correction": True, "gross_mean": 0.01},
            {"mechanism": "C", "significant_after_correction": False, "gross_mean": 0.5},
        ]
        top = pick_top_mechanisms_by_family_a(rows, n=2)
        assert top == ["B", "A"]

    def test_deduplicates_mechanisms_appearing_at_multiple_horizons(self):
        rows = [
            {"mechanism": "A", "significant_after_correction": True, "gross_mean": 0.01},
            {"mechanism": "A", "significant_after_correction": True, "gross_mean": 0.02},
            {"mechanism": "B", "significant_after_correction": True, "gross_mean": 0.005},
        ]
        top = pick_top_mechanisms_by_family_a(rows, n=5)
        assert top == ["A", "B"]


class TestFinalVerdict:
    def _chrono(self, *, holds: bool):
        return {"by_mechanism": {"M2_recent_breakout_up": {"15m": {"train": {"newey_west_net_mean": 0.01 if holds else -0.01}, "validation": {"newey_west_net_mean": 0.01 if holds else -0.01}, "held_out": {"newey_west_net_mean": 0.01 if holds else -0.01}}}}}

    def test_verdict_d_when_nothing_significant(self):
        v = final_verdict([{"significant_after_correction": False}], [], [], self._chrono(holds=True))
        assert v["verdict"] == "D"

    def test_verdict_c_when_significant_but_fails_conservative_cost(self):
        family_a = [{"significant_after_correction": True, "cost_scenarios": {"high_60bps": {"mean_net": -0.001}}}]
        v = final_verdict(family_a, [{"significant_after_correction": True}], [{"significant_after_correction": True}], self._chrono(holds=True))
        assert v["verdict"] == "C"

    def test_verdict_a_only_when_every_criterion_passes(self):
        family_a = [{"significant_after_correction": True, "cost_scenarios": {"high_60bps": {"mean_net": 0.001}}}]
        family_c = [{"significant_after_correction": True}, {"skipped_reason": "x"}]
        v = final_verdict(family_a, [{"significant_after_correction": True}], family_c, self._chrono(holds=True))
        assert v["verdict"] == "A"
        assert v["n_criteria_met"] == v["n_criteria_total"]

    def test_never_hardcodes_a(self):
        import inspect
        from scripts import run_phase_k_economic_analysis as mod
        source = inspect.getsource(mod.final_verdict)
        assert source.index("if n_met == len(criteria):") < source.index('verdict, reason = "A"')


class TestBuildLedger:
    def test_every_family_contributes_and_failures_are_recorded(self):
        family_a = [{"mechanism": "M1", "horizon_minutes": 5, "significant_after_correction": False, "gross_mean": 0.0001, "n": 100}]
        ledger = build_ledger(family_a, [], [], alpha=0.001)
        assert ledger[0]["result"] == "NOT_SIGNIFICANT"
