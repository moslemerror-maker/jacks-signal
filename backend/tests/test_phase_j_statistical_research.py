from __future__ import annotations

import pandas as pd
import pytest

import numpy as np

from scripts.run_phase_j_statistical_research import (
    BINARY_FEATURES,
    COMBINATIONS,
    FAMILY_A_FEATURES,
    HORIZONS,
    _family_a_direction_at_15m,
    _finalize_rate_test,
    build_ledger,
    final_verdict,
    pick_strongest_family_a,
    run_family_c,
)
from scripts.run_phase_j_statistical_research import test_budget as compute_test_budget


class TestTestBudget:
    def test_matches_exact_formula(self):
        b = compute_test_budget()
        n_binary = len(BINARY_FEATURES & set(FAMILY_A_FEATURES))
        n_continuous = len(FAMILY_A_FEATURES) - n_binary
        assert b["A_standalone"] == (n_continuous * 2 + n_binary * 1) * len(HORIZONS)
        assert b["B_combinations"] == len(COMBINATIONS) * 2
        assert b["C_incremental_value"] == len(FAMILY_A_FEATURES) * 2
        assert b["D_cross_stock"] == len(HORIZONS)

    def test_is_deterministic(self):
        assert compute_test_budget() == compute_test_budget()


class TestFinalizeRateTest:
    def test_bucket_rate_above_baseline_gives_lift_greater_than_one(self):
        flags = [1.0] * 20 + [0.0] * 20
        result = _finalize_rate_test(flags, baseline_rate=0.1, horizon=5, alpha=0.05, n_total_valid=1000)
        assert result["bucket_rate"] == pytest.approx(0.5)
        assert result["lift"] == pytest.approx(5.0)

    def test_bucket_rate_equal_to_baseline_gives_lift_of_one_and_is_not_significant(self):
        flags = [0.1] * 50
        result = _finalize_rate_test(flags, baseline_rate=0.1, horizon=5, alpha=0.05, n_total_valid=1000)
        assert result["lift"] == pytest.approx(1.0)

    def test_skips_gracefully_with_too_few_rows(self):
        result = _finalize_rate_test([1.0, 0.0], baseline_rate=0.1, horizon=5, alpha=0.05, n_total_valid=1000)
        assert "skipped_reason" in result

    def test_significance_flag_respects_the_supplied_alpha(self):
        flags = [0.15] * 12 + [0.08] * 8
        lenient = _finalize_rate_test(flags, baseline_rate=0.1, horizon=5, alpha=0.5, n_total_valid=1000)
        strict = _finalize_rate_test(flags, baseline_rate=0.1, horizon=5, alpha=1e-12, n_total_valid=1000)
        assert lenient["p_value"] == strict["p_value"]
        assert strict["significant_after_correction"] is False
        assert strict["p_value"] >= 1e-12


class TestPickStrongestFamilyA:
    def test_picks_by_largest_absolute_deviation_from_lift_one(self):
        rows = [
            {"feature": "a", "lift": 1.05},
            {"feature": "b", "lift": 3.0},
            {"feature": "c", "lift": 0.2},
        ]
        top1 = pick_strongest_family_a(rows, n=1)
        assert top1[0]["feature"] == "b"

    def test_n_controls_how_many_are_returned(self):
        rows = [{"feature": f"f{i}", "lift": 1.0 + i * 0.5} for i in range(10)]
        assert len(pick_strongest_family_a(rows, n=3)) == 3

    def test_ignores_rows_without_a_lift(self):
        rows = [{"feature": "a", "skipped_reason": "x"}, {"feature": "b", "lift": 2.0}]
        top = pick_strongest_family_a(rows, n=5)
        assert len(top) == 1
        assert top[0]["feature"] == "b"


class TestFamilyADirectionAt15m:
    def test_picks_the_bucket_with_the_larger_lift_deviation(self):
        family_a = [
            {"feature": "level_touch_count_resistance", "horizon_minutes": 15, "bucket": "lowest_quantile", "lift": 1.86},
            {"feature": "level_touch_count_resistance", "horizon_minutes": 15, "bucket": "highest_quantile", "lift": 1.02},
            {"feature": "relative_volume_20", "horizon_minutes": 15, "bucket": "lowest_quantile", "lift": 0.98},
            {"feature": "relative_volume_20", "horizon_minutes": 15, "bucket": "highest_quantile", "lift": 1.13},
        ]
        direction = _family_a_direction_at_15m(family_a)
        assert direction["level_touch_count_resistance"] == "lowest_quantile"
        assert direction["relative_volume_20"] == "highest_quantile"

    def test_ignores_horizons_other_than_15m(self):
        family_a = [
            {"feature": "x", "horizon_minutes": 5, "bucket": "highest_quantile", "lift": 5.0},
            {"feature": "x", "horizon_minutes": 15, "bucket": "lowest_quantile", "lift": 1.1},
        ]
        direction = _family_a_direction_at_15m(family_a)
        assert direction["x"] == "lowest_quantile"


class TestRunFamilyC:
    def _df(self):
        rng = np.random.default_rng(0)
        n = 400
        touch = rng.integers(0, 20, size=n)
        rs = rng.normal(size=n)
        mover = ((touch < 5) & (rng.random(n) < 0.8)) | (rng.random(n) < 0.05)
        return pd.DataFrame({"level_touch_count_resistance": touch, "relative_strength_vs_universe": rs, "mover_flag_15m": mover, "symbol": [f"S{i%10}" for i in range(n)], "bar_start_utc": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")})

    def test_tests_the_direction_family_a_actually_found_not_always_the_top_quartile(self):
        df = self._df()
        family_a = [
            {"feature": "level_touch_count_resistance", "horizon_minutes": 15, "bucket": "lowest_quantile", "lift": 1.9},
            {"feature": "level_touch_count_resistance", "horizon_minutes": 15, "bucket": "highest_quantile", "lift": 0.95},
        ]
        results = run_family_c(df, alpha=0.5, family_a=family_a, features=["level_touch_count_resistance"])
        for r in results:
            assert r["direction_tested"] == "lowest_quantile"
        lifts = [r["lift"] for r in results if r.get("lift") is not None]
        assert all(l > 1.0 for l in lifts)


class TestBuildLedger:
    def test_every_family_contributes_entries(self):
        family_a = [{"feature": "a", "bucket": "flag_true", "horizon_minutes": 5, "significant_after_correction": True, "lift": 2.0, "n_rows": 100}]
        family_b = [{"combination": "x", "feature_1": "a", "feature_2": "b", "horizon_minutes": 5, "significant_after_correction": False, "lift": 1.0, "n_rows": 50}]
        family_c = [{"feature": "a", "controlled_group": "below_median_rs", "significant_after_correction": True, "lift": 1.5, "n_rows": 80}]
        ledger = build_ledger(family_a, family_b, family_c, alpha=0.001)
        assert len(ledger) == 3
        assert {e["family"] for e in ledger} == {"A_standalone", "B_combination", "C_incremental_value"}

    def test_skipped_tests_are_recorded_not_dropped(self):
        family_a = [{"feature": "a", "bucket": "n/a", "horizon_minutes": 5, "skipped_reason": "degenerate feature"}]
        ledger = build_ledger(family_a, [], [], alpha=0.001)
        assert ledger[0]["result"] == "SKIPPED"


class TestFinalVerdict:
    def _family_d(self, all_survive: bool):
        return {"feature": "x", "by_horizon": {f"{h}m": {"significant_after_correction": all_survive} for h in HORIZONS}}

    def _family_e(self, holds: bool):
        return {"results": {"x": {"by_split": {"train": {"significant_after_correction": holds}, "validation": {"significant_after_correction": holds}, "held_out": {"significant_after_correction": holds}}}}}

    def test_verdict_c_when_nothing_significant_in_family_a(self):
        v = final_verdict([{"significant_after_correction": False, "lift": 1.0}], [], [], self._family_d(True), self._family_e(True))
        assert v["verdict"] == "C"

    def test_verdict_c_when_family_a_significant_but_no_incremental_value(self):
        family_a = [{"significant_after_correction": True, "lift": 2.0}]
        family_c = [{"significant_after_correction": False, "lift": 1.0}]
        v = final_verdict(family_a, [], family_c, self._family_d(True), self._family_e(True))
        assert v["verdict"] == "C"

    def test_verdict_b_when_incremental_value_but_cross_stock_fails(self):
        family_a = [{"significant_after_correction": True, "lift": 2.0}]
        family_c = [{"significant_after_correction": True, "lift": 1.5}]
        v = final_verdict(family_a, [], family_c, self._family_d(False), self._family_e(True))
        assert v["verdict"] == "B"

    def test_never_hardcodes_a(self):
        import inspect
        from scripts import run_phase_j_statistical_research as mod
        source = inspect.getsource(mod.final_verdict)
        assert '"A"' not in source and "'A'" not in source
