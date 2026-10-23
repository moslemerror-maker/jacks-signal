from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_phase_l_statistical_research import (
    BINARY_QUANTITIES,
    CATEGORICAL_QUANTITIES,
    COMBINATIONS,
    CONTINUOUS_QUANTITIES,
    DIRECTION_THRESHOLD,
    POPULATIONS,
    _finalize_rate_test,
    build_ledger,
    final_verdict,
    pick_survivors,
)
from scripts.run_phase_l_statistical_research import test_budget as compute_test_budget


class TestDirectionThreshold:
    def test_frozen_at_20_bps(self):
        assert DIRECTION_THRESHOLD == 0.0020


class TestTestBudget:
    def test_matches_exact_formula_derived_from_the_same_candidate_lists(self):
        b = compute_test_budget()
        per_pop_per_horizon = len(CONTINUOUS_QUANTITIES) * 2 + len(BINARY_QUANTITIES) + sum(len(s) for s in CATEGORICAL_QUANTITIES.values())
        assert b["A_standalone"] == per_pop_per_horizon * len(POPULATIONS) * 4
        assert b["B_combinations"] == len(COMBINATIONS) * len(POPULATIONS) * 2
        assert b["C_incremental_value"] == (len(CONTINUOUS_QUANTITIES) + len(BINARY_QUANTITIES) + len(CATEGORICAL_QUANTITIES)) * len(POPULATIONS) * 2

    def test_is_deterministic(self):
        assert compute_test_budget() == compute_test_budget()

    def test_row_count_produced_by_family_a_matches_the_budgeted_count(self):
        import numpy as np
        from scripts.run_phase_l_statistical_research import run_family_a

        rng = np.random.default_rng(0)
        n = 3000
        df = pd.DataFrame({
            "symbol": [f"S{i % 10}" for i in range(n)],
            "bar_start_utc": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC"),
            "pop_M4": [True] * n, "pop_M3": [i % 2 == 0 for i in range(n)],
            "g1_support_minus_resistance_touch": rng.normal(size=n),
            "relative_strength_vs_universe": rng.normal(size=n),
            "g5_directional_pressure_10": rng.normal(size=n),
            "g6_opening_range_position": rng.random(n),
            "candle_body_ratio": rng.random(n),
            "overnight_gap": rng.normal(size=n),
            "momentum_acceleration_10": rng.normal(size=n),
            "g2_support_also_fresh": rng.random(n) < 0.2,
            "g3b_above_vwap": rng.random(n) < 0.5,
            "vwap_cross": rng.choice(["RECLAIM", "REJECTION", "NONE"], size=n),
            "g7a_candle_direction": rng.choice([1, -1], size=n),
        })
        for h in (5, 15, 30, 60):
            up = rng.random(n) < 0.4
            down = ~up & (rng.random(n) < 0.6)
            label = np.where(up, "UP", np.where(down, "DOWN", "AMBIGUOUS"))
            df[f"direction_label_{h}m"] = label

        alpha = 0.05 / sum(compute_test_budget().values())
        family_a = run_family_a(df, alpha)
        assert len(family_a) == compute_test_budget()["A_standalone"]

    def test_row_count_produced_by_family_c_matches_the_budgeted_count(self):
        import numpy as np
        from scripts.run_phase_l_statistical_research import run_family_c

        rng = np.random.default_rng(1)
        n = 3000
        df = pd.DataFrame({
            "symbol": [f"S{i % 10}" for i in range(n)],
            "bar_start_utc": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC"),
            "pop_M4": [True] * n, "pop_M3": [i % 2 == 0 for i in range(n)],
            "return_5m": rng.normal(size=n),
            "g1_support_minus_resistance_touch": rng.normal(size=n),
            "relative_strength_vs_universe": rng.normal(size=n),
            "g5_directional_pressure_10": rng.normal(size=n),
            "g6_opening_range_position": rng.random(n),
            "candle_body_ratio": rng.random(n),
            "overnight_gap": rng.normal(size=n),
            "momentum_acceleration_10": rng.normal(size=n),
            "g2_support_also_fresh": rng.random(n) < 0.2,
            "g3b_above_vwap": rng.random(n) < 0.5,
            "vwap_cross": rng.choice(["RECLAIM", "REJECTION", "NONE"], size=n),
            "g7a_candle_direction": rng.choice([1, -1], size=n),
        })
        up = rng.random(n) < 0.4
        down = ~up & (rng.random(n) < 0.6)
        df["direction_label_15m"] = np.where(up, "UP", np.where(down, "DOWN", "AMBIGUOUS"))

        alpha = 0.05 / sum(compute_test_budget().values())
        family_c = run_family_c(df, alpha)
        assert len(family_c) == compute_test_budget()["C_incremental_value"]


class TestFinalizeRateTest:
    def test_bucket_up_rate_above_baseline_gives_lift_greater_than_one(self):
        flags = [1.0] * 30 + [0.0] * 10
        result = _finalize_rate_test(flags, baseline_rate=0.4, horizon=5, alpha=0.5)
        assert result["bucket_up_rate"] == pytest.approx(0.75)
        assert result["lift"] == pytest.approx(0.75 / 0.4)

    def test_skips_gracefully_with_too_few_rows(self):
        result = _finalize_rate_test([1.0, 0.0], baseline_rate=0.4, horizon=5, alpha=0.5)
        assert "skipped_reason" in result


class TestPickSurvivors:
    def test_requires_significance_in_both_family_a_and_both_family_c_halves(self):
        family_a = [{"population": "M4", "quantity": "g1", "bucket": "lowest_quintile", "significant_after_correction": True}]
        family_c_both_sig = [
            {"population": "M4", "quantity": "g1", "controlled_group": "below_median_return5m", "significant_after_correction": True},
            {"population": "M4", "quantity": "g1", "controlled_group": "at_or_above_median_return5m", "significant_after_correction": True},
        ]
        survivors = pick_survivors(family_a, family_c_both_sig)
        assert len(survivors) == 1

    def test_excludes_when_only_one_family_c_half_is_significant(self):
        family_a = [{"population": "M4", "quantity": "g1", "bucket": "lowest_quintile", "significant_after_correction": True}]
        family_c_one_sig = [
            {"population": "M4", "quantity": "g1", "controlled_group": "below_median_return5m", "significant_after_correction": True},
            {"population": "M4", "quantity": "g1", "controlled_group": "at_or_above_median_return5m", "significant_after_correction": False},
        ]
        survivors = pick_survivors(family_a, family_c_one_sig)
        assert survivors == []

    def test_excludes_when_family_a_itself_not_significant(self):
        family_a = [{"population": "M4", "quantity": "g1", "bucket": "lowest_quintile", "significant_after_correction": False}]
        family_c = [
            {"population": "M4", "quantity": "g1", "controlled_group": "below_median_return5m", "significant_after_correction": True},
            {"population": "M4", "quantity": "g1", "controlled_group": "at_or_above_median_return5m", "significant_after_correction": True},
        ]
        assert pick_survivors(family_a, family_c) == []


class TestFinalVerdict:
    def test_verdict_d_with_no_survivors(self):
        v = final_verdict([], [], {}, {})
        assert v["verdict"] == "D"
        assert v["n_survivors"] == 0

    def test_verdict_b_when_survivor_holds_cross_stock_and_chronologically(self):
        survivors = [{"population": "M4", "quantity": "g1", "bucket": "lowest_quintile"}]
        cross_stock = {"M4_g1_lowest_quintile": {"5m": {"significant_after_correction": True}}}
        chrono = {"by_survivor": {"M4_g1_lowest_quintile": {"5m": {"train": {"significant_after_correction": True}, "validation": {"significant_after_correction": True}, "held_out": {"significant_after_correction": True}}}}}
        v = final_verdict([], survivors, cross_stock, chrono)
        assert v["verdict"] == "B"

    def test_verdict_c_when_only_partially_replicates(self):
        survivors = [{"population": "M4", "quantity": "g1", "bucket": "lowest_quintile"}]
        cross_stock = {"M4_g1_lowest_quintile": {"5m": {"significant_after_correction": True}}}
        chrono = {"by_survivor": {"M4_g1_lowest_quintile": {"5m": {"train": {"significant_after_correction": True}, "validation": {"significant_after_correction": False}, "held_out": {"significant_after_correction": True}}}}}
        v = final_verdict([], survivors, cross_stock, chrono)
        assert v["verdict"] == "C"

    def test_never_hardcodes_a(self):
        import inspect
        from scripts import run_phase_l_statistical_research as mod
        source = inspect.getsource(mod.final_verdict)
        assert '"A"' not in source and "'A'" not in source


class TestBuildLedger:
    def test_records_every_family_including_failures(self):
        family_a = [{"quantity": "g1", "bucket": "lowest_quintile", "population": "M4", "horizon_minutes": 5, "significant_after_correction": False, "lift": 1.0, "n_rows": 100}]
        ledger = build_ledger(family_a, [], [], alpha=0.001)
        assert ledger[0]["result"] == "NOT_SIGNIFICANT"
