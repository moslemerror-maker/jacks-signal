from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_phase_v6_statistical_research import ALL_HORIZONS, DISCOVERY_HORIZONS, TOP_N_VALUES
from scripts.run_phase_v7a_opportunity_intensity_tod import (
    TOD_BUCKETS,
    baseline_asymmetry_by_tod,
    final_classification,
    opening_confound_summary,
    pick_survivors_by_tod,
    run_discovery_by_tod,
    test_budget_v7a as compute_test_budget_v7a,
)


def _synthetic_dataset(n_per_bucket: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bucket_level = {"opening": 0.01, "mid": 0.001, "late": 0.001}
    rows = []
    t = pd.Timestamp("2026-01-05 03:45", tz="UTC")
    for tod in TOD_BUCKETS:
        for i in range(n_per_bucket):
            symbol = f"S{i % 50}"
            intensity_rank_bonus = 0.0005 if i < 50 else 0.0
            rows.append({
                "symbol": symbol, "security_id": i % 50,
                "trading_date": pd.Timestamp("2026-01-05").date(),
                "bar_start_utc": t + pd.Timedelta(minutes=i % 50),
                "time_of_day_bucket": tod,
                "relative_volume_20": 3.0 if i < 50 else 1.0,
                "volatility_expansion_ratio": 1.5 if i < 50 else 1.0,
                "rolling_liquidity_group": "high_liquidity",
                **{f"mfe_o1_h{h}": bucket_level[tod] + intensity_rank_bonus + rng.normal(scale=0.0005) for h in ALL_HORIZONS},
                **{f"mae_o1_h{h}": abs(rng.normal(scale=0.0005)) for h in ALL_HORIZONS},
                **{f"gross_return_o1_h{h}": rng.normal(scale=0.001) for h in ALL_HORIZONS},
            })
    return pd.DataFrame(rows)


class TestBaselineByTod:
    def test_opening_baseline_differs_sharply_from_mid_and_late(self):
        df = _synthetic_dataset()
        b_opening = baseline_asymmetry_by_tod(df, 15, "opening")
        b_mid = baseline_asymmetry_by_tod(df, 15, "mid")
        assert b_opening > b_mid * 3

    def test_bucket_conditioned_baseline_differs_from_pooled_baseline(self):
        from scripts.run_phase_v6_statistical_research import baseline_asymmetry
        df = _synthetic_dataset()
        pooled = baseline_asymmetry(df, 15, True)
        opening_conditioned = baseline_asymmetry_by_tod(df, 15, "opening")
        assert opening_conditioned != pytest.approx(pooled)


class TestBudget:
    def test_discovery_budget_is_18(self):
        budget = compute_test_budget_v7a()
        assert budget["discovery_by_tod"] == len(TOD_BUCKETS) * len(TOP_N_VALUES) * len(DISCOVERY_HORIZONS) == 18

    def test_confirmatory_budget_is_72(self):
        budget = compute_test_budget_v7a()
        assert budget["confirmatory_by_tod"] == len(TOD_BUCKETS) * len(TOP_N_VALUES) * len(ALL_HORIZONS) * 2 == 72

    def test_discovery_row_count_matches_budget_on_synthetic_data(self):
        df = _synthetic_dataset()
        alpha = 0.05 / compute_test_budget_v7a()["discovery_by_tod"]
        results = run_discovery_by_tod(df, alpha)
        assert len(results) == compute_test_budget_v7a()["discovery_by_tod"]


class TestPickSurvivorsByTod:
    def test_excludes_significant_but_realized_below_baseline(self):
        results = [{"time_of_day": "opening", "top_n": 10, "horizon_min": 15, "significant": True, "lift": 1.2, "realized_mean": 0.0001, "baseline_mean": 0.0002}]
        assert pick_survivors_by_tod(results) == []

    def test_includes_significant_with_realized_above_baseline(self):
        results = [{"time_of_day": "opening", "top_n": 10, "horizon_min": 15, "significant": True, "lift": 1.2, "realized_mean": 0.0002, "baseline_mean": 0.0001}]
        assert len(pick_survivors_by_tod(results)) == 1

    def test_regression_negative_baseline_sign_flip_is_excluded(self):
        results = [{"time_of_day": "late", "top_n": 50, "horizon_min": 30, "significant": True, "lift": 5.0, "realized_mean": -0.0008, "baseline_mean": -0.0001}]
        assert pick_survivors_by_tod(results) == []


class TestTopNFieldNeverCollidesWithSampleSize:

    def test_discovery_output_top_n_is_the_rank_parameter_not_the_sample_size(self):
        df = _synthetic_dataset()
        alpha = 0.05 / compute_test_budget_v7a()["discovery_by_tod"]
        results = run_discovery_by_tod(df, alpha)
        for row in results:
            assert row["top_n"] in TOP_N_VALUES, row
            assert row["top_n"] != row["n"]


class TestOpeningConfoundSummary:
    def test_reports_a_row_share_and_baseline_for_every_bucket(self):
        df = _synthetic_dataset()
        summary = opening_confound_summary(df)
        assert set(summary["bucket_row_share"].keys()) == set(TOD_BUCKETS)
        assert set(summary["bucket_unconditional_baseline"].keys()) == set(TOD_BUCKETS)
        assert summary["bucket_unconditional_baseline"]["opening"][15] > summary["bucket_unconditional_baseline"]["mid"][15]


class TestFinalClassification:
    def test_classification_a_when_no_survivors(self):
        result = final_classification([], [], [], [])
        assert result["classification"] == "A"

    def test_classification_b_when_survivor_fails_confirmatory(self):
        survivors = [{"time_of_day": "opening", "top_n": 10, "horizon_min": 15, "realized_mean": 0.0002, "baseline_mean": 0.0001}]
        confirmatory = [
            {"split": "validation", "time_of_day": "opening", "top_n": 10, "horizon_min": 15, "significant": False, "realized_mean": 0.0001, "baseline_mean": 0.0001},
            {"split": "held_out", "time_of_day": "opening", "top_n": 10, "horizon_min": 15, "significant": False, "realized_mean": 0.0001, "baseline_mean": 0.0001},
        ]
        cross_stock = [{"significant": False, "realized_mean": 0.0001, "baseline_mean": 0.0001}]
        economic = [{"scenarios": {}}]
        result = final_classification(survivors, confirmatory, cross_stock, economic)
        assert result["classification"] == "B"

    def _replicating_survivor(self):
        survivors = [{"time_of_day": "late", "top_n": 20, "horizon_min": 15, "realized_mean": 0.0002, "baseline_mean": 0.0001}]
        confirmatory = [
            {"split": "validation", "time_of_day": "late", "top_n": 20, "horizon_min": 15, "significant": True, "realized_mean": 0.0003, "baseline_mean": 0.0001},
            {"split": "held_out", "time_of_day": "late", "top_n": 20, "horizon_min": 15, "significant": True, "realized_mean": 0.0003, "baseline_mean": 0.0001},
        ]
        cross_stock = [{"significant": True, "realized_mean": 0.0003, "baseline_mean": 0.0001}]
        return survivors, confirmatory, cross_stock

    def test_classification_b_when_replication_holds_but_gross_mean_is_negative(self):
        survivors, confirmatory, cross_stock = self._replicating_survivor()
        economic = [{"gross_mean": -0.0001, "scenarios": {"A_optimistic": {"mean_net": -0.0002}, "C_conservative": {"mean_net": -0.0003}}}]
        result = final_classification(survivors, confirmatory, cross_stock, economic)
        assert result["classification"] == "B"
        assert result["cells"][0]["economic_hopeless"] is True

    def test_classification_c_when_gross_positive_but_fails_conservative_cost_bar(self):
        survivors, confirmatory, cross_stock = self._replicating_survivor()
        economic = [{"gross_mean": 0.0005, "scenarios": {"A_optimistic": {"mean_net": 0.0002}, "C_conservative": {"mean_net": -0.0001}}}]
        result = final_classification(survivors, confirmatory, cross_stock, economic)
        assert result["classification"] == "C"

    def test_classification_d_when_conservative_bar_is_cleared(self):
        survivors, confirmatory, cross_stock = self._replicating_survivor()
        economic = [{"gross_mean": 0.0008, "scenarios": {"A_optimistic": {"mean_net": 0.0006}, "C_conservative": {"mean_net": 0.0001}}}]
        result = final_classification(survivors, confirmatory, cross_stock, economic)
        assert result["classification"] == "D"
