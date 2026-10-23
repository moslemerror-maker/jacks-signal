from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_phase_u_statistical_research import (
    BUCKETS,
    COST_MODEL,
    ECONOMIC_BAR_SCENARIO,
    EXECUTION_CONVENTIONS,
    HORIZONS,
    IS_LONG_BUCKET,
    LIQUIDITY_TIERS,
    bucket_masks,
    final_verdict,
    pick_survivors,
    run_execution_timing,
    run_liquidity_stratified,
    run_participation_capacity,
    three_way_assessment,
)
from scripts.run_phase_u_statistical_research import test_budget as compute_test_budget


def _synthetic_dataset(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {
        "symbol": np.array([f"S{i % 15}" for i in range(n)]),
        "security_id": np.array([i % 15 for i in range(n)]),
        "trading_date": pd.date_range("2026-01-01", periods=n, freq="5min").date,
        "rolling_liquidity_group": rng.choice(list(LIQUIDITY_TIERS), size=n),
        "rolling_median_daily_traded_value_20d": rng.random(n) * 1e7,
        "avg_1m_volume_20": rng.random(n) * 10000 + 100,
        "entry_price_o1": rng.random(n) * 500 + 50,
    }
    for b in BUCKETS:
        data[b] = rng.random(n) < 0.1
    for _, tag in EXECUTION_CONVENTIONS.values():
        for h in HORIZONS:
            data[f"gross_return_{tag}_h{h}"] = rng.normal(scale=0.004, size=n)
    return pd.DataFrame(data)


class TestBucketsAndDirection:
    def test_four_buckets_correct_implied_sides(self):
        assert IS_LONG_BUCKET == {
            "bottom_1pct_rank15m": True, "bottom_2pct_rank15m": True,
            "top_1pct_rank15m": False, "top_2pct_rank15m": False,
        }


class TestTestBudget:
    def test_matches_frozen_design_totals(self):
        budget = compute_test_budget()
        assert budget["execution_timing"] == 4 * 6 * 2 == 48
        assert budget["liquidity_stratified"] == 4 * 3 * 2 == 24
        assert sum(budget.values()) == 72

    def test_row_counts_match_declared_budget_on_synthetic_data(self):
        df = _synthetic_dataset()
        masks = bucket_masks(df)
        alpha = 0.05 / sum(compute_test_budget().values())
        assert len(run_execution_timing(df, masks, alpha)) == compute_test_budget()["execution_timing"]
        assert len(run_liquidity_stratified(df, masks, alpha)) == compute_test_budget()["liquidity_stratified"]


class TestCostModel:
    def test_four_scenarios_three_tiers_each(self):
        assert set(COST_MODEL.keys()) == {"A_optimistic", "B_moderate", "C_conservative", "D_severe"}
        for scenario, tiers in COST_MODEL.items():
            assert set(tiers.keys()) == set(LIQUIDITY_TIERS)

    def test_cost_increases_with_scenario_severity_within_each_tier(self):
        for tier in LIQUIDITY_TIERS:
            values = [COST_MODEL[s][tier] for s in ("A_optimistic", "B_moderate", "C_conservative", "D_severe")]
            assert values == sorted(values)

    def test_cost_increases_as_liquidity_worsens_within_each_scenario(self):
        for scenario in COST_MODEL:
            tiers = COST_MODEL[scenario]
            assert tiers["high_liquidity"] < tiers["mid_liquidity"] < tiers["low_liquidity"]

    def test_economic_bar_is_the_conservative_scenario(self):
        assert ECONOMIC_BAR_SCENARIO == "C_conservative"

    def test_conservative_low_liquidity_matches_the_projects_existing_60bps_round_trip_convention(self):
        assert COST_MODEL["C_conservative"]["low_liquidity"] == 60.0

    def test_per_event_net_return_subtracts_the_round_trip_value_directly_not_doubled(self):
        from scripts.run_phase_u_statistical_research import _per_event_net_return

        n = 10
        df = pd.DataFrame({
            "gross_return_o1_h15": [0.01] * n,
            "rolling_liquidity_group": ["low_liquidity"] * n,
        })
        mask = pd.Series([True] * n)
        net = _per_event_net_return(df, mask, "next_open", 15, is_long=True, scenario="C_conservative")
        assert net.iloc[0] == pytest.approx(0.01 - 60.0 / 10_000.0)


class TestSignAdjustment:
    def test_short_bucket_return_is_sign_flipped_before_testing(self):
        n = 200
        df = pd.DataFrame({
            "symbol": [f"S{i % 5}" for i in range(n)], "security_id": [i % 5 for i in range(n)],
            "trading_date": pd.date_range("2026-01-01", periods=n, freq="5min").date,
            "top_1pct_rank15m": True, "top_2pct_rank15m": True,
            "bottom_1pct_rank15m": False, "bottom_2pct_rank15m": False,
        })
        for _, tag in EXECUTION_CONVENTIONS.values():
            for h in HORIZONS:
                df[f"gross_return_{tag}_h{h}"] = -0.01
        masks = bucket_masks(df)
        results = run_execution_timing(df, masks, alpha=0.05)
        top_cells = [r for r in results if r["bucket"] == "top_1pct_rank15m"]
        assert all(r["mean"] == pytest.approx(0.01) for r in top_cells)


class TestParticipationCapacity:
    def test_runs_without_missing_columns_on_a_properly_shaped_frame(self):
        df = _synthetic_dataset()
        masks = bucket_masks(df)
        result = run_participation_capacity(df, masks)
        assert set(result.keys()) == set(BUCKETS)
        for bucket_result in result.values():
            for tier_result in bucket_result.values():
                assert "n" in tier_result

    def test_load_dataset_real_files_include_entry_price_o1(self):
        import pandas as pd

        from scripts.run_phase_u_statistical_research import PHASE_T_FILE, load_dataset

        if not PHASE_T_FILE.exists():
            pytest.skip("Phase T output not present in this environment")
        df = load_dataset()
        assert "entry_price_o1" in df.columns


class TestPickSurvivorsAndVerdict:
    def test_requires_significance_and_positive_mean(self):
        rows = [
            {"bucket": "b", "significant_after_correction": True, "mean": 0.001},
            {"bucket": "b", "significant_after_correction": True, "mean": -0.001},
            {"bucket": "b", "significant_after_correction": False, "mean": 0.001},
        ]
        assert len(pick_survivors(rows)) == 1

    def test_three_way_assessment_no_statistical_edge(self):
        out = three_way_assessment([])
        assert out["statistical_edge"] is False
        assert out["economic_edge_at_conservative_scenario"] is False

    def test_three_way_assessment_statistical_but_not_economic(self):
        econ_rows = [{"bucket": "b", "horizon_min": 15, "scenarios": {"C_conservative": {"mean_net": -0.001}}}]
        out = three_way_assessment(econ_rows)
        assert out["statistical_edge"] is True
        assert out["economic_edge_at_conservative_scenario"] is False

    def test_verdict_e_with_no_statistical_survivors(self):
        v = final_verdict(three_way_assessment([]), cross_stock_holds=False, chrono_holds=False, liquidity_extreme_holds=False)
        assert v["verdict"] == "E"

    def test_verdict_c_statistical_but_not_economic(self):
        econ_rows = [{"bucket": "b", "horizon_min": 15, "scenarios": {"C_conservative": {"mean_net": -0.001}}}]
        v = final_verdict(three_way_assessment(econ_rows), cross_stock_holds=True, chrono_holds=True, liquidity_extreme_holds=True)
        assert v["verdict"] == "C"

    def test_verdict_a_when_everything_holds(self):
        econ_rows = [{"bucket": "b", "horizon_min": 15, "scenarios": {"C_conservative": {"mean_net": 0.001}}}]
        v = final_verdict(three_way_assessment(econ_rows), cross_stock_holds=True, chrono_holds=True, liquidity_extreme_holds=True)
        assert v["verdict"] == "A"

    def test_verdict_b_when_economic_holds_but_generalization_does_not(self):
        econ_rows = [{"bucket": "b", "horizon_min": 15, "scenarios": {"C_conservative": {"mean_net": 0.001}}}]
        v = final_verdict(three_way_assessment(econ_rows), cross_stock_holds=True, chrono_holds=False, liquidity_extreme_holds=True)
        assert v["verdict"] == "B"
