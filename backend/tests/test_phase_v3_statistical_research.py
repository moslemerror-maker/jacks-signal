from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_phase_v3_statistical_research import (
    HORIZONS,
    final_mechanism_verdicts,
    mechanism_masks,
    pick_survivors,
    run_economic_evaluation,
    run_incremental,
    run_standalone,
)
from scripts.run_phase_v3_statistical_research import test_budget as compute_test_budget


def _synthetic_dataset(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {
        "symbol": np.array([f"S{i % 12}" for i in range(n)]),
        "security_id": np.array([i % 12 for i in range(n)]),
        "trading_date": pd.date_range("2026-01-01", periods=n, freq="5min").date,
        "bar_start_utc": pd.date_range("2026-01-01", periods=n, freq="5min"),
        "minutes_since_open": rng.integers(0, 375, n),
        "distance_from_session_open": rng.normal(scale=0.01, size=n),
        "consecutive_same_direction_count": rng.integers(0, 10, n),
        "candle_direction": rng.choice([-1, 0, 1], size=n),
        "candle_body_ratio": rng.random(n),
        "relative_volume_20": rng.random(n) * 4,
        "return_5m": rng.normal(scale=0.005, size=n),
        "compression_then_expansion_flag": rng.random(n) < 0.05,
        "breakout_up_meaningful": rng.random(n) < 0.03,
        "breakout_down_meaningful": rng.random(n) < 0.03,
        "rolling_liquidity_group": rng.choice(["mid_liquidity", "high_liquidity"], size=n),
    }
    for h in HORIZONS:
        up = rng.random(n) < 0.33
        down = ~up & (rng.random(n) < 0.5)
        data[f"direction_label_pit_{h}"] = np.where(up, "UP", np.where(down, "DOWN", "NEUTRAL"))
        data[f"gross_return_{h}m"] = rng.normal(scale=0.005, size=n)
    df = pd.DataFrame(data)
    returns_5m = df.groupby("bar_start_utc")["return_5m"].transform("mean")
    df["relative_strength_vs_universe"] = df["return_5m"] - returns_5m
    return df


class TestTestBudget:
    def test_matches_frozen_design_totals(self):
        budget = compute_test_budget()
        assert budget["standalone"] == 4 * 2 * 4 == 32
        assert budget["incremental"] == 4 * 2 * 2 * 2 == 32
        assert sum(budget.values()) == 64

    def test_row_counts_match_declared_budget_on_synthetic_data(self):
        df = _synthetic_dataset()
        masks = mechanism_masks(df)
        alpha = 0.05 / sum(compute_test_budget().values())
        assert len(run_standalone(df, masks, alpha)) == compute_test_budget()["standalone"]
        assert len(run_incremental(df, masks, alpha)) == compute_test_budget()["incremental"]


class TestMechanismMasks:
    def test_exactly_four_mechanisms_each_with_up_and_down(self):
        df = _synthetic_dataset()
        masks = mechanism_masks(df)
        assert len(masks) == 4
        for sides in masks.values():
            assert set(sides.keys()) == {"up", "down"}

    def test_m1_requires_both_breakout_and_decisive_candle(self):
        df = _synthetic_dataset(n=10)
        df.loc[:, "breakout_up_meaningful"] = True
        df.loc[:5, "candle_body_ratio"] = 0.9
        df.loc[5:, "candle_body_ratio"] = 0.1
        masks = mechanism_masks(df)
        m1_up = masks["M1_breakout_continuation"]["up"]
        assert m1_up.iloc[:5].all()
        assert not m1_up.iloc[5:].any()

    def test_m3_requires_both_abnormal_volume_and_minimum_move(self):
        df = _synthetic_dataset(n=4)
        df.loc[:, "relative_volume_20"] = [3.0, 3.0, 0.5, 3.0]
        df.loc[:, "return_5m"] = [0.01, 0.0001, 0.01, -0.01]
        masks = mechanism_masks(df)
        assert masks["M3_abnormal_participation_continuation"]["up"].tolist() == [True, False, False, False]
        assert masks["M3_abnormal_participation_continuation"]["down"].tolist() == [False, False, False, True]

    def test_m4_requires_expansion_and_agreement_between_candle_and_return(self):
        df = _synthetic_dataset(n=4)
        df.loc[:, "compression_then_expansion_flag"] = True
        df.loc[:, "candle_direction"] = [1, 1, -1, -1]
        df.loc[:, "return_5m"] = [0.01, -0.01, -0.01, 0.01]
        masks = mechanism_masks(df)
        assert masks["M4_volatility_expansion_confirmation"]["up"].tolist() == [True, False, False, False]
        assert masks["M4_volatility_expansion_confirmation"]["down"].tolist() == [False, False, True, False]


class TestEconomicSignAdjustment:
    def test_down_side_flips_the_sign_before_cost(self):
        n = 200
        df = pd.DataFrame({
            "symbol": [f"S{i % 5}" for i in range(n)], "trading_date": pd.date_range("2026-01-01", periods=n, freq="5min").date,
            "gross_return_15m": [-0.01] * n, "rolling_liquidity_group": ["high_liquidity"] * n,
            "M1_breakout_continuation_up": [False] * n, "M1_breakout_continuation_down": [True] * n,
        })
        masks = {"M1_breakout_continuation": {"up": pd.Series([False] * n), "down": pd.Series([True] * n)}}
        survivors = [{"mechanism": "M1_breakout_continuation", "side": "down", "horizon_min": 15}]
        econ = run_economic_evaluation(df, masks, survivors)
        assert econ[0]["gross_mean"] == pytest.approx(0.01)


class TestPickSurvivorsAndVerdicts:
    def test_requires_both_standalone_and_all_four_incremental_cells(self):
        standalone = [{"mechanism": "M1_breakout_continuation", "side": "up", "horizon_min": 15, "significant_after_correction": True, "lift": 1.15}]
        incremental_all_sig = [
            {"mechanism": "M1_breakout_continuation", "side": "up", "control": "return_5m", "controlled_group": "below", "significant_after_correction": True},
            {"mechanism": "M1_breakout_continuation", "side": "up", "control": "return_5m", "controlled_group": "above", "significant_after_correction": True},
            {"mechanism": "M1_breakout_continuation", "side": "up", "control": "relative_strength_vs_universe", "controlled_group": "below", "significant_after_correction": True},
            {"mechanism": "M1_breakout_continuation", "side": "up", "control": "relative_strength_vs_universe", "controlled_group": "above", "significant_after_correction": True},
        ]
        assert len(pick_survivors(standalone, incremental_all_sig)) == 1

    def test_excludes_when_only_three_of_four_incremental_cells_significant(self):
        standalone = [{"mechanism": "M1_breakout_continuation", "side": "up", "horizon_min": 15, "significant_after_correction": True, "lift": 1.15}]
        incremental = [
            {"mechanism": "M1_breakout_continuation", "side": "up", "control": "return_5m", "controlled_group": "below", "significant_after_correction": True},
            {"mechanism": "M1_breakout_continuation", "side": "up", "control": "return_5m", "controlled_group": "above", "significant_after_correction": True},
            {"mechanism": "M1_breakout_continuation", "side": "up", "control": "relative_strength_vs_universe", "controlled_group": "below", "significant_after_correction": True},
            {"mechanism": "M1_breakout_continuation", "side": "up", "control": "relative_strength_vs_universe", "controlled_group": "above", "significant_after_correction": False},
        ]
        assert pick_survivors(standalone, incremental) == []

    def test_regression_significant_but_lift_below_one_is_excluded(self):
        standalone = [{"mechanism": "M1_breakout_continuation", "side": "up", "horizon_min": 15, "significant_after_correction": True, "lift": 0.93}]
        incremental_all_sig = [
            {"mechanism": "M1_breakout_continuation", "side": "up", "control": "return_5m", "controlled_group": "below", "significant_after_correction": True},
            {"mechanism": "M1_breakout_continuation", "side": "up", "control": "return_5m", "controlled_group": "above", "significant_after_correction": True},
            {"mechanism": "M1_breakout_continuation", "side": "up", "control": "relative_strength_vs_universe", "controlled_group": "below", "significant_after_correction": True},
            {"mechanism": "M1_breakout_continuation", "side": "up", "control": "relative_strength_vs_universe", "controlled_group": "above", "significant_after_correction": True},
        ]
        assert pick_survivors(standalone, incremental_all_sig) == []

    def test_verdict_failed_with_no_survivors(self):
        v = final_mechanism_verdicts([], [], [], [], {"by_survivor": []}, [], [])
        assert v["M1_breakout_continuation"]["verdict"] == "FAILED"
        assert v["M1_breakout_continuation"]["failed_stage"] == "standalone_or_incremental_significance"

    def test_verdict_failed_at_cross_stock(self):
        survivors = [{"mechanism": "M1_breakout_continuation", "side": "up", "horizon_min": 15}]
        cross_stock = [{"mechanism": "M1_breakout_continuation", "side": "up", "horizon_min": 15, "significant_after_correction": False}]
        v = final_mechanism_verdicts([], [], survivors, cross_stock, {"by_survivor": []}, [], [])
        assert v["M1_breakout_continuation"]["verdict"] == "FAILED"
        assert v["M1_breakout_continuation"]["failed_stage"] == "cross_stock_generalization"

    def test_verdict_statistically_robust_but_economically_unviable(self):
        survivors = [{"mechanism": "M1_breakout_continuation", "side": "up", "horizon_min": 15}]
        cross_stock = [{"mechanism": "M1_breakout_continuation", "side": "up", "horizon_min": 15, "significant_after_correction": True}]
        chrono = {"by_survivor": [{"mechanism": "M1_breakout_continuation", "side": "up", "horizon_min": 15,
                                    "train": {"significant_after_correction": True}, "validation": {"significant_after_correction": True}, "held_out": {"significant_after_correction": True}}]}
        economic = [{"mechanism": "M1_breakout_continuation", "side": "up", "horizon_min": 15, "scenarios": {"C_conservative": {"mean_net": -0.001}}}]
        v = final_mechanism_verdicts([], [], survivors, cross_stock, chrono, [], economic)
        assert v["M1_breakout_continuation"]["verdict"] == "STATISTICALLY_ROBUST_ECONOMICALLY_UNVIABLE"

    def test_verdict_viable_candidate_when_everything_holds(self):
        survivors = [{"mechanism": "M1_breakout_continuation", "side": "up", "horizon_min": 15}]
        cross_stock = [{"mechanism": "M1_breakout_continuation", "side": "up", "horizon_min": 15, "significant_after_correction": True}]
        chrono = {"by_survivor": [{"mechanism": "M1_breakout_continuation", "side": "up", "horizon_min": 15,
                                    "train": {"significant_after_correction": True}, "validation": {"significant_after_correction": True}, "held_out": {"significant_after_correction": True}}]}
        economic = [{"mechanism": "M1_breakout_continuation", "side": "up", "horizon_min": 15, "scenarios": {"C_conservative": {"mean_net": 0.001}}}]
        v = final_mechanism_verdicts([], [], survivors, cross_stock, chrono, [], economic)
        assert v["M1_breakout_continuation"]["verdict"] == "VIABLE_CANDIDATE"
