from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_phase_q_statistical_research import (
    CATEGORICAL_CANDIDATES,
    COMBINATIONS,
    CONTINUOUS_CANDIDATES,
    HORIZONS,
    POPULATIONS,
    combo_breakout_x_volume_x_rs_down,
    combo_breakout_x_volume_x_rs_up,
    final_verdict,
    pick_survivors,
    run_family_a,
    run_family_b,
)
from scripts.run_phase_q_statistical_research import test_budget as compute_test_budget

UTC = "UTC"


def _synthetic_dataset(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "symbol": [f"S{i % 12}" for i in range(n)],
        "bar_start_utc": pd.date_range("2026-01-01", periods=n, freq="min", tz=UTC),
        "trading_date": pd.date_range("2026-01-01", periods=n, freq="min", tz=UTC).date,
        "pop_FULL": [True] * n, "pop_M4": [i % 3 == 0 for i in range(n)], "pop_M3": [i % 2 == 0 for i in range(n)],
        "return_5m": rng.normal(size=n), "relative_strength_vs_universe": rng.normal(size=n),
        "universe_breadth": rng.random(n),
        "recent_breakout_up_opening_range": rng.random(n) < 0.1,
        "recent_breakout_down_opening_range": rng.random(n) < 0.1,
    })
    for col in CONTINUOUS_CANDIDATES:
        df[col] = rng.normal(size=n)
    for col, states in CATEGORICAL_CANDIDATES.items():
        df[col] = rng.choice(states, size=n)
    for h in HORIZONS:
        up = rng.random(n) < 0.4
        down = ~up & (rng.random(n) < 0.6)
        df[f"direction_label_{h}m"] = np.where(up, "UP", np.where(down, "DOWN", "AMBIGUOUS"))
        df[f"gross_return_{h}m"] = rng.normal(scale=0.005, size=n)
    return df


class TestTestBudget:
    def test_is_deterministic(self):
        assert compute_test_budget() == compute_test_budget()

    def test_row_count_produced_by_family_a_matches_the_budgeted_count(self):
        df = _synthetic_dataset()
        alpha = 0.05 / sum(compute_test_budget().values())
        family_a = run_family_a(df, alpha)
        assert len(family_a) == compute_test_budget()["A_standalone"]

    def test_row_count_produced_by_family_b_matches_the_budgeted_count(self):
        df = _synthetic_dataset()
        alpha = 0.05 / sum(compute_test_budget().values())
        family_b = run_family_b(df, alpha)
        assert len(family_b) == compute_test_budget()["B_incremental_value"]

    def test_budget_covers_all_11_families_worth_of_candidates(self):
        n_standalone_quantities = len(CONTINUOUS_CANDIDATES) + len(CATEGORICAL_CANDIDATES) + len(COMBINATIONS)
        assert n_standalone_quantities >= 11


class TestCombinationsAreDirectionallyDisjoint:
    def test_breakout_up_and_breakout_down_combinations_never_overlap(self):
        df = _synthetic_dataset()
        pop_mask = df["pop_FULL"]
        up_mask = combo_breakout_x_volume_x_rs_up(df, pop_mask)
        down_mask = combo_breakout_x_volume_x_rs_down(df, pop_mask)
        assert not (up_mask & down_mask).any()


class TestPickSurvivors:
    def test_requires_significance_in_family_a_and_all_four_family_b_cells(self):
        family_a = [{"population": "FULL", "quantity": "rsi_14", "bucket": "lowest_quintile", "significant_after_correction": True}]
        family_b_all_sig = [
            {"population": "FULL", "quantity": "rsi_14", "control": "return_5m", "controlled_group": "below", "significant_after_correction": True},
            {"population": "FULL", "quantity": "rsi_14", "control": "return_5m", "controlled_group": "above", "significant_after_correction": True},
            {"population": "FULL", "quantity": "rsi_14", "control": "relative_strength_vs_universe", "controlled_group": "below", "significant_after_correction": True},
            {"population": "FULL", "quantity": "rsi_14", "control": "relative_strength_vs_universe", "controlled_group": "above", "significant_after_correction": True},
        ]
        assert len(pick_survivors(family_a, family_b_all_sig)) == 1

    def test_excludes_when_only_three_of_four_family_b_cells_are_significant(self):
        family_a = [{"population": "FULL", "quantity": "rsi_14", "bucket": "lowest_quintile", "significant_after_correction": True}]
        family_b_partial = [
            {"population": "FULL", "quantity": "rsi_14", "control": "return_5m", "controlled_group": "below", "significant_after_correction": True},
            {"population": "FULL", "quantity": "rsi_14", "control": "return_5m", "controlled_group": "above", "significant_after_correction": True},
            {"population": "FULL", "quantity": "rsi_14", "control": "relative_strength_vs_universe", "controlled_group": "below", "significant_after_correction": True},
            {"population": "FULL", "quantity": "rsi_14", "control": "relative_strength_vs_universe", "controlled_group": "above", "significant_after_correction": False},
        ]
        assert pick_survivors(family_a, family_b_partial) == []


class TestFinalVerdict:
    def test_verdict_d_with_no_survivors(self):
        v = final_verdict([], {}, {}, {})
        assert v["verdict"] == "D"
        assert v["n_survivors"] == 0

    def test_verdict_a_requires_all_three_gates_simultaneously(self):
        survivors = [{"population": "FULL", "quantity": "rsi_14", "bucket": "lowest_quintile"}]
        cross_stock = {"k": {"5m": {"significant_after_correction": True}}}
        chrono = {"by_survivor": {"k": {"5m": {"train": {"significant_after_correction": True}, "validation": {"significant_after_correction": True}, "held_out": {"significant_after_correction": True}}}}}
        economic_positive = {"k": {"5m": {"cost_scenarios": {"high_60bps": {"mean_net": 0.001}}}}}
        economic_negative = {"k": {"5m": {"cost_scenarios": {"high_60bps": {"mean_net": -0.001}}}}}

        assert final_verdict(survivors, cross_stock, chrono, economic_positive)["verdict"] == "A"
        assert final_verdict(survivors, cross_stock, chrono, economic_negative)["verdict"] == "B"
