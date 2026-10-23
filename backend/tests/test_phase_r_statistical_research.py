from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from scripts.run_phase_r_statistical_research import (
    HORIZONS,
    build_candidates,
    final_verdict,
    pick_survivors,
    run_economic_evaluation,
    run_family_a,
    run_family_b,
)
from scripts.run_phase_r_statistical_research import test_budget as compute_test_budget

UTC = "UTC"


def _synthetic_dense_dataset(n_timestamps: int = 300, n_symbols: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-05 03:45", periods=n_timestamps, freq="5min", tz=UTC)
    rows = []
    for t in timestamps:
        for s in range(n_symbols):
            rows.append({
                "symbol": f"S{s}", "security_id": s, "bar_start_utc": t,
                "trading_date": t.date(),
                "return_1m": rng.normal(scale=0.002), "return_5m": rng.normal(scale=0.004), "return_15m": rng.normal(scale=0.007),
                "relative_volume_20": rng.random() * 3, "overnight_gap": rng.normal(scale=0.005),
                "distance_from_vwap": rng.normal(scale=0.003),
                "recent_breakout_up_opening_range": rng.random() < 0.1,
                "recent_breakout_down_opening_range": rng.random() < 0.1,
            })
    df = pd.DataFrame(rows)
    df = df.sort_values(["symbol", "bar_start_utc"]).reset_index(drop=True)
    for w in (5, 15):
        df[f"cs_rank_{w}m"] = df.groupby("bar_start_utc")[f"return_{w}m"].rank(pct=True)
    df["relative_strength_vs_universe"] = df["return_1m"] - df.groupby("bar_start_utc")["return_1m"].transform("mean")
    for h in HORIZONS:
        up = rng.random(len(df)) < 0.4
        down = ~up & (rng.random(len(df)) < 0.6)
        df[f"direction_label_{h}m"] = np.where(up, "UP", np.where(down, "DOWN", "AMBIGUOUS"))
        df[f"gross_return_{h}m"] = rng.normal(scale=0.005, size=len(df))
    return df


class TestBuildCandidates:
    def test_produces_exactly_40_frozen_candidates(self):
        df = _synthetic_dense_dataset()
        candidates = build_candidates(df)
        assert len(candidates) == 40

    def test_every_candidate_is_a_boolean_series_no_nulls(self):
        df = _synthetic_dense_dataset()
        candidates = build_candidates(df)
        for name, mask in candidates.items():
            assert mask.dtype == bool, f"{name} is not bool dtype"
            assert not mask.isna().any(), f"{name} has NA values (must be filled to False)"


class TestTestBudget:
    def test_is_deterministic(self):
        df = _synthetic_dense_dataset()
        candidates = build_candidates(df)
        assert compute_test_budget(candidates) == compute_test_budget(candidates)

    def test_row_count_produced_by_family_a_matches_the_budgeted_count(self):
        df = _synthetic_dense_dataset()
        candidates = build_candidates(df)
        budget = compute_test_budget(candidates)
        alpha = 0.05 / sum(budget.values())
        family_a = run_family_a(df, candidates, alpha)
        assert len(family_a) == budget["A_standalone"]

    def test_row_count_produced_by_family_b_matches_the_budgeted_count(self):
        df = _synthetic_dense_dataset()
        candidates = build_candidates(df)
        budget = compute_test_budget(candidates)
        alpha = 0.05 / sum(budget.values())
        family_b = run_family_b(df, candidates, alpha)
        assert len(family_b) == budget["B_incremental_value"]

    def test_matches_the_exact_frozen_totals_from_the_design_document(self):
        df = _synthetic_dense_dataset()
        candidates = build_candidates(df)
        budget = compute_test_budget(candidates)
        assert budget["A_standalone"] == 40 * len(HORIZONS) == 160
        assert budget["B_incremental_value"] == 40 * 2 * 2 == 160
        assert sum(budget.values()) == 320


class TestPickSurvivors:
    def test_requires_significance_in_family_a_and_all_four_family_b_cells(self):
        family_a = [{"quantity": "top_10pct_rank5m", "significant_after_correction": True}]
        family_b_all_sig = [
            {"quantity": "top_10pct_rank5m", "control": "return_5m", "controlled_group": "below", "significant_after_correction": True},
            {"quantity": "top_10pct_rank5m", "control": "return_5m", "controlled_group": "above", "significant_after_correction": True},
            {"quantity": "top_10pct_rank5m", "control": "relative_strength_vs_universe", "controlled_group": "below", "significant_after_correction": True},
            {"quantity": "top_10pct_rank5m", "control": "relative_strength_vs_universe", "controlled_group": "above", "significant_after_correction": True},
        ]
        assert len(pick_survivors(family_a, family_b_all_sig)) == 1

    def test_excludes_when_only_three_of_four_cells_significant(self):
        family_a = [{"quantity": "top_10pct_rank5m", "significant_after_correction": True}]
        family_b = [
            {"quantity": "top_10pct_rank5m", "control": "return_5m", "controlled_group": "below", "significant_after_correction": True},
            {"quantity": "top_10pct_rank5m", "control": "return_5m", "controlled_group": "above", "significant_after_correction": True},
            {"quantity": "top_10pct_rank5m", "control": "relative_strength_vs_universe", "controlled_group": "below", "significant_after_correction": True},
            {"quantity": "top_10pct_rank5m", "control": "relative_strength_vs_universe", "controlled_group": "above", "significant_after_correction": False},
        ]
        assert pick_survivors(family_a, family_b) == []


class TestFinalVerdict:
    def test_verdict_d_with_no_survivors(self):
        v = final_verdict([], {}, {}, {})
        assert v["verdict"] == "D"
        assert v["n_survivors"] == 0


class TestEconomicEvaluationSignFlip:

    def test_a_bucket_with_below_baseline_up_rate_is_evaluated_short_not_long(self):
        n = 200
        df = pd.DataFrame({
            "symbol": ["S0"] * n, "bar_start_utc": pd.date_range("2026-01-01", periods=n, freq="5min", tz=UTC),
            "trading_date": pd.date_range("2026-01-01", periods=n, freq="5min", tz=UTC).date,
        })
        in_bucket = pd.Series([True] * 100 + [False] * 100)
        df["direction_label_5m"] = ["DOWN"] * 80 + ["UP"] * 20 + ["UP"] * 90 + ["DOWN"] * 10
        df["gross_return_5m"] = [-0.01] * 100 + [0.001] * 100
        for h in (15, 30, 60):
            df[f"direction_label_{h}m"] = None
            df[f"gross_return_{h}m"] = np.nan

        candidates = {"fake_leader_bucket": in_bucket}
        survivors = [{"quantity": "fake_leader_bucket"}]
        econ = run_economic_evaluation(df, candidates, survivors)

        detail = econ["fake_leader_bucket"]["5m"]
        assert detail["implied_side"] == "SHORT"
        assert detail["gross_mean"] == pytest.approx(0.01)

    def test_a_bucket_with_above_baseline_up_rate_is_evaluated_long(self):
        n = 200
        df = pd.DataFrame({
            "symbol": ["S0"] * n, "bar_start_utc": pd.date_range("2026-01-01", periods=n, freq="5min", tz=UTC),
            "trading_date": pd.date_range("2026-01-01", periods=n, freq="5min", tz=UTC).date,
        })
        in_bucket = pd.Series([True] * 100 + [False] * 100)
        df["direction_label_5m"] = ["UP"] * 80 + ["DOWN"] * 20 + ["DOWN"] * 90 + ["UP"] * 10
        df["gross_return_5m"] = [0.01] * 100 + [-0.001] * 100
        for h in (15, 30, 60):
            df[f"direction_label_{h}m"] = None
            df[f"gross_return_{h}m"] = np.nan

        candidates = {"fake_laggard_bucket": in_bucket}
        survivors = [{"quantity": "fake_laggard_bucket"}]
        econ = run_economic_evaluation(df, candidates, survivors)

        detail = econ["fake_laggard_bucket"]["5m"]
        assert detail["implied_side"] == "LONG"
        assert detail["gross_mean"] == pytest.approx(0.01)
