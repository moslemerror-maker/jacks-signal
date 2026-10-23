from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_phase_t_statistical_research import (
    BUCKETS,
    CONTEXT_COLUMNS,
    ENTRY_OFFSETS,
    HORIZONS,
    IS_LONG_BUCKET,
    STOP_CANDIDATES,
    TARGET_CANDIDATES,
    bucket_masks,
    final_verdict,
    pick_entry_timing_survivors,
    pick_risk_exit_survivors,
    run_context_association,
    run_entry_timing_horizon,
    run_mfe_vs_mae,
    run_risk_exit_grid,
    simulate_stop_target,
)
from scripts.run_phase_t_statistical_research import test_budget as compute_test_budget


def _synthetic_dataset(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {
        "symbol": np.array([f"S{i % 15}" for i in range(n)]),
        "security_id": np.array([i % 15 for i in range(n)]),
        "trading_date": pd.date_range("2026-01-01", periods=n, freq="5min").date,
        "realized_volatility_20": rng.random(n) * 0.02 + 0.001,
        "relative_volume_20": rng.random(n) * 3,
        "distance_from_ema20_normalized": rng.normal(size=n),
        "ema_20_slope_3": rng.normal(size=n),
        "distance_from_vwap": rng.normal(scale=0.003, size=n),
        "consecutive_same_direction_count": rng.integers(0, 10, n),
        "liquidity_group": rng.choice(["low_liquidity", "mid_liquidity", "high_liquidity"], size=n),
    }
    for b in BUCKETS:
        data[b] = rng.random(n) < 0.1
    for k in ENTRY_OFFSETS:
        for h in HORIZONS:
            data[f"gross_return_o{k}_h{h}"] = rng.normal(scale=0.004, size=n)
            data[f"mfe_o{k}_h{h}"] = np.abs(rng.normal(scale=0.004, size=n))
            data[f"mae_o{k}_h{h}"] = np.abs(rng.normal(scale=0.004, size=n))
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
        assert budget["entry_timing_horizon"] == 4 * 5 * 8 == 160
        assert budget["mfe_vs_mae"] == 4 * 8 == 32
        assert budget["risk_exit_grid"] == 4 * 3 * 3 == 36
        assert budget["context_association"] == 4 * 2 * 6 * 2 == 96
        assert sum(budget.values()) == 324

    def test_row_counts_match_declared_budget_on_synthetic_data(self):
        df = _synthetic_dataset()
        masks = bucket_masks(df)
        alpha = 0.05 / sum(compute_test_budget().values())
        assert len(run_entry_timing_horizon(df, masks, alpha)) == compute_test_budget()["entry_timing_horizon"]
        assert len(run_mfe_vs_mae(df, masks, alpha)) == compute_test_budget()["mfe_vs_mae"]
        assert len(run_risk_exit_grid(df, masks, alpha)) == compute_test_budget()["risk_exit_grid"]
        assert len(run_context_association(df, masks, alpha)) == compute_test_budget()["context_association"]


class TestSignAdjustment:
    def test_short_bucket_return_is_sign_flipped_before_testing(self):
        n = 200
        df = pd.DataFrame({
            "symbol": [f"S{i % 5}" for i in range(n)], "security_id": [i % 5 for i in range(n)],
            "trading_date": pd.date_range("2026-01-01", periods=n, freq="5min").date,
            "top_1pct_rank15m": True, "top_2pct_rank15m": True,
            "bottom_1pct_rank15m": False, "bottom_2pct_rank15m": False,
        })
        for k in ENTRY_OFFSETS:
            for h in HORIZONS:
                df[f"gross_return_o{k}_h{h}"] = -0.01
        masks = bucket_masks(df)
        results = run_entry_timing_horizon(df, masks, alpha=0.05)
        top_cells = [r for r in results if r["bucket"] == "top_1pct_rank15m"]
        assert all(r["mean"] == pytest.approx(0.01) for r in top_cells)


class TestSimulateStopTarget:
    def _sub(self, n=5):
        d = {"realized_volatility_20": np.full(n, 0.01)}
        for h in HORIZONS:
            d[f"mfe_o1_h{h}"] = np.zeros(n)
            d[f"mae_o1_h{h}"] = np.zeros(n)
        d["gross_return_o1_h60"] = np.full(n, 0.005)
        return pd.DataFrame(d)

    def test_no_stop_no_target_falls_back_to_raw_60m_return(self):
        sub = self._sub()
        out = simulate_stop_target(sub, is_long=True, stop_name="none", target_name="none")
        assert np.allclose(out.to_numpy(), 0.005)

    def test_target_hit_before_any_stop_configured_returns_target_level(self):
        sub = self._sub()
        sub.loc[0, "mfe_o1_h5"] = 0.03
        out = simulate_stop_target(sub, is_long=True, stop_name="none", target_name="fixed_2pct")
        assert out.iloc[0] == pytest.approx(0.02)

    def test_stop_hit_before_any_target_configured_returns_negative_stop_level(self):
        sub = self._sub()
        sub.loc[0, "mae_o1_h5"] = 0.03
        out = simulate_stop_target(sub, is_long=True, stop_name="fixed_1pct", target_name="none")
        assert out.iloc[0] == pytest.approx(-0.01)

    def test_same_checkpoint_tie_is_conservatively_resolved_as_stop_first(self):
        sub = self._sub()
        sub.loc[0, "mae_o1_h5"] = 0.03
        sub.loc[0, "mfe_o1_h5"] = 0.03
        out = simulate_stop_target(sub, is_long=True, stop_name="fixed_1pct", target_name="fixed_2pct")
        assert out.iloc[0] == pytest.approx(-0.01)

    def test_earlier_checkpoint_wins_even_if_smaller_index(self):
        sub = self._sub()
        sub.loc[0, "mfe_o1_h2"] = 0.03
        sub.loc[0, "mae_o1_h5"] = 0.03
        out = simulate_stop_target(sub, is_long=True, stop_name="fixed_1pct", target_name="fixed_2pct")
        assert out.iloc[0] == pytest.approx(0.02)

    def test_short_bucket_uses_swapped_favorable_adverse_columns(self):
        sub = self._sub()
        sub.loc[0, "mae_o1_h5"] = 0.03
        sub.loc[0, "mfe_o1_h5"] = 0.03
        out = simulate_stop_target(sub, is_long=False, stop_name="fixed_1pct", target_name="none")
        assert out.iloc[0] == pytest.approx(-0.01)

    def test_vol_scaled_threshold_is_row_specific(self):
        sub = self._sub(n=2)
        sub.loc[:, "realized_volatility_20"] = [0.01, 0.05]
        sub.loc[:, "mfe_o1_h5"] = [0.03, 0.03]
        out = simulate_stop_target(sub, is_long=True, stop_name="none", target_name="vol_2x")
        assert out.iloc[0] == pytest.approx(0.02)
        assert out.iloc[1] == pytest.approx(0.005)


class TestPickSurvivors:
    def test_requires_significance_and_positive_mean(self):
        entry = [
            {"bucket": "b", "entry_offset_min": 1, "horizon_min": 5, "significant_after_correction": True, "mean": 0.001},
            {"bucket": "b", "entry_offset_min": 1, "horizon_min": 10, "significant_after_correction": True, "mean": -0.001},
            {"bucket": "b", "entry_offset_min": 1, "horizon_min": 15, "significant_after_correction": False, "mean": 0.001},
        ]
        assert len(pick_entry_timing_survivors(entry)) == 1

    def test_risk_exit_requires_net_of_60bps_positive(self):
        risk = [
            {"bucket": "b", "stop": "s", "target": "t", "significant_after_correction": True, "mean": 0.01},
            {"bucket": "b", "stop": "s", "target": "t", "significant_after_correction": True, "mean": 0.003},
        ]
        survivors = pick_risk_exit_survivors(risk)
        assert len(survivors) == 1
        assert survivors[0]["mean"] == 0.01


class TestFinalVerdict:
    def test_verdict_d_with_no_survivors(self):
        v = final_verdict([], [], [], {}, {})
        assert v["verdict"] == "D"
        assert v["n_survivors"] == 0
