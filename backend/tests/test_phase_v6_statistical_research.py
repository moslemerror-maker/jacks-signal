from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from scripts.run_phase_v6_statistical_research import (
    ALL_HORIZONS,
    DISCOVERY_HORIZONS,
    RACE_CHECKPOINTS,
    STOP_VOL_MULT,
    TARGET_VOL_MULT,
    TOP_N_VALUES,
    _fav_adv_cols,
    final_verdicts,
    pick_survivors,
    race_target_before_stop,
    run_state_discovery,
    run_top_n_discovery,
    state_masks,
)
from scripts.run_phase_v6_statistical_research import test_budget_discovery as compute_test_budget

UTC = dt.timezone.utc


def _synthetic_dataset(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_symbols = 10
    data: dict[str, np.ndarray] = {
        "symbol": np.array([f"S{i % n_symbols}" for i in range(n)]),
        "security_id": np.array([i % n_symbols for i in range(n)]),
        "trading_date": pd.date_range("2026-01-01", periods=n, freq="5min").date,
        "bar_start_utc": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"),
        "volatility_expansion_ratio": rng.random(n) * 2,
        "relative_volume_20": rng.random(n) * 4,
        "nearest_level_proximity": rng.random(n) * 0.02,
        "wick_asymmetry": rng.normal(size=n),
        "return_5m": rng.normal(scale=0.004, size=n),
        "vwap_reclaim": rng.random(n) < 0.03, "vwap_rejection": rng.random(n) < 0.03,
        "failed_breakout_up": rng.random(n) < 0.02, "failed_breakout_down": rng.random(n) < 0.02,
        "breakout_up_meaningful": rng.random(n) < 0.03, "breakout_down_meaningful": rng.random(n) < 0.03,
        "candle_body_ratio": rng.random(n),
        "consecutive_same_direction_count": rng.integers(0, 10, n), "candle_direction": rng.choice([-1, 0, 1], size=n),
        "compression_then_expansion_flag": rng.random(n) < 0.05,
        "distance_from_session_open": rng.normal(scale=0.01, size=n), "minutes_since_open": rng.integers(0, 375, n),
        "time_of_day_bucket": rng.choice(["opening", "mid", "late"], size=n),
        "realized_volatility_20": rng.random(n) * 0.015 + 0.002,
        "rolling_liquidity_group": rng.choice(["mid_liquidity", "high_liquidity"], size=n),
    }
    for h in RACE_CHECKPOINTS:
        data[f"mfe_o1_h{h}"] = np.abs(rng.normal(scale=0.004 * np.sqrt(h), size=n))
        data[f"mae_o1_h{h}"] = np.abs(rng.normal(scale=0.004 * np.sqrt(h), size=n))
        data[f"gross_return_o1_h{h}"] = rng.normal(scale=0.004 * np.sqrt(h), size=n)
    return pd.DataFrame(data)


class TestFavAdvCols:
    def test_long_favorable_is_mfe(self):
        fav, adv = _fav_adv_cols(15, is_long=True)
        assert fav == "mfe_o1_h15" and adv == "mae_o1_h15"

    def test_short_favorable_is_mae(self):
        fav, adv = _fav_adv_cols(15, is_long=False)
        assert fav == "mae_o1_h15" and adv == "mfe_o1_h15"


class TestRaceTargetBeforeStop:
    def _frame(self, **overrides):
        base = {f"mfe_o1_h{h}": 0.0 for h in RACE_CHECKPOINTS}
        base.update({f"mae_o1_h{h}": 0.0 for h in RACE_CHECKPOINTS})
        base["realized_volatility_20"] = 0.01
        base.update(overrides)
        return pd.DataFrame([base])

    def test_target_wins_when_mfe_crosses_first_long(self):
        df = self._frame(**{"mfe_o1_h5": TARGET_VOL_MULT * 0.01 + 0.001})
        out = race_target_before_stop(df, is_long=True)
        assert out.iloc[0] == 1.0

    def test_stop_wins_when_mae_crosses_first_long(self):
        df = self._frame(**{"mae_o1_h5": STOP_VOL_MULT * 0.01 + 0.001})
        out = race_target_before_stop(df, is_long=True)
        assert out.iloc[0] == 0.0

    def test_tie_at_same_checkpoint_resolved_as_stop_wins(self):
        df = self._frame(**{"mfe_o1_h5": TARGET_VOL_MULT * 0.01 + 0.001, "mae_o1_h5": STOP_VOL_MULT * 0.01 + 0.001})
        out = race_target_before_stop(df, is_long=True)
        assert out.iloc[0] == 0.0

    def test_neither_hit_is_a_loss(self):
        df = self._frame()
        out = race_target_before_stop(df, is_long=True)
        assert out.iloc[0] == 0.0

    def test_short_side_swaps_favorable_adverse(self):
        df = self._frame(**{"mae_o1_h5": TARGET_VOL_MULT * 0.01 + 0.001})
        out = race_target_before_stop(df, is_long=False)
        assert out.iloc[0] == 1.0


class TestStateMasks:
    def test_sixteen_state_side_combinations(self):
        df = _synthetic_dataset()
        states = state_masks(df)
        assert len(states) == 16

    def test_every_state_has_is_long_flag(self):
        df = _synthetic_dataset()
        states = state_masks(df)
        for name, spec in states.items():
            assert isinstance(spec["is_long"], bool), name

    def test_up_down_pairs_are_mutually_exclusive_for_return_conditioned_states(self):
        df = _synthetic_dataset()
        states = state_masks(df)
        up = states["S1_compression_volume_up"]["mask"].fillna(False)
        down = states["S1_compression_volume_down"]["mask"].fillna(False)
        assert not (up & down).any()


class TestTestBudget:
    def test_matches_frozen_totals(self):
        budget = compute_test_budget()
        assert budget["state_asymmetry_and_tbs"] == 16 * len(DISCOVERY_HORIZONS) * 2 == 64
        assert budget["top_n_ranking"] == len(TOP_N_VALUES) * len(DISCOVERY_HORIZONS) == 6
        assert sum(budget.values()) == 70

    def test_row_counts_match_declared_budget_on_synthetic_data(self):
        df = _synthetic_dataset()
        alpha = 0.05 / sum(compute_test_budget().values())
        assert len(run_state_discovery(df, alpha)) == compute_test_budget()["state_asymmetry_and_tbs"]
        assert len(run_top_n_discovery(df, alpha)) == compute_test_budget()["top_n_ranking"]


class TestPickSurvivorsDirectionAware:
    def test_excludes_significant_but_realized_below_baseline(self):
        results = [{"family": "state_asymmetry", "state": "x", "horizon_min": 15, "significant": True, "lift": 1.2, "realized_mean": 0.0001, "baseline_mean": 0.0002}]
        assert pick_survivors(results) == []

    def test_includes_significant_with_realized_above_baseline(self):
        results = [{"family": "state_asymmetry", "state": "x", "horizon_min": 15, "significant": True, "lift": 1.2, "realized_mean": 0.0002, "baseline_mean": 0.0001}]
        assert len(pick_survivors(results)) == 1

    def test_excludes_top_n_ranking_family(self):
        results = [{"family": "top_n_ranking", "n": 10, "horizon_min": 15, "significant": True, "lift": 2.0, "realized_mean": 0.0002, "baseline_mean": 0.0001}]
        assert pick_survivors(results) == []

    def test_regression_negative_baseline_with_misleadingly_large_lift_is_excluded(self):
        results = [{"family": "state_asymmetry", "state": "short_state", "horizon_min": 15, "significant": True,
                    "lift": 18.1, "realized_mean": -0.000719, "baseline_mean": -0.0000397}]
        assert pick_survivors(results) == []


class TestFinalVerdicts:
    def test_none_survived_with_empty_input(self):
        v = final_verdicts([], [], [], [])
        assert v["verdict"] == "NONE_SURVIVED"
