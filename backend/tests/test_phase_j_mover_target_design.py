from __future__ import annotations

import numpy as np

import pandas as pd
import pytest

from scripts.run_phase_j_mover_target_design import (
    MIN_PRIOR_OBSERVATIONS_FOR_T3,
    SELECTION_RATE_RANGE,
    _base_rates_by_group,
    _dispersion_score,
    _expanding_rank_fraction,
    compute_t1,
    compute_t2,
    compute_t3,
    compute_t3_expanding_percentile,
    select_frozen_target,
)


def _df():
    return pd.DataFrame({
        "symbol": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "tier": ["T1", "T1", "T1", "T1", "T2", "T2", "T2", "T2"],
        "label_normalized_move_5m": [0.2, 0.6, 1.2, 2.5, 0.1, 0.4, 0.9, 3.0],
        "mover_future_abs_return_5m": [0.001, 0.004, 0.012, 0.03, 0.0005, 0.002, 0.008, 0.05],
    })


class TestComputeT1T2T3:
    def test_t1_flags_by_absolute_normalized_move(self):
        flag = compute_t1(_df(), 5, 1.0)
        assert flag.tolist() == [False, False, True, True, False, False, False, True]

    def test_t2_flags_by_raw_fraction_threshold(self):
        flag = compute_t2(_df(), 5, 0.01)
        assert flag.tolist() == [False, False, True, True, False, False, False, True]

    def test_t3_is_a_per_symbol_percentile_rank(self):
        flag = compute_t3(_df(), 5, 0.75)
        a_flags = flag[_df()["symbol"] == "A"].tolist()
        assert a_flags == [False, False, True, True]
        b_flags = flag[_df()["symbol"] == "B"].tolist()
        assert b_flags == [False, False, True, True]

    def test_t3_flags_only_the_single_highest_at_a_stricter_threshold(self):
        flag = compute_t3(_df(), 5, 0.9)
        a_flags = flag[_df()["symbol"] == "A"].tolist()
        assert a_flags == [False, False, False, True]


class TestT3IsNotPointInTimeSafe:

    def _long_symbol_history(self) -> pd.DataFrame:
        return pd.DataFrame({
            "symbol": ["A"] * 10,
            "mover_future_abs_return_5m": [0.01, 0.012, 0.014, 0.016, 0.018, 0.5, 0.6, 0.7, 0.8, 0.9],
        })

    def test_an_early_rows_flag_changes_when_later_rows_are_removed(self):
        full = self._long_symbol_history()
        truncated = full.iloc[:5].copy()

        flag_full = compute_t3(full, horizon=5, percentile=0.15).iloc[0]
        flag_truncated = compute_t3(truncated, horizon=5, percentile=0.15).iloc[0]

        assert flag_full != flag_truncated, (
            "if this assertion ever fails, compute_t3 has changed - re-run the Phase K PIT audit "
            "before assuming the leak documented here has been fixed; do not delete this test to "
            "make it pass"
        )

    def test_the_percentile_rank_itself_is_computed_over_the_whole_column_not_a_prefix(self):
        full = self._long_symbol_history()
        rank_col = full.groupby("symbol")["mover_future_abs_return_5m"].rank(pct=True)
        assert rank_col.iloc[0] == pytest.approx(0.10)


class TestExpandingRankFraction:

    def test_strictly_increasing_series_gives_rank_1_once_enough_prior_observations_exist(self):
        values = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = _expanding_rank_fraction(values, min_prior_observations=3)
        assert np.isnan(result[0]) and np.isnan(result[1]) and np.isnan(result[2])
        assert result[3] == pytest.approx(1.0)
        assert result[9] == pytest.approx(1.0)

    def test_a_small_final_value_ranks_near_zero(self):
        values = np.array([5.0, 6, 7, 8, 9, 10, 11, 12, 13, 1.0])
        result = _expanding_rank_fraction(values, min_prior_observations=3)
        assert result[9] == pytest.approx(0.0)

    def test_a_middling_value_ranks_near_the_middle(self):
        values = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 5.5])
        result = _expanding_rank_fraction(values, min_prior_observations=3)
        assert result[9] == pytest.approx(5 / 9)

    def test_nan_input_value_propagates_to_nan_output(self):
        values = np.array([1.0, 2, 3, 4, np.nan, 6])
        result = _expanding_rank_fraction(values, min_prior_observations=2)
        assert np.isnan(result[4])

    def test_nan_values_do_not_count_toward_the_prior_observation_requirement(self):
        values = np.array([1.0, 2, 3, 4, np.nan, 6])
        result = _expanding_rank_fraction(values, min_prior_observations=4)
        assert not np.isnan(result[5])
        result_stricter = _expanding_rank_fraction(values, min_prior_observations=5)
        assert np.isnan(result_stricter[5])

    def test_only_strictly_prior_values_are_ever_consulted(self):
        values = np.array([3.0, 1, 4, 1, 5, 9, 2, 6])
        full = _expanding_rank_fraction(values, min_prior_observations=2)
        truncated = _expanding_rank_fraction(values[:5], min_prior_observations=2)
        for i in range(5):
            if np.isnan(full[i]):
                assert np.isnan(truncated[i])
            else:
                assert full[i] == pytest.approx(truncated[i])


class TestT3ExpandingPercentileIsPointInTimeSafe:

    def _df(self, n: int = 200, seed: int = 3) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "symbol": ["A"] * n,
            "bar_start_utc": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC"),
            "mover_future_abs_return_5m": rng.random(n),
        })

    def test_an_early_rows_label_does_not_change_when_later_rows_are_added(self):
        full = self._df(n=200)
        truncated = full.iloc[:100].copy()
        flags_full = compute_t3_expanding_percentile(full, horizon=5, percentile=0.9, min_prior_observations=10)
        flags_truncated = compute_t3_expanding_percentile(truncated, horizon=5, percentile=0.9, min_prior_observations=10)
        for i in range(100):
            a, b = flags_full.iloc[i], flags_truncated.iloc[i]
            if pd.isna(a):
                assert pd.isna(b)
            else:
                assert a == b, f"row {i} differs: {a!r} vs {b!r} - T3 expanding percentile is no longer PIT-safe"

    def test_multiple_truncation_points_all_agree_with_the_full_computation(self):
        full = self._df(n=150)
        flags_full = compute_t3_expanding_percentile(full, horizon=5, percentile=0.9, min_prior_observations=10)
        for cutoff in (60, 90, 120):
            truncated = full.iloc[:cutoff].copy()
            flags_truncated = compute_t3_expanding_percentile(truncated, horizon=5, percentile=0.9, min_prior_observations=10)
            for i in range(cutoff):
                a, b = flags_full.iloc[i], flags_truncated.iloc[i]
                assert (pd.isna(a) and pd.isna(b)) or a == b

    def test_rows_below_the_minimum_history_requirement_are_unavailable_not_fabricated(self):
        df = self._df(n=30)
        flags = compute_t3_expanding_percentile(df, horizon=5, percentile=0.9, min_prior_observations=20)
        assert flags.iloc[:20].isna().all()
        assert flags.iloc[20:].notna().all()

    def test_default_minimum_matches_the_documented_constant(self):
        assert MIN_PRIOR_OBSERVATIONS_FOR_T3 == 50

    def test_a_sparse_symbol_with_too_little_total_history_is_entirely_unavailable(self):
        sparse_df = self._df(n=30, seed=9)
        flags = compute_t3_expanding_percentile(sparse_df, horizon=5, percentile=0.9)
        assert flags.isna().all()

    def test_chronological_split_boundary_is_irrelevant_to_the_computation(self):
        full = self._df(n=200)
        train_only = full.iloc[:120].copy()
        flags_full = compute_t3_expanding_percentile(full, horizon=5, percentile=0.9, min_prior_observations=10)
        flags_train_only = compute_t3_expanding_percentile(train_only, horizon=5, percentile=0.9, min_prior_observations=10)
        for i in range(120):
            a, b = flags_full.iloc[i], flags_train_only.iloc[i]
            assert (pd.isna(a) and pd.isna(b)) or a == b

    def test_multiple_symbols_never_influence_each_other(self):
        df = pd.concat([self._df(n=80, seed=1).assign(symbol="A"), self._df(n=80, seed=2).assign(symbol="B")], ignore_index=True)
        flags_both = compute_t3_expanding_percentile(df, horizon=5, percentile=0.9, min_prior_observations=10)
        flags_a_alone = compute_t3_expanding_percentile(df[df["symbol"] == "A"].reset_index(drop=True), horizon=5, percentile=0.9, min_prior_observations=10)
        a_rows = df[df["symbol"] == "A"].index
        for pos, idx in enumerate(a_rows):
            a, b = flags_both.loc[idx], flags_a_alone.iloc[pos]
            assert (pd.isna(a) and pd.isna(b)) or a == b


class TestBaseRatesByGroup:
    def test_overall_rate_matches_the_flags_mean(self):
        flag = pd.Series([True, False, True, False])
        df = pd.DataFrame({"tier": ["A", "A", "B", "B"], "price_group": ["x", "x", "y", "y"], "liquidity_group": ["p"] * 4, "volatility_group": ["q"] * 4})
        result = _base_rates_by_group(flag, df)
        assert result["overall"] == pytest.approx(0.5)
        assert result["n_valid"] == 4

    def test_breaks_down_by_every_configured_dimension(self):
        flag = pd.Series([True, True, False, False])
        df = pd.DataFrame({"tier": ["A", "A", "B", "B"], "price_group": ["x"] * 4, "liquidity_group": ["p"] * 4, "volatility_group": ["q"] * 4})
        result = _base_rates_by_group(flag, df)
        assert result["by_group"]["tier"]["A"] == pytest.approx(1.0)
        assert result["by_group"]["tier"]["B"] == pytest.approx(0.0)


class TestDispersionScore:
    def test_zero_dispersion_when_every_group_matches_overall(self):
        rates = {"overall": 0.1, "by_group": {"tier": {"A": 0.1, "B": 0.1}}}
        assert _dispersion_score(rates) == pytest.approx(0.0)

    def test_positive_dispersion_when_groups_differ(self):
        rates = {"overall": 0.1, "by_group": {"tier": {"A": 0.05, "B": 0.2}}}
        assert _dispersion_score(rates) == pytest.approx((0.2 - 0.05) / 0.1)

    def test_none_when_overall_is_zero_or_none(self):
        assert _dispersion_score({"overall": 0.0, "by_group": {}}) is None
        assert _dispersion_score({"overall": None, "by_group": {}}) is None


class TestSelectFrozenTarget:
    def _candidate(self, definition, threshold, horizon, overall, dispersion):
        return {"definition": definition, "horizon_minutes": horizon, "threshold": threshold, "overall": overall, "n_valid": 1000, "by_group": {}, "dispersion_score": dispersion}

    def test_picks_the_lowest_average_dispersion_among_in_range_candidates(self):
        candidates = []
        for h in (5, 15, 30, 60):
            candidates.append(self._candidate("T1", 1.0, h, 0.10, 0.5))
            candidates.append(self._candidate("T3", 0.9, h, 0.10, 0.05))
        result = select_frozen_target(candidates)
        assert result["status"] == "FROZEN"
        assert result["definition"] == "T3"

    def test_excludes_candidates_outside_the_rate_range_at_any_horizon(self):
        candidates = []
        for h in (5, 15, 30, 60):
            rate = 0.99 if h == 60 else 0.10
            candidates.append(self._candidate("T1", 1.0, h, rate, 0.05))
            candidates.append(self._candidate("T3", 0.9, h, 0.15, 0.3))
        result = select_frozen_target(candidates)
        assert result["status"] == "FROZEN"
        assert result["definition"] == "T3"

    def test_no_eligible_candidate_is_reported_honestly_not_forced(self):
        candidates = [self._candidate("T1", 1.0, h, 0.99, 0.05) for h in (5, 15, 30, 60)]
        result = select_frozen_target(candidates)
        assert result["status"] == "NO_ELIGIBLE_CANDIDATE"

    def test_selection_rate_range_is_the_documented_2_to_30_percent(self):
        assert SELECTION_RATE_RANGE == (0.02, 0.30)
