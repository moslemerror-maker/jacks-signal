from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_phase_m_statistical_research import (
    CATEGORICAL_CANDIDATES,
    CONTINUOUS_CANDIDATES,
    HORIZONS,
    NIFTY_50_SYMBOLS,
    NIFTY_BANK_SYMBOLS,
    POPULATIONS,
    add_candidate_1,
    add_candidate_3,
    add_direction_flags,
    classify_and_dedupe_index_rows,
    compute_abnormal_concentration,
    compute_breadth_acceleration,
    compute_cross_sectional_dispersion,
    compute_leader_impulse,
    final_verdict,
    pick_survivors,
    run_family_a,
    run_family_b,
)
from scripts.run_phase_m_statistical_research import test_budget as compute_test_budget

UTC = "UTC"


class TestCandidate1:
    def test_is_a_direct_copy_of_universe_breadth(self):
        df = pd.DataFrame({"universe_breadth": [0.1, 0.5, 0.9]})
        out = add_candidate_1(df)
        assert list(out["c1_universe_breadth"]) == [0.1, 0.5, 0.9]


def _c2_frame(n_times: int, symbols: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    times = pd.date_range("2026-01-05 03:45", periods=n_times, freq="min", tz=UTC)
    rows = []
    for t_idx, t in enumerate(times):
        breadth = rng.random()
        for sym in symbols:
            rows.append({
                "bar_start_utc": t, "symbol": sym,
                "universe_breadth": breadth,
                "return_1m": rng.normal(),
                "realized_volatility_20": abs(rng.normal()) + 0.01,
            })
    return pd.DataFrame(rows)


class TestCandidate2FutureAppendInvariance:

    SYMBOLS = ["A", "B", "C", "D", "E"]

    def test_breadth_acceleration_unaffected_by_future_rows(self):
        df = _c2_frame(20, self.SYMBOLS)
        t_cut = df["bar_start_utc"].unique()[9]

        full = compute_breadth_acceleration(df)
        truncated_df = df[df["bar_start_utc"] <= t_cut].reset_index(drop=True)
        truncated = compute_breadth_acceleration(truncated_df)

        full_at_t = full[df["bar_start_utc"] == t_cut].iloc[0]
        truncated_at_t = truncated[truncated_df["bar_start_utc"] == t_cut].iloc[0]
        assert full_at_t == pytest.approx(truncated_at_t)

    def test_cross_sectional_dispersion_unaffected_by_future_rows(self):
        df = _c2_frame(20, self.SYMBOLS)
        t_cut = df["bar_start_utc"].unique()[9]

        full = compute_cross_sectional_dispersion(df)
        truncated_df = df[df["bar_start_utc"] <= t_cut].reset_index(drop=True)
        truncated = compute_cross_sectional_dispersion(truncated_df)

        full_at_t = full[df["bar_start_utc"] == t_cut].iloc[0]
        truncated_at_t = truncated[truncated_df["bar_start_utc"] == t_cut].iloc[0]
        assert full_at_t == pytest.approx(truncated_at_t)

    def test_abnormal_concentration_unaffected_by_future_rows(self):
        df = _c2_frame(20, self.SYMBOLS)
        t_cut = df["bar_start_utc"].unique()[9]

        full = compute_abnormal_concentration(df)
        truncated_df = df[df["bar_start_utc"] <= t_cut].reset_index(drop=True)
        truncated = compute_abnormal_concentration(truncated_df)

        full_at_t = full[df["bar_start_utc"] == t_cut].iloc[0]
        truncated_at_t = truncated[truncated_df["bar_start_utc"] == t_cut].iloc[0]
        assert full_at_t == pytest.approx(truncated_at_t)

    def test_appending_a_brand_new_future_symbol_does_not_change_earlier_values(self):
        df = _c2_frame(10, self.SYMBOLS)
        t_mid = df["bar_start_utc"].unique()[5]
        before = compute_cross_sectional_dispersion(df)
        before_at_mid = before[df["bar_start_utc"] == t_mid].iloc[0]

        future_time = df["bar_start_utc"].max() + pd.Timedelta(minutes=1)
        intruder = pd.DataFrame([{"bar_start_utc": future_time, "symbol": "ZZZ_NEW", "return_1m": 999.0, "realized_volatility_20": 0.02}])
        df_with_intruder = pd.concat([df, intruder], ignore_index=True)

        after = compute_cross_sectional_dispersion(df_with_intruder)
        after_at_mid = after[df_with_intruder["bar_start_utc"] == t_mid].iloc[0]
        assert before_at_mid == pytest.approx(after_at_mid)


class TestIndexAliasClassification:
    def test_cnx_nifty_and_nifty_50_collapse_to_the_same_index_name(self):
        idx = pd.DataFrame({
            "trade_date": [pd.Timestamp("2015-11-06"), pd.Timestamp("2015-11-10")],
            "symbol": ["CNX NIFTY", "NIFTY 50"],
            "open": [100.0, 200.0], "high": [101.0, 201.0], "low": [99.0, 199.0], "close": [100.5, 200.5],
        })
        out = classify_and_dedupe_index_rows(idx)
        assert set(out["index_name"]) == {"NIFTY_50"}

    def test_bank_nifty_and_nifty_bank_collapse_to_the_same_index_name(self):
        idx = pd.DataFrame({
            "trade_date": [pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-02")],
            "symbol": ["BANK NIFTY", "NIFTY BANK"],
            "open": [100.0, 200.0], "high": [101.0, 201.0], "low": [99.0, 199.0], "close": [100.5, 200.5],
        })
        out = classify_and_dedupe_index_rows(idx)
        assert set(out["index_name"]) == {"NIFTY_BANK"}

    def test_a_date_carrying_both_alias_rows_for_one_index_is_never_double_counted(self):
        idx = pd.DataFrame({
            "trade_date": [pd.Timestamp("2015-11-09"), pd.Timestamp("2015-11-09")],
            "symbol": ["CNX NIFTY", "NIFTY 50"],
            "open": [100.0, 100.0], "high": [101.0, 101.0], "low": [99.0, 99.0], "close": [100.5, 100.5],
        })
        out = classify_and_dedupe_index_rows(idx)
        assert len(out) == 1

    def test_symbol_constants_contain_exactly_the_expected_aliases(self):
        assert set(NIFTY_50_SYMBOLS) == {"NIFTY 50", "CNX NIFTY"}
        assert set(NIFTY_BANK_SYMBOLS) == {"NIFTY BANK", "BANK NIFTY"}


def _stock_frame(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "trading_date": [pd.Timestamp(d).date() for d in dates],
        "symbol": ["RELIANCE"] * len(dates),
        "bar_start_utc": pd.date_range("2026-01-05 03:45", periods=len(dates), freq="D", tz=UTC),
    })


def _idx_frame(dates_and_closes: list[tuple[str, float]], index_name: str) -> pd.DataFrame:
    return pd.DataFrame({
        "trade_date": [pd.Timestamp(d) for d, _ in dates_and_closes],
        "symbol": ["NIFTY 50" if index_name == "NIFTY_50" else "NIFTY BANK"] * len(dates_and_closes),
        "open": [c for _, c in dates_and_closes], "high": [c * 1.01 for _, c in dates_and_closes],
        "low": [c * 0.99 for _, c in dates_and_closes], "close": [c for _, c in dates_and_closes],
        "index_name": [index_name] * len(dates_and_closes),
    })


class TestCandidate3TodaysRowNeverLeaks:
    def test_todays_own_index_close_never_enters_todays_prev_day_return(self):
        stock = _stock_frame(["2026-01-06", "2026-01-07", "2026-01-08"])
        idx_n50 = _idx_frame([("2026-01-05", 100.0), ("2026-01-06", 101.0), ("2026-01-07", 102.0), ("2026-01-08", 9999.0)], "NIFTY_50")
        idx_nb = _idx_frame([("2026-01-05", 200.0), ("2026-01-06", 201.0), ("2026-01-07", 202.0), ("2026-01-08", 8888.0)], "NIFTY_BANK")
        idx = pd.concat([idx_n50, idx_nb], ignore_index=True)

        out = add_candidate_3(stock, idx)
        row_0108 = out[out["trading_date"] == pd.Timestamp("2026-01-08").date()].iloc[0]

        expected_n50 = 102.0 / 101.0 - 1.0
        assert row_0108["c3_nifty50_prev_day_return"] == pytest.approx(expected_n50)
        assert row_0108["c3_nifty50_prev_day_return"] < 1.0

    def test_dates_after_the_last_available_index_row_are_explicitly_nand_not_stale_backfilled(self):
        stock = _stock_frame(["2026-09-10", "2026-09-11", "2026-09-15"])
        idx_n50 = _idx_frame([("2026-09-08", 100.0), ("2026-09-09", 101.0), ("2026-09-10", 102.0), ("2026-09-11", 103.0)], "NIFTY_50")
        idx_nb = _idx_frame([("2026-09-08", 200.0), ("2026-09-09", 201.0), ("2026-09-10", 202.0), ("2026-09-11", 203.0)], "NIFTY_BANK")
        idx = pd.concat([idx_n50, idx_nb], ignore_index=True)

        out = add_candidate_3(stock, idx)
        row_before_cutoff = out[out["trading_date"] == pd.Timestamp("2026-09-11").date()].iloc[0]
        row_after_cutoff = out[out["trading_date"] == pd.Timestamp("2026-09-15").date()].iloc[0]

        assert row_before_cutoff["c3_nifty50_prev_day_return"] == pytest.approx(102.0 / 101.0 - 1.0)
        assert pd.isna(row_after_cutoff["c3_nifty50_prev_day_return"])
        assert pd.isna(row_after_cutoff["c3_niftybank_prev_day_return"])


class TestAddDirectionFlags:
    def test_positive_return_maps_to_up_and_negative_to_down(self):
        df = pd.DataFrame({"c3_nifty50_prev_day_return": [0.01, -0.01, None], "c3_niftybank_prev_day_return": [0.02, -0.02, None]})
        out = add_direction_flags(df)
        assert list(out["c3_nifty50_direction"])[:2] == ["UP", "DOWN"]
        assert pd.isna(out["c3_nifty50_direction"].iloc[2])
        assert list(out["c3_niftybank_direction"])[:2] == ["UP", "DOWN"]
        assert pd.isna(out["c3_niftybank_direction"].iloc[2])


def _c4_frame(symbol_times: dict[str, list[tuple[str, float]]], group_col_values: dict[str, str]) -> pd.DataFrame:
    rows = []
    for sym, series in symbol_times.items():
        for t, ret in series:
            rows.append({"symbol": sym, "bar_start_utc": pd.Timestamp(t, tz=UTC), "return_5m": ret, "grp": group_col_values[sym]})
    return pd.DataFrame(rows)


class TestCandidate4LeaderImpulseCheck1FutureAppendInvariance:
    def test_appending_future_rows_does_not_change_an_already_computed_impulse(self):
        symbol_times = {
            "A": [("2026-01-05 03:45", 0.01), ("2026-01-05 03:50", 0.02), ("2026-01-05 03:55", 0.03)],
            "B": [("2026-01-05 03:45", 0.02), ("2026-01-05 03:50", 0.03), ("2026-01-05 03:55", -0.01)],
            "C": [("2026-01-05 03:45", 0.03), ("2026-01-05 03:50", -0.01), ("2026-01-05 03:55", 0.05)],
        }
        grp = {"A": "G1", "B": "G1", "C": "G1"}
        df = _c4_frame(symbol_times, grp)
        t_mid = pd.Timestamp("2026-01-05 03:50", tz=UTC)

        full = compute_leader_impulse(df, "grp", min_group_size=2)
        full_a_at_mid = full[(df["symbol"] == "A") & (df["bar_start_utc"] == t_mid)].iloc[0]

        truncated_df = df[df["bar_start_utc"] <= t_mid].reset_index(drop=True)
        truncated = compute_leader_impulse(truncated_df, "grp", min_group_size=2)
        truncated_a_at_mid = truncated[(truncated_df["symbol"] == "A") & (truncated_df["bar_start_utc"] == t_mid)].iloc[0]

        assert full_a_at_mid == pytest.approx(truncated_a_at_mid)

    def test_a_brand_new_future_symbol_appended_to_the_group_does_not_change_earlier_impulses(self):
        symbol_times = {
            "A": [("2026-01-05 03:45", 0.01), ("2026-01-05 03:50", 0.02)],
            "B": [("2026-01-05 03:45", 0.02), ("2026-01-05 03:50", 0.03)],
            "C": [("2026-01-05 03:45", 0.03), ("2026-01-05 03:50", -0.01)],
        }
        grp = {"A": "G1", "B": "G1", "C": "G1"}
        df = _c4_frame(symbol_times, grp)
        t_mid = pd.Timestamp("2026-01-05 03:50", tz=UTC)
        before = compute_leader_impulse(df, "grp", min_group_size=2)
        before_a = before[(df["symbol"] == "A") & (df["bar_start_utc"] == t_mid)].iloc[0]

        intruder = pd.DataFrame([
            {"symbol": "ZZZ_NEW", "bar_start_utc": pd.Timestamp("2026-01-05 03:45", tz=UTC), "return_5m": 500.0, "grp": "G1"},
            {"symbol": "ZZZ_NEW", "bar_start_utc": pd.Timestamp("2026-01-05 03:55", tz=UTC), "return_5m": -500.0, "grp": "G1"},
        ])
        df2 = pd.concat([df, intruder], ignore_index=True)
        after = compute_leader_impulse(df2, "grp", min_group_size=2)
        after_a = after[(df2["symbol"] == "A") & (df2["bar_start_utc"] == t_mid)].iloc[0]
        assert before_a == pytest.approx(after_a)


class TestCandidate4LeaderImpulseCheck2SelfExclusion:
    def test_targets_own_prior_value_is_excluded_from_its_own_leader_mean(self):
        symbol_times = {
            "A": [("2026-01-05 03:45", 0.01), ("2026-01-05 03:50", 0.0)],
            "B": [("2026-01-05 03:45", 0.02), ("2026-01-05 03:50", 0.0)],
            "C": [("2026-01-05 03:45", 0.03), ("2026-01-05 03:50", 0.0)],
        }
        grp = {"A": "G1", "B": "G1", "C": "G1"}
        df = _c4_frame(symbol_times, grp)
        t1 = pd.Timestamp("2026-01-05 03:50", tz=UTC)
        impulse = compute_leader_impulse(df, "grp", min_group_size=2)

        impulse_a = impulse[(df["symbol"] == "A") & (df["bar_start_utc"] == t1)].iloc[0]
        impulse_b = impulse[(df["symbol"] == "B") & (df["bar_start_utc"] == t1)].iloc[0]
        impulse_c = impulse[(df["symbol"] == "C") & (df["bar_start_utc"] == t1)].iloc[0]

        assert impulse_a == pytest.approx((0.02 + 0.03) / 2 - 0.01)
        assert impulse_b == pytest.approx((0.01 + 0.03) / 2 - 0.02)
        assert impulse_c == pytest.approx((0.01 + 0.02) / 2 - 0.03)

        naive_inclusive_a = (0.01 + 0.02 + 0.03) / 3 - 0.01
        assert impulse_a != pytest.approx(naive_inclusive_a)


class TestCandidate4LeaderImpulseCheck3TPrevBoundaryExact:
    def test_a_gap_in_one_symbols_series_still_uses_its_own_immediately_prior_row_not_a_neighbor(self):
        symbol_times = {
            "D": [("2026-01-05 03:45", 0.07), ("2026-01-05 03:55", 0.09)],
            "E": [("2026-01-05 03:45", 0.01), ("2026-01-05 03:50", 0.02), ("2026-01-05 03:55", 0.03)],
            "F": [("2026-01-05 03:45", 0.04), ("2026-01-05 03:50", 0.05), ("2026-01-05 03:55", 0.06)],
        }
        grp = {"D": "G1", "E": "G1", "F": "G1"}
        df = _c4_frame(symbol_times, grp)
        t_355 = pd.Timestamp("2026-01-05 03:55", tz=UTC)
        impulse = compute_leader_impulse(df, "grp", min_group_size=2)

        d_prev = df[(df["symbol"] == "D") & (df["bar_start_utc"] == t_355)]
        assert len(d_prev) == 1
        impulse_d = impulse[(df["symbol"] == "D") & (df["bar_start_utc"] == t_355)].iloc[0]
        e_prev_at_355, f_prev_at_355 = 0.02, 0.05
        expected_d_own_prev = 0.07
        expected_impulse_d = (e_prev_at_355 + f_prev_at_355) / 2 - expected_d_own_prev
        assert impulse_d == pytest.approx(expected_impulse_d)


class TestCandidate4LeaderImpulseCheck4GroupMembershipFrozen:
    def test_only_the_passed_group_col_determines_grouping_never_a_lookalike_column(self):
        symbol_times = {
            "A": [("2026-01-05 03:45", 0.01), ("2026-01-05 03:50", 0.0)],
            "B": [("2026-01-05 03:45", 0.02), ("2026-01-05 03:50", 0.0)],
        }
        grp = {"A": "G1", "B": "G1"}
        df = _c4_frame(symbol_times, grp)
        df["fake_sector"] = df["symbol"].map({"A": "IT", "B": "PHARMA"})

        impulse_using_real_group = compute_leader_impulse(df, "grp", min_group_size=1)

        df_scrambled_fake = df.copy()
        df_scrambled_fake["fake_sector"] = df_scrambled_fake["symbol"].map({"A": "PHARMA", "B": "IT"})
        impulse_after_scrambling_decoy = compute_leader_impulse(df_scrambled_fake, "grp", min_group_size=1)

        pd.testing.assert_series_equal(impulse_using_real_group.reset_index(drop=True), impulse_after_scrambling_decoy.reset_index(drop=True))


def _synthetic_family_dataset(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "symbol": [f"S{i % 10}" for i in range(n)],
        "bar_start_utc": pd.date_range("2026-01-01", periods=n, freq="min", tz=UTC),
        "trading_date": pd.date_range("2026-01-01", periods=n, freq="min", tz=UTC).date,
        "pop_M4": [True] * n, "pop_M3": [i % 2 == 0 for i in range(n)],
        "return_5m": rng.normal(size=n), "relative_strength_vs_universe": rng.normal(size=n),
    })
    for col in CONTINUOUS_CANDIDATES:
        if col.startswith("c3_") and col.endswith("direction"):
            continue
        df[col] = rng.random(n) if "abnormal" in col else rng.normal(size=n)
    for col, states in CATEGORICAL_CANDIDATES.items():
        df[col] = rng.choice(states, size=n)
    for h in HORIZONS:
        up = rng.random(n) < 0.4
        down = ~up & (rng.random(n) < 0.6)
        df[f"direction_label_{h}m"] = np.where(up, "UP", np.where(down, "DOWN", "AMBIGUOUS"))
    return df


class TestTestBudget:
    def test_is_deterministic(self):
        assert compute_test_budget() == compute_test_budget()

    def test_row_count_produced_by_family_a_matches_the_budgeted_count(self):
        df = _synthetic_family_dataset()
        alpha = 0.05 / sum(compute_test_budget().values())
        family_a = run_family_a(df, alpha)
        assert len(family_a) == compute_test_budget()["A_standalone"]

    def test_row_count_produced_by_family_b_matches_the_budgeted_count(self):
        df = _synthetic_family_dataset()
        alpha = 0.05 / sum(compute_test_budget().values())
        family_b = run_family_b(df, alpha)
        assert len(family_b) == compute_test_budget()["B_incremental_value"]

    def test_budget_shrinks_correctly_when_candidate_4_columns_are_removed(self):
        import scripts.run_phase_m_statistical_research as mod
        original = list(mod.CONTINUOUS_CANDIDATES)
        try:
            mod.CONTINUOUS_CANDIDATES = [c for c in original if not c.startswith("c4_")]
            df = _synthetic_family_dataset()
            budget = compute_test_budget()
            alpha = 0.05 / sum(budget.values())
            family_a = run_family_a(df, alpha)
            family_b = run_family_b(df, alpha)
            assert len(family_a) == budget["A_standalone"]
            assert len(family_b) == budget["B_incremental_value"]
        finally:
            mod.CONTINUOUS_CANDIDATES = original


class TestPickSurvivors:
    def test_requires_significance_in_family_a_and_all_four_family_b_cells(self):
        family_a = [{"population": "M4", "quantity": "c1_universe_breadth", "bucket": "lowest_quintile", "significant_after_correction": True}]
        family_b_all_sig = [
            {"population": "M4", "quantity": "c1_universe_breadth", "control": "return_5m", "controlled_group": "below_median_return_5m", "significant_after_correction": True},
            {"population": "M4", "quantity": "c1_universe_breadth", "control": "return_5m", "controlled_group": "at_or_above_median_return_5m", "significant_after_correction": True},
            {"population": "M4", "quantity": "c1_universe_breadth", "control": "relative_strength_vs_universe", "controlled_group": "below_median_relative_strength_vs_universe", "significant_after_correction": True},
            {"population": "M4", "quantity": "c1_universe_breadth", "control": "relative_strength_vs_universe", "controlled_group": "at_or_above_median_relative_strength_vs_universe", "significant_after_correction": True},
        ]
        survivors = pick_survivors(family_a, family_b_all_sig)
        assert len(survivors) == 1

    def test_excludes_when_only_three_of_four_family_b_cells_are_significant(self):
        family_a = [{"population": "M4", "quantity": "c1_universe_breadth", "bucket": "lowest_quintile", "significant_after_correction": True}]
        family_b_partial = [
            {"population": "M4", "quantity": "c1_universe_breadth", "control": "return_5m", "controlled_group": "below", "significant_after_correction": True},
            {"population": "M4", "quantity": "c1_universe_breadth", "control": "return_5m", "controlled_group": "above", "significant_after_correction": True},
            {"population": "M4", "quantity": "c1_universe_breadth", "control": "relative_strength_vs_universe", "controlled_group": "below", "significant_after_correction": True},
            {"population": "M4", "quantity": "c1_universe_breadth", "control": "relative_strength_vs_universe", "controlled_group": "above", "significant_after_correction": False},
        ]
        assert pick_survivors(family_a, family_b_partial) == []

    def test_excludes_when_family_a_itself_not_significant(self):
        family_a = [{"population": "M4", "quantity": "c1_universe_breadth", "bucket": "lowest_quintile", "significant_after_correction": False}]
        family_b = [{"population": "M4", "quantity": "c1_universe_breadth", "control": "return_5m", "controlled_group": "x", "significant_after_correction": True}] * 4
        assert pick_survivors(family_a, family_b) == []


class TestFinalVerdict:
    def test_verdict_d_with_no_survivors(self):
        v = final_verdict([], {}, {}, {})
        assert v["verdict"] == "D"
        assert v["n_survivors"] == 0

    def test_never_ranks_or_promotes_when_there_are_no_survivors(self):
        v = final_verdict([], {}, {}, {})
        assert "survivors" not in v or v.get("survivors") in (None, [])
