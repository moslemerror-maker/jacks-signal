from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from scripts.run_phase_h_statistical_research import (
    HORIZONS,
    INTERACTION_HORIZONS,
    INTERACTION_HYPOTHESES,
    MOVER_DEPENDENT_VARS,
    attach_pre_registered_groups,
    bonferroni_test_budget,
    build_research_ledger,
    effect_size_and_tradability,
    model_readiness_verdict,
    population_groups,
)


def _group_defs():
    return {
        "price_group": {"median_avg_close": 100.0},
        "liquidity_group": {"median_avg_daily_turnover": 1000.0},
        "volatility_group": {"median_avg_daily_range_pct": 0.03},
        "per_symbol_stats": {
            "1": {"avg_close": 50.0, "avg_daily_turnover": 500.0, "avg_daily_range_pct": 0.01, "n_days": 40},
            "2": {"avg_close": 150.0, "avg_daily_turnover": 1500.0, "avg_daily_range_pct": 0.05, "n_days": 40},
        },
    }


class TestBonferroniTestBudget:
    def test_budget_matches_exact_formula(self):
        expected = len(population_groups()) * len(HORIZONS) + len(MOVER_DEPENDENT_VARS) * len(HORIZONS) + len(INTERACTION_HYPOTHESES) * len(INTERACTION_HORIZONS)
        assert bonferroni_test_budget() == expected

    def test_budget_is_a_fixed_deterministic_integer(self):
        assert bonferroni_test_budget() == bonferroni_test_budget()


class TestAttachPreRegisteredGroups:
    def test_classifies_price_liquidity_volatility_by_frozen_median(self):
        df = pd.DataFrame({
            "security_id": [1, 2], "universe_average_return": [0.01, -0.01],
        })
        result = attach_pre_registered_groups(df, _group_defs())
        assert result.loc[0, "price_group"] == "low_priced"
        assert result.loc[1, "price_group"] == "higher_priced"
        assert result.loc[0, "liquidity_group"] == "lower_liquidity"
        assert result.loc[1, "liquidity_group"] == "higher_liquidity"
        assert result.loc[0, "volatility_group"] == "low_volatility"
        assert result.loc[1, "volatility_group"] == "high_volatility"

    def test_market_regime_derived_from_universe_average_return_sign(self):
        df = pd.DataFrame({"security_id": [1, 2], "universe_average_return": [0.02, -0.02]})
        result = attach_pre_registered_groups(df, _group_defs())
        assert result.loc[0, "market_regime"] == "MARKET_UP"
        assert result.loc[1, "market_regime"] == "MARKET_DOWN"

    def test_symbol_missing_from_group_definitions_gets_none_not_a_crash(self):
        df = pd.DataFrame({"security_id": [999], "universe_average_return": [0.01]})
        result = attach_pre_registered_groups(df, _group_defs())
        assert result.loc[0, "price_group"] is None


class TestModelReadinessVerdict:
    def _h2(self, populations_significant: dict[str, bool]) -> list[dict]:
        return [{"population": pop, "significant_after_correction": sig, "horizon_minutes": 15} for pop, sig in populations_significant.items()]

    def test_verdict_d_when_full_universe_never_significant(self):
        h2 = self._h2({"full_universe": False, "NIFTY50": False})
        split = {"by_split": {"train": {}, "validation": {}, "held_out": {}}}
        verdict = model_readiness_verdict(h2, split, stability={})
        assert verdict["verdict"] == "D"

    def test_verdict_c_when_significant_but_fails_a_chronological_split(self):
        h2 = self._h2({"full_universe": True})
        split = {"by_split": {
            "train": {"15m": {"significant_after_correction": True}},
            "validation": {"15m": {"significant_after_correction": True}},
            "held_out": {"15m": {"significant_after_correction": False}},
        }}
        verdict = model_readiness_verdict(h2, split, stability={"by_symbol": {"fraction_matching_majority_sign": 1.0}})
        assert verdict["verdict"] == "C"

    def test_verdict_c_when_effect_is_too_concentrated_across_populations(self):
        pops = {p: False for p, _ in population_groups()}
        pops["full_universe"] = True
        h2 = self._h2(pops)
        split = {"by_split": {s: {"15m": {"significant_after_correction": True}} for s in ("train", "validation", "held_out")}}
        verdict = model_readiness_verdict(h2, split, stability={"by_symbol": {"fraction_matching_majority_sign": 1.0}})
        assert verdict["verdict"] == "C"

    def test_never_returns_a_forced_verdict_a(self):
        import inspect
        from scripts import run_phase_h_statistical_research as mod
        source = inspect.getsource(mod.model_readiness_verdict)
        assert '"A"' not in source and "'A'" not in source


class TestEffectSizeAndTradability:
    def test_expected_direction_is_derived_from_the_buckets_own_sign_not_hardcoded(self):
        n = 100
        rs = [i / n for i in range(n)]
        move = [-(r - 0.5) * 2.0 for r in rs]
        df = pd.DataFrame({
            "relative_strength_vs_universe": rs, "label_normalized_move_5m": move,
            "mover_future_abs_return_5m": [abs(m) * 0.01 for m in move], "mover_future_range_pct_5m": [abs(m) * 0.02 for m in move],
            "mfe_5m": [0.01] * n, "mae_5m": [0.01] * n,
            "label_class_5m": ["UP" if m > 0 else "DOWN" for m in move],
        })
        result = effect_size_and_tradability(df, 5)
        assert result["lowest_bucket"]["mean_normalized_move"] > 0
        assert result["lowest_bucket"]["expected_direction"] == "UP"
        assert result["highest_bucket"]["mean_normalized_move"] < 0
        assert result["highest_bucket"]["expected_direction"] == "DOWN"
        assert result["lowest_bucket"]["hit_rate_for_expected_direction"] == pytest.approx(1.0)
        assert result["highest_bucket"]["hit_rate_for_expected_direction"] == pytest.approx(1.0)


class TestBuildResearchLedger:
    def test_every_h2_result_becomes_one_ledger_entry(self):
        h2 = [{"population": "full_universe", "feature": "relative_strength_vs_universe", "horizon_minutes": 5, "significant_after_correction": True, "top_minus_bottom_bucket_mean": -0.3, "n_rows": 1000}]
        ledger = build_research_ledger(h2, [], [], alpha=0.001)
        assert len(ledger) == 1
        assert ledger[0]["result"] == "SIGNIFICANT"

    def test_failed_hypotheses_are_recorded_not_dropped(self):
        h2 = [{"population": "full_universe", "feature": "x", "horizon_minutes": 5, "significant_after_correction": False, "top_minus_bottom_bucket_mean": 0.01, "n_rows": 1000}]
        ledger = build_research_ledger(h2, [], [], alpha=0.001)
        assert ledger[0]["result"] == "NOT SIGNIFICANT"

    def test_skipped_tests_are_recorded_with_their_reason(self):
        h2 = [{"population": "full_universe", "feature": "x", "horizon_minutes": 5, "skipped_reason": "insufficient rows (3)"}]
        ledger = build_research_ledger(h2, [], [], alpha=0.001)
        assert ledger[0]["result"] == "SKIPPED"
        assert "insufficient rows" in ledger[0]["interpretation"]

    def test_h3_and_h4_entries_are_included_alongside_h2(self):
        h3 = [{"feature": "relative_strength_vs_universe", "dependent_variable": "mover_future_abs_return", "horizon_minutes": 5, "significant_after_correction": True, "top_minus_bottom_bucket_mean": 0.02, "n_rows": 500}]
        h4 = [{"interactor": "volatility_expansion_ratio", "horizon_minutes": 5, "groups": {"below_median": {"top_minus_bottom_bucket_mean": -0.1, "significant_after_correction": False}, "at_or_above_median": {"top_minus_bottom_bucket_mean": -0.4, "significant_after_correction": True}}}]
        ledger = build_research_ledger([], h3, h4, alpha=0.001)
        assert len(ledger) == 2
        assert "mover_future_abs_return" in ledger[0]["hypothesis"]
        assert "volatility_expansion_ratio" in ledger[1]["hypothesis"]
