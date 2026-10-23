from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from scripts.run_phase_g_statistical_research import (
    association_analysis,
    class_frequencies,
    coverage_and_missingness,
    model_readiness_verdict,
    pick_strongest_candidate,
)


def _dates(n: int) -> list[dt.date]:
    return [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(n)]


class TestCoverageAndMissingness:
    def test_reports_exact_non_null_fraction_per_column(self):
        df = pd.DataFrame({"a": [1, 2, None, 4], "tier": ["T1", "T1", "T1", "T1"]})
        result = coverage_and_missingness(df)
        assert result["overall_non_null_fraction"]["a"] == pytest.approx(0.75)

    def test_breaks_down_by_tier_independently(self):
        df = pd.DataFrame({"a": [1, None, 3, 4], "tier": ["T1", "T1", "T2", "T2"]})
        result = coverage_and_missingness(df)
        assert result["by_tier_non_null_fraction"]["T1"]["a"] == pytest.approx(0.5)
        assert result["by_tier_non_null_fraction"]["T2"]["a"] == pytest.approx(1.0)


class TestClassFrequencies:
    def test_counts_every_class_including_none(self):
        df = pd.DataFrame({
            "label_class_5m": ["UP", "UP", "DOWN", None],
            "label_class_15m": ["NEUTRAL"] * 4,
            "label_class_30m": ["NEUTRAL"] * 4,
            "tier": ["T1"] * 4, "time_bucket": ["OPENING"] * 4, "intraday_regime": ["RANGE"] * 4,
        })
        result = class_frequencies(df)
        assert result["label_class_5m"]["overall"]["UP"] == 2
        assert result["label_class_5m"]["overall"]["DOWN"] == 1


class TestAssociationAnalysis:
    def test_finds_a_monotonic_relationship_when_one_genuinely_exists(self):
        n = 200
        feature = [i / n for i in range(n)]
        label = [f * 2.0 for f in feature]
        df = pd.DataFrame({
            "relative_volume_20": feature, "label_normalized_move_5m": label,
            "symbol": ["A"] * n, "bar_start_utc": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC"),
        })
        for col in ("vwap_deviation_normalized", "momentum_acceleration_10", "volatility_expansion_ratio",
                    "distance_to_nearest_resistance", "distance_to_nearest_support", "range_position_20",
                    "relative_strength_vs_universe", "realized_vol_vs_tod_baseline", "volume_vs_tod_baseline"):
            df[col] = feature
        for h in (5, 15, 30):
            df[f"label_normalized_move_{h}m"] = label

        results = association_analysis(df)
        rv20_5m = next(r for r in results if r["feature"] == "relative_volume_20" and r["horizon_minutes"] == 5)
        assert rv20_5m["monotonic"] is True
        assert rv20_5m["top_minus_bottom_bucket_mean"] > 0
        assert rv20_5m["pearson_correlation"] == pytest.approx(1.0, abs=1e-6)

    def test_skips_gracefully_when_a_feature_is_degenerate(self):
        n = 100
        df = pd.DataFrame({
            "relative_volume_20": [1.0] * n,
            "label_normalized_move_5m": [0.1] * n,
            "symbol": ["A"] * n, "bar_start_utc": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC"),
        })
        for col in ("vwap_deviation_normalized", "momentum_acceleration_10", "volatility_expansion_ratio",
                    "distance_to_nearest_resistance", "distance_to_nearest_support", "range_position_20",
                    "relative_strength_vs_universe", "realized_vol_vs_tod_baseline", "volume_vs_tod_baseline"):
            df[col] = [1.0] * n
        for h in (5, 15, 30):
            df[f"label_normalized_move_{h}m"] = [0.1] * n

        results = association_analysis(df)
        rv20_5m = next(r for r in results if r["feature"] == "relative_volume_20" and r["horizon_minutes"] == 5)
        assert "skipped_reason" in rv20_5m


class TestPickStrongestCandidate:
    def test_picks_the_largest_absolute_top_minus_bottom_spread(self):
        results = [
            {"feature": "a", "horizon_minutes": 5, "top_minus_bottom_bucket_mean": 0.1},
            {"feature": "b", "horizon_minutes": 5, "top_minus_bottom_bucket_mean": -0.5},
            {"feature": "c", "horizon_minutes": 5, "top_minus_bottom_bucket_mean": 0.3},
        ]
        strongest = pick_strongest_candidate(results)
        assert strongest["feature"] == "b"

    def test_ignores_skipped_entries_with_no_spread_computed(self):
        results = [
            {"feature": "a", "horizon_minutes": 5, "skipped_reason": "insufficient rows (3)"},
            {"feature": "b", "horizon_minutes": 5, "top_minus_bottom_bucket_mean": 0.2},
        ]
        strongest = pick_strongest_candidate(results)
        assert strongest["feature"] == "b"

    def test_returns_none_when_nothing_was_computed(self):
        assert pick_strongest_candidate([{"feature": "a", "skipped_reason": "x"}]) is None


class TestModelReadinessVerdict:
    def test_verdict_d_when_nothing_survives_bonferroni(self):
        association = [{"feature": "a", "horizon_minutes": 5, "low_bucket_significant_after_bonferroni": False, "high_bucket_significant_after_bonferroni": False}]
        verdict = model_readiness_verdict(association, stability={}, split_check={})
        assert verdict["verdict"] == "D"

    def test_verdict_c_when_significant_but_flips_sign_out_of_sample(self):
        association = [{"feature": "a", "horizon_minutes": 5, "low_bucket_significant_after_bonferroni": True, "high_bucket_significant_after_bonferroni": True}]
        stability = {"by_symbol": {"fraction_matching_majority_sign": 1.0}}
        split_check = {"in_sample": {"top_minus_bottom_bucket_mean": 0.5}, "out_of_sample": {"top_minus_bottom_bucket_mean": -0.3}}
        verdict = model_readiness_verdict(association, stability, split_check)
        assert verdict["verdict"] == "C"

    def test_verdict_b_when_significant_and_oos_consistent_but_symbol_sign_unstable(self):
        association = [{"feature": "a", "horizon_minutes": 5, "low_bucket_significant_after_bonferroni": True, "high_bucket_significant_after_bonferroni": True}]
        stability = {"by_symbol": {"fraction_matching_majority_sign": 0.4}}
        split_check = {"in_sample": {"top_minus_bottom_bucket_mean": 0.5}, "out_of_sample": {"top_minus_bottom_bucket_mean": 0.3}}
        verdict = model_readiness_verdict(association, stability, split_check)
        assert verdict["verdict"] == "B"

    def test_never_returns_a_forced_verdict_a(self):
        import inspect
        from scripts import run_phase_g_statistical_research as mod
        source = inspect.getsource(mod.model_readiness_verdict)
        assert '"A"' not in source and "'A'" not in source
