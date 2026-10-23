from __future__ import annotations

import pytest

from scripts.run_phase_h_audit import _quantile, classify_symbols, compute_group_definitions


def _universe():
    return [
        {"symbol": "A", "sector": "S1", "tier": "T1", "security_id": 1},
        {"symbol": "B", "sector": "S1", "tier": "T1", "security_id": 2},
        {"symbol": "C", "sector": "S1", "tier": "T1", "security_id": 3},
    ]


class TestQuantile:
    def test_median_of_odd_length_list(self):
        assert _quantile([1.0, 2.0, 3.0], 0.5) == 2.0

    def test_interpolates_between_points(self):
        assert _quantile([0.0, 10.0], 0.5) == 5.0

    def test_boundaries_return_first_and_last(self):
        vals = [1.0, 2.0, 3.0, 4.0]
        assert _quantile(vals, 0.0) == 1.0
        assert _quantile(vals, 1.0) == 4.0

    def test_raises_on_empty_input(self):
        with pytest.raises(ValueError):
            _quantile([], 0.5)


class TestClassifySymbols:
    def test_symbol_with_no_coverage_row_is_excluded_with_a_reason(self):
        result = classify_symbols(_universe(), coverage_rows={2: {"n_dates": 250, "first_date": "x", "last_date": "y"}, 3: {"n_dates": 250, "first_date": "x", "last_date": "y"}})
        assert len(result["excluded"]) == 1
        assert result["excluded"][0]["symbol"] == "A"
        assert "reason" in result["excluded"][0]

    def test_symbol_below_the_date_threshold_is_sparse_not_excluded(self):
        coverage = {1: {"n_dates": 50, "first_date": "2026-01-01", "last_date": "2026-02-01"}, 2: {"n_dates": 250, "first_date": "x", "last_date": "y"}, 3: {"n_dates": 250, "first_date": "x", "last_date": "y"}}
        result = classify_symbols(_universe(), coverage)
        assert len(result["sparse"]) == 1
        assert result["sparse"][0]["symbol"] == "A"
        assert result["sparse"][0]["n_dates"] == 50
        assert len(result["excluded"]) == 0

    def test_every_universe_symbol_is_classified_exactly_once(self):
        coverage = {1: {"n_dates": 250, "first_date": "x", "last_date": "y"}, 2: {"n_dates": 50, "first_date": "x", "last_date": "y"}}
        result = classify_symbols(_universe(), coverage)
        total = len(result["excluded"]) + len(result["sparse"]) + len(result["healthy"])
        assert total == len(_universe())


class TestComputeGroupDefinitions:
    def test_median_split_thresholds_are_the_true_median(self):
        rows = [
            {"security_id": 1, "avg_daily_turnover": 100.0, "avg_daily_range_pct": 0.01, "avg_close": 10.0, "n_days": 40},
            {"security_id": 2, "avg_daily_turnover": 200.0, "avg_daily_range_pct": 0.02, "avg_close": 20.0, "n_days": 40},
            {"security_id": 3, "avg_daily_turnover": 300.0, "avg_daily_range_pct": 0.03, "avg_close": 30.0, "n_days": 40},
        ]
        defs = compute_group_definitions(rows)
        assert defs["price_group"]["median_avg_close"] == 20.0
        assert defs["liquidity_group"]["median_avg_daily_turnover"] == 200.0
        assert defs["volatility_group"]["median_avg_daily_range_pct"] == 0.02

    def test_volatility_median_excludes_symbols_with_too_few_days(self):
        rows = [
            {"security_id": 1, "avg_daily_turnover": 100.0, "avg_daily_range_pct": 0.99, "avg_close": 10.0, "n_days": 5},
            {"security_id": 2, "avg_daily_turnover": 200.0, "avg_daily_range_pct": 0.02, "avg_close": 20.0, "n_days": 40},
            {"security_id": 3, "avg_daily_turnover": 300.0, "avg_daily_range_pct": 0.03, "avg_close": 30.0, "n_days": 40},
        ]
        defs = compute_group_definitions(rows)
        assert defs["volatility_group"]["median_avg_daily_range_pct"] == pytest.approx(0.025)

    def test_is_deterministic_regardless_of_input_row_order(self):
        rows = [
            {"security_id": 1, "avg_daily_turnover": 300.0, "avg_daily_range_pct": 0.03, "avg_close": 30.0, "n_days": 40},
            {"security_id": 2, "avg_daily_turnover": 100.0, "avg_daily_range_pct": 0.01, "avg_close": 10.0, "n_days": 40},
            {"security_id": 3, "avg_daily_turnover": 200.0, "avg_daily_range_pct": 0.02, "avg_close": 20.0, "n_days": 40},
        ]
        defs_1 = compute_group_definitions(rows)
        defs_2 = compute_group_definitions(list(reversed(rows)))
        assert defs_1["price_group"]["median_avg_close"] == defs_2["price_group"]["median_avg_close"]
