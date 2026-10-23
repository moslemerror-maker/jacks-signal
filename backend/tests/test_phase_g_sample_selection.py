from __future__ import annotations

from scripts.run_phase_g_sample_selection import _rank_pick, select_sample


def _universe():
    return [
        {"symbol": "A", "sector": "S1", "tier": "TIER1", "security_id": 1},
        {"symbol": "B", "sector": "S1", "tier": "TIER1", "security_id": 2},
        {"symbol": "C", "sector": "S1", "tier": "TIER1", "security_id": 3},
        {"symbol": "D", "sector": "S2", "tier": "TIER2", "security_id": 4},
        {"symbol": "E", "sector": "S2", "tier": "TIER2", "security_id": 5},
    ]


def _stats():
    return {
        1: {"avg_daily_volume": 100, "avg_daily_turnover": 1000, "avg_daily_range_pct": 0.01, "avg_close": 10.0, "session_count": 250},
        2: {"avg_daily_volume": 200, "avg_daily_turnover": 5000, "avg_daily_range_pct": 0.05, "avg_close": 20.0, "session_count": 250},
        3: {"avg_daily_volume": 300, "avg_daily_turnover": 3000, "avg_daily_range_pct": 0.03, "avg_close": 5.0, "session_count": 250},
        4: {"avg_daily_volume": 400, "avg_daily_turnover": 9000, "avg_daily_range_pct": 0.02, "avg_close": 500.0, "session_count": 250},
        5: {"avg_daily_volume": 500, "avg_daily_turnover": 1500, "avg_daily_range_pct": 0.08, "avg_close": 1.0, "session_count": 250},
    }


class TestRankPick:
    def test_min_median_max_picks_the_correct_ranked_members(self):
        members = [
            {"symbol": "A", "security_id": 1, "avg_daily_turnover": 1000},
            {"symbol": "B", "security_id": 2, "avg_daily_turnover": 5000},
            {"symbol": "C", "security_id": 3, "avg_daily_turnover": 3000},
        ]
        picks = _rank_pick(members, metric="avg_daily_turnover", quantiles=[0.0, 0.5, 1.0])
        symbols_by_q = {q: m["symbol"] for m, q in picks}
        assert symbols_by_q[0.0] == "A"
        assert symbols_by_q[0.5] == "C"
        assert symbols_by_q[1.0] == "B"

    def test_deduplicates_index_collisions_for_small_groups(self):
        members = [
            {"symbol": "A", "security_id": 1, "avg_daily_turnover": 1000},
            {"symbol": "B", "security_id": 2, "avg_daily_turnover": 5000},
        ]
        picks = _rank_pick(members, metric="avg_daily_turnover", quantiles=[0.0, 0.5, 1.0])
        indices = [id(m) for m, _ in picks]
        assert len(indices) == len(set(indices))

    def test_empty_members_returns_empty(self):
        assert _rank_pick([], metric="avg_daily_turnover", quantiles=[0.0, 0.5, 1.0]) == []

    def test_is_a_pure_function_of_metric_values_not_input_order(self):
        members = [
            {"symbol": "A", "security_id": 1, "avg_daily_turnover": 1000},
            {"symbol": "B", "security_id": 2, "avg_daily_turnover": 5000},
            {"symbol": "C", "security_id": 3, "avg_daily_turnover": 3000},
        ]
        picks_1 = _rank_pick(members, metric="avg_daily_turnover", quantiles=[0.0, 0.5, 1.0])
        picks_2 = _rank_pick(list(reversed(members)), metric="avg_daily_turnover", quantiles=[0.0, 0.5, 1.0])
        assert [m["symbol"] for m, _ in picks_1] == [m["symbol"] for m, _ in picks_2]


class TestSelectSample:
    def test_every_tier_is_represented(self):
        result = select_sample(_universe(), _stats())
        tiers_in_sample = {m["tier"] for m in result["sample"]}
        assert tiers_in_sample == {"TIER1", "TIER2"}

    def test_lowest_price_symbol_is_always_included(self):
        result = select_sample(_universe(), _stats())
        symbols = {m["symbol"] for m in result["sample"]}
        assert "E" in symbols

    def test_result_is_deterministic_across_repeated_calls(self):
        r1 = select_sample(_universe(), _stats())
        r2 = select_sample(_universe(), _stats())
        assert [m["symbol"] for m in r1["sample"]] == [m["symbol"] for m in r2["sample"]]
        assert r1["sample"] == r2["sample"]

    def test_symbols_missing_from_stats_are_silently_excluded_not_crashed_on(self):
        universe = _universe() + [{"symbol": "F", "sector": "S3", "tier": "TIER3", "security_id": 99}]
        result = select_sample(universe, _stats())
        assert result["symbols_with_stats"] == 5
        assert "F" not in {m["symbol"] for m in result["sample"]}

    def test_a_symbol_selected_by_multiple_criteria_keeps_every_reason(self):
        stats = _stats()
        stats[4]["avg_daily_range_pct"] = 0.09
        result = select_sample(_universe(), stats)
        d_entry = next(m for m in result["sample"] if m["symbol"] == "D")
        assert len(d_entry["reasons"]) >= 2
