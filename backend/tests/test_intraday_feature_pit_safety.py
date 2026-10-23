from __future__ import annotations

import datetime as dt

import pytest

from backend.app.research.intraday_features import (
    bar_math,
    labels,
    market_context,
    momentum_oscillators,
    price_action,
    regime,
    vwap_levels,
    volatility_volume,
)
from backend.app.research.intraday_features.bar_math import Bar

UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def _session(trading_date: dt.date, n: int = 360, *, base_price: float = 100.0, seed: int = 0) -> list[Bar]:
    bars = []
    price = base_price
    open_ist = dt.datetime(trading_date.year, trading_date.month, trading_date.day, 9, 15, tzinfo=IST)
    rng_state = seed
    for i in range(n):
        rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
        drift = ((rng_state % 200) - 100) / 10000.0
        o = price
        c = price * (1 + drift)
        h = max(o, c) * 1.001
        l = min(o, c) * 0.999
        vol = 1000 + (rng_state % 5000)
        ts_ist = open_ist + dt.timedelta(minutes=i)
        bars.append(Bar(ts_ist.astimezone(UTC), trading_date, o, h, l, c, float(vol)))
        price = c
    return bars


@pytest.fixture(scope="module")
def multi_session_bars() -> list[Bar]:
    d1, d2, d3 = dt.date(2026, 1, 5), dt.date(2026, 1, 6), dt.date(2026, 1, 7)
    return _session(d1, seed=1) + _session(d2, seed=2) + _session(d3, seed=3)


def _feature_probes():
    return [
        ("return_over_bars_5", lambda b: price_action.return_over_bars(b, 5)),
        ("return_over_bars_30", lambda b: price_action.return_over_bars(b, 30)),
        ("distance_from_session_open", price_action.distance_from_session_open),
        ("distance_from_prev_close", price_action.distance_from_prev_close),
        ("range_position_20", lambda b: price_action.range_position(b, 20)),
        ("range_expansion_20", lambda b: price_action.range_expansion(b, 20)),
        ("momentum_acceleration_10", lambda b: price_action.momentum_acceleration(b, 10)),
        ("candle_body_ratio", lambda b: price_action.candle_body_ratio(b[-1])),
        ("consecutive_same_direction_count", price_action.consecutive_same_direction_count),
        ("is_bullish_engulfing", price_action.is_bullish_engulfing),
        ("minutes_since_open", price_action.minutes_since_open),
        ("time_bucket", price_action.time_bucket),
        ("overnight_gap", price_action.overnight_gap),
        ("realized_volatility_20", lambda b: volatility_volume.realized_volatility(b, 20)),
        ("average_true_range_14", lambda b: volatility_volume.average_true_range(b, 14)),
        ("volatility_expansion_ratio", volatility_volume.volatility_expansion_ratio),
        ("volatility_normalized_return", lambda b: volatility_volume.volatility_normalized_return(b, 5, 20)),
        ("realized_vol_vs_tod_baseline", lambda b: volatility_volume.realized_vol_vs_time_of_day_baseline(b, 10, 2)),
        ("relative_volume_20", lambda b: volatility_volume.relative_volume(b, 20)),
        ("volume_acceleration_10", lambda b: volatility_volume.volume_acceleration(b, 10)),
        ("volume_vs_tod_baseline", lambda b: volatility_volume.volume_vs_time_of_day_baseline(b, 2)),
        ("price_volume_correlation_20", lambda b: volatility_volume.price_volume_correlation(b, 20)),
        ("session_vwap", vwap_levels.session_vwap),
        ("distance_from_vwap", vwap_levels.distance_from_vwap),
        ("vwap_slope_10", lambda b: vwap_levels.vwap_slope(b, 10)),
        ("vwap_cross", vwap_levels.vwap_cross),
        ("vwap_deviation_normalized", lambda b: vwap_levels.vwap_deviation_normalized(b, 20)),
        ("previous_day_high_low_close", vwap_levels.previous_day_high_low_close),
        ("session_high_low_so_far", vwap_levels.session_high_low_so_far),
        ("opening_range_high_low_15", lambda b: vwap_levels.opening_range_high_low(b, 15)),
        ("rolling_swing_points_5_5", lambda b: vwap_levels.rolling_swing_points(b, 5, 5)),
        ("intraday_regime", regime.intraday_regime),
        ("rsi_9", lambda b: momentum_oscillators.relative_strength_index(b, 9)),
        ("rsi_14", lambda b: momentum_oscillators.relative_strength_index(b, 14)),
        ("rsi_14_slope_5", lambda b: momentum_oscillators.rsi_slope(b, 14, 5)),
        ("macd_histogram_12_26_9", lambda b: momentum_oscillators.macd_histogram(b, 12, 26, 9)),
        ("ema_9", lambda b: momentum_oscillators.ema(b, 9)),
        ("ema_20", lambda b: momentum_oscillators.ema(b, 20)),
        ("ema_50", lambda b: momentum_oscillators.ema(b, 50)),
    ]


class TestAdversarialFutureBarIndependence:

    @pytest.mark.parametrize("name,fn", _feature_probes(), ids=[p[0] for p in _feature_probes()])
    def test_value_unchanged_when_future_bars_appended(self, multi_session_bars, name, fn):
        t_index = 400
        t = multi_session_bars[t_index].bar_start_utc

        source_at_t = multi_session_bars[: t_index + 1]
        source_with_future = multi_session_bars[: t_index + 1 + 100]

        bars_at_t = bar_math.bars_as_of(source_at_t, t)
        bars_with_future_filtered = bar_math.bars_as_of(source_with_future, t)
        assert bars_at_t == bars_with_future_filtered

        value_at_t = fn(bars_at_t)
        value_with_future = fn(bars_with_future_filtered)
        assert value_at_t == value_with_future, f"{name} changed when future bars were appended: {value_at_t!r} != {value_with_future!r}"

    @pytest.mark.parametrize("name,fn", _feature_probes(), ids=[p[0] for p in _feature_probes()])
    def test_value_stable_at_multiple_future_horizons(self, multi_session_bars, name, fn):
        t_index = 400
        t = multi_session_bars[t_index].bar_start_utc

        no_future = fn(bar_math.bars_as_of(multi_session_bars[: t_index + 1], t))
        one_future_bar = fn(bar_math.bars_as_of(multi_session_bars[: t_index + 2], t))
        two_hundred_future_bars = fn(bar_math.bars_as_of(multi_session_bars[: min(t_index + 1 + 200, len(multi_session_bars))], t))
        assert no_future == one_future_bar == two_hundred_future_bars


class TestBarsAsOf:

    def test_filters_out_everything_strictly_after_t(self, multi_session_bars):
        t_index = 400
        t = multi_session_bars[t_index].bar_start_utc
        result = bar_math.bars_as_of(multi_session_bars, t)
        assert result == multi_session_bars[: t_index + 1]

    def test_boundary_is_inclusive_of_t_itself(self, multi_session_bars):
        t_index = 400
        t = multi_session_bars[t_index].bar_start_utc
        result = bar_math.bars_as_of(multi_session_bars, t)
        assert result[-1].bar_start_utc == t

    def test_empty_when_t_is_before_the_first_bar(self, multi_session_bars):
        before_first = multi_session_bars[0].bar_start_utc - dt.timedelta(minutes=1)
        assert bar_math.bars_as_of(multi_session_bars, before_first) == []

    def test_result_is_independent_of_how_much_data_exists_after_t_in_the_source(self, multi_session_bars):
        t_index = 400
        t = multi_session_bars[t_index].bar_start_utc
        shortest_source = multi_session_bars[: t_index + 1]
        longest_source = multi_session_bars
        assert bar_math.bars_as_of(shortest_source, t) == bar_math.bars_as_of(longest_source, t)


class TestTodaysFinalHighLowNeverLeaks:

    def test_session_high_low_so_far_excludes_the_days_eventual_extremes(self, multi_session_bars):
        session3_start = 720
        early_cutoff = session3_start + 60
        full_session3 = multi_session_bars[session3_start : session3_start + 360]
        day_final_high = max(b.high for b in full_session3)
        day_final_low = min(b.low for b in full_session3)

        bars_up_to_early_t = multi_session_bars[: early_cutoff + 1]
        high_so_far, low_so_far = vwap_levels.session_high_low_so_far(bars_up_to_early_t)

        later_bars = full_session3[: 61]
        assert high_so_far == max(b.high for b in later_bars)
        assert low_so_far == min(b.low for b in later_bars)
        if day_final_high not in [b.high for b in later_bars]:
            assert high_so_far != day_final_high
        if day_final_low not in [b.low for b in later_bars]:
            assert low_so_far != day_final_low


class TestSwingPointsNeverAnticipateTheFuture:

    def test_an_extreme_bar_within_the_confirmation_window_of_t_is_not_yet_reported(self, multi_session_bars):
        right = 5
        t_index = 400
        bars_up_to_t = list(multi_session_bars[: t_index + 1])
        spiked = bars_up_to_t[:-1] + [bars_up_to_t[-1]._replace(high=1_000_000.0)]

        highs, _ = vwap_levels.rolling_swing_points(spiked, left=5, right=right)
        assert 1_000_000.0 not in highs

        extended = spiked + multi_session_bars[t_index + 1 : t_index + 1 + right]
        highs_later, _ = vwap_levels.rolling_swing_points(extended, left=5, right=right)
        assert 1_000_000.0 in highs_later

    def test_appending_future_bars_can_only_ADD_swing_points_never_change_existing_ones(self, multi_session_bars):
        t_index = 400
        highs_before, lows_before = vwap_levels.rolling_swing_points(multi_session_bars[: t_index + 1], left=5, right=5)
        highs_after, lows_after = vwap_levels.rolling_swing_points(multi_session_bars[: t_index + 1 + 50], left=5, right=5)
        assert set(highs_before).issubset(set(highs_after))
        assert set(lows_before).issubset(set(lows_after))


class TestLabelsCorrectlyUseFutureData:

    def test_label_class_depends_on_future_bars_not_just_bars_up_to_t(self, multi_session_bars):
        t_index = 30
        bars_up_to_t = multi_session_bars[: t_index + 1]
        future_up = [Bar(b.bar_start_utc, b.trading_date, b.open, b.high, b.low, b.open * 1.05, b.volume) for b in multi_session_bars[t_index + 1 : t_index + 6]]
        future_down = [Bar(b.bar_start_utc, b.trading_date, b.open, b.high, b.low, b.open * 0.95, b.volume) for b in multi_session_bars[t_index + 1 : t_index + 6]]

        vol = volatility_volume.realized_volatility(bars_up_to_t, 20) or 0.01
        label_up = labels.forward_movement_label(bars_up_to_t, future_up, horizon_bars=5, vol_normalizer=vol, threshold=0.1)
        label_down = labels.forward_movement_label(bars_up_to_t, future_down, horizon_bars=5, vol_normalizer=vol, threshold=0.1)

        assert label_up.label_class == labels.UP
        assert label_down.label_class == labels.DOWN

    def test_label_undefined_when_future_window_crosses_into_next_session(self, multi_session_bars):
        t_index = 358
        bars_up_to_t = multi_session_bars[: t_index + 1]
        future_bars = multi_session_bars[t_index + 1 : t_index + 1 + 5]
        vol = volatility_volume.realized_volatility(bars_up_to_t, 20)
        result = labels.forward_movement_label(bars_up_to_t, future_bars, horizon_bars=5, vol_normalizer=vol, threshold=0.1)
        assert result is None

    def test_label_undefined_with_insufficient_future_bars(self, multi_session_bars):
        t_index = len(multi_session_bars) - 2
        bars_up_to_t = multi_session_bars[: t_index + 1]
        future_bars = multi_session_bars[t_index + 1 :]
        vol = volatility_volume.realized_volatility(bars_up_to_t, 20)
        assert labels.forward_movement_label(bars_up_to_t, future_bars, horizon_bars=5, vol_normalizer=vol, threshold=0.1) is None

    def test_excursion_label_also_respects_the_session_boundary_rule(self, multi_session_bars):
        t_index = 358
        bars_up_to_t = multi_session_bars[: t_index + 1]
        future_bars = multi_session_bars[t_index + 1 : t_index + 1 + 5]
        assert labels.excursion_label(bars_up_to_t, future_bars, horizon_bars=5) is None


class TestEntryPathResult:

    def test_offset_zero_matches_excursion_label_exactly(self, multi_session_bars):
        t_index = 30
        bars_up_to_t = multi_session_bars[: t_index + 1]
        future_bars = multi_session_bars[t_index + 1 : t_index + 1 + 10]
        ex = labels.excursion_label(bars_up_to_t, future_bars, horizon_bars=5)
        ep = labels.entry_path_result(bars_up_to_t, future_bars, entry_offset_bars=0, horizon_bars=5)
        assert ep.entry_price == pytest.approx(ex.reference_price)
        assert ep.max_upside_excursion == pytest.approx(ex.max_upside_excursion)
        assert ep.max_downside_excursion == pytest.approx(ex.max_downside_excursion)

    def test_offset_one_entry_price_is_the_open_of_the_next_bar(self, multi_session_bars):
        t_index = 30
        bars_up_to_t = multi_session_bars[: t_index + 1]
        future_bars = multi_session_bars[t_index + 1 : t_index + 1 + 10]
        ep = labels.entry_path_result(bars_up_to_t, future_bars, entry_offset_bars=1, horizon_bars=5)
        assert ep.entry_price == pytest.approx(future_bars[0].open)

    def test_horizon_is_measured_from_entry_not_from_t(self, multi_session_bars):
        t_index = 30
        bars_up_to_t = multi_session_bars[: t_index + 1]
        future_bars = multi_session_bars[t_index + 1 : t_index + 1 + 12]
        ep = labels.entry_path_result(bars_up_to_t, future_bars, entry_offset_bars=3, horizon_bars=5)
        window = future_bars[3:8]
        expected_gross = (window[-1].close / future_bars[2].open) - 1.0
        assert ep.gross_return == pytest.approx(expected_gross)
        assert ep.max_upside_excursion == pytest.approx(max(0.0, (max(b.high for b in window) - future_bars[2].open) / future_bars[2].open))

    def test_none_when_entry_bar_itself_crosses_into_next_session(self, multi_session_bars):
        t_index = 359
        bars_up_to_t = multi_session_bars[: t_index + 1]
        future_bars = multi_session_bars[t_index + 1 : t_index + 1 + 10]
        assert labels.entry_path_result(bars_up_to_t, future_bars, entry_offset_bars=1, horizon_bars=5) is None

    def test_none_when_entry_bar_is_same_session_but_horizon_window_crosses_into_next_session(self, multi_session_bars):
        t_index = 355
        bars_up_to_t = multi_session_bars[: t_index + 1]
        future_bars = multi_session_bars[t_index + 1 : t_index + 1 + 10]
        assert future_bars[0].trading_date == bars_up_to_t[-1].trading_date
        assert labels.entry_path_result(bars_up_to_t, future_bars, entry_offset_bars=1, horizon_bars=5) is None

    def test_none_with_insufficient_future_bars(self, multi_session_bars):
        t_index = len(multi_session_bars) - 3
        bars_up_to_t = multi_session_bars[: t_index + 1]
        future_bars = multi_session_bars[t_index + 1 :]
        assert labels.entry_path_result(bars_up_to_t, future_bars, entry_offset_bars=5, horizon_bars=10) is None

    def test_appending_bars_past_the_needed_window_does_not_change_the_result(self, multi_session_bars):
        t_index = 30
        bars_up_to_t = multi_session_bars[: t_index + 1]
        needed = multi_session_bars[t_index + 1 : t_index + 1 + 8]
        extended = needed + [Bar(b.bar_start_utc, b.trading_date, b.open, b.high, b.low, 999999.0, b.volume) for b in multi_session_bars[t_index + 9 : t_index + 20]]
        before = labels.entry_path_result(bars_up_to_t, needed, entry_offset_bars=3, horizon_bars=5)
        after = labels.entry_path_result(bars_up_to_t, extended, entry_offset_bars=3, horizon_bars=5)
        assert before == after

    def test_negative_offset_or_zero_horizon_is_rejected(self, multi_session_bars):
        t_index = 30
        bars_up_to_t = multi_session_bars[: t_index + 1]
        future_bars = multi_session_bars[t_index + 1 : t_index + 1 + 10]
        assert labels.entry_path_result(bars_up_to_t, future_bars, entry_offset_bars=-1, horizon_bars=5) is None
        assert labels.entry_path_result(bars_up_to_t, future_bars, entry_offset_bars=1, horizon_bars=0) is None


class TestNextBarEntryPathResult:

    def test_close_basis_uses_the_next_bars_own_close(self, multi_session_bars):
        t_index = 30
        bars_up_to_t = multi_session_bars[: t_index + 1]
        future_bars = multi_session_bars[t_index + 1 : t_index + 1 + 10]
        ep = labels.next_bar_entry_path_result(bars_up_to_t, future_bars, horizon_bars=5, price_basis="close")
        assert ep.entry_price == pytest.approx(future_bars[0].close)

    def test_vwap_proxy_basis_is_the_typical_price_of_the_next_bar(self, multi_session_bars):
        t_index = 30
        bars_up_to_t = multi_session_bars[: t_index + 1]
        future_bars = multi_session_bars[t_index + 1 : t_index + 1 + 10]
        ep = labels.next_bar_entry_path_result(bars_up_to_t, future_bars, horizon_bars=5, price_basis="vwap_proxy")
        b = future_bars[0]
        assert ep.entry_price == pytest.approx((b.high + b.low + b.close) / 3.0)

    def test_window_matches_offset_one_entry_path_result_exactly(self, multi_session_bars):
        t_index = 30
        bars_up_to_t = multi_session_bars[: t_index + 1]
        future_bars = multi_session_bars[t_index + 1 : t_index + 1 + 10]
        ref = labels.entry_path_result(bars_up_to_t, future_bars, entry_offset_bars=1, horizon_bars=5)
        close_ep = labels.next_bar_entry_path_result(bars_up_to_t, future_bars, horizon_bars=5, price_basis="close")
        ref_future_close = ref.entry_price * (1 + ref.gross_return)
        close_future_close = close_ep.entry_price * (1 + close_ep.gross_return)
        assert close_future_close == pytest.approx(ref_future_close)

    def test_none_when_entry_bar_crosses_into_next_session(self, multi_session_bars):
        t_index = 359
        bars_up_to_t = multi_session_bars[: t_index + 1]
        future_bars = multi_session_bars[t_index + 1 : t_index + 1 + 10]
        assert labels.next_bar_entry_path_result(bars_up_to_t, future_bars, horizon_bars=5, price_basis="close") is None

    def test_none_when_window_crosses_into_next_session(self, multi_session_bars):
        t_index = 355
        bars_up_to_t = multi_session_bars[: t_index + 1]
        future_bars = multi_session_bars[t_index + 1 : t_index + 1 + 10]
        assert future_bars[0].trading_date == bars_up_to_t[-1].trading_date
        assert labels.next_bar_entry_path_result(bars_up_to_t, future_bars, horizon_bars=5, price_basis="close") is None

    def test_appending_bars_past_the_needed_window_does_not_change_the_result(self, multi_session_bars):
        t_index = 30
        bars_up_to_t = multi_session_bars[: t_index + 1]
        needed = multi_session_bars[t_index + 1 : t_index + 1 + 6]
        extended = needed + [Bar(b.bar_start_utc, b.trading_date, b.open, b.high, b.low, 999999.0, b.volume) for b in multi_session_bars[t_index + 7 : t_index + 20]]
        before = labels.next_bar_entry_path_result(bars_up_to_t, needed, horizon_bars=5, price_basis="vwap_proxy")
        after = labels.next_bar_entry_path_result(bars_up_to_t, extended, horizon_bars=5, price_basis="vwap_proxy")
        assert before == after

    def test_invalid_price_basis_is_rejected(self, multi_session_bars):
        t_index = 30
        bars_up_to_t = multi_session_bars[: t_index + 1]
        future_bars = multi_session_bars[t_index + 1 : t_index + 1 + 10]
        with pytest.raises(ValueError):
            labels.next_bar_entry_path_result(bars_up_to_t, future_bars, horizon_bars=5, price_basis="open")


class TestMarketContextCrossSectionalContract:
    def test_relative_strength_is_zero_when_stock_matches_the_universe(self):
        rs = market_context.relative_strength_vs_universe(0.02, 0.02)
        assert rs == pytest.approx(0.0)

    def test_universe_breadth_and_average_are_pure_functions_of_the_supplied_mapping(self):
        returns = {"A": 0.01, "B": -0.02, "C": 0.03}
        assert market_context.universe_breadth(returns) == pytest.approx(2 / 3)
        assert market_context.universe_average_return(returns) == pytest.approx((0.01 - 0.02 + 0.03) / 3)
