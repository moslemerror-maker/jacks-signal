from __future__ import annotations

import uuid
from datetime import date, timedelta

import pytest

from backend.app.research.features.types import Bar
from backend.app.research.position_lifecycle import (
    DailyPortfolioSeriesResult,
    DataQualityIncident,
    ForegoneSignal,
    MissingBarError,
    NextOpenOutcome,
    OpenPositionAtCutoff,
    PositionEvent,
    PositionLifecycleResult,
    SignalDay,
    compute_daily_mark_to_market_portfolio_series,
    simulate_position_lifecycle,
)

_I1 = uuid.UUID(int=1)
_I2 = uuid.UUID(int=2)
_I3 = uuid.UUID(int=3)
_DATES = [date(2024, 1, 1) + timedelta(days=i) for i in range(20)]


def _bars(instrument_closes: dict[date, float]) -> list[Bar]:
    return [Bar(trade_date=d, close=c, high=c, low=c) for d, c in instrument_closes.items()]


def _event(
    instrument_id, entry_date, entry_price, exit_date, exit_price,
    cost_bps=10.0, slippage_bps=5.0, signal_date=None,
) -> PositionEvent:
    cost_fraction = cost_bps / 10000.0
    slippage_fraction = slippage_bps / 10000.0
    gross_return = exit_price / entry_price - 1.0
    transaction_cost = 2.0 * cost_fraction
    slippage = 2.0 * slippage_fraction
    return PositionEvent(
        position_id=f"{instrument_id}:{entry_date.isoformat()}",
        instrument_id=instrument_id,
        signal_date=signal_date or entry_date,
        entry_date=entry_date,
        entry_price=entry_price,
        exit_date=exit_date,
        exit_price=exit_price,
        holding_sessions=(exit_date - entry_date).days,
        gross_return=gross_return,
        transaction_cost=transaction_cost,
        slippage=slippage,
        net_return=gross_return - transaction_cost - slippage,
    )


def _empty_result(events=None, open_at_cutoff=None) -> PositionLifecycleResult:
    return PositionLifecycleResult(
        completed_events=events or [],
        open_at_cutoff=open_at_cutoff or [],
        foregone_signals=[],
    )


class TestHandCalculatedMultiDayMultiPosition:

    def test_every_daily_record_matches_the_hand_computed_values(self):
        d = _DATES
        event_a = _event(_I1, d[1], 100.0, d[4], 112.0)
        event_b = _event(_I2, d[2], 200.0, d[5], 190.0)
        result = _empty_result(events=[event_a, event_b])

        bars = {
            _I1: _bars({d[1]: 102.0, d[2]: 105.0, d[3]: 108.0}),
            _I2: _bars({d[2]: 198.0, d[3]: 195.0, d[4]: 193.0}),
        }
        session_dates = d[0:7]

        series = compute_daily_mark_to_market_portfolio_series(
            result, bars, session_dates, num_sizing_slots=2, max_simultaneous_positions=2,
            transaction_cost_bps_per_side=10.0, slippage_bps_per_side=5.0,
        )
        recs = {r.trade_date: r for r in series.records}

        r = recs[d[0]]
        assert r.gross_return == pytest.approx(0.0)
        assert r.net_return == pytest.approx(0.0)
        assert r.cumulative_equity == pytest.approx(1.0)
        assert r.num_open_positions == 0
        assert r.num_entries == 0 and r.num_exits == 0
        assert r.turnover == pytest.approx(0.0)
        assert r.exposure == pytest.approx(0.0)
        assert r.cash_fraction == pytest.approx(1.0)

        r = recs[d[1]]
        assert r.gross_return == pytest.approx(0.01)
        assert r.net_return == pytest.approx(0.00925)
        assert r.cumulative_equity == pytest.approx(1.00925)
        assert r.num_open_positions == 1
        assert r.num_entries == 1 and r.num_exits == 0
        assert r.turnover == pytest.approx(0.5)
        assert r.exposure == pytest.approx(0.5)
        assert r.cash_fraction == pytest.approx(0.5)

        r = recs[d[2]]
        assert r.gross_return == pytest.approx(0.01)
        assert r.net_return == pytest.approx(0.00925)
        assert r.cumulative_equity == pytest.approx(1.0185)
        assert r.num_open_positions == 2
        assert r.num_entries == 1 and r.num_exits == 0
        assert r.turnover == pytest.approx(0.5)
        assert r.exposure == pytest.approx(1.0)
        assert r.cash_fraction == pytest.approx(0.0)

        r = recs[d[3]]
        assert r.gross_return == pytest.approx(0.0075)
        assert r.net_return == pytest.approx(0.0075)
        assert r.cumulative_equity == pytest.approx(1.026)
        assert r.num_open_positions == 2
        assert r.num_entries == 0 and r.num_exits == 0
        assert r.turnover == pytest.approx(0.0)

        r = recs[d[4]]
        assert r.gross_return == pytest.approx(0.015)
        assert r.net_return == pytest.approx(0.01425)
        assert r.cumulative_equity == pytest.approx(1.04025)
        assert r.num_open_positions == 1
        assert r.num_entries == 0 and r.num_exits == 1
        assert r.turnover == pytest.approx(0.5)
        assert r.exposure == pytest.approx(0.5)

        r = recs[d[5]]
        assert r.gross_return == pytest.approx(-0.0075)
        assert r.net_return == pytest.approx(-0.00825)
        assert r.cumulative_equity == pytest.approx(1.032)
        assert r.num_open_positions == 0
        assert r.num_entries == 0 and r.num_exits == 1
        assert r.turnover == pytest.approx(0.5)
        assert r.exposure == pytest.approx(0.0)

        r = recs[d[6]]
        assert r.gross_return == pytest.approx(0.0)
        assert r.net_return == pytest.approx(0.0)
        assert r.cumulative_equity == pytest.approx(1.032)
        assert r.num_open_positions == 0


class TestExactReconciliationToPositionEventNetReturn:
    def test_isolated_single_position_daily_legs_sum_exactly_to_its_net_return(self):
        d = _DATES
        event = _event(_I1, d[1], 100.0, d[5], 118.0, cost_bps=10.0, slippage_bps=5.0)
        result = _empty_result(events=[event])
        bars = {_I1: _bars({d[1]: 103.0, d[2]: 107.0, d[3]: 109.0, d[4]: 114.0})}
        session_dates = d[0:7]

        series = compute_daily_mark_to_market_portfolio_series(
            result, bars, session_dates, num_sizing_slots=1, max_simultaneous_positions=1,
            transaction_cost_bps_per_side=10.0, slippage_bps_per_side=5.0,
        )
        total_net = sum(r.net_return for r in series.records)
        assert total_net == pytest.approx(event.net_return, rel=1e-9)

    def test_reconciliation_holds_with_a_losing_trade_too(self):
        d = _DATES
        event = _event(_I1, d[1], 200.0, d[4], 180.0, cost_bps=8.0, slippage_bps=3.0)
        result = _empty_result(events=[event])
        bars = {_I1: _bars({d[1]: 195.0, d[2]: 190.0, d[3]: 185.0})}
        session_dates = d[0:6]

        series = compute_daily_mark_to_market_portfolio_series(
            result, bars, session_dates, num_sizing_slots=1, max_simultaneous_positions=1,
            transaction_cost_bps_per_side=8.0, slippage_bps_per_side=3.0,
        )
        total_net = sum(r.net_return for r in series.records)
        assert total_net == pytest.approx(event.net_return, rel=1e-9)


class TestAdditiveEquityCurveReconciliation:
    def test_cumulative_equity_equals_one_plus_running_sum_of_net_return(self):
        d = _DATES
        event = _event(_I1, d[1], 100.0, d[4], 105.0)
        result = _empty_result(events=[event])
        bars = {_I1: _bars({d[1]: 101.0, d[2]: 102.0, d[3]: 103.0})}
        session_dates = d[0:6]

        series = compute_daily_mark_to_market_portfolio_series(
            result, bars, session_dates, num_sizing_slots=1, max_simultaneous_positions=1,
            transaction_cost_bps_per_side=10.0, slippage_bps_per_side=5.0,
        )
        running = 0.0
        for r in series.records:
            running += r.net_return
            assert r.cumulative_equity == pytest.approx(1.0 + running, rel=1e-9)


class TestOvernightGapAndEntryExitPrices:
    def test_entry_day_leg_uses_entry_price_not_any_prior_close(self):
        d = _DATES
        event = _event(_I1, d[2], 150.0, d[3], 155.0)
        result = _empty_result(events=[event])
        bars = {_I1: _bars({d[2]: 153.0})}
        session_dates = d[0:5]

        series = compute_daily_mark_to_market_portfolio_series(
            result, bars, session_dates, num_sizing_slots=1, max_simultaneous_positions=1,
            transaction_cost_bps_per_side=0.0, slippage_bps_per_side=0.0,
        )
        entry_record = next(r for r in series.records if r.trade_date == d[2])
        assert entry_record.gross_return == pytest.approx((153.0 - 150.0) / 150.0)


class TestCostsOccurExactlyOncePerSide:
    def test_middle_days_have_no_cost_delta_between_gross_and_net(self):
        d = _DATES
        event = _event(_I1, d[1], 100.0, d[5], 110.0, cost_bps=10.0, slippage_bps=5.0)
        result = _empty_result(events=[event])
        bars = {_I1: _bars({d[1]: 101.0, d[2]: 103.0, d[3]: 105.0, d[4]: 107.0})}
        session_dates = d[0:6]

        series = compute_daily_mark_to_market_portfolio_series(
            result, bars, session_dates, num_sizing_slots=1, max_simultaneous_positions=1,
            transaction_cost_bps_per_side=10.0, slippage_bps_per_side=5.0,
        )
        by_date = {r.trade_date: r for r in series.records}
        for middle_day in (d[2], d[3], d[4]):
            r = by_date[middle_day]
            assert r.gross_return == pytest.approx(r.net_return, rel=1e-12)
        assert by_date[d[1]].net_return < by_date[d[1]].gross_return
        assert by_date[d[5]].net_return < by_date[d[5]].gross_return
        per_side_charge = (10.0 + 5.0) / 10000.0
        assert (by_date[d[1]].gross_return - by_date[d[1]].net_return) == pytest.approx(per_side_charge)
        assert (by_date[d[5]].gross_return - by_date[d[5]].net_return) == pytest.approx(per_side_charge)


class TestRepeatedSignalsDoNotCreateAdditionalTurnover:
    def test_one_completed_position_yields_exactly_one_entry_and_one_exit_in_mtm(self):
        d = _DATES
        outcome = NextOpenOutcome(
            entry_date=d[1], entry_price=100.0, exit_date=d[4], exit_price=106.0,
            gross_return=0.06, holding_sessions=3,
        )
        signals = [SignalDay(_I1, d[0], True, outcome)]
        for day in d[1:4]:
            signals.append(SignalDay(_I1, day, True, outcome))
        signals.append(SignalDay(_I1, d[4], False, None))
        lifecycle = simulate_position_lifecycle(signals, 10.0, 5.0, max_simultaneous_positions=1)
        assert len(lifecycle.completed_events) == 1

        bars = {_I1: _bars({d[1]: 101.0, d[2]: 102.0, d[3]: 103.0})}
        session_dates = d[0:6]
        series = compute_daily_mark_to_market_portfolio_series(
            lifecycle, bars, session_dates, num_sizing_slots=1, max_simultaneous_positions=1,
            transaction_cost_bps_per_side=10.0, slippage_bps_per_side=5.0,
        )
        assert sum(r.num_entries for r in series.records) == 1
        assert sum(r.num_exits for r in series.records) == 1
        assert sum(r.turnover for r in series.records) == pytest.approx(2 * (1.0 / 1))


class TestPositionCapSelectionDeterministicInMtm:
    def test_only_the_deterministically_selected_instruments_contribute(self):
        d = _DATES
        outcome_high = NextOpenOutcome(d[1], 10.0, d[3], 12.0, 0.2, 2)
        outcome_low = NextOpenOutcome(d[1], 10.0, d[3], 11.0, 0.1, 2)
        signals = [
            SignalDay(_I3, d[0], True, outcome_high),
            SignalDay(_I2, d[0], True, outcome_low),
            SignalDay(_I1, d[0], True, outcome_low),
            SignalDay(_I1, d[1], False, None),
            SignalDay(_I2, d[1], False, None),
            SignalDay(_I3, d[1], False, None),
            SignalDay(_I1, d[2], False, None),
            SignalDay(_I2, d[2], False, None),
            SignalDay(_I3, d[2], False, None),
            SignalDay(_I1, d[3], False, None),
            SignalDay(_I2, d[3], False, None),
            SignalDay(_I3, d[3], False, None),
        ]
        lifecycle = simulate_position_lifecycle(signals, 0.0, 0.0, max_simultaneous_positions=1)
        assert len(lifecycle.completed_events) == 1
        assert lifecycle.completed_events[0].instrument_id == _I1

        bars = {_I1: _bars({d[1]: 10.5, d[2]: 10.8})}
        session_dates = d[0:4]
        series = compute_daily_mark_to_market_portfolio_series(
            lifecycle, bars, session_dates, num_sizing_slots=1, max_simultaneous_positions=1,
            transaction_cost_bps_per_side=0.0, slippage_bps_per_side=0.0,
        )
        assert sum(r.num_entries for r in series.records) == 1
        assert sum(r.num_exits for r in series.records) == 1


class TestOpenPositionMarkedThroughCutoff:
    def test_open_position_contributes_daily_legs_through_the_last_session_and_never_exits(self):
        d = _DATES
        open_pos = OpenPositionAtCutoff(
            position_id=f"{_I1}:{d[1].isoformat()}", instrument_id=_I1,
            signal_date=d[0], entry_date=d[1], entry_price=100.0, scheduled_exit_date=d[10],
        )
        result = _empty_result(open_at_cutoff=[open_pos])
        bars = {_I1: _bars({d[1]: 101.0, d[2]: 103.0, d[3]: 104.0})}
        session_dates = d[0:4]

        series = compute_daily_mark_to_market_portfolio_series(
            result, bars, session_dates, num_sizing_slots=1, max_simultaneous_positions=1,
            transaction_cost_bps_per_side=10.0, slippage_bps_per_side=5.0,
        )
        by_date = {r.trade_date: r for r in series.records}
        assert by_date[d[3]].num_open_positions == 1
        assert sum(r.num_exits for r in series.records) == 0
        entry_charge = (10.0 + 5.0) / 10000.0
        assert (by_date[d[1]].gross_return - by_date[d[1]].net_return) == pytest.approx(entry_charge)
        for held_day in (d[2], d[3]):
            assert by_date[held_day].gross_return == pytest.approx(by_date[held_day].net_return)
        expected_equity = 1.0 + (104.0 / 100.0 - 1.0 - entry_charge)
        assert by_date[d[3]].cumulative_equity == pytest.approx(expected_equity)
        assert result.completed_events == []


class TestNoFutureLeakage:
    def test_appending_bars_and_session_dates_past_the_cutoff_does_not_change_earlier_records(self):
        d = _DATES
        event = _event(_I1, d[1], 100.0, d[3], 104.0)
        result = _empty_result(events=[event])
        bars_short = {_I1: _bars({d[1]: 101.0, d[2]: 102.0})}
        session_dates_short = d[0:4]

        series_short = compute_daily_mark_to_market_portfolio_series(
            result, bars_short, session_dates_short, num_sizing_slots=1,
            max_simultaneous_positions=1, transaction_cost_bps_per_side=10.0,
            slippage_bps_per_side=5.0,
        )

        bars_long = {_I1: _bars({d[1]: 101.0, d[2]: 102.0, d[5]: 999.0})}
        session_dates_long = d[0:8]
        open_extra = OpenPositionAtCutoff(
            position_id="unused", instrument_id=_I2, signal_date=d[4],
            entry_date=d[5], entry_price=50.0, scheduled_exit_date=d[9],
        )
        result_long = _empty_result(events=[event], open_at_cutoff=[])
        series_long = compute_daily_mark_to_market_portfolio_series(
            result_long, bars_long, session_dates_long, num_sizing_slots=1,
            max_simultaneous_positions=1, transaction_cost_bps_per_side=10.0,
            slippage_bps_per_side=5.0,
        )
        records_long_by_date = {r.trade_date: r for r in series_long.records}
        for r in series_short.records:
            assert records_long_by_date[r.trade_date] == r


class TestFixedSlotSizingValidation:
    def test_max_simultaneous_positions_none_is_rejected(self):
        with pytest.raises(ValueError, match="unlimited"):
            compute_daily_mark_to_market_portfolio_series(
                _empty_result(), {}, _DATES[0:2], num_sizing_slots=3,
                max_simultaneous_positions=None, transaction_cost_bps_per_side=0.0,
                slippage_bps_per_side=0.0,
            )

    def test_max_simultaneous_positions_greater_than_num_sizing_slots_is_rejected(self):
        with pytest.raises(ValueError, match="num_sizing_slots"):
            compute_daily_mark_to_market_portfolio_series(
                _empty_result(), {}, _DATES[0:2], num_sizing_slots=2,
                max_simultaneous_positions=3, transaction_cost_bps_per_side=0.0,
                slippage_bps_per_side=0.0,
            )

    def test_num_sizing_slots_equal_to_cap_is_accepted(self):
        series = compute_daily_mark_to_market_portfolio_series(
            _empty_result(), {}, _DATES[0:2], num_sizing_slots=2,
            max_simultaneous_positions=2, transaction_cost_bps_per_side=0.0,
            slippage_bps_per_side=0.0,
        )
        assert isinstance(series, DailyPortfolioSeriesResult)


class TestFixedSizingNeverExceedsExposure:
    def test_exposure_never_exceeds_one_across_a_fully_loaded_scenario(self):
        d = _DATES
        events = [
            _event(_I1, d[1], 100.0, d[4], 110.0),
            _event(_I2, d[1], 50.0, d[4], 45.0),
            _event(_I3, d[1], 200.0, d[4], 220.0),
        ]
        result = _empty_result(events=events)
        bars = {
            _I1: _bars({d[1]: 101.0, d[2]: 103.0, d[3]: 106.0}),
            _I2: _bars({d[1]: 49.0, d[2]: 48.0, d[3]: 46.0}),
            _I3: _bars({d[1]: 205.0, d[2]: 210.0, d[3]: 215.0}),
        }
        session_dates = d[0:6]
        series = compute_daily_mark_to_market_portfolio_series(
            result, bars, session_dates, num_sizing_slots=3, max_simultaneous_positions=3,
            transaction_cost_bps_per_side=10.0, slippage_bps_per_side=5.0,
        )
        for r in series.records:
            assert r.exposure <= 1.0 + 1e-9
            assert r.cash_fraction >= -1e-9

    def test_mismatched_cap_raises_rather_than_silently_exceeding_slots(self):
        d = _DATES
        events = [
            _event(_I1, d[1], 100.0, d[3], 105.0),
            _event(_I2, d[1], 50.0, d[3], 55.0),
        ]
        result = _empty_result(events=events)
        bars = {_I1: _bars({d[1]: 101.0, d[2]: 102.0}), _I2: _bars({d[1]: 51.0, d[2]: 52.0})}
        session_dates = d[0:4]
        with pytest.raises(RuntimeError, match="concurrently open"):
            compute_daily_mark_to_market_portfolio_series(
                result, bars, session_dates, num_sizing_slots=1, max_simultaneous_positions=1,
                transaction_cost_bps_per_side=0.0, slippage_bps_per_side=0.0,
            )


class TestMissingBarFailsLoudNeverGuesses:
    def test_missing_bar_on_entry_date_raises_with_exact_incident(self):
        d = _DATES
        event = _event(_I1, d[1], 100.0, d[3], 105.0)
        result = _empty_result(events=[event])
        bars = {_I1: _bars({})}
        session_dates = d[0:4]

        with pytest.raises(MissingBarError) as exc_info:
            compute_daily_mark_to_market_portfolio_series(
                result, bars, session_dates, num_sizing_slots=1, max_simultaneous_positions=1,
                transaction_cost_bps_per_side=0.0, slippage_bps_per_side=0.0,
            )
        incident = exc_info.value.incident
        assert incident == DataQualityIncident(_I1, d[1], "missing_close_on_entry_date")

    def test_missing_bar_while_held_raises_with_exact_incident(self):
        d = _DATES
        event = _event(_I1, d[1], 100.0, d[4], 110.0)
        result = _empty_result(events=[event])
        bars = {_I1: _bars({d[1]: 101.0, d[3]: 106.0})}
        session_dates = d[0:5]

        with pytest.raises(MissingBarError) as exc_info:
            compute_daily_mark_to_market_portfolio_series(
                result, bars, session_dates, num_sizing_slots=1, max_simultaneous_positions=1,
                transaction_cost_bps_per_side=0.0, slippage_bps_per_side=0.0,
            )
        incident = exc_info.value.incident
        assert incident == DataQualityIncident(_I1, d[2], "missing_close_while_held")

    def test_no_partial_result_is_ever_returned_on_a_missing_bar(self):
        d = _DATES
        event = _event(_I1, d[1], 100.0, d[4], 110.0)
        result = _empty_result(events=[event])
        bars = {_I1: _bars({d[1]: 101.0})}
        session_dates = d[0:5]

        caught = None
        try:
            returned = compute_daily_mark_to_market_portfolio_series(
                result, bars, session_dates, num_sizing_slots=1, max_simultaneous_positions=1,
                transaction_cost_bps_per_side=0.0, slippage_bps_per_side=0.0,
            )
            caught = ("no exception", returned)
        except MissingBarError:
            caught = ("raised", None)
        assert caught == ("raised", None)


class TestUnadjustedPriceMetadata:
    def test_result_is_tagged_unadjusted_raw_ohlc_with_no_corporate_action_adjustment(self):
        d = _DATES
        event = _event(_I1, d[1], 100.0, d[3], 103.0)
        result = _empty_result(events=[event])
        bars = {_I1: _bars({d[1]: 101.0, d[2]: 102.0})}
        series = compute_daily_mark_to_market_portfolio_series(
            result, bars, d[0:4], num_sizing_slots=1, max_simultaneous_positions=1,
            transaction_cost_bps_per_side=0.0, slippage_bps_per_side=0.0,
        )
        assert series.price_basis == "UNADJUSTED_RAW_OHLC"
        assert series.corporate_action_adjustment == "NOT_APPLIED"

    def test_daily_records_are_never_returned_as_a_bare_list_without_the_tags(self):
        d = _DATES
        result = _empty_result()
        series = compute_daily_mark_to_market_portfolio_series(
            result, {}, d[0:2], num_sizing_slots=1, max_simultaneous_positions=1,
            transaction_cost_bps_per_side=0.0, slippage_bps_per_side=0.0,
        )
        assert isinstance(series, DailyPortfolioSeriesResult)
        assert not isinstance(series, list)
        assert hasattr(series, "records")
        assert hasattr(series, "price_basis")
        assert hasattr(series, "corporate_action_adjustment")


class TestEventLevelReturnAndDailyPortfolioReturnRemainSeparate:
    def test_no_single_day_equals_the_full_multiday_position_event_net_return(self):
        d = _DATES
        event = _event(_I1, d[1], 100.0, d[5], 120.0)
        result = _empty_result(events=[event])
        bars = {_I1: _bars({d[1]: 103.0, d[2]: 108.0, d[3]: 112.0, d[4]: 116.0})}
        series = compute_daily_mark_to_market_portfolio_series(
            result, bars, d[0:6], num_sizing_slots=1, max_simultaneous_positions=1,
            transaction_cost_bps_per_side=10.0, slippage_bps_per_side=5.0,
        )
        for r in series.records:
            assert r.net_return != pytest.approx(event.net_return)
        assert sum(r.net_return for r in series.records) == pytest.approx(event.net_return, rel=1e-9)
