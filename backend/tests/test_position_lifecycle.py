from __future__ import annotations

import uuid
from datetime import date, timedelta

import pytest

from backend.app.research.features.types import Bar
from backend.app.research.position_lifecycle import (
    ForegoneSignal,
    NextOpenOutcome,
    OpenPositionAtCutoff,
    PositionEvent,
    SignalDay,
    compute_daily_mark_to_market_portfolio_series,
    event_realization_series_lump_sum_on_exit,
    simulate_position_lifecycle,
)

_I1 = uuid.UUID(int=1)
_I2 = uuid.UUID(int=2)
_I3 = uuid.UUID(int=3)
_DATES = [date(2024, 1, 1) + timedelta(days=i) for i in range(60)]


def _outcome(entry_idx: int, exit_idx: int, entry_price: float, exit_price: float) -> NextOpenOutcome:
    return NextOpenOutcome(
        entry_date=_DATES[entry_idx],
        entry_price=entry_price,
        exit_date=_DATES[exit_idx],
        exit_price=exit_price,
        gross_return=exit_price / entry_price - 1.0,
        holding_sessions=exit_idx - entry_idx,
    )


def _signal(instrument_id, day_idx, is_long, outcome=None):
    return SignalDay(
        instrument_id=instrument_id,
        signal_date=_DATES[day_idx],
        is_long_signal=is_long,
        next_open_outcome=outcome,
    )


class TestBasicEntryExit:
    def test_a_single_signal_produces_exactly_one_completed_event(self):
        signals = [
            _signal(_I1, 0, True, _outcome(1, 6, 100.0, 105.0)),
            SignalDay(_I1, _DATES[6], False, None),
        ]
        result = simulate_position_lifecycle(signals, 0.0, 0.0, None)
        assert len(result.completed_events) == 1
        event = result.completed_events[0]
        assert event.instrument_id == _I1
        assert event.signal_date == _DATES[0]
        assert event.entry_date == _DATES[1]
        assert event.exit_date == _DATES[6]
        assert event.gross_return == pytest.approx(0.05)
        assert result.open_at_cutoff == []
        assert result.foregone_signals == []


class TestExactCostConvention:
    def test_cost_and_slippage_are_charged_per_side_twice_total(self):
        signals = [
            _signal(_I1, 0, True, _outcome(1, 6, 100.0, 105.0)),
            SignalDay(_I1, _DATES[6], False, None),
        ]
        result = simulate_position_lifecycle(
            signals, transaction_cost_bps_per_side=10.0, slippage_bps_per_side=5.0,
            max_simultaneous_positions=None,
        )
        event = result.completed_events[0]
        expected_cost = 2 * (10.0 / 10000.0)
        expected_slippage = 2 * (5.0 / 10000.0)
        assert event.transaction_cost == pytest.approx(expected_cost)
        assert event.slippage == pytest.approx(expected_slippage)
        assert event.net_return == pytest.approx(
            event.gross_return - expected_cost - expected_slippage
        )

    def test_zero_cost_and_slippage_leaves_net_equal_to_gross(self):
        signals = [
            _signal(_I1, 0, True, _outcome(1, 6, 100.0, 105.0)),
            SignalDay(_I1, _DATES[6], False, None),
        ]
        result = simulate_position_lifecycle(signals, 0.0, 0.0, None)
        event = result.completed_events[0]
        assert event.net_return == pytest.approx(event.gross_return)

    def test_rejects_negative_cost_or_slippage(self):
        with pytest.raises(ValueError, match="transaction_cost_bps_per_side"):
            simulate_position_lifecycle([], -1.0, 0.0, None)
        with pytest.raises(ValueError, match="slippage_bps_per_side"):
            simulate_position_lifecycle([], 0.0, -1.0, None)


class TestRepeatedSignalsCreateExactlyOnePosition:
    def test_signal_true_every_day_for_the_whole_holding_period_still_yields_one_position(self):
        outcome = _outcome(1, 6, 100.0, 105.0)
        signals = [
            _signal(_I1, i, True, outcome if i == 0 else _outcome(i + 1, i + 6, 100.0, 105.0))
            for i in range(6)
        ]
        signals.append(SignalDay(_I1, _DATES[6], False, None))
        result = simulate_position_lifecycle(signals, 0.0, 0.0, None)
        assert len(result.completed_events) == 1
        assert result.open_at_cutoff == []
        assert len(result.foregone_signals) == 0

    def test_no_two_overlapping_positions_on_the_same_instrument(self):
        signals = [
            _signal(_I1, 0, True, _outcome(1, 6, 100.0, 105.0)),
            _signal(_I1, 3, True, _outcome(4, 9, 999.0, 1.0)),
            _signal(_I1, 6, True, _outcome(7, 12, 105.0, 110.0)),
            SignalDay(_I1, _DATES[12], False, None),
        ]
        result = simulate_position_lifecycle(signals, 0.0, 0.0, None)
        assert len(result.completed_events) == 2
        first, second = sorted(result.completed_events, key=lambda e: e.entry_date)
        assert first.exit_date <= second.entry_date
        assert all(e.entry_price != 999.0 for e in result.completed_events)


class TestSignalDisappearingWhileHeld:
    def test_holds_to_the_original_scheduled_exit_even_if_signal_turns_false(self):
        signals = [
            _signal(_I1, 0, True, _outcome(1, 6, 100.0, 90.0)),
            _signal(_I1, 2, False, None),
            SignalDay(_I1, _DATES[6], False, None),
        ]
        result = simulate_position_lifecycle(signals, 0.0, 0.0, None)
        assert len(result.completed_events) == 1
        assert result.completed_events[0].exit_date == _DATES[6]
        assert result.completed_events[0].gross_return == pytest.approx(90.0 / 100.0 - 1.0)


class TestOpenAtCutoffExcludedFromCompletedStats:
    def test_a_position_never_reaching_its_scheduled_exit_is_tracked_separately(self):
        signals = [_signal(_I1, 0, True, _outcome(1, 6, 100.0, 105.0))]
        result = simulate_position_lifecycle(signals, 0.0, 0.0, None)
        assert result.completed_events == []
        assert len(result.open_at_cutoff) == 1
        open_pos = result.open_at_cutoff[0]
        assert open_pos.instrument_id == _I1
        assert open_pos.scheduled_exit_date == _DATES[6]

    def test_open_at_cutoff_never_appears_in_completed_events(self):
        signals = [
            _signal(_I1, 0, True, _outcome(1, 6, 100.0, 105.0)),
            _signal(_I2, 50, True, _outcome(51, 56, 200.0, 210.0)),
            SignalDay(instrument_id=_I1, signal_date=_DATES[6], is_long_signal=False, next_open_outcome=None),
        ]
        result = simulate_position_lifecycle(signals, 0.0, 0.0, None)
        completed_ids = {e.instrument_id for e in result.completed_events}
        open_ids = {p.instrument_id for p in result.open_at_cutoff}
        assert completed_ids == {_I1}
        assert open_ids == {_I2}
        assert completed_ids.isdisjoint(open_ids)


class TestPositionCap:
    def test_excess_new_entrants_are_foregone_not_queued(self):
        signals = [
            _signal(_I1, 0, True, _outcome(1, 6, 100.0, 200.0)),
            _signal(_I2, 0, True, _outcome(1, 6, 100.0, 50.0)),
            _signal(_I3, 0, True, _outcome(1, 6, 100.0, 100.01)),
        ]
        result = simulate_position_lifecycle(
            signals, 0.0, 0.0, max_simultaneous_positions=2
        )
        entered_instruments = {e.instrument_id for e in result.completed_events} | {
            p.instrument_id for p in result.open_at_cutoff
        }
        assert len(entered_instruments) == 2
        foregone = [f for f in result.foregone_signals if f.reason == "position_cap"]
        assert len(foregone) == 1

    def test_cap_selection_is_deterministic_by_instrument_id_never_by_future_return(self):
        signals = [
            _signal(_I3, 0, True, _outcome(1, 6, 100.0, 1000.0)),
            _signal(_I1, 0, True, _outcome(1, 6, 100.0, 50.0)),
            _signal(_I2, 0, True, _outcome(1, 6, 100.0, 100.01)),
        ]
        result = simulate_position_lifecycle(
            signals, 0.0, 0.0, max_simultaneous_positions=2
        )
        entered = {e.instrument_id for e in result.completed_events} | {
            p.instrument_id for p in result.open_at_cutoff
        }
        assert entered == {_I1, _I2}
        foregone_instruments = {f.instrument_id for f in result.foregone_signals}
        assert foregone_instruments == {_I3}

    def test_a_closed_slot_frees_room_the_same_day(self):
        signals = [
            _signal(_I1, 0, True, _outcome(1, 6, 100.0, 105.0)),
            SignalDay(instrument_id=_I1, signal_date=_DATES[6], is_long_signal=False, next_open_outcome=None),
            _signal(_I2, 6, True, _outcome(7, 12, 200.0, 210.0)),
        ]
        result = simulate_position_lifecycle(signals, 0.0, 0.0, max_simultaneous_positions=1)
        assert any(e.instrument_id == _I1 for e in result.completed_events)
        entered_i2 = any(e.instrument_id == _I2 for e in result.completed_events) or any(
            p.instrument_id == _I2 for p in result.open_at_cutoff
        )
        assert entered_i2
        assert not any(f.instrument_id == _I2 and f.reason == "position_cap" for f in result.foregone_signals)

    def test_rejects_invalid_cap(self):
        with pytest.raises(ValueError, match="max_simultaneous_positions"):
            simulate_position_lifecycle([], 0.0, 0.0, max_simultaneous_positions=0)


class TestUnresolvedOutcomeForeclosesEntry:
    def test_a_signal_with_no_next_open_outcome_is_foregone_not_entered(self):
        signals = [_signal(_I1, 0, True, outcome=None)]
        result = simulate_position_lifecycle(signals, 0.0, 0.0, None)
        assert result.completed_events == []
        assert result.open_at_cutoff == []
        assert len(result.foregone_signals) == 1
        assert result.foregone_signals[0].reason == "unresolved_next_open_outcome"


class TestDeterministicReproduction:
    def test_identical_inputs_produce_identical_output(self):
        signals = [
            _signal(_I1, 0, True, _outcome(1, 6, 100.0, 105.0)),
            _signal(_I2, 2, True, _outcome(3, 8, 50.0, 48.0)),
        ]
        first = simulate_position_lifecycle(signals, 10.0, 5.0, max_simultaneous_positions=1)
        second = simulate_position_lifecycle(list(signals), 10.0, 5.0, max_simultaneous_positions=1)
        assert first == second

    def test_input_order_does_not_matter(self):
        signals = [
            _signal(_I1, 0, True, _outcome(1, 6, 100.0, 105.0)),
            _signal(_I2, 2, True, _outcome(3, 8, 50.0, 48.0)),
        ]
        forward = simulate_position_lifecycle(signals, 0.0, 0.0, None)
        reversed_result = simulate_position_lifecycle(list(reversed(signals)), 0.0, 0.0, None)
        assert sorted(forward.completed_events, key=lambda e: e.position_id) == sorted(
            reversed_result.completed_events, key=lambda e: e.position_id
        )


class TestEventLevelReturnDistinctFromDailyPortfolioReturn:

    def test_lump_sum_series_attributes_the_whole_return_to_the_exit_day_only(self):
        signals = [
            _signal(_I1, 0, True, _outcome(1, 6, 100.0, 110.0)),
            SignalDay(_I1, _DATES[6], False, None),
        ]
        result = simulate_position_lifecycle(signals, 0.0, 0.0, None)
        series = event_realization_series_lump_sum_on_exit(
            result.completed_events, _DATES[0], _DATES[10]
        )
        for d in _DATES[0:6]:
            assert series[d] == 0.0
        assert series[_DATES[6]] == pytest.approx(0.10)
        for d in _DATES[7:11]:
            assert series[d] == 0.0

    def test_a_multiday_position_does_not_fabricate_a_daily_spike_claim(self):
        signals = [
            _signal(_I1, 0, True, _outcome(1, 6, 100.0, 110.0)),
            SignalDay(_I1, _DATES[6], False, None),
        ]
        result = simulate_position_lifecycle(signals, 0.0, 0.0, None)
        lump_sum = event_realization_series_lump_sum_on_exit(
            result.completed_events, _DATES[0], _DATES[10]
        )
        assert lump_sum[_DATES[6]] == pytest.approx(0.10)

        bars = {
            _I1: [
                Bar(trade_date=_DATES[i], close=close, high=close, low=close)
                for i, close in zip(range(1, 6), [102.0, 104.0, 106.0, 108.0, 109.0])
            ]
        }
        mtm = compute_daily_mark_to_market_portfolio_series(
            result, bars, _DATES[0:11], num_sizing_slots=1, max_simultaneous_positions=1,
            transaction_cost_bps_per_side=0.0, slippage_bps_per_side=0.0,
        )
        mtm_by_date = {r.trade_date: r.net_return for r in mtm.records}
        assert any(mtm_by_date[_DATES[i]] != 0.0 for i in range(1, 6))
        assert mtm_by_date[_DATES[6]] != pytest.approx(0.10)
        assert sum(mtm_by_date.values()) == pytest.approx(sum(lump_sum.values()))

    def test_zero_position_days_are_explicitly_zero_cash(self):
        series = event_realization_series_lump_sum_on_exit([], _DATES[0], _DATES[5])
        assert all(v == 0.0 for v in series.values())
        assert len(series) == 6

    def test_lump_sum_series_requires_ordered_date_range(self):
        with pytest.raises(ValueError, match="end_date"):
            event_realization_series_lump_sum_on_exit([], _DATES[5], _DATES[0])


