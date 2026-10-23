from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import select

from backend.app.db.models import AssetType, FeatureValue, Instrument, LabelValue, MarketBar, SourceFormat
from backend.app.research.features.compute import compute_and_store_features, load_eq_bars
from backend.app.research.features.definitions import register_feature_definitions
from backend.app.research.labels.forward_return_next_open import (
    compute_and_store_next_open_labels,
    register_next_open_label_definition,
)
from backend.app.research.position_lifecycle import (
    DailyPortfolioSeriesResult,
    NextOpenOutcome,
    SignalDay,
    compute_daily_mark_to_market_portfolio_series,
    simulate_position_lifecycle,
)

_N_DAYS = 60
_TRADING_DATES = [date(2024, 1, 1) + timedelta(days=i) for i in range(_N_DAYS)]
_CLOSES = [100.0 + i for i in range(_N_DAYS)]
_PRICE_BY_DATE = dict(zip(_TRADING_DATES, _CLOSES))


@pytest.fixture
def instrument(db_session) -> Instrument:
    inst = Instrument(exchange="NSE", asset_type=AssetType.EQUITY)
    db_session.add(inst)
    db_session.flush()
    return inst


@pytest.fixture
def seeded(db_session, instrument):
    feature_definitions = register_feature_definitions(db_session)
    next_open_label_definition = register_next_open_label_definition(db_session)
    db_session.flush()

    for trade_date, close in zip(_TRADING_DATES, _CLOSES):
        db_session.add(
            MarketBar(
                instrument_id=instrument.id,
                trade_date=trade_date,
                series="EQ",
                timeframe="1D",
                open=close,
                high=close + 1.0,
                low=close - 1.0,
                close=close,
                source_format=SourceFormat.NSE_UDIFF,
                source_file="synthetic-test-fixture",
            )
        )
    db_session.flush()
    compute_and_store_features(db_session, instrument.id, feature_definitions)
    written = compute_and_store_next_open_labels(db_session, instrument.id, next_open_label_definition)
    assert written > 0

    return {
        "instrument": instrument,
        "feature_definitions": feature_definitions,
        "label_definition": next_open_label_definition,
    }


def test_real_stored_features_and_labels_drive_a_correct_full_window_simulation(db_session, seeded):
    instrument = seeded["instrument"]
    feature_definitions = seeded["feature_definitions"]
    label_definition = seeded["label_definition"]
    close_vs_sma20_id = feature_definitions["close_vs_sma20"].id

    feature_by_date = {
        r.computed_for_date: r.value
        for r in db_session.scalars(
            select(FeatureValue).where(
                FeatureValue.instrument_id == instrument.id,
                FeatureValue.feature_definition_id == close_vs_sma20_id,
            )
        ).all()
    }
    label_by_signal_date = {
        r.entry_date: r
        for r in db_session.scalars(
            select(LabelValue).where(
                LabelValue.instrument_id == instrument.id,
                LabelValue.label_definition_id == label_definition.id,
            )
        ).all()
    }

    signal_days = []
    for trade_date in _TRADING_DATES:
        is_long = feature_by_date.get(trade_date, 0.0) > 0
        label_value = label_by_signal_date.get(trade_date)
        next_open_outcome = None
        if label_value is not None:
            entry_execution_date = _TRADING_DATES[_TRADING_DATES.index(trade_date) + 1]
            next_open_outcome = NextOpenOutcome(
                entry_date=entry_execution_date,
                entry_price=_PRICE_BY_DATE[entry_execution_date],
                exit_date=label_value.horizon_end_date,
                exit_price=_PRICE_BY_DATE[label_value.horizon_end_date],
                gross_return=label_value.realized_return,
                holding_sessions=label_definition.horizon_days,
            )
        signal_days.append(SignalDay(instrument.id, trade_date, is_long, next_open_outcome))

    result = simulate_position_lifecycle(signal_days, 10.0, 5.0, max_simultaneous_positions=None)

    assert len(result.completed_events) > 0
    for event in result.completed_events:
        assert event.gross_return == pytest.approx(
            _PRICE_BY_DATE[event.exit_date] / _PRICE_BY_DATE[event.entry_date] - 1.0
        )
        expected_net = event.gross_return - 2.0 * ((10.0 + 5.0) / 10000.0)
        assert event.net_return == pytest.approx(expected_net)

    ordered = sorted(result.completed_events, key=lambda e: e.entry_date)
    for earlier, later in zip(ordered, ordered[1:]):
        assert later.entry_date >= earlier.exit_date

    for foregone in result.foregone_signals:
        assert foregone.reason == "unresolved_next_open_outcome"


def test_real_stored_next_open_label_values_produce_a_completed_position_with_correct_exit_date(
    db_session, seeded
):
    instrument = seeded["instrument"]
    label_definition = seeded["label_definition"]

    label_value = db_session.scalars(
        select(LabelValue)
        .where(
            LabelValue.instrument_id == instrument.id,
            LabelValue.label_definition_id == label_definition.id,
        )
        .order_by(LabelValue.entry_date)
        .limit(1)
    ).first()
    assert label_value is not None

    entry_execution_date = _TRADING_DATES[_TRADING_DATES.index(label_value.entry_date) + 1]

    outcome = NextOpenOutcome(
        entry_date=entry_execution_date,
        entry_price=_PRICE_BY_DATE[entry_execution_date],
        exit_date=label_value.horizon_end_date,
        exit_price=_PRICE_BY_DATE[label_value.horizon_end_date],
        gross_return=label_value.realized_return,
        holding_sessions=label_definition.horizon_days,
    )
    signals = [
        SignalDay(instrument.id, label_value.entry_date, True, outcome),
        SignalDay(instrument.id, label_value.horizon_end_date, False, None),
    ]
    result = simulate_position_lifecycle(signals, 0.0, 0.0, None)
    assert len(result.completed_events) == 1
    event = result.completed_events[0]
    assert event.exit_date == label_value.horizon_end_date
    assert event.gross_return == pytest.approx(label_value.realized_return)
    assert event.net_return == pytest.approx(event.gross_return)


def test_full_pipeline_real_labels_into_lifecycle_into_daily_mtm_series(db_session, seeded):
    instrument = seeded["instrument"]
    feature_definitions = seeded["feature_definitions"]
    label_definition = seeded["label_definition"]
    close_vs_sma20_id = feature_definitions["close_vs_sma20"].id

    feature_by_date = {
        r.computed_for_date: r.value
        for r in db_session.scalars(
            select(FeatureValue).where(
                FeatureValue.instrument_id == instrument.id,
                FeatureValue.feature_definition_id == close_vs_sma20_id,
            )
        ).all()
    }
    label_by_signal_date = {
        r.entry_date: r
        for r in db_session.scalars(
            select(LabelValue).where(
                LabelValue.instrument_id == instrument.id,
                LabelValue.label_definition_id == label_definition.id,
            )
        ).all()
    }

    signal_days = []
    for trade_date in _TRADING_DATES:
        is_long = feature_by_date.get(trade_date, 0.0) > 0
        label_value = label_by_signal_date.get(trade_date)
        next_open_outcome = None
        if label_value is not None:
            entry_execution_date = _TRADING_DATES[_TRADING_DATES.index(trade_date) + 1]
            next_open_outcome = NextOpenOutcome(
                entry_date=entry_execution_date,
                entry_price=_PRICE_BY_DATE[entry_execution_date],
                exit_date=label_value.horizon_end_date,
                exit_price=_PRICE_BY_DATE[label_value.horizon_end_date],
                gross_return=label_value.realized_return,
                holding_sessions=label_definition.horizon_days,
            )
        signal_days.append(SignalDay(instrument.id, trade_date, is_long, next_open_outcome))

    lifecycle = simulate_position_lifecycle(
        signal_days, 10.0, 5.0, max_simultaneous_positions=1
    )
    assert len(lifecycle.completed_events) > 0

    real_bars = load_eq_bars(db_session, instrument.id)
    series = compute_daily_mark_to_market_portfolio_series(
        lifecycle,
        {instrument.id: real_bars},
        _TRADING_DATES,
        num_sizing_slots=1,
        max_simultaneous_positions=1,
        transaction_cost_bps_per_side=10.0,
        slippage_bps_per_side=5.0,
    )
    assert isinstance(series, DailyPortfolioSeriesResult)
    assert series.price_basis == "UNADJUSTED_RAW_OHLC"
    assert series.corporate_action_adjustment == "NOT_APPLIED"
    assert len(series.records) == len(_TRADING_DATES)

    total_daily_net = sum(r.net_return for r in series.records)
    total_event_net = sum(e.net_return for e in lifecycle.completed_events)
    per_side_charge = (10.0 + 5.0) / 10000.0
    for open_pos in lifecycle.open_at_cutoff:
        cutoff_close = next(
            b.close for b in real_bars if b.trade_date == _TRADING_DATES[-1]
        )
        total_event_net += cutoff_close / open_pos.entry_price - 1.0 - per_side_charge
    assert total_daily_net == pytest.approx(total_event_net, rel=1e-9)
