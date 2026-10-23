from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from backend.app.data.canonical.intraday_equity_bar import CanonicalIntradayEquityBar, IntradayProvider
from backend.app.data.canonical.intraday_loader import upsert_intraday_bars
from backend.app.equity_live.live_feature_service import (
    INSUFFICIENT_HISTORY,
    NO_DATA,
    OK,
    STALE,
    compute_liquidity_tiers,
    compute_snapshot,
    compute_universe_snapshot,
    cross_sectional_rank_15m,
)
from scripts.run_phase_r_build_dense_dataset import compute_for_symbol

UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SID = 2885
SYMBOL = "RELIANCE"


def _bar(symbol, sid, day, hour, minute, *, close, volume, open_=None, high=None, low=None):
    bar_start_ist = dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=IST)
    bar_start_utc = bar_start_ist.astimezone(UTC)
    o = open_ if open_ is not None else close
    h = high if high is not None else max(o, close) + 0.5
    l = low if low is not None else min(o, close) - 0.5
    return CanonicalIntradayEquityBar(
        provider=IntradayProvider.DHAN, exchange_segment="NSE_EQ", security_id=sid, trading_symbol=symbol,
        raw_timestamp_epoch=int(bar_start_utc.timestamp()), bar_start_utc=bar_start_utc, bar_start_ist=bar_start_ist,
        trading_date=day, interval_minutes=1, open=o, high=h, low=l, close=close, volume=volume,
        source_provider="dhan", source_dataset="nse_eq_intraday_1m", source_key=f"NSE_EQ-{sid}-1m-{day.isoformat()}-{hour:02d}{minute:02d}",
        source_sha256="a" * 64, normalizer_version="dhan_intraday_equity/1",
    )


def _session_bars(symbol, sid, day, *, n=360, base_price=100.0, seed=0, volume=1000.0):
    bars = []
    price = base_price
    rng_state = seed
    for i in range(n):
        rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
        drift = ((rng_state % 200) - 100) / 10000.0
        hour, minute = 9 + (15 + i) // 60, (15 + i) % 60
        new_price = price * (1 + drift)
        bars.append(_bar(symbol, sid, day, hour, minute, close=new_price, volume=volume + (rng_state % 500), open_=price))
        price = new_price
    return bars


def _seed_sessions(db_session, symbol, sid, *, n_sessions, start_day=dt.date(2026, 8, 1), seed=0, volume=1000.0):
    all_bars = []
    for s in range(n_sessions):
        day = start_day + dt.timedelta(days=s)
        all_bars.extend(_session_bars(symbol, sid, day, seed=seed + s, volume=volume))
    upsert_intraday_bars(db_session, all_bars)
    db_session.commit()
    return all_bars


def _as_of(day, hour, minute):
    return dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=IST).astimezone(UTC)


class TestAdversarialPIT:
    def test_appending_future_bars_does_not_change_the_snapshot(self, db_session):
        day = dt.date(2026, 8, 1)
        _seed_sessions(db_session, SYMBOL, SID, n_sessions=1)
        as_of = _as_of(day, 10, 0)
        before = compute_snapshot(db_session, security_id=SID, symbol=SYMBOL, as_of_utc=as_of)

        future_bars = _session_bars(SYMBOL, SID, day, n=360, base_price=999999.0, seed=777)
        future_bars = [b for b in future_bars if b.bar_start_utc > as_of]
        upsert_intraday_bars(db_session, future_bars)
        db_session.commit()

        after = compute_snapshot(db_session, security_id=SID, symbol=SYMBOL, as_of_utc=as_of)
        assert before == after

    def test_truncated_dataset_matches_as_of_filtered_dataset(self, db_session):
        day = dt.date(2026, 8, 1)
        _seed_sessions(db_session, SYMBOL, SID, n_sessions=1)
        as_of_early = _as_of(day, 9, 30)
        as_of_late = _as_of(day, 14, 0)
        early = compute_snapshot(db_session, security_id=SID, symbol=SYMBOL, as_of_utc=as_of_early)
        late = compute_snapshot(db_session, security_id=SID, symbol=SYMBOL, as_of_utc=as_of_late)
        assert early.n_bars_today < late.n_bars_today
        assert early.latest_bar_utc <= as_of_early
        assert late.latest_bar_utc <= as_of_late


class TestLiveVsBatchEquivalence:
    def test_matches_phase_r_vectorized_formulas_at_several_timestamps(self, db_session):
        day = dt.date(2026, 8, 1)
        bars = _seed_sessions(db_session, SYMBOL, SID, n_sessions=2, seed=5)
        df = pd.DataFrame([
            {"bar_start_utc": b.bar_start_utc, "trading_date": b.trading_date, "open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume}
            for b in bars
        ]).sort_values("bar_start_utc").reset_index(drop=True)
        batch = compute_for_symbol(df)

        for i in [30, 100, 250, 359]:
            as_of = batch.loc[i, "bar_start_utc"]
            live = compute_snapshot(db_session, security_id=SID, symbol=SYMBOL, as_of_utc=as_of)
            row = batch.loc[i]
            assert live.return_1m == pytest.approx(row["return_1m"], abs=1e-9) if not pd.isna(row["return_1m"]) else live.return_1m is None
            assert live.return_5m == pytest.approx(row["return_5m"], abs=1e-9) if not pd.isna(row["return_5m"]) else live.return_5m is None
            assert live.return_15m == pytest.approx(row["return_15m"], abs=1e-9) if not pd.isna(row["return_15m"]) else live.return_15m is None
            if not pd.isna(row["relative_volume_20"]):
                assert live.relative_volume_20 == pytest.approx(row["relative_volume_20"], rel=1e-6)
            if not pd.isna(row["session_vwap"]):
                assert live.session_vwap == pytest.approx(row["session_vwap"], rel=1e-6)
            if not pd.isna(row["distance_from_vwap"]):
                assert live.distance_from_vwap == pytest.approx(row["distance_from_vwap"], rel=1e-6)


class TestDataQuality:
    def test_no_data_for_a_symbol_with_zero_rows(self, db_session):
        snap = compute_snapshot(db_session, security_id=999999, symbol="GHOST", as_of_utc=_as_of(dt.date(2026, 8, 1), 10, 0))
        assert snap.data_quality == NO_DATA
        assert snap.price is None and snap.return_1m is None

    def test_cresto_style_unavailable_symbol(self, db_session):
        snap = compute_snapshot(db_session, security_id=19097, symbol="CRESTO", as_of_utc=_as_of(dt.date(2026, 8, 1), 10, 0))
        assert snap.data_quality == NO_DATA

    def test_011nsetest_style_no_history_symbol(self, db_session):
        snap = compute_snapshot(db_session, security_id=14747, symbol="011NSETEST", as_of_utc=_as_of(dt.date(2026, 8, 1), 10, 0))
        assert snap.data_quality == NO_DATA

    def test_insufficient_history_with_fewer_than_20_bars(self, db_session):
        day = dt.date(2026, 8, 1)
        upsert_intraday_bars(db_session, _session_bars(SYMBOL, SID, day, n=10))
        db_session.commit()
        snap = compute_snapshot(db_session, security_id=SID, symbol=SYMBOL, as_of_utc=_as_of(day, 9, 24))
        assert snap.data_quality == INSUFFICIENT_HISTORY
        assert snap.return_1m is not None

    def test_stale_when_latest_bar_is_old_relative_to_as_of(self, db_session):
        day = dt.date(2026, 8, 1)
        _seed_sessions(db_session, SYMBOL, SID, n_sessions=1)
        latest_bar_time = _as_of(day, 15, 14)
        far_future_as_of = latest_bar_time + dt.timedelta(minutes=30)
        snap = compute_snapshot(db_session, security_id=SID, symbol=SYMBOL, as_of_utc=far_future_as_of)
        assert snap.data_quality == STALE

    def test_ok_with_full_history_and_fresh_bar(self, db_session):
        day = dt.date(2026, 8, 1)
        _seed_sessions(db_session, SYMBOL, SID, n_sessions=1)
        snap = compute_snapshot(db_session, security_id=SID, symbol=SYMBOL, as_of_utc=_as_of(day, 10, 30))
        assert snap.data_quality == OK


class TestRobustness:
    def test_gap_in_the_middle_of_a_session_does_not_crash(self, db_session):
        day = dt.date(2026, 8, 1)
        bars = _session_bars(SYMBOL, SID, day)
        kept = [b for b in bars if not (dt.time(11, 0) <= b.bar_start_ist.time() < dt.time(11, 30))]
        upsert_intraday_bars(db_session, kept)
        db_session.commit()
        snap = compute_snapshot(db_session, security_id=SID, symbol=SYMBOL, as_of_utc=_as_of(day, 14, 0))
        assert snap.data_quality in (OK, INSUFFICIENT_HISTORY)

    def test_partial_universe_coverage_in_compute_universe_snapshot(self, db_session):
        day = dt.date(2026, 8, 1)
        _seed_sessions(db_session, SYMBOL, SID, n_sessions=1)
        securities = [(SID, SYMBOL), (999999, "GHOST")]
        snapshots = compute_universe_snapshot(db_session, securities=securities, as_of_utc=_as_of(day, 10, 0))
        by_symbol = {s.symbol: s for s in snapshots}
        assert by_symbol["GHOST"].data_quality == NO_DATA
        assert by_symbol[SYMBOL].data_quality in (OK, INSUFFICIENT_HISTORY)

    def test_as_of_exactly_at_session_open(self, db_session):
        day = dt.date(2026, 8, 1)
        _seed_sessions(db_session, SYMBOL, SID, n_sessions=1)
        snap = compute_snapshot(db_session, security_id=SID, symbol=SYMBOL, as_of_utc=_as_of(day, 9, 15))
        assert snap.price is not None
        assert snap.n_bars_today == 1

    def test_zero_volume_bar_does_not_crash_relative_volume(self, db_session):
        day = dt.date(2026, 8, 1)
        bars = _session_bars(SYMBOL, SID, day, volume=0.0)
        upsert_intraday_bars(db_session, bars)
        db_session.commit()
        snap = compute_snapshot(db_session, security_id=SID, symbol=SYMBOL, as_of_utc=_as_of(day, 10, 0))
        assert snap.price is not None

    def test_calling_compute_snapshot_twice_is_idempotent(self, db_session):
        day = dt.date(2026, 8, 1)
        _seed_sessions(db_session, SYMBOL, SID, n_sessions=1)
        as_of = _as_of(day, 10, 0)
        first = compute_snapshot(db_session, security_id=SID, symbol=SYMBOL, as_of_utc=as_of)
        second = compute_snapshot(db_session, security_id=SID, symbol=SYMBOL, as_of_utc=as_of)
        assert first == second


class TestLiquidityTiers:
    def test_tertile_assignment_matches_relative_activity(self, db_session):
        day0 = dt.date(2026, 8, 1)
        _seed_sessions(db_session, "LOWLIQ", 101, n_sessions=20, start_day=day0, volume=100.0)
        _seed_sessions(db_session, "MIDLIQ", 102, n_sessions=20, start_day=day0, volume=1000.0)
        _seed_sessions(db_session, "HIGHLIQ", 103, n_sessions=20, start_day=day0, volume=10000.0)
        as_of_date = day0 + dt.timedelta(days=20)
        tiers = compute_liquidity_tiers(db_session, security_ids=[101, 102, 103], as_of_date=as_of_date)
        assert tiers[101] == "low_liquidity"
        assert tiers[102] == "mid_liquidity"
        assert tiers[103] == "high_liquidity"

    def test_defined_with_exactly_20_trailing_sessions_not_off_by_one(self, db_session):
        day0 = dt.date(2026, 8, 1)
        _seed_sessions(db_session, SYMBOL, SID, n_sessions=20, start_day=day0, volume=500.0)
        as_of_date = day0 + dt.timedelta(days=20)
        tiers = compute_liquidity_tiers(db_session, security_ids=[SID], as_of_date=as_of_date)
        assert tiers[SID] is not None

    def test_none_with_fewer_than_20_trailing_sessions(self, db_session):
        day0 = dt.date(2026, 8, 1)
        _seed_sessions(db_session, SYMBOL, SID, n_sessions=5, start_day=day0)
        tiers = compute_liquidity_tiers(db_session, security_ids=[SID], as_of_date=day0 + dt.timedelta(days=5))
        assert tiers[SID] is None

    def test_same_day_volume_never_included(self, db_session):
        day0 = dt.date(2026, 8, 1)
        _seed_sessions(db_session, SYMBOL, SID, n_sessions=20, start_day=day0, volume=100.0)
        as_of_date = day0 + dt.timedelta(days=20)
        before = compute_liquidity_tiers(db_session, security_ids=[SID], as_of_date=as_of_date)
        upsert_intraday_bars(db_session, _session_bars(SYMBOL, SID, as_of_date, volume=999999999.0))
        db_session.commit()
        after = compute_liquidity_tiers(db_session, security_ids=[SID], as_of_date=as_of_date)
        assert before == after


class TestCrossSectionalRank:
    def test_rank_is_none_when_symbol_has_no_return_15m(self, db_session):
        day = dt.date(2026, 8, 1)
        _seed_sessions(db_session, SYMBOL, SID, n_sessions=1)
        snapshots = compute_universe_snapshot(db_session, securities=[(SID, SYMBOL), (999999, "GHOST")], as_of_utc=_as_of(day, 10, 0))
        assert cross_sectional_rank_15m(snapshots, "GHOST") is None

    def test_rank_excludes_the_symbol_itself(self):
        from backend.app.equity_live.live_feature_service import FeatureSnapshot
        now = dt.datetime(2026, 8, 1, 10, 0, tzinfo=UTC)

        def snap(sym, r15m):
            return FeatureSnapshot(
                security_id=1, symbol=sym, as_of_utc=now, trading_date=None, price=100.0, return_1m=None,
                return_5m=None, return_15m=r15m, relative_volume_20=None, realized_volatility_20=None,
                volatility_expansion_ratio=None, distance_from_vwap=None, session_vwap=None, opening_range_high=None,
                opening_range_low=None, overnight_gap=None, recent_breakout_up=None, recent_breakout_down=None,
                distance_from_session_open=None, minutes_since_open=None, relative_strength=None,
                n_bars_today=0, n_bars_total=0, latest_bar_utc=None, data_quality="OK",
            )
        snapshots = [snap("A", 0.01), snap("B", 0.02), snap("C", 0.03)]
        assert cross_sectional_rank_15m(snapshots, "C") == pytest.approx(1.0)
