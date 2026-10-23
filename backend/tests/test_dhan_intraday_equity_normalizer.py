from __future__ import annotations

import datetime as dt
import json

import pytest
from pydantic import ValidationError

from backend.app.data.canonical.intraday_equity_bar import IST, CanonicalIntradayEquityBar, IntradayProvider
from backend.app.data.normalizers import dhan_intraday_equity as norm
from backend.app.data.raw.store import RawStore
from backend.app.data.sources import dhan_equity_historical as deh

UTC = dt.timezone.utc
D = dt.date
SEC_ID = 2885
SYMBOL = "RELIANCE"
FROM, TO = D(2026, 8, 26), D(2026, 8, 26)
KEY = deh.source_key_for(SEC_ID, FROM, TO)

E0, E1, E2 = 1787715900, 1787715960, 1787716020


def _put(store: RawStore, data: dict, *, source_key: str = KEY, now=dt.datetime(2026, 9, 25, 12, 0, tzinfo=UTC)):
    return store.put(
        provider="dhan", dataset=deh.DATASET, source_key=source_key, content=json.dumps(data).encode(), source_url=deh._ENDPOINT,
        http_status=200, response_headers={}, fetcher_version="test/1", media_type="application/json", license_status="UNCLEAR", fetched_at=now,
    )


def three_clean_bars() -> dict:
    return {
        "open": [100.0, 101.0, 102.0], "high": [101.0, 102.0, 103.0], "low": [99.5, 100.5, 101.5],
        "close": [100.8, 101.8, 102.8], "volume": [1000, 1100, 1200], "timestamp": [E0, E1, E2],
    }


@pytest.fixture
def store(tmp_path):
    return RawStore(tmp_path / "raw")


class TestBasicNormalization:
    def test_utc_epoch_to_timezone_aware_timestamp(self, store):
        _put(store, three_clean_bars())
        bars, rejected = norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)
        assert rejected == []
        b = bars[0]
        assert b.bar_start_utc == dt.datetime(2026, 8, 26, 3, 45, 0, tzinfo=UTC)
        assert b.bar_start_utc.tzinfo is not None and b.bar_start_ist.tzinfo is not None

    def test_timestamp_boundary_semantics_first_bar_is_session_open(self, store):
        _put(store, three_clean_bars())
        bars, _ = norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)
        assert bars[0].bar_start_ist == dt.datetime(2026, 8, 26, 9, 15, 0, tzinfo=IST)
        assert bars[0].trading_date == D(2026, 8, 26)

    def test_ohlc_preserved_exactly(self, store):
        raw = three_clean_bars()
        _put(store, raw)
        bars, _ = norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)
        for i, b in enumerate(bars):
            assert (b.open, b.high, b.low, b.close) == (raw["open"][i], raw["high"][i], raw["low"][i], raw["close"][i])

    def test_volume_preserved_exactly(self, store):
        raw = three_clean_bars()
        _put(store, raw)
        bars, _ = norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)
        assert [b.volume for b in bars] == raw["volume"]

    def test_instrument_identity(self, store):
        _put(store, three_clean_bars())
        bars, _ = norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)
        b = bars[0]
        assert (b.provider, b.exchange_segment, b.security_id, b.trading_symbol) == (IntradayProvider.DHAN, "NSE_EQ", SEC_ID, SYMBOL)

    def test_provider_provenance(self, store):
        put_result = _put(store, three_clean_bars())
        bars, _ = norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)
        b = bars[0]
        assert b.source_provider == "dhan" and b.source_dataset == deh.DATASET and b.source_key == KEY
        assert b.source_sha256 == put_result.meta.sha256 and b.normalizer_version == norm.NORMALIZER_VERSION


class TestHardRejection:
    def test_duplicate_timestamp_rejection(self, store):
        raw = three_clean_bars()
        raw["timestamp"][2] = raw["timestamp"][1]
        _put(store, raw)
        bars, rejected = norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)
        assert len(bars) == 1 and len(rejected) == 2
        assert {r.row_index for r in rejected} == {1, 2}
        assert all("duplicate" in r.reason for r in rejected)

    def test_out_of_order_timestamp_rejection(self, store):
        raw = three_clean_bars()
        raw["timestamp"] = [E0, E2, E1]
        _put(store, raw)
        bars, rejected = norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)
        assert any(r.row_index == 2 and "out of order" in r.reason for r in rejected)

    @pytest.mark.parametrize("field, bad_value, why", [
        ("high", 90.0, "high"),
        ("low", 110.0, "low"),
    ])
    def test_invalid_ohlc_rejection(self, store, field, bad_value, why):
        raw = three_clean_bars()
        raw[field][0] = bad_value
        _put(store, raw)
        bars, rejected = norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)
        assert len(bars) == 2 and len(rejected) == 1 and rejected[0].row_index == 0

    def test_non_positive_price_rejection(self, store):
        raw = three_clean_bars()
        raw["open"][0] = 0.0
        raw["high"][0] = 0.0
        raw["low"][0] = 0.0
        raw["close"][0] = 0.0
        _put(store, raw)
        bars, rejected = norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)
        assert len(bars) == 2 and rejected[0].row_index == 0 and "non-positive" in rejected[0].reason

    def test_negative_volume_rejection(self, store):
        raw = three_clean_bars()
        raw["volume"][0] = -5
        _put(store, raw)
        bars, rejected = norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)
        assert len(bars) == 2 and rejected[0].row_index == 0 and "negative volume" in rejected[0].reason

    def test_null_ohlc_value_is_rejected_not_silently_dropped(self, store):
        raw = three_clean_bars()
        raw["open"][0] = None
        _put(store, raw)
        bars, rejected = norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)
        assert len(bars) == 2 and rejected[0].row_index == 0

    def test_null_volume_is_preserved_as_missing_not_rejected(self, store):
        raw = three_clean_bars()
        raw["volume"][0] = None
        _put(store, raw)
        bars, rejected = norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)
        assert rejected == [] and len(bars) == 3
        assert next(b for b in bars if b.raw_timestamp_epoch == E0).volume is None


class TestSessionBoundaryReports:
    def test_complete_session_reports_360(self, store):
        n = 360
        base = E0
        raw = {
            "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n, "close": [100.5] * n,
            "volume": [1000] * n, "timestamp": [base + i * 60 for i in range(n)],
        }
        _put(store, raw)
        bars, rejected = norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)
        assert rejected == []
        reports = norm.session_boundary_reports(bars)
        assert len(reports) == 1
        r = reports[0]
        assert r.bar_count == 360 and r.is_complete_360 is True and r.missing_minute_marks_ist == ()
        assert (r.first_bar_start_ist, r.last_bar_start_ist) == (dt.time(9, 15), dt.time(15, 14))
        assert r.session_shape == norm.SessionShape.MATCHES_CURRENT_GRID

    def test_missing_minutes_are_reported_not_rejected(self, store):
        raw = three_clean_bars()
        _put(store, raw)
        bars, rejected = norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)
        assert rejected == []
        reports = norm.session_boundary_reports(bars)
        r = reports[0]
        assert r.bar_count == 3 and r.is_complete_360 is False
        assert "09:18" in r.missing_minute_marks_ist and "15:14" in r.missing_minute_marks_ist
        assert "09:15" not in r.missing_minute_marks_ist
        assert r.session_shape == norm.SessionShape.UNRECOGNIZED_SHAPE

    def test_known_historical_variant_375_bar_session_is_accepted_not_flagged_as_incomplete_failure(self, store):
        n = 375
        variant_date = dt.date(2026, 6, 29)
        base_ist = dt.datetime(2026, 6, 29, 9, 15, tzinfo=IST)
        base_epoch = int(base_ist.astimezone(UTC).timestamp())
        raw = {
            "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n, "close": [100.5] * n,
            "volume": [1000] * n, "timestamp": [base_epoch + i * 60 for i in range(n)],
        }
        key = deh.source_key_for(SEC_ID, variant_date, variant_date)
        _put(store, raw, source_key=key)
        bars, rejected = norm.normalize_dhan_intraday_artifact(store, source_key=key, trading_symbol=SYMBOL)
        assert rejected == []
        reports = norm.session_boundary_reports(bars)
        r = reports[0]
        assert r.bar_count == 375
        assert r.session_shape == norm.SessionShape.MATCHES_KNOWN_HISTORICAL_VARIANT
        assert r.is_complete_360 is False


class TestReconciliation:
    def test_clean_artifact_reconciles_with_no_discrepancies(self, store):
        raw = three_clean_bars()
        _put(store, raw)
        bars, rejected = norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)
        result = norm.reconcile(store, source_key=KEY, bars=bars, rejected=rejected)
        assert result.discrepancies == ()
        assert result.rows_accounted_for and result.earliest_timestamp_matches and result.latest_timestamp_matches and result.ohlc_matches
        assert result.raw_row_count == 3 and result.normalized_row_count == 3 and result.rejected_row_count == 0

    def test_reconciliation_accounts_for_rejected_rows_too(self, store):
        raw = three_clean_bars()
        raw["timestamp"][2] = raw["timestamp"][1]
        _put(store, raw)
        bars, rejected = norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)
        result = norm.reconcile(store, source_key=KEY, bars=bars, rejected=rejected)
        assert result.rows_accounted_for is True
        assert result.raw_row_count == 3 and result.normalized_row_count == 1 and result.rejected_row_count == 2


class TestMissingValuesAndNoLookahead:
    def test_no_future_looking_fields_on_the_canonical_model(self):
        forbidden_substrings = ("future", "next_", "forward", "label", "return", "target")
        for name in CanonicalIntradayEquityBar.model_fields:
            assert not any(s in name.lower() for s in forbidden_substrings), name


class TestIdempotency:
    def test_normalizing_twice_produces_identical_output(self, store):
        _put(store, three_clean_bars())
        bars1, rejected1 = norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)
        bars2, rejected2 = norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)
        assert bars1 == bars2 and rejected1 == rejected2
        assert [b.model_dump() for b in bars1] == [b.model_dump() for b in bars2]


class TestNoNetworkCalls:
    def test_normalize_and_reconcile_make_no_dhan_api_calls(self, store, monkeypatch):
        import requests

        def _boom(*a, **k):
            raise AssertionError("normalization must never make a network call")

        monkeypatch.setattr(requests.sessions.Session, "request", _boom)
        _put(store, three_clean_bars())
        bars, rejected = norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)
        norm.reconcile(store, source_key=KEY, bars=bars, rejected=rejected)
        norm.session_boundary_reports(bars)
        assert len(bars) == 3


class TestMalformedRawResponse:
    def test_malformed_raw_artifact_raises_before_normalizing_anything(self, store):
        store.put(provider="dhan", dataset=deh.DATASET, source_key=KEY, content=b"<html>not json</html>", source_url=deh._ENDPOINT,
                  http_status=200, response_headers={}, fetcher_version="test/1", license_status="UNCLEAR")
        from backend.app.data.sources.errors import InvalidContentError
        with pytest.raises(InvalidContentError):
            norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)


class TestNormalizationVersion:
    def test_version_string_is_stable_and_present_on_every_bar(self, store):
        n = 5
        raw = {"open": [1.0]*n, "high": [1.0]*n, "low": [1.0]*n, "close": [1.0]*n, "volume": [1]*n, "timestamp": [E0 + i*60 for i in range(n)]}
        _put(store, raw)
        bars, _ = norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)
        assert all(b.normalizer_version == "dhan_intraday_equity/1" for b in bars)


class TestMultipleInstrumentsIsolated:
    def test_two_symbols_do_not_cross_contaminate(self, store):
        key_a = deh.source_key_for(2885, FROM, TO)
        key_b = deh.source_key_for(11536, FROM, TO)
        _put(store, three_clean_bars(), source_key=key_a)
        raw_b = {
            "open": [500.0, 501.0, 502.0], "high": [501.0, 502.0, 503.0], "low": [499.5, 500.5, 501.5],
            "close": [500.8, 501.8, 502.8], "volume": [2000, 2100, 2200], "timestamp": [E0, E1, E2],
        }
        _put(store, raw_b, source_key=key_b)

        bars_a, _ = norm.normalize_dhan_intraday_artifact(store, source_key=key_a, trading_symbol="RELIANCE")
        bars_b, _ = norm.normalize_dhan_intraday_artifact(store, source_key=key_b, trading_symbol="TCS")

        assert all(b.security_id == 2885 and b.trading_symbol == "RELIANCE" for b in bars_a)
        assert all(b.security_id == 11536 and b.trading_symbol == "TCS" for b in bars_b)
        assert [b.open for b in bars_a] == [100.0, 101.0, 102.0]
        assert [b.open for b in bars_b] == [500.0, 501.0, 502.0]


class TestCanonicalOrdering:
    def test_bars_are_returned_in_ascending_bar_start_order(self, store):
        n = 20
        raw = {"open": [1.0]*n, "high": [1.0]*n, "low": [1.0]*n, "close": [1.0]*n, "volume": [1]*n, "timestamp": [E0 + i*60 for i in range(n)]}
        _put(store, raw)
        bars, _ = norm.normalize_dhan_intraday_artifact(store, source_key=KEY, trading_symbol=SYMBOL)
        starts = [b.bar_start_utc for b in bars]
        assert starts == sorted(starts)
