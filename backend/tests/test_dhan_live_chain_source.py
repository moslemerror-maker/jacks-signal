from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any

import pytest

from backend.app.data.sources.dhan_live_chain_source import DhanLiveChainSource
from backend.app.data.sources.dhan_optionchain_client import DhanOptionChainClient, DhanOptionChainError
from backend.app.data.sources.provider_credentials import DhanCredentials
from backend.app.options.availability import session_close_utc
from backend.app.options.expiry_eligibility import NoEligibleExpiryError

D = dt.date
UTC = dt.timezone.utc
CREDS = DhanCredentials(client_id="123", access_token="tok")


@dataclasses.dataclass
class FakeSdk:
    expiry_list_response: dict[str, Any]
    option_chain_response: dict[str, Any] | None = None
    calls: list[tuple] = dataclasses.field(default_factory=list)

    def expiry_list(self, security_id, segment):
        self.calls.append(("expiry_list", security_id, segment))
        return self.expiry_list_response

    def option_chain(self, security_id, segment, expiry):
        self.calls.append(("option_chain", security_id, segment, expiry))
        return self.option_chain_response


def chain_payload(strike="24500.000000") -> dict:
    return {"status": "success", "data": {"data": {"last_price": 24500.0, "oc": {strike: {
        "ce": {"last_price": 140.0, "top_bid_price": 138.0, "top_ask_price": 142.0, "volume": 100, "oi": 500, "implied_volatility": 14.5, "greeks": {"delta": 0.5, "theta": -1, "gamma": 0.01, "vega": 1}},
        "pe": {"last_price": 135.0, "top_bid_price": 133.0, "top_ask_price": 137.0, "volume": 100, "oi": 500, "implied_volatility": 14.0, "greeks": {"delta": -0.5, "theta": -1, "gamma": 0.01, "vega": 1}},
    }}}}}


class TestDhanLiveChainSource:
    def test_uses_eligible_expiry_not_naive_nearest(self):
        today = D(2026, 9, 22)
        next_week = D(2026, 9, 29)
        after_close = session_close_utc(today) + dt.timedelta(minutes=10)
        sdk = FakeSdk({"status": "success", "data": {"data": [str(today), str(next_week)]}}, chain_payload())
        source = DhanLiveChainSource(client=DhanOptionChainClient(CREDS, sdk_client=sdk), security_id=13, segment="IDX_I", underlying="NIFTY")
        snapshot = source.fetch_snapshot(as_of=after_close)
        assert sdk.calls == [("expiry_list", 13, "IDX_I"), ("option_chain", 13, "IDX_I", str(next_week))]
        assert all(q.contract.expiry == next_week for q in snapshot.quotes)

    def test_exactly_two_live_calls_when_expiry_is_already_eligible(self):
        today = D(2026, 9, 22)
        before_close = session_close_utc(today) - dt.timedelta(minutes=5)
        sdk = FakeSdk({"status": "success", "data": {"data": [str(today)]}}, chain_payload())
        source = DhanLiveChainSource(client=DhanOptionChainClient(CREDS, sdk_client=sdk), security_id=13, segment="IDX_I", underlying="NIFTY")
        source.fetch_snapshot(as_of=before_close)
        assert len(sdk.calls) == 2

    def test_snapshot_fields_populated(self):
        today = D(2026, 9, 22)
        before_close = session_close_utc(today) - dt.timedelta(minutes=5)
        sdk = FakeSdk({"status": "success", "data": {"data": [str(today)]}}, chain_payload())
        source = DhanLiveChainSource(client=DhanOptionChainClient(CREDS, sdk_client=sdk), security_id=13, segment="IDX_I", underlying="NIFTY")
        snapshot = source.fetch_snapshot(as_of=before_close)
        assert snapshot.source == "dhan" and snapshot.underlying == "NIFTY"
        assert snapshot.observed_at == before_close
        assert len(snapshot.quotes) == 2
        assert snapshot.raw_response_sha256 is not None and len(snapshot.raw_response_sha256) == 64
        assert snapshot.expiry_list_dates == (today,)

    def test_no_eligible_expiry_propagates(self):
        past = D(2026, 1, 1)
        sdk = FakeSdk({"status": "success", "data": {"data": [str(past)]}})
        source = DhanLiveChainSource(client=DhanOptionChainClient(CREDS, sdk_client=sdk), security_id=13, segment="IDX_I", underlying="NIFTY")
        with pytest.raises(NoEligibleExpiryError):
            source.fetch_snapshot(as_of=dt.datetime(2026, 9, 22, 10, 0, tzinfo=UTC))

    def test_expiry_list_failure_raises(self):
        sdk = FakeSdk({"status": "error"})
        source = DhanLiveChainSource(client=DhanOptionChainClient(CREDS, sdk_client=sdk), security_id=13, segment="IDX_I", underlying="NIFTY")
        with pytest.raises(DhanOptionChainError, match="expiry_list"):
            source.fetch_snapshot(as_of=dt.datetime(2026, 9, 22, 10, 0, tzinfo=UTC))

    def test_option_chain_failure_raises(self):
        today = D(2026, 9, 22)
        before_close = session_close_utc(today) - dt.timedelta(minutes=5)
        sdk = FakeSdk({"status": "success", "data": {"data": [str(today)]}}, {"status": "error"})
        source = DhanLiveChainSource(client=DhanOptionChainClient(CREDS, sdk_client=sdk), security_id=13, segment="IDX_I", underlying="NIFTY")
        with pytest.raises(DhanOptionChainError, match="option_chain"):
            source.fetch_snapshot(as_of=before_close)

    def test_provenance_class_is_production_by_default(self):
        today = D(2026, 9, 22)
        before_close = session_close_utc(today) - dt.timedelta(minutes=5)
        sdk = FakeSdk({"status": "success", "data": {"data": [str(today)]}}, chain_payload())
        source = DhanLiveChainSource(client=DhanOptionChainClient(CREDS, sdk_client=sdk), security_id=13, segment="IDX_I", underlying="NIFTY")
        snapshot = source.fetch_snapshot(as_of=before_close)
        assert all(q.provenance_class == "PRODUCTION_RESEARCH_DATASET" for q in snapshot.quotes)
