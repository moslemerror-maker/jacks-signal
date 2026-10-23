from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from backend.app.data.sources.dhan_optionchain_client import DhanOptionChainClient, DhanOptionChainError, expiry_dates_from_response
from backend.app.data.sources.provider_credentials import DhanCredentials

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


SUCCESS_EXPIRY_LIST = {"status": "success", "data": ["2026-09-23", "2026-09-30"]}
SUCCESS_EXPIRY_LIST_NESTED = {"status": "success", "data": {"data": ["2026-09-23", "2026-09-30"]}}
SUCCESS_CHAIN = {"status": "success", "data": {"data": {"last_price": 24500.0, "oc": {}}}}


class TestFetchChainNearestExpiry:
    def test_success_uses_nearest_expiry(self):
        sdk = FakeSdk(SUCCESS_EXPIRY_LIST, SUCCESS_CHAIN)
        client = DhanOptionChainClient(CREDS, sdk_client=sdk)
        result = client.fetch_chain(13, "IDX_I")
        assert result.expiry_used == "2026-09-23"
        assert result.expiry_selection == "NEAREST_FROM_EXPIRY_LIST"
        assert result.option_chain_response == SUCCESS_CHAIN
        assert sdk.calls == [("expiry_list", 13, "IDX_I"), ("option_chain", 13, "IDX_I", "2026-09-23")]

    def test_real_live_nested_expiry_list_shape_is_handled_correctly(self):
        sdk = FakeSdk(SUCCESS_EXPIRY_LIST_NESTED, SUCCESS_CHAIN)
        client = DhanOptionChainClient(CREDS, sdk_client=sdk)
        result = client.fetch_chain(13, "IDX_I")
        assert result.expiry_used == "2026-09-23"
        assert sdk.calls[1] == ("option_chain", 13, "IDX_I", "2026-09-23")

    def test_expiry_dates_tolerates_both_nesting_depths(self):
        assert expiry_dates_from_response(SUCCESS_EXPIRY_LIST) == ["2026-09-23", "2026-09-30"]
        assert expiry_dates_from_response(SUCCESS_EXPIRY_LIST_NESTED) == ["2026-09-23", "2026-09-30"]

    def test_expiry_dates_unusable_shape_yields_no_dates_not_a_crash(self):
        assert expiry_dates_from_response({"status": "success", "data": "not-a-list-or-dict-of-dates"}) == []
        assert expiry_dates_from_response({"status": "success", "data": {"unexpected_key": [1, 2]}}) == []
        assert expiry_dates_from_response({}) == []

    def test_explicit_expiry_used_when_valid(self):
        sdk = FakeSdk(SUCCESS_EXPIRY_LIST, SUCCESS_CHAIN)
        client = DhanOptionChainClient(CREDS, sdk_client=sdk)
        result = client.fetch_chain(13, "IDX_I", expiry="2026-09-30")
        assert result.expiry_used == "2026-09-30" and result.expiry_selection == "EXPLICIT"

    def test_explicit_expiry_not_in_expiry_list_is_refused(self):
        sdk = FakeSdk(SUCCESS_EXPIRY_LIST, SUCCESS_CHAIN)
        client = DhanOptionChainClient(CREDS, sdk_client=sdk)
        with pytest.raises(DhanOptionChainError, match="not in expiry_list"):
            client.fetch_chain(13, "IDX_I", expiry="2026-10-15")

    def test_expiry_list_failure_status_raises(self):
        sdk = FakeSdk({"status": "error", "errorCode": "DH-905"})
        client = DhanOptionChainClient(CREDS, sdk_client=sdk)
        with pytest.raises(DhanOptionChainError, match="expiry_list"):
            client.fetch_chain(13, "IDX_I")

    def test_expiry_list_empty_dates_raises(self):
        sdk = FakeSdk({"status": "success", "data": []})
        client = DhanOptionChainClient(CREDS, sdk_client=sdk)
        with pytest.raises(DhanOptionChainError, match="no dates"):
            client.fetch_chain(13, "IDX_I")

    def test_option_chain_failure_status_raises(self):
        sdk = FakeSdk(SUCCESS_EXPIRY_LIST, {"status": "error", "errorCode": "DH-999"})
        client = DhanOptionChainClient(CREDS, sdk_client=sdk)
        with pytest.raises(DhanOptionChainError, match="option_chain"):
            client.fetch_chain(13, "IDX_I")

    def test_none_expiry_list_response_raises_not_crashes(self):
        sdk = FakeSdk(None)
        client = DhanOptionChainClient(CREDS, sdk_client=sdk)
        with pytest.raises(DhanOptionChainError):
            client.fetch_chain(13, "IDX_I")

    def test_never_falls_back_to_a_different_expiry_on_failure(self):
        sdk = FakeSdk(SUCCESS_EXPIRY_LIST, {"status": "error"})
        client = DhanOptionChainClient(CREDS, sdk_client=sdk)
        with pytest.raises(DhanOptionChainError):
            client.fetch_chain(13, "IDX_I")
        assert sdk.calls == [("expiry_list", 13, "IDX_I"), ("option_chain", 13, "IDX_I", "2026-09-23")]


class TestRawPassthroughMethods:
    def test_expiry_list_passthrough(self):
        sdk = FakeSdk(SUCCESS_EXPIRY_LIST)
        client = DhanOptionChainClient(CREDS, sdk_client=sdk)
        assert client.expiry_list(13, "IDX_I") == SUCCESS_EXPIRY_LIST

    def test_option_chain_passthrough(self):
        sdk = FakeSdk(SUCCESS_EXPIRY_LIST, SUCCESS_CHAIN)
        client = DhanOptionChainClient(CREDS, sdk_client=sdk)
        assert client.option_chain(13, "IDX_I", "2026-09-23") == SUCCESS_CHAIN
