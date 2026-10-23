from __future__ import annotations

import sys

import pytest

from backend.app.data.sources import breeze_client as bc
from backend.app.data.sources import dhan_client as dc
from backend.app.data.sources import provider_credentials as pc
from backend.app.data.sources import upstox_client as uc


@pytest.fixture(autouse=True)
def no_real_env(monkeypatch):
    monkeypatch.setattr(pc, "load_dotenv", lambda: None)
    for k in ("UPSTOX_ACCESS_TOKEN", "DHAN_CLIENT_ID", "DHAN_ACCESS_TOKEN", "BREEZE_API_KEY", "BREEZE_API_SECRET", "BREEZE_SESSION_TOKEN"):
        monkeypatch.delenv(k, raising=False)


class TestUpstoxCredentials:
    def test_missing(self):
        with pytest.raises(pc.MissingCredentialsError) as exc:
            pc.UpstoxCredentials.from_env()
        assert exc.value.provider == "upstox" and exc.value.missing == ["UPSTOX_ACCESS_TOKEN"]
        assert "UPSTOX_ACCESS_TOKEN" in str(exc.value) and ".env.example" in str(exc.value)

    def test_present(self, monkeypatch):
        monkeypatch.setenv("UPSTOX_ACCESS_TOKEN", "tok123")
        c = pc.UpstoxCredentials.from_env()
        assert c.access_token == "tok123" and c.auth_header() == {"Authorization": "Bearer tok123"}

    def test_whitespace_only_counts_as_missing(self, monkeypatch):
        monkeypatch.setenv("UPSTOX_ACCESS_TOKEN", "   ")
        with pytest.raises(pc.MissingCredentialsError):
            pc.UpstoxCredentials.from_env()


class TestDhanCredentials:
    def test_missing_reports_both(self):
        with pytest.raises(pc.MissingCredentialsError) as exc:
            pc.DhanCredentials.from_env()
        assert set(exc.value.missing) == {"DHAN_CLIENT_ID", "DHAN_ACCESS_TOKEN"}

    def test_missing_reports_only_the_absent_one(self, monkeypatch):
        monkeypatch.setenv("DHAN_CLIENT_ID", "cid")
        with pytest.raises(pc.MissingCredentialsError) as exc:
            pc.DhanCredentials.from_env()
        assert exc.value.missing == ["DHAN_ACCESS_TOKEN"]

    def test_present(self, monkeypatch):
        monkeypatch.setenv("DHAN_CLIENT_ID", "cid")
        monkeypatch.setenv("DHAN_ACCESS_TOKEN", "tok")
        c = pc.DhanCredentials.from_env()
        assert c.auth_headers() == {"access-token": "tok", "dhanClientId": "cid"}


class TestBreezeCredentials:
    def test_missing_all_three(self):
        with pytest.raises(pc.MissingCredentialsError) as exc:
            pc.BreezeCredentials.from_env()
        assert set(exc.value.missing) == {"BREEZE_API_KEY", "BREEZE_API_SECRET", "BREEZE_SESSION_TOKEN"}

    def test_present(self, monkeypatch):
        monkeypatch.setenv("BREEZE_API_KEY", "k")
        monkeypatch.setenv("BREEZE_API_SECRET", "s")
        monkeypatch.setenv("BREEZE_SESSION_TOKEN", "t")
        c = pc.BreezeCredentials.from_env()
        assert (c.api_key, c.api_secret, c.session_token) == ("k", "s", "t")


class TestRedact:
    def test_redacts_every_occurrence_of_each_secret(self):
        out = pc.redact("token=abc123 and again abc123, other=xyz", "abc123", "xyz")
        assert "abc123" not in out and "xyz" not in out and out.count("<redacted>") == 3

    def test_empty_secrets_are_ignored_not_wildcards(self):
        assert pc.redact("hello world", "", "") == "hello world"


class FakeGetSession:
    def __init__(self, response):
        self.response, self.calls = response, []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, headers, timeout))
        return self.response


class FakeResponse:
    def __init__(self, status=200, payload=None, content=b""):
        self.status_code = status
        self._payload = payload
        self.content = content or (b"{}" if payload is None else __import__("json").dumps(payload).encode())
        self.headers = {"Content-Type": "application/json"}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class TestUpstoxUrls:
    def test_expiries_url_encodes_the_instrument_key(self):
        assert uc.expiries_url("NSE_INDEX|Nifty 50") == "https://api.upstox.com/v2/expired-instruments/expiries?instrument_key=NSE_INDEX%7CNifty%2050"

    def test_expired_option_contracts_url(self):
        u = uc.expired_option_contracts_url("NSE_INDEX|Nifty 50", "2024-03-28")
        assert u == "https://api.upstox.com/v2/expired-instruments/option/contract?instrument_key=NSE_INDEX%7CNifty%2050&expiry_date=2024-03-28"

    def test_expired_candle_url(self):
        u = uc.expired_candle_url("NSE_FO|73507|24-04-2025", "day", "2025-04-24", "2025-01-01")
        assert u == "https://api.upstox.com/v2/expired-instruments/historical-candle/NSE_FO%7C73507%7C24-04-2025/day/2025-04-24/2025-01-01"

    def test_unsupported_interval_rejected(self):
        with pytest.raises(ValueError, match="unsupported interval"):
            uc.expired_candle_url("k", "60minute", "2025-01-01", "2025-01-01")


class TestUpstoxClient:
    def client(self, response, monkeypatch):
        monkeypatch.setenv("UPSTOX_ACCESS_TOKEN", "tok123")
        return uc.UpstoxClient(pc.UpstoxCredentials.from_env(), session=FakeGetSession(response))

    def test_expiries_sends_bearer_and_returns_parsed_json(self, monkeypatch):
        c = self.client(FakeResponse(200, {"status": "success", "data": ["2024-10-03", "2025-04-17"]}), monkeypatch)
        r = c.expiries("NSE_INDEX|Nifty 50")
        url, headers, timeout = c._session.calls[0]
        assert headers["Authorization"] == "Bearer tok123" and headers["Accept"] == "application/json" and timeout == 30
        assert r.status_code == 200 and r.json["data"] == ["2024-10-03", "2025-04-17"]

    def test_non_json_body_is_returned_raw_not_raised(self, monkeypatch):
        c = self.client(FakeResponse(200, None, content=b"<html>error</html>"), monkeypatch)
        r = c.expired_option_contracts("NSE_INDEX|Nifty 50", "2024-03-28")
        assert r.json is None and r.content == b"<html>error</html>"

    def test_error_status_is_returned_not_raised(self, monkeypatch):
        c = self.client(FakeResponse(401, {"status": "error", "errors": [{"errorCode": "UDAPI1000"}]}), monkeypatch)
        r = c.expired_candles("NSE_FO|73507|24-04-2025", "day", "2025-01-01", "2024-01-01")
        assert r.status_code == 401 and r.json["status"] == "error"


class FakePostSession:
    def __init__(self, response):
        self.response, self.calls = response, []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append((url, json, headers, timeout))
        return self.response


class TestDhanRollingOptionBody:
    def test_shape(self):
        b = dc.rolling_option_body(security_id="26000", exchange_segment="NSE_FNO", instrument="OPTIDX", expiry_code=1, expiry_flag="WEEK",
                                   strike="ATM", option_type="CALL", from_date="2026-08-01", to_date="2026-08-30")
        assert b == {"exchangeSegment": "NSE_FNO", "interval": "5", "securityId": "26000", "instrument": "OPTIDX", "expiryCode": 1, "expiryFlag": "WEEK",
                    "strike": "ATM", "drvOptionType": "CALL", "requiredData": list(dc.REQUIRED_DATA), "fromDate": "2026-08-01", "toDate": "2026-08-30"}

    @pytest.mark.parametrize("field, value", [("interval", "3"), ("expiry_flag", "DAY"), ("option_type", "BUY")])
    def test_rejects_undocumented_values(self, field, value):
        kwargs = dict(security_id="1", exchange_segment="NSE_FNO", instrument="OPTIDX", expiry_code=1, expiry_flag="WEEK", strike="ATM",
                      option_type="CALL", from_date="a", to_date="b")
        kwargs[field] = value
        with pytest.raises(ValueError):
            dc.rolling_option_body(**kwargs)

    def test_expiry_code_zero_is_rejected(self):
        with pytest.raises(ValueError, match="expiry_code must be one of"):
            dc.rolling_option_body(security_id="13", exchange_segment="NSE_FNO", instrument="OPTIDX", expiry_code=0, expiry_flag="WEEK",
                                   strike="ATM", option_type="CALL", from_date="a", to_date="b")

    @pytest.mark.parametrize("code", [1, 2, 3])
    def test_expiry_code_1_2_3_are_accepted(self, code):
        b = dc.rolling_option_body(security_id="13", exchange_segment="NSE_FNO", instrument="OPTIDX", expiry_code=code, expiry_flag="WEEK",
                                   strike="ATM", option_type="CALL", from_date="a", to_date="b")
        assert b["expiryCode"] == code

    @pytest.mark.parametrize("code", [-1, 4, 99])
    def test_other_expiry_code_values_are_also_rejected(self, code):
        with pytest.raises(ValueError, match="expiry_code must be one of"):
            dc.rolling_option_body(security_id="13", exchange_segment="NSE_FNO", instrument="OPTIDX", expiry_code=code, expiry_flag="WEEK",
                                   strike="ATM", option_type="CALL", from_date="a", to_date="b")


class TestDhanClient:
    def test_posts_with_auth_headers_and_json_body(self, monkeypatch):
        monkeypatch.setenv("DHAN_CLIENT_ID", "cid")
        monkeypatch.setenv("DHAN_ACCESS_TOKEN", "tok")
        resp = FakeResponse(200, {"strike": [], "close": []})
        client = dc.DhanClient(pc.DhanCredentials.from_env(), session=FakePostSession(resp))
        r = client.rolling_option(security_id="26000", exchange_segment="NSE_FNO", instrument="OPTIDX", expiry_code=1, expiry_flag="WEEK",
                                  strike="ATM", option_type="CALL", from_date="2026-08-01", to_date="2026-08-30")
        url, body, headers, timeout = client._session.calls[0]
        assert url == "https://api.dhan.co/v2/charts/rollingoption" and headers["access-token"] == "tok" and headers["dhanClientId"] == "cid"
        assert body["strike"] == "ATM" and body["expiryCode"] == 1 and r.status_code == 200

    def test_rolling_option_rejects_expiry_code_zero_before_any_request_is_sent(self, monkeypatch):
        monkeypatch.setenv("DHAN_CLIENT_ID", "cid")
        monkeypatch.setenv("DHAN_ACCESS_TOKEN", "tok")
        session = FakePostSession(FakeResponse(200, {}))
        client = dc.DhanClient(pc.DhanCredentials.from_env(), session=session)
        with pytest.raises(ValueError, match="expiry_code must be one of"):
            client.rolling_option(security_id="13", exchange_segment="NSE_FNO", instrument="OPTIDX", expiry_code=0, expiry_flag="WEEK",
                                  strike="ATM", option_type="CALL", from_date="2026-08-01", to_date="2026-08-30")
        assert session.calls == []


class FakeSdkClient:

    def __init__(self, response):
        self.response, self.calls = response, []

    def expired_options_data(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class TestExpiredOptionsDataViaSdk:
    def test_parameter_mapping_to_the_official_sdk_is_correct(self):
        sdk = FakeSdkClient({"status": "success", "remarks": "", "data": {"data": {"ce": {}, "pe": None}}})
        dc.expired_options_data_via_sdk(
            sdk, security_id=13, exchange_segment="NSE_FNO", instrument_type="OPTIDX", expiry_flag="WEEK",
            expiry_code=1, strike="ATM+3", option_type="CALL", required_data=["open", "strike", "spot"],
            from_date="2026-09-16", to_date="2026-09-18", interval=1,
        )
        assert sdk.calls == [{
            "security_id": 13, "exchange_segment": "NSE_FNO", "instrument_type": "OPTIDX", "expiry_flag": "WEEK",
            "expiry_code": 1, "strike": "ATM+3", "drv_option_type": "CALL", "required_data": ["open", "strike", "spot"],
            "from_date": "2026-09-16", "to_date": "2026-09-18", "interval": 1,
        }]

    def test_nifty_uses_nse_fno_optidx_security_id_13(self):
        assert dc.NIFTY_SECURITY_ID == 13
        assert dc.NIFTY_EXCHANGE_SEGMENT == "NSE_FNO"
        assert dc.NIFTY_INSTRUMENT == "OPTIDX"

    def test_returns_the_raw_response_unchanged(self):
        sdk = FakeSdkClient({"status": "success", "remarks": "", "data": {"data": {"ce": {"strike": [23400.0]}, "pe": None}}})
        result = dc.expired_options_data_via_sdk(
            sdk, security_id=dc.NIFTY_SECURITY_ID, exchange_segment=dc.NIFTY_EXCHANGE_SEGMENT, instrument_type=dc.NIFTY_INSTRUMENT,
            expiry_flag="WEEK", expiry_code=1, strike="ATM+3", option_type="CALL", required_data=["open", "strike", "spot"],
            from_date="2026-09-16", to_date="2026-09-18",
        )
        assert result == sdk.response

    def test_expiry_code_zero_is_rejected_before_calling_the_sdk(self):
        sdk = FakeSdkClient({})
        with pytest.raises(ValueError, match="expiry_code must be one of"):
            dc.expired_options_data_via_sdk(
                sdk, security_id=13, exchange_segment="NSE_FNO", instrument_type="OPTIDX", expiry_flag="WEEK",
                expiry_code=0, strike="ATM", option_type="CALL", required_data=["open"], from_date="a", to_date="b",
            )
        assert sdk.calls == []

    @pytest.mark.parametrize("code", [1, 2, 3])
    def test_expiry_code_1_2_3_reach_the_sdk(self, code):
        sdk = FakeSdkClient({"status": "success", "remarks": "", "data": {}})
        dc.expired_options_data_via_sdk(
            sdk, security_id=13, exchange_segment="NSE_FNO", instrument_type="OPTIDX", expiry_flag="WEEK",
            expiry_code=code, strike="ATM", option_type="CALL", required_data=["open"], from_date="a", to_date="b",
        )
        assert sdk.calls[0]["expiry_code"] == code


class FakeBreezeSDK:
    def __init__(self, api_key):
        self.api_key = api_key
        self.session_calls = []
        self.historical_calls = []

    def generate_session(self, api_secret, session_token):
        self.session_calls.append((api_secret, session_token))

    def get_historical_data_v2(self, **kwargs):
        self.historical_calls.append(kwargs)
        return {"Success": [], "Status": 200, "Error": None}


class TestBreezeRequest:
    def test_rejects_bad_interval_and_right(self):
        base = dict(stock_code="NIFTY", expiry_date="2026-09-24T06:00:00.000Z", strike_price="25000", right="call", interval="1day",
                    from_date="2026-09-01T06:00:00.000Z", to_date="2026-09-24T06:00:00.000Z")
        bc.HistoricalOptionsRequest(**base)
        with pytest.raises(ValueError):
            bc.HistoricalOptionsRequest(**{**base, "interval": "1hour"})
        with pytest.raises(ValueError):
            bc.HistoricalOptionsRequest(**{**base, "right": "both"})

    def test_as_kwargs_matches_the_documented_field_names(self):
        r = bc.HistoricalOptionsRequest(stock_code="NIFTY", expiry_date="2026-09-24T06:00:00.000Z", strike_price="25000", right="put",
                                        interval="5minute", from_date="a", to_date="b")
        assert r.as_kwargs() == {"interval": "5minute", "from_date": "a", "to_date": "b", "stock_code": "NIFTY", "exchange_code": "NFO",
                                 "product_type": "options", "expiry_date": "2026-09-24T06:00:00.000Z", "right": "put", "strike_price": "25000"}


class TestBreezeClient:
    def make_creds(self, monkeypatch):
        monkeypatch.setenv("BREEZE_API_KEY", "k")
        monkeypatch.setenv("BREEZE_API_SECRET", "s")
        monkeypatch.setenv("BREEZE_SESSION_TOKEN", "t")
        return pc.BreezeCredentials.from_env()

    def test_generates_session_once_and_calls_get_historical_data_v2(self, monkeypatch):
        creds = self.make_creds(monkeypatch)
        sdk = FakeBreezeSDK("k")
        client = bc.BreezeClient(creds, sdk_factory=lambda: sdk)
        req = bc.HistoricalOptionsRequest(stock_code="NIFTY", expiry_date="e", strike_price="25000", right="call", interval="1day", from_date="a", to_date="b")
        client.get_option_historical(req)
        client.get_option_historical(req)
        assert sdk.session_calls == [("s", "t")]
        assert len(sdk.historical_calls) == 2 and sdk.historical_calls[0]["strike_price"] == "25000"

    def test_missing_package_raises_a_clear_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "breeze_connect", None)
        creds = self.make_creds(monkeypatch)
        with pytest.raises(bc.BreezePackageNotInstalled, match="pip install breeze-connect"):
            bc.BreezeClient(creds)
