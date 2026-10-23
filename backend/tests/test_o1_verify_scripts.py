from __future__ import annotations

import datetime as dt
import json
from collections import Counter

import pytest

from backend.app.data.raw.store import RawStore
from backend.app.data.sources.breeze_client import BreezeClient
from backend.app.data.sources.upstox_client import ApiResponse as UpResp
from backend.app.data.sources.upstox_client import UpstoxClient
from scripts import run_o1_verify_breeze as vb
from scripts import run_o1_verify_dhan as vd
from scripts import run_o1_verify_upstox as vu
from scripts.run_o1_0b_sample_gate import load as gate_load
from scripts.run_o1_0b_sample_gate import udiff_bars as gate_udiff_bars


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("UPSTOX_ACCESS_TOKEN", "DHAN_CLIENT_ID", "DHAN_ACCESS_TOKEN", "BREEZE_API_KEY", "BREEZE_API_SECRET", "BREEZE_SESSION_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    from backend.app.data.sources import provider_credentials as pc
    monkeypatch.setattr(pc, "load_dotenv", lambda: None)


class TestMainWithoutCredentials:
    @pytest.mark.parametrize("mod, extra_args", [(vu, []), (vd, []), (vb, ["--expiry", "2026-09-24", "--spot", "25000"])])
    def test_exits_2_and_prints_which_vars_are_missing(self, mod, extra_args, capsys):
        rc = mod.main(extra_args)
        out = capsys.readouterr().out
        assert rc == 2 and "NOT CONFIGURED" in out and ".env.example" in out


class FakeUpstoxClient:

    def __init__(self, expiries, contracts_by_expiry, candles=None, candles_by_type=None, expiries_response=None):
        self._expiries, self._contracts, self._candles, self._candles_by_type = expiries, contracts_by_expiry, candles or [], candles_by_type or {}
        self._oc_calls = Counter()
        self._expiries_response = expiries_response

    def expiries(self, instrument_key):
        if self._expiries_response is not None:
            return self._expiries_response
        return UpResp(200, b"{}", {}, {"status": "success", "data": self._expiries})

    def expired_option_contracts(self, instrument_key, expiry):
        v = self._contracts.get(expiry, [])
        if isinstance(v, tuple):
            call_n = self._oc_calls[expiry]
            self._oc_calls[expiry] += 1
            v = v[min(call_n, len(v) - 1)]
        return UpResp(200, b"{}", {}, {"status": "success", "data": v})

    def expired_candles(self, expired_instrument_key, interval, to_date, from_date):
        cp = expired_instrument_key.split("_")[-1] if "_" in expired_instrument_key else None
        candles = self._candles_by_type.get(cp, self._candles)
        return UpResp(200, b"{}", {}, {"status": "success", "data": {"candles": candles}})


def contract(strike, cp, key, lot=65):
    return {"instrument_key": key, "strike_price": strike, "instrument_type": cp, "lot_size": lot, "weekly": True}


class TestUpstoxVerify:
    def test_reports_earliest_latest_and_per_year_availability(self, tmp_path):
        c = FakeUpstoxClient(["2024-10-03", "2025-04-17", "2025-09-25"], {}, [])
        r = vu.verify(c, instrument_key="NSE_INDEX|Nifty 50", underlying="NIFTY", years=(2024, 2025, 2026))
        assert (r["earliest_expiry_returned"], r["latest_expiry_returned"]) == ("2024-10-03", "2025-09-25")
        assert len(r["years"][2024]["expiries_available_in_year"]) == 1 and len(r["years"][2025]["expiries_available_in_year"]) == 2
        assert r["years"][2026]["expiries_available_in_year"] == [] and r["years"][2026]["tested"] == {}

    def test_contract_stats_including_duplicates_and_strike_intervals(self):
        cs = [contract(24000, "CE", "a"), contract(24000, "PE", "b"), contract(24100, "CE", "c"), contract(24100, "CE", "c2")]
        candles = [["2024-10-03T00:00:00+05:30", 100, 110, 95, 105, 5000, 12000]]
        c = FakeUpstoxClient(["2024-10-03"], {"2024-10-03": cs}, candles)
        r = vu.verify(c, instrument_key="k", underlying="NIFTY", years=(2024,))
        entry = r["years"][2024]["tested"]["2024-10-03"]
        assert (entry["rows"], entry["CE"], entry["PE"], entry["unique_strikes"]) == (4, 3, 1, 2)
        assert entry["lowest_strike"] == 24000 and entry["highest_strike"] == 24100 and entry["strike_intervals"] == [100]
        assert entry["duplicate_strike_type_pairs"] == {"CE 24100": 2}
        assert entry["distinct_lot_sizes"] == 1

    def test_candle_field_presence_and_trade_state_classification(self):
        cs = [contract(24000, "CE", "a")]
        candles = [["t1", 100, 110, 95, 105, 5000, 12000], ["t2", 0, 0, 0, 26.65, 0, 500]]
        c = FakeUpstoxClient(["2024-10-03"], {"2024-10-03": cs}, candles)
        r = vu.verify(c, instrument_key="k", underlying="NIFTY", years=(2024,))
        one = next(iter(r["years"][2024]["tested"]["2024-10-03"]["candles"].values()))
        assert one["day"]["rows"] == 2 and one["day"]["has_oi_field"] and one["day"]["queryable"] is True
        assert one["day"]["trade_states"] == {"TRADED": 1, "NOT_TRADED_PRICE_CARRIED": 1}

    def test_duplicate_timestamps_are_reported(self):
        cs = [contract(24000, "CE", "a")]
        candles = [["t1", 1, 1, 1, 1, 1, 1], ["t1", 1, 1, 1, 1, 1, 1]]
        c = FakeUpstoxClient(["2024-10-03"], {"2024-10-03": cs}, candles)
        r = vu.verify(c, instrument_key="k", underlying="NIFTY", years=(2024,))
        one = next(iter(r["years"][2024]["tested"]["2024-10-03"]["candles"].values()))
        assert one["day"]["duplicate_timestamps"] == 1

    def test_empty_candle_response_is_not_queryable_and_never_crashes(self):
        cs = [contract(24000, "CE", "a")]
        c = FakeUpstoxClient(["2024-10-03"], {"2024-10-03": cs}, [])
        r = vu.verify(c, instrument_key="k", underlying="NIFTY", years=(2024,))
        one = next(iter(r["years"][2024]["tested"]["2024-10-03"]["candles"].values()))
        assert one["day"] == {"status": 200, "queryable": False, "rows": 0, "trade_states": {}, "duplicate_timestamps": 0, "has_oi_field": False}

    def test_raw_responses_are_stored_when_a_store_is_given(self, tmp_path):
        cs = [contract(24000, "CE", "a")]
        c = FakeUpstoxClient(["2024-10-03"], {"2024-10-03": cs}, [[1, 1, 1, 1, 1, 1, 1]])
        store = RawStore(tmp_path / "raw")
        vu.verify(c, instrument_key="k", underlying="NIFTY", years=(2024,), store=store)
        assert store.manifest_entries()
        assert {m.dataset for m in store.manifest_entries()} >= {"expiries", "option_contracts", "candles"}
        assert all(m.provider == "upstox" and m.license_status == "UNCLEAR" for m in store.manifest_entries())

    def test_no_auth_or_token_named_header_is_ever_stored(self, tmp_path):
        c = FakeUpstoxClient(["2024-10-03"], {"2024-10-03": [contract(24000, "CE", "a")]}, [])
        store = RawStore(tmp_path / "raw")
        vu.verify(c, instrument_key="k", underlying="NIFTY", years=(2024,), store=store)
        for m in store.manifest_entries():
            assert not any("auth" in k.lower() or "token" in k.lower() for k in m.response_headers)

    def test_identifier_stability_check_runs_once_and_detects_a_difference(self):
        first_call = [contract(24000, "CE", "a"), contract(24100, "CE", "b")]
        second_call_same = list(first_call)
        c = FakeUpstoxClient(["2024-10-03"], {"2024-10-03": (first_call, second_call_same)}, [])
        r = vu.verify(c, instrument_key="k", underlying="NIFTY", years=(2024,))
        entry = r["years"][2024]["tested"]["2024-10-03"]
        assert entry["identifier_stability_same_session"] == {"identical": True, "only_in_first_call": 0, "only_in_second_call": 0}

    def test_identifier_stability_check_flags_a_real_difference(self):
        first_call = [contract(24000, "CE", "a")]
        second_call_different = [contract(24000, "CE", "a-renumbered")]
        c = FakeUpstoxClient(["2024-10-03"], {"2024-10-03": (first_call, second_call_different)}, [])
        r = vu.verify(c, instrument_key="k", underlying="NIFTY", years=(2024,))
        s = r["years"][2024]["tested"]["2024-10-03"]["identifier_stability_same_session"]
        assert s == {"identical": False, "only_in_first_call": 1, "only_in_second_call": 1}

    def test_stability_check_happens_only_once_across_multiple_expiries(self):
        cs = {"2024-10-03": [contract(24000, "CE", "a")], "2024-11-07": [contract(24000, "CE", "b")]}
        c = FakeUpstoxClient(["2024-10-03", "2024-11-07"], cs, [])
        r = vu.verify(c, instrument_key="k", underlying="NIFTY", years=(2024,))
        tested = r["years"][2024]["tested"]
        assert "identifier_stability_same_session" in tested["2024-10-03"] and "identifier_stability_same_session" not in tested["2024-11-07"]

    def test_plan_upgrade_required_stops_before_any_contract_call(self, tmp_path):
        c = FakeUpstoxClient([], {}, [], expiries_response=UpResp(403, b'{"status":"error"}', {}, {"status": "error"}))
        store = RawStore(tmp_path / "raw")
        r = vu.verify(c, instrument_key="k", underlying="NIFTY", years=(2024, 2025), store=store)
        assert r["plan_upgrade_required"]["status_code"] == 403 and r["years"] == {}
        assert {m.dataset for m in store.manifest_entries()} == {"expiries"}

    def test_plan_upgrade_error_code_without_403_is_also_caught(self):
        c = FakeUpstoxClient([], {}, [], expiries_response=UpResp(400, b"{}", {}, {"status": "error", "errors": [{"errorCode": "UDAPI1149"}]}))
        r = vu.verify(c, instrument_key="k", underlying="NIFTY")
        assert "plan_upgrade_required" in r

    def test_an_ordinary_400_is_not_mistaken_for_a_plan_upgrade(self):
        c = FakeUpstoxClient([], {}, [], expiries_response=UpResp(400, b"{}", {}, {"status": "error", "errors": [{"errorCode": "UDAPI1000"}]}))
        r = vu.verify(c, instrument_key="k", underlying="NIFTY")
        assert "plan_upgrade_required" not in r and r["all_expiries_returned"] == []

    def test_rate_limit_headers_are_captured_when_present(self):
        resp = UpResp(200, b"{}", {"X-RateLimit-Remaining": "49", "Content-Type": "application/json"}, {"status": "success", "data": []})
        c = FakeUpstoxClient([], {}, [], expiries_response=resp)
        r = vu.verify(c, instrument_key="k", underlying="NIFTY")
        assert r["rate_limit_headers_observed"] == {"X-RateLimit-Remaining": "49"}

    def test_no_invented_availability_timestamp(self):
        r = vu.verify(FakeUpstoxClient([], {}, []), instrument_key="k", underlying="NIFTY")
        assert "none is invented" in r["pit_note"] and "documents no publication timestamp" in r["pit_note"]


class TestUpstoxContractStatsHelpers:
    def test_pick_itm_atm_otm_on_ce_only(self):
        cs = [contract(s, "CE", str(s)) for s in (100, 200, 300, 400, 500)]
        picked = vu.pick_itm_atm_otm(cs)
        assert [p["strike_price"] for p in picked] == [100, 300, 500]

    def test_pick_itm_atm_otm_empty(self):
        assert vu.pick_itm_atm_otm([contract(100, "PE", "x")]) == []

    def test_rate_limit_header_matching_is_case_insensitive(self):
        r = UpResp(200, b"", {"x-ratelimit-limit": "500", "Retry-After": "5", "Content-Type": "x"}, None)
        assert vu.rate_limit_headers(r) == {"x-ratelimit-limit": "500"}

    def test_is_plan_upgrade_error(self):
        assert vu.is_plan_upgrade_error(UpResp(403, b"", {}, None)) is True
        assert vu.is_plan_upgrade_error(UpResp(400, b"", {}, {"errors": [{"errorCode": "UDAPI1149"}]})) is True
        assert vu.is_plan_upgrade_error(UpResp(400, b"", {}, {"errors": [{"errorCode": "UDAPI1000"}]})) is False
        assert vu.is_plan_upgrade_error(UpResp(200, b"", {}, {})) is False


class TestUpstoxNseCrossCheck:
    def test_non_cross_check_expiry_returns_none(self):
        assert vu.reconcile_with_nse("2024-03-28", [], {}) is None

    def test_cross_check_expiry_with_no_local_sample_reports_unavailable(self, monkeypatch):
        monkeypatch.setattr("scripts.run_o1_0b_sample_gate.load", lambda family, d: None)
        r = vu.reconcile_with_nse("2026-09-15", [], {})
        assert r == {"available": False, "reason": "local NSE sample file not present (data/raw/nse_fo_sample) - run scripts/run_o1_0b_sample_gate.py first"}

    def test_cross_check_matches_a_real_local_sample_contract(self):
        got = gate_load("udiff", dt.date(2026, 9, 15))
        if got is None:
            pytest.skip("O1.0B NSE sample not present on disk")
        bars = [b for b in gate_udiff_bars(*got) if b.key.underlying == "NIFTY" and b.key.dtype == "IDO" and str(b.key.expiry) == "2026-09-15"]
        one = next(b for b in bars if b.key.cp == "CE")
        strike = one.key.strike_paise / 100
        contracts = [contract(strike, "CE", "up_id", lot=one.lot)]
        tested = {"up_id": {"strike": strike, "type": "CE", "lot_size": one.lot}}
        r = vu.reconcile_with_nse("2026-09-15", contracts, tested)
        assert r["available"] is True
        assert r["contracts"]["up_id"]["strike_identity"] == "MATCH" and r["contracts"]["up_id"]["lot"]["classification"] == "MATCH"

    def test_cross_check_reports_missing_when_upstox_strike_does_not_exist_in_nse(self):
        got = gate_load("udiff", dt.date(2026, 9, 15))
        if got is None:
            pytest.skip("O1.0B NSE sample not present on disk")
        tested = {"fake": {"strike": 1.0, "type": "CE", "lot_size": 65}}
        r = vu.reconcile_with_nse("2026-09-15", [], tested)
        assert r["contracts"]["fake"]["classification"] == "MISSING"


class FakeDhanSdkClient:

    def __init__(self, leg_body: dict | None, status: str = "success"):
        self.leg_body, self.status, self.calls = leg_body, status, []

    def expired_options_data(self, **kwargs):
        self.calls.append(kwargs)
        side = "ce" if kwargs["drv_option_type"] == "CALL" else "pe"
        return {"status": self.status, "remarks": "", "data": {"data": {side: self.leg_body}}}


class TestDhanVerify:
    def test_field_presence_detection(self, tmp_path):
        leg = {"open": [1], "high": [2], "low": [0.5], "close": [1.5], "volume": [100], "oi": [200], "iv": [15.2], "strike": [24000], "spot": [24010]}
        store = RawStore(tmp_path / "raw")
        r = vd.verify(FakeDhanSdkClient(leg), security_id=13, exchange_segment="NSE_FNO", store=store, from_date="2026-08-01", to_date="2026-08-30", strikes=("ATM",))
        e = r["strikes"]["ATM_CALL"]
        assert e["rows"] == 1 and e["has_open"] and e["has_close"] and e["has_volume"] and e["has_oi"] and e["has_iv"] and e["has_spot"]
        assert store.manifest_entries() and all(m.provider == "dhan" for m in store.manifest_entries())

    def test_empty_body_reports_false_not_a_crash(self):
        r = vd.verify(FakeDhanSdkClient({}), security_id=13, exchange_segment="NSE_FNO", from_date="a", to_date="b", strikes=("ATM",))
        e = r["strikes"]["ATM_CALL"]
        assert e == {"status": "success", "rows": 0, "has_open": False, "has_high": False, "has_low": False, "has_close": False, "has_volume": False,
                    "has_oi": False, "has_iv": False, "has_strike_series": False, "has_spot": False, "distinct_strike_values_returned": None}

    def test_tests_both_legs_of_every_requested_strike(self):
        r = vd.verify(FakeDhanSdkClient({"close": [1]}), security_id=13, exchange_segment="NSE_FNO", from_date="a", to_date="b", strikes=("ATM-3", "ATM", "ATM+3"))
        assert set(r["strikes"]) == {"ATM-3_CALL", "ATM-3_PUT", "ATM_CALL", "ATM_PUT", "ATM+3_CALL", "ATM+3_PUT"}

    def test_uses_expiry_code_1_by_default_never_0(self):
        sdk = FakeDhanSdkClient({"close": [1]})
        vd.verify(sdk, security_id=13, exchange_segment="NSE_FNO", from_date="a", to_date="b", strikes=("ATM",))
        assert all(call["expiry_code"] == 1 for call in sdk.calls)
        assert all(call["expiry_code"] != 0 for call in sdk.calls)

    def test_explicit_expiry_code_is_honoured(self):
        sdk = FakeDhanSdkClient({"close": [1]})
        vd.verify(sdk, security_id=13, exchange_segment="NSE_FNO", from_date="a", to_date="b", strikes=("ATM",), expiry_code=3)
        assert all(call["expiry_code"] == 3 for call in sdk.calls)

    def test_nifty_defaults_use_nse_fno_optidx_security_id_13(self):
        assert vd.dc.NIFTY_SECURITY_ID == 13
        assert vd.dc.NIFTY_EXCHANGE_SEGMENT == "NSE_FNO"
        assert vd.STRIKES == ("ATM-3", "ATM", "ATM+3")


import datetime as dt  # noqa: E402


class FakeBreezeClient:
    def __init__(self, result):
        self.result, self.requests = result, []

    def get_option_historical(self, request):
        self.requests.append(request)
        return self.result


class TestBreezeVerify:
    def test_builds_a_strike_ladder_around_spot_and_reports_fields(self, tmp_path):
        result = {"Success": [{"open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10, "open_interest": 500}], "Error": None, "Status": 200}
        store = RawStore(tmp_path / "raw")
        r = vb.verify(FakeBreezeClient(result), stock_code="NIFTY", expiry=dt.date(2026, 9, 24), spot=25000, strike_step=100, lookback_days=30, store=store)
        assert set(r["strikes"]) == {24900, 25000, 25100}
        leg = r["strikes"][25000]["call_1day"]
        assert leg["rows"] == 1 and leg["has_ohlc"] and leg["has_volume"] and leg["has_open_interest"]
        assert store.manifest_entries() and all(m.provider == "breeze" for m in store.manifest_entries())

    def test_missing_open_interest_is_reported_false(self):
        result = {"Success": [{"open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10, "open_interest": None}], "Error": None}
        r = vb.verify(FakeBreezeClient(result), stock_code="NIFTY", expiry=dt.date(2026, 9, 24), spot=25000, strike_step=100, lookback_days=30)
        assert r["strikes"][25000]["call_1day"]["has_open_interest"] is False

    def test_error_field_is_surfaced(self):
        result = {"Success": [], "Error": "Please enter valid Strike Price"}
        r = vb.verify(FakeBreezeClient(result), stock_code="NIFTY", expiry=dt.date(2026, 9, 24), spot=25000, strike_step=100, lookback_days=30)
        assert r["strikes"][25000]["call_1day"]["error_field"] == "Please enter valid Strike Price" and r["strikes"][25000]["call_1day"]["rows"] == 0

    def test_an_sdk_exception_is_recorded_not_raised(self):
        class Boom:
            def get_option_historical(self, request):
                raise RuntimeError("session expired")
        r = vb.verify(Boom(), stock_code="NIFTY", expiry=dt.date(2026, 9, 24), spot=25000, strike_step=100, lookback_days=30)
        assert "session expired" in r["strikes"][25000]["call_1day"]["error"]

    def test_every_leg_is_requested_two_rights_two_intervals(self):
        c = FakeBreezeClient({"Success": [], "Error": None})
        vb.verify(c, stock_code="NIFTY", expiry=dt.date(2026, 9, 24), spot=25000, strike_step=100, lookback_days=30)
        assert len(c.requests) == 3 * 2 * 2
        assert {r.right for r in c.requests} == {"call", "put"} and {r.interval for r in c.requests} == {"1day", "5minute"}
