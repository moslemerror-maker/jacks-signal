from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import io
import json
import zipfile
from decimal import Decimal

import pytest

from scripts import run_o1_0b_sample_gate as g

D = dt.date
UTC = dt.timezone.utc


class TestStrikeIdentity:
    @pytest.mark.parametrize("text, paise", [("22000", 2200000), ("22000.00", 2200000), ("192.5", 19250), ("0.05", 5), ("0", 0), ("  150 ", 15000)])
    def test_exact(self, text, paise):
        assert g.strike_to_paise(text) == (paise, True)

    def test_sub_paise_is_flagged_inexact(self):
        assert g.strike_to_paise("10.005") == (1000, False)
        assert g.strike_to_paise("10.005")[1] is False

    def test_never_a_float(self):
        assert isinstance(g.strike_to_paise("0.1")[0], int)
        assert g.strike_to_paise("0.1") == (10, True)

    @pytest.mark.parametrize("bad", ["", "abc", None])
    def test_unparseable(self, bad):
        assert g.strike_to_paise(bad) == (None, False)


class TestOptionType:
    @pytest.mark.parametrize("raw, expected", [("CE", "CE"), ("PE", "PE"), ("CA", "CA"), ("PA", "PA"), ("XX", None), ("", None), (None, None), (" CE ", "CE")])
    def test_vocabulary(self, raw, expected):
        assert g.norm_cp(raw) == expected

    def test_american_and_european_are_distinct_identities(self):
        e = g.ContractKey("SBIN", "STO", D(2010, 7, 29), 30000, "CE")
        a = g.ContractKey("SBIN", "STO", D(2010, 7, 29), 30000, "CA")
        assert e != a and len({e, a}) == 2


class TestDates:
    def test_legacy_month_case_insensitive(self):
        assert g.parse_legacy_date("29-JAN-2015") == D(2015, 1, 29) == g.parse_legacy_date("29-Jan-2015")

    def test_nse_epoch_is_1980(self):
        assert g.decode_nse_seconds(0) == dt.datetime(1980, 1, 1, tzinfo=UTC)
        assert g.decode_nse_seconds(1391040000).date() == D(2024, 1, 30)

    def test_master_expiry_time_of_day_is_2000_ist_and_date_is_the_ist_date(self):
        x = g.decode_nse_seconds(1391610600)
        assert (x.date(), x.strftime("%H:%M")) == (D(2024, 2, 5), "14:30")


class TestTradeState:
    def d(self, s):
        return Decimal(s)

    def test_traded(self):
        assert g.classify_trade_state(10, self.d(5), self.d(6), self.d(4), self.d(5.5)) == (True, "TRADED")

    def test_carried_close_is_not_an_observation(self):
        assert g.classify_trade_state(0, self.d(0), self.d(0), self.d(0), self.d("26.65")) == (False, "NOT_TRADED_PRICE_CARRIED")

    def test_no_price(self):
        assert g.classify_trade_state(0, self.d(0), self.d(0), self.d(0), self.d(0)) == (False, "NOT_TRADED_NO_PRICE")

    def test_zero_volume_with_ohlc_is_an_anomaly_not_a_trade(self):
        ok, st = g.classify_trade_state(0, self.d(1), self.d(2), self.d(1), self.d(1))
        assert (ok, st) == (False, "ANOMALY_NOT_TRADED_WITH_OHLC")

    def test_traded_with_zero_open_is_an_anomaly(self):
        assert g.classify_trade_state(2, self.d(0), self.d(0), self.d(0), self.d("1361.7")) == (False, "ANOMALY_TRADED_NONPOSITIVE_PRICE")

    def test_close_outside_range_is_an_anomaly(self):
        assert g.classify_trade_state(33, self.d("463.05"), self.d("465.45"), self.d(458), self.d(457)) == (False, "ANOMALY_TRADED_OHLC_ORDER")

    def test_negative_volume(self):
        assert g.classify_trade_state(-1, self.d(1), self.d(1), self.d(1), self.d(1)) == (False, "ANOMALY_NEGATIVE_VOLUME")

    def test_is_traded_iff_state_is_traded(self):
        for vol in (-1, 0, 1):
            for o, h, l, c in ((0, 0, 0, 0), (0, 0, 0, 5), (5, 6, 4, 5), (5, 6, 4, 7), (0, 9, 0, 9)):
                ok, st = g.classify_trade_state(vol, *(Decimal(x) for x in (o, h, l, c)))
                assert ok == (st == "TRADED")


class TestExpirySettleClassification:
    def c(self, settle, close="10", spot="22326.9", strike="22300", cp="CE"):
        return g.classify_expiry_settle(Decimal(settle), Decimal(close), Decimal(spot), Decimal(strike), cp)

    def test_zero_wins_over_zero_intrinsic(self):
        assert self.c("0", strike="22400") == "ZERO"

    def test_underlying_settlement_price(self):
        assert self.c("22326.9") == "UNDERLYING_SETTLEMENT_PRICE"

    def test_intrinsic(self):
        assert self.c("26.9") == "INTRINSIC_VALUE"
        assert self.c("73.1", strike="22400", cp="PE") == "INTRINSIC_VALUE"

    def test_close_and_other(self):
        assert self.c("10", close="10") == "EQUALS_LAST_CLOSE"
        assert self.c("11", close="10") == "OTHER"


class TestAvailability:
    def lm(self, y, m, d, hh, mm):
        return dt.datetime(y, m, d, hh, mm, tzinfo=g.IST).astimezone(UTC)

    def test_observed_last_modified_between_close_and_next_open(self):
        r = g.availability_for_eod_file(D(2024, 7, 5), None, self.lm(2024, 7, 5, 19, 8))
        assert r["availability_basis"] == "OBSERVED_HTTP_LAST_MODIFIED"
        assert r["available_at"] == self.lm(2024, 7, 5, 19, 8)

    def test_reupload_after_next_open_falls_back_to_the_rule(self):
        r = g.availability_for_eod_file(D(2015, 1, 5), None, self.lm(2019, 1, 17, 17, 38))
        assert r["availability_basis"] == "RULE_NEXT_SESSION_OPEN"
        assert r["available_at"] == self.lm(2015, 1, 6, 9, 15)
        assert "REUPLOAD" in r["evidence"]["reason"]

    def test_last_modified_before_close_is_implausible_and_not_trusted(self):
        r = g.availability_for_eod_file(D(2024, 7, 5), None, self.lm(2024, 7, 5, 12, 0))
        assert r["availability_basis"] == "RULE_NEXT_SESSION_OPEN" and "IMPLAUSIBLE" in r["evidence"]["reason"]

    def test_missing_last_modified(self):
        r = g.availability_for_eod_file(D(2024, 7, 5), None, None)
        assert r["availability_basis"] == "RULE_NEXT_SESSION_OPEN" and r["evidence"]["reason"] == "NO_LAST_MODIFIED"

    def test_exactly_at_the_close_or_next_open(self):
        close = g.session_close_utc(D(2024, 7, 5))
        assert g.availability_for_eod_file(D(2024, 7, 5), None, close)["availability_basis"] == "OBSERVED_HTTP_LAST_MODIFIED"
        nxt, _ = g.next_session_open_utc(D(2024, 7, 5))
        assert g.availability_for_eod_file(D(2024, 7, 5), None, nxt)["availability_basis"] == "RULE_NEXT_SESSION_OPEN"

    def test_same_day_close_decision_can_never_see_post_close_data(self):
        close = g.session_close_utc(D(2024, 7, 5))
        for lm in (None, self.lm(2024, 7, 5, 15, 30), self.lm(2024, 7, 5, 19, 8), self.lm(2019, 1, 17, 17, 38), self.lm(2024, 7, 5, 9, 0)):
            assert g.availability_for_eod_file(D(2024, 7, 5), None, lm)["available_at"] >= close

    def test_next_session_uses_observed_sessions_over_the_weekday_fallback(self):
        friday = D(2024, 7, 5)
        nxt, basis = g.next_session_open_utc(friday, [D(2024, 7, 9)])
        assert basis == "OBSERVED_SESSIONS" and nxt.astimezone(g.IST).date() == D(2024, 7, 9)
        nxt, basis = g.next_session_open_utc(friday, [])
        assert basis == "WEEKDAY_FALLBACK" and nxt.astimezone(g.IST).date() == D(2024, 7, 8)

    def test_next_open_is_0915_ist(self):
        nxt, _ = g.next_session_open_utc(D(2024, 7, 4))
        assert nxt.astimezone(g.IST).strftime("%H:%M") == "09:15"

    @pytest.mark.parametrize("hdr", ["Fri, 05 Jul 2024 13:38:00 GMT"])
    def test_last_modified_header_parses_as_utc(self, hdr):
        t = g.parse_last_modified(hdr)
        assert t == dt.datetime(2024, 7, 5, 13, 38, tzinfo=UTC) and t.astimezone(g.IST).strftime("%H:%M") == "19:08"

    @pytest.mark.parametrize("bad", [None, "", "not a date"])
    def test_bad_last_modified(self, bad):
        assert g.parse_last_modified(bad) is None


LEGACY_HEADER = ["INSTRUMENT", "SYMBOL", "EXPIRY_DT", "STRIKE_PR", "OPTION_TYP", "OPEN", "HIGH", "LOW", "CLOSE", "SETTLE_PR", "CONTRACTS", "VAL_INLAKH", "OPEN_INT", "CHG_IN_OI", "TIMESTAMP", ""]
UDIFF_HEADER = ["TradDt", "BizDt", "Sgmt", "Src", "FinInstrmTp", "FinInstrmId", "ISIN", "TckrSymb", "SctySrs", "XpryDt", "FininstrmActlXpryDt", "StrkPric", "OptnTp", "FinInstrmNm",
                "OpnPric", "HghPric", "LwPric", "ClsPric", "LastPric", "PrvsClsgPric", "UndrlygPric", "SttlmPric", "OpnIntrst", "ChngInOpnIntrst", "TtlTradgVol", "TtlTrfVal",
                "TtlNbOfTxsExctd", "SsnId", "NewBrdLotQty", "Rmks", "Rsvd01", "Rsvd02", "Rsvd03", "Rsvd04"]


def legacy_row(instr="OPTIDX", sym="NIFTY", exp="25-JUL-2024", strike="22000", cp="CE", o="0", h="0", l="0", c="26.65", s="26.65", v="0", val="0", oi="0", chg="0", ts="05-JUL-2024"):
    return [instr, sym, exp, strike, cp, o, h, l, c, s, v, val, oi, chg, ts, ""]


def udiff_row(tp="IDO", i="123", sym="NIFTY", exp="2024-07-25", strike="22000.00", cp="CE", o="0", h="0", l="0", c="26.65", s="26.65", v="0", val="0", oi="0", chg="0", d="2024-07-05", lot="25", und="24300"):
    r = [""] * len(UDIFF_HEADER)
    for name, val_ in dict(TradDt=d, FinInstrmTp=tp, FinInstrmId=i, TckrSymb=sym, XpryDt=exp, FininstrmActlXpryDt=exp, StrkPric=strike, OptnTp=cp, OpnPric=o, HghPric=h, LwPric=l, ClsPric=c,
                           UndrlygPric=und, SttlmPric=s, OpnIntrst=oi, ChngInOpnIntrst=chg, TtlTradgVol=v, TtlTrfVal=val, NewBrdLotQty=lot).items():
        r[UDIFF_HEADER.index(name)] = val_
    return r


class TestParsing:
    def test_legacy_row(self):
        (b,) = g.legacy_bars(LEGACY_HEADER, [legacy_row(o="10", h="12", l="9", c="11", s="11.5", v="100", val="2.75", oi="500", chg="-50")])
        assert b.key == g.ContractKey("NIFTY", "IDO", D(2024, 7, 25), 2200000, "CE")
        assert (b.open, b.high, b.low, b.close, b.settle, b.volume, b.oi, b.oi_change, b.turnover) == (10, 12, 9, 11, Decimal("11.5"), 100, 500, -50, Decimal("2.75"))
        assert b.trade_date == D(2024, 7, 5) and b.source_id is None and b.lot is None

    def test_udiff_row(self):
        (b,) = g.udiff_bars(UDIFF_HEADER, [udiff_row(v="7", oi="100", chg="10", lot="65")])
        assert b.key == g.ContractKey("NIFTY", "IDO", D(2024, 7, 25), 2200000, "CE")
        assert (b.volume, b.oi, b.oi_change, b.lot, b.source_id, b.underlying_price) == (7, 100, 10, 65, "123", Decimal(24300))

    def test_futures_have_no_option_type_and_zero_strike(self):
        (b,) = g.legacy_bars(LEGACY_HEADER, [legacy_row(instr="FUTIDX", strike="0", cp="XX")])
        assert b.key.cp is None and b.key.strike_paise == 0 and b.key.dtype == "IDF"
        (u,) = g.udiff_bars(UDIFF_HEADER, [udiff_row(tp="IDF", strike="", cp="")])
        assert u.key.cp is None and u.key.strike_paise == 0

    def test_american_option_types_survive_parsing(self):
        rows = [legacy_row(instr="OPTSTK", sym="SBIN", cp="CA"), legacy_row(instr="OPTSTK", sym="SBIN", cp="PA")]
        bars = g.legacy_bars(LEGACY_HEADER, rows)
        assert [b.key.cp for b in bars] == ["CA", "PA"] and len(g.by_key(bars)[0]) == 2

    def test_unknown_legacy_instrument_is_kept_not_dropped(self):
        (b,) = g.legacy_bars(LEGACY_HEADER, [legacy_row(instr="FUTIVX", sym="INDIAVIX", strike="0", cp="XX")])
        assert b.key.dtype == "OTHER:FUTIVX"

    def test_duplicate_keys_are_reported(self):
        bars = g.legacy_bars(LEGACY_HEADER, [legacy_row(), legacy_row()])
        seen, dups = g.by_key(bars)
        assert len(seen) == 1 and len(dups) == 1

    def test_short_rows_are_skipped(self):
        assert g.legacy_bars(LEGACY_HEADER, [["OPTIDX", "NIFTY"]]) == []

    def test_row_hash_is_deterministic_and_field_sensitive(self):
        a, b = legacy_row(), legacy_row(c="27")
        assert g.row_hash(a) == g.row_hash(list(a)) and g.row_hash(a) != g.row_hash(b)


def bars_from(rows):
    return g.by_key(g.legacy_bars(LEGACY_HEADER, rows))[0]


class TestRenderingReconciliation:
    def test_identical(self):
        legacy = bars_from([legacy_row(v="5", o="1", h="2", l="1", c="2", s="2", oi="10", chg="1")])
        udiff = g.by_key(g.udiff_bars(UDIFF_HEADER, [udiff_row(v="5", o="1", h="2", l="1", c="2", s="2", oi="10", chg="1")]))[0]
        assert g.reconcile_renderings(legacy, udiff)["reconciles"] is True

    @pytest.mark.parametrize("field, ukw, lkw", [("close", dict(c="3"), dict(c="2")), ("volume", dict(v="6"), dict(v="5")), ("oi_change", dict(chg="2"), dict(chg="1")),
                                                   ("settle", dict(s="9"), dict(s="2"))])
    def test_each_disagreement_is_caught(self, field, ukw, lkw):
        base = dict(v="5", o="1", h="2", l="1", c="2", s="2", oi="10", chg="1")
        legacy = bars_from([legacy_row(**{**base, **lkw})])
        udiff = g.by_key(g.udiff_bars(UDIFF_HEADER, [udiff_row(**{**base, **ukw})]))[0]
        r = g.reconcile_renderings(legacy, udiff)
        assert r["reconciles"] is False and field in r["field_disagreements"]

    def test_a_row_in_only_one_rendering_fails(self):
        legacy = bars_from([legacy_row(), legacy_row(strike="22050")])
        udiff = g.by_key(g.udiff_bars(UDIFF_HEADER, [udiff_row()]))[0]
        r = g.reconcile_renderings(legacy, udiff)
        assert r["reconciles"] is False and r["legacy_only"] == 1 and r["udiff_only"] == 0


class TestOiChange:
    def test_reconciles_and_counts_new_listings(self):
        prev = bars_from([legacy_row(oi="100", ts="04-JUL-2024")])
        today = bars_from([legacy_row(oi="130", chg="30"), legacy_row(strike="22050", oi="20", chg="20")])
        r = g.oi_change_reconciliation(prev, today)
        assert r["match_pct"] == 100 and r["exact_both_days"] == 1 and r["exact_new_listing"] == 1

    def test_mismatch_is_caught(self):
        prev = bars_from([legacy_row(oi="100")])
        today = bars_from([legacy_row(oi="130", chg="25")])
        r = g.oi_change_reconciliation(prev, today)
        assert r["mismatch_present_both_days"] == 1 and r["match_pct"] == 0 and r["examples"]

    def test_expired_contracts_are_reported_as_dropped_not_mismatched(self):
        prev = bars_from([legacy_row(oi="100"), legacy_row(strike="22050", oi="40")])
        today = bars_from([legacy_row(oi="100", chg="0")])
        r = g.oi_change_reconciliation(prev, today)
        assert r["match_pct"] == 100 and r["dropped_with_oi"] == 1 and r["dropped_oi_total"] == 40


class TestIdConflicts:
    K1 = g.ContractKey("ESCORTS", "STO", D(2024, 1, 25), 315000, "CE")
    K2 = g.ContractKey("HINDPETRO", "STO", D(2025, 3, 27), 26000, "PE")

    def test_token_reused_for_a_different_contract(self):
        r = g.id_conflicts([("a", self.K1, "114864"), ("b", self.K2, "114864")])
        assert r["ids_mapping_to_several_contracts"] == 1 and r["contracts_with_several_ids"] == 0

    def test_same_contract_under_two_tokens(self):
        r = g.id_conflicts([("a", self.K1, "1"), ("b", self.K1, "2")])
        assert r["contracts_with_several_ids"] == 1 and r["ids_mapping_to_several_contracts"] == 0

    def test_stable(self):
        r = g.id_conflicts([("a", self.K1, "1"), ("b", self.K1, "1"), ("a", self.K2, "2")])
        assert r["ids_mapping_to_several_contracts"] == 0 == r["contracts_with_several_ids"]


class TestSmallHelpers:
    def test_gcd_step(self):
        assert g.gcd_step([5000, 10000, 15000, 5000]) == 5000
        assert g.gcd_step([7000]) is None and g.gcd_step([]) is None

    def test_trade_state_counts(self):
        bars = g.legacy_bars(LEGACY_HEADER, [legacy_row(strike="1", v="5", o="1", h="2", l="1", c="2"), legacy_row(strike="2"), legacy_row(strike="3", c="0", s="0", oi="9")])
        c = g.trade_state_counts(bars)
        assert c["traded"] == 1 and c["not_traded"] == 2 and c["zero_volume_with_oi"] == 1 and c["anomalies"] == {}

    def test_sample_manifest_is_consistent(self):
        keys = [(f, d) for f, d, _ in g.SAMPLE]
        assert len(keys) == len(set(keys))
        for d in g.OVERLAP_DATES:
            assert ("legacy", d) in keys and ("udiff", d) in keys
        for fam, d0, d1 in g.CONSECUTIVE_PAIRS:
            assert (fam, d0) in keys and (fam, d1) in keys and d1 > d0
        for fam, d in g.EXPIRY_SETTLEMENT_DATES:
            assert (fam, d) in keys
        for d in g.MASTER_UDIFF_PAIRS:
            assert ("master", d) in keys and ("udiff", d) in keys
        assert not any(d == D(2023, 6, 29) for _, d, _ in g.SAMPLE)


def zip_bytes(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", text)
    return buf.getvalue()


class FakeResponse:
    def __init__(self, status=200, content=b"", headers=None):
        self.status_code, self.content, self.headers = status, content, headers or {"Content-Type": "application/zip", "Last-Modified": "Fri, 05 Jul 2024 13:38:00 GMT", "ETag": '"abc"'}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, response):
        self.response, self.calls = response, 0

    def get(self, url, headers=None, timeout=None):
        self.calls += 1
        return self.response


CSV = "INSTRUMENT,SYMBOL,HASH\n" + "".join(f"OPTIDX,NIFTY,{hashlib.sha256(str(i).encode()).hexdigest()}\n" for i in range(40))


class TestFetch:
    @pytest.fixture(autouse=True)
    def sandbox(self, tmp_path, monkeypatch):
        monkeypatch.setattr(g, "SAMPLE_ROOT", tmp_path / "raw")

    def test_writes_file_and_sidecar_with_hash_and_headers(self):
        body = zip_bytes(CSV)
        side = g.fetch_one("legacy", D(2024, 7, 5), FakeSession(FakeResponse(200, body)), now=dt.datetime(2026, 9, 21, tzinfo=UTC))
        p = g.sample_path("legacy", D(2024, 7, 5))
        assert p.read_bytes() == body and side["sha256"] == hashlib.sha256(body).hexdigest()
        on_disk = json.loads(g._sidecar(p).read_text())
        assert on_disk == side and side["response_headers"]["Last-Modified"] and side["fetched_at"].startswith("2026-09-21")
        assert side["source_url"].endswith("/2024/JUL/fo05JUL2024bhav.csv.zip")

    def test_second_call_is_idempotent_and_makes_no_request(self):
        s = FakeSession(FakeResponse(200, zip_bytes(CSV)))
        first = g.fetch_one("legacy", D(2024, 7, 5), s)
        assert g.fetch_one("legacy", D(2024, 7, 5), s) == first and s.calls == 1

    def test_tampered_file_is_refused_never_overwritten(self):
        s = FakeSession(FakeResponse(200, zip_bytes(CSV)))
        g.fetch_one("legacy", D(2024, 7, 5), s)
        p = g.sample_path("legacy", D(2024, 7, 5))
        p.write_bytes(b"tampered")
        with pytest.raises(RuntimeError, match="immutable"):
            g.fetch_one("legacy", D(2024, 7, 5), s)
        assert p.read_bytes() == b"tampered" and s.calls == 1

    def test_file_without_sidecar_is_refused(self):
        p = g.sample_path("udiff", D(2024, 7, 5))
        p.parent.mkdir(parents=True)
        p.write_bytes(b"orphan")
        with pytest.raises(RuntimeError, match="without a provenance sidecar"):
            g.fetch_one("udiff", D(2024, 7, 5), FakeSession(FakeResponse(200, zip_bytes(CSV))))
        assert p.read_bytes() == b"orphan"

    def test_404_is_recorded_not_written(self):
        side = g.fetch_one("legacy", D(2023, 6, 29), FakeSession(FakeResponse(404, b"")))
        assert side["available"] is False and not g.sample_path("legacy", D(2023, 6, 29)).exists()

    def test_html_error_page_with_200_is_refused_and_writes_nothing(self):
        r = FakeResponse(200, b"<html>" + b"x" * 1000, {"Content-Type": "text/html"})
        with pytest.raises(RuntimeError, match="unusable body"):
            g.fetch_one("legacy", D(2024, 7, 5), FakeSession(r))
        assert not g.sample_path("legacy", D(2024, 7, 5)).exists()

    def test_master_is_gzip(self):
        body = gzip.compress(CSV.encode())
        r = FakeResponse(200, body, {"Content-Type": "application/x-gzip"})
        side = g.fetch_one("master", D(2026, 9, 18), FakeSession(r))
        assert side["sha256"] == hashlib.sha256(body).hexdigest() and side["source_url"].endswith("NSE_FO_contract_18092026.csv.gz")

    def test_urls_and_paths_for_each_family(self):
        assert g.sample_url("udiff", D(2024, 7, 8)).endswith("/fo/BhavCopy_NSE_FO_0_0_0_20240708_F_0000.csv.zip")
        assert g.sample_url("legacy", D(2015, 1, 5)).endswith("/historical/DERIVATIVES/2015/JAN/fo05JAN2015bhav.csv.zip")
        assert g.sample_path("master", D(2024, 2, 5)).name == "NSE_FO_contract_05022024.csv.gz"
