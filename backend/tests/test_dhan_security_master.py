from __future__ import annotations

import io

import pandas as pd
import pytest

from backend.app.data.sources import dhan_security_master as dsm

_COLUMNS = ["SEM_EXM_EXCH_ID", "SEM_SEGMENT", "SEM_SMST_SECURITY_ID", "SEM_INSTRUMENT_NAME", "SEM_TRADING_SYMBOL", "SEM_SERIES"]
_ROWS = [
    ("NSE", "E", 2885, "EQUITY", "RELIANCE", "EQ"),
    ("NSE", "E", 11536, "EQUITY", "TCS", "EQ"),
    ("NSE", "E", 14366, "EQUITY", "IDEA", "EQ"),
    ("NSE", "E", 99001, "EQUITY", "SMESTOCK", "SM"),
    ("BSE", "E", 99002, "EQUITY", "RELIANCE", "EQ"),
    ("NSE", "C", 99003, "FUTCUR", "USDINR", "XX"),
]


def _csv_bytes() -> bytes:
    df = pd.DataFrame(_ROWS, columns=_COLUMNS)
    return df.to_csv(index=False).encode()


class FakeResponse:
    def __init__(self, content: bytes, status: int = 200):
        self.content, self.status_code = content, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, content: bytes):
        self.content, self.requests = content, []

    def get(self, url, timeout=None):
        self.requests.append((url, timeout))
        return FakeResponse(self.content)


class TestFetchMainboardEquityMaster:
    def test_filters_to_nse_mainboard_eq_only(self):
        master = dsm.fetch_nse_mainboard_equity_master(session=FakeSession(_csv_bytes()))
        assert sorted(master["SEM_TRADING_SYMBOL"]) == ["IDEA", "RELIANCE", "TCS"]
        assert set(master["SEM_EXM_EXCH_ID"]) == {"NSE"} and set(master["SEM_SERIES"]) == {"EQ"}

    def test_uses_the_documented_compact_csv_url(self):
        s = FakeSession(_csv_bytes())
        dsm.fetch_nse_mainboard_equity_master(session=s)
        assert s.requests[0][0] == dsm.COMPACT_CSV_URL


class TestResolveSecurityIds:
    def test_resolves_known_symbols_to_real_security_ids(self):
        master = dsm.fetch_nse_mainboard_equity_master(session=FakeSession(_csv_bytes()))
        resolved = dsm.resolve_security_ids(["RELIANCE", "TCS", "IDEA"], master)
        assert resolved == {"RELIANCE": 2885, "TCS": 11536, "IDEA": 14366}

    def test_unknown_symbol_raises_and_names_every_missing_one(self):
        master = dsm.fetch_nse_mainboard_equity_master(session=FakeSession(_csv_bytes()))
        with pytest.raises(dsm.SymbolNotFoundError) as exc:
            dsm.resolve_security_ids(["RELIANCE", "NOTREAL", "ALSOFAKE"], master)
        assert exc.value.symbols == ["NOTREAL", "ALSOFAKE"]

    def test_sme_series_symbol_is_not_resolvable_even_though_it_shares_exchange_and_instrument_type(self):
        master = dsm.fetch_nse_mainboard_equity_master(session=FakeSession(_csv_bytes()))
        with pytest.raises(dsm.SymbolNotFoundError):
            dsm.resolve_security_ids(["SMESTOCK"], master)
