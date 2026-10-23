from __future__ import annotations

import io
from typing import Mapping, Sequence

import pandas as pd
import requests

COMPACT_CSV_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
_TIMEOUT_SECONDS = 30
_MAINBOARD_EQUITY = {"SEM_EXM_EXCH_ID": "NSE", "SEM_INSTRUMENT_NAME": "EQUITY", "SEM_SERIES": "EQ"}


class SymbolNotFoundError(Exception):

    def __init__(self, symbols: Sequence[str]) -> None:
        self.symbols = list(symbols)
        super().__init__(f"not found in the NSE mainboard equity security master: {self.symbols}")


def fetch_nse_mainboard_equity_master(session: requests.Session | None = None) -> pd.DataFrame:
    s = session or requests
    r = s.get(COMPACT_CSV_URL, timeout=_TIMEOUT_SECONDS)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content), low_memory=False)
    mask = True
    for column, value in _MAINBOARD_EQUITY.items():
        mask = mask & (df[column] == value)
    return df[mask]


def resolve_security_ids(symbols: Sequence[str], master: pd.DataFrame) -> Mapping[str, int]:
    by_symbol = master.drop_duplicates(subset="SEM_TRADING_SYMBOL", keep="first").set_index("SEM_TRADING_SYMBOL")["SEM_SMST_SECURITY_ID"]
    missing = [s for s in symbols if s not in by_symbol.index]
    if missing:
        raise SymbolNotFoundError(missing)
    return {s: int(by_symbol.loc[s]) for s in symbols}
