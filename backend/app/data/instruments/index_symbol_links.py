from __future__ import annotations

INDEX_SYMBOL_SUCCESSOR_LINKS: dict[str, tuple[str, str]] = {
    "NIFTY 50": (
        "CNX NIFTY",
        "NSE renamed CNX NIFTY to NIFTY 50 effective 2015-11-09 (observed: "
        "indicesHistory returns 'CNX NIFTY' for 2015-01-01..2015-11-06 and "
        "'NIFTY 50' from 2015-11-09 onward)",
    ),
    "NIFTY BANK": (
        "BANK NIFTY",
        "NSE renamed BANK NIFTY to NIFTY BANK effective 2015-11-09 (same "
        "transition as CNX NIFTY -> NIFTY 50, observed in the same data)",
    ),
}
