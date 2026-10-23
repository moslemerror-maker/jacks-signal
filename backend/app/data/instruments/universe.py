from __future__ import annotations

from backend.app.data.canonical.schema import CanonicalBar

INDEX_SYMBOLS = ("NIFTY 50", "NIFTY BANK")

EQUITY_BASKET = (
    "RELIANCE",
    "BHARTIARTL",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "TCS",
    "BAJFINANCE",
    "LT",
    "HINDUNILVR",
    "TITAN",
    "SUNPHARMA",
    "INFY",
    "KOTAKBANK",
    "ADANIENT",
    "ADANIPORTS",
    "MARUTI",
    "AXISBANK",
    "M&M",
    "HCLTECH",
    "ITC",
    "ULTRACEMCO",
    "NTPC",
    "BAJAJ-AUTO",
    "ETERNAL",
    "JSWSTEEL",
)

EQUITY_SERIES = frozenset({"EQ", "BE", "BZ"})


def is_equity_series(series: str) -> bool:
    return series in EQUITY_SERIES


def resolve_basket_isins(reference_bars: list[CanonicalBar]) -> dict[str, str]:
    basket = set(EQUITY_BASKET)
    result: dict[str, str] = {}
    for bar in reference_bars:
        if bar.symbol in basket and is_equity_series(bar.series) and bar.isin:
            result[bar.symbol] = bar.isin

    missing = basket - result.keys()
    if missing:
        raise ValueError(
            f"could not resolve ISIN for basket symbols on the reference day: "
            f"{sorted(missing)} - use a more recent reference date, or check for "
            "a typo/rename in EQUITY_BASKET"
        )
    return result
