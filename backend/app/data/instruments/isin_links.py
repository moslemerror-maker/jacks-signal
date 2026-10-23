from __future__ import annotations

ISIN_SUCCESSOR_LINKS: dict[str, tuple[str, str]] = {
    "INE296A01024": (
        "INE296A01016",
        "BAJFINANCE: ISIN changed 2016-09-09 (confirmed by full-history scan "
        "of raw bhavcopy files, 2026-09-14) - consistent with a face-value-"
        "changing stock split. This was the missing link: the chain below "
        "(INE296A01032 -> INE296A01024) was already known, but nothing "
        "linked INE296A01024 back to 2015's actual ISIN, leaving "
        "2015-01-01..2016-09-08 (~1 year 8 months) silently excluded from "
        "the historical backfill.",
    ),
    "INE296A01032": (
        "INE296A01024",
        "BAJFINANCE: ISIN changed 2025-06-16 (confirmed by full-history scan "
        "of raw bhavcopy files, 2026-09-14; date refined from the earlier "
        "'between 2024-07-05 and 2025-09-15' estimate), consistent with a "
        "face-value-changing stock split.",
    ),
    "INE237A01036": (
        "INE237A01028",
        "KOTAKBANK: ISIN changed 2026-01-14 (confirmed by exhaustive full-"
        "history scan of all 25 basket stocks, 2026-09-14 hardening pass; "
        "date refined from the original 'between 2025-09-15 and 2026-09-11' "
        "fixture-comparison estimate), consistent with a face-value-changing "
        "stock split (price dropped ~1970 to ~419, ~1:4.7). Chain confirmed "
        "complete: exactly one transition, continuous coverage both sides, "
        "no earlier undiscovered predecessor in 2015-2026.",
    ),
    "INE040A01034": (
        "INE040A01026",
        "HDFCBANK: ISIN changed 2019-09-20 (confirmed by full-history scan "
        "of raw bhavcopy files, 2026-09-14) - consistent with a face-value-"
        "changing stock split. Previously undiscovered entirely: HDFCBANK "
        "had no link at all, silently excluding 2015-01-01..2019-09-19 "
        "(~4 years 8 months) of history for a top-5-weight NIFTY 50 "
        "constituent.",
    ),
    "INE019A01038": (
        "INE019A01020",
        "JSWSTEEL: ISIN changed 2017-01-05 (confirmed by full-history scan "
        "of raw bhavcopy files, 2026-09-14) - consistent with a face-value-"
        "changing stock split. Previously undiscovered entirely: JSWSTEEL "
        "had no link at all, silently excluding 2015-01-01..2017-01-04 "
        "(~2 years) of history.",
    ),
}


def expand_isin_chain(isin: str) -> set[str]:
    chain = {isin}
    current = isin
    while current in ISIN_SUCCESSOR_LINKS:
        predecessor, _reason = ISIN_SUCCESSOR_LINKS[current]
        if predecessor in chain:
            raise ValueError(f"cycle detected in ISIN_SUCCESSOR_LINKS at {predecessor!r}")
        chain.add(predecessor)
        current = predecessor
    return chain


def all_linked_isins(isin: str) -> set[str]:
    chain = {isin}
    frontier = {isin}
    while frontier:
        next_frontier: set[str] = set()
        for current in frontier:
            if current in ISIN_SUCCESSOR_LINKS:
                predecessor, _reason = ISIN_SUCCESSOR_LINKS[current]
                if predecessor not in chain:
                    next_frontier.add(predecessor)
            for successor, (predecessor, _reason) in ISIN_SUCCESSOR_LINKS.items():
                if predecessor == current and successor not in chain:
                    next_frontier.add(successor)
        chain |= next_frontier
        frontier = next_frontier
    return chain
