from __future__ import annotations

import re
from typing import Literal, Sequence

FetchFailureCategory = Literal[
    "AUTH_FAILURE_SUSPECTED", "TIMEOUT_SUSPECTED", "CONNECTION_FAILURE_SUSPECTED", "PROVIDER_FAILURE_UNKNOWN",
]
FETCH_FAILURE_CATEGORIES: tuple[FetchFailureCategory, ...] = (
    "AUTH_FAILURE_SUSPECTED", "TIMEOUT_SUSPECTED", "CONNECTION_FAILURE_SUSPECTED", "PROVIDER_FAILURE_UNKNOWN",
)

_ALL_NONE_REMARKS = re.compile(
    r"'error_code':\s*None.*?'error_type':\s*None.*?'error_message':\s*None|"
    r"'error_type':\s*None.*?'error_code':\s*None.*?'error_message':\s*None",
    re.DOTALL,
)
_TIMEOUT_SIGNATURES = ("timeout", "timed out", "readtimeout", "connecttimeout")
_CONNECTION_SIGNATURES = (
    "connectionerror", "max retries exceeded", "failed to establish a new connection", "connection refused", "connection aborted",
)


def classify_fetch_failure(detail: str) -> FetchFailureCategory:
    if _ALL_NONE_REMARKS.search(detail):
        return "AUTH_FAILURE_SUSPECTED"
    lowered = detail.lower()
    if any(sig in lowered for sig in _TIMEOUT_SIGNATURES):
        return "TIMEOUT_SUSPECTED"
    if any(sig in lowered for sig in _CONNECTION_SIGNATURES):
        return "CONNECTION_FAILURE_SUSPECTED"
    return "PROVIDER_FAILURE_UNKNOWN"


def ended_on_failure_streak(outcomes: Sequence[str], *, threshold: int | None) -> bool:
    if not threshold or threshold < 1 or len(outcomes) < threshold:
        return False
    return all(o == "FETCH_FAILURE" for o in outcomes[-threshold:])
