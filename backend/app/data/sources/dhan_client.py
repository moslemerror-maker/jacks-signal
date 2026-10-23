from __future__ import annotations

import dataclasses
from typing import Any, Literal, Sequence, get_args

import requests

from .provider_credentials import DhanCredentials

BASE = "https://api.dhan.co/v2"
_TIMEOUT_SECONDS = 30
_INTERVALS = ("1", "5", "15", "25", "60")
REQUIRED_DATA = ("open", "high", "low", "close", "iv", "volume", "strike", "oi", "spot")

NIFTY_SECURITY_ID = 13
NIFTY_EXCHANGE_SEGMENT = "NSE_FNO"
NIFTY_INSTRUMENT = "OPTIDX"

ExpiryCode = Literal[1, 2, 3]
_EXPIRY_CODES: tuple[int, ...] = get_args(ExpiryCode)


def _validate_expiry_code(expiry_code: int) -> None:
    if expiry_code not in _EXPIRY_CODES:
        raise ValueError(
            f"expiry_code must be one of {_EXPIRY_CODES} (1=Near Expiry, 2=Next Expiry, 3=Far Expiry per Dhan's "
            f"official contract) - 0 is not a valid value and previously caused HTTP 400 DH-905 on every request "
            f"(docs/40); got {expiry_code!r}"
        )


@dataclasses.dataclass(frozen=True)
class ApiResponse:
    status_code: int
    content: bytes
    headers: dict[str, str | None]
    json: Any | None


def rolling_option_body(
    *, security_id: str, exchange_segment: str, instrument: str, expiry_code: ExpiryCode, expiry_flag: str, strike: str, option_type: str,
    from_date: str, to_date: str, interval: str = "5", required_data: Sequence[str] = REQUIRED_DATA,
) -> dict[str, Any]:
    _validate_expiry_code(expiry_code)
    if interval not in _INTERVALS:
        raise ValueError(f"interval must be one of {_INTERVALS}, got {interval!r}")
    if expiry_flag not in ("WEEK", "MONTH"):
        raise ValueError(f"expiryFlag must be WEEK or MONTH, got {expiry_flag!r}")
    if option_type not in ("CALL", "PUT"):
        raise ValueError(f"drvOptionType must be CALL or PUT, got {option_type!r}")
    return {
        "exchangeSegment": exchange_segment, "interval": interval, "securityId": security_id, "instrument": instrument, "expiryCode": expiry_code,
        "expiryFlag": expiry_flag, "strike": strike, "drvOptionType": option_type, "requiredData": list(required_data), "fromDate": from_date, "toDate": to_date,
    }


class DhanClient:

    def __init__(self, credentials: DhanCredentials, session: requests.Session | None = None) -> None:
        self._creds = credentials
        self._session = session or requests.Session()

    def rolling_option(self, **kwargs: Any) -> ApiResponse:
        body = rolling_option_body(**kwargs)
        r = self._session.post(f"{BASE}/charts/rollingoption", json=body, headers={**self._creds.auth_headers(), "Content-Type": "application/json"}, timeout=_TIMEOUT_SECONDS)
        try:
            parsed = r.json()
        except ValueError:
            parsed = None
        return ApiResponse(r.status_code, r.content, dict(r.headers), parsed)


def expired_options_data_via_sdk(
    sdk_client: Any, *, security_id: int, exchange_segment: str, instrument_type: str, expiry_flag: str,
    expiry_code: ExpiryCode, strike: str, option_type: str, required_data: Sequence[str],
    from_date: str, to_date: str, interval: int = 1,
) -> dict[str, Any]:
    _validate_expiry_code(expiry_code)
    if expiry_flag not in ("WEEK", "MONTH"):
        raise ValueError(f"expiry_flag must be WEEK or MONTH, got {expiry_flag!r}")
    if option_type not in ("CALL", "PUT"):
        raise ValueError(f"option_type must be CALL or PUT, got {option_type!r}")
    return sdk_client.expired_options_data(
        security_id=security_id, exchange_segment=exchange_segment, instrument_type=instrument_type,
        expiry_flag=expiry_flag, expiry_code=expiry_code, strike=strike, drv_option_type=option_type,
        required_data=list(required_data), from_date=from_date, to_date=to_date, interval=interval,
    )
