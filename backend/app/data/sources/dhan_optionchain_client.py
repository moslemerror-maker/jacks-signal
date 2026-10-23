from __future__ import annotations

import dataclasses
from typing import Any

from .provider_credentials import DhanCredentials

NIFTY_SECURITY_ID = 13
NIFTY_SEGMENT = "IDX_I"


class DhanOptionChainError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class OptionChainRawResult:
    expiry_list_response: dict[str, Any]
    option_chain_response: dict[str, Any]
    expiry_used: str
    expiry_selection: str


def expiry_dates_from_response(el: dict) -> list[str]:
    data = (el or {}).get("data")
    if isinstance(data, dict):
        data = data.get("data")
    return sorted(data) if isinstance(data, list) else []


class DhanOptionChainClient:
    def __init__(self, credentials: DhanCredentials, sdk_client: Any | None = None) -> None:
        self._creds = credentials
        if sdk_client is not None:
            self._sdk = sdk_client
        else:
            from dhanhq import DhanContext, dhanhq
            self._sdk = dhanhq(DhanContext(credentials.client_id, credentials.access_token))

    def expiry_list(self, security_id: int, segment: str) -> dict[str, Any]:
        return self._sdk.expiry_list(security_id, segment)

    def option_chain(self, security_id: int, segment: str, expiry: str) -> dict[str, Any]:
        return self._sdk.option_chain(security_id, segment, expiry)

    def fetch_chain(self, security_id: int, segment: str, *, expiry: str | None = None) -> OptionChainRawResult:
        el = self.expiry_list(security_id, segment)
        if (el or {}).get("status") != "success":
            raise DhanOptionChainError(f"expiry_list did not return status=success: {el!r}")
        dates = expiry_dates_from_response(el)
        if expiry is None:
            if not dates:
                raise DhanOptionChainError(f"expiry_list returned no dates: {el!r}")
            expiry, selection = dates[0], "NEAREST_FROM_EXPIRY_LIST"
        else:
            if expiry not in dates:
                raise DhanOptionChainError(f"requested expiry {expiry!r} is not in expiry_list's own dates {dates!r} - refusing to guess it is valid")
            selection = "EXPLICIT"
        chain = self.option_chain(security_id, segment, expiry)
        if (chain or {}).get("status") != "success":
            raise DhanOptionChainError(f"option_chain did not return status=success for expiry={expiry}: {chain!r}")
        return OptionChainRawResult(expiry_list_response=el, option_chain_response=chain, expiry_used=expiry, expiry_selection=selection)
