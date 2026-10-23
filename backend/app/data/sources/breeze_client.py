from __future__ import annotations

import dataclasses
from typing import Any

from .provider_credentials import BreezeCredentials

INTERVALS = ("1minute", "5minute", "30minute", "1day")
RIGHTS = ("call", "put")


class BreezePackageNotInstalled(RuntimeError):
    def __init__(self) -> None:
        super().__init__("the 'breeze-connect' package is not installed. Run: pip install breeze-connect (or add it to pyproject.toml when this phase begins).")


@dataclasses.dataclass(frozen=True)
class HistoricalOptionsRequest:
    stock_code: str
    expiry_date: str
    strike_price: str
    right: str
    interval: str
    from_date: str
    to_date: str
    exchange_code: str = "NFO"
    product_type: str = "options"

    def __post_init__(self) -> None:
        if self.interval not in INTERVALS:
            raise ValueError(f"interval must be one of {INTERVALS}, got {self.interval!r}")
        if self.right not in RIGHTS:
            raise ValueError(f"right must be one of {RIGHTS}, got {self.right!r}")

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "interval": self.interval, "from_date": self.from_date, "to_date": self.to_date, "stock_code": self.stock_code,
            "exchange_code": self.exchange_code, "product_type": self.product_type, "expiry_date": self.expiry_date, "right": self.right,
            "strike_price": self.strike_price,
        }


class BreezeClient:

    def __init__(self, credentials: BreezeCredentials, sdk_factory=None) -> None:
        self._creds = credentials
        if sdk_factory is not None:
            self._breeze = sdk_factory()
        else:
            try:
                from breeze_connect import BreezeConnect  # type: ignore[import-not-found]
            except ImportError as exc:
                raise BreezePackageNotInstalled() from exc
            self._breeze = BreezeConnect(api_key=credentials.api_key)
        self._session_generated = False

    def _ensure_session(self) -> None:
        if not self._session_generated:
            self._breeze.generate_session(api_secret=self._creds.api_secret, session_token=self._creds.session_token)
            self._session_generated = True

    def get_option_historical(self, request: HistoricalOptionsRequest) -> Any:
        self._ensure_session()
        return self._breeze.get_historical_data_v2(**request.as_kwargs())
