from __future__ import annotations

import dataclasses

from backend.app.config import load_dotenv
from backend.app.data.raw.artifact import check_component
import os


class MissingCredentialsError(Exception):

    def __init__(self, provider: str, missing: list[str]) -> None:
        self.provider = provider
        self.missing = missing
        super().__init__(
            f"{provider}: missing {', '.join(missing)}. Set them in your (gitignored) .env - see .env.example for how to obtain each value. "
            f"Nothing was contacted."
        )


def _require(provider: str, **vars_) -> dict[str, str]:
    load_dotenv()
    missing = [k for k, v in vars_.items() if not v]
    if missing:
        raise MissingCredentialsError(provider, missing)
    return vars_


@dataclasses.dataclass(frozen=True)
class UpstoxCredentials:
    access_token: str

    @classmethod
    def from_env(cls) -> "UpstoxCredentials":
        v = _require("upstox", UPSTOX_ACCESS_TOKEN=os.environ.get("UPSTOX_ACCESS_TOKEN", "").strip())
        return cls(access_token=v["UPSTOX_ACCESS_TOKEN"])

    def auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}


@dataclasses.dataclass(frozen=True)
class DhanCredentials:
    client_id: str
    access_token: str

    @classmethod
    def from_env(cls) -> "DhanCredentials":
        v = _require("dhan", DHAN_CLIENT_ID=os.environ.get("DHAN_CLIENT_ID", "").strip(), DHAN_ACCESS_TOKEN=os.environ.get("DHAN_ACCESS_TOKEN", "").strip())
        return cls(client_id=v["DHAN_CLIENT_ID"], access_token=v["DHAN_ACCESS_TOKEN"])

    def auth_headers(self) -> dict[str, str]:
        return {"access-token": self.access_token, "dhanClientId": self.client_id}


@dataclasses.dataclass(frozen=True)
class BreezeCredentials:
    api_key: str
    api_secret: str
    session_token: str

    @classmethod
    def from_env(cls) -> "BreezeCredentials":
        v = _require(
            "breeze", BREEZE_API_KEY=os.environ.get("BREEZE_API_KEY", "").strip(), BREEZE_API_SECRET=os.environ.get("BREEZE_API_SECRET", "").strip(),
            BREEZE_SESSION_TOKEN=os.environ.get("BREEZE_SESSION_TOKEN", "").strip(),
        )
        return cls(api_key=v["BREEZE_API_KEY"], api_secret=v["BREEZE_API_SECRET"], session_token=v["BREEZE_SESSION_TOKEN"])


def redact(text: str, *secrets: str) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "<redacted>")
    return text


safe_key = check_component
