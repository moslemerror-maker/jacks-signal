from __future__ import annotations

import dataclasses
import re

from backend.app.config import load_dotenv

from .provider_credentials import DhanCredentials, MissingCredentialsError

PreflightStatus = "OK", "MISSING", "MALFORMED"
_JWT_SHAPE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")


@dataclasses.dataclass(frozen=True)
class CredentialPreflightResult:
    provider: str
    status: str
    detail: str

    def __post_init__(self) -> None:
        if self.status not in PreflightStatus:
            raise ValueError(f"status must be one of {PreflightStatus}, got {self.status!r}")


def dhan_credential_preflight() -> CredentialPreflightResult:
    load_dotenv()
    try:
        creds = DhanCredentials.from_env()
    except MissingCredentialsError as exc:
        return CredentialPreflightResult("dhan", "MISSING", str(exc))
    if not creds.client_id.isdigit():
        return CredentialPreflightResult("dhan", "MALFORMED", "DHAN_CLIENT_ID does not look numeric (Dhan client IDs are numeric)")
    if not _JWT_SHAPE.match(creds.access_token):
        return CredentialPreflightResult("dhan", "MALFORMED", "DHAN_ACCESS_TOKEN does not look like a JWT (expected three dot-separated segments)")
    return CredentialPreflightResult(
        "dhan", "OK",
        "DHAN_CLIENT_ID/DHAN_ACCESS_TOKEN present and structurally plausible. Actual validity (not expired/revoked) "
        "is NOT verified here - Dhan tokens last 24h; this requires one real, separately-approved API call.",
    )
