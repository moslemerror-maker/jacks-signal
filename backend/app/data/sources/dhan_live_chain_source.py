from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json

from backend.app.options.expiry_eligibility import select_eligible_expiry
from backend.app.options.live_session import ChainSnapshot
from backend.app.options.provenance import ProvenanceClass

from .dhan_live_quote import parse_dhan_optionchain
from .dhan_optionchain_client import DhanOptionChainClient, DhanOptionChainError, expiry_dates_from_response


@dataclasses.dataclass(frozen=True, slots=True)
class DhanLiveChainSource:
    client: DhanOptionChainClient
    security_id: int
    segment: str
    underlying: str
    provenance_class: ProvenanceClass = "PRODUCTION_RESEARCH_DATASET"

    def fetch_snapshot(self, *, as_of: dt.datetime) -> ChainSnapshot:
        el = self.client.expiry_list(self.security_id, self.segment)
        if (el or {}).get("status") != "success":
            raise DhanOptionChainError(f"expiry_list did not return status=success: {el!r}")
        dates = [dt.date.fromisoformat(d) for d in expiry_dates_from_response(el)]
        eligible_expiry = select_eligible_expiry(dates, as_of=as_of)

        chain = self.client.option_chain(self.security_id, self.segment, eligible_expiry.isoformat())
        if (chain or {}).get("status") != "success":
            raise DhanOptionChainError(f"option_chain did not return status=success for expiry={eligible_expiry}: {chain!r}")

        raw_bytes = json.dumps(chain, default=str, sort_keys=True).encode()
        digest = hashlib.sha256(raw_bytes).hexdigest()
        quotes = parse_dhan_optionchain(
            chain, underlying=self.underlying, expiry=eligible_expiry, observed_at=as_of,
            source_row_hash=digest, provenance_class=self.provenance_class,
        )
        return ChainSnapshot(
            source="dhan", underlying=self.underlying, observed_at=as_of, expiry_list_dates=tuple(dates),
            quotes=tuple(quotes), raw_response_sha256=digest,
        )
