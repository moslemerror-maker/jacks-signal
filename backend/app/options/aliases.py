from __future__ import annotations

import dataclasses
import datetime as dt

from .contracts import ContractIdentity
from .provenance import ProvenanceClass, check_provenance_class


@dataclasses.dataclass(frozen=True, slots=True)
class ContractAlias:

    contract: ContractIdentity
    source: str
    token: str
    first_seen: dt.date
    last_seen: dt.date
    provenance_class: ProvenanceClass

    def __post_init__(self) -> None:
        check_provenance_class(self.provenance_class)
        if not self.source or not self.source.strip():
            raise ValueError("source must be a non-empty provider name")
        if not self.token or not self.token.strip():
            raise ValueError("token must be a non-empty source identifier")
        if self.last_seen < self.first_seen:
            raise ValueError(f"last_seen ({self.last_seen}) is before first_seen ({self.first_seen})")


def resolve(aliases: list[ContractAlias], *, source: str, token: str, as_of: dt.date) -> ContractIdentity | None:
    matches = [a for a in aliases if a.source == source and a.token == token and a.first_seen <= as_of <= a.last_seen]
    if len(matches) > 1:
        raise ValueError(f"{len(matches)} aliases for ({source!r}, {token!r}) cover {as_of} - ambiguous token, cannot resolve: {matches}")
    return matches[0].contract if matches else None
