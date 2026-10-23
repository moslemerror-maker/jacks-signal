from __future__ import annotations

from typing import Literal, get_args

ProvenanceClass = Literal["DEVELOPMENT_FIXTURE", "PRODUCTION_RESEARCH_DATASET"]

PROVENANCE_CLASSES: tuple[str, ...] = get_args(ProvenanceClass)


def check_provenance_class(value: object) -> ProvenanceClass:
    if value not in PROVENANCE_CLASSES:
        raise ValueError(f"provenance_class must be one of {PROVENANCE_CLASSES}, got {value!r}")
    return value  # type: ignore[return-value]
