from __future__ import annotations

import dataclasses
import math
from decimal import Decimal
from fractions import Fraction

SUPPORTED_KINDS = ("SPLIT", "BONUS")
TERM_SCALE = 6
_MAX_ABS = Decimal("100000000000000")


class InvalidTerms(ValueError):
    pass


def _to_decimal(x) -> Decimal:
    if isinstance(x, bool) or x is None:
        raise InvalidTerms(f"not a number: {x!r}")
    if isinstance(x, Decimal):
        d = x
    elif isinstance(x, int):
        d = Decimal(x)
    elif isinstance(x, float):
        if not math.isfinite(x):
            raise InvalidTerms(f"not finite: {x!r}")
        d = Decimal(repr(x))
    else:
        raise InvalidTerms(f"unsupported type {type(x).__name__}: {x!r}")
    if not d.is_finite():
        raise InvalidTerms(f"not finite: {x!r}")
    return d


def canonical_decimal(d: Decimal) -> str:
    d = d.normalize()
    text = format(d, "f")
    return text


@dataclasses.dataclass(frozen=True)
class ExactTerms:
    kind: str
    term_a: Decimal
    term_b: Decimal

    @property
    def a_text(self) -> str:
        return canonical_decimal(self.term_a)

    @property
    def b_text(self) -> str:
        return canonical_decimal(self.term_b)

    @property
    def ratio(self) -> Fraction:
        return Fraction(self.term_a) / Fraction(self.term_b)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "term_a": self.a_text, "term_b": self.b_text}

    def statement(self) -> str:
        if self.kind == "SPLIT":
            return f"SPLIT old face value {self.a_text} -> new face value {self.b_text}"
        return f"BONUS {self.a_text} new for every {self.b_text} existing"


def normalize_terms(kind: str, a, b) -> ExactTerms:
    if kind not in SUPPORTED_KINDS:
        raise InvalidTerms(f"unsupported kind {kind!r}")
    da, db = _to_decimal(a), _to_decimal(b)
    for name, d in (("term_a", da), ("term_b", db)):
        if d <= 0:
            raise InvalidTerms(f"{name} must be positive, got {d}")
        if d > _MAX_ABS:
            raise InvalidTerms(f"{name} is implausibly large: {d}")
        if d != d.quantize(Decimal(1).scaleb(-TERM_SCALE)):
            raise InvalidTerms(f"{name} has more than {TERM_SCALE} decimal places: {d}")
    if kind == "BONUS":
        if da != da.to_integral_value() or db != db.to_integral_value():
            raise InvalidTerms(f"bonus ratio must be whole numbers, got {da}:{db}")
    if kind == "SPLIT" and da == db:
        raise InvalidTerms(f"a split with identical old/new face value {da} is a no-op and almost certainly a parse error")
    return ExactTerms(kind, da.normalize(), db.normalize())


def try_normalize(kind: str, a, b) -> tuple[ExactTerms | None, str | None]:
    try:
        return normalize_terms(kind, a, b), None
    except InvalidTerms as exc:
        return None, str(exc)


def terms_equal(x: ExactTerms | None, y: ExactTerms | None) -> bool:
    return x is not None and y is not None and x.kind == y.kind and x.term_a == y.term_a and x.term_b == y.term_b
