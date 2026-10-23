from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Literal, Sequence, get_args

from .provenance import ProvenanceClass, check_provenance_class

DerivativeClass = Literal["IDF", "STF", "IDO", "STO"]
_DERIVATIVE_CLASSES: tuple[str, ...] = get_args(DerivativeClass)

OptionType = Literal["CE", "PE", "CA", "PA"]
_OPTION_TYPES: tuple[str, ...] = get_args(OptionType)

_OPTIONS_CLASSES = ("IDO", "STO")
_FUTURES_CLASSES = ("IDF", "STF")


@dataclasses.dataclass(frozen=True, slots=True)
class ContractIdentity:

    underlying: str
    derivative_class: DerivativeClass
    expiry: dt.date
    strike_paise: int | None
    option_type: OptionType | None

    def __post_init__(self) -> None:
        if not self.underlying or not self.underlying.strip():
            raise ValueError("underlying must be a non-empty symbol")
        if self.derivative_class not in _DERIVATIVE_CLASSES:
            raise ValueError(f"derivative_class must be one of {_DERIVATIVE_CLASSES}, got {self.derivative_class!r}")
        is_option = self.derivative_class in _OPTIONS_CLASSES
        if is_option:
            if self.strike_paise is None or self.strike_paise <= 0:
                raise ValueError(f"{self.derivative_class} (an option) requires a positive integer strike_paise, got {self.strike_paise!r}")
            if not isinstance(self.strike_paise, int) or isinstance(self.strike_paise, bool):
                raise ValueError(f"strike_paise must be an exact int (never a float), got {type(self.strike_paise).__name__}")
            if self.option_type not in _OPTION_TYPES:
                raise ValueError(f"{self.derivative_class} (an option) requires option_type in {_OPTION_TYPES}, got {self.option_type!r}")
        else:
            if self.strike_paise is not None:
                raise ValueError(f"{self.derivative_class} (a future) must have strike_paise=None, got {self.strike_paise!r}")
            if self.option_type is not None:
                raise ValueError(f"{self.derivative_class} (a future) must have option_type=None, got {self.option_type!r}")

    @property
    def is_option(self) -> bool:
        return self.derivative_class in _OPTIONS_CLASSES

    @property
    def strike_rupees(self) -> str:
        if self.strike_paise is None:
            raise ValueError("no strike on a futures contract")
        rupees, paise = divmod(self.strike_paise, 100)
        return f"{rupees}.{paise:02d}"


@dataclasses.dataclass(frozen=True, slots=True)
class ContractSpec:

    contract: ContractIdentity
    lot_size: int
    effective_from: dt.date
    effective_to: dt.date | None
    provenance_class: ProvenanceClass
    tick_size_paise: int | None = None
    settlement_method: Literal["C", "P"] | None = None
    exercise_style: Literal["E", "A"] | None = None

    def __post_init__(self) -> None:
        check_provenance_class(self.provenance_class)
        if self.lot_size <= 0:
            raise ValueError(f"lot_size must be positive, got {self.lot_size!r}")
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError(f"effective_to ({self.effective_to}) is before effective_from ({self.effective_from})")
        if self.tick_size_paise is not None and self.tick_size_paise <= 0:
            raise ValueError(f"tick_size_paise must be positive if given, got {self.tick_size_paise!r}")

    def covers(self, as_of: dt.date) -> bool:
        return self.effective_from <= as_of and (self.effective_to is None or as_of <= self.effective_to)


def spec_as_of(specs: Sequence[ContractSpec], as_of: dt.date) -> ContractSpec | None:
    covering = [s for s in specs if s.covers(as_of)]
    if len(covering) > 1:
        raise ValueError(f"{len(covering)} ContractSpec entries cover {as_of} for the same contract - overlapping effective windows: {covering}")
    return covering[0] if covering else None
