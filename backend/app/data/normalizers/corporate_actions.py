from __future__ import annotations

import dataclasses
import re

PARSER_VERSION = "stage3.9-c0-v2"

_SPLIT_KW = re.compile(r"\bspl(?:i)?t\b|sub[\s-]*division|\bsub[\s-]*divn\b", re.IGNORECASE)
_BONUS_KW = re.compile(r"\bbonus\b", re.IGNORECASE)
_RIGHTS_KW = re.compile(r"\brights?\b", re.IGNORECASE)
_DEMERGER_KW = re.compile(r"\bdemerger\b|scheme\s+of\s+arrangement", re.IGNORECASE)
_MERGER_KW = re.compile(r"\bmerger\b|amalgamation", re.IGNORECASE)
_DIVIDEND_KW = re.compile(r"\b(?:dividend|divdend|div)\b", re.IGNORECASE)

_CLASS_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("SPLIT", _SPLIT_KW),
    ("BONUS", _BONUS_KW),
    ("RIGHTS", _RIGHTS_KW),
    ("DEMERGER", _DEMERGER_KW),
    ("MERGER", _MERGER_KW),
    ("DIVIDEND", _DIVIDEND_KW),
)

_NUM = r"(\d+(?:\.\d+)?)"
_SPLIT_RATIO_RE = re.compile(
    rf"fr(?:o)?m\s+r[se]\.?\s*{_NUM}\s*(?:/-)?\s*(?:per\s+sh(?:are)?)?\s*to\s+r[se]\.?\s*{_NUM}",
    re.IGNORECASE,
)
_BONUS_RATIO_RE = re.compile(r"\bbonus\b\D*?(\d+)\s*:\s*(\d+)", re.IGNORECASE)
_RIGHTS_RATIO_RE = re.compile(r"\brights?\b\D*?(\d+)\s*:\s*(\d+)", re.IGNORECASE)
_DIVIDEND_AMOUNT_RE = re.compile(
    rf"(?:dividend|divdend|div)\b\D*?r[se]\.?\s*-?\s*{_NUM}\s*(?:/-)?\s*per\b",
    re.IGNORECASE,
)

_COMPONENT_SPLIT_RE = re.compile(r"/(?=[A-Za-z])")
_MULTICLASS_SEPARATORS = (
    re.compile(r"\s+and\s+", re.IGNORECASE),
    re.compile(r"\s*&\s*"),
    re.compile(r"\s*,\s*"),
    re.compile(r"\s*\+\s*"),
)


@dataclasses.dataclass(frozen=True)
class ParsedComponent:
    component_type: str
    raw_text_fragment: str
    face_value_from: float | None = None
    face_value_to: float | None = None
    ratio_new: int | None = None
    ratio_existing: int | None = None
    cash_amount_per_share: float | None = None
    currency: str | None = None
    confidently_parsed: bool = False
    note: str | None = None


def _classes(fragment: str) -> list[str]:
    return [name for name, pattern in _CLASS_PATTERNS if pattern.search(fragment)]


def _split_multiclass(fragment: str) -> list[str] | None:
    wanted = set(_classes(fragment))
    for separator in _MULTICLASS_SEPARATORS:
        parts = [p.strip() for p in separator.split(fragment) if p.strip()]
        if len(parts) < 2:
            continue
        if all(len(_classes(p)) <= 1 for p in parts):
            found = {c for p in parts for c in _classes(p)}
            if found == wanted:
                return parts
    return None


def _parse_single_class_fragment(fragment: str, component_type: str) -> ParsedComponent:
    if component_type == "SPLIT":
        m = _SPLIT_RATIO_RE.search(fragment)
        if m:
            fv_from, fv_to = float(m.group(1)), float(m.group(2))
            if fv_from > 0 and fv_to > 0:
                return ParsedComponent(
                    "SPLIT", fragment, face_value_from=fv_from, face_value_to=fv_to,
                    confidently_parsed=True,
                )
        return ParsedComponent(
            "SPLIT", fragment, note="split recognized but face-value terms not parseable"
        )

    if component_type == "BONUS":
        m = _BONUS_RATIO_RE.search(fragment)
        if m:
            new, existing = int(m.group(1)), int(m.group(2))
            if new > 0 and existing > 0:
                return ParsedComponent(
                    "BONUS", fragment, ratio_new=new, ratio_existing=existing,
                    confidently_parsed=True,
                )
        return ParsedComponent(
            "BONUS", fragment, note="bonus recognized but ratio not parseable"
        )

    if component_type == "RIGHTS":
        m = _RIGHTS_RATIO_RE.search(fragment)
        ratio_new = ratio_existing = None
        if m and int(m.group(1)) > 0 and int(m.group(2)) > 0:
            ratio_new, ratio_existing = int(m.group(1)), int(m.group(2))
        return ParsedComponent(
            "RIGHTS", fragment, ratio_new=ratio_new, ratio_existing=ratio_existing,
            confidently_parsed=False,
            note="rights terms incomplete: issue price is not extracted by this parser "
            "(ratio alone is not a complete rights-issue term set)",
        )

    if component_type == "DIVIDEND":
        m = _DIVIDEND_AMOUNT_RE.search(fragment)
        if m and float(m.group(1)) > 0:
            return ParsedComponent(
                "DIVIDEND", fragment, cash_amount_per_share=float(m.group(1)),
                currency="INR", confidently_parsed=True,
            )
        return ParsedComponent(
            "DIVIDEND", fragment, note="dividend recognized but per-share amount not parseable"
        )

    return ParsedComponent(component_type, fragment)


def parse_subject_components(subject: str) -> list[ParsedComponent]:
    fragments = [f.strip() for f in _COMPONENT_SPLIT_RE.split(subject) if f.strip()]
    if not fragments:
        fragments = [subject.strip()]

    components: list[ParsedComponent] = []
    for fragment in fragments:
        classes = _classes(fragment)
        if len(classes) > 1:
            parts = _split_multiclass(fragment)
            if parts is None:
                components.append(
                    ParsedComponent(
                        "OTHER", fragment,
                        note="fragment names multiple distinct action classes "
                        f"({', '.join(classes)}) and could not be cleanly separated - "
                        "preserved whole rather than classified as one of them",
                    )
                )
                continue
            for part in parts:
                part_classes = _classes(part)
                components.append(
                    _parse_single_class_fragment(part, part_classes[0])
                    if part_classes
                    else ParsedComponent("OTHER", part)
                )
            continue

        components.append(
            _parse_single_class_fragment(fragment, classes[0])
            if classes
            else ParsedComponent("OTHER", fragment)
        )
    return components
