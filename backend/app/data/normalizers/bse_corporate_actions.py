from __future__ import annotations

import dataclasses
import re

_NUM = r"(\d+(?:\.\d+)?)"

_SPLIT_KW = re.compile(r"\bspl(?:i)?t\b|sub[\s-]*division", re.IGNORECASE)
_BONUS_KW = re.compile(r"\bbonus\b", re.IGNORECASE)
_RIGHTS_KW = re.compile(r"\bright(?:s)?\b", re.IGNORECASE)
_DEMERGER_KW = re.compile(r"\bdemerger\b|\bspin[\s-]*off\b|scheme\s+of\s+arrangement", re.IGNORECASE)
_MERGER_KW = re.compile(r"\bmerger\b|amalgamation", re.IGNORECASE)
_DIVIDEND_KW = re.compile(r"\b(?:dividend|divdend|div)\b", re.IGNORECASE)
_CLASS_PATTERNS = (
    ("SPLIT", _SPLIT_KW), ("BONUS", _BONUS_KW), ("RIGHTS", _RIGHTS_KW),
    ("DEMERGER", _DEMERGER_KW), ("MERGER", _MERGER_KW), ("DIVIDEND", _DIVIDEND_KW),
)

_SPLIT_RATIO_RE = re.compile(
    rf"from\s+r[se]\.?\s*{_NUM}\s*(?:/-)?\s*to\s+r[se]\.?\s*{_NUM}", re.IGNORECASE
)
_RATIO_RE = re.compile(r"(\d+)\s*:\s*(\d+)")
_DIVIDEND_AMOUNT_RE = re.compile(rf"r[se]\.?\s*-?\s*{_NUM}", re.IGNORECASE)
_BENIGN_RE = re.compile(r"^(?:annual|extra[\s-]*ordinary)\s+general\s+meeting$|^(?:agm|egm)$", re.IGNORECASE)
_SEPARATORS = (re.compile(r"\s+and\s+", re.IGNORECASE), re.compile(r"\s*&\s*"), re.compile(r"\s*,\s*"))


@dataclasses.dataclass(frozen=True)
class SourceComponent:

    action_class: str
    terms: tuple | None
    fragment: str


def _classes(text: str) -> list[str]:
    return [name for name, pattern in _CLASS_PATTERNS if pattern.search(text)]


def _one(text: str, action_class: str) -> SourceComponent:
    if action_class == "SPLIT":
        m = _SPLIT_RATIO_RE.search(text)
        terms = (float(m.group(1)), float(m.group(2))) if m and float(m.group(1)) > 0 and float(m.group(2)) > 0 else None
        return SourceComponent("SPLIT", terms, text)
    if action_class in ("BONUS", "RIGHTS"):
        m = _RATIO_RE.search(text)
        terms = (int(m.group(1)), int(m.group(2))) if m and int(m.group(1)) > 0 and int(m.group(2)) > 0 else None
        return SourceComponent(action_class, terms, text)
    if action_class == "DIVIDEND":
        m = _DIVIDEND_AMOUNT_RE.search(text)
        terms = (round(float(m.group(1)), 4),) if m and float(m.group(1)) > 0 else None
        return SourceComponent("DIVIDEND", terms, text)
    return SourceComponent(action_class, None, text)


def parse_bse_purpose(purpose: str) -> list[SourceComponent]:
    text = purpose.strip()
    if not text:
        return [SourceComponent("UNCLASSIFIED", None, purpose)]
    if _BENIGN_RE.match(text):
        return []
    classes = _classes(text)
    if not classes:
        return [SourceComponent("UNCLASSIFIED", None, text)]
    if len(classes) == 1:
        return [_one(text, classes[0])]
    for separator in _SEPARATORS:
        parts = [p.strip() for p in separator.split(text) if p.strip()]
        if len(parts) > 1 and all(len(_classes(p)) <= 1 for p in parts):
            if {c for p in parts for c in _classes(p)} == set(classes):
                return [_one(p, _classes(p)[0]) if _classes(p) else SourceComponent("UNCLASSIFIED", None, p) for p in parts]
    return [SourceComponent("UNCLASSIFIED", None, text)]
