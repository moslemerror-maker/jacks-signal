from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Sequence

from backend.app.equity_live.live_feature_service import FeatureSnapshot, cross_sectional_rank_15m

CROSS_SECTIONAL_EXTREME_HIGH = 0.95
CROSS_SECTIONAL_EXTREME_LOW = 0.05
ABNORMAL_RELATIVE_VOLUME = 2.0
VOLATILITY_COMPRESSION_THRESHOLD = 0.7
VOLATILITY_EXPANSION_THRESHOLD = 1.3
OPENING_DRIVE_MAX_MINUTES = 30
OPENING_DRIVE_MIN_MOVE = 0.005

UNKNOWN = "UNKNOWN"
NONE_STATE = "NONE"

TRADABLE_LIQUIDITY_TIERS = ("mid_liquidity", "high_liquidity")


@dataclasses.dataclass(frozen=True)
class CandidateRecord:
    symbol: str
    security_id: int
    timestamp: dt.datetime
    price: float | None
    return_1m: float | None
    return_5m: float | None
    return_15m: float | None
    relative_strength: float | None
    relative_volume: float | None
    volatility_state: str
    breakout_state: str
    opening_drive_state: str
    liquidity_tier: str | None
    candidate_reasons: tuple[str, ...]
    data_quality: str
    direction: str
    decision: str


def classify_volatility_state(snapshot: FeatureSnapshot) -> str:
    ratio = snapshot.volatility_expansion_ratio
    if ratio is None:
        return UNKNOWN
    if ratio >= VOLATILITY_EXPANSION_THRESHOLD:
        return "EXPANDING"
    if ratio <= VOLATILITY_COMPRESSION_THRESHOLD:
        return "COMPRESSED"
    return "NORMAL"


def classify_breakout_state(snapshot: FeatureSnapshot) -> str:
    if snapshot.recent_breakout_up is None and snapshot.recent_breakout_down is None:
        return UNKNOWN
    if snapshot.recent_breakout_up:
        return "BREAKOUT_UP"
    if snapshot.recent_breakout_down:
        return "BREAKOUT_DOWN"
    return NONE_STATE


def classify_opening_drive_state(snapshot: FeatureSnapshot) -> str:
    minutes, distance = snapshot.minutes_since_open, snapshot.distance_from_session_open
    if minutes is None or distance is None:
        return UNKNOWN
    if minutes > OPENING_DRIVE_MAX_MINUTES:
        return NONE_STATE
    if distance >= OPENING_DRIVE_MIN_MOVE:
        return "DRIVE_UP"
    if distance <= -OPENING_DRIVE_MIN_MOVE:
        return "DRIVE_DOWN"
    return NONE_STATE


def _candidate_reasons(snapshot: FeatureSnapshot, rank: float | None, volatility_state: str, breakout_state: str, opening_drive_state: str) -> tuple[str, ...]:
    reasons = []
    if rank is not None and (rank >= CROSS_SECTIONAL_EXTREME_HIGH or rank <= CROSS_SECTIONAL_EXTREME_LOW):
        reasons.append("EXTREME_CROSS_SECTIONAL_MOVE")
    if snapshot.relative_volume_20 is not None and snapshot.relative_volume_20 >= ABNORMAL_RELATIVE_VOLUME:
        reasons.append("ABNORMAL_VOLUME")
    if volatility_state == "EXPANDING":
        reasons.append("VOLATILITY_EXPANDING")
    if breakout_state in ("BREAKOUT_UP", "BREAKOUT_DOWN"):
        reasons.append(breakout_state)
    if opening_drive_state in ("DRIVE_UP", "DRIVE_DOWN"):
        reasons.append("OPENING_" + opening_drive_state)
    return tuple(reasons)


def _decision(snapshot: FeatureSnapshot, liquidity_tier: str | None) -> str:
    if snapshot.data_quality != "OK":
        return "OBSERVE"
    if liquidity_tier in TRADABLE_LIQUIDITY_TIERS:
        return "WAIT"
    return "OBSERVE"


def build_candidate(snapshot: FeatureSnapshot, *, rank: float | None, liquidity_tier: str | None) -> CandidateRecord | None:
    volatility_state = classify_volatility_state(snapshot)
    breakout_state = classify_breakout_state(snapshot)
    opening_drive_state = classify_opening_drive_state(snapshot)
    reasons = _candidate_reasons(snapshot, rank, volatility_state, breakout_state, opening_drive_state)
    if not reasons:
        return None
    return CandidateRecord(
        symbol=snapshot.symbol, security_id=snapshot.security_id, timestamp=snapshot.as_of_utc, price=snapshot.price,
        return_1m=snapshot.return_1m, return_5m=snapshot.return_5m, return_15m=snapshot.return_15m,
        relative_strength=snapshot.relative_strength, relative_volume=snapshot.relative_volume_20,
        volatility_state=volatility_state, breakout_state=breakout_state, opening_drive_state=opening_drive_state,
        liquidity_tier=liquidity_tier, candidate_reasons=reasons, data_quality=snapshot.data_quality,
        direction=UNKNOWN, decision=_decision(snapshot, liquidity_tier),
    )


def scan_universe(snapshots: Sequence[FeatureSnapshot], *, liquidity_tiers: dict[int, str | None]) -> list[CandidateRecord]:
    candidates = []
    for s in snapshots:
        rank = cross_sectional_rank_15m(snapshots, s.symbol)
        candidate = build_candidate(s, rank=rank, liquidity_tier=liquidity_tiers.get(s.security_id))
        if candidate is not None:
            candidates.append(candidate)
    return candidates
