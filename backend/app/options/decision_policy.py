from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal
from typing import Literal, Sequence

from .candidate_generator import CandidateGenerationOutcome
from .probability import probability_metrics_from_distribution
from .recommendation import RecommendationResult

PredictiveProbabilityStatus = Literal["NOT_ESTABLISHED", "ESTABLISHED"]
PREDICTIVE_PROBABILITY_STATUS: PredictiveProbabilityStatus = "NOT_ESTABLISHED"


@dataclasses.dataclass(frozen=True, slots=True)
class DecisionPolicyThresholds:

    min_probability_positive_net_pnl: Decimal
    require_verified_forward: bool


def decide_for_underlying(
    outcomes: Sequence[CandidateGenerationOutcome], *, distribution_factory, timestamp: dt.datetime,
    underlying: str, thresholds: DecisionPolicyThresholds, model_version: str, strategy_version: str,
) -> tuple[RecommendationResult, ...]:
    results = []
    for outcome in outcomes:
        if outcome.candidate is None:
            results.append(RecommendationResult(
                status="NO_TRADE", timestamp=timestamp, underlying=underlying, candidate=None,
                reasons=(f"{outcome.shape}: candidate could not be formed - {outcome.rejected_reason}",),
                data_quality="UNAVAILABLE_INSUFFICIENT_DATA", model_version=model_version, strategy_version=strategy_version,
            ))
            continue

        candidate = outcome.candidate
        reasons: list[str] = []
        probability_metrics = None
        meets_probability_threshold = False

        distribution = distribution_factory(candidate)
        if distribution is None:
            reasons.append(f"{outcome.shape}: insufficient IV/Greeks data on this candidate's own legs to build a probability distribution")
        else:
            probability_metrics = probability_metrics_from_distribution(candidate.structure, distribution)
            p = probability_metrics["POSITIVE_NET_PNL_BY_EXPIRY"]
            reasons.append(f"{outcome.shape}: risk-neutral P(net P&L>0 by expiry) = {p} "
                           f"(RISK-NEUTRAL MODEL BASELINE ONLY - not a validated real-world predictive probability)")
            meets_probability_threshold = p >= thresholds.min_probability_positive_net_pnl
            if not meets_probability_threshold:
                reasons.append(f"{outcome.shape}: probability {p} below configured threshold {thresholds.min_probability_positive_net_pnl}")

        forward_ok = True
        if candidate.forward_resolution is not None and not candidate.forward_resolution.is_verified:
            reasons.append(f"{outcome.shape}: forward is SPOT_PROXY_UNVERIFIED ({candidate.forward_resolution.detail})")
            if thresholds.require_verified_forward:
                forward_ok = False

        if PREDICTIVE_PROBABILITY_STATUS != "ESTABLISHED":
            reasons.append(
                f"{outcome.shape}: PREDICTIVE_PROBABILITY_STATUS is NOT_ESTABLISHED - no validated real-world "
                "probability model exists yet; a risk-neutral baseline alone can never produce TRADE under this policy"
            )

        can_trade = PREDICTIVE_PROBABILITY_STATUS == "ESTABLISHED" and distribution is not None and meets_probability_threshold and forward_ok
        status = "TRADE" if can_trade else "WAIT"
        results.append(RecommendationResult(
            status=status, timestamp=timestamp, underlying=underlying, candidate=candidate if can_trade else None,
            probability_metrics=probability_metrics, reasons=tuple(reasons),
            data_quality=candidate.data_quality, model_version=model_version, strategy_version=strategy_version,
        ))
    return tuple(results)
