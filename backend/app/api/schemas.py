from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["OK", "DB_UNAVAILABLE"]
    database_reachable: bool


class SessionStatusResponse(BaseModel):

    underlying: str
    trading_date: str
    session_state: Literal["NOT_STARTED", "STARTING", "LIVE", "DEGRADED", "PAUSED", "COMPLETED", "FAILED", "UNKNOWN"]
    session_start_timestamp: str | None = None
    last_event_timestamp: str | None = None
    completion_timestamp: str | None = None
    failure_timestamp: str | None = None
    failure_detail: str | None = None
    cycle_count: int
    latest_recommendation_timestamp: str | None = None
    latest_recommendation_status: str | None = None
    latest_recommendation_id: str | None = None
    data_quality: str | None = None
    source: str | None = None


class ContractIdentitySchema(BaseModel):
    underlying: str
    derivative_class: str
    expiry: str | None = None
    strike_paise: int | None = None
    option_type: str | None = None


class LiquidityAssessmentSchema(BaseModel):
    is_acceptable: bool
    reasons: list[str]


class CandidateLegSchema(BaseModel):
    contract: ContractIdentitySchema
    side: str
    entry_price: str
    entry_price_source: str
    quote_timestamp: str | None = None
    quote_status: str | None = None
    last_price: str | None = None
    bid: str | None = None
    ask: str | None = None
    volume: int | None = None
    open_interest: int | None = None
    provider_reported_iv: str | None = None
    provider_reported_greeks: dict[str, str] | None = None
    provenance_class: str | None = None
    liquidity_assessment: LiquidityAssessmentSchema | None = None


class CandidateSnapshotSchema(BaseModel):
    shape: str
    formed: bool
    rejected_reason: str | None = None
    data_quality: str | None = None
    legs: list[CandidateLegSchema] = []


class RecommendationMetadataSchema(BaseModel):
    recommendation_id: str
    decision_timestamp: str
    underlying: str
    status: str
    shape: str | None = None
    data_quality: str
    model_version: str | None = None
    strategy_version: str | None = None
    source: str


class MarketSnapshotSchema(BaseModel):

    cycle_observed_at: str | None = None
    selected_expiry: str | None = None
    forward_mode: str | None = None
    forward_value: str | None = None
    underlying_last_price: str | None = None


class SelectedRecommendationSchema(BaseModel):
    status: str
    selected_shape: str | None = None
    reasons: list[str] = []
    decision_thresholds: dict[str, Any] | None = None


class ProbabilitySchema(BaseModel):

    risk_neutral_probability_metrics: dict[str, str] | None = None
    predictive_probability_status: str | None = None
    note: str | None = None


class NotModeledStatusSchema(BaseModel):
    status: str
    detail: str


class CostRiskStatusSchema(BaseModel):

    costs_slippage_assumptions: NotModeledStatusSchema | dict[str, Any] | None = None
    risk: NotModeledStatusSchema | dict[str, Any] | None = None
    expected_value: NotModeledStatusSchema | dict[str, Any] | None = None


class ProvenanceSchema(BaseModel):
    raw_response_sha256: str | None = None
    description: str


class RecommendationSnapshotResponse(BaseModel):

    metadata: RecommendationMetadataSchema
    market_snapshot: MarketSnapshotSchema
    selected_recommendation: SelectedRecommendationSchema
    candidates: list[CandidateSnapshotSchema]
    probability: ProbabilitySchema
    cost_risk_status: CostRiskStatusSchema
    data_quality: str
    provenance: ProvenanceSchema
