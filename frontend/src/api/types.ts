export interface HealthResponse {
  status: "OK" | "DB_UNAVAILABLE";
  database_reachable: boolean;
}

export type SessionState = "NOT_STARTED" | "STARTING" | "LIVE" | "DEGRADED" | "PAUSED" | "COMPLETED" | "FAILED" | "UNKNOWN";

export interface SessionStatusResponse {
  underlying: string;
  trading_date: string;
  session_state: SessionState;
  session_start_timestamp: string | null;
  last_event_timestamp: string | null;
  completion_timestamp: string | null;
  failure_timestamp: string | null;
  failure_detail: string | null;
  cycle_count: number;
  latest_recommendation_timestamp: string | null;
  latest_recommendation_status: string | null;
  latest_recommendation_id: string | null;
  data_quality: string | null;
  source: string | null;
}

export interface ContractIdentity {
  underlying: string;
  derivative_class: string;
  expiry: string | null;
  strike_paise: number | null;
  option_type: string | null;
}

export interface LiquidityAssessment {
  is_acceptable: boolean;
  reasons: string[];
}

export interface CandidateLeg {
  contract: ContractIdentity;
  side: string;
  entry_price: string;
  entry_price_source: string;
  quote_timestamp: string | null;
  quote_status: string | null;
  last_price: string | null;
  bid: string | null;
  ask: string | null;
  volume: number | null;
  open_interest: number | null;
  provider_reported_iv: string | null;
  provider_reported_greeks: Record<string, string> | null;
  provenance_class: string | null;
  liquidity_assessment: LiquidityAssessment | null;
}

export interface CandidateSnapshot {
  shape: string;
  formed: boolean;
  rejected_reason: string | null;
  data_quality: string | null;
  legs: CandidateLeg[];
}

export interface RecommendationMetadata {
  recommendation_id: string;
  decision_timestamp: string;
  underlying: string;
  status: string;
  shape: string | null;
  data_quality: string;
  model_version: string | null;
  strategy_version: string | null;
  source: string;
}

export interface MarketSnapshot {
  cycle_observed_at: string | null;
  selected_expiry: string | null;
  forward_mode: string | null;
  forward_value: string | null;
  underlying_last_price: string | null;
}

export interface SelectedRecommendation {
  status: string;
  selected_shape: string | null;
  reasons: string[];
  decision_thresholds: Record<string, unknown> | null;
}

export interface Probability {
  risk_neutral_probability_metrics: Record<string, string> | null;
  predictive_probability_status: string | null;
  note: string | null;
}

export interface NotModeledStatus {
  status: string;
  detail: string;
}

export interface CostRiskStatus {
  costs_slippage_assumptions: NotModeledStatus | Record<string, unknown> | null;
  risk: NotModeledStatus | Record<string, unknown> | null;
  expected_value: NotModeledStatus | Record<string, unknown> | null;
}

export interface Provenance {
  raw_response_sha256: string | null;
  description: string;
}

export interface RecommendationSnapshotResponse {
  metadata: RecommendationMetadata;
  market_snapshot: MarketSnapshot;
  selected_recommendation: SelectedRecommendation;
  candidates: CandidateSnapshot[];
  probability: Probability;
  cost_risk_status: CostRiskStatus;
  data_quality: string;
  provenance: Provenance;
}

export function isNotModeledStatus(value: unknown): value is NotModeledStatus {
  return (
    typeof value === "object" &&
    value !== null &&
    "status" in value &&
    (value as { status: unknown }).status === "NOT_MODELED"
  );
}
