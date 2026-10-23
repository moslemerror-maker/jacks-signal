import type { CandidateLeg, CandidateSnapshot, RecommendationSnapshotResponse, SessionState, SessionStatusResponse } from "../api/types";

const BASE_LEG: CandidateLeg = {
  contract: { underlying: "NIFTY", derivative_class: "IDO", expiry: "2026-10-06", strike_paise: 2450000, option_type: "CE" },
  side: "BUY",
  entry_price: "140",
  entry_price_source: "LIVE_QUOTE",
  quote_timestamp: "2026-09-23T05:00:00+00:00",
  quote_status: "QUOTED",
  last_price: "140",
  bid: "138",
  ask: "142",
  volume: 500,
  open_interest: 1000,
  provider_reported_iv: "0.15",
  provider_reported_greeks: { delta: "0.5" },
  provenance_class: "PRODUCTION_RESEARCH_DATASET",
  liquidity_assessment: { is_acceptable: true, reasons: [] },
};

export function leg(overrides: Partial<CandidateLeg> = {}): CandidateLeg {
  return { ...BASE_LEG, ...overrides };
}

export function formedCandidate(shape: string, legs: CandidateLeg[]): CandidateSnapshot {
  return { shape, formed: true, rejected_reason: null, data_quality: "EOD_RESOLVED", legs };
}

export function rejectedCandidate(shape: string, reason: string): CandidateSnapshot {
  return { shape, formed: false, rejected_reason: reason, data_quality: null, legs: [] };
}

const FRESH_ISO = new Date(Date.now() - 30_000).toISOString();
const STALE_ISO = new Date(Date.now() - 20 * 60 * 1000).toISOString();

export function waitResponse(overrides: Partial<RecommendationSnapshotResponse> = {}): RecommendationSnapshotResponse {
  return {
    metadata: {
      recommendation_id: "11111111-1111-1111-1111-111111111111",
      decision_timestamp: FRESH_ISO,
      underlying: "NIFTY",
      status: "WAIT",
      shape: "LONG_CALL_ATM",
      data_quality: "EOD_RESOLVED",
      model_version: "test-v0",
      strategy_version: "test-v0",
      source: "dhan",
    },
    market_snapshot: {
      cycle_observed_at: FRESH_ISO,
      selected_expiry: "2026-10-06",
      forward_mode: "SPOT_PROXY_UNVERIFIED",
      forward_value: "24500",
      underlying_last_price: "24500",
    },
    selected_recommendation: {
      status: "WAIT",
      selected_shape: "LONG_CALL_ATM",
      reasons: [
        "LONG_CALL_ATM: risk-neutral P(net P&L>0 by expiry) = 0.343 (RISK-NEUTRAL MODEL BASELINE ONLY - not a validated real-world predictive probability)",
        "LONG_CALL_ATM: PREDICTIVE_PROBABILITY_STATUS is NOT_ESTABLISHED - no validated real-world probability model exists yet",
      ],
      decision_thresholds: { min_probability_positive_net_pnl: "0.5", require_verified_forward: false },
    },
    candidates: [
      formedCandidate("LONG_CALL_ATM", [leg()]),
      formedCandidate("LONG_PUT_ATM", [leg({ contract: { ...BASE_LEG.contract, option_type: "PE" }, entry_price: "135", last_price: "135", bid: "133", ask: "137" })]),
      formedCandidate("LONG_STRADDLE_ATM", [leg(), leg({ contract: { ...BASE_LEG.contract, option_type: "PE" }, entry_price: "135", last_price: "135" })]),
    ],
    probability: {
      risk_neutral_probability_metrics: { POSITIVE_NET_PNL_BY_EXPIRY: "0.343" },
      predictive_probability_status: "NOT_ESTABLISHED",
      note: "risk-neutral baseline only - NOT a validated real-world predictive probability",
    },
    cost_risk_status: {
      costs_slippage_assumptions: { status: "NOT_MODELED", detail: "not wired into the live path" },
      risk: { status: "NOT_MODELED", detail: "no risk engine exists yet" },
      expected_value: { status: "NOT_MODELED", detail: "EV computation is deferred" },
    },
    data_quality: "EOD_RESOLVED",
    provenance: {
      raw_response_sha256: "deadbeef".repeat(4),
      description: "Project-level digest of this project's own re-serialization of the parsed provider response - NOT a cryptographic digest of the provider's original wire-level HTTP response bytes.",
    },
    ...overrides,
  };
}

export function noTradeResponse(): RecommendationSnapshotResponse {
  return waitResponse({
    metadata: { ...waitResponse().metadata, status: "NO_TRADE", shape: null },
    selected_recommendation: { status: "NO_TRADE", selected_shape: null, reasons: ["no strike in the supplied chain has any usable (QUOTED) leg"], decision_thresholds: null },
    candidates: [
      rejectedCandidate("LONG_CALL_ATM", "no strike in the supplied chain has any usable (QUOTED) leg"),
      rejectedCandidate("LONG_PUT_ATM", "no strike in the supplied chain has any usable (QUOTED) leg"),
      rejectedCandidate("LONG_STRADDLE_ATM", "no strike in the supplied chain has any usable (QUOTED) leg"),
    ],
    probability: { risk_neutral_probability_metrics: null, predictive_probability_status: "NOT_ESTABLISHED", note: null },
  });
}

export function tradeResponse(): RecommendationSnapshotResponse {
  const base = waitResponse();
  return {
    ...base,
    metadata: { ...base.metadata, status: "TRADE" },
    selected_recommendation: { ...base.selected_recommendation, status: "TRADE" },
  };
}

export function staleResponse(): RecommendationSnapshotResponse {
  return waitResponse({
    metadata: { ...waitResponse().metadata, decision_timestamp: STALE_ISO },
  });
}

export function noCandidateResponse(): RecommendationSnapshotResponse {
  return waitResponse({
    metadata: { ...waitResponse().metadata, shape: null },
    selected_recommendation: { status: "WAIT", selected_shape: null, reasons: ["no candidate could be formed"], decision_thresholds: null },
  });
}

const SESSION_FRESH_ISO = new Date(Date.now() - 30_000).toISOString();
const SESSION_STALE_ISO = new Date(Date.now() - 20 * 60 * 1000).toISOString();

export function sessionStatusResponse(overrides: Partial<SessionStatusResponse> = {}): SessionStatusResponse {
  return {
    underlying: "NIFTY",
    trading_date: "2026-09-25",
    session_state: "LIVE",
    session_start_timestamp: "2026-09-25T03:45:03+00:00",
    last_event_timestamp: "2026-09-25T03:45:03+00:00",
    completion_timestamp: null,
    failure_timestamp: null,
    failure_detail: null,
    cycle_count: 15,
    latest_recommendation_timestamp: SESSION_FRESH_ISO,
    latest_recommendation_status: "WAIT",
    latest_recommendation_id: "11111111-1111-1111-1111-111111111111",
    data_quality: "EOD_RESOLVED",
    source: "dhan",
    ...overrides,
  };
}

export function notStartedSessionStatus(): SessionStatusResponse {
  return sessionStatusResponse({
    session_state: "NOT_STARTED", session_start_timestamp: null, last_event_timestamp: null, cycle_count: 0,
    latest_recommendation_timestamp: null, latest_recommendation_status: null, latest_recommendation_id: null,
    data_quality: null, source: null,
  });
}

export function completedSessionStatus(): SessionStatusResponse {
  return sessionStatusResponse({ session_state: "COMPLETED", completion_timestamp: "2026-09-25T09:45:00+00:00" });
}

export function failedSessionStatus(): SessionStatusResponse {
  return sessionStatusResponse({
    session_state: "FAILED", session_start_timestamp: null, cycle_count: 0,
    failure_timestamp: "2026-09-25T03:45:02+00:00",
    failure_detail: "STARTUP FAILURE during 'live source / config construction': ModuleNotFoundError: No module named 'backend'",
    latest_recommendation_timestamp: "2026-09-24T09:40:03+00:00", latest_recommendation_status: "NO_TRADE",
  });
}

export function unknownSessionStatus(): SessionStatusResponse {
  return sessionStatusResponse({
    session_state: "UNKNOWN", session_start_timestamp: null, cycle_count: 0,
    latest_recommendation_timestamp: null, latest_recommendation_status: null, latest_recommendation_id: null,
    data_quality: null, source: null,
  });
}

export function staleSessionStatus(): SessionStatusResponse {
  return sessionStatusResponse({ latest_recommendation_timestamp: SESSION_STALE_ISO });
}

export function noTradeSessionStatus(): SessionStatusResponse {
  return sessionStatusResponse({ latest_recommendation_status: "NO_TRADE" });
}

export const SESSION_STATES_FOR_TESTS: SessionState[] = [
  "NOT_STARTED", "STARTING", "LIVE", "DEGRADED", "PAUSED", "COMPLETED", "FAILED", "UNKNOWN",
];
