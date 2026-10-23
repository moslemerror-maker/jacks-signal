import type { HealthResponse, RecommendationSnapshotResponse, SessionStatusResponse } from "./types";

export type ApiErrorKind = "UNREACHABLE" | "AUTH" | "NOT_FOUND" | "BAD_REQUEST" | "SERVER_ERROR" | "MALFORMED";

export interface ApiError {
  kind: ApiErrorKind;
  message: string;
}

export type ApiResult<T> = { ok: true; data: T } | { ok: false; error: ApiError };

const BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";
const API_KEY = import.meta.env.VITE_API_KEY as string | undefined;

function headers(): HeadersInit {
  const h: Record<string, string> = { Accept: "application/json" };
  if (API_KEY) h["X-API-Key"] = API_KEY;
  return h;
}

function classifyStatus(status: number, detail: string | null): ApiError {
  if (status === 401 || status === 403) return { kind: "AUTH", message: detail ?? "Authentication required" };
  if (status === 404) return { kind: "NOT_FOUND", message: detail ?? "Not found" };
  if (status === 400) return { kind: "BAD_REQUEST", message: detail ?? "Bad request" };
  return { kind: "SERVER_ERROR", message: detail ?? `Server error (${status})` };
}

async function getJson<T>(path: string, isValid: (x: unknown) => x is T): Promise<ApiResult<T>> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, { method: "GET", headers: headers() });
  } catch {
    return { ok: false, error: { kind: "UNREACHABLE", message: "Cannot reach the API" } };
  }

  let body: unknown;
  try {
    body = await response.json();
  } catch {
    return { ok: false, error: { kind: "MALFORMED", message: "The API returned a response that was not valid JSON" } };
  }

  if (!response.ok) {
    const detail = typeof body === "object" && body !== null && "detail" in body ? String((body as { detail: unknown }).detail) : null;
    return { ok: false, error: classifyStatus(response.status, detail) };
  }

  if (!isValid(body)) {
    return { ok: false, error: { kind: "MALFORMED", message: "The API returned a response in an unexpected shape" } };
  }
  return { ok: true, data: body };
}

function isHealthResponse(x: unknown): x is HealthResponse {
  return typeof x === "object" && x !== null && "status" in x && "database_reachable" in x;
}

function isRecommendationSnapshotResponse(x: unknown): x is RecommendationSnapshotResponse {
  if (typeof x !== "object" || x === null) return false;
  const required = ["metadata", "market_snapshot", "selected_recommendation", "candidates", "probability", "cost_risk_status", "data_quality", "provenance"];
  if (!required.every((key) => key in x)) return false;
  const candidates = (x as { candidates: unknown }).candidates;
  return Array.isArray(candidates);
}

export function fetchHealth(): Promise<ApiResult<HealthResponse>> {
  return getJson("/api/health", isHealthResponse);
}

export function fetchLatestRecommendation(underlying: string): Promise<ApiResult<RecommendationSnapshotResponse>> {
  return getJson(`/api/recommendations/latest?underlying=${encodeURIComponent(underlying)}`, isRecommendationSnapshotResponse);
}

function isSessionStatusResponse(x: unknown): x is SessionStatusResponse {
  if (typeof x !== "object" || x === null) return false;
  const required = ["underlying", "trading_date", "session_state", "cycle_count"];
  return required.every((key) => key in x);
}

export function fetchSessionStatus(underlying: string): Promise<ApiResult<SessionStatusResponse>> {
  return getJson(`/api/session/status?underlying=${encodeURIComponent(underlying)}`, isSessionStatusResponse);
}
