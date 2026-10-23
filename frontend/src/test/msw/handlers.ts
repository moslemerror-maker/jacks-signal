import { HttpResponse, http } from "msw";
import type { RecommendationSnapshotResponse, SessionStatusResponse } from "../../api/types";

const BASE = "http://localhost:8000";

export function healthOk() {
  return http.get(`${BASE}/api/health`, () => HttpResponse.json({ status: "OK", database_reachable: true }));
}

export function healthUnreachable() {
  return http.get(`${BASE}/api/health`, () => HttpResponse.error());
}

export function sessionStatusOk(body: SessionStatusResponse) {
  return http.get(`${BASE}/api/session/status`, () => HttpResponse.json(body));
}

export function sessionStatusUnreachable() {
  return http.get(`${BASE}/api/session/status`, () => HttpResponse.error());
}

export function sessionStatusServerError() {
  return http.get(`${BASE}/api/session/status`, () => HttpResponse.json({ detail: "database is not reachable" }, { status: 503 }));
}

export function recommendationSuccess(body: RecommendationSnapshotResponse) {
  return http.get(`${BASE}/api/recommendations/latest`, () => HttpResponse.json(body));
}

export function recommendationNotFound() {
  return http.get(`${BASE}/api/recommendations/latest`, () => HttpResponse.json({ detail: "no recommendation found for underlying='NIFTY'" }, { status: 404 }));
}

export function recommendationUnauthorized() {
  return http.get(`${BASE}/api/recommendations/latest`, () => HttpResponse.json({ detail: "missing or invalid X-API-Key" }, { status: 401 }));
}

export function recommendationServerError() {
  return http.get(`${BASE}/api/recommendations/latest`, () => HttpResponse.json({ detail: "internal error" }, { status: 500 }));
}

export function recommendationUnreachable() {
  return http.get(`${BASE}/api/recommendations/latest`, () => HttpResponse.error());
}

export function recommendationMalformedJson() {
  return http.get(`${BASE}/api/recommendations/latest`, () => new HttpResponse("not json {{{", { status: 200, headers: { "Content-Type": "application/json" } }));
}

export function recommendationMalformedShape() {
  return http.get(`${BASE}/api/recommendations/latest`, () => HttpResponse.json({ unexpected: "shape" }));
}
