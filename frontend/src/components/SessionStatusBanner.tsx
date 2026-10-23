import type { ApiError } from "../api/client";
import type { SessionStatusResponse } from "../api/types";
import { useNow } from "../hooks/useNow";
import { elapsedMs, formatElapsed, formatTimestamp, isStale } from "../lib/time";
import { Value } from "./Value";

const STATE_LABEL: Record<string, string> = {
  NOT_STARTED: "NOT STARTED",
  STARTING: "STARTING",
  LIVE: "LIVE",
  DEGRADED: "DEGRADED",
  PAUSED: "PAUSED",
  COMPLETED: "COMPLETED",
  FAILED: "SESSION FAILED",
  UNKNOWN: "UNKNOWN",
};

const STATE_ICON: Record<string, string> = {
  NOT_STARTED: "\u{1F7E1}",
  STARTING: "\u{1F7E1}",
  LIVE: "\u{1F7E2}",
  DEGRADED: "\u{1F7E0}",
  PAUSED: "\u{26AA}",
  COMPLETED: "\u{1F535}",
  FAILED: "\u{1F534}",
  UNKNOWN: "\u{26AA}",
};

export function SessionStatusBanner({ status }: { status: SessionStatusResponse }) {
  const now = useNow(15_000);
  const recTimestamp = status.latest_recommendation_timestamp;
  const recStale = recTimestamp !== null && isStale(recTimestamp, now);
  const recAge = recTimestamp !== null ? formatElapsed(elapsedMs(recTimestamp, now)) : null;
  const notRunning = status.session_state === "FAILED" || status.session_state === "NOT_STARTED";

  return (
    <div className="panel panel-hero session-status-banner">
      <div className="header-row" style={{ marginBottom: 8 }}>
        <h2 style={{ margin: 0, textTransform: "none", fontSize: 15, color: "var(--text)" }}>
          {status.underlying} Live Session
        </h2>
        <span className="session-state-pill" data-state={status.session_state}>
          <span aria-hidden="true">{STATE_ICON[status.session_state] ?? STATE_ICON.UNKNOWN}</span>{" "}
          {STATE_LABEL[status.session_state] ?? status.session_state}
        </span>
      </div>

      <div className="meta-grid">
        <div className="meta-field">
          <div className="label">Trading date</div>
          <div className="value">{status.trading_date}</div>
        </div>
        <div className="meta-field">
          <div className="label">Session start</div>
          <div className="value">{formatTimestamp(status.session_start_timestamp)}</div>
        </div>
        <div className="meta-field">
          <div className="label">Cycles</div>
          <div className="value">{status.cycle_count}</div>
        </div>
        {status.session_state === "COMPLETED" && (
          <div className="meta-field">
            <div className="label">Completed</div>
            <div className="value">{formatTimestamp(status.completion_timestamp)}</div>
          </div>
        )}
        {status.session_state === "FAILED" && (
          <>
            <div className="meta-field">
              <div className="label">Failure time</div>
              <div className="value">{formatTimestamp(status.failure_timestamp)}</div>
            </div>
            <div className="meta-field" style={{ gridColumn: "1 / -1" }}>
              <div className="label">Reason</div>
              <div className="value">
                <Value value={status.failure_detail} />
              </div>
            </div>
          </>
        )}
      </div>

      <div className="session-status-recommendation">
        <div className="meta-grid">
          <div className="meta-field">
            <div className="label">Latest decision</div>
            <div className="value">
              {status.latest_recommendation_status ? (
                <span className="status-pill" data-kind={status.latest_recommendation_status}>
                  {status.latest_recommendation_status}
                </span>
              ) : (
                <span className="null-value">not available</span>
              )}
            </div>
          </div>
          <div className="meta-field">
            <div className="label">{notRunning ? "Last successful recommendation" : "Last cycle"}</div>
            <div className="value">{formatTimestamp(status.latest_recommendation_timestamp)}</div>
          </div>
          <div className="meta-field">
            <div className="label">Recommendation age</div>
            <div className="value">
              {recAge ?? "—"}
              {recStale && (
                <span className="stale-warning" style={{ marginLeft: 8 }}>
                  STALE
                </span>
              )}
            </div>
          </div>
          <div className="meta-field">
            <div className="label">Data quality</div>
            <div className="value">
              <Value value={status.data_quality} />
            </div>
          </div>
          <div className="meta-field">
            <div className="label">Source</div>
            <div className="value">
              <Value value={status.source} />
            </div>
          </div>
        </div>
        {notRunning && <p className="session-not-running-warning">⚠️ Today's market analysis is not running.</p>}
      </div>
    </div>
  );
}

const SESSION_STATUS_ERROR_MESSAGES: Record<ApiError["kind"], string> = {
  UNREACHABLE: "Session status unavailable — cannot reach the API.",
  AUTH: "Session status unavailable — authentication required.",
  NOT_FOUND: "Session status unavailable — not found.",
  BAD_REQUEST: "Session status unavailable — the request to the API was rejected.",
  SERVER_ERROR: "Session status unavailable — the service is temporarily unable to respond.",
  MALFORMED: "Session status unavailable — the API returned an unexpected response.",
};

export function SessionStatusError({ error }: { error: ApiError }) {
  return (
    <div className="panel error-state" role="alert">
      <p>{SESSION_STATUS_ERROR_MESSAGES[error.kind]}</p>
    </div>
  );
}
