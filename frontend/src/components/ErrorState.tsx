import type { ApiError } from "../api/client";

const MESSAGES: Record<ApiError["kind"], string> = {
  UNREACHABLE: "Recommendation unavailable — cannot reach the API.",
  AUTH: "Authentication required — the configured API key was rejected or is missing.",
  NOT_FOUND: "No recommendation found.",
  BAD_REQUEST: "The request to the API was rejected.",
  SERVER_ERROR: "Recommendation unavailable — the service is temporarily unable to respond.",
  MALFORMED: "Recommendation unavailable — the API returned an unexpected response.",
};

export function ErrorState({ error }: { error: ApiError }) {
  return (
    <div className="error-state" role="alert">
      <p>{MESSAGES[error.kind]}</p>
    </div>
  );
}
