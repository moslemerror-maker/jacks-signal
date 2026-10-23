import { useCallback, useState } from "react";
import { fetchLatestRecommendation, type ApiError } from "../api/client";
import type { RecommendationSnapshotResponse } from "../api/types";
import { usePolling } from "./usePolling";

export type RecommendationState =
  | { status: "loading" }
  | { status: "success"; data: RecommendationSnapshotResponse }
  | { status: "empty" }
  | { status: "error"; error: ApiError };

export const DEFAULT_POLL_INTERVAL_MS = 30_000;

export function useLatestRecommendation(underlying: string, intervalMs: number = DEFAULT_POLL_INTERVAL_MS) {
  const [state, setState] = useState<RecommendationState>({ status: "loading" });
  const [lastRefreshAt, setLastRefreshAt] = useState<Date | null>(null);

  const refresh = useCallback(async () => {
    const result = await fetchLatestRecommendation(underlying);
    setLastRefreshAt(new Date());
    if (result.ok) {
      setState({ status: "success", data: result.data });
    } else if (result.error.kind === "NOT_FOUND") {
      setState({ status: "empty" });
    } else {
      setState({ status: "error", error: result.error });
    }
  }, [underlying]);

  usePolling(refresh, intervalMs);

  return { state, lastRefreshAt, refresh };
}
