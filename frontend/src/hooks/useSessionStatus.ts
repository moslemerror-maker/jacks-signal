import { useCallback, useState } from "react";
import { fetchSessionStatus, type ApiError } from "../api/client";
import type { SessionStatusResponse } from "../api/types";
import { usePolling } from "./usePolling";

export type SessionStatusState =
  | { status: "loading" }
  | { status: "success"; data: SessionStatusResponse }
  | { status: "error"; error: ApiError };

export const DEFAULT_SESSION_STATUS_POLL_INTERVAL_MS = 30_000;

export function useSessionStatus(underlying: string, intervalMs: number = DEFAULT_SESSION_STATUS_POLL_INTERVAL_MS) {
  const [state, setState] = useState<SessionStatusState>({ status: "loading" });

  const refresh = useCallback(async () => {
    const result = await fetchSessionStatus(underlying);
    if (result.ok) {
      setState({ status: "success", data: result.data });
    } else {
      setState({ status: "error", error: result.error });
    }
  }, [underlying]);

  usePolling(refresh, intervalMs);

  return state;
}
