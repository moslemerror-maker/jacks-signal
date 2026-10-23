import { useCallback, useState } from "react";
import { fetchHealth } from "../api/client";
import { usePolling } from "./usePolling";

export type HealthState = "checking" | "ok" | "db_unavailable" | "unreachable";

const HEALTH_POLL_INTERVAL_MS = 30_000;

export function useHealth(): HealthState {
  const [state, setState] = useState<HealthState>("checking");

  const refresh = useCallback(async () => {
    const result = await fetchHealth();
    if (!result.ok) {
      setState("unreachable");
    } else {
      setState(result.data.database_reachable ? "ok" : "db_unavailable");
    }
  }, []);

  usePolling(refresh, HEALTH_POLL_INTERVAL_MS);
  return state;
}
