import type { HealthState } from "../hooks/useHealth";
import { StatusPill } from "./StatusPill";

const HEALTH_LABEL: Record<HealthState, string> = {
  checking: "Checking…",
  ok: "API OK",
  db_unavailable: "DB Unavailable",
  unreachable: "API Unreachable",
};

const HEALTH_KIND: Record<HealthState, string> = {
  checking: "NOT_MODELED",
  ok: "OK",
  db_unavailable: "ERROR",
  unreachable: "ERROR",
};

interface Props {
  health: HealthState;
  lastRefreshAt: Date | null;
  underlying: string;
}

export function Header({ health, lastRefreshAt, underlying }: Props) {
  return (
    <div className="app-header header-row">
      <h1>AEVORYN Options Monitor</h1>
      <div className="app-header-meta">
        <span className="meta-field">
          <span className="label">Underlying</span> <span className="value">{underlying}</span>
        </span>
        <span className="meta-field">
          <span className="label">Last refresh</span>{" "}
          <span className="value">{lastRefreshAt ? lastRefreshAt.toLocaleTimeString() : "—"}</span>
        </span>
        <StatusPill kind={HEALTH_KIND[health]} label={HEALTH_LABEL[health]} />
      </div>
    </div>
  );
}
