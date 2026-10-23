import type { Probability } from "../api/types";
import { StatusPill } from "./StatusPill";

function asPercent(raw: string): string | null {
  const n = Number(raw);
  return Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : null;
}

interface Props {
  probability: Probability;
  decisionThresholds?: Record<string, unknown> | null;
}

export function ProbabilityPanel({ probability, decisionThresholds }: Props) {
  const metrics = probability.risk_neutral_probability_metrics;
  const thresholdRaw = decisionThresholds?.min_probability_positive_net_pnl;
  const thresholdPercent = typeof thresholdRaw === "string" ? asPercent(thresholdRaw) : null;

  return (
    <div className="panel">
      <h2>Probability</h2>

      <div className="stat-cards">
        {metrics && Object.keys(metrics).length > 0 ? (
          Object.entries(metrics).map(([key, value]) => {
            const percent = asPercent(value);
            return (
              <div className="stat-card" key={key}>
                <div className="stat-card-label" title={key}>
                  {key}
                </div>
                <div className="stat-card-value">{percent ?? value}</div>
                {percent && <div className="stat-card-raw">{value}</div>}
              </div>
            );
          })
        ) : (
          <div className="stat-card stat-card-empty">
            <div className="stat-card-label">Risk-neutral probability</div>
            <p className="rejected-reason">No risk-neutral probability was computed for this decision.</p>
          </div>
        )}

        {thresholdRaw !== undefined && thresholdRaw !== null && (
          <div className="stat-card">
            <div className="stat-card-label">Configured threshold</div>
            <div className="stat-card-value">{thresholdPercent ?? String(thresholdRaw)}</div>
            {thresholdPercent && <div className="stat-card-raw">{String(thresholdRaw)}</div>}
          </div>
        )}

        <div className="stat-card">
          <div className="stat-card-label">Predictive probability</div>
          <div className="stat-card-value">
            <StatusPill kind="NOT_MODELED" label={probability.predictive_probability_status ?? "NOT_ESTABLISHED"} />
          </div>
        </div>
      </div>

      <p className="probability-note">
        The figures above are a Black-76 baseline (risk-neutral model output), not a validated real-world
        predictive probability of profit.
        {probability.note ? ` ${probability.note}` : ""}
      </p>
    </div>
  );
}
