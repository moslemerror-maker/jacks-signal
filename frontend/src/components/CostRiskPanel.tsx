import type { CostRiskStatus } from "../api/types";
import { NotModeledOrValue } from "./Value";

export function CostRiskPanel({ costRisk }: { costRisk: CostRiskStatus }) {
  return (
    <div className="panel">
      <h2>Risk / Cost / Expected Value</h2>
      <div className="meta-grid">
        <div className="meta-field">
          <div className="label">Costs / slippage assumptions</div>
          <div className="value">
            <NotModeledOrValue value={costRisk.costs_slippage_assumptions} />
          </div>
        </div>
        <div className="meta-field">
          <div className="label">Risk</div>
          <div className="value">
            <NotModeledOrValue value={costRisk.risk} />
          </div>
        </div>
        <div className="meta-field">
          <div className="label">Expected value</div>
          <div className="value">
            <NotModeledOrValue value={costRisk.expected_value} />
          </div>
        </div>
      </div>
    </div>
  );
}
