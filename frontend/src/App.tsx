import { CandidateComparison } from "./components/CandidateComparison";
import { CostRiskPanel } from "./components/CostRiskPanel";
import { DataQualityPanel } from "./components/DataQualityPanel";
import { DecisionSnapshotBanner } from "./components/DecisionSnapshotBanner";
import { EmptyState } from "./components/EmptyState";
import { ErrorState } from "./components/ErrorState";
import { Header } from "./components/Header";
import { ProbabilityPanel } from "./components/ProbabilityPanel";
import { SelectedRecommendation } from "./components/SelectedRecommendation";
import { SessionStatusBanner, SessionStatusError } from "./components/SessionStatusBanner";
import { useHealth } from "./hooks/useHealth";
import { useLatestRecommendation } from "./hooks/useLatestRecommendation";
import { useSessionStatus } from "./hooks/useSessionStatus";

const UNDERLYING = "NIFTY";

export function App() {
  const health = useHealth();
  const sessionStatus = useSessionStatus(UNDERLYING);
  const { state, lastRefreshAt } = useLatestRecommendation(UNDERLYING);

  return (
    <div className="app-shell">
      <Header health={health} lastRefreshAt={lastRefreshAt} underlying={UNDERLYING} />

      {sessionStatus.status === "loading" && (
        <div className="panel">
          <p>Loading session status{"…"}</p>
        </div>
      )}
      {sessionStatus.status === "error" && <SessionStatusError error={sessionStatus.error} />}
      {sessionStatus.status === "success" && <SessionStatusBanner status={sessionStatus.data} />}

      {state.status === "loading" && (
        <div className="panel">
          <p>Loading{"…"}</p>
        </div>
      )}

      {state.status === "empty" && <EmptyState />}

      {state.status === "error" && <ErrorState error={state.error} />}

      {state.status === "success" && (
        <>
          <DecisionSnapshotBanner metadata={state.data.metadata} marketSnapshot={state.data.market_snapshot} />
          <ProbabilityPanel probability={state.data.probability} decisionThresholds={state.data.selected_recommendation.decision_thresholds} />
          <DataQualityPanel metadata={state.data.metadata} provenance={state.data.provenance} />
          <SelectedRecommendation selected={state.data.selected_recommendation} candidates={state.data.candidates} />
          <CandidateComparison candidates={state.data.candidates} selectedShape={state.data.selected_recommendation.selected_shape} />
          <CostRiskPanel costRisk={state.data.cost_risk_status} />
        </>
      )}
    </div>
  );
}
