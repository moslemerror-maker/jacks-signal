import type { CandidateSnapshot } from "../api/types";
import { LegTable } from "./LegTable";
import { StatusPill } from "./StatusPill";

interface Props {
  candidates: CandidateSnapshot[];
  selectedShape: string | null;
}

export function CandidateComparison({ candidates, selectedShape }: Props) {
  return (
    <div className="panel">
      <h2>Candidate Comparison</h2>
      <div className="candidate-grid">
        {candidates.map((candidate) => (
          <div key={candidate.shape} className={`candidate-card${candidate.shape === selectedShape ? " selected" : ""}`}>
            <div className="header-row" style={{ marginBottom: 8 }}>
              <strong>{candidate.shape}</strong>
              <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                {candidate.shape === selectedShape && <StatusPill kind="OK" label="Selected" />}
                <StatusPill kind={candidate.formed ? "OK" : "NO_TRADE"} label={candidate.formed ? "Formed" : "Not formed"} />
              </div>
            </div>
            {candidate.data_quality && <div className="candidate-data-quality">Data quality: {candidate.data_quality}</div>}
            {candidate.formed ? (
              <LegTable legs={candidate.legs} variant="compact" />
            ) : (
              <p className="rejected-reason">{candidate.rejected_reason ?? "No reason reported."}</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
