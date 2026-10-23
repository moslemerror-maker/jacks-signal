import type { CandidateSnapshot, SelectedRecommendation as SelectedRecommendationType } from "../api/types";
import { LegTable } from "./LegTable";

interface Props {
  selected: SelectedRecommendationType;
  candidates: CandidateSnapshot[];
}

export function SelectedRecommendation({ selected, candidates }: Props) {
  const matched = selected.selected_shape ? candidates.find((c) => c.shape === selected.selected_shape) : undefined;

  return (
    <div className="panel">
      <h2>Selected Option</h2>
      <p className="selected-recommendation-summary">
        <span className="status-pill" data-kind={selected.status}>
          {selected.status}
        </span>
        {selected.selected_shape && <strong className="selected-shape-label">{selected.selected_shape}</strong>}
      </p>
      {selected.reasons.length > 0 && (
        <ul className="reasons-list">
          {selected.reasons.map((reason, idx) => (
            <li key={idx}>{reason}</li>
          ))}
        </ul>
      )}
      {matched && matched.formed ? (
        <LegTable legs={matched.legs} />
      ) : (
        <p className="rejected-reason">No candidate was selected for this decision.</p>
      )}
    </div>
  );
}
