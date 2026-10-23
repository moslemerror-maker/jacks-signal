import type { Provenance, RecommendationMetadata } from "../api/types";
import { formatTimestamp } from "../lib/time";
import { Value } from "./Value";

export function DataQualityPanel({ metadata, provenance }: { metadata: RecommendationMetadata; provenance: Provenance }) {
  return (
    <div className="panel">
      <h2>Data Quality / Provenance</h2>
      <div className="meta-grid">
        <div className="meta-field">
          <div className="label">Data quality</div>
          <div className="value">{metadata.data_quality}</div>
        </div>
        <div className="meta-field">
          <div className="label">Source</div>
          <div className="value">{metadata.source}</div>
        </div>
        <div className="meta-field">
          <div className="label">Decision timestamp</div>
          <div className="value">{formatTimestamp(metadata.decision_timestamp)}</div>
        </div>
        <div className="meta-field">
          <div className="label">Model version</div>
          <div className="value">
            <Value value={metadata.model_version} />
          </div>
        </div>
        <div className="meta-field">
          <div className="label">Strategy version</div>
          <div className="value">
            <Value value={metadata.strategy_version} />
          </div>
        </div>
        <div className="meta-field" style={{ gridColumn: "1 / -1" }}>
          <div className="label">Raw response hash</div>
          <div className="value">
            <Value value={provenance.raw_response_sha256} />
          </div>
          <div className="probability-note">{provenance.description}</div>
        </div>
      </div>
    </div>
  );
}
