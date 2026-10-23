import { useNow } from "../hooks/useNow";
import { elapsedMs, formatElapsed, formatTimestamp, isStale } from "../lib/time";
import type { RecommendationMetadata, MarketSnapshot } from "../api/types";
import { StatusPill } from "./StatusPill";
import { Value } from "./Value";

interface Props {
  metadata: RecommendationMetadata;
  marketSnapshot: MarketSnapshot;
}

export function DecisionSnapshotBanner({ metadata, marketSnapshot }: Props) {
  const now = useNow(15_000);
  const stale = isStale(metadata.decision_timestamp, now);
  const elapsed = elapsedMs(metadata.decision_timestamp, now);

  return (
    <div className="panel panel-hero">
      <div className="banner-decision-snapshot">Decision Snapshot — Not Current Market</div>
      <div className="decision-hero">
        <StatusPill kind={metadata.status} />
        {metadata.shape && <span className="decision-hero-shape">{metadata.shape}</span>}
        {stale && <span className="stale-warning">STALE — OLD DECISION</span>}
      </div>
      <div className="meta-grid">
        <div className="meta-field">
          <div className="label">Decision timestamp</div>
          <div className="value">{formatTimestamp(metadata.decision_timestamp)}</div>
        </div>
        <div className="meta-field">
          <div className="label">Age</div>
          <div className="value">{formatElapsed(elapsed)}</div>
        </div>
        <div className="meta-field">
          <div className="label">Selected expiry</div>
          <div className="value">
            <Value value={marketSnapshot.selected_expiry} />
          </div>
        </div>
        <div className="meta-field">
          <div className="label">Underlying price (at decision time)</div>
          <div className="value">
            <Value value={marketSnapshot.underlying_last_price} />
          </div>
        </div>
        <div className="meta-field">
          <div className="label">Forward mode</div>
          <div className="value">
            <Value value={marketSnapshot.forward_mode} />
          </div>
        </div>
        <div className="meta-field">
          <div className="label">Forward value</div>
          <div className="value">
            <Value value={marketSnapshot.forward_value} />
          </div>
        </div>
        <div className="meta-field">
          <div className="label">Data quality</div>
          <div className="value">{metadata.data_quality}</div>
        </div>
        <div className="meta-field">
          <div className="label">Shape</div>
          <div className="value">
            <Value value={metadata.shape} />
          </div>
        </div>
      </div>
    </div>
  );
}
