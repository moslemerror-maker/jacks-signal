import type { CandidateLeg } from "../api/types";
import { formatTimestamp } from "../lib/time";
import { Value } from "./Value";

function strikeRupees(strikePaise: number | null): string | null {
  if (strikePaise === null) return null;
  return (strikePaise / 100).toString();
}

function greeksLabel(greeks: Record<string, string> | null): string | null {
  if (greeks === null) return null;
  const entries = Object.entries(greeks);
  if (entries.length === 0) return null;
  return entries.map(([k, v]) => `${k}=${v}`).join(", ");
}

function legKey(leg: CandidateLeg, idx: number): string {
  return `${leg.contract.option_type ?? "leg"}-${leg.contract.strike_paise ?? idx}-${idx}`;
}

function liquidityCell(leg: CandidateLeg) {
  if (!leg.liquidity_assessment) return <Value value={null} />;
  return (
    <span
      className={leg.liquidity_assessment.is_acceptable ? "liquidity-ok" : "liquidity-rejected"}
      title={leg.liquidity_assessment.reasons.join("; ") || undefined}
    >
      {leg.liquidity_assessment.is_acceptable ? "Acceptable" : "Rejected"}
    </span>
  );
}

function CompactLegTable({ legs }: { legs: CandidateLeg[] }) {
  return (
    <div className="table-scroll">
      <table className="legs-table legs-table-compact">
        <thead>
          <tr>
            <th>Type</th>
            <th>Strike</th>
            <th>Side</th>
            <th>Entry Price</th>
            <th>LTP</th>
          </tr>
        </thead>
        <tbody>
          {legs.map((leg, idx) => (
            <tr key={legKey(leg, idx)}>
              <td>
                <Value value={leg.contract.option_type} />
              </td>
              <td className="num">
                <Value value={strikeRupees(leg.contract.strike_paise)} />
              </td>
              <td>{leg.side}</td>
              <td className="num">{leg.entry_price}</td>
              <td className="num">
                <Value value={leg.last_price} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FullLegTable({ legs }: { legs: CandidateLeg[] }) {
  return (
    <>
      <div className="table-scroll">
        <table className="legs-table">
          <thead>
            <tr>
              <th>Type</th>
              <th>Strike</th>
              <th>Expiry</th>
              <th>Side</th>
              <th>Entry Price</th>
              <th>Entry Source</th>
              <th>LTP</th>
              <th>Bid</th>
              <th>Ask</th>
              <th>Liquidity</th>
            </tr>
          </thead>
          <tbody>
            {legs.map((leg, idx) => (
              <tr key={legKey(leg, idx)}>
                <td>
                  <Value value={leg.contract.option_type} />
                </td>
                <td className="num">
                  <Value value={strikeRupees(leg.contract.strike_paise)} />
                </td>
                <td>
                  <Value value={leg.contract.expiry} />
                </td>
                <td>{leg.side}</td>
                <td className="num">{leg.entry_price}</td>
                <td>{leg.entry_price_source}</td>
                <td className="num">
                  <Value value={leg.last_price} />
                </td>
                <td className="num">
                  <Value value={leg.bid} />
                </td>
                <td className="num">
                  <Value value={leg.ask} />
                </td>
                <td>{liquidityCell(leg)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <details className="market-details">
        <summary>Market details ({legs.length} leg{legs.length === 1 ? "" : "s"})</summary>
        <div className="table-scroll">
          <table className="legs-table legs-table-secondary">
            <thead>
              <tr>
                <th>Type</th>
                <th>Volume</th>
                <th>OI</th>
                <th>Quote Status</th>
                <th>Observed At</th>
                <th>Provider IV</th>
                <th>Provider Greeks</th>
              </tr>
            </thead>
            <tbody>
              {legs.map((leg, idx) => (
                <tr key={legKey(leg, idx)}>
                  <td>
                    <Value value={leg.contract.option_type} />
                  </td>
                  <td className="num">
                    <Value value={leg.volume} />
                  </td>
                  <td className="num">
                    <Value value={leg.open_interest} />
                  </td>
                  <td>
                    <Value value={leg.quote_status} />
                  </td>
                  <td>{formatTimestamp(leg.quote_timestamp)}</td>
                  <td className="num">
                    <Value value={leg.provider_reported_iv} />
                  </td>
                  <td>
                    <Value value={greeksLabel(leg.provider_reported_greeks)} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </>
  );
}

export function LegTable({ legs, variant = "full" }: { legs: CandidateLeg[]; variant?: "full" | "compact" }) {
  if (legs.length === 0) {
    return <p className="rejected-reason">No legs.</p>;
  }
  return variant === "compact" ? <CompactLegTable legs={legs} /> : <FullLegTable legs={legs} />;
}
