import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ProbabilityPanel } from "./ProbabilityPanel";

describe("ProbabilityPanel", () => {
  it("labels the risk-neutral metric as a Black-76 baseline and shows predictive status verbatim", () => {
    render(
      <ProbabilityPanel
        probability={{
          risk_neutral_probability_metrics: { POSITIVE_NET_PNL_BY_EXPIRY: "0.343" },
          predictive_probability_status: "NOT_ESTABLISHED",
          note: "risk-neutral baseline only",
        }}
      />,
    );
    expect(screen.getByText(/Black-76 baseline/)).toBeInTheDocument();
    expect(screen.getByText("0.343")).toBeInTheDocument();
    expect(screen.getByText("NOT_ESTABLISHED")).toBeInTheDocument();
    expect(document.body.textContent ?? "").not.toMatch(/win probability/i);
  });

  it("shows an explicit message when no risk-neutral metric was computed, never a fabricated number", () => {
    render(<ProbabilityPanel probability={{ risk_neutral_probability_metrics: null, predictive_probability_status: "NOT_ESTABLISHED", note: null }} />);
    expect(screen.getByText("No risk-neutral probability was computed for this decision.")).toBeInTheDocument();
  });
});
