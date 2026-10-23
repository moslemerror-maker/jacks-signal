import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CostRiskPanel } from "./CostRiskPanel";

describe("CostRiskPanel", () => {
  it("renders every NOT_MODELED field as 'Not modeled', never inventing a number", () => {
    render(
      <CostRiskPanel
        costRisk={{
          costs_slippage_assumptions: { status: "NOT_MODELED", detail: "not wired in" },
          risk: { status: "NOT_MODELED", detail: "no risk engine" },
          expected_value: { status: "NOT_MODELED", detail: "deferred" },
        }}
      />,
    );
    expect(screen.getAllByText("Not modeled")).toHaveLength(3);
  });

  it("renders a null field as not available, never as 0", () => {
    render(<CostRiskPanel costRisk={{ costs_slippage_assumptions: null, risk: null, expected_value: null }} />);
    expect(screen.getAllByText("not available")).toHaveLength(3);
    expect(screen.queryByText("0")).not.toBeInTheDocument();
  });
});
