import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { formedCandidate, leg, rejectedCandidate } from "../test/fixtures";
import { CandidateComparison } from "./CandidateComparison";

describe("CandidateComparison", () => {
  it("shows all three shapes even when only one is selected", () => {
    render(
      <CandidateComparison
        candidates={[
          formedCandidate("LONG_CALL_ATM", [leg()]),
          rejectedCandidate("LONG_PUT_ATM", "no PE quote"),
          rejectedCandidate("LONG_STRADDLE_ATM", "strike missing PE quote"),
        ]}
        selectedShape="LONG_CALL_ATM"
      />,
    );
    expect(screen.getByText("LONG_CALL_ATM")).toBeInTheDocument();
    expect(screen.getByText("LONG_PUT_ATM")).toBeInTheDocument();
    expect(screen.getByText("LONG_STRADDLE_ATM")).toBeInTheDocument();
    expect(screen.getByText("no PE quote")).toBeInTheDocument();
  });

  it("labels an unformed shape distinctly, never showing fabricated legs", () => {
    render(<CandidateComparison candidates={[rejectedCandidate("LONG_PUT_ATM", "no PE quote")]} selectedShape={null} />);
    expect(screen.getByText("Not formed")).toBeInTheDocument();
  });
});
