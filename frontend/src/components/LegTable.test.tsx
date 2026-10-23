import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { leg } from "../test/fixtures";
import { LegTable } from "./LegTable";

describe("LegTable", () => {
  it("renders provided quote fields", () => {
    render(<LegTable legs={[leg()]} />);
    expect(screen.getAllByText("140").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("138")).toBeInTheDocument();
    expect(screen.getByText("142")).toBeInTheDocument();
  });

  it("renders missing optional fields as 'not available', never 0 or blank", () => {
    render(
      <LegTable
        legs={[
          leg({
            volume: null,
            open_interest: null,
            provider_reported_iv: null,
            provider_reported_greeks: null,
            bid: null,
            ask: null,
          }),
        ]}
      />,
    );
    const notAvailable = screen.getAllByText("not available");
    expect(notAvailable.length).toBeGreaterThanOrEqual(4);
    expect(screen.queryByText("0")).not.toBeInTheDocument();
  });

  it("renders a rejected liquidity assessment distinctly from an acceptable one", () => {
    render(<LegTable legs={[leg({ liquidity_assessment: { is_acceptable: false, reasons: ["spread too wide"] } })]} />);
    expect(screen.getByText("Rejected")).toBeInTheDocument();
  });

  it("renders strike converted from paise to rupees", () => {
    render(<LegTable legs={[leg({ contract: { underlying: "NIFTY", derivative_class: "IDO", expiry: "2026-10-06", strike_paise: 2450000, option_type: "CE" } })]} />);
    expect(screen.getByText("24500")).toBeInTheDocument();
  });
});
