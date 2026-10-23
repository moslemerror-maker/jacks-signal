import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SessionStatusBanner, SessionStatusError } from "./SessionStatusBanner";
import {
  completedSessionStatus, failedSessionStatus, noTradeSessionStatus, notStartedSessionStatus,
  sessionStatusResponse, staleSessionStatus, unknownSessionStatus,
} from "../test/fixtures";

describe("1. LIVE session display", () => {
  it("shows the LIVE state, cycle count, and session start", () => {
    render(<SessionStatusBanner status={sessionStatusResponse()} />);
    expect(screen.getByText("LIVE")).toBeInTheDocument();
    expect(screen.getByText("15")).toBeInTheDocument();
    expect(screen.queryByText(/not running/i)).not.toBeInTheDocument();
  });
});

describe("2. NOT_STARTED display", () => {
  it("shows NOT STARTED and the not-running warning, never a fabricated decision", () => {
    render(<SessionStatusBanner status={notStartedSessionStatus()} />);
    expect(screen.getByText("NOT STARTED")).toBeInTheDocument();
    expect(screen.getByText(/not running/i)).toBeInTheDocument();
    expect(screen.getAllByText("not available").length).toBeGreaterThan(0);
  });
});

describe("3. COMPLETED display", () => {
  it("shows COMPLETED and the completion timestamp field", () => {
    render(<SessionStatusBanner status={completedSessionStatus()} />);
    expect(screen.getByText("COMPLETED")).toBeInTheDocument();
    expect(screen.getByText("Completed")).toBeInTheDocument();
    expect(screen.queryByText(/not running/i)).not.toBeInTheDocument();
  });
});

describe("4. FAILED display", () => {
  it("shows SESSION FAILED, the failure reason, and the not-running warning", () => {
    render(<SessionStatusBanner status={failedSessionStatus()} />);
    expect(screen.getByText("SESSION FAILED")).toBeInTheDocument();
    expect(screen.getByText(/ModuleNotFoundError/)).toBeInTheDocument();
    expect(screen.getByText(/not running/i)).toBeInTheDocument();
  });
});

describe("5. UNKNOWN display", () => {
  it("shows UNKNOWN rather than guessing a real state", () => {
    render(<SessionStatusBanner status={unknownSessionStatus()} />);
    expect(screen.getByText("UNKNOWN")).toBeInTheDocument();
  });
});

describe("6. WAIT recommendation", () => {
  it("renders WAIT as the latest decision", () => {
    render(<SessionStatusBanner status={sessionStatusResponse({ latest_recommendation_status: "WAIT" })} />);
    expect(screen.getByText("WAIT")).toBeInTheDocument();
  });
});

describe("7. NO_TRADE recommendation", () => {
  it("renders NO_TRADE as the latest decision, never converted to a failure or a WAIT", () => {
    render(<SessionStatusBanner status={noTradeSessionStatus()} />);
    expect(screen.getByText("NO_TRADE")).toBeInTheDocument();
  });
});

describe("8. stale recommendation warning", () => {
  it("shows a STALE warning for a 20-minute-old recommendation", () => {
    render(<SessionStatusBanner status={staleSessionStatus()} />);
    expect(screen.getByText("STALE")).toBeInTheDocument();
  });

  it("shows no STALE warning for a fresh recommendation", () => {
    render(<SessionStatusBanner status={sessionStatusResponse()} />);
    expect(screen.queryByText("STALE")).not.toBeInTheDocument();
  });
});

describe("9. today's FAILED session with yesterday's recommendation", () => {
  it("reports FAILED, not LIVE, while still surfacing yesterday's recommendation as its own fact", () => {
    render(<SessionStatusBanner status={failedSessionStatus()} />);
    expect(screen.getByText("SESSION FAILED")).toBeInTheDocument();
    expect(screen.queryByText("LIVE")).not.toBeInTheDocument();
    expect(screen.getByText("Last successful recommendation")).toBeInTheDocument();
    expect(screen.getByText("NO_TRADE")).toBeInTheDocument();
  });
});

describe("10. API error state", () => {
  it("shows a distinct session-status error, never a fabricated COMPLETED/LIVE/NO_TRADE state", () => {
    render(<SessionStatusError error={{ kind: "SERVER_ERROR", message: "internal error" }} />);
    expect(screen.getByText(/Session status unavailable/)).toBeInTheDocument();
    const bodyText = document.body.textContent ?? "";
    for (const forbidden of ["LIVE", "COMPLETED", "NO_TRADE"]) {
      expect(bodyText).not.toContain(forbidden);
    }
  });
});
