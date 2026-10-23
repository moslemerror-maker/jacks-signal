import { act, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import {
  healthOk,
  recommendationMalformedJson,
  recommendationMalformedShape,
  recommendationNotFound,
  recommendationServerError,
  recommendationSuccess,
  recommendationUnauthorized,
  recommendationUnreachable,
} from "./test/msw/handlers";
import { server } from "./test/msw/server";
import { noCandidateResponse, noTradeResponse, notStartedSessionStatus, staleResponse, tradeResponse, waitResponse } from "./test/fixtures";
import { sessionStatusOk, sessionStatusUnreachable } from "./test/msw/handlers";

beforeEach(() => {
  server.use(healthOk());
  server.use(sessionStatusOk(notStartedSessionStatus()));
});

describe("Session Status: no false LIVE from mere API/browser connectivity", () => {
  it("shows NOT_STARTED, not LIVE, when the API/DB are healthy but no session ran today", async () => {
    server.use(recommendationSuccess(waitResponse()));
    render(<App />);
    await waitFor(() => expect(screen.getByText("NOT STARTED")).toBeInTheDocument());
    expect(screen.queryByText("LIVE")).not.toBeInTheDocument();
  });

  it("shows a distinct session-status error when only the session-status endpoint fails, without hiding a healthy recommendation", async () => {
    server.use(sessionStatusUnreachable());
    server.use(recommendationSuccess(waitResponse()));
    render(<App />);
    await waitFor(() => expect(screen.getByText(/Session status unavailable/)).toBeInTheDocument());
    await waitFor(() => expect(screen.getByText("Decision Snapshot — Not Current Market")).toBeInTheDocument());
  });
});

describe("1/2. successful latest recommendation rendering (WAIT)", () => {
  it("renders the decision status, timestamp and market snapshot", async () => {
    server.use(recommendationSuccess(waitResponse()));
    render(<App />);
    await waitFor(() => expect(screen.getAllByText("WAIT")[0]).toBeInTheDocument());
    expect(screen.getByText("Decision Snapshot — Not Current Market")).toBeInTheDocument();
    expect(screen.getAllByText("2026-10-06").length).toBeGreaterThan(0);
  });
});

describe("3. NO_TRADE rendering", () => {
  it("renders NO_TRADE and rejected-shape reasons, never a fabricated trade", async () => {
    server.use(recommendationSuccess(noTradeResponse()));
    render(<App />);
    await waitFor(() => expect(screen.getAllByText("NO_TRADE").length).toBeGreaterThan(0));
    expect(screen.getByText("No candidate was selected for this decision.")).toBeInTheDocument();
  });
});

describe("4. TRADE rendering (synthetic fixture only)", () => {
  it("renders TRADE verbatim without the frontend altering or blocking it", async () => {
    server.use(recommendationSuccess(tradeResponse()));
    render(<App />);
    await waitFor(() => expect(screen.getAllByText("TRADE").length).toBeGreaterThan(0));
  });
});

describe("5. all three candidate shapes are shown", () => {
  it("never hides an unselected shape", async () => {
    server.use(recommendationSuccess(waitResponse()));
    render(<App />);
    await waitFor(() => expect(screen.getByText("Candidate Comparison")).toBeInTheDocument());
    const panel = screen.getByText("Candidate Comparison").closest(".panel") as HTMLElement;
    const cards = within(panel);
    expect(cards.getByText("LONG_CALL_ATM")).toBeInTheDocument();
    expect(cards.getByText("LONG_PUT_ATM")).toBeInTheDocument();
    expect(cards.getByText("LONG_STRADDLE_ATM")).toBeInTheDocument();
  });
});

describe("6. leg quote fields are rendered", () => {
  it("shows bid/ask/volume/OI/LTP/quote status for the selected candidate's leg", async () => {
    server.use(recommendationSuccess(waitResponse()));
    render(<App />);
    await waitFor(() => expect(screen.getByText("Selected Option")).toBeInTheDocument());
    const panel = screen.getByText("Selected Option").closest(".panel") as HTMLElement;
    const table = within(panel);
    expect(table.getByText("138")).toBeInTheDocument();
    expect(table.getByText("142")).toBeInTheDocument();
    expect(table.getByText("500")).toBeInTheDocument();
    expect(table.getByText("1000")).toBeInTheDocument();
    expect(table.getByText("QUOTED")).toBeInTheDocument();
  });
});

describe("7. risk-neutral probability is clearly labeled, never called win probability", () => {
  it("labels the Black-76 baseline explicitly and never uses forbidden phrasing", async () => {
    server.use(recommendationSuccess(waitResponse()));
    render(<App />);
    await waitFor(() => expect(screen.getByText(/Black-76 baseline/)).toBeInTheDocument());
    const bodyText = document.body.textContent ?? "";
    for (const forbidden of ["win probability", "success probability", "accuracy"]) {
      expect(bodyText.toLowerCase()).not.toContain(forbidden);
    }
  });
});

describe("8. predictive probability NOT_ESTABLISHED", () => {
  it("renders the literal status, never invented", async () => {
    server.use(recommendationSuccess(waitResponse()));
    render(<App />);
    await waitFor(() => expect(screen.getByText("NOT_ESTABLISHED")).toBeInTheDocument());
  });
});

describe("9. NOT_MODELED fields remain explicit", () => {
  it("renders costs/risk/EV as 'Not modeled', never a fabricated number", async () => {
    server.use(recommendationSuccess(waitResponse()));
    render(<App />);
    await waitFor(() => expect(screen.getAllByText("Not modeled").length).toBe(3));
  });
});

describe("10. no selected candidate", () => {
  it("shows an explicit message rather than inventing a trade", async () => {
    server.use(recommendationSuccess(noCandidateResponse()));
    render(<App />);
    await waitFor(() => expect(screen.getByText("No candidate was selected for this decision.")).toBeInTheDocument());
  });
});

describe("11. 404 / no recommendation", () => {
  it("shows a calm empty state, not an error", async () => {
    server.use(recommendationNotFound());
    render(<App />);
    await waitFor(() => expect(screen.getByText(/No recommendation has been generated yet/)).toBeInTheDocument());
  });
});

describe("12. API unavailable", () => {
  it("fails safely rather than showing stale/fabricated data", async () => {
    server.use(recommendationUnreachable());
    render(<App />);
    await waitFor(() => expect(screen.getByText(/cannot reach the API/i)).toBeInTheDocument());
  });
});

describe("13. authentication failure", () => {
  it("shows a distinct auth message and never displays a key value", async () => {
    server.use(recommendationUnauthorized());
    render(<App />);
    await waitFor(() => expect(screen.getByText(/Authentication required/i)).toBeInTheDocument());
    expect(document.body.textContent ?? "").not.toMatch(/X-API-Key.*:.*\S+/);
  });
});

describe("14. malformed response", () => {
  it("handles invalid JSON safely", async () => {
    server.use(recommendationMalformedJson());
    render(<App />);
    await waitFor(() => expect(screen.getByText(/unexpected response/i)).toBeInTheDocument());
  });

  it("handles an unexpected but valid JSON shape safely", async () => {
    server.use(recommendationMalformedShape());
    render(<App />);
    await waitFor(() => expect(screen.getByText(/unexpected response/i)).toBeInTheDocument());
  });
});

describe("15. stale decision handling", () => {
  it("shows a STALE warning for a decision older than 15 minutes, without changing its status", async () => {
    server.use(recommendationSuccess(staleResponse()));
    render(<App />);
    await waitFor(() => expect(screen.getByText("STALE — OLD DECISION")).toBeInTheDocument());
    expect(screen.getAllByText("WAIT")[0]).toBeInTheDocument();
  });

  it("shows no STALE warning for a fresh decision", async () => {
    server.use(recommendationSuccess(waitResponse()));
    render(<App />);
    await waitFor(() => expect(screen.getAllByText("WAIT")[0]).toBeInTheDocument());
    expect(screen.queryByText("STALE — OLD DECISION")).not.toBeInTheDocument();
  });
});

describe("16. refresh behaviour", () => {
  it("polls again and reflects a changed backend response", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      server.use(recommendationSuccess(waitResponse()));
      render(<App />);
      await waitFor(() => expect(screen.getAllByText("WAIT")[0]).toBeInTheDocument());

      server.use(recommendationSuccess(noTradeResponse()));
      await act(async () => {
        await vi.advanceTimersByTimeAsync(30_000);
      });
      await waitFor(() => expect(screen.getAllByText("NO_TRADE").length).toBeGreaterThan(0));
    } finally {
      vi.useRealTimers();
    }
  });

  it("does not poll faster than the configured interval (30s)", async () => {
    let requestCount = 0;
    server.use(
      recommendationSuccess(waitResponse()),
    );
    server.events.on("request:start", ({ request }) => {
      if (request.url.includes("/api/recommendations/latest")) requestCount += 1;
    });
    render(<App />);
    await waitFor(() => expect(screen.getAllByText("WAIT")[0]).toBeInTheDocument());
    await new Promise((resolve) => setTimeout(resolve, 200));
    expect(requestCount).toBe(1);
  });
});

describe("17. no Dhan network calls from the frontend", () => {
  it("every request goes only to the mocked API base - MSW's onUnhandledRequest: 'error' fails on anything else", async () => {
    server.use(recommendationSuccess(waitResponse()));
    render(<App />);
    await waitFor(() => expect(screen.getAllByText("WAIT")[0]).toBeInTheDocument());
  });
});

describe("18. no futures terminology anywhere in rendered output", () => {
  it("never mentions futures/BANKNIFTY", async () => {
    server.use(recommendationSuccess(waitResponse()));
    render(<App />);
    await waitFor(() => expect(screen.getAllByText("WAIT")[0]).toBeInTheDocument());
    const bodyText = (document.body.textContent ?? "").toLowerCase();
    for (const forbidden of ["future", "banknifty"]) {
      expect(bodyText).not.toContain(forbidden);
    }
  });
});

describe("19. basic/responsive rendering", () => {
  it("renders without crashing at a narrow viewport width", async () => {
    Object.defineProperty(window, "innerWidth", { writable: true, configurable: true, value: 375 });
    server.use(recommendationSuccess(waitResponse()));
    render(<App />);
    await waitFor(() => expect(screen.getAllByText("WAIT")[0]).toBeInTheDocument());
  });
});

describe("500/503 server error", () => {
  it("shows a temporary-unavailability message", async () => {
    server.use(recommendationServerError());
    render(<App />);
    await waitFor(() => expect(screen.getByText(/temporarily unable to respond/i)).toBeInTheDocument());
  });
});
