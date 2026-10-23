import { describe, expect, it } from "vitest";
import { STALE_THRESHOLD_MS, elapsedMs, formatElapsed, formatTimestamp, isStale } from "./time";

describe("staleness threshold", () => {
  const decisionAt = "2026-09-23T05:00:00.000Z";

  it("is exactly 15 minutes", () => {
    expect(STALE_THRESHOLD_MS).toBe(15 * 60 * 1000);
  });

  it("is NOT stale at exactly 15 minutes old", () => {
    const now = new Date(decisionAt).getTime() + 15 * 60 * 1000;
    expect(isStale(decisionAt, now)).toBe(false);
  });

  it("IS stale at 15 minutes and 1 second old", () => {
    const now = new Date(decisionAt).getTime() + 15 * 60 * 1000 + 1000;
    expect(isStale(decisionAt, now)).toBe(true);
  });

  it("is not stale for a fresh decision", () => {
    const now = new Date(decisionAt).getTime() + 60 * 1000;
    expect(isStale(decisionAt, now)).toBe(false);
  });
});

describe("elapsedMs", () => {
  it("computes a positive difference for a past timestamp", () => {
    const past = "2026-09-23T05:00:00.000Z";
    const now = new Date(past).getTime() + 5000;
    expect(elapsedMs(past, now)).toBe(5000);
  });
});

describe("formatElapsed", () => {
  it("formats seconds", () => {
    expect(formatElapsed(5000)).toBe("5s ago");
  });
  it("formats minutes", () => {
    expect(formatElapsed(4 * 60 * 1000)).toBe("4m ago");
  });
  it("formats hours and minutes", () => {
    expect(formatElapsed(2 * 3600 * 1000 + 5 * 60 * 1000)).toBe("2h 5m ago");
  });
  it("formats days", () => {
    expect(formatElapsed(3 * 24 * 3600 * 1000)).toBe("3d ago");
  });
});

describe("formatTimestamp", () => {
  it("returns an em dash for null", () => {
    expect(formatTimestamp(null)).toBe("—");
  });
  it("formats a real ISO timestamp without throwing", () => {
    expect(formatTimestamp("2026-09-23T05:00:00.000Z")).not.toBe("—");
  });
});
