import { describe, expect, it } from "vitest";
import { fetchHealth, fetchLatestRecommendation } from "./client";
import {
  healthOk,
  healthUnreachable,
  recommendationMalformedJson,
  recommendationMalformedShape,
  recommendationNotFound,
  recommendationServerError,
  recommendationSuccess,
  recommendationUnauthorized,
  recommendationUnreachable,
} from "../test/msw/handlers";
import { server } from "../test/msw/server";
import { waitResponse } from "../test/fixtures";

describe("fetchLatestRecommendation", () => {
  it("returns ok:true with the parsed body on success", async () => {
    server.use(recommendationSuccess(waitResponse()));
    const result = await fetchLatestRecommendation("NIFTY");
    expect(result.ok).toBe(true);
  });

  it("classifies 404 as NOT_FOUND", async () => {
    server.use(recommendationNotFound());
    const result = await fetchLatestRecommendation("NIFTY");
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error.kind).toBe("NOT_FOUND");
  });

  it("classifies 401 as AUTH", async () => {
    server.use(recommendationUnauthorized());
    const result = await fetchLatestRecommendation("NIFTY");
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error.kind).toBe("AUTH");
  });

  it("classifies 500 as SERVER_ERROR", async () => {
    server.use(recommendationServerError());
    const result = await fetchLatestRecommendation("NIFTY");
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error.kind).toBe("SERVER_ERROR");
  });

  it("classifies a network failure as UNREACHABLE", async () => {
    server.use(recommendationUnreachable());
    const result = await fetchLatestRecommendation("NIFTY");
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error.kind).toBe("UNREACHABLE");
  });

  it("classifies invalid JSON as MALFORMED", async () => {
    server.use(recommendationMalformedJson());
    const result = await fetchLatestRecommendation("NIFTY");
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error.kind).toBe("MALFORMED");
  });

  it("classifies a valid-JSON-but-wrong-shape body as MALFORMED", async () => {
    server.use(recommendationMalformedShape());
    const result = await fetchLatestRecommendation("NIFTY");
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error.kind).toBe("MALFORMED");
  });
});

describe("fetchHealth", () => {
  it("returns ok:true on success", async () => {
    server.use(healthOk());
    const result = await fetchHealth();
    expect(result.ok).toBe(true);
  });

  it("classifies a network failure as UNREACHABLE", async () => {
    server.use(healthUnreachable());
    const result = await fetchHealth();
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error.kind).toBe("UNREACHABLE");
  });
});
