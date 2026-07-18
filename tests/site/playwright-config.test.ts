import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { repoRoot } from "./site-build";

describe("Playwright server isolation", () => {
  it("reuses a preview server only when explicitly requested", () => {
    const config = readFileSync(path.join(repoRoot, "playwright.config.ts"), "utf8");
    expect(config).toContain('reuseExistingServer: process.env.PW_REUSE_SERVER === "1"');
    expect(config).toContain("--host 127.0.0.1 --port 4321");
  });
});
