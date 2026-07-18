import { existsSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { buildSite, withPagefindFetch } from "./site-build";

describe("temporary site-build helpers", () => {
  it("removes the temporary output directory when Astro fails", () => {
    let outputDirectory = "";
    expect(() => buildSite("astro-failure-", (_file, args) => {
      outputDirectory = args.at(-1)!;
      throw new Error("Astro failed");
    })).toThrow("Astro failed");
    expect(existsSync(outputDirectory)).toBe(false);
  });

  it("restores the original fetch implementation when Pagefind setup fails", async () => {
    const originalFetch = globalThis.fetch;
    await expect(withPagefindFetch("/tmp/unused-pagefind", async () => {
      throw new Error("Pagefind setup failed");
    })).rejects.toThrow("Pagefind setup failed");
    expect(globalThis.fetch).toBe(originalFetch);
  });
});
