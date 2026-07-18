import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    // Astro writes a shared .astro prerender cache during a build. Site tests
    // intentionally build in-process, so they must not overlap.
    fileParallelism: false,
  },
});
