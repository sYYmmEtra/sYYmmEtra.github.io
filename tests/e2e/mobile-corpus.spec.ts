import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { expect, test } from "@playwright/test";

const lessons = readdirSync(path.resolve("src/content/ai-daily"))
  .filter((file) => file.endsWith(".md"))
  .map((file) => matter(readFileSync(path.resolve("src/content/ai-daily", file), "utf8")).data.slug as string);

test.use({ viewport: { width: 390, height: 844 }, isMobile: true });

for (const slug of lessons) {
  test(`mobile article ${slug} contains wide content without document overflow`, async ({ page }) => {
    await page.goto(`/ai-daily/${slug}/`);
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(392);
    await expect(await page.locator("pre").evaluateAll((nodes) => nodes.every((node) => node.scrollWidth <= node.clientWidth || getComputedStyle(node).overflowX === "auto"))).toBe(true);
    await expect(await page.locator(".katex-display").evaluateAll((nodes) => nodes.every((node) => node.scrollWidth <= node.clientWidth || getComputedStyle(node).overflowX === "auto"))).toBe(true);
  });
}
