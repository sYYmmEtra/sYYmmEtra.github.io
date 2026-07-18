import { expect, test } from "@playwright/test";

test.use({ viewport: { width: 390, height: 844 }, isMobile: true });

test("mobile article keeps long inline code within the viewport", async ({ page }) => {
  await page.goto("/ai-daily/constrained-decoding-format-tax-tool-routing/");
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
});
