import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

const pages = ["/", "/ai-daily/", "/ai-daily/constrained-decoding-format-tax-tool-routing/", "/projects/", "/about/"];

for (const pagePath of pages) {
  test(`${pagePath} has no serious accessibility violations`, async ({ page }) => {
    await page.goto(pagePath);
    const results = await new AxeBuilder({ page }).analyze();
    expect(results.violations.filter((violation) => ["serious", "critical"].includes(violation.impact ?? ""))).toEqual([]);
  });
}

test("skip link, navigation, theme control, and archive filters are keyboard accessible", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("main")).toBeVisible();
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await page.keyboard.press("Tab");
  const skipLink = page.getByRole("link", { name: "Skip to content" });
  await expect(skipLink).toBeFocused();
  await expect(skipLink).toBeVisible();
  await skipLink.press("Enter");
  await expect(page.locator("main")).toBeFocused();

  await page.goto("/");
  await page.keyboard.press("Tab");
  await page.keyboard.press("Tab");
  await page.keyboard.press("Tab");
  await page.keyboard.press("Tab");
  const archiveLink = page.getByRole("navigation", { name: "Primary navigation" }).getByRole("link", { name: "AI Daily" });
  await expect(archiveLink).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/ai-daily\/$/);

  await page.goto("/");
  for (let index = 0; index < 8; index += 1) await page.keyboard.press("Tab");
  const themeToggle = page.getByRole("button", { name: /theme/i });
  await expect(themeToggle).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator("html")).toHaveAttribute("data-theme-preference", "light");
  await expect(themeToggle).toHaveAccessibleName(/Light theme/i);

  await page.goto("/ai-daily/");
  const trackA = page.getByRole("checkbox", { name: "Track A" });
  for (let index = 0; index < 12; index += 1) {
    if (await trackA.evaluate((element) => element === document.activeElement)) break;
    await page.keyboard.press("Tab");
  }
  await expect(trackA).toBeFocused();
  await trackA.press("Space");
  await expect(trackA).toBeChecked();
});
