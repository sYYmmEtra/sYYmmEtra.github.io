import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import matter from "gray-matter";

const knownChineseTerm = (matter(readFileSync("src/content/ai-daily/lesson-0001.md", "utf8")).data.titleZh as string).slice(0, 2);

test("AI Daily Pagefind search returns Chinese lesson matches", async ({ page }) => {
  await page.goto("/ai-daily/");

  const search = page.locator("input.pagefind-ui__search-input");
  await expect(search).toBeVisible();
  await search.fill(knownChineseTerm);
  await expect(page.locator(".pagefind-ui__result")).not.toHaveCount(0);
});

test("Pagefind results remain below the search control with readable dark-theme text", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "dark" });
  await page.goto("/ai-daily/");
  const search = page.locator("input.pagefind-ui__search-input");
  await search.fill(knownChineseTerm);
  const form = page.locator(".pagefind-ui__form");
  const drawer = page.locator(".pagefind-ui__drawer");
  await expect(drawer).toBeVisible();
  await expect(page.locator(".pagefind-ui__result-link").first()).toBeVisible();
  const geometry = await page.evaluate(() => {
    const input = document.querySelector(".pagefind-ui__search-input")!.getBoundingClientRect();
    const drawer = document.querySelector(".pagefind-ui__drawer")!.getBoundingClientRect();
    const result = document.querySelector(".pagefind-ui__result-link")!;
    const style = getComputedStyle(result);
    return { below: drawer.top >= input.bottom, width: drawer.width, color: style.color, background: getComputedStyle(document.querySelector(".pagefind-ui")!).backgroundColor };
  });
  expect(geometry.below).toBe(true);
  expect(geometry.width).toBeGreaterThan(200);
  expect(geometry.color).not.toBe(geometry.background);
});
