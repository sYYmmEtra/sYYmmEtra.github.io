import { expect, test } from "@playwright/test";

test.use({ viewport: { width: 390, height: 844 }, isMobile: true });

test("mobile navigation closes explicitly and returns focus to its trigger", async ({ page }) => {
  await page.goto("/");

  const trigger = page.getByRole("button", { name: "Open navigation" });
  await expect(trigger).toBeVisible();
  await trigger.click();

  const menu = page.locator("[data-mobile-navigation]");
  const close = menu.getByRole("button", { name: "Close navigation" });
  await expect(menu).toBeVisible();
  await expect(menu.getByRole("link", { name: "Home" })).toBeFocused();

  await close.click();
  await expect(menu).toBeHidden();
  await expect(trigger).toBeFocused();
});

test("mobile navigation closes on outside pointer and leaves page controls available", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Open navigation" }).click();
  await page.locator("main").click();
  await expect(page.locator("[data-mobile-navigation]")).toBeHidden();
});

test("mobile navigation Escape and link activation close the disclosure", async ({ page }) => {
  await page.goto("/");
  const trigger = page.getByRole("button", { name: "Open navigation" });
  await trigger.click();
  await page.keyboard.press("Escape");
  await expect(page.locator("[data-mobile-navigation]")).toBeHidden();
  await expect(trigger).toBeFocused();
  await trigger.click();
  await page.locator("[data-mobile-navigation]").getByRole("link", { name: "AI Daily" }).click();
  await expect(page).toHaveURL(/\/ai-daily\/$/);
});
