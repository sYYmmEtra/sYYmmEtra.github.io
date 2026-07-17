import { execFileSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const distDirectory = path.join(repoRoot, "dist");

function dictionaryKeys(value: Record<string, unknown>, prefix = ""): string[] {
  return Object.entries(value).flatMap(([key, child]) => {
    const name = prefix ? `${prefix}.${key}` : key;
    return child && typeof child === "object" && !Array.isArray(child)
      ? dictionaryKeys(child as Record<string, unknown>, name)
      : [name];
  });
}

describe("Astro site foundation", () => {
  it("defines a reusable display heading with the approved tight typography", () => {
    const styles = readFileSync(
      path.join(repoRoot, "src/styles/global.css"),
      "utf8",
    );

    expect(styles).toMatch(
      /\.display-heading\s*\{(?=[^}]*font-family:\s*inherit;)(?=[^}]*font-optical-sizing:\s*auto;)(?=[^}]*line-height:\s*1\.05;)(?=[^}]*letter-spacing:\s*-0\.0[3-5]em;)[^}]*\}/s,
    );
  });

  it("keeps hover links legible and gives brand and footer links 44px targets", () => {
    const styles = readFileSync(
      path.join(repoRoot, "src/styles/global.css"),
      "utf8",
    );

    expect(styles).toMatch(
      /a:hover\s*\{(?=[^}]*color:\s*var\(--ink\);)(?=[^}]*text-decoration-color:\s*var\(--track-a\);)[^}]*\}/s,
    );
    expect(styles).toMatch(
      /\.site-nav__brand\s*\{(?=[^}]*min-height:\s*2\.75rem;)(?=[^}]*display:\s*flex;)[^}]*\}/s,
    );
    expect(styles).toMatch(
      /\.site-footer a\s*\{(?=[^}]*min-height:\s*2\.75rem;)(?=[^}]*display:\s*inline-flex;)[^}]*\}/s,
    );
  });

  it("keeps a system-aware three-state theme control in sync with OS changes", () => {
    const baseLayout = readFileSync(
      path.join(repoRoot, "src/layouts/BaseLayout.astro"),
      "utf8",
    );
    const themeToggle = readFileSync(
      path.join(repoRoot, "src/components/ThemeToggle.astro"),
      "utf8",
    );

    expect(baseLayout).toContain(
      'stored === "dark" || stored === "light" || stored === "system"',
    );
    expect(baseLayout).toContain(': "system";');
    expect(baseLayout).toContain("dataset.themePreference = preference");
    expect(themeToggle).toContain(
      'const nextPreference = { system: "light", light: "dark", dark: "system" };',
    );
    expect(themeToggle).toContain('localStorage.removeItem("theme")');
    expect(themeToggle).toContain('colorScheme.addEventListener("change"');
    expect(themeToggle).toContain("data-system-label");
  });

  it("loads all generated lessons, keeps bilingual keys aligned, and builds the shared home shell", async () => {
    expect(existsSync(path.join(repoRoot, "src/content.config.ts"))).toBe(true);

    const { en, zhCN } = await import("../../src/data/i18n");
    expect(dictionaryKeys(en).sort()).toEqual(dictionaryKeys(zhCN).sort());

    const sentinel = path.join(
      distDirectory,
      `.routes-test-${process.pid}-${Date.now()}.sentinel`,
    );
    const outputDirectory = mkdtempSync(path.join(tmpdir(), "astro-routes-"));
    mkdirSync(distDirectory, { recursive: true });
    writeFileSync(sentinel, "do not remove\n");

    try {
      execFileSync(
        process.execPath,
        [
          path.join(repoRoot, "node_modules/astro/bin/astro.mjs"),
          "build",
          "--outDir",
          outputDirectory,
        ],
        { cwd: repoRoot, stdio: "pipe" },
      );

      const home = readFileSync(
        path.join(outputDirectory, "index.html"),
        "utf8",
      );
      expect(home).toContain('data-content-entries="13"');
      expect(home).toContain('href="#main-content"');
      expect(home).toMatch(/<html lang="en"[^>]*data-theme=/);
      expect(home).not.toContain('href="/ai-daily/"');
      expect(home).not.toContain('href="/projects/"');
      expect(home).not.toContain('href="/about/"');
      expect(home).toContain("data-system-label");
      expect(readFileSync(sentinel, "utf8")).toBe("do not remove\n");
      expect(existsSync(path.join(outputDirectory, "projects", "index.html"))).toBe(false);
      expect(existsSync(path.join(outputDirectory, "about", "index.html"))).toBe(false);
      expect(existsSync(path.join(outputDirectory, "ai-daily", "index.html"))).toBe(false);
    } finally {
      rmSync(sentinel, { force: true });
      rmSync(outputDirectory, { recursive: true, force: true });
    }
  });
});
