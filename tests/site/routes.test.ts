import { execFileSync } from "node:child_process";
import {
  existsSync,
  readFileSync,
} from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { currentTrackCounts } from "./content-fixture";
import { buildSite, repoRoot } from "./site-build";

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
    expect(styles).toMatch(
      /\.hero__context-link\s*\{(?=[^}]*min-height:\s*2\.75rem;)(?=[^}]*display:\s*inline-flex;)(?=[^}]*align-items:\s*center;)(?=[^}]*padding:)[^}]*\}/s,
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
    const { site } = await import("../../src/data/site");
    expect(dictionaryKeys(en).sort()).toEqual(dictionaryKeys(zhCN).sort());

    const siteBuild = buildSite("astro-routes-");
    const outputDirectory = siteBuild.outputDirectory;

    try {
      const home = readFileSync(
        path.join(outputDirectory, "index.html"),
        "utf8",
      );
      const projects = readFileSync(
        path.join(outputDirectory, "projects", "index.html"),
        "utf8",
      );
      const about = readFileSync(
        path.join(outputDirectory, "about", "index.html"),
        "utf8",
      );
      const zhHome = readFileSync(
        path.join(outputDirectory, "zh", "index.html"),
        "utf8",
      );
      const zhProjects = readFileSync(
        path.join(outputDirectory, "zh", "projects", "index.html"),
        "utf8",
      );
      const zhAbout = readFileSync(
        path.join(outputDirectory, "zh", "about", "index.html"),
        "utf8",
      );
      const expectedCount = Object.values(currentTrackCounts(repoRoot)).reduce(
        (total, count) => total + count,
        0,
      );
      expect(home).toContain(`data-content-entries="${expectedCount}"`);
      expect(home).toContain('href="#main-content"');
      expect(home).toMatch(/<html lang="en"[^>]*data-theme=/);
      expect(home).toContain(`<meta name="description" content="${site.description}">`);
      expect(zhHome).toContain(`<meta name="description" content="${site.descriptionZh}">`);
      expect(home).toContain('href="/ai-daily/"');
      expect(home).toContain('href="/projects/"');
      expect(home).toContain('href="/about/"');
      expect(home).toContain("data-system-label");
      expect(home).toContain('<a href="/" aria-current="page">Home</a>');
      expect(projects).toContain('<a href="/projects/" aria-current="page">Projects</a>');
      expect(about).toContain('<a href="/about/" aria-current="page">About</a>');
      expect(zhHome).toContain('<a href="/zh/" aria-current="page">首页</a>');
      expect(zhProjects).toContain('<a href="/zh/projects/" aria-current="page">项目</a>');
      expect(zhAbout).toContain('<a href="/zh/about/" aria-current="page">关于</a>');
      for (const page of [home, projects, about, zhHome, zhProjects, zhAbout]) {
        expect(page.match(/aria-current="page"/g)).toHaveLength(1);
      }
      expect(home).toContain('<a class="site-nav__language" href="/zh/" lang="zh-CN">中文</a>');
      expect(zhProjects).toContain('<a class="site-nav__language" href="/projects/" lang="en">English</a>');
      expect(home).toContain('class="hero__context-link"');
      expect(existsSync(path.join(outputDirectory, "projects", "index.html"))).toBe(true);
      expect(existsSync(path.join(outputDirectory, "about", "index.html"))).toBe(true);
      expect(existsSync(path.join(outputDirectory, "ai-daily", "index.html"))).toBe(true);
    } finally {
      siteBuild.dispose();
    }
  });
});
