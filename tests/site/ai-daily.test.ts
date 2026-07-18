import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { describe, expect, it } from "vitest";
import { resolveMetadataDisplay, type SidecarMetadata } from "../../scripts/lib/metadata";
import { buildPagefind, buildSite, pagefindLanguages, repoRoot, searchPagefind } from "./site-build";

const lessonCount = () => readdirSync(path.join(repoRoot, "src", "content", "ai-daily"))
  .filter((entry) => entry.endsWith(".md")).length;

describe("AI Daily archive and articles", () => {
  it("builds both archives with one chronological card per current lesson and accessible filter/search wiring", () => {
    const site = buildSite("astro-ai-daily-");
    try {
      const archive = readFileSync(path.join(site.outputDirectory, "ai-daily", "index.html"), "utf8");
      const chineseArchive = readFileSync(path.join(site.outputDirectory, "zh", "ai-daily", "index.html"), "utf8");

      expect(archive.match(/<article[^>]*class="lesson-card /g)).toHaveLength(lessonCount());
      expect(archive).toContain('data-track="A"');
      expect(archive).toContain('data-depth="L1"');
      expect(archive).toContain('data-date="2026-07-17"');
      expect(archive.indexOf("2026-07-17")).toBeLessThan(archive.indexOf("2026-07-06"));
      expect(archive).toContain('data-lesson-filters');
      expect(archive).toContain('data-lesson-no-results');
      expect(archive).toContain('data-lesson-clear');
      expect(archive).toContain('aria-live="polite"');
      expect(archive).toContain('data-lesson-search');
      expect(archive).toContain('/pagefind/pagefind-ui.js');
      expect(archive).toContain('<a href="/ai-daily/" aria-current="page">AI Daily</a>');
      expect(archive).toContain('/ai-daily/prompt-engineering-foundations/');
      expect(chineseArchive).toContain('/ai-daily/prompt-engineering-foundations/');
      expect(chineseArchive).toContain('data-lesson-filters');
      expect(chineseArchive).toContain('/pagefind/pagefind-ui.js');
      expect(chineseArchive).toContain('<a href="/zh/ai-daily/" aria-current="page">AI Daily</a>');
    } finally {
      site.dispose();
    }
  });

  it("builds one canonical Chinese-body detail page per lesson without Chinese duplicates", () => {
    const site = buildSite("astro-ai-daily-");
    try {
      const articleDirectory = path.join(site.outputDirectory, "ai-daily");
      const articlePages = readdirSync(articleDirectory).filter((entry) =>
        existsSync(path.join(articleDirectory, entry, "index.html")),
      );
      const article = readFileSync(
        path.join(articleDirectory, "prompt-engineering-foundations", "index.html"),
        "utf8",
      );
      const middleArticle = readFileSync(
        path.join(articleDirectory, "attention-transformer-architecture", "index.html"),
        "utf8",
      );

      expect(articlePages).toHaveLength(lessonCount());
      expect(existsSync(path.join(site.outputDirectory, "zh", "ai-daily", "prompt-engineering-foundations", "index.html"))).toBe(false);
      expect(article).toMatch(/<article\b(?=[^>]*\blang="zh-CN")(?=[^>]*\bdata-pagefind-body)[^>]*>/);
      expect(article).toContain("提示工程的本质");
      expect(article).toContain("<table");
      expect(article).toContain("<code");
      expect(article).toContain("katex");
      const stylesheetHref = /<link rel="stylesheet" href="([^"]+)"/.exec(article)?.[1];
      expect(stylesheetHref).toMatch(/^\/_astro\/[^"/]+\.css$/);
      const stylesheet = readFileSync(path.join(site.outputDirectory, stylesheetHref!.slice(1)), "utf8");
      expect(stylesheet).toContain(".katex");
      expect(article).toContain('data-pagefind-body');
      expect(article).toContain('<a href="/ai-daily/" aria-current="page">AI Daily</a>');
      expect(article).toContain('data-article-toc');
      expect(article).toContain('data-lesson-pager');
      expect(middleArticle).toContain('?track=B');
      expect(article).toContain("Personal learning note");
      expect(article).toContain("AI-assisted");
      expect(article).toContain("Verify important sources");
      expect(article).toContain('/ai-daily/');
    } finally {
      site.dispose();
    }
  });

  it("builds a Chinese Pagefind index whose Chinese results use canonical article URLs", async () => {
    const site = buildSite("astro-pagefind-");
    try {
      const packageJson = readFileSync(path.join(repoRoot, "package.json"), "utf8");
      buildPagefind(site.outputDirectory);
      const languages = pagefindLanguages(site.outputDirectory);
      const urls = await searchPagefind(site.outputDirectory, "提示工程");

      expect(packageJson).toContain('pagefind --site dist --force-language zh');
      expect(languages).toEqual({ zh: expect.objectContaining({ page_count: lessonCount() }) });
      expect(urls).toContain("https://pagefind.test/ai-daily/prompt-engineering-foundations/");
      expect(urls.some((url) => url.includes("/zh/ai-daily/"))).toBe(false);
    } finally {
      site.dispose();
    }
  });

  it("uses English metadata only while it remains hash-bound and current", () => {
    const parsed = matter(
      readFileSync(path.join(repoRoot, "src/content/ai-daily/lesson-0001.md"), "utf8"),
    ).data as Record<string, unknown>;
    const lesson = {
      ...parsed,
      date: parsed.date instanceof Date ? parsed.date.toISOString().slice(0, 10) : parsed.date,
    } as SidecarMetadata;
    const current = resolveMetadataDisplay(lesson);
    const stale = resolveMetadataDisplay({
      ...lesson,
      source: { ...lesson.source, hash: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" },
      sourceStatusHash: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      metadataStatus: "needs-review",
    });

    expect(current.language).toBe("en");
    expect(stale.language).toBe("zh");
    expect(stale.title).toBe(lesson.titleZh);
    expect(stale.originalInChinese).toBe(true);
  });
});
