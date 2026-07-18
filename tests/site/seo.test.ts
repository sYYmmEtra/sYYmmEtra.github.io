import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { rssItemFromMetadata } from "../../src/lib/rss";
import type { SidecarMetadata } from "../../scripts/lib/metadata";
import { buildSite, repoRoot } from "./site-build";

const origin = "https://syymmetra.github.io";
const lessonCount = () => readdirSync(path.join(repoRoot, "src", "content", "ai-daily"))
  .filter((entry) => entry.endsWith(".md")).length;
const uiPairs = [["/", "/zh/"], ["/ai-daily/", "/zh/ai-daily/"], ["/projects/", "/zh/projects/"], ["/about/", "/zh/about/"]] as const;

function pageFile(outputDirectory: string, pathname: string): string {
  return pathname === "/"
    ? path.join(outputDirectory, "index.html")
    : path.join(outputDirectory, pathname.slice(1), "index.html");
}

function attributes(tag: string): Record<string, string> {
  return Object.fromEntries([...tag.matchAll(/([\w:-]+)="([^"]*)"/g)].map(([, name, value]) => [name, value]));
}

function headTags(document: string): Array<{ name: string; attributes: Record<string, string> }> {
  const head = document.match(/<head>([\s\S]*?)<\/head>/)?.[1] ?? "";
  return [...head.matchAll(/<(link|meta)\b[^>]*>/g)].map((match) => ({ name: match[1], attributes: attributes(match[0]) }));
}

function jsonLd(document: string): unknown[] {
  return [...document.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)].map((match) => JSON.parse(match[1]));
}

function xmlField(item: string, name: string): string {
  const value = item.match(new RegExp(`<${name}>([\\s\\S]*?)<\\/${name}>`))?.[1];
  expect(value, `missing ${name}`).toBeDefined();
  return value!
    .replace(/^<!\[CDATA\[([\s\S]*)\]\]>$/, "$1")
    .replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
}

function sidecar(overrides: Partial<SidecarMetadata> = {}): SidecarMetadata {
  const hash = `sha256:${"a".repeat(64)}`;
  return {
    id: "lesson-0013", source: { file: "lessons/2026-07-17.md", section: 1, hash }, lesson: 13, date: "2026-07-17", track: "A", depth: "L3",
    titleZh: "结构化输出与工具调用", titleEn: "Structured Output & Tool Routing", summaryZh: "结构化输出保证格式合法，但不保证语义正确。", summaryEn: "Structured output makes machine interfaces reliable while demanding independent checks for semantic quality and tool routing decisions in production systems across changing schemas, prompts, data sources, observability, evaluation, safeguards, deployment workflows, maintenance, and responsible engineering practice every day.",
    slug: "structured-output-tool-routing", tags: ["structured-output", "tool-routing"], sourceStatus: "unreviewed", sourceStatusHash: hash, metadataStatus: "current", metadataSourceHash: hash, featured: false,
    ...overrides,
  };
}

describe("SEO and public syndication", () => {
  it("builds one canonical and an exact paired hreflang set for every UI page", () => {
    const site = buildSite("astro-seo-");
    try {
      for (const [englishPath, chinesePath] of uiPairs) {
        for (const [pathname, expectedEnglish, expectedChinese] of [[englishPath, englishPath, chinesePath], [chinesePath, englishPath, chinesePath]] as const) {
          const tags = headTags(readFileSync(pageFile(site.outputDirectory, pathname), "utf8"));
          const canonicals = tags.filter((tag) => tag.name === "link" && tag.attributes.rel === "canonical");
          expect(canonicals).toEqual([{ name: "link", attributes: expect.objectContaining({ rel: "canonical", href: `${origin}${pathname}` }) }]);
          const hreflang = tags.filter((tag) => tag.name === "link" && tag.attributes.rel === "alternate" && tag.attributes.hreflang);
          expect(hreflang.map((tag) => [tag.attributes.hreflang, tag.attributes.href]).sort()).toEqual([
            ["en", `${origin}${expectedEnglish}`], ["x-default", `${origin}${expectedEnglish}`], ["zh-CN", `${origin}${expectedChinese}`],
          ]);
        }
      }
      const home = readFileSync(pageFile(site.outputDirectory, "/"), "utf8");
      const about = readFileSync(pageFile(site.outputDirectory, "/about/"), "utf8");
      expect(jsonLd(home).filter((value) => (value as { "@type"?: string })["@type"] === "Person")).toHaveLength(1);
      expect(about).toContain('src="/avatar.png" width="460" height="460"');
      for (const asset of ["favicon.svg", "og-default.svg", "robots.txt", "avatar.png"]) expect(existsSync(path.join(site.outputDirectory, asset))).toBe(true);
      expect(readFileSync(path.join(site.outputDirectory, "avatar.png")).subarray(0, 8).toString("hex")).toBe("89504e470d0a1a0a");
    } finally { site.dispose(); }
  });

  it("publishes one article canonical and BlogPosting object without locale-detail alternates", () => {
    const site = buildSite("astro-seo-");
    try {
      const pathname = "/ai-daily/constrained-decoding-format-tax-tool-routing/";
      const article = readFileSync(pageFile(site.outputDirectory, pathname), "utf8");
      const tags = headTags(article);
      expect(tags.filter((tag) => tag.name === "link" && tag.attributes.rel === "canonical")).toEqual([{ name: "link", attributes: expect.objectContaining({ href: `${origin}${pathname}` }) }]);
      expect(tags.filter((tag) => tag.attributes.hreflang)).toHaveLength(0);
      const postings = jsonLd(article).filter((value) => (value as { "@type"?: string })["@type"] === "BlogPosting") as Array<Record<string, unknown>>;
      expect(postings).toHaveLength(1);
      expect(postings[0]).toEqual(expect.objectContaining({ inLanguage: "zh-CN", mainEntityOfPage: `${origin}${pathname}`, datePublished: expect.any(String) }));
    } finally { site.dispose(); }
  });

  it("publishes a well-formed newest-first RSS feed of unique canonical article links", () => {
    const site = buildSite("astro-seo-");
    try {
      const rss = readFileSync(path.join(site.outputDirectory, "rss.xml"), "utf8");
      expect(rss).toMatch(/^<\?xml[^>]*\?>\s*<rss\b/);
      expect(rss).not.toMatch(/(?<!&)&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[\da-f]+;)/i);
      const items = [...rss.matchAll(/<item>([\s\S]*?)<\/item>/g)].map((match) => match[1]);
      expect(items).toHaveLength(lessonCount());
      const links = items.map((item) => xmlField(item, "link"));
      const dates = items.map((item) => xmlField(item, "pubDate"));
      expect(new Set(links).size).toBe(lessonCount());
      expect(links.every((link) => link.startsWith(`${origin}/ai-daily/`) && link.endsWith("/"))).toBe(true);
      expect(dates.every((date) => !Number.isNaN(Date.parse(date)))).toBe(true);
      expect(dates.map(Date.parse)).toEqual([...dates.map(Date.parse)].sort((left, right) => right - left));
      for (const item of items) {
        expect(xmlField(item, "title")).not.toBe("");
        expect(xmlField(item, "description")).not.toBe("");
      }
    } finally { site.dispose(); }
  });

  it("uses English RSS metadata only when it remains hash-bound and current", () => {
    expect(rssItemFromMetadata(sidecar())).toEqual(expect.objectContaining({ title: "Structured Output & Tool Routing", description: expect.not.stringContaining("Original in Chinese"), categories: expect.arrayContaining(["structured-output"]) }));
    for (const stale of [sidecar({ metadataStatus: "pending", metadataSourceHash: null }), sidecar({ metadataStatus: "needs-review", metadataSourceHash: `sha256:${"b".repeat(64)}` })]) {
      expect(rssItemFromMetadata(stale)).toEqual(expect.objectContaining({ title: stale.titleZh, description: expect.stringMatching(/^Original in Chinese — /), categories: ["Track A", "L3"] }));
    }
  });
});
