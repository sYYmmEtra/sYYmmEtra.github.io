import {
  readFileSync,
} from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { currentTrackCounts } from "./content-fixture";
import { buildSite, repoRoot } from "./site-build";

describe("Clear Workbench personal pages", () => {
  it("renders the English home with its thesis, learning counts, and usable paths", () => {
    const site = buildSite("astro-home-");

    try {
      const home = readFileSync(path.join(site.outputDirectory, "index.html"), "utf8");

      expect(home).toContain("sYYmmEtra");
      expect(home).toContain("Turning daily AI learning into engineering judgment.");
      for (const [track, count] of Object.entries(currentTrackCounts(repoRoot))) {
        expect(home).toMatch(new RegExp(`Track ${track}[\\s\\S]*>${count}<`));
      }
      expect(home).toContain('href="/ai-daily/"');
      expect(home).toContain('href="https://github.com/sYYmmEtra"');
      expect(home).toContain('href="/projects/"');
      expect(home).toContain('href="/about/"');
      expect(home).toContain('href="/ai-daily/constrained-decoding-format-tax-tool-routing/"');
      expect(home).toContain('href="mailto:private-contact@example.invalid"');
    } finally {
      site.dispose();
    }
  });

  it("renders translated Chinese routes without project cards", () => {
    const site = buildSite("astro-home-");

    try {
      const home = readFileSync(path.join(site.outputDirectory, "zh", "index.html"), "utf8");
      const projects = readFileSync(
        path.join(site.outputDirectory, "zh", "projects", "index.html"),
        "utf8",
      );
      const about = readFileSync(path.join(site.outputDirectory, "zh", "about", "index.html"), "utf8");

      expect(home).toContain("将每日 AI 学习转化为工程判断。");
      expect(home).toContain("学习脉冲");
      expect(home).toContain("轨道 A");
      expect(home).not.toContain("Track A");
      expect(home).not.toContain("Track B");
      expect(home).not.toContain("Track C");
      expect(projects).toContain("尚未发布项目");
      expect(projects).not.toContain("project-card");
      expect(about).toContain("联系我");
    } finally {
      site.dispose();
    }
  });

  it("keeps localized empty-state rendering in the reusable component without creating source fixtures", () => {
    const component = readFileSync(path.join(repoRoot, "src/components/LatestLessons.astro"), "utf8");
    expect(component).toContain("latest.length === 0");
    expect(component).toContain("copy.home.emptyLatestLessons");
    expect(component).not.toContain("writeFileSync");
  });
});
