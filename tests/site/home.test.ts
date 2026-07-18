import { execFileSync } from "node:child_process";
import {
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { currentTrackCounts } from "./content-fixture";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

function buildSite(): { outputDirectory: string; dispose: () => void } {
  const outputDirectory = mkdtempSync(path.join(tmpdir(), "astro-home-"));

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

  return {
    outputDirectory,
    dispose: () => rmSync(outputDirectory, { recursive: true, force: true }),
  };
}

describe("Clear Workbench personal pages", () => {
  it("renders the English home with its thesis, learning counts, and usable paths", () => {
    const site = buildSite();

    try {
      const home = readFileSync(path.join(site.outputDirectory, "index.html"), "utf8");

      expect(home).toContain("sYYmmEtra");
      expect(home).toContain("Turning daily AI learning into engineering judgment.");
      for (const [track, count] of Object.entries(currentTrackCounts(repoRoot))) {
        expect(home).toMatch(new RegExp(`Track ${track}[\\s\\S]*>${count}<`));
      }
      expect(home).toContain('href="#latest-lessons"');
      expect(home).toContain('href="https://github.com/sYYmmEtra"');
      expect(home).toContain('href="/projects/"');
      expect(home).toContain('href="/about/"');
      expect(home).toContain('href="mailto:private-contact@example.invalid"');
    } finally {
      site.dispose();
    }
  });

  it("renders translated Chinese routes without project cards", () => {
    const site = buildSite();

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

  it("renders useful localized empty states for an empty latest-lessons collection", () => {
    const fixturePath = path.join(repoRoot, "src/pages/latest-lessons-empty-fixture.astro");
    writeFileSync(
      fixturePath,
      `---\nimport LatestLessons from "../components/LatestLessons.astro";\n---\n<LatestLessons lessons={[]} />\n<LatestLessons locale="zh-CN" lessons={[]} />\n`,
    );

    let site: ReturnType<typeof buildSite> | undefined;
    try {
      site = buildSite();
      const page = readFileSync(
        path.join(site.outputDirectory, "latest-lessons-empty-fixture", "index.html"),
        "utf8",
      );

      expect(page).toContain("No lessons are available in the current corpus yet.");
      expect(page).toContain("当前学习语料中暂时没有课程。");
      expect(page).not.toContain('class="lesson-list"');
    } finally {
      site?.dispose();
      rmSync(fixturePath, { force: true });
    }
  });
});
