import crypto from "node:crypto";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import YAML from "yaml";
import { afterEach, describe, expect, it, vi } from "vitest";

import { runSync } from "../../scripts/sync-ai-daily";

const temporaryRoots: string[] = [];

async function makeRoots(): Promise<{ sourceRoot: string; websiteRoot: string }> {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "blog-sync-"));
  temporaryRoots.push(root);
  const sourceRoot = path.join(root, "ai-daily");
  const websiteRoot = path.join(root, "site");
  await fs.cp(path.resolve("tests/fixtures"), sourceRoot, { recursive: true });
  await fs.mkdir(websiteRoot, { recursive: true });
  return { sourceRoot, websiteRoot };
}

async function hashTree(root: string): Promise<Record<string, string>> {
  const result: Record<string, string> = {};

  async function walk(directory: string): Promise<void> {
    const entries = await fs.readdir(directory, { withFileTypes: true });
    entries.sort((left, right) => left.name.localeCompare(right.name));
    for (const entry of entries) {
      if (entry.name === ".git") continue;
      const absolute = path.join(directory, entry.name);
      const relative = path.relative(root, absolute);
      if (entry.isDirectory()) await walk(absolute);
      if (entry.isFile()) {
        result[relative] = crypto
          .createHash("sha256")
          .update(await fs.readFile(absolute))
          .digest("hex");
      }
      if (entry.isSymbolicLink()) {
        result[relative] = `symlink:${await fs.readlink(absolute)}`;
      }
    }
  }

  await walk(root);
  return result;
}

async function projection(root: string): Promise<Record<string, string>> {
  const values: Record<string, string> = {};
  async function readDirectory(directory: string, relativeRoot: string): Promise<void> {
    const entries = await fs.readdir(directory, { withFileTypes: true });
    for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
      const absolute = path.join(directory, entry.name);
      const relative = path.join(relativeRoot, entry.name);
      if (entry.isDirectory()) await readDirectory(absolute, relative);
      if (entry.isFile()) values[relative] = await fs.readFile(absolute, "utf8");
    }
  }
  for (const relative of ["metadata", "src/content/ai-daily", "sync-index.json"]) {
    const absolute = path.join(root, relative);
    try {
      const stat = await fs.lstat(absolute);
      if (stat.isDirectory()) {
        await readDirectory(absolute, relative);
      } else {
        values[relative] = await fs.readFile(absolute, "utf8");
      }
    } catch (error) {
      if (!(error instanceof Error && "code" in error && error.code === "ENOENT")) {
        throw error;
      }
    }
  }
  return values;
}

async function readSidecar(websiteRoot: string, id: string): Promise<any> {
  return YAML.parse(
    await fs.readFile(
      path.join(websiteRoot, "metadata/ai-daily", `${id}.yml`),
      "utf8",
    ),
  );
}

function splitGeneratedMarkdown(value: string): {
  frontmatter: any;
  body: string;
} {
  expect(value.startsWith("---\n")).toBe(true);
  const end = value.indexOf("---\n", 4);
  expect(end).toBeGreaterThan(3);
  return {
    frontmatter: YAML.parse(value.slice(4, end)),
    body: value.slice(end + 4),
  };
}

async function replaceLessonBody(
  sourceRoot: string,
  from: string,
  to: string,
): Promise<void> {
  const file = path.join(sourceRoot, "lessons/2026-07-06.md");
  const source = await fs.readFile(file, "utf8");
  await fs.writeFile(file, source.replace(from, to));
}

afterEach(async () => {
  vi.restoreAllMocks();
  await Promise.all(
    temporaryRoots.splice(0).map((root) => fs.rm(root, { recursive: true, force: true })),
  );
});

describe("runSync", () => {
  it("syncs two lessons without changing any source file and ignores non-date lessons", async () => {
    const { sourceRoot, websiteRoot } = await makeRoots();
    await fs.writeFile(path.join(sourceRoot, "protected.txt"), "do not touch\n");
    await fs.writeFile(path.join(sourceRoot, "lessons/not-a-date.md"), "# ignored\n");
    const before = await hashTree(sourceRoot);

    const result = await runSync({ websiteRoot, sourceRoot, enrichMetadata: false });

    expect(result).toEqual({
      created: 2,
      changed: 0,
      unchanged: 0,
      pending: 2,
      lessonIds: ["lesson-0001", "lesson-0002"],
    });
    expect(await hashTree(sourceRoot)).toEqual(before);
    expect(
      (await fs.readdir(path.join(websiteRoot, "src/content/ai-daily"))).sort(),
    ).toEqual(["lesson-0001.md", "lesson-0002.md"]);
    expect(
      (await fs.readdir(path.join(websiteRoot, "metadata/ai-daily"))).sort(),
    ).toEqual(["lesson-0001.yml", "lesson-0002.yml"]);
  });

  it("writes complete deterministic sidecars, content, and a versioned sorted index", async () => {
    const { sourceRoot, websiteRoot } = await makeRoots();
    await runSync({ websiteRoot, sourceRoot, enrichMetadata: false });

    const sidecar = await readSidecar(websiteRoot, "lesson-0001");
    expect(sidecar).toEqual({
      id: "lesson-0001",
      source: {
        file: "lessons/2026-07-06.md",
        section: 1,
        hash: expect.stringMatching(/^sha256:[0-9a-f]{64}$/),
      },
      lesson: 1,
      date: "2026-07-06",
      track: "A",
      depth: "L1",
      titleZh: "提示工程基础与常用模式",
      titleEn: null,
      summaryZh: "学习提示工程的基本结构。",
      summaryEn: null,
      slug: "lesson-0001",
      tags: [],
      sourceStatus: "unreviewed",
      sourceStatusHash: expect.stringMatching(/^sha256:[0-9a-f]{64}$/),
      metadataStatus: "pending",
      metadataSourceHash: null,
      featured: false,
    });
    expect(sidecar.sourceStatusHash).toBe(sidecar.source.hash);

    const source = await fs.readFile(
      path.join(sourceRoot, "lessons/2026-07-06.md"),
      "utf8",
    );
    const secondHeading = source.indexOf("# 📅", 1);
    const generated = splitGeneratedMarkdown(
      await fs.readFile(
        path.join(websiteRoot, "src/content/ai-daily/lesson-0001.md"),
        "utf8",
      ),
    );
    expect(generated.frontmatter).toEqual(sidecar);
    expect(generated.body).toBe(source.slice(0, secondHeading));

    expect(
      JSON.parse(await fs.readFile(path.join(websiteRoot, "sync-index.json"), "utf8")),
    ).toEqual({
      schemaVersion: 1,
      lessons: [
        {
          id: "lesson-0001",
          lesson: 1,
          source: { file: "lessons/2026-07-06.md", section: 1 },
          sourceHash: sidecar.source.hash,
          slug: "lesson-0001",
        },
        {
          id: "lesson-0002",
          lesson: 2,
          source: { file: "lessons/2026-07-06.md", section: 2 },
          sourceHash: expect.stringMatching(/^sha256:[0-9a-f]{64}$/),
          slug: "lesson-0002",
        },
      ],
    });
  });

  it("reports an identical second sync as unchanged with byte-identical output", async () => {
    const { sourceRoot, websiteRoot } = await makeRoots();
    await runSync({ websiteRoot, sourceRoot, enrichMetadata: false });
    const before = await projection(websiteRoot);

    const result = await runSync({ websiteRoot, sourceRoot, enrichMetadata: false });

    expect(result).toMatchObject({ created: 0, changed: 0, unchanged: 2, pending: 2 });
    expect(await projection(websiteRoot)).toEqual(before);
  });

  it("marks a changed source needs-review, resets source review, and preserves editorial fields", async () => {
    const { sourceRoot, websiteRoot } = await makeRoots();
    await runSync({ websiteRoot, sourceRoot, enrichMetadata: false });
    const file = path.join(websiteRoot, "metadata/ai-daily/lesson-0001.yml");
    const sidecar = await readSidecar(websiteRoot, "lesson-0001");
    sidecar.slug = "prompt-engineering-foundations";
    sidecar.featured = true;
    sidecar.sourceStatus = "verified";
    sidecar.titleEn = "Prompt Engineering Foundations";
    sidecar.summaryEn =
      "This lesson introduces practical prompt structure, clear instructions, useful context, constraints, and examples while explaining why careful iteration helps improve reliable model responses for everyday tasks without treating prompting as magic or replacing evaluation and domain judgment.";
    sidecar.tags = ["prompt-engineering"];
    sidecar.metadataStatus = "current";
    sidecar.metadataSourceHash = sidecar.source.hash;
    await fs.writeFile(file, YAML.stringify(sidecar));
    await replaceLessonBody(sourceRoot, "第一课正文。", "第一课正文已更新。");

    const result = await runSync({ websiteRoot, sourceRoot, enrichMetadata: false });
    const changed = await readSidecar(websiteRoot, "lesson-0001");

    expect(result.changed).toBe(1);
    expect(changed.slug).toBe("prompt-engineering-foundations");
    expect(changed.featured).toBe(true);
    expect(changed.sourceStatus).toBe("unreviewed");
    expect(changed.sourceStatusHash).toBe(changed.source.hash);
    expect(changed.metadataStatus).toBe("needs-review");
    expect(changed.metadataSourceHash).not.toBe(changed.source.hash);
  });

  it("enriches only pending or needs-review metadata and accepts only exact current output", async () => {
    const { sourceRoot, websiteRoot } = await makeRoots();
    const calls: string[] = [];
    const enrichLesson = vi.fn(async ({ lesson }: any) => {
      calls.push(lesson.id);
      expect(Object.keys(lesson).sort()).toEqual([
        "bodyMarkdown",
        "id",
        "schemaVersion",
        "sourceHash",
        "summaryZh",
        "titleZh",
        "tldrZh",
      ]);
      expect(JSON.stringify(lesson)).not.toContain(sourceRoot);
      return {
        ok: true as const,
        value: {
          id: lesson.id,
          sourceHash: lesson.sourceHash,
          titleEn: lesson.id === "lesson-0001" ? "Prompt Engineering Foundations" : "Attention and Transformer Architecture",
          summaryEn:
            lesson.id === "lesson-0001"
              ? "This lesson introduces practical prompt structure, clear instructions, useful context, constraints, and examples while explaining why careful iteration helps improve reliable model responses for everyday tasks without treating prompting as magic or replacing evaluation and domain judgment."
              : "This lesson explains how attention connects tokens through learned relevance scores, why multiple heads capture different relationships, and how Transformer blocks combine attention with residual connections and feed-forward layers to build contextual representations used by modern language models.",
          tags: lesson.id === "lesson-0001" ? ["prompt-engineering"] : ["attention", "transformers"],
        },
      };
    });

    await runSync({ websiteRoot, sourceRoot, enrichMetadata: true, enrichLesson });
    expect(calls).toEqual(["lesson-0001", "lesson-0002"]);
    expect((await readSidecar(websiteRoot, "lesson-0001")).metadataStatus).toBe("current");

    calls.length = 0;
    await runSync({ websiteRoot, sourceRoot, enrichMetadata: true, enrichLesson });
    expect(calls).toEqual([]);
  });

  it("keeps Chinese fallback after a soft enrichment failure", async () => {
    const { sourceRoot, websiteRoot } = await makeRoots();
    const result = await runSync({
      websiteRoot,
      sourceRoot,
      enrichMetadata: true,
      enrichLesson: async () => ({
        ok: false,
        reason: "invalid-output",
        message: "stale or malformed output",
      }),
    });

    expect(result.pending).toBe(2);
    const sidecar = await readSidecar(websiteRoot, "lesson-0001");
    expect(sidecar).toMatchObject({
      titleEn: null,
      summaryEn: null,
      tags: [],
      metadataStatus: "pending",
    });
  });

  it("blocks disappearance and reassignment by default but permits an explicit removal operation", async () => {
    const { sourceRoot, websiteRoot } = await makeRoots();
    await runSync({ websiteRoot, sourceRoot, enrichMetadata: false });
    const log = path.join(sourceRoot, "learning-log.md");
    const lesson = path.join(sourceRoot, "lessons/2026-07-06.md");
    await fs.writeFile(log, (await fs.readFile(log, "utf8")).split("\n").filter((line) => !line.startsWith("| 2 |")).join("\n"));
    await fs.writeFile(lesson, (await fs.readFile(lesson, "utf8")).split("# 📅 2026-07-06（第 2 讲）")[0]!);

    await expect(runSync({ websiteRoot, sourceRoot, enrichMetadata: false })).rejects.toThrow(/unexpected lesson removal/i);
    const result = await runSync({ websiteRoot, sourceRoot, enrichMetadata: false, allowRemovals: true });
    expect(result.lessonIds).toEqual(["lesson-0001"]);

    const indexPath = path.join(websiteRoot, "sync-index.json");
    const index = JSON.parse(await fs.readFile(indexPath, "utf8"));
    index.lessons[0].source.section = 2;
    await fs.writeFile(indexPath, `${JSON.stringify(index, null, 2)}\n`);
    await expect(runSync({ websiteRoot, sourceRoot, enrichMetadata: false })).rejects.toThrow(/reassignment/i);
  });

  it("rejects duplicate slugs and malformed existing YAML or index before replacing output", async () => {
    const { sourceRoot, websiteRoot } = await makeRoots();
    await runSync({ websiteRoot, sourceRoot, enrichMetadata: false });
    const original = await projection(websiteRoot);
    const first = await readSidecar(websiteRoot, "lesson-0001");
    const secondPath = path.join(websiteRoot, "metadata/ai-daily/lesson-0002.yml");
    const second = await readSidecar(websiteRoot, "lesson-0002");
    second.slug = first.slug;
    await fs.writeFile(secondPath, YAML.stringify(second));
    const duplicateProjection = await projection(websiteRoot);
    await expect(runSync({ websiteRoot, sourceRoot, enrichMetadata: false })).rejects.toThrow(/duplicate slug/i);
    expect(await projection(websiteRoot)).toEqual(duplicateProjection);

    await fs.writeFile(secondPath, "id: [malformed\n");
    await expect(runSync({ websiteRoot, sourceRoot, enrichMetadata: false })).rejects.toThrow();
    await fs.writeFile(secondPath, YAML.stringify(second));
    await fs.writeFile(path.join(websiteRoot, "sync-index.json"), "{}\n");
    await expect(runSync({ websiteRoot, sourceRoot, enrichMetadata: false })).rejects.toThrow();
    expect(original).not.toEqual(await projection(websiteRoot));
  });

  it("rolls back metadata, content, and index on validation, enrichment, or source-change failure", async () => {
    const { sourceRoot, websiteRoot } = await makeRoots();
    await runSync({ websiteRoot, sourceRoot, enrichMetadata: false });
    const before = await projection(websiteRoot);

    await expect(
      runSync({
        websiteRoot,
        sourceRoot,
        enrichMetadata: false,
        validateCandidate: async () => {
          throw new Error("candidate rejected");
        },
      }),
    ).rejects.toThrow(/candidate rejected/);
    expect(await projection(websiteRoot)).toEqual(before);

    await expect(
      runSync({
        websiteRoot,
        sourceRoot,
        enrichMetadata: true,
        enrichLesson: async () => {
          throw new Error("unsafe staging path");
        },
      }),
    ).rejects.toThrow(/unsafe staging path/);
    expect(await projection(websiteRoot)).toEqual(before);

    await expect(
      runSync({
        websiteRoot,
        sourceRoot,
        enrichMetadata: false,
        beforeCandidateValidation: async () => {
          await fs.appendFile(path.join(sourceRoot, "protected.txt"), "changed\n");
        },
      }),
    ).rejects.toThrow(/source tree changed/i);
    expect(await projection(websiteRoot)).toEqual(before);
  });

  it("rejects a source mutation at the transaction pre-commit check without installing a stale projection", async () => {
    const { sourceRoot, websiteRoot } = await makeRoots();
    await runSync({ websiteRoot, sourceRoot, enrichMetadata: false });
    const before = await projection(websiteRoot);

    await expect(
      runSync({
        websiteRoot,
        sourceRoot,
        enrichMetadata: false,
        beforePreCommitValidation: async () => {
          await fs.appendFile(path.join(sourceRoot, "protected.txt"), "changed\n");
        },
      }),
    ).rejects.toThrow(/source tree changed/i);
    expect(await projection(websiteRoot)).toEqual(before);
  });

  it("rolls back installed projections when source changes before transaction finalization", async () => {
    const { sourceRoot, websiteRoot } = await makeRoots();
    await runSync({ websiteRoot, sourceRoot, enrichMetadata: false });
    await fs.writeFile(
      path.join(websiteRoot, "src/content/ai-daily/lesson-0001.md"),
      "stale generated content\n",
    );
    const before = await projection(websiteRoot);

    await expect(
      runSync({
        websiteRoot,
        sourceRoot,
        enrichMetadata: false,
        beforeFinalIntegrityValidation: async () => {
          await fs.appendFile(path.join(sourceRoot, "protected.txt"), "changed\n");
        },
      }),
    ).rejects.toThrow(/source tree changed/i);
    expect(await projection(websiteRoot)).toEqual(before);
  });

  it("rejects a generated-content leaf swapped to a symlink before it is opened", async () => {
    const { sourceRoot, websiteRoot } = await makeRoots();
    await runSync({ websiteRoot, sourceRoot, enrichMetadata: false });
    const contentFile = path.join(
      websiteRoot,
      "src/content/ai-daily/lesson-0001.md",
    );
    const externalContent = path.join(path.dirname(websiteRoot), "outside.md");
    await fs.writeFile(externalContent, "outside\n");

    await expect(
      runSync({
        websiteRoot,
        sourceRoot,
        enrichMetadata: false,
        beforeExistingContentRead: async (target) => {
          if (path.basename(target) !== "lesson-0001.md") return;
          await fs.rm(contentFile);
          await fs.symlink(externalContent, contentFile);
        },
      }),
    ).rejects.toThrow(/generated content lesson-0001\.md must be a physical regular file/i);
  });

  it("rejects a generated-content directory swapped after identity validation", async () => {
    const { sourceRoot, websiteRoot } = await makeRoots();
    await runSync({ websiteRoot, sourceRoot, enrichMetadata: false });
    const contentRoot = path.join(websiteRoot, "src/content/ai-daily");
    const externalContent = path.join(path.dirname(websiteRoot), "external-content");
    await fs.mkdir(externalContent);
    await fs.writeFile(path.join(externalContent, "lesson-0001.md"), "outside\n");

    await expect(
      runSync({
        websiteRoot,
        sourceRoot,
        enrichMetadata: false,
        beforeExistingContentRead: async (target) => {
          if (path.basename(target) !== "lesson-0001.md") return;
          await fs.rm(contentRoot, { recursive: true });
          await fs.symlink(externalContent, contentRoot);
        },
      }),
    ).rejects.toThrow(/generated content directory changed while reading/i);
  });

  it("rejects a symlinked existing generated-content directory before reading it", async () => {
    const { sourceRoot, websiteRoot } = await makeRoots();
    await runSync({ websiteRoot, sourceRoot, enrichMetadata: false });
    const contentRoot = path.join(websiteRoot, "src/content/ai-daily");
    const externalContent = path.join(path.dirname(websiteRoot), "external-content");
    await fs.mkdir(externalContent);
    await fs.writeFile(path.join(externalContent, "lesson-0001.md"), "outside\n");
    await fs.rm(contentRoot, { recursive: true });
    await fs.symlink(externalContent, contentRoot);

    await expect(
      runSync({ websiteRoot, sourceRoot, enrichMetadata: false }),
    ).rejects.toThrow(/generated content directory must be a physical directory/i);
  });

  it("rejects a symlinked existing generated-content lesson before reading it", async () => {
    const { sourceRoot, websiteRoot } = await makeRoots();
    await runSync({ websiteRoot, sourceRoot, enrichMetadata: false });
    const contentFile = path.join(
      websiteRoot,
      "src/content/ai-daily/lesson-0001.md",
    );
    const externalContent = path.join(path.dirname(websiteRoot), "outside.md");
    await fs.writeFile(externalContent, "outside\n");
    await fs.rm(contentFile);
    await fs.symlink(externalContent, contentFile);

    await expect(
      runSync({ websiteRoot, sourceRoot, enrichMetadata: false }),
    ).rejects.toThrow(/generated content lesson-0001\.md must be a physical regular file/i);
  });

  it("rejects a non-file existing generated-content lesson before reading it", async () => {
    const { sourceRoot, websiteRoot } = await makeRoots();
    await runSync({ websiteRoot, sourceRoot, enrichMetadata: false });
    const contentFile = path.join(
      websiteRoot,
      "src/content/ai-daily/lesson-0001.md",
    );
    await fs.rm(contentFile);
    await fs.mkdir(contentFile);

    await expect(
      runSync({ websiteRoot, sourceRoot, enrichMetadata: false }),
    ).rejects.toThrow(/generated content lesson-0001\.md must be a physical regular file/i);
  });

  it("rejects physical and symlink root overlap before reading source files", async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "blog-overlap-"));
    temporaryRoots.push(root);
    await fs.mkdir(path.join(root, "source"), { recursive: true });
    await expect(
      runSync({
        websiteRoot: root,
        sourceRoot: path.join(root, "source"),
        enrichMetadata: false,
      }),
    ).rejects.toThrow(/physically separate/i);

    const outside = await fs.mkdtemp(path.join(os.tmpdir(), "blog-overlap-outside-"));
    temporaryRoots.push(outside);
    const sourceRoot = path.join(outside, "source");
    await fs.mkdir(sourceRoot);
    const websiteLink = path.join(outside, "website-link");
    await fs.symlink(sourceRoot, websiteLink);
    await expect(
      runSync({ websiteRoot: websiteLink, sourceRoot, enrichMetadata: false }),
    ).rejects.toThrow(/physically separate/i);
  });

  it("rejects a symlinked lessons directory before reading source content", async () => {
    const { sourceRoot, websiteRoot } = await makeRoots();
    const lessonsRoot = path.join(sourceRoot, "lessons");
    const externalLessons = path.join(path.dirname(sourceRoot), "external-lessons");
    await fs.cp(lessonsRoot, externalLessons, { recursive: true });
    await fs.rm(lessonsRoot, { recursive: true });
    await fs.symlink(externalLessons, lessonsRoot);

    await expect(
      runSync({ websiteRoot, sourceRoot, enrichMetadata: false }),
    ).rejects.toThrow(/lessons directory must be a physical directory/i);
  });

  it("rejects unexpected files in the sidecar directory", async () => {
    const { sourceRoot, websiteRoot } = await makeRoots();
    await fs.mkdir(path.join(websiteRoot, "metadata/ai-daily"), { recursive: true });
    await fs.writeFile(path.join(websiteRoot, "metadata/ai-daily/README.md"), "unexpected\n");
    await expect(runSync({ websiteRoot, sourceRoot, enrichMetadata: false })).rejects.toThrow(/unexpected sidecar/i);
  });
});
