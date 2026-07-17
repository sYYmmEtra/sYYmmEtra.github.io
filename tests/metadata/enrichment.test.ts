import { EventEmitter } from "node:events";
import fs from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { PassThrough, Writable } from "node:stream";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ENRICHMENT_CONSTRAINTS,
  validateEnrichment,
  type EnrichmentOutput,
} from "../../scripts/lib/metadata";
import {
  CLI_HELP,
  buildCodexInvocation,
  enrichMetadata,
  runMetadataEnrichmentCli,
  runCodexProcess,
  type CodexInvocation,
  type CodexProcessResult,
  type CodexRunner,
  type StagedLessonInput,
} from "../../scripts/enrich-metadata";

const HASH_A = `sha256:${"a".repeat(64)}`;
const HASH_B = `sha256:${"b".repeat(64)}`;
const promptPath = path.resolve("scripts/metadata-prompt.md");
const schemaPath = path.resolve("scripts/metadata-output.schema.json");
const websiteRoot = path.resolve(".");
const syncTemporaryRoot = path.join(websiteRoot, ".sync-tmp");
const temporaryRoots: string[] = [];
const websiteArtifacts: string[] = [];
let syncTemporaryRootExisted: boolean | undefined;

function words(count: number): string {
  return Array.from({ length: count }, (_, index) => `word${index + 1}`).join(
    " ",
  );
}

function codePointLength(value: string): number {
  return Array.from(value).length;
}

function summaryWithCodePoints(count: number): string {
  const base = words(30);
  return `${base} ${"😀".repeat(count - codePointLength(base) - 1)}`;
}

function validOutput(
  overrides: Partial<EnrichmentOutput> = {},
): EnrichmentOutput {
  return {
    id: "lesson-0013",
    sourceHash: HASH_A,
    titleEn: "Structured Output and Tool Routing",
    summaryEn: words(30),
    tags: ["structured-output", "tool-routing"],
    ...overrides,
  };
}

const stagedLesson: StagedLessonInput = {
  schemaVersion: 1,
  id: "lesson-0013",
  sourceHash: HASH_A,
  titleZh: "结构化输出",
  summaryZh: "结构化输出保证格式合法，但不保证语义正确。",
  tldrZh: null,
  bodyMarkdown: "# 结构化输出\n\n正文。",
};

async function makeTemporaryRoot(): Promise<string> {
  const root = await fs.mkdtemp(
    path.join(tmpdir(), "personal-blog-enrichment-"),
  );
  temporaryRoots.push(root);
  return root;
}

async function ensureSyncTemporaryRoot(): Promise<void> {
  if (syncTemporaryRootExisted === undefined) {
    try {
      await fs.lstat(syncTemporaryRoot);
      syncTemporaryRootExisted = true;
    } catch (error) {
      if (!(error instanceof Error) || !("code" in error) || error.code !== "ENOENT") {
        throw error;
      }
      syncTemporaryRootExisted = false;
    }
  }
  await fs.mkdir(syncTemporaryRoot, { recursive: true, mode: 0o700 });
}

async function makeWebsiteTransaction(): Promise<string> {
  await ensureSyncTemporaryRoot();
  const transaction = await fs.mkdtemp(
    path.join(syncTemporaryRoot, "metadata-cli-test-"),
  );
  websiteArtifacts.push(transaction);
  return transaction;
}

afterEach(async () => {
  vi.useRealTimers();
  for (const artifact of websiteArtifacts.splice(0).reverse()) {
    await fs.rm(artifact, { recursive: true, force: true });
  }
  if (syncTemporaryRootExisted === false) {
    try {
      await fs.rmdir(syncTemporaryRoot);
    } catch (error) {
      if (
        !(error instanceof Error) ||
        !("code" in error) ||
        !["ENOENT", "ENOTEMPTY"].includes(String(error.code))
      ) {
        throw error;
      }
    }
  }
  for (const root of temporaryRoots.splice(0)) {
    await fs.rm(root, { recursive: true, force: true });
  }
});

describe("enrichment validation", () => {
  it("rejects unknown output keys", () => {
    expect(() =>
      validateEnrichment({ ...validOutput(), unexpected: true }),
    ).toThrow();
  });

  it("rejects stale hashes and wrong IDs when expectations are supplied", () => {
    expect(() => validateEnrichment(validOutput(), HASH_B)).toThrow(
      /stale metadata output/i,
    );
    expect(() =>
      validateEnrichment(validOutput(), {
        expectedSourceHash: HASH_A,
        expectedId: "lesson-0012",
      }),
    ).toThrow(/metadata output ID/i);
  });

  it("rejects IDs with more than four digits", () => {
    expect(() =>
      validateEnrichment(validOutput({ id: "lesson-10000" })),
    ).toThrow();
  });

  it.each([
    [29, false],
    [30, true],
    [60, true],
    [61, false],
  ])("enforces the 30-60 English-word summary boundary at %i", (count, valid) => {
    expect(() =>
      validateEnrichment(validOutput({ summaryEn: words(count) })),
    )[valid ? "not" : "to"].toThrow();
  });

  it.each([
    [["Uppercase"]],
    [["two words"]],
    [["duplicate", "duplicate"]],
    [[]],
    [["one", "two", "three", "four", "five", "six"]],
  ])("rejects invalid or duplicate tags: %j", (tags) => {
    expect(() => validateEnrichment(validOutput({ tags }))).toThrow();
  });

  it("enforces title and summary character limits", () => {
    expect(() => validateEnrichment(validOutput({ titleEn: "abc" }))).toThrow();
    expect(() =>
      validateEnrichment(validOutput({ titleEn: "a".repeat(121) })),
    ).toThrow();
    expect(() =>
      validateEnrichment(validOutput({ summaryEn: `${words(30)} ${"x".repeat(500)}` })),
    ).toThrow();
  });

  it("counts English words rather than numeric tokens", () => {
    expect(() =>
      validateEnrichment(
        validOutput({
          summaryEn: Array.from(
            { length: 30 },
            (_, index) => String(10_000 + index),
          ).join(" "),
        }),
      ),
    ).toThrow(/30-60 words/i);
  });

  it("trims English fields before validation and rejects blank or trimmed-short titles", () => {
    const summary = words(30);
    const output = validateEnrichment(
      validOutput({
        titleEn: "  Canonical English Title  ",
        summaryEn: `  ${summary}  `,
      }),
    );

    expect(output.titleEn).toBe("Canonical English Title");
    expect(output.summaryEn).toBe(summary);
    expect(() =>
      validateEnrichment(validOutput({ titleEn: "    " })),
    ).toThrow();
    expect(() =>
      validateEnrichment(validOutput({ titleEn: "  abc  " })),
    ).toThrow();
  });

  it("uses Unicode code-point counts for title and summary length parity", () => {
    expect(() =>
      validateEnrichment(
        validOutput({ titleEn: `${"A".repeat(119)}😀` }),
      ),
    ).not.toThrow();
    expect(() =>
      validateEnrichment(
        validOutput({ titleEn: `${"A".repeat(120)}😀` }),
      ),
    ).toThrow();
    expect(() =>
      validateEnrichment(validOutput({ titleEn: "😀😀😀" })),
    ).toThrow();
    expect(() =>
      validateEnrichment(
        validOutput({ summaryEn: summaryWithCodePoints(600) }),
      ),
    ).not.toThrow();
    expect(() =>
      validateEnrichment(
        validOutput({ summaryEn: summaryWithCodePoints(601) }),
      ),
    ).toThrow();
  });
});

describe("checked-in output schema", () => {
  it("has the exact strict draft-2020-12 shape and Zod-parity constraints", async () => {
    const schema = JSON.parse(await fs.readFile(schemaPath, "utf8"));

    expect(ENRICHMENT_CONSTRAINTS.idPattern).toBe("^lesson-[0-9]{4}$");

    expect(schema).toMatchObject({
      $schema: "https://json-schema.org/draft/2020-12/schema",
      type: "object",
      additionalProperties: false,
      required: ["id", "sourceHash", "titleEn", "summaryEn", "tags"],
      properties: {
        id: {
          type: "string",
          pattern: ENRICHMENT_CONSTRAINTS.idPattern,
        },
        sourceHash: {
          type: "string",
          pattern: ENRICHMENT_CONSTRAINTS.sourceHashPattern,
        },
        summaryEn: {
          type: "string",
          minLength: ENRICHMENT_CONSTRAINTS.summaryMinLength,
          maxLength: ENRICHMENT_CONSTRAINTS.summaryMaxLength,
        },
        tags: {
          type: "array",
          minItems: ENRICHMENT_CONSTRAINTS.tagsMinItems,
          maxItems: ENRICHMENT_CONSTRAINTS.tagsMaxItems,
          uniqueItems: true,
          items: {
            type: "string",
            pattern: ENRICHMENT_CONSTRAINTS.tagPattern,
          },
        },
      },
    });
    expect(schema.properties.titleEn).toEqual({
      type: "string",
      minLength: ENRICHMENT_CONSTRAINTS.titleMinLength,
      maxLength: ENRICHMENT_CONSTRAINTS.titleMaxLength,
    });
    expect(Object.keys(schema.properties)).toEqual([
      "id",
      "sourceHash",
      "titleEn",
      "summaryEn",
      "tags",
    ]);
  });

  it("keeps the approved fixed read-only prompt", async () => {
    await expect(fs.readFile(promptPath, "utf8")).resolves.toBe(
      "Read `lesson.json` in the current directory. Return only the JSON object required by the supplied schema. Echo `id` and `sourceHash` exactly. Translate the Chinese title faithfully into concise technical English. Write a 30–60 word English summary based only on the supplied title, summary, TL;DR, and body. Do not invent claims, verification status, dates, numbers, sources, or links. Produce 1–5 lowercase kebab-case technical tags. Do not edit files.\n",
    );
  });
});

describe("CLI contract", () => {
  it("documents explicit minimized input and staging arguments without AI_DAILY_SOURCE", () => {
    expect(CLI_HELP).toContain("--input <lesson.json>");
    expect(CLI_HELP).toContain("--staging-parent <directory>");
    expect(CLI_HELP).toContain("--timeout-ms <milliseconds>");
    expect(CLI_HELP).toContain("validated enrichment JSON to stdout");
    expect(CLI_HELP).toContain("inside --staging-parent");
    expect(CLI_HELP).toContain(".sync-tmp");
    expect(CLI_HELP).not.toContain("AI_DAILY_SOURCE");
  });

  it("registers the metadata enrichment command only after the entry point exists", async () => {
    const packageJson = JSON.parse(await fs.readFile("package.json", "utf8"));

    expect(packageJson.scripts["metadata:enrich"]).toBe(
      "tsx scripts/enrich-metadata.ts",
    );
  });
});

describe("CLI staging-parent isolation", () => {
  const quietCli = {
    stdout: vi.fn(),
    stderr: vi.fn(),
  };

  it("rejects a staging parent inside AI_DAILY_SOURCE before creating or running", async () => {
    const sourceRoot = await makeTemporaryRoot();
    const stagingParent = path.join(sourceRoot, "transaction");
    await fs.mkdir(stagingParent);
    const before = await fs.readdir(stagingParent);
    const runner = vi.fn() as unknown as CodexRunner;

    await expect(
      runMetadataEnrichmentCli(
        [
          "--input",
          path.join(stagingParent, "lesson.json"),
          "--staging-parent",
          stagingParent,
        ],
        {
          ...quietCli,
          runner,
          parentEnv: { AI_DAILY_SOURCE: sourceRoot },
        },
      ),
    ).rejects.toThrow(/AI Daily source/i);
    expect(runner).not.toHaveBeenCalled();
    expect(await fs.readdir(stagingParent)).toEqual(before);
  });

  it("rejects an arbitrary outside staging directory without AI_DAILY_SOURCE", async () => {
    const stagingParent = await makeTemporaryRoot();
    const before = await fs.readdir(stagingParent);
    const runner = vi.fn() as unknown as CodexRunner;

    await expect(
      runMetadataEnrichmentCli(
        [
          "--input",
          path.join(stagingParent, "lesson.json"),
          "--staging-parent",
          stagingParent,
        ],
        { ...quietCli, runner, parentEnv: {} },
      ),
    ).rejects.toThrow(/\.sync-tmp/i);
    expect(runner).not.toHaveBeenCalled();
    expect(await fs.readdir(stagingParent)).toEqual(before);
  });

  it("rejects the website root and .sync-tmp itself as staging parents", async () => {
    await ensureSyncTemporaryRoot();
    const runner = vi.fn() as unknown as CodexRunner;

    for (const stagingParent of [websiteRoot, syncTemporaryRoot]) {
      const before = await fs.readdir(stagingParent);
      await expect(
        runMetadataEnrichmentCli(
          [
            "--input",
            path.join(stagingParent, "lesson.json"),
            "--staging-parent",
            stagingParent,
          ],
          { ...quietCli, runner, parentEnv: {} },
        ),
      ).rejects.toThrow(/\.sync-tmp/i);
      expect(await fs.readdir(stagingParent)).toEqual(before);
    }
    expect(runner).not.toHaveBeenCalled();
  });

  it("rejects a symlinked escape below website .sync-tmp before creating or running", async () => {
    await ensureSyncTemporaryRoot();
    const outside = await makeTemporaryRoot();
    const linkedParent = path.join(syncTemporaryRoot, "linked-transaction");
    await fs.symlink(outside, linkedParent);
    websiteArtifacts.push(linkedParent);
    const before = await fs.readdir(outside);
    const runner = vi.fn() as unknown as CodexRunner;

    await expect(
      runMetadataEnrichmentCli(
        [
          "--input",
          path.join(linkedParent, "lesson.json"),
          "--staging-parent",
          linkedParent,
        ],
        { ...quietCli, runner, parentEnv: {} },
      ),
    ).rejects.toThrow(/symlink|physical|outside website root/i);
    expect(runner).not.toHaveBeenCalled();
    expect(await fs.readdir(outside)).toEqual(before);
  });
});

describe("CLI input isolation", () => {
  function unusedRunner() {
    return vi.fn(async () => ({
      exitCode: 0,
      signal: null,
      stdout: "",
      stderr: "",
      timedOut: false,
    })) as unknown as CodexRunner;
  }

  async function expectRejectedBeforeStaging(
    stagingParent: string,
    inputFile: string,
    expected: RegExp,
    parentEnv: NodeJS.ProcessEnv = {},
  ): Promise<void> {
    const before = await fs.readdir(stagingParent);
    const runner = unusedRunner();

    await expect(
      runMetadataEnrichmentCli(
        [
          "--input",
          inputFile,
          "--staging-parent",
          stagingParent,
        ],
        {
          runner,
          parentEnv,
          stdout: vi.fn(),
          stderr: vi.fn(),
        },
      ),
    ).rejects.toThrow(expected);
    expect(runner).not.toHaveBeenCalled();
    expect(await fs.readdir(stagingParent)).toEqual(before);
  }

  it("rejects an input file outside the supplied transaction", async () => {
    const stagingParent = await makeWebsiteTransaction();
    const outsideRoot = await makeTemporaryRoot();
    const inputFile = path.join(outsideRoot, "lesson.json");
    await fs.writeFile(inputFile, JSON.stringify(stagedLesson));

    await expectRejectedBeforeStaging(
      stagingParent,
      inputFile,
      /input.*inside.*staging parent|outside.*staging parent/i,
    );
  });

  it("rejects an input file from AI_DAILY_SOURCE", async () => {
    const stagingParent = await makeWebsiteTransaction();
    const sourceRoot = await makeTemporaryRoot();
    const inputFile = path.join(sourceRoot, "lesson.json");
    await fs.writeFile(inputFile, JSON.stringify(stagedLesson));

    await expectRejectedBeforeStaging(
      stagingParent,
      inputFile,
      /input.*AI Daily source|input.*inside.*staging parent/i,
      { AI_DAILY_SOURCE: sourceRoot },
    );
  });

  it("rejects a symlink input without following it", async () => {
    const stagingParent = await makeWebsiteTransaction();
    const outsideRoot = await makeTemporaryRoot();
    const outsideFile = path.join(outsideRoot, "lesson.json");
    const inputFile = path.join(stagingParent, "lesson.json");
    await fs.writeFile(outsideFile, JSON.stringify(stagedLesson));
    await fs.symlink(outsideFile, inputFile);

    await expectRejectedBeforeStaging(
      stagingParent,
      inputFile,
      /input.*symlink|regular non-symlink/i,
    );
  });

  it("rejects a directory input", async () => {
    const stagingParent = await makeWebsiteTransaction();
    const inputFile = path.join(stagingParent, "lesson.json");
    await fs.mkdir(inputFile);

    await expectRejectedBeforeStaging(
      stagingParent,
      inputFile,
      /input.*regular non-symlink file/i,
    );
  });

  it("rejects an input larger than 2 MiB", async () => {
    const stagingParent = await makeWebsiteTransaction();
    const inputFile = path.join(stagingParent, "lesson.json");
    await fs.writeFile(inputFile, Buffer.alloc(2 * 1024 * 1024 + 1, "x"));

    await expectRejectedBeforeStaging(
      stagingParent,
      inputFile,
      /input.*exceeds.*2097152 bytes/i,
    );
  });

  it("rejects a hardlinked input file before staging or running", async () => {
    const stagingParent = await makeWebsiteTransaction();
    const outsideRoot = await makeTemporaryRoot();
    const outsideFile = path.join(outsideRoot, "lesson.json");
    const inputFile = path.join(stagingParent, "lesson.json");
    await fs.writeFile(outsideFile, JSON.stringify(stagedLesson));
    await fs.link(outsideFile, inputFile);

    await expectRejectedBeforeStaging(
      stagingParent,
      inputFile,
      /input.*hardlink|link count|single-link/i,
    );
  });
});

describe("source-path canonicalization policy", () => {
  async function expectSourceTextAllowed(
    parentEnv: NodeJS.ProcessEnv,
    text: string,
  ): Promise<void> {
    const root = await makeTemporaryRoot();
    const stagingParent = path.join(root, "transaction-staging");
    await fs.mkdir(stagingParent, { mode: 0o700 });
    const runner: CodexRunner = async (invocation) => {
      await fs.writeFile(invocation.outputFile, JSON.stringify(validOutput()));
      return {
        exitCode: 0,
        signal: null,
        stdout: "",
        stderr: "",
        timedOut: false,
        reaped: true,
        mayBeAlive: false,
      };
    };

    const result = await enrichMetadata({
      stagingParent,
      assertSafeStagingPath() {},
      lesson: { ...stagedLesson, bodyMarkdown: text },
      readCurrentSourceHash: async () => HASH_A,
      runner,
      schemaPath,
      prompt: "fixed prompt",
      parentEnv,
    });

    expect(result).toEqual({ ok: true, value: validOutput() });
    expect(await fs.readdir(stagingParent)).toEqual([]);
  }

  async function expectLessonPathRejected(
    parentEnv: NodeJS.ProcessEnv,
    exposedPath: string,
  ): Promise<void> {
    const root = await makeTemporaryRoot();
    const stagingParent = path.join(root, "transaction-staging");
    await fs.mkdir(stagingParent, { mode: 0o700 });
    const runner = vi.fn(async () => ({
      exitCode: 0,
      signal: null,
      stdout: "",
      stderr: "",
      timedOut: false,
    })) as unknown as CodexRunner;

    await expect(
      enrichMetadata({
        stagingParent,
        assertSafeStagingPath() {},
        lesson: {
          ...stagedLesson,
          bodyMarkdown: `Do not expose ${exposedPath}/lessons/2026-07-17.md`,
        },
        readCurrentSourceHash: async () => HASH_A,
        runner,
        schemaPath,
        prompt: await fs.readFile(promptPath, "utf8"),
        parentEnv,
      }),
    ).rejects.toThrow(/must not expose the AI Daily source path/i);
    expect(runner).not.toHaveBeenCalled();
    expect(await fs.readdir(stagingParent)).toEqual([]);
  }

  it("handles lexical dot-dot source spelling across env, invocation, and staged input", async () => {
    const root = await makeTemporaryRoot();
    const sourceRoot = path.join(root, "source-root");
    const nested = path.join(sourceRoot, "nested");
    await fs.mkdir(nested, { recursive: true });
    const canonicalSource = await fs.realpath(sourceRoot);
    const lexicalSource = `${nested}${path.sep}..`;
    const canonicalSecret = path.join(canonicalSource, "lessons", "secret.md");
    const parentEnv = {
      AI_DAILY_SOURCE: lexicalSource,
      CANONICAL_SECRET: canonicalSecret,
      UNRELATED_VALUE: "kept",
    };

    const invocation = buildCodexInvocation({
      stagingDir: path.join(root, "website-staging"),
      outputFile: path.join(root, "website-staging", "output.json"),
      schemaPath,
      prompt: "fixed prompt",
      parentEnv,
    });

    expect(invocation.env).not.toHaveProperty("AI_DAILY_SOURCE");
    expect(invocation.env).not.toHaveProperty("CANONICAL_SECRET");
    expect(invocation.env.UNRELATED_VALUE).toBe("kept");
    expect(() =>
      buildCodexInvocation({
        stagingDir: path.join(root, "website-staging"),
        outputFile: path.join(root, "website-staging", "output.json"),
        schemaPath: canonicalSecret,
        prompt: "fixed prompt",
        parentEnv,
      }),
    ).toThrow(/must not expose the AI Daily source path/i);
    await expectLessonPathRejected(parentEnv, canonicalSource);
  });

  it("handles a symlink AI_DAILY_SOURCE alias when values expose its physical realpath", async () => {
    const root = await makeTemporaryRoot();
    const physicalSource = path.join(root, "physical-source");
    const sourceAlias = path.join(root, "source-alias");
    await fs.mkdir(physicalSource);
    await fs.symlink(physicalSource, sourceAlias);
    const physicalSecret = path.join(physicalSource, "private", "secret.md");
    const parentEnv = {
      AI_DAILY_SOURCE: sourceAlias,
      PHYSICAL_SECRET: physicalSecret,
      UNRELATED_VALUE: "kept",
    };

    const invocation = buildCodexInvocation({
      stagingDir: path.join(root, "website-staging"),
      outputFile: path.join(root, "website-staging", "output.json"),
      schemaPath,
      prompt: "fixed prompt",
      parentEnv,
    });

    expect(invocation.env).not.toHaveProperty("AI_DAILY_SOURCE");
    expect(invocation.env).not.toHaveProperty("PHYSICAL_SECRET");
    expect(invocation.env.UNRELATED_VALUE).toBe("kept");
    expect(() =>
      buildCodexInvocation({
        stagingDir: path.join(root, "website-staging"),
        outputFile: path.join(root, "website-staging", "output.json"),
        schemaPath,
        prompt: `Read only ${physicalSecret}`,
        parentEnv,
      }),
    ).toThrow(/must not expose the AI Daily source path/i);
    await expectLessonPathRejected(parentEnv, physicalSource);
  });

  it("physically removes an unrelated-text symlink alias that resolves inside the source root", async () => {
    const root = await makeTemporaryRoot();
    const sourceRoot = path.join(root, "source-root");
    const sourceChild = path.join(sourceRoot, "private-tools");
    const sourceChildAlias = path.join(root, "tool-alias");
    const safeBin = path.join(root, "safe-bin");
    await fs.mkdir(sourceChild, { recursive: true });
    await fs.mkdir(safeBin);
    await fs.symlink(sourceChild, sourceChildAlias);
    expect(sourceChildAlias).not.toContain(sourceRoot);

    const invocation = buildCodexInvocation({
      stagingDir: path.join(root, "website-staging"),
      outputFile: path.join(root, "website-staging", "output.json"),
      schemaPath,
      prompt: "fixed prompt",
      parentEnv: {
        AI_DAILY_SOURCE: sourceRoot,
        ALIASED_TOOL_HOME: sourceChildAlias,
        TOOL_PATH: `${safeBin}${path.delimiter}${sourceChildAlias}`,
        UNRELATED_VALUE: "kept",
      },
    });

    expect(invocation.env).not.toHaveProperty("ALIASED_TOOL_HOME");
    expect(invocation.env.TOOL_PATH).toBe(safeBin);
    expect(invocation.env.UNRELATED_VALUE).toBe("kept");
  });

  it("strips prose punctuation around source aliases in env, prompt, title, and body", async () => {
    const root = await makeTemporaryRoot();
    const physicalSource = path.join(root, "physical-source");
    const sourceAlias = path.join(root, "source-alias");
    const stagingParent = path.join(root, "transaction-staging");
    await fs.mkdir(physicalSource);
    await fs.mkdir(stagingParent);
    await fs.symlink(physicalSource, sourceAlias);
    const parentEnv = {
      AI_DAILY_SOURCE: physicalSource,
      SOURCE_DOT: `See ${sourceAlias}.`,
      SOURCE_COMMA: `See ${sourceAlias}, next`,
      SOURCE_PAREN: `See (${sourceAlias})`,
      SOURCE_CHINESE_PERIOD: `参见${sourceAlias}。`,
      SOURCE_CHINESE_COMMA: `参见${sourceAlias}，继续`,
      SOURCE_CHINESE_PAREN: `参见（${sourceAlias}）`,
      UNRELATED_VALUE: "kept",
    };

    const invocation = buildCodexInvocation({
      stagingDir: path.join(root, "website-staging"),
      outputFile: path.join(root, "website-staging", "output.json"),
      schemaPath,
      prompt: "fixed prompt",
      parentEnv,
    });

    for (const key of [
      "SOURCE_DOT",
      "SOURCE_COMMA",
      "SOURCE_PAREN",
      "SOURCE_CHINESE_PERIOD",
      "SOURCE_CHINESE_COMMA",
      "SOURCE_CHINESE_PAREN",
    ]) {
      expect(invocation.env).not.toHaveProperty(key);
    }
    expect(invocation.env.UNRELATED_VALUE).toBe("kept");

    for (const punctuation of [".", ",", ")", "。", "，", "）", "】", "》"]) {
      expect(() =>
        buildCodexInvocation({
          stagingDir: path.join(root, "website-staging"),
          outputFile: path.join(root, "website-staging", "output.json"),
          schemaPath,
          prompt: `Read ${sourceAlias}${punctuation}`,
          parentEnv,
        }),
      ).toThrow(/must not expose the AI Daily source path/i);
    }

    const runner = vi.fn(async () => ({
      exitCode: 0,
      signal: null,
      stdout: "",
      stderr: "",
      timedOut: false,
    })) as unknown as CodexRunner;
    for (const lesson of [
      { ...stagedLesson, titleZh: `标题 ${sourceAlias}，` },
      { ...stagedLesson, bodyMarkdown: `正文 ${sourceAlias}）` },
    ]) {
      await expect(
        enrichMetadata({
          stagingParent,
          assertSafeStagingPath() {},
          lesson,
          readCurrentSourceHash: async () => HASH_A,
          runner,
          schemaPath,
          prompt: await fs.readFile(promptPath, "utf8"),
          parentEnv,
        }),
      ).rejects.toThrow(/must not expose the AI Daily source path/i);
      expect(await fs.readdir(stagingParent)).toEqual([]);
    }
    expect(runner).not.toHaveBeenCalled();
  });

  it.each(["”", "’", "…", "—", "→", "#"])(
    "uses Unicode punctuation or symbol delimiter %s across env, prompt, and staged content",
    async (delimiter) => {
      const root = await makeTemporaryRoot();
      const physicalSource = path.join(root, "physical-source");
      const sourceAlias = path.join(root, "source-alias");
      const stagingParent = path.join(root, "transaction-staging");
      await fs.mkdir(physicalSource);
      await fs.mkdir(stagingParent);
      await fs.symlink(physicalSource, sourceAlias);
      const parentEnv = {
        AI_DAILY_SOURCE: physicalSource,
        PUNCTUATED_ALIAS: `See ${sourceAlias}${delimiter}continued`,
        UNRELATED_VALUE: "kept",
      };

      const invocation = buildCodexInvocation({
        stagingDir: path.join(root, "website-staging"),
        outputFile: path.join(root, "website-staging", "output.json"),
        schemaPath,
        prompt: "fixed prompt",
        parentEnv,
      });
      expect(invocation.env).not.toHaveProperty("PUNCTUATED_ALIAS");
      expect(invocation.env.UNRELATED_VALUE).toBe("kept");

      expect(() =>
        buildCodexInvocation({
          stagingDir: path.join(root, "website-staging"),
          outputFile: path.join(root, "website-staging", "output.json"),
          schemaPath,
          prompt: `Read ${sourceAlias}${delimiter}continued`,
          parentEnv,
        }),
      ).toThrow(/must not expose the AI Daily source path/i);

      const runner = vi.fn(async () => ({
        exitCode: 0,
        signal: null,
        stdout: "",
        stderr: "",
        timedOut: false,
      })) as unknown as CodexRunner;
      await expect(
        enrichMetadata({
          stagingParent,
          assertSafeStagingPath() {},
          lesson: {
            ...stagedLesson,
            bodyMarkdown: `Body ${sourceAlias}${delimiter}continued`,
          },
          readCurrentSourceHash: async () => HASH_A,
          runner,
          schemaPath,
          prompt: await fs.readFile(promptPath, "utf8"),
          parentEnv,
        }),
      ).rejects.toThrow(/must not expose the AI Daily source path/i);
      expect(runner).not.toHaveBeenCalled();
      expect(await fs.readdir(stagingParent)).toEqual([]);
    },
  );

  it("does not treat a relative source name as a substring of a larger word", async () => {
    const parentEnv = {
      AI_DAILY_SOURCE: "source",
      RELATED_PROSE: "sources remain available",
      UNRELATED_VALUE: "kept",
    };

    const invocation = buildCodexInvocation({
      stagingDir: "/private/site/.sync-tmp/enrichment-1",
      outputFile: "/private/site/.sync-tmp/enrichment-1/output.json",
      schemaPath,
      prompt: "sources remain available",
      parentEnv,
    });

    expect(invocation.env.RELATED_PROSE).toBe("sources remain available");
    expect(invocation.env.UNRELATED_VALUE).toBe("kept");
    await expectSourceTextAllowed(parentEnv, "sources remain available");
  });

  it("does not treat an absolute sibling prefix as the configured source", async () => {
    const root = await makeTemporaryRoot();
    const sourceRoot = path.join(root, "ai-daily");
    const sibling = path.join(root, "ai-daily-backup");
    await fs.mkdir(sourceRoot);
    await fs.mkdir(sibling);
    const parentEnv = {
      AI_DAILY_SOURCE: sourceRoot,
      BACKUP_PATH: sibling,
      UNRELATED_VALUE: "kept",
    };

    const invocation = buildCodexInvocation({
      stagingDir: "/private/site/.sync-tmp/enrichment-1",
      outputFile: "/private/site/.sync-tmp/enrichment-1/output.json",
      schemaPath,
      prompt: `Inspect ${sibling}`,
      parentEnv,
    });

    expect(invocation.env.BACKUP_PATH).toBe(sibling);
    expect(invocation.env.UNRELATED_VALUE).toBe("kept");
    await expectSourceTextAllowed(parentEnv, `Inspect ${sibling}`);
  });
});

describe("Codex invocation and process runner", () => {
  it("builds the exact hardened argv, stdin prompt, and sanitized environment", () => {
    const sourcePath = "/private/source/ai-daily";
    const invocation = buildCodexInvocation({
      stagingDir: "/private/site/.sync-tmp/enrichment-1",
      outputFile: "/private/site/.sync-tmp/enrichment-1/output.json",
      schemaPath: "/private/site/scripts/metadata-output.schema.json",
      prompt: "fixed prompt",
      parentEnv: {
        AI_DAILY_SOURCE: sourcePath,
        PWD: sourcePath,
        PRESERVED: "yes",
      },
    });

    expect(invocation).toMatchObject({
      command: "codex",
      args: [
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--strict-config",
        "--sandbox",
        "read-only",
        "-c",
        "shell_environment_policy.inherit=none",
        "--output-schema",
        "/private/site/scripts/metadata-output.schema.json",
        "-C",
        "/private/site/.sync-tmp/enrichment-1",
        "-o",
        "/private/site/.sync-tmp/enrichment-1/output.json",
        "--color",
        "never",
        "-",
      ],
      cwd: "/private/site/.sync-tmp/enrichment-1",
      stdin: "fixed prompt",
      shell: false,
    });
    expect(invocation.env).toMatchObject({
      PWD: "/private/site/.sync-tmp/enrichment-1",
      PRESERVED: "yes",
    });
    expect(invocation.env).not.toHaveProperty("AI_DAILY_SOURCE");
    expect(JSON.stringify(invocation)).not.toContain(sourcePath);
  });

  it("spawns without a shell and writes the prompt to stdin", async () => {
    let stdin = "";
    let captured:
      | { command: string; args: readonly string[]; options: object }
      | undefined;
    const child = new EventEmitter() as EventEmitter & {
      stdin: Writable;
      stdout: PassThrough;
      stderr: PassThrough;
      kill: ReturnType<typeof vi.fn>;
    };
    child.stdin = new Writable({
      write(chunk, _encoding, callback) {
        stdin += chunk.toString();
        callback();
      },
    });
    child.stdout = new PassThrough();
    child.stderr = new PassThrough();
    child.kill = vi.fn(() => true);
    const spawnImpl = ((
      command: string,
      args: readonly string[],
      options: object,
    ) => {
      captured = { command, args, options };
      return child;
    }) as never;
    const invocation: CodexInvocation = {
      command: "codex",
      args: ["exec", "-"],
      cwd: "/tmp/staging",
      env: { PWD: "/tmp/staging" },
      stdin: "prompt via stdin",
      shell: false,
      outputFile: "/tmp/staging/output.json",
    };

    const resultPromise = runCodexProcess(invocation, {
      timeoutMs: 1_000,
      spawnImpl,
    });
    child.stderr.write("warning only");
    child.emit("close", 0, null);

    await expect(resultPromise).resolves.toMatchObject({
      exitCode: 0,
      stderr: "warning only",
      timedOut: false,
    });
    expect(captured).toMatchObject({
      command: "codex",
      args: ["exec", "-"],
      options: {
        cwd: "/tmp/staging",
        env: { PWD: "/tmp/staging" },
        shell: false,
      },
    });
    expect(stdin).toBe("prompt via stdin");
  });

  it("preserves child spawn errors that occur before termination begins", async () => {
    const child = new EventEmitter() as EventEmitter & {
      stdin: PassThrough;
      stdout: PassThrough;
      stderr: PassThrough;
      kill: ReturnType<typeof vi.fn>;
    };
    child.stdin = new PassThrough();
    child.stdout = new PassThrough();
    child.stderr = new PassThrough();
    child.kill = vi.fn(() => true);
    const spawnError = Object.assign(new Error("spawn codex ENOENT"), {
      code: "ENOENT",
    });

    const resultPromise = runCodexProcess(
      {
        command: "codex",
        args: ["exec", "-"],
        cwd: "/tmp/staging",
        env: {},
        stdin: "prompt",
        shell: false,
        outputFile: "/tmp/staging/output.json",
      },
      { timeoutMs: 1_000, spawnImpl: (() => child) as never },
    );
    child.emit("error", spawnError);

    await expect(resultPromise).resolves.toMatchObject({
      exitCode: null,
      signal: null,
      timedOut: false,
      reaped: true,
      mayBeAlive: false,
      error: spawnError,
    });
    expect(child.kill).not.toHaveBeenCalled();
  });

  it("caps oversized stdout and stderr with deterministic truncation markers", async () => {
    const child = new EventEmitter() as EventEmitter & {
      stdin: PassThrough;
      stdout: PassThrough;
      stderr: PassThrough;
      kill: ReturnType<typeof vi.fn>;
    };
    child.stdin = new PassThrough();
    child.stdout = new PassThrough();
    child.stderr = new PassThrough();
    child.kill = vi.fn(() => true);

    const resultPromise = runCodexProcess(
      {
        command: "codex",
        args: ["exec", "-"],
        cwd: "/tmp/staging",
        env: {},
        stdin: "prompt",
        shell: false,
        outputFile: "/tmp/staging/output.json",
      },
      { timeoutMs: 1_000, spawnImpl: (() => child) as never },
    );
    child.stdout.write("x".repeat(70_000));
    child.stderr.write("y".repeat(70_000));
    child.emit("close", 0, null);

    const result = await resultPromise;
    expect(Buffer.byteLength(result.stdout)).toBeLessThanOrEqual(64 * 1024);
    expect(Buffer.byteLength(result.stderr)).toBeLessThanOrEqual(64 * 1024);
    expect(result.stdout).toMatch(/\n\.\.\.\[truncated\]\n$/);
    expect(result.stderr).toMatch(/\n\.\.\.\[truncated\]\n$/);
  });

  it("terminates with SIGTERM and reports a safely reaped timeout when close follows", async () => {
    vi.useFakeTimers();
    const child = new EventEmitter() as EventEmitter & {
      stdin: Writable;
      stdout: PassThrough;
      stderr: PassThrough;
      kill: ReturnType<typeof vi.fn>;
    };
    child.stdin = new PassThrough();
    child.stdout = new PassThrough();
    child.stderr = new PassThrough();
    child.kill = vi.fn((signal) => {
      queueMicrotask(() => child.emit("close", null, signal));
      return true;
    });

    const resultPromise = runCodexProcess(
      {
        command: "codex",
        args: ["exec", "-"],
        cwd: "/tmp/staging",
        env: {},
        stdin: "prompt",
        shell: false,
        outputFile: "/tmp/staging/output.json",
      },
      {
        timeoutMs: 25,
        terminateGraceMs: 10,
        reapTimeoutMs: 10,
        spawnImpl: (() => child) as never,
      },
    );
    await vi.advanceTimersByTimeAsync(25);

    await expect(resultPromise).resolves.toMatchObject({
      timedOut: true,
      reaped: true,
      mayBeAlive: false,
    });
    expect(child.kill).toHaveBeenCalledWith("SIGTERM");
  });

  it("treats asynchronous child errors during termination as diagnostic until the bounded reap deadline", async () => {
    vi.useFakeTimers();
    const child = new EventEmitter() as EventEmitter & {
      stdin: PassThrough;
      stdout: PassThrough;
      stderr: PassThrough;
      exitCode: number | null;
      signalCode: NodeJS.Signals | null;
      kill: ReturnType<typeof vi.fn>;
    };
    child.stdin = new PassThrough();
    child.stdout = new PassThrough();
    child.stderr = new PassThrough();
    child.exitCode = null;
    child.signalCode = null;
    child.kill = vi.fn((signal) => {
      if (signal === "SIGTERM") {
        queueMicrotask(() => {
          child.emit("error", new Error("kill failed asynchronously"));
        });
      }
      return true;
    });

    const resultPromise = runCodexProcess(
      {
        command: "codex",
        args: ["exec", "-"],
        cwd: "/tmp/staging",
        env: {},
        stdin: "prompt",
        shell: false,
        outputFile: "/tmp/staging/output.json",
      },
      {
        timeoutMs: 25,
        terminateGraceMs: 10,
        reapTimeoutMs: 10,
        spawnImpl: (() => child) as never,
      },
    );
    await vi.advanceTimersByTimeAsync(45);

    await expect(resultPromise).resolves.toMatchObject({
      timedOut: true,
      reaped: false,
      mayBeAlive: true,
      error: expect.objectContaining({
        message: "kill failed asynchronously",
      }),
    });
    expect(child.kill).toHaveBeenNthCalledWith(1, "SIGTERM");
    expect(child.kill).toHaveBeenNthCalledWith(2, "SIGKILL");
  });

  it.each([
    ["returns false", () => false],
    ["throws", () => { throw new Error("kill failed"); }],
  ])("settles after bounded TERM/KILL/reap deadlines when kill %s", async (
    _label,
    killImplementation,
  ) => {
    vi.useFakeTimers();
    const child = new EventEmitter() as EventEmitter & {
      stdin: PassThrough;
      stdout: PassThrough;
      stderr: PassThrough;
      exitCode: number | null;
      signalCode: NodeJS.Signals | null;
      kill: ReturnType<typeof vi.fn>;
    };
    child.stdin = new PassThrough();
    child.stdout = new PassThrough();
    child.stderr = new PassThrough();
    child.exitCode = null;
    child.signalCode = null;
    child.kill = vi.fn(killImplementation);

    let result: CodexProcessResult | undefined;
    void runCodexProcess(
      {
        command: "codex",
        args: ["exec", "-"],
        cwd: "/tmp/staging",
        env: {},
        stdin: "prompt",
        shell: false,
        outputFile: "/tmp/staging/output.json",
      },
      {
        timeoutMs: 25,
        terminateGraceMs: 10,
        reapTimeoutMs: 10,
        spawnImpl: (() => child) as never,
      },
    ).then((value) => {
      result = value;
    });

    await vi.advanceTimersByTimeAsync(45);

    expect(result).toMatchObject({
      timedOut: true,
      reaped: false,
      mayBeAlive: true,
    });
    expect(child.kill).toHaveBeenNthCalledWith(1, "SIGTERM");
    expect(child.kill).toHaveBeenNthCalledWith(2, "SIGKILL");
    expect(child.stdin.destroyed).toBe(true);
    expect(child.stdout.destroyed).toBe(true);
    expect(child.stderr.destroyed).toBe(true);
  });

  it("keeps a safe error sink after forcing settlement of an unreaped child", async () => {
    vi.useFakeTimers();
    const child = new EventEmitter() as EventEmitter & {
      stdin: PassThrough;
      stdout: PassThrough;
      stderr: PassThrough;
      exitCode: number | null;
      signalCode: NodeJS.Signals | null;
      kill: ReturnType<typeof vi.fn>;
    };
    child.stdin = new PassThrough();
    child.stdout = new PassThrough();
    child.stderr = new PassThrough();
    child.exitCode = null;
    child.signalCode = null;
    child.kill = vi.fn(() => true);

    const resultPromise = runCodexProcess(
      {
        command: "codex",
        args: ["exec", "-"],
        cwd: "/tmp/staging",
        env: {},
        stdin: "prompt",
        shell: false,
        outputFile: "/tmp/staging/output.json",
      },
      {
        timeoutMs: 25,
        terminateGraceMs: 10,
        reapTimeoutMs: 10,
        spawnImpl: (() => child) as never,
      },
    );
    await vi.advanceTimersByTimeAsync(45);
    const result = await resultPromise;

    expect(() => {
      child.emit("error", new Error("late child error"));
    }).not.toThrow();
    expect(result).toMatchObject({
      timedOut: true,
      reaped: false,
      mayBeAlive: true,
    });
    expect(result.error).toBeUndefined();
  });

  it("settles a never-close child as safely dead when exit state becomes observable", async () => {
    vi.useFakeTimers();
    const child = new EventEmitter() as EventEmitter & {
      stdin: PassThrough;
      stdout: PassThrough;
      stderr: PassThrough;
      exitCode: number | null;
      signalCode: NodeJS.Signals | null;
      kill: ReturnType<typeof vi.fn>;
    };
    child.stdin = new PassThrough();
    child.stdout = new PassThrough();
    child.stderr = new PassThrough();
    child.exitCode = null;
    child.signalCode = null;
    child.kill = vi.fn((signal) => {
      if (signal === "SIGKILL") {
        child.signalCode = "SIGKILL";
      }
      return true;
    });

    const resultPromise = runCodexProcess(
      {
        command: "codex",
        args: ["exec", "-"],
        cwd: "/tmp/staging",
        env: {},
        stdin: "prompt",
        shell: false,
        outputFile: "/tmp/staging/output.json",
      },
      {
        timeoutMs: 25,
        terminateGraceMs: 10,
        reapTimeoutMs: 10,
        spawnImpl: (() => child) as never,
      },
    );
    await vi.advanceTimersByTimeAsync(45);

    await expect(resultPromise).resolves.toMatchObject({
      timedOut: true,
      reaped: true,
      mayBeAlive: false,
    });
  });
});

describe("enrichment orchestration", () => {
  async function exercise(
    runner: CodexRunner,
    options: {
      currentHash?: string;
      parentEnv?: NodeJS.ProcessEnv;
      lesson?: StagedLessonInput;
      inspectStaging?: (invocation: CodexInvocation) => Promise<void>;
    } = {},
  ) {
    const root = await makeTemporaryRoot();
    const stagingParent = path.join(root, "transaction-staging");
    await fs.mkdir(stagingParent, { mode: 0o700 });
    const prompt = await fs.readFile(promptPath, "utf8");
    const guardedPaths: string[] = [];
    const wrappedRunner: CodexRunner = async (invocation, timeoutMs) => {
      await options.inspectStaging?.(invocation);
      return runner(invocation, timeoutMs);
    };

    const result = await enrichMetadata({
      stagingParent,
      assertSafeStagingPath(target) {
        const relative = path.relative(stagingParent, target);
        if (
          target !== stagingParent &&
          (relative === "" ||
            relative === ".." ||
            relative.startsWith(`..${path.sep}`) ||
            path.isAbsolute(relative))
        ) {
          throw new Error("unsafe staging path");
        }
        guardedPaths.push(target);
      },
      lesson: options.lesson ?? stagedLesson,
      readCurrentSourceHash: async () => options.currentHash ?? HASH_A,
      runner: wrappedRunner,
      schemaPath,
      prompt,
      parentEnv: options.parentEnv,
      timeoutMs: 100,
    });

    expect(await fs.readdir(stagingParent)).toEqual([]);
    return { result, guardedPaths, stagingParent };
  }

  function processResult(
    overrides: Partial<CodexProcessResult> = {},
  ): CodexProcessResult {
    return {
      exitCode: 0,
      signal: null,
      stdout: "",
      stderr: "",
      timedOut: false,
      ...overrides,
    };
  }

  function outputRunner(
    content: string | Buffer,
    result: CodexProcessResult = processResult(),
  ): CodexRunner {
    return async (invocation) => {
      await fs.writeFile(invocation.outputFile, content);
      return result;
    };
  }

  it("stages only minimized input with restrictive modes and returns valid output", async () => {
    const sourcePath = "/private/source/ai-daily";
    const { result, guardedPaths, stagingParent } = await exercise(
      outputRunner(JSON.stringify(validOutput())),
      {
        parentEnv: {
          AI_DAILY_SOURCE: sourcePath,
          PWD: sourcePath,
          SAFE_PARENT_VALUE: "kept",
        },
        async inspectStaging(invocation) {
          const staged = JSON.parse(
            await fs.readFile(path.join(invocation.cwd, "lesson.json"), "utf8"),
          );
          const directoryMode = (await fs.stat(invocation.cwd)).mode & 0o777;
          const inputMode =
            (await fs.stat(path.join(invocation.cwd, "lesson.json"))).mode &
            0o777;

          expect(staged).toEqual(stagedLesson);
          expect(Object.keys(staged)).toEqual([
            "schemaVersion",
            "id",
            "sourceHash",
            "titleZh",
            "summaryZh",
            "tldrZh",
            "bodyMarkdown",
          ]);
          expect(directoryMode).toBe(0o700);
          expect(inputMode).toBe(0o600);
          expect(invocation.env).not.toHaveProperty("AI_DAILY_SOURCE");
          expect(invocation.env.PWD).toBe(invocation.cwd);
          expect(invocation.env.SAFE_PARENT_VALUE).toBe("kept");
          expect(JSON.stringify({ staged, invocation })).not.toContain(
            sourcePath,
          );
        },
      },
    );

    expect(result).toEqual({ ok: true, value: validOutput() });
    expect(guardedPaths).toContain(stagingParent);
    expect(guardedPaths.some((target) => target.endsWith("lesson.json"))).toBe(
      true,
    );
  });

  it("refuses to stage the known AI Daily source path", async () => {
    const sourcePath = "/private/source/ai-daily";

    await expect(
      exercise(outputRunner(JSON.stringify(validOutput())), {
        parentEnv: { AI_DAILY_SOURCE: sourcePath },
        lesson: {
          ...stagedLesson,
          bodyMarkdown: `Do not expose ${sourcePath}/lessons/2026-07-17.md`,
        },
      }),
    ).rejects.toThrow(/must not expose the AI Daily source path/i);
  });

  it("returns stale-source with no enrichment when the hash changes just before return", async () => {
    const { result } = await exercise(
      outputRunner(JSON.stringify(validOutput())),
      { currentHash: HASH_B },
    );

    expect(result).toMatchObject({ ok: false, reason: "stale-source" });
    expect(result).not.toHaveProperty("value");
  });

  it.each([
    [
      "Codex ENOENT",
      async () =>
        processResult({
          exitCode: null,
          error: Object.assign(new Error("spawn codex ENOENT"), {
            code: "ENOENT",
          }),
        }),
      "codex-unavailable",
    ],
    [
      "timeout",
      async () =>
        processResult({
          exitCode: null,
          timedOut: true,
          reaped: true,
          mayBeAlive: false,
        }),
      "timeout",
    ],
    [
      "nonzero exit",
      async () => processResult({ exitCode: 2, stderr: "failure" }),
      "process-failed",
    ],
  ])("returns a typed soft failure for %s and cleans staging", async (
    _label,
    runner,
    reason,
  ) => {
    const { result } = await exercise(runner as CodexRunner);

    expect(result).toMatchObject({ ok: false, reason });
  });

  it("preserves staging and throws when a runner reports an unreaped process", async () => {
    const root = await makeTemporaryRoot();
    const stagingParent = path.join(root, "transaction-staging");
    await fs.mkdir(stagingParent, { mode: 0o700 });

    await expect(
      enrichMetadata({
        stagingParent,
        assertSafeStagingPath() {},
        lesson: stagedLesson,
        readCurrentSourceHash: async () => HASH_A,
        runner: async () =>
          processResult({
            exitCode: null,
            timedOut: true,
            reaped: false,
            mayBeAlive: true,
          }),
        schemaPath,
        prompt: await fs.readFile(promptPath, "utf8"),
      }),
    ).rejects.toThrow(/process may still be alive|unreaped.*preserved/i);
    const preserved = await fs.readdir(stagingParent);
    expect(preserved).toHaveLength(1);
    expect(preserved[0]).toMatch(/^metadata-enrichment-/);
  });

  it("allows stderr warnings when Codex exits zero with valid output", async () => {
    const { result } = await exercise(
      outputRunner(
        JSON.stringify(validOutput()),
        processResult({ stderr: "warning: harmless" }),
      ),
    );

    expect(result).toEqual({ ok: true, value: validOutput() });
  });

  it.each([
    ["missing output", async () => processResult()],
    ["malformed output", outputRunner("not json")],
    [
      "unknown output key",
      outputRunner(JSON.stringify({ ...validOutput(), unknown: true })),
    ],
    ["oversized output", outputRunner(Buffer.alloc(65_537, "x"))],
  ])("returns invalid-output for %s and cleans staging", async (
    _label,
    runner,
  ) => {
    const { result } = await exercise(runner as CodexRunner);

    expect(result).toMatchObject({ ok: false, reason: "invalid-output" });
  });

  it("rejects a symlink output without following it and cleans staging", async () => {
    const root = await makeTemporaryRoot();
    const outside = path.join(root, "outside.json");
    await fs.writeFile(outside, JSON.stringify(validOutput()));
    const runner: CodexRunner = async (invocation) => {
      await fs.symlink(outside, invocation.outputFile);
      return processResult();
    };

    const { result } = await exercise(runner);

    expect(result).toMatchObject({ ok: false, reason: "invalid-output" });
    await expect(fs.readFile(outside, "utf8")).resolves.toBe(
      JSON.stringify(validOutput()),
    );
  });

  it("rejects a hardlinked output file and cleans only the staging link", async () => {
    const root = await makeTemporaryRoot();
    const outside = path.join(root, "outside.json");
    await fs.writeFile(outside, JSON.stringify(validOutput()));
    const runner: CodexRunner = async (invocation) => {
      await fs.link(outside, invocation.outputFile);
      return processResult();
    };

    const { result } = await exercise(runner);

    expect(result).toMatchObject({ ok: false, reason: "invalid-output" });
    await expect(fs.readFile(outside, "utf8")).resolves.toBe(
      JSON.stringify(validOutput()),
    );
    expect((await fs.stat(outside)).nlink).toBe(1);
  });

  it("throws path-safety failures and removes a child created before a child guard fails", async () => {
    const root = await makeTemporaryRoot();
    const stagingParent = path.join(root, "transaction-staging");
    await fs.mkdir(stagingParent, { mode: 0o700 });
    let calls = 0;

    await expect(
      enrichMetadata({
        stagingParent,
        assertSafeStagingPath() {
          calls += 1;
          if (calls === 2) {
            throw new Error("unsafe child path");
          }
        },
        lesson: stagedLesson,
        readCurrentSourceHash: async () => HASH_A,
        runner: outputRunner(JSON.stringify(validOutput())),
        schemaPath,
        prompt: await fs.readFile(promptPath, "utf8"),
      }),
    ).rejects.toThrow(/unsafe child path/);
    expect(await fs.readdir(stagingParent)).toEqual([]);
  });

  it("detects staging-parent replacement before lesson write and preserves the original child", async () => {
    const root = await makeTemporaryRoot();
    const stagingParent = path.join(root, "transaction-staging");
    const displacedParent = path.join(root, "displaced-staging");
    await fs.mkdir(stagingParent, { mode: 0o700 });
    const runner = vi.fn() as unknown as CodexRunner;
    let replaced = false;

    await expect(
      enrichMetadata({
        stagingParent,
        async assertSafeStagingPath(target) {
          if (!replaced && target.endsWith("lesson.json")) {
            replaced = true;
            await fs.rename(stagingParent, displacedParent);
            await fs.mkdir(stagingParent, { mode: 0o700 });
          }
        },
        lesson: stagedLesson,
        readCurrentSourceHash: async () => HASH_A,
        runner,
        schemaPath,
        prompt: await fs.readFile(promptPath, "utf8"),
      }),
    ).rejects.toThrow(/staging parent.*identity changed/i);
    expect(runner).not.toHaveBeenCalled();
    expect(await fs.readdir(stagingParent)).toEqual([]);
    const preservedChildren = await fs.readdir(displacedParent);
    expect(preservedChildren).toHaveLength(1);
    await expect(
      fs.stat(path.join(displacedParent, preservedChildren[0]!, "lesson.json")),
    ).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("detects staging-child replacement before cleanup and never deletes the replacement", async () => {
    const root = await makeTemporaryRoot();
    const stagingParent = path.join(root, "transaction-staging");
    const displacedChild = path.join(root, "displaced-child");
    await fs.mkdir(stagingParent, { mode: 0o700 });

    await expect(
      enrichMetadata({
        stagingParent,
        assertSafeStagingPath() {},
        lesson: stagedLesson,
        readCurrentSourceHash: async () => HASH_A,
        runner: async (invocation) => {
          await fs.rename(invocation.cwd, displacedChild);
          await fs.mkdir(invocation.cwd, { mode: 0o700 });
          await fs.writeFile(path.join(invocation.cwd, "replacement.txt"), "keep");
          return processResult({ exitCode: 2, reaped: true, mayBeAlive: false });
        },
        schemaPath,
        prompt: await fs.readFile(promptPath, "utf8"),
      }),
    ).rejects.toThrow(/staging directory.*identity changed/i);
    await expect(
      fs.readFile(
        path.join(stagingParent, path.basename(displacedChild), "replacement.txt"),
        "utf8",
      ),
    ).rejects.toMatchObject({ code: "ENOENT" });
    const replacements = await fs.readdir(stagingParent);
    expect(replacements).toHaveLength(1);
    await expect(
      fs.readFile(
        path.join(stagingParent, replacements[0]!, "replacement.txt"),
        "utf8",
      ),
    ).resolves.toBe("keep");
    await expect(fs.stat(displacedChild)).resolves.toMatchObject({});
  });
});
