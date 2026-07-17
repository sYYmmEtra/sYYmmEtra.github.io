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
const temporaryRoots: string[] = [];

function words(count: number): string {
  return Array.from({ length: count }, (_, index) => `word${index + 1}`).join(
    " ",
  );
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

afterEach(async () => {
  vi.useRealTimers();
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
});

describe("checked-in output schema", () => {
  it("has the exact strict draft-2020-12 shape and Zod-parity constraints", async () => {
    const schema = JSON.parse(await fs.readFile(schemaPath, "utf8"));

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
        titleEn: {
          type: "string",
          minLength: ENRICHMENT_CONSTRAINTS.titleMinLength,
          maxLength: ENRICHMENT_CONSTRAINTS.titleMaxLength,
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
    expect(CLI_HELP).not.toContain("AI_DAILY_SOURCE");
  });

  it("registers the metadata enrichment command only after the entry point exists", async () => {
    const packageJson = JSON.parse(await fs.readFile("package.json", "utf8"));

    expect(packageJson.scripts["metadata:enrich"]).toBe(
      "tsx scripts/enrich-metadata.ts",
    );
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

  it("kills a process after the configured timeout", async () => {
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
    child.kill = vi.fn(() => {
      queueMicrotask(() => child.emit("close", null, "SIGKILL"));
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
      { timeoutMs: 25, spawnImpl: (() => child) as never },
    );
    await vi.advanceTimersByTimeAsync(25);

    await expect(resultPromise).resolves.toMatchObject({ timedOut: true });
    expect(child.kill).toHaveBeenCalledWith("SIGKILL");
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
      async () => processResult({ exitCode: null, timedOut: true }),
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
});
