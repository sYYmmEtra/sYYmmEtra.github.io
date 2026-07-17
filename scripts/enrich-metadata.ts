import { constants } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import { spawn, type ChildProcess } from "node:child_process";
import { fileURLToPath } from "node:url";
import { z } from "zod";

import {
  LessonIdSchema,
  SourceHashSchema,
  validateEnrichment,
  type EnrichmentOutput,
} from "./lib/metadata";

const SCRIPT_DIRECTORY = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_SCHEMA_PATH = path.join(
  SCRIPT_DIRECTORY,
  "metadata-output.schema.json",
);
const DEFAULT_PROMPT_PATH = path.join(SCRIPT_DIRECTORY, "metadata-prompt.md");
const MAX_OUTPUT_BYTES = 64 * 1024;
const DEFAULT_TIMEOUT_MS = 120_000;

export const StagedLessonInputSchema = z
  .object({
    schemaVersion: z.literal(1),
    id: LessonIdSchema,
    sourceHash: SourceHashSchema,
    titleZh: z.string().min(1),
    summaryZh: z.string().min(1),
    tldrZh: z.string().nullable(),
    bodyMarkdown: z.string(),
  })
  .strict();

export type StagedLessonInput = z.infer<typeof StagedLessonInputSchema>;

export interface CodexInvocation {
  command: "codex";
  args: readonly string[];
  cwd: string;
  env: NodeJS.ProcessEnv;
  stdin: string;
  shell: false;
  outputFile: string;
}

export interface BuildCodexInvocationOptions {
  stagingDir: string;
  outputFile: string;
  schemaPath: string;
  prompt: string;
  parentEnv?: NodeJS.ProcessEnv;
}

function removeKnownSourcePathFromEnvironment(
  environment: NodeJS.ProcessEnv,
  sourcePath: string | undefined,
): void {
  delete environment.AI_DAILY_SOURCE;
  if (!sourcePath) {
    return;
  }

  for (const [key, value] of Object.entries(environment)) {
    if (value?.includes(sourcePath)) {
      delete environment[key];
    }
  }
}

export function buildCodexInvocation(
  options: BuildCodexInvocationOptions,
): CodexInvocation {
  const parentEnvironment = options.parentEnv ?? process.env;
  const knownSourcePath = parentEnvironment.AI_DAILY_SOURCE;
  const exposedValues = [
    options.stagingDir,
    options.outputFile,
    options.schemaPath,
    options.prompt,
  ];
  if (
    knownSourcePath &&
    exposedValues.some((value) => value.includes(knownSourcePath))
  ) {
    throw new Error("Codex invocation must not expose the AI Daily source path");
  }

  const environment = { ...parentEnvironment };
  removeKnownSourcePathFromEnvironment(environment, knownSourcePath);
  environment.PWD = options.stagingDir;

  return {
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
      options.schemaPath,
      "-C",
      options.stagingDir,
      "-o",
      options.outputFile,
      "--color",
      "never",
      "-",
    ],
    cwd: options.stagingDir,
    env: environment,
    stdin: options.prompt,
    shell: false,
    outputFile: options.outputFile,
  };
}

export interface CodexProcessResult {
  exitCode: number | null;
  signal: NodeJS.Signals | null;
  stdout: string;
  stderr: string;
  timedOut: boolean;
  error?: Error & { code?: string };
}

export interface RunCodexProcessOptions {
  timeoutMs: number;
  spawnImpl?: typeof spawn;
}

function asProcessError(error: unknown): Error & { code?: string } {
  if (error instanceof Error) {
    return error;
  }
  return new Error(String(error));
}

export async function runCodexProcess(
  invocation: CodexInvocation,
  options: RunCodexProcessOptions,
): Promise<CodexProcessResult> {
  const spawnImpl = options.spawnImpl ?? spawn;

  return new Promise((resolve) => {
    let child: ChildProcess;
    try {
      child = spawnImpl(invocation.command, [...invocation.args], {
        cwd: invocation.cwd,
        env: invocation.env,
        shell: false,
        stdio: ["pipe", "pipe", "pipe"],
      });
    } catch (error) {
      resolve({
        exitCode: null,
        signal: null,
        stdout: "",
        stderr: "",
        timedOut: false,
        error: asProcessError(error),
      });
      return;
    }

    let stdout = "";
    let stderr = "";
    let timedOut = false;
    let settled = false;

    child.stdout?.setEncoding("utf8");
    child.stderr?.setEncoding("utf8");
    child.stdout?.on("data", (chunk: string) => {
      stdout += chunk;
    });
    child.stderr?.on("data", (chunk: string) => {
      stderr += chunk;
    });
    child.stdin?.on("error", () => {
      // Process exit/error is authoritative; EPIPE on stdin is only a symptom.
    });

    const finish = (
      exitCode: number | null,
      signal: NodeJS.Signals | null,
      error?: Error & { code?: string },
    ) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timeout);
      resolve({
        exitCode,
        signal,
        stdout,
        stderr,
        timedOut,
        ...(error === undefined ? {} : { error }),
      });
    };

    const timeout = setTimeout(() => {
      timedOut = true;
      try {
        child.kill("SIGKILL");
      } catch (error) {
        finish(null, null, asProcessError(error));
      }
    }, options.timeoutMs);
    timeout.unref();

    child.once("error", (error) => {
      finish(null, null, asProcessError(error));
    });
    child.once("close", (exitCode, signal) => {
      finish(exitCode, signal);
    });
    child.stdin?.end(invocation.stdin, "utf8");
  });
}

export type CodexRunner = (
  invocation: CodexInvocation,
  timeoutMs: number,
) => Promise<CodexProcessResult>;

export type EnrichmentFailureReason =
  | "codex-unavailable"
  | "process-failed"
  | "timeout"
  | "invalid-output"
  | "stale-source";

export type EnrichmentResult =
  | { ok: true; value: EnrichmentOutput }
  | {
      ok: false;
      reason: EnrichmentFailureReason;
      message: string;
    };

export interface EnrichMetadataOptions {
  stagingParent: string;
  assertSafeStagingPath: (
    target: string,
  ) => void | Promise<void>;
  lesson: StagedLessonInput;
  readCurrentSourceHash: () => Promise<string>;
  runner?: CodexRunner;
  schemaPath?: string;
  prompt?: string;
  parentEnv?: NodeJS.ProcessEnv;
  timeoutMs?: number;
}

function processFailure(
  reason: EnrichmentFailureReason,
  message: string,
): EnrichmentResult {
  return { ok: false, reason, message };
}

function errorCode(error: unknown): string | undefined {
  return error instanceof Error && "code" in error
    ? String(error.code)
    : undefined;
}

function isWithin(parent: string, target: string): boolean {
  const relative = path.relative(parent, target);
  return (
    relative !== "" &&
    relative !== ".." &&
    !relative.startsWith(`..${path.sep}`) &&
    !path.isAbsolute(relative)
  );
}

async function assertOutputAbsent(outputFile: string): Promise<void> {
  try {
    await fs.lstat(outputFile);
  } catch (error) {
    if (errorCode(error) === "ENOENT") {
      return;
    }
    throw error;
  }
  throw new Error("Codex output path must be absent before invocation");
}

async function readValidatedOutput(
  outputFile: string,
  expectedId: string,
  expectedSourceHash: string,
): Promise<EnrichmentResult> {
  let outputStat;
  try {
    outputStat = await fs.lstat(outputFile);
  } catch (error) {
    if (errorCode(error) === "ENOENT") {
      return processFailure("invalid-output", "Codex did not create output");
    }
    throw error;
  }

  if (outputStat.isSymbolicLink() || !outputStat.isFile()) {
    return processFailure(
      "invalid-output",
      "Codex output must be a regular non-symlink file",
    );
  }
  if (outputStat.size > MAX_OUTPUT_BYTES) {
    return processFailure(
      "invalid-output",
      `Codex output exceeds ${MAX_OUTPUT_BYTES} bytes`,
    );
  }

  let handle;
  try {
    handle = await fs.open(
      outputFile,
      constants.O_RDONLY | constants.O_NOFOLLOW,
    );
  } catch (error) {
    if (errorCode(error) === "ELOOP") {
      return processFailure(
        "invalid-output",
        "Codex output must not be a symlink",
      );
    }
    throw error;
  }

  let content: Buffer;
  try {
    const openedStat = await handle.stat();
    if (!openedStat.isFile() || openedStat.size > MAX_OUTPUT_BYTES) {
      return processFailure(
        "invalid-output",
        "Codex output is not a regular file within the size limit",
      );
    }
    content = await handle.readFile();
  } finally {
    await handle.close();
  }

  if (content.byteLength > MAX_OUTPUT_BYTES) {
    return processFailure(
      "invalid-output",
      `Codex output exceeds ${MAX_OUTPUT_BYTES} bytes`,
    );
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(content.toString("utf8"));
  } catch {
    return processFailure("invalid-output", "Codex output is not valid JSON");
  }

  try {
    return {
      ok: true,
      value: validateEnrichment(parsed, {
        expectedId,
        expectedSourceHash,
      }),
    };
  } catch (error) {
    return processFailure(
      "invalid-output",
      `Codex output failed validation: ${asProcessError(error).message}`,
    );
  }
}

function classifyProcessResult(result: CodexProcessResult): EnrichmentResult | null {
  if (result.timedOut) {
    return processFailure("timeout", "Codex enrichment timed out");
  }
  if (result.error?.code === "ENOENT") {
    return processFailure(
      "codex-unavailable",
      "Codex executable is unavailable",
    );
  }
  if (result.error !== undefined || result.exitCode !== 0) {
    const detail = result.error?.message || result.stderr.trim();
    return processFailure(
      "process-failed",
      detail
        ? `Codex enrichment failed: ${detail}`
        : `Codex enrichment exited with code ${String(result.exitCode)}`,
    );
  }
  return null;
}

export async function enrichMetadata(
  options: EnrichMetadataOptions,
): Promise<EnrichmentResult> {
  const lesson = StagedLessonInputSchema.parse(options.lesson);
  const stagingParent = path.resolve(options.stagingParent);
  const schemaPath = path.resolve(options.schemaPath ?? DEFAULT_SCHEMA_PATH);
  const prompt =
    options.prompt ?? (await fs.readFile(DEFAULT_PROMPT_PATH, "utf8"));
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const knownSourcePath = (options.parentEnv ?? process.env).AI_DAILY_SOURCE;
  if (
    knownSourcePath &&
    JSON.stringify(lesson).includes(knownSourcePath)
  ) {
    throw new Error("Staged lesson must not expose the AI Daily source path");
  }
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0) {
    throw new Error("timeoutMs must be a positive integer");
  }

  await options.assertSafeStagingPath(stagingParent);
  const parentStat = await fs.lstat(stagingParent);
  if (parentStat.isSymbolicLink() || !parentStat.isDirectory()) {
    throw new Error("Staging parent must be a physical directory");
  }

  let stagingDir: string | undefined;
  let candidate: EnrichmentResult | undefined;
  try {
    stagingDir = await fs.mkdtemp(
      path.join(stagingParent, "metadata-enrichment-"),
    );
    await fs.chmod(stagingDir, 0o700);
    if (!isWithin(stagingParent, stagingDir)) {
      throw new Error("Unique staging directory escaped its parent");
    }
    await options.assertSafeStagingPath(stagingDir);

    const inputFile = path.join(stagingDir, "lesson.json");
    const outputFile = path.join(stagingDir, "output.json");
    await options.assertSafeStagingPath(inputFile);
    await options.assertSafeStagingPath(outputFile);
    await fs.writeFile(inputFile, `${JSON.stringify(lesson, null, 2)}\n`, {
      encoding: "utf8",
      flag: "wx",
      mode: 0o600,
    });
    await fs.chmod(inputFile, 0o600);
    await assertOutputAbsent(outputFile);

    const invocation = buildCodexInvocation({
      stagingDir,
      outputFile,
      schemaPath,
      prompt,
      parentEnv: options.parentEnv,
    });
    const runner =
      options.runner ??
      ((nextInvocation, nextTimeoutMs) =>
        runCodexProcess(nextInvocation, { timeoutMs: nextTimeoutMs }));

    let processResult: CodexProcessResult;
    try {
      processResult = await runner(invocation, timeoutMs);
    } catch (error) {
      candidate =
        errorCode(error) === "ENOENT"
          ? processFailure(
              "codex-unavailable",
              "Codex executable is unavailable",
            )
          : processFailure(
              "process-failed",
              `Codex runner failed: ${asProcessError(error).message}`,
            );
    }

    if (candidate === undefined) {
      const failedProcess = classifyProcessResult(processResult!);
      candidate =
        failedProcess ??
        (await readValidatedOutput(outputFile, lesson.id, lesson.sourceHash));
    }
  } finally {
    if (stagingDir !== undefined) {
      await fs.rm(stagingDir, { recursive: true, force: true });
    }
  }

  if (candidate === undefined) {
    throw new Error("Enrichment completed without a result");
  }
  if (!candidate.ok) {
    return candidate;
  }

  const currentHash = SourceHashSchema.parse(
    await options.readCurrentSourceHash(),
  );
  if (currentHash !== lesson.sourceHash) {
    return processFailure(
      "stale-source",
      `Source changed during enrichment: staged ${lesson.sourceHash}, current ${currentHash}`,
    );
  }
  return candidate;
}

export const CLI_HELP = `Usage:
  npm run metadata:enrich -- --input <lesson.json> --staging-parent <directory> [--timeout-ms <milliseconds>]

Arguments:
  --input <lesson.json>          Minimized schemaVersion 1 lesson record.
  --staging-parent <directory>   Existing website-transaction staging directory.
  --timeout-ms <milliseconds>    Optional positive timeout; defaults to 120000.
  --help                         Show this help.

The command prints validated enrichment JSON to stdout and never writes sidecar metadata.`;

interface CliArguments {
  input: string;
  stagingParent: string;
  timeoutMs?: number;
}

function parseCliArguments(argv: readonly string[]): CliArguments | "help" {
  if (argv.includes("--help")) {
    return "help";
  }

  let input: string | undefined;
  let stagingParent: string | undefined;
  let timeoutMs: number | undefined;

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    const value = argv[index + 1];
    if (argument === "--input" || argument === "--staging-parent") {
      if (!value || value.startsWith("--")) {
        throw new Error(`${argument} requires a value`);
      }
      if (argument === "--input") {
        input = value;
      } else {
        stagingParent = value;
      }
      index += 1;
      continue;
    }
    if (argument === "--timeout-ms") {
      if (!value || !/^\d+$/.test(value)) {
        throw new Error("--timeout-ms requires a positive integer");
      }
      timeoutMs = Number(value);
      if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0) {
        throw new Error("--timeout-ms requires a positive integer");
      }
      index += 1;
      continue;
    }
    throw new Error(`Unknown argument: ${String(argument)}`);
  }

  if (!input || !stagingParent) {
    throw new Error("--input and --staging-parent are required");
  }
  return { input, stagingParent, ...(timeoutMs === undefined ? {} : { timeoutMs }) };
}

function assertCliStagingPath(parent: string, target: string): void {
  if (target === parent) {
    return;
  }
  if (!isWithin(parent, target)) {
    throw new Error("Staging path is outside the supplied staging parent");
  }
}

async function readCliLesson(inputFile: string): Promise<StagedLessonInput> {
  return StagedLessonInputSchema.parse(
    JSON.parse(await fs.readFile(inputFile, "utf8")),
  );
}

async function main(): Promise<void> {
  let parsed: CliArguments | "help";
  try {
    parsed = parseCliArguments(process.argv.slice(2));
  } catch (error) {
    console.error(`${asProcessError(error).message}\n\n${CLI_HELP}`);
    process.exitCode = 2;
    return;
  }

  if (parsed === "help") {
    console.log(CLI_HELP);
    return;
  }

  const inputFile = path.resolve(parsed.input);
  const stagingParent = await fs.realpath(parsed.stagingParent);
  const lesson = await readCliLesson(inputFile);
  const result = await enrichMetadata({
    stagingParent,
    assertSafeStagingPath: (target) =>
      assertCliStagingPath(stagingParent, target),
    lesson,
    readCurrentSourceHash: async () =>
      (await readCliLesson(inputFile)).sourceHash,
    timeoutMs: parsed.timeoutMs,
  });

  if (!result.ok) {
    console.error(`${result.reason}: ${result.message}`);
    process.exitCode = 1;
    return;
  }
  console.log(JSON.stringify(result.value, null, 2));
}

const invokedPath = process.argv[1]
  ? path.resolve(process.argv[1])
  : undefined;
if (invokedPath === fileURLToPath(import.meta.url)) {
  void main().catch((error) => {
    console.error(asProcessError(error).message);
    process.exitCode = 1;
  });
}
