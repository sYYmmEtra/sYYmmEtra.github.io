import { constants, realpathSync } from "node:fs";
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
import {
  assertDisjointRoots,
  assertSafeWritePath,
  resolveExistingPath,
} from "./lib/paths";

const SCRIPT_DIRECTORY = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_SCHEMA_PATH = path.join(
  SCRIPT_DIRECTORY,
  "metadata-output.schema.json",
);
const DEFAULT_PROMPT_PATH = path.join(SCRIPT_DIRECTORY, "metadata-prompt.md");
const MAX_OUTPUT_BYTES = 64 * 1024;
const MAX_CLI_INPUT_BYTES = 2 * 1024 * 1024;
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

interface SourcePathPolicy {
  textVariants: readonly string[];
  absoluteRoots: readonly string[];
}

function tryResolvePhysicalPath(value: string): string | undefined {
  let ancestor = path.resolve(value);
  const missingSuffix: string[] = [];

  while (true) {
    try {
      return path.resolve(realpathSync(ancestor), ...missingSuffix);
    } catch (error) {
      const code = errorCode(error);
      if (code !== "ENOENT" && code !== "ENOTDIR") {
        return undefined;
      }
      const parent = path.dirname(ancestor);
      if (parent === ancestor) {
        return undefined;
      }
      missingSuffix.unshift(path.basename(ancestor));
      ancestor = parent;
    }
  }
}

function unique(values: readonly (string | undefined)[]): string[] {
  return [...new Set(values.filter((value): value is string => Boolean(value)))];
}

function createSourcePathPolicy(
  sourcePath: string | undefined,
): SourcePathPolicy | undefined {
  if (!sourcePath) {
    return undefined;
  }

  const resolvedSource = path.resolve(sourcePath);
  const physicalSource = tryResolvePhysicalPath(sourcePath);
  return {
    textVariants: unique([sourcePath, resolvedSource, physicalSource]),
    absoluteRoots: unique([resolvedSource, physicalSource]),
  };
}

function isEqualOrDescendant(parent: string, candidate: string): boolean {
  const relative = path.relative(parent, candidate);
  return (
    relative === "" ||
    (relative !== ".." &&
      !relative.startsWith(`..${path.sep}`) &&
      !path.isAbsolute(relative))
  );
}

function absolutePathExposesSource(
  policy: SourcePathPolicy,
  value: string,
): boolean {
  if (!path.isAbsolute(value)) {
    return false;
  }
  const candidatePaths = unique([
    path.resolve(value),
    tryResolvePhysicalPath(value),
  ]);
  return policy.absoluteRoots.some((sourceRoot) =>
    candidatePaths.some((candidate) =>
      isEqualOrDescendant(sourceRoot, candidate),
    ),
  );
}

function valueExposesSource(
  policy: SourcePathPolicy,
  value: string,
): boolean {
  if (policy.textVariants.some((variant) => value.includes(variant))) {
    return true;
  }
  if (absolutePathExposesSource(policy, value)) {
    return true;
  }

  const absoluteTokens = value.match(/\/[^\s"'`<>|\\]+/g) ?? [];
  return absoluteTokens.some((candidate) =>
    absolutePathExposesSource(policy, candidate),
  );
}

function isPathLikeEnvironmentKey(key: string): boolean {
  return key === "PATH" || key.endsWith("PATH") || key.endsWith("PATHS");
}

function sanitizeEnvironment(
  environment: NodeJS.ProcessEnv,
  policy: SourcePathPolicy | undefined,
): void {
  delete environment.AI_DAILY_SOURCE;
  if (!policy) {
    return;
  }

  for (const [key, value] of Object.entries(environment)) {
    if (value === undefined) {
      continue;
    }
    if (isPathLikeEnvironmentKey(key) && value.includes(path.delimiter)) {
      const safeEntries = value
        .split(path.delimiter)
        .filter(
          (entry) => entry === "" || !valueExposesSource(policy, entry),
        );
      if (safeEntries.length === 0) {
        delete environment[key];
      } else {
        environment[key] = safeEntries.join(path.delimiter);
      }
      continue;
    }
    if (valueExposesSource(policy, value)) {
      delete environment[key];
    }
  }
}

export function buildCodexInvocation(
  options: BuildCodexInvocationOptions,
): CodexInvocation {
  const parentEnvironment = options.parentEnv ?? process.env;
  const sourcePolicy = createSourcePathPolicy(
    parentEnvironment.AI_DAILY_SOURCE,
  );
  const exposedValues = [
    options.stagingDir,
    options.outputFile,
    options.schemaPath,
    options.prompt,
  ];
  if (
    sourcePolicy &&
    exposedValues.some((value) => valueExposesSource(sourcePolicy, value))
  ) {
    throw new Error("Codex invocation must not expose the AI Daily source path");
  }

  const environment = { ...parentEnvironment };
  sanitizeEnvironment(environment, sourcePolicy);
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
  const sourcePolicy = createSourcePathPolicy(
    (options.parentEnv ?? process.env).AI_DAILY_SOURCE,
  );
  if (
    sourcePolicy &&
    valueExposesSource(sourcePolicy, JSON.stringify(lesson))
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
  --input <lesson.json>          Existing minimized schemaVersion 1 record inside --staging-parent (max 2 MiB).
  --staging-parent <directory>   Existing physical transaction directory strictly below <website>/.sync-tmp/.
  --timeout-ms <milliseconds>    Optional positive timeout; defaults to 120000.
  --help                         Show this help.

The command prints validated enrichment JSON to stdout and never writes sidecar metadata.`;

interface CliArguments {
  input: string;
  stagingParent: string;
  timeoutMs?: number;
}

export interface MetadataEnrichmentCliOptions {
  runner?: CodexRunner;
  parentEnv?: NodeJS.ProcessEnv;
  stdout?: (message: string) => void;
  stderr?: (message: string) => void;
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

function isPhysicalDescendant(parent: string, candidate: string): boolean {
  const relative = path.relative(parent, candidate);
  return (
    relative !== "" &&
    relative !== ".." &&
    !relative.startsWith(`..${path.sep}`) &&
    !path.isAbsolute(relative)
  );
}

async function requirePhysicalDirectory(
  target: string,
  description: string,
): Promise<void> {
  const stat = await fs.lstat(target);
  if (stat.isSymbolicLink() || !stat.isDirectory()) {
    throw new Error(`${description} must be an existing physical directory`);
  }
}

async function resolveCliStagingParent(
  websiteRoot: string,
  requestedParent: string,
  parentEnv: NodeJS.ProcessEnv,
): Promise<string> {
  const requested = path.resolve(requestedParent);
  await requirePhysicalDirectory(requested, "Staging parent");
  const physicalParent = resolveExistingPath(requested);

  const sourceRootValue = parentEnv.AI_DAILY_SOURCE;
  if (sourceRootValue) {
    const sourceRoot = resolveExistingPath(sourceRootValue);
    try {
      assertDisjointRoots(physicalParent, sourceRoot);
    } catch {
      throw new Error("Staging parent must not overlap the AI Daily source");
    }
  }

  const syncTemporaryRoot = path.join(websiteRoot, ".sync-tmp");
  await requirePhysicalDirectory(
    syncTemporaryRoot,
    "Website .sync-tmp root",
  );
  const physicalSyncTemporaryRoot = resolveExistingPath(syncTemporaryRoot);

  try {
    assertSafeWritePath(websiteRoot, physicalParent);
  } catch {
    throw new Error(
      "Staging parent must be a physical directory strictly inside website .sync-tmp",
    );
  }
  if (!isPhysicalDescendant(physicalSyncTemporaryRoot, physicalParent)) {
    throw new Error(
      "Staging parent must be a physical directory strictly inside website .sync-tmp",
    );
  }

  return physicalParent;
}

function assertCliStagingPath(parent: string, target: string): void {
  if (path.resolve(target) === parent) {
    return;
  }
  assertSafeWritePath(parent, target);
}

async function readCliLesson(
  websiteRoot: string,
  stagingParent: string,
  requestedInput: string,
  parentEnv: NodeJS.ProcessEnv,
): Promise<StagedLessonInput> {
  const inputFile = path.resolve(requestedInput);
  let initialStat;
  try {
    initialStat = await fs.lstat(inputFile);
  } catch (error) {
    if (errorCode(error) === "ENOENT") {
      throw new Error(
        "CLI input must be an existing regular non-symlink file",
      );
    }
    throw error;
  }
  if (initialStat.isSymbolicLink() || !initialStat.isFile()) {
    throw new Error("CLI input must be a regular non-symlink file");
  }
  if (initialStat.size > MAX_CLI_INPUT_BYTES) {
    throw new Error(`CLI input exceeds ${MAX_CLI_INPUT_BYTES} bytes`);
  }

  const physicalInput = resolveExistingPath(inputFile);
  const sourceRootValue = parentEnv.AI_DAILY_SOURCE;
  if (sourceRootValue) {
    const sourceRoot = resolveExistingPath(sourceRootValue);
    try {
      assertDisjointRoots(physicalInput, sourceRoot);
    } catch {
      throw new Error("CLI input must not overlap the AI Daily source");
    }
  }

  try {
    assertSafeWritePath(websiteRoot, physicalInput);
    assertSafeWritePath(stagingParent, physicalInput);
  } catch {
    throw new Error(
      "CLI input must be strictly inside the supplied staging parent",
    );
  }

  let handle;
  try {
    handle = await fs.open(
      physicalInput,
      constants.O_RDONLY | constants.O_NOFOLLOW,
    );
  } catch (error) {
    if (errorCode(error) === "ELOOP") {
      throw new Error("CLI input must be a regular non-symlink file");
    }
    throw error;
  }

  let content: Buffer;
  try {
    const openedStat = await handle.stat();
    if (!openedStat.isFile()) {
      throw new Error("CLI input must be a regular non-symlink file");
    }
    if (
      openedStat.dev !== initialStat.dev ||
      openedStat.ino !== initialStat.ino
    ) {
      throw new Error("CLI input changed while it was being opened");
    }
    if (openedStat.size > MAX_CLI_INPUT_BYTES) {
      throw new Error(`CLI input exceeds ${MAX_CLI_INPUT_BYTES} bytes`);
    }
    content = await handle.readFile();
  } finally {
    await handle.close();
  }

  if (content.byteLength > MAX_CLI_INPUT_BYTES) {
    throw new Error(`CLI input exceeds ${MAX_CLI_INPUT_BYTES} bytes`);
  }
  return StagedLessonInputSchema.parse(JSON.parse(content.toString("utf8")));
}

export async function runMetadataEnrichmentCli(
  argv: readonly string[],
  options: MetadataEnrichmentCliOptions = {},
): Promise<number> {
  const stdout = options.stdout ?? console.log;
  const stderr = options.stderr ?? console.error;
  let parsed: CliArguments | "help";
  try {
    parsed = parseCliArguments(argv);
  } catch (error) {
    stderr(`${asProcessError(error).message}\n\n${CLI_HELP}`);
    return 2;
  }

  if (parsed === "help") {
    stdout(CLI_HELP);
    return 0;
  }

  const parentEnv = options.parentEnv ?? process.env;
  const websiteRoot = resolveExistingPath(
    path.resolve(SCRIPT_DIRECTORY, ".."),
  );
  const stagingParent = await resolveCliStagingParent(
    websiteRoot,
    parsed.stagingParent,
    parentEnv,
  );
  const inputFile = path.resolve(parsed.input);
  const lesson = await readCliLesson(
    websiteRoot,
    stagingParent,
    inputFile,
    parentEnv,
  );
  const result = await enrichMetadata({
    stagingParent,
    assertSafeStagingPath: (target) =>
      assertCliStagingPath(stagingParent, target),
    lesson,
    readCurrentSourceHash: async () =>
      (
        await readCliLesson(
          websiteRoot,
          stagingParent,
          inputFile,
          parentEnv,
        )
      ).sourceHash,
    runner: options.runner,
    parentEnv,
    timeoutMs: parsed.timeoutMs,
  });

  if (!result.ok) {
    stderr(`${result.reason}: ${result.message}`);
    return 1;
  }
  stdout(JSON.stringify(result.value, null, 2));
  return 0;
}

async function main(): Promise<void> {
  process.exitCode = await runMetadataEnrichmentCli(process.argv.slice(2));
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
