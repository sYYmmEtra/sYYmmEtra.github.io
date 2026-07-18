import { execFile } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

import { assertDisjointRoots } from "./lib/paths";

const execFileAsync = promisify(execFile);
const SCRIPT_DIRECTORY = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_WEBSITE_ROOT = path.resolve(SCRIPT_DIRECTORY, "..");
const MAX_DIAGNOSTIC_LENGTH = 4096;
const EXPECTED_ORIGINS = new Set([
  "https://github.com/sYYmmEtra/sYYmmEtra.github.io.git",
  "git@github.com:sYYmmEtra/sYYmmEtra.github.io.git",
]);
const ALLOWED_PATHS = ["metadata/ai-daily", "src/content/ai-daily", "sync-index.json"] as const;

export interface CommandResult {
  code: number;
  stdout: string;
  stderr: string;
}

export interface CommandOptions {
  cwd: string;
}

export type CommandRunner = (
  command: string,
  args: readonly string[],
  options: CommandOptions,
) => Promise<CommandResult>;

export interface PublishOptions {
  /** Test seam. The CLI always resolves its root from this script's location. */
  websiteRoot?: string;
  /** Test seam. Production requires AI_DAILY_SOURCE. */
  sourceRoot?: string;
  runner?: CommandRunner;
  logger?: (message: string) => void;
}

export interface PublishResult {
  committed: boolean;
  changedPaths: string[];
}

export interface PorcelainStatusEntry {
  index: string;
  worktree: string;
  paths: string[];
}

export class PublishError extends Error {}

function output(value: string | Buffer | undefined): string {
  return (value ?? "").toString();
}

async function defaultRunner(
  command: string,
  args: readonly string[],
  options: CommandOptions,
): Promise<CommandResult> {
  try {
    const result = await execFileAsync(command, [...args], {
      cwd: options.cwd,
      encoding: "utf8",
      maxBuffer: 1024 * 1024,
      timeout: 10 * 60 * 1000,
    });
    return { code: 0, stdout: output(result.stdout), stderr: output(result.stderr) };
  } catch (error) {
    const result = error as { code?: number; stdout?: string | Buffer; stderr?: string | Buffer; message?: string };
    return {
      code: typeof result.code === "number" ? result.code : 1,
      stdout: output(result.stdout),
      stderr: output(result.stderr) || result.message || "command could not be started",
    };
  }
}

function trimOutput(value: string): string {
  return value.replace(/\r?\n$/, "");
}

function commandText(command: string, args: readonly string[]): string {
  return [command, ...args].join(" ");
}

function diagnosticDetail(result: CommandResult): string {
  const selected = result.stderr || result.stdout;
  const sanitized = selected
    .replace(/\r\n?/g, "\n")
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]/g, "?")
    .trim();
  if (sanitized.length <= MAX_DIAGNOSTIC_LENGTH) return sanitized;
  return `${sanitized.slice(0, MAX_DIAGNOSTIC_LENGTH)}… [truncated]`;
}

function failureMessage(label: string, command: string, args: readonly string[], result: CommandResult): string {
  const detail = diagnosticDetail(result);
  return `${label}: ${commandText(command, args)}${detail ? `: ${detail}` : ""}`;
}

function commandFailureMessage(command: string, args: readonly string[], result: CommandResult): string {
  const detail = diagnosticDetail(result);
  return `${commandText(command, args)} failed${detail ? `: ${detail}` : ""}`;
}

async function requireCommand(
  runner: CommandRunner,
  cwd: string,
  command: string,
  args: readonly string[],
): Promise<CommandResult> {
  const result = await runner(command, args, { cwd });
  if (result.code !== 0) {
    throw new PublishError(commandFailureMessage(command, args, result));
  }
  return result;
}

async function physicalDirectory(value: string, label: string): Promise<string> {
  const resolved = await fs.realpath(value);
  const stat = await fs.lstat(resolved);
  if (stat.isSymbolicLink() || !stat.isDirectory()) {
    throw new PublishError(`${label} must be a physical directory`);
  }
  return resolved;
}

/** Parse Git porcelain v1 -z without lossy line splitting or quoting. */
export function parsePorcelainStatus(value: string): PorcelainStatusEntry[] {
  if (!value) return [];
  const fields = value.split("\0");
  if (fields.at(-1) === "") fields.pop();
  const entries: PorcelainStatusEntry[] = [];
  for (let index = 0; index < fields.length; index += 1) {
    const field = fields[index]!;
    if (field.length < 4 || field[2] !== " ") {
      throw new PublishError("Malformed git status output");
    }
    const status = { index: field[0]!, worktree: field[1]!, paths: [field.slice(3)] };
    if (status.index === "R" || status.index === "C" || status.worktree === "R" || status.worktree === "C") {
      const original = fields[++index];
      if (original === undefined) throw new PublishError("Malformed rename status output");
      status.paths.push(original);
    }
    entries.push(status);
  }
  return entries;
}

function allowedPath(value: string): boolean {
  return (
    value === "sync-index.json" ||
    value.startsWith("metadata/ai-daily/") ||
    value.startsWith("src/content/ai-daily/")
  );
}

function safeRelativePath(value: string): boolean {
  return (
    value.length > 0 &&
    !path.posix.isAbsolute(value) &&
    !value.includes("\\") &&
    !value.split("/").some((part) => part === "" || part === "." || part === "..")
  );
}

async function assertNoSymlinkAlias(root: string, relative: string): Promise<void> {
  if (!safeRelativePath(relative)) throw new PublishError(`Unsafe repository path: ${JSON.stringify(relative)}`);
  let current = root;
  for (const component of relative.split("/")) {
    current = path.join(current, component);
    try {
      const stat = await fs.lstat(current);
      if (stat.isSymbolicLink()) throw new PublishError(`Symlinked publication path is not allowed: ${relative}`);
    } catch (error) {
      if (error instanceof PublishError) throw error;
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
      throw error;
    }
  }
}

async function assertAllowedPaths(root: string, paths: readonly string[], description: string): Promise<void> {
  for (const candidate of paths) {
    if (!allowedPath(candidate)) throw new PublishError(`Unexpected ${description} path: ${JSON.stringify(candidate)}`);
    await assertNoSymlinkAlias(root, candidate);
  }
}

async function readStatus(runner: CommandRunner, root: string): Promise<PorcelainStatusEntry[]> {
  const result = await requireCommand(runner, root, "git", ["status", "--porcelain=v1", "-z"]);
  return parsePorcelainStatus(result.stdout);
}

async function assertSafeStatus(runner: CommandRunner, root: string): Promise<void> {
  const entries = await readStatus(runner, root);
  for (const entry of entries) {
    if (entry.index === "!" && entry.worktree === "!") continue;
    await assertAllowedPaths(root, entry.paths, "dirty");
    if (entry.index !== " " && entry.index !== "?") {
      throw new PublishError("Refusing publication with pre-existing staged changes");
    }
  }
}

/** After this script stages generated files, staging is expected but must stay allowlisted. */
async function assertOnlyAllowedStatus(runner: CommandRunner, root: string): Promise<void> {
  const entries = await readStatus(runner, root);
  for (const entry of entries) {
    if (entry.index === "!" && entry.worktree === "!") continue;
    await assertAllowedPaths(root, entry.paths, "dirty");
  }
}

function parseResolvedRemoteUrls(value: string, label: string): string[] {
  if (value.includes("\0") || !value.endsWith("\n")) {
    throw new PublishError(`Malformed ${label} output`);
  }
  const values = value.slice(0, -1).split("\n");
  if (values.length === 0 || values.some((url) => url.length === 0 || url.includes("\r"))) {
    throw new PublishError(`Malformed ${label} output`);
  }
  return values;
}

async function readResolvedRemoteUrls(
  runner: CommandRunner,
  root: string,
  args: readonly string[],
  label: string,
): Promise<string[]> {
  return parseResolvedRemoteUrls(
    (await requireCommand(runner, root, "git", args)).stdout,
    label,
  );
}

async function assertRepository(runner: CommandRunner, root: string): Promise<void> {
  const topLevel = trimOutput((await requireCommand(runner, root, "git", ["rev-parse", "--show-toplevel"])).stdout);
  let physicalTopLevel: string;
  try {
    physicalTopLevel = await physicalDirectory(topLevel, "Git repository root");
  } catch (error) {
    if (error instanceof PublishError) throw error;
    throw new PublishError("Git repository root cannot be resolved physically");
  }
  if (physicalTopLevel !== root) throw new PublishError("Publication must run from the website repository root");

  const branch = await runner("git", ["symbolic-ref", "--quiet", "--short", "HEAD"], { cwd: root });
  if (branch.code !== 0 || trimOutput(branch.stdout) !== "main") {
    throw new PublishError("Publication must run on main with a non-detached HEAD");
  }

  const remotes = trimOutput((await requireCommand(runner, root, "git", ["remote"])).stdout)
    .split(/\r?\n/)
    .filter(Boolean);
  if (remotes.length === 0) throw new PublishError("No origin remote is configured");
  if (remotes.length !== 1 || remotes[0] !== "origin") {
    throw new PublishError("Publication requires exactly one remote named origin");
  }
  const fetchUrls = await readResolvedRemoteUrls(
    runner,
    root,
    ["remote", "get-url", "--all", "origin"],
    "origin fetch URL",
  );
  if (fetchUrls.length !== 1) {
    throw new PublishError("Origin requires exactly one fetch URL");
  }
  if (!EXPECTED_ORIGINS.has(fetchUrls[0]!)) {
    throw new PublishError(`Unexpected origin remote: ${JSON.stringify(fetchUrls[0])}`);
  }

  // Git resolves pushurl fallback and url.* rewrite rules for this query.
  const effectivePushUrls = await readResolvedRemoteUrls(
    runner,
    root,
    ["remote", "get-url", "--push", "--all", "origin"],
    "origin push URL",
  );
  for (const pushUrl of effectivePushUrls) {
    if (!EXPECTED_ORIGINS.has(pushUrl)) {
      throw new PublishError(`Unexpected origin push URL: ${JSON.stringify(pushUrl)}`);
    }
  }
}

export function resolveConfiguredSourceRoot(sourceRoot?: string): string {
  const configured = sourceRoot || process.env.AI_DAILY_SOURCE;
  if (!configured) {
    throw new PublishError(
      "AI_DAILY_SOURCE is required and must point to the read-only AI Daily repository",
    );
  }
  return configured;
}

async function assertSourceIsolation(root: string, sourceRoot?: string): Promise<void> {
  const configuredSource = resolveConfiguredSourceRoot(sourceRoot);
  try {
    assertDisjointRoots(root, await physicalDirectory(configuredSource, "AI_DAILY_SOURCE"));
  } catch (error) {
    if (error instanceof PublishError) throw error;
    throw new PublishError(error instanceof Error ? error.message : "Website and AI Daily source roots must be physically separate");
  }
}

async function runGates(runner: CommandRunner, root: string): Promise<void> {
  const gates: Array<[string, string[]]> = [
    ["npm", ["test"]],
    ["npx", ["tsc", "--noEmit"]],
    ["npm", ["run", "build"]],
    ["npm", ["run", "links:check"]],
    ["npm", ["run", "test:e2e"]],
  ];
  for (const [command, args] of gates) {
    const result = await runner(command, args, { cwd: root });
    if (result.code !== 0) throw new PublishError(failureMessage("Gate failed", command, args, result));
  }
}

async function stagedPaths(runner: CommandRunner, root: string, command: readonly string[]): Promise<string[]> {
  const result = await requireCommand(runner, root, "git", command);
  const values = result.stdout.split("\0");
  if (values.at(-1) === "") values.pop();
  return values;
}

export async function runPublish(options: PublishOptions = {}): Promise<PublishResult> {
  const root = await physicalDirectory(options.websiteRoot ?? DEFAULT_WEBSITE_ROOT, "Website root");
  const runner = options.runner ?? defaultRunner;
  const logger = options.logger ?? console.log;

  await assertSourceIsolation(root, options.sourceRoot);
  await assertRepository(runner, root);
  await assertSafeStatus(runner, root);
  await runGates(runner, root);

  // Validation may create generated output, so every safety condition is checked again.
  await assertRepository(runner, root);
  await assertSafeStatus(runner, root);
  await requireCommand(runner, root, "git", ["add", "--", ...ALLOWED_PATHS]);
  const staged = await stagedPaths(runner, root, ["diff", "--cached", "--name-only", "-z"]);
  await assertAllowedPaths(root, staged, "staged");
  if (staged.length === 0) {
    logger("No generated changes to publish; no commit was created.");
    return { committed: false, changedPaths: [] };
  }

  await assertRepository(runner, root);
  await assertOnlyAllowedStatus(runner, root);
  await requireCommand(runner, root, "git", [
    "commit",
    "--only",
    "-m",
    `chore: publish AI Daily sync (${staged.length} generated files)`,
    "--",
    ...ALLOWED_PATHS,
  ]);

  const committed = await stagedPaths(runner, root, ["show", "--format=", "--name-only", "-z", "HEAD"]);
  await assertAllowedPaths(root, committed, "committed");
  if (committed.length === 0) throw new PublishError("Publication commit contains no generated paths");

  await assertRepository(runner, root);
  await assertSafeStatus(runner, root);
  const push = await runner("git", ["push", "origin", "main"], { cwd: root });
  if (push.code !== 0) {
    const recovery = `${failureMessage("Push failed", "git", ["push", "origin", "main"], push)}. Your local generated commit is preserved. Retry exactly: git push origin main`;
    logger(recovery);
    throw new PublishError(recovery);
  }
  logger(`Published ${committed.length} generated path${committed.length === 1 ? "" : "s"}.`);
  return { committed: true, changedPaths: committed };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  runPublish().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
