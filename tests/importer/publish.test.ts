import { execFile } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  PublishError,
  parsePorcelainStatus,
  resolveConfiguredSourceRoot,
  runPublish,
  type CommandRunner,
} from "../../scripts/publish";

const execFileAsync = promisify(execFile);
const temporaryRoots: string[] = [];
const gitEnvironments = new Map<string, NodeJS.ProcessEnv>();
let previousSourceRoot: string | undefined;
let testSourceRoot = "";

const EXPECTED_REMOTE = "https://github.com/sYYmmEtra/sYYmmEtra.github.io.git";
const ATTACKER_REMOTE = "https://github.com/attacker/sYYmmEtra.github.io.git";

async function makeRepository(): Promise<string> {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "blog-publish-"));
  temporaryRoots.push(root);
  const home = path.join(root, "home");
  const xdgConfig = path.join(root, "xdg-config");
  await fs.mkdir(home);
  await fs.mkdir(xdgConfig);
  const environment: NodeJS.ProcessEnv = { ...process.env };
  for (const key of Object.keys(environment)) {
    if (
      key === "GIT_CONFIG_GLOBAL" ||
      key === "GIT_CONFIG_SYSTEM" ||
      key === "GIT_CONFIG_COUNT" ||
      key === "GIT_CONFIG_PARAMETERS" ||
      /^GIT_CONFIG_(?:KEY|VALUE)_\d+$/.test(key)
    ) {
      delete environment[key];
    }
  }
  environment.GIT_CONFIG_NOSYSTEM = "1";
  environment.HOME = home;
  environment.XDG_CONFIG_HOME = xdgConfig;
  await execFileAsync("git", ["init", "-q", root], { env: environment });
  const physicalRoot = await fs.realpath(root);
  gitEnvironments.set(physicalRoot, environment);
  return physicalRoot;
}

function gitEnvironment(root: string): NodeJS.ProcessEnv {
  const environment = gitEnvironments.get(root);
  if (!environment) throw new Error(`Missing isolated Git environment for ${root}`);
  return environment;
}

function nul(paths: string[]): string {
  return paths.join("\0") + (paths.length ? "\0" : "");
}

interface FakeState {
  branch?: string;
  remote?: string;
  fetchUrls?: string[];
  pushUrls?: string[];
  remotes?: string;
  statuses?: string[];
  staged?: string[];
  gateFailure?: string;
  gateStderr?: string;
  gateStdout?: string;
  pushFailure?: boolean;
  pushStderr?: string;
  pushStdout?: string;
}

function fakeRunner(root: string, state: FakeState = {}): {
  runner: CommandRunner;
  calls: Array<{ command: string; args: readonly string[] }>;
} {
  const calls: Array<{ command: string; args: readonly string[] }> = [];
  let statusCall = 0;
  const runner: CommandRunner = async (command, args, options) => {
    expect(options.cwd).toBe(root);
    calls.push({ command, args });
    const key = `${command} ${args.join(" ")}`;
    if (key === "git rev-parse --show-toplevel") return { code: 0, stdout: `${root}\n`, stderr: "" };
    if (key === "git symbolic-ref --quiet --short HEAD") {
      return { code: state.branch === undefined || state.branch === "main" ? 0 : 1, stdout: `${state.branch ?? "main"}\n`, stderr: "" };
    }
    if (key === "git remote") return { code: 0, stdout: state.remotes ?? "origin\n", stderr: "" };
    if (key === "git remote get-url --all origin") {
      return { code: 0, stdout: `${(state.fetchUrls ?? [state.remote ?? EXPECTED_REMOTE]).join("\n")}\n`, stderr: "" };
    }
    if (key === "git remote get-url --push --all origin") {
      const effectivePushUrls = state.pushUrls && state.pushUrls.length > 0
        ? state.pushUrls
        : state.fetchUrls ?? [state.remote ?? EXPECTED_REMOTE];
      return { code: 0, stdout: `${effectivePushUrls.join("\n")}\n`, stderr: "" };
    }
    if (key === "git status --porcelain=v1 -z") {
      const statuses = state.statuses ?? ["", "", "", ""];
      return { code: 0, stdout: statuses[Math.min(statusCall++, statuses.length - 1)]!, stderr: "" };
    }
    if (key === "git diff --cached --name-only -z") return { code: 0, stdout: nul(state.staged ?? ["metadata/ai-daily/lesson-0001.yml", "src/content/ai-daily/lesson-0001.md", "sync-index.json"]), stderr: "" };
    if (key === "git show --format= --name-only -z HEAD") return { code: 0, stdout: nul(state.staged ?? ["metadata/ai-daily/lesson-0001.yml", "src/content/ai-daily/lesson-0001.md", "sync-index.json"]), stderr: "" };
    if (key === "git push origin main") return { code: state.pushFailure ? 1 : 0, stdout: state.pushStdout ?? "", stderr: state.pushFailure ? state.pushStderr ?? "denied" : "" };
    if (state.gateFailure && key === state.gateFailure) return { code: 1, stdout: state.gateStdout ?? "", stderr: state.gateStderr ?? "failed gate" };
    return { code: 0, stdout: "", stderr: "" };
  };
  return { runner, calls };
}

function effectiveUrlRunner(root: string, state: FakeState = {}): {
  runner: CommandRunner;
  calls: Array<{ command: string; args: readonly string[] }>;
} {
  const fake = fakeRunner(root, state);
  const runner: CommandRunner = async (command, args, options) => {
    if (
      command === "git" &&
      (args.join(" ") === "remote get-url --all origin" ||
        args.join(" ") === "remote get-url --push --all origin")
    ) {
      fake.calls.push({ command, args });
      try {
        const result = await execFileAsync(command, [...args], { cwd: options.cwd, encoding: "utf8", env: gitEnvironment(root) });
        return { code: 0, stdout: result.stdout, stderr: result.stderr };
      } catch (error) {
        const result = error as { code?: number; stdout?: string; stderr?: string };
        return { code: result.code ?? 1, stdout: result.stdout ?? "", stderr: result.stderr ?? "" };
      }
    }
    return fake.runner(command, args, options);
  };
  return { runner, calls: fake.calls };
}

beforeEach(async () => {
  previousSourceRoot = process.env.AI_DAILY_SOURCE;
  testSourceRoot = await fs.mkdtemp(path.join(os.tmpdir(), "blog-publish-source-"));
  temporaryRoots.push(testSourceRoot);
  process.env.AI_DAILY_SOURCE = testSourceRoot;
});

afterEach(async () => {
  if (previousSourceRoot === undefined) delete process.env.AI_DAILY_SOURCE;
  else process.env.AI_DAILY_SOURCE = previousSourceRoot;
  await Promise.all(temporaryRoots.splice(0).map((root) => fs.rm(root, { recursive: true, force: true })));
  gitEnvironments.clear();
});

describe("parsePorcelainStatus", () => {
  it("preserves spaces, newlines, and both names of a rename from NUL status output", () => {
    const parsed = parsePorcelainStatus("R  metadata/ai-daily/new name.yml\0metadata/ai-daily/old\nname.yml\0?? src/content/ai-daily/new file.md\0");
    expect(parsed).toEqual([
      { index: "R", worktree: " ", paths: ["metadata/ai-daily/new name.yml", "metadata/ai-daily/old\nname.yml"] },
      { index: "?", worktree: "?", paths: ["src/content/ai-daily/new file.md"] },
    ]);
  });
});

describe("runPublish", () => {
  it("removes inherited Git configuration injection variables from temporary repos", async () => {
    const keys = ["GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0", "GIT_CONFIG_PARAMETERS"];
    const previous = Object.fromEntries(keys.map((key) => [key, process.env[key]]));
    Object.assign(process.env, {
      GIT_CONFIG_GLOBAL: "/tmp/untrusted-global",
      GIT_CONFIG_SYSTEM: "/tmp/untrusted-system",
      GIT_CONFIG_COUNT: "1",
      GIT_CONFIG_KEY_0: "url.https://attacker.invalid/.insteadOf",
      GIT_CONFIG_VALUE_0: EXPECTED_REMOTE,
      GIT_CONFIG_PARAMETERS: "'user.name=Injected User' 'url.https://attacker.invalid/.insteadOf=https://github.com/sYYmmEtra/sYYmmEtra.github.io.git'",
    });
    try {
      const root = await makeRepository();
      const environment = gitEnvironment(root);
      for (const key of keys) expect(environment[key]).toBeUndefined();
      expect(Object.keys(environment).some((key) => /^GIT_CONFIG_(?:KEY|VALUE)_\d+$/.test(key))).toBe(false);
      expect(environment.GIT_CONFIG_NOSYSTEM).toBe("1");
      await execFileAsync("git", ["-C", root, "remote", "add", "origin", EXPECTED_REMOTE], { env: environment });
      const resolved = await execFileAsync("git", ["-C", root, "remote", "get-url", "--all", "origin"], { env: environment, encoding: "utf8" });
      expect(resolved.stdout).toBe(`${EXPECTED_REMOTE}\n`);
      await expect(execFileAsync("git", ["-C", root, "config", "--get", "user.name"], { env: environment })).rejects.toMatchObject({ code: 1 });
    } finally {
      for (const key of keys) {
        if (previous[key] === undefined) delete process.env[key];
        else process.env[key] = previous[key];
      }
    }
  });

  it("requires AI_DAILY_SOURCE when no source root is provided", () => {
    delete process.env.AI_DAILY_SOURCE;
    expect(() => resolveConfiguredSourceRoot()).toThrow(
      "AI_DAILY_SOURCE is required and must point to the read-only AI Daily repository",
    );
  });
  it.each([
    ["an unexpected remote URL", { remote: "https://github.com/sYYmmEtra/sYYmmEtra.github.io.evil.git" }, /Unexpected origin remote/],
    ["a trailing-slash remote URL", { remote: `${EXPECTED_REMOTE}/` }, /Unexpected origin remote/],
    ["multiple fetch URLs", { fetchUrls: [EXPECTED_REMOTE, EXPECTED_REMOTE] }, /exactly one fetch URL/],
    ["a malicious configured push URL", { pushUrls: ["https://github.com/sYYmmEtra/sYYmmEtra.github.io.evil.git"] }, /Unexpected origin push URL/],
    ["mixed allowed and malicious push URLs", { pushUrls: [EXPECTED_REMOTE, "git@github.com:attacker/sYYmmEtra.github.io.git"] }, /Unexpected origin push URL/],
    ["no origin remote", { remotes: "" }, /No origin remote/],
    ["multiple remotes", { remotes: "origin\nupstream\n" }, /exactly one remote/],
    ["a non-main branch", { branch: "feature/sync" }, /must run on main/],
    ["an unrelated dirty path", { statuses: [nul(["?? src/pages/index.astro"])] }, /Unexpected dirty path/],
  ])("rejects %s before publication", async (_name, state, message) => {
    const root = await makeRepository();
    const { runner, calls } = fakeRunner(root, state);
    await expect(runPublish({ websiteRoot: root, runner })).rejects.toThrow(message);
    expect(calls.some((call) => call.args[0] === "add" || call.args[0] === "commit" || call.args[0] === "push")).toBe(false);
  });

  it("rejects pre-existing staging even when it is an allowed generated path", async () => {
    const root = await makeRepository();
    const { runner } = fakeRunner(root, { statuses: [nul(["M  metadata/ai-daily/lesson-0001.yml"])] });
    await expect(runPublish({ websiteRoot: root, runner })).rejects.toThrow(/pre-existing staged changes/);
  });

  it("refuses a website root that overlaps the configured AI Daily source through an alias", async () => {
    const root = await makeRepository();
    const alias = path.join(root, "source-alias");
    await fs.symlink(root, alias);
    const previous = process.env.AI_DAILY_SOURCE;
    process.env.AI_DAILY_SOURCE = alias;
    try {
      const { runner } = fakeRunner(root);
      await expect(runPublish({ websiteRoot: root, runner })).rejects.toThrow(/physically separate/);
    } finally {
      if (previous === undefined) delete process.env.AI_DAILY_SOURCE;
      else process.env.AI_DAILY_SOURCE = previous;
    }
  });

  it.each([
    ["insteadOf", "insteadOf"],
    ["pushInsteadOf", "pushInsteadOf"],
  ])("rejects an allowed raw origin rewritten by url.%s", async (_name, rewriteKey) => {
    const root = await makeRepository();
    await execFileAsync("git", ["-C", root, "remote", "add", "origin", EXPECTED_REMOTE], { env: gitEnvironment(root) });
    await execFileAsync("git", ["-C", root, "config", `url.${ATTACKER_REMOTE}.${rewriteKey}`, EXPECTED_REMOTE], { env: gitEnvironment(root) });
    const { runner, calls } = effectiveUrlRunner(root);
    await expect(runPublish({ websiteRoot: root, runner })).rejects.toThrow(/Unexpected origin (remote|push URL)/);
    expect(calls.some((call) => call.command === "npm" || call.args[0] === "add" || call.args[0] === "push")).toBe(false);
  });

  it("runs gates before explicitly staging only generated projections and commits their count", async () => {
    const root = await makeRepository();
    const { runner, calls } = fakeRunner(root);
    await runPublish({ websiteRoot: root, runner });
    const add = calls.find((call) => call.command === "git" && call.args[0] === "add");
    expect(add?.args).toEqual(["add", "--", "metadata/ai-daily", "src/content/ai-daily", "sync-index.json"]);
    const addIndex = calls.findIndex((call) => call === add);
    const firstGateIndex = calls.findIndex((call) => call.command === "npm" && call.args[0] === "test");
    expect(firstGateIndex).toBeGreaterThan(-1);
    expect(addIndex).toBeGreaterThan(firstGateIndex);
    expect(calls.find((call) => call.command === "git" && call.args[0] === "commit")?.args.join(" ")).toContain("3 generated files");
    expect(calls.find((call) => call.command === "git" && call.args[0] === "push")?.args).toEqual(["push", "origin", "main"]);
    expect(calls.filter((call) => call.args.join(" ") === "remote get-url --all origin")).toHaveLength(4);
    expect(calls.filter((call) => call.args.join(" ") === "remote get-url --push --all origin")).toHaveLength(4);
  });

  it("allows its own allowlisted staging while still checking it before commit", async () => {
    const root = await makeRepository();
    const stagedStatus = nul([
      "M  metadata/ai-daily/lesson-0001.yml",
      "M  src/content/ai-daily/lesson-0001.md",
      "M  sync-index.json",
    ]);
    const { runner, calls } = fakeRunner(root, { statuses: ["", "", stagedStatus, ""] });
    await runPublish({ websiteRoot: root, runner });
    expect(calls.some((call) => call.command === "git" && call.args[0] === "commit")).toBe(true);
  });

  it("does not commit or push when the generated staged diff is empty", async () => {
    const root = await makeRepository();
    const { runner, calls } = fakeRunner(root, { staged: [] });
    const result = await runPublish({ websiteRoot: root, runner });
    expect(result).toMatchObject({ committed: false, changedPaths: [] });
    expect(calls.some((call) => call.args[0] === "commit" || call.args[0] === "push")).toBe(false);
  });

  it("accepts a deletion within the generated projection allowlist", async () => {
    const root = await makeRepository();
    const { runner, calls } = fakeRunner(root, {
      statuses: [nul([" D src/content/ai-daily/lesson-0001.md"])],
      staged: ["src/content/ai-daily/lesson-0001.md"],
    });
    await runPublish({ websiteRoot: root, runner });
    expect(calls.find((call) => call.command === "git" && call.args[0] === "add")?.args).toEqual([
      "add", "--", "metadata/ai-daily", "src/content/ai-daily", "sync-index.json",
    ]);
    expect(calls.some((call) => call.command === "git" && call.args[0] === "push")).toBe(true);
  });

  it.each([
    "npm test",
    "npx tsc --noEmit",
    "npm run build",
    "npm run links:check",
    "npm run test:e2e",
  ])("stops before staging when the %s gate fails", async (gateFailure) => {
    const root = await makeRepository();
    const { runner, calls } = fakeRunner(root, { gateFailure });
    await expect(runPublish({ websiteRoot: root, runner })).rejects.toThrow(`Gate failed: ${gateFailure}: failed gate`);
    expect(calls.some((call) => call.args[0] === "add" || call.args[0] === "commit" || call.args[0] === "push")).toBe(false);
  });

  it("includes bounded sanitized gate stderr and falls back to stdout", async () => {
    const root = await makeRepository();
    const { runner } = fakeRunner(root, { gateFailure: "npm test", gateStderr: "bad\0stderr" });
    await expect(runPublish({ websiteRoot: root, runner })).rejects.toThrow("Gate failed: npm test: bad?stderr");
    const fallback = fakeRunner(root, { gateFailure: "npm test", gateStderr: "", gateStdout: "stdout detail" });
    await expect(runPublish({ websiteRoot: root, runner: fallback.runner })).rejects.toThrow("Gate failed: npm test: stdout detail");
    const truncated = fakeRunner(root, { gateFailure: "npm test", gateStderr: "x".repeat(5000) });
    const error = await runPublish({ websiteRoot: root, runner: truncated.runner }).catch((failure) => failure);
    expect(error).toBeInstanceOf(PublishError);
    expect(error.message).toContain("[truncated]");
    expect(error.message.length).toBeLessThan(4_300);
  });

  it("preserves the local commit on push failure and logs the exact retry command", async () => {
    const root = await makeRepository();
    const messages: string[] = [];
    const { runner, calls } = fakeRunner(root, { pushFailure: true, pushStderr: "remote denied\0credential" });
    await expect(runPublish({ websiteRoot: root, runner, logger: (message) => messages.push(message) })).rejects.toThrow("Push failed: git push origin main: remote denied?credential");
    expect(calls.some((call) => call.args[0] === "commit")).toBe(true);
    expect(calls.some((call) => call.args[0] === "reset" || call.args[0] === "revert" || call.args[0] === "amend")).toBe(false);
    expect(messages).toContain("Push failed: git push origin main: remote denied?credential. Your local generated commit is preserved. Retry exactly: git push origin main");
  });
});
