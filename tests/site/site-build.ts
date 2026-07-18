import { execFileSync } from "node:child_process";
import { readFileSync, mkdtempSync, rmSync } from "node:fs";
import path from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath, pathToFileURL } from "node:url";

export const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

type CommandRunner = (file: string, args: string[], options: { cwd: string; stdio: "pipe" }) => unknown;

export function buildSite(
  prefix: string,
  runCommand: CommandRunner = execFileSync as unknown as CommandRunner,
): { outputDirectory: string; dispose: () => void } {
  const outputDirectory = mkdtempSync(path.join(tmpdir(), prefix));
  try {
    runCommand(
      process.execPath,
      [path.join(repoRoot, "node_modules/astro/bin/astro.mjs"), "build", "--outDir", outputDirectory],
      { cwd: repoRoot, stdio: "pipe" },
    );
  } catch (error) {
    rmSync(outputDirectory, { recursive: true, force: true });
    throw error;
  }
  return { outputDirectory, dispose: () => rmSync(outputDirectory, { recursive: true, force: true }) };
}

export function buildPagefind(outputDirectory: string): void {
  execFileSync(
    path.join(repoRoot, "node_modules/.bin/pagefind"),
    ["--site", outputDirectory, "--force-language", "zh"],
    { cwd: repoRoot, stdio: "pipe" },
  );
}

export function pagefindLanguages(outputDirectory: string): Record<string, { page_count: number }> {
  return JSON.parse(
    readFileSync(path.join(outputDirectory, "pagefind/pagefind-entry.json"), "utf8"),
  ).languages;
}

export async function withPagefindFetch<T>(outputDirectory: string, action: () => Promise<T>): Promise<T> {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const requestUrl = new URL(
      typeof input === "string" ? input : input instanceof URL ? input.href : input.url,
    );
    if (requestUrl.origin !== "https://pagefind.test") return originalFetch(input);
    const relativePath = decodeURIComponent(requestUrl.pathname).replace(/^\/+/, "");
    if (relativePath.split("/").includes("..")) return new Response("Not found", { status: 404 });
    return new Response(readFileSync(path.join(outputDirectory, relativePath)), { status: 200 });
  };

  try {
    return await action();
  } finally {
    globalThis.fetch = originalFetch;
  }
}

export async function searchPagefind(outputDirectory: string, query: string): Promise<string[]> {
  return withPagefindFetch(outputDirectory, async () => {
    const moduleUrl = `${pathToFileURL(path.join(outputDirectory, "pagefind/pagefind.js")).href}?test=${Date.now()}`;
    const pagefind = await import(moduleUrl);
    try {
    await pagefind.options({ basePath: "https://pagefind.test/pagefind/", baseUrl: "https://pagefind.test/" });
    await pagefind.init();
    const result = await pagefind.search(query);
    return Promise.all(result.results.map(async (entry: { data: () => Promise<{ url: string }> }) => (await entry.data()).url));
    } finally {
      await pagefind.destroy();
    }
  });
}
