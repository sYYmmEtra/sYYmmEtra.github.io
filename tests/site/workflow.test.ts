import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { parse } from "yaml";
import { describe, expect, it } from "vitest";

const repositoryRoot = path.resolve(import.meta.dirname, "../..");
const workflowPath = path.join(repositoryRoot, ".github/workflows/deploy.yml");
const packagePath = path.join(repositoryRoot, "package.json");

type Workflow = Record<string, unknown>;

function workflow(): Workflow | undefined {
  if (!existsSync(workflowPath)) return undefined;
  return parse(readFileSync(workflowPath, "utf8")) as Workflow;
}

function jobs(document: Workflow): Record<string, Workflow> {
  return document.jobs as Record<string, Workflow>;
}

function runSteps(job: Workflow): string[] {
  return ((job.steps as Workflow[] | undefined) ?? [])
    .map((step) => step.run)
    .filter((run): run is string => typeof run === "string");
}

function uses(job: Workflow): string[] {
  return ((job.steps as Workflow[] | undefined) ?? [])
    .map((step) => step.uses)
    .filter((action): action is string => typeof action === "string");
}

describe("GitHub Pages deployment workflow", () => {
  it("deploys only main pushes and manual runs", () => {
    const document = workflow();
    expect(document).toBeDefined();
    if (!document) return;

    // YAML 1.1 parsers can coerce `on` to a boolean, so accept both parsed forms.
    const triggers = (document.on ?? document.true) as Workflow;
    expect(triggers.push).toEqual({ branches: ["main"] });
    expect(triggers.workflow_dispatch).toEqual({});
    expect(triggers.pull_request).toBeUndefined();
    expect(document.concurrency).toEqual({ group: "pages", "cancel-in-progress": false });
  });

  it("uses only official pinned-major Pages actions with least-privilege jobs", () => {
    const document = workflow();
    expect(document).toBeDefined();
    if (!document) return;

    expect(document.permissions).toEqual({ contents: "read" });
    const build = jobs(document).build;
    const deploy = jobs(document).deploy;
    expect(build["runs-on"]).toBe("ubuntu-latest");
    expect(uses(build)).toEqual(expect.arrayContaining([
      "actions/checkout@v5",
      "actions/setup-node@v5",
      "actions/configure-pages@v5",
      "actions/upload-pages-artifact@v4",
    ]));
    expect(deploy.needs).toBe("build");
    expect(deploy["runs-on"]).toBe("ubuntu-latest");
    expect(deploy.permissions).toEqual({ pages: "write", "id-token": "write" });
    expect(deploy.environment).toEqual({ name: "github-pages", url: "${{ steps.deployment.outputs.page_url }}" });
    expect(uses(deploy)).toEqual(["actions/deploy-pages@v4"]);
    expect((deploy.steps as Workflow[]).find((step) => step.uses === "actions/deploy-pages@v4")?.id).toBe("deployment");
    expect([...uses(build), ...uses(deploy)].every((action) => action.startsWith("actions/"))).toBe(true);
  });

  it("installs, tests, type-checks, builds, and uploads the generated site", () => {
    const document = workflow();
    expect(document).toBeDefined();
    if (!document) return;

    const build = jobs(document).build;
    const setupNode = (build.steps as Workflow[]).find((step) => step.uses === "actions/setup-node@v5");
    const upload = (build.steps as Workflow[]).find((step) => step.uses === "actions/upload-pages-artifact@v4");
    const commands = runSteps(build).join("\n");
    expect(setupNode?.with).toEqual({ "node-version": "24", cache: "npm" });
    expect(upload?.with).toEqual({ path: "./dist" });
    for (const command of ["npm ci", "npm test", "npx tsc --noEmit", "npm run content:validate", "npm run build"]) {
      expect(commands).toContain(command);
    }
    expect(commands.match(/\bnpm run build\b/g)).toHaveLength(1);
    const packageJson = JSON.parse(readFileSync(packagePath, "utf8")) as { scripts: Record<string, string> };
    expect(packageJson.scripts.build).toContain("npm run search:index");
    expect(packageJson.scripts["search:index"]).toContain("pagefind");
    expect(packageJson.scripts["content:validate"]).toBe("tsx scripts/validate-committed-content.ts");
    expect(commands.indexOf("npm test")).toBeLessThan(commands.indexOf("npx tsc --noEmit"));
    expect(commands.indexOf("npx tsc --noEmit")).toBeLessThan(commands.indexOf("npm run content:validate"));
    expect(commands.indexOf("npm run content:validate")).toBeLessThan(commands.indexOf("npm run build"));
  });

  it("validates committed output with local smoke checks and network-free internal links", () => {
    const document = workflow();
    expect(document).toBeDefined();
    if (!document) return;

    const commands = runSteps(jobs(document).build).join("\n");
    for (const output of ["dist/index.html", "dist/ai-daily/index.html", "dist/zh/ai-daily/index.html", "dist/rss.xml", "dist/sitemap-index.xml", "dist/pagefind/pagefind.js"]) {
      expect(commands).toContain(output);
    }
    expect(commands).toMatch(/find dist\/ai-daily/);
    expect(commands).toMatch(/find dist\/zh\/ai-daily/);
    expect(commands).toContain("sync-index.json");
    expect(commands).not.toMatch(/(?:article_count|lesson_count).*13|13.*(?:article_count|lesson_count)/);
    expect(commands).toContain('import { LinkChecker, LinkState } from "linkinator"');
    expect(commands).toContain("checkFragments: true");
    expect(commands).not.toMatch(/\bport\s*:/);
    expect(commands).toContain('server.listen(0, "127.0.0.1")');
    expect(commands).toContain("candidate.origin !== internalOrigin");
    expect(commands).toContain("linksToSkip: (url) =>");
    expect(commands).toContain("LinkState.BROKEN");
    expect(commands).toContain("set -euo pipefail");
  });

  it("never invokes generation, synchronization, enrichment, or an external source", () => {
    const document = workflow();
    expect(document).toBeDefined();
    if (!document) return;

    const yaml = readFileSync(workflowPath, "utf8");
    expect(yaml).not.toMatch(/\b(?:claude|codex|openai|model)\b/i);
    expect(yaml).not.toMatch(/AI_DAILY_SOURCE|npm run sync|sync-ai-daily|metadata:enrich|enrich-metadata|\.\.\/ai-daily|\bsource\b/i);
  });
});
