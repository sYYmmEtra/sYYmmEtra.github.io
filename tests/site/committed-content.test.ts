import { existsSync } from "node:fs";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { afterEach, describe, expect, it } from "vitest";

const repositoryRoot = path.resolve(import.meta.dirname, "../..");
const validatorPath = path.join(repositoryRoot, "scripts/validate-committed-content.ts");
const temporaryRoots: string[] = [];

type ValidationResult = { lessonCount: number; lessonIds: string[] };
type ValidatorModule = {
  validateCommittedContent: (websiteRoot: string) => Promise<ValidationResult>;
};

async function validator(): Promise<ValidatorModule | undefined> {
  if (!existsSync(validatorPath)) return undefined;
  return import(pathToFileURL(validatorPath).href) as Promise<ValidatorModule>;
}

async function copiedProjection(): Promise<string> {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "committed-content-"));
  temporaryRoots.push(root);
  await fs.cp(path.join(repositoryRoot, "sync-index.json"), path.join(root, "sync-index.json"));
  await fs.mkdir(path.join(root, "metadata"), { recursive: true });
  await fs.mkdir(path.join(root, "src/content"), { recursive: true });
  await fs.cp(path.join(repositoryRoot, "metadata/ai-daily"), path.join(root, "metadata/ai-daily"), { recursive: true });
  await fs.cp(path.join(repositoryRoot, "src/content/ai-daily"), path.join(root, "src/content/ai-daily"), { recursive: true });
  return root;
}

async function requireValidator(): Promise<ValidatorModule> {
  const loaded = await validator();
  expect(loaded).toBeDefined();
  if (!loaded) throw new Error("validator module is unavailable");
  return loaded;
}

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map((root) => fs.rm(root, { recursive: true, force: true })));
});

describe("committed content validation", () => {
  it("accepts the checked-in projection with a data-derived lesson count", async () => {
    const { validateCommittedContent } = await requireValidator();
    const result = await validateCommittedContent(repositoryRoot);

    expect(result.lessonCount).toBeGreaterThan(0);
    expect(result.lessonIds).toHaveLength(result.lessonCount);
  });

  it("rejects a missing committed article with an actionable set error", async () => {
    const { validateCommittedContent } = await requireValidator();
    const root = await copiedProjection();
    await fs.rm(path.join(root, "src/content/ai-daily/lesson-0001.md"));

    await expect(validateCommittedContent(root)).rejects.toThrow("Content IDs differ from sync index");
  });

  it("rejects an unexpected projection entry", async () => {
    const { validateCommittedContent } = await requireValidator();
    const root = await copiedProjection();
    await fs.writeFile(path.join(root, "metadata/ai-daily/unexpected.yml"), "id: lesson-0001\n");

    await expect(validateCommittedContent(root)).rejects.toThrow("Unexpected sidecar entry: unexpected.yml");
  });

  it("rejects mismatched generated frontmatter", async () => {
    const { validateCommittedContent } = await requireValidator();
    const root = await copiedProjection();
    const target = path.join(root, "src/content/ai-daily/lesson-0001.md");
    const content = await fs.readFile(target, "utf8");
    await fs.writeFile(target, content.replace("slug: prompt-engineering-foundations", "slug: wrong-slug"));

    await expect(validateCommittedContent(root)).rejects.toThrow("Generated frontmatter does not match sidecar: lesson-0001");
  });

  it("rejects index assignments that do not match the projection", async () => {
    const { validateCommittedContent } = await requireValidator();
    const root = await copiedProjection();
    const target = path.join(root, "sync-index.json");
    const index = JSON.parse(await fs.readFile(target, "utf8"));
    index.lessons[0].slug = "wrong-slug";
    await fs.writeFile(target, `${JSON.stringify(index, null, 2)}\n`);

    await expect(validateCommittedContent(root)).rejects.toThrow("Sync index assignment does not match sidecar: lesson-0001");
  });

  it("rejects an empty corpus with a validator-specific error", async () => {
    const { validateCommittedContent } = await requireValidator();
    const root = await copiedProjection();
    await fs.writeFile(path.join(root, "sync-index.json"), '{"schemaVersion":1,"lessons":[]}\n');
    await fs.rm(path.join(root, "metadata/ai-daily"), { recursive: true });
    await fs.rm(path.join(root, "src/content/ai-daily"), { recursive: true });
    await fs.mkdir(path.join(root, "metadata/ai-daily"), { recursive: true });
    await fs.mkdir(path.join(root, "src/content/ai-daily"), { recursive: true });

    await expect(validateCommittedContent(root)).rejects.toThrow("Committed content index must contain at least one lesson");
  });

  it("rejects duplicate sync index slugs without changing sync schema semantics", async () => {
    const { validateCommittedContent } = await requireValidator();
    const root = await copiedProjection();
    const target = path.join(root, "sync-index.json");
    const index = JSON.parse(await fs.readFile(target, "utf8"));
    index.lessons[1].slug = index.lessons[0].slug;
    await fs.writeFile(target, `${JSON.stringify(index, null, 2)}\n`);

    await expect(validateCommittedContent(root)).rejects.toThrow("Duplicate committed sync index slug");
  });

  it("rejects symlinked committed projection entries", async () => {
    const { validateCommittedContent } = await requireValidator();
    const root = await copiedProjection();
    const target = path.join(root, "metadata/ai-daily/lesson-0001.yml");
    const replacement = path.join(root, "sidecar.yml");
    await fs.rename(target, replacement);
    await fs.symlink(replacement, target);

    await expect(validateCommittedContent(root)).rejects.toThrow("Unexpected sidecar entry: lesson-0001.yml");
  });
});
