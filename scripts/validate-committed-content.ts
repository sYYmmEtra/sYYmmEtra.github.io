import { constants as fsConstants } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import YAML from "yaml";

import { SidecarMetadataSchema, type SidecarMetadata } from "./lib/metadata";
import {
  SyncIndexSchema,
  type SyncIndex,
  type SyncIndexLesson,
} from "./lib/sync-index";

const SCRIPT_DIRECTORY = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_WEBSITE_ROOT = path.resolve(SCRIPT_DIRECTORY, "..");
const SIDECAR_FILE = /^(lesson-[0-9]{4})\.yml$/;
const CONTENT_FILE = /^(lesson-[0-9]{4})\.md$/;

export interface CommittedContentValidationResult {
  lessonCount: number;
  lessonIds: string[];
}

function isDescendant(root: string, target: string): boolean {
  const relative = path.relative(root, target);
  return relative !== "" && relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

function assertPathInsideRoot(root: string, target: string, label: string): void {
  if (!isDescendant(root, target)) {
    throw new Error(`${label} is outside the website root`);
  }
}

async function assertPhysicalDirectory(target: string, label: string): Promise<void> {
  const stat = await fs.lstat(target);
  if (stat.isSymbolicLink() || !stat.isDirectory()) {
    throw new Error(`${label} must be a physical directory`);
  }
}

async function projectionDirectory(
  websiteRoot: string,
  physicalRoot: string,
  relative: string,
  label: string,
): Promise<string> {
  let current = websiteRoot;
  for (const part of relative.split("/")) {
    current = path.join(current, part);
    await assertPhysicalDirectory(current, `${label} path`);
  }
  assertPathInsideRoot(physicalRoot, await fs.realpath(current), label);
  return current;
}

async function readPhysicalText(
  physicalRoot: string,
  target: string,
  label: string,
): Promise<string> {
  const stat = await fs.lstat(target);
  if (stat.isSymbolicLink() || !stat.isFile()) {
    throw new Error(`${label} must be a physical regular file`);
  }
  assertPathInsideRoot(physicalRoot, await fs.realpath(target), label);

  const handle = await fs.open(target, fsConstants.O_RDONLY | fsConstants.O_NOFOLLOW);
  try {
    if (!(await handle.stat()).isFile()) {
      throw new Error(`${label} must be a physical regular file`);
    }
    return await handle.readFile({ encoding: "utf8" });
  } finally {
    await handle.close();
  }
}

async function projectionEntries(
  physicalRoot: string,
  directory: string,
  matcher: RegExp,
  label: string,
): Promise<Array<{ id: string; filename: string; text: string }>> {
  const entries = await fs.readdir(directory, { withFileTypes: true });
  const result: Array<{ id: string; filename: string; text: string }> = [];

  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    const match = matcher.exec(entry.name);
    if (!match || entry.isSymbolicLink() || !entry.isFile()) {
      throw new Error(`Unexpected ${label} entry: ${entry.name}`);
    }
    result.push({
      id: match[1]!,
      filename: entry.name,
      text: await readPhysicalText(physicalRoot, path.join(directory, entry.name), `${label} ${entry.name}`),
    });
  }
  return result;
}

function parseIndex(text: string): SyncIndex {
  let index: SyncIndex;
  try {
    index = SyncIndexSchema.parse(JSON.parse(text));
  } catch (error) {
    throw new Error("Malformed sync-index.json", { cause: error });
  }
  if (index.lessons.length === 0) {
    throw new Error("Committed content index must contain at least one lesson");
  }
  const slugs = new Set<string>();
  for (const lesson of index.lessons) {
    if (slugs.has(lesson.slug)) {
      throw new Error(`Duplicate committed sync index slug: ${lesson.slug}`);
    }
    slugs.add(lesson.slug);
  }
  return index;
}

function parseSidecar(text: string, filename: string): SidecarMetadata {
  try {
    return SidecarMetadataSchema.parse(YAML.parse(text));
  } catch (error) {
    throw new Error(`Malformed sidecar ${filename}`, { cause: error });
  }
}

function canonicalSidecarText(sidecar: SidecarMetadata): string {
  return YAML.stringify(SidecarMetadataSchema.parse(sidecar), { lineWidth: 0 });
}

function parseGeneratedFrontmatter(text: string, id: string): { raw: string; sidecar: SidecarMetadata } {
  if (!text.startsWith("---\n")) {
    throw new Error(`Generated content is missing frontmatter: ${id}`);
  }
  const end = text.indexOf("---\n", 4);
  if (end < 4) {
    throw new Error(`Generated content has unterminated frontmatter: ${id}`);
  }
  const raw = text.slice(4, end);
  try {
    return { raw, sidecar: SidecarMetadataSchema.parse(YAML.parse(raw)) };
  } catch (error) {
    throw new Error(`Malformed generated frontmatter: ${id}`, { cause: error });
  }
}

function assertExactIdSet(label: string, expected: readonly string[], actual: readonly string[]): void {
  if (expected.length !== actual.length || expected.some((id, index) => id !== actual[index])) {
    throw new Error(`${label} IDs differ from sync index`);
  }
}

function assertIndexAssignment(index: SyncIndexLesson, sidecar: SidecarMetadata): void {
  if (
    index.lesson !== sidecar.lesson ||
    index.source.file !== sidecar.source.file ||
    index.source.section !== sidecar.source.section ||
    index.sourceHash !== sidecar.source.hash ||
    index.slug !== sidecar.slug
  ) {
    throw new Error(`Sync index assignment does not match sidecar: ${index.id}`);
  }
}

export async function validateCommittedContent(
  websiteRoot = DEFAULT_WEBSITE_ROOT,
): Promise<CommittedContentValidationResult> {
  const resolvedRoot = path.resolve(websiteRoot);
  await assertPhysicalDirectory(resolvedRoot, "Website root");
  const physicalRoot = await fs.realpath(resolvedRoot);

  const sidecarDirectory = await projectionDirectory(resolvedRoot, physicalRoot, "metadata/ai-daily", "Sidecar directory");
  const contentDirectory = await projectionDirectory(resolvedRoot, physicalRoot, "src/content/ai-daily", "Content directory");
  const indexText = await readPhysicalText(physicalRoot, path.join(resolvedRoot, "sync-index.json"), "sync-index.json");
  const index = parseIndex(indexText);
  const expectedIds = index.lessons.map((lesson) => lesson.id);

  const sidecars = await projectionEntries(physicalRoot, sidecarDirectory, SIDECAR_FILE, "sidecar");
  const sidecarById = new Map<string, SidecarMetadata>();
  for (const entry of sidecars) {
    const sidecar = parseSidecar(entry.text, entry.filename);
    if (sidecar.id !== entry.id) {
      throw new Error(`Sidecar filename ${entry.filename} does not match contained ID ${sidecar.id}`);
    }
    if (sidecarById.has(sidecar.id)) {
      throw new Error(`Duplicate sidecar ID ${sidecar.id}`);
    }
    sidecarById.set(sidecar.id, sidecar);
  }
  assertExactIdSet("Sidecar", expectedIds, [...sidecarById.keys()].sort());

  const content = await projectionEntries(physicalRoot, contentDirectory, CONTENT_FILE, "content");
  const contentById = new Map<string, { raw: string; sidecar: SidecarMetadata }>();
  for (const entry of content) {
    const frontmatter = parseGeneratedFrontmatter(entry.text, entry.id);
    if (frontmatter.sidecar.id !== entry.id) {
      throw new Error(`Content filename ${entry.filename} does not match contained ID ${frontmatter.sidecar.id}`);
    }
    if (contentById.has(entry.id)) {
      throw new Error(`Duplicate content ID ${entry.id}`);
    }
    contentById.set(entry.id, frontmatter);
  }
  assertExactIdSet("Content", expectedIds, [...contentById.keys()].sort());

  for (const lesson of index.lessons) {
    const sidecar = sidecarById.get(lesson.id)!;
    const frontmatter = contentById.get(lesson.id)!;
    assertIndexAssignment(lesson, sidecar);
    if (canonicalSidecarText(frontmatter.sidecar) !== canonicalSidecarText(sidecar) || frontmatter.raw !== canonicalSidecarText(sidecar)) {
      throw new Error(`Generated frontmatter does not match sidecar: ${lesson.id}`);
    }
  }

  return { lessonCount: expectedIds.length, lessonIds: expectedIds };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  validateCommittedContent().then(
    (result) => console.log(`Validated ${result.lessonCount} committed lesson projections.`),
    (error: unknown) => {
      console.error(error);
      process.exitCode = 1;
    },
  );
}
