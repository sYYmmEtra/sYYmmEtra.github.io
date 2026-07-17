import crypto from "node:crypto";
import { constants as fsConstants } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import YAML from "yaml";
import { z } from "zod";

import {
  enrichMetadata as defaultEnrichMetadata,
  StagedLessonInputSchema,
  type EnrichMetadataOptions,
  type EnrichmentResult,
} from "./enrich-metadata";
import { parseLearningLog } from "./lib/learning-log";
import { discoverLessonFiles, splitLessonSegments } from "./lib/lessons";
import { matchSegmentsToLog, type MatchedLesson } from "./lib/matcher";
import {
  applyEnrichment,
  assertUniqueMetadataSlugs,
  createMetadataSidecar,
  reconcileMetadataWithSource,
  SidecarMetadataSchema,
  type SidecarMetadata,
} from "./lib/metadata";
import { assertDisjointRoots, resolveExistingPath } from "./lib/paths";
import {
  assertNoUnexpectedRemovals,
  withSyncTransaction,
  type SyncCandidatePaths,
  type StableLessonAssignment,
} from "./lib/transaction";

const SCRIPT_DIRECTORY = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_WEBSITE_ROOT = path.resolve(SCRIPT_DIRECTORY, "..");
const SIDECAR_FILE = /^(lesson-[0-9]{4})\.yml$/;

const SyncIndexLessonSchema = z
  .object({
    id: z.string().regex(/^lesson-[0-9]{4}$/),
    lesson: z.number().int().positive(),
    source: z
      .object({
        file: z.string().regex(/^lessons\/\d{4}-\d{2}-\d{2}\.md$/),
        section: z.number().int().positive(),
      })
      .strict(),
    sourceHash: z.string().regex(/^sha256:[0-9a-f]{64}$/),
    slug: z.string().regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/),
  })
  .strict()
  .superRefine((record, context) => {
    const expected = `lesson-${String(record.lesson).padStart(4, "0")}`;
    if (record.id !== expected) {
      context.addIssue({
        code: "custom",
        path: ["id"],
        message: `Lesson ${record.lesson} requires canonical ID ${expected}`,
      });
    }
  });

const SyncIndexSchema = z
  .object({
    schemaVersion: z.literal(1),
    lessons: z.array(SyncIndexLessonSchema),
  })
  .strict()
  .superRefine((index, context) => {
    const ids = new Set<string>();
    for (const [position, lesson] of index.lessons.entries()) {
      if (ids.has(lesson.id)) {
        context.addIssue({
          code: "custom",
          path: ["lessons", position, "id"],
          message: `Duplicate sync index ID ${lesson.id}`,
        });
      }
      ids.add(lesson.id);
    }
    const sorted = [...index.lessons].sort(
      (left, right) => left.lesson - right.lesson || left.id.localeCompare(right.id),
    );
    if (index.lessons.some((lesson, position) => lesson.id !== sorted[position]?.id)) {
      context.addIssue({
        code: "custom",
        path: ["lessons"],
        message: "Sync index lessons must be sorted by lesson number",
      });
    }
  });

type SyncIndex = z.infer<typeof SyncIndexSchema>;
type SyncIndexLesson = z.infer<typeof SyncIndexLessonSchema>;

interface SourceSnapshotEntry {
  kind: "file" | "symlink";
  digest: string;
}

type SourceSnapshot = Record<string, SourceSnapshotEntry>;

export type SyncEnrichLesson = (
  options: EnrichMetadataOptions,
) => Promise<EnrichmentResult>;

export interface RunSyncOptions {
  websiteRoot: string;
  sourceRoot: string;
  enrichMetadata: boolean;
  allowRemovals?: boolean;
  enrichLesson?: SyncEnrichLesson;
  logger?: (message: string) => void;
  /** Test seam invoked inside the candidate transaction before deterministic validation. */
  beforeCandidateValidation?: () => void | Promise<void>;
  /** Test seam invoked after candidate validation and immediately before commit. */
  beforePreCommitValidation?: () => void | Promise<void>;
  /** Test seam invoked after installs but before the transaction finalizes. */
  beforeFinalIntegrityValidation?: () => void | Promise<void>;
  /** Test seam invoked after generated-content directory validation but before a file opens. */
  beforeExistingContentRead?: (target: string) => void | Promise<void>;
  /** Test seam for an additional read-only candidate assertion. */
  validateCandidate?: (
    candidate: SyncCandidatePaths,
  ) => void | Promise<void>;
}

export interface SyncResult {
  created: number;
  changed: number;
  unchanged: number;
  pending: number;
  lessonIds: string[];
}

interface LessonProjection {
  match: MatchedLesson;
  sidecar: SidecarMetadata;
  sidecarText: string;
  contentText: string;
}

interface PhysicalIdentity {
  dev: number;
  ino: number;
}

function errorCode(error: unknown): string | undefined {
  return error instanceof Error && "code" in error
    ? String(error.code)
    : undefined;
}

async function pathExists(target: string): Promise<boolean> {
  try {
    await fs.lstat(target);
    return true;
  } catch (error) {
    if (errorCode(error) === "ENOENT") return false;
    throw error;
  }
}

async function assertPhysicalFile(target: string, label: string): Promise<void> {
  const stat = await fs.lstat(target);
  if (stat.isSymbolicLink() || !stat.isFile()) {
    throw new Error(`${label} must be a physical regular file`);
  }
}

async function assertPhysicalDirectory(target: string, label: string): Promise<void> {
  const stat = await fs.lstat(target);
  if (stat.isSymbolicLink() || !stat.isDirectory()) {
    throw new Error(`${label} must be a physical directory`);
  }
}

function physicalIdentity(stat: { dev: number; ino: number }): PhysicalIdentity {
  return { dev: stat.dev, ino: stat.ino };
}

function samePhysicalIdentity(
  left: PhysicalIdentity,
  right: PhysicalIdentity,
): boolean {
  return left.dev === right.dev && left.ino === right.ino;
}

async function capturePhysicalDirectory(
  target: string,
  label: string,
): Promise<PhysicalIdentity> {
  const stat = await fs.lstat(target);
  if (stat.isSymbolicLink() || !stat.isDirectory()) {
    throw new Error(`${label} must be a physical directory`);
  }
  return physicalIdentity(stat);
}

async function assertPhysicalDirectoryIdentity(
  target: string,
  label: string,
  expected: PhysicalIdentity,
): Promise<void> {
  const stat = await fs.lstat(target);
  if (stat.isSymbolicLink() || !stat.isDirectory()) {
    throw new Error(`${label} changed while reading`);
  }
  const actual = physicalIdentity(stat);
  if (!samePhysicalIdentity(actual, expected)) {
    throw new Error(`${label} changed while reading`);
  }
}

async function snapshotSourceTree(sourceRoot: string): Promise<SourceSnapshot> {
  const snapshot: SourceSnapshot = {};

  async function walk(directory: string): Promise<void> {
    const entries = await fs.readdir(directory, { withFileTypes: true });
    entries.sort((left, right) => left.name.localeCompare(right.name));
    for (const entry of entries) {
      if (entry.name === ".git") continue;
      const absolute = path.join(directory, entry.name);
      const relative = path.relative(sourceRoot, absolute).split(path.sep).join("/");
      if (entry.isDirectory()) {
        await walk(absolute);
      } else if (entry.isFile()) {
        snapshot[relative] = {
          kind: "file",
          digest: crypto
            .createHash("sha256")
            .update(await fs.readFile(absolute))
            .digest("hex"),
        };
      } else if (entry.isSymbolicLink()) {
        snapshot[relative] = {
          kind: "symlink",
          digest: crypto
            .createHash("sha256")
            .update(await fs.readlink(absolute))
            .digest("hex"),
        };
      } else {
        throw new Error(`Unsupported source filesystem entry: ${relative}`);
      }
    }
  }

  await walk(sourceRoot);
  return snapshot;
}

function assertSameSourceSnapshot(
  before: SourceSnapshot,
  after: SourceSnapshot,
): void {
  if (JSON.stringify(after) !== JSON.stringify(before)) {
    throw new Error("AI Daily source tree changed during sync");
  }
}

function serializeSidecar(sidecar: SidecarMetadata): string {
  return YAML.stringify(SidecarMetadataSchema.parse(sidecar), {
    lineWidth: 0,
  });
}

function renderMarkdown(sidecar: SidecarMetadata, rawBody: string): string {
  return `---\n${serializeSidecar(sidecar)}---\n${rawBody}`;
}

function createIndexLesson(sidecar: SidecarMetadata): SyncIndexLesson {
  return SyncIndexLessonSchema.parse({
    id: sidecar.id,
    lesson: sidecar.lesson,
    source: {
      file: sidecar.source.file,
      section: sidecar.source.section,
    },
    sourceHash: sidecar.source.hash,
    slug: sidecar.slug,
  });
}

function serializeIndex(index: SyncIndex): string {
  return `${JSON.stringify(SyncIndexSchema.parse(index), null, 2)}\n`;
}

async function loadExistingSidecars(
  websiteRoot: string,
): Promise<Map<string, SidecarMetadata>> {
  const directory = path.join(websiteRoot, "metadata/ai-daily");
  if (!(await pathExists(directory))) return new Map();

  const directoryStat = await fs.lstat(directory);
  if (directoryStat.isSymbolicLink() || !directoryStat.isDirectory()) {
    throw new Error("AI Daily sidecar path must be a physical directory");
  }

  const result = new Map<string, SidecarMetadata>();
  const entries = await fs.readdir(directory, { withFileTypes: true });
  entries.sort((left, right) => left.name.localeCompare(right.name));
  for (const entry of entries) {
    const match = SIDECAR_FILE.exec(entry.name);
    if (!match || !entry.isFile() || entry.isSymbolicLink()) {
      throw new Error(`Unexpected sidecar entry: ${entry.name}`);
    }
    const target = path.join(directory, entry.name);
    await assertPhysicalFile(target, `Sidecar ${entry.name}`);
    let parsed: unknown;
    try {
      parsed = YAML.parse(await fs.readFile(target, "utf8"));
    } catch (error) {
      throw new Error(`Malformed sidecar YAML ${entry.name}`, { cause: error });
    }
    const sidecar = SidecarMetadataSchema.parse(parsed);
    const expectedId = match[1]!;
    if (sidecar.id !== expectedId) {
      throw new Error(
        `Sidecar filename ${entry.name} does not match contained ID ${sidecar.id}`,
      );
    }
    if (result.has(sidecar.id)) {
      throw new Error(`Duplicate sidecar ID ${sidecar.id}`);
    }
    result.set(sidecar.id, sidecar);
  }
  return result;
}

async function loadExistingIndex(websiteRoot: string): Promise<SyncIndex | undefined> {
  const target = path.join(websiteRoot, "sync-index.json");
  if (!(await pathExists(target))) return undefined;
  await assertPhysicalFile(target, "sync-index.json");
  let parsed: unknown;
  try {
    parsed = JSON.parse(await fs.readFile(target, "utf8"));
  } catch (error) {
    throw new Error("Malformed sync-index.json", { cause: error });
  }
  return SyncIndexSchema.parse(parsed);
}

function stableAssignment(record: SyncIndexLesson): StableLessonAssignment {
  return {
    id: record.id,
    lesson: record.lesson,
    source: record.source,
    sourceHash: record.sourceHash,
  };
}

function sidecarForMatch(
  match: MatchedLesson,
  existing: SidecarMetadata | undefined,
): SidecarMetadata {
  const source = {
    file: `lessons/${match.segment.file}`,
    section: match.segment.section,
    hash: match.segment.hash,
  };
  const base = existing
    ? reconcileMetadataWithSource(existing, match.segment.hash)
    : createMetadataSidecar({
        source,
        lesson: match.log.lesson,
        date: match.log.date,
        track: match.log.track,
        depth: match.log.depth,
        titleZh: match.segment.titleZh,
        summaryZh: match.log.summaryZh,
      });

  return SidecarMetadataSchema.parse({
    ...base,
    id: match.id,
    source,
    lesson: match.log.lesson,
    date: match.log.date,
    track: match.log.track,
    depth: match.log.depth,
    titleZh: match.segment.titleZh,
    summaryZh: match.log.summaryZh,
  });
}

async function readCurrentSegmentHash(
  sourceRoot: string,
  match: MatchedLesson,
): Promise<string> {
  const segments = await splitLessonSegments(
    path.join(sourceRoot, "lessons", match.segment.file),
  );
  const current = segments.find(
    (segment) => segment.section === match.segment.section,
  );
  if (!current) {
    throw new Error(
      `Source section disappeared during enrichment: ${match.segment.file} section ${match.segment.section}`,
    );
  }
  return current.hash;
}

async function enrichProjection(
  projection: LessonProjection,
  options: RunSyncOptions,
  candidate: SyncCandidatePaths,
  sourceRoot: string,
): Promise<LessonProjection> {
  if (
    !options.enrichMetadata ||
    projection.sidecar.metadataStatus === "current"
  ) {
    return projection;
  }

  const lesson = StagedLessonInputSchema.parse({
    schemaVersion: 1,
    id: projection.sidecar.id,
    sourceHash: projection.sidecar.source.hash,
    titleZh: projection.sidecar.titleZh,
    summaryZh: projection.sidecar.summaryZh,
    tldrZh: null,
    bodyMarkdown: projection.match.segment.raw,
  });
  const stagingParent = candidate.metadata;
  candidate.assertSafeWritePath(stagingParent);
  const result = await (options.enrichLesson ?? defaultEnrichMetadata)({
    stagingParent,
    assertSafeStagingPath: async (target) => candidate.assertSafeWritePath(target),
    lesson,
    readCurrentSourceHash: () =>
      readCurrentSegmentHash(sourceRoot, projection.match),
    parentEnv: { ...process.env, AI_DAILY_SOURCE: sourceRoot },
  });
  if (!result.ok) {
    options.logger?.(
      `${projection.sidecar.id}: metadata enrichment skipped (${result.reason}: ${result.message})`,
    );
    return projection;
  }

  const sidecar = applyEnrichment(projection.sidecar, result.value);
  return {
    ...projection,
    sidecar,
    sidecarText: serializeSidecar(sidecar),
    contentText: renderMarkdown(sidecar, projection.match.segment.raw),
  };
}

async function copyOtherMetadata(
  websiteRoot: string,
  candidate: SyncCandidatePaths,
): Promise<void> {
  const current = path.join(websiteRoot, "metadata");
  if (!(await pathExists(current))) return;
  const stat = await fs.lstat(current);
  if (stat.isSymbolicLink() || !stat.isDirectory()) {
    throw new Error("Website metadata must be a physical directory");
  }
  for (const entry of await fs.readdir(current, { withFileTypes: true })) {
    if (entry.name === "ai-daily") continue;
    const source = path.join(current, entry.name);
    const destination = path.join(candidate.metadata, entry.name);
    candidate.assertSafeWritePath(destination);
    await fs.cp(source, destination, { recursive: true, preserveTimestamps: false });
  }
}

async function validateWrittenCandidate(
  candidate: SyncCandidatePaths,
  projections: readonly LessonProjection[],
  index: SyncIndex,
): Promise<void> {
  const expectedSidecars = projections.map(
    (projection) => `${projection.sidecar.id}.yml`,
  );
  const sidecarDirectory = path.join(candidate.metadata, "ai-daily");
  const actualSidecars = (await fs.readdir(sidecarDirectory)).sort();
  if (JSON.stringify(actualSidecars) !== JSON.stringify(expectedSidecars)) {
    throw new Error("Candidate sidecar set is incomplete or contains extras");
  }

  for (const projection of projections) {
    const sidecarPath = path.join(
      sidecarDirectory,
      `${projection.sidecar.id}.yml`,
    );
    const contentPath = path.join(
      candidate.content,
      `${projection.sidecar.id}.md`,
    );
    const sidecarText = await fs.readFile(sidecarPath, "utf8");
    const parsedSidecar = SidecarMetadataSchema.parse(YAML.parse(sidecarText));
    if (serializeSidecar(parsedSidecar) !== projection.sidecarText) {
      throw new Error(`Candidate sidecar is not deterministic: ${projection.sidecar.id}`);
    }
    if ((await fs.readFile(contentPath, "utf8")) !== projection.contentText) {
      throw new Error(`Candidate content does not match source: ${projection.sidecar.id}`);
    }
  }

  const contentFiles = (await fs.readdir(candidate.content)).sort();
  const expectedContent = projections.map(
    (projection) => `${projection.sidecar.id}.md`,
  );
  if (JSON.stringify(contentFiles) !== JSON.stringify(expectedContent)) {
    throw new Error("Candidate content set is incomplete or contains extras");
  }
  const candidateIndex = SyncIndexSchema.parse(
    JSON.parse(await fs.readFile(candidate.index, "utf8")),
  );
  if (serializeIndex(candidateIndex) !== serializeIndex(index)) {
    throw new Error("Candidate sync index is not deterministic");
  }
}

async function readOptionalPhysicalText(
  target: string,
  label: string,
): Promise<string | undefined> {
  try {
    const stat = await fs.lstat(target);
    if (stat.isSymbolicLink() || !stat.isFile()) {
      throw new Error(`${label} must be a physical regular file`);
    }
    return await fs.readFile(target, "utf8");
  } catch (error) {
    if (errorCode(error) === "ENOENT") return undefined;
    throw error;
  }
}

async function readOptionalGeneratedContentText(
  target: string,
  label: string,
  directory: string,
  directoryIdentity: PhysicalIdentity,
  beforeRead: RunSyncOptions["beforeExistingContentRead"],
): Promise<string | undefined> {
  await assertPhysicalDirectoryIdentity(
    directory,
    "Generated content directory",
    directoryIdentity,
  );
  await beforeRead?.(target);
  await assertPhysicalDirectoryIdentity(
    directory,
    "Generated content directory",
    directoryIdentity,
  );

  let handle;
  try {
    handle = await fs.open(
      target,
      fsConstants.O_RDONLY | fsConstants.O_NOFOLLOW,
    );
  } catch (error) {
    if (errorCode(error) === "ENOENT") return undefined;
    if (errorCode(error) === "ELOOP") {
      throw new Error(`${label} must be a physical regular file`);
    }
    throw error;
  }

  try {
    const stat = await handle.stat();
    if (!stat.isFile()) {
      throw new Error(`${label} must be a physical regular file`);
    }
    const fileIdentity = physicalIdentity(stat);
    const content = await handle.readFile({ encoding: "utf8" });
    await assertPhysicalDirectoryIdentity(
      directory,
      "Generated content directory",
      directoryIdentity,
    );
    const current = await fs.lstat(target);
    if (
      current.isSymbolicLink() ||
      !current.isFile() ||
      !samePhysicalIdentity(physicalIdentity(current), fileIdentity)
    ) {
      throw new Error(`${label} changed while reading`);
    }
    return content;
  } finally {
    await handle.close();
  }
}

function calculateCounts(
  projections: readonly LessonProjection[],
  existingSidecars: ReadonlyMap<string, SidecarMetadata>,
  existingTexts: ReadonlyMap<string, { sidecar?: string; content?: string }>,
): Omit<SyncResult, "pending" | "lessonIds"> {
  let created = 0;
  let changed = 0;
  let unchanged = 0;
  for (const projection of projections) {
    if (!existingSidecars.has(projection.sidecar.id)) {
      created += 1;
      continue;
    }
    const existing = existingTexts.get(projection.sidecar.id);
    if (
      existing?.sidecar === projection.sidecarText &&
      existing.content === projection.contentText
    ) {
      unchanged += 1;
    } else {
      changed += 1;
    }
  }
  return { created, changed, unchanged };
}

export async function runSync(options: RunSyncOptions): Promise<SyncResult> {
  const websiteRoot = path.resolve(resolveExistingPath(options.websiteRoot));
  const sourceRoot = path.resolve(resolveExistingPath(options.sourceRoot));
  assertDisjointRoots(websiteRoot, sourceRoot);

  const sourceSnapshot = await snapshotSourceTree(sourceRoot);
  const learningLogPath = path.join(sourceRoot, "learning-log.md");
  const lessonsRoot = path.join(sourceRoot, "lessons");
  await assertPhysicalFile(learningLogPath, "AI Daily learning-log.md");
  await assertPhysicalDirectory(lessonsRoot, "AI Daily lessons directory");
  const lessonFiles = await discoverLessonFiles(lessonsRoot);
  const rows = parseLearningLog(await fs.readFile(learningLogPath, "utf8"));
  const segments = (
    await Promise.all(lessonFiles.map((file) => splitLessonSegments(file)))
  ).flat();
  const matches = matchSegmentsToLog(segments, rows).sort(
    (left, right) => left.log.lesson - right.log.lesson,
  );
  if (matches.length !== rows.length) {
    throw new Error(
      `Learning log contains ${rows.length} rows but only ${matches.length} lesson segments matched`,
    );
  }

  const existingSidecars = await loadExistingSidecars(websiteRoot);
  const existingIndex = await loadExistingIndex(websiteRoot);
  const initialProjections = matches.map((match) => {
    const sidecar = sidecarForMatch(match, existingSidecars.get(match.id));
    return {
      match,
      sidecar,
      sidecarText: serializeSidecar(sidecar),
      contentText: renderMarkdown(sidecar, match.segment.raw),
    } satisfies LessonProjection;
  });
  assertUniqueMetadataSlugs(initialProjections.map(({ sidecar }) => sidecar));

  const currentAssignments = initialProjections.map(({ sidecar }) =>
    stableAssignment(createIndexLesson(sidecar)),
  );
  assertNoUnexpectedRemovals({
    previous: (existingIndex?.lessons ?? []).map(stableAssignment),
    current: currentAssignments,
    allowRemovals: options.allowRemovals ?? false,
  });

  const existingTexts = new Map<
    string,
    { sidecar?: string; content?: string }
  >();
  const existingContentDirectory = path.join(
    websiteRoot,
    "src/content/ai-daily",
  );
  const existingContentDirectoryIdentity = (
    await pathExists(existingContentDirectory)
  )
    ? await capturePhysicalDirectory(
        existingContentDirectory,
        "Generated content directory",
      )
    : undefined;
  for (const projection of initialProjections) {
    existingTexts.set(projection.sidecar.id, {
      sidecar: await readOptionalPhysicalText(
        path.join(
          websiteRoot,
          "metadata/ai-daily",
          `${projection.sidecar.id}.yml`,
        ),
        `Sidecar ${projection.sidecar.id}.yml`,
      ),
      content: existingContentDirectoryIdentity
        ? await readOptionalGeneratedContentText(
            path.join(
              websiteRoot,
              "src/content/ai-daily",
              `${projection.sidecar.id}.md`,
            ),
            `Generated content ${projection.sidecar.id}.md`,
            existingContentDirectory,
            existingContentDirectoryIdentity,
            options.beforeExistingContentRead,
          )
        : undefined,
    });
  }

  let finalProjections: LessonProjection[] = [];
  let finalIndex: SyncIndex | undefined;
  await withSyncTransaction({
    websiteRoot,
    writer: async (candidate) => {
      await copyOtherMetadata(websiteRoot, candidate);
      const sidecarDirectory = path.join(candidate.metadata, "ai-daily");
      candidate.assertSafeWritePath(sidecarDirectory);
      await fs.mkdir(sidecarDirectory, { recursive: true });

      finalProjections = [];
      for (const projection of initialProjections) {
        finalProjections.push(
          await enrichProjection(projection, options, candidate, sourceRoot),
        );
      }
      assertUniqueMetadataSlugs(finalProjections.map(({ sidecar }) => sidecar));
      finalIndex = SyncIndexSchema.parse({
        schemaVersion: 1,
        lessons: finalProjections.map(({ sidecar }) => createIndexLesson(sidecar)),
      });

      for (const projection of finalProjections) {
        const sidecarPath = path.join(
          sidecarDirectory,
          `${projection.sidecar.id}.yml`,
        );
        const contentPath = path.join(
          candidate.content,
          `${projection.sidecar.id}.md`,
        );
        candidate.assertSafeWritePath(sidecarPath);
        candidate.assertSafeWritePath(contentPath);
        await fs.writeFile(sidecarPath, projection.sidecarText, "utf8");
        await fs.writeFile(contentPath, projection.contentText, "utf8");
      }
      candidate.assertSafeWritePath(candidate.index);
      await fs.writeFile(candidate.index, serializeIndex(finalIndex), "utf8");
    },
    validator: async (candidate) => {
      await options.beforeCandidateValidation?.();
      assertSameSourceSnapshot(
        sourceSnapshot,
        await snapshotSourceTree(sourceRoot),
      );
      if (!finalIndex) throw new Error("Sync candidate index was not created");
      await validateWrittenCandidate(candidate, finalProjections, finalIndex);
      await options.validateCandidate?.(candidate);
    },
    beforeCommit: async () => {
      await options.beforePreCommitValidation?.();
      assertSameSourceSnapshot(
        sourceSnapshot,
        await snapshotSourceTree(sourceRoot),
      );
    },
    beforeFinalize: async () => {
      await options.beforeFinalIntegrityValidation?.();
      assertSameSourceSnapshot(
        sourceSnapshot,
        await snapshotSourceTree(sourceRoot),
      );
    },
  });
  const counts = calculateCounts(
    finalProjections,
    existingSidecars,
    existingTexts,
  );
  return {
    ...counts,
    pending: finalProjections.filter(
      ({ sidecar }) => sidecar.metadataStatus !== "current",
    ).length,
    lessonIds: finalProjections.map(({ sidecar }) => sidecar.id),
  };
}

interface CliArguments {
  allowRemovals: boolean;
  enrichMetadata: boolean;
}

function parseCliArguments(argv: readonly string[]): CliArguments {
  let allowRemovals = false;
  let enrichMetadata = true;
  for (const argument of argv) {
    if (argument === "--allow-removals") {
      allowRemovals = true;
    } else if (argument === "--no-enrich") {
      enrichMetadata = false;
    } else {
      throw new Error(`Unknown argument: ${argument}`);
    }
  }
  return { allowRemovals, enrichMetadata };
}

async function runCli(): Promise<void> {
  const sourceRoot = process.env.AI_DAILY_SOURCE;
  if (!sourceRoot) {
    throw new Error(
      "AI_DAILY_SOURCE is required and must point to the read-only AI Daily repository",
    );
  }
  const cli = parseCliArguments(process.argv.slice(2));
  const result = await runSync({
    websiteRoot: DEFAULT_WEBSITE_ROOT,
    sourceRoot,
    enrichMetadata: cli.enrichMetadata,
    allowRemovals: cli.allowRemovals,
    logger: (message) => process.stderr.write(`${message}\n`),
  });
  process.stdout.write(
    `created=${result.created} changed=${result.changed} unchanged=${result.unchanged} pending=${result.pending}\n`,
  );
  process.stdout.write(`lessons=${result.lessonIds.join(",")}\n`);
}

const invokedPath = process.argv[1]
  ? pathToFileURL(path.resolve(process.argv[1])).href
  : undefined;
if (invokedPath === import.meta.url) {
  runCli().catch((error: unknown) => {
    process.stderr.write(
      `Sync failed: ${error instanceof Error ? error.message : String(error)}\n`,
    );
    process.exitCode = 1;
  });
}
