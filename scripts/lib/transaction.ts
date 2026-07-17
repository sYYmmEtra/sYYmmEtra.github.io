import { randomUUID } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

import {
  assertSafeWritePath as assertWebsiteWritePath,
  resolveExistingPath,
} from "./paths";

export interface StableLessonAssignment {
  id: string;
  lesson: number;
  source: {
    file: string;
    section: number;
  };
  /** Source content can change without changing the stable assignment identity. */
  sourceHash?: string;
}

export interface StableAssignmentGuard {
  previous: readonly StableLessonAssignment[];
  current: readonly StableLessonAssignment[];
  allowRemovals: boolean;
}

function isStableLessonId(value: unknown): value is string {
  if (typeof value !== "string") {
    return false;
  }

  const match = /^lesson-(\d{4,})$/.exec(value);
  if (!match) {
    return false;
  }

  const lesson = Number(match[1]);
  return (
    Number.isSafeInteger(lesson) &&
    lesson > 0 &&
    `lesson-${String(lesson).padStart(4, "0")}` === value
  );
}

function assertUniqueIds(ids: readonly string[], label: string): void {
  const seen = new Set<string>();

  for (const id of ids) {
    if (!isStableLessonId(id)) {
      throw new Error(`Invalid lesson ID in ${label}`);
    }
    if (seen.has(id)) {
      throw new Error(`Duplicate lesson ID ${id} in ${label}`);
    }
    seen.add(id);
  }
}

function assertValidAssignments(
  assignments: readonly StableLessonAssignment[],
  label: string,
): void {
  for (const assignment of assignments) {
    if (!assignment || !isStableLessonId(assignment.id)) {
      throw new Error(`Invalid lesson assignment ID in ${label}`);
    }
    if (
      !Number.isSafeInteger(assignment.lesson) ||
      assignment.lesson < 1 ||
      !assignment.source ||
      typeof assignment.source.file !== "string" ||
      assignment.source.file.length === 0 ||
      !Number.isSafeInteger(assignment.source.section) ||
      assignment.source.section < 1
    ) {
      throw new Error(
        `Invalid lesson assignment identity in ${label}: source file and section plus a positive lesson are required`,
      );
    }
  }

  assertUniqueIds(
    assignments.map((assignment) => assignment.id),
    label,
  );
}

function describeAssignment(assignment: StableLessonAssignment): string {
  return `lesson ${assignment.lesson} at ${assignment.source.file} section ${assignment.source.section}`;
}

export function assertNoUnexpectedRemovals(
  previousIds: readonly string[],
  currentIds: readonly string[],
  allowRemovals: boolean,
): void;
export function assertNoUnexpectedRemovals(
  guard: StableAssignmentGuard,
): void;
export function assertNoUnexpectedRemovals(
  previousOrGuard: readonly string[] | StableAssignmentGuard,
  currentIds?: readonly string[],
  allowRemovals?: boolean,
): void {
  if (Array.isArray(previousOrGuard)) {
    const previousIds = previousOrGuard;
    const current = currentIds ?? [];
    assertUniqueIds(previousIds, "previous IDs");
    assertUniqueIds(current, "current IDs");

    if (!allowRemovals) {
      const currentSet = new Set(current);
      const missing = previousIds.filter((id) => !currentSet.has(id));
      if (missing.length > 0) {
        throw new Error(
          `Unexpected lesson removal: missing ${missing.join(", ")}`,
        );
      }
    }
    return;
  }

  const guard = previousOrGuard as StableAssignmentGuard;
  assertValidAssignments(guard.previous, "previous assignments");
  assertValidAssignments(guard.current, "current assignments");
  if (guard.allowRemovals) {
    return;
  }

  const currentById = new Map(
    guard.current.map((assignment) => [assignment.id, assignment]),
  );
  for (const previous of guard.previous) {
    const current = currentById.get(previous.id);
    if (!current) {
      throw new Error(`Unexpected lesson removal: missing ${previous.id}`);
    }

    if (
      previous.lesson !== current.lesson ||
      previous.source.file !== current.source.file ||
      previous.source.section !== current.source.section
    ) {
      throw new Error(
        `Unexpected lesson reassignment for ${previous.id}: ${describeAssignment(previous)} changed to ${describeAssignment(current)}`,
      );
    }
  }
}

export interface SyncCandidatePaths {
  readonly metadata: string;
  readonly content: string;
  readonly index: string;
  readonly assertSafeWritePath: (target: string) => void;
}

type CandidateProjectionPaths = Pick<
  SyncCandidatePaths,
  "metadata" | "content" | "index"
>;

export type SyncDestination = "metadata" | "content" | "index";
export type SyncCommitPhase = "backup" | "install";

export interface SyncCommitStep {
  phase: SyncCommitPhase;
  destination: SyncDestination;
}

export interface SyncTransactionOptions {
  websiteRoot: string;
  writer(candidate: SyncCandidatePaths): void | Promise<void>;
  validator(candidate: SyncCandidatePaths): void | Promise<void>;
  /** Test seam for deterministic rollback coverage; called after each rename. */
  afterCommitStep?(step: SyncCommitStep): void | Promise<void>;
}

interface ProjectionDestination {
  name: SyncDestination;
  candidate: string;
  current: string;
  backup: string;
}

async function pathExists(target: string): Promise<boolean> {
  try {
    await fs.lstat(target);
    return true;
  } catch (error) {
    if (
      error instanceof Error &&
      "code" in error &&
      error.code === "ENOENT"
    ) {
      return false;
    }
    throw error;
  }
}

function assertTransactionPath(websiteRoot: string, target: string): void {
  assertWebsiteWritePath(websiteRoot, target);
}

async function safeMkdir(
  websiteRoot: string,
  target: string,
  options: { recursive?: boolean } = {},
): Promise<void> {
  assertTransactionPath(websiteRoot, target);
  await fs.mkdir(target, options);
}

async function safeRename(
  websiteRoot: string,
  source: string,
  destination: string,
): Promise<void> {
  assertTransactionPath(websiteRoot, source);
  assertTransactionPath(websiteRoot, destination);
  await fs.rename(source, destination);
}

async function safeRemove(
  websiteRoot: string,
  target: string,
): Promise<void> {
  assertTransactionPath(websiteRoot, target);
  await fs.rm(target, { recursive: true, force: true });
}

async function validateCandidateEntry(
  websiteRoot: string,
  candidateRoot: string,
  target: string,
): Promise<void> {
  assertTransactionPath(websiteRoot, target);
  assertWebsiteWritePath(candidateRoot, target);
  const stat = await fs.lstat(target);

  if (stat.isSymbolicLink()) {
    throw new Error(
      `Candidate contains a symlink and may escape the website root: ${target}`,
    );
  }
  if (stat.isDirectory()) {
    for (const entry of await fs.readdir(target)) {
      await validateCandidateEntry(
        websiteRoot,
        candidateRoot,
        path.join(target, entry),
      );
    }
    return;
  }
  if (!stat.isFile()) {
    throw new Error(
      `Candidate contains an unsupported filesystem entry: ${target}`,
    );
  }
}

async function validateCandidateProjection(
  websiteRoot: string,
  candidateRoot: string,
  candidate: CandidateProjectionPaths,
): Promise<void> {
  const metadata = await fs.lstat(candidate.metadata);
  if (metadata.isSymbolicLink()) {
    throw new Error(
      "Candidate metadata is a symlink and may escape the website root",
    );
  }
  if (!metadata.isDirectory()) {
    throw new Error("Candidate metadata must be a physical directory");
  }
  const content = await fs.lstat(candidate.content);
  if (content.isSymbolicLink()) {
    throw new Error(
      "Candidate content is a symlink and may escape the website root",
    );
  }
  if (!content.isDirectory()) {
    throw new Error("Candidate content must be a physical directory");
  }
  const index = await fs.lstat(candidate.index);
  if (index.isSymbolicLink()) {
    throw new Error(
      "Candidate sync-index.json is a symlink and may escape the website root",
    );
  }
  if (!index.isFile()) {
    throw new Error("Candidate sync-index.json must be a physical file");
  }

  await validateCandidateEntry(websiteRoot, candidateRoot, candidate.metadata);
  await validateCandidateEntry(websiteRoot, candidateRoot, candidate.content);
  await validateCandidateEntry(websiteRoot, candidateRoot, candidate.index);
}

async function ensureDestinationParent(
  websiteRoot: string,
  destination: string,
  createdParents: string[],
): Promise<void> {
  const parent = path.dirname(destination);
  const relative = path.relative(websiteRoot, parent);
  if (!relative) {
    return;
  }

  let current = websiteRoot;
  for (const part of relative.split(path.sep)) {
    current = path.join(current, part);
    assertTransactionPath(websiteRoot, current);
    if (await pathExists(current)) {
      const stat = await fs.stat(current);
      if (!stat.isDirectory()) {
        throw new Error(`Destination parent is not a directory: ${current}`);
      }
      continue;
    }

    await safeMkdir(websiteRoot, current);
    createdParents.push(current);
  }
}

async function removeCreatedParents(
  websiteRoot: string,
  createdParents: readonly string[],
): Promise<void> {
  for (const parent of [...createdParents].reverse()) {
    assertTransactionPath(websiteRoot, parent);
    try {
      await fs.rmdir(parent);
    } catch (error) {
      if (
        error instanceof Error &&
        "code" in error &&
        (error.code === "ENOENT" || error.code === "ENOTEMPTY")
      ) {
        continue;
      }
      throw error;
    }
  }
}

async function rollbackProjection(
  websiteRoot: string,
  destinations: readonly ProjectionDestination[],
  installed: ReadonlySet<SyncDestination>,
  backedUp: ReadonlySet<SyncDestination>,
  createdParents: readonly string[],
): Promise<void> {
  for (const destination of [...destinations].reverse()) {
    if (installed.has(destination.name)) {
      await safeRemove(websiteRoot, destination.current);
    }
  }

  for (const destination of [...destinations].reverse()) {
    if (backedUp.has(destination.name)) {
      await ensureDestinationParent(websiteRoot, destination.current, []);
      await safeRename(
        websiteRoot,
        destination.backup,
        destination.current,
      );
    }
  }

  await removeCreatedParents(websiteRoot, createdParents);
}

async function removeSyncTmpIfCreated(
  websiteRoot: string,
  syncTmp: string,
  syncTmpCreated: boolean,
): Promise<void> {
  if (!syncTmpCreated) {
    return;
  }

  assertTransactionPath(websiteRoot, syncTmp);
  try {
    await fs.rmdir(syncTmp);
  } catch (error) {
    if (
      error instanceof Error &&
      "code" in error &&
      (error.code === "ENOENT" || error.code === "ENOTEMPTY")
    ) {
      return;
    }
    throw error;
  }
}

export async function withSyncTransaction(
  options: SyncTransactionOptions,
): Promise<void> {
  const websiteRoot = path.resolve(resolveExistingPath(options.websiteRoot));
  const syncTmp = path.join(websiteRoot, ".sync-tmp");
  assertTransactionPath(websiteRoot, syncTmp);
  const syncTmpCreated = !(await pathExists(syncTmp));
  await safeMkdir(websiteRoot, syncTmp, { recursive: true });

  const transactionRoot = path.join(syncTmp, randomUUID());
  const candidateRoot = path.join(transactionRoot, "candidate");
  const backupRoot = path.join(transactionRoot, "backup");
  await safeMkdir(websiteRoot, candidateRoot, { recursive: true });

  const canonicalCandidate: CandidateProjectionPaths = {
    metadata: path.join(candidateRoot, "metadata"),
    content: path.join(candidateRoot, "src/content/ai-daily"),
    index: path.join(candidateRoot, "sync-index.json"),
  };
  const candidate: SyncCandidatePaths = Object.freeze({
    metadata: canonicalCandidate.metadata,
    content: canonicalCandidate.content,
    index: canonicalCandidate.index,
    assertSafeWritePath(target: string): void {
      assertTransactionPath(websiteRoot, target);
      assertWebsiteWritePath(candidateRoot, target);
    },
  });

  const destinations: ProjectionDestination[] = [
    {
      name: "metadata",
      candidate: canonicalCandidate.metadata,
      current: path.join(websiteRoot, "metadata"),
      backup: path.join(backupRoot, "metadata"),
    },
    {
      name: "content",
      candidate: canonicalCandidate.content,
      current: path.join(websiteRoot, "src/content/ai-daily"),
      backup: path.join(backupRoot, "content"),
    },
    {
      name: "index",
      candidate: canonicalCandidate.index,
      current: path.join(websiteRoot, "sync-index.json"),
      backup: path.join(backupRoot, "sync-index.json"),
    },
  ];
  const installed = new Set<SyncDestination>();
  const backedUp = new Set<SyncDestination>();
  const createdParents: string[] = [];

  try {
    await safeMkdir(websiteRoot, canonicalCandidate.metadata, {
      recursive: true,
    });
    await safeMkdir(websiteRoot, canonicalCandidate.content, {
      recursive: true,
    });
    await options.writer(candidate);
    await validateCandidateProjection(
      websiteRoot,
      candidateRoot,
      canonicalCandidate,
    );
    await options.validator(candidate);
    await validateCandidateProjection(
      websiteRoot,
      candidateRoot,
      canonicalCandidate,
    );

    for (const destination of destinations) {
      assertTransactionPath(websiteRoot, destination.candidate);
      assertTransactionPath(websiteRoot, destination.current);
      assertTransactionPath(websiteRoot, destination.backup);
    }

    await safeMkdir(websiteRoot, backupRoot, { recursive: true });
    for (const destination of destinations) {
      if (!(await pathExists(destination.current))) {
        continue;
      }

      await safeRename(
        websiteRoot,
        destination.current,
        destination.backup,
      );
      backedUp.add(destination.name);
      await options.afterCommitStep?.({
        phase: "backup",
        destination: destination.name,
      });
    }

    for (const destination of destinations) {
      await ensureDestinationParent(
        websiteRoot,
        destination.current,
        createdParents,
      );
      await safeRename(
        websiteRoot,
        destination.candidate,
        destination.current,
      );
      installed.add(destination.name);
      await options.afterCommitStep?.({
        phase: "install",
        destination: destination.name,
      });
    }
  } catch (error) {
    let rollbackError: unknown;
    try {
      await rollbackProjection(
        websiteRoot,
        destinations,
        installed,
        backedUp,
        createdParents,
      );
    } catch (caught) {
      rollbackError = caught;
    }

    let cleanupError: unknown;
    try {
      await safeRemove(websiteRoot, transactionRoot);
      await removeSyncTmpIfCreated(websiteRoot, syncTmp, syncTmpCreated);
    } catch (caught) {
      cleanupError = caught;
    }

    if (rollbackError || cleanupError) {
      throw new AggregateError(
        [error, rollbackError, cleanupError].filter(Boolean),
        `Sync transaction failed and recovery was incomplete: ${error instanceof Error ? error.message : String(error)}`,
        { cause: error },
      );
    }
    throw error;
  }

  await safeRemove(websiteRoot, transactionRoot);
  await removeSyncTmpIfCreated(websiteRoot, syncTmp, syncTmpCreated);
}
