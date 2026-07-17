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

    const expectedId = `lesson-${String(assignment.lesson).padStart(4, "0")}`;
    if (assignment.id !== expectedId) {
      throw new Error(
        `Invalid lesson assignment ${assignment.id} in ${label}: lesson ${assignment.lesson} requires ID ${expectedId}`,
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
export type SyncFilesystemOperation =
  | "mkdir"
  | "rename"
  | "remove"
  | "rmdir";
export type SyncFilesystemPhase =
  | "setup"
  | "backup"
  | "install"
  | "rollback-remove"
  | "rollback-restore"
  | "rollback-parent-cleanup"
  | "precommit-cleanup"
  | "postcommit-cleanup";

export interface SyncCommitStep {
  phase: SyncCommitPhase;
  destination: SyncDestination;
}

export interface SyncFilesystemOperationContext {
  operation: SyncFilesystemOperation;
  phase: SyncFilesystemPhase;
  destination?: SyncDestination;
  path: string;
  sourcePath?: string;
}

export class SyncRollbackError extends AggregateError {
  readonly committed = false as const;

  constructor(
    errors: readonly unknown[],
    readonly recoveryPath: string,
    cause: unknown,
  ) {
    super(
      errors,
      `Sync transaction failed before commit and recovery is incomplete. Preserved recovery artifacts at ${recoveryPath}`,
      { cause },
    );
    this.name = "SyncRollbackError";
  }
}

export class SyncPostCommitCleanupError extends AggregateError {
  readonly committed = true as const;

  constructor(
    errors: readonly unknown[],
    readonly recoveryPath: string,
    cause: unknown,
  ) {
    super(
      errors,
      `Sync transaction committed, but cleanup failed. Recovery artifacts may remain at ${recoveryPath}`,
      { cause },
    );
    this.name = "SyncPostCommitCleanupError";
  }
}

export interface SyncTransactionOptions {
  websiteRoot: string;
  writer(candidate: SyncCandidatePaths): void | Promise<void>;
  validator(candidate: SyncCandidatePaths): void | Promise<void>;
  /** Runs after candidate validation and immediately before backups or installs. */
  beforeCommit?(): void | Promise<void>;
  /** Runs after all destinations are installed but before the transaction commits. */
  beforeFinalize?(): void | Promise<void>;
  /** Test seam for deterministic rollback coverage; called after each rename. */
  afterCommitStep?(step: SyncCommitStep): void | Promise<void>;
  /** Narrow fault seam called immediately before a guarded filesystem mutation. */
  beforeFilesystemOperation?(
    operation: SyncFilesystemOperationContext,
  ): void | Promise<void>;
}

interface ProjectionDestination {
  name: SyncDestination;
  candidate: string;
  current: string;
  backup: string;
}

interface FilesystemOperationScope {
  phase: SyncFilesystemPhase;
  destination?: SyncDestination;
}

type FilesystemFaultInjector =
  SyncTransactionOptions["beforeFilesystemOperation"];

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

async function injectFilesystemFault(
  faultInjector: FilesystemFaultInjector,
  operation: SyncFilesystemOperationContext,
): Promise<void> {
  await faultInjector?.(operation);
}

async function safeMkdir(
  websiteRoot: string,
  target: string,
  options: { recursive?: boolean },
  faultInjector: FilesystemFaultInjector,
  scope: FilesystemOperationScope,
): Promise<void> {
  assertTransactionPath(websiteRoot, target);
  await injectFilesystemFault(faultInjector, {
    operation: "mkdir",
    ...scope,
    path: target,
  });
  await fs.mkdir(target, options);
}

async function safeRename(
  websiteRoot: string,
  source: string,
  destination: string,
  faultInjector: FilesystemFaultInjector,
  scope: FilesystemOperationScope,
): Promise<void> {
  assertTransactionPath(websiteRoot, source);
  assertTransactionPath(websiteRoot, destination);
  await injectFilesystemFault(faultInjector, {
    operation: "rename",
    ...scope,
    path: destination,
    sourcePath: source,
  });
  await fs.rename(source, destination);
}

async function safeRemove(
  websiteRoot: string,
  target: string,
  faultInjector: FilesystemFaultInjector,
  scope: FilesystemOperationScope,
): Promise<void> {
  assertTransactionPath(websiteRoot, target);
  await injectFilesystemFault(faultInjector, {
    operation: "remove",
    ...scope,
    path: target,
  });
  await fs.rm(target, { recursive: true, force: true });
}

async function safeRmdir(
  websiteRoot: string,
  target: string,
  faultInjector: FilesystemFaultInjector,
  scope: FilesystemOperationScope,
): Promise<void> {
  assertTransactionPath(websiteRoot, target);
  await injectFilesystemFault(faultInjector, {
    operation: "rmdir",
    ...scope,
    path: target,
  });
  await fs.rmdir(target);
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
  faultInjector: FilesystemFaultInjector,
  scope: FilesystemOperationScope,
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

    await safeMkdir(websiteRoot, current, {}, faultInjector, scope);
    createdParents.push(current);
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function recoveryStepError(message: string, cause: unknown): Error {
  return new Error(`${message}: ${errorMessage(cause)}`, { cause });
}

class ArtifactCleanupError extends Error {
  constructor(
    readonly artifactPath: string,
    cause: unknown,
  ) {
    super(
      `Failed to clean transaction artifact ${artifactPath}: ${errorMessage(cause)}`,
      { cause },
    );
    this.name = "ArtifactCleanupError";
  }
}

async function removeCreatedParents(
  websiteRoot: string,
  createdParents: readonly string[],
  faultInjector: FilesystemFaultInjector,
): Promise<unknown[]> {
  const errors: unknown[] = [];

  for (const parent of [...createdParents].reverse()) {
    try {
      await safeRmdir(websiteRoot, parent, faultInjector, {
        phase: "rollback-parent-cleanup",
      });
    } catch (error) {
      if (
        error instanceof Error &&
        "code" in error &&
        (error.code === "ENOENT" || error.code === "ENOTEMPTY")
      ) {
        continue;
      }
      errors.push(
        recoveryStepError(
          `Failed to remove transaction-created parent ${parent} during rollback`,
          error,
        ),
      );
    }
  }

  return errors;
}

async function rollbackProjection(
  websiteRoot: string,
  destinations: readonly ProjectionDestination[],
  installed: ReadonlySet<SyncDestination>,
  backedUp: ReadonlySet<SyncDestination>,
  createdParents: readonly string[],
  faultInjector: FilesystemFaultInjector,
): Promise<unknown[]> {
  const errors: unknown[] = [];
  const failedRemovals = new Set<SyncDestination>();

  for (const destination of [...destinations].reverse()) {
    if (installed.has(destination.name)) {
      try {
        await safeRemove(
          websiteRoot,
          destination.current,
          faultInjector,
          {
            phase: "rollback-remove",
            destination: destination.name,
          },
        );
      } catch (error) {
        failedRemovals.add(destination.name);
        errors.push(
          recoveryStepError(
            `Failed to remove installed ${destination.name} projection at ${destination.current} during rollback`,
            error,
          ),
        );
      }
    }
  }

  for (const destination of [...destinations].reverse()) {
    if (
      !backedUp.has(destination.name) ||
      failedRemovals.has(destination.name)
    ) {
      continue;
    }

    try {
      await ensureDestinationParent(
        websiteRoot,
        destination.current,
        [],
        faultInjector,
        {
          phase: "rollback-restore",
          destination: destination.name,
        },
      );
      await safeRename(
        websiteRoot,
        destination.backup,
        destination.current,
        faultInjector,
        {
          phase: "rollback-restore",
          destination: destination.name,
        },
      );
    } catch (error) {
      errors.push(
        recoveryStepError(
          `Failed to restore ${destination.name} backup from ${destination.backup} during rollback`,
          error,
        ),
      );
    }
  }

  errors.push(
    ...(await removeCreatedParents(
      websiteRoot,
      createdParents,
      faultInjector,
    )),
  );
  return errors;
}

async function removeSyncTmpIfCreated(
  websiteRoot: string,
  syncTmp: string,
  syncTmpCreated: boolean,
  faultInjector: FilesystemFaultInjector,
  phase: "precommit-cleanup" | "postcommit-cleanup",
): Promise<void> {
  if (!syncTmpCreated) {
    return;
  }

  try {
    await safeRmdir(websiteRoot, syncTmp, faultInjector, { phase });
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

async function cleanupTransactionArtifacts(
  websiteRoot: string,
  transactionRoot: string,
  syncTmp: string,
  syncTmpCreated: boolean,
  faultInjector: FilesystemFaultInjector,
  phase: "precommit-cleanup" | "postcommit-cleanup",
): Promise<void> {
  try {
    await safeRemove(websiteRoot, transactionRoot, faultInjector, { phase });
  } catch (error) {
    throw new ArtifactCleanupError(transactionRoot, error);
  }

  try {
    await removeSyncTmpIfCreated(
      websiteRoot,
      syncTmp,
      syncTmpCreated,
      faultInjector,
      phase,
    );
  } catch (error) {
    throw new ArtifactCleanupError(syncTmp, error);
  }
}

export async function withSyncTransaction(
  options: SyncTransactionOptions,
): Promise<void> {
  const websiteRoot = path.resolve(resolveExistingPath(options.websiteRoot));
  const syncTmp = path.join(websiteRoot, ".sync-tmp");
  const transactionRoot = path.join(syncTmp, randomUUID());
  const candidateRoot = path.join(transactionRoot, "candidate");
  const backupRoot = path.join(transactionRoot, "backup");
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
  let syncTmpCreated = false;
  let syncTmpReady = false;
  let committed = false;

  try {
    assertTransactionPath(websiteRoot, syncTmp);
    syncTmpCreated = !(await pathExists(syncTmp));
    await safeMkdir(
      websiteRoot,
      syncTmp,
      { recursive: true },
      options.beforeFilesystemOperation,
      { phase: "setup" },
    );
    syncTmpReady = true;
    await safeMkdir(
      websiteRoot,
      candidateRoot,
      { recursive: true },
      options.beforeFilesystemOperation,
      { phase: "setup" },
    );
    await safeMkdir(
      websiteRoot,
      canonicalCandidate.metadata,
      { recursive: true },
      options.beforeFilesystemOperation,
      {
        phase: "setup",
        destination: "metadata",
      },
    );
    await safeMkdir(
      websiteRoot,
      canonicalCandidate.content,
      { recursive: true },
      options.beforeFilesystemOperation,
      {
        phase: "setup",
        destination: "content",
      },
    );
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
    await options.beforeCommit?.();

    for (const destination of destinations) {
      assertTransactionPath(websiteRoot, destination.candidate);
      assertTransactionPath(websiteRoot, destination.current);
      assertTransactionPath(websiteRoot, destination.backup);
    }

    await safeMkdir(
      websiteRoot,
      backupRoot,
      { recursive: true },
      options.beforeFilesystemOperation,
      { phase: "backup" },
    );
    for (const destination of destinations) {
      if (!(await pathExists(destination.current))) {
        continue;
      }

      await safeRename(
        websiteRoot,
        destination.current,
        destination.backup,
        options.beforeFilesystemOperation,
        { phase: "backup", destination: destination.name },
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
        options.beforeFilesystemOperation,
        { phase: "install", destination: destination.name },
      );
      await safeRename(
        websiteRoot,
        destination.candidate,
        destination.current,
        options.beforeFilesystemOperation,
        { phase: "install", destination: destination.name },
      );
      installed.add(destination.name);
      await options.afterCommitStep?.({
        phase: "install",
        destination: destination.name,
      });
    }

    await options.beforeFinalize?.();

    // Explicit commit point: all three candidate projections are installed and finalized.
    committed = true;
  } catch (error) {
    if (committed || !syncTmpReady) {
      throw error;
    }

    const rollbackErrors = await rollbackProjection(
      websiteRoot,
      destinations,
      installed,
      backedUp,
      createdParents,
      options.beforeFilesystemOperation,
    );
    if (rollbackErrors.length > 0) {
      throw new SyncRollbackError(
        [error, ...rollbackErrors],
        transactionRoot,
        error,
      );
    }

    try {
      await cleanupTransactionArtifacts(
        websiteRoot,
        transactionRoot,
        syncTmp,
        syncTmpCreated,
        options.beforeFilesystemOperation,
        "precommit-cleanup",
      );
    } catch (cleanupError) {
      const recoveryPath =
        cleanupError instanceof ArtifactCleanupError
          ? cleanupError.artifactPath
          : transactionRoot;
      throw new SyncRollbackError(
        [
          error,
          recoveryStepError(
            `Rollback succeeded, but transaction artifact cleanup failed at ${transactionRoot}`,
            cleanupError,
          ),
        ],
        recoveryPath,
        error,
      );
    }
    throw error;
  }

  try {
    await cleanupTransactionArtifacts(
      websiteRoot,
      transactionRoot,
      syncTmp,
      syncTmpCreated,
      options.beforeFilesystemOperation,
      "postcommit-cleanup",
    );
  } catch (cleanupError) {
    const recoveryPath =
      cleanupError instanceof ArtifactCleanupError
        ? cleanupError.artifactPath
        : transactionRoot;
    throw new SyncPostCommitCleanupError(
      [
        recoveryStepError(
          `Committed projection cleanup failed at ${transactionRoot}`,
          cleanupError,
        ),
      ],
      recoveryPath,
      cleanupError,
    );
  }
}
