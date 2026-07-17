import fs from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  assertNoUnexpectedRemovals,
  type StableLessonAssignment,
  type SyncCandidatePaths,
  type SyncDestination,
  type SyncFilesystemOperationContext,
  type SyncPostCommitCleanupError,
  type SyncRollbackError,
  withSyncTransaction,
} from "../../scripts/lib/transaction";

const temporaryRoots: string[] = [];

async function makeWebsiteRoot(): Promise<string> {
  const root = await fs.mkdtemp(
    path.join(await fs.realpath(tmpdir()), "personal-blog-transaction-"),
  );
  temporaryRoots.push(root);
  return root;
}

async function writeFile(
  filename: string,
  contents: string | Uint8Array,
): Promise<void> {
  await fs.mkdir(path.dirname(filename), { recursive: true });
  await fs.writeFile(filename, contents);
}

async function seedProjection(websiteRoot: string): Promise<void> {
  await Promise.all([
    writeFile(
      path.join(websiteRoot, "metadata", "lesson-0001.yml"),
      new Uint8Array([0, 1, 2, 255]),
    ),
    writeFile(
      path.join(websiteRoot, "src/content/ai-daily", "lesson-0001.md"),
      "old content\n",
    ),
    writeFile(path.join(websiteRoot, "sync-index.json"), '{"old":true}\n'),
    writeFile(path.join(websiteRoot, "public", "unrelated.txt"), "keep me\n"),
  ]);
}

async function snapshotFiles(root: string): Promise<Record<string, string>> {
  const snapshot: Record<string, string> = {};

  async function visit(directory: string): Promise<void> {
    for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
      const absolute = path.join(directory, entry.name);
      const relative = path.relative(root, absolute);
      if (entry.isDirectory()) {
        await visit(absolute);
      } else if (entry.isFile()) {
        snapshot[relative] = (await fs.readFile(absolute)).toString("base64");
      }
    }
  }

  await visit(root);
  return snapshot;
}

async function writeCandidate(
  candidate: SyncCandidatePaths,
  marker: string,
): Promise<void> {
  await Promise.all([
    writeFile(
      path.join(candidate.metadata, "lesson-0002.yml"),
      `${marker} metadata\n`,
    ),
    writeFile(
      path.join(candidate.content, "lesson-0002.md"),
      `${marker} content\n`,
    ),
    writeFile(candidate.index, JSON.stringify({ marker }) + "\n"),
  ]);
}

async function validateCandidate(
  candidate: SyncCandidatePaths,
  marker: string,
): Promise<void> {
  const values = await Promise.all([
    fs.readFile(path.join(candidate.metadata, "lesson-0002.yml"), "utf8"),
    fs.readFile(path.join(candidate.content, "lesson-0002.md"), "utf8"),
    fs.readFile(candidate.index, "utf8"),
  ]);

  expect(values).toEqual([
    `${marker} metadata\n`,
    `${marker} content\n`,
    JSON.stringify({ marker }) + "\n",
  ]);
}

async function expectNoTransactionArtifacts(websiteRoot: string): Promise<void> {
  const syncTmp = path.join(websiteRoot, ".sync-tmp");
  try {
    expect(await fs.readdir(syncTmp)).toEqual([]);
  } catch (error) {
    expect(error).toMatchObject({ code: "ENOENT" });
  }
}

function assignment(
  overrides: Partial<StableLessonAssignment> = {},
): StableLessonAssignment {
  return {
    id: "lesson-0001",
    lesson: 1,
    source: { file: "2026-07-06.md", section: 1 },
    sourceHash: "sha256:old",
    ...overrides,
  };
}

const syncDestinations = ["metadata", "content", "index"] as const satisfies
  readonly SyncDestination[];

function currentDestinationPath(
  websiteRoot: string,
  destination: SyncDestination,
): string {
  if (destination === "metadata") {
    return path.join(websiteRoot, "metadata");
  }
  if (destination === "content") {
    return path.join(websiteRoot, "src/content/ai-daily");
  }
  return path.join(websiteRoot, "sync-index.json");
}

function oldProjectionFile(
  root: string,
  destination: SyncDestination,
): string {
  if (destination === "metadata") {
    return path.join(root, "metadata", "lesson-0001.yml");
  }
  if (destination === "content") {
    return path.join(root, "content", "lesson-0001.md");
  }
  return path.join(root, "sync-index.json");
}

function currentOldProjectionFile(
  websiteRoot: string,
  destination: SyncDestination,
): string {
  if (destination === "metadata") {
    return path.join(websiteRoot, "metadata", "lesson-0001.yml");
  }
  if (destination === "content") {
    return path.join(
      websiteRoot,
      "src/content/ai-daily",
      "lesson-0001.md",
    );
  }
  return path.join(websiteRoot, "sync-index.json");
}

function currentNewProjectionFile(
  websiteRoot: string,
  destination: SyncDestination,
): string {
  if (destination === "metadata") {
    return path.join(websiteRoot, "metadata", "lesson-0002.yml");
  }
  if (destination === "content") {
    return path.join(
      websiteRoot,
      "src/content/ai-daily",
      "lesson-0002.md",
    );
  }
  return path.join(websiteRoot, "sync-index.json");
}

function expectedOldProjection(destination: SyncDestination): Buffer {
  if (destination === "metadata") {
    return Buffer.from([0, 1, 2, 255]);
  }
  if (destination === "content") {
    return Buffer.from("old content\n");
  }
  return Buffer.from('{"old":true}\n');
}

function expectedNewProjection(destination: SyncDestination): Buffer {
  if (destination === "metadata") {
    return Buffer.from("new metadata\n");
  }
  if (destination === "content") {
    return Buffer.from("new content\n");
  }
  return Buffer.from('{"marker":"new"}\n');
}

async function captureError(work: () => Promise<void>): Promise<unknown> {
  try {
    await work();
    return undefined;
  } catch (error) {
    return error;
  }
}

async function expectOtherDestinationsRestored(
  websiteRoot: string,
  failedDestination: SyncDestination,
): Promise<void> {
  for (const destination of syncDestinations) {
    if (destination === failedDestination) {
      continue;
    }
    expect(await fs.readFile(currentOldProjectionFile(websiteRoot, destination)))
      .toEqual(expectedOldProjection(destination));
  }
}

afterEach(async () => {
  for (const root of temporaryRoots.splice(0)) {
    await fs.rm(root, { recursive: true, force: true });
  }
});

describe("assertNoUnexpectedRemovals", () => {
  it("keeps the implementation-plan string-array call and blocks disappearance", () => {
    expect(() =>
      assertNoUnexpectedRemovals(
        ["lesson-0001", "lesson-0002"],
        ["lesson-0001"],
        false,
      ),
    ).toThrow(/unexpected lesson removal.*lesson-0002/i);
  });

  it("rejects duplicate IDs in either string set", () => {
    expect(() =>
      assertNoUnexpectedRemovals(
        ["lesson-0001", "lesson-0001"],
        ["lesson-0001"],
        false,
      ),
    ).toThrow(/duplicate lesson id.*lesson-0001/i);

    expect(() =>
      assertNoUnexpectedRemovals(
        ["lesson-0001"],
        ["lesson-0001", "lesson-0001"],
        true,
      ),
    ).toThrow(/duplicate lesson id.*lesson-0001/i);
  });

  it("protects source file/section for a known ID", () => {
    const previous = [assignment()];

    expect(() =>
      assertNoUnexpectedRemovals({
        previous,
        current: [
          assignment({
            source: { file: "2026-07-07.md", section: 2 },
          }),
        ],
        allowRemovals: false,
      }),
    ).toThrow(
      /unexpected lesson reassignment.*2026-07-06\.md.*section 1.*2026-07-07\.md.*section 2/is,
    );
  });

  it.each([
    {
      label: "previous",
      previous: [assignment({ lesson: 2 })],
      current: [] as StableLessonAssignment[],
    },
    {
      label: "current",
      previous: [] as StableLessonAssignment[],
      current: [assignment({ lesson: 2 })],
    },
    {
      label: "both",
      previous: [assignment({ lesson: 2 })],
      current: [assignment({ lesson: 2 })],
    },
  ])(
    "rejects an ID/lesson mismatch in $label assignments even when removals are enabled",
    ({ previous, current }) => {
      expect(() =>
        assertNoUnexpectedRemovals({
          previous,
          current,
          allowRemovals: true,
        }),
      ).toThrow(/invalid lesson assignment.*lesson-0001.*lesson 2/i);
    },
  );

  it("allows source hash changes because hashes describe edits, not identity", () => {
    expect(() =>
      assertNoUnexpectedRemovals({
        previous: [assignment({ sourceHash: "sha256:old" })],
        current: [assignment({ sourceHash: "sha256:new" })],
        allowRemovals: false,
      }),
    ).not.toThrow();
  });

  it("allows known removals and reassignments only when explicitly enabled", () => {
    expect(() =>
      assertNoUnexpectedRemovals({
        previous: [assignment(), assignment({ id: "lesson-0002", lesson: 2 })],
        current: [
          assignment({
            source: { file: "2026-07-09.md", section: 3 },
          }),
        ],
        allowRemovals: true,
      }),
    ).not.toThrow();
  });

  it("still rejects malformed or duplicate current assignments when removals are enabled", () => {
    expect(() =>
      assertNoUnexpectedRemovals({
        previous: [assignment()],
        current: [assignment({ id: "not-a-stable-id" })],
        allowRemovals: true,
      }),
    ).toThrow(/invalid lesson assignment.*id/i);

    expect(() =>
      assertNoUnexpectedRemovals({
        previous: [assignment()],
        current: [assignment({ source: { file: "", section: 0 } })],
        allowRemovals: true,
      }),
    ).toThrow(/invalid lesson assignment.*source/i);

    expect(() =>
      assertNoUnexpectedRemovals({
        previous: [assignment()],
        current: [assignment(), assignment()],
        allowRemovals: true,
      }),
    ).toThrow(/duplicate lesson id.*lesson-0001/i);
  });
});

describe("withSyncTransaction", () => {
  it.each(syncDestinations)(
    "preserves the %s backup when rollback removal fails",
    async (failedDestination) => {
      const websiteRoot = await makeWebsiteRoot();
      await seedProjection(websiteRoot);

      const error = await captureError(() =>
        withSyncTransaction({
          websiteRoot,
          writer: (candidate) => writeCandidate(candidate, "new"),
          validator: (candidate) => validateCandidate(candidate, "new"),
          afterCommitStep: ({ phase, destination }) => {
            if (phase === "install" && destination === "index") {
              throw new Error("forced commit failure");
            }
          },
          beforeFilesystemOperation: (
            operation: SyncFilesystemOperationContext,
          ) => {
            if (
              operation.phase === "rollback-remove" &&
              operation.operation === "remove" &&
              operation.destination === failedDestination
            ) {
              throw new Error(
                `forced rollback removal failure for ${failedDestination}`,
              );
            }
          },
        }),
      );

      expect(error).toBeInstanceOf(AggregateError);
      expect(error).toMatchObject({
        name: "SyncRollbackError",
        committed: false,
        recoveryPath: expect.any(String),
      });
      const recoveryPath = (error as SyncRollbackError).recoveryPath;
      expect(await fs.stat(recoveryPath)).toMatchObject({});
      expect(
        await fs.readFile(
          oldProjectionFile(
            path.join(recoveryPath, "backup"),
            failedDestination,
          ),
        ),
      ).toEqual(expectedOldProjection(failedDestination));
      expect(
        await fs.readFile(
          currentNewProjectionFile(websiteRoot, failedDestination),
        ),
      ).toEqual(expectedNewProjection(failedDestination));
      await expectOtherDestinationsRestored(
        websiteRoot,
        failedDestination,
      );
    },
  );

  it.each(syncDestinations)(
    "preserves the %s backup when rollback restoration fails",
    async (failedDestination) => {
      const websiteRoot = await makeWebsiteRoot();
      await seedProjection(websiteRoot);

      const error = await captureError(() =>
        withSyncTransaction({
          websiteRoot,
          writer: (candidate) => writeCandidate(candidate, "new"),
          validator: (candidate) => validateCandidate(candidate, "new"),
          afterCommitStep: ({ phase, destination }) => {
            if (phase === "install" && destination === "index") {
              throw new Error("forced commit failure");
            }
          },
          beforeFilesystemOperation: (
            operation: SyncFilesystemOperationContext,
          ) => {
            if (
              operation.phase === "rollback-restore" &&
              operation.operation === "rename" &&
              operation.destination === failedDestination
            ) {
              throw new Error(
                `forced rollback restoration failure for ${failedDestination}`,
              );
            }
          },
        }),
      );

      expect(error).toBeInstanceOf(AggregateError);
      expect(error).toMatchObject({
        name: "SyncRollbackError",
        committed: false,
        recoveryPath: expect.any(String),
      });
      const recoveryPath = (error as SyncRollbackError).recoveryPath;
      expect(await fs.stat(recoveryPath)).toMatchObject({});
      expect(
        await fs.readFile(
          oldProjectionFile(
            path.join(recoveryPath, "backup"),
            failedDestination,
          ),
        ),
      ).toEqual(expectedOldProjection(failedDestination));
      await expect(
        fs.stat(currentDestinationPath(websiteRoot, failedDestination)),
      ).rejects.toMatchObject({ code: "ENOENT" });
      await expectOtherDestinationsRestored(
        websiteRoot,
        failedDestination,
      );
    },
  );

  it("aggregates independent rollback failures and preserves both backups", async () => {
    const websiteRoot = await makeWebsiteRoot();
    await seedProjection(websiteRoot);

    const error = await captureError(() =>
      withSyncTransaction({
        websiteRoot,
        writer: (candidate) => writeCandidate(candidate, "new"),
        validator: (candidate) => validateCandidate(candidate, "new"),
        afterCommitStep: ({ phase, destination }) => {
          if (phase === "install" && destination === "index") {
            throw new Error("forced commit failure");
          }
        },
        beforeFilesystemOperation: (
          operation: SyncFilesystemOperationContext,
        ) => {
          if (
            operation.phase === "rollback-remove" &&
            operation.destination === "metadata"
          ) {
            throw new Error("forced metadata removal failure");
          }
          if (
            operation.phase === "rollback-restore" &&
            operation.destination === "content"
          ) {
            throw new Error("forced content restoration failure");
          }
        },
      }),
    );

    expect(error).toBeInstanceOf(AggregateError);
    const rollbackError = error as SyncRollbackError;
    const messages = rollbackError.errors.map(String).join("\n");
    expect(messages).toMatch(/forced commit failure/);
    expect(messages).toMatch(/forced metadata removal failure/);
    expect(messages).toMatch(/forced content restoration failure/);
    expect(
      await fs.readFile(
        oldProjectionFile(
          path.join(rollbackError.recoveryPath, "backup"),
          "metadata",
        ),
      ),
    ).toEqual(expectedOldProjection("metadata"));
    expect(
      await fs.readFile(
        oldProjectionFile(
          path.join(rollbackError.recoveryPath, "backup"),
          "content",
        ),
      ),
    ).toEqual(expectedOldProjection("content"));
    expect(await fs.readFile(currentOldProjectionFile(websiteRoot, "index")))
      .toEqual(expectedOldProjection("index"));
  });

  it("reports committed=true and preserves backups when post-commit cleanup fails", async () => {
    const websiteRoot = await makeWebsiteRoot();
    await seedProjection(websiteRoot);

    const error = await captureError(() =>
      withSyncTransaction({
        websiteRoot,
        writer: (candidate) => writeCandidate(candidate, "new"),
        validator: (candidate) => validateCandidate(candidate, "new"),
        beforeFilesystemOperation: (
          operation: SyncFilesystemOperationContext,
        ) => {
          if (
            operation.phase === "postcommit-cleanup" &&
            operation.operation === "remove"
          ) {
            throw new Error("forced post-commit cleanup failure");
          }
        },
      }),
    );

    expect(error).toBeInstanceOf(AggregateError);
    expect(error).toMatchObject({
      name: "SyncPostCommitCleanupError",
      committed: true,
      recoveryPath: expect.any(String),
    });
    const cleanupError = error as SyncPostCommitCleanupError;
    expect(await fs.stat(cleanupError.recoveryPath)).toMatchObject({});
    for (const destination of syncDestinations) {
      expect(
        await fs.readFile(currentNewProjectionFile(websiteRoot, destination)),
      ).toEqual(expectedNewProjection(destination));
      expect(
        await fs.readFile(
          oldProjectionFile(
            path.join(cleanupError.recoveryPath, "backup"),
            destination,
          ),
        ),
      ).toEqual(expectedOldProjection(destination));
    }
    expect(
      await fs.readFile(path.join(websiteRoot, "public", "unrelated.txt"), "utf8"),
    ).toBe("keep me\n");
  });

  it("reports the remaining sync-tmp artifact when parent cleanup fails after commit", async () => {
    const websiteRoot = await makeWebsiteRoot();
    await seedProjection(websiteRoot);

    const error = await captureError(() =>
      withSyncTransaction({
        websiteRoot,
        writer: (candidate) => writeCandidate(candidate, "new"),
        validator: (candidate) => validateCandidate(candidate, "new"),
        beforeFilesystemOperation: (
          operation: SyncFilesystemOperationContext,
        ) => {
          if (
            operation.phase === "postcommit-cleanup" &&
            operation.operation === "rmdir"
          ) {
            throw new Error("forced sync-tmp cleanup failure");
          }
        },
      }),
    );

    const cleanupError = error as SyncPostCommitCleanupError;
    expect(cleanupError).toMatchObject({
      name: "SyncPostCommitCleanupError",
      committed: true,
      recoveryPath: path.join(websiteRoot, ".sync-tmp"),
    });
    expect(await fs.readdir(cleanupError.recoveryPath)).toEqual([]);
    for (const destination of syncDestinations) {
      expect(
        await fs.readFile(currentNewProjectionFile(websiteRoot, destination)),
      ).toEqual(expectedNewProjection(destination));
    }
  });

  it("cleans newly created transaction artifacts after an initial setup failure", async () => {
    const websiteRoot = await makeWebsiteRoot();
    await seedProjection(websiteRoot);
    const before = await snapshotFiles(websiteRoot);
    let writerCalled = false;

    await expect(
      withSyncTransaction({
        websiteRoot,
        writer: async (candidate) => {
          writerCalled = true;
          await writeCandidate(candidate, "new");
        },
        validator: (candidate) => validateCandidate(candidate, "new"),
        beforeFilesystemOperation: (
          operation: SyncFilesystemOperationContext,
        ) => {
          if (
            operation.phase === "setup" &&
            operation.operation === "mkdir" &&
            operation.destination === "content"
          ) {
            throw new Error("forced initial setup failure");
          }
        },
      }),
    ).rejects.toThrow(/forced initial setup failure/);

    expect(writerCalled).toBe(false);
    expect(await snapshotFiles(websiteRoot)).toEqual(before);
    await expectNoTransactionArtifacts(websiteRoot);
  });

  it("freezes callback paths so mutation cannot redirect validation or commit", async () => {
    const websiteRoot = await makeWebsiteRoot();

    await withSyncTransaction({
      websiteRoot,
      writer: async (candidate) => {
        const canonicalMetadata = candidate.metadata;
        const redirectedMetadata = path.join(
          path.dirname(canonicalMetadata),
          "redirected-metadata",
        );
        await fs.mkdir(redirectedMetadata);
        await writeFile(
          path.join(redirectedMetadata, "lesson-redirected.yml"),
          "redirected metadata\n",
        );

        const mutable = candidate as { metadata: string };
        expect(() => {
          mutable.metadata = redirectedMetadata;
        }).toThrow(TypeError);
        expect(Object.isFrozen(candidate)).toBe(true);
        expect(candidate.metadata).toBe(canonicalMetadata);

        await writeCandidate(candidate, "canonical");
      },
      validator: (candidate) => validateCandidate(candidate, "canonical"),
    });

    expect(
      await fs.readFile(
        path.join(websiteRoot, "metadata", "lesson-0002.yml"),
        "utf8",
      ),
    ).toBe("canonical metadata\n");
    await expect(
      fs.stat(path.join(websiteRoot, "metadata", "lesson-redirected.yml")),
    ).rejects.toMatchObject({ code: "ENOENT" });
    await expectNoTransactionArtifacts(websiteRoot);
  });

  it("leaves the exact old projection intact when candidate validation fails", async () => {
    const websiteRoot = await makeWebsiteRoot();
    await seedProjection(websiteRoot);
    const before = await snapshotFiles(websiteRoot);

    await expect(
      withSyncTransaction({
        websiteRoot,
        writer: (candidate) => writeCandidate(candidate, "invalid"),
        validator: async () => {
          throw new Error("candidate validation failed");
        },
      }),
    ).rejects.toThrow(/candidate validation failed/);

    expect(await snapshotFiles(websiteRoot)).toEqual(before);
    await expectNoTransactionArtifacts(websiteRoot);
  });

  it("rolls all three destinations back after a mid-commit failure", async () => {
    const websiteRoot = await makeWebsiteRoot();
    await seedProjection(websiteRoot);
    const before = await snapshotFiles(websiteRoot);

    await expect(
      withSyncTransaction({
        websiteRoot,
        writer: (candidate) => writeCandidate(candidate, "new"),
        validator: (candidate) => validateCandidate(candidate, "new"),
        afterCommitStep: ({ phase, destination }) => {
          if (phase === "install" && destination === "metadata") {
            throw new Error("forced mid-commit failure");
          }
        },
      }),
    ).rejects.toThrow(/forced mid-commit failure/);

    expect(await snapshotFiles(websiteRoot)).toEqual(before);
    await expectNoTransactionArtifacts(websiteRoot);
  });

  it("replaces metadata, content, and index together and preserves unrelated files", async () => {
    const websiteRoot = await makeWebsiteRoot();
    await seedProjection(websiteRoot);

    await withSyncTransaction({
      websiteRoot,
      writer: (candidate) => writeCandidate(candidate, "new"),
      validator: (candidate) => validateCandidate(candidate, "new"),
    });

    expect(
      await fs.readFile(
        path.join(websiteRoot, "metadata", "lesson-0002.yml"),
        "utf8",
      ),
    ).toBe("new metadata\n");
    await expect(
      fs.stat(path.join(websiteRoot, "metadata", "lesson-0001.yml")),
    ).rejects.toMatchObject({ code: "ENOENT" });
    expect(
      await fs.readFile(
        path.join(websiteRoot, "src/content/ai-daily", "lesson-0002.md"),
        "utf8",
      ),
    ).toBe("new content\n");
    expect(await fs.readFile(path.join(websiteRoot, "sync-index.json"), "utf8"))
      .toBe('{"marker":"new"}\n');
    expect(
      await fs.readFile(path.join(websiteRoot, "public", "unrelated.txt"), "utf8"),
    ).toBe("keep me\n");
    await expectNoTransactionArtifacts(websiteRoot);
  });

  it("installs all three candidates when destinations were initially absent", async () => {
    const websiteRoot = await makeWebsiteRoot();
    await writeFile(path.join(websiteRoot, "unrelated.txt"), "unchanged");

    await withSyncTransaction({
      websiteRoot,
      writer: (candidate) => writeCandidate(candidate, "first"),
      validator: (candidate) => validateCandidate(candidate, "first"),
    });

    expect(
      await fs.readFile(
        path.join(websiteRoot, "metadata", "lesson-0002.yml"),
        "utf8",
      ),
    ).toBe("first metadata\n");
    expect(
      await fs.readFile(
        path.join(websiteRoot, "src/content/ai-daily", "lesson-0002.md"),
        "utf8",
      ),
    ).toBe("first content\n");
    expect(await fs.readFile(path.join(websiteRoot, "sync-index.json"), "utf8"))
      .toBe('{"marker":"first"}\n');
    expect(await fs.readFile(path.join(websiteRoot, "unrelated.txt"), "utf8"))
      .toBe("unchanged");
    await expectNoTransactionArtifacts(websiteRoot);
  });

  it("rejects a candidate symlink escape before modifying destinations", async () => {
    const websiteRoot = await makeWebsiteRoot();
    const outsideRoot = await makeWebsiteRoot();
    await seedProjection(websiteRoot);
    await writeFile(path.join(outsideRoot, "sentinel.txt"), "outside unchanged");
    const before = await snapshotFiles(websiteRoot);

    await expect(
      withSyncTransaction({
        websiteRoot,
        writer: async (candidate) => {
          await fs.rm(candidate.metadata, { recursive: true });
          await fs.symlink(outsideRoot, candidate.metadata);
          await writeFile(
            path.join(candidate.content, "lesson-0002.md"),
            "candidate content",
          );
          await writeFile(candidate.index, "{}\n");
        },
        validator: async () => undefined,
      }),
    ).rejects.toThrow(/candidate.*symlink|outside website root/i);

    expect(await snapshotFiles(websiteRoot)).toEqual(before);
    expect(await fs.readFile(path.join(outsideRoot, "sentinel.txt"), "utf8"))
      .toBe("outside unchanged");
    await expectNoTransactionArtifacts(websiteRoot);
  });

  it("rejects a destination parent symlink escape before commit", async () => {
    const websiteRoot = await makeWebsiteRoot();
    const outsideRoot = await makeWebsiteRoot();
    await writeFile(path.join(websiteRoot, "metadata", "old.yml"), "old");
    await writeFile(path.join(websiteRoot, "sync-index.json"), "old index");
    await fs.symlink(outsideRoot, path.join(websiteRoot, "src"));
    await writeFile(path.join(outsideRoot, "sentinel.txt"), "outside unchanged");
    const before = await snapshotFiles(websiteRoot);

    await expect(
      withSyncTransaction({
        websiteRoot,
        writer: (candidate) => writeCandidate(candidate, "new"),
        validator: (candidate) => validateCandidate(candidate, "new"),
      }),
    ).rejects.toThrow(/outside website root/i);

    expect(await snapshotFiles(websiteRoot)).toEqual(before);
    expect(await fs.readFile(path.join(outsideRoot, "sentinel.txt"), "utf8"))
      .toBe("outside unchanged");
    await expectNoTransactionArtifacts(websiteRoot);
  });

  it("exposes a physical guard that rejects parent traversal in writer paths", async () => {
    const websiteRoot = await makeWebsiteRoot();

    await expect(
      withSyncTransaction({
        websiteRoot,
        writer: async (candidate) => {
          candidate.assertSafeWritePath(
            `${candidate.metadata}/../../../../../escaped.txt`,
          );
        },
        validator: async () => undefined,
      }),
    ).rejects.toThrow(/outside website root/i);

    await expectNoTransactionArtifacts(websiteRoot);
  });
});
