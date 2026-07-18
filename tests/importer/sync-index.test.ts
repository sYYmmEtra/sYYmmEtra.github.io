import { existsSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { describe, expect, it } from "vitest";

const repositoryRoot = path.resolve(import.meta.dirname, "../..");
const modulePath = path.join(repositoryRoot, "scripts/lib/sync-index.ts");

type SyncIndexModule = {
  SyncIndexSchema: { safeParse: (value: unknown) => { success: boolean } };
};

async function syncIndexModule(): Promise<SyncIndexModule | undefined> {
  if (!existsSync(modulePath)) return undefined;
  return import(pathToFileURL(modulePath).href) as Promise<SyncIndexModule>;
}

async function requireSyncIndexModule(): Promise<SyncIndexModule> {
  const loaded = await syncIndexModule();
  expect(loaded).toBeDefined();
  if (!loaded) throw new Error("sync-index module is unavailable");
  return loaded;
}

function lesson(lessonNumber: number, slug = `lesson-${lessonNumber}`): Record<string, unknown> {
  return {
    id: `lesson-${String(lessonNumber).padStart(4, "0")}`,
    lesson: lessonNumber,
    source: { file: "lessons/2026-07-06.md", section: lessonNumber },
    sourceHash: `sha256:${"a".repeat(64)}`,
    slug,
  };
}

describe("shared sync index schema", () => {
  it("preserves empty indexes for sync while validating strict canonical records", async () => {
    const { SyncIndexSchema } = await requireSyncIndexModule();
    expect(SyncIndexSchema.safeParse({ schemaVersion: 1, lessons: [] }).success).toBe(true);
    expect(SyncIndexSchema.safeParse({ schemaVersion: 1, lessons: [lesson(1)] }).success).toBe(true);
  });

  it("rejects unknown fields, out-of-order lessons, and noncanonical IDs", async () => {
    const { SyncIndexSchema } = await requireSyncIndexModule();
    expect(SyncIndexSchema.safeParse({ schemaVersion: 1, lessons: [{ ...lesson(1), unexpected: true }] }).success).toBe(false);
    expect(SyncIndexSchema.safeParse({ schemaVersion: 1, lessons: [lesson(2), lesson(1)] }).success).toBe(false);
    expect(SyncIndexSchema.safeParse({ schemaVersion: 1, lessons: [{ ...lesson(1), id: "lesson-0002" }] }).success).toBe(false);
  });
});
