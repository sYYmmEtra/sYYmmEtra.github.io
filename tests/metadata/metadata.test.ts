import { describe, expect, it } from "vitest";

import {
  SidecarMetadataSchema,
  applyEnrichment,
  assertUniqueMetadataSlugs,
  createMetadataSidecar,
  createMetadataSlug,
  reconcileMetadataWithSource,
  resolveMetadataDisplay,
  setSourceReviewStatus,
  type SidecarMetadata,
} from "../../scripts/lib/metadata";

const HASH_A = `sha256:${"a".repeat(64)}`;
const HASH_B = `sha256:${"b".repeat(64)}`;

function words(count: number): string {
  return Array.from({ length: count }, (_, index) => `word${index + 1}`).join(
    " ",
  );
}

function pendingSidecar(
  overrides: Partial<SidecarMetadata> = {},
): SidecarMetadata {
  return {
    id: "lesson-0013",
    source: {
      file: "lessons/2026-07-17.md",
      section: 1,
      hash: HASH_A,
    },
    lesson: 13,
    date: "2026-07-17",
    track: "A",
    depth: "L3",
    titleZh: "结构化输出",
    titleEn: null,
    summaryZh: "结构化输出保证格式合法，但不保证语义正确。",
    summaryEn: null,
    slug: "lesson-0013",
    tags: [],
    sourceStatus: "unreviewed",
    sourceStatusHash: HASH_A,
    metadataStatus: "pending",
    metadataSourceHash: null,
    featured: false,
    ...overrides,
  };
}

function currentSidecar(
  overrides: Partial<SidecarMetadata> = {},
): SidecarMetadata {
  return {
    ...pendingSidecar(),
    titleEn: "Structured Output and Tool Routing",
    summaryEn: words(34),
    slug: "structured-output-and-tool-routing",
    tags: ["structured-output", "tool-routing"],
    sourceStatus: "verified",
    metadataStatus: "current",
    metadataSourceHash: HASH_A,
    featured: true,
    ...overrides,
  };
}

describe("SidecarMetadataSchema", () => {
  it("rejects unknown keys", () => {
    expect(
      SidecarMetadataSchema.safeParse({
        ...pendingSidecar(),
        unexpected: true,
      }).success,
    ).toBe(false);
  });

  it("requires the canonical ID to agree with the lesson number", () => {
    expect(
      SidecarMetadataSchema.safeParse({
        ...pendingSidecar(),
        id: "lesson-0012",
      }).success,
    ).toBe(false);
    expect(
      SidecarMetadataSchema.safeParse({
        ...pendingSidecar({
          id: "lesson-10000",
          lesson: 10_000,
          slug: "lesson-10000",
        }),
      }).success,
    ).toBe(true);
  });

  it("requires the sidecar date to agree with source.file", () => {
    expect(
      SidecarMetadataSchema.safeParse({
        ...pendingSidecar(),
        date: "2026-07-18",
      }).success,
    ).toBe(false);
  });

  it("binds source review state to the current source hash", () => {
    expect(
      SidecarMetadataSchema.safeParse({
        ...pendingSidecar(),
        sourceStatus: "verified",
        sourceStatusHash: HASH_B,
      }).success,
    ).toBe(false);
  });

  it.each([
    ["English title", { titleEn: "Unexpected English" }],
    ["English summary", { summaryEn: words(30) }],
    ["tags", { tags: ["unexpected"] }],
    ["metadata hash", { metadataSourceHash: HASH_A }],
  ])("rejects pending metadata with %s", (_label, change) => {
    expect(
      SidecarMetadataSchema.safeParse({
        ...pendingSidecar(),
        ...change,
      }).success,
    ).toBe(false);
  });

  it.each([
    ["missing title", { titleEn: null }],
    ["missing summary", { summaryEn: null }],
    ["empty tags", { tags: [] }],
    ["wrong metadata hash", { metadataSourceHash: HASH_B }],
  ])("rejects current metadata with %s", (_label, change) => {
    expect(
      SidecarMetadataSchema.safeParse({
        ...currentSidecar(),
        ...change,
      }).success,
    ).toBe(false);
  });

  it("accepts needs-review with either no English set or a complete stale set", () => {
    const empty = {
      ...pendingSidecar(),
      metadataStatus: "needs-review",
    };
    const stale = {
      ...currentSidecar(),
      source: { ...currentSidecar().source, hash: HASH_B },
      sourceStatus: "unreviewed",
      sourceStatusHash: HASH_B,
      metadataStatus: "needs-review",
    };

    expect(SidecarMetadataSchema.safeParse(empty).success).toBe(true);
    expect(SidecarMetadataSchema.safeParse(stale).success).toBe(true);
  });

  it.each([
    ["partial stale English", { titleEn: "Stale title" }],
    [
      "current hash",
      {
        titleEn: "Stale title",
        summaryEn: words(30),
        tags: ["stale-tag"],
        metadataSourceHash: HASH_A,
      },
    ],
  ])("rejects needs-review with %s", (_label, change) => {
    expect(
      SidecarMetadataSchema.safeParse({
        ...pendingSidecar(),
        metadataStatus: "needs-review",
        ...change,
      }).success,
    ).toBe(false);
  });
});

describe("metadata creation and transitions", () => {
  it("creates a new unreviewed pending sidecar with Chinese fallback fields", () => {
    const sidecar = createMetadataSidecar({
      source: {
        file: "lessons/2026-07-17.md",
        section: 1,
        hash: HASH_A,
      },
      lesson: 13,
      date: "2026-07-17",
      track: "A",
      depth: "L3",
      titleZh: "结构化输出",
      summaryZh: "中文摘要",
    });

    expect(sidecar).toEqual(pendingSidecar({ summaryZh: "中文摘要" }));
  });

  it("uses valid pre-publication English enrichment to assign the initial slug", () => {
    const sidecar = createMetadataSidecar(
      {
        source: {
          file: "lessons/2026-07-17.md",
          section: 1,
          hash: HASH_A,
        },
        lesson: 13,
        date: "2026-07-17",
        track: "A",
        depth: "L3",
        titleZh: "结构化输出",
        summaryZh: "中文摘要",
      },
      {
        id: "lesson-0013",
        sourceHash: HASH_A,
        titleEn: "Café & Tool Routing",
        summaryEn: words(30),
        tags: ["tool-routing"],
      },
    );

    expect(sidecar).toMatchObject({
      titleEn: "Café & Tool Routing",
      slug: "cafe-and-tool-routing",
      metadataStatus: "current",
      metadataSourceHash: HASH_A,
    });
  });

  it("returns the same valid sidecar when the source hash is unchanged", () => {
    const sidecar = currentSidecar();

    expect(reconcileMetadataWithSource(sidecar, HASH_A)).toBe(sidecar);
  });

  it("invalidates English display and source review when the hash changes", () => {
    const sidecar = currentSidecar();

    const next = reconcileMetadataWithSource(sidecar, HASH_B);

    expect(next).toMatchObject({
      id: sidecar.id,
      slug: sidecar.slug,
      source: { ...sidecar.source, hash: HASH_B },
      sourceStatus: "unreviewed",
      sourceStatusHash: HASH_B,
      metadataStatus: "needs-review",
      titleEn: sidecar.titleEn,
      summaryEn: sidecar.summaryEn,
      tags: sidecar.tags,
      metadataSourceHash: HASH_A,
    });
  });

  it("moves pending metadata to needs-review without inventing stale English", () => {
    const next = reconcileMetadataWithSource(pendingSidecar(), HASH_B);

    expect(next).toMatchObject({
      metadataStatus: "needs-review",
      titleEn: null,
      summaryEn: null,
      tags: [],
      metadataSourceHash: null,
    });
  });

  it("sets source review only for the exact current source hash", () => {
    expect(() =>
      setSourceReviewStatus(pendingSidecar(), "verified", HASH_B),
    ).toThrow(/stale source review/i);

    expect(
      setSourceReviewStatus(pendingSidecar(), "partially-verified", HASH_A),
    ).toMatchObject({
      sourceStatus: "partially-verified",
      sourceStatusHash: HASH_A,
    });
  });

  it("applies exact-current enrichment without changing source review or slug", () => {
    const sidecar = pendingSidecar({
      sourceStatus: "partially-verified",
    });

    const enriched = applyEnrichment(sidecar, {
      id: sidecar.id,
      sourceHash: HASH_A,
      titleEn: "A New English Title",
      summaryEn: words(35),
      tags: ["new-title"],
    });

    expect(enriched).toMatchObject({
      slug: "lesson-0013",
      sourceStatus: "partially-verified",
      sourceStatusHash: HASH_A,
      metadataStatus: "current",
      metadataSourceHash: HASH_A,
    });
  });

  it("does not allow Codex output to set source review state", () => {
    expect(() =>
      applyEnrichment(pendingSidecar(), {
        id: "lesson-0013",
        sourceHash: HASH_A,
        titleEn: "A New English Title",
        summaryEn: words(35),
        tags: ["new-title"],
        sourceStatus: "verified",
      }),
    ).toThrow();
  });
});

describe("slug and display behavior", () => {
  it("creates deterministic ASCII slugs without transliterating Chinese", () => {
    expect(
      createMetadataSlug("  Café & O'Reilly’s APIs — déjà vu  ", "lesson-0013"),
    ).toBe("cafe-and-oreillys-apis-deja-vu");
    expect(createMetadataSlug("结构化输出", "lesson-0013")).toBe(
      "lesson-0013",
    );
  });

  it("collapses separators, trims them, and caps slugs at 96 characters", () => {
    const slug = createMetadataSlug(
      `-- ${"a".repeat(70)} ${"b".repeat(70)} --`,
      "lesson-0013",
    );

    expect(slug).toHaveLength(96);
    expect(slug).toMatch(/^[a-z0-9]+(?:-[a-z0-9]+)*$/);
  });

  it("returns English only for metadata bound to the current source hash", () => {
    expect(resolveMetadataDisplay(currentSidecar())).toEqual({
      title: "Structured Output and Tool Routing",
      summary: words(34),
      tags: ["structured-output", "tool-routing"],
      language: "en",
      languageState: "english",
      originalInChinese: false,
    });
  });

  it("falls back to Chinese and hides stale tags for non-current metadata", () => {
    const stale = reconcileMetadataWithSource(currentSidecar(), HASH_B);

    expect(resolveMetadataDisplay(stale)).toEqual({
      title: stale.titleZh,
      summary: stale.summaryZh,
      tags: [],
      language: "zh",
      languageState: "original-in-chinese",
      originalInChinese: true,
    });
  });

  it("falls back instead of displaying English when a current marker has a mismatched hash", () => {
    const mismatched = {
      ...currentSidecar(),
      source: { ...currentSidecar().source, hash: HASH_B },
      sourceStatus: "unreviewed",
      sourceStatusHash: HASH_B,
    } as SidecarMetadata;

    expect(resolveMetadataDisplay(mismatched)).toMatchObject({
      title: mismatched.titleZh,
      summary: mismatched.summaryZh,
      tags: [],
      languageState: "original-in-chinese",
    });
  });

  it("reports both lesson IDs for a duplicate slug", () => {
    const first = currentSidecar();
    const second = currentSidecar({
      id: "lesson-0014",
      lesson: 14,
      source: {
        file: "lessons/2026-07-18.md",
        section: 1,
        hash: HASH_B,
      },
      date: "2026-07-18",
      sourceStatusHash: HASH_B,
      metadataSourceHash: HASH_B,
    });

    expect(() => assertUniqueMetadataSlugs([first, second])).toThrow(
      /lesson-0013.*lesson-0014|lesson-0014.*lesson-0013/,
    );
  });
});
