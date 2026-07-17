import { z } from "zod";

export const ENRICHMENT_CONSTRAINTS = {
  idPattern:
    "^lesson-(?:0{3}[1-9]|0{2}[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3,})$",
  sourceHashPattern: "^sha256:[0-9a-f]{64}$",
  titleMinLength: 4,
  titleMaxLength: 120,
  summaryMinLength: 40,
  summaryMaxLength: 600,
  summaryMinWords: 30,
  summaryMaxWords: 60,
  tagsMinItems: 1,
  tagsMaxItems: 5,
  tagPattern: "^[a-z0-9]+(?:-[a-z0-9]+)*$",
} as const;

const SOURCE_FILE_PATTERN = /^lessons\/\d{4}-\d{2}-\d{2}\.md$/;
const SLUG_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const ENGLISH_WORD_PATTERN =
  /[A-Za-z0-9]*[A-Za-z][A-Za-z0-9]*(?:[-'’][A-Za-z0-9]*[A-Za-z][A-Za-z0-9]*)*/g;

export const SourceStatusSchema = z.enum([
  "unreviewed",
  "partially-verified",
  "verified",
]);
export const MetadataStatusSchema = z.enum([
  "pending",
  "needs-review",
  "current",
]);
export const LessonIdSchema = z
  .string()
  .regex(new RegExp(ENRICHMENT_CONSTRAINTS.idPattern));
export const SourceHashSchema = z
  .string()
  .regex(new RegExp(ENRICHMENT_CONSTRAINTS.sourceHashPattern));

function isValidCalendarDate(value: string): boolean {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) {
    return false;
  }

  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const date = new Date(Date.UTC(year, month - 1, day));

  return (
    date.getUTCFullYear() === year &&
    date.getUTCMonth() === month - 1 &&
    date.getUTCDate() === day
  );
}

const DateSchema = z.string().refine(isValidCalendarDate, {
  message: "Expected a valid YYYY-MM-DD date",
});
const SourceFileSchema = z
  .string()
  .regex(SOURCE_FILE_PATTERN)
  .refine(
    (value) =>
      isValidCalendarDate(value.slice("lessons/".length, -".md".length)),
    { message: "Source file must contain a valid calendar date" },
  );
const TagSchema = z
  .string()
  .regex(new RegExp(ENRICHMENT_CONSTRAINTS.tagPattern));
const EnglishTitleSchema = z
  .string()
  .min(ENRICHMENT_CONSTRAINTS.titleMinLength)
  .max(ENRICHMENT_CONSTRAINTS.titleMaxLength);

export function countEnglishWords(value: string): number {
  return value.match(ENGLISH_WORD_PATTERN)?.length ?? 0;
}

const EnglishSummarySchema = z
  .string()
  .min(ENRICHMENT_CONSTRAINTS.summaryMinLength)
  .max(ENRICHMENT_CONSTRAINTS.summaryMaxLength)
  .superRefine((value, context) => {
    const wordCount = countEnglishWords(value);
    if (
      wordCount < ENRICHMENT_CONSTRAINTS.summaryMinWords ||
      wordCount > ENRICHMENT_CONSTRAINTS.summaryMaxWords
    ) {
      context.addIssue({
        code: "custom",
        message: `English summary must contain ${ENRICHMENT_CONSTRAINTS.summaryMinWords}-${ENRICHMENT_CONSTRAINTS.summaryMaxWords} words`,
      });
    }
  });

const TagListSchema = z
  .array(TagSchema)
  .max(ENRICHMENT_CONSTRAINTS.tagsMaxItems)
  .superRefine((tags, context) => {
    if (new Set(tags).size !== tags.length) {
      context.addIssue({
        code: "custom",
        message: "Tags must be unique",
      });
    }
  });

export const EnrichmentOutputSchema = z
  .object({
    id: LessonIdSchema,
    sourceHash: SourceHashSchema,
    titleEn: EnglishTitleSchema,
    summaryEn: EnglishSummarySchema,
    tags: TagListSchema.min(ENRICHMENT_CONSTRAINTS.tagsMinItems),
  })
  .strict();

const SidecarSourceSchema = z
  .object({
    file: SourceFileSchema,
    section: z.number().int().positive(),
    hash: SourceHashSchema,
  })
  .strict();

const SidecarBaseSchema = z
  .object({
    id: LessonIdSchema,
    source: SidecarSourceSchema,
    lesson: z.number().int().positive(),
    date: DateSchema,
    track: z.enum(["A", "B", "C"]),
    depth: z.enum(["L1", "L2", "L3", "L4"]),
    titleZh: z.string().min(1),
    titleEn: EnglishTitleSchema.nullable(),
    summaryZh: z.string().min(1),
    summaryEn: EnglishSummarySchema.nullable(),
    slug: z.string().min(1).max(96).regex(SLUG_PATTERN),
    tags: TagListSchema,
    sourceStatus: SourceStatusSchema,
    sourceStatusHash: SourceHashSchema,
    metadataStatus: MetadataStatusSchema,
    metadataSourceHash: SourceHashSchema.nullable(),
    featured: z.boolean(),
  })
  .strict();

function canonicalLessonId(lesson: number): string {
  return `lesson-${String(lesson).padStart(4, "0")}`;
}

function sourceDate(file: string): string {
  return file.slice("lessons/".length, -".md".length);
}

export const SidecarMetadataSchema = SidecarBaseSchema.superRefine(
  (sidecar, context) => {
    if (sidecar.id !== canonicalLessonId(sidecar.lesson)) {
      context.addIssue({
        code: "custom",
        path: ["id"],
        message: "Sidecar ID must be canonical for its lesson number",
      });
    }
    if (sidecar.date !== sourceDate(sidecar.source.file)) {
      context.addIssue({
        code: "custom",
        path: ["date"],
        message: "Sidecar date must agree with source.file",
      });
    }
    if (sidecar.sourceStatusHash !== sidecar.source.hash) {
      context.addIssue({
        code: "custom",
        path: ["sourceStatusHash"],
        message: "Source review status must be bound to the current source hash",
      });
    }

    const hasNoEnglishSet =
      sidecar.titleEn === null &&
      sidecar.summaryEn === null &&
      sidecar.tags.length === 0 &&
      sidecar.metadataSourceHash === null;
    const hasCompleteEnglishSet =
      sidecar.titleEn !== null &&
      sidecar.summaryEn !== null &&
      sidecar.tags.length >= ENRICHMENT_CONSTRAINTS.tagsMinItems &&
      sidecar.metadataSourceHash !== null;

    if (sidecar.metadataStatus === "pending" && !hasNoEnglishSet) {
      context.addIssue({
        code: "custom",
        path: ["metadataStatus"],
        message: "Pending metadata must not contain English metadata",
      });
    }

    if (sidecar.metadataStatus === "current") {
      if (!hasCompleteEnglishSet) {
        context.addIssue({
          code: "custom",
          path: ["metadataStatus"],
          message: "Current metadata requires a complete English metadata set",
        });
      }
      if (sidecar.metadataSourceHash !== sidecar.source.hash) {
        context.addIssue({
          code: "custom",
          path: ["metadataSourceHash"],
          message: "Current metadata must be bound to the current source hash",
        });
      }
    }

    if (sidecar.metadataStatus === "needs-review") {
      if (!hasNoEnglishSet && !hasCompleteEnglishSet) {
        context.addIssue({
          code: "custom",
          path: ["metadataStatus"],
          message:
            "Needs-review metadata must contain either no English set or one complete stale set",
        });
      }
      if (
        hasCompleteEnglishSet &&
        sidecar.metadataSourceHash === sidecar.source.hash
      ) {
        context.addIssue({
          code: "custom",
          path: ["metadataSourceHash"],
          message: "Needs-review English metadata must be stale",
        });
      }
    }
  },
);

export const CreateMetadataSidecarInputSchema = z
  .object({
    source: SidecarSourceSchema,
    lesson: z.number().int().positive(),
    date: DateSchema,
    track: z.enum(["A", "B", "C"]),
    depth: z.enum(["L1", "L2", "L3", "L4"]),
    titleZh: z.string().min(1),
    summaryZh: z.string().min(1),
    featured: z.boolean().optional(),
  })
  .strict();

export type SourceStatus = z.infer<typeof SourceStatusSchema>;
export type MetadataStatus = z.infer<typeof MetadataStatusSchema>;
export type EnrichmentOutput = z.infer<typeof EnrichmentOutputSchema>;
export type SidecarMetadata = z.infer<typeof SidecarMetadataSchema>;
export type CreateMetadataSidecarInput = z.infer<
  typeof CreateMetadataSidecarInputSchema
>;

export interface EnrichmentExpectations {
  expectedSourceHash?: string;
  expectedId?: string;
}

function parseEnrichmentExpectations(
  expected?: string | EnrichmentExpectations,
  expectedId?: string,
): EnrichmentExpectations {
  if (typeof expected === "string") {
    return { expectedSourceHash: expected, expectedId };
  }
  return expected ?? {};
}

export function validateEnrichment(
  value: unknown,
  expected?: string | EnrichmentExpectations,
  expectedId?: string,
): EnrichmentOutput {
  const output = EnrichmentOutputSchema.parse(value);
  const expectations = parseEnrichmentExpectations(expected, expectedId);

  if (
    expectations.expectedSourceHash !== undefined &&
    output.sourceHash !== SourceHashSchema.parse(expectations.expectedSourceHash)
  ) {
    throw new Error(
      `Stale metadata output: expected ${expectations.expectedSourceHash}, received ${output.sourceHash}`,
    );
  }
  if (
    expectations.expectedId !== undefined &&
    output.id !== LessonIdSchema.parse(expectations.expectedId)
  ) {
    throw new Error(
      `Metadata output ID does not match: expected ${expectations.expectedId}, received ${output.id}`,
    );
  }

  return output;
}

export function createMetadataSlug(value: string, fallbackId: string): string {
  const fallback = LessonIdSchema.parse(fallbackId);
  const slug = value
    .normalize("NFKD")
    .replace(/\p{M}+/gu, "")
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/['’]/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "");
  const capped = slug.slice(0, 96).replace(/-+$/g, "");

  return capped || fallback;
}

export function createMetadataSidecar(
  value: CreateMetadataSidecarInput,
  enrichment?: unknown,
): SidecarMetadata {
  const input = CreateMetadataSidecarInputSchema.parse(value);
  const id = canonicalLessonId(input.lesson);
  const output =
    enrichment === undefined
      ? undefined
      : validateEnrichment(enrichment, {
          expectedSourceHash: input.source.hash,
          expectedId: id,
        });

  return SidecarMetadataSchema.parse({
    id,
    source: input.source,
    lesson: input.lesson,
    date: input.date,
    track: input.track,
    depth: input.depth,
    titleZh: input.titleZh,
    titleEn: output?.titleEn ?? null,
    summaryZh: input.summaryZh,
    summaryEn: output?.summaryEn ?? null,
    slug:
      output === undefined
        ? id
        : createMetadataSlug(output.titleEn, id),
    tags: output?.tags ?? [],
    sourceStatus: "unreviewed",
    sourceStatusHash: input.source.hash,
    metadataStatus: output === undefined ? "pending" : "current",
    metadataSourceHash: output?.sourceHash ?? null,
    featured: input.featured ?? false,
  });
}

export function reconcileMetadataWithSource(
  value: SidecarMetadata,
  currentHash: string,
): SidecarMetadata {
  const sidecar = SidecarMetadataSchema.parse(value);
  const hash = SourceHashSchema.parse(currentHash);

  if (sidecar.source.hash === hash) {
    return value;
  }

  return SidecarMetadataSchema.parse({
    ...sidecar,
    source: { ...sidecar.source, hash },
    sourceStatus: "unreviewed",
    sourceStatusHash: hash,
    metadataStatus: "needs-review",
  });
}

export function applyEnrichment(
  value: SidecarMetadata,
  enrichment: unknown,
): SidecarMetadata {
  const sidecar = SidecarMetadataSchema.parse(value);
  const output = validateEnrichment(enrichment, {
    expectedSourceHash: sidecar.source.hash,
    expectedId: sidecar.id,
  });

  return SidecarMetadataSchema.parse({
    ...sidecar,
    titleEn: output.titleEn,
    summaryEn: output.summaryEn,
    tags: output.tags,
    metadataStatus: "current",
    metadataSourceHash: output.sourceHash,
  });
}

export function setSourceReviewStatus(
  value: SidecarMetadata,
  status: SourceStatus,
  reviewedHash: string,
): SidecarMetadata {
  const sidecar = SidecarMetadataSchema.parse(value);
  const nextStatus = SourceStatusSchema.parse(status);
  const hash = SourceHashSchema.parse(reviewedHash);

  if (hash !== sidecar.source.hash) {
    throw new Error(
      `Stale source review: expected ${sidecar.source.hash}, received ${hash}`,
    );
  }

  return SidecarMetadataSchema.parse({
    ...sidecar,
    sourceStatus: nextStatus,
    sourceStatusHash: hash,
  });
}

export interface ResolvedMetadataDisplay {
  title: string;
  summary: string;
  tags: string[];
  language: "en" | "zh";
  languageState: "english" | "original-in-chinese";
  originalInChinese: boolean;
}

export function resolveMetadataDisplay(
  value: SidecarMetadata,
): ResolvedMetadataDisplay {
  const sidecar = SidecarBaseSchema.parse(value);
  const englishIsCurrent =
    sidecar.metadataStatus === "current" &&
    sidecar.metadataSourceHash === sidecar.source.hash &&
    sidecar.titleEn !== null &&
    sidecar.summaryEn !== null;

  if (englishIsCurrent) {
    return {
      title: sidecar.titleEn!,
      summary: sidecar.summaryEn!,
      tags: [...sidecar.tags],
      language: "en",
      languageState: "english",
      originalInChinese: false,
    };
  }

  return {
    title: sidecar.titleZh,
    summary: sidecar.summaryZh,
    tags: [],
    language: "zh",
    languageState: "original-in-chinese",
    originalInChinese: true,
  };
}

export function assertUniqueMetadataSlugs(
  sidecars: readonly SidecarMetadata[],
): void {
  const owners = new Map<string, string>();

  for (const value of sidecars) {
    const sidecar = SidecarMetadataSchema.parse(value);
    const existingId = owners.get(sidecar.slug);
    if (existingId !== undefined) {
      throw new Error(
        `Duplicate slug "${sidecar.slug}" for ${existingId} and ${sidecar.id}`,
      );
    }
    owners.set(sidecar.slug, sidecar.id);
  }
}
