import { z } from "zod";

export const SyncIndexLessonSchema = z
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

export const SyncIndexSchema = z
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

export type SyncIndex = z.infer<typeof SyncIndexSchema>;
export type SyncIndexLesson = z.infer<typeof SyncIndexLessonSchema>;
