import type { LearningLogRow } from "./learning-log";
import type { LessonSegment } from "./lessons";

export interface MatchedLesson {
  id: string;
  segment: LessonSegment;
  log: LearningLogRow;
}

interface IndexedSegment {
  index: number;
  value: LessonSegment;
  normalizedTopic: string;
}

interface IndexedRow {
  index: number;
  value: LearningLogRow;
  normalizedTopic: string;
}

const PRESENTATION_PUNCTUATION = new Set([
  ":",
  "：",
  ",",
  "，",
  ".",
  "。",
  ";",
  "；",
  "!",
  "！",
  "?",
  "？",
  "、",
  "·",
  "・",
  "•",
  "(",
  ")",
  "（",
  "）",
  "[",
  "]",
  "【",
  "】",
  "{",
  "}",
  "《",
  "》",
  "〈",
  "〉",
  '"',
  "'",
  "“",
  "”",
  "‘",
  "’",
  "`",
  "-",
  "‐",
  "‑",
  "‒",
  "–",
  "—",
  "―",
  "−",
]);
const WHITE_SPACE = /\p{White_Space}/u;

function normalizeTopic(value: string): string {
  return [...value.normalize("NFKC").toLowerCase()]
    .filter(
      (character) =>
        !WHITE_SPACE.test(character) &&
        !PRESENTATION_PUNCTUATION.has(character),
    )
    .join("");
}

function describeSegment(segment: LessonSegment): string {
  return `${segment.file} section ${segment.section} (${segment.date}, ${JSON.stringify(
    segment.titleZh,
  )})`;
}

function describeRow(row: LearningLogRow): string {
  return `lesson ${row.lesson} (${row.date}, ${JSON.stringify(row.topic)})`;
}

function describeRows(rows: readonly IndexedRow[]): string {
  return rows.length === 0
    ? "none"
    : rows.map((candidate) => describeRow(candidate.value)).join("; ");
}

function stableLessonId(lesson: number): string {
  return `lesson-${String(lesson).padStart(4, "0")}`;
}

function assertValidLogRows(rows: readonly LearningLogRow[]): void {
  const seenIds = new Map<string, number>();

  rows.forEach((row, index) => {
    const rowNumber = index + 1;
    if (!Number.isSafeInteger(row.lesson) || row.lesson < 1) {
      throw new Error(
        `Invalid log lesson number ${String(row.lesson)} at log row ${rowNumber} (${row.date}, ${JSON.stringify(row.topic)})`,
      );
    }

    const id = stableLessonId(row.lesson);
    const previousIndex = seenIds.get(id);
    if (previousIndex !== undefined) {
      throw new Error(
        `Duplicate lesson ID ${id} in log rows: log row ${previousIndex + 1} (${describeRow(rows[previousIndex]!)}) conflicts with log row ${rowNumber} (${describeRow(row)})`,
      );
    }
    seenIds.set(id, index);
  });
}

function groupByTopic<T extends { normalizedTopic: string }>(
  values: readonly T[],
): Map<string, T[]> {
  const groups = new Map<string, T[]>();

  for (const value of values) {
    const group = groups.get(value.normalizedTopic) ?? [];
    group.push(value);
    groups.set(value.normalizedTopic, group);
  }

  return groups;
}

export function matchSegmentsToLog(
  segments: readonly LessonSegment[],
  rows: readonly LearningLogRow[],
): MatchedLesson[] {
  assertValidLogRows(rows);

  const indexedSegments: IndexedSegment[] = segments.map((value, index) => ({
    index,
    value,
    normalizedTopic: normalizeTopic(value.titleZh),
  }));
  const indexedRows: IndexedRow[] = rows.map((value, index) => ({
    index,
    value,
    normalizedTopic: normalizeTopic(value.topic),
  }));
  const matches: Array<MatchedLesson | undefined> = new Array(segments.length);
  const usedRows = new Map<number, IndexedSegment>();
  const usedIds = new Map<string, IndexedSegment>();

  function assign(segment: IndexedSegment, row: IndexedRow): void {
    const previousSegment = usedRows.get(row.index);
    if (previousSegment) {
      throw new Error(
        `Reused log row ${describeRow(row.value)} for ${describeSegment(previousSegment.value)} and ${describeSegment(segment.value)}`,
      );
    }

    const id = stableLessonId(row.value.lesson);
    const previousIdSegment = usedIds.get(id);
    if (previousIdSegment) {
      throw new Error(
        `Duplicate lesson ID ${id} for ${describeSegment(previousIdSegment.value)} and ${describeSegment(segment.value)}`,
      );
    }

    usedRows.set(row.index, segment);
    usedIds.set(id, segment);
    matches[segment.index] = { id, segment: segment.value, log: row.value };
  }

  for (const segment of indexedSegments) {
    const explicitLesson = segment.value.lesson;
    if (explicitLesson === undefined) {
      continue;
    }

    const candidates = indexedRows.filter(
      (candidate) => candidate.value.lesson === explicitLesson,
    );
    if (candidates.length === 0) {
      throw new Error(
        `Explicit lesson mapping failed for lesson ${explicitLesson} at ${describeSegment(segment.value)}: no candidate log row; available candidates: ${describeRows(indexedRows)}`,
      );
    }
    if (candidates.length > 1) {
      throw new Error(
        `Explicit lesson mapping failed for lesson ${explicitLesson} at ${describeSegment(segment.value)}: duplicate candidates: ${describeRows(candidates)}`,
      );
    }

    const [candidate] = candidates;
    if (candidate!.value.date !== segment.value.date) {
      throw new Error(
        `Explicit lesson mapping conflict for lesson ${explicitLesson} at ${describeSegment(segment.value)}: segment date ${segment.value.date} conflicts with candidate date ${candidate!.value.date} (${describeRow(candidate!.value)})`,
      );
    }

    assign(segment, candidate!);
  }

  const implicitByDate = new Map<string, IndexedSegment[]>();
  for (const segment of indexedSegments) {
    if (segment.value.lesson !== undefined) {
      continue;
    }

    const group = implicitByDate.get(segment.value.date) ?? [];
    group.push(segment);
    implicitByDate.set(segment.value.date, group);
  }

  for (const [date, dateSegments] of implicitByDate) {
    const sameDateRows = indexedRows.filter(
      (candidate) =>
        candidate.value.date === date && !usedRows.has(candidate.index),
    );
    const segmentGroups = groupByTopic(dateSegments);
    const rowGroups = groupByTopic(sameDateRows);

    for (const [topic, topicSegments] of segmentGroups) {
      const topicRows = rowGroups.get(topic) ?? [];
      if (topicSegments.length === 1 && topicRows.length === 1) {
        assign(topicSegments[0]!, topicRows[0]!);
      }
    }

    const remainingSegments = dateSegments.filter(
      (segment) => matches[segment.index] === undefined,
    );
    const remainingRows = sameDateRows.filter(
      (candidate) => !usedRows.has(candidate.index),
    );
    if (remainingSegments.length === 0) {
      continue;
    }

    const completeOrderMapping =
      remainingSegments.length > 0 &&
      remainingSegments.length === remainingRows.length;

    if (completeOrderMapping) {
      const orderedSegments = [...remainingSegments].sort(
        (left, right) =>
          left.value.section - right.value.section || left.index - right.index,
      );
      const orderedRows = [...remainingRows].sort(
        (left, right) => left.index - right.index,
      );

      orderedSegments.forEach((segment, index) => {
        assign(segment, orderedRows[index]!);
      });
      continue;
    }

    for (const segment of remainingSegments) {
      const candidates = remainingRows.filter(
        (candidate) =>
          candidate.normalizedTopic !== "" &&
          candidate.normalizedTopic === segment.normalizedTopic,
      );
      if (candidates.length > 0) {
        throw new Error(
          `Ambiguous lesson mapping for ${describeSegment(segment.value)}; same-date topic candidates: ${describeRows(candidates)}. Section order is allowed only for a complete one-to-one remaining set.`,
        );
      }

      throw new Error(
        `Unmatched lesson segment ${describeSegment(segment.value)}; same-date candidates: ${describeRows(remainingRows)}`,
      );
    }
  }

  const unmatched = indexedSegments.find(
    (segment) => matches[segment.index] === undefined,
  );
  if (unmatched) {
    throw new Error(
      `Unmatched lesson segment ${describeSegment(unmatched.value)}`,
    );
  }

  return matches as MatchedLesson[];
}
