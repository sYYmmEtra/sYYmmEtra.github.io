import type { Root, TableRow } from "mdast";
import { toString } from "mdast-util-to-string";
import remarkGfm from "remark-gfm";
import remarkParse from "remark-parse";
import { unified } from "unified";

export type Track = "A" | "B" | "C";
export type Depth = "L1" | "L2" | "L3" | "L4";

const markdownParser = unified().use(remarkParse).use(remarkGfm);
const LEARNING_LOG_HEADER = [
  "讲次",
  "日期",
  "轨道",
  "主题",
  "深度",
  "一句话要点",
  "下次可深入",
] as const;

export interface LearningLogRow {
  lesson: number;
  date: string;
  track: Track;
  topic: string;
  depth: Depth;
  summaryZh: string;
  nextZh: string;
}

function isValidDate(value: string): boolean {
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

function invalidRow(lineNumber: number, reason: string): never {
  throw new Error(`Invalid learning log row at line ${lineNumber}: ${reason}`);
}

function rowLineNumber(row: TableRow): number {
  const lineNumber = row.position?.start.line;
  if (lineNumber === undefined) {
    throw new Error("Learning log table row is missing a source line");
  }
  return lineNumber;
}

export function parseLearningLog(source: string): LearningLogRow[] {
  const tree = markdownParser.parse(source) as Root;
  const rows: LearningLogRow[] = [];

  for (const node of tree.children) {
    if (node.type !== "table") {
      continue;
    }

    const [header, ...dataRows] = node.children;
    const headerCells = header?.children.map((cell) => toString(cell).trim());
    if (
      !headerCells ||
      headerCells.length !== LEARNING_LOG_HEADER.length ||
      !LEARNING_LOG_HEADER.every(
        (expected, index) => headerCells[index] === expected,
      )
    ) {
      continue;
    }

    for (const row of dataRows) {
      const cells = row.children.map((cell) => toString(cell).trim());
      const lineNumber = rowLineNumber(row);

      if (cells.length !== 7) {
        invalidRow(
          lineNumber,
          "expected seven fields, found " + cells.length,
        );
      }

      const lessonCell = cells[0]!;
      const date = cells[1]!;
      const trackCell = cells[2]!;
      const topic = cells[3]!;
      const depthCell = cells[4]!;
      const summaryZh = cells[5]!;
      const nextZh = cells[6]!;
      const lesson = Number(lessonCell);

      if (
        !/^\d+$/.test(lessonCell) ||
        !Number.isSafeInteger(lesson) ||
        lesson < 1
      ) {
        invalidRow(lineNumber, "lesson must be a positive integer");
      }
      if (!isValidDate(date)) {
        invalidRow(lineNumber, "date must be a valid YYYY-MM-DD value");
      }
      if (!(["A", "B", "C"] as const).includes(trackCell as Track)) {
        invalidRow(lineNumber, "track must be A, B, or C");
      }
      if (!topic) {
        invalidRow(lineNumber, "topic must not be empty");
      }
      if (!(["L1", "L2", "L3", "L4"] as const).includes(depthCell as Depth)) {
        invalidRow(lineNumber, "depth must be L1, L2, L3, or L4");
      }
      if (!summaryZh) {
        invalidRow(lineNumber, "summary must not be empty");
      }
      if (!nextZh) {
        invalidRow(lineNumber, "next must not be empty");
      }

      rows.push({
        lesson,
        date,
        track: trackCell as Track,
        topic,
        depth: depthCell as Depth,
        summaryZh,
        nextZh,
      });
    }
  }

  return rows;
}
