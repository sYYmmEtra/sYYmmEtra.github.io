import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import type { Root } from "mdast";
import { toString } from "mdast-util-to-string";
import remarkGfm from "remark-gfm";
import remarkParse from "remark-parse";
import { unified } from "unified";

const DATE_LESSON_FILE = /^\d{4}-\d{2}-\d{2}\.md$/;
const STRUCTURED_LESSON_HEADING = /^(?<date>\d{4}-\d{2}-\d{2})(?:（第\s*(?<parenLesson>\d+)\s*讲）)?\s*·\s*(?:讲\s*(?<inlineLesson>\d+)\s*·\s*)?轨道\s*(?<track>[ABC])\s*·\s*(?<topic>.+)\s*·\s*深度\s+(?<depth>L[1-4])\s*$/u;
const markdownParser = unified().use(remarkParse).use(remarkGfm);

export interface LessonSegment {
  file: string;
  section: number;
  lesson?: number;
  date: string;
  titleZh: string;
  raw: string;
  hash: string;
}

interface ParsedLessonHeading {
  titleZh: string;
  lesson?: number;
  date?: string;
}

interface LessonHeading {
  index: number;
  text: string;
}

function findLessonHeadings(source: string): LessonHeading[] {
  const tree = markdownParser.parse(source) as Root;
  const headings: LessonHeading[] = [];

  for (const node of tree.children) {
    if (node.type !== "heading" || node.depth !== 1) {
      continue;
    }

    const text = toString(node).trim();
    if (!text.startsWith("📅")) {
      continue;
    }

    const offset = node.position?.start.offset;
    if (offset === undefined) {
      throw new Error("Lesson heading is missing a source offset");
    }

    const lineStart = source.lastIndexOf("\n", offset - 1) + 1;
    headings.push({
      index: lineStart,
      text: text.slice("📅".length).trim(),
    });
  }

  return headings;
}

function parseStructuredLessonHeading(
  text: string,
): ParsedLessonHeading | undefined {
  const groups = STRUCTURED_LESSON_HEADING.exec(text)?.groups;
  if (!groups || (groups.parenLesson && groups.inlineLesson)) {
    return undefined;
  }

  const lessonText = groups.parenLesson ?? groups.inlineLesson;
  const lesson = lessonText === undefined ? undefined : Number(lessonText);
  if (
    lesson !== undefined &&
    (!Number.isSafeInteger(lesson) || lesson < 1)
  ) {
    return undefined;
  }

  return {
    date: groups.date!,
    titleZh: groups.topic!.trim(),
    ...(lesson === undefined ? {} : { lesson }),
  };
}

function assertHeadingDateMatchesFile(
  headingDate: string,
  filenameDate: string,
  sourceFile: string,
  section: number,
): void {
  if (headingDate !== filenameDate) {
    throw new Error(
      `Lesson heading date ${headingDate} does not match filename date ${filenameDate} in ${sourceFile} section ${section}`,
    );
  }
}

function looksLikeStructuredLessonHeading(text: string): boolean {
  return (
    /^\d{4}-\d{2}-\d{2}/.test(text) &&
    (/·\s*(?:讲\s*\d+|轨道|深度)/u.test(text) ||
      /（第\s*\d+\s*讲）/u.test(text))
  );
}

function parseLessonHeading(
  text: string,
  date: string,
  sourceFile: string,
  section: number,
): ParsedLessonHeading {
  const structured = parseStructuredLessonHeading(text);
  if (structured) {
    assertHeadingDateMatchesFile(
      structured.date!,
      date,
      sourceFile,
      section,
    );
    return structured;
  }

  if (looksLikeStructuredLessonHeading(text)) {
    throw new Error(
      `Malformed structured lesson heading in ${sourceFile} section ${section}`,
    );
  }

  const simpleDate = /^(\d{4}-\d{2}-\d{2})(?!\d)/.exec(text)?.[1];
  let titleZh = text;
  if (simpleDate) {
    assertHeadingDateMatchesFile(simpleDate, date, sourceFile, section);
    titleZh = titleZh.slice(simpleDate.length).trim();
  }

  return {
    titleZh: titleZh.replace(/^[—–\-:：|]\s*/, "").trim(),
  };
}

function isValidCalendarDate(value: string): boolean {
  const [yearText, monthText, dayText] = value.split("-");
  const year = Number(yearText);
  const month = Number(monthText);
  const day = Number(dayText);
  const date = new Date(Date.UTC(year, month - 1, day));

  return (
    date.getUTCFullYear() === year &&
    date.getUTCMonth() === month - 1 &&
    date.getUTCDate() === day
  );
}

export async function discoverLessonFiles(root: string): Promise<string[]> {
  const entries = await fs.readdir(root, { withFileTypes: true });

  return entries
    .filter(
      (entry) =>
        entry.isFile() &&
        DATE_LESSON_FILE.test(entry.name) &&
        isValidCalendarDate(entry.name.slice(0, -3)),
    )
    .map((entry) => path.join(root, entry.name))
    .sort();
}

export async function splitLessonSegments(
  file: string,
): Promise<LessonSegment[]> {
  const source = await fs.readFile(file, "utf8");
  const normalized = source.replace(/^\uFEFF/, "").replace(/\r\n/g, "\n");
  const headings = findLessonHeadings(normalized);
  const sourceFile = path.basename(file);
  const date = sourceFile.replace(/\.md$/, "");

  return headings.map((heading, index) => {
    const start = heading.index;
    const end = headings[index + 1]?.index ?? normalized.length;
    const raw = normalized.slice(start, end);
    const parsedHeading = parseLessonHeading(
      heading.text,
      date,
      sourceFile,
      index + 1,
    );

    const digest = createHash("sha256").update(raw).digest("hex");

    return {
      file: sourceFile,
      section: index + 1,
      ...(parsedHeading.lesson === undefined
        ? {}
        : { lesson: parsedHeading.lesson }),
      date,
      titleZh: parsedHeading.titleZh,
      raw,
      hash: `sha256:${digest}`,
    };
  });
}
