import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

const DATE_LESSON_FILE = /^\d{4}-\d{2}-\d{2}\.md$/;

export interface LessonSegment {
  file: string;
  section: number;
  date: string;
  titleZh: string;
  raw: string;
  hash: string;
}

interface LessonHeading {
  index: number;
  text: string;
}

interface MarkdownFence {
  marker: "`" | "~";
  length: number;
}

function findLessonHeadings(source: string): LessonHeading[] {
  const headings: LessonHeading[] = [];
  let fence: MarkdownFence | undefined;
  let lineStart = 0;

  while (lineStart <= source.length) {
    const newline = source.indexOf("\n", lineStart);
    const lineEnd = newline === -1 ? source.length : newline;
    const line = source.slice(lineStart, lineEnd);
    const fenceMatch = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(line);

    if (fence) {
      if (
        fenceMatch &&
        fenceMatch[1]![0] === fence.marker &&
        fenceMatch[1]!.length >= fence.length &&
        !fenceMatch[2]!.trim()
      ) {
        fence = undefined;
      }
    } else if (fenceMatch) {
      fence = {
        marker: fenceMatch[1]![0] as MarkdownFence["marker"],
        length: fenceMatch[1]!.length,
      };
    } else if (line.startsWith("# 📅")) {
      headings.push({ index: lineStart, text: line });
    }

    if (newline === -1) {
      break;
    }
    lineStart = newline + 1;
  }

  return headings;
}

export async function discoverLessonFiles(root: string): Promise<string[]> {
  const entries = await fs.readdir(root, { withFileTypes: true });

  return entries
    .filter((entry) => entry.isFile() && DATE_LESSON_FILE.test(entry.name))
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
    let titleZh = heading.text.slice("# 📅".length).trim();

    if (titleZh.startsWith(date)) {
      titleZh = titleZh.slice(date.length).trim();
    }
    titleZh = titleZh.replace(/^[—–\-:：|]\s*/, "").trim();

    const digest = createHash("sha256").update(raw).digest("hex");

    return {
      file: sourceFile,
      section: index + 1,
      date,
      titleZh,
      raw,
      hash: `sha256:${digest}`,
    };
  });
}
