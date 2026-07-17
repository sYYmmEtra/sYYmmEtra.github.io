export type Track = "A" | "B" | "C";
export type Depth = "L1" | "L2" | "L3" | "L4";

export interface LearningLogRow {
  lesson: number;
  date: string;
  track: Track;
  topic: string;
  depth: Depth;
  summaryZh: string;
  nextZh: string;
}

function parseTableCells(line: string): string[] | undefined {
  const trimmed = line.trim();
  if (!trimmed.includes("|")) {
    return undefined;
  }

  const withoutLeadingPipe = trimmed.startsWith("|")
    ? trimmed.slice(1)
    : trimmed;
  const withoutOuterPipes = withoutLeadingPipe.endsWith("|")
    ? withoutLeadingPipe.slice(0, -1)
    : withoutLeadingPipe;

  const cells: string[] = [];
  let cell = "";

  for (const character of withoutOuterPipes) {
    if (character !== "|") {
      cell += character;
      continue;
    }

    let precedingBackslashes = 0;
    for (let index = cell.length - 1; cell[index] === "\\"; index -= 1) {
      precedingBackslashes += 1;
    }

    if (precedingBackslashes % 2 === 1) {
      cell = `${cell.slice(0, -1)}|`;
      continue;
    }

    cells.push(cell.trim());
    cell = "";
  }

  cells.push(cell.trim());

  return cells;
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

export function parseLearningLog(source: string): LearningLogRow[] {
  const rows: LearningLogRow[] = [];

  source.split("\n").forEach((line, index) => {
    const cells = parseTableCells(line);
    if (!cells || !/^\d+$/.test(cells[0] ?? "")) {
      return;
    }

    const lineNumber = index + 1;
    if (cells.length !== 7) {
      invalidRow(
        lineNumber,
        `expected seven fields, found ${cells.length}`,
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

    if (!Number.isSafeInteger(lesson) || lesson < 1) {
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
  });

  return rows;
}
