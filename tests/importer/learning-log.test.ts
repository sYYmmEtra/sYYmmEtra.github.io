import fs from "node:fs/promises";
import { describe, expect, it } from "vitest";

import { parseLearningLog } from "../../scripts/lib/learning-log";

const tableHeader =
  "| Lesson | Date | Track | Topic | Depth | Summary Zh | Next Zh |";
const tableDivider = "| --- | --- | --- | --- | --- | --- | --- |";

function learningLogTable(...rows: string[]): string {
  return [tableHeader, tableDivider, ...rows].join("\n");
}

it("parses typed learning-log rows", async () => {
  const source = await fs.readFile("tests/fixtures/learning-log.md", "utf8");
  const rows = parseLearningLog(source);

  expect(rows).toMatchObject([
    { lesson: 1, date: "2026-07-06", track: "A", depth: "L1" },
    { lesson: 2, date: "2026-07-06", track: "B", depth: "L2" },
  ]);
});

it("parses escaped pipes inside Markdown table cells", () => {
  const source = learningLogTable(
    "| 3 | 2026-07-07 | C | 模型 A \\| 模型 B | L3 | 比较两种模型。 | 继续实验。 |",
  );

  const rows = parseLearningLog(source);

  expect(rows).toMatchObject([
    { lesson: 3, topic: "模型 A | 模型 B", track: "C", depth: "L3" },
  ]);
});

it("ignores numeric table rows inside fenced examples", () => {
  const source = [
    "```md",
    learningLogTable(
      "| 99 | 2026-07-07 | C | 围栏示例 | L3 | 不应导入。 | 不应导入。 |",
    ),
    "```",
    "",
    learningLogTable(
      "| 3 | 2026-07-07 | C | 实际课程 | L3 | 应导入。 | 继续学习。 |",
    ),
  ].join("\n");

  expect(parseLearningLog(source).map((row) => row.lesson)).toEqual([3]);
});

it("decodes an escaped pipe at the end of the final cell", () => {
  const source = learningLogTable(
    "| 4 | 2026-07-08 | A | 末尾管道 | L1 | 摘要 | next \\|",
  );

  expect(parseLearningLog(source)[0]?.nextZh).toBe("next |");
});

describe("learning-log validation", () => {
  const validCells = [
    "1",
    "2026-07-06",
    "A",
    "提示工程基础与常用模式",
    "L1",
    "中文摘要",
    "后续主题",
  ];

  it.each([
    { field: "lesson", index: 0, value: "0" },
    { field: "date", index: 1, value: "2026/07/06" },
    { field: "track", index: 2, value: "D" },
    { field: "topic", index: 3, value: "" },
    { field: "depth", index: 4, value: "L5" },
    { field: "summary", index: 5, value: "" },
    { field: "next", index: 6, value: "" },
  ])("reports the source line for an invalid $field", ({ field, index, value }) => {
    const cells = [...validCells];
    cells[index] = value;
    const source = learningLogTable(`| ${cells.join(" | ")} |`);

    expect(() => parseLearningLog(source)).toThrow(
      new RegExp(`line 3.*${field}`, "i"),
    );
  });

  it("reports the source line when a numeric row has the wrong column count", () => {
    const source = learningLogTable(
      "| 2 | 2026-07-06 | A | 主题 | L1 | 摘要 |",
    );

    expect(() => parseLearningLog(source)).toThrow(/line 3.*seven fields/i);
  });

  it("reports the source line for a malformed candidate lesson ID", () => {
    const source = learningLogTable(
      "| 2x | 2026-07-06 | A | 主题 | L1 | 摘要 | 后续 |",
    );

    expect(() => parseLearningLog(source)).toThrow(/line 3.*lesson/i);
  });
});
