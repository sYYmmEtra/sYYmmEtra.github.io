import fs from "node:fs/promises";
import { describe, expect, it } from "vitest";

import { parseLearningLog } from "../../scripts/lib/learning-log";

it("parses typed learning-log rows", async () => {
  const source = await fs.readFile("tests/fixtures/learning-log.md", "utf8");
  const rows = parseLearningLog(source);

  expect(rows).toMatchObject([
    { lesson: 1, date: "2026-07-06", track: "A", depth: "L1" },
    { lesson: 2, date: "2026-07-06", track: "B", depth: "L2" },
  ]);
});

it("parses escaped pipes inside Markdown table cells", () => {
  const source =
    "| 3 | 2026-07-07 | C | 模型 A \\| 模型 B | L3 | 比较两种模型。 | 继续实验。 |";

  const rows = parseLearningLog(source);

  expect(rows).toMatchObject([
    { lesson: 3, topic: "模型 A | 模型 B", track: "C", depth: "L3" },
  ]);
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
    const source = ["# Log", "", `| ${cells.join(" | ")} |`].join("\n");

    expect(() => parseLearningLog(source)).toThrow(
      new RegExp(`line 3.*${field}`, "i"),
    );
  });

  it("reports the source line when a numeric row has the wrong column count", () => {
    const source = [
      "# Log",
      "",
      "| 2 | 2026-07-06 | A | 主题 | L1 | 摘要 |",
    ].join("\n");

    expect(() => parseLearningLog(source)).toThrow(/line 3.*seven fields/i);
  });
});
