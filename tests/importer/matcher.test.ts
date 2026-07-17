import { describe, expect, it } from "vitest";

import type { LearningLogRow } from "../../scripts/lib/learning-log";
import type { LessonSegment } from "../../scripts/lib/lessons";
import { matchSegmentsToLog } from "../../scripts/lib/matcher";

function segment(
  overrides: Partial<LessonSegment> & Pick<LessonSegment, "section" | "titleZh">,
): LessonSegment {
  return {
    file: "2026-07-06.md",
    date: "2026-07-06",
    raw: `# ${overrides.titleZh}`,
    hash: `sha256:segment-${overrides.section}`,
    ...overrides,
  };
}

function row(
  overrides: Partial<LearningLogRow> & Pick<LearningLogRow, "lesson" | "topic">,
): LearningLogRow {
  return {
    date: "2026-07-06",
    track: "A",
    depth: "L1",
    summaryZh: `summary ${overrides.lesson}`,
    nextZh: `next ${overrides.lesson}`,
    ...overrides,
  };
}

describe("matchSegmentsToLog", () => {
  it("maps two 2026-07-06 segments to stable lesson IDs", () => {
    const segments = [
      segment({ section: 1, titleZh: "提示工程基础与常用模式" }),
      segment({
        section: 2,
        titleZh: "注意力机制与 Transformer 架构",
      }),
    ];
    const rows = [
      row({ lesson: 1, topic: "提示工程基础与常用模式" }),
      row({ lesson: 2, topic: "注意力机制与 Transformer 架构" }),
    ];

    const matches = matchSegmentsToLog(segments, rows);

    expect(matches.map((item) => item.id)).toEqual([
      "lesson-0001",
      "lesson-0002",
    ]);
    expect(matches[0]).toEqual({
      id: "lesson-0001",
      segment: segments[0],
      log: rows[0],
    });
  });

  it("fails instead of guessing between duplicate same-date topic candidates", () => {
    const segments = [segment({ section: 1, titleZh: "同一主题" })];
    const rows = [
      row({ lesson: 1, topic: "同一主题" }),
      row({ lesson: 2, topic: "同一主题", track: "B", depth: "L2" }),
    ];

    expect(() => matchSegmentsToLog(segments, rows)).toThrow(
      /ambiguous lesson mapping.*2026-07-06\.md.*section 1.*lesson 1.*lesson 2/is,
    );
  });

  it("reserves explicit lesson identities before matching implicit segments", () => {
    const segments = [
      segment({ section: 1, titleZh: "同一主题" }),
      segment({ section: 2, lesson: 1, titleZh: "标题可以不同" }),
    ];
    const rows = [
      row({ lesson: 1, topic: "同一主题" }),
      row({ lesson: 2, topic: "同一主题" }),
    ];

    const matches = matchSegmentsToLog(segments, rows);

    expect(matches.map((item) => item.id)).toEqual([
      "lesson-0002",
      "lesson-0001",
    ]);
  });

  it("uses an explicit lesson number even when its topic differs", () => {
    const explicit = segment({
      section: 3,
      lesson: 7,
      titleZh: "源标题与日志标题不同",
    });
    const explicitRow = row({ lesson: 7, topic: "日志中的正式标题" });

    expect(matchSegmentsToLog([explicit], [explicitRow])).toEqual([
      { id: "lesson-0007", segment: explicit, log: explicitRow },
    ]);
  });

  it("rejects a missing explicit lesson candidate with source diagnostics", () => {
    const explicit = segment({
      file: "2026-07-08.md",
      section: 4,
      lesson: 99,
      date: "2026-07-08",
      titleZh: "缺少日志行",
    });

    expect(() => matchSegmentsToLog([explicit], [row({ lesson: 1, topic: "其他" })]))
      .toThrow(/explicit lesson mapping.*lesson 99.*2026-07-08\.md.*section 4/is);
  });

  it("preflights duplicate explicit candidates as duplicate stable IDs", () => {
    const explicit = segment({ section: 2, lesson: 3, titleZh: "显式课程" });
    const rows = [
      row({ lesson: 3, topic: "候选一" }),
      row({ lesson: 3, topic: "候选二", track: "B" }),
    ];

    expect(() => matchSegmentsToLog([explicit], rows)).toThrow(
      /duplicate lesson id.*lesson-0003.*log row 1.*log row 2/is,
    );
  });

  it("preflights duplicate stable IDs even when both rows are unused", () => {
    const source = segment({ section: 1, titleZh: "已使用课程" });
    const rows = [
      row({ lesson: 1, topic: "已使用课程" }),
      row({ lesson: 9, date: "2026-07-07", topic: "未使用一" }),
      row({ lesson: 9, date: "2026-07-08", topic: "未使用二" }),
    ];

    expect(() => matchSegmentsToLog([source], rows)).toThrow(
      /duplicate lesson id.*lesson-0009.*log row 2.*log row 3/is,
    );
  });

  it("preflights non-positive and non-safe log lesson numbers", () => {
    for (const lesson of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1]) {
      expect(() =>
        matchSegmentsToLog([], [row({ lesson, topic: "无效课程" })]),
      ).toThrow(/invalid log lesson number.*log row 1/i);
    }
  });

  it("rejects an explicit lesson whose log row has a conflicting date", () => {
    const explicit = segment({
      file: "2026-07-08.md",
      section: 1,
      lesson: 3,
      date: "2026-07-08",
      titleZh: "显式课程",
    });
    const conflicting = row({
      lesson: 3,
      date: "2026-07-07",
      topic: "显式课程",
    });

    expect(() => matchSegmentsToLog([explicit], [conflicting])).toThrow(
      /explicit lesson mapping.*lesson 3.*date.*2026-07-08.*2026-07-07/is,
    );
  });

  it("normalizes Unicode, case, whitespace, and punctuation", () => {
    const source = segment({
      section: 1,
      titleZh: "  Ｔｒａｎｓｆｏｒｍｅｒ：提示工程（基础）  ",
    });
    const log = row({ lesson: 8, topic: "transformer 提示工程 - 基础" });
    const unrelated = row({ lesson: 9, topic: "其他主题" });

    expect(matchSegmentsToLog([source], [log, unrelated])[0]?.id).toBe(
      "lesson-0008",
    );
  });

  it("does not semantically fuzz when complete order fallback is unavailable", () => {
    const source = segment({ section: 1, titleZh: "提示工程基础" });
    const rows = [
      row({ lesson: 1, topic: "提示工程进阶" }),
      row({ lesson: 2, topic: "另一课程" }),
    ];

    expect(() => matchSegmentsToLog([source], rows)).toThrow(
      /unmatched lesson segment.*2026-07-06\.md.*section 1.*提示工程进阶/is,
    );
  });

  it("preserves semantic symbols instead of normalizing them away", () => {
    const cases = [
      { source: "C#", collapsed: "C" },
      { source: "C++", collapsed: "C" },
      { source: "A/B", collapsed: "AB" },
      { source: "R&D", collapsed: "RD" },
    ];

    cases.forEach(({ source, collapsed }, index) => {
      expect(() =>
        matchSegmentsToLog(
          [segment({ section: 1, titleZh: source })],
          [
            row({ lesson: 30 + index * 2, topic: collapsed }),
            row({ lesson: 31 + index * 2, topic: "占位课程" }),
          ],
        ),
      ).toThrow(/unmatched lesson segment/i);
    });
  });

  it("restricts implicit candidates to the same date", () => {
    const source = segment({
      file: "2026-07-07.md",
      section: 1,
      date: "2026-07-07",
      titleZh: "同名课程",
    });
    const wrongDate = row({
      lesson: 1,
      date: "2026-07-06",
      topic: "同名课程",
    });

    expect(() => matchSegmentsToLog([source], [wrongDate])).toThrow(
      /unmatched lesson segment.*2026-07-07\.md.*section 1/is,
    );
  });

  it("uses section order only for a complete one-to-one duplicate-topic set", () => {
    const segments = [
      segment({ section: 2, titleZh: "重复主题" }),
      segment({ section: 1, titleZh: "重复主题" }),
    ];
    const rows = [
      row({ lesson: 11, topic: "重复主题" }),
      row({ lesson: 12, topic: "重复主题" }),
    ];

    const matches = matchSegmentsToLog(segments, rows);

    expect(matches.map((item) => item.id)).toEqual([
      "lesson-0012",
      "lesson-0011",
    ]);
  });

  it("uses section order for a complete same-date set whose topics differ", () => {
    const segments = [
      segment({ section: 2, titleZh: "源主题二" }),
      segment({ section: 1, titleZh: "源主题一" }),
    ];
    const rows = [
      row({ lesson: 21, topic: "日志主题甲" }),
      row({ lesson: 22, topic: "日志主题乙" }),
    ];

    const matches = matchSegmentsToLog(segments, rows);

    expect(matches.map((item) => item.id)).toEqual([
      "lesson-0022",
      "lesson-0021",
    ]);
  });

  it("rejects duplicate stable IDs produced by distinct log rows", () => {
    const segments = [
      segment({ section: 1, titleZh: "主题一" }),
      segment({ section: 2, titleZh: "主题二" }),
    ];
    const rows = [
      row({ lesson: 4, topic: "主题一" }),
      row({ lesson: 4, topic: "主题二" }),
    ];

    expect(() => matchSegmentsToLog(segments, rows)).toThrow(
      /duplicate lesson id.*lesson-0004/i,
    );
  });

  it("rejects reuse of an explicit log row", () => {
    const segments = [
      segment({ section: 1, lesson: 5, titleZh: "第一处" }),
      segment({ section: 2, lesson: 5, titleZh: "第二处" }),
    ];

    expect(() =>
      matchSegmentsToLog(segments, [row({ lesson: 5, topic: "日志行" })]),
    ).toThrow(/reused log row.*lesson 5.*section 1.*section 2/is);
  });
});
