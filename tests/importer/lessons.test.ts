import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  discoverLessonFiles,
  splitLessonSegments,
} from "../../scripts/lib/lessons";

const fixtureRoot = path.resolve("tests/fixtures/lessons");
const temporaryRoots: string[] = [];

async function writeTemporaryLesson(source: string): Promise<string> {
  const root = await fs.mkdtemp(path.join(tmpdir(), "personal-blog-lessons-"));
  const file = path.join(root, "2026-07-06.md");
  temporaryRoots.push(root);
  await fs.writeFile(file, source, "utf8");
  return file;
}

afterEach(async () => {
  for (const root of temporaryRoots.splice(0)) {
    await fs.rm(root, { recursive: true, force: true });
  }
});

describe("lesson discovery", () => {
  it("includes only YYYY-MM-DD.md files", async () => {
    const files = await discoverLessonFiles(fixtureRoot);

    expect(files.map((file) => path.basename(file))).toEqual([
      "2026-07-06.md",
    ]);
  });
});

describe("lesson splitting", () => {
  it("splits multiple lessons in one date file", async () => {
    const segments = await splitLessonSegments(
      path.join(fixtureRoot, "2026-07-06.md"),
    );

    expect(segments).toHaveLength(2);
    expect(segments.map((item) => item.section)).toEqual([1, 2]);
  });

  it("normalizes BOM and CRLF while preserving all other segment whitespace", async () => {
    const file = await writeTemporaryLesson(
      "\uFEFF# 📅 2026-07-06 — 第一课\r\n\r\n正文一  \r\n## 📅 内部标题\r\n\r\n# 📅 2026-07-06 — 第二课\r\n\t正文二",
    );

    const segments = await splitLessonSegments(file);

    const expectedRaw = [
      "# 📅 2026-07-06 — 第一课\n\n正文一  \n## 📅 内部标题\n\n",
      "# 📅 2026-07-06 — 第二课\n\t正文二",
    ];

    expect(segments.map((segment) => segment.raw)).toEqual(expectedRaw);
    expect(segments.map((segment) => segment.hash)).toEqual(
      expectedRaw.map(
        (raw) => `sha256:${createHash("sha256").update(raw).digest("hex")}`,
      ),
    );
  });

  it("does not split heading-like lines inside fenced code blocks", async () => {
    const file = await writeTemporaryLesson(
      [
        "# 📅 2026-07-06 — 第一课",
        "",
        "```md",
        "# 📅 2026-07-06 — 反引号代码示例",
        "```",
        "",
        "~~~text",
        "# 📅 2026-07-06 — 波浪线代码示例",
        "~~~",
        "",
        "# 📅 2026-07-06 — 第二课",
        "正文",
      ].join("\n"),
    );

    const segments = await splitLessonSegments(file);

    expect(segments.map((segment) => segment.titleZh)).toEqual([
      "第一课",
      "第二课",
    ]);
    expect(segments[0]?.raw).toContain("# 📅 2026-07-06 — 反引号代码示例");
    expect(segments[0]?.raw).toContain("# 📅 2026-07-06 — 波浪线代码示例");
  });

  it("preserves whether the source ends with a final newline", async () => {
    const withoutNewline = await writeTemporaryLesson(
      "# 📅 2026-07-06 — 无换行\n正文",
    );
    const withNewline = await writeTemporaryLesson(
      "# 📅 2026-07-06 — 有换行\n正文\n",
    );

    const [withoutSegment] = await splitLessonSegments(withoutNewline);
    const [withSegment] = await splitLessonSegments(withNewline);

    expect(withoutSegment?.raw).toBe("# 📅 2026-07-06 — 无换行\n正文");
    expect(withSegment?.raw).toBe("# 📅 2026-07-06 — 有换行\n正文\n");
  });

  it("returns source metadata, Chinese titles, and hashes of normalized segments", async () => {
    const segments = await splitLessonSegments(
      path.join(fixtureRoot, "2026-07-06.md"),
    );

    expect(segments).toMatchObject([
      {
        file: "2026-07-06.md",
        section: 1,
        date: "2026-07-06",
        titleZh: "提示工程基础与常用模式",
      },
      {
        file: "2026-07-06.md",
        section: 2,
        date: "2026-07-06",
        titleZh: "注意力机制与 Transformer 架构",
      },
    ]);

    for (const segment of segments) {
      const expectedHash = createHash("sha256")
        .update(segment.raw)
        .digest("hex");
      expect(segment.hash).toBe(`sha256:${expectedHash}`);
      expect(segment.hash).toMatch(/^sha256:[0-9a-f]{64}$/);
    }
  });
});
