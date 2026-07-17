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

async function writeTemporaryLessons(
  files: Record<string, string>,
): Promise<string> {
  const root = await fs.mkdtemp(path.join(tmpdir(), "personal-blog-lessons-"));
  temporaryRoots.push(root);
  await Promise.all(
    Object.entries(files).map(([filename, source]) =>
      fs.writeFile(path.join(root, filename), source, "utf8"),
    ),
  );
  return root;
}

async function writeTemporaryLesson(
  source: string,
  filename = "2026-07-06.md",
): Promise<string> {
  const root = await writeTemporaryLessons({ [filename]: source });
  return path.join(root, filename);
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

  it("excludes calendar-invalid date filenames", async () => {
    const root = await writeTemporaryLessons({
      "2026-02-28.md": "# 📅 Valid",
      "2026-02-30.md": "# 📅 Invalid",
    });

    const files = await discoverLessonFiles(root);

    expect(files.map((file) => path.basename(file))).toEqual([
      "2026-02-28.md",
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

  it("ignores H1 lesson headings nested inside list items", async () => {
    const file = await writeTemporaryLesson(
      [
        "# 📅 2026-07-06 — 根课程一",
        "正文",
        "",
        "- 列表项",
        "",
        "  # 📅 2026-07-06 — 嵌套课程",
        "",
        "# 📅 2026-07-06 — 根课程二",
      ].join("\n"),
    );

    const segments = await splitLessonSegments(file);

    expect(segments.map((segment) => segment.titleZh)).toEqual([
      "根课程一",
      "根课程二",
    ]);
    expect(segments[0]?.raw).toContain("  # 📅 2026-07-06 — 嵌套课程");
  });

  it("does not treat a backtick-containing info string as a fence opener", async () => {
    const file = await writeTemporaryLesson(
      [
        "# 📅 2026-07-06 — 第一课",
        "```bad`info",
        "# 📅 2026-07-06 — 第二课",
      ].join("\n"),
    );

    const segments = await splitLessonSegments(file);

    expect(segments.map((segment) => segment.titleZh)).toEqual([
      "第一课",
      "第二课",
    ]);
  });

  it("recognizes up to three leading spaces but not four", async () => {
    const file = await writeTemporaryLesson(
      [
        "# 📅 2026-07-06 — 零空格",
        "正文一",
        " # 📅 2026-07-06 — 一空格",
        "正文二",
        "  # 📅 2026-07-06 — 二空格",
        "正文三",
        "   # 📅 2026-07-06 — 三空格",
        "正文四",
        "    # 📅 2026-07-06 — 四空格代码",
      ].join("\n"),
    );

    const segments = await splitLessonSegments(file);

    expect(segments.map((segment) => segment.titleZh)).toEqual([
      "零空格",
      "一空格",
      "二空格",
      "三空格",
    ]);
    expect(segments[3]?.raw).toContain("    # 📅 2026-07-06 — 四空格代码");
  });

  it("excludes optional ATX closing hashes from the title only", async () => {
    const source = "  # 📅 2026-07-06 — Normal ###   \n正文";
    const file = await writeTemporaryLesson(source);

    const [segment] = await splitLessonSegments(file);

    expect(segment?.titleZh).toBe("Normal");
    expect(segment?.raw).toBe(source);
    expect(segment?.hash).toBe(
      `sha256:${createHash("sha256").update(source).digest("hex")}`,
    );
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

  it("preserves segment offsets after astral emoji content", async () => {
    const first = "# 📅 2026-07-06 — 第一课\n😀 astral content\n";
    const second = "# 📅 2026-07-06 — 第二课\n正文";
    const file = await writeTemporaryLesson(`${first}${second}`);

    const segments = await splitLessonSegments(file);

    expect(segments.map((segment) => segment.raw)).toEqual([first, second]);
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
    expect(segments[0]).not.toHaveProperty("lesson");
    expect(segments[1]?.lesson).toBe(2);

    for (const segment of segments) {
      const expectedHash = createHash("sha256")
        .update(segment.raw)
        .digest("hex");
      expect(segment.hash).toBe(`sha256:${expectedHash}`);
      expect(segment.hash).toMatch(/^sha256:[0-9a-f]{64}$/);
    }
  });

  it("extracts a separate explicit lesson number and preserves dots inside the topic", async () => {
    const titleZh =
      "进阶 RAG 检索工程（Hybrid · RRF · Rerank · Contextual Retrieval）";
    const source = `# 📅 2026-07-14 · 讲10 · 轨道A · ${titleZh} · 深度 L3\n正文`;
    const file = await writeTemporaryLesson(source, "2026-07-14.md");

    const [segment] = await splitLessonSegments(file);

    expect(segment).toMatchObject({
      date: "2026-07-14",
      lesson: 10,
      titleZh,
      raw: source,
    });
  });

  it("reports malformed structured headings with file and section context", async () => {
    const file = await writeTemporaryLesson(
      [
        "# 📅 2026-07-06 — 第一课",
        "正文",
        "# 📅 2026-07-06 · 轨道A · 缺少深度",
      ].join("\n"),
    );

    await expect(splitLessonSegments(file)).rejects.toThrow(
      /malformed structured lesson heading.*2026-07-06\.md.*section 2/i,
    );
  });
});
