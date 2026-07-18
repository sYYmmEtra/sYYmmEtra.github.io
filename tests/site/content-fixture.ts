import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import matter from "gray-matter";

export function currentTrackCounts(repoRoot: string): Record<"A" | "B" | "C", number> {
  const counts = { A: 0, B: 0, C: 0 };
  const contentDirectory = path.join(repoRoot, "src/content/ai-daily");

  for (const filename of readdirSync(contentDirectory)) {
    if (!filename.endsWith(".md")) continue;
    const { data } = matter(readFileSync(path.join(contentDirectory, filename), "utf8"));
    const track: unknown = data.track;
    if (track === "A" || track === "B" || track === "C") {
      counts[track] += 1;
    }
  }

  return counts;
}
