import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";
import { SidecarMetadataSchema } from "../scripts/lib/metadata";

const aiDaily = defineCollection({
  loader: glob({
    base: "./src/content/ai-daily",
    pattern: "**/*.md",
  }),
  // Generated frontmatter is the sidecar projection. Reusing its schema keeps
  // the public collection bound to the same editorial and identity rules.
  schema: z.preprocess((value) => {
    if (
      value !== null &&
      typeof value === "object" &&
      "date" in value &&
      value.date instanceof Date
    ) {
      return {
        ...value,
        date: value.date.toISOString().slice(0, 10),
      };
    }
    return value;
  }, SidecarMetadataSchema),
});

export const collections = { aiDaily };
