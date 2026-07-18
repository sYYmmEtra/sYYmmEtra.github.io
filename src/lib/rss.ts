import {
  resolveMetadataDisplay,
  type SidecarMetadata,
} from "../../scripts/lib/metadata";

export interface RssItem {
  title: string;
  description: string;
  link: string;
  pubDate: Date;
  categories: string[];
}

export function rssItemFromMetadata(lesson: SidecarMetadata): RssItem {
  const display = resolveMetadataDisplay(lesson);
  return {
    title: display.title,
    description: `${display.originalInChinese ? "Original in Chinese — " : ""}${display.summary}`,
    link: `/ai-daily/${lesson.slug}/`,
    pubDate: new Date(`${lesson.date}T00:00:00Z`),
    categories: [`Track ${lesson.track}`, lesson.depth, ...display.tags],
  };
}
