import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import { rssItemFromMetadata } from "../lib/rss";
import { site } from "../data/site";

export async function GET() {
  const lessons = (await getCollection("aiDaily")).sort((left, right) =>
    right.data.date.localeCompare(left.data.date) || right.data.lesson - left.data.lesson,
  );
  return rss({
    title: `${site.name} — AI Daily`,
    description: site.description,
    site: site.siteUrl,
    items: lessons.map((lesson) => rssItemFromMetadata(lesson.data)),
  });
}
