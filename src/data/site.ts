export const site = {
  name: "sYYmmEtra",
  role: "Software engineer learning and practicing AI engineering",
  description:
    "Personal notes on software, artificial intelligence, and deliberate learning.",
  descriptionZh: "关于软件、人工智能与刻意学习的个人笔记。",
  siteUrl: "https://syymmetra.github.io",
  githubUrl: "https://github.com/sYYmmEtra",
  email: "private-contact@example.invalid",
} as const;

export const navigation = [
  { key: "home", href: "/", available: true },
  { key: "archive", href: "/ai-daily/", available: true },
  { key: "projects", href: "/projects/", available: true },
  { key: "about", href: "/about/", available: true },
] as const;

export type NavigationKey = (typeof navigation)[number]["key"];
