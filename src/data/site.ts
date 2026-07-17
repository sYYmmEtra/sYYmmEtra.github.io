export const site = {
  name: "sYYmmEtra",
  role: "Software engineer learning and practicing AI engineering",
  description:
    "Personal notes on software, artificial intelligence, and deliberate learning.",
  siteUrl: "https://syymmetra.github.io",
  githubUrl: "https://github.com/sYYmmEtra",
  email: "private-contact@example.invalid",
} as const;

export const navigation = [
  { key: "home", href: "/", available: true },
  { key: "archive", href: "/ai-daily/", available: false },
  { key: "projects", href: "/projects/", available: false },
  { key: "about", href: "/about/", available: false },
] as const;

export type NavigationKey = (typeof navigation)[number]["key"];
